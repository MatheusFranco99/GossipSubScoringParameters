#!/bin/env python3

import sys
from typing import Dict

from eth2spec.phase0.spec import (
    uint64,
    MAX_VALIDATORS_PER_COMMITTEE,
    MAX_COMMITTEES_PER_SLOT,
    SLOTS_PER_EPOCH,
    SECONDS_PER_SLOT,
    TARGET_COMMITTEE_SIZE,
    TARGET_AGGREGATORS_PER_COMMITTEE,
    # ATTESTATION_SUBNET_COUNT,
)

ATTESTATION_SUBNET_COUNT = 64

# <- INPUT constants for GossipSub parameter
D = 8


def get_committee_count_per_slot(active_validators: uint64) -> uint64:
    """
    This function does also exist in eth2sepc but calculates the number of active validators within the function,
    this is basically the same function but with the number of active validators as parameter.
    """
    return max(
        uint64(1),
        min(
            MAX_COMMITTEES_PER_SLOT,
            active_validators // SLOTS_PER_EPOCH // TARGET_COMMITTEE_SIZE,
        ),
    )


def expected_aggregator_count_per_slot(active_validators: uint64) -> float:
    committees = get_committee_count_per_slot(active_validators) * SLOTS_PER_EPOCH

    smaller_committee_size = active_validators // committees
    num_larger_committees = active_validators - smaller_committee_size * committees

    modulo_smaller = max(1, smaller_committee_size // TARGET_AGGREGATORS_PER_COMMITTEE)
    modulo_larger = max(
        1, (smaller_committee_size + 1) // TARGET_AGGREGATORS_PER_COMMITTEE
    )

    return (
        float((committees - num_larger_committees) * smaller_committee_size)
        / modulo_smaller
        + float(num_larger_committees * (smaller_committee_size + 1)) / modulo_larger
    ) / SLOTS_PER_EPOCH


class TopicParams:
    topic_weight: float
    time_in_mesh_weight: float
    time_in_mesh_quantum: float  # in seconds
    time_in_mesh_cap: float
    first_message_deliveries_weight: float
    first_message_deliveries_decay: float
    first_message_deliveries_cap: float
    mesh_message_deliveries_weight: float
    mesh_message_deliveries_decay: float
    mesh_message_deliveries_threshold: float
    mesh_message_deliveries_cap: float
    mesh_message_deliveries_activation: float  # in seconds
    mesh_message_deliveries_window: float  # in seconds
    mesh_failure_penalty_weight: float
    mesh_failure_penalty_decay: float
    invalid_message_deliveries_weight: float
    invalid_message_deliveries_decay: float

    def __repr__(self):
        return str(self.__dict__)

    def max_positive_score(self):
        return (
            self.first_message_deliveries_weight * self.first_message_deliveries_cap
            + self.time_in_mesh_weight * self.time_in_mesh_cap
        ) * self.topic_weight


class ScoringParams:
    gossip_threshold: float
    publish_threshold: float
    graylist_threshold: float
    accept_px_threshold: float
    oppertunistic_graft_threshold: float
    decay_interval: float  # in seconds
    decay_to_zero: float
    retain_score: float  # in seconds
    app_specific_weight: float
    ip_colocation_factor_weight: float
    ip_colocation_factor_threshold: uint64
    behaviour_penalty_threshold: float
    behaviour_penalty_weight: float
    behaviour_penalty_decay: float
    topic_score_cap: float
    topic_params: Dict[str, TopicParams]

    def __repr__(self):
        return str(self.__dict__)


def score_parameter_decay_with_base(
    decay_time_in_seconds: float, decay_interval_in_seconds: float, decay_to_zero: float
) -> float:
    """
    computes the decay to use such that a value of 1 decays to 0 (using the decay_to_zero parameter and the
    decay_interval) within the specified decay_time_in_seconds
    """

    ticks = decay_time_in_seconds / decay_interval_in_seconds
    return pow(decay_to_zero, 1 / ticks)


def slots(amount: float) -> float:
    return amount * float(SECONDS_PER_SLOT)


def epochs(amount: float) -> float:
    return slots(amount * float(SLOTS_PER_EPOCH))


def decay_convergence(decay: float, rate: float) -> float:
    """
    Computes the limit to which a decay process will convert if it has the given issuaence rate per decay interval and
    the given decay factor.

    This is basically the convergence for a score that decays [DECAY] and at the same time increases WITH [RATE] (addition, not multiplication).
    So, after k periods, the score is:
    S = S0 * decay^k + rate * decay^k + rate * decay^k-1 + .... + rate * decay
    Going to the limit, the second part is
    rate / (1 - decay)
    Rate here is not a good name though, could be Increment
    """
    return rate / (1 - decay)


def threshold(decay: float, required_rate: float) -> float:
    """
    Computes a threshold value if we require at least the given rate with the given decay
    (In fact we require strictly more than the given rate, since the rate will reach the threshold only at infinity)
    """

    # we multiply the limit with the decay so that we are save to reach the threshold even directly after a decay event
    return decay_convergence(decay, required_rate) * decay


def get_scoring_params(active_validators: uint64) -> ScoringParams:
    """Given number of validators, returns the score parameter"""
    max_validators = (
        MAX_VALIDATORS_PER_COMMITTEE * MAX_COMMITTEES_PER_SLOT * SLOTS_PER_EPOCH
    )
    assert SLOTS_PER_EPOCH <= active_validators <= max_validators

    params = ScoringParams()

    # Defines Topic Weight for each subnet
    beacon_block_weight = 0.5
    beacon_aggregate_proof_weight = 0.5
    beacon_attestation_subnet_weight = 1.0 / float(ATTESTATION_SUBNET_COUNT)
    voluntary_exit_weight = 0.05
    proposer_slashing_weight = 0.05
    attester_slashing_weight = 0.05

    # first we need to compute the theoretical maximum positive score
    # OBS: it does not include the P5 (app specific weight), but it's not related to the topic score
    max_in_mesh_score = 10 # <-- INPUT (max w1 * P1)
    max_first_message_deliveries_score = 40 # <-- INPUT (max w2 * P2)
    max_positive_score = (max_in_mesh_score + max_first_message_deliveries_score) * (
        beacon_block_weight
        + beacon_aggregate_proof_weight
        + beacon_attestation_subnet_weight * ATTESTATION_SUBNET_COUNT
        + voluntary_exit_weight
        + proposer_slashing_weight
        + attester_slashing_weight
    )

    params.topic_score_cap = max_positive_score / 2 # <-- INPUT (he assigns that half the score comes from the topics params. and half from P5, P6 and P7)

    # global parameters
    # -> Thresholds
    params.gossip_threshold = -4000  # as in report
    params.publish_threshold = -8000
    params.graylist_threshold = -16000
    params.accept_px_threshold = 100  # only for boot nodes with enough AppSpecificScore
    params.oppertunistic_graft_threshold = 5  # needs to be tested
    # -> Decay and retain
    params.decay_interval = slots(1) # <-- INPUT (but it's a good heuristic and better change the decay accordingly. Also, Score function is heavy and we should care with low decays intervals)
    params.decay_to_zero = 0.01 # <-- INPUT -> will break decay_convergence calls if changed
    params.retain_score = epochs(100)

    # ==============
    # P5 APP SPECIFIC
    # ==============
    params.app_specific_weight = 1 # <-- INPUT

    # ==============
    # P6 IP COLOCATION
    # ==============
    params.ip_colocation_factor_threshold = 3 # <-- INPUT
    params.ip_colocation_factor_weight = -params.topic_score_cap # Can zero the topic score even if it previously had full score

    # helper function
    def score_parameter_decay(decay_time_in_seconds: float) -> float:
        """
        computes the decay to use such that a value of 1 decays to 0 (using the DecayToZero parameter) within the
        specified decay_time_in_seconds
        (Time wanted to decay 1 to 0 -> decay rate)
        """

        return score_parameter_decay_with_base(
            decay_time_in_seconds, params.decay_interval, params.decay_to_zero
        )

    # further global parameters
    # ==============
    # P7 BEHAVIOUR PENALTY
    # ==============
    params.behaviour_penalty_decay = score_parameter_decay(epochs(10))
    params.behaviour_penalty_threshold = 6  # as in lotus


    # we want to ignore gossip for a peer if he has more than 10 behaviour penalties per epoch
    # which weight we should put in order to reach a threshold given a certain increment rate
    target_value = (
        decay_convergence(params.behaviour_penalty_decay, 10 / float(SLOTS_PER_EPOCH)) # -> score convergence decaying and increasing by the 2nd argument amount
        - params.behaviour_penalty_threshold
    )
    params.behaviour_penalty_weight = params.gossip_threshold / (target_value**2) # squares because libp2p squares the deviation. <--INPUT (i think it could be the graylist threshold instead)

    def topic_params(
        topic_weight: float,
        expected_message_rate: float,
        first_message_decay_time_in_seconds: float,
        mesh_message_decay_time_in_seconds: float = 0.0,
        mesh_message_cap_factor: float = 0.0,
        mesh_message_activation_in_seconds: float = 0.0,
    ) -> TopicParams:
        topic_params = TopicParams()
        topic_params.topic_weight = topic_weight

        # ==============
        # P1 TIME IN MESH
        # ==============
        topic_params.time_in_mesh_quantum = float(SECONDS_PER_SLOT) # One point per slot <-- INPUT (could be changed but )
        topic_params.time_in_mesh_cap = 3600 / topic_params.time_in_mesh_quantum # <-- INPUT 1 hour / 12 seconds -> 300.
        topic_params.time_in_mesh_weight = (
            max_in_mesh_score / topic_params.time_in_mesh_cap # w1 = MaxTimeInMeshScore / max P1
        )

        # ==============
        # P2 FIRST DELIVERY
        # ==============
        topic_params.first_message_deliveries_decay = score_parameter_decay(
            first_message_decay_time_in_seconds
        )
        # we assume every peer in the mesh sends the same amount of first message deliveries and cap the rate at twice
        # the value
        topic_params.first_message_deliveries_cap = decay_convergence(
            topic_params.first_message_deliveries_decay, 2 * expected_message_rate*12 / D # convergence value for Decay and equal first delivery (cap at twice)
        )
        topic_params.first_message_deliveries_weight = (
            max_first_message_deliveries_score
            / topic_params.first_message_deliveries_cap # w2 = MaxFirstDeliveryScore / max P2
        )

        # ==============
        # P3 AND P3B MSG DELIVERY RATE AND MESH FAILURE
        # ==============
        if mesh_message_decay_time_in_seconds > 0.0:
            topic_params.mesh_message_deliveries_decay = score_parameter_decay(
                mesh_message_decay_time_in_seconds
            )
            # a peer must send us at least 1/50 of the regular messages in time, very conservative limit
            topic_params.mesh_message_deliveries_threshold = threshold(
                topic_params.mesh_message_deliveries_decay, expected_message_rate*12 / 50
            )
            topic_params.mesh_message_deliveries_weight = -max_positive_score / (
                topic_params.topic_weight
                * topic_params.mesh_message_deliveries_threshold**2
            )
            topic_params.mesh_message_deliveries_cap = (
                mesh_message_cap_factor * topic_params.mesh_message_deliveries_threshold
            )
            topic_params.mesh_message_deliveries_activation = (
                mesh_message_activation_in_seconds
            )
            topic_params.mesh_message_deliveries_window = 2

            topic_params.mesh_failure_penalty_weight = (
                topic_params.mesh_message_deliveries_weight
            )
            topic_params.mesh_failure_penalty_decay = (
                topic_params.mesh_message_deliveries_decay
            )
        else:
            topic_params.mesh_message_deliveries_weight = 0
            topic_params.mesh_message_deliveries_threshold = 0
            topic_params.mesh_failure_penalty_weight = 0


        # ==============
        # P4 INVALID MSG
        # ==============
        topic_params.invalid_message_deliveries_weight = (  # Should we use here the same logic above for P7 for instance, using an allowed rate of malicious messages per slot?
            -max_positive_score / topic_params.topic_weight
                                                            # Better would be to take Graylist threshold into account
        )
        topic_params.invalid_message_deliveries_decay = score_parameter_decay(
            epochs(50) # <-- INPUT
        )

        return topic_params


    committees_per_slot = get_committee_count_per_slot(active_validators)
    multiple_bursts_per_subnet_per_epoch = (
        committees_per_slot >= 2 * ATTESTATION_SUBNET_COUNT // SLOTS_PER_EPOCH
    )
    params.topic_params = {
        "beacon_block": topic_params(
            beacon_block_weight, 1, epochs(20), epochs(5), 3, epochs(1)
        ),
        "beacon_aggregate_and_proof": topic_params(
            beacon_aggregate_proof_weight,
            expected_aggregator_count_per_slot(active_validators),
            epochs(1),
            epochs(2),
            4,
            epochs(1),
        ),
        "beacon_attestation_subnet": topic_params(
            beacon_attestation_subnet_weight,
            float(active_validators)
            / float(ATTESTATION_SUBNET_COUNT)
            / float(SLOTS_PER_EPOCH),
            epochs(1) if multiple_bursts_per_subnet_per_epoch else epochs(4),
            epochs(4) if multiple_bursts_per_subnet_per_epoch else epochs(16),
            16,
            slots(SLOTS_PER_EPOCH // 2 + 1)
            if multiple_bursts_per_subnet_per_epoch
            else epochs(3),
        ),
        "voluntary_exit": topic_params(
            voluntary_exit_weight, 4 / float(SLOTS_PER_EPOCH), epochs(100)
        ),
        "proposer_slashing": topic_params(
            proposer_slashing_weight, 1 / 5 / float(SLOTS_PER_EPOCH), epochs(100)
        ),
        "attester_slashing": topic_params(
            attester_slashing_weight, 1 / 5 / float(SLOTS_PER_EPOCH), epochs(100)
        ),
    }

    return params


def print_testground_toml(active_validators: uint64):
    """
    Receives a number of validator and
    Returns the TOML file to test with testground
    Specifies 15 subnets:
        - beacon block
        - beacon aggregate and proof
        - 10 beacon attestation
        - voluntary exit
        - proposer slashing
        - attester slashing

    Anyway, scoring parameters don't matter for 1 or X subnets.
    """
    params = get_scoring_params(active_validators)

    # config
    tab = "  "
    subnets = 10

    # print topic_config
    print("#config for {} validators".format(active_validators))
    print("TOPIC_CONFIG = [")
    print(
        "{} {{ id = 'beacon_block', message_rate = '1/{}s', message_size = '123 KiB' }},".format(
            tab, SECONDS_PER_SLOT
        )
    )
    print(
        "{} {{ id = 'beacon_aggregate_and_proof', message_rate = '{}/{}s', message_size = '680 B' }},".format(
            tab, expected_aggregator_count_per_slot(active_validators), SECONDS_PER_SLOT
        )
    )
    for i in range(subnets):
        print(
            "{} {{ id = 'beacon_attestation_{}', message_rate = '{}/{}s', message_size = '480 B' }},".format(
                tab,
                i,
                float(active_validators) / ATTESTATION_SUBNET_COUNT / SLOTS_PER_EPOCH,
                SECONDS_PER_SLOT,
            )
        )
    print(
        "{} {{ id = 'voluntary_exit', message_rate = '4/{}s', message_size = '112 B' }},".format(
            tab, SLOTS_PER_EPOCH * SECONDS_PER_SLOT
        )
    )
    print(
        "{} {{ id = 'proposer_slashing', message_rate = '1/{}s', message_size = '400 B' }},".format(
            tab, 100 * SLOTS_PER_EPOCH * SECONDS_PER_SLOT
        )
    )
    print(
        "{} {{ id = 'attester_slashing', message_rate = '1/{}s', message_size = '33 KiB' }},".format(
            tab, 100 * SLOTS_PER_EPOCH * SECONDS_PER_SLOT
        )
    )
    print("]")
    print()

    # print peer_score_params
    print("[PEER_SCORE_PARAMS]")
    print()
    print("TopicSoreCap = {}".format(params.topic_score_cap))
    print("AppSpecificWeight = {}".format(params.app_specific_weight))
    print("IPColocationFactorWeight = {}".format(params.ip_colocation_factor_weight))
    print(
        "IPColocationFactorThreshold = {}".format(params.ip_colocation_factor_threshold)
    )
    print("BehaviourPenaltyWeight = {}".format(params.behaviour_penalty_weight))
    print("BehaviourPenaltyThreshold = {}".format(params.behaviour_penalty_threshold))
    print("BehaviourPenaltyDecay = {}".format(params.behaviour_penalty_decay))
    print("DecayInterval = '{}s'".format(params.decay_interval))
    print("DecayToZero = {}".format(params.decay_to_zero))
    print("RetainScore = {}".format(params.retain_score))

    print()

    print("{}[PEER_SCORE_PARAMS.Thresholds]".format(tab))
    print("{}GossipThreshold = {}".format(tab, params.gossip_threshold))
    print("{}PublishThreshold = {}".format(tab, params.publish_threshold))
    print("{}GraylistThreshold = {}".format(tab, params.graylist_threshold))
    print("{}AcceptPXThreshold = {}".format(tab, params.accept_px_threshold))

    def print_topic(key, name):
        print()
        print("{}[PEER_SCORE_PARAMS.Topics.{}]".format(tab, name))
        tparams = params.topic_params[key]
        print("{}TopicWeight = {}".format(tab, tparams.topic_weight))
        print()
        print("{}TimeInMeshWeight = {}".format(tab, tparams.time_in_mesh_weight))
        print("{}TimeInMeshQuantum = {}".format(tab, tparams.time_in_mesh_quantum))
        print("{}TimeInMeshCap = {}".format(tab, tparams.time_in_mesh_cap))
        print()
        print(
            "{}FirstMessageDeliveriesWeight = {}".format(
                tab, tparams.first_message_deliveries_weight
            )
        )
        print(
            "{}FirstMessageDeliveriesDecay = {}".format(
                tab, tparams.first_message_deliveries_decay
            )
        )
        print(
            "{}FirstMessageDeliveriesCap = {}".format(
                tab, tparams.first_message_deliveries_cap
            )
        )
        print()
        print(
            "{}MeshMessageDeliveriesWeight = {}".format(
                tab, tparams.mesh_message_deliveries_weight
            )
        )
        if tparams.mesh_message_deliveries_weight < 0:
            print(
                "{}MeshMessageDeliveriesDecay = {}".format(
                    tab, tparams.mesh_message_deliveries_decay
                )
            )
            print(
                "{}MeshMessageDeliveriesThreshold = {}".format(
                    tab, tparams.mesh_message_deliveries_threshold
                )
            )
            print(
                "{}MeshMessageDeliveriesCap = {}".format(
                    tab, tparams.mesh_message_deliveries_cap
                )
            )
            print(
                "{}MeshMessageDeliveriesActivation = '{}s'".format(
                    tab, tparams.mesh_message_deliveries_activation
                )
            )
            print(
                "{}MeshMessageDeliveriesWindow = '{}s'".format(
                    tab, tparams.mesh_message_deliveries_window
                )
            )
        print()
        print(
            "{}MeshFailurePenaltyWeight = {}".format(
                tab, tparams.mesh_failure_penalty_weight
            )
        )
        if tparams.mesh_failure_penalty_weight < 0:
            print(
                "{}MeshFailurePenaltyDecay = {}".format(
                    tab, tparams.mesh_failure_penalty_decay
                )
            )
        print()
        print(
            "{}InvalidMessageDeliveriesWeight = {}".format(
                tab, tparams.invalid_message_deliveries_weight
            )
        )
        print(
            "{}InvalidMessageDeliveriesDecay = {}".format(
                tab, tparams.invalid_message_deliveries_decay
            )
        )

    print_topic("beacon_block", "beacon_block")
    print_topic("beacon_aggregate_and_proof", "beacon_aggregate_and_proof")
    for i in range(subnets):
        print_topic("beacon_attestation_subnet", "beacon_attestation_{}".format(i))
    print_topic("voluntary_exit", "voluntary_exit")
    print_topic("proposer_slashing", "proposer_slashing")
    print_topic("attester_slashing", "attester_slashing")


if __name__ == "__main__":
    print_testground_toml(uint64(int(sys.argv[1])))
