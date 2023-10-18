#!/bin/env python3

import sys
from typing import Dict
import math

from eth2spec.phase0.spec import (
    uint64,
    SLOTS_PER_EPOCH,
    SECONDS_PER_SLOT,
)

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


def get_scoring_params(D: uint64 = 6,
                       num_subnets: uint64 = 128,
                       total_topics_weight: float = 4.0,
                       max_in_mesh_score: float = 10, # (max w1 * P1)
                       max_first_message_deliveries_score: float = 40, # (max w2 * P2)
                       decay_interval: float = slots(1), # We can change but it's a good heuristic and better change the decay accordingly. Also, Score function is heavy and we should care with low decays intervals
                       decay_to_zero: float = 0.01,
                       retain_score: float = epochs(100),
                       topic_score_cap_percentage_from_total: float = 0.5, # max percentage that topics score may contribute to max total score
                       app_specific_weight: float = 0,
                       ip_colocation_factor_threshold: int = 10,
                       behaviour_penalty_decay_1_0_seconds: float = epochs(10),
                       behaviour_penalty_threshold: int = 6,
                       behaviour_penalty_limit_rate: float = 10 / float(SLOTS_PER_EPOCH),
                       msg_rate: float = 5.128,
                       time_in_mesh_quantum: float = float(SECONDS_PER_SLOT), # Can be changed but it's a good heuristic
                       time_in_mesh_quantum_cap: float = 3600, # Cap will be time_in_mesh_quantum_cap / time_in_mesh_quantum (e.g. 3600 / 12 -> 300)
                       first_message_decay_time_in_seconds: float = epochs(4),
                       mesh_message_decay_time_in_seconds: float = epochs(16),
                       mesh_message_cap_factor: float = 16,
                       mesh_message_activation_in_seconds: float = epochs(3),
                       mesh_message_deliveries_threshold_minimum_fraction: float = 1/50, # Peer should send us at least 1/50 of the expected number of messages.
                       mesh_message_deliveries_window: float = 2, # Time after first delivery that is considered "near-first" (in seconds)
                       invalid_message_deliveries_decaytime_1_0: float = epochs(100),
                       invalid_messages_limit_rate = 10 / float(SLOTS_PER_EPOCH),
                       ) -> ScoringParams:
    """Given number of validators, returns the score parameter"""


    params = ScoringParams()

    # Defines Topic Weight for each subnet
    topics_weight = total_topics_weight / num_subnets

    # first we need to compute the theoretical maximum positive score
    # OBS: it does not include the P5 (app specific weight), but it's not related to the topic score
    max_positive_score = (max_in_mesh_score + max_first_message_deliveries_score) * ( topics_weight * num_subnets)

    params.topic_score_cap = max_positive_score * topic_score_cap_percentage_from_total

    # global parameters
    # -> Thresholds
    params.gossip_threshold = -4000  # as in report
    params.publish_threshold = -8000
    params.graylist_threshold = -16000
    params.accept_px_threshold = 100  # only for boot nodes with enough AppSpecificScore
    params.oppertunistic_graft_threshold = 5  # needs to be tested

    # -> Decay and retain
    params.decay_interval = decay_interval
    params.decay_to_zero = decay_to_zero
    params.retain_score = retain_score

    # ==============
    # P5 APP SPECIFIC
    # ==============
    params.app_specific_weight = app_specific_weight

    # ==============
    # P6 IP COLOCATION
    # ==============
    params.ip_colocation_factor_threshold = ip_colocation_factor_threshold # <-- INPUT
    params.ip_colocation_factor_weight = -params.topic_score_cap # Drop to zero the topic score even if it previously had full score

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
    params.behaviour_penalty_decay = score_parameter_decay(behaviour_penalty_decay_1_0_seconds)
    params.behaviour_penalty_threshold = behaviour_penalty_threshold  # as in lotus


    # we want to ignore gossip for a peer if he has more than 10 behaviour penalties per epoch
    # which weight we should put in order to reach a threshold given a certain increment rate
    target_value = (
        decay_convergence(params.behaviour_penalty_decay, behaviour_penalty_limit_rate) # -> score convergence decaying and increasing by the 2nd argument amount
        - params.behaviour_penalty_threshold
    )
    # Here we can use gossip_threshold, publish_threshold or graylist threshold. The behaviour penalty, however, is related to control messages, so that's way it's used the gossip threshold
    params.behaviour_penalty_weight = params.gossip_threshold / (target_value**2) # squares because libp2p squares the deviation.

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
        topic_params.time_in_mesh_quantum = time_in_mesh_quantum
        topic_params.time_in_mesh_cap = time_in_mesh_quantum_cap / time_in_mesh_quantum
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
            topic_params.first_message_deliveries_decay, 2 * expected_message_rate / D # convergence value for Decay and equal first delivery (cap at twice)
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
            # a peer must send us at least mesh_message_deliveries_threshold_minimum_fraction of the regular messages in time, very conservative limit
            topic_params.mesh_message_deliveries_threshold = threshold(
                topic_params.mesh_message_deliveries_decay, expected_message_rate * mesh_message_deliveries_threshold_minimum_fraction
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
            topic_params.mesh_message_deliveries_window = mesh_message_deliveries_window

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


        topic_params.invalid_message_deliveries_decay = score_parameter_decay(invalid_message_deliveries_decaytime_1_0)
        # Should we use here the same logic above for P6 for instance, using an allowed rate of malicious messages per slot?
        topic_params.invalid_message_deliveries_weight = (
            params.graylist_threshold / (topic_params.topic_weight * 20 * 20) # It's better to take Graylist threshold into account
        )

        return topic_params

    params.topic_params = {
        f"subnet_{i}": topic_params(
            topics_weight,
            msg_rate*decay_interval, # Multiplied since we want the rate per decay interval
            first_message_decay_time_in_seconds = first_message_decay_time_in_seconds,
            mesh_message_decay_time_in_seconds= mesh_message_decay_time_in_seconds,
            mesh_message_cap_factor = mesh_message_cap_factor,
            mesh_message_activation_in_seconds = mesh_message_activation_in_seconds)
        for i in range(num_subnets)
    }


    return params


def print_testground_toml(subnets: uint64, msg_rate: float, msg_size: int, params: ScoringParams, show_all: bool = False):


    # config
    tab = "  "

    # print topic_config
    print("TOPIC_CONFIG = [")
    for i in range(subnets):
        print(
            "{} {{ id = 'subnet_{}', message_rate = '{}/{}s', message_size = '{} B' }},".format(
                tab,
                i,
                msg_rate,
                1,
                msg_size
            )
        )
        if not show_all:
            break
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

    if show_all:
        for i in range(subnets):
            print_topic(f"subnet_{i}", f"subnet_{i}")
    else:
        print_topic(f"subnet_{0}", f"subnet_{0}")


if __name__ == "__main__":

    num_subnets = 128
    msg_rate = 600/117
    msg_size = 200 # not used. Only for testground simulation
    scoringParams = get_scoring_params(num_subnets=num_subnets,msg_rate=msg_rate,D=8)

    print_testground_toml(num_subnets,msg_rate,msg_size,scoringParams, show_all = False)
