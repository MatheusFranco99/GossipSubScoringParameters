""" GossipSub Parameters """

from dataclasses import dataclass, field
from generate_scoring_params import slots, epochs


@dataclass
class Threshold:
    """ Threshold values, except the `0` threshold"""
    gossip_threshold: float = -4000
    publish_threshoold: float = -8000
    graylist_threshold: float = -1600
    accept_px_treshold: float = 100
    opportunistic_graft_threshold: float = 5


@dataclass
class ScoreParameters:
    """ All score parameters of the GossipSub """
    threshold: Threshold = field(default_factory=Threshold)

    decay_interval: float = slots(1)
    decay_to_zero: float = 0.01

    retain_score: float = epochs(100)

    topic_weight: float = 0.03125
    topic_score_cap: float = 100

    # P1
    time_in_mesh_weight: float = 0.033333333
    time_in_mesh_quantum: float = 12000000000
    time_in_mesh_cap: float = 300

    # P2
    first_msg_deliveries_weight: float = 3.186933212480587
    first_msg_deliveries_decay: float = 0.5623413251903491
    first_msg_deliveries_cap: float = 12.55125141730395

    # P3
    mesh_msg_deliveries_weight: float = -16.199996132888096
    mesh_msg_deliveries_decay: float = 0.7498942093324558
    mesh_msg_deliveries_threshold: float = 1.3176158490012766
    mesh_msg_deliveries_cap: float = 21.081853584020426
    mesh_msg_deliveries_window: float = 2000000000
    mesh_msg_deliveries_activation: float = 384000000000

    # P3b
    mesh_failure_penalty_weight: float = -0.01222
    mesh_failure_penalty_decay: float = 0.7498

    # P4
    invalid_msg_deliveries_weight: float = 0
    invalid_msg_deliveries_decay: float = 0.1

    # P5
    app_specific_weight: float = 0

    # P6
    ip_colocation_factor_weight: float = -300
    ip_colocation_factor_threshold: float = 3

    # P7
    behaviour_penalty_weight: float = -400
    behaviour_penalty_threshold: float = 6
    behaviour_penalty_decay: float = 0.1

