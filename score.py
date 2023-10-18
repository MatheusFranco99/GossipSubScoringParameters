""" This file is a python implementation of https://github.com/libp2p/go-libp2p-pubsub/blob/master/score.go, with the adjustments oritented to the simulator"""

from dataclasses import dataclass, field
from typing import Dict
from score_parameters import ScoreParameters

@dataclass
class PeerStats:
    """State of a a certain peer"""

    # P1
    in_mesh: bool = True
    mesh_time: float = 0.0

    # P2
    first_message_deliveries: float = 0.0

    # P3
    mesh_message_deliveries: float = 0.0
    mesh_message_deliveries_active: bool = True
    mesh_failure_penalty: float = 0.0

    # P4
    invalid_message_deliveries: float = 0.0

    # P6
    ip_allocation: int = 0

    # P7
    behaviour_penalty: float = 0.0


@dataclass
class ScoreHistory:
    """ Stores score values with timestamps. Used to keep track of how a score evolved through time for a certain peer. """
    peer_id: int = 0
    history: Dict = field(default_factory= lambda : {key: [] for key in ['p1','p2','p3','p3b','p4','p5','p6','p7','topic','score']})

    def insert(self,key,value,time):
        """Add to history"""
        self.history[key] += [(value,time)]

    def sort(self):
        for key in self.history:
            self.history[key] = sorted(self.history[key], key = lambda x: x[1])




@dataclass
class PeerScore:
    """Score system of a certain peer. Handles score update for all its known peers."""
    params: ScoreParameters = field(default_factory = ScoreParameters)

    peer_stats: Dict[int, PeerStats] = field(default_factory = lambda: {}) # node id -> Peer Stats

    deliveries: Dict[int, float] = field(default_factory= lambda : {}) # Msg id -> timestamp

    record: Dict[int, ScoreHistory] = field(default_factory= lambda: {}) # node id -> Score History

    def add_peer(self, peer_id: int):
        """Add new peer"""
        self.peer_stats[peer_id] = PeerStats()
        self.record[peer_id] = ScoreHistory(peer_id=peer_id)

    def score(self, peer_id: int, time: float) -> float:
        """Calculate the score using the current state"""
        if peer_id not in self.peer_stats:
            raise KeyError(f"Peer id {peer_id} not in stats {self.peer_stats.keys()}")

        stats = self.peer_stats[peer_id]

        score = 0
        topic_score = 0

        p1 = 0
        if stats.in_mesh:
            p1 = time / self.params.time_in_mesh_quantum
            p1 = min(p1,self.params.time_in_mesh_cap)

        topic_score += p1 * self.params.time_in_mesh_weight
        self.record[peer_id].insert('p1',p1*self.params.time_in_mesh_weight,time)
        
        p2 = stats.first_message_deliveries
        topic_score += p2 * self.params.first_msg_deliveries_weight
        self.record[peer_id].insert('p2',p2 * self.params.first_msg_deliveries_weight,time)

        p3 = 0
        if time > self.params.mesh_msg_deliveries_activation:
            if stats.mesh_message_deliveries < self.params.mesh_msg_deliveries_threshold:
                deficit = self.params.mesh_msg_deliveries_threshold - stats.mesh_message_deliveries
                p3 = deficit * deficit

        topic_score += p3 * self.params.mesh_msg_deliveries_weight
        self.record[peer_id].insert('p3',p3 * self.params.mesh_msg_deliveries_weight,time)

        p3b = stats.mesh_failure_penalty
        topic_score += p3b * self.params.mesh_failure_penalty_weight
        self.record[peer_id].insert('p3b',p3b * self.params.mesh_failure_penalty_weight,time)

        p4 = stats.invalid_message_deliveries ** 2
        topic_score += p4 * self.params.invalid_msg_deliveries_weight
        self.record[peer_id].insert('p4',p4 * self.params.invalid_msg_deliveries_weight,time)

        score = topic_score * self.params.topic_weight

        score = min(score,self.params.topic_score_cap)
        self.record[peer_id].insert('topic',topic_score,time)

        # p5 = 0

        p6 = 0
        if stats.ip_allocation > self.params.ip_colocation_factor_threshold:
            surpluss = stats.ip_allocation - self.params.ip_colocation_factor_threshold
            p6 = surpluss ** 2

        score += p6 * self.params.ip_colocation_factor_weight
        self.record[peer_id].insert('p6',p6 * self.params.ip_colocation_factor_weight,time)
        
        p7 = 0
        if stats.behaviour_penalty > self.params.behaviour_penalty_threshold:
            excess = stats.behaviour_penalty - self.params.behaviour_penalty_threshold
            p7 = excess**2
        score += p7 * self.params.behaviour_penalty_weight
        self.record[peer_id].insert('p7',p7 * self.params.behaviour_penalty_weight,time)

        self.record[peer_id].insert('score',score,time)

        return score

    def add_penalty(self,node_id: int, count: int, timestamp: float):
        """Increase penalty stats for peer"""
        if node_id not in self.peer_stats:
            raise KeyError(f"Peer id {node_id} not in stats {self.peer_stats.keys()}")

        if self.score(node_id,timestamp) < self.params.threshold.gossip_threshold:
            return

        self.peer_stats[node_id].behaviour_penalty += count

        self.score(node_id,timestamp)

    def add_invalid(self,node_id: int, count: int, timestamp: float):
        """Increase invalid msgs counter for peer"""
        if node_id not in self.peer_stats:
            raise KeyError(f"Peer id {node_id} not in stats {self.peer_stats.keys()}")


        if self.score(node_id,timestamp) < self.params.threshold.graylist_threshold:
            return

        self.peer_stats[node_id].invalid_message_deliveries += count

        self.score(node_id,timestamp)

    def add_new_ip(self,node_id: int, count: int, timestamp: float):
        """Increase number of ips for peer"""
        if node_id not in self.peer_stats:
            raise KeyError(f"Peer id {node_id} not in stats {self.peer_stats.keys()}")

        self.peer_stats[node_id].ip_allocation += count

        self.score(node_id,timestamp)

    def add_msg(self, msg_id: int, sender: int, timestamp: float):
        """Process message"""

        if self.score(sender,timestamp) < self.params.threshold.graylist_threshold:
            return

        if msg_id in self.deliveries:
            self.add_duplicate_msg_delivery(sender,msg_id,timestamp)
        else:
            self.add_first_msg_delivery(sender,msg_id,timestamp)

        self.score(sender,timestamp)

    def add_first_msg_delivery(self,node_id: int, msg_id: int, time: float):
        """Process message first time seen"""
        if node_id not in self.peer_stats:
            raise KeyError(f"Peer id {node_id} not in stats {self.peer_stats.keys()}")

        self.deliveries[msg_id] = time

        self.peer_stats[node_id].first_message_deliveries += 1
        self.peer_stats[node_id].first_message_deliveries = min(self.peer_stats[node_id].first_message_deliveries, self.params.first_msg_deliveries_cap)

        if not self.peer_stats[node_id].in_mesh:
            return


        self.peer_stats[node_id].mesh_message_deliveries += 1
        self.peer_stats[node_id].mesh_message_deliveries = min(self.peer_stats[node_id].mesh_message_deliveries, self.params.mesh_msg_deliveries_cap)

    def add_duplicate_msg_delivery(self,node_id: int, msg_id: int, time: float):
        """Process message already seen"""
        if node_id not in self.peer_stats:
            raise KeyError(f"Peer id {node_id} not in stats {self.peer_stats.keys()}")

        if not self.peer_stats[node_id].in_mesh:
            return

        # if it's "far" from first received time, don't add to mesh message deliveries
        if time - self.deliveries[msg_id] > self.params.mesh_msg_deliveries_window:
            return

        self.peer_stats[node_id].mesh_message_deliveries += 1
        self.peer_stats[node_id].mesh_message_deliveries = min(self.peer_stats[node_id].mesh_message_deliveries, self.params.mesh_msg_deliveries_cap)




    def decays(self, timestamp: float):
        """Update peers stats by decaying the counters"""
        for node_id in self.peer_stats:
            stats = self.peer_stats[node_id]

            def update(value, decay) -> float:
                value *= decay
                if value < self.params.decay_to_zero:
                    value = 0
                return value

            stats.first_message_deliveries = update(stats.first_message_deliveries,self.params.first_msg_deliveries_decay)
            stats.mesh_message_deliveries = update(stats.mesh_message_deliveries,self.params.mesh_msg_deliveries_decay)
            stats.mesh_failure_penalty = update(stats.mesh_failure_penalty,self.params.mesh_failure_penalty_decay)
            stats.invalid_message_deliveries = update(stats.invalid_message_deliveries,self.params.invalid_msg_deliveries_decay)
            stats.behaviour_penalty = update(stats.behaviour_penalty,self.params.behaviour_penalty_decay)

            self.score(node_id,timestamp)
