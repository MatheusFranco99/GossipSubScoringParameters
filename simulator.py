from dataclasses import dataclass, field
from enum import Enum
import heapq
import json
import random
import sys

from matplotlib import pyplot as plt
plt.style.use("ggplot")
from generate_scoring_params import slots
import toml
from typing import Dict, List
from score import PeerScore

from score_parameters import ScoreParameters, Threshold

import logging


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)


@dataclass
class EventType(Enum):
    """Type of event"""
    MESSAGE = 1
    DECAY = 2
    INVALID = 3
    NEW_IP = 4
    BEHAVIOUR_PENALTY = 5

    def __repr__(self) -> str:
        return f"{self.name}"
    def __str__(self) -> str:
        if self.name == "INVALID":
            return "IN"
        elif self.name == "NEW_IP":
            return "IP"
        elif self.name == "BEHAVIOUR_PENALTY":
            return "BH"
        elif self.name == "DECAY":
            return "DC"
        elif self.name == "MESSAGE":
            return "MSG"
        else:
            return "unknown"

@dataclass
class Event:
    """Event to be handled by the simulator"""
    e_type: EventType
    msg_id: int
    timestamp: float
    sender: int

    def __lt__(self, other) -> bool:
        return self.timestamp < other.timestamp

    def __repr__(self) -> str:
        return f"{self.e_type} - {self.msg_id} - {self.sender} - {self.timestamp}"


class Behaviour:
    def __init__(self, num_per_slot: int = 10):
        self.num_per_slot = num_per_slot
    """ Sends malicious events for a given slot. `Timestamp` in events will be edited by simulator and `Sender` by the node"""
    def invalid_messages(self, slot: int):
        """Invalid messages events for slot "slot" """
        return []
    def new_ips(self, slot: int):
        """New IP connection events for slot "slot" """
        return []
    def behaviour_penalties(self, slot: int):
        """behaviour penalties events for slot "slot" """
        return []

class BadBehaviourInvalid(Behaviour):
    """ Bad Behaviour that sends invalid messages """
    def invalid_messages(self, slot: int):
        return [Event(e_type=EventType.INVALID,msg_id=-1,timestamp=0,sender=-1) for _ in range(self.num_per_slot)]
class BadBehaviourNewIP(Behaviour):
    """ Bad Behaviour that sends new IPSs """
    def new_ips(self, slot: int):
        return [Event(e_type=EventType.NEW_IP,msg_id=-1,timestamp=0,sender=-1) for _ in range(self.num_per_slot)]
class BadBehaviourPenalty(Behaviour):
    """ Bad Behaviour that sends behaviour penalties """
    def behaviour_penalties(self, slot: int):
        return [Event(e_type=EventType.BEHAVIOUR_PENALTY,msg_id=-1,timestamp=0,sender=-1) for _ in range(self.num_per_slot)]
class BadBehaviourMix1(Behaviour):
    """ Bad Behaviour that mix invalid messages and behaviour penalties """
    def invalid_messages(self, slot: int):
        return [Event(e_type=EventType.INVALID,msg_id=-1,timestamp=0,sender=-1) for _ in range(10)]
    def behaviour_penalties(self, slot: int):
        return [Event(e_type=EventType.BEHAVIOUR_PENALTY,msg_id=-1,timestamp=0,sender=-1) for _ in range(10)]
class BadBehaviourPenalty12PerEpoch(Behaviour):
    """ Bad Behaviour that sends behaviour penalties """
    def behaviour_penalties(self, slot: int):
        epochs_slot = slot%32
        if epochs_slot in [0,3,6,9,12,15,17,18,21,24,27,30]:
            return [Event(e_type=EventType.BEHAVIOUR_PENALTY,msg_id=-1,timestamp=0,sender=-1)]
        else:
            return []


@dataclass
class Node:
    """ Node in the network """
    ID: int
    D: int
    score_parameters: ScoreParameters = field(default_factory = ScoreParameters)
    time: int = 0
    mesh: set = field(default_factory = set)
    behaviour: Behaviour = field(default_factory= Behaviour)
    peer_score:PeerScore = field(default_factory= PeerScore)
    events: list = field(default_factory= lambda: [])

    def connect(self, node_ID: int):
        """ Graft new peer. Used for setup. """
        self.mesh.add(node_ID)
        self.peer_score.add_peer(node_ID)

    def set_behaviour(self,behaviour: Behaviour):
        """Set new behaviour"""
        self.behaviour = behaviour

    def decay(self, timestamp: float):
        """Update score counters due to decay event"""
        self.peer_score.decays(timestamp)

    def process_event(self, event: Event):
        """Process new event"""
        logging.debug(f"Processing event {event}")
        self.events.append(event)
        if event.e_type is EventType.MESSAGE:
            self.peer_score.add_msg(event.msg_id, event.sender, event.timestamp)
        elif event.e_type is EventType.INVALID:
            self.peer_score.add_invalid(event.sender, 1, event.timestamp)
        elif event.e_type is EventType.BEHAVIOUR_PENALTY:
            self.peer_score.add_penalty(event.sender, 1, event.timestamp)
        elif event.e_type is EventType.NEW_IP:
            self.peer_score.add_new_ip(event.sender,1, event.timestamp)

    def call_and_set_sender(self,slot: int, func: callable) -> List[Event]:
        """ Internal function used by the following 3 functions. Used for setting the `sender` field correctly in the events created by the behaviour """
        events = func(slot)
        for ev in events:
            ev.sender = self.ID
        return events

    def invalid_messages(self, slot: int) -> List[Event]:
        """ Asks behaviour for invalid events and returns them to the simulator """
        return self.call_and_set_sender(slot,self.behaviour.invalid_messages)
    def new_ips(self, slot: int) -> List[Event]:
        """ Asks behaviour for new IPs events and returns them to the simulator """
        return self.call_and_set_sender(slot,self.behaviour.new_ips)
    def behaviour_penalties(self, slot: int) -> List[Event]:
        """ Asks behaviour for behaviour penalties events and returns them to the simulator """
        return self.call_and_set_sender(slot,self.behaviour.behaviour_penalties)


class Simulator:
    """ Simulator class.
        > Does the setup of the network creating the nodes
        > Focus on the view of only one node (node 0).
        > Process a queue of events until it's empty and the simulation is done
    """
    def __init__(self, num_nodes: int = 72,
                 D: int = 8,
                 score_parameters: ScoreParameters = ScoreParameters(),
                 messages_per_slot: int = 10,
                 behaviours: Dict[int,Behaviour] = {},
                 ):
        self.num_nodes = num_nodes
        self.D = D
        self.messages_per_slot = messages_per_slot
        self.decay_interval = score_parameters.decay_interval

        self.nodes_map = {node_id: Node(ID = node_id, D = D, score_parameters = score_parameters, peer_score=PeerScore(params=score_parameters)) for node_id in range(self.num_nodes)}

        for i in range(1,1+D):
            self.nodes_map[0].connect(i)

        self.sending_peers = list(range(1,1+D))

        self.behaviours_input = behaviours
        for node_id in behaviours:
            self.nodes_map[node_id].set_behaviour(behaviours[node_id])

        self.events = []

    def setup(self, num_slots = 10):
        """ Setup the simulation:
            > add message sending events
            > add bad behaviour events
            > add decay events
        """

        msg_id = 0
        for slot_i in range(num_slots):
            for _ in range(self.messages_per_slot):
                msg_id += 1
                for peer in self.sending_peers:
                    self.add_event(Event(e_type=EventType.MESSAGE, msg_id=msg_id, sender = peer, timestamp= slots(slot_i) +  random.random()*slots(1)*0.7))

            for peer in self.sending_peers:
                invalid_messages = self.nodes_map[peer].invalid_messages(slot_i)
                new_ips = self.nodes_map[peer].new_ips(slot_i)
                behaviour_penalties = self.nodes_map[peer].behaviour_penalties(slot_i)

                for invalid_msg in invalid_messages:
                    invalid_msg.timestamp = slots(slot_i) + random.random()*slots(1)*0.7
                    self.add_event(invalid_msg)

                for behaviour_p in behaviour_penalties:
                    behaviour_p.timestamp = slots(slot_i) + random.random()*slots(1)*0.7
                    self.add_event(behaviour_p)

                for new_ip in new_ips:
                    new_ip.timestamp = slots(slot_i) + random.random()*slots(1)*0.7
                    self.add_event(new_ip)


        decay_time = self.decay_interval
        while decay_time < slots(num_slots):
            self.add_event(Event(e_type=EventType.DECAY, timestamp=decay_time, msg_id=-1, sender=-1))
            decay_time += self.decay_interval


    def run(self, num_slots = 10):
        """Run the simulation by running all events"""
        self.setup(num_slots=num_slots)

        while self.has_event():
            event = self.get_event()
            if event.e_type is EventType.DECAY:
                self.nodes_map[0].decay(event.timestamp)
            else:
                self.nodes_map[0].process_event(event)

        logging.info("Finishing the simulation. No more events.")
        return

    def add_event(self, event: Event) -> None:
        """Puts event in min heap"""
        heapq.heappush(self.events,event)


    def get_event(self) -> Event:
        """Get most recent event"""
        event = heapq.heappop(self.events)
        return event

    def has_event(self) -> bool:
        """Check if heap is not empty"""
        return len(self.events) > 0

    def report(self, num_slots: int, fname: str | None = None, title_description: str | None = None):
        """Get node's 0 score record"""

        # Score record of node 0
        record =  self.nodes_map[0].peer_score.record

        # Plot score record for each peer with defined behaviour
        for node_id in self.behaviours_input.keys():

            record[node_id].sort()

            num_plots = len(record[node_id].history)

            fig,ax = plt.subplots(ncols=3, nrows= num_plots//3 if num_plots%3 == 0 else num_plots//3 + 1, figsize = (15,8))


            # Plot each score stats
            subplot_idx = 0
            for p_type in record[node_id].history:

                times = [y for (_, y) in record[node_id].history[p_type]]
                score = [x for (x, _) in record[node_id].history[p_type]]

                ax[subplot_idx//3, subplot_idx%3].plot(times,score,label=p_type)
                ax[subplot_idx//3, subplot_idx%3].set_ylabel(p_type)

                subplot_idx += 1

            # Plot the score against the threshold values
            gossip_t = score_parameters.threshold.gossip_threshold
            publish_t = score_parameters.threshold.publish_threshoold
            gray_t = score_parameters.threshold.graylist_threshold

            times = [y for (_, y) in record[node_id].history["score"]]
            score = [x for (x, _) in record[node_id].history["score"]]

            ax[subplot_idx//3, subplot_idx%3].plot(times,score,label="score")
            ax[subplot_idx//3, subplot_idx%3].axhline(gossip_t,xmin = 0, xmax = slots(num_slots), label = "gossip threshold", linestyle = "--", color = "gray")
            ax[subplot_idx//3, subplot_idx%3].axhline(publish_t,xmin = 0, xmax = slots(num_slots), label = "publish threshold", linestyle = "--", color = "black")
            ax[subplot_idx//3, subplot_idx%3].axhline(gray_t,xmin = 0, xmax = slots(num_slots), label = "gray threshold", linestyle = "--", color = "purple")
            ax[subplot_idx//3, subplot_idx%3].legend()

            subplot_idx += 1


            # Plot the malicious events with P4, P6 and P7
            node_events = [e for e in simulator.nodes_map[0].events if e.e_type is not EventType.MESSAGE and e.e_type is not EventType.DECAY and e.sender == node_id]
            events_types = [e.e_type for e in node_events]
            events_timestamps = [e.timestamp for e in node_events]

            for event_type, event_timestamp in zip(events_types, events_timestamps):
                # Plot vertical lines at event timestamps
                plt.axvline(x=event_timestamp, color='r', linestyle='--')

                # Plot markers
                ax[subplot_idx//3, subplot_idx%3].scatter(event_timestamp, 0, color='r', marker='o')

            # Add event texts
            for i in range(len(events_types)):
                ax[subplot_idx//3, subplot_idx%3].text(x = events_timestamps[i], y = 0, s = str(events_types[i]), fontsize = 6)

            for p_type in ["p4", "p6", "p7"]:
                times = [y for (_, y) in record[node_id].history[p_type]]
                p_lst = [x for (x, _) in record[node_id].history[p_type]]
                ax[subplot_idx//3, subplot_idx%3].plot(times,p_lst,label=p_type)

            ax[subplot_idx//3, subplot_idx%3].legend()

            # Show
            if title_description is not None:
                fig.suptitle(f"Score evolution for {title_description}")
            if fname is not None:
                plt.savefig(f"images/{fname}.png")
            plt.show()


def read_toml(fname: str) -> dict:
    """Read toml file with configuration"""
    with open(fname,'r') as f:
        return toml.load(f)


if __name__ == "__main__":
    config = read_toml('ssv.toml')
    logging.info("Score Parameters:")
    logging.info(json.dumps(config,indent=4))

    score_parameters = ScoreParameters(
        threshold = Threshold(gossip_threshold=config['PEER_SCORE_PARAMS']['Thresholds']['GossipThreshold'],
                              publish_threshoold=config['PEER_SCORE_PARAMS']['Thresholds']['PublishThreshold'],
                              graylist_threshold=config['PEER_SCORE_PARAMS']['Thresholds']['GraylistThreshold'],
                              accept_px_treshold=config['PEER_SCORE_PARAMS']['Thresholds']['AcceptPXThreshold']
                              ),
        decay_interval=float(config["PEER_SCORE_PARAMS"]["DecayInterval"][:-1]),
        decay_to_zero=config["PEER_SCORE_PARAMS"]["DecayToZero"],
        retain_score=config["PEER_SCORE_PARAMS"]["RetainScore"],

        topic_weight=config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["TopicWeight"],
        topic_score_cap=config["PEER_SCORE_PARAMS"]["TopicSoreCap"],

        time_in_mesh_weight=config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["TimeInMeshWeight"],
        time_in_mesh_quantum=config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["TimeInMeshQuantum"],
        time_in_mesh_cap=config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["TimeInMeshCap"],

        first_msg_deliveries_weight = config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["FirstMessageDeliveriesWeight"],
        first_msg_deliveries_decay = config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["FirstMessageDeliveriesDecay"],
        first_msg_deliveries_cap = config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["FirstMessageDeliveriesCap"],

        mesh_msg_deliveries_weight= config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["MeshMessageDeliveriesWeight"],
        mesh_msg_deliveries_decay= config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["MeshMessageDeliveriesDecay"],
        mesh_msg_deliveries_threshold= config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["MeshMessageDeliveriesThreshold"],
        mesh_msg_deliveries_cap= config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["MeshMessageDeliveriesCap"],
        mesh_msg_deliveries_activation= float(config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["MeshMessageDeliveriesActivation"][:-1]),
        mesh_msg_deliveries_window= float(config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["MeshMessageDeliveriesWindow"][:-1]),

        mesh_failure_penalty_weight= config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["MeshFailurePenaltyWeight"],
        mesh_failure_penalty_decay= config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["MeshFailurePenaltyDecay"],

        invalid_msg_deliveries_weight= config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["InvalidMessageDeliveriesWeight"],
        invalid_msg_deliveries_decay= config["PEER_SCORE_PARAMS"]["Topics"]["subnet_0"]["InvalidMessageDeliveriesDecay"],

        app_specific_weight=config["PEER_SCORE_PARAMS"]["AppSpecificWeight"],

        ip_colocation_factor_weight=config["PEER_SCORE_PARAMS"]["IPColocationFactorWeight"],
        ip_colocation_factor_threshold=config["PEER_SCORE_PARAMS"]["IPColocationFactorThreshold"],

        behaviour_penalty_weight=config["PEER_SCORE_PARAMS"]["BehaviourPenaltyWeight"],
        behaviour_penalty_threshold=config["PEER_SCORE_PARAMS"]["BehaviourPenaltyThreshold"],
        behaviour_penalty_decay=config["PEER_SCORE_PARAMS"]["BehaviourPenaltyDecay"],
        )

    # Good behavior
    print("Good behavior results")
    simulator = Simulator(num_nodes=72,D = 8, score_parameters=score_parameters,messages_per_slot=5 * 12,
                          behaviours={
                                        1: Behaviour(),
                                      })


    num_slots = 1000
    simulator.run(num_slots=num_slots)
    simulator.report(num_slots, fname = "honest", title_description = "honest peer")


    # Invalid messages 1 per slot
    print("Invalid messages 1 per slot results")
    simulator = Simulator(num_nodes=72,D = 8, score_parameters=score_parameters,messages_per_slot=5 * 12,
                          behaviours={
                                        2: BadBehaviourInvalid(num_per_slot=1),
                                      })


    num_slots = 25
    simulator.run(num_slots=num_slots)
    simulator.report(num_slots, fname = "invalid_1_per_slot", title_description = "peer sending 1 invalid message per slot")


    # Invalid messages 10 per slot
    print("Invalid messages 10 per slot results")
    simulator = Simulator(num_nodes=72,D = 8, score_parameters=score_parameters,messages_per_slot=5 * 12,
                          behaviours={
                                        3: BadBehaviourInvalid(num_per_slot=10),
                                      })


    num_slots = 5
    simulator.run(num_slots=num_slots)
    simulator.report(num_slots, fname = "invalid_10_per_slot",  title_description = "peer sending 10 invalid message per slot")


    # IP allocation
    print("IP allocation results")
    simulator = Simulator(num_nodes=72,D = 8, score_parameters=score_parameters,messages_per_slot=5 * 12,
                          behaviours={
                                        4: BadBehaviourNewIP(num_per_slot=1),
                                      })


    num_slots = 30
    simulator.run(num_slots=num_slots)
    simulator.report(num_slots, fname = "ip_colocation_1_per_slot", title_description = "peer sending allocating 1 node with same IP per slot")


    # Behavior penalties
    print("Behavior penalties results")
    simulator = Simulator(num_nodes=72,D = 8, score_parameters=score_parameters,messages_per_slot=5 * 12,
                          behaviours={
                                        5: BadBehaviourPenalty(num_per_slot=10),
                                      })


    num_slots = 4
    simulator.run(num_slots=num_slots)
    simulator.report(num_slots, fname = "behavior_penalty_10_per_slot",  title_description = "peer doing 10 behavior penalties per slot")

