from generate_scoring_params import score_parameter_decay_with_base,decay_convergence,epochs,slots,SLOTS_PER_EPOCH

decay_interval = slots(1)
decay_to_zero = 0.1

gossip_threshold = -4000

def score_parameter_decay(decay_time_in_seconds: float) -> float:
        """
        computes the decay to use such that a value of 1 decays to 0 (using the DecayToZero parameter) within the
        specified decay_time_in_seconds
        (Time wanted to decay 1 to 0 -> decay rate)
        """

        return score_parameter_decay_with_base(
            decay_time_in_seconds, decay_interval, decay_to_zero
        )


behaviour_penalty_decay = score_parameter_decay(epochs(10))
behaviour_penalty_threshold = 6  # as in lotus


# we want to ignore gossip for a peer if he has more than 10 behaviour penalties per epoch
# which weight we should put in order to reach a threshold given a certain increment rate
target_value = (
    decay_convergence(behaviour_penalty_decay, 10 / float(SLOTS_PER_EPOCH)) # -> score convergence decaying and increasing by the 2nd argument amount
    - behaviour_penalty_threshold
)
behaviour_penalty_weight = gossip_threshold / (target_value**2) # squares because libp2p squares the deviation


print(f"{gossip_threshold=}")
print(f"{behaviour_penalty_decay=}")
print(f"{behaviour_penalty_threshold=}")
print(f"{target_value=}")
print(f"{behaviour_penalty_weight=}")
print(f"{decay_convergence(behaviour_penalty_decay, 10 / float(SLOTS_PER_EPOCH))=}")

p7 = 0
p7_11 = 0
x = []
scores = []
scores_11 = []
p7_lst = []
p7_lst_11 = []
x_max = 1000

print_only_once = True

for i in range(x_max):
    p7 = p7 + 10 / float(SLOTS_PER_EPOCH)
    p7 = p7 * behaviour_penalty_decay
    p7_11 = (p7_11 + 11/ float(SLOTS_PER_EPOCH)) * behaviour_penalty_decay
    x += [i]
    p7_lst += [p7]
    p7_lst_11 += [p7_11]
    scores += [pow(max(p7-behaviour_penalty_threshold,0),2) * behaviour_penalty_weight]
    scores_11 += [pow(max(p7_11-behaviour_penalty_threshold,0),2) * behaviour_penalty_weight]


import matplotlib.pyplot as plt
plt.style.use("ggplot")

# plt.plot(x,p7_lst)
plt.plot(x,scores, linestyle = "--",label="P7 score (with 10 messages)")
plt.plot(x,scores_11, linestyle = "--",label="P7 score (with 11 messages)")
plt.axhline(gossip_threshold,0,x_max,linestyle="--",color = "gray",label="GossipThreshold")
plt.xlabel("Slots")
plt.legend()
plt.show()
