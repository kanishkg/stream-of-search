import json
import os
import numpy as np

import matplotlib.pyplot as plt
from analysis.plot_utils import *

with open("", "r") as f:
    ratings = json.load(f)

with open("", "r") as f:
    correctness = json.load(f)

with open("", "r") as f:
    node_alignments = json.load(f)

search_strategies = ["dfs-sum", "dfs-mult", "bfs-sum-1", "bfs-sum-2", "bfs-sum-3", "bfs-sum-4", "bfs-sum-5",
                        "bfs-mult-1", "bfs-mult-2", "bfs-mult-3", "bfs-mult-4", "bfs-mult-5"]

correlations_seen = {}
correlations_unseen = {}
correlations_combined = {}
search_strategies_model = search_strategies + ["st"]
for s1 in search_strategies_model:
    correlations_seen[s1] = {}
    correlations_unseen[s1] = {}
    correlations_combined[s1] = {}
    s1_arr = np.concatenate((np.array(correctness["seen"][s1]), np.array(correctness["unseen"][s1])))
    for s2 in search_strategies_model:
        correlations_seen[s1][s2] = np.corrcoef(correctness["seen"][s1], correctness["seen"][s2])[0, 1]
        correlations_unseen[s1][s2] = np.corrcoef(correctness["unseen"][s1], correctness["unseen"][s2])[0, 1]
        s2_arr = np.concatenate((np.array(correctness["seen"][s2]), np.array(correctness["unseen"][s2])))
        correlations_combined[s1][s2] = np.corrcoef(s1_arr, s2_arr)[0, 1]

print("Accuracy Correlations:")
# print(correlations_combined)
# plot correlation matrix

create_correlation_heatmap(correlations_combined, search_strategies_model, search_strategies, target="st", save_title="plots/accuracy_correlation_st_combined")

# get accuracy correlation between Star3 model and strategies with each other as a grid
correlations_star3 = {}
search_strategies_star3 = search_strategies + ["star3"]
for s1 in search_strategies_star3:
    correlations_star3[s1] = {}
    s1_arr = np.concatenate((np.array(correctness["seen"][s1]), np.array(correctness["unseen"][s1])))
    for s2 in search_strategies_star3:
        s2_arr = np.concatenate((np.array(correctness["seen"][s2]), np.array(correctness["unseen"][s2])))
        correlations_star3[s1][s2] = np.corrcoef(s1_arr, s2_arr)[0, 1]

print("Accuracy Correlations for STaR model iteration 3:")

create_correlation_heatmap(correlations_star3, search_strategies_star3, search_strategies, target="star3", save_title="plots/accuracy_correlation_star3_combined")

# apa
correlations_apa = {}
search_strategies_apa = search_strategies + ["apa"]
for s1 in search_strategies_apa:
    correlations_apa[s1] = {}
    s1_arr = np.concatenate((np.array(correctness["seen"][s1]), np.array(correctness["unseen"][s1])))
    for s2 in search_strategies_apa:
        s2_arr = np.concatenate((np.array(correctness["seen"][s2]), np.array(correctness["unseen"][s2])))
        correlations_apa[s1][s2] = np.corrcoef(s1_arr, s2_arr)[0, 1]

print("Accuracy Correlations for APA model:")
create_correlation_heatmap(correlations_apa, search_strategies_apa, search_strategies, target="apa", save_title="plots/accuracy_correlation_apa_combined")

# node alignments
average_node_alignments = {
    "seen": {},
    "unseen": {},
    "combined": {}
}
for s1, subdict in node_alignments["seen"].items():
    average_node_alignments["seen"][s1] = {}
    for s2, alignments in subdict.items():
        average_node_alignments["seen"][s1][s2] = sum(alignments) / len(alignments)

for s1, subdict in node_alignments["unseen"].items():
    average_node_alignments["unseen"][s1] = {}
    for s2, alignments in subdict.items():
        average_node_alignments["unseen"][s1][s2] = sum(alignments) / len(alignments)

for s1, subdict in node_alignments["seen"].items():
    average_node_alignments["combined"][s1] = {}
    for s2, alignments in subdict.items():
        average_node_alignments["combined"][s1][s2] = (sum(alignments) + sum(node_alignments["unseen"][s1][s2])) / len(alignments + node_alignments["unseen"][s1][s2])


# plot node alignments for sos
search_strategies_model = search_strategies + ["st"]
create_correlation_heatmap(average_node_alignments["combined"], search_strategies_model, search_strategies, target="st", save_title="plots/node_alignment_combined")

search_strategies_star3 = search_strategies + ["star3"]
create_correlation_heatmap(average_node_alignments["combined"], search_strategies_star3, search_strategies, target="star3", save_title="plots/node_alignment_star3_combined")

# search_strategies_apa = search_strategies + ["apa"]
# create_correlation_heatmap(average_node_alignments["combined"], search_strategies_apa, search_strategies, target="apa", save_title="node_alignment_apa_combined")

print("Difference in node alignment values for STaR 3 and ST Models wrt the same search strategy:")
for s in search_strategies:
    print(f"{s}: STaR-{s} = {average_node_alignments['combined']['star3'][s]}, ST-{s} = {average_node_alignments['combined']['st'][s]}. "
            f"\n  Difference = {abs(average_node_alignments['combined']['star3'][s] - average_node_alignments['combined']['st'][s])}")

print("Difference in node alignment values for APA and ST Models wrt the same search strategy:")
for s in search_strategies:
    print(f"{s}: APA-{s} = {average_node_alignments['combined']['apa'][s]}, ST-{s} = {average_node_alignments['combined']['st'][s]}. "
            f"\n  Difference = {abs(average_node_alignments['combined']['apa'][s] - average_node_alignments['combined']['st'][s])}")