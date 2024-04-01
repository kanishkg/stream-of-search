import json
import os
import numpy as np
import pandas as pd

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

# calculate difference between alignments
diffs_star = []
for s in search_strategies:
    diff = average_node_alignments['combined']['star3'][s] - average_node_alignments['combined']['st'][s]
    diffs_star.append(diff)
    
diffs_apa = []
for s in search_strategies:
    diff = average_node_alignments['combined']['apa'][s] - average_node_alignments['combined']['st'][s]
    diffs_apa.append(diff)

# plot diffs
diff_data = {'Star3': diffs_star, 'APA': diffs_apa}
diff_df = pd.DataFrame(diff_data, index=search_strategies)

# plot the heatmap
plt.figure(figsize=(8, 6))
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
cmap = sns.color_palette("YlGnBu", as_cmap=True)
sns.heatmap(diff_df, cmap=cmap, annot=True, cbar_kws={'label': 'Difference'})
plt.title('Difference in Average Node Alignments')
plt.xlabel('Alignment Type')
plt.ylabel('Search Strategy')
plt.tight_layout()
plt.savefig("plots/alignment_diff_heatmap.svg")

