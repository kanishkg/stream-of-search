import os
import json
import random
import argparse
import re

from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPTNeoForCausalLM
from datasets import load_dataset, DatasetDict, Dataset

from countdown_utils import *
from countdown_bfs import bfs
from countdown_dfs import dfs
# from util import *

from math import sqrt
from scipy.stats import norm

def evaluate(num_tests, models, seen, node_alignments, ratings, correctness, search_strategies, max_rating, tokenizer):
    for i in tqdm(range(3000,num_tests)):
        # grab the target and nums
        matches = re.search("Current State: (\d+):\[(\d+), (\d+), (\d+), (\d+)\]", models["st"]["trajectories"][i])
        target, nums = int(matches.group(1)), [int(matches.group(2)), int(matches.group(3)), int(matches.group(4)),
                                               int(matches.group(5))]
        # for node alignment
        search_trees = dict()
        for name in models:
            trajectory = models[name]["trajectories"][i]
            # seen is a string "seen" or "unseen"
            model_tree = SearchTree()
            model_tree.parse_search_trajectory(trajectory)
            # ratings[seen][name].append(model_tree.rating)
            # correctness[seen][name].append(model_tree.correctness)
            if name == "st" or name == "star3" or name == "apa":
                search_trees[name] = model_tree
            rating = metric_fn(trajectory.split(tokenizer.bos_token)[1], mode="sft")[0]
            ratings[seen][name].append(rating)
            if rating > 0:
                correctness[seen][name].append(1.)
            else:
                correctness[seen][name].append(0.)
            # accuracies[seen][name].append(metric_fn(trajectory)[0])
        for strategy in search_strategies:
            if strategy == "dfs-sum":
                search_path = dfs(target, nums, heuristic=sum_heuristic, threshold=target)
            elif strategy == "dfs-mult":
                search_path = dfs(target, nums, heuristic=mult_heuristic, threshold=target)
            elif strategy == "bfs-sum-1":
                search_path = bfs(target, nums, 1, heuristic=sum_heuristic)
            elif strategy == "bfs-sum-2":
                search_path = bfs(target, nums, 2, heuristic=sum_heuristic)
            elif strategy == "bfs-sum-3":
                search_path = bfs(target, nums, 3, heuristic=sum_heuristic)
            elif strategy == "bfs-sum-4":
                search_path = bfs(target, nums, 4, heuristic=sum_heuristic)
            elif strategy == "bfs-sum-5":
                search_path = bfs(target, nums, 5, heuristic=sum_heuristic)
            elif strategy == "bfs-mult-1":
                search_path = bfs(target, nums, 1, heuristic=mult_heuristic)
            elif strategy == "bfs-mult-2":
                search_path = bfs(target, nums, 2, heuristic=mult_heuristic)
            elif strategy == "bfs-mult-3":
                search_path = bfs(target, nums, 3, heuristic=mult_heuristic)
            elif strategy == "bfs-mult-4":
                search_path = bfs(target, nums, 4, heuristic=mult_heuristic)
            elif strategy == "bfs-mult-5":
                search_path = bfs(target, nums, 5, heuristic=mult_heuristic)
            else:
                raise ValueError(f"Search type {strategy} not supported")

            # Clip search_path to 4096 tokens (fits within one context window)
            clipped_path = ""
            token_count = 0
            for word in search_path.split(' '):
                tokens = tokenizer.tokenize(word)
                if token_count + len(tokens) < 4096:
                    clipped_path += word + ' '
                    token_count += len(tokens)
                else:
                    break
            if "Goal Reached" in search_path:
                search_path = clipped_path.split("Goal Reached", 1)[0] + "Goal Reached"
                symbol_accuracy = 1.
                symbol_rating = 1. - simple_rating(search_path) / max_rating
            else:
                symbol_accuracy = 0.
                symbol_rating = 0.
            symbol_tree = SearchTree()
            symbol_tree.parse_search_trajectory(search_path)
            if args.debug:
                print(search_path)
                print("Tree parse rating", symbol_tree.rating)
                print("Non parse rating", symbol_rating)
            # assert (symbol_tree.rating == symbol_rating)
            search_trees[strategy] = symbol_tree
            # accuracies[seen][strategy].append(metric_fn(search_path)[0])
            rating = metric_fn(search_path)[0]
            if rating > 0:
                correctness[seen][strategy].append(1.)
            else:
                correctness[seen][strategy].append(0.)
            ratings[seen][strategy].append(rating)
        # get node alignments matrix (should be symmetric along diagonal and diagonal values should all be 1)
        for s1 in search_strategies + ["st", "star3", "apa"]:
            for s2 in search_strategies + ["st", "star3", "apa"]:
                node_alignments[seen][s1][s2].append(get_node_alignment(search_trees[s1], search_trees[s2]))

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

search_strategies = ["dfs-sum", "dfs-mult", "bfs-sum-1", "bfs-sum-2", "bfs-sum-3", "bfs-sum-4", "bfs-sum-5",
                        "bfs-mult-1", "bfs-mult-2", "bfs-mult-3", "bfs-mult-4", "bfs-mult-5"]

trajectories_dir = ""
st_seen_file = "st_test.json"
st_unseen_file = "st_test_target.json"
star1_seen_file = "star1_test.json"
star1_unseen_file = "star1_test_target.json"
star2_seen_file = "star2_test.json"
star2_unseen_file = "star2_test_target.json"
star3_seen_file = "star3_test.json"
star3_unseen_file = "star3_test_target.json"
ot_seen_file = "ot_test.json"
ot_unseen_file = "ot_test_target.json"
apa_seen_file = "apa_test.json"
apa_unseen_file = "apa_test_target.json"

with open(os.path.join(trajectories_dir, st_seen_file), "r") as json_file:
    st_seen = json.load(json_file)
with open(os.path.join(trajectories_dir, st_unseen_file), "r") as json_file:
    st_unseen = json.load(json_file)
with open(os.path.join(trajectories_dir, star1_seen_file), "r") as json_file:
    star1_seen = json.load(json_file)
with open(os.path.join(trajectories_dir, star1_unseen_file), "r") as json_file:
    star1_unseen = json.load(json_file)
with open(os.path.join(trajectories_dir, star2_seen_file), "r") as json_file:
    star2_seen = json.load(json_file)
with open(os.path.join(trajectories_dir, star2_unseen_file), "r") as json_file:
    star2_unseen = json.load(json_file)
with open(os.path.join(trajectories_dir, star3_seen_file), "r") as json_file:
    star3_seen = json.load(json_file)
with open(os.path.join(trajectories_dir, star3_unseen_file), "r") as json_file:
    star3_unseen = json.load(json_file)
with open(os.path.join(trajectories_dir, apa_seen_file), "r") as json_file:
    apa_seen = json.load(json_file)
with open(os.path.join(trajectories_dir, apa_unseen_file), "r") as json_file:
    apa_unseen = json.load(json_file)


correctness = {
    "seen": {
        # search with backtracking
        "st": [],
        # average of 12 symbolic strategies
        "average_symbolic": [],
        # STaR first iteration
        "star1": [],
        # STaR second iteration
        "star2": [],
        # STaR third iteration
        "star3": [],
        "apa": []
    },
    "unseen": {
        "st": [],
        "average_symbolic": [],
        "star1": [],
        "star2": [],
        "star3": [],
        "apa": []
    }
}

ratings = {
    "seen": {
        # search with backtracking
        "st": [],
        # average of 12 symbolic strategies
        "average_symbolic": [],
        # STaR first iteration
        "star1": [],
        "star2": [],
        "star3": [],
        "apa": []
    },
    "unseen": {
        "st": [],
        "average_symbolic": [],
        "star1": [],
        "star2": [],
        "star3": [],
        "apa": []
    }
}
for s in search_strategies:
    ratings["seen"][s] = []
    ratings["unseen"][s] = []
    correctness["seen"][s] = []
    correctness["unseen"][s] = []
    # accuracies["seen"][s] = []
    # accuracies["unseen"][s] = []

max_rating = 1152 # 4 input nums

context_len = 4096

node_alignments = {"seen": dict(), "unseen": dict()}
for s1 in search_strategies + ["st", "star3", "apa"]:
    node_alignments["seen"][s1] = {}
    node_alignments["unseen"][s1] = {}
    for s2 in search_strategies + ["st", "star3", "apa"]:
        node_alignments["seen"][s1][s2] = []
        node_alignments["unseen"][s1][s2] = []

models_seen = {"st": st_seen, "star1": star1_seen, "star2": star2_seen, "star3": star3_seen, "apa": apa_seen}
num_tests_seen = len(st_seen["trajectories"])

models_unseen = {"st": st_unseen, "star1": star1_unseen, "star2": star2_unseen, "star3": star3_unseen, "apa": apa_unseen}
num_tests_unseen = len(st_unseen["trajectories"])

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
evaluate(num_tests_seen, models_seen, "seen", node_alignments, ratings, correctness, search_strategies,
            max_rating, tokenizer)
evaluate(num_tests_unseen, models_unseen, "unseen", node_alignments, ratings, correctness, search_strategies,
            max_rating, tokenizer)

with open("", "w") as f:
    json.dump(ratings, f, indent=4)

with open("", "w") as f:
    json.dump(correctness, f, indent=4)

with open("", "w") as f:
    json.dump(node_alignments, f, indent=4)
