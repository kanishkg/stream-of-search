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

def evaluate_accuracy(num_tests, models, seen, search_strategies, accuracies, tokenizer):
    for i in tqdm(range(num_tests)):
        # grab the target and nums
        matches = re.search("Current State: (\d+):\[(\d+), (\d+), (\d+), (\d+)\]", models["st"]["trajectories"][i])
        target, nums = int(matches.group(1)), [int(matches.group(2)), int(matches.group(3)), int(matches.group(4)),
                                               int(matches.group(5))]
        # for node alignment
        search_trees = dict()
        for name in models:
            trajectory = models[name]["trajectories"][i]
            # seen is a string "seen" or "unseen"
            if metric_fn(trajectory.split(tokenizer.bos_token)[1], mode="sft")[0] > 0:

                accuracies[seen][name].append(1.)
            else:
                accuracies[seen][name].append(0.)

            # accuracies[seen][name].append(metric_fn(trajectory.split(tokenizer.bos_token)[1], mode="sft")[0])
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
            if metric_fn(search_path)[0] > 0:
                accuracies[seen][strategy].append(1.)
            else:
                accuracies[seen][strategy].append(0.)

parser = argparse.ArgumentParser()
parser.add_argument("--all", action="store_true")
parser.add_argument("--accuracy", action="store_true")
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
# apa_seen_file = "apa_test.json"
# apa_unseen_file = "apa_test_target.json"

with open(os.path.join(trajectories_dir, ot_seen_file), "r") as json_file:
    ot_seen = json.load(json_file)
with open(os.path.join(trajectories_dir, ot_unseen_file), "r") as json_file:
    ot_unseen = json.load(json_file)
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
# with open(os.path.join(trajectories_dir, apa_seen_file), "r") as json_file:
#     apa_seen = json.load(json_file)
# with open(os.path.join(trajectories_dir, apa_unseen_file), "r") as json_file:
#     apa_unseen = json.load(json_file)
# MEASURED BY METRIC FUNCTION
accuracies = {
    "seen": {
        "ot": [],
        "st": [],
        "star1": [],
        "star2": [],
        "star3": [],
        # "apa": []
    },
    "unseen": {
        "ot": [],
        "st": [],
        "star1": [],
        "star2": [],
        "star3": [],
        # "apa": []
    }
}

for s in search_strategies:
    accuracies["seen"][s] = []
    accuracies["unseen"][s] = []

max_rating = 1152  # 4 input nums

context_len = 4096

models_seen = {"ot": ot_seen, "st": st_seen, "star1": star1_seen, "star2": star2_seen, "star3": star3_seen}
num_tests_seen = len(st_seen["trajectories"])

models_unseen = {"ot": ot_unseen, "st": st_unseen, "star1": star1_unseen, "star2": star2_unseen, "star3": star3_unseen}
num_tests_unseen = len(st_unseen["trajectories"])

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
evaluate_accuracy(num_tests_seen, models_seen, "seen", search_strategies, accuracies, tokenizer)
evaluate_accuracy(num_tests_unseen, models_unseen, "unseen", search_strategies, accuracies, tokenizer)

# store accuracy dict
with open("", "w") as f:
    json.dump(accuracies, f)    

seen_symbolic_accuracy = 0
unseen_symbolic_accuracy = 0
for strategy in search_strategies:
    if strategy in accuracies["seen"]:
        seen_symbolic_accuracy += sum(accuracies["seen"][strategy])
    if strategy in accuracies["unseen"]:
        unseen_symbolic_accuracy += sum(accuracies["unseen"][strategy])
# Weight each symbolic algorithm the same
seen_symbolic_accuracy /= (len(search_strategies) * num_tests_seen)
print(f"Average Accuracy Summary for {num_tests_seen} Seen Target Tests:")
print("Average symbolic accuracy:", seen_symbolic_accuracy)

seen_ot_accuracy = sum(accuracies["seen"]["ot"]) / num_tests_seen
seen_st_accuracy = sum(accuracies["seen"]["st"]) / num_tests_seen
seen_star1_accuracy = sum(accuracies["seen"]["star1"]) / num_tests_seen
seen_star2_accuracy = sum(accuracies["seen"]["star2"]) / num_tests_seen
seen_star3_accuracy = sum(accuracies["seen"]["star3"]) / num_tests_seen
print("Average ot accuracy:", seen_ot_accuracy)
print("Average st accuracy:", seen_st_accuracy)
print("Average star1 accuracy:", seen_star1_accuracy)
print("Average star2 accuracy:", seen_star2_accuracy)
print("Average star3 accuracy:", seen_star3_accuracy)

unseen_symbolic_accuracy /= (len(search_strategies) * num_tests_unseen)
print(f"Average Accuracy Summary for {num_tests_unseen} Unseen Target Tests:")
print("Average symbolic accuracy:", unseen_symbolic_accuracy)

unseen_ot_accuracy = sum(accuracies["seen"]["ot"]) / num_tests_seen
unseen_st_accuracy = sum(accuracies["unseen"]["st"]) / num_tests_unseen
unseen_star1_accuracy = sum(accuracies["unseen"]["star1"]) / num_tests_unseen
unseen_star2_accuracy = sum(accuracies["unseen"]["star2"]) / num_tests_unseen
unseen_star3_accuracy = sum(accuracies["unseen"]["star3"]) / num_tests_unseen
print("Average ot accuracy:", unseen_ot_accuracy)
print("Average st accuracy:", unseen_st_accuracy)
print("Average star1 accuracy:", unseen_star1_accuracy)
print("Average star2 accuracy:", unseen_star2_accuracy)
print("Average star3 accuracy:", unseen_star3_accuracy)
