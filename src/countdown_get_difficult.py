import json
import argparse
import tqdm

from countdown_utils import *
from countdown_bfs import bfs
from countdown_dfs import dfs

parser = argparse.ArgumentParser(description='Get the difficult questions')
parser.add_argument('-n','--num', type=int, default=10, help='Number of qs')
parser.add_argument('-d', '--data', type=str, default='data/val.json', help='Path to the data file')
parser.add_argument('--output', type=str, default='data/difficult.json', help='Path to the output file')


args = parser.parse_args()
with open(args.data, 'r') as f:
    data = json.load(f)

unsolved_questions = []
difficult_questions = []

search_strategies = ["dfs-sum", "dfs-mult", "bfs-sum-1", "bfs-sum-2", "bfs-sum-3", "bfs-sum-4", "bfs-sum-5",
                        "bfs-mult-1", "bfs-mult-2", "bfs-mult-3", "bfs-mult-4", "bfs-mult-5"]


for d in tqdm.tqdm(data):
    rating = d['rating']
    if rating == 0:
        unsolved_questions.append(d)
    
    solved = False
    for strategy in search_strategies:
        target = d['target']
        nums = d['nums']
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
        if 'Goal Reached' in search_path:
            solved = True
            break
    if not solved:
        difficult_questions.append(d)

    if len(difficult_questions) == args.num:
        break

# Save the difficult questions
with open(args.output + 'difficult.json', 'w') as f:
    json.dump(difficult_questions, f, indent=4)

with open(args.output + 'unsolved.json', 'w') as f:
    json.dump(unsolved_questions, f, indent=4)