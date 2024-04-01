import os
import json
import argparse

from countdown_utils import *

parser = argparse.ArgumentParser(description='Control stitch_dataset params.')
parser.add_argument("--input", type=str, default=None,
                    help='a path to the dir')
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--prefix", type=str, default=None)
parser.add_argument("--filter", type=str, default=None)
parser.add_argument("--filter_value", type=float, default=None)

args = parser.parse_args()

base_files = sorted([f for f in os.listdir(args.input) if args.prefix in f])
print(base_files)
new_file = {
    "trajectories": [],
    "ratings": [],
    "reasons": [],
}

for f in base_files:
    with open(os.path.join(args.input, f), 'r') as file:
        data = json.load(file)
        new_file["trajectories"] += data["trajectories"]
        new_file["ratings"] += data["ratings"]
        new_file["reasons"] += data["reasons"]

filtered_data = {
    "trajectories": [],
    "ratings": [],
    "reasons": [],
}

if args.filter is not None:
    if args.filter == "thresh":
        assert 0 <= float(args.filter_value) <= 1, "Threshold must be between 0 and 1"
        for i, r in enumerate(new_file["ratings"]):
            if r > float(args.filter_value):
                # remove bos,eos,padding token
                new_file["trajectories"][i] = new_file["trajectories"][i].replace('<|endoftext|>', '')
                # trim till the Goal Reached token
                if "Goal Reached" in new_file["trajectories"][i]:
                    new_file["trajectories"][i] = new_file["trajectories"][i].split("Goal Reached")[0] + "Goal Reached"
                filtered_data["trajectories"].append(new_file["trajectories"][i])
                filtered_data["reasons"].append(new_file["reasons"][i])
                filtered_data["ratings"].append(new_file["ratings"][i])


    elif args.filter == "percentile":
        assert 0 <= float(args.filter_value) <= 1, "Percentile must be between 0 and 1"
        # sort dict by ratings
        sorted_ratings = sorted(new_file["ratings"])
        # get the index of the filter value percentile
        filter_index = int(len(sorted_ratings) * float(args.filter_value))
        # get the rating at the filter index
        filter_rating = sorted_ratings[filter_index]
        # filter the ratings
        for i, r in enumerate(new_file["ratings"]):
            if r >= filter_rating:
                # remove bos,eos,padding token
                new_file["trajectories"][i] = new_file["trajectories"][i].replace('<|endoftext|>', '')
                # trim till the Goal Reached token
                if "Goal Reached" in new_file["trajectories"][i]:
                    new_file["trajectories"][i] = new_file["trajectories"][i].split("Goal Reached")[0] + "Goal Reached"
                filtered_data["trajectories"].append(new_file["trajectories"][i])
                filtered_data["reasons"].append(new_file["reasons"][i])
                filtered_data["ratings"].append(new_file["ratings"][i])

    else:
        raise ValueError("Invalid filter type")

print(f"Length of new_file: {len(filtered_data['trajectories'])}")

formatted_data = []
for i in range(len(filtered_data["trajectories"])):
    target, nums = get_target_nums(filtered_data["trajectories"][i], mode="sft")
    formatted_data.append({
        "nums": nums,
        "target": target,
        "solution": ["No solution available"],
        "rating": filtered_data["ratings"][i],
        "search_path": filtered_data["trajectories"][i],
        "optimal_path": "No optimal path available",
        "heuristic": "base-0",
        "search_type": "base-0",
        })

with open(args.output, 'w') as file:
    json.dump(formatted_data, file, indent=4)
