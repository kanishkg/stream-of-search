'''
Adds optimal paths to existing countdown dataset
'''

import argparse
import random
import json
import tqdm


from countdown import CountDown

parser = argparse.ArgumentParser()
parser.add_argument("--data_file", type=str, default=None, help="Path to data file")

args = parser.parse_args()
with open(args.data_file, "r") as f:
    data = json.load(f)
    
new_data = []
for sample in tqdm.tqdm(data):
    target = sample["target"]
    nums = sample["nums"]
    solution = sample["solution"]
    cd = CountDown(target, len(nums))
    trace = cd.convert_to_path(target, nums, solution)
    sample["optimal_path"] = trace
    new_data.append(sample)

with open(args.data_file.split('.')[0]+'_converted.json', "w") as f:
    json.dump(new_data, f, indent=4)