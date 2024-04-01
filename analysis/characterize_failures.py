"""
This file parses trajectories and characterizes the failures in the trajectory according to four buckets:
arithmetic, exploration (jumping logical steps), formatting and other.
"""

import argparse
import copy
import re
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from countdown_utils import *
import numpy as np
import math

def plot_metric_against_len(lengths, metric, metric_name):
    plt.scatter(lengths, metric, color='blue')
    plt.title(f"{metric_name} against Length of Trajectory")
    plt.xlabel("Length of Trajectory")
    plt.ylabel(f"{metric_name}")

def std_to_95_ci(std, n):
    """
    Translates standard deviation to a 95% confidence interval.

    Args:
        std (float): Standard deviation of the sample.
        n (int): Sample size.

    Returns:
        float: margin of error
    """
    # Calculate the standard error
    se = std / math.sqrt(n)
    
    # Calculate the margin of error for a 95% confidence interval
    margin_of_error = 1.96 * se
    
    return margin_of_error

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Characterize the failures in the trajectories')
    parser.add_argument('--data_file', type=str, help='The file containing the trajectories', default='')
    args = parser.parse_args()
    data_file = args.data_file
    with open(data_file, "r") as json_file:
        data = json.load(json_file)
    if type(data) == list:
        old_data = copy.deepcopy(data)
        data = {"trajectories": [d["search_path"] for d in old_data]}
    search_trees = []
    correctness = []
    goal_reached = []
    ratings = []
    num_total_nodes = []
    num_nodes_in_correct_path = []
    errors = []
    num_arithmetic_errors = []
    num_formatting_errors = []
    num_other_errors = []
    num_exploration_errors = []
    num_errors = []

    total_num = len(data["trajectories"])
    num_correct = 0
    ave_rating = 0
    num_goal_reached = 0
    ave_num_total_nodes_all = 0

    ave_num_total_nodes_correct = 0
    ave_num_nodes_in_correct_path = 0

    lengths_of_inputs = []
    lengths_of_correct_inputs = []

    for trajectory in tqdm(data["trajectories"], total=len(data["trajectories"])):
        lengths_of_inputs.append(len(trajectory))
        tree = SearchTree()
        result = tree.parse_search_trajectory(trajectory)
        search_trees.append(tree)
        correctness.append(tree.correctness)
        goal_reached.append(tree.goal_reached)
        ratings.append(tree.rating)
        num_total_nodes.append(tree.num_total_nodes)
        num_nodes_in_correct_path.append(tree.num_nodes_in_correct_path)
        errors.append(tree.errors)

        num_arithmetic_errors.append(len(tree.errors["arithmetic"]))
        num_formatting_errors.append(len(tree.errors["formatting"]))
        num_other_errors.append(len(tree.errors["other"]))
        num_exploration_errors.append(len(tree.errors["exploration"]))
        num_errors.append(len(tree.errors["arithmetic"]) + len(tree.errors["formatting"]) + len(tree.errors["other"]) +
                          len(tree.errors["exploration"]))
        if tree.goal_reached == 1:
            num_goal_reached += 1
        ave_num_total_nodes_all += tree.num_total_nodes
        if tree.correctness == 1:
            num_correct += 1
            ave_num_nodes_in_correct_path += tree.num_nodes_in_correct_path
            ave_num_total_nodes_correct += tree.num_total_nodes
            lengths_of_correct_inputs.append(tree.num_total_nodes)

    ave_num_arithmetic_errors = sum(num_arithmetic_errors) / total_num
    ave_num_formatting_errors = sum(num_formatting_errors) / total_num
    ave_num_other_errors = sum(num_other_errors) / total_num
    ave_num_exploration_errors = sum(num_exploration_errors) / total_num
    ave_num_total_errors = sum(num_errors) / total_num

    std_arithmetic_errors = np.std(np.array(num_arithmetic_errors))
    ci_arithmetic_errors = std_to_95_ci(std_arithmetic_errors, total_num)
    std_formatting_errors = np.std(np.array(num_formatting_errors))
    ci_formatting_errors = std_to_95_ci(std_formatting_errors, total_num)
    std_other_errors = np.std(np.array(num_other_errors))
    ci_other_errors = std_to_95_ci(std_other_errors, total_num)
    std_exploration_errors = np.std(np.array(num_exploration_errors))
    ci_exploration_errors = std_to_95_ci(std_exploration_errors, total_num)

    ave_rating = sum(ratings) / total_num
    ave_num_total_nodes_all = sum(num_total_nodes) / total_num
    std_num_total_nodes = np.std(np.array(num_total_nodes))
    ci_num_total_nodes = std_to_95_ci(std_num_total_nodes, total_num)
    if num_correct > 0:
        ave_num_nodes_in_correct_path = sum(num_nodes_in_correct_path) / num_correct
        std_num_nodes_in_correct_path = np.std(np.array(num_nodes_in_correct_path))
        ci_num_nodes_in_correct_path = std_to_95_ci(std_num_nodes_in_correct_path, num_correct)
        ave_num_total_nodes_correct = sum(num_total_nodes) / num_correct
        std_num_total_nodes_correct = np.std(np.array(num_total_nodes))
        ci_num_total_nodes_correct = std_to_95_ci(std_num_total_nodes_correct, num_correct)
    

    with open("analysis.json", "w") as json_file:
        json.dump({"trajectories": data["trajectories"], "ratings": ratings, "goal_reached": goal_reached,
                   "num_total_nodes": num_total_nodes, "num_nodes_in_correct_path": num_nodes_in_correct_path,
                   "num_arithmetic_errors": num_arithmetic_errors, "num_formatting_errors": num_formatting_errors,
                   "num_other_errors": num_other_errors, "num_exploration_errors": num_exploration_errors,}, json_file)

    print("Total num of trajectories:", total_num)
    # includes the goal reached statement
    print("Num goal reached:", num_goal_reached)
    # the path to goal reached is correct
    print("Num correct trajectories:", num_correct)
    print("Average rating:", ave_rating)
    print("Average number of total nodes in all trajectories:", ave_num_total_nodes_all, "stdev:", std_num_total_nodes, "95% CI:", ci_num_total_nodes)
    print("Average number of total nodes in correct trajectories:", ave_num_total_nodes_correct, "stdev:", std_num_total_nodes_correct, "95% CI:", ci_num_total_nodes_correct)
    print("Average number of total nodes in correct path:", ave_num_nodes_in_correct_path, "stdev:", std_num_nodes_in_correct_path, "95% CI:", ci_num_nodes_in_correct_path)

    print("Average number of arithmetic error per trajectory:", ave_num_arithmetic_errors, "stdev:", std_arithmetic_errors, "95% CI:", ci_arithmetic_errors)
    print("Average number of formatting error per trajectory:", ave_num_formatting_errors, "stdev:", std_formatting_errors, "95% CI:", ci_formatting_errors)
    print("Average number of other error per trajectory:", ave_num_other_errors, "stdev:", std_other_errors, "95% CI:", ci_other_errors)
    print("Average number of exploration error per trajectory:", ave_num_exploration_errors, "stdev:", std_exploration_errors, "95% CI:", ci_exploration_errors)

    plot_metric_against_len(lengths_of_inputs, num_total_nodes, "Num Total Nodes")
    plot_metric_against_len(lengths_of_correct_inputs, num_nodes_in_correct_path, "Num Total Nodes in Correct Path")
    plot_metric_against_len(lengths_of_inputs, num_errors, "Num Errors")