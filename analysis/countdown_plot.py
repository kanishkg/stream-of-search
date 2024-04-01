import json
from analysis.plot_utils import *

search_strategies = ["dfs-sum", "dfs-mult", "bfs-sum-1", "bfs-sum-2", "bfs-sum-3", "bfs-sum-4", "bfs-sum-5",
                         "bfs-mult-1", "bfs-mult-2", "bfs-mult-3", "bfs-mult-4", "bfs-mult-5"]
search_weights = [0.25, 0.25] + [0.05] * 10

with open("", "r") as f:
    accuracies = json.load(f)   

means = {'seen': {}, 'unseen': {}}
errors = {'seen': {}, 'unseen': {}}

num_tests_seen = len(accuracies["seen"]["ot"])
num_tests_unseen = len(accuracies["unseen"]["ot"])

seen_symbolic_accuracies = [sum(accuracies["seen"][strategy])/len(accuracies["seen"][strategy]) \
    for s, strategy in enumerate(search_strategies)]
unseen_symbolic_accuracies = [sum(accuracies["unseen"][strategy])/len(accuracies["unseen"][strategy]) \
    for s, strategy in enumerate(search_strategies)]

means['seen']['symbolic'] = sum([seen_symbolic_accuracies[s] * search_weights[s] for s in range(len(search_strategies))])
means['unseen']['symbolic'] = sum([unseen_symbolic_accuracies[s] * search_weights[s] for s in range(len(search_strategies))])

means['seen']['ot'] = sum(accuracies["seen"]["ot"]) / num_tests_seen
means['seen']['st'] = sum(accuracies["seen"]["st"]) / num_tests_seen
means['seen']['star1'] = sum(accuracies["seen"]["star1"]) / num_tests_seen
means['seen']['star2'] = sum(accuracies["seen"]["star2"]) / num_tests_seen
means['seen']['star3'] = sum(accuracies["seen"]["star3"]) / num_tests_seen
# means['seen']['apa'] = sum(accuracies["seen"]["apa"]) / num_tests_seen
means['seen']['apa'] = 0.5652
means['seen']['starapa'] = 0.5654

means['unseen']['ot'] = sum(accuracies["unseen"]["ot"]) / num_tests_unseen
means['unseen']['st'] = sum(accuracies["unseen"]["st"]) / num_tests_unseen
means['unseen']['star1'] = sum(accuracies["unseen"]["star1"]) / num_tests_unseen
means['unseen']['star2'] = sum(accuracies["unseen"]["star2"]) / num_tests_unseen
means['unseen']['star3'] = sum(accuracies["unseen"]["star3"]) / num_tests_unseen
# means['unseen']['apa'] = sum(accuracies["unseen"]["apa"]) / num_tests_unseen
means['unseen']['apa'] = 0.5423
means['unseen']['starapa'] = 0.5382

errors['seen']['symbolic'] = binomial_confidence_interval(means['seen']['symbolic'] * num_tests_seen, num_tests_seen, confidence_level=0.95)
errors['unseen']['symbolic'] = binomial_confidence_interval(means['unseen']['symbolic'] * num_tests_unseen, num_tests_unseen, confidence_level=0.95)
errors['seen']['ot'] = binomial_confidence_interval(means['seen']['ot'] * num_tests_seen, num_tests_seen, confidence_level=0.95)
errors['seen']['st'] = binomial_confidence_interval(means['seen']['st'] * num_tests_seen, num_tests_seen, confidence_level=0.95)
errors['seen']['star1'] = binomial_confidence_interval(means['seen']['star1'] * num_tests_seen, num_tests_seen, confidence_level=0.95)
errors['seen']['star2'] = binomial_confidence_interval(means['seen']['star2'] * num_tests_seen, num_tests_seen, confidence_level=0.95)
errors['seen']['star3'] = binomial_confidence_interval(means['seen']['star3'] * num_tests_seen, num_tests_seen, confidence_level=0.95)
errors['seen']['apa'] = binomial_confidence_interval(means['seen']['apa'] * num_tests_seen, num_tests_seen, confidence_level=0.95)
errors['unseen']['ot'] = binomial_confidence_interval(means['unseen']['ot'] * num_tests_unseen, num_tests_unseen, confidence_level=0.95)
errors['unseen']['st'] = binomial_confidence_interval(means['unseen']['st'] * num_tests_unseen, num_tests_unseen, confidence_level=0.95)
errors['unseen']['star1'] = binomial_confidence_interval(means['unseen']['star1'] * num_tests_unseen, num_tests_unseen, confidence_level=0.95)
errors['unseen']['star2'] = binomial_confidence_interval(means['unseen']['star2'] * num_tests_unseen, num_tests_unseen, confidence_level=0.95)
errors['unseen']['star3'] = binomial_confidence_interval(means['unseen']['star3'] * num_tests_unseen, num_tests_unseen, confidence_level=0.95)
errors['unseen']['apa'] = binomial_confidence_interval(means['unseen']['apa'] * num_tests_unseen, num_tests_unseen, confidence_level=0.95)
errors['seen']['starapa'] = binomial_confidence_interval(means['seen']['starapa'] * num_tests_seen, num_tests_seen, confidence_level=0.95)
errors['unseen']['starapa'] = binomial_confidence_interval(means['unseen']['starapa'] * num_tests_unseen, num_tests_unseen, confidence_level=0.95)

print('plotting exp1')
keys_to_plot1 = ['symbolic', 'ot', 'st']
plot_bars(means, errors, x_keys=keys_to_plot1, 
    x_labels=['Symbolic\nAverage', "Optimal\nPath", "Stream\nof Search"],
    l_keys=['seen', 'unseen'], ymin=0.1, ymax=0.61, yticks=[0.2, 0.3, 0.4, 0.5, 0.6], save_title="plots/exp1",
    save_format="svg", ar=(11,12), font="Avenir", fontsize=24, ylabel="Accuracy")

print('plotting exp2')
keys_to_plot2 = ['st', 'star3', 'apa']
plot_bars(means, errors, keys_to_plot2, 
    ["Stream\nof Search", "Stream of Search\n+ STaR", "Stream of Search\n+APA"],
    ['seen', 'unseen'],ymin=0.45, ymax=0.62, yticks=[0.45, 0.5, 0.55, 0.6], save_title="plots/exp2",
    save_format="svg", ar=(11,12), font="Avenir", fontsize=24, ylabel="Accuracy")

print('plotting star')
keys_to_plot3 = ['st', 'star1', 'star2', 'star3']
plot_bars(means, errors, keys_to_plot3,
    ["Stream\nof Search", "STaR 1", "STaR 2", "STaR 3"],
    ['seen', 'unseen'], ymin=0.45, ymax=0.62, yticks=[0.45, 0.5, 0.55, 0.6], save_title="plots/star",
    save_format="svg", ar=(12,12), font="Avenir", fontsize=24, ylabel="Accuracy")


