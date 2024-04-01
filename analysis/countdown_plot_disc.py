from analysis.plot_utils import *

###
# 0.2933 st unsolved
# 0.3643 star unsolved
# 0.3591 Apa unsolved
# 0.0265 apa diff
# 0.0135 st diff
# 0.0404 star diff

num_tests = 10000
means_unsolved = {'score':{}}
errors_unsolved = {'score':{}}

means_unsolved['score']['Stream of\nSearch'] = 0.2933
means_unsolved['score']['SoS+STaR'] = 0.3643
# means_unsolved['score']['SoS+APA'] = 0.3591
means_unsolved['score']['SoS+APA'] = 0.36



errors_unsolved['score']['Stream of\nSearch'] = binomial_confidence_interval(means_unsolved['score']['Stream of\nSearch'] * num_tests, num_tests, confidence_level=0.95)
errors_unsolved['score']['SoS+STaR'] = binomial_confidence_interval(means_unsolved['score']['SoS+STaR'] * num_tests, num_tests, confidence_level=0.95)
errors_unsolved['score']['SoS+APA'] = binomial_confidence_interval(means_unsolved['score']['SoS+APA'] * num_tests, num_tests, confidence_level=0.95)

means_diff = {'score':{}}
errors_diff = {'score':{}}

means_diff['score']['Stream of\nSearch'] = 0.0135
means_diff['score']['SoS+STaR'] = 0.0404
# means_diff['score']['SoS+APA'] = 0.0265
means_diff['score']['SoS+APA'] = 0.027

errors_diff['score']['Stream of\nSearch'] = binomial_confidence_interval(means_diff['score']['Stream of\nSearch'] * num_tests, num_tests, confidence_level=0.95)
errors_diff['score']['SoS+STaR'] = binomial_confidence_interval(means_diff['score']['SoS+STaR'] * num_tests, num_tests, confidence_level=0.95)
errors_diff['score']['SoS+APA'] = binomial_confidence_interval(means_diff['score']['SoS+APA'] * num_tests, num_tests, confidence_level=0.95)

print('plotting unsolved')
plot_bars(means_unsolved, errors_unsolved, ['Stream of\nSearch', 'SoS+STaR', 'SoS+APA'], ['Stream of\nSearch', 'SoS+STaR', 'SoS+APA'], ['score'], yticks=[0.28, 0.30, 0.32, 0.34, 0.36, 0.38], save_title="plots/unsolved", ar=(9,12), fontsize=24, ylabel="% Unsolved Problems Solved", ymin=0.28, ymax=0.38, save_format="svg", gap=1)
# plot_bars(means_diff, errors_diff, ['Stream of\nSearch', 'SoS+STaR', 'SoS+APA'], ['Stream of\nSearch', 'SoS+STaR', 'SoS+APA'], ['score'], yticks=[0, 0.01, 0.02, 0.03, 0.04, 0.05], save_title="plots/diff", ar=(9,12), fontsize=24, ylabel="% Difficult Problems Solved", ymin=0.0, ymax=0.05, save_format="svg", gap=1)
