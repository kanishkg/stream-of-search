from analysis.plot_utils import *

num_tests = 10000
means_err = {'test':{}, 'test_target': {}}
errors_err = {'test':{}, 'test_target': {}}

means_err['test']['Stream of\nSearch'] = 2.247
means_err['test']['SoS+STaR'] = 1.927
means_err['test']['SoS+APA'] = 0.668
errors_err['test']['Stream of\nSearch'] = 0.097
errors_err['test']['SoS+STaR'] = 0.084
errors_err['test']['SoS+APA'] = 0.036

means_err['test_target']['Stream of\nSearch'] = 2.547
means_err['test_target']['SoS+STaR'] = 2.179
means_err['test_target']['SoS+APA'] = 0.785
errors_err['test_target']['Stream of\nSearch'] = 0.102
errors_err['test_target']['SoS+STaR'] = 0.084
errors_err['test_target']['SoS+APA'] = 0.039

means_len = {'test':{}, 'test_target': {}}
errors_len = {'test':{}, 'test_target': {}}

means_len['test']['Stream of\nSearch'] = 75.043
means_len['test']['SoS+STaR'] = 66.176
means_len['test']['SoS+APA'] = 64.733
errors_len['test']['Stream of\nSearch'] = 0.640
errors_len['test']['SoS+STaR'] = 0.641
errors_len['test']['SoS+APA'] = 0.652

means_len['test_target']['Stream of\nSearch'] = 81.747
means_len['test_target']['SoS+STaR'] = 72.252
means_len['test_target']['SoS+APA'] = 70.849
errors_len['test_target']['Stream of\nSearch'] = 0.647
errors_len['test_target']['SoS+STaR'] =  0.652
errors_len['test_target']['SoS+APA'] = 0.655

plot_bars(means_err, errors_err, ['Stream of\nSearch', 'SoS+STaR', 'SoS+APA'], ['Stream of\nSearch', 'SoS+STaR', 'SoS+APA'], ['test', 'test_target'], yticks=[0,0.5,1,1.5,2,2.5,3], save_title="plots/err", ar=(11,12), fontsize=24, ylabel="Arithmetic Errors per Trajectory", ymin=0, ymax=3, save_format="svg")
plot_bars(means_len, errors_len, ['Stream of\nSearch', 'SoS+STaR', 'SoS+APA'], ['Stream of\nSearch', 'SoS+STaR', 'SoS+APA'], ['test', 'test_target'], yticks=[60, 65, 70, 75, 80, 85], save_title="plots/len", ar=(11,12), fontsize=24, ylabel="Average Length of Solution", ymin=60, ymax=85, save_format="svg")
