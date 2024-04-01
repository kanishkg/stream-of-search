from math import sqrt
from scipy.stats import norm
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def change_saturation(rgb, change=0.6):
    """
    Changes the saturation for the plotted bars, rgb is from sns.colorblind (used change=0.6 in paper)
    """
    hsv = colorsys.rgb_to_hsv(rgb[0], rgb[1], rgb[2])
    saturation = max(0, min(hsv[1] * change, 1))
    return colorsys.hsv_to_rgb(hsv[0], saturation, hsv[2])



def binomial_confidence_interval(successes, trials, confidence_level=0.95):
    p = successes / trials
    z = norm.ppf((1 + confidence_level) / 2)

    interval = z * sqrt((p * (1 - p)) / trials)
    return interval
    # return (lower_bound, upper_bound)


def plot_bars(means, sems, x_keys, x_labels, l_keys, yticks,
            save_title="acc", save_format="png", ar=(14,6), barWidth=1.2, gap=1.3,
            font="Avenir", fontsize=24, ylabel="Accuracy", ymin=0, ymax=1):
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.despine(left=True, bottom=False)
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    plt.rcParams["font.size"] = fontsize
    keys = x_keys
    labels = x_labels
    assert len(keys) == len(labels)

    # Initialize figure
    plt.figure(figsize=ar)

    # Determine the number of model methods for dynamic positioning
    num_lkeys = len(l_keys)
    num_categories = len(keys)

    # Calculate positions with increased gap
    positions = np.arange(len(labels)) * (barWidth * num_lkeys + gap)

    # Adjust the positions for x-ticks to be at the center of each group of bars
    adjusted_positions = positions + (barWidth * (num_lkeys-1)) / 2

    # Color palette
    colorblind_palette = sns.color_palette('colorblind', n_colors=num_categories)
    # colorblind_palette = sns.color_palette('colorblind', n_colors=2)
        
    colorblind_palette = [change_saturation(color, 0.6) for color in colorblind_palette]
    # Plot bars
    for i, legend_key in enumerate(l_keys):
        print(legend_key)
        mean = [means[legend_key][key] for key in keys if legend_key in means]
        interval = [sems[legend_key][key] for key in keys if legend_key in sems]
        print(mean, interval)
        print(positions + i*barWidth)
        # plt.bar(positions + i*barWidth, mean, yerr=interval, width=barWidth, color=colorblind_palette[i+1])
        plt.bar(positions + i*barWidth, mean, yerr=interval, width=barWidth, color=colorblind_palette[i])



    # Adjusting the plot
    plt.ylabel(ylabel)
    plt.xticks(adjusted_positions, labels, rotation=0)
    sns.despine(left=True, bottom=True)
    plt.grid(True, which='major', axis='y', linestyle='-', color='lightgrey', alpha=0.5)
    plt.tick_params(axis='y', length=0)
    plt.tight_layout()
    plt.ylim(ymin, ymax)
    plt.yticks(yticks)
    plt.savefig(save_title+'.'+save_format)


def create_correlation_heatmap(correlations_combined, search_strategies_model, search_strategies, target="SoS",
                              save_format="svg", save_title="accuracy_correlation_st_combined"):
    fig, ax = plt.subplots(figsize=(10, 8))  # Adjust the figure size as needed

    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # Create a DataFrame for the correlation matrix
    corr_matrix = pd.DataFrame([[correlations_combined[s1][s2] for s1 in search_strategies_model] for s2 in search_strategies_model],
                                columns=search_strategies + [target], index=search_strategies + [target])

    # Create the heatmap using Seaborn
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, cbar_kws={'label': 'Correlation'}, ax=ax)

    ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels by 45 degrees


    fig.tight_layout()
    plt.savefig(f"{save_title}.{save_format}")
    plt.close()