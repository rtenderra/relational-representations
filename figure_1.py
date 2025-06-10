# %% DESCRIPTION
"""
This script recreates the task performance plots (Figure 1) from the manuscript.
It visualizes performance in two tasks:
1. Object Location Memory (OLM)
2. Object Location Arrangement (OLA)
"""

# %% IMPORTS 
# tools for data loading
import os
# tools for data manipulation
import numpy as np
import pandas as pd
# tools for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# %% CONFIG
# Define directories using cross-platform paths
data_dir = os.path.expanduser('~/owncloud/projects/relational-representations')
data_file = os.path.join(data_dir, 'data', 'data.csv')
plot_dir = os.path.join(data_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# %% LOAD DATA
data = pd.read_csv(data_file)
# filter for participants with reliable data for current task
data = data[~np.isnan(data['HC_right_relational'])]

n_subs = data['participant_id'].nunique()
print(f'Number of subjects: {n_subs}')

# %% AESTETICS
# Plot size settings
ylim = (0, 70)
yticks = [0, 20, 40, 60]
ytick_labels = ['0', '20', '40', '60']

# Boxplot settings
boxplot_size = (2, 4)
brightness = 0.35
boxplot_color = (brightness, brightness + 0.3, brightness + 0.3)

# Swarmplot settings
swarmplot_size = (3, 4)
swarm_color = 'black'

# %% FUNCTIONS
def plot_performance(data, column, label, filename_base):
    """
    Create and save a boxplot and swarmplot for a performance column.
    
    Parameters:
    - data: DataFrame containing the data
    - column: str, name of the performance column
    - label: str, label for the y-axis
    - filename_base: str, base name for saving plots
    """
    y = data[column]

    # Boxplot
    fig = plt.figure(figsize=boxplot_size)
    sns.boxplot(
        x=np.ones(len(y)), 
        y=y, 
        color=boxplot_color, 
        width=0.25,
        notch=True, 
        linewidth=0.5, 
        orient='v',
        flierprops=dict(
            marker='o', 
            markersize=2.15, 
            markerfacecolor='black', 
            markeredgecolor='black', 
            alpha=0.5
        )
    )
    plt.ylim(*ylim)
    plt.xticks([])
    plt.yticks(yticks, ytick_labels, fontsize=6)
    plt.ylabel(label, fontsize=6)
    plt.title(column)
    plt.tight_layout()
    fig.savefig(
        os.path.join(
            plot_dir,
            f'{filename_base}_boxplot.svg'
            ),
        dpi=300
    )
    plt.show()

    # Swarmplot
    fig = plt.figure(figsize=swarmplot_size)
    sns.swarmplot(
        x=np.ones(len(y)), 
        y=y,
        color=swarm_color, 
        size=2.15, 
        alpha=0.5
    )
    plt.ylim(*ylim)
    plt.xticks([])
    plt.yticks(yticks, ytick_labels, fontsize=6)
    plt.ylabel(label, fontsize=6)
    plt.title(column)
    plt.tight_layout()
    fig.savefig(
        os.path.join(
            plot_dir, 
            f'{filename_base}_swarmplot.svg'
        ), 
        dpi=300
    )
    plt.show()


def print_summary_stats(data, column, label):
    """Prints mean and standard deviation of a given column."""
    mean_val = np.round(data[column].mean(), 2)
    std_val = np.round(data[column].std(), 2)
    print(f'Mean {label}: {mean_val}')
    print(f'SD {label}: {std_val}')


# %% PLOTS AND STATS
plot_performance(data, 'olm_performance', 'Distance from target [% of max]', 'figure_1_olm_performance')
print_summary_stats(data, 'olm_performance', 'object location memory')

plot_performance(data, 'ola_performance', 'Distance from target [% of max]', 'figure_1_ola_performance')
print_summary_stats(data, 'ola_performance', 'object location arrangement')

# %%
