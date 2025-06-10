# %% DESCRIPTION
"""
This script recreates the correlation plot and statistical results for 
Figure 4 in the manuscript. It focuses on the relationship between gf 
and right hippocampal item familiarity signals.
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

# tools for statistical analysis
from scipy import stats
import pingouin as pg  # for Bayesian and partial correlations

# %% CONFIGURATION
data_dir = os.path.expanduser('~/owncloud/projects/relational-representations')
data_file = os.path.join(data_dir, 'data', 'data.csv')
plot_dir = os.path.join(data_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# %% AESTHETICS
main_color = (0, 0.3, 0.3)

# %% CUSTOM FUNCTIONS
def jitter(data, range=0.075):
    """Add jitter based on KDE density, scaled to a fixed range."""
    n = len(data)
    density = stats.gaussian_kde(data)(data)
    jitter = np.random.randn(n) * density
    jitter = (jitter / np.abs(jitter).max()) * range
    return jitter

def zoom(data, zoom=0.15):
    """Extend axis limits slightly beyond min and max."""
    data = np.asarray(data)
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    margin = (dmax - dmin) * zoom
    return dmin - margin, dmax + margin

def plot_group_distribution(y, ylabel, title, filename):
    """Plot and save a boxplot + jitter overlay."""
    plt.figure(figsize=(2.5, 4))
    plt.boxplot(y, positions=[1], widths=0.3, notch=True)
    plt.scatter(jitter(y) + 1.3, y, alpha=0.3, color=main_color)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xticks([1], ['prepost'], fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=12)
    plt.ylim(zoom(y))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation(x, y, xlabel, ylabel, title, filename):
    """Plot and save a scatter plot with regression line."""
    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, color=main_color, alpha=0.5)
    sns.regplot(x=x, y=y, color=main_color, scatter=False)
    plt.axhline(0, color='grey', linestyle='--')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=12)
    plt.xlim(zoom(x))
    plt.ylim(zoom(y))
    plt.xticks([85, 100, 115, 130, 145], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_dir, filename), 
        dpi=300)
    plt.show()

def print_ttest_stats(values, label):
    """Run and print classical one-sample t-test."""
    ttest = stats.ttest_1samp(values, 0, alternative='greater', nan_policy='omit')
    print(f'{label} - One-sample t-test: t = {ttest.statistic:.2f}, p = {ttest.pvalue:.3f}, n = {len(values)}')
   
def print_correlation_stats(x, y, label):
    """Run and print Pearson correlation."""
    corr = stats.pearsonr(x, y, alternative='greater')
    print(f'{label} - Pearson correlation: r = {corr.statistic:.2f}, p = {corr.pvalue:.3f}')

# %% LOAD DATA
data = pd.read_csv(data_file)
# filter for participants with reliable data for current task
data = data[~np.isnan(data['HC_right_nonrelational'])]
n_subs = data['participant_id'].nunique()
print(f'Number of subjects: {n_subs}')

# %% PLOT: BOX + JITTER
plot_group_distribution(
    y=data['HC_right_nonrelational'].values,
    ylabel='rHC item familiarity signal',
    title='All subjects',
    filename='figure_4_all.svg'
)
print_ttest_stats(data['HC_right_nonrelational'].values, 'All subjects')

# %% PLOT: CORRELATION GF × ITEM FAMILIARITY SIGNAL
plot_correlation(
    x=data['gf'].values,
    y=data['HC_right_nonrelational'].values,
    xlabel='gf',
    ylabel='rHC item familiarity signal',
    title='Correlation between gf and item familiarity signal',
    filename='figure_4_correlation_gf_nonrelational.svg'
)

print_correlation_stats(
    x=data['gf'].values,
    y=data['HC_right_nonrelational'].values,
    label='Item familiarity'
)
   
#%%

