# %% DESCRIPTION
"""
This script recreates the correlation and group comparison plots for Figure 2 in the manuscript.
It visualizes and analyzes:
1. Group differences in map-like representations based on gf (fluid intelligence)
2. Correlations between gf and right hippocampal relational representations
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
import pingouin as pg # for Bayesian and partial correlations

# %% CONFIGURATION
data_dir = os.path.expanduser('~/owncloud/projects/relational-representations')
data_file = os.path.join(data_dir, 'data', 'data.csv')
plot_dir = os.path.join(data_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# %% AESTHETICS
main_color = (0, 0.3, 0.3)

# %% CUSTOM FUNCTIONS
def jitter(data, range=0.075):
    """    Add jitter to the data based on the density of the data points."""
    # get the number of data points
    n = len(data)
    # get density
    density = stats.gaussian_kde(data)(data)
    
    # for each data point, draw from a gaussian with mean=0 and std=density
    jitter = np.random.randn(n) * density
    # scale the jitter to the range
    jitter = (jitter / abs(jitter).max()) * range

    return jitter

def zoom(data, zoom=0.15):
    """Zoom out the axis limits by a given percentage."""
    lims = (np.nanmin(data), np.nanmax(data))
    lims = (lims[0] - (lims[1]-lims[0]) * zoom, lims[1] + (lims[1]-lims[0]) * zoom)
    return lims

def plot_group_distribution(y, group_name, ylabel, filename):
    """
    Create and save a boxplot with overlaid jittered points.
    """
    plt.figure(figsize=(2.5, 4))
    plt.boxplot(
        y, 
        positions=[1], 
        widths=0.3, 
        notch=True
    )
    plt.scatter(
        jitter(y) + 1.3, 
        y, 
        alpha=0.3, 
        color=main_color
    )
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xticks([1], ['prepost'], fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(group_name, fontsize=12)
    plt.ylim(zoom(y, zoom=0.15))
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plot_dir, 
            filename), 
        dpi=300)
    plt.show()

def plot_correlation(x, y, xlabel, ylabel, filename):
    """
    Create and save a scatter plot with regression line and confidence band.
    """
    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, color=main_color, alpha=0.5)
    sns.regplot(x=x, y=y, color=main_color, scatter=False)
    plt.axhline(0, color='grey', linestyle='--')
    plt.xlim(zoom(x))
    plt.ylim(zoom(y))
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title('Correlation between gf and map-like representations', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plot_dir, 
            filename
        ), 
        dpi=300, 
    )
    plt.show()

def print_ttest_stats(group, values):
    """
    Run and print classical one-sample t-tests against 0.
    """
    ttest = stats.ttest_1samp(
        values, 
        0, 
        alternative='greater',
        nan_policy='omit'
    )

    print(f'T-test {group} gf: t={ttest.statistic:.2f}, p={ttest.pvalue:.3f}, n={len(values)}')
   

# %% LOAD DATA
data = pd.read_csv(data_file)
# filter for participants with reliable data for current task
data = data[~np.isnan(data['HC_right_relational'])]

n_subs = data['participant_id'].nunique()
print(f'Number of subjects: {n_subs}')

# %% GROUPS BY GF PERCENTILES
percentile = 100 / 3
gf_high = np.percentile(data['gf'], 100 - percentile)
gf_low = np.percentile(data['gf'], percentile)

ix_high = data['gf'] >= gf_high
ix_low = data['gf'] <= gf_low

print(f'Upper percentile gf = {gf_high:.1f}, n={ix_high.sum()}')
print(f'Lower percentile gf = {gf_low:.1f}, n={ix_low.sum()}')

# %% PLOT HIGH GF GROUP
plot_group_distribution(
    y=data['HC_right_relational'][ix_high].values,
    group_name='High gf',
    ylabel='rHC map-like representations',
    filename='figure_2_high_gf.svg'
)
print_ttest_stats('higher', data['HC_right_relational'][ix_high].values)

# %% PLOT LOW GF GROUP
print_ttest_stats('lower', data['HC_right_relational'][ix_low].values)

# %% CORRELATION: GF × MAP-LIKE REPRESENTATIONS
plot_correlation(
    x=data['gf'],
    y=data['HC_right_relational'],
    xlabel='gf',
    ylabel='HC map-like representations',
    filename='figure_2_correlation_gf_map_like.svg'
)

# Pearson correlation
corr = stats.pearsonr(data['gf'], data['HC_right_relational'], alternative='greater')
print(f'Pearson r = {corr.statistic:.2f}, p = {corr.pvalue:.3f}')

# %% PARTIAL CORRELATION
covariates = ['olm_performance', 'ola_performance', 'olp_performance']
partial = pg.partial_corr(
    data=data,
    x='gf', y='HC_right_relational',
    covar=covariates,
    method='pearson',
    alternative='greater'
)
print('\nPartial correlation (controlling for task performance variables):')
print(partial)

# %%

