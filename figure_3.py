# %% DESCRIPTION
"""
This script recreates the group comparison plots and statistical results
for Figure 3 in the manuscript. It focuses on the differences in 2D-ness
of neural and behavioural representations.
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
import statsmodels.api as sm # statsmodels for OLS regression

# %% CONFIGURATION
data_dir = os.path.expanduser('~/owncloud/projects/relational-representations')
data_file = os.path.join(data_dir, 'data', 'data.csv')
plot_dir = os.path.join(data_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# %% CUSTOM FUNCTIONS
def jitter(data, range=0.075):
    """Add jitter based on KDE density, scaled to a fixed range."""
    n = len(data)
    density = stats.gaussian_kde(data)(data)
    jitter = np.random.randn(n) * density
    jitter = (jitter / np.abs(jitter).max()) * range
    return jitter

def plot_mean_comparison(data_low, data_high, title, xlabel, ylabel, ylim=None, palette='flare', filename=''):
    """Boxplot with scatter and jitter for group comparison."""
    plt.figure(figsize=(2.5, 4))
    colors = sns.color_palette(palette, 2)

    sns.boxplot(data=[data_low, data_high], palette=colors, width=0.4, notch=True)

    plt.scatter(jitter(data_low, range=0.2) - 0.7, data_low, color=colors[0], alpha=0.25, s=15)
    plt.scatter(jitter(data_high, range=0.2) + 1.7, data_high, color=colors[1], alpha=0.25, s=15)

    plt.xticks([0, 1], ['Lower', 'Higher'])
    plt.xlim([-1.25, 2.25])
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir, 
            filename
        ), 
        dpi=300, 
    )

    plt.show()

def print_ttest_stats(var_higher, var_lower, label):
    """Run and print classical two-sample t-test."""
    ttest = stats.ttest_ind(var_higher, var_lower, alternative='greater', nan_policy='omit')
    print(f'{label} - Two-sample t-test: t = {ttest.statistic:.2f}, p = {ttest.pvalue:.3f}, n_higher = {len(var_higher)}, n_lower = {len(var_higher)}')

# %% LOAD DATA
data = pd.read_csv(f'{data_dir}/data/data.csv')
# filter for participants with reliable data for current task
data = data[~np.isnan(data['HC_right_relational'])]
n_subs = data['participant_id'].nunique()
print(f'Number of subjects: {n_subs}')

# %% EXCLUDE INVALID ENTRIES
# Exclude rows with missing values for any of the principal components (removed due to potentially random responses)
ix_nan = data[['PC1_neural', 'PC2_neural', 'PC1_behavioural', 'PC2_behavioural']].isna().any(axis=1)

# %% GROUP DEFINITION BASED ON GF
percentile = 33.33
gf_high = np.percentile(data['gf'], 100 - percentile)
gf_low = np.percentile(data['gf'], percentile)

ix_high = (data['gf'] >= gf_high) & ~ix_nan
ix_low = (data['gf'] <= gf_low) & ~ix_nan

print(f'Upper gf threshold = {gf_high:.2f}, n_high = {ix_high.sum()}')
print(f'Lower gf threshold = {gf_low:.2f}, n_low = {ix_low.sum()}')

# %% CALCULATE 2D-NESS
# Sum PC1 and PC2 to get overall dimensionality
data['2Dness_neural'] = data['PC1_neural'] + data['PC2_neural']
data['2Dness_behavioural'] = data['PC1_behavioural'] + data['PC2_behavioural']

# %% PLOT 2D-NESS: NEURAL REPRESENTATIONS
plot_mean_comparison(
    data['2Dness_neural'][ix_low].values,
    data['2Dness_neural'][ix_high].values,
    title='2D-ness of Neural Representations',
    xlabel='gf',
    ylabel='2D-ness',
    ylim=(-0.5, 1.25),
    palette='crest',
    filename='figure_3_2Dness_neural.svg'
)

print_ttest_stats(
    data['2Dness_neural'][ix_high].values,
    data['2Dness_neural'][ix_low].values,
    '2D-ness of Neural Representations'
)

# %% INDIVIDUAL PCs
# Can this effect be explained by PC1 or PC2 individually?

# PC1
print_ttest_stats(
    data['PC1_neural'][ix_high].values,
    data['PC1_neural'][ix_low].values,
    'Variance Explained by PC1 of Neural Representations'
)

# PC2
print_ttest_stats(
    data['PC2_neural'][ix_high].values,
    data['PC2_neural'][ix_low].values,
    'Variance Explained by PC2 of Neural Representations'
)

# %% PLOT 2D-NESS: BEHAVIOURAL REPRESENTATIONS
plot_mean_comparison(
    data['2Dness_behavioural'][ix_low].values,
    data['2Dness_behavioural'][ix_high].values,
    title='2D-ness of Behavioural Representations',
    xlabel='gf',
    ylabel='2D-ness',
    ylim=(-0.5, 1.25), 
    filename='figure_3_2Dness_behaviour.svg'
)

print_ttest_stats(
    data['2Dness_behavioural'][ix_high].values,
    data['2Dness_behavioural'][ix_low].values,
    '2D-ness of Behavioural Representations'
)

# %% REGRESSION: ACCOUNT FOR UNCERTAINTY
# Model behavioural 2D-ness controlling for old_uncertainty
X = sm.add_constant(data.loc[~ix_nan, 'old_uncertainty'])
y = data.loc[~ix_nan, '2Dness_behavioural']
model = sm.OLS(y, X).fit()
residuals = model.resid

# %% PLOT RESIDUALS
plot_mean_comparison(
    residuals[ix_low[~ix_nan]].values,
    residuals[ix_high[~ix_nan]].values,
    title='Residuals (Behavioural 2D-ness)',
    xlabel='gf',
    ylabel='Residuals',
    ylim=(-0.5, 0.5),
    filename='figure_3_2Dness_behaviour_residuals.svg'
)

print_ttest_stats(
    residuals[ix_high[~ix_nan]].values,
    residuals[ix_low[~ix_nan]].values,
    'Residuals (Behavioural 2D-ness)'
)
# %%