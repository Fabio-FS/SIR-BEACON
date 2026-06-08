import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
Lx = 2.29
Ly = 2.16

def bootstrap_mph(df, behavior_type="masks", n_bootstrap=1000, seed=42):
    np.random.seed(seed)
    df = df.dropna(subset=[f'{behavior_type}_self'])
    n_samples = len(df)
    
    b_m = np.zeros(n_bootstrap)
    b_p = np.zeros(n_bootstrap)
    b_h = np.zeros(n_bootstrap)

    SARM = precalculate_single_agent_raw_matrices_vectorized(df, behavior_type=behavior_type)
    h_list = np.linspace(1, 5, 500)
    
    for i in tqdm(range(n_bootstrap), desc=f"Bootstrapping MPH for {behavior_type}"):
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = df.iloc[bootstrap_indices]

        behavior_distribution = extract_behavior_distribution_vectorized(bootstrap_sample, behavior_type)
        raw_matrix = np.sum(SARM[bootstrap_indices, :, :], axis=0)
        data_matrix = normalize_vectorized(raw_matrix, behavior_distribution)

        self_behavior = bootstrap_sample[f'{behavior_type}_self'].values.astype(int)
        rescaled_behavior = (self_behavior - 1) / 4
        
        CM_synth = synth_matrices_vectorized(behavior_distribution, h_list)
        #DIFFs = np.sum(np.square(data_matrix[None, :, :] - CM_synth), axis=(1, 2))


        DIFFs = np.sum(np.abs(data_matrix[None, :, :] - CM_synth), axis=(1, 2))
        
        b_m[i] = np.mean(rescaled_behavior)
        b_p[i] = 4 * np.var(rescaled_behavior)
        b_h[i] = h_list[np.argmin(DIFFs)]

    M_CI = np.percentile(b_m, [2.5, 97.5])
    P_CI = np.percentile(b_p, [2.5, 97.5])
    H_CI = np.percentile(b_h, [2.5, 97.5])

    return {
        "M_CI": [M_CI[0], np.mean(b_m), M_CI[1]],
        "P_CI": [P_CI[0], np.mean(b_p), P_CI[1]],
        "H_CI": [H_CI[0], np.mean(b_h), H_CI[1]],
        "M": b_m,
        "P": b_p,
        "H": b_h
    }


def precalculate_single_agent_raw_matrices_vectorized(df, behavior_type="masks"):
    df = df.dropna()
    n_bins = 5
    
    if behavior_type == "vacc":
        cols = [f'{behavior_type}_others0{i+1}' for i in range(n_bins)]
    else:
        cols = [
            f'{behavior_type}_others_never', 
            f'{behavior_type}_others_sometimes',
            f'{behavior_type}_others_half',
            f'{behavior_type}_others_often',
            f'{behavior_type}_others_always'
        ]
    
    self_indices = (df[f'{behavior_type}_self'].values - 1).astype(int)
    contacts = df[cols].values
    
    S_A_R_M = np.zeros((len(df), n_bins, n_bins))
    S_A_R_M[np.arange(len(df)), self_indices] = contacts
    
    return S_A_R_M


def extract_behavior_distribution_vectorized(df, behavior_type):
    self_behavior = df[f'{behavior_type}_self'].values.astype(int)
    behavior_vector = np.bincount(self_behavior - 1, minlength=5) / len(self_behavior)
    return behavior_vector


def normalize_vectorized(M, pop_distribution):
    """Single matrix normalize"""
    M = M / (M @ pop_distribution)[:, None]
    TC = np.sum(np.outer(pop_distribution, pop_distribution) * M)
    M = M / TC
    return M


def synth_matrices_vectorized(distribution, h_list):
    n_groups = 5
    positions = np.linspace(0, 1, n_groups)
    diffs = np.abs(positions[:, None] - positions[None, :])
    
    weights = np.exp(-h_list[:, None, None] * diffs[None, :, :])
    C = weights * n_groups * n_groups
    
    # Batch normalize: C is [N, 5, 5], distribution is [5]
    C = C / (C @ distribution)[:, :, None]
    TC = np.sum(np.outer(distribution, distribution)[None, :, :] * C, axis=(1, 2), keepdims=True)
    C = C / TC
    
    return C

def generate_raw_matrix(df, behavior_type="masks"):
    n_bins = 5
    df_clean = df.dropna()
    contact_matrix = np.zeros((n_bins, n_bins))
    
    if behavior_type == "vacc":
        cols = [f'{behavior_type}_others0{i+1}' for i in range(n_bins)]
    else:
        cols = [
            f'{behavior_type}_others_never', 
            f'{behavior_type}_others_sometimes',
            f'{behavior_type}_others_half',
            f'{behavior_type}_others_often',
            f'{behavior_type}_others_always'
        ]
    for i, row in df_clean.iterrows():
        self_idx = int(row[f'{behavior_type}_self']) - 1
        for j, col in enumerate(cols):
            contact_matrix[self_idx, j] += row[col]
    
    return contact_matrix


def generate_contact_matrix(df, behavior_type = "masks"):
    behavior_distribution = extract_behavior_distribution_vectorized(df, behavior_type)
    raw_matrix = generate_raw_matrix(df, behavior_type)
    contact_matrix = normalize_vectorized(raw_matrix, behavior_distribution)
    return contact_matrix


def plot_bootstrap_heatmap(R_M, R_T, R_V, bins=20, figsize=(7.09, 6)):
    """
    Create a 3x3 grid of heatmaps for bootstrap results.
    
    Args:
        R_M, R_T, R_V: Results from bootstrap_mph for masks, testing, vaccines
        bins: Number of bins for 2D histograms
        figsize: Figure size (width, height)
    
    Returns:
        fig, axes: figure and dictionary of axes with keys (row, col)
    """
    
    results = [R_M, R_T, R_V]
    behavior_labels = ['Masks', 'Testing', 'Vaccines']
    
    # Row specifications: (x_key, y_key, x_label, y_label, x_range, y_range)
    rows = [
        ('M', 'H', 'Mean', 'Homophily', [0, 1], [0, 6]),
        ('P', 'H', 'Polarization', 'Homophily', [0, 1], [0, 6]),
        ('P', 'M', 'Polarization', 'Mean', [0, 1], [0, 1])
    ]
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 4, width_ratios=[1, 1, 1, 0.08], 
                         height_ratios=[1, 0.3, 1, 0.15, 1],
                         hspace=0.05, wspace=0.15)
    
    # Calculate global vmin/vmax as percentages
    all_percentages = []
    for row_idx, (x_key, y_key, _, _, x_range, y_range) in enumerate(rows):
        for result in results:
            x_data = result[x_key]
            y_data = result[y_key]
            n_bootstrap = len(x_data)
            counts, _, _ = np.histogram2d(x_data, y_data, bins=bins, range=[x_range, y_range])
            percentages = (counts / n_bootstrap) * 100
            all_percentages.extend(percentages.flatten())
    
    vmin = 0
    vmax = max(all_percentages)
    
    colorbar_hist = None
    axes = {}
    
    for row_idx, (x_key, y_key, x_label, y_label, x_range, y_range) in enumerate(rows):
        for col, (result, label) in enumerate(zip(results, behavior_labels)):
            grid_row = row_idx * 2
            ax = fig.add_subplot(gs[grid_row, col])
            axes[(row_idx, col)] = ax
            
            x_data = result[x_key]
            y_data = result[y_key]
            n_bootstrap = len(x_data)
            
            # Create histogram with counts
            counts, xedges, yedges = np.histogram2d(x_data, y_data, bins=bins, range=[x_range, y_range])
            percentages = (counts / n_bootstrap) * 100
            
            # Plot as percentages
            h = ax.pcolormesh(xedges, yedges, percentages.T, cmap='Blues', 
                             vmin=vmin, vmax=vmax)
            
            if colorbar_hist is None:
                colorbar_hist = h
            
            if row_idx == 2:
                ax.set_xlabel(x_label, fontsize=8)
            if col == 0:
                ax.set_ylabel(y_label, fontsize=8)
            if row_idx == 0:
                ax.set_title(label, fontsize=9, pad=8)
            
            if col == 0:
                ax.tick_params(axis='y', which='major', labelsize=7)
                ax.locator_params(axis='y', nbins=4)
            else:
                ax.tick_params(axis='y', labelleft=False)
                
            if row_idx == 0 or row_idx == 2:
                ax.tick_params(axis='x', which='major', labelsize=7)
                ax.locator_params(axis='x', nbins=4)
            else:
                ax.tick_params(axis='x', labelbottom=False)
    
    cbar_ax = fig.add_subplot(gs[:, 3])
    axes['colorbar'] = cbar_ax
    cbar = plt.colorbar(colorbar_hist, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Percentage of bootstrap samples', fontsize=8)
    
    plt.tight_layout(pad=0.5)
    
    return fig, axes

def plot_histogram_distribution(distribution, figsize=(Lx, Ly), save_path=None):

    colors = ['#276419', '#7FBC41', '#F7F7F7', '#DE77AE', '#8E0152']
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(1, 6)  # Categories 1-5
    distribution = np.flipud(distribution)
    for i in range(5):
        ax.bar(x[i], distribution[i], width=1.0, color=colors[i], edgecolor='black', linewidth=1)    
    ax.set_ylim(0, 0.5)

    ax.set_xticks([1,2,3,4,5])
    ax.set_xticklabels([])

    ax.set_yticks([0,0.25,0.5])
    ax.set_yticklabels([])

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    
    # remove top and left spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)    
    # Set background to transparent
    fig.patch.set_alpha(0.0)
    fig.patch.set_visible(False)
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_contact_matrix(matrix,  Lx=Lx, Ly=Ly, path = None):
    fig, ax = plt.subplots(figsize=(Lx, Ly))
    ax.imshow(np.flipud(matrix), cmap="Blues", interpolation='nearest', vmin = 0, vmax = 6)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.patch.set_alpha(0.0)
    fig.patch.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Get value from the original matrix (not the flipped one)
            value = np.round(matrix[matrix.shape[0]-1-i, j],1)
            # Add text annotation
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", 
                   color="black" if value < 0.5*np.max(matrix) else "white")
    if path:
        fig.savefig(path, dpi=300, bbox_inches='tight')
    
    return fig, ax