import numpy as np
import pandas as pd
from src.utils.Contact_Matrix import create_contact_matrix
from src.utils.visualization.core import create_standalone_colorbar, Lx, Ly, discretize_cmaps
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec  # Add this import

def extract_behavior_distribution(df, behavior_type):
    df = df.dropna()
    
    # Extract self-reported behavior
    self_behavior = df[f'{behavior_type}_self'].astype(int)
    
    # Count occurrences of each behavior level (1-5)
    n_bins = 5
    behavior_vector = np.zeros(n_bins)
    for i in range(n_bins):
        behavior_vector[i] = np.sum(self_behavior == i+1)
    
    # Normalize to get distribution
    behavior_vector = behavior_vector / np.sum(behavior_vector)
    
    return behavior_vector
def generate_raw_matrix(df, behavior_type = "masks"):
    n_bins = 5
    df_clean = df.dropna()
    contact_matrix = np.zeros((n_bins, n_bins))
    
    # Define columns based on behavior type
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
    for i, row in df.iterrows():
        self_idx = int(row[f'{behavior_type}_self']) - 1
        for j, col in enumerate(cols):
            if pd.notna(row[col]):
                contact_matrix[self_idx, j] += row[col]
    
    return contact_matrix

def generate_contact_matrix(df, behavior_type = "masks"):
    behavior_distribution  = extract_behavior_distribution(df, behavior_type)
    raw_matrix = generate_raw_matrix(df, behavior_type)
    contact_matrix = normalize(raw_matrix, behavior_distribution)
    return contact_matrix

def plot_histogram_distribution(distribution, figsize=(Lx, Ly), save_path=None):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
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
def plot_contact_matrix(matrix,  Lx, Ly, path = None):
    fig, ax = plt.subplots(figsize=(Lx, Ly))
    ax.imshow(np.flipud(matrix), cmap="Blues", interpolation='nearest', vmin = 0, vmax = 2.6)
    
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
def bootstrap_pol_mean(df, behavior_type, n_bootstrap=1000, seed=42):

    np.random.seed(seed)
    
    # Get clean dataset with no NAs in the target behavior
    df_clean = df.dropna(subset=[f'{behavior_type}_self'])
    n_samples = len(df_clean)
    
    # Arrays to store bootstrap results
    bootstrap_means = np.zeros(n_bootstrap)
    bootstrap_polarizations = np.zeros(n_bootstrap)
    
    # Perform bootstrap
    for i in tqdm(range(n_bootstrap), desc=f"Bootstrapping {behavior_type}"):
        # Sample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_sample = df_clean.iloc[bootstrap_indices]
        
        # Get self-reported behavior and rescale from 1-5 to 0-1
        self_behavior = bootstrap_sample[f'{behavior_type}_self'].astype(int)
        rescaled_behavior = (self_behavior - 1) / 4  # Rescale from 1-5 to 0-1
        
        # Store results
        bootstrap_means[i] = np.mean(rescaled_behavior)
        bootstrap_polarizations[i] = 4 *np.var(rescaled_behavior)
    
    # Calculate summary statistics
    mean_estimate = np.mean(bootstrap_means)
    polarization_estimate = np.mean(bootstrap_polarizations)
    
    # Calculate 95% confidence intervals
    mean_ci = np.percentile(bootstrap_means, [2.5, 97.5])
    polarization_ci = np.percentile(bootstrap_polarizations, [2.5, 97.5])
    
    return {
        'bootstrap_means': bootstrap_means,
        'bootstrap_polarizations': bootstrap_polarizations,
        'mean_estimate': mean_estimate,
        'mean_ci': mean_ci,
        'polarization_estimate': polarization_estimate,
        'polarization_ci': polarization_ci
    }

def generate_raw_matrix_single_agent(df, k, behavior_type="masks"):
    """
    Generate raw contact matrix based on data from a single agent k.
    
    Args:
        df: DataFrame with survey data
        k: Index of the agent to sample
        behavior_type: Type of behavior ("masks", "testing", or "vacc")
    
    Returns:
        Raw contact matrix for single agent
    """
    n_bins = 5
    contact_matrix = np.zeros((n_bins, n_bins))
    
    # Get data for agent k only
    row = df.iloc[k]
    
    # Define columns based on behavior type
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
    
    # Agent's self-reported behavior
    self_idx = int(row[f'{behavior_type}_self']) - 1
    
    # Fill contact matrix row for this agent
    for j, col in enumerate(cols):
        if pd.notna(row[col]):
            contact_matrix[self_idx, j] = row[col]
    
    return contact_matrix


def precalculate_single_agent_raw_matrices(df, behavior_type="masks"):

    df = df.dropna()
    S_A_R_M = np.zeros((len(df), 5, 5))
    for i in tqdm(range(len(df))):
        S_A_R_M[i,:,:] = generate_raw_matrix_single_agent(df, i, behavior_type = behavior_type)
    return S_A_R_M


def normalize(M, pop_distribution) -> np.ndarray:
    """Normalize each column of the matrix to sum to 1"""
    M = M / np.sum(M, axis=1, keepdims=True)
    TC = np.sum(np.outer(pop_distribution, pop_distribution) * M) # Total contacts
    M = M / TC # Normalize to total contacts
    return M



def create_contact_matrix(n_groups, homophilic_tendency, pop_distribution) -> np.ndarray:
    positions = np.linspace(0, 1, n_groups)
    diffs = np.abs(positions[:, None] - positions[None, :])
    weights = np.exp(-homophilic_tendency * diffs)
    C = weights  * n_groups * n_groups
    
    C = normalize(C, pop_distribution)   
    return C

def synth_matrices(distribution, h_min = 1, h_max = 5, N_h = 1000):
    CM_H = np.zeros((N_h, 5, 5))
    h_list = np.linspace(h_min, h_max, N_h)
    for i in range(N_h):
        CM_H[i,:,:] = create_contact_matrix(5, h_list[i], distribution)
    return CM_H, h_list

def metric(CM_real, CM_synth):
    return np.sum(np.abs(CM_real - CM_synth))

def bootstrap_homophily(df, behavior_type, n_bootstrap=1000, seed=42):

    np.random.seed(seed)
    df = df.dropna(subset=[f'{behavior_type}_self'])
    # for each respondant we precalculate their contribution to the raw matrix (SARM)
    SARM = precalculate_single_agent_raw_matrices(df, behavior_type=behavior_type)
    
    # Arrays to store all bootstrap results
    bootstrapped_homophily = np.zeros(n_bootstrap)
    bootstrapped_means = np.zeros(n_bootstrap)
    bootstrapped_polarizations = np.zeros(n_bootstrap)

    for i in tqdm(range(n_bootstrap), desc=f"Bootstrapping {behavior_type}"):
        # Sample with replacement
        bootstrap_indices = np.random.choice(len(df), size=len(df), replace=True)
        bootstrap_sample = df.iloc[bootstrap_indices]
        
        # Sum the raw matrices for the bootstrapped sample
        raw_matrix = np.sum(SARM[bootstrap_indices, :, :], axis=0)
        
        # Obtain the behavior distribution of the selected bootstrap sample
        behavior_distribution = extract_behavior_distribution(bootstrap_sample, behavior_type)
        
        # Calculate mean and polarization for this bootstrap sample
        self_behavior = bootstrap_sample[f'{behavior_type}_self'].astype(int)
        rescaled_behavior = (self_behavior - 1) / 4  # Rescale from 1-5 to 0-1
        bootstrapped_means[i] = np.mean(rescaled_behavior)
        bootstrapped_polarizations[i] = 4 * np.var(rescaled_behavior)
        
        # Create the contact matrix
        data_matrix = normalize(raw_matrix, behavior_distribution)
        
        # Calculate homophily
        CM_synth, h_list = synth_matrices(behavior_distribution, h_min=1, h_max=5, N_h=500)
        DIFFs = np.zeros(len(h_list))
        DIFFs = np.sum(np.abs(data_matrix[None, :, :] - CM_synth), axis=(1, 2))
        
        bootstrapped_homophily[i] = h_list[np.argmin(DIFFs[:])]

    return {
        'bootstrap_homophily': bootstrapped_homophily,
        'bootstrap_means': bootstrapped_means,
        'bootstrap_polarizations': bootstrapped_polarizations,
        'homophily_estimate': np.mean(bootstrapped_homophily),
        'mean_estimate': np.mean(bootstrapped_means),
        'polarization_estimate': np.mean(bootstrapped_polarizations),
        'homophily_ci': np.percentile(bootstrapped_homophily, [2.5, 97.5]),
        'mean_ci': np.percentile(bootstrapped_means, [2.5, 97.5]),
        'polarization_ci': np.percentile(bootstrapped_polarizations, [2.5, 97.5])
    }


def plot_3x3_bootstrap_heatmaps(results_dict, bins=20, figsize=(7.09, 6)):
    """
    Create a 3x3 grid of heatmaps for bootstrap results.
    
    Rows: Pol vs Hom, Mean vs Hom, Pol vs Mean
    Columns: Masks, Testing, Vaccines
    
    Args:
        results_dict: Dictionary with keys 'masks', 'testing', 'vacc'
                     Each containing bootstrap_polarizations, bootstrap_means, bootstrap_homophily
        bins: Number of bins for 2D histograms
        figsize: Figure size (width, height)
    
    Returns:
        fig, axes: matplotlib figure object and axes dictionary
    """
    behaviors = ['masks', 'testing', 'vacc']
    behavior_labels = ['Masks', 'Testing', 'Vaccines']
    
    # Row specifications: (x_key, y_key, x_label, y_label)
    # Switched first two rows as requested
    rows = [
        ('bootstrap_means', 'bootstrap_homophily', 'Mean', 'Homophily'),
        ('bootstrap_polarizations', 'bootstrap_homophily', 'Polarization', 'Homophily'),
        ('bootstrap_polarizations', 'bootstrap_means', 'Polarization', 'Mean')
    ]
    
    # Create figure with adjusted subplot spacing to accommodate colorbar
    fig = plt.figure(figsize=figsize)
    
    # Add space between rows 1-2 and 2-3 for different y-axes
    # height_ratios: [row1, space1, row2, space2, row3]
    gs = fig.add_gridspec(5, 4, width_ratios=[1, 1, 1, 0.08], 
                         height_ratios=[1, 0.3, 1, 0.15, 1],
                         hspace=0.05, wspace=0.15)
    
    # Calculate global vmin and vmax across all data
    all_counts = []
    axes = {}
    for row, (x_key, y_key, x_label, y_label) in enumerate(rows):
        for col, (behavior, behavior_label) in enumerate(zip(behaviors, behavior_labels)):
            x_data = results_dict[behavior][x_key]
            y_data = results_dict[behavior][y_key]
            counts, _, _ = np.histogram2d(x_data, y_data, bins=bins)
            all_counts.extend(counts.flatten())
    
    vmin = 0  # Always start from 0 for counts
    vmax = max(all_counts)
    
    # Store one histogram for colorbar
    colorbar_hist = None
    
    # Set fixed axis ranges for consistency
    pol_range = [0, 1]
    mean_range = [0, 1]
    hom_range = [0, 6]
    
    for row, (x_key, y_key, x_label, y_label) in enumerate(rows):
        for col, (behavior, behavior_label) in enumerate(zip(behaviors, behavior_labels)):
            # Map logical rows to grid positions (skip spacing rows)
            grid_row = row * 2  # 0->0, 1->2, 2->4
            ax = fig.add_subplot(gs[grid_row, col])
            axes[(row, col)] = ax
            
            # Get data
            x_data = results_dict[behavior][x_key]
            y_data = results_dict[behavior][y_key]
            
            # Set consistent ranges for all plots
            if x_key == 'bootstrap_polarizations':
                x_range = pol_range
            elif x_key == 'bootstrap_means':
                x_range = mean_range
            else:  # homophily
                x_range = hom_range
                
            if y_key == 'bootstrap_homophily':
                y_range = hom_range
            elif y_key == 'bootstrap_means':
                y_range = mean_range
            else:  # polarizations
                y_range = pol_range
            
            # Create heatmap with global scale and consistent ranges
            h = ax.hist2d(x_data, y_data, bins=bins, cmap='Blues', 
                         vmin=vmin, vmax=vmax, range=[x_range, y_range])
            
            # Store for colorbar
            if colorbar_hist is None:
                colorbar_hist = h[3]
            
            # Smaller font sizes for compact layout
            fontsize_labels = 8
            fontsize_title = 9
            
            # Labels with smaller fonts
            if row == 2:  # Only bottom row gets x-labels
                ax.set_xlabel(x_label, fontsize=fontsize_labels)
            if col == 0:  # Only leftmost column gets y-labels
                ax.set_ylabel(y_label, fontsize=fontsize_labels)
            
            # Title only on top row
            if row == 0:
                ax.set_title(behavior_label, fontsize=fontsize_title, pad=8)
            
            # Tick labels and ticks - now with spacing between different y-axes
            if col == 0:  # Only leftmost column gets y-tick labels
                ax.tick_params(axis='y', which='major', labelsize=7)
                ax.locator_params(axis='y', nbins=4)
            else:
                ax.tick_params(axis='y', labelleft=False)
                
            # Show x-tick labels on rows 0 and 2 (different y-axes), hide on row 1
            if row == 0 or row == 2:  # First and third logical rows get x-tick labels
                ax.tick_params(axis='x', which='major', labelsize=7)
                ax.locator_params(axis='x', nbins=4)
            else:  # Middle row (row 1) - no x-tick labels
                ax.tick_params(axis='x', labelbottom=False)
    
    # Add single colorbar spanning all rows
    cbar_ax = fig.add_subplot(gs[:, 3])  # Spans all 5 grid rows
    cbar = plt.colorbar(colorbar_hist, cax=cbar_ax, label='Count')
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Count', fontsize=8)
    
    # Use tight_layout with padding
    plt.tight_layout(pad=0.5)
    
    return fig, axes