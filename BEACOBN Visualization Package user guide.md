# BEACON Visualization Package: User Guide

This guide demonstrates how to use the BEACON visualization package to analyze and visualize results from epidemic model simulations. The package provides tools for plotting parameter sweeps, sensitivity analysis, and time series data.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Example 1: Mask Compliance vs. Polarization](#example-1-mask-compliance-vs-polarization)
4. [Example 2: Homophily vs. Polarization](#example-2-homophily-vs-polarization)
5. [Advanced Visualizations](#advanced-visualizations)
   - [Parameter Slices](#parameter-slices)
   - [Sensitivity Analysis](#sensitivity-analysis)
   - [Trajectory Visualization](#trajectory-visualization)
6. [Customization Options](#customization-options)
7. [API Reference](#api-reference)

## Installation

The visualization package is included as part of the BEACON framework in the `src/utils/visualization` directory. No additional installation is required.

## Basic Usage

To use the visualization functions, import them from the appropriate modules:

```python
# Import basic visualization functions
from src.utils.visualization import plot_sweep_results, plot_multiple_metrics

# For advanced analysis
from src.utils.visualization import plot_parameter_slice, analyze_parameter_sensitivity
from src.utils.visualization.trajectory import plot_time_series, plot_heatmap_over_time
```

## Example 1: Mask Compliance vs. Polarization

Let's visualize how mask compliance and population polarization affect epidemic outcomes using the SIRM model:

```python
import jax.numpy as jnp
from src.models import SIRM
from src.utils.batch_sweep import sweep_two_parameters
from src.utils.visualization import plot_sweep_results, plot_multiple_metrics, print_sweep_results

# Define parameter ranges
NB = 51
NP = 51
mask_max_range = {"m": 0, "M": 1, "n": NB}  # Maximum mask compliance
pol_range = {"m": 0, "M": 1, "n": NP}       # Population polarization

# Run parameter sweep
results_MASKS = sweep_two_parameters(
    model_module=SIRM,
    param1_name="mu_max",           # parameter 1 name
    param1_range=mask_max_range,    # parameter 1 range
    param2_name="beta_params",      # parameter 2 name
    param2_range=pol_range,         # parameter 2 range
    custom_base_params={
        'beta_M': 0.25,             # Maximum susceptibility
        'recovery_rate': 0.1,       # Recovery rate
        'dT': 1,                    # Time step
        'homophilic_tendency': 0,   # No homophily
        'SPB_exponent': 1           # Linear behavior pattern
    },
    simulated_days=1000,            # Simulation duration
    population_size=100,            # Number of population compartments
    batch_size=1000                 # Process 1000 simulations at once
)

# Print a summary of the results
print_sweep_results(results_MASKS)

# Basic visualization: Total infections
fig1 = plot_sweep_results(
    results_MASKS, 
    metric="infections",
    cmap="viridis_r",
    title_prefix="Effect of Mask Compliance and Polarization on "
)

# Multiple metrics visualization
fig2 = plot_multiple_metrics(
    results_MASKS,
    metrics=["infections", "r0", "S"],
    cmaps=["viridis_r", "plasma", "Blues"],
    fig_size=(18, 6)
)

# Save the figures
fig1.savefig("mask_pol_infections.png", dpi=300, bbox_inches='tight')
fig2.savefig("mask_pol_metrics.png", dpi=300, bbox_inches='tight')
```

This code will produce two figures:
1. A heatmap showing how total infections vary with mask compliance and polarization
2. A multi-panel figure showing infections, R0, and final susceptible population

### Example Output:

The first figure (fig1) will show that higher mask compliance reduces total infections, but this effect diminishes as polarization increases. This occurs because polarization creates population segments that adopt varying degrees of mask-wearing, limiting the overall protective effect.

## Example 2: Homophily vs. Polarization

Now, let's examine how social network structure (homophily) and polarization interact:

```python
import jax.numpy as jnp
from src.models import SIRM
from src.utils.batch_sweep import sweep_two_parameters
from src.utils.visualization import plot_sweep_results, plot_multiple_metrics

# Define parameter ranges
NB = 51
NP = 51
homophilic_tendency = {"m": -10, "M": 10, "n": NB}  # Homophily parameter
pol_range = {"m": 0, "M": 1, "n": NP}               # Population polarization

# Run parameter sweep
results_HOM = sweep_two_parameters(
    model_module=SIRM,
    param1_name="beta_params",           # parameter 1 name
    param1_range=pol_range,              # parameter 1 range
    param2_name="homophilic_tendency",   # parameter 2 name
    param2_range=homophilic_tendency,    # parameter 2 range
    custom_base_params={
        'beta_M': 0.35,            # Maximum susceptibility
        'recovery_rate': 0.1,      # Recovery rate
        'dT': 1,                   # Time step
        'SPB_exponent': 1,         # Linear behavior pattern
        'mu_max': 1,               # Maximum mask compliance
        'mu_min': 0                # Minimum mask compliance
    },
    simulated_days=1000,
    population_size=100,
    batch_size=1000
)

# Visualize results with contours
fig3 = plot_multiple_metrics(
    results_HOM,
    metrics=["infections", "r0"],
    cmaps=["viridis_r", "plasma"],
    fig_size=(16, 8),
    contour_values=[
        [0.3, 0.5, 0.7, 0.9],  # Contour levels for infections
        [1.0, 2.0, 3.0, 4.0]   # Contour levels for R0
    ],
    contour_colors=[
        ["white"],              # Contour color for infections
        ["#f7f7f7", "#d9d9d9", "#bdbdbd"]  # Contour colors for R0
    ]
)

# Save the figure
fig3.savefig("homophily_pol_contours.png", dpi=300, bbox_inches='tight')
```

This example explores how homophily affects epidemic spread across polarized populations. Positive homophily means people interact more with those similar to them, while negative homophily means more interactions with dissimilar individuals.

### Example Output:

The figure will reveal how homophily can amplify or mitigate the effects of polarization on disease spread. For instance, high homophily can isolate high-risk groups, potentially reducing overall infection rates in some scenarios.

## Advanced Visualizations

### Parameter Slices

For sweeps over many parameters, you can visualize 2D slices:

```python
from src.utils.batch_sweep import sweep_n_parameters
from src.utils.visualization import plot_parameter_slice

# Define a 3D parameter sweep
param_specs = {
    "mu_max": {"m": 0, "M": 1, "n": 10},           # Mask compliance
    "beta_params": {"m": 0, "M": 1, "n": 10},      # Polarization
    "recovery_rate": {"m": 0.05, "M": 0.2, "n": 4} # Recovery rate
}

# Run 3D sweep
results_3D = sweep_n_parameters(
    model_module=SIRM,
    param_specs=param_specs,
    simulated_days=1000
)

# Visualize a 2D slice with recovery_rate fixed at 0.1
slice_plot = plot_parameter_slice(
    results_3D,
    fixed_params={"recovery_rate": 0.1},
    output_key="r0",
    cmap="plasma",
    add_contours=True
)

# For a 1D slice, fix two parameters
from src.utils.visualization import plot_1d_parameter_slice

# Visualize how mask compliance affects outcomes at fixed polarization and recovery rate
slice_1d = plot_1d_parameter_slice(
    results_3D,
    fixed_params={"beta_params": 0.5, "recovery_rate": 0.1},
    output_key="infections",
    color="darkblue"
)
```

### Sensitivity Analysis [BETA VERSION. Don't TRUST IT YET!]

Analyze which parameters have the greatest impact on outcomes:

```python
from src.utils.visualization import analyze_parameter_sensitivity

# Analyze parameter sensitivity
sensitivity = analyze_parameter_sensitivity(
    results_3D,
    output_key="infections",
    plot=True,
    fig_size=(10, 8),
    save_path="sensitivity_analysis.png"
)

# Access sensitivity metrics
for param, metrics in sensitivity.items():
    print(f"{param}: Normalized sensitivity = {metrics['normalized_sensitivity']:.4f}")
```

### Trajectory Visualization

Visualize how compartments evolve over time:

```python
from src.utils.visualization.trajectory import plot_time_series, plot_heatmap_over_time

# Run a simulation with trajectory output
results_traj = SIRM.run_simulation(
    beta_params=(2, 2),  # Symmetric distribution (no polarization)
    params={
        'beta_M': 0.25,
        'recovery_rate': 0.1,
        'mu_max': 0.8,
        'dT': 0.25
    },
    simulated_days=200,
    return_trajectory=True  # Return full trajectory instead of just final state
)

# Unpack results
trajectory, r0, h = results_traj

# Plot time series for S, I, R compartments
compartment_names = ["Susceptible", "Infected", "Recovered"]
time_points = jnp.arange(0, 200, 0.25)  # Time points (days)

ts_plot = plot_time_series(
    trajectory,
    compartment_names=compartment_names,
    time_points=time_points,
    title="SIR Model Dynamics with Masks",
    colors=["blue", "red", "green"],
    plot_sum=True  # Also plot sum of all compartments (should be 1.0)
)

# Plot a heatmap showing Infected compartment distribution over time
heatmap = plot_heatmap_over_time(
    trajectory,
    compartment_idx=1,  # Index 1 is Infected
    compartment_name="Infected",
    time_points=time_points,
    y_label="Population Group",
    contours=True,
    cmap="YlOrRd"
)
```

## Customization Options

Most visualization functions support common customization parameters:

- `fig_size`: Control the figure dimensions
- `cmap`: Choose a colormap for heatmaps
- `save_path`: Path to save the figure
- `title`/`title_prefix`: Set or modify figure titles
- `vmin`/`vmax`: Set color scale limits
- `contour_values`/`contour_colors`: Add contour lines with custom levels and colors

For publication-ready plots, some functions support a `final_params` dictionary:

```python
# Create publication-quality figure
from src.utils.visualization.core import Lx, Ly

fig_pub = plot_multiple_metrics(
    results_MASKS,
    metrics=["infections"],
    final_params={
        'Lx': Lx,  # Standard figure width for publications
        'Ly': Ly,  # Standard figure height
        'xticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'yticks': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'vmin': 0,
        'vmax': 1
    }
)
```

## API Reference

The visualization package is organized into the following modules:

- `core.py`: Core utilities including `discretize_cmaps`, `print_sweep_results`, `reshape_results`
- `heatmaps.py`: Heatmap visualizations with `plot_sweep_results`, `plot_multiple_metrics`
- `slices.py`: Parameter slice visualizations with `plot_parameter_slice`, `plot_1d_parameter_slice`
- `sensitivity.py`: Sensitivity analysis with `analyze_parameter_sensitivity`
- `trajectory.py`: Time series visualization with `plot_time_series`, `plot_parameter_trajectories`, `plot_heatmap_over_time`
- `common.py`: Internal utilities used by other modules

For detailed documentation of each function, refer to their docstrings in the module source code.