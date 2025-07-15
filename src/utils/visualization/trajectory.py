# File: src/utils/visualization/trajectory.py
"""
Trajectory visualization functions for epidemic models
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

from ..distributions import pol_mean_to_ab

def plot_time_series(
    trajectory: Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray]],
    compartment_names: Optional[List[str]] = None,
    time_points: Optional[np.ndarray] = None,
    title: str = "Compartment Dynamics",
    colors: Optional[List[str]] = None,
    fig_size: Tuple[int, int] = (10, 6),
    y_label: str = "Population Fraction",
    x_label: str = "Time",
    legend_loc: str = "best",
    linestyles: Optional[List[str]] = None,
    grid: bool = False,
    alpha: float = 1.0,
    plot_sum: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series data from model trajectories
    
    Args:
        trajectory: Either a tuple of arrays (one per compartment) or a dictionary
                   mapping compartment names to time series arrays
        compartment_names: List of compartment names (used if trajectory is a tuple)
        time_points: Time points for the x-axis (default: use indices)
        title: Plot title
        colors: List of colors for each compartment (default: use matplotlib defaults)
        fig_size: Figure size (width, height)
        y_label: Label for y-axis
        x_label: Label for x-axis
        legend_loc: Location for the legend
        linestyles: List of linestyles for each compartment
        grid: Whether to add grid lines
        alpha: Transparency for the lines
        plot_sum: Whether to also plot the sum of all compartments (should be 1.0)
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    # Handle different input formats
    if isinstance(trajectory, tuple):
        # Convert tuple of arrays to dictionary with provided names
        if compartment_names is None:
            # Generate default names if none provided
            compartment_names = [f"Compartment {i+1}" for i in range(len(trajectory))]
        elif len(compartment_names) != len(trajectory):
            raise ValueError("Number of compartment names must match number of trajectory arrays")
            
        data = {}
        for name, traj in zip(compartment_names, trajectory):
            # Extract population totals for each time step for plotting
            if len(traj.shape) > 1:
                # If we have a 2D array (time, population_groups), sum across population groups
                data[name] = np.array(jnp.sum(traj, axis=1))
            else:
                # If already 1D (just time), use as is
                data[name] = np.array(traj)
    elif isinstance(trajectory, dict):
        # Use the dictionary directly
        data = {}
        for name, traj in trajectory.items():
            # Extract population totals for each time step for plotting
            if len(traj.shape) > 1:
                # If we have a 2D array (time, population_groups), sum across population groups
                data[name] = np.array(jnp.sum(traj, axis=1))
            else:
                # If already 1D (just time), use as is
                data[name] = np.array(traj)
        compartment_names = list(data.keys())
    else:
        raise ValueError("Trajectory must be either a tuple of arrays or a dictionary")
    
    # Create time points if not provided
    if time_points is None:
        # Get length of first trajectory
        first_traj = list(data.values())[0]
        time_points = np.arange(len(first_traj))
    else:
        # Ensure time_points has the right length
        first_traj = list(data.values())[0]
        if len(time_points) != len(first_traj):
            # If lengths don't match, create appropriate time points
            if len(time_points) == 1 + len(first_traj):
                # Special case: time_points includes t=0 and data starts at t=1
                time_points = time_points[:-1]
            else:
                # General case: create evenly spaced time points
                time_points = np.linspace(np.min(time_points), np.max(time_points), len(first_traj))
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Setup default colors if not provided
    if colors is None:
        # Use default matplotlib color cycle
        colors = [f"C{i}" for i in range(len(compartment_names))]
    elif len(colors) < len(compartment_names):
        # Extend colors if not enough provided
        colors = colors + [f"C{i}" for i in range(len(colors), len(compartment_names))]
    
    # Setup default linestyles if not provided
    if linestyles is None:
        linestyles = ['-'] * len(compartment_names)
    elif len(linestyles) < len(compartment_names):
        # Extend linestyles if not enough provided
        linestyles = linestyles + ['-'] * (len(compartment_names) - len(linestyles))
    
    # Plot each compartment
    for i, name in enumerate(compartment_names):
        ax.plot(
            time_points,
            data[name],
            color=colors[i],
            linestyle=linestyles[i],
            label=name,
            alpha=alpha
        )
    
    # Plot sum of all compartments if requested
    if plot_sum:
        sum_data = np.zeros_like(time_points, dtype=float)
        for traj in data.values():
            sum_data += traj
        
        ax.plot(
            time_points,
            sum_data,
            color='black',
            linestyle='--',
            label='Sum',
            alpha=0.7
        )
    
    # Add labels, title, legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    
    # Add grid if requested
    if grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits to accommodate all data
    ax.set_ylim(0, max(1.0, np.max([np.max(traj) for traj in data.values()]) * 1.05))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
def plot_heatmap_over_time(
    trajectory: Tuple[np.ndarray, ...],
    compartment_idx: int = 0,
    compartment_name: Optional[str] = None,
    time_points: Optional[np.ndarray] = None,
    population_axis: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    fig_size: Tuple[int, int] = (10, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    contours: bool = False,
    contour_levels: Optional[List[float]] = None,
    contour_colors: str = "white",
    x_label: str = "Time",
    y_label: str = "Population Index",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a heatmap of a compartment's value over time and population index
    
    Args:
        trajectory: Tuple of trajectory arrays (one per compartment)
        compartment_idx: Index of the compartment to plot
        compartment_name: Name of the compartment (None for default)
        time_points: Array of time points (None for default)
        population_axis: Array of population indices or values (None for default)
        title: Plot title (None for auto-generated)
        cmap: Colormap to use
        fig_size: Figure size (width, height)
        vmin, vmax: Color scale limits (None for auto)
        contours: Whether to add contour lines
        contour_levels: Levels for contour lines (None for auto)
        contour_colors: Colors for contour lines
        x_label: Label for x-axis
        y_label: Label for y-axis
        save_path: Path to save figure (None to not save)
        
    Returns:
        Matplotlib figure
    """
    # Get time series data for the selected compartment
    if compartment_idx >= len(trajectory):
        raise ValueError(f"Compartment index {compartment_idx} out of range (max: {len(trajectory)-1})")
    
    compartment_data = np.array(trajectory[compartment_idx])
    
    # Generate default compartment name if not provided
    if compartment_name is None:
        compartment_name = f"Compartment {compartment_idx+1}"
    
    # Generate default time points if not provided
    if time_points is None:
        time_points = np.arange(compartment_data.shape[0])
    else:
        # Ensure time_points has the right length
        if len(time_points) != compartment_data.shape[0]:
            # If lengths don't match, create appropriate time points
            if len(time_points) == 1 + compartment_data.shape[0]:
                # Special case: time_points includes t=0 and data starts at t=1
                time_points = time_points[:-1]
            else:
                # General case: create evenly spaced time points
                time_points = np.linspace(np.min(time_points), np.max(time_points), compartment_data.shape[0])
    
    # Generate default population axis if not provided
    if population_axis is None:
        population_axis = np.arange(compartment_data.shape[1])
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Create heatmap
    im = ax.pcolormesh(
        time_points,
        population_axis,
        compartment_data.T,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{compartment_name} Value")
    
    # Add contours if requested
    if contours:
        if contour_levels is None:
            # Auto-generate levels
            min_val = np.nanmin(compartment_data)
            max_val = np.nanmax(compartment_data)
            contour_levels = np.linspace(min_val, max_val, 5)[1:-1]  # Skip min and max
        
        # Generate grid for contour (needed because pcolormesh uses cell edges)
        X, Y = np.meshgrid(time_points, population_axis)
        
        # Make sure X and Y have the same shape as compartment_data.T
        if X.shape != compartment_data.T.shape:
            # Create new meshgrid with correct dimensions
            x_new = np.linspace(time_points[0], time_points[-1], compartment_data.shape[0])
            y_new = np.linspace(population_axis[0], population_axis[-1], compartment_data.shape[1])
            X, Y = np.meshgrid(x_new, y_new)
        
        contour = ax.contour(
            X, Y, compartment_data.T, 
            levels=contour_levels,
            colors=contour_colors,
            linewidths=1.0,
            alpha=0.8
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if title is None:
        title = f"{compartment_name} Evolution Over Time"
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_parameter_trajectories(
    trajectories: Dict[str, Tuple[np.ndarray, ...]],
    param_values: List[float],
    param_name: str,
    compartments: Optional[List[int]] = None,
    compartment_names: Optional[List[str]] = None,
    time_points: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    colormap: str = "viridis",
    fig_size: Tuple[int, int] = (12, 8),
    grid: bool = True,
    alpha: float = 0.7,
    normalize: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot trajectories for multiple parameter values
    
    Args:
        trajectories: Dictionary mapping parameter values (as strings) to trajectory tuples
        param_values: List of parameter values corresponding to the dictionary keys
        param_name: Name of the parameter being varied
        compartments: List of compartment indices to plot (default: plot all)
        compartment_names: List of compartment names
        time_points: Time points for the x-axis (default: use indices)
        title: Plot title (None for auto-generated)
        colormap: Colormap to use for parameter values
        fig_size: Figure size (width, height)
        grid: Whether to add grid lines
        alpha: Transparency for the lines
        normalize: Whether to normalize trajectories to their max value
        save_path: Path to save the figure (if None, figure is not saved)
        
    Returns:
        Matplotlib figure object
    """
    # Convert param_values to strings to match dictionary keys
    param_keys = [str(val) for val in param_values]
    
    # Determine number of compartments from the first trajectory
    first_traj = trajectories[param_keys[0]]
    n_compartments = len(first_traj)
    
    # If compartments not specified, plot all
    if compartments is None:
        compartments = list(range(n_compartments))
    
    # If compartment_names not provided, use defaults
    if compartment_names is None:
        compartment_names = [f"Compartment {i+1}" for i in range(n_compartments)]
    elif len(compartment_names) < n_compartments:
        # Extend names if not enough provided
        compartment_names = compartment_names + [f"Compartment {i+1}" for i in range(len(compartment_names), n_compartments)]
    
    # Create time points if not provided
    if time_points is None:
        # Get length of the first trajectory's first compartment
        time_points = np.arange(len(first_traj[0]))
    
    # Create figure with one subplot per selected compartment
    n_plots = len(compartments)
    fig, axes = plt.subplots(n_plots, 1, figsize=fig_size, sharex=True)
    if n_plots == 1:
        axes = [axes]  # Make iterable for single compartment case
    
    # Create colormap
    cmap = plt.get_cmap(colormap)
    norm = plt.Normalize(min(param_values), max(param_values))
    
    # Plot each parameter value in each compartment subplot
    for i, compartment_idx in enumerate(compartments):
        ax = axes[i]
        
        # Get compartment name
        compartment_name = compartment_names[compartment_idx]
        
        # Plot trajectory for each parameter value
        for j, (param_key, param_val) in enumerate(zip(param_keys, param_values)):
            # Get trajectory data for this compartment
            traj_data = trajectories[param_key][compartment_idx]
            
            # Normalize if requested
            if normalize and np.max(traj_data) > 0:
                traj_data = traj_data / np.max(traj_data)
            
            # Plot with color based on parameter value
            ax.plot(
                time_points,
                traj_data,
                color=cmap(norm(param_val)),
                alpha=alpha,
                label=f"{param_name}={param_val:.3g}" if i == 0 else None  # Only add label in first subplot
            )
        
        # Set title for this subplot
        ax.set_title(f"{compartment_name} Dynamics")
        
        # Add grid if requested
        if grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add y-label
        if normalize:
            ax.set_ylabel(f"Normalized\n{compartment_name}")
        else:
            ax.set_ylabel(compartment_name)
    
    # Add x-label to bottom subplot
    axes[-1].set_xlabel("Time")
    
    # Add overall title if provided or create default
    if title is None:
        title = f"Effect of {param_name} on Compartment Dynamics"
    fig.suptitle(title, fontsize=14)
    
    # Add legend in first subplot
    if axes:
        axes[0].legend(loc='best')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label=param_name)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def run_polarization_homophily_trajectories(
    model_module,
    polarization_values,
    homophily_values,
    color_matrix,
    custom_params=None,
    initial_infected_prop=1e-4,
    population_size=100,
    simulated_days=1000,
):
    """
    Run simulations for different combinations of polarization and homophily.
    
    Args:
        model_module: The model module to use (e.g., SIRM, SIRT, SIRV)
        polarization_values: List of polarization values to simulate
        homophily_values: List of homophily values to simulate
        color_matrix: 2D array of colors for visualization
        custom_params: Custom parameters for the model (None to use defaults)
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        simulated_days: Number of days to simulate
    
    Returns:
        Dictionary with trajectories and metadata for visualization
    """
    # Import utility to convert polarization to beta parameters
    from ..distributions import pol_to_alpha
    
    # Set up model parameters
    if custom_params is None:
        model_params = model_module.get_default_params()
    else:
        model_params = dict(model_module.get_default_params())
        model_params.update(custom_params)
    
    # Get model name and compartment names
    model_name, compartment_names = model_module.get_compartment_info()
    
    # Dictionary to store trajectories
    trajectories = {}
    colors = {}
    labels = {}
    
    # Calculate time steps
    dT = model_params.get('dT', 1)
    n_steps = int(simulated_days / dT)
    time_points = jnp.arange(0, simulated_days + dT, dT)
    
    # Run simulations for each combination
    for i, pol in enumerate(polarization_values):
        for j, hom in enumerate(homophily_values):
            # Convert polarization to alpha for beta distribution
            alpha = pol_to_alpha(pol)
            beta_params = (alpha, alpha)  # Symmetric distribution
            
            # Set homophilic tendency
            sim_params = dict(model_params)
            sim_params['homophilic_tendency'] = hom
            
            # Run simulation with trajectory output
            results = model_module.run_simulation(
                beta_params=beta_params,
                params=sim_params,
                simulated_days=simulated_days,
                initial_infected_prop=initial_infected_prop,
                population_size=population_size,
                use_contact_matrix=True,
                return_trajectory=True,
            )
            
            # Store results with key that identifies the parameters
            key = f"pol_{pol}_hom_{hom}"
            trajectories[key] = results[0]  # First element contains trajectories
            colors[key] = color_matrix[i, j]
            labels[key] = f"P={pol}, H={hom}"
    
    return {
        "trajectories": trajectories,
        "colors": colors,
        "labels": labels,
        "time_points": time_points,
        "compartment_names": compartment_names,
        "model_name": model_name
    }

def plot_polarization_homophily_trajectories(
    model_module,
    polarization_values,
    homophily_values,
    color_matrix,
    compartment_to_plot=None,
    custom_params=None,
    initial_infected_prop=1e-4,
    population_size=100,
    simulated_days=1000,
    fig_size=(10, 6),
    title=None,
    legend=False,
    line_alpha=0.8,  # Renamed to avoid confusion
    save_path=None,
):
    """
    Run simulations and plot trajectories for polarization and homophily combinations.
    
    Args:
        model_module: The model module to use (e.g., SIRM, SIRT, SIRV)
        polarization_values: List of polarization values to simulate
        homophily_values: List of homophily values to simulate
        color_matrix: 2D array of colors where color_matrix[i,j] is the color for 
                     the combination of polarization_values[i] and homophily_values[j]
        compartment_to_plot: Which compartments to plot (None for all, or list of indices/names)
        custom_params: Custom parameters for the model (None to use defaults)
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        simulated_days: Number of days to simulate
        fig_size: Figure size tuple
        title: Plot title (None for auto-generated)
        legend: Whether to show the legend
        line_alpha: Transparency for the lines (0-1)
        save_path: Path to save the figure
    
    Returns:
        Tuple of (figure, trajectories_dict)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    from ..distributions import pol_to_alpha
    
    # Check color matrix dimensions
    if color_matrix.shape != (len(polarization_values), len(homophily_values)):
        color_matrix = color_matrix.T
        if color_matrix.shape != (len(polarization_values), len(homophily_values)):
            raise ValueError(f"Color matrix shape doesn't match polarization and homophily values")
    
    # Set up model parameters
    if custom_params is None:
        model_params = model_module.get_default_params()
    else:
        model_params = dict(model_module.get_default_params())
        model_params.update(custom_params)
    
    # Get model name and compartment names
    model_name, compartment_names = model_module.get_compartment_info()
    
    # Define line styles based on homophily values
    # Lightest (first) = dotted, medium (second) = dashed, highest (third) = solid
    linestyles = [':','--','-']
    
    # Dictionary to store trajectories
    trajectories = {}
    
    # Calculate time steps
    dT = model_params.get('dT', 1)
    n_steps = int(simulated_days / dT)
    time_points = np.arange(0, simulated_days + dT, dT)
    
    # Run simulations for each combination
    for i, pol in enumerate(polarization_values):
        for j, hom in enumerate(homophily_values):
            # Convert polarization to alpha for beta distribution
            alpha = pol_to_alpha(pol)
            beta_params = (alpha, alpha)  # Symmetric distribution
            
            # Set homophilic tendency
            sim_params = dict(model_params)
            sim_params['homophilic_tendency'] = hom
            
            # Run simulation with trajectory output
            results = model_module.run_simulation(
                beta_params=beta_params,
                params=sim_params,
                simulated_days=simulated_days,
                initial_infected_prop=initial_infected_prop,
                population_size=population_size,
                use_contact_matrix=True,
                return_trajectory=True,
            )
            
            # Store results with key that identifies the parameters
            key = f"pol_{pol}_hom_{hom}"
            trajectories[key] = results[0]  # First element contains trajectories
    
    # Determine which compartments to plot
    if compartment_to_plot is None:
        # Plot all compartments
        compartments_to_show = list(range(len(compartment_names)))
    elif isinstance(compartment_to_plot, list):
        # List of compartment indices or names
        compartments_to_show = []
        for comp in compartment_to_plot:
            if isinstance(comp, int):
                compartments_to_show.append(comp)
            elif isinstance(comp, str) and comp in compartment_names:
                compartments_to_show.append(compartment_names.index(comp))
    elif isinstance(compartment_to_plot, (int, str)):
        # Single compartment
        if isinstance(compartment_to_plot, int):
            compartments_to_show = [compartment_to_plot]
        else:
            compartments_to_show = [compartment_names.index(compartment_to_plot)] if compartment_to_plot in compartment_names else []
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot each combination
    for i, pol in enumerate(polarization_values):
        for j, hom in enumerate(homophily_values):
            key = f"pol_{pol}_hom_{hom}"
            if key in trajectories:
                traj = trajectories[key]
                color = color_matrix[i, j]
                linestyle = linestyles[j % len(linestyles)]  # Line style based on homophily value
                
                # Plot selected compartments
                for comp_idx in compartments_to_show:
                    if comp_idx < len(traj):
                        # Sum across population groups for each time step
                        comp_data = np.array(jnp.sum(traj[comp_idx], axis=1))
                        
                        # Create label based on parameters and compartment
                        if len(compartments_to_show) > 1:
                            label = f"{compartment_names[comp_idx]} (P={pol}, H={hom})"
                        else:
                            label = f"P={pol}, H={hom}"
                            
                        ax.plot(
                            time_points[:len(comp_data)],  # Time points
                            comp_data,  # Compartment data
                            color=color,
                            linestyle=linestyle,
                            label=label,
                            alpha=line_alpha,  # Using the renamed parameter
                        )
    
    # Set labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Population Fraction")
    
    if title is None:
        if len(compartments_to_show) == 1:
            title = f"{model_name}: {compartment_names[compartments_to_show[0]]} Trajectories"
        else:
            title = f"{model_name}: Compartment Trajectories"
    
    ax.set_title(title)
    
    # Add legend if requested
    if legend:
        ax.legend(loc="best")
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, trajectories


def run_single_simulation(polarization, homophilic_tendency, fixed_mean, PARAMS, model, simulated_days=1000,
                          initial_infected_prop = 1e-4, population_size = 5):
    """
    Helper function to run a single simulation with given parameters.
    
    Returns:
    --------
    tuple 
        (S, I, R) where each is a sum over population compartments
    """
    # Set up parameters for this simulation
    params = PARAMS.copy()
    params.update({
        'homophilic_tendency': homophilic_tendency,
        'fixed_mean': fixed_mean
    })
    
    # Calculate beta parameters from polarization and mean
    beta_params = pol_mean_to_ab(polarization, params['fixed_mean'])
    
    # Run the simulation
    states_trajectory, r0, homophily = model.run_simulation(
        beta_params=beta_params,
        params=params,
        simulated_days=simulated_days,
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=True,
        return_trajectory=True
    )
    
    # Unpack and sum the state trajectories across population compartments
    S, I, R, *_ = states_trajectory
    S_total = jnp.sum(S, axis=1)
    I_total = jnp.sum(I, axis=1)
    R_total = jnp.sum(R, axis=1)
    
    return S_total, I_total, R_total