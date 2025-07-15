"""
Example demonstrating the refactored visualization approach using parse_2_parameters and plot_metrics
"""
import matplotlib.pyplot as plt
import numpy as np

# Import the simulation model
from src.models import SIRM  # SIRM model example

# Import the sweep function
from src.utils.batch_sweep import sweep_two_parameters

# Import the visualization functions
from src.utils.visualization import (
    plot_multiple_metrics,  # Traditional approach
    parse_2_parameters,    # New data processing
    plot_metrics           # New visualization
)

def main():
    """Run example simulation and visualize with both old and new approaches"""
    print("Running simulation with SIRM model...")
    
    # Run parameter sweep
    results = sweep_two_parameters(
        model_module=SIRM,
        param1_name="polarization_mean",
        param1_range={"m": 0.0, "M": 1.0, "n": 20},
        param2_name="mask_efficacy",
        param2_range={"m": 0.0, "M": 1.0, "n": 20},
        simulated_days=250,
        custom_base_params={
            "mask_usage": 0.5
        }
    )
    
    # -------------------------------
    # Traditional approach
    # -------------------------------
    print("Creating visualization using traditional approach...")
    
    # Plot multiple metrics the traditional way
    fig1 = plot_multiple_metrics(
        results=results,
        metrics=["infections", "r0"],
        fig_size=(12, 6),
        save_path="traditional_approach.png"
    )
    
    # -------------------------------
    # New refactored approach
    # -------------------------------
    print("Creating visualization using new refactored approach...")
    
    # Step 1: Parse the simulation results to prepare data for visualization
    parsed_data = parse_2_parameters(
        results=results,
        metrics=["infections", "r0"]
    )
    
    # Step 2: Create visualization using the parsed data
    fig2 = plot_metrics(
        parsed_data=parsed_data,
        fig_size=(12, 6),
        save_path="refactored_approach.png"
    )
    
    # -------------------------------
    # Benefits of the refactored approach
    # -------------------------------
    print("Demonstrating benefits of refactored approach...")
    
    # 1. Reuse parsed data for different visualizations
    # Create a second visualization with different options
    fig3 = plot_metrics(
        parsed_data=parsed_data,
        cmaps={"infections": "hot_r", "r0": "viridis"},
        contour_values={"infections": [0.2, 0.4, 0.6, 0.8]},
        contour_colors={"infections": ["white"]},
        fig_size=(12, 6),
        save_path="refactored_approach_customized.png"
    )
    
    # 2. Create a single-metric plot with custom parameters
    # Extract just the infections data
    infections_data = {
        'data': {k: v for k, v in parsed_data['data'].items() if k == 'infections'},
        'metadata': {k: v for k, v in parsed_data['metadata'].items() if k == 'infections'},
        'axis_config': parsed_data['axis_config'],
        'model_name': parsed_data['model_name']
    }
    
    # Plot just the infections with custom parameters
    fig4 = plot_metrics(
        parsed_data=infections_data,
        cmaps={"infections": "hot_r"},
        fig_size=(8, 6),
        final_params={
            'Lx': 4,
            'Ly': 4,
            'xticks': np.linspace(0, 1, 6),
            'yticks': np.linspace(0, 1, 6)
        },
        save_path="refactored_approach_single.png"
    )
    
    # Show all figures
    plt.show()

if __name__ == "__main__":
    main() 