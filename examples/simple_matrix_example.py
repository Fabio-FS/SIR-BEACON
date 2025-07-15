"""
Example demonstrating the simplified matrix extraction approach
"""
import matplotlib.pyplot as plt
import numpy as np

# Import the simulation model
from src.models import SIRM  # SIRM model example

# Import the sweep function
from src.utils.batch_sweep import sweep_two_parameters

# Import the visualization functions
from src.utils.visualization import simple_parse_2_parameters

def main():
    """Run example simulation and extract a simple matrix"""
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
    
    # Extract just the matrix for infections using the simplified function
    infections_matrix = simple_parse_2_parameters(results, "infections")
    
    # Print shape and sample of the matrix
    print(f"Matrix shape: {infections_matrix.shape}")
    print("Sample from matrix (first 5x5):")
    print(infections_matrix[:5, :5])
    
    # Extract R0 matrix
    r0_matrix = simple_parse_2_parameters(results, "r0")
    print(f"R0 matrix shape: {r0_matrix.shape}")
    
    # Basic visualization with just the matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot infections matrix with simple imshow
    im1 = ax1.imshow(infections_matrix, origin='lower', aspect='auto', cmap='viridis')
    ax1.set_title("Infections")
    plt.colorbar(im1, ax=ax1)
    
    # Plot R0 matrix
    im2 = ax2.imshow(r0_matrix, origin='lower', aspect='auto', cmap='plasma')
    ax2.set_title("R0")
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig("simple_matrix_example.png")
    plt.show()

if __name__ == "__main__":
    main() 