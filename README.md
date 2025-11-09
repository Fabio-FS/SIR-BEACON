# BEACON: Behavioral Epidemic Analysis with COntact patterNs

BEACON is a computational framework for modeling epidemics with heterogeneous social structures and behaviors. It implements a suite of compartmental models to simulate how social heterogeneity affects the efficacy of interventions like mask-wearing, testing, and vaccination.

## Key Features

- **Modular Structure**: Easy-to-extend framework for creating new models
- **Social Heterogeneity**: Models population heterogeneity through polarization parameters
- **Contact Patterns**: Implements flexible contact matrices with control over homophily
- **Fast Performance**: Built on JAX for efficient computation and parallel parameter sweeps
- **Versatile API**: Simple interface for running simulations and parameter sweeps

## Included Models

- **SIRM**: SIR model with mask-wearing (reduces susceptibility)
- **SIRT**: SIR model with testing-based isolation (increases recovery rate)
- **SIRV**: SIR model with vaccination (removes individuals from susceptible pool)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/beacon.git
cd beacon

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
from src.models import SIRM
from src.utils.batch_sweep import sweep_two_parameters

# Define parameter ranges
pol_range = {"m": 0, "M": 1, "n": 10}  # Polarization range
mask_range = {"m": 0, "M": 1, "n": 10}  # Mask effectiveness range

# Run parameter sweep
states, r0s, hs, param_grid = sweep_two_parameters(
    model_module=SIRM,
    param1_name="mu_max",        # Maximum mask effectiveness
    param1_range=mask_range,
    param2_name="beta_params",   # Controls population polarization
    param2_range=pol_range,
    n_steps=1000                 # Number of simulation steps
)

# Analyze results
S, I, R = states  # Final states for all parameter combinations
```

Check the `examples` directory for more detailed usage examples.

## Creating New Models

To create a new epidemic model compatible with BEACON:

1. Create a new file `models/YourModel.py`
2. Implement the required interface:
   - `get_default_params()`: Returns default model parameters
   - `run_simulation()`: Main simulation function
   - Core dynamics functions like `step()`, `initialize_states()`, etc.
3. Use your model with the existing batch sweep functionality

```python
# Example of a minimal model implementation
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional

def get_default_params() -> Dict[str, Any]:
    """Return default parameters for the model"""
    return {
        'parameter1': 0.5,
        'parameter2': 0.1,
        # Add other parameters...
    }

def run_simulation(beta_params, params, n_steps, **kwargs):
    """Main simulation function"""
    # Implement your simulation logic
    # ...
    return final_states, r0, h
```

See existing models for complete implementation examples.

## API Reference

### Model Interface

Each model should implement:

- `get_default_params()`: Returns default model parameters
- `run_simulation()`: Runs a complete simulation with given parameters
- `step()`: Executes one time step of the model dynamics
- `initialize_states()`: Initializes model states
- `calculate_r0()`: Calculates basic reproduction number

### Batch Sweeping

The main functions for parameter sweeping are:

- `batch_sweep()`: Low-level function for running many simulations in parallel
- `sweep_two_parameters()`: High-level function for sweeping over two parameters
- `create_parameter_grid()`: Creates a grid of parameter combinations

## License

MIT

## Citation

If you use BEACON in your research, please cite:
```
@article{beacon2025,
  title={BEACON: Behavioral Epidemic Analysis with COntact patterNs},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```