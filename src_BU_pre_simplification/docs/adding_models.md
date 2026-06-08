# Adding New Models to BEACON

This guide explains how to extend the BEACON framework by adding your own epidemic models.

## Model Structure

Each model in BEACON is a separate Python module implementing a common interface. This approach ensures that all models can be used with the existing parameter sweeping and analysis tools.

## Step 1: Create a New Model File

Create a new Python file in the `models` directory, e.g., `models/MyNewModel.py`.

## Step 2: Implement the Required Interface

Your model needs to implement the following functions:

### 1. `get_default_params()`

This function returns a dictionary of default parameter values for your model.

```python
def get_default_params() -> Dict[str, Any]:
    """Return default parameters for the model"""
    return {
        'parameter1': 0.5,  # Description of parameter1
        'parameter2': 0.1,  # Description of parameter2
        'dT': 0.25,         # Time step
        'recovery_rate': 0.1,  # Recovery rate
        'homophilic_tendency': 0,  # Homophily parameter
        # Add other parameters...
    }
```

### 2. `initialize_states()`

This function initializes the compartments for your model based on a beta distribution of the population.

```python
def initialize_states(beta_params: Tuple[float, float], 
                     initial_infected_prop: float, 
                     population_size: int) -> StateType:
    """Initialize model states"""
    from ..utils.distributions import my_beta_asymmetric
    
    # Generate normalized population distribution
    populations = my_beta_asymmetric(beta_params[0], beta_params[1], population_size, norm=1.0)
    
    # Initialize compartments (example for a SIR-type model)
    S = populations * (1 - initial_infected_prop)
    I = populations * initial_infected_prop
    R = jnp.zeros_like(populations)
    
    # Add any additional compartments your model needs
    
    return S, I, R  # Return a tuple of all compartments
```

### 3. `step()`

This function executes one time step of your model dynamics.

```python
@jax.jit
def step(state: StateType,
        parameter1: float,
        parameter2: jnp.ndarray,
        contact_matrix: Optional[jnp.ndarray] = None,
        use_contact_matrix: bool = False,
        dT: float = 1.0) -> StateType:
    """Execute one time step of the model"""
    # Unpack state
    S, I, R = state  # Adjust based on your compartments
    
    # Calculate force of infection
    infection_force = calculate_infection_force(I, contact_matrix, use_contact_matrix)
    
    # Calculate transitions between compartments
    new_infections = parameter1 * S * infection_force * dT
    new_recoveries = parameter2 * I * dT
    
    # Update compartments
    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries
    
    return S_new, I_new, R_new  # Return updated state
```

### 4. `calculate_r0()`

This function calculates the basic reproduction number for your model.

```python
def calculate_r0(parameter1: float, 
                parameter2: jnp.ndarray, 
                contact_matrix: jnp.ndarray, 
                populations: jnp.ndarray) -> float:
    """Calculate R0 for the model"""
    from ..utils.R0 import R0_calculation_function  # Use appropriate R0 function
    
    # Implement R0 calculation logic
    return R0_calculation_function(parameter1, parameter2, contact_matrix, populations)
```

### 5. `run_simulation()`

This is the main function that runs a complete simulation with your model.

```python
def run_simulation(beta_params: Tuple[float, float],
                  params: ParamType,
                  n_steps: int,
                  initial_infected_prop: float = 1e-4,
                  population_size: int = 100,
                  use_contact_matrix: bool = False,
                  return_trajectory: bool = False,
                  custom_behavior_distribution: Optional[jnp.ndarray] = None,
                  custom_contact_matrix: Optional[jnp.ndarray] = None,
                  custom_populations: Optional[jnp.ndarray] = None) -> Tuple:
    """Run a complete simulation"""
    from ..utils.Contact_Matrix import create_contact_matrix
    
    # Initialize the model
    # Set up contact matrix, behavior patterns, etc.
    # Run the simulation logic
    # Return final state or trajectory
    
    # See existing models for complete implementation examples
```

## Step 3: Make Your Model Available for Import

For the framework to recognize your model, you might need to update the `models/__init__.py` file:

```python
from .SIRM import *
from .SIRT import *
from .SIRV import *
from .MyNewModel import *  # Add your model
```

## Complete Example

Here's a simple skeleton for a new model:

```python
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Optional

# Type definitions
StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]  # Adjust based on your compartments
ParamType = Dict[str, Any]

def get_default_params() -> Dict[str, Any]:
    """Return default parameters for the model"""
    return {
        'parameter1': 0.5,
        'parameter2': 0.1,
        'dT': 0.25,
        'recovery_rate': 0.1,
        'homophilic_tendency': 0
    }

def generate_behavior_pattern(population_size: int, params: ParamType) -> jnp.ndarray:
    """Generate behavior pattern across population compartments"""
    x = jnp.linspace(0, 1, num=population_size)
    return params['parameter1'] * jnp.power(x, params.get('SPB_exponent', 1.0))

def initialize_states(beta_params: Tuple[float, float], 
                     initial_infected_prop: float, 
                     population_size: int) -> StateType:
    """Initialize model states"""
    from ..utils.distributions import my_beta_asymmetric
    
    populations = my_beta_asymmetric(beta_params[0], beta_params[1], population_size, norm=1.0)
    
    S = populations * (1 - initial_infected_prop)
    I = populations * initial_infected_prop
    R = jnp.zeros_like(populations)
    
    return S, I, R

@jax.jit
def calculate_infection_force(I: jnp.ndarray, 
                             contact_matrix: Optional[jnp.ndarray] = None, 
                             use_contact_matrix: bool = False) -> jnp.ndarray:
    """Calculate force of infection"""
    return jax.lax.cond(
        jnp.logical_and(use_contact_matrix, contact_matrix is not None),
        lambda _: contact_matrix @ I,
        lambda _: jnp.full_like(I, jnp.sum(I)),
        operand=None
    )

@jax.jit
def step(state: StateType,
        parameter1: float,
        parameter2: jnp.ndarray,
        contact_matrix: Optional[jnp.ndarray] = None,
        use_contact_matrix: bool = False,
        dT: float = 1.0) -> StateType:
    """Execute one time step of the model"""
    S, I, R = state
    
    infection_force = calculate_infection_force(I, contact_matrix, use_contact_matrix)
    
    new_infections = parameter1 * S * infection_force * dT
    new_recoveries = parameter2 * I * dT
    
    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries
    
    return S_new, I_new, R_new

def calculate_r0(parameter1: float, 
                parameter2: jnp.ndarray, 
                contact_matrix: jnp.ndarray, 
                populations: jnp.ndarray) -> float:
    """Calculate R0 for the model"""
    # Implement R0 calculation
    return 0.0  # Placeholder

def run_simulation(beta_params: Tuple[float, float],
                  params: ParamType,
                  n_steps: int,
                  initial_infected_prop: float = 1e-4,
                  population_size: int = 100,
                  use_contact_matrix: bool = False,
                  return_trajectory: bool = False,
                  custom_behavior_distribution: Optional[jnp.ndarray] = None,
                  custom_contact_matrix: Optional[jnp.ndarray] = None,
                  custom_populations: Optional[jnp.ndarray] = None) -> Tuple:
    """Run a complete simulation"""
    from ..utils.Contact_Matrix import create_contact_matrix
    
    # Use provided populations or generate from beta distribution
    if custom_populations is not None:
        POP = custom_populations
        initial_state = (
            POP * (1 - initial_infected_prop),
            POP * initial_infected_prop,
            jnp.zeros_like(POP)
        )
        population_size = len(POP)
    else:
        initial_state = initialize_states(beta_params, initial_infected_prop, population_size)
        POP = initial_state[0] + initial_state[1] + initial_state[2]
    
    # Use provided contact matrix or create one
    if custom_contact_matrix is not None:
        C = custom_contact_matrix
        use_contact_matrix = True
    else:
        C = create_contact_matrix(population_size, params['homophilic_tendency'], POP)
        use_contact_matrix = use_contact_matrix or params['homophilic_tendency'] != 0
    
    # Use provided behavior distribution or generate one
    if custom_behavior_distribution is not None:
        behavior_pattern = custom_behavior_distribution
    else:
        behavior_pattern = generate_behavior_pattern(population_size, params)
    
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    parameter1 = params['parameter1']
    parameter2 = params['parameter2']
    
    r0 = jnp.where(is_valid, calculate_r0(parameter1, behavior_pattern, C, POP), jnp.nan)
    obs_h = 0.0  # No observed homophily for this model
    
    # Run simulation with or without trajectory
    # ... Implement similar to existing models
    
    # Return results
    return final_state_or_trajectory, r0, obs_h
```

## Step 4: Test Your Model

Once you've implemented your model, test it with a simple simulation:

```python
from src.models import MyNewModel

# Get default parameters
params = MyNewModel.get_default_params()

# Run a simulation
states, r0, h = MyNewModel.run_simulation(
    beta_params=(1.0, 1.0),
    params=params,
    n_steps=100,
    population_size=50
)

# Check the results
print(f"R0: {r0}")
print(f"Final susceptible: {sum(states[0])}")
print(f"Final infected: {sum(states[1])}")
print(f"Final recovered: {sum(states[2])}")
```

## Step 5: Use with Parameter Sweeping

Your model should now be compatible with the batch sweeping functionality:

```python
from src.models import MyNewModel
from src.utils.batch_sweep import sweep_two_parameters

# Define parameter ranges
param1_range = {"m": 0, "M": 1, "n": 10}
param2_range = {"m": 0, "M": 1, "n": 10}

# Run parameter sweep
states, r0s, hs, param_grid = sweep_two_parameters(
    model_module=MyNewModel,
    param1_name="parameter1",
    param1_range=param1_range,
    param2_name="parameter2",
    param2_range=param2_range,
    n_steps=1000
)

# Analyze and visualize results
# ...
```

## Advanced Tips

1. **Optimize Performance**: Use JAX's JIT and vectorization capabilities for optimal performance.

2. **Consistent Interfaces**: Keep parameter and function names consistent with existing models when appropriate.

3. **Document Parameters**: Clearly document what each parameter does to make your model accessible to others.

4. **Test Edge Cases**: Test your model with extreme parameter values and ensure it handles them gracefully.

5. **Share Your Model**: Consider contributing your model back to the BEACON project if it might be useful to others.