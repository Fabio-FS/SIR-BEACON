import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple, Callable, List, Union

# Type definitions
ParamType = Dict[str, Any]
StateType = Union[Tuple[jnp.ndarray, ...], List[jnp.ndarray]]

def generate_behavior_pattern(population_size: int, params: ParamType, behavior_key: Union[str, float], min_val: float = 0.0) -> jnp.ndarray:
    """Generate behavior pattern across population compartments
    
    Args:
        population_size: Number of population compartments
        params: Model parameters
        behavior_key: Key for the behavior parameter in params (e.g., 'mu_max') 
                     or the actual max value to use
        min_val: Minimum value for the behavior (default: 0.0)
        
    Returns:
        Array of behavior values across population compartments
    """
    x = jnp.linspace(0, 1, num=population_size)
    
    # Handle different parameter formats
    if isinstance(behavior_key, str):
        # behavior_key is a parameter name
        if isinstance(params[behavior_key], tuple):
            # For parameters stored as (min, max) tuples
            max_value = params[behavior_key][1]
        else:
            # For scalar parameters
            max_value = params[behavior_key]
    else:
        # behavior_key is already the max value
        max_value = behavior_key
        
    # Generate the behavior pattern and add the minimum value
    return min_val + (max_value - min_val) * jnp.power(x, params.get('SPB_exponent', 1.0))

def create_initial_states(populations: jnp.ndarray, 
                          initial_infected_prop: float,
                          n_health_states: int = 3) -> Tuple[jnp.ndarray, ...]:
    """Create initial state arrays for a given population distribution
    
    Args:
        populations: Population distribution across compartments
        initial_infected_prop: Initial proportion of infected individuals
        n_health_states: Number of health states in the model
        
    Returns:
        Tuple of initial state arrays
    """
    # Initialize compartments
    states = []
    
    # First compartment is always Susceptible (S)
    states.append(populations * (1 - initial_infected_prop))
    
    # Second compartment is always Infected (I)
    states.append(populations * initial_infected_prop)
    
    # Remaining compartments start with zero population
    for _ in range(2, n_health_states):
        states.append(jnp.zeros_like(populations))
    
    return tuple(states)


def initialize_states(beta_params: Tuple[float, float], 
                     initial_infected_prop: float, 
                     population_size: int,
                     n_health_states: int = 3) -> StateType:
    """Initialize model states based on a beta distribution of the population
    
    Args:
        beta_params: Parameters for the beta distribution
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        n_health_states: Number of health states in the model (3 for SIR, 4 for SIRV)
        
    Returns:
        Tuple of initial state arrays
    """
    from ..utils.distributions import my_beta_asymmetric
    
    # Generate normalized population distribution
    populations = my_beta_asymmetric(beta_params[0], beta_params[1], population_size, norm=1.0)
    
    # Initialize common compartments
    S = populations * (1 - initial_infected_prop)
    I = populations * initial_infected_prop
    R = jnp.zeros_like(populations)
    
    # Initialize first 3 standard compartments (SIR)
    states = [S, I, R]
    
    # Add additional compartments if needed
    for _ in range(3, n_health_states):
        states.append(jnp.zeros_like(populations))
    
    return tuple(states)


def run_simulation(
    model_step_fn: Callable,
    r0_calculation_fn: Callable,
    beta_params: Tuple[float, float],
    params: ParamType,
    simulated_days: float,
    initial_infected_prop: float = 1e-4,
    population_size: int = 100,
    use_contact_matrix: bool = False,
    return_trajectory: bool = False,
    custom_behavior_distribution: Optional[jnp.ndarray] = None,
    custom_contact_matrix: Optional[jnp.ndarray] = None,
    custom_populations: Optional[jnp.ndarray] = None,
    n_health_states: int = 3,
    prepare_step_params_fn: Optional[Callable] = None
) -> Tuple:
    """Generic simulation function for compartment models
    
    Args:
        model_step_fn: Function that executes one step of the model
        r0_calculation_fn: Function that calculates R0 for the model
        beta_params: Parameters for the beta distribution of population
        params: Model parameters
        simulated_days: Number of days to simulate
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        use_contact_matrix: Whether to use the contact matrix
        return_trajectory: Whether to return the full trajectory or just the final state
        custom_behavior_distribution: Optional custom behavior distribution
        custom_contact_matrix: Optional custom contact matrix
        custom_populations: Optional custom population distribution
        n_health_states: Number of health states in the model (3 for SIR, 4 for SIRV)
        prepare_step_params_fn: Function to prepare parameters for the step function
        
    Returns:
        If return_trajectory is True:
            Tuple of (states_trajectory, r0, obs_h)
        Otherwise:
            Tuple of (final_state, r0, obs_h)
    """
    from ..utils.Contact_Matrix import create_contact_matrix
    
    # Calculate the number of steps based on simulated_days and time step dT
    dT = params.get('dT', 1)  # Get dT from params or use default 0.25
    n_steps = int(simulated_days / dT)
    
    # Use provided populations or generate from beta distribution
    if custom_populations is not None:
        # Use custom population distribution
        POP = custom_populations
        population_size = len(POP)
        initial_state = create_initial_states(POP, initial_infected_prop, n_health_states)
    else:
        # Initialize states from beta distribution and extract the population
        initial_state = initialize_states(beta_params, initial_infected_prop, population_size, n_health_states)
        # Sum all compartments to get the total population
        POP = sum(initial_state)
    
    # Use provided contact matrix or create one
    if custom_contact_matrix is not None:
        C = custom_contact_matrix
        use_contact_matrix = True
    else:
        C = create_contact_matrix(population_size, params.get('homophilic_tendency', 0), POP)
        use_contact_matrix = use_contact_matrix or params.get('homophilic_tendency', 0) != 0
    
    # Prepare parameters for step function using the provided function
    step_params = prepare_step_params_fn(params, custom_behavior_distribution, population_size) if prepare_step_params_fn else {}
    
    # Check if parameters are valid
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    # Calculate R0
    r0 = jnp.where(is_valid, r0_calculation_fn(step_params, C, POP), jnp.nan)
    obs_h = 0.0  # No observed homophily for this model
    
    if return_trajectory:
        # Initialize trajectory storage
        trajectories = jnp.zeros((n_health_states, n_steps + 1, population_size))
        nan_trajectories = jnp.full((n_health_states, n_steps + 1, population_size), jnp.nan)
        
        # Store initial state
        for i in range(n_health_states):
            trajectories = trajectories.at[i, 0].set(initial_state[i])
        
        def body_fun(carry, t):
            state, trajs = carry
            # Execute step with the specific model step function
            next_state = model_step_fn(state, C, use_contact_matrix, params["dT"], **step_params)
            
            # Update trajectories
            for i in range(n_health_states):
                trajs = trajs.at[i, t + 1].set(next_state[i])
                
            return (next_state, trajs), None
        
        (_, trajectories), _ = jax.lax.scan(
            body_fun,
            (initial_state, trajectories),
            jnp.arange(n_steps)
        )
        
        # Extract results from trajectories
        valid_result = tuple(trajectories[i] for i in range(n_health_states))
        invalid_result = tuple(nan_trajectories[i] for i in range(n_health_states))
        
        # Use where to conditionally return valid or invalid results
        # This avoids using Python control flow that would fail with tracers
        result = tuple(
            jnp.where(is_valid, valid, invalid) 
            for valid, invalid in zip(valid_result, invalid_result)
        )
        
        return result, r0, obs_h
    else:
        # Only calculate final state
        nan_state = tuple(jnp.full(population_size, jnp.nan) for _ in range(n_health_states))
        
        def body_fun(t, state):
            return model_step_fn(state, C, use_contact_matrix, params["dT"], **step_params)

        final_state = jax.lax.fori_loop(0, n_steps, body_fun, initial_state)
        
        # Use where to conditionally return valid or invalid results
        # This avoids using Python control flow that would fail with tracers
        result = tuple(
            jnp.where(is_valid, valid, invalid) 
            for valid, invalid in zip(final_state, nan_state)
        )
        
        return result, r0, obs_h


@jax.jit
def calculate_infection_force(I: jnp.ndarray, 
                             contact_matrix: Optional[jnp.ndarray] = None, 
                             use_contact_matrix: bool = False) -> jnp.ndarray:
    """Calculate force of infection with or without contact structure
    
    Args:
        I: Array of infected individuals across compartments
        contact_matrix: Optional contact matrix for structured mixing
        use_contact_matrix: Whether to use the contact matrix
        
    Returns:
        Force of infection for each compartment
    """
    return jax.lax.cond(
        jnp.logical_and(use_contact_matrix, contact_matrix is not None),
        lambda _: contact_matrix @ I,
        lambda _: jnp.full_like(I, jnp.sum(I)),
        operand=None
    )