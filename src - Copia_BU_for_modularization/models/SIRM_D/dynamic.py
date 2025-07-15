import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from ...utils.distributions import my_beta_asymmetric

StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
ParamType = Dict[str, Any]

def generate_susceptibility(beta_range: Tuple[float, float], N_COMPARTMENTS: int, SPB_exponent: float = 1.0) -> jnp.ndarray:
    """Generate susceptibility values with non-linear interpolation (default linear)"""
    beta_low, beta_high = beta_range
    x = jnp.linspace(0, 1, num=N_COMPARTMENTS)
    return beta_high + jnp.power(x, SPB_exponent) * (beta_low - beta_high)

def initialize_states(
    beta_params: Tuple[float, float],
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100
) -> StateType:
    """Initialize states for fixed-size system normalized to total population 1"""
    # Generate normalized population distribution
    populations = my_beta_asymmetric(beta_params[0], beta_params[1], N_COMPARTMENTS, norm=1.0)
    
    # Initialize compartments
    S = populations * (1 - initial_infected_prop)
    I = populations * initial_infected_prop
    R = jnp.zeros_like(populations)
    
    return S, I, R

@jax.jit
def SIRM_D_step(
    state: StateType,
    susceptibility: jnp.ndarray,
    gamma: float,
    dT: float = 1.0
) -> StateType:
    """
    Simulate one step of the disconnected SIRM model.
    Each compartment only interacts with itself.
    """
    S, I, R = state
    
    # Calculate total population in each compartment
    N = S + I + R  # Population of each compartment
    
    # Each compartment only sees its own infected population normalized by its own total population
    # No interaction between compartments
    new_infections = susceptibility * S * (I / N) * dT  # Force of infection normalized by compartment population
    new_recoveries = gamma * I * dT

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return S_new, I_new, R_new

@jax.jit
def execute_sim_SIRM_D_final(
    initial_state: StateType,
    params: ParamType,
    n_steps: int,
    beta_params: Tuple[float, float],
    nan_state: StateType,
    susceptibilities: jnp.ndarray
) -> Tuple[StateType, float]:
    """Execute simulation until final state"""
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    gamma = params['recovery_rate']

    def body_fun(_, state):
        return SIRM_D_step(state, susceptibilities, gamma, dT=params["dT"])

    final_state = jax.lax.fori_loop(0, n_steps, body_fun, initial_state)
    
    # For disconnected case, R0 is simply the max of susceptibility/gamma across compartments
    r0 = jnp.where(is_valid, jnp.max(susceptibilities/gamma), jnp.nan)
    
    # Return placeholder 0.0 for h to maintain interface compatibility
    return jax.tree_map(
        lambda v, n: jnp.where(is_valid, v, n),
        final_state,
        nan_state
    ), r0, 0.0

def sim_SIRM_D_final(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0,
    **kwargs  # Accept but ignore extra parameters for compatibility
) -> Tuple[StateType, float]:
    """
    Simulate disconnected SIRM model to final state.
    Each compartment evolves independently.
    """
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    
    nan_state = (
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan)
    )
    
    susceptibilities = generate_susceptibility(
        params['susceptibility_rates'], 
        N_COMPARTMENTS,
        SPB_exponent = SPB_exponent
    )
    
    return execute_sim_SIRM_D_final(
        initial_state, 
        params, 
        n_steps, 
        beta_params, 
        nan_state,
        susceptibilities
    )

def execute_sim_SIRM_D_trajectory(
    initial_state: StateType,
    params: ParamType,
    n_steps: int,
    beta_params: Tuple[float, float],
    trajectories: jnp.ndarray,
    nan_trajectories: StateType,
    susceptibilities: jnp.ndarray
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]:
    """Execute simulation recording full trajectory"""
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    gamma = params['recovery_rate']
    
    # For disconnected case, R0 is simply the max of susceptibility/gamma across compartments
    r0 = jnp.where(is_valid, jnp.max(susceptibilities/gamma), jnp.nan)
    
    trajectories = trajectories.at[0, 0].set(initial_state[0])
    trajectories = trajectories.at[1, 0].set(initial_state[1])
    trajectories = trajectories.at[2, 0].set(initial_state[2])
    
    def body_fun(carry, t):
        state, trajectories = carry
        next_state = SIRM_D_step(state, susceptibilities, gamma, dT=params["dT"])
        trajectories = trajectories.at[0, t + 1].set(next_state[0])
        trajectories = trajectories.at[1, t + 1].set(next_state[1])
        trajectories = trajectories.at[2, t + 1].set(next_state[2])
        return (next_state, trajectories), None
    
    (_, trajectories), _ = jax.lax.scan(
        body_fun,
        (initial_state, trajectories),
        jnp.arange(n_steps)
    )
    
    valid_result = (trajectories[0], trajectories[1], trajectories[2])
    
    return jax.tree_map(
        lambda v, n: jnp.where(is_valid, v, n),
        valid_result,
        (nan_trajectories[0], nan_trajectories[1], nan_trajectories[2])
    ), r0

def sim_SIRM_D_trajectory(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0,
    **kwargs  # Accept but ignore extra parameters for compatibility
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]:
    """
    Simulate disconnected SIRM model recording full trajectory.
    Each compartment evolves independently.
    """
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    trajectories = jnp.zeros((3, n_steps + 1, N_COMPARTMENTS))
    nan_trajectories = jnp.full((3, n_steps + 1, N_COMPARTMENTS), jnp.nan)
    
    susceptibilities = generate_susceptibility(
        params['susceptibility_rates'], 
        N_COMPARTMENTS,
        SPB_exponent = SPB_exponent
    )
    
    return execute_sim_SIRM_D_trajectory(
        initial_state, 
        params, 
        n_steps, 
        beta_params,
        trajectories, 
        nan_trajectories,
        susceptibilities
    )