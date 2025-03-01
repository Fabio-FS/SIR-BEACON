import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from ...utils.distributions import homogeneous_distribution

StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
ParamType = Dict[str, Any]

def generate_mask_wearing(mu_max: float, N_COMPARTMENTS: int, SPB_exponent: float = 1.0) -> jnp.ndarray:
    """Generate mask wearing values with non-linear interpolation (default linear)"""
    x = jnp.linspace(0, 1, num=N_COMPARTMENTS)
    mask = mu_max * jnp.power(x, SPB_exponent)
    return mask

def calculate_susceptibilities(beta_M: float, mask_wearing: jnp.ndarray) -> jnp.ndarray:
    """Calculate effective susceptibilities based on mask wearing behavior"""
    return beta_M * (1 - mask_wearing)

def initialize_states(
    beta_params: Tuple[float, float],
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100
) -> StateType:
    """Initialize states for fixed-size system normalized to total population 1"""
    from ...utils.distributions import my_beta_asymmetric
    
    # Generate normalized population distribution
    populations = my_beta_asymmetric(beta_params[0], beta_params[1], N_COMPARTMENTS, norm=1.0)
    
    # Initialize compartments
    S = populations * (1 - initial_infected_prop)
    I = populations * initial_infected_prop
    R = jnp.zeros_like(populations)
    
    return S, I, R

@jax.jit
def maskSIRD_step(
    state: StateType,
    susceptibilities: jnp.ndarray,
    gamma: float,
    dT: float = 1.0
) -> StateType:
    """Execute one time step of the maskSIR_D model.
    In this model, each compartment only interacts with itself."""
    S, I, R = state
    N = S + I + R  # Population of each compartment
    
    
    
    new_infections = susceptibilities * S * (I / N) * dT
    new_recoveries = gamma * I * dT

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return S_new, I_new, R_new

@jax.jit
def execute_sim_maskSIRD_final(
    initial_state: StateType,
    params: ParamType,
    n_steps: int,
    beta_params: Tuple[float, float],
    nan_state: StateType,
    susceptibilities: jnp.ndarray,
    mask_wearing: jnp.ndarray
) -> Tuple[StateType, float, float]:
    """Execute complete simulation of maskSIR_D model"""
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    gamma = params['recovery_rate']
    
    # For maskSIR_D, R0 is simply susceptibility/gamma for each compartment
    r0 = jnp.where(is_valid, jnp.max(susceptibilities/gamma), jnp.nan)
    obs_h = 0.0  # No homophily in non-mixing model

    def body_fun(_, state):
        return maskSIRD_step(state, susceptibilities, gamma, dT=params["dT"])

    final_state = jax.lax.fori_loop(0, n_steps, body_fun, initial_state)
    
    return jax.tree_map(
        lambda v, n: jnp.where(is_valid, v, n),
        final_state,
        nan_state
    ), r0, obs_h

def sim_maskSIRD_final(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0
) -> Tuple[StateType, float, float]:
    """Run complete maskSIR_D simulation with final state output"""
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    
    nan_state = (
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan)
    )
    
    # Generate mask wearing behavior and calculate susceptibilities
    mask_wearing = generate_mask_wearing(
        params['mu_max'],  # Maximum mask wearing value
        N_COMPARTMENTS,
        SPB_exponent
    )
    
    susceptibilities = calculate_susceptibilities(
        params['beta_M'],  # Maximum susceptibility
        mask_wearing
    )
    
    return execute_sim_maskSIRD_final(
        initial_state, 
        params, 
        n_steps, 
        beta_params, 
        nan_state,
        susceptibilities,
        mask_wearing
    )

@jax.jit
def execute_sim_maskSIRD_trajectory(
    initial_state: StateType,
    params: ParamType,
    n_steps: int,
    beta_params: Tuple[float, float],
    trajectories: jnp.ndarray,
    nan_trajectories: StateType,
    susceptibilities: jnp.ndarray,
    mask_wearing: jnp.ndarray
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], float, float]:
    """Execute complete simulation of maskSIR_D model with trajectory output"""
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    gamma = params['recovery_rate']
    
    # Calculate R0 as before
    r0 = jnp.where(is_valid, jnp.max(susceptibilities/gamma), jnp.nan)
    obs_h = 0.0
    
    trajectories = trajectories.at[0, 0].set(initial_state[0])
    trajectories = trajectories.at[1, 0].set(initial_state[1])
    trajectories = trajectories.at[2, 0].set(initial_state[2])
    
    def body_fun(carry, t):
        state, trajectories = carry
        next_state = maskSIRD_step(
            state,
            susceptibilities,
            gamma,
            params["dT"]
        )
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
        nan_trajectories
    ), r0, obs_h

def sim_maskSIRD_trajectory(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], float, float]:
    """Run complete maskSIR_D simulation with trajectory output"""
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    
    trajectories = jnp.zeros((3, n_steps + 1, N_COMPARTMENTS))
    nan_trajectories = jnp.full((3, n_steps + 1, N_COMPARTMENTS), jnp.nan)
    
    # Generate mask wearing behavior and calculate susceptibilities
    mask_wearing = generate_mask_wearing(
        params['mu_max'],
        N_COMPARTMENTS,
        SPB_exponent
    )
    
    susceptibilities = calculate_susceptibilities(
        params['beta_M'],
        mask_wearing
    )
    
    return execute_sim_maskSIRD_trajectory(
        initial_state, 
        params, 
        n_steps, 
        beta_params,
        trajectories, 
        nan_trajectories,
        susceptibilities,
        mask_wearing
    )