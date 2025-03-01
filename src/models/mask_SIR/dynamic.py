import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from ...utils.Contact_Matrix import create_contact_matrix
from ...utils.R0 import R0_maskSIR
from ...utils.distributions import homogeneous_distribution

StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
ParamType = Dict[str, Any]

def generate_mask_wearing(mu_max: float, N_COMPARTMENTS: int, SPB_exponent: float = 1.0) -> jnp.ndarray:
    x = jnp.linspace(0, 1, num=N_COMPARTMENTS)
    mask = mu_max * jnp.power(x, SPB_exponent)
    #jax.debug.print("mask wearing values: {}", mask)
    return mask

def calculate_susceptibilities(beta_M: float, mask_wearing: jnp.ndarray) -> jnp.ndarray:
    susc = beta_M * (1 - mask_wearing)
    #jax.debug.print("calculated susceptibilities: {}", susc)
    #jax.debug.print("susceptibilities: {}", susc)
    #jax.debug.print("masks: {}", mask_wearing)
    return susc

def initialize_states(
    beta_params: Tuple[float, float],
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100
) -> StateType:
    """Initialize states for fixed-size system normalized to total population 1"""
    from ...utils.distributions import my_beta_asymmetric
    
    # Generate normalized population distribution
    populations = my_beta_asymmetric(beta_params[0], beta_params[1], N_COMPARTMENTS, norm=1.0)
    #jax.debug.print("initial populations: {}", populations)
    # Initialize compartments
    S = populations * (1 - initial_infected_prop)
    I = populations * initial_infected_prop
    R = jnp.zeros_like(populations)
    #jax.debug.print("initial S: {}", S)
    #jax.debug.print("initial I: {}", I)
    
    return S, I, R

@jax.jit
def maskSIR_IF_homogeneous(I: jnp.ndarray) -> jnp.ndarray:
    return jnp.full_like(I, jnp.sum(I))

@jax.jit
def maskSIR_IF_structured(I: jnp.ndarray, contact_matrix: jnp.ndarray) -> jnp.ndarray:
    return contact_matrix @ I

@jax.jit
def maskSIR_step(
    state: StateType,
    susceptibilities: jnp.ndarray,
    gamma: float,
    contact_matrix: jnp.ndarray = None,
    use_contact_matrix: bool = False,
    dT: float = 1.0
) -> StateType:
    S, I, R = state

    infection_force = jax.lax.cond(
        use_contact_matrix,
        lambda i: maskSIR_IF_structured(i, contact_matrix),
        lambda i: maskSIR_IF_homogeneous(i),
        state[1]
    )
    
    #jax.debug.print("infection_force: {}", infection_force)
    #jax.debug.print("susceptibilities: {}", susceptibilities)
    #jax.debug.print("S: {}", S)

    new_infections = susceptibilities * S * infection_force * dT
    new_recoveries = gamma * I * dT

    #jax.debug.print("new_infections: {}", new_infections)
    #jax.debug.print("new_recoveries: {}", new_recoveries)

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return S_new, I_new, R_new

@jax.jit
def execute_sim_maskSIR_final(
    initial_state: StateType,
    contact_matrix: jnp.ndarray,
    params: ParamType,
    n_steps: int,
    beta_params: Tuple[float, float],
    nan_state: StateType,
    use_contact_matrix: bool,
    susceptibilities: jnp.ndarray,
    mask_wearing: jnp.ndarray
) -> Tuple[StateType, float, float]:
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    POP = initial_state[0] + initial_state[1] + initial_state[2]
    obs_h = 0.0
    
    gamma = params['recovery_rate']
    r0 = jnp.where(is_valid, R0_maskSIR(params['beta_M'], mask_wearing, gamma, contact_matrix, POP), jnp.nan)

    def body_fun(_, state):
        return maskSIR_step(state, susceptibilities, gamma, contact_matrix, use_contact_matrix, dT=params["dT"])

    final_state = jax.lax.fori_loop(0, n_steps, body_fun, initial_state)
    
    return jax.tree_map(
        lambda v, n: jnp.where(is_valid, v, n),
        final_state,
        nan_state
    ), r0, obs_h

def sim_maskSIR_final(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0
) -> Tuple[StateType, float, float]:
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    POP = initial_state[0] + initial_state[1] + initial_state[2]
    C = create_contact_matrix(N_COMPARTMENTS, params['homophilic_tendency'], POP)
    
    use_contact_matrix = jax.lax.cond(
        jnp.logical_or(use_contact_matrix, params['homophilic_tendency'] != 0),
        lambda x: True,   # Use contact matrix if either condition is true
        lambda x: False,  # Only use homogeneous mixing if both conditions are false
        None
    )
    
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
    
    return execute_sim_maskSIR_final(
        initial_state, 
        C, 
        params, 
        n_steps, 
        beta_params, 
        nan_state, 
        use_contact_matrix,
        susceptibilities,
        mask_wearing
    )

def execute_sim_maskSIR_trajectory(*args, **kwargs):
    print("Warning: maskSIR trajectory simulation not yet implemented for mask-wearing model")
    raise NotImplementedError("Trajectory simulation not yet implemented for mask-wearing model")

def sim_maskSIR_trajectory(*args, **kwargs):
    print("Warning: maskSIR trajectory simulation not yet implemented for mask-wearing model")
    raise NotImplementedError("Trajectory simulation not yet implemented for mask-wearing model")