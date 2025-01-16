import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from ...utils.Contact_Matrix import create_contact_matrix
from ...utils.R0 import R0_SIRM  ##??## # Should there be a specific R0_SIRV function?
from ...utils.distributions import homogeneous_distribution

StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
ParamType = Dict[str, Any]

def generate_vaccination_rates(vacc_range: Tuple[float, float], N_COMPARTMENTS: int, exponent: float = 1.0) -> jnp.ndarray:
    """Generate vaccination values with non-linear interpolation (default linear)"""
    vacc_low, vacc_high = vacc_range
    return vacc_high + jnp.power(homogeneous_distribution(N_COMPARTMENTS,0,1), exponent) * (vacc_low - vacc_high)

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
    V = jnp.zeros_like(populations)
    
    return S, I, R, V

@jax.jit
def SIRV_IF_homogeneous(I: jnp.ndarray) -> jnp.ndarray:
    # Return an array filled with the sum, matching the shape of input I
    return jnp.full_like(I, jnp.sum(I))

@jax.jit
def SIRV_IF_structured(I: jnp.ndarray, contact_matrix: jnp.ndarray) -> jnp.ndarray:
    return contact_matrix @ I

@jax.jit
def SIRV_step(
    state: StateType,
    vaccination_rates: jnp.ndarray,
    susceptibility: float,
    gamma: float,
    contact_matrix: jnp.ndarray = None,
    use_contact_matrix: bool = False,
    dT: float = 1.0
) -> StateType:
    S, I, R, V = state
    
    # Vaccination process
    new_vaccinations = vaccination_rates * S * dT
    
    infection_force = jax.lax.cond(
        use_contact_matrix,
        lambda i: SIRV_IF_structured(i, contact_matrix),
        lambda i: SIRV_IF_homogeneous(i),
        I
    )
    
    new_infections = susceptibility * S * infection_force * dT
    new_recoveries = gamma * I * dT
    
    S_new = S - new_infections - new_vaccinations
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries
    V_new = V + new_vaccinations
    
    return S_new, I_new, R_new, V_new

@jax.jit
def execute_sim_SIRV_final(
    initial_state: StateType,
    contact_matrix: jnp.ndarray,
    params: ParamType,
    n_steps: int,
    beta_params: Tuple[float, float],
    nan_state: StateType,
    use_contact_matrix: bool,
    vaccination_rates: jnp.ndarray
) -> Tuple[StateType, float, float]:
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    POP = initial_state[0] + initial_state[1] + initial_state[2] + initial_state[3]
    r0 = jnp.where(is_valid, R0_SIRM(jnp.full_like(POP, params['susceptibility_rate']), params['recovery_rate'], contact_matrix, POP), jnp.nan)  ##??## # Check if R0 calculation is correct for SIRV
    obs_h = 0.0  ##??## # Do we need observable homophily for SIRV?
    
    gamma = params['recovery_rate']
    susceptibility = params['susceptibility_rate']
    
    def body_fun(_, state):
        return SIRV_step(
            state,
            vaccination_rates,
            susceptibility,
            gamma,
            contact_matrix,
            use_contact_matrix,
            dT=params["dT"]
        )
    
    final_state = jax.lax.fori_loop(0, n_steps, body_fun, initial_state)
    
    return jax.tree_map(
        lambda v, n: jnp.where(is_valid, v, n),
        final_state,
        nan_state
    ), r0, obs_h

def sim_SIRV_final(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0
) -> Tuple[StateType, float, float]:
    params['SPB_exponent'] = SPB_exponent
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    POP = initial_state[0] + initial_state[1] + initial_state[2] + initial_state[3]
    C = create_contact_matrix(N_COMPARTMENTS, params['homophilic_tendency'], POP)
    
    use_contact_matrix = jax.lax.cond(
        params['homophilic_tendency'] != 0,
        lambda x: True,   # Use contact matrix if homophily is non-zero
        lambda x: False,  # Only use homogeneous mixing if homophily is zero
        None
    )
    
    nan_state = (
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan)
    )
    
    vaccination_rates = generate_vaccination_rates(
        params['vaccination_rates'], 
        N_COMPARTMENTS,
        params['SPB_exponent']
    )
    
    return execute_sim_SIRV_final(
        initial_state, 
        C, 
        params, 
        n_steps, 
        beta_params, 
        nan_state, 
        use_contact_matrix,
        vaccination_rates
    )

def execute_sim_SIRV_trajectory(
    initial_state: StateType,
    contact_matrix: jnp.ndarray,
    params: ParamType,
    n_steps: int,
    beta_params: Tuple[float, float],
    trajectories: jnp.ndarray,
    nan_trajectories: StateType,
    use_contact_matrix: bool,
    vaccination_rates: jnp.ndarray
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], float, float]:
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    POP = initial_state[0] + initial_state[1] + initial_state[2] + initial_state[3]
    r0 = jnp.where(is_valid, R0_SIRM(jnp.full_like(POP, params['susceptibility_rate']), params['recovery_rate'], contact_matrix, POP), jnp.nan)
    obs_h = 0.0  ##??## # Same question about observable homophily
    
    gamma = params['recovery_rate']
    susceptibility = params['susceptibility_rate']
    
    trajectories = trajectories.at[0, 0].set(initial_state[0])
    trajectories = trajectories.at[1, 0].set(initial_state[1])
    trajectories = trajectories.at[2, 0].set(initial_state[2])
    trajectories = trajectories.at[3, 0].set(initial_state[3])
    
    def body_fun(carry, t):
        state, trajectories = carry
        next_state = SIRV_step(
            state,
            vaccination_rates,
            susceptibility,
            gamma,
            contact_matrix,
            use_contact_matrix,
            params["dT"]
        )
        trajectories = trajectories.at[0, t + 1].set(next_state[0])
        trajectories = trajectories.at[1, t + 1].set(next_state[1])
        trajectories = trajectories.at[2, t + 1].set(next_state[2])
        trajectories = trajectories.at[3, t + 1].set(next_state[3])
        return (next_state, trajectories), None
    
    (_, trajectories), _ = jax.lax.scan(
        body_fun,
        (initial_state, trajectories),
        jnp.arange(n_steps)
    )
    
    valid_result = (trajectories[0], trajectories[1], trajectories[2], trajectories[3])
    
    return jax.tree_map(
        lambda v, n: jnp.where(is_valid, v, n),
        valid_result,
        nan_trajectories
    ), r0, obs_h

def sim_SIRV_trajectory(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], float, float]:
    params['SPB_exponent'] = SPB_exponent
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    POP = initial_state[0] + initial_state[1] + initial_state[2] + initial_state[3]
    C = create_contact_matrix(N_COMPARTMENTS, params['homophilic_tendency'], POP)
    
    trajectories = jnp.zeros((4, n_steps + 1, N_COMPARTMENTS))
    nan_trajectories = jnp.full((4, n_steps + 1, N_COMPARTMENTS), jnp.nan)
    
    vaccination_rates = generate_vaccination_rates(
        params['vaccination_rates'], 
        N_COMPARTMENTS,
        params['SPB_exponent']
    )
    
    use_contact_matrix = jax.lax.cond(
        params['homophilic_tendency'] != 0,
        lambda x: True,
        lambda x: False,
        None
    )
    
    return execute_sim_SIRV_trajectory(
        initial_state, 
        C, 
        params, 
        n_steps, 
        beta_params,
        trajectories, 
        nan_trajectories,
        use_contact_matrix,
        vaccination_rates
    )