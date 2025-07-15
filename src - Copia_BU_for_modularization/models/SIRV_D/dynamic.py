import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
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
def SIRVD_step(
    state: StateType,
    susceptibility: float,
    gamma: float,
    vaccination_rates: jnp.ndarray,
    dT: float = 1.0
) -> StateType:
    """Execute one time step of the SIRV_D model.
    In this model, each compartment only interacts with itself."""
    S, I, R, V = state
    N = S + I + R + V  # Total population in each compartment
    
    # Vaccination process (from S to V)
    new_vaccinations = vaccination_rates * S * dT
    
    # Infection process (from S to I)
    new_infections = susceptibility * S * (I / N) * dT
    
    # Recovery process (from I to R)
    new_recoveries = gamma * I * dT
    
    S_new = S - new_infections - new_vaccinations
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries
    V_new = V + new_vaccinations
    
    return S_new, I_new, R_new, V_new

@jax.jit
def execute_sim_SIRVD_final(
    initial_state: StateType,
    params: ParamType,
    n_steps: int,
    beta_params: Tuple[float, float],
    nan_state: StateType,
    vaccination_rates: jnp.ndarray
) -> Tuple[StateType, float, float]:
    """Execute complete simulation of SIRV_D model"""
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    gamma = params['recovery_rate']
    susceptibility = params['susceptibility_rate']

    def body_fun(_, state):
        return SIRVD_step(
            state,
            susceptibility,
            gamma,
            vaccination_rates,
            dT=params["dT"]
        )

    final_state = jax.lax.fori_loop(0, n_steps, body_fun, initial_state)
    
    return jax.tree_map(
        lambda v, n: jnp.where(is_valid, v, n),
        final_state,
        nan_state
    ), -1.0, 0.0

def sim_SIRVD_final(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0
) -> Tuple[StateType, float, float]:
    """Run complete SIRV_D simulation with final state output"""
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    
    nan_state = (
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan)
    )
    
    # Generate vaccination rates for each compartment
    vaccination_rates = generate_vaccination_rates(
        params['vaccination_rates'], 
        N_COMPARTMENTS,
        params['SPB_exponent']
    )
    
    return execute_sim_SIRVD_final(
        initial_state, 
        params, 
        n_steps, 
        beta_params, 
        nan_state,
        vaccination_rates
    )

def sim_SIRVD_trajectory(*args, **kwargs):
    """Run complete SIRV_D simulation with trajectory output"""
    print("Warning: SIRV_D trajectory simulation not yet implemented")
    raise NotImplementedError("Trajectory simulation not yet implemented for SIRV_D model")