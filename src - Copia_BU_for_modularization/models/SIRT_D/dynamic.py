import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
from ...utils.distributions import homogeneous_distribution

StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
ParamType = Dict[str, Any]

def generate_testing_rates(test_range: Tuple[float, float], N_COMPARTMENTS: int, exponent: float = 1.0) -> jnp.ndarray:
    """Generate testing rate values with non-linear interpolation (default linear)"""
    test_low, test_high = test_range
    return test_high + jnp.power(homogeneous_distribution(N_COMPARTMENTS,0,1), exponent) * (test_low - test_high)

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
def SIRTD_step(
    state: StateType,
    susceptibility: float,
    gammaS: jnp.ndarray,
    dT: float = 1.0
) -> StateType:
    """Execute one time step of the SIRT_D model.
    In this model, each compartment only interacts with itself."""
    S, I, R = state
    N = S + I + R  # Population of each compartment
    
    new_infections = susceptibility * S * (I / N) * dT
    new_recoveries = gammaS * I * dT

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return S_new, I_new, R_new

@jax.jit
def execute_sim_SIRTD_final(
    initial_state: StateType,
    params: ParamType,
    n_steps: int,
    beta_params: Tuple[float, float],
    nan_state: StateType,
    testing_rates: jnp.ndarray
) -> Tuple[StateType, float, float]:
    """Execute complete simulation of SIRT_D model"""
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    gammaS = params['recovery_rate'] + testing_rates
    susceptibility = params['susceptibility_rate']

    def body_fun(_, state):
        return SIRTD_step(state, susceptibility, gammaS, dT=params["dT"])

    final_state = jax.lax.fori_loop(0, n_steps, body_fun, initial_state)
    
    return jax.tree_map(
        lambda v, n: jnp.where(is_valid, v, n),
        final_state,
        nan_state
    ), -1.0, 0.0

def sim_SIRTD_final(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0
) -> Tuple[StateType, float, float]:
    """Run complete SIRT_D simulation with final state output"""
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    
    nan_state = (
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan)
    )
    
    # Generate testing rates for each compartment
    testing_rates = generate_testing_rates(
        params['testing_rates'], 
        N_COMPARTMENTS,
        params['SPB_exponent']
    )
    
    return execute_sim_SIRTD_final(
        initial_state, 
        params, 
        n_steps, 
        beta_params, 
        nan_state,
        testing_rates
    )

def sim_SIRTD_trajectory(*args, **kwargs):
    """Run complete SIRT_D simulation with trajectory output"""
    print("Warning: SIRT_D trajectory simulation not yet implemented")
    raise NotImplementedError("Trajectory simulation not yet implemented for SIRT_D model")