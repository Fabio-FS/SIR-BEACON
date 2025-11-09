import jax
import jax.numpy as jnp
from typing import Tuple, List, Dict, Any, Optional
from ..utils.model_utils import generate_behavior_pattern, calculate_infection_force, run_simulation as generic_run_simulation
from ..utils.R0 import power_iteration

# Type definitions
StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]  # S, I, R, V
ParamType = Dict[str, Any]

def get_default_params() -> Dict[str, Any]:
    """Return default parameters for the SIRV model"""
    return {
        'beta_M': 0.6,               # Maximum susceptibility
        'vaccination_rates': (0, 0.05),  # (min, max) vaccination rates
        'recovery_rate': 0.1,        # Recovery rate (gamma)
        'dT': 0.25,                  # Time step
        'homophilic_tendency': 0,    # Homophily parameter (h=0 gives uniform mixing)
        'SPB_exponent': 1.0          # Exponent for behavior pattern generation (1.0 = linear)
    }

@jax.jit
def step(state: StateType,
         susceptibilities: jnp.ndarray,
         gamma: float,
         vaccination_rates: jnp.ndarray,
         contact_matrix: Optional[jnp.ndarray] = None,
         use_contact_matrix: bool = False,
         dT: float = 1.0) -> StateType:
    """Execute one time step of the SIRV model"""
    S, I, R, V = state
    
    infection_force = calculate_infection_force(I, contact_matrix, use_contact_matrix)
    
    new_infections = susceptibilities * S * infection_force * dT
    new_recoveries = gamma * I * dT
    new_vaccinations = vaccination_rates * S * dT

    S_new = S - new_infections - new_vaccinations
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries
    V_new = V + new_vaccinations

    return S_new, I_new, R_new, V_new

@jax.jit
def create_ngm_SIRV(
    gamma: float,
    susceptibility: jnp.ndarray,
    populations: jnp.ndarray,
    C: jnp.ndarray
) -> jnp.ndarray:
    """Create next generation matrix from model parameters"""
    pop_fractions = populations / jnp.sum(populations)
    return (1/gamma) * jnp.diag(susceptibility * pop_fractions) @ C

def calculate_r0(susceptibilities: jnp.ndarray, 
                gamma: float, 
                contact_matrix: jnp.ndarray, 
                populations: jnp.ndarray) -> float:
    """Calculate R0 using the largest eigenvalue of the next generation matrix"""
    ngm = create_ngm_SIRV(gamma, susceptibilities, populations, contact_matrix)
    return power_iteration(ngm)


# metadata:
def get_compartment_info() -> Tuple[str, List[str]]:
    """Return model name and compartment names"""
    return "SIRV", ["S", "I", "R", "V"]




def prepare_step_params(params: ParamType, 
                       custom_behavior_distribution: Optional[jnp.ndarray], 
                       population_size: int) -> Dict[str, Any]:
    """Prepare parameters for the SIRV step function"""
    # Use provided behavior distribution or generate one for vaccination rates
    if custom_behavior_distribution is not None:
        vaccination_rates = custom_behavior_distribution
    else:
        # Generate vaccination rates across population
        min_rate = params.get('vaccination_rate_min', params.get('vaccination_rates', (0, 0.05))[0])
        max_rate = params.get('vaccination_rate_max', params.get('vaccination_rates', (0, 0.05))[1])
        vaccination_rates = generate_behavior_pattern(population_size, params, max_rate, min_val=min_rate)
    
    susceptibilities = jnp.full(population_size, params['beta_M'])
    gamma = params['recovery_rate']
    
    return {
        'susceptibilities': susceptibilities,
        'gamma': gamma,
        'vaccination_rates': vaccination_rates
    }

def adapter_step_fn(state: StateType, 
                   contact_matrix: jnp.ndarray, 
                   use_contact_matrix: bool, 
                   dT: float, 
                   **step_params) -> StateType:
    """Adapter function to connect SIRV step function with generic run_simulation"""
    return step(state, 
               step_params['susceptibilities'], 
               step_params['gamma'], 
               step_params['vaccination_rates'],
               contact_matrix, 
               use_contact_matrix, 
               dT)

def adapter_r0_calculation(step_params: Dict[str, Any], 
                          contact_matrix: jnp.ndarray, 
                          populations: jnp.ndarray) -> float:
    """Adapter function for R0 calculation"""
    return calculate_r0(step_params['susceptibilities'], 
                       step_params['gamma'],
                       contact_matrix, 
                       populations)

def run_simulation(beta_params: Tuple[float, float],
                  params: ParamType,
                  simulated_days: float,
                  initial_infected_prop: float = 1e-4,
                  population_size: int = 100,
                  use_contact_matrix: bool = False,
                  return_trajectory: bool = False,
                  custom_behavior_distribution: Optional[jnp.ndarray] = None,
                  custom_contact_matrix: Optional[jnp.ndarray] = None,
                  custom_populations: Optional[jnp.ndarray] = None) -> Tuple:
    """Run a complete SIRV simulation
    
    Args:
        beta_params: Parameters for the beta distribution of population
        params: Model parameters
        simulated_days: Number of simulated days
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        use_contact_matrix: Whether to use the contact matrix
        return_trajectory: Whether to return the full trajectory or just the final state
        custom_behavior_distribution: Optional custom vaccination rate distribution
        custom_contact_matrix: Optional custom contact matrix
        custom_populations: Optional custom population distribution
        
    Returns:
        If return_trajectory is True:
            Tuple of (states_trajectory, r0, obs_h)
        Otherwise:
            Tuple of (final_state, r0, obs_h)
    """
    return generic_run_simulation(
        model_step_fn=adapter_step_fn,
        r0_calculation_fn=adapter_r0_calculation,
        beta_params=beta_params,
        params=params,
        simulated_days=simulated_days,
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=use_contact_matrix,
        return_trajectory=return_trajectory,
        custom_behavior_distribution=custom_behavior_distribution,
        custom_contact_matrix=custom_contact_matrix,
        custom_populations=custom_populations,
        prepare_step_params_fn=prepare_step_params,
        n_health_states=4  # SIRV model has 4 health states (S, I, R, V)
    )