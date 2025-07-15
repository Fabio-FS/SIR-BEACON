import jax
import jax.numpy as jnp
from typing import Tuple, List, Dict, Any, Optional
from ..utils.model_utils import generate_behavior_pattern, calculate_infection_force, run_simulation as generic_run_simulation
from ..utils.R0 import power_iteration

# Type definitions
StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
ParamType = Dict[str, Any]

def get_default_params() -> Dict[str, Any]:
    """Return default parameters for the SIRT model"""
    return {
        'beta_M': 0.2,           # Maximum susceptibility
        'testing_rates': (0, 0.05),  # (min, max) testing rates
        'recovery_rate': 0.1,    # Base recovery rate (gamma)
        'dT': 0.25,              # Time step
        'homophilic_tendency': 0,  # Homophily parameter (h=0 gives uniform mixing)
        'SPB_exponent': 1.0      # Exponent for behavior pattern generation (1.0 = linear)
    }

@jax.jit
def step(state: StateType,
         susceptibilities: jnp.ndarray,
         gamma: float,
         testing_effect: jnp.ndarray,
         contact_matrix: Optional[jnp.ndarray] = None,
         use_contact_matrix: bool = False,
         dT: float = 1.0) -> StateType:
    """Execute one time step of the SIRT model"""
    S, I, R = state
    
    infection_force = calculate_infection_force(I, contact_matrix, use_contact_matrix)
    
    new_infections = susceptibilities * S * infection_force * dT
    
    # Testing increases recovery rate
    effective_gamma = gamma + testing_effect
    new_recoveries = effective_gamma * I * dT

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return S_new, I_new, R_new

@jax.jit
def create_ngm_SIRT(
    gamma: float,
    susceptibility: jnp.ndarray,
    testing_effect: jnp.ndarray,
    populations: jnp.ndarray,
    C: jnp.ndarray
) -> jnp.ndarray:
    """Create next generation matrix from model parameters"""
    pop_fractions = populations / jnp.sum(populations)
    effective_gamma = gamma + testing_effect
    return jnp.diag(susceptibility * pop_fractions) @ jnp.diag(1/effective_gamma) @ C

def calculate_r0(susceptibilities: jnp.ndarray, 
                gamma: float, 
                testing_effect: jnp.ndarray,
                contact_matrix: jnp.ndarray, 
                populations: jnp.ndarray) -> float:
    """Calculate R0 using the largest eigenvalue of the next generation matrix"""
    ngm = create_ngm_SIRT(gamma, susceptibilities, testing_effect, populations, contact_matrix)
    return power_iteration(ngm)


# metadata:
def get_compartment_info() -> Tuple[str, List[str]]:
    """Return model name and compartment names"""
    return "SIRT", ["S", "I", "R"]



def prepare_step_params(params: ParamType, 
                       custom_behavior_distribution: Optional[jnp.ndarray], 
                       population_size: int) -> Dict[str, Any]:
    """Prepare parameters for the SIRT step function"""
    # Use provided behavior distribution or generate one for testing rates
    if custom_behavior_distribution is not None:
        testing_rates = custom_behavior_distribution
    else:
        # Get min and max testing rates
        min_rate = params.get('testing_rate_min', 0.0)
        # Handle both formats: testing_rate_max or testing_rates tuple
        if 'testing_rate_max' in params:
            max_rate = params['testing_rate_max']
        else:
            max_rate = params.get('testing_rates', (0, 0.05))[1]
            
        # Generate testing rates that vary across population based on position in the distribution
        # This ensures polarization affects testing behavior
        testing_rates = generate_behavior_pattern(population_size, params, max_rate, min_val=min_rate)
    
    # Susceptibility is uniform across population
    susceptibilities = jnp.full(population_size, params['beta_M'])
    gamma = params['recovery_rate']
    
    # Calculate testing effect on recovery rate
    testing_effect = testing_rates
    
    return {
        'susceptibilities': susceptibilities,
        'gamma': gamma,
        'testing_effect': testing_effect
    }

def adapter_step_fn(state: StateType, 
                   contact_matrix: jnp.ndarray, 
                   use_contact_matrix: bool, 
                   dT: float, 
                   **step_params) -> StateType:
    """Adapter function to connect SIRT step function with generic run_simulation"""
    return step(state, 
               step_params['susceptibilities'], 
               step_params['gamma'], 
               step_params['testing_effect'],
               contact_matrix, 
               use_contact_matrix, 
               dT)

def adapter_r0_calculation(step_params: Dict[str, Any], 
                          contact_matrix: jnp.ndarray, 
                          populations: jnp.ndarray) -> float:
    """Adapter function for R0 calculation"""
    return calculate_r0(step_params['susceptibilities'], 
                       step_params['gamma'],
                       step_params['testing_effect'],
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
    """Run a complete SIRT simulation
    
    Args:
        beta_params: Parameters for the beta distribution of population
        params: Model parameters
        simulated_days: Number of simulated days
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        use_contact_matrix: Whether to use the contact matrix
        return_trajectory: Whether to return the full trajectory or just the final state
        custom_behavior_distribution: Optional custom testing rate distribution
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
        n_health_states=3  # SIR model has 3 health states
    )