import jax
import jax.numpy as jnp
from typing import Tuple, List, Dict, Any, Optional
from ..utils.model_utils import generate_behavior_pattern, calculate_infection_force, run_simulation as generic_run_simulation
from ..utils.R0 import power_iteration

# Type definitions
StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
ParamType = Dict[str, Any]

def get_default_params() -> Dict[str, Any]:
    """Return default parameters for the SIRM model"""
    return {
        'beta_M': 0.35,           # Maximum susceptibility
        'mu_max': 1,           # Maximum mask-wearing value
        'recovery_rate': 0.1,    # Recovery rate (gamma)
        'dT': 1,              # Time step
        'homophilic_tendency': 0,  # Homophily parameter (h=0 gives uniform mixing)
        'SPB_exponent': 1.0      # Exponent for behavior pattern generation (1.0 = linear)
    }

 
@jax.jit
def step(state: StateType,
         susceptibilities: jnp.ndarray,
         gamma: float,
         contact_matrix: Optional[jnp.ndarray] = None,
         use_contact_matrix: bool = False,
         dT: float = 1.0) -> StateType:
    """Execute one time step of the SIRM model"""
    S, I, R = state
    
    infection_force = calculate_infection_force(I, contact_matrix, use_contact_matrix)
    #jax.debug.print("Contact matrix = {}", contact_matrix)
    #jax.debug.print("S = {}", S)
    #jax.debug.print("I = {}", I)
    #jax.debug.print("infection_force = {}", infection_force)
    #jax.debug.print("_____________")
    #jax.debug.print("I = {}", I)
    #jax.debug.print("infection_force = {}", infection_force)
    #jax.debug.print("contact_matrix = {}", contact_matrix)
    new_infections = susceptibilities * S * infection_force * dT
    new_recoveries = gamma * I * dT

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return S_new, I_new, R_new






# Things to calculate R0:

@jax.jit
def create_ngm_SIRM(
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
    ngm = create_ngm_SIRM(gamma, susceptibilities, populations, contact_matrix)
    return power_iteration(ngm)



# metadata:

def get_compartment_info() -> Tuple[str, List[str]]:
    """Return model name and compartment names"""
    return "SIRM", ["S", "I", "R"]



def prepare_step_params(params: ParamType, 
                       custom_behavior_distribution: Optional[jnp.ndarray], 
                       population_size: int) -> Dict[str, Any]:
    """Prepare parameters for the SIRM step function"""
    # Use provided behavior distribution or generate one
    if custom_behavior_distribution is not None:
        mask_wearing = custom_behavior_distribution
    else:
        # Get min and max mask-wearing values
        min_mask = params.get('mu_min', 0.0)
        max_mask = params['mu_max']
        mask_wearing = generate_behavior_pattern(population_size, params, max_mask, min_val=min_mask)
    
    susceptibilities = params['beta_M'] * (1 - mask_wearing)
    gamma = params['recovery_rate']
    
    return {
        'susceptibilities': susceptibilities,
        'gamma': gamma
    }

def adapter_step_fn(state: StateType, 
                   contact_matrix: jnp.ndarray, 
                   use_contact_matrix: bool, 
                   dT: float, 
                   **step_params) -> StateType:
    """Adapter function to connect SIRM step function with generic run_simulation"""
    return step(state, 
               step_params['susceptibilities'], 
               step_params['gamma'], 
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
    """Run a complete SIRM simulation
    
    Args:
        beta_params: Parameters for the beta distribution of population
        params: Model parameters
        simulated_days: Number of simulated days
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        use_contact_matrix: Whether to use the contact matrix
        return_trajectory: Whether to return the full trajectory or just the final state
        custom_behavior_distribution: Optional custom mask-wearing distribution
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