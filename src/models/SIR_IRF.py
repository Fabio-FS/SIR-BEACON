import jax
import jax.numpy as jnp
from typing import Tuple, List, Dict, Any, Optional
from ..utils.model_utils import run_simulation as generic_run_simulation
from ..utils.R0 import power_iteration

# Type definitions
StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]  # S, I, R, behaviors
ParamType = Dict[str, Any]

def get_default_params() -> Dict[str, Any]:
    """Return default parameters for the SIR_IRF model"""
    return {
        'beta_M': 0.2,           # Maximum susceptibility
        'recovery_rate': 0.1,     # Recovery rate (gamma)
        'dT': 1,                  # Time step
        'logistic_k': 10.0,       # Steepness of the logistic function for behavior change
        'logistic_x0': 0.1,       # Midpoint of the logistic function for behavior change
        'inertia': [0.8, 0.8, 1.0, 1.0, 1.0],  # Inertia parameters for each behavior class
                                  # [Rational, Herder, Never, Always, Meh]
        'behavior_distribution': [0.3, 0.3, 0.1, 0.1, 0.2]  # Proportion of each behavior class
                                  # [Rational, Herder, Never, Always, Meh]
    }

@jax.jit
def calculate_transition_probabilities(
    current_behavior: jnp.ndarray,  # Current behavior state (mu_i)
    prevalence: jnp.ndarray,        # Current infection prevalence (I/N)
    fraction_protecting: float,      # Fraction of total population currently self-protecting
    behavior_types: jnp.ndarray,     # Array of behavior types (0=Rational, 1=Herder, 2=Never, 3=Always, 4=Meh)
    inertia: jnp.ndarray,            # Inertia parameters for each behavior type
    logistic_k: float,               # Steepness of the logistic function
    logistic_x0: float               # Midpoint of the logistic function
) -> jnp.ndarray:
    """
    Calculate transition probabilities for behavioral states based on behavior type,
    current prevalence, and fraction of population that is protecting.
    
    Args:
        current_behavior: Current behavior state (mu_i)
        prevalence: Current infection prevalence (I/N) for each population group
        fraction_protecting: Fraction of total population currently self-protecting
        behavior_types: Array of behavior types (0=Rational, 1=Herder, 2=Never, 3=Always, 4=Meh)
        inertia: Inertia parameters for each behavior type
        logistic_k: Steepness of the logistic function
        logistic_x0: Midpoint of the logistic function
        
    Returns:
        New behavior values (mu_i)
    """
    # Initialize new behavior as a copy of current
    new_behavior = current_behavior
    
    # Logistic function: f(x) = 1 / (1 + exp(-k * (x - x0)))
    # For Rationals: responds to prevalence positively, fraction_protecting negatively
    # For Herders: responds to both prevalence and fraction_protecting positively
    
    # Calculate base probability from prevalence using logistic function
    base_prob_prevalence = 1.0 / (1.0 + jnp.exp(-logistic_k * (prevalence - logistic_x0)))
    
    # Calculate influence from others' behavior using logistic function
    base_prob_others = 1.0 / (1.0 + jnp.exp(-logistic_k * (fraction_protecting - logistic_x0)))
    
    # Process each behavior type
    
    # Rational: increases with prevalence, decreases with others' protection
    rational_mask = (behavior_types == 0)
    rational_inertia = inertia[0]
    rational_prob = base_prob_prevalence * (1.0 - base_prob_others)
    new_behavior = jnp.where(
        rational_mask,
        (1 - rational_inertia) * rational_prob + rational_inertia * current_behavior,
        new_behavior
    )
    
    # Herder: increases with both prevalence and others' protection
    herder_mask = (behavior_types == 1)
    herder_inertia = inertia[1]
    herder_prob = base_prob_prevalence * base_prob_others
    new_behavior = jnp.where(
        herder_mask,
        (1 - herder_inertia) * herder_prob + herder_inertia * current_behavior,
        new_behavior
    )
    
    # Never: always 0
    never_mask = (behavior_types == 2)
    new_behavior = jnp.where(never_mask, 0.0, new_behavior)
    
    # Always: always 1
    always_mask = (behavior_types == 3)
    new_behavior = jnp.where(always_mask, 1.0, new_behavior)
    
    # Meh: always 0.5
    meh_mask = (behavior_types == 4)
    new_behavior = jnp.where(meh_mask, 0.5, new_behavior)
    
    return new_behavior

@jax.jit
def step(state: StateType,
         beta_M: float,
         gamma: float,
         behavior_types: jnp.ndarray,
         inertia: jnp.ndarray,
         logistic_k: float,
         logistic_x0: float,
         dT: float = 1.0) -> StateType:
    """Execute one time step of the SIR_IRF model with well-mixed population"""
    S, I, R, behaviors = state
    
    # Total population
    N = S + I + R
    
    # Calculate prevalence (I/N) for each group
    prevalence = I / N
    
    # Calculate fraction of the total population that is currently self-protecting
    # This is weighted by population size
    pop_weighted_behaviors = behaviors * N
    fraction_protecting = jnp.sum(pop_weighted_behaviors) / jnp.sum(N)
    
    # Update behaviors based on current state
    new_behaviors = calculate_transition_probabilities(
        behaviors, prevalence, fraction_protecting,
        behavior_types, inertia, logistic_k, logistic_x0
    )
    
    # Calculate susceptibilities based on behaviors
    susceptibilities = beta_M * (1.0 - new_behaviors)
    
    # Calculate force of infection in a well-mixed population
    # All compartments see the same total infected population
    total_I = jnp.sum(I)
    total_N = jnp.sum(N)
    infection_force = total_I / total_N
    
    # Calculate transitions
    new_infections = susceptibilities * S * infection_force * dT
    new_recoveries = gamma * I * dT

    # Update compartments
    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return S_new, I_new, R_new, new_behaviors

# Next generation matrix for R0 calculation
@jax.jit
def create_ngm_SIR_IRF(
    gamma: float,
    susceptibility: jnp.ndarray,
    populations: jnp.ndarray
) -> jnp.ndarray:
    """Create next generation matrix from model parameters for well-mixed population"""
    # For a well-mixed population, we create a matrix where everyone contacts everyone
    n = len(populations)
    pop_fractions = populations / jnp.sum(populations)
    
    # In a well-mixed population, the contact matrix is all ones
    # Each column sums to 1 (proper probability distribution)
    C = jnp.ones((n, n)) / n
    
    return (1/gamma) * jnp.diag(susceptibility * pop_fractions) @ C

def calculate_r0(susceptibilities: jnp.ndarray, 
                gamma: float, 
                populations: jnp.ndarray) -> float:
    """Calculate R0 using the largest eigenvalue of the next generation matrix"""
    ngm = create_ngm_SIR_IRF(gamma, susceptibilities, populations)
    return power_iteration(ngm)

# metadata:
def get_compartment_info() -> Tuple[str, List[str]]:
    """Return model name and compartment names"""
    return "SIR_IRF", ["S", "I", "R", "B"]

def initialize_behavior_types(population_size: int, behavior_distribution: List[float]) -> jnp.ndarray:
    """
    Initialize behavior types for the population based on distribution.
    
    Args:
        population_size: Number of population compartments
        behavior_distribution: Proportion of each behavior type [Rational, Herder, Never, Always, Meh]
        
    Returns:
        Array of behavior types (0=Rational, 1=Herder, 2=Never, 3=Always, 4=Meh)
    """
    # Ensure distribution sums to 1
    behavior_distribution = jnp.array(behavior_distribution)
    behavior_distribution = behavior_distribution / jnp.sum(behavior_distribution)
    
    # Calculate cumulative distribution
    cum_dist = jnp.cumsum(behavior_distribution)
    
    # Generate evenly spaced population groups
    positions = jnp.linspace(0, 1, num=population_size, endpoint=False) + 1/(2*population_size)
    
    # Assign behavior types based on position in distribution
    behavior_types = jnp.zeros(population_size, dtype=jnp.int32)
    
    # Rational (type 0)
    behavior_types = jnp.where(positions < cum_dist[0], 0, behavior_types)
    
    # Herder (type 1)
    behavior_types = jnp.where(
        (positions >= cum_dist[0]) & (positions < cum_dist[1]), 
        1, 
        behavior_types
    )
    
    # Never (type 2)
    behavior_types = jnp.where(
        (positions >= cum_dist[1]) & (positions < cum_dist[2]), 
        2, 
        behavior_types
    )
    
    # Always (type 3)
    behavior_types = jnp.where(
        (positions >= cum_dist[2]) & (positions < cum_dist[3]), 
        3, 
        behavior_types
    )
    
    # Meh (type 4)
    behavior_types = jnp.where(positions >= cum_dist[3], 4, behavior_types)
    
    return behavior_types

def initialize_behaviors(behavior_types: jnp.ndarray) -> jnp.ndarray:
    """
    Initialize behavior values based on behavior types.
    
    Args:
        behavior_types: Array of behavior types (0=Rational, 1=Herder, 2=Never, 3=Always, 4=Meh)
        
    Returns:
        Initial behavior values (mu_i)
    """
    # Start with small random values for Rational and Herder
    behaviors = jnp.random.uniform(key=jax.random.PRNGKey(0), shape=behavior_types.shape, minval=0.0, maxval=0.1)
    
    # Set fixed values for Never, Always, and Meh
    behaviors = jnp.where(behavior_types == 2, 0.0, behaviors)  # Never
    behaviors = jnp.where(behavior_types == 3, 1.0, behaviors)  # Always
    behaviors = jnp.where(behavior_types == 4, 0.5, behaviors)  # Meh
    
    return behaviors

def prepare_step_params(params: ParamType, 
                       custom_behavior_distribution: Optional[jnp.ndarray], 
                       population_size: int) -> Dict[str, Any]:
    """Prepare parameters for the SIR_IRF step function"""
    
    # Initialize behavior types
    if custom_behavior_distribution is not None:
        behavior_types = custom_behavior_distribution
    else:
        # Use default behavior distribution from params
        behavior_types = initialize_behavior_types(
            population_size, 
            params.get('behavior_distribution', [0.3, 0.3, 0.1, 0.1, 0.2])
        )
    
    # Get other parameters
    beta_M = params['beta_M']
    gamma = params['recovery_rate']
    inertia = jnp.array(params.get('inertia', [0.8, 0.8, 1.0, 1.0, 1.0]))
    logistic_k = params.get('logistic_k', 10.0)
    logistic_x0 = params.get('logistic_x0', 0.1)
    
    return {
        'beta_M': beta_M,
        'gamma': gamma,
        'behavior_types': behavior_types,
        'inertia': inertia,
        'logistic_k': logistic_k,
        'logistic_x0': logistic_x0
    }

def adapter_step_fn(state: StateType, 
                   contact_matrix: jnp.ndarray, 
                   use_contact_matrix: bool, 
                   dT: float, 
                   **step_params) -> StateType:
    """Adapter function to connect SIR_IRF step function with generic run_simulation"""
    # For this model, we ignore contact_matrix and use_contact_matrix
    return step(state, 
               step_params['beta_M'], 
               step_params['gamma'],
               step_params['behavior_types'],
               step_params['inertia'],
               step_params['logistic_k'],
               step_params['logistic_x0'],
               dT)

def adapter_r0_calculation(step_params: Dict[str, Any], 
                          contact_matrix: jnp.ndarray, 
                          populations: jnp.ndarray) -> float:
    """Adapter function for R0 calculation"""
    # Calculate initial susceptibilities based on behavior types
    behaviors = initialize_behaviors(step_params['behavior_types'])
    susceptibilities = step_params['beta_M'] * (1.0 - behaviors)
    
    # Ignore contact_matrix for well-mixed population
    return calculate_r0(susceptibilities, step_params['gamma'], populations)

def create_initial_state(populations: jnp.ndarray, 
                        initial_infected_prop: float,
                        behavior_types: jnp.ndarray) -> StateType:
    """Create initial state for the SIR_IRF model"""
    # Initialize SIR compartments
    S = populations * (1 - initial_infected_prop)
    I = populations * initial_infected_prop
    R = jnp.zeros_like(populations)
    
    # Initialize behaviors based on types
    behaviors = initialize_behaviors(behavior_types)
    
    return S, I, R, behaviors

def run_simulation(beta_params: Tuple[float, float],
                  params: ParamType,
                  simulated_days: float,
                  initial_infected_prop: float = 1e-4,
                  population_size: int = 100,
                  use_contact_matrix: bool = False,  # Ignored for well-mixed population
                  return_trajectory: bool = False,
                  custom_behavior_distribution: Optional[jnp.ndarray] = None,
                  custom_contact_matrix: Optional[jnp.ndarray] = None,  # Ignored for well-mixed population
                  custom_populations: Optional[jnp.ndarray] = None) -> Tuple:
    """Run a complete SIR_IRF simulation
    
    Args:
        beta_params: Parameters for the beta distribution of population
        params: Model parameters
        simulated_days: Number of simulated days
        initial_infected_prop: Initial proportion of infected individuals
        population_size: Number of population compartments
        use_contact_matrix: Ignored for well-mixed population
        return_trajectory: Whether to return the full trajectory or just the final state
        custom_behavior_distribution: Optional custom behavior distribution
        custom_contact_matrix: Ignored for well-mixed population
        custom_populations: Optional custom population distribution
        
    Returns:
        If return_trajectory is True:
            Tuple of (states_trajectory, r0, obs_h)
        Otherwise:
            Tuple of (final_state, r0, obs_h)
    """
    step_params = prepare_step_params(params, custom_behavior_distribution, population_size)
    
    # Create custom initial state function
    def custom_init_fn(populations, initial_infected_prop, n_health_states):
        return create_initial_state(populations, initial_infected_prop, step_params['behavior_types'])
    
    return generic_run_simulation(
        model_step_fn=adapter_step_fn,
        r0_calculation_fn=adapter_r0_calculation,
        beta_params=beta_params,
        params=params,
        simulated_days=simulated_days,
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=False,  # Always false for well-mixed population
        return_trajectory=return_trajectory,
        custom_behavior_distribution=None,  # Already handled in prepare_step_params
        custom_contact_matrix=None,  # Ignored for well-mixed population
        custom_populations=custom_populations,
        prepare_step_params_fn=prepare_step_params,
        n_health_states=4,  # S, I, R, behaviors
        custom_init_fn=custom_init_fn  # Use custom initialization
    )