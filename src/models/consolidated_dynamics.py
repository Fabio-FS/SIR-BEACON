import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Callable, Optional
from ..utils.Contact_Matrix import create_contact_matrix
from ..utils.R0 import R0_maskSIR, R0_SIRT, R0_SIRM

# Common type definitions
StateType = Tuple[jnp.ndarray, ...]
ParamType = Dict[str, Any]

# Common pattern generation function
def generate_behavior_pattern(max_value: float, N_COMPARTMENTS: int, exponent: float = 1.0) -> jnp.ndarray:
    """Generate behavior pattern values with non-linear interpolation"""
    x = jnp.linspace(0, 1, num=N_COMPARTMENTS)
    return max_value * jnp.power(x, exponent)

def initialize_states(
    beta_params: Tuple[float, float],
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    include_vaccinated: bool = False
) -> StateType:
    """Initialize states for fixed-size system normalized to total population 1"""
    from ..utils.distributions import my_beta_asymmetric
    
    # Generate normalized population distribution
    populations = my_beta_asymmetric(beta_params[0], beta_params[1], N_COMPARTMENTS, norm=1.0)
    
    # Initialize compartments
    S = populations * (1 - initial_infected_prop)
    I = populations * initial_infected_prop
    R = jnp.zeros_like(populations)
    
    if include_vaccinated:
        V = jnp.zeros_like(populations)
        return S, I, R, V
    
    return S, I, R

@jax.jit
def calculate_infection_force(I: jnp.ndarray, contact_matrix: Optional[jnp.ndarray] = None, 
                            use_contact_matrix: bool = False) -> jnp.ndarray:
    """Calculate force of infection with or without contact structure"""
    # Use jax.lax.cond instead of if/else
    return jax.lax.cond(
        jnp.logical_and(use_contact_matrix, contact_matrix is not None),
        lambda _: contact_matrix @ I,
        lambda _: jnp.full_like(I, jnp.sum(I)),
        operand=None
    )

# Model-specific step functions
@jax.jit
def mask_sir_step(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    susceptibilities: jnp.ndarray,
    gamma: float,
    contact_matrix: Optional[jnp.ndarray] = None,
    use_contact_matrix: bool = False,
    dT: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Execute one time step of the SIR model with mask-wearing"""
    S, I, R = state
    
    infection_force = calculate_infection_force(I, contact_matrix, use_contact_matrix)
    
    new_infections = susceptibilities * S * infection_force * dT
    new_recoveries = gamma * I * dT

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return S_new, I_new, R_new

@jax.jit
def testing_sir_step(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    susceptibility: float,
    gammaS: jnp.ndarray,
    contact_matrix: Optional[jnp.ndarray] = None,
    use_contact_matrix: bool = False,
    dT: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Execute one time step of the SIR model with testing-based isolation"""
    S, I, R = state

    infection_force = calculate_infection_force(I, contact_matrix, use_contact_matrix)

    new_infections = susceptibility * S * infection_force * dT
    new_recoveries = gammaS * I * dT

    S_new = S - new_infections
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries

    return S_new, I_new, R_new

@jax.jit
def vaccination_sir_step(
    state: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    vaccination_rates: jnp.ndarray,
    susceptibility: float,
    gamma: float,
    contact_matrix: Optional[jnp.ndarray] = None,
    use_contact_matrix: bool = False,
    dT: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Execute one time step of the SIR model with vaccination"""
    S, I, R, V = state
    
    # Vaccination process
    new_vaccinations = vaccination_rates * S * dT
    
    infection_force = calculate_infection_force(I, contact_matrix, use_contact_matrix)
    
    new_infections = susceptibility * S * infection_force * dT
    new_recoveries = gamma * I * dT
    
    S_new = S - new_infections - new_vaccinations
    I_new = I + new_infections - new_recoveries
    R_new = R + new_recoveries
    V_new = V + new_vaccinations
    
    return S_new, I_new, R_new, V_new



# Main simulation functions
def sim_maskSIR_final(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0
) -> Tuple[StateType, float, float]:
    """Run complete maskSIR simulation with final state output"""
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    POP = initial_state[0] + initial_state[1] + initial_state[2]
    C = create_contact_matrix(N_COMPARTMENTS, params['homophilic_tendency'], POP)
    
    use_contact_matrix = jax.lax.cond(
        jnp.logical_or(use_contact_matrix, params['homophilic_tendency'] != 0),
        lambda x: True,
        lambda x: False,
        None
    )
    
    nan_state = (
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan)
    )
    
    # Generate mask wearing behavior and calculate susceptibilities
    mask_wearing = generate_behavior_pattern(
        params['mu_max'],
        N_COMPARTMENTS,
        SPB_exponent
    )
    
    susceptibilities = params['beta_M'] * (1 - mask_wearing)
    
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    gamma = params['recovery_rate']
    r0 = jnp.where(is_valid, R0_maskSIR(params['beta_M'], mask_wearing, gamma, C, POP), jnp.nan)
    obs_h = 0.0
    
    def body_fun(_, state):
        return mask_sir_step(state, susceptibilities, gamma, C, use_contact_matrix, dT=params["dT"])

    final_state = jax.lax.fori_loop(0, n_steps, body_fun, initial_state)
    
    return jax.tree_map(
        lambda v, n: jnp.where(is_valid, v, n),
        final_state,
        nan_state
    ), r0, obs_h

def sim_SIRT_final(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0
) -> Tuple[StateType, float, float]:
    """Run complete SIRT simulation with final state output"""
    params['SPB_exponent'] = SPB_exponent
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
    POP = initial_state[0] + initial_state[1] + initial_state[2]
    C = create_contact_matrix(N_COMPARTMENTS, params['homophilic_tendency'], POP)
    
    use_contact_matrix = jax.lax.cond(
        params['homophilic_tendency'] != 0,
        lambda x: True,
        lambda x: False,
        None
    )
    
    nan_state = (
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan)
    )
    
    testing_rates = generate_behavior_pattern(
        params['testing_rates'][1],
        N_COMPARTMENTS,
        params['SPB_exponent']
    )
    
    gammaS = params['recovery_rate'] + testing_rates
    
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    susceptibility = params['susceptibility_rate']
    r0 = jnp.where(is_valid, R0_SIRT(susceptibility, gammaS, C, POP), jnp.nan)
    obs_h = 0.0
    
    def body_fun(_, state):
        return testing_sir_step(
            state, 
            susceptibility,
            gammaS, 
            C, 
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
    """Run complete SIRV simulation with final state output"""
    params['SPB_exponent'] = SPB_exponent
    initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS, include_vaccinated=True)
    POP = initial_state[0] + initial_state[1] + initial_state[2] + initial_state[3]
    C = create_contact_matrix(N_COMPARTMENTS, params['homophilic_tendency'], POP)
    
    use_contact_matrix = jax.lax.cond(
        params['homophilic_tendency'] != 0,
        lambda x: True,
        lambda x: False,
        None
    )
    
    nan_state = (
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan),
        jnp.full(N_COMPARTMENTS, jnp.nan)
    )
    
    vaccination_rates = generate_behavior_pattern(
        params['vaccination_rates'][1],
        N_COMPARTMENTS,
        params['SPB_exponent']
    )
    
    a, b = beta_params
    is_valid = (a > 0) & (b > 0)
    
    gamma = params['recovery_rate']
    susceptibility = params['susceptibility_rate']
    
    # Use SIRM R0 calculation for SIRV as well
    r0 = jnp.where(is_valid, R0_SIRM(jnp.full_like(POP, susceptibility), gamma, C, POP), jnp.nan)
    obs_h = 0.0
    
    def body_fun(_, state):
        return vaccination_sir_step(
            state,
            vaccination_rates,
            susceptibility,
            gamma,
            C,
            use_contact_matrix,
            dT=params["dT"]
        )
    
    final_state = jax.lax.fori_loop(0, n_steps, body_fun, initial_state)
    
    return jax.tree_map(
        lambda v, n: jnp.where(is_valid, v, n),
        final_state,
        nan_state
    ), r0, obs_h


def sim_maskSIR_trajectory(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0,
    behavior_distribution: Optional[jnp.ndarray] = None,
    custom_contact_matrix: Optional[jnp.ndarray] = None,
    custom_populations: Optional[jnp.ndarray] = None
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], float, float]:
    """Run maskSIR simulation and return full trajectory of states
    
    Args:
        beta_params: Alpha and beta parameters for population distribution
        params: Dictionary with model parameters (mu_max, beta_M, recovery_rate, etc.)
        n_steps: Number of simulation steps
        initial_infected_prop: Initial proportion of infected individuals
        N_COMPARTMENTS: Number of behavioral compartments
        SPB_exponent: Exponent for behavior pattern (default=1.0 for linear)
        behavior_distribution: Optional custom mask-wearing distribution
        custom_contact_matrix: Optional custom contact matrix
        custom_populations: Optional custom population distribution
        
    Returns:
        Tuple of:
        - State trajectories (S, I, R arrays of shape [n_steps+1, N_COMPARTMENTS])
        - Basic reproduction number (R0)
        - Observed homophily
    """
    # Initialize populations
    if custom_populations is not None:
        POP = custom_populations
        N_COMPARTMENTS = len(POP)
        initial_state = (
            POP * (1 - initial_infected_prop),
            POP * initial_infected_prop,
            jnp.zeros_like(POP)
        )
    else:
        initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
        POP = initial_state[0] + initial_state[1] + initial_state[2]
    
    # Set up contact matrix
    use_contact_matrix = True if custom_contact_matrix is not None else params.get('homophilic_tendency', 0) != 0
    C = custom_contact_matrix if custom_contact_matrix is not None else create_contact_matrix(
        N_COMPARTMENTS, params.get('homophilic_tendency', 0), POP
    )
    
    # Set up mask-wearing behavior
    mask_wearing = behavior_distribution if behavior_distribution is not None else generate_behavior_pattern(
        params['mu_max'], N_COMPARTMENTS, SPB_exponent
    )
    susceptibilities = params['beta_M'] * (1 - mask_wearing)
    
    # Calculate R0
    gamma = params['recovery_rate']
    r0 = R0_maskSIR(params['beta_M'], mask_wearing, gamma, C, POP)
    
    # Initialize trajectories
    S_trajectory = jnp.zeros((n_steps + 1, N_COMPARTMENTS))
    I_trajectory = jnp.zeros((n_steps + 1, N_COMPARTMENTS))
    R_trajectory = jnp.zeros((n_steps + 1, N_COMPARTMENTS))
    
    # Store initial state
    S_trajectory = S_trajectory.at[0].set(initial_state[0])
    I_trajectory = I_trajectory.at[0].set(initial_state[1])
    R_trajectory = R_trajectory.at[0].set(initial_state[2])
    
    # Run simulation
    state = initial_state
    dT = params.get('dT', 0.25)
    
    for t in range(n_steps):
        state = mask_sir_step(state, susceptibilities, gamma, C, use_contact_matrix, dT)
        S_trajectory = S_trajectory.at[t + 1].set(state[0])
        I_trajectory = I_trajectory.at[t + 1].set(state[1])
        R_trajectory = R_trajectory.at[t + 1].set(state[2])
    
    return (S_trajectory, I_trajectory, R_trajectory), r0, 0.0


def sim_SIRT_trajectory(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0,
    behavior_distribution: Optional[jnp.ndarray] = None,
    custom_contact_matrix: Optional[jnp.ndarray] = None,
    custom_populations: Optional[jnp.ndarray] = None
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], float, float]:
    """Run SIRT simulation and return full trajectory of states
    
    Args:
        beta_params: Alpha and beta parameters for population distribution
        params: Dictionary with model parameters (testing_rates, susceptibility_rate, recovery_rate, etc.)
        n_steps: Number of simulation steps
        initial_infected_prop: Initial proportion of infected individuals
        N_COMPARTMENTS: Number of behavioral compartments
        SPB_exponent: Exponent for behavior pattern (default=1.0 for linear)
        behavior_distribution: Optional custom testing rate distribution
        custom_contact_matrix: Optional custom contact matrix
        custom_populations: Optional custom population distribution
        
    Returns:
        Tuple of:
        - State trajectories (S, I, R arrays of shape [n_steps+1, N_COMPARTMENTS])
        - Basic reproduction number (R0)
        - Observed homophily
    """
    # Initialize populations
    if custom_populations is not None:
        POP = custom_populations
        N_COMPARTMENTS = len(POP)
        initial_state = (
            POP * (1 - initial_infected_prop),
            POP * initial_infected_prop,
            jnp.zeros_like(POP)
        )
    else:
        initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS)
        POP = initial_state[0] + initial_state[1] + initial_state[2]
    
    # Set up contact matrix
    use_contact_matrix = True if custom_contact_matrix is not None else params.get('homophilic_tendency', 0) != 0
    C = custom_contact_matrix if custom_contact_matrix is not None else create_contact_matrix(
        N_COMPARTMENTS, params.get('homophilic_tendency', 0), POP
    )
    
    # Set up testing behavior
    testing_rates = behavior_distribution if behavior_distribution is not None else generate_behavior_pattern(
        params['testing_rates'][1], N_COMPARTMENTS, SPB_exponent
    )
    
    # Adjusted recovery rate based on testing
    gammaS = params['recovery_rate'] + testing_rates
    
    # Calculate R0
    susceptibility = params['susceptibility_rate']
    r0 = R0_SIRT(susceptibility, gammaS, C, POP)
    
    # Initialize trajectories
    S_trajectory = jnp.zeros((n_steps + 1, N_COMPARTMENTS))
    I_trajectory = jnp.zeros((n_steps + 1, N_COMPARTMENTS))
    R_trajectory = jnp.zeros((n_steps + 1, N_COMPARTMENTS))
    
    # Store initial state
    S_trajectory = S_trajectory.at[0].set(initial_state[0])
    I_trajectory = I_trajectory.at[0].set(initial_state[1])
    R_trajectory = R_trajectory.at[0].set(initial_state[2])
    
    # Run simulation
    state = initial_state
    dT = params.get('dT', 0.25)
    
    for t in range(n_steps):
        state = testing_sir_step(state, susceptibility, gammaS, C, use_contact_matrix, dT)
        S_trajectory = S_trajectory.at[t + 1].set(state[0])
        I_trajectory = I_trajectory.at[t + 1].set(state[1])
        R_trajectory = R_trajectory.at[t + 1].set(state[2])
    
    return (S_trajectory, I_trajectory, R_trajectory), r0, 0.0


def sim_SIRV_trajectory(
    beta_params: Tuple[float, float],
    params: ParamType,
    n_steps: int,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    SPB_exponent: float = 1.0,
    behavior_distribution: Optional[jnp.ndarray] = None,
    custom_contact_matrix: Optional[jnp.ndarray] = None,
    custom_populations: Optional[jnp.ndarray] = None
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], float, float]:
    """Run SIRV simulation and return full trajectory of states
    
    Args:
        beta_params: Alpha and beta parameters for population distribution
        params: Dictionary with model parameters (vaccination_rates, susceptibility_rate, recovery_rate, etc.)
        n_steps: Number of simulation steps
        initial_infected_prop: Initial proportion of infected individuals
        N_COMPARTMENTS: Number of behavioral compartments
        SPB_exponent: Exponent for behavior pattern (default=1.0 for linear)
        behavior_distribution: Optional custom vaccination rate distribution
        custom_contact_matrix: Optional custom contact matrix
        custom_populations: Optional custom population distribution
        
    Returns:
        Tuple of:
        - State trajectories (S, I, R, V arrays of shape [n_steps+1, N_COMPARTMENTS])
        - Basic reproduction number (R0)
        - Observed homophily
    """
    # Initialize populations
    if custom_populations is not None:
        POP = custom_populations
        N_COMPARTMENTS = len(POP)
        initial_state = (
            POP * (1 - initial_infected_prop),
            POP * initial_infected_prop,
            jnp.zeros_like(POP),
            jnp.zeros_like(POP)
        )
    else:
        initial_state = initialize_states(beta_params, initial_infected_prop, N_COMPARTMENTS, include_vaccinated=True)
        POP = initial_state[0] + initial_state[1] + initial_state[2] + initial_state[3]
    
    # Set up contact matrix
    use_contact_matrix = True if custom_contact_matrix is not None else params.get('homophilic_tendency', 0) != 0
    C = custom_contact_matrix if custom_contact_matrix is not None else create_contact_matrix(
        N_COMPARTMENTS, params.get('homophilic_tendency', 0), POP
    )
    
    # Set up vaccination behavior
    vaccination_rates = behavior_distribution if behavior_distribution is not None else generate_behavior_pattern(
        params['vaccination_rates'][1], N_COMPARTMENTS, SPB_exponent
    )
    
    # Calculate R0
    gamma = params['recovery_rate']
    susceptibility = params['susceptibility_rate']
    # Use SIRM R0 calculation for SIRV as well
    r0 = R0_SIRM(jnp.full_like(POP, susceptibility), gamma, C, POP)
    
    # Initialize trajectories
    S_trajectory = jnp.zeros((n_steps + 1, N_COMPARTMENTS))
    I_trajectory = jnp.zeros((n_steps + 1, N_COMPARTMENTS))
    R_trajectory = jnp.zeros((n_steps + 1, N_COMPARTMENTS))
    V_trajectory = jnp.zeros((n_steps + 1, N_COMPARTMENTS))
    
    # Store initial state
    S_trajectory = S_trajectory.at[0].set(initial_state[0])
    I_trajectory = I_trajectory.at[0].set(initial_state[1])
    R_trajectory = R_trajectory.at[0].set(initial_state[2])
    V_trajectory = V_trajectory.at[0].set(initial_state[3])
    
    # Run simulation
    state = initial_state
    dT = params.get('dT', 0.25)
    
    for t in range(n_steps):
        state = vaccination_sir_step(state, vaccination_rates, susceptibility, gamma, C, use_contact_matrix, dT)
        S_trajectory = S_trajectory.at[t + 1].set(state[0])
        I_trajectory = I_trajectory.at[t + 1].set(state[1])
        R_trajectory = R_trajectory.at[t + 1].set(state[2])
        V_trajectory = V_trajectory.at[t + 1].set(state[3])
    
    return (S_trajectory, I_trajectory, R_trajectory, V_trajectory), r0, 0.0

