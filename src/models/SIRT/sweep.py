import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from ...utils.batch_sweep import batch_sweep
from ...utils.distributions import pol_mean_to_ab, pol_to_alpha, homogeneous_distribution
from .dynamic import sim_SIRT_final, StateType

def update_test_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new testing rate and alpha values"""
    params['testing_rates'] = (0, values[0])     # min and max testing rates
    params['beta_params'] = (values[1], values[1])      # both parameters equal
    return params

def update_h_pol_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new homophily and alpha values"""
    params['homophilic_tendency'] = values[0]
    params['beta_params'] = (values[1], values[1])   # both parameters equal for polarization
    return params

def update_pol_mean_params(params: Dict[str, Any], values: jnp.ndarray) -> Dict[str, Any]:
    """Updates parameters with new alpha and beta values derived from variance and mean"""
    alpha, beta = pol_mean_to_ab(values[0], values[1])
    params['beta_params'] = (alpha, beta)
    return params

def sweep_pol_SPB_SIRT(
    test_max_range: dict = {"m": 0.001, "M": 0.05, "n": 10},
    pol_range: dict = {"m": 0.01, "M": 1, "n": 10},
    h: float = 2.0,
    dT: float = 0.25,
    T: int = 1000,
    recovery_rate: float = 0.1,
    susceptibility_rate: float = 0.6,
    batch_size: int = 1000,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray, jnp.ndarray]:
    """Sweep over testing rate (SPB) and polarization values"""
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'homophilic_tendency': h,
        'beta_params': (2.0, 2.0),
        'susceptibility_rate': susceptibility_rate,
        'testing_rates': (0.0, 0.05),
        'SPB_exponent': SPB_exponent
    }
    
    test_vals = homogeneous_distribution(test_max_range["n"], test_max_range["m"], test_max_range["M"])
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    alpha_vals = pol_to_alpha(pol_vals)
    
    test_mesh, alpha_mesh = jnp.meshgrid(test_vals, alpha_vals)
    param_ranges = jnp.stack([test_mesh.ravel(), alpha_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    return batch_sweep(
        simulation_fn=sim_SIRT_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_test_pol_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS=N_COMPARTMENTS,
        use_contact_matrix=use_contact_matrix,
        SPB_exponent=SPB_exponent
    )

def sweep_pol_mean_SIRT(
    pol_range: dict = {"m": 1/100/2, "M": 1-1/100/2, "n": 100},
    mean_range: dict = {"m": 1/100/2, "M": 1-1/100/2, "n": 100},
    h: float = 0,
    dT: float = 1,
    T: int = 1000,
    recovery_rate: float = 0.1,
    susceptibility_rate: float = 0.6,
    test_rate: float = 0.05,  # Fixed testing rate
    batch_size: int = 1000,
    use_contact_matrix: bool = False,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray, jnp.ndarray]:
    """Sweep over polarization and mean values"""
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'homophilic_tendency': h,
        'susceptibility_rate': susceptibility_rate,
        'testing_rates': (0.0, test_rate),  # Fixed test rate
        'beta_params': (1.0, 1.0),
        'SPB_exponent': SPB_exponent
    }
    
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    mean_vals = homogeneous_distribution(mean_range["n"], mean_range["m"], mean_range["M"]) 
    
    pol_mesh, mean_mesh = jnp.meshgrid(pol_vals, mean_vals)
    param_ranges = jnp.stack([pol_mesh.ravel(), mean_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    return batch_sweep(
        simulation_fn=sim_SIRT_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_pol_mean_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS=N_COMPARTMENTS,
        use_contact_matrix=use_contact_matrix,
        SPB_exponent=SPB_exponent
    )

def sweep_hom_pol_SIRT(
    h_range: dict = {"m": -2.0, "M": 2.0, "n": 10},
    pol_range: dict = {"m": 0.01, "M": 1, "n": 10},
    dT: float = 1,
    T: int = 1000,
    recovery_rate: float = 0.1,
    susceptibility_rate: float = 0.6,
    test_rate: float = 0.05,  # Fixed testing rate
    batch_size: int = 1000,
    initial_infected_prop: float = 1e-4,
    SPB_exponent: float = 1.0,
    use_contact_matrix: bool = True,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray, jnp.ndarray]:
    """Sweep over homophily and polarization values"""
    base_params = {
        'recovery_rate': recovery_rate,
        'dT': dT,
        'susceptibility_rate': susceptibility_rate,
        'testing_rates': (0.0, test_rate),
        'beta_params': (2.0, 2.0),
        'SPB_exponent': SPB_exponent
    }
    
    h_vals = homogeneous_distribution(h_range["n"], h_range["m"], h_range["M"])
    pol_vals = homogeneous_distribution(pol_range["n"], pol_range["m"], pol_range["M"])
    alpha_vals = pol_to_alpha(pol_vals)
    
    h_mesh, alpha_mesh = jnp.meshgrid(h_vals, alpha_vals)
    param_ranges = jnp.stack([h_mesh.ravel(), alpha_mesh.ravel()], axis=1)
    
    n_steps = int(T / dT)
    
    return batch_sweep(
        simulation_fn=sim_SIRT_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        param_updater=update_h_pol_params,
        initial_infected_prop=initial_infected_prop,
        N_COMPARTMENTS=N_COMPARTMENTS,
        use_contact_matrix=use_contact_matrix,
        SPB_exponent=SPB_exponent
    )