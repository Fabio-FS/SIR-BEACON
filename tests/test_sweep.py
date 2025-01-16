# test_sweep.py
import jax.numpy as jnp
from models.variable_susceptibility import (
    simulate_variable_susceptibility_hom_final,
    simulate_variable_susceptibility_pol_final
)
from utils.parallel import (
    parameter_sweep,
    batch_parameter_sweep,
    create_h_sweep,
    create_susceptibility_sweep,
    create_beta_sweep
)

def test_h_sweep(
    h_min: float = -5.0,
    h_max: float = 5.0,
    n_h: int = 50,
    dt: float = 0.25,
    T: int = 1000,
    recovery_rate: float = 0.1,
    transmission_rates: tuple = (0.1, 0.5),
    batch_size: int = 1000
):
    """Test parameter sweep varying only homophilic tendency"""
    base_params = {
        'transmission_rates': transmission_rates,
        'recovery_rate': recovery_rate,
        'dt': dt,
        'beta_params': (2.0, 2.0)
    }
    
    param_ranges = create_h_sweep(h_min, h_max, n_h)
    n_steps = int(T / dt)

    return batch_parameter_sweep(
        simulation_fn=simulate_variable_susceptibility_hom_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        sweep_type='h'
    )

def test_susceptibility_symmetric_sweep(
    susc_min: float = 0.2,
    susc_max: float = 0.8,
    n_susc: int = 10,
    ab_min: float = 0.5,
    ab_max: float = 5.0,
    n_ab: int = 10,
    h: float = 2.0,
    dt: float = 0.25,
    T: int = 1000,
    recovery_rate: float = 0.1,
    batch_size: int = 1000
):
    """Test parameter sweep varying susceptibility range and symmetric beta params"""
    base_params = {
        'recovery_rate': recovery_rate,
        'dt': dt,
        'homophilic_tendency': h,
        'beta_params': (2.0, 2.0),
        'transmission_rates': (0.1, 0.5)
    }
    
    param_ranges = create_susceptibility_sweep(
        susc_min, susc_max, n_susc,
        ab_min, ab_max, n_ab
    )
    n_steps = int(T / dt)
    
    if h == 0:
        return batch_parameter_sweep(
            simulation_fn=simulate_variable_susceptibility_pol_final,
            param_ranges=param_ranges,
            base_params=base_params,
            n_steps=n_steps,
            batch_size=batch_size,
            sweep_type='susc'
        )
    return batch_parameter_sweep(
        simulation_fn=simulate_variable_susceptibility_hom_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        sweep_type='susc'
    )

def test_asymmetric_beta_sweep(
    range_a: dict = {"m": 0.5, "M": 5.0, "n": 50},
    range_b: dict = {"m": 0.5, "M": 5.0, "n": 50},
    h: float = 2,
    dt: float = 0.25,
    recovery_rate: float = 0.1,
    transmission_rates: tuple = (0, 0.6),
    T: int = 1000,
    batch_size: int = 4000
):
    """Test parameter sweep varying asymmetric beta parameters"""
    base_params = {
        'transmission_rates': transmission_rates,
        'recovery_rate': recovery_rate,
        'dt': dt,
        'homophilic_tendency': h,
        'beta_params': (1.0, 1.0)
    }
    
    param_ranges = create_beta_sweep(
        range_a["m"], range_a["M"], range_a["n"],
        range_b["m"], range_b["M"], range_b["n"]
    )
    n_steps = int(T / dt)
    
    if h == 0:
        return batch_parameter_sweep(
            simulation_fn=simulate_variable_susceptibility_pol_final,
            param_ranges=param_ranges,
            base_params=base_params,
            n_steps=n_steps,
            batch_size=batch_size,
            sweep_type='beta'
        )
    return batch_parameter_sweep(
        simulation_fn=simulate_variable_susceptibility_hom_final,
        param_ranges=param_ranges,
        base_params=base_params,
        n_steps=n_steps,
        batch_size=batch_size,
        sweep_type='beta'
    )