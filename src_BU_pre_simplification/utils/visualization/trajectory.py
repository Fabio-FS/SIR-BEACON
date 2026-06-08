# File: src/utils/visualization/trajectory.py
"""
Trajectory visualization functions for epidemic models
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

from ..distributions import pol_mean_to_ab

def run_single_simulation(polarization, homophilic_tendency, fixed_mean, PARAMS, model, simulated_days=1000,
                          initial_infected_prop = 1e-4, population_size = 5):
    """
    Helper function to run a single simulation with given parameters.
    
    Returns:
    --------
    tuple 
        (S, I, R) where each is a sum over population compartments
    """
    # Set up parameters for this simulation
    params = PARAMS.copy()
    params.update({
        'homophilic_tendency': homophilic_tendency,
        'fixed_mean': fixed_mean
    })
    
    # Calculate beta parameters from polarization and mean
    beta_params = pol_mean_to_ab(polarization, params['fixed_mean'])
    
    # Run the simulation
    states_trajectory, r0, homophily = model.run_simulation(
        beta_params=beta_params,
        params=params,
        simulated_days=simulated_days,
        initial_infected_prop=initial_infected_prop,
        population_size=population_size,
        use_contact_matrix=True,
        return_trajectory=True
    )
    
    # Unpack and sum the state trajectories across population compartments
    S, I, R, *_ = states_trajectory
    S_total = jnp.sum(S, axis=1)
    I_total = jnp.sum(I, axis=1)
    R_total = jnp.sum(R, axis=1)
    
    return S_total, I_total, R_total