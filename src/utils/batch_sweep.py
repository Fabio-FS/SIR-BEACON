import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable, Tuple, Union
from functools import partial

StateType = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]

def batch_sweep(
    simulation_fn: Callable,
    param_ranges: jnp.ndarray,
    base_params: Dict[str, Any],
    n_steps: int,
    batch_size: int,
    param_updater: Callable,
    initial_infected_prop: float = 1e-4,
    use_contact_matrix: bool = False,
    SPB_exponent: float = 1,
    N_COMPARTMENTS: int = 100
) -> Tuple[StateType, jnp.ndarray, jnp.ndarray]:
    total_params = len(param_ranges)
    batch_size = min(batch_size, total_params)
    n_batches = (total_params + batch_size - 1) // batch_size
    results = []
    r0_results = []
    h_results = []
    
    def run_single_sim(param_values):
        sim_params = dict(base_params)
        sim_params = param_updater(sim_params, param_values)
        return simulation_fn(
            beta_params=sim_params.get('beta_params', base_params['beta_params']),
            params=sim_params,
            n_steps=n_steps,
            initial_infected_prop=initial_infected_prop,
            N_COMPARTMENTS=N_COMPARTMENTS,
            use_contact_matrix = use_contact_matrix,
            SPB_exponent = SPB_exponent
        )
    
    vectorized_sim = jax.vmap(run_single_sim, in_axes=0)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_params)
        batch_params = param_ranges[start_idx:end_idx]
        
        batch_states, batch_r0s, batch_hs = vectorized_sim(batch_params)
        results.append(batch_states)
        r0_results.append(batch_r0s)
        h_results.append(batch_hs)
    
    final_states = jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *results)
    final_r0s = jnp.concatenate(r0_results)
    final_hs = jnp.concatenate(h_results)
    
    return final_states, final_r0s, final_hs



def batch_sweep_SIRV(
    simulation_fn: Callable,
    param_ranges: jnp.ndarray,
    base_params: Dict[str, Any],
    n_steps: int,
    batch_size: int,
    param_updater: Callable,
    initial_infected_prop: float = 1e-4,
    N_COMPARTMENTS: int = 100,
    use_contact_matrix: bool = False,
    SPB_exponent: float = 1
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    total_params = len(param_ranges)
    batch_size = min(batch_size, total_params)
    n_batches = (total_params + batch_size - 1) // batch_size
    results = []
    r0_results = []
    h_results = []
    
    def run_single_sim(param_values):
        sim_params = dict(base_params)
        sim_params = param_updater(sim_params, param_values)
        return simulation_fn(
            beta_params=sim_params.get('beta_params', base_params['beta_params']),
            params=sim_params,
            n_steps=n_steps,
            initial_infected_prop=initial_infected_prop,
            N_COMPARTMENTS=N_COMPARTMENTS,
            use_contact_matrix = use_contact_matrix,
            SPB_exponent = SPB_exponent
        )
    
    vectorized_sim = jax.vmap(run_single_sim, in_axes=0)
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_params)
        batch_params = param_ranges[start_idx:end_idx]
        
        batch_states, batch_r0s, batch_hs = vectorized_sim(batch_params)
        results.append(batch_states)
        r0_results.append(batch_r0s)
        h_results.append(batch_hs)
    
    final_states = jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *results)
    final_r0s = jnp.concatenate(r0_results)
    final_hs = jnp.concatenate(h_results)
    
    return final_states, final_r0s, final_hs