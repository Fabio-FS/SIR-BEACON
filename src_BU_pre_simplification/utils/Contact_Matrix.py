import jax
import jax.numpy as jnp
from typing import Tuple, Optional

def normalize(M,pop):
    M = M / (M @ pop)[:, None]
    TC = jnp.sum(jnp.outer(pop, pop) * M) # Total contacts
    M = M / TC # Normalize to total contacts
    return M 
    
def create_contact_matrix(n_groups, homophilic_tendency, group_sizes, Cm_external = None, norm = True) -> jnp.ndarray:
    if Cm_external is None:
        positions = jnp.linspace(0, 1, n_groups)
        diffs = jnp.abs(positions[:, None] - positions[None, :])
        weights = jnp.exp(-homophilic_tendency * diffs)
        C = weights  * n_groups * n_groups
    else:
        C = Cm_external

    if norm:
        C = normalize(C,group_sizes)
    return C
