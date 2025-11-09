import jax
import jax.numpy as jnp
from typing import Tuple, Optional


def normalize(M) -> jnp.ndarray:
    """Normalize each column of the matrix to sum to 1"""
    return M / jnp.sum(M, axis=1, keepdims=True)

def create_contact_matrix(n_groups, homophilic_tendency, group_sizes, Cm_external = None, norm = True) -> jnp.ndarray:
    if Cm_external is None:
        positions = jnp.linspace(0, 1, n_groups)
        diffs = jnp.abs(positions[:, None] - positions[None, :])
        weights = jnp.exp(-homophilic_tendency * diffs)
        C = weights  * n_groups * n_groups
    else:
        C = Cm_external

    if norm:
        C = normalize(C)
        TC = jnp.sum(jnp.outer(group_sizes, group_sizes) * C) # Total contacts
        C = C / TC # Normalize to total contacts
    
    return C
