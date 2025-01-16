import jax
import jax.numpy as jnp
from typing import Tuple, Optional

@jax.jit
def matrix_scaling(
    matrix: jnp.ndarray,
    max_iters: int = 1000,
    threshold: float = 1e-6
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Perform matrix scaling to normalize rows and columns to sum to 1.
    
    Args:
        matrix: Input matrix to normalize
        max_iters: Maximum number of iterations
        threshold: Convergence threshold
        
    Returns:
        Tuple of:
        - Normalized matrix
        - Row scaling vector
        - Column scaling vector
    """
    n, m = matrix.shape
    ones_n = jnp.ones(n)
    ones_m = jnp.ones(m)
    
    return sinkhorn_norm(
        -jnp.log(matrix + 1e-10),  # Convert to cost matrix
        ones_n / n,                 # Uniform row distribution
        ones_m / m,                 # Uniform column distribution
        epsilon=1.0,
        max_iters=max_iters,
        threshold=threshold
    )

def sinkhorn_norm(
    cost_matrix: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
    epsilon: float = 1e-1,
    max_iters: int = 1000,
    threshold: float = 1e-4
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute Sinkhorn normalization to obtain optimal transport matrix.
    
    Args:
        cost_matrix: Matrix of costs/distances between points
        a: Source distribution (row marginals)
        b: Target distribution (column marginals)
        epsilon: Regularization parameter
        max_iters: Maximum number of iterations
        threshold: Convergence threshold
        
    Returns:
        Tuple of:
        - Transport matrix P
        - u vector (row scaling)
        - v vector (column scaling)
    """
    # Initialize scaling vectors
    u = jnp.ones_like(a)
    v = jnp.ones_like(b)
    
    # Compute kernel matrix
    K = jnp.exp(-cost_matrix / epsilon)
    
    def body_fun(state):
        u, v, i = state
        
        # Update u
        v_K = v * K
        u_new = a / (v_K.sum(axis=1) + 1e-10)
        
        # Update v
        u_K = u_new[:, None] * K
        v_new = b / (u_K.sum(axis=0) + 1e-10)
        
        return u_new, v_new, i + 1
    
    def cond_fun(state):
        u, v, i = state
        
        # Check convergence on marginals
        P = u[:, None] * K * v
        row_sums = P.sum(axis=1)
        col_sums = P.sum(axis=0)
        
        row_diff = jnp.abs(row_sums - a).max()
        col_diff = jnp.abs(col_sums - b).max()
        
        return (row_diff > threshold) & (col_diff > threshold) & (i < max_iters)
    
    # Run Sinkhorn iterations
    init_state = (u, v, 0)
    u_final, v_final, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
    
    # Compute final transport matrix
    P = u_final[:, None] * K * v_final
    
    return P, u_final, v_final

def create_contact_matrix(n_groups: int, homophilic_tendency: float, populations: jnp.ndarray) -> jnp.ndarray:
    """Create contact matrix where Cᵢⱼ is the probability of j contacting i.
    
    Args:
        n_groups: Number of groups
        homophilic_tendency: h parameter (h=0 gives uniform mixing)
        populations: Array of population sizes for each group
        
    Returns:
        Contact matrix C where Cᵢⱼ is the probability of j contacting i
    """
    # Generate distance-based weights
    positions = jnp.linspace(0, 1, n_groups)
    diffs = jnp.abs(positions[:, None] - positions[None, :])
    weights = jnp.exp(-homophilic_tendency * diffs)
    
    # For each column j, normalize by sum of weighted populations
    # This ensures each column sums to 1 (total probability = 1)
    C = weights / jnp.sum(weights, axis=0)[None, :]
    C2, _, _ = matrix_scaling(C)
    C2 = C2 * n_groups * n_groups
    return C2