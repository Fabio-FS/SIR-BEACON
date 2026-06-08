import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable, Tuple, Union
from functools import partial

def pol_to_alpha(pol: jnp.ndarray) -> jnp.ndarray:
    """Convert polarization values to alpha values"""
    return 1/(pol*2) - 0.5


# DEPRECATED! THIS WORKS CORRECTLY ONLY FOR A CONTINUOUS BETA DISTRIBUTION
# We only have 5 bins, so the error would become quite consistent:

#0.10 -> 0.18
#0.25 -> 0.40
#0.50 -> 0.66
#0.75 -> 0.85
def pol_mean_to_ab_old(pol: jnp.ndarray, m : jnp.ndarray = 0.5) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert polarization and mean to alpha, beta parameters of beta distribution
    Args:
        p: polarization values
        m: Mean values
    
    Returns:
        Tuple of (alpha, beta) parameters
    """
    a = -m*(4*(m-1)*m+pol)/pol
    b = 4*m*(m-1)**2/pol +m - 1
    return a, b



def pol_mean_to_ab(pol, m=0.5):
    """Find Beta (a, b) such that the 5-bin discretization has mean m and 4*Var = pol.
    Returns negative a or b if (m, pol) is infeasible."""
    from scipy.optimize import root
    from scipy.stats import beta as sbeta
    import numpy as np
    
    def solve(pol_val, m_val):
        pol_val = float(pol_val)
        m_val = float(m_val)
        
        if pol_val > 4 * m_val * (1 - m_val):
            return np.array([-1.0, -1.0])
        
        N = 5
        edges = np.linspace(0, 1, N + 1)
        levels = np.linspace(0, 1, N)
        
        def residual(log_ab):
            a, b = np.exp(log_ab[0]), np.exp(log_ab[1])
            q = sbeta.cdf(edges[1:], a, b) - sbeta.cdf(edges[:-1], a, b)
            q = q / q.sum()
            mean = np.sum(q * levels)
            p = 4 * np.sum(q * (levels - mean) ** 2)
            return [mean - m_val, p - pol_val]
        
        sol = root(residual, x0=[0.0, 0.0], method="hybr")
        return np.exp(sol.x)
    
    out_dtype = jnp.result_type(pol, m)
    
    result = jax.pure_callback(
        lambda p, mm: solve(p, mm).astype(out_dtype),
        jax.ShapeDtypeStruct((2,), out_dtype),
        pol, m
    )
    return result[0], result[1]


def my_beta_asymmetric(a: float, b: float, n_groups: int, norm: float = 1.0) -> jnp.ndarray:
    """Generate population sizes based on asymmetric beta distribution with better handling of extreme polarization"""
    # Use the CDF to compute the probability mass in each bin
    from jax.scipy.stats import beta as jbeta
    
    # Create bin edges that divide [0,1] into n_groups equal segments
    bin_edges = jnp.linspace(0, 1, n_groups + 1)
    
    # Calculate probability mass in each bin using the CDF
    bin_probs = jnp.array([jbeta.cdf(bin_edges[i+1], a, b) - jbeta.cdf(bin_edges[i], a, b) for i in range(n_groups)])
    
    # Normalize and scale
    return bin_probs / jnp.sum(bin_probs) * norm

def homogeneous_distribution(n_groups: int, min: float, max: float):
    return jnp.linspace(min+jnp.abs(max-min)/n_groups/2, max-jnp.abs(max-min)/n_groups/2, n_groups)