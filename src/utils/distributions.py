import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable, Tuple, Union
from functools import partial

def pol_to_alpha(pol: jnp.ndarray) -> jnp.ndarray:
    """Convert polarization values to alpha values"""
    return 1/(pol*2) - 0.5

def pol_mean_to_ab(pol: jnp.ndarray, m : jnp.ndarray = 0.5) -> Tuple[jnp.ndarray, jnp.ndarray]:
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