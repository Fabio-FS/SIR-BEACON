import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable, Tuple, Union
from functools import partial

def pol_to_alpha(pol: jnp.ndarray) -> jnp.ndarray:
    """Convert polarization values to alpha values"""
    return 1/(pol*2) - 0.5

def pol_mean_to_ab(pol: jnp.ndarray, m: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Convert variance and mean to alpha, beta parameters of beta distribution
    Args:
        p: polarization values
        m: Mean values
    
    Returns:
        Tuple of (alpha, beta) parameters
    """
    a = -m*(4*(m-1)*m+pol)/pol
    b = 4*m*(m-1)**2/pol +m - 1
    return a, b

def my_beta_symmetric(a: float, n_groups: int, norm: float = 1.0) -> jnp.ndarray:
    """Generate population sizes based on symmetric beta distribution
    
    Args:
        a: Shape parameter (alpha = beta for symmetry)
        n_groups: Number of subpopulations
        norm: Total population size
        
    Returns:
        Array of population sizes
    """
    x = jnp.linspace(1/n_groups/2, 1-1/n_groups/2, n_groups)
     
    from jax.scipy.stats import beta as jbeta
    y = jbeta.pdf(x, a, a)
    y = y / jnp.sum(y) * norm
    return y

def my_beta_asymmetric(a: float, b: float, n_groups: int, norm: float = 1.0) -> jnp.ndarray:
    """Generate population sizes based on asymmetric beta distribution
    
    Args:
        a: First shape parameter (alpha)
        b: Second shape parameter (beta)
        n_groups: Number of subpopulations
        norm: Total population size
        
    Returns:
        Array of population sizes
    """
    x = jnp.linspace(1/n_groups/2, 1-1/n_groups/2, n_groups)
    
    from jax.scipy.stats import beta as jbeta
    y = jbeta.pdf(x, a, b)
    y = y / jnp.sum(y) * norm
    return y

def homogeneous_distribution(n_groups: int, min: float, max: float):
    return jnp.linspace(min+jnp.abs(max-min)/n_groups/2, max-jnp.abs(max-min)/n_groups/2, n_groups)