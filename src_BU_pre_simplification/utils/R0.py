# utils/r0_calculation.py
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any
#from core.interaction import create_ngm, create_rectangular_contact_matrix
#from models.SIRB import generate_susceptibility


# In R0.py

@jax.jit
def create_ngm_maskSIR(
    gamma: float,
    beta_M: float,
    mask_wearing: jnp.ndarray,
    populations: jnp.ndarray,
    C: jnp.ndarray
) -> jnp.ndarray:
    """Create next generation matrix for maskSIR model
    
    Args:
        gamma: Recovery rate
        beta_M: Maximum susceptibility
        mask_wearing: Array of mask-wearing values for each compartment
        populations: Array of population sizes
        C: Contact matrix
    """
    pop_fractions = populations / jnp.sum(populations)
    susceptibilities = beta_M * (1 - mask_wearing)  # Calculate effective susceptibilities
    return (1/gamma) * jnp.diag(susceptibilities * pop_fractions) @ C

def R0_maskSIR(
    beta_M: float,
    mask_wearing: jnp.ndarray,
    gamma: float,
    C: jnp.ndarray,
    populations: jnp.ndarray
) -> float:
    """Calculate R0 for maskSIR model using the largest eigenvalue of the next generation matrix
    
    Args:
        beta_M: Maximum susceptibility
        mask_wearing: Array of mask-wearing values for each compartment
        gamma: Recovery rate
        C: Contact matrix
        populations: Array of population sizes
    
    Returns:
        R0 value (largest eigenvalue of next generation matrix)
    """
    ngm = create_ngm_maskSIR(gamma, beta_M, mask_wearing, populations, C)
    return power_iteration(ngm)







@jax.jit
def create_ngm_SIRM(
    gamma: float,
    susceptibility: jnp.ndarray,
    populations: jnp.ndarray,
    C: jnp.ndarray
) -> jnp.ndarray:
    """Create next generation matrix from model parameters"""
    pop_fractions = populations / jnp.sum(populations)
    return (1/gamma) * jnp.diag(susceptibility * pop_fractions) @ C


@jax.jit
def create_ngm_SIRT(
    gammaS: float,
    susceptibility: jnp.ndarray,
    populations: jnp.ndarray,
    C: jnp.ndarray
) -> jnp.ndarray:
    """Create next generation matrix from model parameters"""
    pop_fractions = populations / jnp.sum(populations)
    return jnp.diag(susceptibility * pop_fractions / gammaS) @ C







@jax.jit
def power_iteration(matrix: jnp.ndarray, num_iterations: int = 100) -> float:
    """Calculate largest eigenvalue using power iteration method"""
    n = matrix.shape[0]
    vector = jnp.ones(n)
    
    def body_fun(_, v):
        v_new = matrix @ v
        return v_new / jnp.linalg.norm(v_new)
    
    vector = jax.lax.fori_loop(0, num_iterations, body_fun, vector)
    return jnp.vdot(matrix @ vector, vector) / jnp.vdot(vector, vector)



def R0_SIRM(
    susceptibilities: jnp.ndarray,
    gamma: float,
    C: jnp.ndarray,
    populations: jnp.ndarray
) -> float:
    """Calculate R0 using the largest eigenvalue of the next generation matrix"""
    
    ngm = create_ngm_SIRM(gamma, susceptibilities, populations, C)

    return power_iteration(ngm)


def R0_SIRV(
    susceptibilities: jnp.ndarray,
    gamma: float,
    C: jnp.ndarray,
    populations: jnp.ndarray
) -> float:
    """Calculate R0 using the largest eigenvalue of the next generation matrix"""
    
    ngm = create_ngm_SIRM(gamma, susceptibilities, populations, C)

    return power_iteration(ngm)

def R0_SIRT(
    susceptibility: float,
    gammaS: jnp.ndarray,
    C: jnp.ndarray,
    populations: jnp.ndarray
) -> float:
    """Calculate R0 for SIRT model using the largest eigenvalue of the next generation matrix
    
    In SIRT, the recovery rate varies by group due to testing, while transmission rate is fixed.
    """

    
    ngm = create_ngm_SIRT(gammaS, susceptibility, populations, C)
    
    return power_iteration(ngm)

# def generate_susceptibility(beta_range: Tuple[float, float], N_COMPARTMENTS: int) -> jnp.ndarray:
#     """Generate linearly interpolated susceptibility values"""
#     beta_low, beta_high = beta_range
#     return beta_high + jnp.linspace(0, 1, N_COMPARTMENTS) * (beta_low - beta_high)