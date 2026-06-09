import jax.numpy as jnp
from jax.scipy.stats import beta as jbeta


def beta_populations(a, b, n_groups):
    """Discretized Beta(a,b): fraction of population in each of n_groups bins."""
    edges = jnp.linspace(0, 1, n_groups + 1)
    mass = jbeta.cdf(edges[1:], a, b) - jbeta.cdf(edges[:-1], a, b)
    return mass / jnp.sum(mass)


def contact_matrix(n_groups, h, pop):
    """Homophilic contact matrix, row-normalized then total-contact-normalized."""
    pos = jnp.linspace(0, 1, n_groups)
    diffs = jnp.abs(pos[:, None] - pos[None, :])
    C = jnp.exp(-h * diffs) * n_groups * n_groups
    C = C / (C @ pop)[:, None]
    C = C / jnp.sum(jnp.outer(pop, pop) * C)
    return C


def pol_mean_to_ab(pol, mean):
    """Convert (polarization, mean) to Beta distribution (a, b) parameters."""
    a = -mean * (4 * (mean - 1) * mean + pol) / pol
    b = 4 * mean * (mean - 1)**2 / pol + mean - 1
    return a, b