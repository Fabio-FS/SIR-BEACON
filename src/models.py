import jax
import jax.numpy as jnp


@jax.jit
def step_SIRM(state, beh, C, params):
    S, I, R = state
    susc = params['beta_M'] * (1 - beh)
    force = C @ I
    new_inf = susc * S * force * params['dT']
    new_rec = params['recovery_rate'] * I * params['dT']
    return jnp.stack([
        S - new_inf,
        I + new_inf - new_rec,
        R + new_rec,
    ])


@jax.jit
def step_SIRT(state, beh, C, params):
    S, I, R = state
    susc = params['beta_M']
    force = C @ I
    new_inf = susc * S * force * params['dT']
    new_rec = (params['recovery_rate'] + beh) * I * params['dT']
    return jnp.stack([
        S - new_inf,
        I + new_inf - new_rec,
        R + new_rec,
    ])


@jax.jit
def step_SIRV(state, beh, C, params):
    S, I, R, V = state
    susc = params['beta_M']
    force = C @ I
    new_inf = susc * S * force * params['dT']
    new_rec = params['recovery_rate'] * I * params['dT']
    new_vac = beh * S * params['dT']
    return jnp.stack([
        S - new_inf - new_vac,
        I + new_inf - new_rec,
        R + new_rec,
        V + new_vac,
    ])