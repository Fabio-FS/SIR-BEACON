import jax
import jax.numpy as jnp




def simulate(step_fn, state0, beh, C, params, n_steps):
    """Run a simulation and return the full trajectory.
    Returns shape (n_compartments, n_steps + 1, n_groups)."""
    def body(state, _):
        next_state = step_fn(state, beh, C, params)
        return next_state, next_state
    _, traj = jax.lax.scan(body, state0, jnp.arange(n_steps))
    traj = jnp.concatenate([state0[:, None, :], traj.transpose(1, 0, 2)], axis=1)
    return traj


def simulate_final(step_fn, state0, beh, C, params, n_steps):
    """Run a simulation and return only the final state.
    Returns shape (n_compartments, n_groups)."""
    def body(_, state):
        return step_fn(state, beh, C, params)
    return jax.lax.fori_loop(0, n_steps, body, state0)