# sweep.py
import jax
import jax.numpy as jnp
import numpy as np
from .contact import beta_populations, contact_matrix, pol_mean_to_ab
from .simulate import simulate, simulate_final

def _ab_grid(pol_vals, second_vals, second_is_mean, fixed_mean):
    """Build (a, b) arrays of shape (len(pol_vals), len(second_vals))."""
    pol_vals = np.asarray(pol_vals)
    second_vals = np.asarray(second_vals)
    a_grid = np.empty((len(pol_vals), len(second_vals)))
    b_grid = np.empty((len(pol_vals), len(second_vals)))
    for i, pol in enumerate(pol_vals):
        for j, x in enumerate(second_vals):
            mean = x if second_is_mean else fixed_mean
            a, b = pol_mean_to_ab(float(pol), float(mean))
            a_grid[i, j] = float(a)
            b_grid[i, j] = float(b)
    return jnp.asarray(a_grid), jnp.asarray(b_grid)


def sweep_pol_homophily(step_fn, build_state0, beh, pol_vals, h_vals, params,
                        n_steps, mean=0.5, n_groups=5):
    a_grid, b_grid = _ab_grid(pol_vals, h_vals, second_is_mean=False, fixed_mean=mean)
    h_grid = jnp.broadcast_to(jnp.asarray(h_vals)[None, :], a_grid.shape)
    
    def one_run(a, b, h):
        pop = beta_populations(a, b, n_groups)
        C = contact_matrix(n_groups, h, pop)
        state0 = build_state0(pop)
        return simulate_final(step_fn, state0, beh, C, params, n_steps)
    
    flat = jax.vmap(one_run)(a_grid.ravel(), b_grid.ravel(), h_grid.ravel())
    return flat.reshape(a_grid.shape + flat.shape[1:])


def sweep_pol_mean(step_fn, build_state0, beh, pol_vals, mean_vals, params,
                   n_steps, h=0.0, n_groups=5):
    a_grid, b_grid = _ab_grid(pol_vals, mean_vals, second_is_mean=True, fixed_mean=None)
    
    def one_run(a, b):
        pop = beta_populations(a, b, n_groups)
        C = contact_matrix(n_groups, h, pop)
        state0 = build_state0(pop)
        return simulate_final(step_fn, state0, beh, C, params, n_steps)
    
    flat = jax.vmap(one_run)(a_grid.ravel(), b_grid.ravel())
    return flat.reshape(a_grid.shape + flat.shape[1:])


def sweep_extra_infections(step_fn, build_state0, beh, beta_vals, rect_coords,
                           params, n_steps, mean=0.5, n_groups=5,
                           pol_baseline=1e-3):
    """For each beta_0 in beta_vals, compute (I_min, I_max, I_baseline) over the
    4 corners of rect_coords.
    
    rect_coords = [pol0, h0, dpol, dh] (same convention as plotting).
    pol_baseline: small polarization used as the 'homogeneous' reference.
    
    Returns:
        I_corners: (len(beta_vals), 4)  -- infection size at each corner
        I_baseline: (len(beta_vals),)   -- infection size at (pol_baseline, h=0)
    """
    pol0, h0, dpol, dh = rect_coords
    corner_pols = jnp.array([pol0, pol0 + dpol, pol0,        pol0 + dpol])
    corner_hs   = jnp.array([h0,   h0,          h0 + dh,    h0 + dh])
    
    # Pre-compute (a, b) for all corners and the baseline, host-side.
    a_corners = np.empty(4)
    b_corners = np.empty(4)
    for i in range(4):
        a, b = pol_mean_to_ab(float(corner_pols[i]), mean)
        a_corners[i] = float(a)
        b_corners[i] = float(b)
    a_base, b_base = pol_mean_to_ab(pol_baseline, mean)
    a_base, b_base = float(a_base), float(b_base)
    
    a_corners = jnp.asarray(a_corners)
    b_corners = jnp.asarray(b_corners)
    
    def one_run(a, b, h, beta_0):
        pop = beta_populations(a, b, n_groups)
        C = contact_matrix(n_groups, h, pop)
        state0 = build_state0(pop)
        p = {**params, 'beta_M': beta_0}
        final = simulate_final(step_fn, state0, beh, C, p, n_steps)
        # infection size: 1 - sum(S) - sum(V if present)
        n_comp = final.shape[0]
        S_sum = final[0].sum()
        V_sum = jnp.where(n_comp == 4, final[-1].sum(), 0.0)
        return 1.0 - S_sum - V_sum
    
    # Vmap: over beta_0 (outer), over corner (inner)
    corner_fn = jax.vmap(one_run, in_axes=(0, 0, 0, None))   # over corners
    full_fn   = jax.vmap(corner_fn, in_axes=(None, None, None, 0))  # over beta
    I_corners = full_fn(a_corners, b_corners, corner_hs, beta_vals)
    
    # Baseline: vmap over beta_0 only
    base_fn = jax.vmap(lambda beta_0: one_run(a_base, b_base, 0.0, beta_0))
    I_baseline = base_fn(beta_vals)
    
    return I_corners, I_baseline


def trajectories_at_corners(step_fn, build_state0, beh, rect_coords, beta_0,
                            params, n_steps, mean=0.5, n_groups=5,
                            pol_baseline=1e-3):
    """Run simulations at the 4 corners of rect_coords and at the baseline.
    
    Returns:
        I_corners: (4, n_steps+1) -- I(t) summed across groups, per corner
        I_baseline: (n_steps+1,)  -- I(t) summed across groups, at baseline
    """
    pol0, h0, dpol, dh = rect_coords
    corner_pols = jnp.array([pol0, pol0 + dpol, pol0,        pol0 + dpol])
    corner_hs   = jnp.array([h0,   h0,          h0 + dh,    h0 + dh])
    
    # Pre-compute (a, b) host-side
    a_corners = np.empty(4)
    b_corners = np.empty(4)
    for i in range(4):
        a, b = pol_mean_to_ab(float(corner_pols[i]), mean)
        a_corners[i] = float(a)
        b_corners[i] = float(b)
    a_base, b_base = pol_mean_to_ab(pol_baseline, mean)
    a_base, b_base = float(a_base), float(b_base)
    a_corners = jnp.asarray(a_corners)
    b_corners = jnp.asarray(b_corners)
    
    p = {**params, 'beta_M': beta_0}
    
    def one_run(a, b, h):
        pop = beta_populations(a, b, n_groups)
        C = contact_matrix(n_groups, h, pop)
        state0 = build_state0(pop)
        traj = simulate(step_fn, state0, beh, C, p, n_steps)
        # cumulative infected = 1 - S(t) - V(t if present)
        n_comp = traj.shape[0]
        cum_inf = 1.0 - traj[0].sum(axis=-1)
        if n_comp == 4:
            cum_inf = cum_inf - traj[-1].sum(axis=-1)
        return cum_inf   # shape (n_steps+1,)
    
    I_corners = jax.vmap(one_run)(a_corners, b_corners, corner_hs)
    I_baseline = one_run(a_base, b_base, 0.0)
    return I_corners, I_baseline

def sweep_point_vs_baseline(step_fn, build_state0, beh, beta_vals, pol, h,
                            params, n_steps, mean=0.5, n_groups=5,
                            pol_baseline=1e-3):
    """For each beta_0 in beta_vals, infection size at (pol, h) and at the
    baseline (pol_baseline, h=0), same mean.

    Returns:
        I_point:    (len(beta_vals),)
        I_baseline: (len(beta_vals),)
    """
    a_pt, b_pt = pol_mean_to_ab(float(pol), float(mean))
    a_pt, b_pt = float(a_pt), float(b_pt)
    a_base, b_base = pol_mean_to_ab(float(pol_baseline), float(mean))
    a_base, b_base = float(a_base), float(b_base)

    def one_run(a, b, h, beta_0):
        pop = beta_populations(a, b, n_groups)
        C = contact_matrix(n_groups, h, pop)
        state0 = build_state0(pop)
        p = {**params, 'beta_M': beta_0}
        final = simulate_final(step_fn, state0, beh, C, p, n_steps)
        n_comp = final.shape[0]
        S_sum = final[0].sum()
        V_sum = jnp.where(n_comp == 4, final[-1].sum(), 0.0)
        return 1.0 - S_sum - V_sum

    point_fn = jax.vmap(lambda beta_0: one_run(a_pt, b_pt, h, beta_0))
    base_fn = jax.vmap(lambda beta_0: one_run(a_base, b_base, 0.0, beta_0))
    return point_fn(beta_vals), base_fn(beta_vals)