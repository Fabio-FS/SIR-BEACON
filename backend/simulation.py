import jax.numpy as jnp

from sir_model.contact import beta_populations, contact_matrix, pol_mean_to_ab
from sir_model.simulate import simulate
from sir_model.models import step_SIRM, step_SIRT, step_SIRV
from sir_model.sweep import sweep_pol_homophily, sweep_point_vs_baseline


N_GROUPS = 5
INIT_INF = 1e-6
DT = 0.25
N_STEPS = 3000
RECOVERY_RATE = 0.1

MU_MAX = 0.8
TESTING_RATE_MAX = 0.198
VACCINATION_RATE_MAX = 0.01

GRID_POINTS = 60
POL_VALS = jnp.linspace(0.01, 0.99, GRID_POINTS)
H_VALS = jnp.linspace(0.0, 6.0, GRID_POINTS)
R0_VALS = jnp.linspace(0.5, 5.0, GRID_POINTS)


def make_behavior(model):
    if model == "SIRM":
        return jnp.linspace(0, MU_MAX, N_GROUPS)
    if model == "SIRT":
        return jnp.linspace(0, TESTING_RATE_MAX, N_GROUPS)
    if model == "SIRV":
        return jnp.linspace(0, VACCINATION_RATE_MAX, N_GROUPS)


def get_step_fn(model):
    if model == "SIRM":
        return step_SIRM
    if model == "SIRT":
        return step_SIRT
    if model == "SIRV":
        return step_SIRV


def build_state0_3(pop):
    return jnp.stack([pop * (1 - INIT_INF), pop * INIT_INF, jnp.zeros_like(pop)])


def build_state0_4(pop):
    Z = jnp.zeros_like(pop)
    return jnp.stack([pop * (1 - INIT_INF), pop * INIT_INF, Z, Z])


def get_state_builder(model):
    if model == "SIRV":
        return build_state0_4
    return build_state0_3


def make_params(r0):
    return {"beta_M": r0 * RECOVERY_RATE, "recovery_rate": RECOVERY_RATE, "dT": DT}


def population(pol, mean=0.5):
    """Fraction of population in each behavior group."""
    a, b = pol_mean_to_ab(pol, mean)
    pop = beta_populations(a, b, N_GROUPS)
    return pop.tolist()


def matrix(pol, h, mean=0.5):
    """Contact matrix for the given structure."""
    a, b = pol_mean_to_ab(pol, mean)
    pop = beta_populations(a, b, N_GROUPS)
    C = contact_matrix(N_GROUPS, h, pop)
    return C.tolist()


def run_trajectory(model, r0, pol, h, mean=0.5):
    """Single epidemic. Returns S(t), I(t), R(t) summed across groups,
    plus V(t) for SIR-V."""
    a, b = pol_mean_to_ab(pol, mean)
    pop = beta_populations(a, b, N_GROUPS)
    C = contact_matrix(N_GROUPS, h, pop)
    state0 = get_state_builder(model)(pop)
    beh = make_behavior(model)
    params = make_params(r0)
    traj = simulate(get_step_fn(model), state0, beh, C, params, N_STEPS)

    out = {
        "S": traj[0].sum(axis=-1).tolist(),
        "I": traj[1].sum(axis=-1).tolist(),
        "R": traj[2].sum(axis=-1).tolist(),
    }
    if model == "SIRV":
        out["V"] = traj[3].sum(axis=-1).tolist()
    return out


def run_sweep_polarization(model, r0, h, mean=0.5):
    """I(infinity) as polarization varies, homophily fixed."""
    beh = make_behavior(model)
    params = make_params(r0)
    final = sweep_pol_homophily(
        get_step_fn(model), get_state_builder(model), beh,
        POL_VALS, [h], params, N_STEPS, mean=mean, n_groups=N_GROUPS,
    )
    infected = total_infected(final)
    return {"pol": POL_VALS.tolist(), "infected": infected[:, 0].tolist()}


def run_sweep_homophily(model, r0, pol, mean=0.5):
    """I(infinity) as homophily varies, polarization fixed."""
    beh = make_behavior(model)
    params = make_params(r0)
    final = sweep_pol_homophily(
        get_step_fn(model), get_state_builder(model), beh,
        [pol], H_VALS, params, N_STEPS, mean=mean, n_groups=N_GROUPS,
    )
    infected = total_infected(final)
    return {"h": H_VALS.tolist(), "infected": infected[0, :].tolist()}


def run_sweep_severity(model, pol, h, mean=0.5):
    """Infected total vs disease severity (R0), at fixed pol and h.
    Returns the structured-model curve and the homogeneous baseline."""
    beh = make_behavior(model)
    params = make_params(1.0)
    beta_vals = R0_VALS * RECOVERY_RATE
    I_point, I_baseline = sweep_point_vs_baseline(
        get_step_fn(model), get_state_builder(model), beh,
        beta_vals, pol, h, params, N_STEPS, mean=mean, n_groups=N_GROUPS,
    )
    return {
        "r0": R0_VALS.tolist(),
        "structured": I_point.tolist(),
        "baseline": I_baseline.tolist(),
    }


def total_infected(final):
    """final has shape (n_pol, n_h, n_comp, n_groups). Returns (n_pol, n_h)."""
    n_comp = final.shape[2]
    S_sum = final[:, :, 0, :].sum(axis=-1)
    if n_comp == 4:
        V_sum = final[:, :, -1, :].sum(axis=-1)
        return 1.0 - S_sum - V_sum
    return 1.0 - S_sum