# test_optimized_sweeps.py
import jax.numpy as jnp
from test_sweep import (
    test_h_sweep,
    test_susceptibility_symmetric_sweep,
    test_asymmetric_beta_sweep
)

def run_optimization_tests():
    # Test 1: Quick run with end states only
    print("Testing h sweep with end states only...")
    S_h, I_h, R_h = test_h_sweep(
        n_h=5,
        T=100,
        save_trajectory=False,
        batch_size=2
    )
    print(f"H sweep output shapes: {S_h.shape}, {I_h.shape}, {R_h.shape}")
    
    # Test 2: Full trajectory with small parameter space
    print("\nTesting h sweep with full trajectory...")
    S_h, I_h, R_h = test_h_sweep(
        n_h=5,
        T=100,
        save_trajectory=True,
        batch_size=2
    )
    print(f"H sweep with trajectory shapes: {S_h.shape}, {I_h.shape}, {R_h.shape}")
    
    # Test 3: Asymmetric beta sweep
    print("\nTesting asymmetric beta sweep...")
    range_a = {"m": 1.0, "M": 2.0, "n": 3}
    range_b = {"m": 1.0, "M": 2.0, "n": 3}
    S_b, I_b, R_b = test_asymmetric_beta_sweep(
        range_a=range_a,
        range_b=range_b,
        T=100,
        save_trajectory=False,
        batch_size=2
    )
    print(f"Beta sweep output shapes: {S_b.shape}, {I_b.shape}, {R_b.shape}")

    # Validate results
    def validate_results(S, I, R, name):
        print(f"\nValidating {name}...")
        total = S + I + R
        if len(total.shape) == 2:  # end states only
            is_conserved = jnp.allclose(total.sum(axis=-1), 1.0)
        else:  # full trajectory
            is_conserved = jnp.allclose(total.sum(axis=-1), 1.0)
        
        print(f"Population conserved: {is_conserved}")
        print(f"All values non-negative: {(S>=0).all() and (I>=0).all() and (R>=0).all()}")
    
    validate_results(S_h, I_h, R_h, "h sweep")
    validate_results(S_b, I_b, R_b, "beta sweep")