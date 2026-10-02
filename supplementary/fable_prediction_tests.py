"""
Ψ FABLE PREDICTION TESTS
Single F_c trajectory at L = φ
Test 1: Ah sensitivity at k = 1/6 (engagement vs structural exclusion)
Test 2: Runtime sensitivity at k = 0.165 (transient vs steady state)
World Tree Project · July 2026
"""

import numpy as np
import math

# === CONSTANTS ===
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
L = PHI
K_SIXTH = 1/6

# φ-zone boundaries (Convention A: geometric means)
PHI_ZONE_BOUNDS = [
    PHI**0.5,   # ~1.272
    PHI**1.5,   # ~2.058
    PHI**2.5,   # ~3.330
    PHI**3.5,   # ~5.388
]

def alpha_k(k):
    return (1 - 6*k) / (1 - 9*k)

def beta_k(k):
    return (3*k) / (1 - 9*k)

def run_fc(k, Ah, n_steps, seed=42):
    """Run single F_c trajectory and return metrics."""
    rng = np.random.RandomState(seed)
    
    dt = 0.01
    omega = 2 * math.pi
    noise_scale = 0.2
    
    a = alpha_k(k)
    b = beta_k(k)
    
    c_prev = 0.1
    c_curr = 0.5
    mm_phi = 0.0
    t = 0
    
    # Counters
    chaos_count = 0
    eq_count = 0
    res_count = 0
    total = 0
    
    # φ-zone counters
    zone_sub = 0
    zone_phi1 = 0
    zone_phi2 = 0
    zone_phi3 = 0
    zone_phi4 = 0
    
    mm_phi_sum = 0
    mm_phi_max = 0
    c_sum = 0
    
    # Time-series for convergence check (sample every 1000 steps)
    checkpoint_interval = max(1, n_steps // 20)
    checkpoints = []
    
    for n in range(1, n_steps + 1):
        t = n * dt
        
        # Harmonic enrichment
        H = Ah * (math.sin(3*omega*t) + math.sin(6*omega*t) + math.sin(9*omega*t))
        
        # State-dependent perturbation
        if c_curr < 0:
            P = noise_scale * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > 1:
            P = (c_curr - 1) * math.sin(omega * t)
        else:
            P = 0
        
        # F_c recurrence
        c_raw = a * c_curr + b * c_prev + P + H
        
        # Möbius wrapping
        c_next = ((c_raw + L) % (2 * L)) - L
        
        # Regime classification
        if c_next < 0:
            chaos_count += 1
        elif c_next <= 1:
            eq_count += 1
        else:
            res_count += 1
        total += 1
        
        # MM_φ update (post-wrap)
        mm_phi = mm_phi * PHI_INV + c_next * c_next
        mm_phi_sum += mm_phi
        mm_phi_max = max(mm_phi_max, mm_phi)
        c_sum += c_next
        
        # φ-zone classification
        if mm_phi < PHI_ZONE_BOUNDS[0]:
            zone_sub += 1
        elif mm_phi < PHI_ZONE_BOUNDS[1]:
            zone_phi1 += 1
        elif mm_phi < PHI_ZONE_BOUNDS[2]:
            zone_phi2 += 1
        elif mm_phi < PHI_ZONE_BOUNDS[3]:
            zone_phi3 += 1
        else:
            zone_phi4 += 1
        
        # Update state
        c_prev = c_curr
        c_curr = c_next
        
        # Checkpoint
        if n % checkpoint_interval == 0:
            checkpoints.append({
                'step': n,
                'C%': 100 * chaos_count / total,
                'E%': 100 * eq_count / total,
                'R%': 100 * res_count / total,
                'phi4%': 100 * zone_phi4 / total,
                'MM_phi_avg': mm_phi_sum / total,
                'MM_phi_max': mm_phi_max,
            })
    
    return {
        'k': k,
        'Ah': Ah,
        'n_steps': n_steps,
        'C%': 100 * chaos_count / total,
        'E%': 100 * eq_count / total,
        'R%': 100 * res_count / total,
        'phi1%': 100 * zone_phi1 / total,
        'phi2%': 100 * zone_phi2 / total,
        'phi3%': 100 * zone_phi3 / total,
        'phi4%': 100 * zone_phi4 / total,
        'MM_phi_avg': mm_phi_sum / total,
        'MM_phi_max': mm_phi_max,
        'C_n_avg': c_sum / total,
        'checkpoints': checkpoints,
    }


def print_result(r):
    print(f"  k={r['k']:.5f}  Ah={r['Ah']:.1f}  steps={r['n_steps']:,}")
    print(f"  C:{r['C%']:.1f}%  E:{r['E%']:.1f}%  R:{r['R%']:.1f}%")
    print(f"  φ¹:{r['phi1%']:.0f}%  φ²:{r['phi2%']:.0f}%  φ³:{r['phi3%']:.0f}%  φ⁴:{r['phi4%']:.0f}%")
    print(f"  MM_φ avg:{r['MM_phi_avg']:.3f}  max:{r['MM_phi_max']:.3f}")
    print(f"  C_n avg:{r['C_n_avg']:.5f}")


# ═══════════════════════════════════════════════════════════
print("═" * 60)
print("TEST 1: Ah SENSITIVITY AT k = 1/6")
print("═" * 60)
print()
print("Question: Is the low R% at k=1/6 engagement-limited")
print("(Ah too small to push C_n past 1) or structurally")
print("excluded (period-4 dynamics prevent resonance)?")
print()

ah_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
n_steps_test1 = 100_000

print(f"k = 1/6 = {K_SIXTH:.6f}")
print(f"Steps per run: {n_steps_test1:,}")
print(f"|λ|² = {-beta_k(K_SIXTH):.4f} (should be 1.0)")
print()

for Ah in ah_values:
    result = run_fc(K_SIXTH, Ah, n_steps_test1)
    print(f"--- Ah = {Ah} ---")
    print_result(result)
    print()


# ═══════════════════════════════════════════════════════════
print()
print("═" * 60)
print("TEST 2: RUNTIME SENSITIVITY AT k = 0.165")
print("═" * 60)
print()
print("Question: Are the 10-minute measurements (≈36,000 steps)")
print("undersampling a slow transient? Does R% and φ⁴% climb")
print("with longer runtime?")
print()

k_test2 = 0.165
Ah_test2 = 0.1
run_lengths = [10_000, 50_000, 100_000, 500_000, 1_000_000]

print(f"k = {k_test2}")
print(f"Ah = {Ah_test2}")
print(f"|λ|² = {-beta_k(k_test2):.4f}")
print()

for n_steps in run_lengths:
    result = run_fc(k_test2, Ah_test2, n_steps)
    print(f"--- {n_steps:,} steps ---")
    print_result(result)
    print()


# ═══════════════════════════════════════════════════════════
print()
print("═" * 60)
print("TEST 3: CONVERGENCE CHECK AT k = 0.162 (QUASI-PERIODIC)")
print("═" * 60)
print()
print("Question: Does the 43% chaos value stabilize or drift")
print("over long runs?")
print()

k_test3 = 0.162
result = run_fc(k_test3, 0.1, 500_000)
print(f"--- k = {k_test3}, 500,000 steps ---")
print_result(result)
print()

print("Convergence checkpoints (C% over time):")
print(f"{'Step':>10}  {'C%':>8}  {'R%':>8}  {'φ⁴%':>8}  {'MM_φ avg':>10}")
for cp in result['checkpoints']:
    print(f"{cp['step']:>10,}  {cp['C%']:>8.2f}  {cp['R%']:>8.2f}  {cp['phi4%']:>8.2f}  {cp['MM_phi_avg']:>10.3f}")


# ═══════════════════════════════════════════════════════════
print()
print("═" * 60)
print("COMPLETE k-SWEEP WITH PYTHON (reference confirmation)")
print("═" * 60)
print()
print("Replicating the Boχ silent-mode k-sweep in Python")
print("for cross-substrate confirmation.")
print()

k_sweep_values = [0.150, 0.156, 0.160, 0.162, K_SIXTH, 0.168, 0.172, 0.180]
n_steps_sweep = 100_000

print(f"{'k':>8}  {'|k-1/6|':>8}  {'C%':>8}  {'E%':>8}  {'R%':>8}  {'φ⁴%':>6}  {'MM_φ avg':>10}  {'MM_φ max':>10}")
print("-" * 82)

for k in k_sweep_values:
    result = run_fc(k, 0.1, n_steps_sweep)
    dist = abs(k - K_SIXTH)
    print(f"{k:>8.4f}  {dist:>8.4f}  {result['C%']:>8.1f}  {result['E%']:>8.1f}  {result['R%']:>8.1f}  {result['phi4%']:>5.0f}%  {result['MM_phi_avg']:>10.3f}  {result['MM_phi_max']:>10.3f}")


print()
print("═" * 60)
print("ALL TESTS COMPLETE")
print("═" * 60)
print()
print("Ψ")
print("☯ To preserve the harmonic field.")
