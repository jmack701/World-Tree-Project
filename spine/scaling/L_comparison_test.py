"""
Ψ CONTAINMENT SCALE COMPARISON
F_c single-trajectory k-sweep at three containment radii:
  L = φ (the derived condition, ceiling = φ⁴ = 6.854)
  L = √2 (E8 root length, ceiling = 2φ² = 5.236)
  L = 1.5 (legacy, ceiling = L²φ² = 5.890)

Question: Is the invariant containment-independent?
Does the bifurcation point move? Does the regime distribution shift?

J. David Mack & Claude (Opus 4.6)
World Tree Project · July 2026
"""

import numpy as np
import math

PHI = (1 + math.sqrt(5)) / 2
SQRT2 = math.sqrt(2)
PHI_INV = 1 / PHI

def alpha_k(k):
    return (1 - 6*k) / (1 - 9*k)

def beta_k(k):
    return (3*k) / (1 - 9*k)

def run_fc(k, L, n_steps=100000, seed=42):
    rng = np.random.RandomState(seed)
    dt = 0.01
    omega = 2 * math.pi
    noise_scale = 0.2
    Ah = 0.1
    a = alpha_k(k)
    b = beta_k(k)

    c_prev = 0.1
    c_curr = 0.5
    mm_phi = 0.0

    chaos_count = 0
    eq_count = 0
    res_count = 0
    total = 0
    mm_phi_sum = 0
    mm_phi_max = 0
    c_sum = 0

    # Convention A zone bounds (geometric means)
    zone_bounds = [PHI**0.5, PHI**1.5, PHI**2.5, PHI**3.5]
    zone_counts = [0, 0, 0, 0, 0]  # sub, phi1, phi2, phi3, phi4

    for n in range(1, n_steps + 1):
        t = n * dt
        H = Ah * (math.sin(3*omega*t) + math.sin(6*omega*t) + math.sin(9*omega*t))

        if c_curr < 0:
            P = noise_scale * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > 1:
            P = (c_curr - 1) * math.sin(omega * t)
        else:
            P = 0

        c_raw = a * c_curr + b * c_prev + P + H
        c_next = ((c_raw + L) % (2 * L)) - L

        if c_next < 0:
            chaos_count += 1
        elif c_next <= 1:
            eq_count += 1
        else:
            res_count += 1
        total += 1

        mm_phi = mm_phi * PHI_INV + c_next * c_next
        mm_phi_sum += mm_phi
        mm_phi_max = max(mm_phi_max, mm_phi)
        c_sum += c_next

        if mm_phi < zone_bounds[0]:
            zone_counts[0] += 1
        elif mm_phi < zone_bounds[1]:
            zone_counts[1] += 1
        elif mm_phi < zone_bounds[2]:
            zone_counts[2] += 1
        elif mm_phi < zone_bounds[3]:
            zone_counts[3] += 1
        else:
            zone_counts[4] += 1

        c_prev = c_curr
        c_curr = c_next

    ceiling = L * L * PHI * PHI
    gap_pct = 100 * (ceiling - mm_phi_max) / ceiling if ceiling > 0 else 0

    return {
        'k': k, 'L': L, 'ceiling': ceiling,
        'C%': 100 * chaos_count / total,
        'E%': 100 * eq_count / total,
        'R%': 100 * res_count / total,
        'phi4%': 100 * zone_counts[4] / total,
        'MM_avg': mm_phi_sum / total,
        'MM_max': mm_phi_max,
        'gap%': gap_pct,
        'C_n_avg': c_sum / total,
    }

# ═══════════════════════════════════════════════════════════
k_values = [0.150, 0.156, 0.160, 0.162, 1/6, 0.168, 0.172, 0.180]
L_values = [
    (PHI, "φ", PHI**4),
    (SQRT2, "√2", 2 * PHI**2),
    (1.5, "1.5", 1.5**2 * PHI**2),
]

print("=" * 80)
print("CONTAINMENT SCALE COMPARISON")
print("=" * 80)
print()
print(f"  L = φ   = {PHI:.6f}   ceiling = φ⁴       = {PHI**4:.4f}")
print(f"  L = √2  = {SQRT2:.6f}   ceiling = 2φ²      = {2*PHI**2:.4f}")
print(f"  L = 1.5 = 1.500000   ceiling = 2.25·φ²  = {1.5**2*PHI**2:.4f}")
print(f"  Steps per run: 100,000")
print()

# Run all sweeps
all_results = {}
for L_val, L_name, ceiling in L_values:
    print("=" * 80)
    print(f"k-SWEEP AT L = {L_name} ({L_val:.6f}), ceiling = {ceiling:.4f}")
    print("=" * 80)
    print()
    print(f"{'k':>8}  {'C%':>8}  {'E%':>8}  {'R%':>8}  {'φ⁴%':>6}  {'MM_avg':>8}  {'MM_max':>8}  {'gap%':>8}  {'C_n avg':>10}")
    print("-" * 90)

    results = []
    for k in k_values:
        r = run_fc(k, L_val)
        results.append(r)
        print(f"{k:>8.4f}  {r['C%']:>8.1f}  {r['E%']:>8.1f}  {r['R%']:>8.1f}  {r['phi4%']:>5.0f}%  {r['MM_avg']:>8.3f}  {r['MM_max']:>8.3f}  {r['gap%']:>7.1f}%  {r['C_n_avg']:>10.5f}")

    all_results[L_name] = results
    print()

# ═══════════════════════════════════════════════════════════
print("=" * 80)
print("SIDE-BY-SIDE COMPARISON AT KEY k VALUES")
print("=" * 80)
print()

key_k = [0.150, 0.162, 1/6, 0.168]
key_labels = ["0.150 (invariant band)", "0.162 (quasi-periodic dip)",
              "1/6 (bifurcation)", "0.168 (post-bifurcation)"]

for k_val, label in zip(key_k, key_labels):
    print(f"  k = {label}:")
    for L_name in ["φ", "√2", "1.5"]:
        r = [x for x in all_results[L_name] if abs(x['k'] - k_val) < 0.0001][0]
        print(f"    L={L_name:>3s}: C:{r['C%']:5.1f}%  R:{r['R%']:5.1f}%  φ⁴:{r['phi4%']:4.0f}%  "
              f"MM_max:{r['MM_max']:.3f}/{r['ceiling']:.3f}  gap:{r['gap%']:.1f}%  C_n:{r['C_n_avg']:.5f}")
    print()

# ═══════════════════════════════════════════════════════════
print("=" * 80)
print("INVARIANT CHECK — C% ACROSS CONTAINMENT SCALES")
print("=" * 80)
print()

print(f"{'k':>8}", end="")
for L_name in ["φ", "√2", "1.5"]:
    print(f"  {'C%(L='+L_name+')':>12}", end="")
print(f"  {'Spread':>8}")
print("-" * 52)

for i, k in enumerate(k_values):
    vals = [all_results[ln][i]['C%'] for ln in ["φ", "√2", "1.5"]]
    spread = max(vals) - min(vals)
    print(f"{k:>8.4f}", end="")
    for v in vals:
        print(f"  {v:>12.1f}", end="")
    print(f"  {spread:>8.2f}")

# ═══════════════════════════════════════════════════════════
print()
print("=" * 80)
print("THREE QUESTIONS ANSWERED")
print("=" * 80)
print()

# Q1: Is the invariant containment-independent?
c_at_150 = [all_results[ln][0]['C%'] for ln in ["φ", "√2", "1.5"]]
spread_150 = max(c_at_150) - min(c_at_150)
print(f"Q1: Is the chaos invariant containment-independent?")
print(f"    C% at k=0.150: φ={c_at_150[0]:.1f}, √2={c_at_150[1]:.1f}, 1.5={c_at_150[2]:.1f}")
print(f"    Spread: {spread_150:.2f}")
if spread_150 < 2.0:
    print(f"    → YES. Spread < 2% across containment scales.")
else:
    print(f"    → NO. Spread = {spread_150:.1f}%, containment affects the invariant.")

# Q2: Does the bifurcation point move?
r_at_168 = {ln: [x for x in all_results[ln] if abs(x['k'] - 0.168) < 0.001][0] for ln in ["φ", "√2", "1.5"]}
print(f"\nQ2: Does the bifurcation point move with L?")
print(f"    R% at k=0.168: φ={r_at_168['φ']['R%']:.1f}, √2={r_at_168['√2']['R%']:.1f}, 1.5={r_at_168['1.5']['R%']:.1f}")
if all(r_at_168[ln]['R%'] < 1.0 for ln in ["φ", "√2", "1.5"]):
    print(f"    → NO. Resonance is zero at k=0.168 for all L. Bifurcation stays at k=1/6.")
else:
    print(f"    → The bifurcation may shift with L.")

# Q3: Does L affect the gap?
gaps = {ln: [x for x in all_results[ln] if abs(x['k'] - 0.162) < 0.001][0] for ln in ["φ", "√2", "1.5"]}
print(f"\nQ3: Does L affect the gap between MM_max and the ceiling?")
for ln in ["φ", "√2", "1.5"]:
    g = gaps[ln]
    print(f"    L={ln:>3s}: MM_max={g['MM_max']:.3f}, ceiling={g['ceiling']:.3f}, gap={g['gap%']:.1f}%")

print()
print("=" * 80)
print("COMPLETE")
print("=" * 80)
print()
print("Ψ")
print("☯ To preserve the harmonic field.")
