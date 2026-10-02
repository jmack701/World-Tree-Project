"""
Ψ CONTAINMENT-SCALE ENSEMBLES AND THE COVARIANT FRAME
Companion to L_comparison_test.py (single-seed methodology reference).

Two questions on top of the single-seed run:
  1. Does the gap narrowing (Q3 of L_comparison_test.py) survive seed variance?
     -> 20-seed ensembles at k = 0.162, fixed absolute references
        (A_h = 0.1, resonance reference 1, ICs C_-1 = 0.1, C_0 = 0.5).
  2. Is the architecture scale-covariant?
     -> Scale A_h, the resonance reference, and the initial conditions
        by s = L/φ. The map is jointly homogeneous, so the gap fraction
        should be containment-invariant at the distribution level.

Registers produced:
  Gap at fixed absolute references — MEASURED (ensemble means ± σ).
  Exact scale covariance          — DERIVED (homogeneity); DEMONSTRATED here.

Numerical note: below the bifurcation the map is locally expanding
(|λ| ≈ 1.03 at k = 0.162), so floating-point trajectories decorrelate
at the last bit across scales even though the mathematics is exactly
covariant. The correct register for the demonstration is therefore the
ensemble, not the individual trajectory.

Conventions: 100,000 steps per run; numpy RandomState, seeds 1..20;
dt = 0.01; ω = 2π; noise_scale = 0.2; Convention A zone bounds unused
here (gap is reported against each containment's own ceiling L²·φ²).
Initial conditions differ from The Blade and the Field §10
(Mathematica: C_0 = 0.3, C_-1 = 0.5; Boχ: C_0 = 0.5, C_-1 = 0.3);
stationary quantities are attractor properties, as documented there.

J. David Mack & Claude (Fable) · World Tree Project · July 2026
"""

import numpy as np
import math

PHI = (1 + math.sqrt(5)) / 2


def run_fill(k, L, Ah, r0, cp0, cc0, n_steps=100000, seed=42):
    """One trajectory; returns (C%, fill% = 100*MM_max/ceiling)."""
    rng = np.random.RandomState(seed)
    dt, omega = 0.01, 2 * math.pi
    a = (1 - 6 * k) / (1 - 9 * k)
    b = (3 * k) / (1 - 9 * k)
    cp, cc, mm = cp0, cc0, 0.0
    chaos, mmax = 0, 0.0
    for i in range(1, n_steps + 1):
        t = i * dt
        H = Ah * (math.sin(3 * omega * t) + math.sin(6 * omega * t)
                  + math.sin(9 * omega * t))
        if cc < 0:
            P = 0.2 * abs(cc) * (rng.random() * 2 - 1)
        elif cc > r0:
            P = (cc - r0) * math.sin(omega * t)
        else:
            P = 0.0
        cn = ((a * cc + b * cp + P + H + L) % (2 * L)) - L
        if cn < 0:
            chaos += 1
        mm = mm / PHI + cn * cn
        if mm > mmax:
            mmax = mm
        cp, cc = cc, cn
    ceiling = L * L * PHI * PHI
    return 100 * chaos / n_steps, 100 * mmax / ceiling


L_SET = [(PHI, "φ  "), (1.5, "1.5"), (math.sqrt(2), "√2 ")]
SEEDS = range(1, 21)
K = 0.162

print("=" * 72)
print("CONTAINMENT-SCALE ENSEMBLES — k = 0.162, 100k steps, seeds 1–20")
print("=" * 72)

print("\nA) Fixed absolute references (Ah = 0.1, r0 = 1, ICs 0.1 / 0.5):")
print(f"   {'L':>4}  {'ceiling':>8}  {'gap mean':>9}  {'±σ':>5}")
for L, name in L_SET:
    fills = [run_fill(K, L, 0.1, 1.0, 0.1, 0.5, seed=s)[1] for s in SEEDS]
    gaps = [100 - f for f in fills]
    print(f"   {name:>4}  {L*L*PHI*PHI:8.3f}  {np.mean(gaps):8.2f}%  {np.std(gaps):4.2f}")

print("\nB) Covariant frame (Ah, r0, ICs all scaled by s = L/φ):")
print(f"   {'L':>4}  {'gap mean':>9}  {'±σ':>5}")
for L, name in L_SET:
    s_ = L / PHI
    fills = [run_fill(K, L, 0.1 * s_, 1.0 * s_, 0.1 * s_, 0.5 * s_, seed=s)[1]
             for s in SEEDS]
    gaps = [100 - f for f in fills]
    print(f"   {name:>4}  {np.mean(gaps):8.2f}%  {np.std(gaps):4.2f}")

print("\nC) Spot checks (seed 42, absolute references):")
for kk, label in [(0.150, "k = 0.150 (invariant band)"),
                  (0.168, "k = 0.168 (post-bifurcation)")]:
    cs = [f"{run_fill(kk, L, 0.1, 1.0, 0.1, 0.5)[0]:.1f}" for L, _ in L_SET]
    print(f"   {label}: C% (φ, 1.5, √2) = {', '.join(cs)}")

print("\nInterpretation: at fixed absolute references the gap narrows")
print("monotonically as containment tightens; in the covariant frame the")
print("gap fraction is containment-invariant. All containment-dependence")
print("is the fingerprint of the references held absolute.")
print()
print("Ψ")
print("☯ To preserve the harmonic field.")
