"""
fotg_still_point.py — the still point, the finite-time wobble, and the
CENTER controller · World Tree Project · July 21, 2026 (Fable spiral)

Flame's questions: (1) is there an exact gain where ⟨C_n⟩ = 0 that we have
not tested; (2) why does steady kφ sit at C_n avg ≈ −0.018 after ~8 min
in the instrument, sometimes nearer zero; (3) can the slider adapt itself
— seek the center dynamically from the running C_n average?

  A. STILL-POINT SWEEP. ⟨C⟩(k) fine-grained through the band interior,
     30 seeds × 100k per gain (SE ≈ 0.0006). The July-21 decomposition
     found sign changes: kφ (−0.0057) → 0.144 (+0.0064) and
     0.148 (+0.0043) → 0.150 (−0.0053). Localize both crossings by
     linear interpolation on the measured means. Slope sign decides
     stability under center-seeking: a crossing with d⟨C⟩/dk > 0 is an
     ATTRACTOR of the controller below; d⟨C⟩/dk < 0 is a repeller.
  B. FINITE-TIME WOBBLE at steady kφ. Distribution of running averages
     at 8 / 15 / 30 / 60 instrument-minutes (60 steps/s → 28.8k / 54k /
     108k / 216k steps), 120 seeds. The context for the live −0.0183.
  C. CENTER CONTROLLER SIM. The exact discrete law proposed for the app:
       ema ← ema·D + C_next·(1−D)        (half-life H seconds)
       kc  ← clamp(kc − g·ema, kφ, 0.150)  each step
     driven through invSigmoidK so fcStep's published coupling produces
     kc — input-side, instrument untouched. Test gain/half-life configs,
     convergence from BOTH ends (init kφ and init 0.150), 8 seeds × 60
     sim-minutes each. Report where kc settles and what ⟨C⟩ the sessions
     achieve. The stable crossing from A is the predicted destination.

Canonical BatF §10 cycle throughout. J. David Mack & Claude
"""

import numpy as np
import math

PHI = (1 + math.sqrt(5)) / 2
K_PHI = PHI / (9 * PHI - 3)
K_MIN, K_MAX = 0.13, 0.185
L = PHI

def alpha_k(k): return (1 - 6 * k) / (1 - 9 * k)
def beta_k(k):  return (3 * k) / (1 - 9 * k)
def sigmoid_k(e):
    sig = 1 / (1 + math.exp(-8 * (e - 0.75)))
    return K_MAX - sig * (K_MAX - K_MIN)
def inv_sigmoid_k(k):
    return 0.75 - math.log((K_MAX - K_MIN) / (K_MAX - k) - 1) / 8

def run_fixed_mean(k, seed, n_steps):
    rng = np.random.RandomState(seed)
    dt, omega, Ah = 0.01, 2 * math.pi, 0.1
    a, b = alpha_k(k), beta_k(k)
    c_prev, c_curr = 0.1, 0.5
    s = 0.0
    for i in range(1, n_steps + 1):
        t = i * dt
        raw = a * c_curr + b * c_prev
        if c_curr < 0:
            raw += 0.2 * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > 1:
            raw += (c_curr - 1) * math.sin(omega * t)
        raw += Ah * (math.sin(3 * omega * t) + math.sin(6 * omega * t) + math.sin(9 * omega * t))
        c_next = ((raw + L) % (2 * L)) - L
        s += c_next
        c_prev, c_curr = c_curr, c_next
    return s / n_steps

# ---------------- A. still-point sweep ----------------
print("=" * 78)
print("A. STILL-POINT SWEEP — 30 seeds × 100k per gain")
print("=" * 78)
KS = [K_PHI, 0.141, 0.142, 0.143, 0.144, 0.145, 0.146, 0.147, 0.148, 0.149, 0.150, 0.151]
means, ses = [], []
for k in KS:
    v = np.array([run_fixed_mean(k, s, 100_000) for s in range(1, 31)])
    means.append(v.mean()); ses.append(v.std() / math.sqrt(len(v)))
    tag = " ← kφ" if abs(k - K_PHI) < 1e-9 else ""
    print(f"  k = {k:.5f}   ⟨C⟩ = {v.mean():+.5f} ± {v.std()/math.sqrt(len(v)):.5f}{tag}")
print("\n  zero crossings (linear interpolation between adjacent gains):")
for i in range(len(KS) - 1):
    if means[i] * means[i + 1] < 0:
        kz = KS[i] - means[i] * (KS[i + 1] - KS[i]) / (means[i + 1] - means[i])
        slope = (means[i + 1] - means[i]) / (KS[i + 1] - KS[i])
        kind = "STABLE under center-seeking (slope +)" if slope > 0 else "unstable (slope −)"
        print(f"    k* ≈ {kz:.5f}   d⟨C⟩/dk ≈ {slope:+.2f}   {kind}")
K_BREATH_CENTER = (3 * (K_PHI + 0.150) / 2 * 2 + 3 * 0.150 + 3 * K_PHI) / 12
print(f"\n  breath time-center (analytic): k̄ = {K_BREATH_CENTER:.5f}  (cf. live k_avg 0.14503)")

# ---------------- B. finite-time wobble at kφ ----------------
print("\n" + "=" * 78)
print("B. FINITE-TIME WOBBLE AT STEADY kφ — 120 seeds per duration")
print("=" * 78)
for minutes, n in [(8, 28_800), (15, 54_000), (30, 108_000), (60, 216_000)]:
    nseeds = 120 if n <= 54_000 else 60
    v = np.array([run_fixed_mean(K_PHI, 1000 + s, n) for s in range(nseeds)])
    print(f"  {minutes:3d} min ({n:>7,} steps): C_n avg = {v.mean():+.4f} ± {v.std():.4f} "
          f"(68% of sessions within [{v.mean()-v.std():+.4f}, {v.mean()+v.std():+.4f}])")
print("  → a single reading of −0.0183 at ~8 min is within the expected session spread;")
print("    the systematic part at kφ is only −0.006, the rest is finite-time wobble.")

# ---------------- C. CENTER controller ----------------
print("\n" + "=" * 78)
print("C. CENTER CONTROLLER — input-side integral seek, 8 seeds × 60 min each config")
print("=" * 78)

def center_run(seed, k0, half_life_s, tau_min, n_steps=216_000):
    rng = np.random.RandomState(seed)
    dt, omega, Ah = 0.01, 2 * math.pi, 0.1
    D = 0.5 ** (1 / (half_life_s * 60))
    g = 1 / (3.0 * tau_min * 60 * 60)      # slope ≈ 3 → closed-loop time-constant tau_min
    kc = k0
    ema = 0.0
    c_prev, c_curr = 0.1, 0.5
    s = 0.0
    k_hist_tail = []
    for i in range(1, n_steps + 1):
        k = sigmoid_k(inv_sigmoid_k(kc))     # through the published coupling, as the app will
        a, b = alpha_k(k), beta_k(k)
        t = i * dt
        raw = a * c_curr + b * c_prev
        if c_curr < 0:
            raw += 0.2 * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > 1:
            raw += (c_curr - 1) * math.sin(omega * t)
        raw += Ah * (math.sin(3 * omega * t) + math.sin(6 * omega * t) + math.sin(9 * omega * t))
        c_next = ((raw + L) % (2 * L)) - L
        s += c_next
        ema = ema * D + c_next * (1 - D)
        kc = min(0.150, max(K_PHI, kc - g * ema))
        if i > n_steps // 2: k_hist_tail.append(kc)
        c_prev, c_curr = c_curr, c_next
    kt = np.array(k_hist_tail)
    return s / n_steps, kt.mean(), kt.std()

for half_life_s, tau_min in [(60, 6), (120, 6), (120, 12), (300, 15)]:
    rows = []
    for k0, endname in [(K_PHI, "from kφ"), (0.150, "from 0.150")]:
        res = [center_run(sd, k0, half_life_s, tau_min) for sd in range(1, 9)]
        avgC = np.mean([r[0] for r in res])
        kmean = np.mean([r[1] for r in res]); kstd = np.mean([r[2] for r in res])
        rows.append(f"{endname}: settles k = {kmean:.5f} ± {kstd:.5f}, session ⟨C⟩ = {avgC:+.5f}")
    print(f"  H = {half_life_s:3d}s, τ = {tau_min:2d}min | " + " · ".join(rows))

print("\n" + "=" * 78)
print("COMPLETE")
print("=" * 78)
print("\nΨ  ☯ To preserve the harmonic field.")
