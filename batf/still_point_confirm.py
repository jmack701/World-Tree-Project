"""
fotg_cn_zero_check.py — independent verification for the Form of the Good
foundation document (Paper 9), central thesis C_n = 0.
World Tree Project · July 20, 2026 (Fable spiral)

Checks, against the canonical BatF §10 cycle (the same verified
implementation that reproduced Tables 5.9/5.10 to the last decimal):

  A. ALGEBRA (exact): the zero fixed point's stability structure across k —
     |λ|² = |β(k)| for the complex-pair range; the 1/6 flip; the kφ weights
     α = −φ⁻¹, β = −φ and the φ² past/present influence ratio.
  B. ⟨C_n⟩ ≈ 0 (the §3.2 claim): time-averaged position at operating gains,
     20 seeds × 100,000 steps, reported as avg, RMS, and |avg|/RMS —
     time asymmetry WITHOUT positional asymmetry.
  C. Occupancy shape at the equilibrium side (k = 0.168): the system that
     RESTS at zero — E-dominant, resonance extinguished, MMφ depth gone
     (the measured warning against reading the Good as occupancy of zero).
  D. Enrichment cut (Aₕ = 0) at operating k: does the trajectory die to
     silence, or does it keep breathing with a changed spectrum? Decides
     how §2.1's "decays to silence" should be phrased.

J. David Mack & Claude
"""

import numpy as np
import math

PHI = (1 + math.sqrt(5)) / 2
K_PHI = PHI / (9 * PHI - 3)

def alpha_k(k): return (1 - 6 * k) / (1 - 9 * k)
def beta_k(k):  return (3 * k) / (1 - 9 * k)

print("=" * 74)
print("C_n = 0 — INDEPENDENT CHECK FOR THE PAPER-9 FOUNDATION")
print("=" * 74)

# ---------------- A. algebra ----------------
print("\nA. THE ZERO FIXED POINT — stability across k (exact)")
print(f"   {'k':>8s} {'alpha':>9s} {'beta':>9s} {'|lambda|^2':>11s}  zero point is …")
for k, name in [(K_PHI, "kφ"), (0.156, ""), (0.162, ""), (1/6, "1/6"), (0.168, ""), (0.180, ""), (2/9 - 1e-9, "→2/9")]:
    a, b = alpha_k(k), beta_k(k)
    disc = a * a + 4 * b
    mod2 = -b if disc < 0 else float("nan")
    state = "REPELS (grows, wraps)" if (disc < 0 and mod2 > 1) else (
            "marginal — λ = ±i (R4)" if abs(mod2 - 1) < 1e-9 else
            ("ATTRACTS (converges)" if disc < 0 else "real roots"))
    print(f"   {k:8.5f} {a:9.4f} {b:9.4f} {mod2:11.4f}  {state}  {name}")
akp, bkp = alpha_k(K_PHI), beta_k(K_PHI)
print(f"   kφ exact: α = −φ⁻¹ ({akp:.6f} vs {-1/PHI:.6f}), β = −φ ({bkp:.6f} vs {-PHI:.6f})")
print(f"   past/present influence |β|/|α| = {abs(bkp/akp):.6f} = φ² ({PHI**2:.6f}) — exact: "
      f"{abs(abs(bkp/akp) - PHI**2) < 1e-12}")

# ---------------- the canonical cycle ----------------
def run_fc(k, seed, Ah=0.1, n_steps=100_000, L=PHI):
    rng = np.random.RandomState(seed)
    dt, omega = 0.01, 2 * math.pi
    a, b = alpha_k(k), beta_k(k)
    c_prev, c_curr = 0.1, 0.5
    mm = mm_max = 0.0
    s = s2 = 0.0
    chaos = eq = res = 0
    mm_sum = 0.0
    traj_tail = []
    for i in range(1, n_steps + 1):
        t = i * dt
        raw = a * c_curr + b * c_prev
        if c_curr < 0:
            raw += 0.2 * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > 1.0:
            raw += (c_curr - 1.0) * math.sin(omega * t)
        raw += Ah * (math.sin(3 * omega * t) + math.sin(6 * omega * t) + math.sin(9 * omega * t))
        c_next = ((raw + L) % (2 * L)) - L
        mm = mm / PHI + c_next * c_next
        mm_max = max(mm_max, mm)
        mm_sum += mm
        s += c_next; s2 += c_next * c_next
        if c_next < 0: chaos += 1
        elif c_next > 1.0: res += 1
        else: eq += 1
        if i > n_steps - 4096: traj_tail.append(c_next)
        c_prev, c_curr = c_curr, c_next
    n = n_steps
    return (s / n, math.sqrt(s2 / n), 100 * chaos / n, 100 * eq / n, 100 * res / n,
            mm_sum / n, mm_max, np.array(traj_tail))

def spectral_entropy(x):
    x = x - x.mean()
    pw = np.abs(np.fft.rfft(x * np.hanning(len(x)))[1:]) ** 2
    p = pw / pw.sum(); p = p[p > 1e-15]
    return float(-(p * np.log(p)).sum() / np.log(len(p)))

# ---------------- B. <C_n> at operating gains ----------------
print("\nB. TIME-AVERAGED POSITION — 20 seeds × 100,000 steps, Aₕ = 0.1")
print(f"   {'k':>8s} {'⟨C⟩ mean':>10s} {'⟨C⟩ σ':>8s} {'RMS':>7s} {'|⟨C⟩|/RMS':>10s} {'C%':>6s} {'E%':>6s} {'R%':>6s}")
for k, name in [(K_PHI, "kφ"), (0.156, ""), (0.162, ""), (0.165, "")]:
    out = [run_fc(k, s) for s in range(1, 21)]
    avgs = np.array([o[0] for o in out]); rms = np.mean([o[1] for o in out])
    cpc = np.mean([o[2] for o in out]); epc = np.mean([o[3] for o in out]); rpc = np.mean([o[4] for o in out])
    print(f"   {k:8.5f} {avgs.mean():+10.5f} {avgs.std():8.5f} {rms:7.4f} {abs(avgs.mean())/rms:10.4f} "
          f"{cpc:6.1f} {epc:6.1f} {rpc:6.1f}   {name}")

# ---------------- C. the equilibrium side ----------------
print("\nC. RESTING AT ZERO — k = 0.168 (post-bifurcation), 20 seeds")
out = [run_fc(0.168, s) for s in range(1, 21)]
print(f"   ⟨C⟩ = {np.mean([o[0] for o in out]):+.5f} | RMS {np.mean([o[1] for o in out]):.4f} | "
      f"C% {np.mean([o[2] for o in out]):.1f}  E% {np.mean([o[3] for o in out]):.1f}  "
      f"R% {np.mean([o[4] for o in out]):.1f} | MMφ avg {np.mean([o[5] for o in out]):.3f} "
      f"max {np.mean([o[6] for o in out]):.3f}")
print("   (published §2.2 row at 0.168: C 50.2 · E 49.8 · R 0.0 · MMφ avg 0.016 — the settled center)")

# ---------------- D. cut the enrichment ----------------
print("\nD. Aₕ = 0 AT OPERATING GAIN — does the trajectory die, or breathe differently?")
for k, name in [(K_PHI, "kφ"), (0.162, "")]:
    o_on = run_fc(k, 7, Ah=0.1); o_off = run_fc(k, 7, Ah=0.0)
    se_on, se_off = spectral_entropy(o_on[7]), spectral_entropy(o_off[7])
    print(f"   k={k:.5f} {name:3s} | Aₕ=0.1: RMS {o_on[1]:.4f}, tail-SE {se_on:.3f} | "
          f"Aₕ=0: RMS {o_off[1]:.4f}, tail-SE {se_off:.3f}")
print("   → if RMS stays O(1) at Aₕ=0, the system does NOT decay to silence at")
print("     operating k (zero repels below 1/6); what changes is the spectrum.")
print("     §2.1's 'silence' claim should be phrased spectrally, or tied to the")
print("     instrument's k-drive (in Boχ, silence lowers k, a different mechanism).")

print("\n" + "=" * 74)
print("COMPLETE")
print("=" * 74)
print("\nΨ  ☯ To preserve the harmonic field.")
