"""
fotg_cn_sweep.py — ⟨C_n⟩ across the operating window (companion to
fotg_cn_zero_check.py). Maps where the time-displacement balance holds
and where it breaks. 20 seeds × 100,000 steps per gain, canonical
BatF §10 cycle. World Tree Project · July 20, 2026 (Fable spiral).
"""

import numpy as np
import math

PHI = (1 + math.sqrt(5)) / 2
K_PHI = PHI / (9 * PHI - 3)

def alpha_k(k): return (1 - 6 * k) / (1 - 9 * k)
def beta_k(k):  return (3 * k) / (1 - 9 * k)

def run_fc(k, seed, n_steps=100_000, L=PHI):
    rng = np.random.RandomState(seed)
    dt, omega = 0.01, 2 * math.pi
    a, b = alpha_k(k), beta_k(k)
    c_prev, c_curr = 0.1, 0.5
    s = s2 = 0.0
    res = 0
    for i in range(1, n_steps + 1):
        t = i * dt
        raw = a * c_curr + b * c_prev
        if c_curr < 0:
            raw += 0.2 * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > 1.0:
            raw += (c_curr - 1.0) * math.sin(omega * t)
        raw += 0.1 * (math.sin(3 * omega * t) + math.sin(6 * omega * t) + math.sin(9 * omega * t))
        c_next = ((raw + L) % (2 * L)) - L
        s += c_next; s2 += c_next * c_next
        if c_next > 1.0: res += 1
        c_prev, c_curr = c_curr, c_next
    return s / n_steps, math.sqrt(s2 / n_steps), 100 * res / n_steps

print("⟨C_n⟩ SWEEP — where the balance holds (20 seeds × 100k per gain)")
print(f"{'k':>9s} {'⟨C⟩':>10s} {'seed σ':>8s} {'RMS':>7s} {'|⟨C⟩|/RMS':>10s} {'R%':>6s}")
KS = [0.140, K_PHI, 0.144, 0.148, 0.150, 0.152, 0.156, 0.158, 0.160, 0.162, 0.164, 0.165]
rows = []
for k in sorted(KS):
    out = [run_fc(k, s) for s in range(1, 21)]
    avg = np.mean([o[0] for o in out]); sd = np.std([o[0] for o in out])
    rms = np.mean([o[1] for o in out]); rpc = np.mean([o[2] for o in out])
    tag = "  ← kφ" if abs(k - K_PHI) < 1e-9 else ""
    rows.append((k, avg, rms))
    print(f"{k:9.5f} {avg:+10.5f} {sd:8.5f} {rms:7.4f} {abs(avg)/rms:10.4f} {rpc:6.1f}{tag}")
best = min(rows, key=lambda r: abs(r[1]) / r[2])
print(f"\nminimum |⟨C⟩|/RMS at k = {best[0]:.5f}")
print("\nΨ  ☯ To preserve the harmonic field.")
