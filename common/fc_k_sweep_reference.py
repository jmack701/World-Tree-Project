"""fc_k_sweep_reference.py — canonical nine-point k-sweep for The Spine §2.2.

The deposited reference for Table §2.2. Implements the BatF §10 cycle exactly
(Steps 1, 1b, 2, 3, 4, 6; regime classification; Convention-A phi4 zone).
Initial conditions: the §10 prescribed set (C0 = 0.3, C-1 = 0.5, MMphi(0) per spec).
Single run, single seed. All values in the published table and its cross-substrate
prose are drawn from this script's output verbatim.

Sweep header standard: steps, seed, initial conditions, and thresholds print first.
World Tree Project · July 2026 · seed 42.
"""
import numpy as np
import math

PHI = (1 + 5**0.5) / 2
PHI_INV = 1 / PHI
N_STEPS = 100_000
SEED = 42
C0, CM1 = 0.3, 0.5
AH, NOISE_SCALE, RES_REF = 0.1, 0.2, 1.0
DT, OMEGA = 0.01, 2 * math.pi
Z4_THRESHOLD = PHI ** 3.5          # Convention A geometric-mean boundary

def alpha(k): return (1 - 6 * k) / (1 - 9 * k)
def beta(k):  return 3 * k / (1 - 9 * k)

def run_fc(k):
    rng = np.random.RandomState(SEED)
    a, b = alpha(k), beta(k)
    c_prev, c_curr = CM1, C0
    mm = CM1 * CM1 * PHI_INV + C0 * C0
    cc = ec = rc = z4 = 0
    mm_sum = mm_max = c_sum = 0.0
    for n in range(1, N_STEPS + 1):
        t = n * DT
        c_raw = a * c_curr + b * c_prev                                  # Step 1
        if c_curr < 0:                                                   # Step 1b
            c_raw += NOISE_SCALE * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > RES_REF:
            c_raw += (c_curr - RES_REF) * math.sin(OMEGA * t)
        c_raw += AH * (math.sin(3*OMEGA*t) + math.sin(6*OMEGA*t) + math.sin(9*OMEGA*t))  # Step 2
        c_next = ((c_raw + PHI) % (2 * PHI)) - PHI                       # Step 3 (L = phi)
        mm = mm * PHI_INV + c_next * c_next                              # Step 4 (post-wrap)
        if c_next < 0:      cc += 1
        elif c_next <= 1:   ec += 1
        else:               rc += 1
        z4 += (mm >= Z4_THRESHOLD)
        mm_sum += mm; mm_max = max(mm_max, mm); c_sum += c_next
        c_prev, c_curr = c_curr, c_next                                  # Step 6
    N = N_STEPS
    return dict(C=100*cc/N, E=100*ec/N, R=100*rc/N, phi4=100*z4/N,
                mm_avg=mm_sum/N, mm_max=mm_max, c_avg=c_sum/N)

if __name__ == "__main__":
    K_VALUES = [0.150, 0.156, 0.160, 0.162, 0.165, 1/6, 0.168, 0.172, 0.180]
    print("fc_k_sweep_reference.py | steps=%d | seed=%d | C0=%.1f C-1=%.1f | L=phi"
          % (N_STEPS, SEED, C0, CM1))
    print("Ah=%.1f noise=%.1f | Convention A phi4 zone: MMphi >= phi^3.5 = %.3f"
          % (AH, NOISE_SCALE, Z4_THRESHOLD))
    print("-" * 78)
    print(f"{'k':>8} | {'C%':>5} {'E%':>5} {'R%':>5} | {'phi4%':>5} | {'MM_avg':>7} {'MM_max':>7} | {'C_avg':>7}")
    for k in K_VALUES:
        r = run_fc(k)
        label = "1/6" if abs(k - 1/6) < 1e-9 else f"{k:.4f}"
        print(f"{label:>8} | {r['C']:5.1f} {r['E']:5.1f} {r['R']:5.1f} | {r['phi4']:5.0f} | "
              f"{r['mm_avg']:7.3f} {r['mm_max']:7.3f} | {r['c_avg']:7.3f}")
