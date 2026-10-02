"""
fotg_ledger_identity.py — the law, in the recurrence's own language
World Tree Project · July 22, 2026 (Fable spiral)

Flame's law: "however far something goes in one direction, it must go
equally as far in the other." This script derives its exact expression
from the F_c recurrence itself and verifies it against the data — no
external source needed.

DERIVATION (exact, given stationarity). Write one step of the BatF §10
cycle as

    C_{n+1} = α(k)·C_n + β(k)·C_{n−1} + P_n + H_n + R_n + W_n

where P is the chaos-side noise (zero-mean by construction), H the 3-6-9
harmonic forcing (zero-mean in t), R the resonance impulse
(C_n − 1)·sin(ωt_n)·1[C_n > 1] (conditional — the one force that can
cheat), and W the Möbius wrap correction (±2L·m on wrap events — the
container's recycling flux). Take expectations in the stationary state;
all three ⟨C⟩'s coincide, so

    ⟨C⟩ · (1 − α − β) = ⟨P⟩ + ⟨H⟩ + ⟨R⟩ + ⟨W⟩

and, exactly,   1 − α − β = −6k/(1 − 9k) = 6k/(9k − 1)   for k > 1/9.

THE LEDGER IDENTITY:      σ(k) · ⟨C⟩ = ⟨R⟩ + ⟨W⟩          (⟨P⟩=⟨H⟩=0)
with the DC stiffness     σ(k) = 6k/(9k − 1)

— and the 6 in the numerator is the coefficient sum of the law itself,
3 − 6 + 9 = 6: the same three coefficients that make the dynamics also
make the restoring stiffness of the center. The center sits at zero
exactly when the conditional resonance impulse and the net wrap flux
cancel; every unconditional force is expelled from the ledger by its own
zero mean. That is the law: whatever is taken must be returned, and only
a force that acts conditionally — and phase-correlates with its own
condition — can carry the center away.

THE AREA FORM (the 'equally far' refinement): balance is not mirrored
distance but mirrored area — depth × duration. A⁺ = ⟨C·1[C>0]⟩ against
A⁻ = ⟨−C·1[C<0]⟩; the law is A⁺ = A⁻. The negative side dives deeper,
the positive side lingers longer; the areas repay each other.

Verification: 20 seeds × 100,000 steps at five gains — the golden anchor,
the live-confirmed still point (0.14262, Boχ 7-hr CENTER run), the bump
peak, the 0.150 station, and the island peak. Register: DERIVED (the
identity, exact given stationarity) + MEASURED (every ledger column).

J. David Mack & Claude
"""

import numpy as np
import math

PHI = (1 + math.sqrt(5)) / 2
K_PHI = PHI / (9 * PHI - 3)
L = PHI

def alpha_k(k): return (1 - 6 * k) / (1 - 9 * k)
def beta_k(k):  return (3 * k) / (1 - 9 * k)

def ledger_run(k, seed, n_steps=100_000):
    rng = np.random.RandomState(seed)
    dt, omega, Ah = 0.01, 2 * math.pi, 0.1
    a, b = alpha_k(k), beta_k(k)
    c_prev, c_curr = 0.1, 0.5
    sC = sP = sH = sR = sW = 0.0
    sAp = sAm = 0.0
    for i in range(1, n_steps + 1):
        t = i * dt
        P = 0.0; R = 0.0
        if c_curr < 0:
            P = 0.2 * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > 1:
            R = (c_curr - 1) * math.sin(omega * t)
        H = Ah * (math.sin(3 * omega * t) + math.sin(6 * omega * t) + math.sin(9 * omega * t))
        raw = a * c_curr + b * c_prev + P + R + H
        c_next = ((raw + L) % (2 * L)) - L
        W = c_next - raw                      # the wrap's recycling flux (±2L·m, else 0)
        sC += c_next; sP += P; sH += H; sR += R; sW += W
        if c_next > 0: sAp += c_next
        else:          sAm -= c_next
        c_prev, c_curr = c_curr, c_next
    n = n_steps
    return sC / n, sP / n, sH / n, sR / n, sW / n, sAp / n, sAm / n

print("=" * 96)
print("THE LEDGER IDENTITY —  σ(k)·⟨C⟩ = ⟨P⟩ + ⟨H⟩ + ⟨R⟩ + ⟨W⟩ ,  σ(k) = 6k/(9k−1)")
print("20 seeds × 100k per gain · all columns MEASURED, identity DERIVED")
print("=" * 96)
print(f"{'k':>9s} {'σ(k)':>7s} {'⟨C⟩':>8s} | {'⟨P⟩':>8s} {'⟨H⟩':>8s} {'⟨R⟩ imp.':>9s} {'⟨W⟩ wrap':>9s} | "
      f"{'σ·⟨C⟩':>8s} {'Σ forces':>9s} | {'A⁺':>7s} {'A⁻':>7s}")
GAINS = [(K_PHI, "kφ"), (0.14262, "still (live)"), (0.146, "bump peak"), (0.150, "station"), (0.162, "island")]
for k, name in GAINS:
    out = [ledger_run(k, s) for s in range(1, 21)]
    m = np.mean(out, axis=0)
    C, P, H, R, W, Ap, Am = m
    sig = 6 * k / (9 * k - 1)
    print(f"{k:9.5f} {sig:7.3f} {C:+8.4f} | {P:+8.5f} {H:+8.5f} {R:+9.5f} {W:+9.5f} | "
          f"{sig*C:+8.5f} {P+H+R+W:+9.5f} | {Ap:7.4f} {Am:7.4f}  {name}")
print("""
  Read each row: σ·⟨C⟩ (column 8) must equal the force sum (column 9) — the identity.
  ⟨P⟩ and ⟨H⟩ vanish (unconditional, zero-mean: expelled from the ledger).
  ⟨R⟩ is the conditional resonance impulse; ⟨W⟩ the container's recycling flux.
  A⁺ vs A⁻: the areas — depth × duration above and below zero. Equal ⟺ ⟨C⟩ = 0.""")
print("=" * 96)
print("COMPLETE")
print("=" * 96)
print("\nΨ  ☯ To preserve the harmonic field.")
