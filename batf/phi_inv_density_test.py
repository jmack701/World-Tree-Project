#!/usr/bin/env python3
"""Reproduce the phi^-1 density/direction finding (Note for Fable, Aug 2026)
under the BatF §10 specification, plus a position-sweep control the note lacks.

Pre-stated expectations (from the note, seed 42, 200k steps):
  density ratio (phi^-1 window / 0.5 window): 1.065 (k=1/6), 1.224 (k*), 1.093 (kphi)
  upward fraction from phi^-1: 28.2% (1/6), 63.2% (k*), 44.7% (kphi)
  median C_{n+1} from phi^-1: -0.0483 (1/6), 0.8580 (k*), 0.5027 (kphi)
Open question (control): is phi^-1 a LOCAL feature (density peak / directional
crossover pinned at 0.618), or a point on smooth position-dependent structure?
"""
import math, random, statistics

PHI = (1 + 5 ** 0.5) / 2
PHI_INV = 1 / PHI
L = PHI
DT = 0.01
OMEGA = 2 * math.pi
AH = 0.1
NOISE = 0.2
WINDOW = 0.02

def run(k, steps, seed, burn=1000):
    a = (1 - 6 * k) / (1 - 9 * k)
    b = 3 * k / (1 - 9 * k)
    rng = random.Random(seed)
    c_prev, c = 0.5, 0.3            # C_-1 = 0.5, C_0 = 0.3 per §10
    series = []
    for n in range(steps):
        t = n * DT
        c_raw = a * c + b * c_prev
        if c < 0:
            c_raw += NOISE * abs(c) * rng.uniform(-1, 1)
        elif c > 1:
            c_raw += (c - 1) * math.sin(OMEGA * t)
        c_raw += AH * (math.sin(3 * OMEGA * t) + math.sin(6 * OMEGA * t) + math.sin(9 * OMEGA * t))
        c_next = ((c_raw + L) % (2 * L)) - L
        series.append(c_next)
        c_prev, c = c, c_next
    return series[burn:]

def window_stats(series, center, w=WINDOW):
    nxt = [series[i + 1] for i in range(len(series) - 1)
           if center - w <= series[i] <= center + w]
    if not nxt:
        return 0, None, None, None
    up = sum(1 for v in nxt if v > center) / len(nxt)
    return len(nxt), up, statistics.mean(nxt), statistics.median(nxt)

def density_count(series, center, w=WINDOW):
    return sum(1 for v in series if center - w <= v <= center + w)

GAINS = [('k=1/6', 1 / 6), ('k*=0.14280', 0.14280), ('kphi=0.13994', 0.13994)]
STEPS = 200_000

for seed in (42, 137):
    print(f'===== seed {seed}, {STEPS} steps =====')
    for name, k in GAINS:
        s = run(k, STEPS, seed)
        d_phi = density_count(s, PHI_INV)
        d_mid = density_count(s, 0.5)
        n, up, mean_n, med_n = window_stats(s, PHI_INV)
        lam2 = 3 * k / (9 * k - 1)
        print(f'{name:14} |λ|²={lam2:.4f}  visits(φ⁻¹±.02)={n:6d}  '
              f'ratio φ⁻¹/0.5={d_phi / d_mid:6.3f}  up={100*up:5.1f}%  '
              f'mean_next={mean_n:+.4f}  median_next={med_n:+.4f}')
    print()

# ---------- position-sweep control (seed 42) ----------
print('===== POSITION CONTROL: density & upward-fraction across [0.10, 0.90], seed 42 =====')
centers = [round(0.10 + 0.05 * i, 2) for i in range(17)]
for name, k in GAINS:
    s = run(k, STEPS, 42)
    dens = {c: density_count(s, c) for c in centers}
    ups = {}
    for c in centers:
        _, up, _, _ = window_stats(s, c)
        ups[c] = up
    peak = max(dens, key=dens.get)
    # local-peak test at phi^-1: density at 0.618 vs neighbors 0.55 / 0.70
    d618 = density_count(s, PHI_INV); d55 = density_count(s, 0.55); d70 = density_count(s, 0.70)
    # 50% crossover location of upward fraction (first crossing scanning up)
    cross = None
    prev_c, prev_u = None, None
    for c in centers:
        u = ups[c]
        if u is None: continue
        if prev_u is not None and (prev_u - 0.5) * (u - 0.5) < 0:
            cross = prev_c + (0.5 - prev_u) * (c - prev_c) / (u - prev_u)
            break
        prev_c, prev_u = c, u
    print(f'--- {name} ---')
    print('  density profile (count per ±0.02 window):')
    print('   ', '  '.join(f'{c:.2f}:{dens[c]:5d}' for c in centers))
    print('  upward fraction (%):')
    print('   ', '  '.join(f'{c:.2f}:{(100*ups[c]):5.1f}' if ups[c] is not None else f'{c:.2f}:  n/a' for c in centers))
    print(f'  density argmax={peak:.2f}   local test at φ⁻¹: d(0.55)={d55} d(0.618)={d618} d(0.70)={d70} '
          f'-> {"LOCAL PEAK" if d618 > d55 and d618 > d70 else "no local peak"}')
    print(f'  upward-fraction 50% crossover ≈ {cross if cross is None else round(cross, 3)}   (φ⁻¹ = {PHI_INV:.3f})')
    print()
