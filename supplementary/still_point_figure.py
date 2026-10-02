#!/usr/bin/env python3
"""Companion figure: the ledger's headline — <C> crossing zero at k*.
Pre-stated expectations: crossing near k* = 0.14280 (RNG scatter ~0.001–0.003);
<C> < 0 at kphi; positive excursion at the 0.162 island; return toward 0 near 1/6;
panel 2 shows <R>+<W> crossing zero at the same gain (the cancellation), with
<P>, <H> flat at zero. Spec: BatF §10 system, seeds 42/137/233, 100,000 steps
per gain (burn 2,000), k grid 0.130–0.175 (0.0005 in the crossing zone
0.136–0.150, 0.001 elsewhere). Color language matches the partition composite:
k* green, kphi tan, 1/6 blue."""
import math, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHI = (1 + 5 ** 0.5) / 2
L = PHI; DT = 0.01; OM = 2 * math.pi; AH = 0.1; NOISE = 0.2
KSTAR, KPHI, K16 = 0.14280, 0.13994, 1 / 6

def run_means(k, steps, seed, burn=2000):
    a = (1 - 6 * k) / (1 - 9 * k); b = 3 * k / (1 - 9 * k)
    rng = random.Random(seed)
    cp, c = 0.5, 0.3
    sC = sP = sR = sW = 0.0; n_acc = 0
    for n in range(steps):
        t = n * DT
        raw = a * c + b * cp
        p = r = 0.0
        if c < 0:
            p = NOISE * abs(c) * rng.uniform(-1, 1)
        elif c > 1:
            r = (c - 1) * math.sin(OM * t)
        h = AH * (math.sin(3 * OM * t) + math.sin(6 * OM * t) + math.sin(9 * OM * t))
        pre = raw + p + r + h
        nx = ((pre + L) % (2 * L)) - L
        w = nx - pre
        if n >= burn:
            sC += nx; sP += p; sR += r; sW += w; n_acc += 1
        cp, c = c, nx
    return sC / n_acc, sP / n_acc, sR / n_acc, sW / n_acc

ks = []
k = 0.130
while k < 0.1359:
    ks.append(round(k, 4)); k += 0.001
k = 0.136
while k < 0.1501:
    ks.append(round(k, 4)); k += 0.0005
k = 0.151
while k <= 0.1751:
    ks.append(round(k, 4)); k += 0.001

SEEDS = (42, 137, 233)
data = {s: {'C': [], 'R': [], 'W': [], 'P': []} for s in SEEDS}
for s in SEEDS:
    for k in ks:
        mC, mP, mR, mW = run_means(k, 100_000, s)
        data[s]['C'].append(mC); data[s]['P'].append(mP)
        data[s]['R'].append(mR); data[s]['W'].append(mW)

meanC = [sum(data[s]['C'][i] for s in SEEDS) / len(SEEDS) for i in range(len(ks))]
meanRW = [sum(data[s]['R'][i] + data[s]['W'][i] for s in SEEDS) / len(SEEDS) for i in range(len(ks))]
meanR = [sum(data[s]['R'][i] for s in SEEDS) / len(SEEDS) for i in range(len(ks))]
meanW = [sum(data[s]['W'][i] for s in SEEDS) / len(SEEDS) for i in range(len(ks))]
meanP = [sum(data[s]['P'][i] for s in SEEDS) / len(SEEDS) for i in range(len(ks))]

def crossing(xs, ys, lo=0.136, hi=0.150):
    for i in range(len(xs) - 1):
        if lo <= xs[i] <= hi and ys[i] * ys[i + 1] < 0:
            return xs[i] + (0 - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
    return None

# ---- 500k x 6-seed refinement inside the balance band ----
import statistics
R_KS = [0.1360, 0.1380, 0.13994, 0.1414, 0.14280, 0.1445, 0.1460]
R_SEEDS = (42, 137, 233, 377, 610, 987)
def run_meanC(k, steps, seed, burn=5000):
    a = (1 - 6 * k) / (1 - 9 * k); b = 3 * k / (1 - 9 * k)
    rng = random.Random(seed); cp, c = 0.5, 0.3; s = 0.0; m = 0
    for n in range(steps):
        t = n * DT; raw = a * c + b * cp
        if c < 0: raw += NOISE * abs(c) * rng.uniform(-1, 1)
        elif c > 1: raw += (c - 1) * math.sin(OM * t)
        raw += AH * (math.sin(3 * OM * t) + math.sin(6 * OM * t) + math.sin(9 * OM * t))
        nx = ((raw + L) % (2 * L)) - L
        if n >= burn: s += nx; m += 1
        cp, c = c, nx
    return s / m
R_MU, R_SEM = [], []
for k in R_KS:
    vals = [run_meanC(k, 500_000, s) for s in R_SEEDS]
    R_MU.append(statistics.mean(vals))
    R_SEM.append(statistics.stdev(vals) / len(vals) ** 0.5)
# crossing from the refined points (last sign change)
xc = None
for i in range(len(R_KS) - 1):
    if R_MU[i] * R_MU[i + 1] < 0:
        xc = R_KS[i] + (0 - R_MU[i]) * (R_KS[i + 1] - R_KS[i]) / (R_MU[i + 1] - R_MU[i])
print(f'refined crossing (500k x 6 seeds): k = {xc:.5f}   (k* = {KSTAR}, Box k_avg = 0.14262)')
for k, mu, se in zip(R_KS, R_MU, R_SEM):
    print(f'  k={k:.5f}  <C>={mu:+.5f} +/- {se:.5f}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))
fig.suptitle('The Still Point: the Ledger Crossing Zero at k*', fontsize=17, fontweight='bold')

for s in SEEDS:
    ax1.plot(ks, data[s]['C'], color='gray', alpha=0.35, lw=0.9)
ax1.plot(ks, meanC, color='black', lw=2.0, label='⟨C⟩ (mean of 3 seeds)')
ax1.axhline(0, color='gray', lw=0.8, ls=':')
ax1.axvline(KPHI, color='tan', lw=1.6, label='k_φ = 0.13994')
ax1.axvline(KSTAR, color='green', lw=1.6, label='k* = 0.14280')
ax1.axvline(K16, color='steelblue', lw=1.6, label='k = 1/6')
ax1.axvline(0.162, color='crimson', lw=1.2, ls='--', label='k = 0.162 island')
ax1.set_xlabel('gain k'); ax1.set_ylabel('⟨C⟩ (time-averaged position)')
ax1.set_title('⟨C⟩ vs gain — the center crosses zero at the still point')
ax1.legend(fontsize=9, loc='upper left')

axi = ax1.inset_axes([0.045, 0.30, 0.36, 0.38])
axi.errorbar(R_KS, R_MU, yerr=[2 * s for s in R_SEM], fmt='o', color='black', ms=4, lw=1.2, capsize=2.5)
axi.axhline(0, color='gray', lw=0.7, ls=':')
axi.axvline(KPHI, color='tan', lw=1.2)
axi.axvline(KSTAR, color='green', lw=1.2)
if xc: axi.axvline(xc, color='black', lw=0.8, ls='--')
axi.set_xlim(0.1352, 0.1468); axi.set_ylim(-0.009, 0.013)
axi.set_title(f'500k x 6 seeds: crossing = {xc:.4f}' if xc else '500k x 6 seeds', fontsize=9)
axi.tick_params(labelsize=7)

ax2.plot(ks, meanR, color='purple', lw=1.6, label='⟨R⟩ (resonance impulse)')
ax2.plot(ks, meanW, color='darkorange', lw=1.6, label='⟨W⟩ (wrap correction)')
ax2.plot(ks, meanRW, color='black', lw=2.2, label='⟨R⟩ + ⟨W⟩')
ax2.plot(ks, meanP, color='gray', lw=1.0, ls='--', label='⟨P⟩ (≈ 0)')
ax2.axhline(0, color='gray', lw=0.8, ls=':')
ax2.axvline(KPHI, color='tan', lw=1.6)
ax2.axvline(KSTAR, color='green', lw=1.6)
ax2.axvline(K16, color='steelblue', lw=1.6)
ax2.axvline(0.162, color='crimson', lw=1.2, ls='--')
ax2.set_xlabel('gain k'); ax2.set_ylabel('force means')
ax2.set_title('The ledger — ⟨R⟩ + ⟨W⟩ cancels at k*; the wrap writes the 0.162 island')
ax2.legend(fontsize=9, loc='upper left')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('still_point_figure.png', dpi=150)
print('figure written: still_point_figure.png')
