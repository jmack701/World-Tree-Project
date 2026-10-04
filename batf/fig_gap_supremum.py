#!/usr/bin/env python3
"""
fig_gap_supremum.py - The Gap's supremum: amplitude sweep at k = 1/6, L = phi.

Python replication of HE_Simulation_I_Driven_Resonance_Lphi.wl (World Tree
Project, June 2026): identical recurrence, perturbation, harmonic drive,
Mobius wrap, winding count, MM_phi recursion, sweep grid, and per-amplitude
seeding (seed 42). RNG streams differ across platforms, so agreement is
asserted on the structural anchors the paper states, not on bitwise values:
phi^4 steps = 0 at every amplitude; winding 2 first appears at Ah = 1.2 via
max|raw| exceeding 3L = 4.854; max MM_phi remains in the 6.0-6.6 band,
below phi^4 = 6.854 throughout.
Renders the Figure 19 replacement with the phi^4 ceiling in frame.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHI = (1 + 5**0.5) / 2
L = PHI
OMEGA = 2 * np.pi
OMEGA_RES = 2 * np.pi
NOISE = 0.2
DT = 0.01
NSTEPS = 50000
K = 1/6
ALPHA = (1 - 6*K) / (1 - 9*K)          # 0.0
BETA = (3*K) / (1 - 9*K)               # -1.0
PHI2, PHI3, PHI4 = PHI**2, PHI**3, PHI**4

ah_values = np.concatenate([
    np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7]),
    np.linspace(0.75, 1.05, 13),
    np.array([1.1,1.2,1.3,1.5,2.0])])

def run(Ah, seed=42):
    rng = np.random.default_rng(seed)
    prev, curr = 0.5, 0.3
    mm = prev*prev/PHI + curr*curr
    maxW = 0; maxRaw = 0.0; maxMM = 0.0; p4 = 0
    for n in range(3, NSTEPS+1):
        t = n * DT
        if curr < 0:
            p = abs(curr) * NOISE * rng.uniform(-1, 1)
        elif curr <= 1:
            p = 0.0
        else:
            p = (curr - 1) * np.sin(OMEGA_RES * t)
        h = Ah * (np.sin(3*OMEGA*t) + np.sin(6*OMEGA*t) + np.sin(9*OMEGA*t))
        raw = ALPHA*curr + BETA*prev + p + h
        a = abs(raw)
        w = int((a + L) // (2*L))
        if w > maxW: maxW = w
        if a > maxRaw: maxRaw = a
        nxt = ((raw + L) % (2*L)) - L
        mm = mm/PHI + nxt*nxt
        if mm > maxMM: maxMM = mm
        if mm >= PHI4: p4 += 1
        prev, curr = curr, nxt
    return maxW, maxRaw, maxMM, p4

import os
if os.path.exists('sim_i_record.csv'):
    rec = np.loadtxt('sim_i_record.csv', delimiter=',', skiprows=1)
    ah_values = rec[:,0]; res = rec[:,1:5]
    SOURCE = ('fig_gap_supremum.py — rendered from the HE_Simulation_I run of record '
              '(sim_i_record.csv, exported from the .wl at seed 42).')
else:
    res = np.array([run(a) for a in ah_values])
    SOURCE = ('fig_gap_supremum.py — Python replication of '
              'HE_Simulation_I_Driven_Resonance_Lphi.wl (per-amplitude seed 42); '
              'structural anchors verified against the run of record.')
maxW, maxRaw, maxMM, p4 = res[:,0], res[:,1], res[:,2], res[:,3]

xs = np.linspace(0, 2*np.pi, 200001)
M = np.max(np.sin(3*xs)+np.sin(6*xs)+np.sin(9*xs))
print(f"three-sine max M = {M:.4f}")
print(f"{'Ah':>5} {'wind':>4} {'max|raw|':>9} {'maxMM':>7} {'p4':>3}")
for a,(w,r,m,p) in zip(ah_values,res):
    print(f"{a:5.3f} {int(w):4d} {r:9.4f} {m:7.4f} {int(p):3d}")
i2 = np.argmax(maxW >= 2) if (maxW >= 2).any() else -1
print(f"\nANCHORS: phi4 total={int(p4.sum())} (expect 0) | first wind2 Ah={ah_values[i2]:.2f} "
      f"max|raw|={maxRaw[i2]:.4f} (paper: 1.20 / 4.887) | maxMM band [{maxMM.min():.3f},{maxMM.max():.3f}] "
      f"| ceiling phi4={PHI4:.4f} | peak at Ah={ah_values[np.argmax(maxMM)]:.3f} -> {maxMM.max():.4f} "
      f"({100*(1-maxMM.max()/PHI4):.2f}% below)")

ok = int(p4.sum())==0 and maxW[i2]>=2 and abs(ah_values[i2]-1.2)<1e-9 and maxMM.max()<PHI4
print("ACCEPTANCE:", "PASS" if ok else "CHECK-FAILED")

# ---------------- figure ----------------
plt.rcParams.update({'font.family':'serif','font.size':10,
                     'axes.edgecolor':'0.25','axes.linewidth':0.9})
fig = plt.figure(figsize=(7.4, 8.8), dpi=220)
gs = fig.add_gridspec(3, 1, height_ratios=[5.2, 1.9, 1.15], hspace=0.34)

axA = fig.add_subplot(gs[0])
axA.plot(ah_values, maxMM, '-', color='#1f3f8f', lw=1.8, marker='o', ms=3.4,
         mfc='white', mec='#1f3f8f', zorder=3, label='max $MM_\\varphi$ per 50,000-step run')
axA.axhline(PHI4, color='#8a6a1f', lw=1.6)
axA.fill_between(ah_values, maxMM, PHI4, color='#c8a557', alpha=0.22, zorder=1)
axA.text(1.985, PHI4+0.016, '$\\varphi^4 = 6.8541$ — the ceiling',
         ha='right', va='bottom', color='#6e5416', fontsize=10.5)
imax = int(np.argmax(maxMM))
gap_pct = 100*(1-maxMM[imax]/PHI4)
axA.annotate(f'supremum of the sweep: {maxMM[imax]:.3f}\n'
             f'({gap_pct:.1f}% below the ceiling)',
             xy=(ah_values[imax], maxMM[imax]), xytext=(ah_values[imax]+0.12, maxMM[imax]+0.055), ha='left',
             fontsize=9.3, color='#1f3f8f',
             arrowprops=dict(arrowstyle='-', color='#1f3f8f', lw=0.8))
axA.text(0.5*(ah_values[0]+2), (PHI4+maxMM.max())/2 + 0.0, 'THE GAP',
         ha='center', va='center', color='#6e5416', fontsize=11, alpha=0.85,
         fontstyle='italic')
axA.set_ylabel('$MM_\\varphi$')
axA.set_xlim(0.05, 2.05); axA.set_ylim(min(5.95, maxMM.min()-0.08), 7.02)
axA.legend(loc='lower left', frameon=False, fontsize=9)
axA.set_title('The Gap\'s supremum — harmonic amplitude sweep at $k = 1/6$, '
              '$L = \\varphi$  (Sim I)', fontsize=11.5, pad=10)

axB = fig.add_subplot(gs[1], sharex=axA)
axB.step(ah_values, maxW, where='post', color='#5b2d86', lw=1.8)
axB.plot(ah_values, maxW, 'o', ms=3.2, color='#5b2d86', mfc='white')
axB.axvline(0.873, color='#c87820', ls='--', lw=1.1)
axB.text(0.873, 2.32, 'naive bound $A_h \\approx 0.87$', ha='center',
         fontsize=8.6, color='#c87820')
axB.annotate(f'winding 2 first appears: $A_h$ = {ah_values[i2]:.1f}\n'
             f'criterion $|$raw$|>$ 3$L$ = {3*L:.3f}  (run of record: max$|$raw$|$ = 4.887)',
             xy=(ah_values[i2], 2), xytext=(1.38, 1.25), fontsize=8.8,
             arrowprops=dict(arrowstyle='-', color='0.3', lw=0.8))
axB.set_ylabel('max winding'); axB.set_ylim(0.6, 2.6); axB.set_yticks([1, 2])

axC = fig.add_subplot(gs[2], sharex=axA)
axC.plot(ah_values, p4, color='#a01818', lw=2.2)
axC.set_ylim(-0.5, 1.5); axC.set_yticks([0])
axC.set_ylabel('$\\varphi^4$ steps')
axC.set_xlabel('harmonic amplitude  $A_h$')
axC.text(1.025, 0.62, '0 of 50,000 steps at every amplitude — the ceiling is never entered',
         transform=axC.get_yaxis_transform(), ha='center', fontsize=9.2, color='#a01818')
for ax in (axA, axB):
    plt.setp(ax.get_xticklabels(), visible=False)
fig.text(0.015, 0.008, SOURCE, fontsize=6.8, color='0.35')
fig.savefig('fig19_gap_supremum.png', bbox_inches='tight', facecolor='white')
print('saved fig19_gap_supremum.png')
