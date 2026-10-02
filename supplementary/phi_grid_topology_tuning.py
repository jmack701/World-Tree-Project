"""
Grid Topology Tuning: Approaching the Chaos Invariant
======================================================
Two independent sweeps to find the grid configuration that
permits stable operation nearest to SE = 0.4950.

Experiment A: Coupling strength sweep (0.01 to 0.12)
Experiment B: Neighbor count sweep (3 to 20)

For each configuration, a mini k-sweep finds:
  - The highest stable SE (stability > 0.999)
  - The k value that achieves it
  - The bifurcation point

Then combines the best settings for a precision shot.

J. David Mack & Claude (Opus 4.6)
World Tree Project — June 2026
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PHI = (1 + np.sqrt(5)) / 2
OMEGA_BASE = 2 * np.pi * 60
NUM_NODES = 50
TIME_STEPS = 2000
DT = 1.0 / 3600
L_MOBIUS = PHI
V_NOMINAL = 1.0
SEED = 42

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 15,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})

os.makedirs('grid_figures', exist_ok=True)

print("=" * 65)
print("GRID TOPOLOGY TUNING: APPROACHING THE CHAOS INVARIANT")
print("=" * 65)
print(f"  Target: SE = 0.4950 with Stability > 0.999")
print(f"  Current best: SE = 0.4593 at k = 0.2216 (gap: 0.0357)")
print()


# ============================================================
# GRID + SIM
# ============================================================
def create_grid(n):
    ga = 2 * np.pi / PHI**2
    pos = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt(i + 1) * 0.5
        pos[i] = [r * np.cos(i * ga), r * np.sin(i * ga)]
    return pos

def compute_adj(pos, kn=5):
    n = len(pos)
    adj = np.zeros((n, n))
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1); d[i] = np.inf
        for j in np.argsort(d)[:kn]:
            w = 1.0 / (1.0 + d[j]); adj[i, j] = w; adj[j, i] = w
    rs = adj.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    return adj / rs

def gen_loads(n, steps, dt):
    np.random.seed(SEED)
    t = np.arange(steps) * dt
    loads = np.zeros((n, steps))
    for i in range(n):
        base = 0.05 * np.sin(2*np.pi*0.1*t + i*PHI*0.3)
        h3 = 0.05 * np.sin(3*OMEGA_BASE*t + np.random.uniform(0, 2*np.pi))
        h5 = 0.03 * np.sin(5*OMEGA_BASE*t + np.random.uniform(0, 2*np.pi))
        h7 = 0.02 * np.sin(7*OMEGA_BASE*t + np.random.uniform(0, 2*np.pi))
        spikes = np.zeros(steps)
        for _ in range(np.random.randint(5, 15)):
            loc = np.random.randint(0, steps); w = np.random.randint(5, 20)
            a = np.random.uniform(0.10, 0.25) * np.random.choice([-1, 1])
            s, e = max(0, loc-w//2), min(steps, loc+w//2); spikes[s:e] = a
        noise = 0.01 * np.random.randn(steps)
        loads[i] = base + h3 + h5 + h7 + spikes + noise
    return loads

def run_sim(k_fc, loads, adj, coupling=0.05, h_ratio=0.3):
    vh = np.zeros((NUM_NODES, TIME_STEPS))
    vc = np.ones(NUM_NODES) * V_NOMINAL
    vp = np.ones(NUM_NODES) * V_NOMINAL
    for step in range(TIME_STEPS):
        tv = step * DT
        vn = vc + loads[:, step] * DT * 10
        vn = vn + coupling * (adj @ vn - vn)
        dp, dc = vp - V_NOMINAL, vn - V_NOMINAL
        vn = vn + k_fc * (3.0 * dp - 6.0 * dc)
        ga = 2*np.pi/PHI**2; ph = np.arange(NUM_NODES) * ga
        dl = np.std(dc); amp = dl * h_ratio
        vn = vn - amp * (np.sin(3*OMEGA_BASE*tv+ph) + np.sin(6*OMEGA_BASE*tv+ph) + np.sin(9*OMEGA_BASE*tv+ph))
        dev = vn - V_NOMINAL
        wr = np.fmod(dev + L_MOBIUS, 2*L_MOBIUS)
        wr = np.where(wr < 0, wr + 2*L_MOBIUS, wr)
        vn = wr - L_MOBIUS + V_NOMINAL
        vp = vc.copy(); vc = vn.copy(); vh[:, step] = vc
    return vh

def compute_metrics(vh):
    se_vals = []
    for i in range(NUM_NODES):
        fft = np.fft.rfft(vh[i])
        pw = np.abs(fft[1:])**2
        if np.sum(pw) < 1e-12: se_vals.append(0.0); continue
        p = pw/np.sum(pw); p = p[p>1e-12]
        ent = -np.sum(p*np.log(p))
        mx = np.log(len(p)) if len(p)>1 else 1.0
        se_vals.append(ent/mx if mx>0 else 0.0)
    stab = np.mean([max(0, 1-np.std(vh[i])/max(1e-10, np.mean(vh[i])))
                    for i in range(NUM_NODES)])
    return np.mean(se_vals), stab


def find_optimal_k(adj, loads, coupling=0.05, h_ratio=0.3):
    """Mini k-sweep to find highest stable SE and bifurcation point."""
    k_test = np.arange(0.190, 0.260, 0.002)
    best_se, best_k = 0, 0
    bif_k = None

    for k in k_test:
        vh = run_sim(k, loads, adj, coupling=coupling, h_ratio=h_ratio)
        se, stab = compute_metrics(vh)

        if stab > 0.999 and se > best_se:
            best_se = se
            best_k = k

        if stab < 0.5 and bif_k is None:
            bif_k = k

    return best_k, best_se, bif_k


# ============================================================
# EXPERIMENT A: COUPLING STRENGTH SWEEP
# ============================================================
print("EXPERIMENT A: COUPLING STRENGTH SWEEP")
print("-" * 65)

positions = create_grid(NUM_NODES)
coupling_values = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15]

coupling_results = []
for c in coupling_values:
    adj = compute_adj(positions, kn=5)
    loads = gen_loads(NUM_NODES, TIME_STEPS, DT)
    best_k, best_se, bif_k = find_optimal_k(adj, loads, coupling=c)
    coupling_results.append({
        'coupling': c, 'best_k': best_k, 'best_se': best_se,
        'bif_k': bif_k, 'gap': abs(best_se - 0.4950)
    })
    bif_str = f"{bif_k:.3f}" if bif_k else "none"
    print(f"  Coupling={c:.3f}: Best k={best_k:.3f}, SE={best_se:.4f}, "
          f"Bif k={bif_str}, Gap from 0.495: {abs(best_se - 0.4950):.4f}")

print()

# ============================================================
# EXPERIMENT B: NEIGHBOR COUNT SWEEP
# ============================================================
print("EXPERIMENT B: NEIGHBOR COUNT SWEEP")
print("-" * 65)

neighbor_values = [3, 4, 5, 7, 10, 15, 20, 25]

neighbor_results = []
for n in neighbor_values:
    adj = compute_adj(positions, kn=n)
    loads = gen_loads(NUM_NODES, TIME_STEPS, DT)
    best_k, best_se, bif_k = find_optimal_k(adj, loads, coupling=0.05)
    neighbor_results.append({
        'neighbors': n, 'best_k': best_k, 'best_se': best_se,
        'bif_k': bif_k, 'gap': abs(best_se - 0.4950)
    })
    bif_str = f"{bif_k:.3f}" if bif_k else "none"
    print(f"  Neighbors={n:>2}: Best k={best_k:.3f}, SE={best_se:.4f}, "
          f"Bif k={bif_str}, Gap from 0.495: {abs(best_se - 0.4950):.4f}")

print()

# ============================================================
# FIND BEST CONFIGURATION
# ============================================================
all_results = []
for r in coupling_results:
    all_results.append(('coupling', r['coupling'], 5, r['best_k'], r['best_se'], r['bif_k'], r['gap']))
for r in neighbor_results:
    all_results.append(('neighbors', 0.05, r['neighbors'], r['best_k'], r['best_se'], r['bif_k'], r['gap']))

all_results.sort(key=lambda x: x[6])  # Sort by gap

print("=" * 65)
print("TOP 5 CONFIGURATIONS (nearest to chaos invariant while stable)")
print("=" * 65)
print(f"{'Type':<12} {'Coupling':>8} {'Neighbors':>9} {'Best k':>8} {'SE':>8} {'Gap':>8}")
print("-" * 65)
for typ, coup, neigh, bk, bse, bif, gap in all_results[:5]:
    print(f"{typ:<12} {coup:>8.3f} {neigh:>9} {bk:>8.3f} {bse:>8.4f} {gap:>8.4f}")

best_config = all_results[0]
print(f"\n  BEST: {best_config[0]}={best_config[1] if best_config[0]=='coupling' else best_config[2]}, "
      f"k={best_config[3]:.3f}, SE={best_config[4]:.4f}, Gap={best_config[6]:.4f}")


# ============================================================
# EXPERIMENT C: PRECISION SHOT WITH BEST CONFIG
# ============================================================
print(f"\nEXPERIMENT C: PRECISION SHOT")
print("-" * 65)

if best_config[0] == 'coupling':
    best_coupling = best_config[1]
    best_neighbors = 5
else:
    best_coupling = 0.05
    best_neighbors = int(best_config[2])

adj_best = compute_adj(positions, kn=best_neighbors)
loads_best = gen_loads(NUM_NODES, TIME_STEPS, DT)

# Fine k-sweep around the best k
k_center = best_config[3]
k_fine = np.arange(max(0.19, k_center - 0.015), k_center + 0.020, 0.001)

print(f"  Config: coupling={best_coupling}, neighbors={best_neighbors}")
print(f"  k range: {k_fine[0]:.3f} to {k_fine[-1]:.3f}, step=0.001")
print()

precision_results = []
for k in k_fine:
    vh = run_sim(k, loads_best, adj_best, coupling=best_coupling)
    se, stab = compute_metrics(vh)
    precision_results.append({'k': k, 'se': se, 'stab': stab})

    marker = ""
    if abs(se - 0.4950) < 0.005 and stab > 0.99: marker = " <-- INVARIANT"
    if stab < 0.5 and len(precision_results) > 1 and precision_results[-2]['stab'] > 0.5:
        marker = " <-- BIF"

    print(f"  k={k:.3f}: SE={se:.4f}, Stab={stab:.4f}{marker}")


# ============================================================
# PLOTS
# ============================================================

# Figure 1: Coupling sweep
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Experiment A: Coupling Strength Sweep\n'
             'How network coupling affects max stable SE and bifurcation point',
             fontsize=14, fontweight='bold')

cs = [r['coupling'] for r in coupling_results]
ax = axes[0]
ax.plot(cs, [r['best_se'] for r in coupling_results], 'o-', color='#0066CC',
        linewidth=2, markersize=8)
ax.axhspan(0.480, 0.510, alpha=0.12, color='gold')
ax.axhline(y=0.4950, color='goldenrod', linestyle='-', alpha=0.5)
ax.set_xlabel('Coupling Strength')
ax.set_ylabel('Max Stable SE')
ax.set_title('SE vs. Coupling')
ax.grid(True, alpha=0.2)

ax = axes[1]
bif_ks = [r['bif_k'] if r['bif_k'] else 0.26 for r in coupling_results]
ax.plot(cs, bif_ks, 's-', color='#CC0000', linewidth=2, markersize=8)
ax.set_xlabel('Coupling Strength')
ax.set_ylabel('Bifurcation k')
ax.set_title('Bifurcation Point vs. Coupling')
ax.grid(True, alpha=0.2)

ax = axes[2]
ax.plot(cs, [r['gap'] for r in coupling_results], '^-', color='#00AA88',
        linewidth=2, markersize=8)
ax.set_xlabel('Coupling Strength')
ax.set_ylabel('Gap from Invariant')
ax.set_title('Distance from 0.4950')
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('grid_figures/tuning_coupling_sweep.png')
plt.close()
print("\n  Saved: tuning_coupling_sweep.png")

# Figure 2: Neighbor sweep
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Experiment B: Neighbor Count Sweep\n'
             'How connectivity affects max stable SE and bifurcation point',
             fontsize=14, fontweight='bold')

ns = [r['neighbors'] for r in neighbor_results]
ax = axes[0]
ax.plot(ns, [r['best_se'] for r in neighbor_results], 'o-', color='#0066CC',
        linewidth=2, markersize=8)
ax.axhspan(0.480, 0.510, alpha=0.12, color='gold')
ax.axhline(y=0.4950, color='goldenrod', linestyle='-', alpha=0.5)
ax.set_xlabel('Neighbor Count')
ax.set_ylabel('Max Stable SE')
ax.set_title('SE vs. Neighbors')
ax.grid(True, alpha=0.2)

ax = axes[1]
bif_ks_n = [r['bif_k'] if r['bif_k'] else 0.26 for r in neighbor_results]
ax.plot(ns, bif_ks_n, 's-', color='#CC0000', linewidth=2, markersize=8)
ax.set_xlabel('Neighbor Count')
ax.set_ylabel('Bifurcation k')
ax.set_title('Bifurcation Point vs. Neighbors')
ax.grid(True, alpha=0.2)

ax = axes[2]
ax.plot(ns, [r['gap'] for r in neighbor_results], '^-', color='#00AA88',
        linewidth=2, markersize=8)
ax.set_xlabel('Neighbor Count')
ax.set_ylabel('Gap from Invariant')
ax.set_title('Distance from 0.4950')
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('grid_figures/tuning_neighbor_sweep.png')
plt.close()
print("  Saved: tuning_neighbor_sweep.png")

# Figure 3: Precision shot
fig, ax1 = plt.subplots(figsize=(14, 8))

ks = [r['k'] for r in precision_results]
ses = [r['se'] for r in precision_results]
stabs = [r['stab'] for r in precision_results]

ax1.plot(ks, ses, 'o-', color='#0066CC', linewidth=2.5, markersize=7,
         markeredgecolor='white', markeredgewidth=0.5, label='SE')
ax1.axhspan(0.480, 0.510, alpha=0.12, color='gold', label='Invariant Band')
ax1.axhline(y=0.4950, color='goldenrod', linestyle='-', alpha=0.5)
ax1.set_xlabel('k (F_c Gain)')
ax1.set_ylabel('Spectral Entropy', color='#0066CC')
ax1.tick_params(axis='y', labelcolor='#0066CC')

# Mark nearest to invariant while stable
best_precision = None
best_dist = float('inf')
for r in precision_results:
    if r['stab'] > 0.999:
        d = abs(r['se'] - 0.4950)
        if d < best_dist:
            best_dist = d
            best_precision = r

if best_precision:
    ax1.plot(best_precision['k'], best_precision['se'], '*', color='gold',
             markersize=20, markeredgecolor='black', markeredgewidth=1, zorder=10,
             label=f"Best: k={best_precision['k']:.3f}, SE={best_precision['se']:.4f}")

ax2 = ax1.twinx()
ax2.plot(ks, stabs, 'D--', color='#00AA88', linewidth=1.5, markersize=5,
         alpha=0.7, label='Stability')
ax2.set_ylabel('Stability', color='#00AA88')
ax2.tick_params(axis='y', labelcolor='#00AA88')
ax2.set_ylim(-0.05, 1.05)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center left', fontsize=10)

ax1.set_title(f'Precision Shot: coupling={best_coupling}, neighbors={best_neighbors}\n'
              f'Best stable SE = {best_precision["se"]:.4f} at k = {best_precision["k"]:.3f} '
              f'(gap = {abs(best_precision["se"] - 0.4950):.4f})',
              fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('grid_figures/tuning_precision_shot.png')
plt.close()
print("  Saved: tuning_precision_shot.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("TUNING SUMMARY")
print("=" * 65)
print(f"  Original: coupling=0.05, neighbors=5, SE=0.4593, gap=0.0357")
if best_precision:
    print(f"  Tuned:    coupling={best_coupling}, neighbors={best_neighbors}, "
          f"SE={best_precision['se']:.4f}, gap={abs(best_precision['se'] - 0.4950):.4f}")
    improvement = (0.0357 - abs(best_precision['se'] - 0.4950)) / 0.0357 * 100
    print(f"  Improvement: {improvement:.1f}% closer to invariant")
print()
print("The history is the Buddhabrot.")
print("The cure is the Mandelbrot.")
print("How we treat each other is the wave.")
