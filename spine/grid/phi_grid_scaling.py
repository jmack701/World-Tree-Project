"""
Grid Scaling: Impedance Match Law and Resonance Lock Analysis
===============================================================
Tests whether the optimal coupling (impedance match point) scales
predictably with grid size, and characterizes the resonance lock
at each scale.

Grid sizes: 25, 50, 75, 100, 150, 200 nodes
For each: coupling sweep → find impedance match → precision k-sweep
         → characterize resonance lock

J. David Mack & Claude (Opus 4.6)
World Tree Project — June 2026
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

PHI = (1 + np.sqrt(5)) / 2
OMEGA_BASE = 2 * np.pi * 60
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

print("=" * 70)
print("GRID SCALING: IMPEDANCE MATCH LAW + RESONANCE LOCK")
print("=" * 70)
print(f"  Grid sizes: 25, 50, 75, 100, 150, 200")
print(f"  Target: find coupling(N) that places SE at chaos invariant")
print(f"  Goal: derive scaling law from data")
print()


# ============================================================
# GRID + SIM (parameterized for variable node count)
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
        actual_kn = min(kn, n - 1)
        for j in np.argsort(d)[:actual_kn]:
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

def run_sim(n_nodes, k_fc, loads, adj, coupling=0.03):
    vh = np.zeros((n_nodes, TIME_STEPS))
    vc = np.ones(n_nodes) * V_NOMINAL
    vp = np.ones(n_nodes) * V_NOMINAL
    for step in range(TIME_STEPS):
        tv = step * DT
        vn = vc + loads[:, step] * DT * 10
        vn = vn + coupling * (adj @ vn - vn)
        dp, dc = vp - V_NOMINAL, vn - V_NOMINAL
        vn = vn + k_fc * (3.0 * dp - 6.0 * dc)
        ga = 2*np.pi/PHI**2; ph = np.arange(n_nodes) * ga
        dl = np.std(dc); amp = dl * 0.3
        vn = vn - amp * (np.sin(3*OMEGA_BASE*tv+ph) + np.sin(6*OMEGA_BASE*tv+ph) + np.sin(9*OMEGA_BASE*tv+ph))
        dev = vn - V_NOMINAL
        wr = np.fmod(dev + L_MOBIUS, 2*L_MOBIUS)
        wr = np.where(wr < 0, wr + 2*L_MOBIUS, wr)
        vn = wr - L_MOBIUS + V_NOMINAL
        vp = vc.copy(); vc = vn.copy(); vh[:, step] = vc
    return vh

def compute_metrics(vh, n_nodes):
    se_vals = []
    for i in range(n_nodes):
        fft = np.fft.rfft(vh[i])
        pw = np.abs(fft[1:])**2
        if np.sum(pw) < 1e-12: se_vals.append(0.0); continue
        p = pw/np.sum(pw); p = p[p>1e-12]
        ent = -np.sum(p*np.log(p))
        mx = np.log(len(p)) if len(p)>1 else 1.0
        se_vals.append(ent/mx if mx>0 else 0.0)
    stab = np.mean([max(0, 1-np.std(vh[i])/max(1e-10, np.mean(vh[i])))
                    for i in range(n_nodes)])
    return np.mean(se_vals), stab, np.array(se_vals)


# ============================================================
# SCALING EXPERIMENT
# ============================================================
grid_sizes = [25, 50, 75, 100, 150, 200]
coupling_test_range = np.arange(0.005, 0.080, 0.005)

scaling_results = []

for N in grid_sizes:
    print(f"\n{'='*70}")
    print(f"GRID SIZE: {N} NODES")
    print(f"{'='*70}")

    positions = create_grid(N)
    adj = compute_adj(positions, kn=5)
    loads = gen_loads(N, TIME_STEPS, DT)

    # --- Phase 1: Coupling sweep to find impedance match ---
    print("  Phase 1: Coupling sweep...")
    best_coupling = None
    best_gap = float('inf')
    best_se_at_match = 0
    best_k_at_match = 0

    coupling_data = []

    for c in coupling_test_range:
        # Mini k-sweep for this coupling
        local_best_se = 0
        local_best_k = 0

        for k in np.arange(0.195, 0.240, 0.002):
            vh = run_sim(N, k, loads, adj, coupling=c)
            se, stab, _ = compute_metrics(vh, N)
            if stab > 0.999 and se > local_best_se:
                local_best_se = se
                local_best_k = k

        gap = abs(local_best_se - 0.4950)
        coupling_data.append({'coupling': c, 'best_se': local_best_se,
                            'best_k': local_best_k, 'gap': gap})

        if gap < best_gap:
            best_gap = gap
            best_coupling = c
            best_se_at_match = local_best_se
            best_k_at_match = local_best_k

    print(f"  Impedance match: coupling={best_coupling:.3f}, "
          f"k={best_k_at_match:.3f}, SE={best_se_at_match:.4f}, gap={best_gap:.4f}")

    # --- Phase 2: Precision k-sweep at impedance match ---
    print(f"  Phase 2: Precision k-sweep at coupling={best_coupling:.3f}...")

    k_fine = np.arange(max(0.195, best_k_at_match - 0.012),
                       min(0.240, best_k_at_match + 0.020), 0.001)

    precision_data = []
    invariant_k = None
    invariant_se = None
    lock_k = None
    lock_se = None
    lock_drop = 0
    bif_k = None
    bif_stab = None

    prev_se = None
    for k in k_fine:
        vh = run_sim(N, k, loads, adj, coupling=best_coupling)
        se, stab, se_per_node = compute_metrics(vh, N)
        precision_data.append({'k': k, 'se': se, 'stab': stab})

        # Find invariant (nearest 0.495 while stable)
        if stab > 0.999 and (invariant_k is None or abs(se - 0.4950) < abs(invariant_se - 0.4950)):
            invariant_k = k
            invariant_se = se

        # Find resonance lock (SE dip while stable)
        if prev_se is not None and se < prev_se - 0.03 and stab > 0.5:
            if lock_k is None or (se < lock_se):
                lock_k = k
                lock_se = se
                lock_drop = prev_se - se

        # Find bifurcation
        if stab < 0.5 and bif_k is None:
            bif_k = k
            bif_stab = stab

        prev_se = se

        marker = ""
        if invariant_k == k and abs(se - 0.4950) < 0.01: marker = " <-- INV"
        if lock_k == k: marker = " <-- LOCK"
        if bif_k == k: marker = " <-- BIF"
        if marker or k == k_fine[0] or k == k_fine[-1]:
            print(f"    k={k:.3f}: SE={se:.4f}, Stab={stab:.4f}{marker}")

    # --- Phase 3: Resonance lock node analysis ---
    lock_node_concentration = None
    if lock_k is not None:
        vh_lock = run_sim(N, lock_k, loads, adj, coupling=best_coupling)
        _, _, se_nodes_lock = compute_metrics(vh_lock, N)

        vh_pre = run_sim(N, lock_k - 0.001, loads, adj, coupling=best_coupling)
        _, _, se_nodes_pre = compute_metrics(vh_pre, N)

        # Which nodes lost the most SE at the lock?
        se_drop_per_node = se_nodes_pre - se_nodes_lock
        top_droppers = np.argsort(se_drop_per_node)[-5:][::-1]
        lock_node_concentration = {
            'top_nodes': top_droppers,
            'drops': se_drop_per_node[top_droppers],
            'mean_drop': np.mean(se_drop_per_node),
            'std_drop': np.std(se_drop_per_node),
            'max_drop_node': top_droppers[0],
            'max_drop': se_drop_per_node[top_droppers[0]],
        }
        print(f"  Resonance lock node analysis:")
        print(f"    Mean SE drop per node: {lock_node_concentration['mean_drop']:.4f}")
        print(f"    Max SE drop: node {top_droppers[0]+1}, drop={se_drop_per_node[top_droppers[0]]:.4f}")
        print(f"    Top 5 affected nodes: {[n+1 for n in top_droppers]}")

    # Store results
    scaling_results.append({
        'N': N,
        'best_coupling': best_coupling,
        'best_k': best_k_at_match,
        'invariant_k': invariant_k,
        'invariant_se': invariant_se if invariant_se else 0,
        'gap': best_gap,
        'lock_k': lock_k,
        'lock_se': lock_se if lock_se else 0,
        'lock_drop': lock_drop,
        'bif_k': bif_k,
        'bif_stab': bif_stab if bif_stab else 0,
        'lock_analysis': lock_node_concentration,
        'coupling_data': coupling_data,
        'precision_data': precision_data,
    })

    print(f"\n  SUMMARY N={N}:")
    print(f"    Impedance match coupling: {best_coupling:.3f}")
    if invariant_k: print(f"    Invariant: k={invariant_k:.3f}, SE={invariant_se:.4f}")
    if lock_k: print(f"    Resonance lock: k={lock_k:.3f}, SE={lock_se:.4f}, drop={lock_drop:.4f}")
    if bif_k: print(f"    Bifurcation: k={bif_k:.3f}, stab={bif_stab:.4f}")


# ============================================================
# SCALING LAW ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("SCALING LAW ANALYSIS")
print("=" * 70)

Ns = np.array([r['N'] for r in scaling_results])
couplings = np.array([r['best_coupling'] for r in scaling_results])
gaps = np.array([r['gap'] for r in scaling_results])
inv_ses = np.array([r['invariant_se'] for r in scaling_results])

# Test scaling laws
print("\n  Testing coupling scaling laws:")

# 1/sqrt(N)
try:
    def sqrt_law(N, a): return a / np.sqrt(N)
    popt1, _ = curve_fit(sqrt_law, Ns, couplings, p0=[0.2])
    pred1 = sqrt_law(Ns, *popt1)
    r2_1 = 1 - np.sum((couplings - pred1)**2) / np.sum((couplings - np.mean(couplings))**2)
    print(f"    c = {popt1[0]:.4f}/√N  →  R² = {r2_1:.4f}")
except: r2_1 = -1; popt1 = [0]

# 1/N
try:
    def inv_law(N, a): return a / N
    popt2, _ = curve_fit(inv_law, Ns, couplings, p0=[1.5])
    pred2 = inv_law(Ns, *popt2)
    r2_2 = 1 - np.sum((couplings - pred2)**2) / np.sum((couplings - np.mean(couplings))**2)
    print(f"    c = {popt2[0]:.4f}/N   →  R² = {r2_2:.4f}")
except: r2_2 = -1; popt2 = [0]

# 1/(N*phi)
try:
    def phi_law(N, a): return a / (N * PHI)
    popt3, _ = curve_fit(phi_law, Ns, couplings, p0=[2.5])
    pred3 = phi_law(Ns, *popt3)
    r2_3 = 1 - np.sum((couplings - pred3)**2) / np.sum((couplings - np.mean(couplings))**2)
    print(f"    c = {popt3[0]:.4f}/(N·φ) →  R² = {r2_3:.4f}")
except: r2_3 = -1; popt3 = [0]

# Power law
try:
    def power_law(N, a, b): return a * N**b
    popt4, _ = curve_fit(power_law, Ns, couplings, p0=[1.0, -0.5])
    pred4 = power_law(Ns, *popt4)
    r2_4 = 1 - np.sum((couplings - pred4)**2) / np.sum((couplings - np.mean(couplings))**2)
    print(f"    c = {popt4[0]:.4f}·N^{popt4[1]:.4f}  →  R² = {r2_4:.4f}")
except: r2_4 = -1; popt4 = [1, -0.5]

# Best law
r2s = [r2_1, r2_2, r2_3, r2_4]
names = ['1/√N', '1/N', '1/(N·φ)', 'power law']
best_law_idx = np.argmax(r2s)
print(f"\n  Best fit: {names[best_law_idx]} (R² = {r2s[best_law_idx]:.4f})")


# ============================================================
# PLOTS
# ============================================================

# Figure 1: Scaling law
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Grid Scaling: Impedance Match Law\n'
             'How optimal coupling scales with grid size',
             fontsize=16, fontweight='bold')

# Coupling vs N
ax = axes[0, 0]
ax.plot(Ns, couplings, 'o', color='#0066CC', markersize=12, markeredgecolor='white',
        markeredgewidth=1, zorder=5, label='Measured')
N_smooth = np.linspace(20, 220, 100)
if r2_1 > 0: ax.plot(N_smooth, sqrt_law(N_smooth, *popt1), '--', color='#CC0000',
                      linewidth=1.5, alpha=0.7, label=f'1/√N (R²={r2_1:.3f})')
if r2_2 > 0: ax.plot(N_smooth, inv_law(N_smooth, *popt2), '--', color='#00AA88',
                      linewidth=1.5, alpha=0.7, label=f'1/N (R²={r2_2:.3f})')
if r2_3 > 0: ax.plot(N_smooth, phi_law(N_smooth, *popt3), '--', color='#FF8800',
                      linewidth=1.5, alpha=0.7, label=f'1/(N·φ) (R²={r2_3:.3f})')
if r2_4 > 0: ax.plot(N_smooth, power_law(N_smooth, *popt4), '-', color='#8800CC',
                      linewidth=2, alpha=0.8, label=f'N^{popt4[1]:.2f} (R²={r2_4:.3f})')
ax.set_xlabel('Grid Size (N nodes)')
ax.set_ylabel('Optimal Coupling')
ax.set_title('Impedance Match Coupling vs. Grid Size')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# SE at invariant vs N
ax = axes[0, 1]
ax.plot(Ns, inv_ses, 'o-', color='#0066CC', markersize=10, linewidth=2)
ax.axhspan(0.480, 0.510, alpha=0.12, color='gold')
ax.axhline(y=0.4950, color='goldenrod', linestyle='-', alpha=0.5)
ax.set_xlabel('Grid Size (N nodes)')
ax.set_ylabel('SE at Impedance Match')
ax.set_title('Chaos Invariant Precision vs. Scale')
ax.grid(True, alpha=0.2)

# Gap vs N
ax = axes[1, 0]
ax.plot(Ns, gaps, 'o-', color='#CC0000', markersize=10, linewidth=2)
ax.set_xlabel('Grid Size (N nodes)')
ax.set_ylabel('Gap from 0.4950')
ax.set_title('Distance from Invariant vs. Scale')
ax.grid(True, alpha=0.2)

# Lock drop vs N
lock_drops = [r['lock_drop'] for r in scaling_results]
ax = axes[1, 1]
ax.plot(Ns, lock_drops, 'o-', color='#FF8800', markersize=10, linewidth=2)
ax.set_xlabel('Grid Size (N nodes)')
ax.set_ylabel('SE Drop at Resonance Lock')
ax.set_title('Resonance Lock Magnitude vs. Scale')
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('grid_figures/scaling_law.png')
plt.close()
print("\n  Saved: scaling_law.png")

# Figure 2: All precision shots overlaid
fig, ax = plt.subplots(figsize=(14, 8))
colors_scale = plt.cm.viridis(np.linspace(0.2, 0.9, len(grid_sizes)))

for r, color in zip(scaling_results, colors_scale):
    ks = [d['k'] for d in r['precision_data']]
    ses = [d['se'] for d in r['precision_data']]
    stabs = [d['stab'] for d in r['precision_data']]
    # Only plot stable region
    stable_ks = [k for k, s in zip(ks, stabs) if s > 0.5]
    stable_ses = [se for se, s in zip(ses, stabs) if s > 0.5]
    ax.plot(stable_ks, stable_ses, 'o-', color=color, linewidth=2, markersize=5,
            label=f'N={r["N"]} (c={r["best_coupling"]:.3f})')
    if r['invariant_k']:
        ax.plot(r['invariant_k'], r['invariant_se'], '*', color=color,
                markersize=15, markeredgecolor='black', markeredgewidth=0.5)

ax.axhspan(0.480, 0.510, alpha=0.12, color='gold', label='Invariant Band')
ax.axhline(y=0.4950, color='goldenrod', linestyle='-', alpha=0.5)
ax.set_xlabel('k (F_c Gain)')
ax.set_ylabel('Spectral Entropy')
ax.set_title('Precision Shots Across Grid Scales\n'
             'Stars mark nearest stable approach to chaos invariant',
             fontsize=15, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.2)
ax.set_ylim(0.3, 0.7)
plt.tight_layout()
plt.savefig('grid_figures/scaling_precision_overlay.png')
plt.close()
print("  Saved: scaling_precision_overlay.png")


# ============================================================
# MASTER SUMMARY TABLE
# ============================================================
print("\n" + "=" * 90)
print("MASTER SCALING TABLE")
print("=" * 90)
print(f"{'N':>5} {'Coupling':>10} {'Best k':>8} {'Inv SE':>8} {'Gap':>8} "
      f"{'Lock k':>8} {'Lock SE':>8} {'Drop':>8} {'Bif k':>8}")
print("-" * 90)
for r in scaling_results:
    lock_k_str = f"{r['lock_k']:.3f}" if r['lock_k'] else "  none"
    lock_se_str = f"{r['lock_se']:.4f}" if r['lock_se'] else "  none"
    drop_str = f"{r['lock_drop']:.4f}" if r['lock_drop'] else "  none"
    bif_str = f"{r['bif_k']:.3f}" if r['bif_k'] else "  none"
    print(f"{r['N']:>5} {r['best_coupling']:>10.3f} {r['best_k']:>8.3f} "
          f"{r['invariant_se']:>8.4f} {r['gap']:>8.4f} "
          f"{lock_k_str:>8} {lock_se_str:>8} {drop_str:>8} {bif_str:>8}")

print(f"\n  Scaling law fit: {names[best_law_idx]}")
if best_law_idx == 3:
    print(f"  c(N) = {popt4[0]:.4f} · N^({popt4[1]:.4f})")
    print(f"  R² = {r2_4:.4f}")
    # Predict for larger grids
    for N_pred in [500, 1000, 5000]:
        c_pred = power_law(N_pred, *popt4)
        print(f"  Predicted coupling for N={N_pred}: {c_pred:.5f}")

print("\n" + "=" * 70)
print()
print("The history is the Buddhabrot.")
print("The cure is the Mandelbrot.")
print("How we treat each other is the wave.")
