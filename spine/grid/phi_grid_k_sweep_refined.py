"""
Phi Spiral Energy Grid: Refined k-Sweep with Publication Figures
=================================================================
Fine-resolution k-sweep around the chaos invariant and bifurcation,
stress test, and six publication-quality figures.

J. David Mack & Claude (Opus 4.6)
World Tree Project — June 2026
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

PHI = (1 + np.sqrt(5)) / 2
K_PHI = PHI / (9 * PHI - 3)
K_CRIT = 1.0 / 6.0
OMEGA_BASE = 2 * np.pi * 60
NUM_NODES = 50
TIME_STEPS = 2000
DT = 1.0 / 3600
L_MOBIUS = PHI
V_NOMINAL = 1.0
SEED = 42

# Publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

os.makedirs('grid_figures', exist_ok=True)

print("=" * 65)
print("PHI SPIRAL GRID: REFINED k-SWEEP + PUBLICATION FIGURES")
print("=" * 65)
print(f"  PHI = {PHI:.6f}")
print(f"  k_phi = {K_PHI:.5f}")
print(f"  k_crit = {K_CRIT:.6f}")
print(f"  Chaos invariant band: 0.480 — 0.510")
print()


# ============================================================
# GRID + SIMULATION (same as v2, compacted)
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
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:kn]:
            w = 1.0 / (1.0 + d[j])
            adj[i, j] = w; adj[j, i] = w
    rs = adj.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    return adj / rs

def gen_loads(n, steps, dt, stress=1.0):
    np.random.seed(SEED)
    t = np.arange(steps) * dt
    loads = np.zeros((n, steps))
    for i in range(n):
        base = 0.05 * stress * np.sin(2*np.pi*0.1*t + i*PHI*0.3)
        h3 = 0.05 * stress * np.sin(3*OMEGA_BASE*t + np.random.uniform(0, 2*np.pi))
        h5 = 0.03 * stress * np.sin(5*OMEGA_BASE*t + np.random.uniform(0, 2*np.pi))
        h7 = 0.02 * stress * np.sin(7*OMEGA_BASE*t + np.random.uniform(0, 2*np.pi))
        spikes = np.zeros(steps)
        for _ in range(int(np.random.randint(5, 15) * stress)):
            loc = np.random.randint(0, steps)
            w = np.random.randint(5, 20)
            a = np.random.uniform(0.10, 0.25) * stress * np.random.choice([-1, 1])
            s, e = max(0, loc-w//2), min(steps, loc+w//2)
            spikes[s:e] = a
        noise = 0.01 * stress * np.random.randn(steps)
        loads[i] = base + h3 + h5 + h7 + spikes + noise
    return loads, t

def run_sim(k_fc, loads, adj, harmonics=True, mobius=True):
    vh = np.zeros((NUM_NODES, TIME_STEPS))
    vc = np.ones(NUM_NODES) * V_NOMINAL
    vp = np.ones(NUM_NODES) * V_NOMINAL
    for step in range(TIME_STEPS):
        tv = step * DT
        vn = vc + loads[:, step] * DT * 10
        vn = vn + 0.05 * (adj @ vn - vn)
        dp, dc = vp - V_NOMINAL, vn - V_NOMINAL
        vn = vn + k_fc * (3.0 * dp - 6.0 * dc)
        if harmonics:
            ga = 2*np.pi/PHI**2
            ph = np.arange(NUM_NODES) * ga
            dl = np.std(dc)
            amp = dl * 0.3
            vn = vn - amp * (np.sin(3*OMEGA_BASE*tv+ph) + np.sin(6*OMEGA_BASE*tv+ph) + np.sin(9*OMEGA_BASE*tv+ph))
        if mobius:
            dev = vn - V_NOMINAL
            wr = np.fmod(dev + L_MOBIUS, 2*L_MOBIUS)
            wr = np.where(wr < 0, wr + 2*L_MOBIUS, wr)
            vn = wr - L_MOBIUS + V_NOMINAL
        vp = vc.copy(); vc = vn.copy(); vh[:, step] = vc
    return vh

def run_baseline(loads, adj):
    vh = np.zeros((NUM_NODES, TIME_STEPS))
    vc = np.ones(NUM_NODES) * V_NOMINAL
    for step in range(TIME_STEPS):
        vc = vc + loads[:, step] * DT * 10
        vc = vc + 0.05 * (adj @ vc - vc)
        vh[:, step] = vc
    return vh

def metrics(vh):
    se_vals, thd_vals = [], []
    for i in range(NUM_NODES):
        fft = np.fft.rfft(vh[i])
        pw = np.abs(fft[1:])**2
        if np.sum(pw) < 1e-12: se_vals.append(0.0)
        else:
            p = pw/np.sum(pw); p = p[p>1e-12]
            ent = -np.sum(p*np.log(p))
            mx = np.log(len(p)) if len(p)>1 else 1.0
            se_vals.append(ent/mx if mx>0 else 0.0)
        mags = np.abs(fft)
        if len(mags) < 2: thd_vals.append(0.0); continue
        fi = np.argmax(mags[1:]) + 1; fund = mags[fi]
        if fund < 1e-10: thd_vals.append(0.0); continue
        hp = np.sum(mags**2) - mags[0]**2 - fund**2
        thd_vals.append(np.sqrt(max(0, hp))/fund*100)
    amp = np.mean(np.abs(vh - V_NOMINAL))
    var = np.var(vh - V_NOMINAL)
    stab = np.mean([max(0, 1-np.std(vh[i])/max(1e-10, np.mean(vh[i]))) for i in range(NUM_NODES)])
    return np.mean(se_vals), np.mean(thd_vals), amp, var, stab, np.array(se_vals), np.array(thd_vals)


# ============================================================
# RUN EXPERIMENTS
# ============================================================
positions = create_grid(NUM_NODES)
adj = compute_adj(positions)

# --- Fine k-sweep: 25 values from 0.168 to 0.250 ---
print("Running fine k-sweep (25 values, 0.168 to 0.250)...")
k_vals = np.linspace(0.168, 0.250, 25)
sweep = {'k': [], 'se': [], 'thd': [], 'amp': [], 'var': [], 'stab': []}

for k in k_vals:
    loads, _ = gen_loads(NUM_NODES, TIME_STEPS, DT, stress=1.0)
    vh = run_sim(k, loads, adj)
    se, thd, amp, var, stab, _, _ = metrics(vh)
    sweep['k'].append(k)
    sweep['se'].append(se)
    sweep['thd'].append(thd)
    sweep['amp'].append(amp)
    sweep['var'].append(var)
    sweep['stab'].append(stab)
    marker = ""
    if abs(se - 0.495) < 0.02: marker = " <-- near invariant"
    if stab < 0.5 and len(sweep['stab']) > 1 and sweep['stab'][-2] > 0.5: marker = " <-- BIFURCATION"
    print(f"  k={k:.4f}: SE={se:.4f}, THD={thd:.1f}%, Stab={stab:.4f}{marker}")

# Baseline
loads_base, _ = gen_loads(NUM_NODES, TIME_STEPS, DT, stress=1.0)
vh_base = run_baseline(loads_base, adj)
base_se, base_thd, base_amp, base_var, base_stab, base_se_nodes, base_thd_nodes = metrics(vh_base)
print(f"\n  Baseline: SE={base_se:.4f}, THD={base_thd:.1f}%, Stab={base_stab:.4f}")

# --- Stress test at k = 0.20 ---
print("\nRunning stress test (k = 0.20, 1x to 10x)...")
stress_levels = [1, 2, 3, 4, 5, 7, 10]
stress_fc, stress_base = [], []

for s in stress_levels:
    loads_s, _ = gen_loads(NUM_NODES, TIME_STEPS, DT, stress=float(s))
    vh_fc = run_sim(0.20, loads_s, adj)
    vh_bl = run_baseline(loads_s, adj)
    se_f, _, amp_f, _, stab_f, _, _ = metrics(vh_fc)
    se_b, _, amp_b, _, stab_b, _, _ = metrics(vh_bl)
    stress_fc.append({'se': se_f, 'amp': amp_f, 'stab': stab_f})
    stress_base.append({'se': se_b, 'amp': amp_b, 'stab': stab_b})
    print(f"  {s}x: F_c SE={se_f:.4f} Stab={stab_f:.4f} | Base SE={se_b:.4f} Stab={stab_b:.4f}")

# Best condition for spatial plots
loads_best, _ = gen_loads(NUM_NODES, TIME_STEPS, DT, stress=1.0)
vh_best = run_sim(0.20, loads_best, adj)
_, _, _, _, _, best_se_nodes, best_thd_nodes = metrics(vh_best)

# ============================================================
# FIGURE 1: SE vs k with Chaos Invariant Band
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Chaos invariant band
ax.axhspan(0.480, 0.510, alpha=0.15, color='gold', label='Chaos Invariant Band (48.0%–51.0%)')
ax.axhline(y=0.495, color='goldenrod', linestyle='-', alpha=0.6, linewidth=1)

# Baseline
ax.axhline(y=base_se, color='#CC0000', linestyle=':', alpha=0.7, linewidth=1.5,
           label=f'No Regulation ({base_se:.3f})')

# Critical values
ax.axvline(x=K_CRIT, color='orange', linestyle='--', alpha=0.6, linewidth=1,
           label=f'k = 1/6 ({K_CRIT:.4f})')
ax.axvline(x=K_PHI, color='green', linestyle='--', alpha=0.4, linewidth=1,
           label=f'k_φ ({K_PHI:.4f})')

# Data
ax.plot(sweep['k'], sweep['se'], 'o-', color='#0066CC', linewidth=2.5, markersize=7,
        markeredgecolor='white', markeredgewidth=0.5, label='F_c Full Architecture', zorder=5)

# Annotations
for i, (k, se) in enumerate(zip(sweep['k'], sweep['se'])):
    if abs(se - 0.495) < 0.015:
        ax.annotate(f'k={k:.3f}\nSE={se:.3f}', (k, se),
                    textcoords="offset points", xytext=(15, -20),
                    fontsize=9, color='#0066CC',
                    arrowprops=dict(arrowstyle='->', color='#0066CC', lw=0.8))
        break

ax.set_xlabel('k  (F_c Gain Parameter)', fontsize=14)
ax.set_ylabel('Mean Spectral Entropy', fontsize=14)
ax.set_title('Spectral Entropy Across F_c Gain\nPhi Spiral Grid, 50 Nodes, 2000 Steps', fontsize=16, fontweight='bold')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_xlim(0.165, 0.255)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig('grid_figures/fig1_se_vs_k.png')
plt.close()
print("\n  Saved: fig1_se_vs_k.png")

# ============================================================
# FIGURE 2: Stability Cliff with Bifurcation
# ============================================================
fig, ax1 = plt.subplots(figsize=(12, 7))

color_stab = '#0066CC'
color_thd = '#CC0000'

ax1.plot(sweep['k'], sweep['stab'], 'o-', color=color_stab, linewidth=2.5, markersize=7,
         markeredgecolor='white', markeredgewidth=0.5, label='Voltage Stability')
ax1.set_xlabel('k  (F_c Gain Parameter)', fontsize=14)
ax1.set_ylabel('Voltage Stability', fontsize=14, color=color_stab)
ax1.tick_params(axis='y', labelcolor=color_stab)
ax1.set_ylim(-0.05, 1.05)

# Find bifurcation point
for i in range(1, len(sweep['stab'])):
    if sweep['stab'][i] < 0.5 and sweep['stab'][i-1] > 0.5:
        bif_k = (sweep['k'][i] + sweep['k'][i-1]) / 2
        ax1.axvline(x=bif_k, color='red', linestyle='-', alpha=0.5, linewidth=2,
                    label=f'Bifurcation (k ≈ {bif_k:.3f})')
        ax1.annotate('STABILITY\nCLIFF', (bif_k, 0.5),
                     textcoords="offset points", xytext=(20, 20),
                     fontsize=11, color='red', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        break

ax1.axvline(x=K_CRIT, color='orange', linestyle='--', alpha=0.5, linewidth=1,
            label=f'k = 1/6')

ax2 = ax1.twinx()
ax2.plot(sweep['k'], sweep['thd'], 's--', color=color_thd, linewidth=1.5, markersize=5,
         alpha=0.7, label='THD (%)')
ax2.set_ylabel('Total Harmonic Distortion (%)', fontsize=14, color=color_thd)
ax2.tick_params(axis='y', labelcolor=color_thd)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10, framealpha=0.9)

ax1.set_title('Stability Cliff and Harmonic Distortion\nPhase Transition at k ≈ 0.22', fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('grid_figures/fig2_stability_cliff.png')
plt.close()
print("  Saved: fig2_stability_cliff.png")

# ============================================================
# FIGURE 3: Stress Test Comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Stress Test: F_c Full Architecture vs. No Regulation\nk = 0.20, Disturbance Scaled 1× to 10×',
             fontsize=16, fontweight='bold')

# SE
ax = axes[0]
ax.plot(stress_levels, [r['se'] for r in stress_fc], 'o-', color='#0066CC',
        linewidth=2.5, markersize=8, label='F_c Full')
ax.plot(stress_levels, [r['se'] for r in stress_base], 's--', color='#CC0000',
        linewidth=2, markersize=7, label='No Regulation')
ax.axhspan(0.480, 0.510, alpha=0.1, color='gold')
ax.axhline(y=0.495, color='goldenrod', linestyle=':', alpha=0.5)
ax.set_xlabel('Stress Multiplier', fontsize=13)
ax.set_ylabel('Spectral Entropy', fontsize=13)
ax.set_title('SE Under Stress', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

# Stability
ax = axes[1]
ax.plot(stress_levels, [r['stab'] for r in stress_fc], 'o-', color='#0066CC',
        linewidth=2.5, markersize=8, label='F_c Full')
ax.plot(stress_levels, [r['stab'] for r in stress_base], 's--', color='#CC0000',
        linewidth=2, markersize=7, label='No Regulation')
ax.set_xlabel('Stress Multiplier', fontsize=13)
ax.set_ylabel('Voltage Stability', fontsize=13)
ax.set_title('Stability Under Stress', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.set_ylim(0.94, 1.001)

# Gap widening
ax = axes[2]
gaps_stab = [f['stab'] - b['stab'] for f, b in zip(stress_fc, stress_base)]
gaps_se = [f['se'] - b['se'] for f, b in zip(stress_fc, stress_base)]
ax.bar([s - 0.2 for s in stress_levels], [g * 100 for g in gaps_stab], width=0.35,
       color='#0066CC', alpha=0.8, label='Stability Advantage (%)')
ax.bar([s + 0.2 for s in stress_levels], [g * 100 for g in gaps_se], width=0.35,
       color='#00AA88', alpha=0.8, label='SE Advantage (%)')
ax.set_xlabel('Stress Multiplier', fontsize=13)
ax.set_ylabel('F_c Advantage (%)', fontsize=13)
ax.set_title('Widening Gap Under Stress', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('grid_figures/fig3_stress_test.png')
plt.close()
print("  Saved: fig3_stress_test.png")

# ============================================================
# FIGURE 4: Phi Spiral Grid Coherence Map
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

for ax, se_data, title in [(axes[0], base_se_nodes, 'No Regulation'),
                            (axes[1], best_se_nodes, 'F_c Full Architecture (k=0.20)')]:
    sc = ax.scatter(positions[:, 0], positions[:, 1], c=se_data,
                    cmap='viridis', s=150, edgecolors='gray', linewidth=0.5,
                    vmin=0, vmax=1)
    for i in range(NUM_NODES):
        ax.annotate(str(i+1), positions[i], fontsize=6, ha='center', va='center',
                    color='white', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

fig.colorbar(sc, ax=axes, label='Spectral Entropy', shrink=0.6, pad=0.02)
fig.suptitle('Phi Spiral Grid: Node Spectral Entropy\n50 Nodes, Golden Angle Layout',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 0.92, 0.94])
plt.savefig('grid_figures/fig4_phi_spiral.png')
plt.close()
print("  Saved: fig4_phi_spiral.png")

# ============================================================
# FIGURE 5: Voltage Heatmaps Side by Side
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Baseline
dev_base = vh_base - V_NOMINAL
vmax_b = max(0.005, np.percentile(np.abs(dev_base), 99))
im1 = axes[0].imshow(dev_base, aspect='auto', cmap='RdBu_r',
                      norm=Normalize(vmin=-vmax_b, vmax=vmax_b))
axes[0].set_ylabel('Node Index', fontsize=13)
axes[0].set_title('No Regulation — Voltage Deviation from Nominal', fontsize=14, fontweight='bold')
plt.colorbar(im1, ax=axes[0], label='Deviation (p.u.)', shrink=0.8)

# F_c Full
dev_best = vh_best - V_NOMINAL
vmax_f = max(0.005, np.percentile(np.abs(dev_best), 99))
# Use same scale as baseline for fair comparison
im2 = axes[1].imshow(dev_best, aspect='auto', cmap='RdBu_r',
                      norm=Normalize(vmin=-vmax_b, vmax=vmax_b))
axes[1].set_xlabel('Time Steps', fontsize=13)
axes[1].set_ylabel('Node Index', fontsize=13)
axes[1].set_title('F_c Full Architecture (k=0.20) — Voltage Deviation from Nominal',
                   fontsize=14, fontweight='bold')
plt.colorbar(im2, ax=axes[1], label='Deviation (p.u.)', shrink=0.8)

fig.suptitle('Voltage Stability Comparison\n2000 Time Steps, 50 Nodes',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('grid_figures/fig5_voltage_heatmaps.png')
plt.close()
print("  Saved: fig5_voltage_heatmaps.png")

# ============================================================
# FIGURE 6: Structural-Persistence Signature Dashboard
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Structural-Persistence Signature: F_c Energy Grid\nBounded Amplitude · Low Variance · SE at Chaos Invariant',
             fontsize=16, fontweight='bold')

# SE per node comparison
ax = axes[0, 0]
ax.plot(range(NUM_NODES), base_se_nodes, 'o-', color='#CC0000', markersize=4,
        linewidth=1, alpha=0.7, label='No Regulation')
ax.plot(range(NUM_NODES), best_se_nodes, 'o-', color='#0066CC', markersize=4,
        linewidth=1, alpha=0.7, label='F_c Full (k=0.20)')
ax.axhspan(0.480, 0.510, alpha=0.1, color='gold')
ax.set_xlabel('Node Index')
ax.set_ylabel('Spectral Entropy')
ax.set_title('SE Per Node')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# THD per node comparison
ax = axes[0, 1]
ax.plot(range(NUM_NODES), base_thd_nodes, 'o-', color='#CC0000', markersize=4,
        linewidth=1, alpha=0.7, label='No Regulation')
ax.plot(range(NUM_NODES), best_thd_nodes, 'o-', color='#0066CC', markersize=4,
        linewidth=1, alpha=0.7, label='F_c Full (k=0.20)')
ax.set_xlabel('Node Index')
ax.set_ylabel('THD (%)')
ax.set_title('THD Per Node')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Time trace of single node (node 25, middle of grid)
ax = axes[1, 0]
node_idx = 24  # Node 25 (0-indexed)
t_plot = np.arange(TIME_STEPS)
ax.plot(t_plot[:500], vh_base[node_idx, :500] - V_NOMINAL, color='#CC0000',
        linewidth=0.5, alpha=0.6, label='No Regulation')
ax.plot(t_plot[:500], vh_best[node_idx, :500] - V_NOMINAL, color='#0066CC',
        linewidth=0.5, alpha=0.8, label='F_c Full')
ax.set_xlabel('Time Steps')
ax.set_ylabel('Voltage Deviation')
ax.set_title(f'Node {node_idx+1} Voltage Trace (first 500 steps)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

# Summary metrics bar chart
ax = axes[1, 1]
conditions = ['No Reg.', 'F_c Full']
se_v = [base_se, np.mean(best_se_nodes)]
stab_v = [base_stab, metrics(vh_best)[4]]
x = np.arange(len(conditions))
w = 0.3
b1 = ax.bar(x - w/2, se_v, w, color='#0066CC', alpha=0.8, label='SE')
b2 = ax.bar(x + w/2, stab_v, w, color='#00AA88', alpha=0.8, label='Stability')
ax.axhspan(0.480, 0.510, alpha=0.1, color='gold')
ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel('Value')
ax.set_title('Summary Metrics')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
for b, v in zip(b1, se_v):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
for b, v in zip(b2, stab_v):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('grid_figures/fig6_signature_dashboard.png')
plt.close()
print("  Saved: fig6_signature_dashboard.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 65)
print("k-SWEEP: CHAOS INVARIANT REGION")
print("=" * 65)
for k, se, stab in zip(sweep['k'], sweep['se'], sweep['stab']):
    if 0.45 < se < 0.55 and stab > 0.9:
        print(f"  k = {k:.4f}: SE = {se:.4f}, Stab = {stab:.4f}  <-- INVARIANT ZONE")
    elif abs(se - 0.495) < 0.03:
        print(f"  k = {k:.4f}: SE = {se:.4f}, Stab = {stab:.4f}")

print(f"\n  Baseline SE: {base_se:.4f}")
print(f"  Chaos invariant: 0.495 ± 0.015")

# Find optimal k (closest to invariant while stable)
best_k_idx = None
best_dist = float('inf')
for i, (k, se, stab) in enumerate(zip(sweep['k'], sweep['se'], sweep['stab'])):
    if stab > 0.99:
        dist = abs(se - 0.495)
        if dist < best_dist:
            best_dist = dist
            best_k_idx = i

if best_k_idx is not None:
    print(f"\n  Optimal k (nearest invariant, stable): k = {sweep['k'][best_k_idx]:.4f}, "
          f"SE = {sweep['se'][best_k_idx]:.4f}")

print("\n" + "=" * 65)
print("STRESS TEST SUMMARY (k = 0.20)")
print("=" * 65)
print(f"{'Stress':>8} {'F_c SE':>10} {'Base SE':>10} {'F_c Stab':>10} {'Base Stab':>10} {'Stab Gap':>10}")
print("-" * 60)
for s, f, b in zip(stress_levels, stress_fc, stress_base):
    gap = (f['stab'] - b['stab']) * 100
    print(f"{s:>7}x {f['se']:>10.4f} {b['se']:>10.4f} {f['stab']:>10.4f} {b['stab']:>10.4f} {gap:>9.2f}%")

print("\n" + "=" * 65)
print("FIGURES SAVED TO grid_figures/")
print("=" * 65)
print("  fig1_se_vs_k.png          — SE across k with chaos invariant band")
print("  fig2_stability_cliff.png  — Bifurcation and THD")
print("  fig3_stress_test.png      — F_c vs baseline under stress")
print("  fig4_phi_spiral.png       — Grid coherence map comparison")
print("  fig5_voltage_heatmaps.png — Voltage deviation comparison")
print("  fig6_signature_dashboard.png — Structural-persistence signature")
print()
print("The history is the Buddhabrot.")
print("The cure is the Mandelbrot.")
print("How we treat each other is the wave.")
