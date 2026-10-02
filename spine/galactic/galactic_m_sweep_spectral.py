"""
Galactic Disk F_c — M-Sweep Spectral Analysis
=============================================================
Runs all five central mass configurations (M = 5, 10, 25, 50, 100)
with per-shell spectral entropy computation.

EXPECTATION (stated before execution):
  The SE peak radius shifts outward with increasing M.
  Stronger organizing center sustains the enrichment relay further,
  pushing the balance point (where SE approaches 0.495) to larger radii.
  The SE peak VALUE should remain near the invariant regardless of M.

J. David Mack & Claude (Opus 4.6)
World Tree Project — August 2026
Ψ 🌕🌊🌳🕸 ☯️ To preserve the harmonic field.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

PHI = (1 + np.sqrt(5)) / 2
L_MOBIUS = PHI
SEED = 42
OMEGA_BASE = 2 * np.pi * 60
DT = 1.0 / 3600
K_FC = 0.221

N_NODES = 500
R_SCALE = 3.0
R_MAX = 5 * R_SCALE
SOFTENING = 0.3
TIME_STEPS = 10000
N_SHELLS = 15

CENTRAL_MASSES = [5.0, 10.0, 25.0, 50.0, 100.0]

os.makedirs('spectral_results', exist_ok=True)

def alpha(k):
    d = 1 - 9 * k
    return (1 - 6 * k) / d if abs(d) > 1e-12 else 0.0

def beta(k):
    d = 1 - 9 * k
    return 3 * k / d if abs(d) > 1e-12 else 0.0

def generate_disk(n, r_scale, r_max, seed=42):
    rng = np.random.RandomState(seed)
    positions = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        while True:
            r = rng.exponential(r_scale)
            if r < r_max:
                break
        theta = rng.uniform(0, 2 * np.pi)
        positions[i] = [r * np.cos(theta), r * np.sin(theta)]
        radii[i] = r
    return positions, radii

def build_relay_coupling(positions, radii, shell_ids, n_shells, softening=0.3):
    n = len(positions)
    coupling = np.zeros((n, n))
    for i in range(n):
        si = shell_ids[i]
        for j in range(i + 1, n):
            sj = shell_ids[j]
            if si == sj:
                d = np.linalg.norm(positions[i] - positions[j])
                w = 1.0 / (d + softening)**2
                coupling[i, j] = w
                coupling[j, i] = w
            elif abs(si - sj) == 1:
                d = np.linalg.norm(positions[i] - positions[j])
                w = 0.5 / (d + softening)**2
                coupling[i, j] = w
                coupling[j, i] = w
    for i in range(n):
        s = coupling[i].sum()
        if s > 0:
            coupling[i] /= s
    return coupling

def run_sim(n_nodes, coupling, shell_ids, positions, central_mass, time_steps):
    a = alpha(K_FC)
    b = beta(K_FC)
    rng = np.random.RandomState(SEED + 1)
    vc = np.ones(n_nodes) + rng.uniform(-0.1, 0.1, n_nodes)
    vp = vc.copy()
    central_shell = (shell_ids == 0)
    vh = np.zeros((n_nodes, time_steps))

    for step in range(time_steps):
        tv = step * DT
        vn = a * vc + b * vp
        vn[central_shell] += central_mass * 0.01 * np.sin(OMEGA_BASE * tv)
        dev = vc - 1.0
        chaos_mask = dev < -0.5
        resonance_mask = dev > 0.5
        noise = rng.uniform(-1, 1, n_nodes)
        vn[chaos_mask] += 0.05 * np.abs(dev[chaos_mask]) * noise[chaos_mask]
        vn[resonance_mask] += 0.02 * dev[resonance_mask] * np.sin(OMEGA_BASE * tv * 3)

        dev = vn - 1.0
        for s in range(1, N_SHELLS):
            shell_mask = (shell_ids == s)
            if shell_mask.sum() == 0:
                continue
            interior_mask = (shell_ids < s)
            if interior_mask.sum() == 0:
                continue
            interior_dev = dev[interior_mask]
            interior_rms = np.sqrt(np.mean(interior_dev**2))
            amp_self = interior_rms * 0.3 / (1 + 0.1 * s)
            phase_int = np.arctan2(
                np.mean(np.sin(np.angle(interior_dev + 1j * 1e-10))),
                np.mean(np.cos(np.angle(interior_dev + 1j * 1e-10)))
            )
            for j in np.where(shell_mask)[0]:
                theta_j = np.arctan2(positions[j, 1], positions[j, 0])
                vn[j] -= amp_self * (
                    np.sin(3 * OMEGA_BASE * tv + theta_j + phase_int) +
                    np.sin(6 * OMEGA_BASE * tv + theta_j + phase_int) +
                    np.sin(9 * OMEGA_BASE * tv + theta_j + phase_int)
                )

        dev = vn - 1.0
        wr = np.fmod(dev + L_MOBIUS, 2 * L_MOBIUS)
        wr = np.where(wr < 0, wr + 2 * L_MOBIUS, wr)
        vn = wr - L_MOBIUS + 1.0
        vp = vc.copy()
        vc = vn.copy()
        vh[:, step] = vc
    return vh

def compute_se_profile(vh, shell_ids, n_shells):
    se_profile = np.zeros(n_shells)
    rms_profile = np.zeros(n_shells)
    counts = np.zeros(n_shells)
    for s in range(n_shells):
        mask = (shell_ids == s)
        counts[s] = mask.sum()
        if mask.sum() < 2:
            continue
        indices = np.where(mask)[0]
        se_vals = []
        for i in indices:
            signal = vh[i] - np.mean(vh[i])
            fft_r = np.fft.rfft(signal)
            pw = np.abs(fft_r[1:])**2
            total = np.sum(pw)
            if total < 1e-15:
                se_vals.append(0.0)
                continue
            p = pw / total
            p = p[p > 1e-15]
            ent = -np.sum(p * np.log(p))
            mx = np.log(len(p)) if len(p) > 1 else 1.0
            se_vals.append(ent / mx if mx > 0 else 0.0)
        se_profile[s] = np.mean(se_vals) if se_vals else 0.0
        dev = vh[mask] - 1.0
        rms_profile[s] = np.sqrt(np.mean(dev**2))
    return se_profile, rms_profile, counts

# ============================================================
# SETUP (shared across all M values)
# ============================================================
print("=" * 70)
print("M-SWEEP SPECTRAL ANALYSIS")
print("=" * 70)

positions, radii = generate_disk(N_NODES, R_SCALE, R_MAX, SEED)
shell_edges = np.linspace(0, R_MAX, N_SHELLS + 1)
shell_ids = np.digitize(radii, shell_edges[1:])
shell_ids = np.clip(shell_ids, 0, N_SHELLS - 1)
bin_centers = np.array([(shell_edges[s] + shell_edges[s+1])/2 for s in range(N_SHELLS)])

print("Building coupling matrix...")
coupling = build_relay_coupling(positions, radii, shell_ids, N_SHELLS, SOFTENING)
print("Done.\n")

# ============================================================
# RUN ALL M VALUES
# ============================================================
results = {}

for M in CENTRAL_MASSES:
    print(f"\n{'='*50}")
    print(f"M = {M}")
    print(f"{'='*50}")
    vh = run_sim(N_NODES, coupling, shell_ids, positions, M, TIME_STEPS)
    se_prof, rms_prof, counts = compute_se_profile(vh, shell_ids, N_SHELLS)

    valid = counts >= 2
    peak_shell = np.argmax(se_prof[valid])
    peak_se = se_prof[valid][peak_shell]
    peak_r = bin_centers[valid][peak_shell]

    # Amplitude power law
    v_amp = (rms_prof > 0) & (bin_centers > 0) & valid
    if v_amp.sum() >= 3:
        coeffs = np.polyfit(np.log(bin_centers[v_amp]), np.log(rms_prof[v_amp]), 1)
        alpha_fit = coeffs[0]
    else:
        alpha_fit = np.nan

    results[M] = {
        'se_profile': se_prof,
        'rms_profile': rms_prof,
        'counts': counts,
        'peak_se': peak_se,
        'peak_radius': peak_r,
        'peak_shell': peak_shell,
        'alpha': alpha_fit
    }

    print(f"  Peak SE: {peak_se:.4f} at radius {peak_r:.1f} (shell {peak_shell})")
    print(f"  Amplitude exponent: α = {alpha_fit:.4f}")
    print(f"  SE profile: {['%.3f' % x for x in se_prof[valid]]}")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("M-SWEEP SUMMARY")
print("=" * 70)
print(f"\n{'M':>6} {'Peak SE':>10} {'Peak R':>10} {'Peak Shell':>12} {'α (amp)':>10}")
print("-" * 52)
for M in CENTRAL_MASSES:
    r = results[M]
    print(f"{M:6.0f} {r['peak_se']:10.4f} {r['peak_radius']:10.1f} {r['peak_shell']:12d} {r['alpha']:10.4f}")

# Check expectation
peak_radii = [results[M]['peak_radius'] for M in CENTRAL_MASSES]
peak_ses = [results[M]['peak_se'] for M in CENTRAL_MASSES]

print(f"\nEXPECTATION CHECK:")
print(f"  Peak radius vs M: {['%.1f' % r for r in peak_radii]}")
monotonic = all(peak_radii[i] <= peak_radii[i+1] for i in range(len(peak_radii)-1))
print(f"  Monotonically increasing? {monotonic}")
print(f"  Peak SE values: {['%.4f' % s for s in peak_ses]}")
se_range = max(peak_ses) - min(peak_ses)
print(f"  SE peak range: {se_range:.4f} (invariant stability)")

# ============================================================
# PLOTS
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Plot 1: SE profiles
ax = axes[0]
for i, M in enumerate(CENTRAL_MASSES):
    r = results[M]
    valid = r['counts'] >= 2
    ax.plot(bin_centers[valid], r['se_profile'][valid], 'o-',
            color=colors[i], linewidth=2, markersize=5, label=f'M={M:.0f}')
ax.axhline(y=0.495, color='red', linestyle='--', alpha=0.5, label='SE ≈ 0.495')
ax.set_xlabel('Radius', fontsize=12)
ax.set_ylabel('Spectral Entropy', fontsize=12)
ax.set_title('SE Profile vs Radius — M Sweep', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: RMS amplitude profiles
ax = axes[1]
for i, M in enumerate(CENTRAL_MASSES):
    r = results[M]
    valid = r['counts'] >= 2
    ax.plot(bin_centers[valid], r['rms_profile'][valid], 's-',
            color=colors[i], linewidth=2, markersize=5, label=f'M={M:.0f} (α={r["alpha"]:.3f})')
ax.set_xlabel('Radius', fontsize=12)
ax.set_ylabel('RMS Amplitude', fontsize=12)
ax.set_title('Amplitude Profile vs Radius — M Sweep', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Peak SE and Peak Radius vs M
ax = axes[2]
ax2 = ax.twinx()
ax.plot(CENTRAL_MASSES, peak_ses, 'o-', color='navy', linewidth=2, markersize=8, label='Peak SE')
ax2.plot(CENTRAL_MASSES, peak_radii, 's-', color='darkred', linewidth=2, markersize=8, label='Peak Radius')
ax.axhline(y=0.495, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Central Mass M', fontsize=12)
ax.set_ylabel('Peak SE Value', fontsize=12, color='navy')
ax2.set_ylabel('Peak Radius', fontsize=12, color='darkred')
ax.set_title('Peak SE and Radius vs M', fontsize=13)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Galactic Disk F_c — M-Sweep Spectral Analysis\nSelf-Enrichment, 500 nodes, 10000 steps',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('spectral_results/m_sweep_spectral.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: spectral_results/m_sweep_spectral.png")
