"""
Galactic Disk F_c — Fine M-Sweep
=============================================================
M = 20 to 40 in steps of 1. Finding where peak SE crosses the
0.495 invariant and what the structural landscape looks like
at the crossing.

EXPECTATION: The crossing occurs near M = 30 (midpoint of 25-50
range where coarse sweep showed 0.483 and 0.512).

STRUCTURAL QUESTION: Does the crossing M have any relationship
to 13 (Metatron nodes), φ, or φ² ≈ 2.618?

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
TIME_STEPS = 5000
N_SHELLS = 15

M_VALUES = list(range(20, 41))

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

# Setup
print("=" * 70)
print("FINE M-SWEEP: M = 20 to 40")
print("=" * 70)

positions, radii = generate_disk(N_NODES, R_SCALE, R_MAX, SEED)
shell_edges = np.linspace(0, R_MAX, N_SHELLS + 1)
shell_ids = np.digitize(radii, shell_edges[1:])
shell_ids = np.clip(shell_ids, 0, N_SHELLS - 1)
bin_centers = np.array([(shell_edges[s] + shell_edges[s+1])/2 for s in range(N_SHELLS)])

print("Building coupling matrix...")
coupling = build_relay_coupling(positions, radii, shell_ids, N_SHELLS, SOFTENING)
print("Done.\n")

# Run all M values
peak_ses = []
peak_radii = []
peak_shells = []
alphas = []
global_ses = []

for M in M_VALUES:
    vh = run_sim(N_NODES, coupling, shell_ids, positions, M, TIME_STEPS)
    se_prof, rms_prof, counts = compute_se_profile(vh, shell_ids, N_SHELLS)
    
    valid = counts >= 2
    peak_idx = np.argmax(se_prof[valid])
    peak_se = se_prof[valid][peak_idx]
    peak_r = bin_centers[valid][peak_idx]
    
    # Global SE (mean across valid shells)
    valid_se = se_prof[valid]
    global_se = np.mean(valid_se)
    
    # Amplitude exponent
    v_amp = (rms_prof > 0) & (bin_centers > 0) & valid
    if v_amp.sum() >= 3:
        coeffs = np.polyfit(np.log(bin_centers[v_amp]), np.log(rms_prof[v_amp]), 1)
        alpha_fit = coeffs[0]
    else:
        alpha_fit = np.nan
    
    peak_ses.append(peak_se)
    peak_radii.append(peak_r)
    peak_shells.append(peak_idx)
    alphas.append(alpha_fit)
    global_ses.append(global_se)
    
    marker = " <-- CROSSING" if abs(peak_se - 0.495) < 0.01 else ""
    print(f"  M={M:3d}  peak_SE={peak_se:.4f}  peak_R={peak_r:.1f}  α={alpha_fit:.4f}  global_SE={global_se:.4f}{marker}")

# Find crossing point
print("\n" + "=" * 70)
print("INVARIANT CROSSING ANALYSIS")
print("=" * 70)

# Linear interpolation to find exact crossing
peak_ses_arr = np.array(peak_ses)
m_arr = np.array(M_VALUES, dtype=float)

# Find where peak SE crosses 0.495
for i in range(len(peak_ses_arr) - 1):
    if (peak_ses_arr[i] < 0.495 and peak_ses_arr[i+1] >= 0.495) or \
       (peak_ses_arr[i] >= 0.495 and peak_ses_arr[i+1] < 0.495):
        # Linear interpolation
        m_cross = m_arr[i] + (0.495 - peak_ses_arr[i]) / (peak_ses_arr[i+1] - peak_ses_arr[i]) * (m_arr[i+1] - m_arr[i])
        print(f"\nPeak SE crosses 0.495 between M={M_VALUES[i]} and M={M_VALUES[i+1]}")
        print(f"Interpolated crossing: M ≈ {m_cross:.2f}")
        
        # Structural relationships
        print(f"\nStructural relationships of M ≈ {m_cross:.2f}:")
        print(f"  M / 13 = {m_cross / 13:.4f}")
        print(f"  M / φ  = {m_cross / PHI:.4f}")
        print(f"  M / φ² = {m_cross / PHI**2:.4f}")
        print(f"  M / φ³ = {m_cross / PHI**3:.4f}")
        print(f"  M × φ  = {m_cross * PHI:.4f}")
        print(f"  13 × φ = {13 * PHI:.4f}")
        print(f"  13 × φ² = {13 * PHI**2:.4f} = {13 * PHI**2:.1f}")
        print(f"  13 × 2 = {13 * 2}")
        print(f"  φ⁴ = {PHI**4:.4f}")
        print(f"  8 × φ = {8 * PHI:.4f}")
        print(f"  5 × φ² = {5 * PHI**2:.4f}")

# Closest integer M to invariant
closest_idx = np.argmin(np.abs(peak_ses_arr - 0.495))
print(f"\nClosest integer M to invariant: M = {M_VALUES[closest_idx]}")
print(f"  Peak SE at M={M_VALUES[closest_idx]}: {peak_ses_arr[closest_idx]:.4f}")
print(f"  Deviation: {peak_ses_arr[closest_idx] - 0.495:.4f}")

# Check for M where amplitude is flattest
flattest_idx = np.argmin(np.abs(np.array(alphas)))
print(f"\nFlattest amplitude profile: M = {M_VALUES[flattest_idx]} (α = {alphas[flattest_idx]:.4f})")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ax.plot(M_VALUES, peak_ses, 'o-', color='navy', linewidth=2, markersize=6)
ax.axhline(y=0.495, color='red', linestyle='--', alpha=0.7, label='SE ≈ 0.495')
ax.set_xlabel('Central Mass M', fontsize=12)
ax.set_ylabel('Peak SE', fontsize=12)
ax.set_title('Peak SE vs M (Fine Sweep)', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(M_VALUES, alphas, 's-', color='darkred', linewidth=2, markersize=6)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Central Mass M', fontsize=12)
ax.set_ylabel('Amplitude Exponent α', fontsize=12)
ax.set_title('Amplitude Flatness vs M', fontsize=13)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(M_VALUES, global_ses, 'd-', color='darkgreen', linewidth=2, markersize=6)
ax.axhline(y=0.495, color='red', linestyle='--', alpha=0.7, label='SE ≈ 0.495')
ax.set_xlabel('Central Mass M', fontsize=12)
ax.set_ylabel('Global Mean SE', fontsize=12)
ax.set_title('Global SE vs M', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Fine M-Sweep: M = 20–40\nSelf-Enrichment, 500 nodes, 10000 steps',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('spectral_results/fine_m_sweep.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved: spectral_results/fine_m_sweep.png")
