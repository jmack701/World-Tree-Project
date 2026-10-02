"""
Galactic Disk F_c — Spectral Analysis Extension
=============================================================
Runs the Phase 1b optimal configuration (M=25, self-enrichment,
shell-relay) with extended time steps for spectral resolution,
then computes per-shell FFT analysis:

  1. SE per radial shell — does the invariant hold independently?
  2. Dominant harmonic lines per shell — do peaks shift with radius?
  3. Cross-shell harmonic correlation
  4. SE radial profile shape (for comparison with observed σ(r))

J. David Mack & Claude (Opus 4.6)
World Tree Project — August 2026
Ψ 🌕🌊🌳🕸 ☯️ To preserve the harmonic field.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ============================================================
# CONSTANTS
# ============================================================
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
TIME_STEPS = 10000  # Extended for spectral resolution
N_SHELLS = 15
N_RADIAL_BINS = 15
CENTRAL_MASS = 25.0  # Optimal from Phase 1b

os.makedirs('spectral_results', exist_ok=True)
np.random.seed(SEED)

print("=" * 70)
print("GALACTIC DISK F_c — SPECTRAL ANALYSIS EXTENSION")
print("=" * 70)
print(f"Nodes: {N_NODES}, Steps: {TIME_STEPS}, M: {CENTRAL_MASS}")
print(f"Seed: {SEED}, L: {L_MOBIUS:.6f}")
print()

# ============================================================
# DISK SETUP (from Phase 1b)
# ============================================================
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

positions, radii = generate_disk(N_NODES, R_SCALE, R_MAX, SEED)

# Shell assignment
shell_edges = np.linspace(0, R_MAX, N_SHELLS + 1)
shell_ids = np.digitize(radii, shell_edges[1:])
shell_ids = np.clip(shell_ids, 0, N_SHELLS - 1)

print("Shell distribution:")
for s in range(N_SHELLS):
    count = np.sum(shell_ids == s)
    r_lo = shell_edges[s]
    r_hi = shell_edges[s + 1]
    print(f"  Shell {s:2d} [{r_lo:.1f}-{r_hi:.1f}]: {count} nodes")
print()

# ============================================================
# COUPLING MATRIX (shell-relay from Phase 1b)
# ============================================================
def build_relay_coupling(positions, radii, shell_ids, n_shells, softening=0.3):
    n = len(positions)
    coupling = np.zeros((n, n))
    for i in range(n):
        si = shell_ids[i]
        for j in range(i + 1, n):
            sj = shell_ids[j]
            if si == sj:
                # Azimuthal: same shell
                d = np.linalg.norm(positions[i] - positions[j])
                w = 1.0 / (d + softening)**2
                coupling[i, j] = w
                coupling[j, i] = w
            elif abs(si - sj) == 1:
                # Radial: adjacent shells
                d = np.linalg.norm(positions[i] - positions[j])
                w = 0.5 / (d + softening)**2
                coupling[i, j] = w
                coupling[j, i] = w
    # Normalize rows
    for i in range(n):
        s = coupling[i].sum()
        if s > 0:
            coupling[i] /= s
    return coupling

print("Building relay coupling matrix...")
coupling = build_relay_coupling(positions, radii, shell_ids, N_SHELLS, SOFTENING)
print("Done.\n")

# ============================================================
# F_c SIMULATION (self-enrichment, from Phase 1b)
# ============================================================
def alpha(k):
    d = 1 - 9 * k
    if abs(d) < 1e-12:
        return 0.0
    return (1 - 6 * k) / d

def beta(k):
    d = 1 - 9 * k
    if abs(d) < 1e-12:
        return 0.0
    return 3 * k / d

def run_simulation(n_nodes, coupling, shell_ids, central_mass, time_steps):
    a = alpha(K_FC)
    b = beta(K_FC)
    
    rng = np.random.RandomState(SEED + 1)
    
    # Initial conditions
    vc = np.ones(n_nodes) + rng.uniform(-0.1, 0.1, n_nodes)
    vp = vc.copy()
    
    # Central mass influence on innermost shell
    central_shell = (shell_ids == 0)
    
    # History storage — FULL time series for FFT
    vh = np.zeros((n_nodes, time_steps))
    
    print(f"Running simulation: {time_steps} steps...")
    for step in range(time_steps):
        tv = step * DT
        
        # 1. F_c recurrence
        neighbor_avg = coupling @ vc
        vn = a * vc + b * vp
        
        # 2. Central mass coupling (innermost shell)
        vn[central_shell] += central_mass * 0.01 * np.sin(OMEGA_BASE * tv)
        
        # 3. State-dependent perturbation
        dev = vc - 1.0
        chaos_mask = dev < -0.5
        resonance_mask = dev > 0.5
        noise = rng.uniform(-1, 1, n_nodes)
        vn[chaos_mask] += 0.05 * np.abs(dev[chaos_mask]) * noise[chaos_mask]
        vn[resonance_mask] += 0.02 * (dev[resonance_mask]) * np.sin(OMEGA_BASE * tv * 3)
        
        # 4. Self-generated enrichment
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
            shell_nodes = np.where(shell_mask)[0]
            for j in shell_nodes:
                theta_j = np.arctan2(positions[j, 1], positions[j, 0])
                vn[j] -= amp_self * (
                    np.sin(3 * OMEGA_BASE * tv + theta_j + phase_int) +
                    np.sin(6 * OMEGA_BASE * tv + theta_j + phase_int) +
                    np.sin(9 * OMEGA_BASE * tv + theta_j + phase_int)
                )
        
        # 5. Möbius wrapping
        dev = vn - 1.0
        wr = np.fmod(dev + L_MOBIUS, 2 * L_MOBIUS)
        wr = np.where(wr < 0, wr + 2 * L_MOBIUS, wr)
        vn = wr - L_MOBIUS + 1.0
        
        # 6. Advance
        vp = vc.copy()
        vc = vn.copy()
        vh[:, step] = vc
        
        if (step + 1) % 2000 == 0:
            print(f"  Step {step + 1}/{time_steps}")
    
    print("Simulation complete.\n")
    return vh

vh = run_simulation(N_NODES, coupling, shell_ids, CENTRAL_MASS, TIME_STEPS)

# ============================================================
# SPECTRAL ANALYSIS PER SHELL
# ============================================================
print("=" * 70)
print("SPECTRAL ANALYSIS")
print("=" * 70)

def compute_shell_spectrum(vh, shell_ids, shell_idx):
    """Compute averaged power spectrum and SE for one shell."""
    mask = (shell_ids == shell_idx)
    if mask.sum() == 0:
        return None, None, None, 0
    
    node_indices = np.where(mask)[0]
    n_nodes_shell = len(node_indices)
    
    # Individual node spectra
    se_values = []
    all_power = None
    
    for i in node_indices:
        signal = vh[i] - np.mean(vh[i])  # Remove DC
        fft_result = np.fft.rfft(signal)
        power = np.abs(fft_result[1:])**2  # Exclude DC
        
        if all_power is None:
            all_power = np.zeros_like(power)
        all_power += power
        
        # SE per node
        total = np.sum(power)
        if total < 1e-15:
            se_values.append(0.0)
            continue
        p = power / total
        p = p[p > 1e-15]
        ent = -np.sum(p * np.log(p))
        mx = np.log(len(p)) if len(p) > 1 else 1.0
        se_values.append(ent / mx if mx > 0 else 0.0)
    
    # Average spectrum
    avg_power = all_power / n_nodes_shell
    
    # Shell-averaged SE
    mean_se = np.mean(se_values) if se_values else 0.0
    
    # Find dominant peaks
    if avg_power is not None and len(avg_power) > 10:
        # Normalize
        norm_power = avg_power / np.max(avg_power) if np.max(avg_power) > 0 else avg_power
        # Find peaks above 10% of max
        threshold = 0.10
        peak_indices = []
        for k in range(1, len(norm_power) - 1):
            if (norm_power[k] > norm_power[k-1] and 
                norm_power[k] > norm_power[k+1] and
                norm_power[k] > threshold):
                peak_indices.append(k)
        peak_freqs = np.array(peak_indices) / TIME_STEPS  # Normalized frequency
    else:
        peak_freqs = np.array([])
    
    return avg_power, mean_se, peak_freqs, n_nodes_shell

# Compute for each shell
shell_se = []
shell_peaks = []
shell_spectra = []
shell_counts = []

print(f"\n{'Shell':>5} {'Radius':>10} {'Nodes':>6} {'SE':>8} {'Peaks':>6}")
print("-" * 45)

for s in range(N_SHELLS):
    r_center = (shell_edges[s] + shell_edges[s+1]) / 2
    spectrum, se, peaks, count = compute_shell_spectrum(vh, shell_ids, s)
    
    shell_se.append(se if se is not None else 0.0)
    shell_peaks.append(peaks if peaks is not None else np.array([]))
    shell_spectra.append(spectrum)
    shell_counts.append(count)
    
    n_peaks = len(peaks) if peaks is not None else 0
    if count > 0:
        print(f"{s:5d} {r_center:10.2f} {count:6d} {se:8.4f} {n_peaks:6d}")
    else:
        print(f"{s:5d} {r_center:10.2f} {count:6d}      ---    ---")

# ============================================================
# INVARIANT TEST: Does SE ≈ 0.495 hold per shell?
# ============================================================
print("\n" + "=" * 70)
print("INVARIANT TEST: SE PER SHELL")
print("=" * 70)

valid_se = [(s, shell_se[s]) for s in range(N_SHELLS) if shell_counts[s] > 2]
se_values_only = [se for _, se in valid_se]
global_se = np.mean(se_values_only)
se_std = np.std(se_values_only)

print(f"\nGlobal mean SE across shells: {global_se:.4f} ± {se_std:.4f}")
print(f"Target invariant: 0.495")
print(f"Deviation from target: {abs(global_se - 0.495):.4f}")
print(f"\nPer-shell range: [{min(se_values_only):.4f}, {max(se_values_only):.4f}]")

# Check how many shells individually fall near 0.495
near_invariant = sum(1 for se in se_values_only if abs(se - 0.495) < 0.05)
print(f"Shells within ±0.05 of 0.495: {near_invariant}/{len(se_values_only)}")

# ============================================================
# RADIAL PROFILES (amplitude + SE)
# ============================================================
bin_centers = np.array([(shell_edges[s] + shell_edges[s+1])/2 for s in range(N_SHELLS)])
rms_profile = np.zeros(N_SHELLS)
for s in range(N_SHELLS):
    mask = (shell_ids == s)
    if mask.sum() == 0:
        continue
    dev = vh[mask] - 1.0
    rms_profile[s] = np.sqrt(np.mean(dev**2))

# Power law fit for amplitude
valid_amp = (rms_profile > 0) & (bin_centers > 0)
if valid_amp.sum() >= 3:
    log_r = np.log(bin_centers[valid_amp])
    log_a = np.log(rms_profile[valid_amp])
    alpha_fit, _ = np.polyfit(log_r, log_a, 1)
    print(f"\nAmplitude power law exponent: α = {alpha_fit:.3f}")
else:
    alpha_fit = np.nan

# ============================================================
# CROSS-SHELL HARMONIC CORRELATION
# ============================================================
print("\n" + "=" * 70)
print("CROSS-SHELL HARMONIC ANALYSIS")
print("=" * 70)

# Compare peak frequencies between adjacent shells
print(f"\n{'Shell Pair':>12} {'Shared Peaks':>14} {'Unique Inner':>14} {'Unique Outer':>14}")
print("-" * 58)

for s in range(N_SHELLS - 1):
    if len(shell_peaks[s]) == 0 or len(shell_peaks[s+1]) == 0:
        continue
    # Find shared peaks (within tolerance)
    tol = 2.0 / TIME_STEPS  # 2 frequency bins tolerance
    shared = 0
    for p1 in shell_peaks[s]:
        for p2 in shell_peaks[s+1]:
            if abs(p1 - p2) < tol:
                shared += 1
                break
    unique_inner = len(shell_peaks[s]) - shared
    unique_outer = len(shell_peaks[s+1]) - shared
    print(f"  {s:2d}-{s+1:2d}      {shared:14d} {unique_inner:14d} {unique_outer:14d}")

# ============================================================
# PLOTS
# ============================================================

# Plot 1: SE profile vs radius
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
valid = np.array(shell_counts) > 0
ax.plot(bin_centers[valid], np.array(shell_se)[valid], 'o-', color='navy', linewidth=2)
ax.axhline(y=0.495, color='red', linestyle='--', alpha=0.7, label='SE ≈ 0.495 invariant')
ax.set_xlabel('Radius')
ax.set_ylabel('Spectral Entropy')
ax.set_title('SE Profile vs Radius')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: RMS amplitude profile
ax = axes[0, 1]
ax.plot(bin_centers[valid], rms_profile[valid], 's-', color='darkred', linewidth=2)
if not np.isnan(alpha_fit):
    r_fit = bin_centers[valid]
    ax.plot(r_fit, np.exp(np.polyval([alpha_fit, np.log(rms_profile[valid][0]) - alpha_fit * np.log(r_fit[0])], np.log(r_fit))),
            '--', color='gray', alpha=0.7, label=f'α = {alpha_fit:.3f}')
ax.set_xlabel('Radius')
ax.set_ylabel('RMS Amplitude')
ax.set_title('Amplitude Profile vs Radius')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Spectral waterfall (power spectra stacked by shell)
ax = axes[1, 0]
max_freq_idx = min(200, TIME_STEPS // 2)  # Show first 200 frequency bins
for s in range(N_SHELLS):
    if shell_spectra[s] is not None and shell_counts[s] > 0:
        spec = shell_spectra[s][:max_freq_idx]
        if np.max(spec) > 0:
            spec_norm = spec / np.max(spec)
            ax.plot(range(max_freq_idx), spec_norm + s * 0.3, alpha=0.7, linewidth=0.8)
ax.set_xlabel('Frequency bin')
ax.set_ylabel('Shell (offset)')
ax.set_title('Spectral Waterfall by Shell')
ax.grid(True, alpha=0.3)

# Plot 4: SE vs amplitude scatter
ax = axes[1, 1]
valid_both = (np.array(shell_counts) > 0)
ax.scatter(rms_profile[valid_both], np.array(shell_se)[valid_both], 
           c=bin_centers[valid_both], cmap='viridis', s=80, edgecolors='black', linewidth=0.5)
ax.set_xlabel('RMS Amplitude')
ax.set_ylabel('Spectral Entropy')
ax.set_title('SE vs Amplitude (colored by radius)')
cb = plt.colorbar(ax.collections[0], ax=ax)
cb.set_label('Radius')
ax.grid(True, alpha=0.3)

plt.suptitle(f'Galactic Disk F_c — Spectral Analysis\nM={CENTRAL_MASS}, N={N_NODES}, Steps={TIME_STEPS}, Self-Enrichment',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('spectral_results/spectral_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nDashboard saved: spectral_results/spectral_dashboard.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Amplitude exponent: α = {alpha_fit:.4f}")
print(f"Global SE: {global_se:.4f} ± {se_std:.4f}")
print(f"Invariant target: 0.495")
print(f"Shells near invariant (±0.05): {near_invariant}/{len(se_values_only)}")
print(f"SE range across shells: [{min(se_values_only):.4f}, {max(se_values_only):.4f}]")
print(f"\nKey question: Does SE hold independently per shell?")
if se_std < 0.05:
    print(f"  YES — SE is consistent across shells (σ = {se_std:.4f})")
    print(f"  Each radial zone converges to the same spectral equilibrium.")
else:
    print(f"  PARTIAL — SE varies across shells (σ = {se_std:.4f})")
    print(f"  The invariant may be a global rather than local property.")
print(f"\nKey question: Does the SE profile shape match σ(r)?")
print(f"  The SE profile is plotted for visual comparison with")
print(f"  observed velocity dispersion profiles.")
print("=" * 70)
