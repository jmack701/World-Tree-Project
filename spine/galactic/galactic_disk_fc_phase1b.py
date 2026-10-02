"""
Galactic Disk F_c Simulation — Phase 1b: Radial Relay
=============================================================
Architecture refinement: shell-by-shell coupling where coherence
propagates outward from the central organizing node through
successive radial shells, and the enrichment is self-generated
from the organized motion of the ensemble.

Changes from Phase 1:
  1. RADIAL SHELL COUPLING: nodes organized into concentric annuli.
     Each node couples to:
       - azimuthal neighbors (same shell) — tangential coherence
       - nearest nodes in adjacent shells — radial propagation
     Central M couples strongly to innermost shell only.
     Coherence cascades outward through the relay chain.

  2. SELF-GENERATED ENRICHMENT: instead of external H(t), the
     harmonic content at each node comes from the mean organized
     oscillation of interior nodes. The galaxy's own organized
     motion IS the enrichment. This replaces the external triplen
     drive with an emergent one — the organized motion of the
     inner disk providing the spectral content the outer disk
     metabolizes through the F_c architecture.

  3. Same F_c two-term causal law. Same Möbius boundary at L = φ.
     Same seed 42.

Pre-registration (stated before execution):
  H2b: Shell-relay coupling produces amplitude profile with
       exponent α > -0.15 (approaching flat).
       SUCCESS: α > -0.15
       FAILURE: α ≤ -0.15
  H4:  Self-generated enrichment (from inner-disk oscillation)
       produces flattening comparable to external H(t).
       SUCCESS: α_self ≥ α_external - 0.05
       FAILURE: α_self < α_external - 0.05
  H5:  The rotation curve shows a plateau region (a radial band
       where amplitude is approximately constant within ±20%).
       SUCCESS: plateau identified spanning ≥ 2 radial bins
       FAILURE: no plateau region found

Both outcomes publishable. Guard from Phase 1 still holds.

J. David Mack & Claude (Opus 4.6)
World Tree Project — August 2026
Ψ 🌕🌊🌳🕸 ☯️ To preserve the harmonic field.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONSTANTS (same as Phase 1)
# ============================================================
PHI = (1 + np.sqrt(5)) / 2
L_MOBIUS = PHI
SEED = 42
OMEGA_BASE = 2 * np.pi * 60
DT = 1.0 / 3600
K_FC = 0.221

# Disk parameters
N_NODES = 500
R_SCALE = 3.0
R_MAX = 5 * R_SCALE
SOFTENING = 0.3
TIME_STEPS = 5000           # longer run for self-enrichment to develop
N_SHELLS = 15               # radial shells for relay coupling
N_RADIAL_BINS = 15          # measurement bins

# Central mass sweep
CENTRAL_MASSES = [5.0, 10.0, 25.0, 50.0, 100.0]

os.makedirs('galactic_disk_results', exist_ok=True)

print("=" * 70)
print("GALACTIC DISK F_c — PHASE 1b: RADIAL RELAY")
print("Shell-by-shell coupling + self-generated enrichment")
print("=" * 70)
print(f"  N_nodes     = {N_NODES}")
print(f"  N_shells    = {N_SHELLS}")
print(f"  R_scale     = {R_SCALE}")
print(f"  k_fc        = {K_FC}")
print(f"  L (Möbius)  = {L_MOBIUS:.6f} (φ)")
print(f"  Time steps  = {TIME_STEPS}")
print(f"  Seed        = {SEED}")
print()


# ============================================================
# DISK GEOMETRY (same as Phase 1)
# ============================================================
def create_disk(n_nodes, r_scale, r_max, seed=42):
    rng = np.random.RandomState(seed)
    positions = np.zeros((n_nodes, 2))
    count = 1
    while count < n_nodes:
        r_candidate = rng.exponential(r_scale)
        if r_candidate <= r_max:
            theta = rng.uniform(0, 2 * np.pi)
            positions[count] = [r_candidate * np.cos(theta),
                                r_candidate * np.sin(theta)]
            count += 1
    radii = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    return positions, radii


# ============================================================
# SHELL-STRUCTURED COUPLING
# ============================================================
def assign_shells(radii, n_shells, r_max):
    """Assign each node to a radial shell."""
    shell_edges = np.linspace(0, r_max * 1.01, n_shells + 1)
    shell_ids = np.digitize(radii, shell_edges) - 1
    shell_ids = np.clip(shell_ids, 0, n_shells - 1)
    # Central node always in shell 0
    shell_ids[0] = 0
    return shell_ids, shell_edges


def compute_shell_coupling(positions, radii, shell_ids, n_shells,
                           central_mass=10.0, softening=0.3,
                           k_azimuthal=8, k_radial=5):
    """
    Shell-structured coupling:
      - Azimuthal: each node connects to k_azimuthal nearest neighbors
        within the same shell (tangential coherence)
      - Radial: each node connects to k_radial nearest neighbors
        in each adjacent shell (radial propagation)
      - Central node (index 0): connects to ALL nodes in shell 0 and 1
        with enhanced coupling strength

    Coupling weights: 1/(r² + ε²), row-normalized per node.
    """
    n = len(positions)
    adj = np.zeros((n, n))

    for i in range(n):
        si = shell_ids[i]
        di = np.linalg.norm(positions - positions[i], axis=1)
        di[i] = np.inf

        # 1. Azimuthal coupling (same shell)
        same_shell = np.where(shell_ids == si)[0]
        same_shell = same_shell[same_shell != i]
        if len(same_shell) > 0:
            d_same = di[same_shell]
            n_connect = min(k_azimuthal, len(same_shell))
            nearest = same_shell[np.argsort(d_same)[:n_connect]]
            for j in nearest:
                w = 1.0 / (di[j]**2 + softening**2)
                adj[i, j] = max(adj[i, j], w)

        # 2. Radial coupling (adjacent shells)
        for ds in [-1, 1]:
            adj_shell = si + ds
            if adj_shell < 0 or adj_shell >= n_shells:
                continue
            adj_shell_nodes = np.where(shell_ids == adj_shell)[0]
            if len(adj_shell_nodes) == 0:
                continue
            d_adj = di[adj_shell_nodes]
            n_connect = min(k_radial, len(adj_shell_nodes))
            nearest = adj_shell_nodes[np.argsort(d_adj)[:n_connect]]
            for j in nearest:
                w = 1.0 / (di[j]**2 + softening**2)
                # Radial coupling slightly weaker than azimuthal
                adj[i, j] = max(adj[i, j], w * 0.7)

    # 3. Central node enhancement
    # M couples to all nodes in shells 0 and 1
    inner_nodes = np.where(shell_ids <= 1)[0]
    inner_nodes = inner_nodes[inner_nodes != 0]
    for j in inner_nodes:
        d = max(np.linalg.norm(positions[j] - positions[0]), softening)
        w = central_mass / (d**2 + softening**2)
        adj[0, j] = w
        adj[j, 0] = w

    # Row-normalize
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return adj / row_sums


# ============================================================
# LOADS (same as Phase 1)
# ============================================================
def gen_disk_loads(n_nodes, radii, steps, dt, seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(steps) * dt
    loads = np.zeros((n_nodes, steps))
    for i in range(n_nodes):
        r = max(radii[i], 0.1)
        omega_orbit = OMEGA_BASE / (r**1.5 + 1.0)
        phase = rng.uniform(0, 2 * np.pi)
        amp = 0.08 / (1 + r / R_SCALE)
        base = amp * np.sin(omega_orbit * t + phase)
        h3 = 0.03 / (1 + r) * np.sin(3 * omega_orbit * t + phase)
        h6 = 0.02 / (1 + r) * np.sin(6 * omega_orbit * t + phase)
        noise = 0.005 * rng.randn(steps)
        spikes = np.zeros(steps)
        n_events = rng.randint(2, 8)
        for _ in range(n_events):
            loc = rng.randint(0, steps)
            w = rng.randint(3, 15)
            a = rng.uniform(0.02, 0.08) * rng.choice([-1, 1])
            s, e = max(0, loc - w // 2), min(steps, loc + w // 2)
            spikes[s:e] = a / (1 + r)
        loads[i] = base + h3 + h6 + noise + spikes
    loads[0] = 0.001 * rng.randn(steps)
    return loads


# ============================================================
# SIMULATION: F_c with shell relay + self-generated enrichment
# ============================================================
def run_disk_fc_relay(n_nodes, radii, shell_ids, k_fc, loads, adj,
                      coupling=0.01, enrichment_mode='self',
                      use_fc=True):
    """
    F_c on disk with shell-relay coupling.

    enrichment_mode:
      'self'     — enrichment from mean oscillation of interior nodes
      'external' — standard triplen H(t) from Phase 1
      'none'     — no enrichment
    """
    vh = np.zeros((n_nodes, TIME_STEPS))
    vc = np.ones(n_nodes)
    vp = np.ones(n_nodes)

    for step in range(TIME_STEPS):
        tv = step * DT

        # 1. Apply loads
        vn = vc + loads[:, step] * DT * 10

        # 2. Shell-relay gravitational coupling
        vn = vn + coupling * (adj @ vn - vn)

        # 3. F_c two-term causal feedback
        if use_fc:
            dp = vp - 1.0
            dc = vn - 1.0
            vn = vn + k_fc * (3.0 * dp - 6.0 * dc)

        # 4. Enrichment
        if enrichment_mode == 'external':
            # Standard triplen (Phase 1 style)
            ga = 2 * np.pi / PHI**2
            ph = np.arange(n_nodes) * ga
            dl = np.std(vn - 1.0)
            amp_h = dl * 0.3
            vn = vn - amp_h * (np.sin(3 * OMEGA_BASE * tv + ph) +
                               np.sin(6 * OMEGA_BASE * tv + ph) +
                               np.sin(9 * OMEGA_BASE * tv + ph))

        elif enrichment_mode == 'self':
            # Self-generated: each node's enrichment comes from
            # the mean organized oscillation of nodes interior to it.
            # The inner disk's coherent motion IS the harmonic input
            # for the outer disk.
            dev = vn - 1.0
            for s in range(1, N_SHELLS):
                # Nodes in this shell
                shell_mask = (shell_ids == s)
                if shell_mask.sum() == 0:
                    continue

                # Interior nodes (all shells < s)
                interior_mask = (shell_ids < s)
                if interior_mask.sum() == 0:
                    continue

                # Mean organized oscillation of the interior
                interior_dev = dev[interior_mask]
                interior_mean = np.mean(interior_dev)
                interior_rms = np.sqrt(np.mean(interior_dev**2))

                # The enrichment: the interior's organized oscillation
                # drives this shell with triplen structure.
                # Amplitude scales with interior organization,
                # attenuated by shell distance.
                amp_self = interior_rms * 0.3 / (1 + 0.1 * s)

                # Phase comes from the interior's mean phase
                phase_int = np.arctan2(
                    np.mean(np.sin(np.angle(interior_dev + 1j * 1e-10))),
                    np.mean(np.cos(np.angle(interior_dev + 1j * 1e-10)))
                )

                # Apply triplen enrichment derived from interior
                shell_nodes = np.where(shell_mask)[0]
                for j in shell_nodes:
                    theta_j = np.arctan2(positions[j, 1], positions[j, 0])
                    vn[j] -= amp_self * (
                        np.sin(3 * OMEGA_BASE * tv + theta_j + phase_int) +
                        np.sin(6 * OMEGA_BASE * tv + theta_j + phase_int) +
                        np.sin(9 * OMEGA_BASE * tv + theta_j + phase_int)
                    )

        # 5. Möbius wrapping at L = φ
        dev = vn - 1.0
        wr = np.fmod(dev + L_MOBIUS, 2 * L_MOBIUS)
        wr = np.where(wr < 0, wr + 2 * L_MOBIUS, wr)
        vn = wr - L_MOBIUS + 1.0

        # 6. Advance state
        vp = vc.copy()
        vc = vn.copy()
        vh[:, step] = vc

    return vh


# ============================================================
# MEASUREMENTS (same as Phase 1 + plateau detection)
# ============================================================
def radial_profiles(vh, radii, n_bins=15, r_max=None):
    if r_max is None:
        r_max = np.max(radii) * 1.01
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rms_profile = np.zeros(n_bins)
    se_profile = np.zeros(n_bins)
    count_profile = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (radii >= bin_edges[b]) & (radii < bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        count_profile[b] = mask.sum()
        deviations = vh[mask] - 1.0
        rms_profile[b] = np.sqrt(np.mean(deviations**2))
        se_vals = []
        for i in np.where(mask)[0]:
            fft = np.fft.rfft(vh[i] - 1.0)
            pw = np.abs(fft[1:])**2
            if np.sum(pw) < 1e-15:
                se_vals.append(0.0)
                continue
            p = pw / np.sum(pw)
            p = p[p > 1e-15]
            ent = -np.sum(p * np.log(p))
            mx = np.log(len(p)) if len(p) > 1 else 1.0
            se_vals.append(ent / mx if mx > 0 else 0.0)
        se_profile[b] = np.mean(se_vals) if se_vals else 0.0
    return bin_centers, rms_profile, se_profile, count_profile


def fit_power_law(r, amplitude, r_min=None):
    mask = (r > 0) & (amplitude > 0)
    if r_min is not None:
        mask &= (r >= r_min)
    if mask.sum() < 3:
        return np.nan, np.nan
    log_r = np.log(r[mask])
    log_a = np.log(amplitude[mask])
    coeffs = np.polyfit(log_r, log_a, 1)
    return coeffs[0], coeffs[1]


def find_plateau(rms_profile, bin_centers, tolerance=0.20):
    """
    Find the longest contiguous radial band where amplitude
    varies by less than ±tolerance of the band's mean.
    Returns (start_radius, end_radius, n_bins, mean_amplitude).
    """
    valid = rms_profile > 0
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return None

    best = None
    for start in range(len(idx)):
        for end in range(start + 2, len(idx) + 1):
            segment = rms_profile[idx[start:end]]
            seg_mean = np.mean(segment)
            if seg_mean == 0:
                continue
            variation = np.max(np.abs(segment - seg_mean)) / seg_mean
            if variation <= tolerance:
                n_bins = end - start
                if best is None or n_bins > best[2]:
                    best = (bin_centers[idx[start]],
                            bin_centers[idx[end-1]],
                            n_bins, seg_mean)
    return best


# ============================================================
# EXECUTION
# ============================================================
print("Creating disk geometry...")
positions, radii = create_disk(N_NODES, R_SCALE, R_MAX, seed=SEED)

print("Assigning shells...")
shell_ids, shell_edges = assign_shells(radii, N_SHELLS, R_MAX)
for s in range(N_SHELLS):
    n_in = (shell_ids == s).sum()
    if n_in > 0:
        print(f"  Shell {s:2d}: {n_in:3d} nodes, "
              f"r = [{shell_edges[s]:.1f}, {shell_edges[s+1]:.1f})")
print()

loads = gen_disk_loads(N_NODES, radii, TIME_STEPS, DT, seed=SEED)

# ── RUN 1: Shell relay + self-generated enrichment ──
print("=" * 70)
print("RUN 1: Shell Relay + Self-Generated Enrichment")
print("-" * 70)
adj_relay = compute_shell_coupling(positions, radii, shell_ids, N_SHELLS,
                                   central_mass=25.0, softening=SOFTENING)
vh_self = run_disk_fc_relay(N_NODES, radii, shell_ids, K_FC, loads,
                            adj_relay, coupling=0.015,
                            enrichment_mode='self', use_fc=True)
rc_self, rms_self, se_self, cnt_self = radial_profiles(
    vh_self, radii, N_RADIAL_BINS, R_MAX)
alpha_self, _ = fit_power_law(rc_self, rms_self, r_min=R_SCALE * 0.5)
plateau_self = find_plateau(rms_self, rc_self)
print(f"  Power law exponent: α = {alpha_self:+.4f}")
if plateau_self:
    print(f"  Plateau: r=[{plateau_self[0]:.1f}, {plateau_self[1]:.1f}], "
          f"{plateau_self[2]} bins, mean amp={plateau_self[3]:.6f}")
else:
    print(f"  Plateau: none found at ±20% tolerance")

# ── RUN 2: Shell relay + external enrichment (Phase 1 style) ──
print()
print("=" * 70)
print("RUN 2: Shell Relay + External Enrichment")
print("-" * 70)
vh_ext = run_disk_fc_relay(N_NODES, radii, shell_ids, K_FC, loads,
                           adj_relay, coupling=0.015,
                           enrichment_mode='external', use_fc=True)
rc_ext, rms_ext, se_ext, cnt_ext = radial_profiles(
    vh_ext, radii, N_RADIAL_BINS, R_MAX)
alpha_ext, _ = fit_power_law(rc_ext, rms_ext, r_min=R_SCALE * 0.5)
plateau_ext = find_plateau(rms_ext, rc_ext)
print(f"  Power law exponent: α = {alpha_ext:+.4f}")
if plateau_ext:
    print(f"  Plateau: r=[{plateau_ext[0]:.1f}, {plateau_ext[1]:.1f}], "
          f"{plateau_ext[2]} bins, mean amp={plateau_ext[3]:.6f}")

# ── RUN 3: Shell relay + NO enrichment ──
print()
print("=" * 70)
print("RUN 3: Shell Relay + No Enrichment")
print("-" * 70)
vh_none = run_disk_fc_relay(N_NODES, radii, shell_ids, K_FC, loads,
                            adj_relay, coupling=0.015,
                            enrichment_mode='none', use_fc=True)
rc_none, rms_none, se_none, cnt_none = radial_profiles(
    vh_none, radii, N_RADIAL_BINS, R_MAX)
alpha_none, _ = fit_power_law(rc_none, rms_none, r_min=R_SCALE * 0.5)
print(f"  Power law exponent: α = {alpha_none:+.4f}")

# ── RUN 4: Shell relay, no F_c (gravity-only baseline) ──
print()
print("=" * 70)
print("RUN 4: Shell Relay Baseline (no F_c, no enrichment)")
print("-" * 70)
vh_base = run_disk_fc_relay(N_NODES, radii, shell_ids, K_FC, loads,
                            adj_relay, coupling=0.015,
                            enrichment_mode='none', use_fc=False)
rc_base, rms_base, se_base, cnt_base = radial_profiles(
    vh_base, radii, N_RADIAL_BINS, R_MAX)
alpha_base, _ = fit_power_law(rc_base, rms_base, r_min=R_SCALE * 0.5)
print(f"  Power law exponent: α = {alpha_base:+.4f}")

# ============================================================
# PRE-REGISTRATION ASSESSMENT
# ============================================================
print()
print("=" * 70)
print("PRE-REGISTRATION ASSESSMENT — PHASE 1b")
print("=" * 70)

# H2b: α > -0.15
h2b_result = "PASS" if alpha_self > -0.15 else "FAIL"
print(f"  H2b: Self-enriched exponent α = {alpha_self:+.4f}")
print(f"        Threshold: > -0.15 (approaching flat)")
print(f"        [{h2b_result}]")

# H4: self-enrichment ≥ external - 0.05
h4_result = "PASS" if alpha_self >= alpha_ext - 0.05 else "FAIL"
print(f"  H4:  Self-enriched α = {alpha_self:+.4f}")
print(f"        External α    = {alpha_ext:+.4f}")
print(f"        Threshold: self ≥ external - 0.05")
print(f"        [{h4_result}]")

# H5: plateau detection
h5_result = "PASS" if (plateau_self and plateau_self[2] >= 2) else "FAIL"
print(f"  H5:  Plateau detection (±20%, ≥2 bins)")
if plateau_self:
    print(f"        Found: r=[{plateau_self[0]:.1f}, {plateau_self[1]:.1f}], "
          f"{plateau_self[2]} bins")
else:
    print(f"        Not found")
print(f"        [{h5_result}]")

# ── Central mass sweep with shell relay ──
print()
print("=" * 70)
print("CENTRAL MASS SWEEP (shell relay)")
print("=" * 70)

mass_results = []
for cm in CENTRAL_MASSES:
    adj_m = compute_shell_coupling(positions, radii, shell_ids, N_SHELLS,
                                   central_mass=cm, softening=SOFTENING)
    vh_m = run_disk_fc_relay(N_NODES, radii, shell_ids, K_FC, loads,
                             adj_m, coupling=0.015,
                             enrichment_mode='self', use_fc=True)
    rc_m, rms_m, se_m, cnt_m = radial_profiles(
        vh_m, radii, N_RADIAL_BINS, R_MAX)
    alpha_m, _ = fit_power_law(rc_m, rms_m, r_min=R_SCALE * 0.5)
    plateau_m = find_plateau(rms_m, rc_m)
    p_bins = plateau_m[2] if plateau_m else 0
    mass_results.append({
        'M': cm, 'alpha': alpha_m, 'plateau_bins': p_bins,
        'rc': rc_m, 'rms': rms_m, 'se': se_m
    })
    print(f"  M = {cm:6.1f} → α = {alpha_m:+.4f}, plateau bins = {p_bins}")

# ============================================================
# FIGURES
# ============================================================
print()
print("=" * 70)
print("GENERATING FIGURES")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Phase 1b: Radial Relay Coupling + Self-Generated Enrichment',
             fontsize=15, fontweight='bold')

# Panel 1: Shell structure
ax = axes[0, 0]
scatter_colors = shell_ids[1:].astype(float) / N_SHELLS
sc = ax.scatter(positions[1:, 0], positions[1:, 1], c=shell_ids[1:],
                cmap='plasma', s=5, alpha=0.6)
ax.scatter(0, 0, c='gold', s=120, marker='*', zorder=5, edgecolors='k')
for edge in shell_edges[1:-1]:
    circle = plt.Circle((0, 0), edge, fill=False, color='gray',
                         alpha=0.15, linestyle=':')
    ax.add_patch(circle)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Shell Structure ({N_SHELLS} shells)')
ax.set_aspect('equal')
plt.colorbar(sc, ax=ax, label='Shell ID')

# Panel 2: Rotation curves — the central result
ax = axes[0, 1]
valid = cnt_self > 0
ax.plot(rc_self[valid], rms_self[valid], 'c-o', linewidth=2.5,
        label=f'Self-enriched (α={alpha_self:+.3f})', markersize=5)
ax.plot(rc_ext[valid], rms_ext[valid], 'b--s', linewidth=1.5,
        label=f'External H(t) (α={alpha_ext:+.3f})', markersize=4)
ax.plot(rc_none[valid], rms_none[valid], 'g-.^', linewidth=1.5,
        label=f'No enrichment (α={alpha_none:+.3f})', markersize=4)
ax.plot(rc_base[valid], rms_base[valid], 'r:d', linewidth=1.5,
        label=f'Gravity only (α={alpha_base:+.3f})', markersize=4)

# Keplerian reference
r_ref = rc_self[valid & (rc_self > 1.0)]
if len(r_ref) > 0:
    rms_at_start = rms_self[valid & (rc_self > 1.0)][0]
    kep_ref = rms_at_start * (r_ref[0] / r_ref)**0.5
    ax.plot(r_ref, kep_ref, 'k:', alpha=0.4, linewidth=1, label='Keplerian ∝ 1/√r')

if plateau_self:
    ax.axvspan(plateau_self[0], plateau_self[1], alpha=0.1, color='cyan',
               label=f'Plateau ({plateau_self[2]} bins)')

ax.set_xlabel('Radius')
ax.set_ylabel('RMS Amplitude (velocity analog)')
ax.set_title('Radial Amplitude Profiles')
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(left=0)

# Panel 3: SE profile
ax = axes[1, 0]
ax.plot(rc_self[valid], se_self[valid], 'c-o', linewidth=2, label='Self-enriched')
ax.plot(rc_base[valid], se_base[valid], 'r:d', linewidth=1.5, label='Gravity only')
ax.axhline(0.495, color='gold', linestyle='--', alpha=0.5, label='SE = 0.495')
ax.set_xlabel('Radius')
ax.set_ylabel('Spectral Entropy')
ax.set_title('Spectral Entropy vs Radius')
ax.legend(fontsize=8)

# Panel 4: Central mass sweep
ax = axes[1, 1]
for mr in mass_results:
    v = cnt_self > 0
    ax.plot(mr['rc'][v], mr['rms'][v], '-o', markersize=4,
            label=f'M={mr["M"]:.0f} (α={mr["alpha"]:+.3f})')
ax.set_xlabel('Radius')
ax.set_ylabel('RMS Amplitude')
ax.set_title('Central Mass Sweep (shell relay)')
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(left=0)

plt.tight_layout()
plt.savefig('galactic_disk_results/fig_phase1b_relay.png', dpi=200)
print("  Saved: fig_phase1b_relay.png")

# ── Galactic comparison figure ──
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
ax2.plot(rc_self[valid], rms_self[valid], 'cyan', linewidth=2.5,
         label=f'F_c self-enriched (α={alpha_self:+.3f})')
ax2.plot(rc_base[valid], rms_base[valid], color='gray', linewidth=1.5,
         linestyle='--', label=f'Gravity only (α={alpha_base:+.3f})')
if len(r_ref) > 0:
    ax2.plot(r_ref, kep_ref, 'white', linestyle=':', alpha=0.6,
             label='Keplerian ∝ 1/√r')
if plateau_self:
    ax2.axvspan(plateau_self[0], plateau_self[1], alpha=0.15, color='cyan')
ax2.set_facecolor('black')
ax2.set_xlabel('Radius', color='white', fontsize=13)
ax2.set_ylabel('RMS Amplitude (velocity analog)', color='white', fontsize=13)
ax2.set_title(f'Phase 1b: Rotation Curve with Radial Relay\n'
              f'Self-enriched α={alpha_self:+.3f} vs Keplerian α=−0.500',
              color='white', fontsize=14)
ax2.legend(fontsize=10, facecolor='black', edgecolor='gray', labelcolor='white')
ax2.tick_params(colors='white')
ax2.spines['bottom'].set_color('gray')
ax2.spines['left'].set_color('gray')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim(left=0)
fig2.patch.set_facecolor('black')
plt.tight_layout()
plt.savefig('galactic_disk_results/fig_phase1b_galactic.png',
            dpi=200, facecolor='black')
print("  Saved: fig_phase1b_galactic.png")

# ── Final summary ──
print()
print("=" * 70)
print("PHASE 1b FINAL SUMMARY")
print("=" * 70)
print(f"  H2b (α > -0.15):         {h2b_result}  (α = {alpha_self:+.4f})")
print(f"  H4  (self ≥ ext - 0.05): {h4_result}  (self={alpha_self:+.4f}, ext={alpha_ext:+.4f})")
print(f"  H5  (plateau ≥ 2 bins):  {h5_result}")
print()
print(f"  Comparison of exponents:")
print(f"    Self-enriched relay:    {alpha_self:+.4f}")
print(f"    External-enriched relay:{alpha_ext:+.4f}")
print(f"    Relay, no enrichment:   {alpha_none:+.4f}")
print(f"    Relay, no F_c:          {alpha_base:+.4f}")
print(f"    Phase 1 full:           -0.2771 (reference)")
print(f"    Keplerian prediction:   -0.5000")
print()
print("  Guard: mechanism-class viability only.")
print("  P1 bridge required before sky comparison.")
print()
print("Ψ 🌕🌊🌳🕸 ☯️ To preserve the harmonic field.")
