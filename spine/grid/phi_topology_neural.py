"""
Phi-Topology Neural Dynamics Test — RECREATION
===============================================
Original: session script, June 28, 2026 (Math Data Analysis spiral,
/home/claude/phi_topology_neural.py — not preserved in the archive).
This recreation: July 23, 2026 (Fable spiral). If the original file is
exported from the June 28 conversation, the original supersedes this file.

PURPOSE (recovered): Test whether neuron placement topology (phi-spiral vs
random vs grid) differentiates the spectral-entropy organization produced
by static F_c feedback on a tanh neural substrate, against the chaos
invariant target SE = 0.4950.

RECOVERED FROM THE RECORD (June 28 session fragments; Spine v6 §4.4, §4.6;
Diamond Core Rewrite Spiral Continuity):
  - Three placements: PHI-SPIRAL (golden angle), RANDOM, GRID
  - tanh activation substrate (Spine §4.4)
  - Static F_c feedback through k-nearest-neighbor coupling:
      3·(previous state) − 6·(current state) + 9·(harmonic enrichment)
  - Recorded findings: baseline SE ≈ 0.805 without feedback;
    ≈ 0.53 under feedback (recorded readings 0.540 / 0.534);
    invariant 0.4950 NOT reached; placements do NOT differentiate
  - Closing decision logic (recovered near-verbatim; see bottom)
  - Per-topology multi-run results structure

RECONSTRUCTION CHOICES (not preserved in the record; flagged, tunable):
  - N_NODES = 50, T_STEPS = 2000, TRANSIENT = 200
  - Coupling: kn = 5 nearest neighbors, weights 1/(1+d), row-normalized,
    leak RHO = 0.9 on the coupled term (family convention, grid scripts)
  - Drive: i.i.d. Gaussian noise, sigma = 0.3, per node per step
  - F_c gain K_FC = 0.165 (field value; family convention)
  - Enrichment H(t): amplitude 0.3, phases = node index × golden angle
    (identical phase rule for all three placements, for comparability),
    time scaling tau = 0.1·step (family convention)
  - Seeds: 42, 137, 369, 7, 13 (5 runs per condition)
  - PRIMARY SE metric: per-node temporal FFT normalized Shannon entropy,
    averaged over nodes (grid-family convention — the 0.4950 target is
    that family's invariant). SECONDARY: covariance-eigenvalue SE across
    units (Diamond-Core convention), reported for completeness.
  - First branch of the decision logic (condition text not preserved):
    reconstructed as phi within 0.02 of target while others are not.

STATUS: recreation for regenerability. Exact values are parameterization-
dependent; the June 28 readings above are the record. This file preserves
protocol and decision logic.

J. David Mack & Claude
World Tree Project — recreation July 2026 (original June 2026)
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
GOLDEN_ANGLE = 2 * np.pi / PHI**2
TARGET_SE = 0.4950

N_NODES = 50
T_STEPS = 2000
TRANSIENT = 200
KN = 5
RHO = 0.9
DRIVE_SIGMA = 0.3
K_FC = 0.165
H_AMP = 0.3
TAU_SCALE = 0.1
SEEDS = [42, 137, 369, 7, 13]

print("=" * 65)
print("PHI-TOPOLOGY NEURAL DYNAMICS TEST  (RECREATION of June 28 script)")
print("=" * 65)
print(f"  Target: SE = {TARGET_SE:.4f}  (chaos invariant, grid family)")
print(f"  Recorded (June 28): baseline ~0.805; feedback ~0.53 (0.540/0.534)")
print(f"  N={N_NODES}, T={T_STEPS}, k_fc={K_FC}, seeds={SEEDS}")
print()


# ============================================================
# PLACEMENTS
# ============================================================
def phi_positions(n):
    pos = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt(i + 1) * 0.5
        pos[i] = [r * np.cos(i * GOLDEN_ANGLE), r * np.sin(i * GOLDEN_ANGLE)]
    return pos


def random_positions(n, rng):
    rmax = np.sqrt(n) * 0.5
    r = rmax * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)


def grid_positions(n):
    side = int(np.ceil(np.sqrt(n)))
    extent = np.sqrt(n) * 0.5
    xs = np.linspace(-extent, extent, side)
    ys = np.linspace(-extent, extent, side)
    pts = [(x, y) for y in ys for x in xs]
    return np.array(pts[:n])


def compute_adj(pos, kn=KN):
    n = len(pos)
    adj = np.zeros((n, n))
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:kn]:
            w = 1.0 / (1.0 + d[j])
            adj[i, j] = w
            adj[j, i] = w
    rs = adj.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return adj / rs


# ============================================================
# DYNAMICS
# ============================================================
def run_dynamics(adj, seed, use_fc):
    rng = np.random.default_rng(seed)
    n = adj.shape[0]
    phases = np.arange(n) * GOLDEN_ANGLE
    h = np.zeros(n)
    h_prev = np.zeros(n)
    traj = np.zeros((n, T_STEPS))
    for t in range(T_STEPS):
        tau = t * TAU_SCALE
        drive = rng.normal(0.0, DRIVE_SIGMA, n)
        pre = RHO * (adj @ h) + drive
        if use_fc:
            H = H_AMP * (np.sin(3 * tau + phases)
                         + np.sin(6 * tau + phases)
                         + np.sin(9 * tau + phases))
            pre = pre + K_FC * (3.0 * h_prev - 6.0 * h + 9.0 * H)
        h_next = np.tanh(pre)
        h_prev = h
        h = h_next
        traj[:, t] = h
    return traj[:, TRANSIENT:]


# ============================================================
# METRICS
# ============================================================
def se_temporal(traj):
    """Per-node temporal FFT normalized Shannon entropy, node-averaged
    (grid-family convention; the 0.4950 target lives in this metric)."""
    vals = []
    for i in range(traj.shape[0]):
        fft = np.fft.rfft(traj[i])
        pw = np.abs(fft[1:]) ** 2
        s = pw.sum()
        if s < 1e-12:
            vals.append(0.0)
            continue
        p = pw / s
        p = p[p > 1e-12]
        ent = -np.sum(p * np.log(p))
        mx = np.log(len(p)) if len(p) > 1 else 1.0
        vals.append(ent / mx if mx > 0 else 0.0)
    return float(np.mean(vals))


def se_covariance(traj):
    """Covariance-eigenvalue SE across units (Diamond-Core convention),
    reported as the secondary metric."""
    cov = np.cov(traj)
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 1e-12]
    if len(ev) == 0:
        return 0.0
    ev = ev / ev.sum()
    ent = -np.sum(ev * np.log(ev + 1e-12))
    mx = np.log(len(ev))
    return float(ent / mx) if mx > 0 else 0.0


# ============================================================
# EXPERIMENT
# ============================================================
topo_rng = np.random.default_rng(42)
placements = {
    'PHI-SPIRAL': phi_positions(N_NODES),
    'RANDOM': random_positions(N_NODES, topo_rng),
    'GRID': grid_positions(N_NODES),
}

results = {name: [] for name in placements}
baselines = {name: [] for name in placements}

for name, pos in placements.items():
    adj = compute_adj(pos)
    print(f"{name}")
    for seed in SEEDS:
        traj_b = run_dynamics(adj, seed, use_fc=False)
        traj_f = run_dynamics(adj, seed, use_fc=True)
        se_b = se_temporal(traj_b)
        se_f = se_temporal(traj_f)
        cov_f = se_covariance(traj_f)
        baselines[name].append({'se': se_b})
        results[name].append({'se': se_f, 'se_cov': cov_f})
        print(f"  seed {seed:>3}: baseline SE={se_b:.4f}   "
              f"feedback SE={se_f:.4f}   (cov SE={cov_f:.4f})")
    mb = np.mean([r['se'] for r in baselines[name]])
    mf = np.mean([r['se'] for r in results[name]])
    print(f"  mean: baseline {mb:.4f} -> feedback {mf:.4f}")
    print()

# ============================================================
# DECISION LOGIC (recovered near-verbatim from the June 28 record;
# first branch condition reconstructed — see header)
# ============================================================
mean_phi = np.mean([r['se'] for r in results['PHI-SPIRAL']])
mean_rand = np.mean([r['se'] for r in results['RANDOM']])
mean_grid = np.mean([r['se'] for r in results['GRID']])
phi_gap = abs(mean_phi - TARGET_SE)
others_gap = min(abs(mean_rand - TARGET_SE), abs(mean_grid - TARGET_SE))

print("=" * 65)
if phi_gap < 0.02 and others_gap >= 0.02:
    print("  The topology carries the invariant.")
    print("  Phi-based neural architecture is structurally distinct.")
elif phi_gap < 0.02:
    print("  PHI-SPIRAL achieves SE near 0.495.")
    print("  But other topologies may also achieve it.")
    print("  The F_c feedback may matter more than the topology.")
else:
    print("  No topology achieved SE near 0.495.")
    print("  Neural activations (tanh nonlinearity) may resist")
    print("  the harmonic structure the invariant requires.")
    print("  The substrate matters.")

print()
print(f"  Target:  SE = {TARGET_SE:.4f} (chaos invariant)")
print(f"  Phi:     SE = {mean_phi:.4f}")
print(f"  Random:  SE = {mean_rand:.4f}")
print(f"  Grid:    SE = {mean_grid:.4f}")
print()
