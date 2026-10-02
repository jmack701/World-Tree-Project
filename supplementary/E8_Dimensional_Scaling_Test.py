"""
Ψ E8 DIMENSIONAL SCALING TEST
═══════════════════════════════════════════════════════════
Does the F_c invariant survive on higher-dimensional phi-geometry?
Extends phi_grid_showcase.py architecture to:
  Test 1: 600-cell (4D, 120 vertices)
  Test 2: E8 root system (8D, 240 vectors)
  Control: 2D phi-spiral (100 nodes, confirmed baseline)

Same coupling, loads, F_c feedback, and measurement as the
working energy grid — only the node placement changes.

J. David Mack & Claude (Opus 4.6)
World Tree Project · July 2026
"""

import numpy as np
import math
import itertools

PHI = (1 + np.sqrt(5)) / 2
OMEGA_BASE = 2 * np.pi * 60
TIME_STEPS = 2000
DT = 1.0 / 3600
L_MOBIUS = PHI
V_NOMINAL = 1.0
SEED = 42

# ═══════════════════════════════════════════════════════════
# VERTEX CONSTRUCTORS
# ═══════════════════════════════════════════════════════════

def build_phi_spiral_2d(n):
    """2D phi-spiral — the confirmed baseline."""
    ga = 2 * np.pi / PHI**2
    pos = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt(i + 1) * 0.5
        pos[i] = [r * np.cos(i * ga), r * np.sin(i * ga)]
    return pos


def build_600_cell():
    """600-cell: 120 vertices in 4D with φ-based coordinates."""
    vertices = []
    
    # 8 vertices: permutations of (±1, 0, 0, 0)
    for i in range(4):
        for s in [-1, 1]:
            v = [0, 0, 0, 0]
            v[i] = s
            vertices.append(v)
    
    # 16 vertices: (±1/2, ±1/2, ±1/2, ±1/2)
    for signs in itertools.product([-1, 1], repeat=4):
        vertices.append([s * 0.5 for s in signs])
    
    # 96 vertices: even permutations of (0, ±1/2, ±φ/2, ±1/(2φ))
    base = [0, 0.5, PHI / 2, 1 / (2 * PHI)]
    # All permutations of 4 elements
    for perm in itertools.permutations(range(4)):
        # Check if even permutation
        inv = 0
        p = list(perm)
        for i in range(4):
            for j in range(i + 1, 4):
                if p[i] > p[j]:
                    inv += 1
        if inv % 2 != 0:
            continue
        permuted = [base[p[i]] for i in range(4)]
        for signs in itertools.product([-1, 1], repeat=4):
            vertices.append([permuted[i] * signs[i] for i in range(4)])
    
    # Remove duplicates
    unique = []
    seen = set()
    for v in vertices:
        key = tuple(round(x, 8) for x in v)
        if key not in seen:
            seen.add(key)
            unique.append(v)
    
    return np.array(unique)


def build_e8_roots():
    """E8 root system: 240 vectors in 8D."""
    roots = []
    
    # D8 roots: choose 2 positions from 8, assign ±1
    for pos in itertools.combinations(range(8), 2):
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                v = [0] * 8
                v[pos[0]] = s1
                v[pos[1]] = s2
                roots.append(v)
    
    # Half-spin vectors: (±1/2)^8 with even number of minus signs
    for n in range(256):
        bits = [(n >> i) & 1 for i in range(8)]
        signs = [2 * b - 1 for b in bits]
        if signs.count(-1) % 2 == 0:
            roots.append([s * 0.5 for s in signs])
    
    # Remove duplicates
    unique = []
    seen = set()
    for v in roots:
        key = tuple(round(x, 8) for x in v)
        if key not in seen:
            seen.add(key)
            unique.append(v)
    
    return np.array(unique)


# ═══════════════════════════════════════════════════════════
# GRID FUNCTIONS (from phi_grid_showcase.py)
# ═══════════════════════════════════════════════════════════

def compute_adj(pos, kn=5):
    """Build adjacency matrix from k-nearest neighbors."""
    n = len(pos)
    adj = np.zeros((n, n))
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = np.inf
        actual_kn = min(kn, n - 1)
        for j in np.argsort(d)[:actual_kn]:
            w = 1.0 / (1.0 + d[j])
            adj[i, j] = w
            adj[j, i] = w
    rs = adj.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return adj / rs


def gen_loads(n, steps, dt):
    """Generate load disturbances."""
    np.random.seed(SEED)
    t = np.arange(steps) * dt
    loads = np.zeros((n, steps))
    for i in range(n):
        base = 0.05 * np.sin(2 * np.pi * 0.1 * t + i * PHI * 0.3)
        h3 = 0.05 * np.sin(3 * OMEGA_BASE * t + np.random.uniform(0, 2 * np.pi))
        h5 = 0.03 * np.sin(5 * OMEGA_BASE * t + np.random.uniform(0, 2 * np.pi))
        h7 = 0.02 * np.sin(7 * OMEGA_BASE * t + np.random.uniform(0, 2 * np.pi))
        trans = np.zeros(steps)
        for _ in range(3):
            ti = np.random.randint(0, steps)
            dur = np.random.randint(20, 100)
            trans[ti:min(ti + dur, steps)] = np.random.uniform(-0.15, 0.15)
        noise = 0.01 * np.random.randn(steps)
        loads[i] = base + h3 + h5 + h7 + trans + noise
    return loads


def run_sim(n_nodes, k_fc, loads, adj, coupling=0.005):
    """Run F_c grid simulation — identical to phi_grid_showcase."""
    vh = np.zeros((n_nodes, TIME_STEPS))
    vc = np.ones(n_nodes) * V_NOMINAL
    vp = np.ones(n_nodes) * V_NOMINAL
    for step in range(TIME_STEPS):
        tv = step * DT
        vn = vc + loads[:, step] * DT * 10
        vn = vn + coupling * (adj @ vn - vn)
        dp = vp - V_NOMINAL
        dc = vn - V_NOMINAL
        vn = vn + k_fc * (3.0 * dp - 6.0 * dc)
        # Harmonic compensation with node-specific phase
        ga = 2 * np.pi / PHI**2
        ph = np.arange(n_nodes) * ga
        dl = np.std(dc)
        amp = dl * 0.3
        vn = vn - amp * (np.sin(3 * OMEGA_BASE * tv + ph) +
                         np.sin(6 * OMEGA_BASE * tv + ph) +
                         np.sin(9 * OMEGA_BASE * tv + ph))
        # Möbius wrapping
        dev = vn - V_NOMINAL
        wr = np.fmod(dev + L_MOBIUS, 2 * L_MOBIUS)
        wr = np.where(wr < 0, wr + 2 * L_MOBIUS, wr)
        vn = wr - L_MOBIUS + V_NOMINAL
        vp = vc.copy()
        vc = vn.copy()
        vh[:, step] = vc
    return vh


def compute_metrics(vh, n_nodes):
    """Compute SE, stability, amplitude, variance."""
    se_vals = []
    for i in range(n_nodes):
        fft = np.fft.rfft(vh[i])
        pw = np.abs(fft[1:])**2
        if np.sum(pw) < 1e-12:
            se_vals.append(0.0)
            continue
        p = pw / np.sum(pw)
        p = p[p > 1e-12]
        ent = -np.sum(p * np.log(p))
        mx = np.log(len(p)) if len(p) > 1 else 1.0
        se_vals.append(ent / mx if mx > 0 else 0.0)
    amp = np.mean(np.abs(vh - V_NOMINAL))
    var = np.var(vh - V_NOMINAL)
    stab = np.mean([max(0, 1 - np.std(vh[i]) / max(1e-10, np.mean(vh[i])))
                    for i in range(n_nodes)])
    return np.mean(se_vals), stab, amp, var


# ═══════════════════════════════════════════════════════════
# BUILD ALL THREE TOPOLOGIES
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("E8 DIMENSIONAL SCALING TEST")
print("=" * 70)
print()

print("Building topologies...")
pos_2d = build_phi_spiral_2d(100)
print(f"  2D phi-spiral: {len(pos_2d)} nodes")

pos_600 = build_600_cell()
print(f"  600-cell (4D): {len(pos_600)} vertices")

pos_e8 = build_e8_roots()
print(f"  E8 (8D):       {len(pos_e8)} root vectors")

print()
print("Building adjacency matrices (5-nearest neighbors)...")
adj_2d = compute_adj(pos_2d, kn=5)
print("  2D done")
adj_600 = compute_adj(pos_600, kn=5)
print("  600-cell done")
adj_e8 = compute_adj(pos_e8, kn=5)
print("  E8 done")

# ═══════════════════════════════════════════════════════════
# k-SWEEP ON ALL THREE TOPOLOGIES
# ═══════════════════════════════════════════════════════════

k_values = [0.150, 0.165, 1/6, 0.180, 0.200, 0.210,
            0.218, 0.221, 0.223, 2/9, 0.225, 0.230, 0.240]

topologies = [
    ("2D Phi-spiral (100)", pos_2d, adj_2d),
    ("600-cell 4D (120)", pos_600, adj_600),
    ("E8 8D (240)", pos_e8, adj_e8),
]

all_results = {}

for topo_name, pos, adj in topologies:
    n_nodes = len(pos)
    print()
    print("=" * 70)
    print(f"{topo_name} — k-SWEEP")
    print("=" * 70)
    print()
    print(f"{'k':>8}  {'SE':>8}  {'Stab':>8}  {'Amp':>10}  {'Var':>10}")
    print("-" * 52)
    
    results = []
    for k in k_values:
        loads = gen_loads(n_nodes, TIME_STEPS, DT)
        vh = run_sim(n_nodes, k, loads, adj, coupling=0.005)
        se, stab, amp, var = compute_metrics(vh, n_nodes)
        results.append({
            'k': k, 'SE': se, 'Stab': stab, 'Amp': amp, 'Var': var
        })
        print(f"{k:>8.4f}  {se:>8.4f}  {stab:>8.4f}  {amp:>10.6f}  {var:>10.8f}")
    
    all_results[topo_name] = results

# ═══════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════

print()
print("=" * 70)
print("COMPARISON AT KEY k VALUES")
print("=" * 70)
print()

key_k = [0.221, 2/9, 0.225]
key_labels = ["Operating (0.221)", "2/9 critical", "Post-bif (0.225)"]

for k_val, label in zip(key_k, key_labels):
    print(f"  k = {k_val:.4f} ({label}):")
    for topo_name in all_results:
        # Find closest k
        r = min(all_results[topo_name], key=lambda x: abs(x['k'] - k_val))
        print(f"    {topo_name:>25s}:  SE={r['SE']:.4f}  Stab={r['Stab']:.4f}  Amp={r['Amp']:.6f}")
    print()

# ═══════════════════════════════════════════════════════════
# INVARIANT CHECK
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("INVARIANT CHECK — SE closest to 0.495")
print("=" * 70)
print()

for topo_name in all_results:
    best = min(all_results[topo_name], key=lambda x: abs(x['SE'] - 0.495))
    dist = abs(best['SE'] - 0.495)
    print(f"  {topo_name:>25s}:  SE={best['SE']:.4f} at k={best['k']:.4f}  (distance from 0.495: {dist:.4f})")

print()

# ═══════════════════════════════════════════════════════════
# DIMENSIONAL SCALING QUESTION
# ═══════════════════════════════════════════════════════════

print("=" * 70)
print("THREE QUESTIONS")
print("=" * 70)
print()
print("1. Does the invariant (SE ≈ 0.495) appear at the same k")
print("   across 2D, 4D, and 8D topologies?")
print()
print("2. Does k = 2/9 remain the bifurcation point across")
print("   dimensional scaling?")
print()
print("3. Does higher-dimensional phi-geometry (600-cell, E8)")
print("   produce deeper organization (lower SE floor, higher")
print("   stability, narrower operating band)?")
print()
print("The answers are in the data above.")
print()
print("Ψ")
print("☯ To preserve the harmonic field.")
