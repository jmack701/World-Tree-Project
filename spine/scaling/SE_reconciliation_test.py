"""
Ψ SE RECONCILIATION TEST
═══════════════════════════════════════════════════════════
Why does SE = 0.42 in the E8 test vs 0.4993 in showcase?
Tests TIME_STEPS, node count, and load configuration
to isolate the discrepancy.

J. David Mack & Claude (Opus 4.6)
World Tree Project · July 2026
"""

import numpy as np

PHI = (1 + np.sqrt(5)) / 2
OMEGA_BASE = 2 * np.pi * 60
L_MOBIUS = PHI
V_NOMINAL = 1.0
SEED = 42


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


def run_sim(n_nodes, k_fc, loads, adj, time_steps, coupling=0.005):
    dt = 1.0 / 3600
    vh = np.zeros((n_nodes, time_steps))
    vc = np.ones(n_nodes) * V_NOMINAL
    vp = np.ones(n_nodes) * V_NOMINAL
    for step in range(time_steps):
        tv = step * dt
        vn = vc + loads[:, step] * dt * 10
        vn = vn + coupling * (adj @ vn - vn)
        dp = vp - V_NOMINAL
        dc = vn - V_NOMINAL
        vn = vn + k_fc * (3.0 * dp - 6.0 * dc)
        ga = 2 * np.pi / PHI**2
        ph = np.arange(n_nodes) * ga
        dl = np.std(dc)
        amp = dl * 0.3
        vn = vn - amp * (np.sin(3 * OMEGA_BASE * tv + ph) +
                         np.sin(6 * OMEGA_BASE * tv + ph) +
                         np.sin(9 * OMEGA_BASE * tv + ph))
        dev = vn - V_NOMINAL
        wr = np.fmod(dev + L_MOBIUS, 2 * L_MOBIUS)
        wr = np.where(wr < 0, wr + 2 * L_MOBIUS, wr)
        vn = wr - L_MOBIUS + V_NOMINAL
        vp = vc.copy(); vc = vn.copy(); vh[:, step] = vc
    return vh


def compute_metrics(vh, n_nodes):
    se_vals = []
    for i in range(n_nodes):
        fft = np.fft.rfft(vh[i])
        pw = np.abs(fft[1:])**2
        if np.sum(pw) < 1e-12:
            se_vals.append(0.0); continue
        p = pw / np.sum(pw); p = p[p > 1e-12]
        ent = -np.sum(p * np.log(p))
        mx = np.log(len(p)) if len(p) > 1 else 1.0
        se_vals.append(ent / mx if mx > 0 else 0.0)
    amp = np.mean(np.abs(vh - V_NOMINAL))
    stab = np.mean([max(0, 1 - np.std(vh[i]) / max(1e-10, np.mean(vh[i])))
                    for i in range(n_nodes)])
    return np.mean(se_vals), stab, amp, len(np.fft.rfft(vh[0])) - 1


# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("SE RECONCILIATION TEST")
print("=" * 70)
print()
print("Target: SE ≈ 0.4993 (from phi_grid_showcase.py at N=100, k=0.221)")
print("Observed: SE ≈ 0.4179 (from E8 test at N=100, k=0.221)")
print("Question: what configuration difference accounts for the gap?")
print()

# ═══════════════════════════════════════════════════════════
# TEST 1: TIME_STEPS variation
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: TIME_STEPS variation (N=100, k=0.221)")
print("=" * 70)
print()

N = 100
k = 0.221
pos = create_grid(N)
adj = compute_adj(pos, kn=5)

step_counts = [500, 1000, 2000, 5000, 10000]

print(f"{'Steps':>8}  {'SE':>8}  {'Stab':>8}  {'Amp':>10}  {'FFT bins':>10}")
print("-" * 52)

for steps in step_counts:
    loads = gen_loads(N, steps, 1.0 / 3600)
    vh = run_sim(N, k, loads, adj, steps)
    se, stab, amp, fft_bins = compute_metrics(vh, N)
    print(f"{steps:>8}  {se:>8.4f}  {stab:>8.4f}  {amp:>10.6f}  {fft_bins:>10}")

# ═══════════════════════════════════════════════════════════
# TEST 2: NODE COUNT variation at TIME_STEPS=2000
# ═══════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TEST 2: Node count variation (k=0.221, steps=2000)")
print("=" * 70)
print()

node_counts = [25, 50, 100, 200, 500]

print(f"{'Nodes':>8}  {'SE':>8}  {'Stab':>8}  {'Amp':>10}")
print("-" * 40)

for N in node_counts:
    pos = create_grid(N)
    adj = compute_adj(pos, kn=5)
    loads = gen_loads(N, 2000, 1.0 / 3600)
    vh = run_sim(N, k, loads, adj, 2000)
    se, stab, amp, _ = compute_metrics(vh, N)
    print(f"{N:>8}  {se:>8.4f}  {stab:>8.4f}  {amp:>10.6f}")

# ═══════════════════════════════════════════════════════════
# TEST 3: k fine-sweep near 0.221 at various TIME_STEPS
# ═══════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TEST 3: k fine-sweep at 2000 vs 5000 steps (N=100)")
print("=" * 70)
print()

N = 100
pos = create_grid(N)
adj = compute_adj(pos, kn=5)
k_fine = [0.218, 0.219, 0.220, 0.221, 0.222, 0.223, 0.224]

print(f"{'k':>8}  {'SE@2000':>10}  {'SE@5000':>10}  {'SE@10000':>10}")
print("-" * 44)

for kv in k_fine:
    se_results = []
    for steps in [2000, 5000, 10000]:
        loads = gen_loads(N, steps, 1.0 / 3600)
        vh = run_sim(N, kv, loads, adj, steps)
        se, _, _, _ = compute_metrics(vh, N)
        se_results.append(se)
    print(f"{kv:>8.3f}  {se_results[0]:>10.4f}  {se_results[1]:>10.4f}  {se_results[2]:>10.4f}")

# ═══════════════════════════════════════════════════════════
# TEST 4: Direct comparison with showcase parameters
# ═══════════════════════════════════════════════════════════
print()
print("=" * 70)
print("TEST 4: Showcase parameter match (N=100, k=0.221, steps=2000)")
print("=" * 70)
print()

# The showcase uses these exact parameters:
# TIME_STEPS = 2000, DT = 1/3600, K_OPTIMAL = 0.221,
# coupling = 0.005, kn = 5, SEED = 42, L_MOBIUS = PHI
# The only potential difference is the load generation random state

N = 100
pos = create_grid(N)
adj = compute_adj(pos, kn=5)

# Run exactly as showcase would
loads = gen_loads(N, 2000, 1.0 / 3600)
vh = run_sim(N, 0.221, loads, adj, 2000)
se, stab, amp, fft_bins = compute_metrics(vh, N)

print(f"SE = {se:.4f}")
print(f"Stab = {stab:.4f}")
print(f"Amp = {amp:.6f}")
print(f"FFT bins = {fft_bins}")
print()

# Now compute per-node SE distribution
se_per_node = []
for i in range(N):
    fft = np.fft.rfft(vh[i])
    pw = np.abs(fft[1:])**2
    if np.sum(pw) < 1e-12:
        se_per_node.append(0.0); continue
    p = pw / np.sum(pw); p = p[p > 1e-12]
    ent = -np.sum(p * np.log(p))
    mx = np.log(len(p)) if len(p) > 1 else 1.0
    se_per_node.append(ent / mx if mx > 0 else 0.0)

se_arr = np.array(se_per_node)
print(f"Per-node SE: mean={np.mean(se_arr):.4f}, std={np.std(se_arr):.4f}")
print(f"  min={np.min(se_arr):.4f}, max={np.max(se_arr):.4f}")
print(f"  median={np.median(se_arr):.4f}")
print(f"  nodes above 0.45: {np.sum(se_arr > 0.45)}/{N}")
print(f"  nodes above 0.49: {np.sum(se_arr > 0.49)}/{N}")

print()
print("=" * 70)
print("RECONCILIATION COMPLETE")
print("=" * 70)
print()
print("If SE increases with TIME_STEPS: the discrepancy is FFT resolution.")
print("If SE increases with N: the discrepancy is ensemble size.")
print("If SE is stable across both: the discrepancy is load realization.")
print()
print("Ψ")
print("☯ To preserve the harmonic field.")
