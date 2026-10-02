"""node_level_k13.py — the thirteenth question, powered.

Instruments (EXPECTATIONS_node_level.md, written before this run):
  1. Identity gate at the published 2,000-step configuration.
  2. Bridge: the failed probe rerun at 20 seeds, 2,000 steps (expect unresolved).
  3. Powered probe: 20 seeds x 20,000 steps, per-node SE / wrap rate / amplitude /
     32-band spectral shape on K13.
  4. Clustering: 3-way agglomerative on seed-averaged spectral shapes, no
     positional information supplied. Does the dynamics recover {1, 6, 6}?
  5. Ablation control: minus-center vs minus-inner vs minus-outer at N=12 —
     identical load matrices at fixed seed; geometry the only variable.
  6. 64-grid nucleus cross-check under the same power.

Engine mathematics byte-identical to E8_Dimensional_Scaling_Test_v2.py;
additions are run length and per-node instrumentation only.
J. David Mack & Claude (Fable 5) - World Tree Project - seed set 42-61
"""
import numpy as np, itertools, json, math

src = open("/home/claude/metatron_chain_grid_sweep.py").read()
head = src.split("# ------------------------------------------------------------------\n# SWEEP")[0]
NS = {}; exec(head, NS)
PHI = NS['PHI']; DT = NS['DT']; OMEGA_BASE = NS['OMEGA_BASE']
V_NOMINAL = NS['V_NOMINAL']; L_MOBIUS = NS['L_MOBIUS']
compute_adj = NS['compute_adj']; adj_from_edges = NS['adj_from_edges']

SEEDS = list(range(42, 62))

def gen_loads(n, steps, seed):
    np.random.seed(seed)
    t = np.arange(steps) * DT
    loads = np.zeros((n, steps))
    for i in range(n):
        base = 0.05 * np.sin(2 * np.pi * 0.1 * t + i * PHI * 0.3)
        h3 = 0.05 * np.sin(3 * OMEGA_BASE * t + np.random.uniform(0, 2 * np.pi))
        h5 = 0.03 * np.sin(5 * OMEGA_BASE * t + np.random.uniform(0, 2 * np.pi))
        h7 = 0.02 * np.sin(7 * OMEGA_BASE * t + np.random.uniform(0, 2 * np.pi))
        trans = np.zeros(steps)
        for _ in range(int(np.random.randint(5, 15))):
            loc = np.random.randint(0, steps); w = np.random.randint(5, 20)
            a = np.random.uniform(0.10, 0.25) * np.random.choice([-1, 1])
            s, e = max(0, loc - w // 2), min(steps, loc + w // 2)
            trans[s:e] = a
        loads[i] = base + h3 + h5 + h7 + trans + 0.01 * np.random.randn(steps)
    return loads

def run_sim(n, k, loads, adj, steps, record_wraps=False):
    """Byte-identical dynamics; optional per-node wrap-event counting (instrumentation)."""
    vh = np.zeros((n, steps))
    vc = np.ones(n) * V_NOMINAL; vp = np.ones(n) * V_NOMINAL
    ph = np.arange(n) * (2 * np.pi / PHI**2)
    wraps = np.zeros(n)
    for step in range(steps):
        tv = step * DT
        vn = vc + loads[:, step] * DT * 10
        vn = vn + 0.05 * (adj @ vn - vn)
        dp = vp - V_NOMINAL; dc = vn - V_NOMINAL
        vn = vn + k * (3.0 * dp - 6.0 * dc)
        dl = np.std(dc); amp = dl * 0.3
        vn = vn - amp * (np.sin(3 * OMEGA_BASE * tv + ph) +
                         np.sin(6 * OMEGA_BASE * tv + ph) +
                         np.sin(9 * OMEGA_BASE * tv + ph))
        dev = vn - V_NOMINAL
        if record_wraps: wraps += (np.abs(dev) > L_MOBIUS)
        wr = np.fmod(dev + L_MOBIUS, 2 * L_MOBIUS)
        wr = np.where(wr < 0, wr + 2 * L_MOBIUS, wr)
        vn = wr - L_MOBIUS + V_NOMINAL
        vp = vc.copy(); vc = vn.copy()
        vh[:, step] = vc
    return vh, wraps / steps

def per_node_obs(vh, nbands=32):
    n, steps = vh.shape
    se = np.zeros(n); shape = np.zeros((n, nbands)); ampl = np.zeros(n)
    for i in range(n):
        fft = np.fft.rfft(vh[i] - vh[i].mean())
        pw = np.abs(fft[1:])**2
        p = pw / pw.sum(); pp = p[p > 1e-14]
        se[i] = -np.sum(pp * np.log(pp)) / np.log(len(p))
        # 32-band log-spectral shape
        bands = np.array_split(pw, nbands)
        bp = np.array([b.mean() for b in bands]); bp = np.log(bp + 1e-20)
        shape[i] = (bp - bp.mean()) / (bp.std() + 1e-12)
        ampl[i] = np.std(vh[i])
    return se, shape, ampl

MET = NS['build_metatron_13']()
MET_EDGES = [(i, j) for i in range(13) for j in range(i + 1, 13)]
ADJ13 = adj_from_edges(MET, MET_EDGES)
CLASSES = {"center": [0], "inner": list(range(1, 7)), "outer": list(range(7, 13))}

# ------------------------------------------------------------ 1. identity gate
print("=" * 78)
print("IDENTITY GATE (published configuration, 2,000 steps, seed 42)")
ok = True
for name, builder, pub in [("2D", NS['build_phi_spiral_2d'](100), (0.4932, 0.4847)),
                            ("600", NS['build_600_cell'](), (0.4989, 0.4940)),
                            ("E8", NS['build_e8_roots'](), (0.4996, 0.4995))]:
    adj = compute_adj(builder, 5); n = len(builder)
    ld = gen_loads(n, 2000, 42)
    for kk, p in zip([0.221, 2/9], pub):
        vh, _ = run_sim(n, kk, ld, adj, 2000)
        se = per_node_obs(vh)[0].mean()
        m = abs(se - p) < 5e-5; ok &= m
        print(f"  {name:>4s} SE({kk:.4f}) = {se:.4f} (pub {p})  {m}")
print(f"GATE: {'PASSED' if ok else 'FAILED'}"); assert ok

# ------------------------------------------------------------ 2+3. bridge and powered probe
def probe(steps, seeds, k=0.221):
    ses, shapes, wrs, amps = [], [], [], []
    for sd in seeds:
        ld = gen_loads(13, steps, sd)
        vh, wr = run_sim(13, k, ld, ADJ13, steps, record_wraps=True)
        se, sh, am = per_node_obs(vh)
        ses.append(se); shapes.append(sh); wrs.append(wr); amps.append(am)
    return np.array(ses), np.array(shapes), np.array(wrs), np.array(amps)

def class_stats(ses):
    out = {}
    for c, ids in CLASSES.items():
        m = ses[:, ids].mean(axis=1)          # per-seed class mean
        out[c] = (m.mean(), m.std(ddof=1) / math.sqrt(len(m)), m)
    d = out["outer"][2] - out["center"][2]
    tstat = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    signs = int((d > 0).sum())
    return out, tstat, signs

print()
print("BRIDGE — 2,000 steps x 20 seeds (expect unresolved)")
seB, _, _, _ = probe(2000, SEEDS)
stB, tB, sB = class_stats(seB)
for c in ["center", "inner", "outer"]:
    print(f"  {c:>6s}: {stB[c][0]:.4f} ± {stB[c][1]:.4f} (sem)")
print(f"  outer−center: t = {tB:.2f}, seeds with outer>center: {sB}/20")

print()
print("POWERED — 20,000 steps x 20 seeds")
seP, shP, wrP, amP = probe(20000, SEEDS)
stP, tP, sP = class_stats(seP)
for c in ["center", "inner", "outer"]:
    print(f"  {c:>6s}: SE {stP[c][0]:.4f} ± {stP[c][1]:.4f}   wrap/step {wrP[:, CLASSES[c]].mean():.4f}   amp {amP[:, CLASSES[c]].mean():.4f}")
print(f"  outer−center: t = {tP:.2f}, seeds with outer>center: {sP}/20")
print(f"  E-N1 bar (t ≥ 3, ≥16/20, direction center<inner<outer): "
      f"{'MET' if (tP >= 3 and sP >= 16 and stP['center'][0] < stP['inner'][0] < stP['outer'][0]) else 'NOT MET'}")

# ------------------------------------------------------------ 4. clustering (no positions supplied)
mean_shape = shP.mean(axis=0)                     # 13 x 32 seed-averaged
D = np.zeros((13, 13))
for i in range(13):
    for j in range(13):
        D[i, j] = np.linalg.norm(mean_shape[i] - mean_shape[j])
# simple average-linkage agglomerative to 3 clusters
clusters = [[i] for i in range(13)]
while len(clusters) > 3:
    best = (1e18, None, None)
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            dd = np.mean([D[i, j] for i in clusters[a] for j in clusters[b]])
            if dd < best[0]: best = (dd, a, b)
    _, a, b = best
    clusters[a] = clusters[a] + clusters[b]; del clusters[b]
part = sorted([sorted(c) for c in clusters], key=len)
target = [[0], list(range(1, 7)), list(range(7, 13))]
recovered = part == target
print()
print("CLUSTERING — 3-way agglomerative on seed-averaged spectral shapes")
print(f"  partition: {part}")
print(f"  orbit classes {{1,6,6}} recovered exactly: {recovered}")

# ------------------------------------------------------------ 5. ablation control
print()
print("ABLATION — N=12, identical loads per seed across all three ablations")
ABL = {"minus-center": [i for i in range(13) if i != 0],
       "minus-inner": [i for i in range(13) if i != 1],
       "minus-outer": [i for i in range(13) if i != 7]}
KEY = [0.221, 0.223, 2/9, 0.225]
abl = {a: {k: [] for k in KEY} for a in ABL}
for sd in SEEDS:
    ld12 = gen_loads(12, 20000, sd)
    for aname, keep in ABL.items():
        pos = MET[keep]
        edges = [(i, j) for i in range(12) for j in range(i + 1, 12)]
        adj = adj_from_edges(pos, edges)
        for kk in KEY:
            vh, _ = run_sim(12, kk, ld12, adj, 20000)
            abl[aname][kk].append(per_node_obs(vh)[0].mean())
print(f"{'ablation':>14s}  " + "  ".join(f"SE({('2/9' if abs(k-2/9)<1e-9 else k)})" for k in KEY))
for aname in ABL:
    print(f"{aname:>14s}  " + "  ".join(f"{np.mean(abl[aname][k]):.4f}±{np.std(abl[aname][k],ddof=1)/math.sqrt(20):.4f}" for k in KEY))
dc = np.array(abl["minus-center"][0.223]); di = np.array(abl["minus-inner"][0.223]); do = np.array(abl["minus-outer"][0.223])
t_ci = (dc - di).mean() / ((dc - di).std(ddof=1) / math.sqrt(20))
t_co = (dc - do).mean() / ((dc - do).std(ddof=1) / math.sqrt(20))
t_io = (di - do).mean() / ((di - do).std(ddof=1) / math.sqrt(20))
print(f"  lock-row paired t: center-vs-inner {t_ci:.2f}, center-vs-outer {t_co:.2f}, inner-vs-outer {t_io:.2f}")

# ------------------------------------------------------------ 6. 64-grid nucleus cross-check
print()
print("64-GRID — six classes, 20 seeds x 20,000 steps, k = 0.221")
exec(open("/home/claude/te64_container_sweep.py").read().split("# ------------------------------------------------------------------\n# IDENTITY GATE")[0].split('print("=" * 78)')[0], NS)
v64, t64, e64 = NS['build_64_grid']()
adj64 = adj_from_edges(v64, e64)
center = np.array([2.0, 2.0, 2.0])
def cls64(v):
    off = tuple(np.sort(np.abs(v - center))[::-1])
    return {(0.,0.,0.):"C1", (1.,1.,0.):"C2", (2.,0.,0.):"C3",
            (2.,1.,1.):"C4", (2.,2.,0.):"C5", (2.,2.,2.):"C6"}[off]
c64 = {}
for i, v in enumerate(v64): c64.setdefault(cls64(v), []).append(i)
se64 = []
for sd in SEEDS:
    ld = gen_loads(63, 20000, sd)
    vh, _ = run_sim(63, 0.221, ld, adj64, 20000)
    se64.append(per_node_obs(vh)[0])
se64 = np.array(se64)
R64 = {"C1": 0.0, "C2": math.sqrt(2), "C3": 2.0, "C4": math.sqrt(6), "C5": 2*math.sqrt(2), "C6": 2*math.sqrt(3)}
print("  " + "  ".join(f"{c} (r={R64[c]:.2f}): {se64[:, c64[c]].mean(axis=1).mean():.4f}±{se64[:, c64[c]].mean(axis=1).std(ddof=1)/math.sqrt(20):.4f}" for c in ["C1","C2","C3","C4","C5","C6"]))
mm = [se64[:, c64[c]].mean(axis=1).mean() for c in ["C1","C2","C3","C4","C5","C6"]]
print(f"  monotone increasing with radius: {all(mm[i] < mm[i+1] for i in range(5))}")

json.dump(dict(bridge={c: stB[c][:2] for c in stB}, powered={c: stP[c][:2] for c in stP},
               t_powered=tP, signs=sP, partition=part, recovered=bool(recovered),
               ablation={a: {str(k): [float(np.mean(abl[a][k])), float(np.std(abl[a][k],ddof=1)/math.sqrt(20))] for k in KEY} for a in ABL},
               t_lock=dict(ci=t_ci, co=t_co, io=t_io),
               grid64={c: [float(se64[:, c64[c]].mean()), float(se64[:, c64[c]].mean(axis=1).std(ddof=1)/math.sqrt(20))] for c in c64}),
          open("/home/claude/node_level_results.json", "w"), indent=1, default=float)
print()
print("Saved node_level_results.json")
print("Psi - To preserve the harmonic field.")
