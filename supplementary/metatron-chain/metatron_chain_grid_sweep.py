"""metatron_chain_grid_sweep.py — the F_c grid law on the Metatron projection chain.

Chain: Metatron's Cube (2D, K13) -> icosahedron / dodecahedron / rhombic
triacontahedron (3D, Ih) -> 600-cell / 120-cell (4D, H4 dual pair) -> E8 (8D).

Engine: E8_Dimensional_Scaling_Test_v2.py VERBATIM (c = 0.05, 2,000 steps,
seed 42, 13-gain sweep, 5-NN distance-weighted row-normalized coupling).
Identity gate: the published 2D / 600-cell / E8 rows must reproduce before
any new container is read. Natural-degree variants run beside the 5-NN rule
where the container's own connectivity differs (Spine 6.2 companion).

New containers this run: Metatron K13, icosahedron(12), dodecahedron(20),
rhombic triacontahedron(32, true dual construction, verified), 120-cell(600).

J. David Mack & Claude (Fable 5) - World Tree Project - August 2026 - seed 42
"""
import numpy as np
import itertools, json, math

PHI = (1 + np.sqrt(5)) / 2
OMEGA_BASE = 2 * np.pi * 60
TIME_STEPS = 2000
DT = 1.0 / 3600
L_MOBIUS = PHI
V_NOMINAL = 1.0
SEED = 42

# ------------------------------------------------------------------
# ENGINE — verbatim from E8_Dimensional_Scaling_Test_v2.py
# ------------------------------------------------------------------
def build_phi_spiral_2d(n):
    ga = 2 * np.pi / PHI**2
    pos = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt(i + 1) * 0.5
        pos[i] = [r * np.cos(i * ga), r * np.sin(i * ga)]
    return pos

def build_600_cell():
    vertices = []
    for i in range(4):
        for s in [-1, 1]:
            v = [0, 0, 0, 0]; v[i] = s; vertices.append(v)
    for signs in itertools.product([-1, 1], repeat=4):
        vertices.append([s * 0.5 for s in signs])
    base = [0, 0.5, PHI / 2, 1 / (2 * PHI)]
    for perm in itertools.permutations(range(4)):
        inv = 0; p = list(perm)
        for i in range(4):
            for j in range(i + 1, 4):
                if p[i] > p[j]: inv += 1
        if inv % 2 != 0: continue
        permuted = [base[p[i]] for i in range(4)]
        for signs in itertools.product([-1, 1], repeat=4):
            vertices.append([permuted[i] * signs[i] for i in range(4)])
    unique, seen = [], set()
    for v in vertices:
        key = tuple(round(x, 8) for x in v)
        if key not in seen: seen.add(key); unique.append(v)
    return np.array(unique)

def build_e8_roots():
    roots = []
    for pos in itertools.combinations(range(8), 2):
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                v = [0] * 8; v[pos[0]] = s1; v[pos[1]] = s2; roots.append(v)
    for n in range(256):
        bits = [(n >> i) & 1 for i in range(8)]
        signs = [2 * b - 1 for b in bits]
        if signs.count(-1) % 2 == 0:
            roots.append([s * 0.5 for s in signs])
    unique, seen = [], set()
    for v in roots:
        key = tuple(round(x, 8) for x in v)
        if key not in seen: seen.add(key); unique.append(v)
    return np.array(unique)

def compute_adj(pos, kn=5):
    n = len(pos)
    adj = np.zeros((n, n))
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = np.inf
        actual_kn = min(kn, n - 1)
        for j in np.argsort(d)[:actual_kn]:
            w = 1.0 / (1.0 + d[j])
            adj[i, j] = w; adj[j, i] = w
    rs = adj.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    return adj / rs

def adj_from_edges(pos, edges):
    """Natural-edge adjacency: same 1/(1+d) weighting and row normalization
    as the engine's compute_adj, connectivity from the container's own edges."""
    n = len(pos)
    adj = np.zeros((n, n))
    for i, j in edges:
        d = np.linalg.norm(pos[i] - pos[j])
        w = 1.0 / (1.0 + d)
        adj[i, j] = w; adj[j, i] = w
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
        for _ in range(int(np.random.randint(5, 15))):
            loc = np.random.randint(0, steps)
            w = np.random.randint(5, 20)
            a = np.random.uniform(0.10, 0.25) * np.random.choice([-1, 1])
            s, e = max(0, loc - w // 2), min(steps, loc + w // 2)
            trans[s:e] = a
        noise = 0.01 * np.random.randn(steps)
        loads[i] = base + h3 + h5 + h7 + trans + noise
    return loads

def run_sim(n_nodes, k_fc, loads, adj, coupling=0.05):
    vh = np.zeros((n_nodes, TIME_STEPS))
    vc = np.ones(n_nodes) * V_NOMINAL
    vp = np.ones(n_nodes) * V_NOMINAL
    ga = 2 * np.pi / PHI**2
    ph = np.arange(n_nodes) * ga
    for step in range(TIME_STEPS):
        tv = step * DT
        vn = vc + loads[:, step] * DT * 10
        vn = vn + coupling * (adj @ vn - vn)
        dp = vp - V_NOMINAL
        dc = vn - V_NOMINAL
        vn = vn + k_fc * (3.0 * dp - 6.0 * dc)
        dl = np.std(dc)
        amp = dl * 0.3
        vn = vn - amp * (np.sin(3 * OMEGA_BASE * tv + ph) +
                         np.sin(6 * OMEGA_BASE * tv + ph) +
                         np.sin(9 * OMEGA_BASE * tv + ph))
        dev = vn - V_NOMINAL
        wr = np.fmod(dev + L_MOBIUS, 2 * L_MOBIUS)
        wr = np.where(wr < 0, wr + 2 * L_MOBIUS, wr)
        vn = wr - L_MOBIUS + V_NOMINAL
        vp = vc.copy(); vc = vn.copy()
        vh[:, step] = vc
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
    var = np.var(vh - V_NOMINAL)
    stab = np.mean([max(0, 1 - np.std(vh[i]) / max(1e-10, np.mean(vh[i])))
                    for i in range(n_nodes)])
    return np.mean(se_vals), stab, amp, var

# ------------------------------------------------------------------
# NEW CONSTRUCTORS — each verified before use
# ------------------------------------------------------------------
def build_metatron_13():
    """Metatron's Cube node set: 1 center + inner hex (r=1) + outer hex (r=2).
    Its 78 lines are all pairwise connections: the complete graph K13."""
    pos = [(0.0, 0.0)]
    for ring_r in (1.0, 2.0):
        for j in range(6):
            a = j * np.pi / 3
            pos.append((ring_r * np.cos(a), ring_r * np.sin(a)))
    return np.array(pos)

def build_icosahedron():
    v = []
    for a, b in itertools.product([-1, 1], repeat=2):
        v.append((0, a, b * PHI)); v.append((a, b * PHI, 0)); v.append((b * PHI, 0, a))
    return np.array(sorted(set(v)))

def build_dodecahedron():
    v = list(itertools.product([-1, 1], repeat=3))
    for a, b in itertools.product([-1, 1], repeat=2):
        v.append((0, a / PHI, b * PHI)); v.append((a / PHI, b * PHI, 0)); v.append((b * PHI, 0, a / PHI))
    return np.array(sorted(set(tuple(x) for x in v)))

def min_dist_edges(pos, tol=1e-6):
    n = len(pos)
    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    dmat[np.arange(n), np.arange(n)] = np.inf
    dmin = dmat.min()
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)
             if abs(dmat[i, j] - dmin) < tol * max(1.0, dmin)]
    return edges, dmin

def build_rhombic_triacontahedron():
    """RT as the polar dual of the icosidodecahedron. Returns (pos, edges).
    Verified downstream: 32 vertices, 60 edges, degrees {3:20, 5:12},
    golden rhombus faces (diagonal ratio phi)."""
    ico = build_icosahedron()
    ico_edges, _ = min_dist_edges(ico)                     # 30 icosa edges
    mid = np.array([(ico[i] + ico[j]) / 2 for i, j in ico_edges])  # 30 icosidodeca vertices
    edge_index = {frozenset(e): idx for idx, e in enumerate(ico_edges)}
    # icosa faces: triangles of mutually adjacent vertices
    adjset = {i: set() for i in range(len(ico))}
    for i, j in ico_edges: adjset[i].add(j); adjset[j].add(i)
    tri_faces = []
    for i in range(len(ico)):
        for j in adjset[i]:
            for k2 in adjset[i]:
                if j < k2 and k2 in adjset[j] and i < j:
                    tri_faces.append((i, j, k2))
    # icosidodecahedron faces:
    # 20 triangles = midpoints of each icosa face's 3 edges
    # 12 pentagons = midpoints of each icosa vertex's 5 incident edges
    faces = []       # list of lists of midpoint indices
    face_type = []   # 'T' (-> 3-valent RT vertex) or 'P' (-> 5-valent)
    for (i, j, k2) in tri_faces:
        faces.append([edge_index[frozenset((i, j))], edge_index[frozenset((j, k2))],
                      edge_index[frozenset((i, k2))]])
        face_type.append('T')
    for i in range(len(ico)):
        faces.append([edge_index[frozenset((i, j))] for j in adjset[i]])
        face_type.append('P')
    # RT vertices: polar reciprocal of each face plane (centroid / |centroid|^2)
    rt_pos = []
    for f in faces:
        g = mid[f].mean(axis=0)
        rt_pos.append(g / np.dot(g, g))
    rt_pos = np.array(rt_pos)
    # RT edges: faces sharing an icosidodeca edge (two consecutive midpoints)
    pair_faces = {}
    for fi, f in enumerate(faces):
        m = len(f)
        # consecutive pairs on the face boundary: order pentagon midpoints by angle
        if face_type[fi] == 'P':
            g = mid[f].mean(axis=0)
            # order around the vertex axis
            axis = g / np.linalg.norm(g)
            ref = mid[f[0]] - g; ref -= axis * np.dot(ref, axis)
            ref /= np.linalg.norm(ref)
            ref2 = np.cross(axis, ref)
            ang = [math.atan2(np.dot(mid[x] - g, ref2), np.dot(mid[x] - g, ref)) for x in f]
            f_ord = [x for _, x in sorted(zip(ang, f))]
        else:
            f_ord = f
        m = len(f_ord)
        for a in range(m):
            key = frozenset((f_ord[a], f_ord[(a + 1) % m]))
            pair_faces.setdefault(key, []).append(fi)
    rt_edges = []
    for key, fl in pair_faces.items():
        if len(fl) == 2:
            rt_edges.append((fl[0], fl[1]))
    return rt_pos, rt_edges, face_type

def build_120_cell():
    """600 vertices of the 120-cell (circumradius sqrt(8), edge 2/phi^2).
    Verified downstream: count 600, all norms sqrt(8), natural degree 4."""
    p, p2, pi1, pi2 = PHI, PHI**2, 1/PHI, 1/PHI**2
    r5 = math.sqrt(5)
    all_perm_sets = [
        (0, 0, 2, 2),
        (1, 1, 1, r5),
        (pi2, p, p, p),
        (pi1, pi1, pi1, p2),
    ]
    even_perm_sets = [
        (0, pi2, 1, p2),
        (0, pi1, p, r5),
        (pi1, 1, p, 2),
    ]
    def signed_perms(base, even_only):
        out = []
        for perm in itertools.permutations(range(4)):
            if even_only:
                inv = sum(1 for i in range(4) for j in range(i + 1, 4)
                          if perm[i] > perm[j])
                if inv % 2 != 0: continue
            arr = [base[perm[i]] for i in range(4)]
            for signs in itertools.product([-1, 1], repeat=4):
                out.append(tuple(arr[i] * signs[i] for i in range(4)))
        return out
    verts = []
    for b in all_perm_sets: verts += signed_perms(b, even_only=False)
    for b in even_perm_sets: verts += signed_perms(b, even_only=True)
    unique, seen = [], set()
    for v in verts:
        key = tuple(round(x, 8) for x in v)
        if key not in seen: seen.add(key); unique.append(v)
    return np.array(unique)

# ------------------------------------------------------------------
# CONSTRUCTION VERIFICATION
# ------------------------------------------------------------------
def degree_counts(n, edges):
    deg = np.zeros(n, int)
    for i, j in edges: deg[i] += 1; deg[j] += 1
    vals, cnts = np.unique(deg, return_counts=True)
    return dict(zip(vals.tolist(), cnts.tolist()))

print("=" * 78)
print("CONSTRUCTION VERIFICATION")
print("=" * 78)

met = build_metatron_13()
met_edges = [(i, j) for i in range(13) for j in range(i + 1, 13)]
print(f"Metatron K13: {len(met)} nodes, {len(met_edges)} edges (expect 13, 78)")

ico = build_icosahedron()
ico_edges, d_ico = min_dist_edges(ico)
print(f"Icosahedron: {len(ico)} vertices, {len(ico_edges)} edges, degrees {degree_counts(12, ico_edges)}"
      f" (expect 12, 30, all 5); edge = {d_ico:.6f} (expect 2)")

dod = build_dodecahedron()
dod_edges, d_dod = min_dist_edges(dod)
print(f"Dodecahedron: {len(dod)} vertices, {len(dod_edges)} edges, degrees {degree_counts(20, dod_edges)}"
      f" (expect 20, 30, all 3); edge = {d_dod:.6f} (expect 2/phi = {2/PHI:.6f})")

rt_pos, rt_edges, rt_types = build_rhombic_triacontahedron()
rt_deg = degree_counts(len(rt_pos), rt_edges)
print(f"Rhombic triacontahedron: {len(rt_pos)} vertices, {len(rt_edges)} edges, degrees {rt_deg}"
      f" (expect 32, 60, {{3:20, 5:12}})")
# golden-rhombus check: reconstruct one face and measure diagonal ratio
# faces of RT are dual to icosidodeca vertices: the 4 RT vertices around one midpoint
mid_faces = {}
for fi, f in enumerate([None]):
    pass
# quick check via edge lengths instead: all RT edges equal?
els = [np.linalg.norm(rt_pos[i] - rt_pos[j]) for i, j in rt_edges]
print(f"  RT edge lengths: min {min(els):.6f} max {max(els):.6f} (equal => true RT)")
# diagonal ratio: for a 3-valent vertex v and a 5-valent neighbor w sharing a face,
# face = (v, w, v', w'); use the two vertex families' radii to get diagonals
r3 = np.linalg.norm(rt_pos[[i for i, t in enumerate(rt_types) if t == 'T'][0]])
r5v = np.linalg.norm(rt_pos[[i for i, t in enumerate(rt_types) if t == 'P'][0]])
print(f"  RT vertex radii: 3-valent {r3:.6f}, 5-valent {r5v:.6f}")

c600 = build_600_cell()
c600_edges, d600 = min_dist_edges(c600)
print(f"600-cell: {len(c600)} vertices; edge = {d600:.6f} (expect 1/phi = {1/PHI:.6f}); "
      f"natural degree {degree_counts(120, c600_edges)}")

c120 = build_120_cell()
norms = np.linalg.norm(c120, axis=1)
c120_edges, d120 = min_dist_edges(c120)
print(f"120-cell: {len(c120)} vertices (expect 600); norms {norms.min():.6f}-{norms.max():.6f} "
      f"(expect sqrt8 = {math.sqrt(8):.6f}); edge = {d120:.6f} (expect 2/phi^2 = {2/PHI**2:.6f}); "
      f"natural degree {degree_counts(600, c120_edges)} (expect all 4)")

e8 = build_e8_roots()
print(f"E8: {len(e8)} roots (expect 240)")

# rhombus face check for RT: find one face (cycle v3 - v5 - v3' - v5')
# take a 3-valent vertex a and one 5-valent neighbor b; their common neighbors
rt_adj = {i: set() for i in range(len(rt_pos))}
for i, j in rt_edges: rt_adj[i].add(j); rt_adj[j].add(i)
a = [i for i, t in enumerate(rt_types) if t == 'T'][0]
b = next(iter(rt_adj[a]))
common = [c for c in rt_adj[a] & set().union(*[rt_adj[x] for x in [b]]) if c != a]
# face = a, b, opposite of a across the face, opposite of b:
# the two vertices adjacent to both a and b complete two different faces; take one
cands = [c for c in rt_adj[b] if c != a and len(rt_adj[c] & rt_adj[a] & {b} | set()) >= 0]
face_found = None
for c in rt_adj[b]:
    if c == a: continue
    for dv in rt_adj[a]:
        if dv == b: continue
        if dv in rt_adj[c]:
            face_found = (a, b, c, dv); break
    if face_found: break
if face_found:
    A, B, C, D = [rt_pos[x] for x in face_found]
    diag1 = np.linalg.norm(C - A); diag2 = np.linalg.norm(D - B)
    ratio = max(diag1, diag2) / min(diag1, diag2)
    # planarity
    nvec = np.cross(B - A, C - A)
    planar = abs(np.dot(D - A, nvec)) / (np.linalg.norm(nvec) + 1e-12)
    print(f"  RT face check: diagonal ratio {ratio:.6f} (expect phi = {PHI:.6f}); "
          f"planarity residual {planar:.2e}")

# ------------------------------------------------------------------
# SWEEP
# ------------------------------------------------------------------
k_values = [0.150, 0.165, 1/6, 0.180, 0.200, 0.210,
            0.218, 0.221, 0.223, 2/9, 0.225, 0.230, 0.240]

containers = [
    ("2D phi-spiral (100) [identity]",      build_phi_spiral_2d(100), ("knn", 5)),
    ("Metatron K13 complete (13, 2D)",      met,   ("edges", met_edges)),
    ("Metatron 5-NN (13, 2D)",              met,   ("knn", 5)),
    ("Icosahedron (12, 3D) nat=5NN",        ico,   ("knn", 5)),
    ("Dodecahedron 5-NN (20, 3D)",          dod,   ("knn", 5)),
    ("Dodecahedron natural-3 (20, 3D)",     dod,   ("edges", dod_edges)),
    ("Rhombic triaconta 5-NN (32, 3D)",     rt_pos, ("knn", 5)),
    ("Rhombic triaconta natural (32, 3D)",  rt_pos, ("edges", rt_edges)),
    ("600-cell 5-NN (120, 4D) [identity]",  c600,  ("knn", 5)),
    ("120-cell 5-NN (600, 4D) [NEW]",       c120,  ("knn", 5)),
    ("120-cell natural-4 (600, 4D) [NEW]",  c120,  ("edges", c120_edges)),
    ("E8 5-NN (240, 8D) [identity]",        e8,    ("knn", 5)),
]

all_results = {}
for name, pos, (mode, arg) in containers:
    n = len(pos)
    adj = compute_adj(pos, kn=arg) if mode == "knn" else adj_from_edges(pos, arg)
    loads = gen_loads(n, TIME_STEPS, DT)
    rows = []
    for k in k_values:
        vh = run_sim(n, k, loads, adj, coupling=0.05)
        se, stab, amp, var = compute_metrics(vh, n)
        rows.append(dict(k=k, SE=se, Stab=stab, Amp=amp, Var=var))
    all_results[name] = rows
    print()
    print(f"--- {name} ---")
    print(f"{'k':>8}  {'SE':>8}  {'Stab':>8}  {'Amp':>10}")
    for r in rows:
        lbl = "1/6" if abs(r['k'] - 1/6) < 1e-9 else ("2/9" if abs(r['k'] - 2/9) < 1e-9 else f"{r['k']:.4f}")
        print(f"{lbl:>8}  {r['SE']:>8.4f}  {r['Stab']:>8.4f}  {r['Amp']:>10.6f}")

# identity gate report
print()
print("=" * 78)
print("IDENTITY GATE (published: 2D 0.4932/0.4847; 600-cell 0.4989/0.4940; E8 0.4996/0.4995)")
for name in ["2D phi-spiral (100) [identity]", "600-cell 5-NN (120, 4D) [identity]", "E8 5-NN (240, 8D) [identity]"]:
    r221 = min(all_results[name], key=lambda x: abs(x['k'] - 0.221))
    r29 = min(all_results[name], key=lambda x: abs(x['k'] - 2/9))
    print(f"  {name:>38s}: SE(0.221)={r221['SE']:.4f}  SE(2/9)={r29['SE']:.4f}")

print()
print("CHAIN SUMMARY at k = 0.221 / 2/9 / 0.223 lock / 0.225 collapse (stab)")
for name in all_results:
    g = lambda kk, f: min(all_results[name], key=lambda x: abs(x['k'] - kk))[f]
    print(f"  {name:>38s}: SE {g(0.221,'SE'):.4f} / {g(2/9,'SE'):.4f} / lock {g(0.223,'SE'):.4f} "
          f"/ cliff stab {g(0.225,'Stab'):.4f} (0.221 stab {g(0.221,'Stab'):.4f}, amp {g(0.221,'Amp'):.6f})")

with open("/home/claude/metatron_chain_results.json", "w") as f:
    json.dump(all_results, f, indent=1)
print()
print("Saved metatron_chain_results.json")
print("Psi - To preserve the harmonic field.")
