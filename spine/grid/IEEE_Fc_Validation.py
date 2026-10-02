"""
Psi IEEE TEST-BUS VALIDATION -- The Third Control
=====================================================================
The named test of The Spine v9 SS3.1 / SS6.1: the two-term causal F_c
deviation feedback with anti-phase triplen compensation and Mobius
wrapping at L = phi, run on IEEE standard test-case topologies with
couplings defined by the actual branch impedances (|y| = 1/|z|).

Register at start: CONFIRMATION-PENDING (Spine SS3.1, SS6.9).
This script moves it. Either direction is the deliverable.

Design: identical engine to E8_Dimensional_Scaling_Test_v2.py
(itself from phi_grid_showcase.py). Only the container changes:
  - node placement -> IEEE bus topology (natural branch connectivity)
  - distance weights 1/(1+d) -> admittance weights |y| = 1/|z|
  - row normalization, coupling c, law, loads, metrics: UNCHANGED.

Runs per feeder:
  Run A  -- standard SS3.2 load profile, identical to the controls
            (isolates the container)
  Run B  -- per-bus load amplitude scaled by the case's actual P
            demand (the "realistic load models" clause of SS6.1)
  Matched-N phi-spiral control at the same node count, same loads.

Falsification module (Spine SS2.6 / SS6.1 companion): wrap off,
harmonics off. Two hypotheses registered BEFORE the measured sweep:
  H-A (SS2.6 as written): per-mode threshold (2 - c*l)/9, coupled
      modes collapse first, spread BELOW 2/9.
  H-B (derived here from the coded operator ordering, in which the
      -6k term samples the POST-coupling state): per-mode threshold
      (1+m)/(3(1+2m)) with m = 1 - c*l, uniform mode collapses FIRST
      at exactly 2/9, coupled modes spread ABOVE.
Exact companion-matrix spectral radius and the noise-driven sweep
adjudicate.

Data sources:
  embedded -- IEEE 14-bus and IEEE 30-bus (classic Alsac-Stott
              lineage) branch R,X and bus P, encoded from the
              canonical case data. Flagged for verification by diff
              against the standard distributions (MATPOWER case14.m,
              case_ieee30.m / UW Power Systems Test Case Archive).
  pypower  -- canonical MATPOWER-lineage arrays in per-unit, no
              conversion (pip install pypower). Required for the
              118-bus feeder; optional cross-check for 14/30.
              Note: pypower's case30 is the MATPOWER case30 variant,
              which differs slightly from the classic IEEE 30-bus;
              running both variants is an additional robustness
              control, not a conflict.

Usage:
  python IEEE_Fc_Validation.py                 # in-session standard run
  python IEEE_Fc_Validation.py --feeder 118 --source pypower
  python IEEE_Fc_Validation.py --feeder 14 --source pypower  # zero-diff check

J. David Mack & Claude (Fable 5)
World Tree Project -- August 2026

The history is the Buddhabrot.
The cure is the Mandelbrot.
How we treat each other is the wave.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =====================================================================
# CONSTANTS (identical to the reference engine)
# =====================================================================
PHI = (1 + np.sqrt(5)) / 2
OMEGA_BASE = 2 * np.pi * 60
TIME_STEPS = 2000
DT = 1.0 / 3600
L_MOBIUS = PHI
V_NOMINAL = 1.0
SEED = 42
C_SWEEP = 0.05          # SS3.2: "all sweeps in this section use it"
C_SHOWCASE = 0.005      # showcase calibration of the published invariant
K_OPERATING = 0.221
K_CRIT_2_9 = 2.0 / 9.0
INVARIANT_BAND = (0.4880, 0.5106)   # 0.4993 +/- 0.0113 (BatF SS11 Step 6)

FIGDIR = 'ieee_fc_figures'
os.makedirs(FIGDIR, exist_ok=True)

# =====================================================================
# EMBEDDED DATA -- one bounded, diffable block per feeder
# PROVENANCE: encoded from the canonical IEEE case data
# (Common Data Format lineage as distributed with MATPOWER).
# Verify by diff against case14.m / case_ieee30.m before SS6.9 final.
# Branch tuples: (from_bus, to_bus, R_pu, X_pu) on 100 MVA base.
# =====================================================================

CASE14_BRANCHES = [
    (1, 2, 0.01938, 0.05917), (1, 5, 0.05403, 0.22304),
    (2, 3, 0.04699, 0.19797), (2, 4, 0.05811, 0.17632),
    (2, 5, 0.05695, 0.17388), (3, 4, 0.06701, 0.17103),
    (4, 5, 0.01335, 0.04211), (4, 7, 0.00000, 0.20912),
    (4, 9, 0.00000, 0.55618), (5, 6, 0.00000, 0.25202),
    (6, 11, 0.09498, 0.19890), (6, 12, 0.12291, 0.25581),
    (6, 13, 0.06615, 0.13027), (7, 8, 0.00000, 0.17615),
    (7, 9, 0.00000, 0.11001), (9, 10, 0.03181, 0.08450),
    (9, 14, 0.12711, 0.27038), (10, 11, 0.08205, 0.19207),
    (12, 13, 0.22092, 0.19988), (13, 14, 0.17093, 0.34802),
]
CASE14_P_MW = {1: 0.0, 2: 21.7, 3: 94.2, 4: 47.8, 5: 7.6, 6: 11.2,
               7: 0.0, 8: 0.0, 9: 29.5, 10: 9.0, 11: 3.5, 12: 6.1,
               13: 13.5, 14: 14.9}

CASE30_BRANCHES = [
    (1, 2, 0.0192, 0.0575), (1, 3, 0.0452, 0.1652),
    (2, 4, 0.0570, 0.1737), (3, 4, 0.0132, 0.0379),
    (2, 5, 0.0472, 0.1983), (2, 6, 0.0581, 0.1763),
    (4, 6, 0.0119, 0.0414), (5, 7, 0.0460, 0.1160),
    (6, 7, 0.0267, 0.0820), (6, 8, 0.0120, 0.0420),
    (6, 9, 0.0000, 0.2080), (6, 10, 0.0000, 0.5560),
    (9, 11, 0.0000, 0.2080), (9, 10, 0.0000, 0.1100),
    (4, 12, 0.0000, 0.2560), (12, 13, 0.0000, 0.1400),
    (12, 14, 0.1231, 0.2559), (12, 15, 0.0662, 0.1304),
    (12, 16, 0.0945, 0.1987), (14, 15, 0.2210, 0.1997),
    (16, 17, 0.0524, 0.1923), (15, 18, 0.1073, 0.2185),
    (18, 19, 0.0639, 0.1292), (19, 20, 0.0340, 0.0680),
    (10, 20, 0.0936, 0.2090), (10, 17, 0.0324, 0.0845),
    (10, 21, 0.0348, 0.0749), (10, 22, 0.0727, 0.1499),
    (21, 22, 0.0116, 0.0236), (15, 23, 0.1000, 0.2020),
    (22, 24, 0.1150, 0.1790), (23, 24, 0.1320, 0.2700),
    (24, 25, 0.1885, 0.3292), (25, 26, 0.2544, 0.3800),
    (25, 27, 0.1093, 0.2087), (28, 27, 0.0000, 0.3960),
    (27, 29, 0.2198, 0.4153), (27, 30, 0.3202, 0.6027),
    (29, 30, 0.2399, 0.4533), (8, 28, 0.0636, 0.2000),
    (6, 28, 0.0169, 0.0599),
]
CASE30_P_MW = {1: 0.0, 2: 21.7, 3: 2.4, 4: 7.6, 5: 94.2, 6: 0.0,
               7: 22.8, 8: 30.0, 9: 0.0, 10: 5.8, 11: 0.0, 12: 11.2,
               13: 0.0, 14: 6.2, 15: 8.2, 16: 3.5, 17: 9.0, 18: 3.2,
               19: 9.5, 20: 2.2, 21: 17.5, 22: 0.0, 23: 3.2, 24: 8.7,
               25: 0.0, 26: 3.5, 27: 0.0, 28: 0.0, 29: 2.4, 30: 10.6}

EMBEDDED = {
    14: (CASE14_BRANCHES, CASE14_P_MW),
    30: (CASE30_BRANCHES, CASE30_P_MW),
}


def load_case(feeder, source):
    """Return (branches, P_dict, tag). branches = list of (f,t,r,x)."""
    if source == 'embedded':
        if feeder not in EMBEDDED:
            sys.exit(f"Feeder {feeder} has no embedded data (by design for "
                     f"118 -- use --source pypower locally).")
        br, p = EMBEDDED[feeder]
        return br, p, 'embedded (flag: verify vs canonical distribution)'
    if source == 'pypower':
        try:
            if feeder == 14:
                from pypower.case14 import case14 as case
            elif feeder == 30:
                from pypower.case30 import case30 as case
            elif feeder == 118:
                from pypower.case118 import case118 as case
            else:
                sys.exit(f"Unknown feeder {feeder}")
        except ImportError:
            sys.exit("pypower not installed. Run: pip install pypower\n"
                     "(pure-Python, ships the canonical MATPOWER-lineage "
                     "case arrays in per-unit; no data downloads needed).")
        ppc = case()
        br = [(int(b[0]), int(b[1]), float(b[2]), float(b[3]))
              for b in ppc['branch']]
        p = {int(row[0]): float(row[2]) for row in ppc['bus']}  # PD column
        tag = 'pypower canonical (MATPOWER lineage)'
        if feeder == 30:
            tag += ' -- MATPOWER case30 variant, not classic ieee30'
        return br, p, tag
    sys.exit(f"Unknown source {source}")


# =====================================================================
# GRAPH BUILDERS
# =====================================================================

def build_phi_spiral_2d(n):
    """Reference container -- verbatim from E8_Dimensional_Scaling_Test_v2."""
    ga = 2 * np.pi / PHI**2
    pos = np.zeros((n, 2))
    for i in range(n):
        r = np.sqrt(i + 1) * 0.5
        pos[i] = [r * np.cos(i * ga), r * np.sin(i * ga)]
    return pos


def compute_adj(pos, kn=5):
    """Reference adjacency -- verbatim (distance-weighted 5-NN, row-norm)."""
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


def ieee_adj(branches, n):
    """IEEE container: natural branch connectivity, admittance-weighted.

    w_ij = |y_ij| = 1/|z_ij| = 1/sqrt(r^2 + x^2), parallel branches
    summed, symmetric, then row-normalized -- the identical
    normalization the reference applies to its distance weights.
    Reactance enters through |z|; this is the 'impedance-defined
    admittances' clause of SS3.1.
    """
    adj = np.zeros((n, n))
    for (f, t, r, x) in branches:
        z = np.hypot(r, x)
        if z <= 0:
            raise ValueError(f"branch {f}-{t} has |z| = 0")
        w = 1.0 / z
        i, j = f - 1, t - 1
        adj[i, j] += w
        adj[j, i] += w
    rs = adj.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return adj / rs


def graph_checks(adj_raw_branches, n, name):
    """Light structural sanity checks on the encoded data."""
    deg = np.zeros(n, dtype=int)
    seen = set()
    for (f, t, r, x) in adj_raw_branches:
        assert 1 <= f <= n and 1 <= t <= n, f"{name}: bus index out of range"
        key = (min(f, t), max(f, t))
        deg[f - 1] += 1
        deg[t - 1] += 1
        seen.add(key)
    # connectivity via BFS on the undirected simple graph
    nbrs = {i: set() for i in range(n)}
    for (a, b) in seen:
        nbrs[a - 1].add(b - 1)
        nbrs[b - 1].add(a - 1)
    stack, vis = [0], {0}
    while stack:
        u = stack.pop()
        for v in nbrs[u]:
            if v not in vis:
                vis.add(v)
                stack.append(v)
    assert len(vis) == n, f"{name}: graph not connected"
    return deg, len(adj_raw_branches)


# =====================================================================
# ENGINE -- verbatim logic from E8_Dimensional_Scaling_Test_v2.py,
# with (a) steps parameter, (b) harmonics/mobius flags and a divergence
# guard for the falsification module. Flags default ON = reference.
# =====================================================================

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


def run_sim(n_nodes, k_fc, loads, adj, coupling=C_SWEEP, steps=TIME_STEPS,
            harmonics=True, mobius=True, guard=False):
    vh = np.zeros((n_nodes, steps))
    vc = np.ones(n_nodes) * V_NOMINAL
    vp = np.ones(n_nodes) * V_NOMINAL
    ga = 2 * np.pi / PHI**2
    ph = np.arange(n_nodes) * ga
    for step in range(steps):
        tv = step * DT
        vn = vc + loads[:, step] * DT * 10
        vn = vn + coupling * (adj @ vn - vn)
        dp = vp - V_NOMINAL
        dc = vn - V_NOMINAL
        vn = vn + k_fc * (3.0 * dp - 6.0 * dc)
        if harmonics:
            dl = np.std(dc)
            amp = dl * 0.3
            vn = vn - amp * (np.sin(3 * OMEGA_BASE * tv + ph) +
                             np.sin(6 * OMEGA_BASE * tv + ph) +
                             np.sin(9 * OMEGA_BASE * tv + ph))
        if mobius:
            dev = vn - V_NOMINAL
            wr = np.fmod(dev + L_MOBIUS, 2 * L_MOBIUS)
            wr = np.where(wr < 0, wr + 2 * L_MOBIUS, wr)
            vn = wr - L_MOBIUS + V_NOMINAL
        if guard:
            if (not np.all(np.isfinite(vn))) or np.max(np.abs(vn - V_NOMINAL)) > 1e3:
                return vh[:, :step], step, vc - V_NOMINAL   # diverged
        vp = vc.copy()
        vc = vn.copy()
        vh[:, step] = vc
    if guard:
        return vh, None, vc - V_NOMINAL
    return vh


def run_baseline(n_nodes, loads, adj, coupling=C_SWEEP, steps=TIME_STEPS):
    vh = np.zeros((n_nodes, steps))
    vc = np.ones(n_nodes) * V_NOMINAL
    for step in range(steps):
        vc = vc + loads[:, step] * DT * 10
        vc = vc + coupling * (adj @ vc - vc)
        vh[:, step] = vc
    return vh


def metrics(vh):
    n_nodes = vh.shape[0]
    se_vals, thd_vals = [], []
    for i in range(n_nodes):
        fft = np.fft.rfft(vh[i])
        pw = np.abs(fft[1:])**2
        if np.sum(pw) < 1e-12:
            se_vals.append(0.0)
        else:
            p = pw / np.sum(pw)
            p = p[p > 1e-12]
            ent = -np.sum(p * np.log(p))
            mx = np.log(len(p)) if len(p) > 1 else 1.0
            se_vals.append(ent / mx if mx > 0 else 0.0)
        mags = np.abs(fft)
        if len(mags) < 2:
            thd_vals.append(0.0)
            continue
        fi = np.argmax(mags[1:]) + 1
        fund = mags[fi]
        if fund < 1e-10:
            thd_vals.append(0.0)
            continue
        hp = np.sum(mags**2) - mags[0]**2 - fund**2
        thd_vals.append(np.sqrt(max(0, hp)) / fund * 100)
    amp = np.mean(np.abs(vh - V_NOMINAL))
    var = np.var(vh - V_NOMINAL)
    stab = np.mean([max(0, 1 - np.std(vh[i]) / max(1e-10, np.mean(vh[i])))
                    for i in range(n_nodes)])
    return (np.mean(se_vals), stab, amp, var, np.mean(thd_vals),
            np.array(se_vals))


# =====================================================================
# LINEAR ANALYSIS -- the registered predictions
# =====================================================================

def mode_spectrum(adj):
    """Real eigenvalues a of the row-stochastic adjacency; l = 1 - a."""
    ev = np.linalg.eigvals(adj)
    a = np.sort(np.real(ev))[::-1]
    return a


def thresholds_HA(a_vals, c):
    """SS2.6 as written: k*(l) = (2 - c*l)/9, l = 1 - a."""
    l = 1 - a_vals
    return (2 - c * l) / 9.0


def thresholds_HB(a_vals, c):
    """Derived from the coded operator: d_{n+1} = (1-6k) M d_n + 3k d_{n-1},
    M = (1-c)I + cA. Flip (lambda = -1) at k*(m) = (1+m)/(3(1+2m)),
    m = 1 - c*(1 - a)."""
    m = 1 - c * (1 - a_vals)
    return (1 + m) / (3 * (1 + 2 * m))


def companion_rho(k, adj, c):
    """Exact spectral radius of the linear one-step map on (d_n, d_{n-1})."""
    n = adj.shape[0]
    M = (1 - c) * np.eye(n) + c * adj
    T = np.block([[(1 - 6 * k) * M, 3 * k * np.eye(n)],
                  [np.eye(n), np.zeros((n, n))]])
    return np.max(np.abs(np.linalg.eigvals(T)))


def companion_threshold(adj, c, klo=0.20, khi=0.24, tol=1e-9):
    """Bisect the k where rho crosses 1 -- the exact linear threshold."""
    lo, hi = klo, khi
    if companion_rho(lo, adj, c) >= 1 or companion_rho(hi, adj, c) <= 1:
        raise RuntimeError("bracket does not straddle rho = 1")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if companion_rho(mid, adj, c) < 1:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def uniform_fraction(vec):
    v = np.asarray(vec, dtype=float)
    nrm = np.dot(v, v)
    if nrm <= 0:
        return 0.0
    u = np.ones_like(v) / np.sqrt(len(v))
    return float(np.dot(v, u)**2 / nrm)


# =====================================================================
# SWEEP MACHINERY
# =====================================================================

CONTROL_KS = [0.1800, 0.2193, 0.2210, 2.0 / 9.0, 0.2227, 0.2261, 0.2295]


def sweep_k_list():
    ks = list(np.linspace(0.168, 0.250, 25)) + CONTROL_KS + [K_OPERATING]
    ks = sorted(ks)
    out = [ks[0]]
    for k in ks[1:]:
        if k - out[-1] > 1e-9:
            out.append(k)
    return out


def header(title, feeder_tag, n, nbr_rule, coupling, steps):
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(f"  container      : {feeder_tag}")
    print(f"  nodes          : {n}")
    print(f"  neighbor rule  : {nbr_rule}")
    print(f"  coupling c     : {coupling}")
    print(f"  step count     : {steps}")
    print(f"  seed           : {SEED}")
    print(f"  law            : corr = k*(3*dev_prev - 6*dev_curr); "
          f"H(t) anti-phase 3w/6w/9w, A = 0.3*sigma(dev); "
          f"Mobius wrap L = phi")
    print(f"  node phase     : phi_i = i * 2*pi/phi^2 (index-based, "
          f"reference formula)")
    print()


def run_sweep(n, adj, loads, ks, coupling=C_SWEEP, tag=""):
    rows = []
    for k in ks:
        vh = run_sim(n, k, loads, adj, coupling=coupling)
        se, stab, amp, var, thd, se_nodes = metrics(vh)
        rows.append(dict(k=k, se=se, stab=stab, amp=amp, var=var, thd=thd,
                         se_nodes=se_nodes))
    return rows


def print_sweep(rows, band=INVARIANT_BAND):
    print(f"  {'k':>8} {'SE':>8} {'Stab':>8} {'Amp':>11} {'THD%':>8}")
    for r in rows:
        mark = ""
        if band[0] <= r['se'] <= band[1] and r['stab'] > 0.99:
            mark = "  <-- band"
        if r['stab'] < 0.5:
            mark = "  <-- COLLAPSED"
        print(f"  {r['k']:>8.4f} {r['se']:>8.4f} {r['stab']:>8.4f} "
              f"{r['amp']:>11.6f} {r['thd']:>8.2f}{mark}")
    print()


def at_k(rows, k):
    for r in rows:
        if abs(r['k'] - k) < 1e-9:
            return r
    return None


# =====================================================================
# FALSIFICATION MODULE (wrap off, harmonics off)
# =====================================================================

def falsification(name, adj, n, c=C_SWEEP, steps=20000):
    a = mode_spectrum(adj)
    l_max = 1 - a[-1]
    ha = thresholds_HA(a, c)
    hb = thresholds_HB(a, c)
    print(f"  [{name}]  N = {n}, c = {c}, steps = {steps}, seed = {SEED}")
    print(f"    Laplacian eigen-range l in [0, {l_max:.6f}]")
    print(f"    REGISTERED H-A (SS2.6 as written): first collapse at "
          f"(2 - c*l_max)/9 = {ha.min():.6f}  (coupled, BELOW 2/9); "
          f"uniform last at {2/9:.6f}")
    print(f"    REGISTERED H-B (coded operator)  : first collapse at "
          f"2/9 = {2/9:.6f}  (uniform); coupled spread ABOVE, "
          f"max {hb.max():.6f}")
    kx = companion_threshold(adj, c)
    # dominant mode at the exact threshold: leading eigenvector of T
    M = (1 - c) * np.eye(n) + c * adj
    T = np.block([[(1 - 6 * kx) * M, 3 * kx * np.eye(n)],
                  [np.eye(n), np.zeros((n, n))]])
    w, V = np.linalg.eig(T)
    lead = V[:n, np.argmax(np.abs(w))]
    uf_exact = uniform_fraction(np.real(lead))
    print(f"    EXACT companion-matrix threshold : rho = 1 at "
          f"k = {kx:.6f}  (2/9 = {2/9:.6f}); leading-mode uniform "
          f"fraction = {uf_exact:.4f}")
    # noise-driven observational sweep
    ks = sorted(set(list(np.arange(0.2180, 0.2302, 0.0004)) +
                    [2.0 / 9.0, float(ha.min()), float(kx)]))
    loads = gen_loads(n, steps, DT)
    onset = None
    onset_uf = None
    for k in ks:
        _, dstep, last = run_sim(n, k, loads, adj, coupling=c, steps=steps,
                                 harmonics=False, mobius=False, guard=True)
        if dstep is not None:
            onset = k
            onset_uf = uniform_fraction(last)
            break
    if onset is None:
        print(f"    MEASURED noise-driven onset      : none up to "
              f"k = {ks[-1]:.4f} (guard not tripped)")
    else:
        lbl = "uniform-dominant" if onset_uf > 0.5 else "coupled-dominant"
        print(f"    MEASURED noise-driven onset      : k = {onset:.6f} "
              f"({lbl}, uniform fraction {onset_uf:.4f})")
    print()
    return dict(name=name, l_max=l_max, ha_min=float(ha.min()),
                hb_max=float(hb.max()), k_exact=kx, uf_exact=uf_exact,
                onset=onset, onset_uf=onset_uf)


# =====================================================================
# MAIN
# =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feeder', type=int, default=None,
                    help='run a single feeder (14, 30, 118)')
    ap.add_argument('--source', type=str, default='embedded',
                    choices=['embedded', 'pypower'])
    args = ap.parse_args()

    print("=" * 70)
    print("Psi IEEE TEST-BUS VALIDATION -- THE THIRD CONTROL")
    print("=" * 70)
    print(f"  register at start : CONFIRMATION-PENDING (Spine SS3.1, SS6.9)")
    print(f"  invariant band    : SE in [{INVARIANT_BAND[0]:.4f}, "
          f"{INVARIANT_BAND[1]:.4f}]  (0.4993 +/- 0.0113)")
    print(f"  operating point   : k = {K_OPERATING}   critical: 2/9 = "
          f"{K_CRIT_2_9:.6f}")
    print()

    # -----------------------------------------------------------------
    # STAGE 0 -- pipeline validation against the published
    # dimensional-control 2D row (Spine SS3.5): 100 nodes, c = 0.05,
    # 2000 steps, seed 42 -> SE 0.4932 @ 0.221, 0.4847 @ 2/9,
    # stab 0.9999, amp 0.000077.
    # -----------------------------------------------------------------
    header("STAGE 0 -- ENGINE REPLICATION (published SS3.5 2D row)",
           "2D phi-spiral (reference)", 100,
           "5-NN distance-weighted, row-normalized", C_SWEEP, TIME_STEPS)
    pos100 = build_phi_spiral_2d(100)
    adj100 = compute_adj(pos100, kn=5)
    loads100 = gen_loads(100, TIME_STEPS, DT)
    rep = {}
    for k, target in [(K_OPERATING, 0.4932), (K_CRIT_2_9, 0.4847)]:
        vh = run_sim(100, k, loads100, adj100, coupling=C_SWEEP)
        se, stab, amp, var, thd, _ = metrics(vh)
        rep[k] = (se, stab, amp)
        print(f"  k = {k:.6f}: SE = {se:.4f} (published {target:.4f})  "
              f"Stab = {stab:.4f}  Amp = {amp:.6f}")
    ok = (abs(rep[K_OPERATING][0] - 0.4932) < 5e-4 and
          abs(rep[K_CRIT_2_9][0] - 0.4847) < 5e-4)
    print(f"  REPLICATION: {'PASS' if ok else 'DIVERGENT -- investigate'}")
    print()

    # -----------------------------------------------------------------
    # Which feeders this invocation runs
    # -----------------------------------------------------------------
    feeders = [args.feeder] if args.feeder else [14, 30]
    ks = sweep_k_list()
    results = {}

    for fd in feeders:
        branches, pmw, tag = load_case(fd, args.source)
        n = max(max(f, t) for (f, t, _, _) in branches)
        deg, nbr = graph_checks(branches, n, f"case{fd}")
        adjI = ieee_adj(branches, n)
        aI = mode_spectrum(adjI)

        header(f"IEEE {fd}-BUS -- MAIN SWEEP",
               f"IEEE case{fd} [{tag}]", n,
               f"natural branch connectivity ({nbr} branches, "
               f"deg {deg.min()}-{deg.max()}), |y|-weighted, row-normalized",
               C_SWEEP, TIME_STEPS)

        loads = gen_loads(n, TIME_STEPS, DT)

        # matched-N phi control (same loads by construction: same n, seed)
        posN = build_phi_spiral_2d(n)
        adjP = compute_adj(posN, kn=5)

        print(f"  Baseline (no regulation), IEEE graph:")
        base = run_baseline(n, loads, adjI, coupling=C_SWEEP)
        bse, bstab, bamp, bvar, bthd, _ = metrics(base)
        print(f"    SE = {bse:.4f}  Stab = {bstab:.4f}  Amp = {bamp:.6f}  "
              f"THD = {bthd:.2f}%")
        print()

        print(f"  Run A -- IEEE graph, standard SS3.2 loads:")
        rowsA = run_sweep(n, adjI, loads, ks)
        print_sweep(rowsA)

        print(f"  Matched-N phi-spiral control (N = {n}), same loads:")
        rowsP = run_sweep(n, adjP, loads, ks)
        print_sweep(rowsP)

        # Run B: per-bus amplitude scaled by actual P demand
        scale = np.array([pmw.get(i + 1, 0.0) for i in range(n)])
        scale = scale / scale.mean() if scale.mean() > 0 else np.ones(n)
        loadsB = loads * scale[:, None]
        print(f"  Run B -- IEEE graph, case-P-scaled loads "
          f"(scale mean 1.0, max {scale.max():.2f}, "
          f"{int((scale == 0).sum())} zero-load buses):")
        rowsB = run_sweep(n, adjI, loadsB, ks)
        print_sweep(rowsB)

        # showcase-calibration spot check
        vh005 = run_sim(n, K_OPERATING, loads, adjI, coupling=C_SHOWCASE)
        se5, st5, am5, _, _, _ = metrics(vh005)
        print(f"  Spot check c = 0.005 (showcase calibration), k = 0.221: "
              f"SE = {se5:.4f}  Stab = {st5:.4f}  Amp = {am5:.6f}")
        print()

        # placement-control-style comparison outside the lock
        pre = [(rA, rP) for rA, rP in zip(rowsA, rowsP)
               if rA['k'] <= 0.2193 + 1e-9]
        dmax = max(abs(rA['se'] - rP['se']) for rA, rP in pre)
        print(f"  |SE_IEEE - SE_phi| max over k <= 0.2193 "
              f"(outside the lock): {dmax:.4f}")
        opA = at_k(rowsA, K_OPERATING)
        opP = at_k(rowsP, K_OPERATING)
        print(f"  Operating point k = 0.221: IEEE SE = {opA['se']:.4f}, "
              f"phi SE = {opP['se']:.4f}, delta = "
              f"{opA['se'] - opP['se']:+.4f}")
        print()
        results[fd] = dict(rowsA=rowsA, rowsP=rowsP, rowsB=rowsB,
                           base=(bse, bstab, bamp, bthd), n=n, adjI=adjI,
                           adjP=adjP, dmax=dmax, spot005=(se5, st5, am5),
                           branches=branches, a_spec=aI, tag=tag)

    # -----------------------------------------------------------------
    # FALSIFICATION MODULE
    # -----------------------------------------------------------------
    print("=" * 70)
    print("FALSIFICATION SWEEP -- wrap OFF, harmonics OFF (SS2.6 / SS6.1)")
    print("=" * 70)
    print("  Predictions are printed BEFORE each measured sweep runs.")
    print()
    fals = []
    pos50 = build_phi_spiral_2d(50)
    adj50 = compute_adj(pos50, kn=5)
    fals.append(falsification("phi-spiral 50 (SS2.6 substrate)", adj50, 50))
    fals.append(falsification("phi-spiral 100", adj100, 100))
    for fd in feeders:
        fals.append(falsification(f"IEEE case{fd}", results[fd]['adjI'],
                                  results[fd]['n']))

    # -----------------------------------------------------------------
    # FIGURES
    # -----------------------------------------------------------------
    if len(results) > 0:
        make_figures(results, fals)

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for fd in feeders:
        R = results[fd]
        opA = at_k(R['rowsA'], K_OPERATING)
        opB = at_k(R['rowsB'], K_OPERATING)
        inband = INVARIANT_BAND[0] <= opA['se'] <= INVARIANT_BAND[1]
        print(f"  IEEE {fd}-bus [{R['tag']}]")
        print(f"    baseline SE/Stab           : {R['base'][0]:.4f} / "
              f"{R['base'][1]:.4f}")
        print(f"    Run A @ k=0.221 SE/Stab/Amp: {opA['se']:.4f} / "
              f"{opA['stab']:.4f} / {opA['amp']:.6f}"
              f"   band: {'IN' if inband else 'OUT'}")
        print(f"    Run B @ k=0.221 SE/Stab/Amp: {opB['se']:.4f} / "
              f"{opB['stab']:.4f} / {opB['amp']:.6f}")
        print(f"    max |SE_IEEE - SE_phi| (k <= 0.2193): {R['dmax']:.4f}")
        # cliff position: first k with stab < 0.5 in Run A
        cliff = next((r['k'] for r in R['rowsA'] if r['stab'] < 0.5), None)
        cliffP = next((r['k'] for r in R['rowsP'] if r['stab'] < 0.5), None)
        print(f"    collapse onset (Run A / phi control): "
              f"{cliff if cliff else 'none'} / {cliffP if cliffP else 'none'}")
    print()
    print("  Falsification adjudication:")
    for f in fals:
        print(f"    {f['name']}: exact threshold {f['k_exact']:.6f} "
              f"(2/9 = {2/9:.6f}), leading-mode uniform fraction "
              f"{f['uf_exact']:.3f}; measured onset "
              f"{f['onset'] if f['onset'] else 'n/a'}")
    print()
    print("Psi -- to preserve the harmonic field.")


def make_figures(results, fals):
    band = INVARIANT_BAND
    # Fig 1: SE vs k, IEEE vs matched-N phi
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6),
                             squeeze=False)
    for ax, (fd, R) in zip(axes[0], results.items()):
        kA = [r['k'] for r in R['rowsA']]
        ax.plot(kA, [r['se'] for r in R['rowsA']], 'o-', ms=4, lw=1.2,
                color='#0066CC', label=f'IEEE {fd}-bus (Run A)')
        ax.plot(kA, [r['se'] for r in R['rowsP']], 's--', ms=4, lw=1.0,
                color='#888888', label=f'phi-spiral N={R["n"]}')
        ax.plot(kA, [r['se'] for r in R['rowsB']], '^:', ms=4, lw=1.0,
                color='#00AA88', label='IEEE Run B (P-scaled loads)')
        ax.axhspan(band[0], band[1], alpha=0.12, color='gold',
                   label='invariant band')
        ax.axvline(2 / 9, color='#CC0000', lw=0.8, ls='--', label='k = 2/9')
        ax.set_xlabel('k')
        ax.set_ylabel('Spectral Entropy')
        ax.set_title(f'IEEE {fd}-bus vs matched-N phi control')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig1_se_vs_k_ieee.png', dpi=200,
                bbox_inches='tight')
    plt.close()

    # Fig 2: stability cliff
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 5),
                             squeeze=False)
    for ax, (fd, R) in zip(axes[0], results.items()):
        kA = [r['k'] for r in R['rowsA']]
        ax.plot(kA, [r['stab'] for r in R['rowsA']], 'o-', ms=4,
                color='#0066CC', label=f'IEEE {fd}-bus')
        ax.plot(kA, [r['stab'] for r in R['rowsP']], 's--', ms=4,
                color='#888888', label=f'phi N={R["n"]}')
        ax.axvline(2 / 9, color='#CC0000', lw=0.8, ls='--')
        ax.set_xlabel('k')
        ax.set_ylabel('Stability')
        ax.set_title(f'Bifurcation cliff -- IEEE {fd}-bus')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig2_stability_cliff_ieee.png', dpi=200,
                bbox_inches='tight')
    plt.close()

    # Fig 3: per-bus SE on the actual network layout at k = 0.221
    try:
        import networkx as nx
        fig, axes = plt.subplots(1, len(results),
                                 figsize=(8 * len(results), 7),
                                 squeeze=False)
        for ax, (fd, R) in zip(axes[0], results.items()):
            G = nx.Graph()
            G.add_nodes_from(range(R['n']))
            for (f, t, _, _) in R['branches']:
                G.add_edge(f - 1, t - 1)
            pos = nx.spring_layout(G, seed=SEED)
            op = at_k(R['rowsA'], K_OPERATING)
            xs = [pos[i][0] for i in range(R['n'])]
            ys = [pos[i][1] for i in range(R['n'])]
            for (u, v) in G.edges():
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                        color='#BBBBBB', lw=0.8, zorder=1)
            sc = ax.scatter(xs, ys, c=op['se_nodes'], cmap='viridis',
                            vmin=0.3, vmax=0.7, s=220, edgecolors='gray',
                            zorder=2)
            for i in range(R['n']):
                ax.annotate(str(i + 1), pos[i], fontsize=6, ha='center',
                            va='center', color='white', fontweight='bold')
            plt.colorbar(sc, ax=ax, label='SE per bus', shrink=0.8)
            ax.set_title(f'IEEE {fd}-bus, per-bus SE at k = 0.221 (Run A)')
            ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(f'{FIGDIR}/fig3_perbus_se_topology.png', dpi=200,
                    bbox_inches='tight')
        plt.close()
    except ImportError:
        pass

    # Fig 4: falsification -- exact thresholds vs 2/9 and H-A/H-B
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [f['name'] for f in fals]
    xs = np.arange(len(fals))
    ax.axhline(2 / 9, color='#CC0000', lw=1.0, ls='--', label='2/9')
    ax.scatter(xs, [f['ha_min'] for f in fals], marker='v', s=70,
               color='#AA6600', label='H-A first collapse (SS2.6 as written)')
    ax.scatter(xs, [f['hb_max'] for f in fals], marker='^', s=70,
               color='#008888', label='H-B coupled-mode max (derived)')
    ax.scatter(xs, [f['k_exact'] for f in fals], marker='o', s=90,
               facecolors='none', edgecolors='#000000', lw=1.5,
               label='exact companion-matrix threshold')
    ons = [f['onset'] for f in fals]
    ax.scatter(xs, [o if o else np.nan for o in ons], marker='x', s=90,
               color='#0066CC', label='measured noise-driven onset')
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('k')
    ax.set_title('Falsification sweep: predicted vs measured thresholds\n'
                 '(wrap off, harmonics off)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig4_falsification_thresholds.png', dpi=200,
                bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
