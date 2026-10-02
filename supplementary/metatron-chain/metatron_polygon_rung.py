"""metatron_polygon_rung.py — the 2D rung of the projection chain in the F_c dynamics.

Closed form (derived before measurement, EXPECTATIONS_prestated.md):
in the complex window, |lambda|^2 = m = 3k/(9k-1), alpha = -(m-1), beta = -m,
and the eigenvalue rotation angle theta satisfies cos(theta) = -(m-1)/(2*sqrt(m)).
For a target theta, u = |lambda| solves u^2 + 2*cos(theta)*u - 1 = 0, and
k = m/(9m-3) with m = u^2. Reachable range: m in (1/3, inf) => theta in
(arccos(1/sqrt(3)) ~ 54.74 deg, 180 deg).

Special gains (the polygon ladder): the commensurate rotation angles select
n-fold directional symmetry classes. The 72/108/144-degree members are the H2
(pentagonal/decagonal) family — the 2D shadow of the H3 -> H4 -> E8 chain.
Two land on golden moduli exactly: theta = 120 deg <=> |lambda| = phi;
theta = 60 deg <=> |lambda| = 1/phi.

Measurement: BatF §10 cycle verbatim (fc_k_sweep_reference.py mechanics,
seed 42, C0 = 0.3, C-1 = 0.5, Ah = 0.1, noise 0.2, L = phi), 200,000 steps
per gain. Readouts:
  R1  per-step rotation: circular mean of the normal-frame angle increment
      over consecutive Wind-0 pairs, against the derived theta(k).
  R2  angular power spectrum of the Wind-0 normal-frame density (720 bins);
      dominant low mode against the commensurability order q.
  R3  4D delay-embedding participation ratio (node Computation 1, honest form).

J. David Mack & Claude (Fable 5) - World Tree Project - August 2026 - seed 42
"""
import numpy as np
import math, json

PHI = (1 + 5**0.5) / 2
PHI_INV = 1 / PHI
N_STEPS = 200_000
SEED = 42
C0, CM1 = 0.3, 0.5
AH, NOISE_SCALE, RES_REF = 0.1, 0.2, 1.0
DT, OMEGA = 0.01, 2 * math.pi
L = PHI

def alpha(k): return (1 - 6 * k) / (1 - 9 * k)
def beta(k):  return 3 * k / (1 - 9 * k)

def k_for_theta(theta_deg):
    c = math.cos(math.radians(theta_deg))
    u = -c + math.sqrt(c * c + 1)          # positive root of u^2 + 2c u - 1 = 0
    m = u * u
    return m / (9 * m - 3), u, m

# ------------------------------------------------------------------
# The ladder — derived, then eigenvalue-verified
# ------------------------------------------------------------------
LADDER = [
    ("12-fold dodecagon", 150.0, 12),
    ("5-fold pentagram",  144.0, 5),
    ("8-fold octagon",    135.0, 8),
    ("3-fold triangle",   120.0, 3),   # |lambda| = phi exactly
    ("10-fold decagon",   108.0, 10),
    ("kphi control",      None,  None),  # incommensurate, |lambda|^2 = phi
    ("4-fold square",     90.0,  4),   # k = 1/6, the published anchor
    ("5-fold pentagon",   72.0,  5),   # convergent side
    ("6-fold hexagon",    60.0,  6),   # |lambda| = 1/phi exactly
]

print("=" * 88)
print("THE POLYGON LADDER — derived gains, eigenvalue-verified")
print("=" * 88)
rows = []
for name, th, q in LADDER:
    if th is None:
        k = PHI / (9 * PHI - 3)
        m = 3 * k / (9 * k - 1); u = math.sqrt(m)
        th_actual = math.degrees(math.acos(-(m - 1) / (2 * u)))
        q_txt = "incomm."
    else:
        k, u, m = k_for_theta(th)
        th_actual = th
        q_txt = str(q)
    # eigenvalue verification
    a, b = alpha(k), beta(k)
    lam = np.roots([1, -a, -b])
    lam_c = lam[np.argmax(np.abs(np.imag(lam)))]
    r_num = abs(lam_c); th_num = math.degrees(math.atan2(abs(lam_c.imag), lam_c.real))
    rows.append(dict(name=name, k=k, theta=th_actual, q=q_txt, u=u, m=m))
    tag = ""
    if abs(u - PHI) < 1e-12: tag = "  <- |lambda| = phi EXACT"
    if abs(u - PHI_INV) < 1e-12: tag = "  <- |lambda| = 1/phi EXACT"
    if th is None: tag = "  <- |lambda|^2 = phi (kphi)"
    print(f"  {name:>18s}: k = {k:.6f}  theta = {th_actual:9.4f}  q = {q_txt:>7s}  "
          f"|lam| = {u:.6f}  [eig check: r = {r_num:.6f}, theta = {th_num:.4f}]{tag}")
print(f"\n  Domain bound: m -> 1/3+ as k -> inf  =>  theta_min = arccos(1/sqrt(3)) = "
      f"{math.degrees(math.acos(1/math.sqrt(3))):.4f} deg (the magic angle).")
print(f"  Forcing period = 100 steps; q divides 100 for q in (4, 5, 10) — noted, not modeled.")

# ------------------------------------------------------------------
# Trajectory measurement — §10 cycle verbatim + winding + normal frame
# ------------------------------------------------------------------
def normal_frame(k):
    a, b = alpha(k), beta(k)
    disc = a * a + 4 * b
    s = math.sqrt(-disc)                    # complex window: disc < 0
    # M = [[a, b],[1,0]]; lambda = (a + i s)/2; v = (lambda, 1)
    # T columns u = Re v = (a/2, 1), w = Im v = (s/2, 0); T^-1 M T = r R(-theta)
    T = np.array([[a / 2, s / 2], [1.0, 0.0]])
    Tinv = np.linalg.inv(T)
    return Tinv

def run_gain(k, n_steps=N_STEPS):
    rng = np.random.RandomState(SEED)
    a, b = alpha(k), beta(k)
    Tinv = normal_frame(k)
    c_prev, c_curr = CM1, C0
    mm = CM1 * CM1 * PHI_INV + C0 * C0
    wind0 = np.zeros(n_steps, bool)
    xs = np.zeros(n_steps); ys = np.zeros(n_steps)   # state (C_n, C_{n-1}) BEFORE step? we store post-step pair
    cn_hist = np.zeros(n_steps)
    for n in range(1, n_steps + 1):
        t = n * DT
        c_raw = a * c_curr + b * c_prev
        if c_curr < 0:
            c_raw += NOISE_SCALE * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > RES_REF:
            c_raw += (c_curr - RES_REF) * math.sin(OMEGA * t)
        c_raw += AH * (math.sin(3*OMEGA*t) + math.sin(6*OMEGA*t) + math.sin(9*OMEGA*t))
        w0 = abs(c_raw) <= L
        c_next = ((c_raw + L) % (2 * L)) - L
        mm = mm * PHI_INV + c_next * c_next
        i = n - 1
        wind0[i] = w0
        xs[i] = c_next; ys[i] = c_curr          # post-step state vector (C_n+1, C_n)
        cn_hist[i] = c_next
        c_prev, c_curr = c_curr, c_next
    # normal-frame coordinates of post-step states
    Z = Tinv @ np.vstack([xs, ys])
    psi = np.arctan2(Z[1], Z[0])
    return dict(wind0=wind0, psi=psi, cn=cn_hist, Z=Z)

def angular_spectrum(psi, bins=720, max_mode=24):
    h, _ = np.histogram(psi, bins=bins, range=(-np.pi, np.pi))
    h = h.astype(float)
    if h.sum() == 0: return None, None
    h = h / h.sum()
    F = np.fft.rfft(h - h.mean())
    P = np.abs(F[1:max_mode + 1]) ** 2
    P = P / (P.sum() + 1e-30)
    return P, int(np.argmax(P[1:16]) + 2)   # dominant mode among 2..16

def circ_stats(d):
    if len(d) == 0: return None, None
    z = np.exp(1j * d)
    mean = math.degrees(math.atan2(np.mean(np.imag(z)), np.mean(np.real(z))))
    R = abs(np.mean(z))
    return mean, R

print()
print("=" * 88)
print("MEASUREMENT — 200,000 steps per gain, seed 42, BatF §10 cycle")
print("=" * 88)
print(f"{'gain':>18s} {'k':>9s} {'theta_pred':>10s} {'q':>7s} {'W0%':>6s} "
      f"{'|dpsi| meas':>11s} {'circR':>6s} {'dom mode':>8s} {'P(q)/P(dom)':>12s}")

results = {}
for row in rows:
    k = row['k']; name = row['name']
    out = run_gain(k)
    w0 = out['wind0']; psi = out['psi']
    # R1: consecutive Wind-0 pairs
    pair = w0[1:] & w0[:-1]
    dpsi = np.angle(np.exp(1j * (psi[1:][pair] - psi[:-1][pair])))
    dmean, dR = circ_stats(dpsi)
    # R2: angular spectrum of Wind-0 states
    P, dom = angular_spectrum(psi[w0])
    q = row['q']
    pq = ""
    if P is not None and q not in ("incomm.",):
        qi = int(q)
        if 2 <= qi <= 24:
            pq = f"{P[qi - 1]:.3f}/{P[dom - 1]:.3f}"
    results[name] = dict(k=k, theta=row['theta'], q=q, w0_frac=float(w0.mean()),
                         dpsi_mean=dmean, dpsi_R=dR,
                         spectrum=(P.tolist() if P is not None else None), dom_mode=dom,
                         psi_w0=psi[w0][:120000].tolist() if w0.sum() else [],
                         Zx=out['Z'][0][w0][::max(1, w0.sum() // 60000)].tolist(),
                         Zy=out['Z'][1][w0][::max(1, w0.sum() // 60000)].tolist())
    print(f"{name:>18s} {k:>9.6f} {row['theta']:>10.4f} {str(q):>7s} {100*w0.mean():>6.1f} "
          f"{abs(dmean) if dmean is not None else float('nan'):>11.4f} "
          f"{dR if dR is not None else float('nan'):>6.3f} {dom:>8d} {pq:>12s}")

# ------------------------------------------------------------------
# R3: 4D delay-embedding participation ratio (node Computation 1, honest form)
# ------------------------------------------------------------------
print()
print("R3 — 4D delay embedding (C_n, C_n-1, C_n-2, C_n-3): participation ratio of covariance")
for name, kk in [("4-fold square (k=1/6)", 1/6), ("kphi", PHI/(9*PHI-3))]:
    out = run_gain(kk, n_steps=100_000)
    c = out['cn']
    X = np.vstack([c[3:], c[2:-1], c[1:-2], c[:-3]])
    C = np.cov(X)
    ev = np.linalg.eigvalsh(C)
    pr = (ev.sum() ** 2) / (np.sum(ev ** 2))
    print(f"  {name:>22s}: eigenvalues {np.round(ev[::-1], 5)}  PR = {pr:.3f}  (expect ~2)")

with open("/home/claude/polygon_rung_results.json", "w") as f:
    json.dump({k2: {kk: vv for kk, vv in v2.items() if kk not in ("psi_w0",)}
               for k2, v2 in results.items()}, f)
np.save("/home/claude/polygon_rung_psi.npy",
        np.array([results[r['name']]['k'] for r in rows]))
with open("/home/claude/polygon_rung_full.json", "w") as f:
    json.dump(results, f)
print()
print("Saved polygon_rung_results.json / polygon_rung_full.json")
print("Psi - To preserve the harmonic field.")
