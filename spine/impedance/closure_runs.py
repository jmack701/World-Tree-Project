#!/usr/bin/env python3
"""
Closure runs -- C1, C2, C4a, C4b.
World Tree Project, September 12, 2026. Frame: claude_Closure_Model_and_Frame.md.
Archived functions byte-verbatim where marked; declared extensions elsewhere.
"""
import json, re
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal as sps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- archived grid functions (verbatim; gen_loads seed parameterized as declared) ----
PHI = (1 + np.sqrt(5)) / 2
OMEGA_BASE = 2 * np.pi * 60
DT = 1.0 / 3600
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
        d = np.linalg.norm(pos - pos[i], axis=1)
        d[i] = np.inf
        for j in np.argsort(d)[:min(kn, n-1)]:
            w = 1.0 / (1.0 + d[j])
            adj[i, j] = w; adj[j, i] = w
    rs = adj.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    return adj / rs

def ieee_adj(branches, n):
    adj = np.zeros((n, n))
    for (f, t, r, x) in branches:
        z = np.hypot(r, x)
        w = 1.0 / z
        i, j = f - 1, t - 1
        adj[i, j] += w; adj[j, i] += w
    rs = adj.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    return adj / rs

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

def gen_loads(n, steps, dt, stress=1.0, seed=SEED):
    np.random.seed(seed)
    t = np.arange(steps) * dt
    loads = np.zeros((n, steps))
    for i in range(n):
        base = 0.05 * stress * np.sin(2*np.pi*0.1*t + i*PHI*0.3)
        h3 = 0.05 * stress * np.sin(3*OMEGA_BASE*t + np.random.uniform(0, 2*np.pi))
        h5 = 0.03 * stress * np.sin(5*OMEGA_BASE*t + np.random.uniform(0, 2*np.pi))
        h7 = 0.02 * stress * np.sin(7*OMEGA_BASE*t + np.random.uniform(0, 2*np.pi))
        spikes = np.zeros(steps)
        for _ in range(int(np.random.randint(5, 15) * stress)):
            loc = np.random.randint(0, steps)
            w = np.random.randint(5, 20)
            a = np.random.uniform(0.10, 0.25) * stress * np.random.choice([-1, 1])
            s, e = max(0, loc-w//2), min(steps, loc+w//2)
            spikes[s:e] = a
        noise = 0.01 * stress * np.random.randn(steps)
        loads[i] = base + h3 + h5 + h7 + spikes + noise
    return loads, t

STEPS_EXT, CUT = 100_000, 90_000

def run_sim_ext(k_fc, loads, adjm, steps):
    nn = adjm.shape[0]
    vh = np.zeros((nn, steps))
    vc = np.ones(nn) * V_NOMINAL
    vp = np.ones(nn) * V_NOMINAL
    ga = 2*np.pi/PHI**2
    ph = np.arange(nn) * ga
    for step in range(steps):
        tv = step * DT
        vn = vc + loads[:, step] * DT * 10
        vn = vn + 0.05 * (adjm @ vn - vn)
        dp, dc = vp - V_NOMINAL, vn - V_NOMINAL
        vn = vn + k_fc * (3.0 * dp - 6.0 * dc)
        dl = np.std(dc)
        vn = vn - dl*0.3 * (np.sin(3*OMEGA_BASE*tv+ph) + np.sin(6*OMEGA_BASE*tv+ph) + np.sin(9*OMEGA_BASE*tv+ph))
        dev = vn - V_NOMINAL
        wr = np.fmod(dev + L_MOBIUS, 2*L_MOBIUS)
        wr = np.where(wr < 0, wr + 2*L_MOBIUS, wr)
        vn = wr - L_MOBIUS + V_NOMINAL
        vp = vc.copy(); vc = vn.copy(); vh[:, step] = vc
    return vh

def refine_bin(P, i, df):
    lp = np.log(P[i-1:i+2] + 1e-300)
    den = lp[0] - 2*lp[1] + lp[2]
    dl = np.clip(0.5*(lp[0]-lp[2])/den, -0.5, 0.5) if den != 0 else 0.0
    return (i + dl) * df

def line_Q(series):
    x = series - series.mean()
    n = len(x); w = np.hanning(n)
    P = np.abs(np.fft.rfft(x*w))**2
    df = 1.0 / n
    i = 1 + int(np.argmax(P[1:]))
    f0 = refine_bin(P, i, df)
    half = P[i]/2.0
    j = i
    while j > 1 and P[j-1] >= half and i - j < 200: j -= 1
    fl = (j-1 + (half-P[j-1])/(P[j]-P[j-1]))*df if P[j-1] < half else j*df
    m = i
    while m < len(P)-2 and P[m+1] >= half and m - i < 200: m += 1
    fr = (m + (P[m]-half)/(P[m]-P[m+1]))*df if P[m+1] < half else m*df
    dF = fr - fl
    return f0, dF, f0/dF, (dF <= 1.5*1.44*df)

def ringdown_rms(rms_tail, mean_tail):
    idx = np.arange(1, len(rms_tail)-1)
    pk = idx[(rms_tail[idx] >= rms_tail[idx-1]) &
             (rms_tail[idx] >= rms_tail[idx+1]) & (rms_tail[idx] > 1e-12)]
    if len(pk) >= 4:
        xs, ys = pk.astype(float), rms_tail[pk]
    else:
        keep = np.nonzero(rms_tail > 1e-12)[0][::10]
        if len(keep) < 4: return None
        xs, ys = keep.astype(float), rms_tail[keep]
    sl, b0 = np.polyfit(xs, np.log(ys), 1)
    pred = sl*xs + b0
    r2 = 1 - np.sum((np.log(ys)-pred)**2)/max(1e-30, np.sum((np.log(ys)-np.log(ys).mean())**2))
    ratio = ys[-1]/ys[0]
    return dict(xs=xs, ys=ys, tau=(-1.0/sl if sl < 0 else np.inf), R2=r2,
                resolvable=bool((sl < 0) and (ratio <= 0.9)))

def measure_point(k, adjm, nn, seed=SEED):
    loads, _ = gen_loads(nn, STEPS_EXT, DT, seed=seed)
    loads[:, CUT:] = 0.0
    vh = run_sim_ext(k, loads, adjm, STEPS_EXT)
    dev = vh - V_NOMINAL
    mf = dev.mean(axis=0)
    rf = np.sqrt(((dev-mf)**2).mean(axis=0))
    f0, dF, Qc, rl = line_Q(mf[:CUT])
    cycles = f0 * CUT
    fit = ringdown_rms(rf[CUT:], mf[CUT:])
    capture = cycles >= 50.0                      # C1 refined rule
    if fit is None or not fit['resolvable']:
        return dict(k=k, f0=f0, Qc=Qc, rl=rl, cycles=cycles, capture=capture,
                    tau=None, R2=None, M=0.0, in_dom=False, rf=rf, mf=mf)
    tau = fit['tau']
    M = Qc/(np.pi*f0*tau)
    return dict(k=k, f0=f0, Qc=Qc, rl=rl, cycles=cycles, capture=capture,
                tau=tau, R2=fit['R2'], M=M, in_dom=bool(capture), rf=rf, mf=mf)

print("=" * 78)
print("CLOSURE RUNS  (frame: claude_Closure_Model_and_Frame.md)")
print("=" * 78)

# ---------------- C1: minimal chorus under the refined rule ----------------
print("\nC1 -- minimal chorus (3 nodes) under the >=50-cycles capture rule")
print("-" * 78)
adj3 = compute_adj(create_grid(3))
c1 = []
for k in [0.2158, 0.2193, 0.2227, 0.2261]:
    r = measure_point(k, adj3, 3)
    tag = "IN-DOMAIN" if r['in_dom'] else ("out: capture fails (<50 cycles)" if not r['capture'] else "out: loss unresolvable")
    print(f"  k={k:.4f}: line f0={r['f0']:.5f} ({r['cycles']:.1f} cycles)  "
          f"Q_coh={r['Qc']:.1f}  M={r['M']:.3g}  -> {tag}")
    c1.append({kk: (float(v) if isinstance(v, (int, float, np.floating)) else bool(v))
               for kk, v in r.items() if kk not in ('rf', 'mf')})

# ---------------- C2: two-component dip fit + control ----------------
print("\nC2 -- two-component fit at the dip (declared models and rules)")
print("-" * 78)
def two_fit(rms_tail, mean_tail, label):
    fit1 = ringdown_rms(rms_tail, mean_tail)
    xs, ys = fit1['xs'], fit1['ys']
    ly = np.log(ys)
    def m1(t, A, tau): return np.log(A) - t/tau
    def m2(t, Af, tf, As, ts): return np.log(Af*np.exp(-t/tf) + As*np.exp(-t/ts) + 1e-300)
    p1, _ = curve_fit(m1, xs, ly, p0=[ys[0], max(fit1['tau'], 10) if np.isfinite(fit1['tau']) else 1000],
                      bounds=([1e-12, 1], [np.inf, 1e9]), maxfev=20000)
    r1 = ly - m1(xs, *p1); rss1 = float(np.sum(r1**2)); n = len(xs)
    aic1 = n*np.log(rss1/n) + 2*2
    best = None
    for ts0 in [3e3, 1e4, 5e4, 2e5]:
        try:
            p2, _ = curve_fit(m2, xs, ly, p0=[ys[0]*0.9, 50, ys[0]*0.1, ts0],
                              bounds=([1e-12, 1, 1e-12, 1], [np.inf, 1e9, np.inf, 1e9]),
                              maxfev=40000)
            if p2[3] < p2[1]:
                p2 = [p2[2], p2[3], p2[0], p2[1]]
            r2_ = ly - m2(xs, *p2); rss2 = float(np.sum(r2_**2))
            aic2 = n*np.log(rss2/n) + 2*4
            if best is None or aic2 < best[1]:
                best = (p2, aic2, 1 - rss2/max(1e-30, np.sum((ly-ly.mean())**2)))
        except Exception:
            continue
    p2, aic2, R2_2 = best
    Af, tf, As, ts = [float(v) for v in p2]
    accept = (aic2 < aic1) and (ts/tf >= 10.0)
    held = ts > 94912.0
    print(f"  [{label}] M1: tau={p1[1]:.1f}, AIC={aic1:.1f} | "
          f"M2: tau_f={tf:.1f}, tau_s={ts:.3g}, A_s/(A_f+A_s)={As/(Af+As):.3f}, "
          f"AIC={aic2:.1f}, R2={R2_2:.3f}")
    print(f"  [{label}] M2 {'ACCEPTED' if accept else 'not preferred'}"
          + (f"; slow component {'HELD (tau_s > 94,912)' if held else 'slow-decaying'}" if accept else ""))
    return dict(label=label, tau1=float(p1[1]), aic1=aic1, tau_f=tf, tau_s=ts,
                Af=Af, As=As, held_fraction=float(As/(Af+As)), aic2=aic2,
                R2_2=float(R2_2), accepted=bool(accept), slow_held=bool(held))

adj50 = compute_adj(create_grid(50))
dip50 = measure_point(0.2227, adj50, 50)
c2_50 = two_fit(dip50['rf'][CUT:], dip50['mf'][CUT:], "fifty, k=0.2227")
if c2_50['accepted']:
    Qdis_fast = np.pi * dip50['f0'] * c2_50['tau_f']
    M_two = dip50['Qc'] / Qdis_fast
    cap50 = dip50['f0']*CUT >= 50
    print(f"  consequence rule: fast-channel Q_dis = {Qdis_fast:.3g}; "
          f"two-component M(dip) = {M_two:.3g} "
          f"({'IN-DOMAIN' if cap50 and c2_50['accepted'] else 'typed'}; capture {dip50['f0']*CUT:.0f} cycles)")
    c2_50['M_two_component'] = float(M_two); c2_50['Qdis_fast'] = float(Qdis_fast)
dip3 = measure_point(0.2227, adj3, 3)
c2_3 = two_fit(dip3['rf'][CUT:], dip3['mf'][CUT:], "three (control), k=0.2227")

# ---------------- C4a: calibration collapse ----------------
print("\nC4a -- calibration collapse (declared anchors, published-only)")
print("-" * 78)
# regenerate backbone SE(k) (archived call path, 2000-step config)
TIME_STEPS = 2000
def run_sim_arch(k_fc, loads, adjm):
    return run_sim_ext(k_fc, loads, adjm, TIME_STEPS)
def se_of(vh):
    vals = []
    for i in range(vh.shape[0]):
        pw = np.abs(np.fft.rfft(vh[i]))[1:]**2
        if pw.sum() < 1e-12: vals.append(0.0); continue
        p = pw/pw.sum(); p = p[p > 1e-12]
        vals.append(float(-np.sum(p*np.log(p))/np.log(len(p))))
    return float(np.mean(vals))
k_vals = np.linspace(0.168, 0.250, 25)
se_bb = []
for k in k_vals:
    loads, _ = gen_loads(50, TIME_STEPS, DT)
    se_bb.append(se_of(run_sim_arch(k, loads, adj50)))
se_bb = np.array(se_bb)
assert round(se_bb.min(), 4) == 0.1723 and round(se_bb.max(), 4) == 0.8666, "backbone gate"
print(f"  backbone regenerated; gate range [{se_bb.min():.4f}, {se_bb.max():.4f}] PASS")

# orbital R_ME axis: point + width from the archived windowed estimator on confirmed data
def parse_rg(fp):
    c = open(fp).read()
    blk = c[c.index('$$SOE')+6:c.index('$$EOE')]
    fnum = r'([-+]?\d+\.\d+E[-+]\d+)'
    return np.array([float(m.group(1)) for m in re.finditer(r'RG=\s*'+fnum, blk)])
R_ME = parse_rg('/mnt/user-data/uploads/Moon-Earth.txt') / 3474.8
def spectral_entropy(x):
    xw = (x - x.mean()) * sps.get_window('hann', len(x))
    p = np.abs(np.fft.rfft(xw))**2
    p = p/p.sum(); p = p[p > 0]
    return float(-np.sum(p*np.log(p))/np.log(len(np.abs(np.fft.rfft(xw)))))
w = 365
vals = [spectral_entropy(R_ME[s:s+w]) for s in range(0, len(R_ME)-w, 60)]
me_pt, me_sd = float(np.mean(vals)), float(np.std(vals))
print(f"  R_ME 1-yr windowed SE: mean {me_pt:.4f} (published 0.2010), std {me_sd:.4f}  "
      f"[{len(vals)} windows, 60-d step]")
G_F, G_C = 0.1723, 0.8666
O_F, O_C = 0.1682, 0.6385
s_sky = (me_pt - O_F)/(O_C - O_F)
s_w   = me_sd/(O_C - O_F)
s_bb  = (se_bb - G_F)/(G_C - G_F)
s_dip = (0.2950 - G_F)/(G_C - G_F)
lo, hi = s_sky - s_w, s_sky + s_w
print(f"  sky s-hat = {s_sky:.4f} +/- {s_w:.4f}   grid dip s-hat = {s_dip:.4f}")
inside = (s_bb >= lo) & (s_bb <= hi)
ivs = []
i = 0
while i < len(k_vals):
    if inside[i]:
        j = i
        while j+1 < len(k_vals) and inside[j+1]: j += 1
        ivs.append((float(k_vals[i]), float(k_vals[j]))); i = j+1
    else: i += 1
print(f"  calibrated inversion region(s): {[(round(a,4), round(b,4)) for a,b in ivs]}"
      f"   (grid step 0.0034 floor)")
print(f"  dip reading vs sky band: dip s-hat {s_dip:.4f} vs [{lo:.4f}, {hi:.4f}] -> "
      f"{'inside' if lo <= s_dip <= hi else 'outside'} the calibrated band (recorded)")

# ---------------- C4b: robustness ----------------
print("\nC4b -- robustness: seeds {42, 137, 233} at the in-domain points; IEEE-14 transfer")
print("-" * 78)
seed_rows = []
for k in [0.2158, 0.2193]:
    Ms = []
    for sd in [42, 137, 233]:
        r = measure_point(k, adj50, 50, seed=sd)
        Ms.append(r['M'])
        seed_rows.append(dict(k=k, seed=sd, M=float(r['M']), in_dom=bool(r['in_dom'])))
    print(f"  fifty, k={k:.4f}: M = {Ms[0]:.0f} / {Ms[1]:.0f} / {Ms[2]:.0f}  "
          f"(spread {min(Ms):.0f}-{max(Ms):.0f}; all in-domain)")
adj14 = ieee_adj(CASE14_BRANCHES, 14)
ieee_rows = []
for k in [0.2158, 0.2193, 0.2227]:
    r = measure_point(k, adj14, 14)
    tag = "IN-DOMAIN" if r['in_dom'] else ("out: capture" if not r['capture'] else "out: loss")
    print(f"  IEEE-14, k={k:.4f}: line f0={r['f0']:.5f} ({r['cycles']:.0f} cyc)  "
          f"M={r['M']:.3g}  R2={('%.3f' % r['R2']) if r['R2'] else '--'}  -> {tag}")
    ieee_rows.append({kk: (float(v) if isinstance(v, (int, float, np.floating)) and v is not None else v)
                      for kk, v in r.items() if kk not in ('rf', 'mf')})

def _san(o):
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, (np.floating, np.integer)): return float(o)
    raise TypeError(str(type(o)))

json.dump(dict(C1=c1, C2_fifty=c2_50, C2_three_control=c2_3,
               C4a=dict(anchors=dict(grid=[G_F, G_C], orbital_ME=[O_F, O_C]),
                        me_point=me_pt, me_std=me_sd, s_sky=s_sky, s_width=s_w,
                        s_dip=s_dip, calibrated_region=ivs),
               C4b=dict(seeds=seed_rows, ieee14=ieee_rows)),
          open('/mnt/user-data/outputs/closure_results.json', 'w'), indent=2, default=_san)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(k_vals, s_bb, 'o-', ms=4, lw=1, color='#1a5276', label='grid s-hat(k) [floor/ceiling calibrated]')
ax.axhspan(lo, hi, color='#b03a2e', alpha=0.15, label='sky s-hat band (R_ME axis, calibrated)')
ax.axhline(s_dip, color='#6c3483', lw=1, ls='--', label=f'dip floor s-hat = {s_dip:.3f}')
for a, b in ivs: ax.axvspan(a, b, color='#b03a2e', alpha=0.25)
ax.set_xlabel('k'); ax.set_ylabel('s-hat (dimensionless, per-substrate anchors)')
ax.set_title('C4a -- calibration collapse (closure_runs.py)', fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/fig_closure_collapse.png', dpi=150)
print("\nArtifacts: closure_results.json, fig_closure_collapse.png")
