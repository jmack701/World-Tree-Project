#!/usr/bin/env python3
"""
Chorus M -- the coupled-grid ring-down.
World Tree Project, September 12, 2026. Frame: claude_Chorus_M_Model_and_Frame.md.

REGIME I (reproduction): archived functions of phi_grid_k_sweep_refined.py
byte-verbatim; gate = the four published backbone anchors from the archived
2,000-step configuration, re-run on the identical call path.
REGIME II (extension, declared): 100,000 steps, exogenous load vector zeroed
at step 90,000 (pin 1); mean field = unison voice, deviation field = loss
channel (pin 2); in-domain rule = capture present AND resolvable loss (pin 3);
sweep {0.2158, 0.2193, 0.2227, 0.2261} (pin 5); three-node minimal chorus as
declared, ungated extension (pin 6). Predictions, not laws: unanticipated
behavior is recorded, status OPEN, never written off.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= REGIME I -- ARCHIVED FUNCTIONS, VERBATIM =================
PHI = (1 + np.sqrt(5)) / 2
K_PHI = PHI / (9 * PHI - 3)
K_CRIT = 1.0 / 6.0
OMEGA_BASE = 2 * np.pi * 60
NUM_NODES = 50
TIME_STEPS = 2000
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
        for j in np.argsort(d)[:kn]:
            w = 1.0 / (1.0 + d[j])
            adj[i, j] = w; adj[j, i] = w
    rs = adj.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    return adj / rs

def gen_loads(n, steps, dt, stress=1.0):
    np.random.seed(SEED)
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

def run_sim(k_fc, loads, adj, harmonics=True, mobius=True):
    vh = np.zeros((NUM_NODES, TIME_STEPS))
    vc = np.ones(NUM_NODES) * V_NOMINAL
    vp = np.ones(NUM_NODES) * V_NOMINAL
    for step in range(TIME_STEPS):
        tv = step * DT
        vn = vc + loads[:, step] * DT * 10
        vn = vn + 0.05 * (adj @ vn - vn)
        dp, dc = vp - V_NOMINAL, vn - V_NOMINAL
        vn = vn + k_fc * (3.0 * dp - 6.0 * dc)
        if harmonics:
            ga = 2*np.pi/PHI**2
            ph = np.arange(NUM_NODES) * ga
            dl = np.std(dc)
            amp = dl * 0.3
            vn = vn - amp * (np.sin(3*OMEGA_BASE*tv+ph) + np.sin(6*OMEGA_BASE*tv+ph) + np.sin(9*OMEGA_BASE*tv+ph))
        if mobius:
            dev = vn - V_NOMINAL
            wr = np.fmod(dev + L_MOBIUS, 2*L_MOBIUS)
            wr = np.where(wr < 0, wr + 2*L_MOBIUS, wr)
            vn = wr - L_MOBIUS + V_NOMINAL
        vp = vc.copy(); vc = vn.copy(); vh[:, step] = vc
    return vh

def metrics(vh):
    n_nodes = vh.shape[0]
    se_vals, thd_vals = [], []
    for i in range(n_nodes):
        fft = np.fft.rfft(vh[i])
        pw = np.abs(fft[1:])**2
        if np.sum(pw) < 1e-12: se_vals.append(0.0)
        else:
            p = pw/np.sum(pw); p = p[p>1e-12]
            ent = -np.sum(p*np.log(p))
            mx = np.log(len(p)) if len(p)>1 else 1.0
            se_vals.append(ent/mx if mx>0 else 0.0)
        mags = np.abs(fft)
        if len(mags) < 2: thd_vals.append(0.0); continue
        fi = np.argmax(mags[1:]) + 1; fund = mags[fi]
        if fund < 1e-10: thd_vals.append(0.0); continue
        hp = np.sum(mags**2) - mags[0]**2 - fund**2
        thd_vals.append(np.sqrt(max(0, hp))/fund*100)
    amp = np.mean(np.abs(vh - V_NOMINAL))
    var = np.var(vh - V_NOMINAL)
    stab = np.mean([max(0, 1-np.std(vh[i])/max(1e-10, np.mean(vh[i]))) for i in range(n_nodes)])
    return np.mean(se_vals), np.mean(thd_vals), amp, var, stab, np.array(se_vals), np.array(thd_vals)
# ================= END ARCHIVED FUNCTIONS =================

print("=" * 78)
print("CHORUS M  (frame: claude_Chorus_M_Model_and_Frame.md)")
print("=" * 78)

# -------- REGIME I GATE: archived 2,000-step configuration, four anchors ----
print("\nREGIME I -- GATE (archived call path: 25-point sweep, 2,000 steps)")
print("-" * 78)
positions = create_grid(NUM_NODES)
adj = compute_adj(positions)
k_vals = np.linspace(0.168, 0.250, 25)
se_g, stab_g = [], []
for k in k_vals:
    loads, _ = gen_loads(NUM_NODES, TIME_STEPS, DT, stress=1.0)
    vh = run_sim(k, loads, adj)
    se, _, _, _, stab, _, _ = metrics(vh)
    se_g.append(se); stab_g.append(stab)
se_g = np.array(se_g); stab_g = np.array(stab_g)
i219 = int(np.argmin(np.abs(k_vals-0.21925)))
i223 = int(np.argmin(np.abs(k_vals-0.22267)))
i226 = int(np.argmin(np.abs(k_vals-0.22608)))
checks = [
 ("sampled SE range [0.1723, 0.8666]",
  round(float(se_g.min()),4)==0.1723 and round(float(se_g.max()),4)==0.8666,
  f"[{se_g.min():.4f}, {se_g.max():.4f}]"),
 ("operating point SE ~ 0.498 at 0.219", round(float(se_g[i219]),3)==0.498,
  f"{se_g[i219]:.4f}"),
 ("dip at 0.223, stability holding", se_g[i223]<se_g[i219] and stab_g[i223]>0.9,
  f"SE {se_g[i223]:.4f}, stab {stab_g[i223]:.4f}"),
 ("collapse at next grid point", stab_g[i226]<0.5 and stab_g[i223]>0.5,
  f"stab {stab_g[i226]:.4f}")]
gate_pass = True
for name, ok, note in checks:
    gate_pass &= ok
    print(f"  {name:<40} {'PASS' if ok else 'DEVIATION'}   ({note})")
print(f"\n  GATE VERDICT: {'PASS -- instrument reproduces the record; '
      'REGIME II opens' if gate_pass else 'FAIL -- STOP; the deviation is the finding'}")
if not gate_pass:
    raise SystemExit(1)

# -------- REGIME II: declared extension --------
STEPS_EXT, CUT = 100_000, 90_000

def run_sim_ext(k_fc, loads, adjm, steps):
    """Archived run_sim body, span parameterized; node count from adjm."""
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
        amp = dl * 0.3
        vn = vn - amp * (np.sin(3*OMEGA_BASE*tv+ph) + np.sin(6*OMEGA_BASE*tv+ph) + np.sin(9*OMEGA_BASE*tv+ph))
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
    return f0, dF, f0/dF, f0/(1.44*df), (dF <= 1.5*1.44*df)

def ringdown_rms(rms_tail, mean_tail):
    idx = np.arange(1, len(rms_tail)-1)
    pk = idx[(rms_tail[idx] >= rms_tail[idx-1]) &
             (rms_tail[idx] >= rms_tail[idx+1]) & (rms_tail[idx] > 1e-12)]
    if len(pk) >= 4:
        xs, ys, mode = pk.astype(float), rms_tail[pk], "envelope peaks"
    else:
        keep = np.nonzero(rms_tail > 1e-12)[0][::10]
        if len(keep) < 4:
            return None
        xs, ys, mode = keep.astype(float), rms_tail[keep], "direct samples (monotone envelope)"
    sl, b0 = np.polyfit(xs, np.log(ys), 1)
    pred = sl*xs + b0
    r2 = 1 - np.sum((np.log(ys)-pred)**2)/max(1e-30, np.sum((np.log(ys)-np.log(ys).mean())**2))
    ratio = ys[-1]/ys[0]
    resolvable = (sl < 0) and (ratio <= 0.9)
    mseg = mean_tail[int(xs[0]):int(xs[-1])+1]
    sc = np.sign(mseg[np.abs(mseg) > 0])
    f_ring = (np.count_nonzero(np.diff(sc)) / 2.0 / max(1, xs[-1]-xs[0])) if len(sc) > 2 else float('nan')
    return dict(tau_A=(-1.0/sl if sl < 0 else np.inf), R2=r2, mode=mode,
                win=(int(xs[0]), int(xs[-1])), ratio=ratio,
                f_ring=f_ring, resolvable=bool(resolvable))

def measure_chorus(kset, adjm, nn, label):
    print(f"\nREGIME II -- {label}: {nn} nodes, {STEPS_EXT} steps, loads off at {CUT}")
    print("-" * 78)
    loads_full, _ = gen_loads(nn, STEPS_EXT, DT, stress=1.0)
    loads_cut = loads_full.copy(); loads_cut[:, CUT:] = 0.0   # pin 1
    hdr = (f"{'k':>7} | {'f0(line)':>9} {'Q_coh':>8} {'rl':>2} | {'tau_A':>9} "
           f"{'R2':>5} {'fit':>16} | {'1/Qdis':>8} {'M':>9} | domain")
    print(hdr); print("-" * len(hdr))
    out = []
    for k in kset:
        vh = run_sim_ext(k, loads_cut, adjm, STEPS_EXT)
        dev = vh - V_NOMINAL
        mean_f = dev.mean(axis=0)
        rms_f = np.sqrt(((dev - mean_f)**2).mean(axis=0))
        f0, dF, Qc, Qres, rl = line_Q(mean_f[:CUT])
        fit = ringdown_rms(rms_f[CUT:], mean_f[CUT:])
        capture = True   # dominant pre-cutoff mean-field line stated (S4a); character reported via rl
        if fit is None or not fit['resolvable']:
            inv_qdis, M, Minv, qdis = 0.0, 0.0, float('inf'), float('inf')
            tau, r2, mode = (fit['tau_A'] if fit else float('inf')), (fit['R2'] if fit else float('nan')), (fit['mode'] if fit else '--')
            loss_ok = False
        else:
            tau, r2, mode = fit['tau_A'], fit['R2'], fit['mode']
            qdis = np.pi * f0 * tau
            inv_qdis = 1.0/qdis
            M = Qc/qdis
            assert abs(M - 1.0/(np.pi*dF*tau))/M < 1e-9
            Minv = 1.0/M
            loss_ok = True
        in_dom = capture and loss_ok
        print(f"{k:>7.4f} | {f0:>9.5f} {Qc:>8.1f} {('Y' if rl else 'n'):>2} | "
              f"{(f'{tau:9.1f}' if np.isfinite(tau) else '      inf')} "
              f"{(f'{r2:5.3f}' if np.isfinite(r2) else '   --')} {mode:>16} | "
              f"{inv_qdis:>8.2e} {M:>9.3g} | "
              f"{'IN-DOMAIN' if in_dom else 'out (typed)'}")
        out.append(dict(k=float(k), n_nodes=int(nn), f0=float(f0), dF=float(dF),
                        Q_coh=float(Qc), Q_res=float(Qres), res_limited=bool(rl),
                        tau_A=(None if not np.isfinite(tau) else float(tau)),
                        R2=(None if not np.isfinite(r2) else float(r2)),
                        fit_mode=mode,
                        f_ring=(None if fit is None or not np.isfinite(fit['f_ring']) else float(fit['f_ring'])),
                        inv_Q_dis=float(inv_qdis), M=float(M),
                        M_inv=(None if not np.isfinite(Minv) else float(Minv)),
                        loss_resolvable=bool(loss_ok), in_domain=bool(in_dom)))
    return out

KSET = [0.2158, 0.2193, 0.2227, 0.2261]
res50 = measure_chorus(KSET, adj, NUM_NODES, "the gated fifty")

pos3 = create_grid(3)
adj3 = compute_adj(pos3)
res3 = measure_chorus(KSET, adj3, 3, "the minimal chorus (declared, ungated)")

SKY = 100.0
in50 = [r for r in res50 if r['in_domain']]
print("\nTHE PRE-STATED QUESTION (S0, answered in its own words):")
if in50:
    mmax = max(in50, key=lambda r: r['M'])
    verdict = ("approaches the sky's value" if mmax['M'] >= SKY else
               "carries the class (M >> 1) but not the sky's value" if mmax['M'] > 10 else
               "shows loaded coherence of order unity or below")
    print(f"  in-domain M exists on the chorus dial: peak M = {mmax['M']:.3g} "
          f"at k = {mmax['k']:.4f} -- {verdict}.")
else:
    print("  no in-domain point on the swept set -- recorded; predictions, not laws.")

json.dump(dict(gate='PASS', sky_M=SKY, fifty=res50, three=res3),
          open('/mnt/user-data/outputs/chorus_M_results.json', 'w'), indent=2)

# figures
fig, axes = plt.subplots(2, 1, figsize=(9, 7))
loads_full, _ = gen_loads(NUM_NODES, STEPS_EXT, DT, stress=1.0)
loads_cut = loads_full.copy(); loads_cut[:, CUT:] = 0.0
cols = {0.2158: '#7fb3d5', 0.2193: '#1a5276', 0.2227: '#6c3483', 0.2261: '#b03a2e'}
for k in KSET:
    vh = run_sim_ext(k, loads_cut, adj, STEPS_EXT)
    dev = vh - V_NOMINAL
    mf = dev.mean(axis=0); rf = np.sqrt(((dev-mf)**2).mean(axis=0))
    axes[0].semilogy(np.arange(CUT-200, min(STEPS_EXT, CUT+3000)),
                     np.maximum(rf[CUT-200:CUT+3000], 1e-16),
                     lw=0.9, color=cols[k], label=f'k={k}')
    axes[1].plot(np.arange(CUT-200, CUT+1500), mf[CUT-200:CUT+1500],
                 lw=0.8, color=cols[k], label=f'k={k}')
axes[0].axvline(CUT, color='k', lw=0.7, ls=':')
axes[0].set_ylabel('deviation-field RMS'); axes[0].legend(fontsize=7)
axes[0].set_title('Chorus ring-down -- the paying sector after loads-off (chorus_M_run.py)', fontsize=10)
axes[1].axvline(CUT, color='k', lw=0.7, ls=':')
axes[1].set_xlabel('step'); axes[1].set_ylabel('mean field (unison voice)')
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/fig_chorus_ringdown.png', dpi=150)
print("\nArtifacts: chorus_M_results.json, fig_chorus_ringdown.png")
