"""
Galactic Disk F_c Simulation — Phase 1c: Perturbation Winding
=============================================================
Question: does an impulsive perturbation of the Phase 1b disc wind
into a persistent, datable phase spiral (organized memory), or does
it disperse (memory lost)? This is the computational half of the
Gaia phase-spiral comparison proposed for The Spine §7.5.

Method: twin runs. A reference run and a kicked run share identical
geometry, loads, coupling, and seed; the kicked run receives a single
impulsive displacement delta = +0.02 on every disc node (central M
excluded) at step t0 = 1500. The response is the exact difference
d_i(t) = v_kick − v_ref (deterministic model — the difference is the
response and nothing else). Shell-mean responses give per-shell
phases via (d, d')/omega normalization, unwrapped in time; winding
W(t) = [Theta_inner(t) − Theta_outer(t)] / 2pi.

Configurations (all twin pairs, same kick):
  A. Fc + self-enrichment  (Phase 1b RUN 1 configuration — the claim)
  B. Fc, no enrichment     (isolates the Fc feedback layer)
  C. no Fc, no enrichment  (gravity-relay control — load-clock kinematics)

Pre-registration (stated before execution):
  H6 (winding):    In configuration A, W(t) grows approximately
                   linearly over the post-kick window.
                   SUCCESS: linear fit R^2 >= 0.90 with positive slope.
                   FAILURE: otherwise.
  H7 (datable):    Slope fitted on the first half of the window
                   predicts W at the end of the window.
                   SUCCESS: |W_pred − W_meas| / W_meas <= 0.15.
                   FAILURE: otherwise.
  H8 (organized):  At the first time W >= 2.5 wraps, the radial
                   phase profile Theta(shell) is monotone
                   (|Spearman rho| >= 0.80 across analysis shells),
                   and the mean response envelope at the end of the
                   window remains above 10% of its post-kick peak.
                   SUCCESS: both hold. FAILURE: either fails.
  Reported without threshold: measured winding rate against the
  load-clock prediction Delta omega_bar / 2pi; control comparison.

Kick amplitude fixed in advance at 0.02; no retuning. If the
response leaves the linear regime, that is the result and is
reported as such. Both outcomes publishable. The Phase 1 guard
holds: mechanism-class viability only; no bridge to physical units.

Stepper, geometry, loads, and coupling are copied verbatim from
galactic_disk_fc_phase1b.py (RUN 1 parameters: k_fc = 0.221,
coupling = 0.015, central mass 25, seed 42). The only stepper
change: configurable step count and the kick injection point.

J. David Mack & Claude (Fable 5)
World Tree Project — September 2026
Psi  moon-wave-tree-web  To preserve the harmonic field.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# CONSTANTS (Phase 1b, verbatim)
# ============================================================
PHI = (1 + np.sqrt(5)) / 2
L_MOBIUS = PHI
SEED = 42
OMEGA_BASE = 2 * np.pi * 60
DT = 1.0 / 3600
K_FC = 0.221

N_NODES = 500
R_SCALE = 3.0
R_MAX = 5 * R_SCALE
SOFTENING = 0.3
N_SHELLS = 15

# Phase 1c protocol
TIME_STEPS = 9000
KICK_STEP = 1500
KICK_AMP = 0.02
CENTRAL_MASS = 25.0
COUPLING = 0.015

os.makedirs('phase1c_results', exist_ok=True)

print("=" * 70)
print("GALACTIC DISK F_c — PHASE 1c: PERTURBATION WINDING")
print(f"  steps={TIME_STEPS}  kick at t0={KICK_STEP}  amp={KICK_AMP}")
print(f"  k_fc={K_FC}  coupling={COUPLING}  M={CENTRAL_MASS}  seed={SEED}")
print("=" * 70)

# ============================================================
# GEOMETRY / SHELLS / LOADS / COUPLING (Phase 1b, verbatim)
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


def assign_shells(radii, n_shells, r_max):
    shell_edges = np.linspace(0, r_max * 1.01, n_shells + 1)
    shell_ids = np.digitize(radii, shell_edges) - 1
    shell_ids = np.clip(shell_ids, 0, n_shells - 1)
    shell_ids[0] = 0
    return shell_ids, shell_edges


def compute_shell_coupling(positions, radii, shell_ids, n_shells,
                           central_mass=10.0, softening=0.3,
                           k_azimuthal=8, k_radial=5):
    n = len(positions)
    adj = np.zeros((n, n))
    for i in range(n):
        si = shell_ids[i]
        di = np.linalg.norm(positions - positions[i], axis=1)
        di[i] = np.inf
        same_shell = np.where(shell_ids == si)[0]
        same_shell = same_shell[same_shell != i]
        if len(same_shell) > 0:
            d_same = di[same_shell]
            n_connect = min(k_azimuthal, len(same_shell))
            nearest = same_shell[np.argsort(d_same)[:n_connect]]
            for j in nearest:
                w = 1.0 / (di[j]**2 + softening**2)
                adj[i, j] = max(adj[i, j], w)
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
                adj[i, j] = max(adj[i, j], w * 0.7)
    inner_nodes = np.where(shell_ids <= 1)[0]
    inner_nodes = inner_nodes[inner_nodes != 0]
    for j in inner_nodes:
        d = max(np.linalg.norm(positions[j] - positions[0]), softening)
        w = central_mass / (d**2 + softening**2)
        adj[0, j] = w
        adj[j, 0] = w
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return adj / row_sums


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
# STEPPER (Phase 1b law verbatim; configurable steps + kick)
# ============================================================
def run_disk_fc_relay(n_nodes, radii, shell_ids, k_fc, loads, adj,
                      coupling=0.01, enrichment_mode='self',
                      use_fc=True, steps=TIME_STEPS,
                      kick_step=None, kick_vec=None):
    vh = np.zeros((n_nodes, steps))
    vc = np.ones(n_nodes)
    vp = np.ones(n_nodes)
    for step in range(steps):
        if kick_step is not None and step == kick_step:
            vc = vc + kick_vec          # impulsive displacement of the current state
        tv = step * DT
        vn = vc + loads[:, step] * DT * 10
        vn = vn + coupling * (adj @ vn - vn)
        if use_fc:
            dp = vp - 1.0
            dc = vn - 1.0
            vn = vn + k_fc * (3.0 * dp - 6.0 * dc)
        if enrichment_mode == 'external':
            ga = 2 * np.pi / PHI**2
            ph = np.arange(n_nodes) * ga
            dl = np.std(vn - 1.0)
            amp_h = dl * 0.3
            vn = vn - amp_h * (np.sin(3 * OMEGA_BASE * tv + ph) +
                               np.sin(6 * OMEGA_BASE * tv + ph) +
                               np.sin(9 * OMEGA_BASE * tv + ph))
        elif enrichment_mode == 'self':
            dev = vn - 1.0
            for s in range(1, N_SHELLS):
                shell_mask = (shell_ids == s)
                if shell_mask.sum() == 0:
                    continue
                interior_mask = (shell_ids < s)
                if interior_mask.sum() == 0:
                    continue
                interior_dev = dev[interior_mask]
                interior_rms = np.sqrt(np.mean(interior_dev**2))
                amp_self = interior_rms * 0.3 / (1 + 0.1 * s)
                phase_int = np.arctan2(
                    np.mean(np.sin(np.angle(interior_dev + 1j * 1e-10))),
                    np.mean(np.cos(np.angle(interior_dev + 1j * 1e-10)))
                )
                shell_nodes = np.where(shell_mask)[0]
                for j in shell_nodes:
                    theta_j = np.arctan2(positions[j, 1], positions[j, 0])
                    vn[j] -= amp_self * (
                        np.sin(3 * OMEGA_BASE * tv + theta_j + phase_int) +
                        np.sin(6 * OMEGA_BASE * tv + theta_j + phase_int) +
                        np.sin(9 * OMEGA_BASE * tv + theta_j + phase_int)
                    )
        dev = vn - 1.0
        wr = np.fmod(dev + L_MOBIUS, 2 * L_MOBIUS)
        wr = np.where(wr < 0, wr + 2 * L_MOBIUS, wr)
        vn = wr - L_MOBIUS + 1.0
        vp = vc.copy()
        vc = vn.copy()
        vh[:, step] = vc
    return vh


# ============================================================
# ANALYSIS
# ============================================================
def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def shell_phases(d_resp, shells_used, omega_bar, t0):
    """Per-shell temporally-unwrapped phase and amplitude from the
    twin-difference response, using (d, d'/omega) coordinates."""
    T = d_resp.shape[1]
    thetas, amps = {}, {}
    for s in shells_used:
        d = d_resp[s]
        dd = np.gradient(d) / DT
        x, y = d, dd / omega_bar[s]
        th_raw = np.unwrap(np.arctan2(y, x))
        thetas[s] = th_raw
        amps[s] = np.sqrt(x**2 + y**2)
    return thetas, amps


def run_pair(tag, enrichment_mode, use_fc, kick_vec):
    print(f"\n-- configuration {tag}: enrichment={enrichment_mode}, "
          f"use_fc={use_fc} --")
    vh_ref = run_disk_fc_relay(N_NODES, radii, shell_ids, K_FC, loads,
                               adj_relay, coupling=COUPLING,
                               enrichment_mode=enrichment_mode,
                               use_fc=use_fc, steps=TIME_STEPS)
    vh_kick = run_disk_fc_relay(N_NODES, radii, shell_ids, K_FC, loads,
                                adj_relay, coupling=COUPLING,
                                enrichment_mode=enrichment_mode,
                                use_fc=use_fc, steps=TIME_STEPS,
                                kick_step=KICK_STEP, kick_vec=kick_vec)
    d_nodes = vh_kick - vh_ref
    # shell means (disc shells only; node 0 = central M excluded)
    d_shell = np.zeros((N_SHELLS, TIME_STEPS))
    for s in range(N_SHELLS):
        m = (shell_ids == s)
        m[0] = False
        if m.sum() > 0:
            d_shell[s] = d_nodes[m].mean(axis=0)
    print(f"   max|d| over run: {np.abs(d_nodes).max():.4f}   "
          f"(Mobius half-width phi = {PHI:.4f})")
    return d_shell, float(np.abs(d_nodes).max())


# ============================================================
# EXECUTION
# ============================================================
positions, radii = create_disk(N_NODES, R_SCALE, R_MAX, seed=SEED)
shell_ids, shell_edges = assign_shells(radii, N_SHELLS, R_MAX)
loads = gen_disk_loads(N_NODES, radii, TIME_STEPS, DT, seed=SEED)
adj_relay = compute_shell_coupling(positions, radii, shell_ids, N_SHELLS,
                                   central_mass=CENTRAL_MASS,
                                   softening=SOFTENING)

kick = np.full(N_NODES, KICK_AMP)
kick[0] = 0.0                                     # disc kicked; central M untouched

# analysis shells: populated disc shells
counts = np.array([((shell_ids == s) & (np.arange(N_NODES) != 0)).sum()
                   for s in range(N_SHELLS)])
shells_used = [s for s in range(1, N_SHELLS) if counts[s] >= 8]
s_in, s_out = shells_used[0], shells_used[-1]
print(f"analysis shells: {shells_used}  (inner={s_in}, outer={s_out})")

# load-clock prediction: shell-mean orbital frequency
omega_bar = np.zeros(N_SHELLS)
for s in range(N_SHELLS):
    m = (shell_ids == s); m[0] = False
    if m.sum() > 0:
        rr = np.maximum(radii[m], 0.1)
        omega_bar[s] = np.mean(OMEGA_BASE / (rr**1.5 + 1.0))
slope_pred = (omega_bar[s_in] - omega_bar[s_out]) * DT / (2 * np.pi)  # wraps/step
print(f"load-clock predicted winding rate: {slope_pred:.5f} wraps/step "
      f"({slope_pred*100:.3f} per 100 steps)")

results = {}
dA, maxA = run_pair('A (Fc + self-enrichment)', 'self', True, kick)
dB, maxB = run_pair('B (Fc, no enrichment)', 'none', True, kick)
dC, maxC = run_pair('C (no Fc control)', 'none', False, kick)

post = slice(KICK_STEP + 50, TIME_STEPS)          # settle 50 steps, then measure
tt = np.arange(TIME_STEPS)

summary = {}
for tag, d_shell in (('A', dA), ('B', dB), ('C', dC)):
    thetas, amps = shell_phases(d_shell, shells_used, omega_bar, KICK_STEP)
    W = (thetas[s_in] - thetas[s_out]) / (2 * np.pi)
    W = W - W[KICK_STEP + 50]                      # zero at measurement start
    # H6: linearity over post-kick window
    x = tt[post].astype(float); y = W[post]
    A_ = np.vstack([x, np.ones_like(x)]).T
    coef, res_, *_ = np.linalg.lstsq(A_, y, rcond=None)
    slope, icpt = coef
    yhat = A_ @ coef
    ss_res = float(((y - yhat)**2).sum())
    ss_tot = float(((y - y.mean())**2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # H7: date from first-half slope
    half = slice(KICK_STEP + 50, KICK_STEP + 50 + (TIME_STEPS - KICK_STEP - 50)//2)
    xh = tt[half].astype(float); yh = W[half]
    ch, *_ = np.linalg.lstsq(np.vstack([xh, np.ones_like(xh)]).T, yh,
                             rcond=None)
    W_end_pred = ch[0] * x[-1] + ch[1]
    date_err = abs(W_end_pred - y[-1]) / abs(y[-1]) if y[-1] != 0 else np.inf
    # H8: monotone profile at W = 2.5; persistence
    idx25 = np.argmax(np.abs(W) >= 2.5)
    t25 = int(idx25) if np.abs(W).max() >= 2.5 else None
    if t25:
        prof = np.array([thetas[s][t25] for s in shells_used])
        rho = spearman(np.array(shells_used, float), prof)
    else:
        rho = 0.0
    env = np.mean([amps[s] for s in shells_used], axis=0)
    peak = env[KICK_STEP + 50:KICK_STEP + 600].max()
    tail = env[-200:].mean()
    persist = tail / peak if peak > 0 else 0.0
    summary[tag] = dict(slope=float(slope), r2=float(r2),
                        slope_ratio=float(slope / slope_pred),
                        date_err=float(date_err), t25=t25,
                        rho=float(rho), persist=float(persist),
                        W_end=float(y[-1]), W=W, env=env, thetas=thetas)
    print(f"\nconfig {tag}: slope={slope:+.5f} wraps/step  R2={r2:.4f}  "
          f"slope/pred={slope/slope_pred:+.3f}")
    print(f"          W(end)={y[-1]:+.2f} wraps  date_err={date_err*100:.1f}%  "
          f"rho(profile@2.5)={rho:+.3f}  persist(tail/peak)={persist:.3f}")

# ---- pre-registration assessment on configuration A ----
A = summary['A']
h6 = "PASS" if (A['r2'] >= 0.90 and A['slope'] > 0) else "FAIL"
h7 = "PASS" if A['date_err'] <= 0.15 else "FAIL"
h8 = "PASS" if (abs(A['rho']) >= 0.80 and A['persist'] >= 0.10) else "FAIL"
print("\n" + "=" * 70)
print("PRE-REGISTRATION ASSESSMENT — PHASE 1c (configuration A)")
print(f"  H6 winding  (R2>=0.90, slope>0):        [{h6}]  "
      f"R2={A['r2']:.4f} slope={A['slope']:+.5f}")
print(f"  H7 datable  (end-W predicted +/-15%):   [{h7}]  "
      f"err={A['date_err']*100:.1f}%")
print(f"  H8 organized (|rho|>=0.80, tail>=10%):  [{h8}]  "
      f"rho={A['rho']:+.3f} persist={A['persist']:.3f}")
print(f"  reported: slope/load-clock prediction = {A['slope_ratio']:+.3f}; "
      f"control C ratio = {summary['C']['slope_ratio']:+.3f}")
print("=" * 70)

# ============================================================
# FIGURES
# ============================================================
Wmax_t = summary['A']['t25'] or (KICK_STEP + 400)
# choose a display time near 2.5 wraps for the spiral portrait
t_disp = Wmax_t

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Phase 1c: Perturbation Winding in the F$_c$ Disc '
             f'(kick {KICK_AMP} at step {KICK_STEP}, seed {SEED})',
             fontsize=14, fontweight='bold')

ax = axes[0, 0]                                    # polar spiral portrait
th_prof = np.array([summary['A']['thetas'][s][t_disp] for s in shells_used])
r_prof = np.array([0.5 * (shell_edges[s] + shell_edges[s + 1])
                   for s in shells_used])
ax = plt.subplot(2, 2, 1, projection='polar')
ax.plot(th_prof, r_prof, '-o', color='c', linewidth=2, markersize=5)
for th, r, s in zip(th_prof, r_prof, shells_used):
    ax.annotate(str(s), (th, r), fontsize=7, color='gray')
ax.set_title(f'Radial phase profile (no organized spiral), step {t_disp} '
             f'(W = {abs(summary["A"]["W"][t_disp]):.2f} wraps)', fontsize=10)

ax = axes[0, 1]                                    # W(t)
for tag, c in (('A', 'c'), ('B', 'orange'), ('C', 'gray')):
    ax.plot(tt[post], np.abs(summary[tag]['W'][post]), color=c,
            label=f"{tag}: slope/pred={summary[tag]['slope_ratio']:+.2f}, "
                  f"R\u00b2={summary[tag]['r2']:.3f}")
xfit = tt[post].astype(float)
ax.plot(xfit, np.abs(A['slope'] * xfit +
        (A['W'][post][0] - A['slope'] * xfit[0])), 'k--', alpha=0.5,
        label='A linear fit')
ax.set_xlabel('step'); ax.set_ylabel('|W| (wraps)')
ax.set_title('Wrap count W(t): no coherent winding develops'); ax.legend(fontsize=8)

ax = axes[1, 0]                                    # shell x time phase chevrons
S = len(shells_used)
img = np.zeros((S, TIME_STEPS - KICK_STEP))
for k_, s in enumerate(shells_used):
    img[k_] = np.mod(summary['A']['thetas'][s][KICK_STEP:], 2 * np.pi)
im = ax.imshow(img, aspect='auto', origin='lower', cmap='twilight',
               extent=[KICK_STEP, TIME_STEPS, shells_used[0], shells_used[-1]])
ax.set_xlabel('step'); ax.set_ylabel('shell')
ax.set_title('Response phase (mod 2\u03c0): no differential winding')
plt.colorbar(im, ax=ax, label='\u03b8 mod 2\u03c0')

ax = axes[1, 1]                                    # persistence envelope
for tag, c in (('A', 'c'), ('B', 'orange'), ('C', 'gray')):
    ax.semilogy(tt[KICK_STEP:], summary[tag]['env'][KICK_STEP:], color=c,
                label=f"{tag} (tail/peak={summary[tag]['persist']:.2f})")
ax.set_xlabel('step'); ax.set_ylabel('mean shell response amplitude')
ax.set_title('Persistence: organized memory vs dispersal')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('phase1c_results/fig_phase1c_winding.png', dpi=200)
print("saved: fig_phase1c_winding.png")

# paper-style single figure (black, polar) in the Phase 1b idiom
fig2 = plt.figure(figsize=(8, 8))
ax2 = plt.subplot(111, projection='polar')
for tstep, alpha in ((KICK_STEP + 150, 0.35), (t_disp, 1.0)):
    thp = np.array([summary['A']['thetas'][s][tstep] for s in shells_used])
    ax2.plot(thp, r_prof, '-o', color='cyan', alpha=alpha, linewidth=2,
             markersize=4,
             label=f"step {tstep}  (W={abs(summary['A']['W'][tstep]):.2f})")
ax2.set_facecolor('black'); fig2.patch.set_facecolor('black')
ax2.tick_params(colors='white')
ax2.set_title('F$_c$ disc, Phase 1b architecture: non-winding response (diagnostic)',
              color='white', fontsize=12)
ax2.legend(facecolor='black', edgecolor='gray', labelcolor='white',
           fontsize=9, loc='lower left')
plt.tight_layout()
plt.savefig('phase1c_results/fig_phase1c_spiral.png', dpi=200,
            facecolor='black')
print("saved: fig_phase1c_spiral.png")

out = {k: {kk: vv for kk, vv in v.items()
           if kk not in ('W', 'env', 'thetas')} for k, v in summary.items()}
out['protocol'] = dict(steps=TIME_STEPS, kick_step=KICK_STEP,
                       kick_amp=KICK_AMP, seed=SEED, k_fc=K_FC,
                       coupling=COUPLING, central_mass=CENTRAL_MASS,
                       shells_used=shells_used,
                       slope_pred_wraps_per_step=float(slope_pred),
                       max_abs_response=dict(A=maxA, B=maxB, C=maxC))
out['preregistration'] = dict(H6=h6, H7=h7, H8=h8)
with open('phase1c_results/phase1c_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print("saved: phase1c_results.json")
print("\nGuard: mechanism-class viability only. No bridge to physical units.")
print("Psi  moon-wave-tree-web  To preserve the harmonic field.")
