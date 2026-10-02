#!/usr/bin/env python3
"""
Q Bridge -- Step 4: Q_phase.
World Tree Project, September 12, 2026. Frame: claude_Q_Bridge_Model_and_Frame.md.

REGIME I (reproduction): archived demodulation verbatim from
impedance_match_phase_confirmation.py at W=730, step=365; gate = Table 5.15
to published decimals before the lag test runs.
REGIME II (extension): S2 lag decision rule on the harmonic-closure phases,
then the S1 registered mapping (rejected branch coded, invoked only if the
test rules). Deterministic throughout; no random stream.
"""
import re, json
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DT_DAYS = 1.0
MOON_DIAMETER = 3474.8

# ================= REGIME I -- ARCHIVED CODE, VERBATIM =================
def parse_horizons_full(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    soe = content.index('$$SOE'); eoe = content.index('$$EOE')
    block = content[soe+6:eoe].strip()
    jd, X, Y, Z, RG = [], [], [], [], []
    fnum = r'([-+]?\d+\.\d+E[-+]\d+|[-+]?\d+\.\d+)'
    for line in block.split('\n'):
        line = line.strip()
        if '= A.D.' in line:
            jd.append(float(line.split('=')[0]))
        elif line.startswith('X ='):
            m = re.findall(r'[XYZ]\s*=\s*' + fnum, line)
            X.append(float(m[0])); Y.append(float(m[1])); Z.append(float(m[2]))
        elif line.startswith('LT='):
            m = re.search(r'RG=\s*' + fnum, line)
            RG.append(float(m.group(1)))
    return (np.array(jd), np.array(X), np.array(Y), np.array(Z), np.array(RG))

def compute_power(data):
    n = len(data)
    x = data - np.mean(data)
    w = signal.get_window('hann', n)
    fft_vals = np.fft.rfft(x * w)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=DT_DAYS)
    return power, freqs

_, _, _, _, rg_me = parse_horizons_full('/mnt/user-data/uploads/Moon-Earth.txt')
R_ME = rg_me / MOON_DIAMETER
N = len(R_ME)
pw, fr = compute_power(R_ME)
df = fr[1] - fr[0]
t = np.arange(N) * DT_DAYS

def refine(pw, f0):
    i = int(round(f0/df)); lo, hi = max(1, i-3), min(len(pw)-2, i+3)
    i = lo + int(np.argmax(pw[lo:hi+1]))
    lp = np.log(pw[i-1:i+2]+1e-300); den = lp[0]-2*lp[1]+lp[2]
    dl = np.clip(0.5*(lp[0]-lp[2])/den, -0.5, 0.5) if den != 0 else 0
    return fr[i]+dl*df

f_anom = refine(pw, 1/27.55); f_syn = refine(pw, 1/29.5306)
f_ev_t = 2*f_syn - f_anom; f_hf_t = 2*f_anom
x = R_ME - np.mean(R_ME)

def circ_var(th): return 1.0 - abs(np.mean(np.exp(1j*th)))
def circ_mean_deg(th): return np.degrees(np.angle(np.mean(np.exp(1j*th))))

Wp, step = 730, 365
w = signal.get_window('hann', Wp)
comp = {'anomalistic': f_anom, 'synodic': f_syn,
        'evection': f_ev_t, 'half': f_hf_t}
ph = {k: [] for k in comp}
centers = []
for s0 in range(0, N-Wp, step):
    seg = x[s0:s0+Wp]*w; tt = t[s0:s0+Wp]
    centers.append(s0 + Wp/2)
    for k, fc in comp.items():
        ph[k].append(np.angle(np.sum(seg*np.exp(-1j*2*np.pi*fc*tt))))
ph = {k: np.array(v) for k, v in ph.items()}
centers = np.array(centers)
cl_ev = np.angle(np.exp(1j*(ph['evection']-(2*ph['synodic']-ph['anomalistic']))))
cl_hf = np.angle(np.exp(1j*(ph['half']-2*ph['anomalistic'])))
d_as  = np.angle(np.exp(1j*(ph['anomalistic']-ph['synodic'])))
# ================= END ARCHIVED CODE =================

print("=" * 76)
print("Q BRIDGE -- STEP 4: Q_phase  (frame: claude_Q_Bridge_Model_and_Frame.md)")
print("=" * 76)
nw = len(centers)
print(f"\n{nw} windows of {Wp} d at {step}-d steps; refined f_anom -> "
      f"{1/f_anom:.4f} d")

print("\nREGIME I -- REPRODUCTION GATE (Table 5.15, archived demodulation)")
print("-" * 76)
drift = np.polyfit(centers, np.unwrap(d_as), 1)[0] * np.degrees(1) * 365.25
gate_vals = [
    ("anomalistic CV",            circ_var(ph['anomalistic']), 0.0002, 4),
    ("evection CV",               circ_var(ph['evection']),    0.0007, 4),
    ("half-anomalistic CV",       circ_var(ph['half']),        0.0011, 4),
    ("harmonic closure CV",       circ_var(cl_hf),             0.0003, 4),
    ("harmonic closure mean deg", circ_mean_deg(cl_hf),      -179.98,  2),
    ("synodic CV",                circ_var(ph['synodic']),     0.9684, 4),
    ("evection closure CV",       circ_var(cl_ev),             0.8409, 4),
    ("evection closure mean deg", circ_mean_deg(cl_ev),        +1.58,  2),
    ("anom-syn difference CV",    circ_var(d_as),              0.9682, 4),
    ("anom-syn drift deg/yr",     drift,                      +40.91,  2),
]
gate_pass = True
for name, got, pub, dec in gate_vals:
    ok = round(got, dec) == round(pub, dec)
    gate_pass &= ok
    print(f"  {name:<26} {got:>10.{dec}f}   published {pub:>9.{dec}f}   "
          f"{'PASS' if ok else 'DEVIATION'}")
print(f"\n  GATE VERDICT: {'PASS -- instrument reproduces the record; '
      'REGIME II opens' if gate_pass else 'FAIL -- STOP; the deviation is the finding'}")
if not gate_pass:
    raise SystemExit(1)

print("\nREGIME II -- EXTENSION (S2 lag decision rule; S1 mapping)")
print("-" * 76)
# lag test on the registered input: harmonic-closure phases
theta = cl_hf
lags, V = [], []
for k in range(2, nw):                      # lag >= 730 d only
    dphi = np.angle(np.exp(1j*(theta[k:] - theta[:-k])))
    lags.append(k*step); V.append(np.mean(dphi**2))
lags = np.array(lags, float); V = np.array(V)

# M0: constant; M1: a + b*L (unweighted OLS over lag bins)
c0 = V.mean(); rss0 = np.sum((V-c0)**2)
A = np.vstack([np.ones_like(lags), lags]).T
coef, rss1v, *_ = np.linalg.lstsq(A, V, rcond=None)
a1, b1 = coef; res1 = V - A@coef; rss1 = np.sum(res1**2)
n = len(V)
sigma2 = rss1/(n-2)
se_b = np.sqrt(sigma2/np.sum((lags-lags.mean())**2))
t_b = b1/se_b
aic0 = n*np.log(rss0/n) + 2*1
aic1 = n*np.log(rss1/n) + 2*2
reject = (b1 > 0) and (t_b >= 3.0) and (aic1 < aic0)
print(f"  lag bins: {n} (730 d to {int(lags[-1]):,} d)")
print(f"  M0 constant: V = {c0:.3e} rad^2")
print(f"  M1 linear:   V = {a1:.3e} + {b1:.3e}*L;  slope t = {t_b:+.2f}  "
      f"(threshold >= +3.00)")
print(f"  AIC: constant {aic0:.1f} vs linear {aic1:.1f}  -> "
      f"{'linear preferred' if aic1 < aic0 else 'constant preferred'}")
print(f"  DECISION: {'REJECT white-phase -> random-walk branch applies' if reject
      else 'white-phase-noise model CONFIRMED (flat); registered branch applies'}")

# S1 registered mapping (or S2 rejected branch, only as ruled)
T_span = centers[-1] - centers[0]
CV_in = circ_var(cl_hf)          # registered input, gate-reproduced
CV_xc = circ_var(ph['anomalistic'])  # stated cross-check
if not reject:
    Q_phase = f_anom * T_span
    branch = "white phase noise (S1): Q_phase = f0 * T_span, lower bound"
    rw_val = None
else:
    Q_phase = np.pi * f_anom * T_span / (6.0 * CV_in)
    branch = "random walk (S2 rejected branch): Q_phase = pi*f0*T/(6*CV)"
    rw_val = Q_phase
print(f"\n  T_span (first-to-last window center) = {T_span:,.0f} d")
print(f"  branch: {branch}")
print(f"  Q_phase >= {Q_phase:,.1f}"
      + ("" if not reject else "  (point estimate)"))
print(f"  carrier power retention e^(-2*CV): input CV {CV_in:.4f} -> "
      f"{np.exp(-2*CV_in):.5f};  cross-check CV {CV_xc:.4f} -> "
      f"{np.exp(-2*CV_xc):.5f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))
ax1.plot(lags/365.25, V, 'o', ms=3, color='#1a5276', label='V(L) = <dphi^2>')
ax1.axhline(c0, color='#1e8449', lw=1.2, label=f'M0 constant {c0:.2e}')
ax1.plot(lags/365.25, a1 + b1*lags, '--', color='#b03a2e', lw=1,
         label=f'M1 linear (slope t = {t_b:+.2f})')
ax1.set_xlabel('window-center lag (yr)'); ax1.set_ylabel('V(L)  (rad$^2$)')
ax1.set_title('S2 lag decision rule -- harmonic-closure phase pairs, '
              'lag >= 730 d', fontsize=10)
ax1.legend(fontsize=8)
ax2.plot(1925 + centers/365.25, np.degrees(cl_hf), 'o-', ms=2.5, lw=0.7,
         color='#6c3483')
ax2.axhline(-179.98, color='#b03a2e', lw=0.9, ls='--', label='-179.98 deg')
ax2.set_xlabel('window center (yr)')
ax2.set_ylabel('harmonic closure phase (deg)')
ax2.set_title('Registered input: half - 2*anomalistic closure, 99 windows',
              fontsize=10)
ax2.legend(fontsize=8)
fig.suptitle('Q Bridge step 4 (q_bridge_qphase.py)', fontsize=10)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/fig_qbridge_lagtest.png', dpi=150)

json.dump({'gate': 'PASS', 'n_windows': int(nw), 'T_span_days': float(T_span),
           'f_anom_cpd': float(f_anom), 'lag_test': {
               'n_bins': int(n), 'const': float(c0), 'slope': float(b1),
               'slope_t': float(t_b), 'aic_const': float(aic0),
               'aic_linear': float(aic1), 'reject_white_phase': bool(reject)},
           'branch': branch, 'Q_phase_lower_bound': float(Q_phase),
           'CV_input': float(CV_in), 'CV_crosscheck': float(CV_xc),
           'carrier_retention_input': float(np.exp(-2*CV_in)),
           'carrier_retention_crosscheck': float(np.exp(-2*CV_xc))},
          open('/mnt/user-data/outputs/q_bridge_qphase_results.json', 'w'),
          indent=2)
print("\nArtifacts: q_bridge_qphase_results.json, fig_qbridge_lagtest.png")
