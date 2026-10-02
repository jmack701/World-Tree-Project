#!/usr/bin/env python3
"""
Geocentric completion of the three-system comparison.
World Tree Project, July 2026. Instruction relayed from the Opus spiral.

PRE-STATED PREDICTIONS (recorded in the conversation before execution):
 P1  Two-ellipse Kepler baseline (zero fitted parameters; standard J2000
     elements; a from data means per the section 5.2 convention)
     reproduces each geocentric series' full-span SE within ~0.01.
 P2  dSE_EM and dSE_EJ positive but < +0.01 at every scale; Moon-Earth
     remains the outlier by >= 1 order of magnitude in both geometries.
 P3  Resolvable components phase-lock (CV < 0.01 at synodic-resolving
     windows); measured synodic frequencies satisfy f_syn = 1/T_E - 1/T_p
     to sub-bin.
 P4  Line forests identify on the integer basis n1*f_E + n2*f_p.
 P5  No sub-classification emerges; the discriminator remains the
     enrichment fraction.
"""
import numpy as np
from scipy import signal

d = np.load('a100_results.npz')
t, fr = d['t'], d['fr']
R_EM, R_EJ = d['R_EM'], d['R_EJ']
pw_EM, pw_EJ = d['pw_EM'], d['pw_EJ']
df = fr[1]-fr[0]; N = len(t)
JD0 = 2424151.5           # series start, 1925-01-01 TDB
J2000 = 2451545.0
D_MARS, D_JUP = 6779.84, 139822.0
AU = 149_597_870.7

# Standard J2000 mean elements (Standish approximate elements, declared;
# angles deg; a taken from the data means per the registered convention)
ELEM = {
 'earth':  dict(e=0.01671123, i=-0.00001531, Om=0.0,          w_bar=102.93768193, L0=100.46457166, T=365.256363),
 'mars':   dict(e=0.09339410, i=1.84969142,  Om=49.55953891,  w_bar=-23.94362959, L0=-4.55343205,  T=686.980),
 'jupiter':dict(e=0.04838624, i=1.30439695,  Om=100.47390909, w_bar=14.72847983,  L0=34.39644051,  T=4332.589),
}
A_DATA = {'earth': 149_618_830.0, 'mars': 228_941_831.0, 'jupiter': 777_946_125.0}  # km, Table 5.18

def kepler_vec(body, t_days):
    E = ELEM[body]; a = A_DATA[body]
    n = 360.0 / E['T']                              # deg/day
    L = E['L0'] + n * (JD0 - J2000 + t_days)        # mean longitude
    M = np.radians(np.mod(L - E['w_bar'], 360.0))
    ecc = E['e']
    Ea = M.copy()
    for _ in range(25):
        Ea = Ea - (Ea - ecc*np.sin(Ea) - M)/(1 - ecc*np.cos(Ea))
    nu = 2*np.arctan2(np.sqrt(1+ecc)*np.sin(Ea/2), np.sqrt(1-ecc)*np.cos(Ea/2))
    r = a*(1-ecc**2)/(1+ecc*np.cos(nu))
    w = np.radians(E['w_bar'] - E['Om'])            # argument of perihelion
    Om = np.radians(E['Om']); inc = np.radians(E['i'])
    u = nu + w
    x = r*(np.cos(Om)*np.cos(u) - np.sin(Om)*np.sin(u)*np.cos(inc))
    y = r*(np.sin(Om)*np.cos(u) + np.cos(Om)*np.sin(u)*np.cos(inc))
    z = r*np.sin(u)*np.sin(inc)
    return np.stack([x, y, z])

rE = kepler_vec('earth', t); rM = kepler_vec('mars', t); rJ = kepler_vec('jupiter', t)
kep_EM = np.linalg.norm(rM - rE, axis=0) / D_MARS
kep_EJ = np.linalg.norm(rJ - rE, axis=0) / D_JUP

def se(x):
    n = len(x); xw = (x-np.mean(x))*signal.get_window('hann', n)
    p = np.abs(np.fft.rfft(xw))**2; tot = p.sum()
    if tot == 0: return 0.0
    pn = p/tot; pn = pn[pn>0]
    return -np.sum(pn*np.log(pn))/np.log(len(p))

def windowed(x, Lw, step):
    return float(np.mean([se(x[s:s+Lw]) for s in range(0, len(x)-Lw, step)]))

print("="*74)
print("GEOCENTRIC COMPLETION -- two-ellipse baselines, dSE, phases, decomposition")
print("="*74)

print("\n--- Baseline sanity (P1): full-span SE and gross agreement ---")
for name, act, kep in [('R_EM', R_EM, kep_EM), ('R_EJ', R_EJ, kep_EJ)]:
    corr = np.corrcoef(act, kep)[0,1]
    print(f"  {name}: actual mean {act.mean():9.2f} / baseline mean {kep.mean():9.2f}"
          f" | corr {corr:.5f} | SE act {se(act):.4f} / SE kep {se(kep):.4f}"
          f" | dSE(full) {se(act)-se(kep):+.4f}")

print("\n--- dSE sweep (P2), same scales as Table 5.19 ---")
scales = [(365, 60), (1461, 243), (5844, 974), (None, None)]
hdr = f"{'system':<16}" + "".join(f"{(str(s[0]) if s[0] else 'FULL'):>10}" for s in scales)
print(hdr); print("-"*len(hdr))
rows = {}
for name, act, kep in [('Earth-Mars', R_EM, kep_EM), ('Earth-Jupiter', R_EJ, kep_EJ)]:
    vals = []
    for Lw, st in scales:
        vals.append((se(act)-se(kep)) if Lw is None else (windowed(act,Lw,st)-windowed(kep,Lw,st)))
    rows[name] = vals
    print(f"{name:<16}" + "".join(f"{v:>+10.4f}" for v in vals))
print(f"{'Moon-Earth ref':<16}" + "".join(f"{v:>+10.4f}" for v in [0.0328, 0.0351, 0.0291, 0.0233]))

print("\n--- Phase analysis (P3): 16-yr windows (5844 d), 4-yr step ---")
def refine(pw, f0):
    i = int(round(f0/df)); lo,hi = max(1,i-3), min(len(pw)-2,i+3)
    i = lo+int(np.argmax(pw[lo:hi+1]))
    lp = np.log(pw[i-1:i+2]+1e-300); den = lp[0]-2*lp[1]+lp[2]
    dl = np.clip(0.5*(lp[0]-lp[2])/den,-0.5,0.5) if den!=0 else 0
    return fr[i]+dl*df
def circ_var(th): return 1.0 - abs(np.mean(np.exp(1j*np.array(th))))
def phase_cv(x, comps, Wp=5844, step=1461):
    w = signal.get_window('hann', Wp); xm = x - x.mean()
    out = {}
    for nm, fc in comps.items():
        ph = [np.angle(np.sum(xm[s:s+Wp]*w*np.exp(-1j*2*np.pi*fc*t[s:s+Wp])))
              for s in range(0, N-Wp, step)]
        out[nm] = (circ_var(ph), len(ph))
    return out

fE = 1/365.256363
f_syn_M_meas = refine(pw_EM, 1/779.94); f_syn_J_meas = refine(pw_EJ, 1/398.88)
f_syn_M_pred = fE - 1/686.980; f_syn_J_pred = fE - 1/4332.589
print(f"  synodic identity, Mars:    measured {1/f_syn_M_meas:8.3f} d  vs 1/T_E-1/T_p {1/f_syn_M_pred:8.3f} d"
      f"  (delta {abs(f_syn_M_meas-f_syn_M_pred)/df:.2f} bins)")
print(f"  synodic identity, Jupiter: measured {1/f_syn_J_meas:8.3f} d  vs 1/T_E-1/T_p {1/f_syn_J_pred:8.3f} d"
      f"  (delta {abs(f_syn_J_meas-f_syn_J_pred)/df:.2f} bins)")

cv_EM = phase_cv(R_EM, {'Mars synodic': f_syn_M_meas,
                        'Mars anomalistic': refine(pw_EM, 1/686.98),
                        'annual': refine(pw_EM, fE)})
cv_EJ = phase_cv(R_EJ, {'Jupiter synodic': f_syn_J_meas,
                        'half synodic': refine(pw_EJ, 2*f_syn_J_meas),
                        'annual': refine(pw_EJ, fE)})
for label, cvs in [('Earth-Mars', cv_EM), ('Earth-Jupiter', cv_EJ)]:
    for nm, (cv, nw) in cvs.items():
        print(f"  {label:<14} {nm:<18} CV = {cv:.4f}  ({nw} windows)")

print("\n--- Spectral decomposition (P4): top peaks on n1*f_E + n2*f_p basis ---")
def local_floor(p, i, w=10):
    lo,hi = max(0,i-w), min(len(p), i+w+1)
    return np.median(np.concatenate([p[lo:i], p[i+1:hi]]))
def decompose(pw, fp, label, extra=None):
    peaks = []
    for i in range(2, len(pw)-1):
        fl = local_floor(pw, i)
        if fl>0 and pw[i] > 3*fl and pw[i]>=pw[i-1] and pw[i]>=pw[i+1]:
            peaks.append((pw[i], i))
    peaks.sort(reverse=True)
    print(f"  {label}:")
    shown = 0
    for p, i in peaks:
        if shown >= 10: break
        f = fr[i]; best = (None, 9e9)
        for n1 in range(-4,5):
            for n2 in range(-4,5):
                fc = n1*fE + n2*fp
                if fc <= 0: continue
                dd = abs(fc-f)
                if dd < best[1]: best = ((n1,n2), dd)
        ident = ""
        if best[1] < 1.5*df:
            n1, n2 = best[0]
            ident = f"[{n1:+d}E{n2:+d}P]"
        elif extra:
            for enm, ef in extra:
                if abs(ef-f) < 1.5*df: ident = f"[{enm}]"; break
        print(f"    {1/f:10.3f} d  SNR {p/local_floor(pw,i):12.1f}x  {ident}")
        shown += 1
f_lun = 1/29.5306
decompose(pw_EM, 1/686.980, "Earth-Mars", extra=[('lunar', f_lun), ('MJ synodic', 1/816.4)])
decompose(pw_EJ, 1/4332.589, "Earth-Jupiter", extra=[('lunar', f_lun)])
print("\n(P = planet orbital frequency; E = Earth orbital frequency)")
