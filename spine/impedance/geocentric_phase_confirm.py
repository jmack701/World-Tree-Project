#!/usr/bin/env python3
"""Confirmation sweep for the geocentric mixed components.
PRE-STATED: EM annual and EJ annual collapse at triplet-resolving windows
(separations 1.64e-4 / 2.19e-4 cpd vs bandwidths 1.14e-4 @24yr, 8.3e-5 @33yr);
Mars anomalistic collapses slowest (441x line 1.64e-4 from an 18,916x giant)."""
import numpy as np
from scipy import signal
d = np.load('a100_results.npz')
t, fr = d['t'], d['fr']; df = fr[1]-fr[0]; N = len(t)
R_EM, R_EJ, pw_EM, pw_EJ = d['R_EM'], d['R_EJ'], d['pw_EM'], d['pw_EJ']
fE = 1/365.256363
def refine(pw, f0):
    i = int(round(f0/df)); lo,hi = max(1,i-3), min(len(pw)-2,i+3)
    i = lo+int(np.argmax(pw[lo:hi+1]))
    lp = np.log(pw[i-1:i+2]+1e-300); den = lp[0]-2*lp[1]+lp[2]
    dl = np.clip(0.5*(lp[0]-lp[2])/den,-0.5,0.5) if den!=0 else 0
    return fr[i]+dl*df
def cv_at(x, fc, Wp, step):
    w = signal.get_window('hann', Wp); xm = x - x.mean()
    ph = [np.angle(np.sum(xm[s:s+Wp]*w*np.exp(-1j*2*np.pi*fc*t[s:s+Wp])))
          for s in range(0, N-Wp, step)]
    return 1.0 - abs(np.mean(np.exp(1j*np.array(ph)))), len(ph)
targets = [
 ('EM annual (sep 1.64e-4, mix +2E-2P)', R_EM, refine(pw_EM, fE)),
 ('EM Mars anomalistic (441x by 18,916x giant)', R_EM, refine(pw_EM, 1/686.98)),
 ('EJ annual (sep 2.19e-4, mix synodic)', R_EJ, refine(pw_EJ, fE)),
]
print(f"{'component':<46}{'16yr':>10}{'24yr':>10}{'33yr':>10}")
for nm, x, fc in targets:
    row = []
    for Wp, st in [(5844,1461),(8766,1461),(12053,2922)]:
        cv, nw = cv_at(x, fc, Wp, st)
        row.append(f"{cv:.4f}({nw})")
    print(f"{nm:<46}" + "".join(f"{r:>10}" for r in row))
print("\nreference lock levels: Mars synodic 0.0009 | Jupiter synodic 0.0003 (16 yr)")
