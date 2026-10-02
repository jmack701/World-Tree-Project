#!/usr/bin/env python3
"""
Placement Phase -- Step 1: the M-sweep on scalar F_c.
World Tree Project, September 12, 2026. Frame: claude_Placement_Model_and_Frame.md.

REGIME I (reproduction): fc_k_sweep_reference.py dynamics verbatim; gate = all
nine published Spine 2.2 rows at printed precision before any M is measured.
REGIME II (extension): ring-down Q_dis (pin 2, pin 5 semantics), line Q_coh
(inherited DERIVED estimator), M and M^-1 with regime labels (pin 3) and the
finite-pair convention (pin 4). Seed 42 throughout; single RNG stream.
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================= REGIME I -- ARCHIVED DYNAMICS, VERBATIM =================
PHI = (1 + 5**0.5) / 2
PHI_INV = 1 / PHI
N_STEPS = 100_000
SEED = 42
C0, CM1 = 0.3, 0.5
AH, NOISE_SCALE, RES_REF = 0.1, 0.2, 1.0
DT, OMEGA = 0.01, 2 * math.pi
Z4_THRESHOLD = PHI ** 3.5

def alpha(k): return (1 - 6 * k) / (1 - 9 * k)
def beta(k):  return 3 * k / (1 - 9 * k)

def run_fc(k):
    rng = np.random.RandomState(SEED)
    a, b = alpha(k), beta(k)
    c_prev, c_curr = CM1, C0
    mm = CM1 * CM1 * PHI_INV + C0 * C0
    cc = ec = rc = z4 = 0
    mm_sum = mm_max = c_sum = 0.0
    for n in range(1, N_STEPS + 1):
        t = n * DT
        c_raw = a * c_curr + b * c_prev                                  # Step 1
        if c_curr < 0:                                                   # Step 1b
            c_raw += NOISE_SCALE * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > RES_REF:
            c_raw += (c_curr - RES_REF) * math.sin(OMEGA * t)
        c_raw += AH * (math.sin(3*OMEGA*t) + math.sin(6*OMEGA*t) + math.sin(9*OMEGA*t))  # Step 2
        c_next = ((c_raw + PHI) % (2 * PHI)) - PHI                       # Step 3 (L = phi)
        mm = mm * PHI_INV + c_next * c_next                              # Step 4 (post-wrap)
        if c_next < 0:      cc += 1
        elif c_next <= 1:   ec += 1
        else:               rc += 1
        z4 += (mm >= Z4_THRESHOLD)
        mm_sum += mm; mm_max = max(mm_max, mm); c_sum += c_next
        c_prev, c_curr = c_curr, c_next                                  # Step 6
    N = N_STEPS
    return dict(C=100*cc/N, E=100*ec/N, R=100*rc/N, phi4=100*z4/N,
                mm_avg=mm_sum/N, mm_max=mm_max, c_avg=c_sum/N)
# ================= END ARCHIVED DYNAMICS =================

print("=" * 78)
print("PLACEMENT -- STEP 1: M-SWEEP  (frame: claude_Placement_Model_and_Frame.md)")
print("=" * 78)

# -------- REGIME I GATE: all nine published Spine 2.2 rows --------
PUB = {  # k: (C, E, R, phi4, MM_avg, MM_max)
 0.150: (50.5, 10.2, 39.2,  3, 4.033, 6.341),
 0.156: (50.1,  4.2, 45.7, 18, 4.784, 6.556),
 0.160: (44.8,  1.9, 53.2, 29, 5.059, 6.611),
 0.162: (42.7,  2.9, 54.4, 31, 5.019, 6.556),
 0.165: (47.8, 14.8, 37.4, 13, 4.132, 6.603),
 1/6:   (49.9, 32.5, 17.5,  2, 2.150, 6.462),
 0.168: (50.2, 49.8,  0.0,  0, 0.016, 0.650),
 0.172: (49.7, 50.3,  0.0,  0, 0.014, 0.452),
 0.180: (49.7, 50.3,  0.0,  0, 0.016, 0.384)}
print("\nREGIME I -- REPRODUCTION GATE (Spine 2.2, nine rows, archived dynamics)")
print("-" * 78)
gate_pass = True
for k, pub in PUB.items():
    r = run_fc(k)
    got = (round(r['C'],1), round(r['E'],1), round(r['R'],1),
           round(r['phi4']), round(r['mm_avg'],3), round(r['mm_max'],3))
    exp = (pub[0], pub[1], pub[2], pub[3], pub[4], pub[5])
    ok = got == exp
    gate_pass &= ok
    lab = "1/6" if abs(k-1/6) < 1e-9 else f"{k:.4f}"
    print(f"  k={lab:>7}: C/E/R {got[0]:>5.1f}/{got[1]:>5.1f}/{got[2]:>5.1f}  "
          f"phi4 {got[3]:>2d}  MM {got[4]:.3f}/{got[5]:.3f}   "
          f"{'PASS' if ok else 'DEVIATION vs ' + str(exp)}")
print(f"\n  GATE VERDICT: {'PASS -- instrument reproduces the record; '
      'REGIME II opens' if gate_pass else 'FAIL -- STOP; the deviation is the finding'}")
if not gate_pass:
    raise SystemExit(1)

# -------- REGIME II: M-sweep (cutoff at 90k; wrap stays ON) --------
CUT = 90_000
def run_fc_ringdown(k):
    """Identical dynamics; records C series; all three exogenous terms off
    for n > CUT (pin 5); Mobius wrap stays ON. Same seed, same stream."""
    rng = np.random.RandomState(SEED)
    a, b = alpha(k), beta(k)
    c_prev, c_curr = CM1, C0
    mm = CM1 * CM1 * PHI_INV + C0 * C0
    series = np.empty(N_STEPS)
    mm_sum = 0.0
    for n in range(1, N_STEPS + 1):
        t = n * DT
        c_raw = a * c_curr + b * c_prev
        if n <= CUT:
            if c_curr < 0:
                c_raw += NOISE_SCALE * abs(c_curr) * (rng.random() * 2 - 1)
            elif c_curr > RES_REF:
                c_raw += (c_curr - RES_REF) * math.sin(OMEGA * t)
            c_raw += AH * (math.sin(3*OMEGA*t) + math.sin(6*OMEGA*t) + math.sin(9*OMEGA*t))
        c_next = ((c_raw + PHI) % (2 * PHI)) - PHI
        mm = mm * PHI_INV + c_next * c_next
        mm_sum += mm
        series[n-1] = c_next
        c_prev, c_curr = c_curr, c_next
    return series, mm_sum / N_STEPS

def refine_bin(power, i, df):
    lp = np.log(power[i-1:i+2] + 1e-300)
    den = lp[0] - 2*lp[1] + lp[2]
    dl = np.clip(0.5*(lp[0]-lp[2])/den, -0.5, 0.5) if den != 0 else 0.0
    return (i + dl) * df

def line_Q(series):
    """Inherited DERIVED estimator on the pre-cutoff span."""
    x = series - series.mean()
    n = len(x)
    w = np.hanning(n)
    P = np.abs(np.fft.rfft(x*w))**2
    df = 1.0 / n                       # cycles/step
    i = 1 + int(np.argmax(P[1:]))
    f0 = refine_bin(P, i, df)
    half = P[i] / 2.0
    j = i
    while j > 1 and P[j-1] >= half and i - j < 200: j -= 1
    fl = (j-1 + (half-P[j-1])/(P[j]-P[j-1]))*df if P[j-1] < half else j*df
    m = i
    while m < len(P)-2 and P[m+1] >= half and m - i < 200: m += 1
    fr = (m + (P[m]-half)/(P[m]-P[m+1]))*df if P[m+1] < half else m*df
    dF = fr - fl
    Qc = f0 / dF
    Qres = f0 / (1.44*df)
    return f0, dF, Qc, Qres, (dF <= 1.5*1.44*df)

def ringdown_fit(series):
    """Pin-2 fit on steps CUT..end; returns tau_A, R2, window, f_ring, resolvable."""
    tail = series[CUT:]
    idx = np.arange(1, len(tail)-1)
    pk = idx[(np.abs(tail[idx]) >= np.abs(tail[idx-1])) &
             (np.abs(tail[idx]) >= np.abs(tail[idx+1])) &
             (np.abs(tail[idx]) > 1e-12)]
    if len(pk) < 4:
        return None
    A = np.abs(tail[pk])
    sl, b0 = np.polyfit(pk.astype(float), np.log(A), 1)
    pred = sl*pk + b0
    ss = 1 - np.sum((np.log(A)-pred)**2)/max(1e-30, np.sum((np.log(A)-np.log(A).mean())**2))
    ratio = A[-1] / A[0]
    resolvable = (sl < 0) and (ratio <= 0.9)
    sc = np.sign(tail[pk[0]:pk[-1]+1])
    f_ring = np.count_nonzero(np.diff(sc[sc != 0])) / 2.0 / max(1, pk[-1]-pk[0])
    return dict(tau_A=(-1.0/sl if sl < 0 else np.inf), R2=ss,
                win=(int(pk[0]), int(pk[-1])), n_pk=len(pk),
                ratio=ratio, f_ring=f_ring, resolvable=bool(resolvable))

K_PHI = PHI / (9*PHI - 3)
SWEEP = [K_PHI, 0.150, 0.156, 0.160, 0.162, 0.165, 1/6, 0.168, 0.172, 0.180, 0.2227]
SKY_M = 100.0

print("\nREGIME II -- M-SWEEP (eleven points; pin-5 cutoff at step 90,000; wrap ON)")
print("-" * 78)
rows = []
hdr = (f"{'k':>8} {'regime':>10} {'MMavg':>6} | {'f0':>7} {'Q_coh':>7} {'rl':>2} | "
       f"{'tau_A':>8} {'tau_an':>7} {'R2':>5} | {'1/Qdis':>8} {'M':>9} {'M^-1':>9}")
print(hdr); print("-" * len(hdr))
for k in SWEEP:
    series, mmavg = run_fc_ringdown(k)
    regime = "sustained" if k < 1/6 - 1e-12 else ("marginal" if abs(k-1/6) < 1e-9 else "convergent")
    f0, dF, Qc, Qres, rl = line_Q(series[:CUT])
    fit = ringdown_fit(series)
    disc = alpha(k)**2 + 4*beta(k)
    lam = math.sqrt(abs(beta(k))) if disc < 0 else max(abs((alpha(k)+math.sqrt(disc))/2),
                                                       abs((alpha(k)-math.sqrt(disc))/2))
    tau_an = (-1.0/math.log(lam)) if lam < 1 else float('inf')
    if fit is None or not fit['resolvable']:
        inv_qdis, M, Minv = 0.0, 0.0, float('inf')
        qdis = float('inf'); tau_s = fit['tau_A'] if fit else float('inf')
        r2 = fit['R2'] if fit else float('nan'); fr_ = fit['f_ring'] if fit else float('nan')
    else:
        tau_s = fit['tau_A']; r2 = fit['R2']; fr_ = fit['f_ring']
        qdis = math.pi * f0 * tau_s
        inv_qdis = 1.0/qdis
        M = Qc / qdis
        Mid = 1.0/(math.pi * dF * tau_s)      # named identity check
        assert abs(M - Mid)/M < 1e-9
        Minv = 1.0/M
    lab = "1/6" if abs(k-1/6) < 1e-9 else f"{k:.4f}"
    print(f"{lab:>8} {regime:>10} {mmavg:>6.3f} | {f0:>7.4f} {Qc:>7.1f} {('Y' if rl else 'n'):>2} | "
          f"{(f'{tau_s:8.1f}' if np.isfinite(tau_s) else '     inf')} "
          f"{(f'{tau_an:7.1f}' if np.isfinite(tau_an) else '    inf')} "
          f"{(f'{r2:5.3f}' if np.isfinite(r2) else '   --')} | "
          f"{inv_qdis:>8.2e} {M:>9.3g} {(f'{Minv:9.3g}' if np.isfinite(Minv) else '      inf')}")
    rows.append(dict(k=k, regime=regime, mm_avg=mmavg, f0=f0, dF=dF, Q_coh=Qc,
                     Q_res=Qres, res_limited=bool(rl),
                     tau_A=(None if not np.isfinite(tau_s) else tau_s),
                     tau_analytic=(None if not np.isfinite(tau_an) else tau_an),
                     R2=(None if not np.isfinite(r2) else r2),
                     f_ring=(None if not np.isfinite(fr_) else fr_),
                     inv_Q_dis=inv_qdis, M=M,
                     M_inv=(None if not np.isfinite(Minv) else Minv),
                     in_domain=(regime == "sustained")))

cross = [r['k'] for r in rows if r['in_domain'] and r['M'] >= SKY_M]
print("\nPre-stated questions:")
print(f"  in-domain (sustained) crossing region M >= 10^2: "
      f"{cross if cross else 'EMPTY'}")
mpk = max((r for r in rows), key=lambda r: r['M'])
print(f"  M peak on the typed full curve: k = {mpk['k']:.4f} (regime {mpk['regime']})")
print(f"  toward criticality (sustained side): M -> 0 by the finite pair; "
      f"M^-1 diverges; 1/Q_dis = 0 at every unresolvable point")

json.dump(dict(gate='PASS', sky_M=SKY_M, rows=rows,
               in_domain_crossing_k=cross),
          open('/mnt/user-data/outputs/placement_step1_results.json', 'w'), indent=2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ks = [r['k'] for r in rows]; Ms = [max(r['M'], 1e-3) for r in rows]
sus = [r['in_domain'] for r in rows]
ax1.axvspan(0.160, 0.164, color='#1e8449', alpha=0.15, label='the Hold')
ax1.axvspan(0.2227, 0.2261, color='#6c3483', alpha=0.15, label='coupled-lock band')
ax1.axhline(SKY_M, color='#b03a2e', lw=1.2, ls='--', label='sky M >= 10^2 (one-sided)')
ax1.semilogy([k for k,s in zip(ks,sus) if s], [m for m,s in zip(Ms,sus) if s],
             'o', color='#1a5276', label='sustained (in-domain): M = 0 shown at floor')
ax1.semilogy([k for k,s in zip(ks,sus) if not s], [m for m,s in zip(Ms,sus) if not s],
             's', mfc='none', mec='#e67e22', label='convergent/marginal (typed)')
ax1.axvline(1/6, color='k', lw=0.7, ls=':')
ax1.set_ylabel('M (type-labeled)'); ax1.legend(fontsize=7, loc='center left')
ax1.set_title('Placement step 1 -- M(k) on scalar F_c (placement_step1_msweep.py)', fontsize=10)
ax2.plot(ks, [r['inv_Q_dis'] for r in rows], 'o-', ms=4, lw=0.8, color='#1a5276')
ax2.axvline(1/6, color='k', lw=0.7, ls=':')
ax2.set_xlabel('k'); ax2.set_ylabel('1/Q_dis')
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/fig_placement_msweep.png', dpi=150)
print("\nArtifacts: placement_step1_results.json, fig_placement_msweep.png")
