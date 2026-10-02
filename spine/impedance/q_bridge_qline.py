#!/usr/bin/env python3
"""
Q Bridge -- Step 3: Q_line (six registered components).
World Tree Project, September 12, 2026. Frame: claude_Q_Bridge_Model_and_Frame.md.

REGIME I (reproduction): archived functions verbatim from
impedance_match_100yr_analysis.py; gate = Table 5.12/5.13 detections
(refined frequencies + SNRs) must reproduce before any width is measured.
REGIME II (extension): half-power widths, Q_line, Q_res, verdicts, per S5
of the frame. Deterministic throughout; no random stream.
"""
import re, json
import numpy as np
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DT_DAYS = 1.0
SUN_DIAMETER, MOON_DIAMETER = 1391016.0, 3474.8

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

def compute_spectral_entropy(data, window='hann'):
    n = len(data)
    x = data - np.mean(data)
    w = signal.get_window(window, n)
    xw = x * w
    fft_vals = np.fft.rfft(xw)
    power = np.abs(fft_vals) ** 2
    tot = np.sum(power)
    if tot == 0:
        return 0.0, power, np.fft.rfftfreq(n, d=DT_DAYS)
    p = power / tot
    pn = p[p > 0]
    se = -np.sum(pn * np.log(pn)) / np.log(len(p))
    freqs = np.fft.rfftfreq(n, d=DT_DAYS)
    return se, power, freqs

def find_peaks_with_noise(power, freqs, noise_window=10, threshold=3.0):
    peaks = []
    for i in range(1, len(power) - 1):
        lo = max(0, i - noise_window)
        hi = min(len(power), i + noise_window + 1)
        local = np.concatenate([power[lo:i], power[i+1:hi]])
        nf = np.median(local)
        if nf > 0 and power[i] > threshold * nf:
            if power[i] >= power[i-1] and power[i] >= power[i+1]:
                peaks.append({'freq': freqs[i],
                              'period_days': 1.0/freqs[i] if freqs[i] > 0 else np.inf,
                              'power': power[i],
                              'snr': power[i]/nf,
                              'bin': i})
    peaks.sort(key=lambda x: x['power'], reverse=True)
    return peaks

def refine_peak_freq(power, freqs, f_guess):
    i = int(np.argmin(np.abs(freqs - f_guess)))
    lo, hi = max(1, i-3), min(len(power)-2, i+3)
    i = lo + int(np.argmax(power[lo:hi+1]))
    if i <= 0 or i >= len(power)-1:
        return freqs[i], i
    lp = np.log(power[i-1:i+2] + 1e-300)
    denom = (lp[0] - 2*lp[1] + lp[2])
    delta = 0.5 * (lp[0] - lp[2]) / denom if denom != 0 else 0.0
    delta = np.clip(delta, -0.5, 0.5)
    df = freqs[1] - freqs[0]
    return freqs[i] + delta * df, i
# ================= END ARCHIVED CODE =================

print("=" * 76)
print("Q BRIDGE -- STEP 3: Q_line   (frame: claude_Q_Bridge_Model_and_Frame.md)")
print("=" * 76)

jd_me, _, _, _, rg_me = parse_horizons_full('/mnt/user-data/uploads/Moon-Earth.txt')
N = len(rg_me)
R_ME = rg_me / MOON_DIAMETER
_, power, freqs = compute_spectral_entropy(R_ME)
dfbin = freqs[1] - freqs[0]
T_rec = N * DT_DAYS

print(f"\nN = {N} daily records; bin width = {dfbin:.3e} cyc/d "
      f"(published: 2.738e-05)")

# -------- REGIME I GATE: Table 5.12 / 5.13 reproduction --------
print("\nREGIME I -- REPRODUCTION GATE (Table 5.12 / 5.13, archived estimator)")
print("-" * 76)
peaks = find_peaks_with_noise(power, freqs)
bypin = {p['bin']: p for p in peaks}

def match_within_two_bins(period_pred):
    f_pred = 1.0 / period_pred
    bin_pred = f_pred / dfbin
    cands = [p for p in peaks if abs(p['bin'] - bin_pred) <= 2.0]
    return max(cands, key=lambda p: p['power']) if cands else None

# published gate values: (name, predicted period, found period, SNR)
GATE = [("anomalistic",      27.55,    27.5460, 223282.),
        ("synodic",          29.5306,  29.5279,  44705.),
        ("evection",         31.81,    31.8171, 767950.),
        ("half-anomalistic", 13.78,    13.7782, 847485.)]
gate_rows, gate_pass = [], True
for name, ppred, pfound_pub, snr_pub in GATE:
    pk = match_within_two_bins(ppred)
    ok = (pk is not None and abs(pk['period_days'] - pfound_pub) < 5e-4
          and abs(pk['snr'] - snr_pub)/snr_pub < 5e-5)
    gate_pass &= ok
    gate_rows.append((name, pk, ok))
    print(f"  {name:<16} found {pk['period_days']:9.4f} d  SNR {pk['snr']:>11,.0f}x"
          f"   published {pfound_pub:9.4f} d / {snr_pub:>9,.0f}x   "
          f"{'PASS' if ok else 'DEVIATION'}")

# low-frequency neighborhood, Table 5.13 (raw, x local floor)
def floor_ratio(i, nw=10):
    lo, hi = max(0, i-nw), min(len(power), i+nw+1)
    local = np.concatenate([power[lo:i], power[i+1:hi]])
    return power[i] / np.median(local)
T513 = {5: 8.03, 6: 12.58, 10: 234.4, 11: 3136.4, 12: 1756.3}
print("  low-frequency neighborhood (x local floor):")
for b, pub in T513.items():
    r = floor_ratio(b)
    ok = abs(r - pub)/pub < 2e-3
    gate_pass &= ok
    print(f"    bin {b:>2}  period {1/freqs[b]:8.1f} d   {r:9.2f}x   "
          f"published {pub:9.2f}x   {'PASS' if ok else 'DEVIATION'}")

# refined frequencies (published: 27.5548 / 29.5308 / 31.8116 / 13.7772; apsidal 3234.4)
REF = [("anomalistic", 1/27.55, 27.5548), ("synodic", 1/29.5306, 29.5308),
       ("evection", 1/31.81, 31.8116), ("half-anomalistic", 1/13.78, 13.7772),
       ("apsidal", 1/3231.50, 3234.4)]
refined = {}
print("  parabolic refinement:")
for name, fg, pub in REF:
    fref, ibin = refine_peak_freq(power, freqs, fg)
    refined[name] = (fref, ibin)
    per = 1.0/fref
    tol = 0.05 if name == "apsidal" else 5e-4
    ok = abs(per - pub) < (0.5 if name == "apsidal" else 1e-3)
    gate_pass &= ok
    print(f"    {name:<16} refined {per:9.4f} d   published {pub:9.4f} d   "
          f"{'PASS' if ok else 'DEVIATION'}")

print(f"\n  GATE VERDICT: {'PASS -- instrument reproduces the record; '
      'REGIME II opens' if gate_pass else 'FAIL -- STOP; the deviation is the finding'}")
if not gate_pass:
    raise SystemExit(1)

# -------- REGIME II EXTENSION: half-power widths, Q_line --------
print("\nREGIME II -- EXTENSION (S5 protocol: half-power width, raw grid, "
      "linear interpolation)")
print("-" * 76)
HANN_FWHM_BINS = 1.44
w_window = HANN_FWHM_BINS * dfbin

def half_power_width(ibin, f_lo=None, f_hi=None, cap=60):
    Pp = power[ibin]; half = Pp / 2.0
    i = ibin
    while i > 1 and power[i-1] >= half and (ibin - i) < cap:
        i -= 1
        if f_lo is not None and freqs[i] < f_lo: break
    fl = freqs[i-1] + (freqs[i] - freqs[i-1]) * \
         (half - power[i-1]) / (power[i] - power[i-1]) if power[i-1] < half \
         else freqs[i]
    j = ibin
    while j < len(power)-2 and power[j+1] >= half and (j - ibin) < cap:
        j += 1
        if f_hi is not None and freqs[j] > f_hi: break
    fr_ = freqs[j] + (freqs[j+1] - freqs[j]) * \
          (power[j] - half) / (power[j] - power[j+1]) if power[j+1] < half \
          else freqs[j]
    return fr_ - fl, fl, fr_

# synodic inter-line gap: midpoints to triplet neighbors 29.268 / 29.793 d
f_gap_lo = 0.5*(1/29.5279 + 1/29.793)
f_gap_hi = 0.5*(1/29.5279 + 1/29.268)

results = {}
print(f"  window FWHM = {HANN_FWHM_BINS} bins = {w_window:.3e} cyc/d;  "
      f"1.5x tolerance = {1.5*w_window:.3e} cyc/d\n")
hdr = (f"  {'component':<16} {'f0 (d)':>9} {'dF (cyc/d)':>11} {'dF (bins)':>9} "
       f"{'Q_line':>8} {'Q_res':>7}  verdict")
print(hdr); print("  " + "-" * (len(hdr)-2))
for name, pk, _ in gate_rows:
    fref = refined[name][0]
    lo, hi = (f_gap_lo, f_gap_hi) if name == "synodic" else (None, None)
    dF, fl, fr_ = half_power_width(pk['bin'], lo, hi)
    Q = fref / dF
    Qres = fref / (HANN_FWHM_BINS * dfbin)
    limited = dF <= 1.5 * w_window
    verdict = "resolution-limited: intrinsic Q >= Q_res" if limited \
              else "INTRINSICALLY BROADENED (disqualifier a)"
    results[name] = dict(period_refined_d=1/fref, f0_cpd=fref,
                         width_cpd=dF, width_bins=dF/dfbin, Q_line=Q,
                         Q_res=Qres, resolution_limited=bool(limited),
                         half_crossings_cpd=[fl, fr_])
    print(f"  {name:<16} {1/fref:>9.4f} {dF:>11.3e} {dF/dfbin:>9.3f} "
          f"{Q:>8.1f} {Qres:>7.1f}  {verdict}")

# precession lines: brackets only (forbidden-reading clause)
print("\n  precession lines (+/-1-bin brackets; bounds only; no tidal-Q comparison):")
for name, ibin, pub_note in [("apsidal", 11, "refined 3,234.4 d"),
                             ("nodal", 6, "feature spans bins 5-6")]:
    br = [1/freqs[ibin+1], 1/freqs[ibin-1]]
    cyc = T_rec/ (1/freqs[ibin])
    Qres = cyc / HANN_FWHM_BINS
    results[name] = dict(bin=ibin, period_bracket_d=br,
                         cycles_in_span=cyc, Q_res_windowing=Qres,
                         note="bracket only; excluded from numerical comparison")
    print(f"  {name:<16} bin {ibin}: bracket [{br[0]:,.0f}, {br[1]:,.0f}] d  "
          f"({cyc:.1f} cycles in span; window-limited Q_res {Qres:.1f}); {pub_note}")

# figure
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for ax, (name, pk, _) in zip(axes.flat, gate_rows):
    r = results[name]; ib = pk['bin']
    sl = slice(max(0, ib-9), ib+10)
    ax.semilogy(freqs[sl]*1e3, power[sl], 'o-', ms=3, lw=1, color='#1a5276')
    ax.axhline(pk['power']/2, color='#b03a2e', lw=0.9, ls='--')
    for fc in r['half_crossings_cpd']:
        ax.axvline(fc*1e3, color='#b03a2e', lw=0.8, ls=':')
    ax.set_title(f"{name}: dF={r['width_bins']:.2f} bins, "
                 f"Q_line={r['Q_line']:.0f} (Q_res {r['Q_res']:.0f})", fontsize=9)
    ax.set_xlabel('frequency (1e-3 cyc/d)', fontsize=8)
    ax.tick_params(labelsize=7)
fig.suptitle('Q Bridge step 3 -- half-power widths, four short-period lines '
             '(q_bridge_qline.py)', fontsize=10)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/fig_qbridge_linewidths.png', dpi=150)

with open('/mnt/user-data/outputs/q_bridge_qline_results.json', 'w') as f:
    json.dump({'gate': 'PASS', 'bin_width_cpd': dfbin,
               'window_fwhm_cpd': w_window, 'components': results}, f, indent=2)
print("\nArtifacts: q_bridge_qline_results.json, fig_qbridge_linewidths.png")
