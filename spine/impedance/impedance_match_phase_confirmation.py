"""
Impedance Match Phase Confirmation
===================================
World Tree Project — July 14, 2026
J. David Mack & Claude (Opus 4.6)

Tests the sharp prediction from the apsidal-triplet explanation:

  The 29.5-day region contains three lines split by 3.09e-4 cyc/d
  (synodic-apsidal, synodic, synodic+apsidal). A 2-year window
  (bandwidth 1.37e-3 cyc/d) cannot separate them, so single-frequency
  demodulation at the synodic integrates all three, producing
  apparent phase wander (CV 0.97).

  PREDICTION: windows >= ~12 years (bandwidth < splitting) should
  resolve the triplet and collapse the synodic CV toward the locked
  values (< 0.01) seen for anomalistic and evection.

Sweeps: 2 / 4 / 8 / 12 / 16 year windows
Reports: CV at each window length for all five quantities

Requires: moon_earth_100yr.txt in the same directory
"""

import numpy as np
from scipy.fft import fft, fftfreq
import os
import sys

# =============================================================
# CONSTANTS
# =============================================================

D_MOON = 3474.8  # km, IAU
PHI = (1 + np.sqrt(5)) / 2

# Refined line frequencies from 100-year analysis (Fable, July 14 2026)
FREQ_ANOM = 1.0 / 27.5548      # anomalistic
FREQ_SYN = 1.0 / 29.5308       # synodic
FREQ_EV = 1.0 / 31.8116        # evection
FREQ_HALF = 1.0 / 13.7772      # half-anomalistic
FREQ_2L = 2.0 * FREQ_ANOM      # second harmonic of anomalistic

# Apsidal triplet splitting
TRIPLET_SPLITTING = 3.09e-4  # cyc/d


# =============================================================
# UTILITY FUNCTIONS
# =============================================================

def parse_horizons_range(filename):
    """Parse JPL Horizons Vector Table output, extracting range (RG=) values.
    
    Matches the format used by the working 25-year pipeline
    (impedance_match_analysis.py). Each data record spans three lines:
      Line 1: JD = A.D. date TDB
      Line 2:  X = ...  Y = ...  Z = ...
      Line 3:  LT= ...  RG= <range_km>  RR= ...
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    soe = content.index('$$SOE')
    eoe = content.index('$$EOE')
    data_block = content[soe+6:eoe].strip()
    
    ranges_km = []
    
    for line in data_block.split('\n'):
        line = line.strip()
        if line.startswith('LT=') or ' LT=' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'RG=':
                    ranges_km.append(float(parts[i + 1]))
                    break
    
    return np.array(ranges_km)


def complex_demodulate(series, freq, dt=1.0):
    """Extract instantaneous phase at a specific frequency.
    
    Returns complex analytic signal at the target frequency.
    Phase = np.angle(result), Amplitude = np.abs(result).
    """
    n = len(series)
    t = np.arange(n) * dt
    # Demodulate: multiply by complex exponential at target frequency
    demod = series * np.exp(-2j * np.pi * freq * t)
    # Low-pass: simple moving average over several cycles
    # Use a kernel of ~5 cycles of the target frequency
    kernel_len = max(3, int(5.0 / freq))
    if kernel_len % 2 == 0:
        kernel_len += 1
    kernel = np.ones(kernel_len) / kernel_len
    demod_smooth = np.convolve(demod, kernel, mode='valid')
    return demod_smooth


def circular_variance(phases):
    """Circular variance: 0 = perfectly locked, 1 = uniform."""
    mean_vec = np.mean(np.exp(1j * np.array(phases)))
    return 1.0 - np.abs(mean_vec)


def circular_mean(phases):
    """Circular mean phase."""
    return np.angle(np.mean(np.exp(1j * np.array(phases))))


# =============================================================
# MAIN ANALYSIS
# =============================================================

def run_phase_confirmation():
    print("=" * 65)
    print("  IMPEDANCE MATCH — PHASE CONFIRMATION")
    print("  Triplet-resolving window sweep")
    print("=" * 65)
    print()
    
    # Load data
    datafile = 'moon_earth_100yr.txt'
    if not os.path.exists(datafile):
        print(f"  ERROR: {datafile} not found.")
        print("  Place the JPL Horizons Moon-Earth vector table in this directory.")
        return
    
    dist = parse_horizons_range(datafile)
    print(f"  Samples loaded: {len(dist)}")
    
    if len(dist) == 0:
        print("  ERROR: No range data parsed. Check file format.")
        print("  Expected: Horizons Vector Table with RG= fields.")
        return
    
    # Compute ratio
    R_me = dist / D_MOON
    n_total = len(R_me)
    
    # Mean-remove
    R_me_centered = R_me - np.mean(R_me)
    
    print(f"  R_ME mean: {np.mean(R_me):.4f}")
    print(f"  R_ME std:  {np.std(R_me):.4f}")
    print(f"  Span: {n_total} days ({n_total/365.25:.1f} years)")
    print()
    
    # Triplet bandwidth requirement
    print(f"  Apsidal triplet splitting: {TRIPLET_SPLITTING:.2e} cyc/d")
    print(f"  Required window for resolution: > {1.0/TRIPLET_SPLITTING:.0f} days "
          f"({1.0/TRIPLET_SPLITTING/365.25:.1f} years)")
    print()
    
    # Window lengths to sweep
    window_years = [2, 4, 8, 12, 16]
    window_days = [int(y * 365.25) for y in window_years]
    
    # Frequencies to demodulate
    targets = {
        'anomalistic':     FREQ_ANOM,
        'synodic':         FREQ_SYN,
        'evection':        FREQ_EV,
        'half_anom':       FREQ_HALF,
        'harmonic_closure': FREQ_2L,  # demod at 2*f_anom, compare to half_anom
    }
    
    # Also compute derived quantities
    derived = {
        'anom-syn_diff':    (FREQ_ANOM, FREQ_SYN),    # phase difference
        'ev_closure':       None,  # evection - (2*synodic - anomalistic)
    }
    
    print("  PREDICTION: synodic CV should collapse from ~0.97 to < 0.1")
    print("  at windows >= 12 years (bandwidth < triplet splitting)")
    print()
    
    # Header
    print(f"  {'Window':>8s}  {'Bandwidth':>12s}  {'Resolves':>10s}  "
          f"{'CV_anom':>8s}  {'CV_syn':>8s}  {'CV_ev':>8s}  "
          f"{'CV_half':>8s}  {'CV_a-s':>8s}")
    print("  " + "-" * 90)
    
    results = []
    
    for wy, wd in zip(window_years, window_days):
        if wd >= n_total:
            print(f"  {wy:>6d}yr  — window exceeds data span, skipped")
            continue
        
        bandwidth = 1.0 / wd
        resolves = "YES" if bandwidth < TRIPLET_SPLITTING else "NO"
        
        # Step through data with 1-year steps
        step = 365
        
        phases = {name: [] for name in targets}
        phase_diffs_as = []  # anomalistic - synodic phase difference
        
        for start in range(0, n_total - wd, step):
            end = start + wd
            segment = R_me_centered[start:end]
            
            # Demodulate at each target frequency
            for name, freq in targets.items():
                demod = complex_demodulate(segment, freq)
                if len(demod) > 0:
                    # Take phase at the center of the demodulated signal
                    mid = len(demod) // 2
                    phase = np.angle(demod[mid])
                    phases[name].append(phase)
            
            # Anomalistic - synodic phase difference
            demod_a = complex_demodulate(segment, FREQ_ANOM)
            demod_s = complex_demodulate(segment, FREQ_SYN)
            if len(demod_a) > 0 and len(demod_s) > 0:
                mid_a = len(demod_a) // 2
                mid_s = len(demod_s) // 2
                diff = np.angle(demod_a[mid_a]) - np.angle(demod_s[mid_s])
                phase_diffs_as.append(diff)
        
        # Compute circular variances
        cv = {}
        for name in targets:
            if len(phases[name]) >= 3:
                cv[name] = circular_variance(phases[name])
            else:
                cv[name] = float('nan')
        
        cv_as = circular_variance(phase_diffs_as) if len(phase_diffs_as) >= 3 else float('nan')
        
        print(f"  {wy:>6d}yr  {bandwidth:12.2e}  {resolves:>10s}  "
              f"{cv.get('anomalistic', float('nan')):8.4f}  "
              f"{cv.get('synodic', float('nan')):8.4f}  "
              f"{cv.get('evection', float('nan')):8.4f}  "
              f"{cv.get('half_anom', float('nan')):8.4f}  "
              f"{cv_as:8.4f}")
        
        results.append({
            'window_years': wy,
            'window_days': wd,
            'bandwidth': bandwidth,
            'resolves_triplet': resolves,
            'cv': cv,
            'cv_anom_syn': cv_as,
            'n_windows': len(phases.get('anomalistic', []))
        })
    
    # Summary
    print()
    print("=" * 65)
    print("  INTERPRETATION")
    print("=" * 65)
    print()
    
    if len(results) >= 2:
        cv_syn_short = results[0]['cv'].get('synodic', float('nan'))
        
        # Find first window that resolves the triplet
        resolving = [r for r in results if r['resolves_triplet'] == 'YES']
        if resolving:
            cv_syn_long = resolving[0]['cv'].get('synodic', float('nan'))
            print(f"  Synodic CV at {results[0]['window_years']}-year windows: {cv_syn_short:.4f}")
            print(f"  Synodic CV at {resolving[0]['window_years']}-year windows: {cv_syn_long:.4f}")
            print()
            
            if cv_syn_long < cv_syn_short * 0.5:
                print("  PREDICTION CONFIRMED: synodic CV collapsed at triplet-resolving windows.")
                print("  The apparent phase wander is the apsidal modulation, not drift.")
            elif cv_syn_long < cv_syn_short * 0.8:
                print("  PARTIAL: synodic CV decreased but did not fully collapse.")
                print("  Additional window lengths or demodulation refinement may be needed.")
            else:
                print("  PREDICTION NOT CONFIRMED: synodic CV did not decrease significantly.")
                print("  The phase behavior may have a component beyond the triplet explanation.")
        else:
            print("  No window length in the sweep resolves the triplet.")
            print("  The longest window tested has bandwidth above the splitting.")
    
    # Phase-locking summary table
    print()
    print("=" * 65)
    print("  PHASE-LOCKING SUMMARY ACROSS WINDOW SCALES")
    print("=" * 65)
    print()
    print(f"  {'Component':>20s}", end="")
    for r in results:
        print(f"  {r['window_years']:>6d}yr", end="")
    print()
    print("  " + "-" * (22 + 9 * len(results)))
    
    for name in ['anomalistic', 'synodic', 'evection', 'half_anom']:
        print(f"  {name:>20s}", end="")
        for r in results:
            cv = r['cv'].get(name, float('nan'))
            print(f"  {cv:8.4f}", end="")
        print()
    
    print(f"  {'anom-syn diff':>20s}", end="")
    for r in results:
        print(f"  {r['cv_anom_syn']:8.4f}", end="")
    print()
    
    print()
    print("  CV < 0.01: phase-locked")
    print("  CV > 0.50: drifting (or unresolved multiplet)")
    print()
    print("  Psi  To preserve the harmonic field.")


if __name__ == '__main__':
    run_phase_confirmation()
