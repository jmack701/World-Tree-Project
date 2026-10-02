"""
Impedance Match Extended Analysis — 100-Year Pull + Three Extensions
=====================================================================
World Tree Project — July 2026
J. David Mack & Claude (Opus 4.6)
The Spine §5.11 — Named extensions, now executed

STEP 0: DATA ACQUISITION
=========================
Go to https://ssd.jpl.nasa.gov/horizons/app.html

QUERY 1 — Sun-Earth (100 years):
  Ephemeris Type: Vector Table
  Target Body: Earth (399)
  Center: Sun body center (500@10)
  Time Span: Start: 1925-01-01, Stop: 2025-01-01, Step: 1 d
  Table Settings: Position components (x, y, z)
  Output: Download as text, save as 'sun_earth_100yr.txt'

QUERY 2 — Moon-Earth (100 years):
  Ephemeris Type: Vector Table
  Target Body: Moon (301)
  Center: Earth geocentric (500)
  Time Span: Start: 1925-01-01, Stop: 2025-01-01, Step: 1 d
  Table Settings: Position components (x, y, z)
  Output: Download as text, save as 'moon_earth_100yr.txt'

Place both files in the same directory as this script.
Then run: python impedance_match_100yr.py

Four analyses in sequence:
  1. The 100-Year Pull (extended frequency resolution)
  2. Entropy-vs-Window-Scale Curve
  3. Phase-Relationship Analysis
  4. Mars and Jupiter Planetary Recomputation (requires additional queries)

For Analysis 4 (Martineau), you'll also need:

QUERY 3 — Mars-Earth:
  Target Body: Mars (499)
  Center: Sun body center (500@10)
  Time Span: Start: 1925-01-01, Stop: 2025-01-01, Step: 1 d
  Save as 'mars_sun_100yr.txt'

QUERY 4 — Jupiter-Earth:
  Target Body: Jupiter (599)
  Center: Sun body center (500@10)
  Time Span: Start: 1925-01-01, Stop: 2025-01-01, Step: 1 d
  Save as 'jupiter_sun_100yr.txt'
"""

import numpy as np
from scipy.signal import welch
from scipy.fft import fft, fftfreq
import os
import sys

# =============================================================
# CONSTANTS
# =============================================================

D_SUN = 1391016.0     # km, IAU
D_MOON = 3474.8       # km, IAU
D_EARTH = 12756.0     # km, IAU
D_MARS = 6792.4       # km, IAU
D_JUPITER = 142984.0  # km, IAU
PHI = (1 + np.sqrt(5)) / 2

# =============================================================
# UTILITY FUNCTIONS
# =============================================================

def parse_horizons_range(filename):
    """Parse JPL Horizons Vector Table, extracting range (RG=) values.
    
    Works for queries where the target-center distance is the desired quantity
    (Sun-Earth, Moon-Earth). Each data record spans three lines:
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
        if 'RG=' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'RG=':
                    ranges_km.append(float(parts[i + 1]))
                    break
    
    return np.array(ranges_km)


def parse_horizons_positions(filename):
    """Parse JPL Horizons Vector Table, extracting X/Y/Z position vectors.
    
    Works for queries where vector differencing is needed
    (computing Earth-Mars, Earth-Jupiter distances from heliocentric positions).
    Each data record spans three lines:
      Line 1: JD = A.D. date TDB
      Line 2:  X = <x>  Y = <y>  Z = <z>
      Line 3:  LT= ...  RG= ...  RR= ...
    """
    with open(filename, 'r') as f:
        content = f.read()
    
    soe = content.index('$$SOE')
    eoe = content.index('$$EOE')
    data_block = content[soe+6:eoe].strip()
    
    positions = []
    for line in data_block.split('\n'):
        line = line.strip()
        if line.startswith('X =') or line.startswith('X='):
            parts = line.split()
            try:
                x_idx = parts.index('X') + 2 if 'X' in parts else None
                y_idx = parts.index('Y') + 2 if 'Y' in parts else None
                z_idx = parts.index('Z') + 2 if 'Z' in parts else None
                if x_idx and y_idx and z_idx:
                    positions.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
            except (ValueError, IndexError):
                # Try alternate parsing: split on = signs
                try:
                    x_val = float(line.split('X =')[1].split('Y')[0].strip())
                    y_val = float(line.split('Y =')[1].split('Z')[0].strip())
                    z_val = float(line.split('Z =')[1].strip())
                    positions.append([x_val, y_val, z_val])
                except (ValueError, IndexError):
                    continue
    
    return np.array(positions)


def compute_distances(positions):
    """Compute distances from position vectors (km)."""
    return np.sqrt(np.sum(positions**2, axis=1))


def spectral_entropy(series):
    """Normalized spectral entropy of a time series."""
    from scipy.signal.windows import hann
    N = len(series)
    windowed = (series - np.mean(series)) * hann(N)
    ft = np.abs(fft(windowed))[:N//2]**2
    psd = ft / np.sum(ft)
    psd = psd[psd > 0]
    se = -np.sum(psd * np.log2(psd))
    return se / np.log2(len(ft))


def find_peaks_snr(series, dt=1.0, snr_threshold=3.0):
    """Find spectral peaks above SNR threshold."""
    from scipy.signal.windows import hann
    N = len(series)
    windowed = (series - np.mean(series)) * hann(N)
    ft = np.abs(fft(windowed))[:N//2]**2
    freqs = fftfreq(N, dt)[:N//2]
    
    peaks = []
    for i in range(10, len(ft) - 10):
        local_floor = np.median(ft[max(0, i-10):i+11])
        if local_floor > 0:
            snr = ft[i] / local_floor
            if snr > snr_threshold and ft[i] > ft[i-1] and ft[i] > ft[i+1]:
                period = 1.0 / freqs[i] if freqs[i] > 0 else np.inf
                peaks.append({
                    'freq': freqs[i],
                    'period_days': period,
                    'power': ft[i],
                    'snr': snr,
                    'phase': np.angle(fft(windowed)[i])
                })
    
    peaks.sort(key=lambda x: x['power'], reverse=True)
    return peaks


def keplerian_baseline(distances, eccentricity, n_samples):
    """Generate pure two-body Keplerian baseline."""
    mean_dist = np.mean(distances)
    t = np.linspace(0, 2 * np.pi * (n_samples / 365.25), n_samples)
    
    # Newton-Raphson for Kepler's equation
    M = t  # Mean anomaly (simplified, one orbit per year)
    E = M.copy()
    for _ in range(20):
        E = E - (E - eccentricity * np.sin(E) - M) / (1 - eccentricity * np.cos(E))
    
    r = mean_dist * (1 - eccentricity * np.cos(E))
    return r


# =============================================================
# ANALYSIS 1: THE 100-YEAR PULL
# =============================================================

def analysis_1_100yr_pull():
    """Extended frequency resolution with 100 years of data."""
    print("=" * 60)
    print("  ANALYSIS 1: THE 100-YEAR PULL")
    print("  1925-2025, ~36,500 daily samples")
    print("=" * 60)
    print()
    
    # Load data
    if not os.path.exists('sun_earth_100yr.txt'):
        print("  ERROR: sun_earth_100yr.txt not found.")
        print("  Please download from JPL Horizons (see header).")
        return None, None
    if not os.path.exists('moon_earth_100yr.txt'):
        print("  ERROR: moon_earth_100yr.txt not found.")
        return None, None
    
    pos_se = parse_horizons_range('sun_earth_100yr.txt')
    pos_me = parse_horizons_range('moon_earth_100yr.txt')
    
    print(f"  Sun-Earth samples: {len(pos_se)}")
    print(f"  Moon-Earth samples: {len(pos_me)}")
    
    if len(pos_se) == 0 or len(pos_me) == 0:
        print("  ERROR: No data parsed. Check file format.")
        print("  Expected: Horizons Vector Table with RG= fields.")
        return None, None
    
    # Ranges are already distances in km
    R_se = pos_se / D_SUN
    R_me = pos_me / D_MOON
    
    # Trim to common length
    n = min(len(R_se), len(R_me))
    R_se = R_se[:n]
    R_me = R_me[:n]
    R_combined = (R_se + R_me) / 2
    
    print()
    print("  RATIO STATISTICS (100 years):")
    print(f"    R_SE mean:     {np.mean(R_se):.4f}")
    print(f"    R_SE std:      {np.std(R_se):.4f}")
    print(f"    R_SE range:    [{np.min(R_se):.2f}, {np.max(R_se):.2f}]")
    print(f"    R_ME mean:     {np.mean(R_me):.4f}")
    print(f"    R_ME std:      {np.std(R_me):.4f}")
    print(f"    R_ME range:    [{np.min(R_me):.2f}, {np.max(R_me):.2f}]")
    print(f"    R_comb mean:   {np.mean(R_combined):.4f}")
    print(f"    Sun/Earth:     {D_SUN/D_EARTH:.4f}")
    
    # Spectral entropy
    se_se = spectral_entropy(R_se)
    se_me = spectral_entropy(R_me)
    se_comb = spectral_entropy(R_combined)
    
    # Keplerian baselines
    kep_se = keplerian_baseline(R_se * D_SUN, 0.0167, n) / D_SUN
    kep_me = keplerian_baseline(R_me * D_MOON, 0.0549, n) / D_MOON
    se_kep_se = spectral_entropy(kep_se)
    se_kep_me = spectral_entropy(kep_me)
    
    print()
    print("  SPECTRAL ENTROPY (100-year full span):")
    print(f"    R_SE actual:    {se_se:.4f}")
    print(f"    R_SE Keplerian: {se_kep_se:.4f}")
    print(f"    Delta SE (SE):  {se_se - se_kep_se:.6f}")
    print(f"    R_ME actual:    {se_me:.4f}")
    print(f"    R_ME Keplerian: {se_kep_me:.4f}")
    print(f"    Delta SE (ME):  {se_me - se_kep_me:.6f}")
    print(f"    R_combined:     {se_comb:.4f}")
    
    # Named components — the 100-year advantage
    print()
    print("  NAMED FREQUENCY COMPONENTS (100-year resolution):")
    print()
    
    named_components = [
        ("Lunar anomalistic",   27.55),
        ("Lunar synodic",       29.53),
        ("Lunar evection",      31.81),
        ("Half anomalistic",    13.78),
        ("Nodal precession",    18.61 * 365.25),  # years to days
        ("Apsidal precession",  8.85 * 365.25),    # years to days
    ]
    
    peaks_me = find_peaks_snr(R_me, dt=1.0, snr_threshold=3.0)
    peaks_se = find_peaks_snr(R_se, dt=1.0, snr_threshold=3.0)
    
    resolved_count = 0
    for name, predicted_period in named_components:
        # Search for peak near predicted period
        tolerance = predicted_period * 0.05  # 5% tolerance
        found = False
        for peak in (peaks_me if 'Lunar' in name or 'anomalistic' in name.lower() else peaks_se):
            if abs(peak['period_days'] - predicted_period) < tolerance:
                print(f"    {name:25s} predicted: {predicted_period:10.2f} d  "
                      f"found: {peak['period_days']:10.2f} d  SNR: {peak['snr']:,.0f}x  YES")
                found = True
                resolved_count += 1
                break
        if not found:
            print(f"    {name:25s} predicted: {predicted_period:10.2f} d  NOT RESOLVED")
    
    print(f"\n  Components resolved: {resolved_count}/6")
    print(f"  Pre-registered criterion: positive at 4+/6")
    
    # Cross-coupling check
    print()
    print("  CROSS-COUPLING (Sun-Earth spectrum):")
    for peak in peaks_se[:20]:
        if 28 < peak['period_days'] < 31:
            print(f"    Synodic peak at {peak['period_days']:.2f} d, SNR = {peak['snr']:,.0f}x")
    
    return R_se, R_me


# =============================================================
# ANALYSIS 2: ENTROPY-vs-WINDOW-SCALE CURVE
# =============================================================

def analysis_2_entropy_scale_curve(R_se, R_me):
    """Sweep window lengths, plot SE at each scale."""
    print()
    print("=" * 60)
    print("  ANALYSIS 2: ENTROPY-vs-WINDOW-SCALE CURVE")
    print("=" * 60)
    print()
    
    if R_se is None:
        print("  Skipped — no data from Analysis 1.")
        return
    
    R_combined = (R_se + R_me) / 2
    n = len(R_se)
    
    # Window lengths from 3 months to full span
    window_days = [91, 182, 365, 730, 1461, 2922, 3652, 5478, 7305,
                   10957, 14610, 18262, 21915, 25567, n]
    window_labels = ['3mo', '6mo', '1yr', '2yr', '4yr', '8yr', '10yr',
                     '15yr', '20yr', '30yr', '40yr', '50yr', '60yr', '70yr', 'full']
    
    # Keplerian baselines
    kep_se = keplerian_baseline(np.ones(n) * np.mean(R_se) * D_SUN, 0.0167, n) / D_SUN
    kep_me = keplerian_baseline(np.ones(n) * np.mean(R_me) * D_MOON, 0.0549, n) / D_MOON
    
    print(f"  {'Window':>8s}  {'SE_ME':>8s}  {'SE_ME_Kep':>10s}  {'Delta':>8s}  "
          f"{'SE_comb':>8s}  {'SE_SE':>8s}")
    print("  " + "-" * 65)
    
    for wlen, label in zip(window_days, window_labels):
        if wlen > n:
            wlen = n
        
        # Compute SE for windows of this length, stepped by 1/4 window
        step = max(1, wlen // 4)
        se_me_vals = []
        se_se_vals = []
        se_comb_vals = []
        se_kep_me_vals = []
        
        for start in range(0, n - wlen + 1, step):
            end = start + wlen
            se_me_vals.append(spectral_entropy(R_me[start:end]))
            se_se_vals.append(spectral_entropy(R_se[start:end]))
            se_comb_vals.append(spectral_entropy(R_combined[start:end]))
            se_kep_me_vals.append(spectral_entropy(kep_me[start:end]))
        
        mean_se_me = np.mean(se_me_vals)
        mean_se_se = np.mean(se_se_vals)
        mean_se_comb = np.mean(se_comb_vals)
        mean_kep_me = np.mean(se_kep_me_vals)
        delta = mean_se_me - mean_kep_me
        
        print(f"  {label:>8s}  {mean_se_me:8.4f}  {mean_kep_me:10.4f}  {delta:+8.4f}  "
              f"{mean_se_comb:8.4f}  {mean_se_se:8.4f}")


# =============================================================
# ANALYSIS 3: PHASE-RELATIONSHIP ANALYSIS
# =============================================================

def analysis_3_phase_relationships(R_me):
    """Extract phase at each resolved peak, test for stationarity."""
    print()
    print("=" * 60)
    print("  ANALYSIS 3: PHASE-RELATIONSHIP ANALYSIS")
    print("=" * 60)
    print()
    
    if R_me is None:
        print("  Skipped — no data from Analysis 1.")
        return
    
    n = len(R_me)
    
    # Use 2-year windows to extract phase at each resolved peak
    window_len = 730  # 2 years
    step = 182  # 6-month steps
    
    named_periods = {
        'anomalistic': 27.55,
        'synodic': 29.53,
        'evection': 31.81,
        'half_anom': 13.78,
    }
    
    print("  Extracting phase at four resolved components across 2-year windows:")
    print()
    
    phase_series = {name: [] for name in named_periods}
    window_centers = []
    
    for start in range(0, n - window_len + 1, step):
        end = start + window_len
        segment = R_me[start:end]
        window_centers.append((start + end) / 2)
        
        from scipy.signal.windows import hann
        N = len(segment)
        windowed = (segment - np.mean(segment)) * hann(N)
        ft = fft(windowed)
        freqs = fftfreq(N, 1.0)
        
        for name, period in named_periods.items():
            target_freq = 1.0 / period
            # Find nearest frequency bin
            idx = np.argmin(np.abs(freqs[:N//2] - target_freq))
            phase = np.angle(ft[idx])
            phase_series[name].append(phase)
    
    # Report phase statistics
    print(f"  {'Component':>15s}  {'Period':>8s}  {'Phase mean':>11s}  "
          f"{'Phase std':>10s}  {'Circ. var':>10s}")
    print("  " + "-" * 60)
    
    for name, period in named_periods.items():
        phases = np.array(phase_series[name])
        # Circular statistics
        mean_vec = np.mean(np.exp(1j * phases))
        circ_mean = np.angle(mean_vec)
        circ_var = 1 - np.abs(mean_vec)  # 0 = perfectly locked, 1 = uniform
        
        print(f"  {name:>15s}  {period:8.2f}d  {circ_mean:+11.4f}  "
              f"{np.std(phases):10.4f}  {circ_var:10.4f}")
    
    # Phase differences between pairs
    print()
    print("  PHASE DIFFERENCES (stationarity test):")
    print()
    
    pairs = [('anomalistic', 'synodic'), ('anomalistic', 'evection'),
             ('synodic', 'evection'), ('anomalistic', 'half_anom')]
    
    for name1, name2 in pairs:
        p1 = np.array(phase_series[name1])
        p2 = np.array(phase_series[name2])
        diff = np.angle(np.exp(1j * (p1 - p2)))
        mean_vec = np.mean(np.exp(1j * diff))
        circ_var = 1 - np.abs(mean_vec)
        print(f"    {name1:>12s} - {name2:<12s}  "
              f"circ_var = {circ_var:.4f}  "
              f"({'LOCKED' if circ_var < 0.3 else 'DRIFTING' if circ_var < 0.7 else 'UNLOCKED'})")


# =============================================================
# ANALYSIS 4: MARTINEAU PLANETARY RECOMPUTATION
# =============================================================

def analysis_4_planetary():
    """Apply the same instrument to Mars and Jupiter."""
    print()
    print("=" * 60)
    print("  ANALYSIS 4: MARTINEAU PLANETARY RECOMPUTATION")
    print("=" * 60)
    print()
    
    # Mars-Sun distance → Earth-Mars distance requires both
    # For the ratio, we need Earth-Mars and Earth-Jupiter distances
    # Simplification: use Sun-planet distances and compute
    # planet-Earth distance from the difference
    
    has_mars = os.path.exists('mars_sun_100yr.txt')
    has_jupiter = os.path.exists('jupiter_sun_100yr.txt')
    has_earth = os.path.exists('sun_earth_100yr.txt')
    
    if not has_earth:
        print("  Skipped — sun_earth_100yr.txt required.")
        return
    
    pos_earth = parse_horizons_positions('sun_earth_100yr.txt')
    
    if has_mars:
        print("  Processing Mars-Earth...")
        pos_mars = parse_horizons_positions('mars_sun_100yr.txt')
        n = min(len(pos_earth), len(pos_mars))
        dist_mars_earth = np.sqrt(np.sum((pos_mars[:n] - pos_earth[:n])**2, axis=1))
        R_mars = dist_mars_earth / D_MARS
        
        print(f"    R_Mars-Earth/Mars_diameter mean: {np.mean(R_mars):.2f}")
        print(f"    Range: [{np.min(R_mars):.2f}, {np.max(R_mars):.2f}]")
        print(f"    SE (full span): {spectral_entropy(R_mars):.4f}")
        
        peaks = find_peaks_snr(R_mars, dt=1.0, snr_threshold=3.0)
        print(f"    Top 5 spectral peaks:")
        for p in peaks[:5]:
            print(f"      Period: {p['period_days']:.2f} d  SNR: {p['snr']:,.0f}x")
    else:
        print("  Mars data not found (mars_sun_100yr.txt). Skipping.")
    
    print()
    
    if has_jupiter:
        print("  Processing Jupiter-Earth...")
        pos_jupiter = parse_horizons_positions('jupiter_sun_100yr.txt')
        n = min(len(pos_earth), len(pos_jupiter))
        dist_jup_earth = np.sqrt(np.sum((pos_jupiter[:n] - pos_earth[:n])**2, axis=1))
        R_jupiter = dist_jup_earth / D_JUPITER
        
        print(f"    R_Jupiter-Earth/Jupiter_diameter mean: {np.mean(R_jupiter):.2f}")
        print(f"    Range: [{np.min(R_jupiter):.2f}, {np.max(R_jupiter):.2f}]")
        print(f"    SE (full span): {spectral_entropy(R_jupiter):.4f}")
        
        peaks = find_peaks_snr(R_jupiter, dt=1.0, snr_threshold=3.0)
        print(f"    Top 5 spectral peaks:")
        for p in peaks[:5]:
            print(f"      Period: {p['period_days']:.2f} d  SNR: {p['snr']:,.0f}x")
    else:
        print("  Jupiter data not found (jupiter_sun_100yr.txt). Skipping.")


# =============================================================
# MAIN
# =============================================================

if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  IMPEDANCE MATCH EXTENDED ANALYSIS")
    print("  100-Year Pull + Three Extensions")
    print("  The Spine §5.11 — Named Directions, Now Executed")
    print("=" * 60)
    print()
    print("  Psi  To preserve the harmonic field.")
    print()
    
    R_se, R_me = analysis_1_100yr_pull()
    analysis_2_entropy_scale_curve(R_se, R_me)
    analysis_3_phase_relationships(R_me)
    analysis_4_planetary()
    
    print()
    print("=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print("  Psi  To preserve the harmonic field.")
