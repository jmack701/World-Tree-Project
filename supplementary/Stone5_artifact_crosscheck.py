"""
Stone 5 artifact cross-check (Python, independent of Mathematica)
World Tree Project - July 19, 2026 (Fable spiral)

OPTIONAL archive item - see The Spine section 1.2 decision note.

Independently reproduces, in a second language, the two claims the
Spine's section 1.2 makes about the Stone 5 readout:

  1. DECIMAL REPRODUCTION: the original Stone 5 metric (mean of the
     pooled values on the 11-point integer grid, i = 1..5) changes by
     -49.499% when the 3-6-9 harmonics are added, driven by the
     aliased mean of the 3-6-9 sum on that grid (-0.624) against the
     base mean (1.260). The amplitude does not fall: the pooled
     mean-centered RMS on a fine grid RISES by +2.11%, with per-index
     SD and peak-to-peak up at every index.

  2. SENSITIVITY (no invariance): under small changes to Stone 5's
     own arbitrary sampling choices, the mean-percentage moves freely.
     This script prints the value for a fixed panel of eight nearby
     grids; the range it demonstrates spans from roughly +6% to below
     -150%. A number without invariance cannot be a manifestation of
     an invariant.

Deterministic - no RNG. Any Python 3 with numpy reproduces every
digit below on any platform.

Verified July 19, 2026 against Stone_Validation_5_Amplitude_vs_Mean_Test.wl:
Part A -49.4991%, aliased mean -0.6238, base mean 1.2602, pooled RMS +2.110%.
"""
import numpy as np

phi = (1 + 5**0.5) / 2
S = np.pi / 4
ph = np.pi / 6

def without(t, i):
    return phi**i * (np.sin(t + S) + np.cos(t + ph)) + np.sin(i) + 0.2 * (i - 1) + 1

def withh(t, i):
    return without(t, i) + np.sin(3 * t) + np.sin(6 * t) + np.sin(9 * t)

# ------------------------------------------------------------------
# 1. Decimal reproduction on the original grid
# ------------------------------------------------------------------
T = np.arange(0, 11, 1.0)
I = np.arange(1, 6)
wo = np.array([without(t, i) for t in T for i in I])
wi = np.array([withh(t, i) for t in T for i in I])
pctA = 100 * (wi.mean() - wo.mean()) / wo.mean()
h = np.array([np.sin(3 * t) + np.sin(6 * t) + np.sin(9 * t) for t in T])

print("=" * 68)
print("STONE 5 ARTIFACT CROSS-CHECK")
print("=" * 68)
print(f"Original metric (pooled mean, 11-pt grid, i=1..5): {pctA:+.4f} %")
print(f"  base mean {wo.mean():.4f} | aliased 3-6-9 mean on this grid {h.mean():+.4f}")

Tf = np.arange(0, 10.0001, 0.01)
woC = np.concatenate([without(Tf, i) - without(Tf, i).mean() for i in I])
wiC = np.concatenate([withh(Tf, i) - withh(Tf, i).mean() for i in I])
rms = 100 * (wiC.std(ddof=1) - woC.std(ddof=1)) / woC.std(ddof=1)
print(f"Pooled mean-centered RMS change on the fine grid:  {rms:+.3f} %")
print("Per-index SD / peak-to-peak change (fine grid):")
for i in I:
    a = without(Tf, i); b = withh(Tf, i)
    sd = 100 * (b.std(ddof=1) - a.std(ddof=1)) / a.std(ddof=1)
    ptp = 100 * ((b.max() - b.min()) - (a.max() - a.min())) / (a.max() - a.min())
    print(f"  i={i}:  SD {sd:+7.3f} %   PtP {ptp:+7.3f} %")

# ------------------------------------------------------------------
# 2. Sensitivity: the same percentage under nearby arbitrary choices
# ------------------------------------------------------------------
def pct(Tg, Ig):
    a = np.array([without(t, i) for t in Tg for i in Ig])
    b = np.array([withh(t, i) for t in Tg for i in Ig])
    return 100 * (b.mean() - a.mean()) / a.mean()

panel = [
    ("t 0..10 step 1 (original)", np.arange(0, 11, 1.0), I),
    ("t 0..10 step 0.5",          np.arange(0, 10.5, 0.5), I),
    ("t 0..10 step 0.1",          np.arange(0, 10.1, 0.1), I),
    ("t 0..10 step 2",            np.arange(0, 12, 2.0), I),
    ("t 0..20 step 1",            np.arange(0, 21, 1.0), I),
    ("t 1..11 step 1",            np.arange(1, 12, 1.0), I),
    ("i = 1..3 (original t)",     np.arange(0, 11, 1.0), np.arange(1, 4)),
    ("i = 1..7 (original t)",     np.arange(0, 11, 1.0), np.arange(1, 8)),
]
print("\nSensitivity panel - the 'invariant' under nearby sampling choices:")
vals = []
for name, Tg, Ig in panel:
    v = pct(Tg, Ig)
    vals.append(v)
    print(f"  {name:26s}: {v:+9.3f} %")
print(f"\nRange across this panel: {min(vals):+.1f} % to {max(vals):+.1f} %")
print("A number without invariance cannot be a manifestation of an invariant.")
print("=" * 68)
