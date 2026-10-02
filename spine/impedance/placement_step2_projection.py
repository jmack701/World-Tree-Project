#!/usr/bin/env python3
"""
Placement Phase -- Steps 2 & 3: SE-axis projection and concordance.
World Tree Project, September 12, 2026. Frame: claude_Placement_Model_and_Frame.md.

REGIME I (reproduction): phi_grid_k_sweep_refined.py executed VERBATIM (exec of
the archived bytes); gate = published anchors before any inversion.
REGIME II (extension): the fenced band inversion SE in 0.2548 +/- 0.0122, then
Step-3 concordance against the pre-named three outcomes.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 78)
print("PLACEMENT -- STEPS 2 & 3  (frame: claude_Placement_Model_and_Frame.md)")
print("=" * 78)
print("\nREGIME I -- executing phi_grid_k_sweep_refined.py VERBATIM ...\n")

import io, contextlib, os
os.chdir('/home/claude')
code = open('/mnt/project/phi_grid_k_sweep_refined.py').read()
buf = io.StringIO()
ns = {}
with contextlib.redirect_stdout(buf):
    exec(compile(code, 'phi_grid_k_sweep_refined.py', 'exec'), ns)
out = buf.getvalue()
open('/mnt/user-data/outputs/placement_backbone_stdout.txt', 'w').write(out)
for line in out.splitlines():
    if line.strip().startswith('k=') or 'Baseline' in line:
        print(" ", line.strip())

k_vals = np.array(ns['sweep']['k'])
se     = np.array(ns['sweep']['se'])
stab   = np.array(ns['sweep']['stab'])
dk     = k_vals[1] - k_vals[0]

print("\nREGIME I GATE (published anchors)")
print("-" * 78)
checks = []
checks.append(("sampled SE range [0.1723, 0.8666]",
               round(float(se.min()),4) == 0.1723 and round(float(se.max()),4) == 0.8666,
               f"got [{se.min():.4f}, {se.max():.4f}]"))
i219 = int(np.argmin(np.abs(k_vals - 0.21925)))
checks.append(("operating point SE ~ 0.498 at k = 0.219",
               round(float(se[i219]),3) == 0.498, f"got {se[i219]:.4f} at k={k_vals[i219]:.5f}"))
i223 = int(np.argmin(np.abs(k_vals - 0.22267)))
checks.append(("lock dip at k ~ 0.223, stability holding",
               (se[i223] < se[i219]) and (stab[i223] > 0.9),
               f"SE {se[i223]:.4f}, stab {stab[i223]:.4f}"))
i226 = int(np.argmin(np.abs(k_vals - 0.22608)))
checks.append(("discontinuous collapse at the next grid point",
               (stab[i226] < 0.5) and (stab[i223] > 0.5),
               f"stab {stab[i226]:.4f} at k={k_vals[i226]:.5f}"))
gate_pass = True
for name, ok, note in checks:
    gate_pass &= ok
    print(f"  {name:<44} {'PASS' if ok else 'DEVIATION'}   ({note})")
print(f"\n  GATE VERDICT: {'PASS -- backbone reproduces the record; '
      'REGIME II opens' if gate_pass else 'FAIL -- STOP; the deviation is the finding'}")
if not gate_pass:
    raise SystemExit(1)

print("\nREGIME II -- THE PROJECTION (S7 of the frame; fences verbatim)")
print("-" * 78)
LO, HI = 0.2548 - 0.0122, 0.2548 + 0.0122
print(f"  band: SE in [{LO:.4f}, {HI:.4f}]   (0.2548 +/- 0.0122)   "
      f"k-side floor Dk = {dk:.4f}")

def band_intervals(k, y, lo, hi):
    inside = (y >= lo) & (y <= hi)
    ivs = []
    i = 0
    while i < len(k):
        if inside[i]:
            a = k[i]
            if i > 0 and not inside[i-1]:
                for level in (lo, hi):
                    if min(y[i-1], y[i]) <= level <= max(y[i-1], y[i]):
                        a = k[i-1] + (level - y[i-1])*(k[i]-k[i-1])/(y[i]-y[i-1]); break
            j = i
            while j+1 < len(k) and inside[j+1]: j += 1
            b = k[j]
            if j+1 < len(k):
                for level in (lo, hi):
                    if min(y[j], y[j+1]) <= level <= max(y[j], y[j+1]):
                        b = k[j] + (level - y[j])*(k[j+1]-k[j])/(y[j+1]-y[j]); break
            ivs.append((a, b)); i = j+1
        else:
            i += 1
    return ivs

proj = band_intervals(k_vals, se, LO, HI)
for a, b in proj:
    print(f"  projected k-region: [{a:.4f}, {b:.4f}]  (+/- {dk:.4f} floor)")
print('  fences: "ratio-series SE and grid-deviation SE are cousins, not twins; '
      'the projection carries stated width, never a point." '
      '"This is a projection -- coordinates on the model\'s dial -- never an '
      'identification"; typed-not-numerical stands permanently.')

print("\nSTEP 3 -- CONCORDANCE (S8; three pre-named outcomes)")
print("-" * 78)
step1 = json.load(open('/mnt/user-data/outputs/placement_step1_results.json'))
cross = step1['in_domain_crossing_k']
print(f"  Step-1 in-domain crossing region (sustained, M >= 10^2): "
      f"{cross if cross else 'EMPTY'}")
HOLD = (0.160, 0.164); CLOCK = (0.2227, 0.2261)
def overlaps(iv, band): return not (iv[1] < band[0] or iv[0] > band[1])
if not cross:
    outcome = ("NO OVERLAP -- the Step-1 crossing region is empty; the sky is "
               "not on this dial, and the correspondence stays typed at its "
               "current level.")
else:
    hits_cl = any(overlaps(p, CLOCK) for p in proj)
    hits_h  = any(overlaps(p, HOLD) for p in proj)
    outcome = ("overlap in the coupled-lock band" if hits_cl else
               "overlap in the Hold band" if hits_h else "no overlap")
in_hold = [p for p in proj if overlaps(p, HOLD)]
in_cl   = [p for p in proj if overlaps(p, CLOCK)]
print(f"  projection vs the named bands (context): Hold {'yes' if in_hold else 'no'}; "
      f"coupled-lock {'yes' if in_cl else 'no'}")
print(f"  OUTCOME: {outcome}")

json.dump(dict(gate='PASS', band=[LO, HI], dk_floor=float(dk),
               projection_intervals=[[float(a), float(b)] for a, b in proj],
               step1_crossing=cross, outcome=outcome,
               backbone=dict(k=[float(x) for x in k_vals],
                             se=[float(x) for x in se],
                             stab=[float(x) for x in stab])),
          open('/mnt/user-data/outputs/placement_step2_results.json', 'w'), indent=2)

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.plot(k_vals, se, 'o-', ms=4, lw=1, color='#1a5276', label='backbone SE(k), verbatim regeneration')
ax.axhspan(LO, HI, color='#b03a2e', alpha=0.12, label='sky band 0.2548 +/- 0.0122')
for a, b in proj:
    ax.axvspan(a, b, color='#b03a2e', alpha=0.25)
ax.axvspan(*HOLD, color='#1e8449', alpha=0.15, label='the Hold (scalar; below backbone floor)')
ax.axvspan(*CLOCK, color='#6c3483', alpha=0.18, label='coupled-lock band')
ax.axhline(0.2548, color='#b03a2e', lw=0.8, ls='--')
ax.set_xlabel('k'); ax.set_ylabel('SE (grid-deviation estimator)')
ax.set_title('Placement step 2 -- fenced projection on the grid backbone '
             '(placement_step2_projection.py)', fontsize=10)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig('/mnt/user-data/outputs/fig_placement_projection.png', dpi=150)
print("\nArtifacts: placement_step2_results.json, fig_placement_projection.png, "
      "placement_backbone_stdout.txt")
