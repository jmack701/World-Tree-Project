"""weight_family_sweep.py — the Spine's final PROPOSED session.
FAMILY (the passage's raw form): C[n+1] = C[n] + k*(a*C[n-1] - b*C[n])
RULES FIXED BEFORE ANY RESULT (this header is the registration):
  1. Flip law (DERIVED, one line): lambda=-1 root at 2-(a+b)k=0 -> k_flip = 2/S.
     Verified numerically per pair by bisection on the undriven, unwrapped
     homogeneous family (growth vs decay over 5k steps, tol 1e-6).
  2. Environment held canonical and identical for all pairs: dt=0.01, w=2pi,
     harmonics 0.1*(sin3wt+sin6wt+sin9wt), chaos-side noise 0.2|C|*U(-1,1) at C<0,
     resonance term (C-1)*sin(wt) at C>1, Mobius wrap L=phi. Only (a,b) varies.
  3. SE estimator: Shannon entropy of the normalized rfft power of C_n
     (DC dropped), last 80k of 100k steps, per seed, mean over 20 seeds.
     Decimal always.
  4. SE_edge station: k = 0.98*(2/S) - shared across all sum-9 pairs because
     the landmark itself is sum-only. Mini-curve: 9 k-points per pair.
  5. Verdict rule: family-invariant if all four sum-9 SE_edge in [0.48,0.51];
     ratio-load-bearing if only (3,6) is in band and all others deviate >0.02;
     anything between reported as measured, typed. Either outcome instructs.
  6. Sum-variation spot checks: (2,4) flip -> 1/3; (4,8) flip -> 1/6 (digits
     echo the canonical 1/6 - two objects, typed, never identified).
  7. Validation gate: vectorized 20-seed stepper matches the scalar reference
     to <1e-12 on one (pair,k) before any result counts.
"""
import numpy as np, json, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHI = (1 + 5 ** 0.5) / 2
SEEDS = list(range(1, 21))

def step_vec(C, P, k, a, b, t, RS):
    raw = C + k * (a * P - b * C)
    for i in range(len(C)):
        if C[i] < 0:
            raw[i] += 0.2 * abs(C[i]) * RS[i].uniform(-1, 1)
    raw = np.where(C > 1, raw + (C - 1) * math.sin(2 * math.pi * t), raw)
    raw = raw + 0.1 * (math.sin(6 * math.pi * t) + math.sin(12 * math.pi * t) + math.sin(18 * math.pi * t))
    return ((raw + PHI) % (2 * PHI)) - PHI

def run_pair(a, b, k, n=100_000):
    RS = [np.random.RandomState(s) for s in SEEDS]
    C = np.full(len(SEEDS), 0.5); P = np.full(len(SEEDS), 0.1)
    keep = np.empty((n, len(SEEDS)))
    for i in range(n):
        C, P = step_vec(C, P, k, a, b, (i + 1) * 0.01, RS), C
        keep[i] = C
    x = keep[20_000:]
    ses = []
    for s in range(len(SEEDS)):
        p = np.abs(np.fft.rfft(x[:, s]))[1:] ** 2
        p = p / p.sum()
        ses.append(float(-(p * np.log(p + 1e-300)).sum() / math.log(len(p))))
    return float(np.mean(ses)), float(np.std(ses))

def scalar_ref(a, b, k, n=2_000):
    rs = np.random.RandomState(1)
    C, P = 0.5, 0.1
    out = []
    for i in range(n):
        raw = C + k * (a * P - b * C)
        if C < 0: raw += 0.2 * abs(C) * rs.uniform(-1, 1)
        if C > 1: raw += (C - 1) * math.sin(2 * math.pi * (i + 1) * 0.01)
        raw += 0.1 * (math.sin(6 * math.pi * (i + 1) * 0.01) + math.sin(12 * math.pi * (i + 1) * 0.01) + math.sin(18 * math.pi * (i + 1) * 0.01))
        C, P = ((raw + PHI) % (2 * PHI)) - PHI, C
        out.append(C)
    return np.array(out)

def vec_ref(a, b, k, n=2_000):
    RS = [np.random.RandomState(1)]
    C = np.array([0.5]); P = np.array([0.1])
    out = []
    for i in range(n):
        C, P = step_vec(C, P, k, a, b, (i + 1) * 0.01, RS), C
        out.append(C[0])
    return np.array(out)

def flip_bisect(a, b):
    def grows(k):
        C, P = 0.01, 0.0
        for _ in range(5_000):
            C, P = C + k * (a * P - b * C), C
            if abs(C) > 1e6: return True
        return abs(C) > 1e-3
    lo, hi = 0.01, 1.0
    if grows(lo) == grows(hi): return float("nan")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if grows(mid) == grows(lo): lo = mid
        else: hi = mid
    return 0.5 * (lo + hi)

# ---- gate 7: validation ----
d = np.max(np.abs(scalar_ref(3, 6, 0.20) - vec_ref(3, 6, 0.20)))
print(f"validation gate: max scalar-vs-vector diff = {d:.2e}")
assert d < 1e-12

results = {"pairs": {}, "spots": {}}
S9 = [(1, 8), (2, 7), (3, 6), (4, 5)]
kf9 = 2 / 9
edge = round(0.98 * kf9, 6)
KGRID = [round(x, 4) for x in np.linspace(0.10, 0.22, 9)]
print(f"shared flip 2/9 = {kf9:.6f}; SE_edge station k = {edge}")
for a, b in S9:
    kf = flip_bisect(a, b)
    se_e, sd_e = run_pair(a, b, edge)
    curve = [(k, *run_pair(a, b, k)) for k in KGRID]
    results["pairs"][f"{a},{b}"] = {"flip_measured": kf, "flip_predicted": kf9,
                                    "SE_edge": se_e, "SE_edge_sd": sd_e, "curve": curve}
    print(f"({a},{b}): flip {kf:.6f} (pred {kf9:.6f}, Δ {abs(kf-kf9):.2e}) | SE_edge {se_e:.4f} ± {sd_e:.4f}")
for a, b in [(2, 4), (4, 8)]:
    kf = flip_bisect(a, b)
    results["spots"][f"{a},{b}"] = {"flip_measured": kf, "flip_predicted": 2 / (a + b)}
    print(f"spot ({a},{b}): flip {kf:.6f} (pred {2/(a+b):.6f}, Δ {abs(kf-2/(a+b)):.2e})")

band = all(0.48 <= results["pairs"][f"{a},{b}"]["SE_edge"] <= 0.51 for a, b in S9)
only36 = (0.48 <= results["pairs"]["3,6"]["SE_edge"] <= 0.51) and all(
    abs(results["pairs"][f"{a},{b}"]["SE_edge"] - 0.495) > 0.02 for a, b in S9 if (a, b) != (3, 6))
verdict = "FAMILY-INVARIANT" if band else ("RATIO-LOAD-BEARING" if only36 else "INTERMEDIATE (typed)")
results["verdict"] = verdict
print("VERDICT (per pre-stated rule):", verdict)
json.dump(results, open("weight_family_results.json", "w"), indent=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
prs = [f"{a},{b}" for a, b in S9] + ["2,4", "4,8"]
meas = [results["pairs"].get(p, results["spots"].get(p))["flip_measured"] for p in prs]
pred = [results["pairs"].get(p, results["spots"].get(p))["flip_predicted"] for p in prs]
ax1.scatter(pred, meas, s=60); mx = max(pred) * 1.1
ax1.plot([0, mx], [0, mx], "k--", lw=0.8)
for p, x, y in zip(prs, pred, meas): ax1.annotate(p, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax1.set_xlabel("flip predicted 2/(a+b)"); ax1.set_ylabel("flip measured (bisection)")
ax1.set_title("Landmark law: k_flip = 2/(a+b)")
xs = np.arange(len(S9)); ses = [results["pairs"][f"{a},{b}"]["SE_edge"] for a, b in S9]
sds = [results["pairs"][f"{a},{b}"]["SE_edge_sd"] for a, b in S9]
ax2.bar(xs, ses, yerr=sds, capsize=4, color=["tab:blue" if p != (3, 6) else "tab:green" for p in S9])
ax2.axhspan(0.48, 0.51, color="gold", alpha=0.3); ax2.axhline(0.495, color="gray", ls=":", lw=0.8)
ax2.set_xticks(xs); ax2.set_xticklabels([f"{a}:{b}" for a, b in S9])
ax2.set_ylabel("SE_edge at k = 0.98·(2/9)"); ax2.set_title(f"The ratio question — verdict: {verdict}")
plt.savefig("fig_weight_family.png", dpi=150)
print("artifacts: weight_family_results.json, fig_weight_family.png")
