"""polygon_return_order.py — the wrap-robust commensurability instrument.

R2 found the stationary angular comb only at k = 1/6: wrap events rephase the
global comb (the winding decomposition's edge-band geometry owns the density).
The local statistic survives wrapping: over any run of j+1 consecutive Wind-0
steps, the direction advances by j*theta. For a commensurate angle of order q,
j = q is the first lag at which the advance returns to 0 (mod 360). For the
incommensurate control (kphi), no low lag returns.

Readout per gain: for j = 1..14, circular mean and concentration R of
Delta_j psi over all runs of j consecutive Wind-0 increments; predicted value
j*theta mod 360; the measured return order q_hat = first j with |mean| < 6 deg
and R > 0.8 (minimum 200 samples). Prediction: q_hat = q on the divergent
commensurate gains; no return at kphi.

Same §10 cycle, seed 42, 400,000 steps per gain (longer runs to populate
deep-j segments at low Wind-0 fractions).
"""
import numpy as np
import math, json

PHI = (1 + 5**0.5) / 2
PHI_INV = 1 / PHI
SEED = 42
C0, CM1 = 0.3, 0.5
AH, NOISE_SCALE, RES_REF = 0.1, 0.2, 1.0
DT, OMEGA = 0.01, 2 * math.pi
L = PHI

def alpha(k): return (1 - 6 * k) / (1 - 9 * k)
def beta(k):  return 3 * k / (1 - 9 * k)

def normal_frame(k):
    a, b = alpha(k), beta(k)
    disc = a * a + 4 * b
    s = math.sqrt(-disc)
    T = np.array([[a / 2, s / 2], [1.0, 0.0]])
    return np.linalg.inv(T)

def run_gain(k, n_steps):
    rng = np.random.RandomState(SEED)
    a, b = alpha(k), beta(k)
    Tinv = normal_frame(k)
    c_prev, c_curr = CM1, C0
    wind0 = np.zeros(n_steps, bool)
    xs = np.zeros(n_steps); ys = np.zeros(n_steps)
    for n in range(1, n_steps + 1):
        t = n * DT
        c_raw = a * c_curr + b * c_prev
        if c_curr < 0:
            c_raw += NOISE_SCALE * abs(c_curr) * (rng.random() * 2 - 1)
        elif c_curr > RES_REF:
            c_raw += (c_curr - RES_REF) * math.sin(OMEGA * t)
        c_raw += AH * (math.sin(3*OMEGA*t) + math.sin(6*OMEGA*t) + math.sin(9*OMEGA*t))
        w0 = abs(c_raw) <= L
        c_next = ((c_raw + L) % (2 * L)) - L
        i = n - 1
        wind0[i] = w0; xs[i] = c_next; ys[i] = c_curr
        c_prev, c_curr = c_curr, c_next
    Z = Tinv @ np.vstack([xs, ys])
    psi = np.arctan2(Z[1], Z[0])
    return wind0, psi

GAINS = [
    ("5-fold pentagram", 0.1202399, 144.0, 5),
    ("8-fold octagon",   0.1220081, 135.0, 8),
    ("3-fold triangle",  0.1273220, 120.0, 3),
    ("10-fold decagon",  0.1357284, 108.0, 10),
    ("kphi control",     PHI/(9*PHI-3), 104.0598, None),
    ("4-fold square",    1/6,       90.0,  4),
]
# recompute exact k for the ladder members (avoid rounding)
def k_for_theta(th):
    c = math.cos(math.radians(th))
    u = -c + math.sqrt(c * c + 1)
    m = u * u
    return m / (9 * m - 3)
GAINS = [(n, (k_for_theta(t) if q is not None else k), t, q) for (n, k, t, q) in GAINS]

N_STEPS = 400_000
MIN_SAMP = 200
print("=" * 96)
print("RETURN-ORDER TEST — Delta_j psi over runs of consecutive Wind-0 increments (400k steps, seed 42)")
print("=" * 96)
all_out = {}
for name, k, th, q in GAINS:
    w0, psi = run_gain(k, N_STEPS)
    # consecutive-Wind-0 increment mask: increment n->n+1 is 'clean' if step n+1 is Wind 0
    clean = w0[1:]
    table = []
    qhat = None
    for j in range(1, 15):
        # positions where j consecutive clean increments start
        ok = np.ones(len(psi) - j, bool)
        for jj in range(j):
            ok &= clean[jj:len(clean) - (j - 1 - jj)]
        idx = np.where(ok)[0]
        if len(idx) < MIN_SAMP:
            table.append((j, None, None, len(idx))); continue
        d = np.angle(np.exp(1j * (psi[idx + j] - psi[idx])))
        z = np.exp(1j * d)
        mean = math.degrees(math.atan2(np.mean(np.imag(z)), np.mean(np.real(z))))
        R = float(abs(np.mean(z)))
        pred = ((j * th + 180) % 360) - 180
        table.append((j, mean, R, len(idx), pred))
        if qhat is None and abs(mean) < 6 and R > 0.8 and j > 1:
            qhat = j
    all_out[name] = dict(k=k, theta=th, q=q, qhat=qhat, table=table)
    print(f"\n--- {name}  (k = {k:.6f}, theta = {th:.4f}, predicted order q = {q}) ---")
    print(f"{'j':>3s} {'pred j*th':>10s} {'measured':>10s} {'circR':>7s} {'N':>8s}")
    for row in table:
        if row[1] is None:
            print(f"{row[0]:>3d} {'—':>10s} {'(n<200)':>10s} {'—':>7s} {row[3]:>8d}")
        else:
            j, mean, R, N, pred = row
            mark = "  <== RETURN" if (abs(mean) < 6 and R > 0.8 and j > 1) else ""
            print(f"{j:>3d} {pred:>10.2f} {mean:>10.2f} {R:>7.3f} {N:>8d}{mark}")
    print(f"  measured return order q_hat = {qhat}   (predicted {q})")

with open("/home/claude/return_order_results.json", "w") as f:
    json.dump({n: {kk: vv for kk, vv in v.items() if kk != 'table'} for n, v in all_out.items()}, f)
    
print()
print("Psi - To preserve the harmonic field.")
