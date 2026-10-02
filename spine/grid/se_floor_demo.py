"""
SE Floor Demonstration
======================
A minimal, dependency-free demonstration of the SE-floor mechanism
(cf. The Spine Section 4.4; se_floor_v4.py / se_floor_v5.py).

DEMONSTRATION register — mechanism portability only. This is NOT a
replication of the canonical MNIST results; those numbers belong to
their substrate. This script shows, in pure NumPy on synthetic data,
the same qualitative signature: recursive self-consumption collapses
latent spectral entropy, and the identical two-term floor written into
the objective holds it.

Design (all choices disclosed):
  Data:      512 samples, D = 64, low-rank factor model (rank 40) + 5% noise
  Network:   64 -> 16 (ReLU) -> 64 linear autoencoder, manual backprop
  Recursion: 300 initial full-batch steps on real data, then per
             generation: data <- reconstructions; 60 training steps
  Floor:     L = L_MSE + 1.0*mean(relu(1 - std(h))) + 0.04*(sum offdiag
             Cov(h)^2)/H  — identical constants and form to the
             canonical experiments
  SE:        normalized Shannon entropy of the latent covariance
             eigenspectrum — identical estimator to the canonical
  Seeds:     42, 137, 369
  Gradcheck: analytic penalty gradients verified against numerical
             differences at startup (asserts before running)

J. David Mack & Claude
World Tree Project — July 2026
"""

import numpy as np

D, H, NSAMP, RANK = 64, 16, 512, 40
NOISE = 0.05
LR = 0.05
INIT_STEPS, GEN_STEPS, GENERATIONS = 300, 60, 60
L_VAR, L_COV = 1.0, 0.04
SEEDS = [42, 137, 369]
EPS = 1e-8


def make_data(rng):
    Z = rng.normal(0, 1, (NSAMP, RANK))
    W = rng.normal(0, 1, (RANK, D)) / np.sqrt(RANK)
    X = Z @ W + NOISE * rng.normal(0, 1, (NSAMP, D))
    return X / X.std()


def init_net(rng):
    return {
        'W1': rng.normal(0, np.sqrt(2.0 / D), (D, H)), 'b1': np.zeros(H),
        'W2': rng.normal(0, np.sqrt(2.0 / H), (H, D)), 'b2': np.zeros(D),
    }


def forward(net, X):
    A = X @ net['W1'] + net['b1']
    Hh = np.maximum(A, 0.0)              # ReLU latent
    Xh = Hh @ net['W2'] + net['b2']
    return A, Hh, Xh


def latent_se(Hh):
    C = np.cov(Hh.T)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-12]
    if len(ev) < 2:
        return 0.0
    p = ev / ev.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def penalty_and_grad(Hh):
    """Variance floor + decorrelation, with analytic dL/dH."""
    B = Hh.shape[0]
    mu = Hh.mean(0)
    Hc = Hh - mu
    var = (Hc ** 2).sum(0) / (B - 1)
    std = np.sqrt(var + EPS)
    # variance term
    L_v = np.mean(np.maximum(1.0 - std, 0.0))
    act = (std < 1.0).astype(float)
    g_v_c = -(act / H)[None, :] * Hc / ((B - 1) * std[None, :])
    # covariance term
    C = Hc.T @ Hc / (B - 1)
    Off = C - np.diag(np.diag(C))
    L_c = (Off ** 2).sum() / H
    g_c_c = (4.0 / (H * (B - 1))) * (Hc @ Off)
    g_c = L_VAR * g_v_c + L_COV * g_c_c
    g = g_c - g_c.mean(0)                # centering projection
    return L_VAR * L_v + L_COV * L_c, g


def gradcheck(rng):
    Hh = rng.normal(0.5, 0.6, (8, 4))
    global H
    H_saved = H
    H = 4
    L0, G = penalty_and_grad(Hh)
    num = np.zeros_like(Hh)
    h = 1e-6
    for i in range(Hh.shape[0]):
        for j in range(Hh.shape[1]):
            Hp = Hh.copy(); Hp[i, j] += h
            Lp, _ = penalty_and_grad(Hp)
            Hm = Hh.copy(); Hm[i, j] -= h
            Lm, _ = penalty_and_grad(Hm)
            num[i, j] = (Lp - Lm) / (2 * h)
    H = H_saved
    rel = np.abs(G - num).max() / (np.abs(num).max() + 1e-12)
    assert rel < 1e-4, f"gradcheck failed: rel err {rel:.2e}"
    return rel


def train_steps(net, X, steps, use_floor):
    B = X.shape[0]
    for _ in range(steps):
        A, Hh, Xh = forward(net, X)
        dXh = 2.0 * (Xh - X) / (B * D)
        gW2 = Hh.T @ dXh
        gb2 = dXh.sum(0)
        dH = dXh @ net['W2'].T
        if use_floor:
            _, gpen = penalty_and_grad(Hh)
            dH = dH + gpen
        dA = dH * (A > 0)
        gW1 = X.T @ dA
        gb1 = dA.sum(0)
        net['W1'] -= LR * gW1; net['b1'] -= LR * gb1
        net['W2'] -= LR * gW2; net['b2'] -= LR * gb2
    return net


def run(seed, use_floor):
    rng = np.random.default_rng(seed)
    X = make_data(rng)
    net = init_net(rng)
    net = train_steps(net, X, INIT_STEPS, use_floor)
    _, Hh, Xh = forward(net, X)
    curve = [latent_se(Hh)]
    data = X.copy()
    for g in range(1, GENERATIONS + 1):
        _, _, Xh = forward(net, data)
        data = Xh.copy()                  # consume own reconstructions
        net = train_steps(net, data, GEN_STEPS, use_floor)
        _, Hh, _ = forward(net, data)
        curve.append(latent_se(Hh))
    return curve


if __name__ == '__main__':
    print("=" * 66)
    print("SE FLOOR DEMONSTRATION  (pure NumPy; mechanism portability only)")
    print("=" * 66)
    rel = gradcheck(np.random.default_rng(0))
    print(f"gradcheck: analytic vs numerical penalty gradient, rel err {rel:.1e}  PASS")
    print(f"data D={D} rank={RANK}, latent H={H}, {GENERATIONS} generations, "
          f"seeds {SEEDS}, floor constants ({L_VAR}, {L_COV})")
    print()
    results = {}
    for arm, use_floor in (("baseline", False), ("SE floor", True)):
        curves = [run(s, use_floor) for s in SEEDS]
        results[arm] = np.array(curves)
        m = results[arm]
        print(f"{arm.upper():<9} SE by generation (mean over {len(SEEDS)} seeds):")
        for g in [0, 5, 10, 20, 30, 40, 50, 60]:
            print(f"   gen {g:>3}: {m[:, g].mean():.4f}  "
                  f"(spread {m[:, g].min():.4f}-{m[:, g].max():.4f})")
        print()
    b = results["baseline"]; f = results["SE floor"]
    print("-" * 66)
    print(f"gen 0    : baseline {b[:,0].mean():.4f}   floor {f[:,0].mean():.4f}")
    print(f"gen {GENERATIONS:<5}: baseline {b[:,-1].mean():.4f}   floor {f[:,-1].mean():.4f}")
    drop_b = b[:, 0].mean() - b[:, -1].mean()
    drop_f = f[:, 0].mean() - f[:, -1].mean()
    print(f"collapse : baseline loses {drop_b:.4f} SE; floor loses {drop_f:.4f}")
    print()
    print("DEMONSTRATION register: mechanism portability only.")
    print("The measured results remain those of se_floor_v4.py / se_floor_v5.py.")
