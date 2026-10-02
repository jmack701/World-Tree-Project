# Metatron Chain — Expectations, stated before execution
Session-scale pre-statement (not a timestamped pre-registration; written before any run in this session).
Spiral: Fable 5, August 2026.

## Instrument 1 — Grid law on the projection-chain containers
Engine: E8_Dimensional_Scaling_Test_v2.py verbatim (c = 0.05, 2,000 steps, seed 42, 13-gain sweep).
Identity gate: the 2D phi-spiral (100), 600-cell, and E8 rows must reproduce the published values
(SE 0.4932 / 0.4989 / 0.4996 at k = 0.221; 0.4847 / 0.4940 / 0.4995 at 2/9) before any new container is read.

New containers: Metatron's Cube graph (13 nodes, complete K13 = 78 edges, the figure's own connectivity),
icosahedron (12), dodecahedron (20), rhombic triacontahedron (32, true dual construction),
120-cell (600 vertices, 4D — the untested dual of the 600-cell), each under the engine's 5-NN rule and,
where natural degree differs, under natural-edge connectivity.

E1. The invariant band (0.48–0.51) is attained inside the operating envelope on every container,
    and the collapse lands at/just past k = 2/9 on every container (per-node threshold algebra, Spine §2.6).
E2. The 120-cell matches the 600-cell at the operating point within the dimensional-control spread (~0.006):
    duality introduces no invariant-level difference. Lock depth is left open (no expectation staked —
    this is the §6.2 NOT CONFIRMED row).
E3. Small-N containers (13–32 nodes) show larger finite-size scatter but the same plateau and cliff positions.

## Instrument 2 — The polygon rung (trajectory instrument, BatF §10 cycle)
Closed form, derived before measurement: in the complex window, |λ|² = m = 3k/(9k−1),
α = −(m−1), β = −m, and the eigenvalue rotation angle satisfies cos θ = −(m−1)/(2√m).
Solving u² + 2cosθ·u − 1 = 0 for u = |λ| gives the gain for any target angle.
Derived special gains (to be verified numerically in-script before measurement):
  θ = 150° → 12-fold, k ≈ 0.11942
  θ = 144° → 5-fold (pentagram), k ≈ 0.12024
  θ = 135° → 8-fold, k ≈ 0.12201
  θ = 120° → 3-fold, k ≈ 0.12732 — |λ| = φ exactly (u² − u − 1 = 0)
  θ = 108° → 10-fold (decagon), k ≈ 0.13573
  θ = 104.06° → incommensurate control, k = kφ (|λ|² = φ)
  θ = 90°  → 4-fold, k = 1/6 — the published anchor (four-armed spiral, R4)
  θ = 72°  → 5-fold (pentagon), k ≈ 0.28685 (convergent side)
  θ = 60°  → 6-fold, k ≈ 0.87268 — |λ| = φ⁻¹ exactly
Domain note derived alongside: m ∈ (1/3, ∞) on this branch, so θ is bounded below by
arccos(1/√3) ≈ 54.74° — no gain reaches angles below the magic angle.

E4. The measured per-step rotation Δψ in the eigen-normal frame equals the derived θ(k)
    across the window (circular mean within ~1°).
E5. At commensurate gains the angular power spectrum of the normal-frame density peaks at the
    commensurability order q (4 at 1/6 — the anchor that must reproduce; 3 at 0.12732; 10 at 0.13573;
    5 at 0.12024; 8 at 0.12201; 12 at 0.11942), and shows no crisp low-order peak at kφ (control).
    The 5/10-fold gains are the H2 (pentagonal) rung — the 2D shadow of the H3 → H4 → E8 chain.
Failure at E4/E5 is reportable as-is and routes to the winding decomposition, not to a verdict.

## Instrument 3 — Node Computation 1, reframed honestly
The F_c state is (Cₙ, Cₙ₋₁): two-dimensional. A 4D initial-condition volume over (C₀,C₁,C₂,C₃)
does not exist for a second-order recurrence. Faithful 4D reads: (a) the 4D delay embedding of a
trajectory — expected intrinsic dimension ≈ 2 (participation ratio of the delay covariance);
(b) the grid law ON H4 topologies — which is Instrument 1's 600-cell and 120-cell rows.
E6. Participation ratio of the 4D delay covariance ≈ 2 at k = 1/6 and kφ. The H4-symmetry question
    is answered by the grid instrument, not the delay embedding.

Ψ
🌕🌊🌳🕸
☯️ To preserve the harmonic field.
