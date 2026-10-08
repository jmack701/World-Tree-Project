# World Tree Sims — Simulation Archive of the Lumin Framework

Complete simulation code and source data for the Lumin Framework papers, published October 8, 2026.
Every script named in the papers appears here under its exact printed filename; each figure caption in
the papers credits its generating script. Companion to the live instruments — Boχ at https://box.eternityprocess.com and the Polygon Ladder at
https://box.eternityprocess.com/ladder — and to the papers below.

**Papers**
- *The Blade and the Field* — https://doi.org/10.5281/zenodo.23064205
- *The Spine* — https://doi.org/10.5281/zenodo.23064215
- *Form of the Good* — https://doi.org/10.5281/zenodo.23064223

**Software records** · Boχ https://doi.org/10.5281/zenodo.23064277 · Polygonal Ladder https://doi.org/10.5281/zenodo.23064291
**Pre-registration** · *The Impedance Match at Two Scales* (July 30, 2026) — https://doi.org/10.5281/zenodo.23064337

## Layout

    batf/                 Blade and the Field verification set — HE_Simulation Mathematica suite + core Python
    spine/grid/           Phi-spiral energy grid, IEEE bus validation, neural substrate
    spine/scaling/        Dimensional scaling (600-cell, E8), weight-family sweeps, L-boundary tests
    spine/impedance/      Impedance match at two scales — 100-yr JPL analysis, Q bridge, placement, chorus, closure
    spine/galactic/       Galactic disk Fc program (phases 1b/1c, M sweeps, spectral analysis)
    common/               Scripts cited by both papers (canonical k-sweep reference, grid showcase)
    data/ephemeris/       JPL Horizons DE441 series, 1925–2025 (Sun–Earth, Moon–Earth, Mars, Jupiter) — frozen input data
    figures/              Reference renders of the published figures (all regenerate from the scripts)
    supplementary/        Research archive beyond the verification set, including the Metatron chain and
                          density-map investigations with their pre-stated expectation files

## Verification set — every artifact named in the papers

Run environments: Python 3.10+ (`pip install -r requirements.txt`) for `.py`; Wolfram Mathematica 13+ for `.wl`.
The canonical reproduction protocol is *The Blade and the Field* §12; all k-sweep table values derive from
`common/fc_k_sweep_reference.py`.

The archive preserves each script exactly as its documented run executed it, including the input and
output paths of the original development environment (`/mnt/...` locations). Before running a script,
point its input paths at this repository’s copies — the ephemeris inputs live in `data/ephemeris/` — and
its output paths wherever the artifacts should land. The computation is unchanged by the path edit;
`SHA256SUMS` certifies the archived bytes, and the papers’ printed values are the reproduction targets.

| Artifact | Cited in | Location |
|---|---|---|
| `E8_Dimensional_Scaling_Test_v2.py` | Spine §3.5, §3.7, §4.3, §4.8 | `spine/scaling/` |
| `HE_Simulation_A_Extended_Lphi.wl` | BatF §1, §5, §9, §12 | `batf/` |
| `HE_Simulation_A_Forward_Bias_Lphi.wl` | BatF §1, §5, §12 | `batf/` |
| `HE_Simulation_D_kphi_Void_Lphi.wl` | BatF §1, §3, §12 | `batf/` |
| `HE_Simulation_G_MMphi_Power_Levels.wl` | BatF §10, §12 | `batf/` |
| `HE_Simulation_H_Winding_Lphi.wl` | BatF §6, §12 | `batf/` |
| `HE_Simulation_I_Driven_Resonance_Lphi.wl` | BatF §1, §7, §12 | `batf/` |
| `HE_Simulation_J_3D_Winding_Lphi.wl` | BatF §1, §7, §12 | `batf/` |
| `HE_Simulation_K_Dynamic_k_Lphi.wl` | BatF §1, §7, §12 | `batf/` |
| `HE_Simulation_L_Zone_III_Lphi.wl` | BatF §1, §3, §12 | `batf/` |
| `HE_Simulation_M_Luminbrot_3D.wl` | BatF §1, §3, §12 | `batf/` |
| `HE_Simulation_M_Phi_Boundary_Sweep_k16.wl` | BatF §1, §12 | `batf/` |
| `HE_Simulation_M_Phi_Boundary_Sweep_v2.wl` | BatF §1, §12 | `batf/` |
| `HE_Simulation_M_Return_Map.wl` | BatF §3, §12 | `batf/` |
| `IEEE_Fc_Validation.py` | Spine §2.6, §3.5, §6.1 | `spine/grid/` |
| `L_comparison_test.py` | Spine §5.9 | `spine/scaling/` |
| `L_ensemble_covariance_test.py` | Spine §5.9 | `spine/scaling/` |
| `SE_reconciliation_test.py` | Spine §3.2, §3.7 | `spine/scaling/` |
| `chorus_M_run.py` | Spine §3, §3.8, §6.2 | `spine/impedance/` |
| `closure_runs.py` | Spine §3, §3.8, §6.2 | `spine/impedance/` |
| `fc_k_sweep_reference.py` | BatF §11; Spine §2.2, §4.7 | `common/` |
| `fotg_ledger_identity.py` | BatF §4, §12 | `batf/` |
| `fig_gap_supremum.py` | BatF §7, §12 (Figure 19 renderer) | `batf/` |
| `fotg_still_point.py` | BatF §4, §12 | `batf/` |
| `galactic_disk_fc_phase1b.py` | Spine §7.4, §7.7 | `spine/galactic/` |
| `galactic_disk_fc_phase1c.py` | Spine §7.5, §7.7 | `spine/galactic/` |
| `galactic_fine_m_sweep.py` | Spine §7.4, §7.7 | `spine/galactic/` |
| `galactic_m_sweep_spectral.py` | Spine §7.4 | `spine/galactic/` |
| `galactic_spectral_analysis.py` | Spine §7.1, §7.4, §7.7 | `spine/galactic/` |
| `geocentric_completion.py` | Spine §5 | `spine/impedance/` |
| `geocentric_phase_confirm.py` | Spine §5 | `spine/impedance/` |
| `impedance_match_100yr_analysis.py` | Spine §5.11, §5.12, §5.13, §5.14, §5.17 | `spine/impedance/` |
| `impedance_match_phase_confirmation.py` | Spine §5.11, §5.13, §5.17 | `spine/impedance/` |
| `make_figure_5A_v2.py` | Spine §5 | `spine/impedance/` |
| `make_figures_100yr.py` | Spine §5, §5.11, §5.12, §5.13, §5.14 | `spine/impedance/` |
| `phi_grid_k_sweep_refined.py` | Spine §2.7, §3.2, §3.7, §5 | `spine/grid/` |
| `phi_grid_scaling.py` | Spine §3.2, §3.7 | `spine/grid/` |
| `phi_grid_showcase.py` | BatF §1, §9, §12; Spine §2.7, §3.2, §3.7, §4.8 | `common/` |
| `phi_inv_density_test.py` | BatF §4, §12 | `batf/` |
| `phi_topology_neural.py` | Spine §4.2, §4.7, §4.8 | `spine/grid/` |
| `placement_step1_msweep.py` | Spine §3, §3.8 | `spine/impedance/` |
| `placement_step2_projection.py` | Spine §3 | `spine/impedance/` |
| `q_bridge_qline.py` | Spine §5.16, §5.17 | `spine/impedance/` |
| `q_bridge_qphase.py` | Spine §5.16, §5.17 | `spine/impedance/` |
| `se_floor_demo.py` | Spine §4.6 | `spine/grid/` |
| `se_floor_v4.py` | Spine §4.4, §4.6, §4.7, §4.8 | `spine/grid/` |
| `se_floor_v5.py` | Spine §4.4, §4.6, §4.7, §4.8 | `spine/grid/` |
| `still_point_confirm.py` | BatF §4, §12 | `batf/` |
| `weight_family_ext_results.json` | Spine §2.6 | `spine/scaling/` |
| `weight_family_results.json` | Spine §2.6 | `spine/scaling/` |
| `weight_family_sweep.py` | Spine §2.6, §2.8, §6.3 | `spine/scaling/` |
| `weight_family_sweep_ext.py` | Spine §2.6, §2.8, §6.3 | `spine/scaling/` |

## Verifying integrity

    sha256sum -c SHA256SUMS

## License

© 2025-2026 John David Mack. World Tree Project™ Licensed under Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0). Patent pending. See LICENSE.

John David Mack — in collaboration with Instruments of Mind · johndavid@eternityprocess.com · eternityprocess.com
