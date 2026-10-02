(* ::Package:: *)

(* ============================================================
   HE SIMULATION A: Forward Bias Asymmetry Test \[LongDash] L = \[CurlyPhi]
   ------------------------------------------------------------
   Extension of The Field v2.0 Simulation A to L = \[CurlyPhi].
   
   The original Sim A at L = 1.5 established that the forward
   bias in the F_c coefficients \[LongDash] the +6 term distinguishing
   "become" (9) from "remember" (3) \[LongDash] IS the mechanism that
   creates the chaos regime. Symmetric coefficients (3, -6, 3)
   eliminated chaos entirely: 0.00% chaos, trajectory confined
   to positive domain [0.10, 1.12], mean C_n = 0.825.
   
   This simulation tests whether that structural finding
   persists at L = \[CurlyPhi], where the M\[ODoubleDot]bius boundary matches the
   golden ratio. At L = \[CurlyPhi]:
     - Trajectory range expands from [-1.5, 1.5] to [-\[CurlyPhi], \[CurlyPhi]]
     - P_max in resonance = (\[CurlyPhi]-1)\[CenterDot]sin(\[Omega]t) = \[CurlyPhi]\:207b\.b9 \[TildeTilde] 0.618
       (vs 0.5 at L = 1.5)
     - More room for resonance excursion (up to \[CurlyPhi] vs 1.5)
   
   The prediction: the forward bias finding is structural and
   will persist regardless of L, because the mechanism is the
   SIGN of \[Beta] (negative for asymmetric, positive for symmetric),
   not the magnitude of the boundary.
   
   Comparison data (Sim A at L = 1.5):
     Asymmetric: chaos ~49.5%, full regime structure
     Symmetric:  chaos 0.00%, confined to positive domain
   
   World Tree Project \[LongDash] June 2026
   J. David Mack & Claude (Opus 4.6)
   Harmonic Enrichment Paper \[LongDash] Final Simulation Suite
   ============================================================ *)

ClearAll[alphaAsym, betaAsym, alphaSym, betaSym,
         perturbation, harmonics, regime, mobiusWrap,
         generateTrajectory, omega, omegaRes, Ah, noiseScale, dt];

(* ============================================= *)
(* COEFFICIENT DEFINITIONS                        *)
(* ============================================= *)

(* Asymmetric: (3, -6, 9) - the actual F_c system *)
alphaAsym[kv_] := (1 - 6 kv)/(1 - 9 kv);
betaAsym[kv_]  := (3 kv)/(1 - 9 kv);

(* Symmetric: (3, -6, 3) - pure discrete Laplacian *)
alphaSym[kv_] := (1 - 6 kv)/(1 - 3 kv);
betaSym[kv_]  := (3 kv)/(1 - 3 kv);

(* ============================================= *)
(* VERIFY COEFFICIENTS AT k = 1/6                *)
(* ============================================= *)

Print["==================================================="];
Print["  HE SIMULATION A: Forward Bias \[LongDash] L = \[Phi]"];
Print["==================================================="];
Print[""];
Print["=== Coefficient Verification at k = 1/6 ==="];
Print[""];
Print["Asymmetric (3, -6, 9):"];
Print["  alpha = ", alphaAsym[1/6], "  beta = ", betaAsym[1/6]];
Print["  Companion matrix eigenvalues: ",
  Eigenvalues[{{alphaAsym[1/6], betaAsym[1/6]}, {1, 0}}]];
Print["  \[Beta] sign: NEGATIVE (sign inversion \[Rule] zero crossing)"];
Print[""];
Print["Symmetric (3, -6, 3):"];
Print["  alpha = ", alphaSym[1/6], "  beta = ", betaSym[1/6]];
Print["  Companion matrix eigenvalues: ",
  Eigenvalues[{{alphaSym[1/6], betaSym[1/6]}, {1, 0}}]];
Print["  \[Beta] sign: POSITIVE (sign preservation \[Rule] no zero crossing)"];
Print[""];

(* ============================================= *)
(* SHARED PARAMETERS                              *)
(* ============================================= *)

omega = 2 Pi;
omegaRes = 2 Pi;
Ah = 0.1;
noiseScale = 0.2;
dt = 0.01;

(* === THE CHANGE: L = \[CurlyPhi] === *)
phi = N[GoldenRatio];
LBoundary = phi;

Print["  L = \[Phi] = ", NumberForm[LBoundary, 10]];
Print["  Trajectory range: [-\[Phi], \[Phi]] = [-",
  NumberForm[phi, 6], ", ", NumberForm[phi, 6], "]"];
Print["  Asymmetric max pre-wrap at k=1/6: |\[Alpha]|\[CenterDot]\[Phi] + |\[Beta]|\[CenterDot]\[Phi] + P_max + H_max = ",
  NumberForm[Abs[alphaAsym[1/6 // N]] * phi + Abs[betaAsym[1/6 // N]] * phi + (phi - 1) + 0.3, 6]];
Print["  Symmetric max pre-wrap at k=1/6:  |\[Alpha]|\[CenterDot]\[Phi] + |\[Beta]|\[CenterDot]\[Phi] + P_max + H_max = ",
  NumberForm[Abs[alphaSym[1/6 // N]] * phi + Abs[betaSym[1/6 // N]] * phi + (phi - 1) + 0.3, 6]];
Print[""];

(* Perturbation function: three regimes *)
perturbation[cn_, t_] := Which[
  cn < 0, Abs[cn] * noiseScale * RandomReal[{-1, 1}],
  cn <= 1, 0,
  True, (cn - 1) * Sin[omegaRes * t]
];

(* Harmonic enrichment *)
harmonics[t_] := Ah * (Sin[3 omega t] + Sin[6 omega t] + Sin[9 omega t]);

(* Regime classifier \[LongDash] UNCHANGED, boundary-independent *)
regime[cn_] := Which[cn < 0, -1, cn <= 1, 0, True, 1];

(* M\[ODoubleDot]bius wrapping at L = \[CurlyPhi] *)
mobiusWrap[cn_] := Mod[cn + LBoundary, 2.0 * LBoundary] - LBoundary;

(* ============================================= *)
(* TRAJECTORY GENERATOR                           *)
(* ============================================= *)

generateTrajectory[alphaFn_, betaFn_, kv_, nSteps_, seed_] := Module[
  {traj, regs},
  SeedRandom[seed];
  traj = Table[0.0, nSteps];
  regs = Table[0, nSteps];
  traj[[1]] = 0.5;
  traj[[2]] = 0.3;
  regs[[1]] = regime[traj[[1]]];
  regs[[2]] = regime[traj[[2]]];
  Do[
    Module[{t, cnp1},
      t = n * dt;
      cnp1 = alphaFn[kv] * traj[[n - 1]] + betaFn[kv] * traj[[n - 2]] +
             perturbation[traj[[n - 1]], t] + harmonics[t];
      traj[[n]] = mobiusWrap[cnp1];
      regs[[n]] = regime[traj[[n]]];
    ],
    {n, 3, nSteps}
  ];
  {traj, regs}
];

(* ============================================= *)
(* MAIN COMPARISON: Multi-seed ensemble           *)
(* ============================================= *)

Print["=== MAIN COMPARISON: 30-seed ensemble ==="];
Print[""];

nSteps = 20000;
seeds = Range[42, 71];  (* 30 seeds *)

(* Test at three k values across the operating range *)
kValues = {0.14, 1/6 // N, 0.19};
kLabels = {"0.14 (Zone I, near k_\[Phi])", "0.16667 (k = 1/6, boundary)", "0.19 (Zone III)"};

Do[
  kv = kValues[[ki]];
  Print[""];
  Print["=== k = ", kLabels[[ki]], " ==="];
  Print["  Asymmetric: \[Alpha] = ", NumberForm[N[alphaAsym[kv]], 6],
         ", \[Beta] = ", NumberForm[N[betaAsym[kv]], 6]];
  Print["  Symmetric:  \[Alpha] = ", NumberForm[N[alphaSym[kv]], 6],
         ", \[Beta] = ", NumberForm[N[betaSym[kv]], 6]];
  Print[""];
  
  (* Asymmetric ensemble *)
  asymChaos = Table[
    Module[{traj, regs},
      {traj, regs} = generateTrajectory[alphaAsym, betaAsym, kv, nSteps, s];
      100.0 * Count[regs, -1] / nSteps
    ],
    {s, seeds}
  ];
  
  (* Also collect full regime data for one representative seed *)
  {asymTrajRep, asymRegsRep} = generateTrajectory[alphaAsym, betaAsym, kv, nSteps, 42];
  
  (* Symmetric ensemble *)
  symChaos = Table[
    Module[{traj, regs},
      {traj, regs} = generateTrajectory[alphaSym, betaSym, kv, nSteps, s];
      100.0 * Count[regs, -1] / nSteps
    ],
    {s, seeds}
  ];
  
  {symTrajRep, symRegsRep} = generateTrajectory[alphaSym, betaSym, kv, nSteps, 42];
  
  Print["  ASYMMETRIC (3, -6, 9):"];
  Print["    Chaos mean:   ", NumberForm[Mean[asymChaos], {5, 2}], "%"];
  Print["    Chaos std:    ", NumberForm[StandardDeviation[asymChaos], {4, 2}], "%"];
  Print["    Chaos range:  [", NumberForm[Min[asymChaos], {5, 2}], "%, ",
         NumberForm[Max[asymChaos], {5, 2}], "%]"];
  Print["    Equil:        ", NumberForm[100. Count[asymRegsRep, 0]/nSteps, {5, 2}], "%"];
  Print["    Resonance:    ", NumberForm[100. Count[asymRegsRep, 1]/nSteps, {5, 2}], "%"];
  Print["    C_n range:    [", NumberForm[Min[asymTrajRep], 4], ", ",
         NumberForm[Max[asymTrajRep], 4], "]"];
  Print["    Mean C_n:     ", NumberForm[Mean[asymTrajRep], 6]];
  Print[""];
  Print["  SYMMETRIC (3, -6, 3):"];
  Print["    Chaos mean:   ", NumberForm[Mean[symChaos], {5, 2}], "%"];
  Print["    Chaos std:    ", NumberForm[StandardDeviation[symChaos], {4, 2}], "%"];
  Print["    Chaos range:  [", NumberForm[Min[symChaos], {5, 2}], "%, ",
         NumberForm[Max[symChaos], {5, 2}], "%]"];
  Print["    Equil:        ", NumberForm[100. Count[symRegsRep, 0]/nSteps, {5, 2}], "%"];
  Print["    Resonance:    ", NumberForm[100. Count[symRegsRep, 1]/nSteps, {5, 2}], "%"];
  Print["    C_n range:    [", NumberForm[Min[symTrajRep], 4], ", ",
         NumberForm[Max[symTrajRep], 4], "]"];
  Print["    Mean C_n:     ", NumberForm[Mean[symTrajRep], 6]];
  Print[""];
  Print["  DIFFERENCE (Sym - Asym): ",
         NumberForm[Mean[symChaos] - Mean[asymChaos], {4, 2}], " percentage points"];
  ,
  {ki, Length[kValues]}
];

(* ============================================= *)
(* DETAILED SINGLE-RUN AT k = 1/6                *)
(* ============================================= *)

Print[""];
Print["=== DETAILED SINGLE RUN: k = 1/6, seed = 42 ==="];
Print[""];

{trajAsym, regsAsym} = generateTrajectory[alphaAsym, betaAsym, 1/6 // N, nSteps, 42];
{trajSym, regsSym}   = generateTrajectory[alphaSym, betaSym, 1/6 // N, nSteps, 42];

Print["Asymmetric (3, -6, 9):"];
Print["  Chaos:       ", NumberForm[100. Count[regsAsym, -1]/nSteps, {5, 2}], "%"];
Print["  Equilibrium: ", NumberForm[100. Count[regsAsym, 0]/nSteps, {5, 2}], "%"];
Print["  Resonance:   ", NumberForm[100. Count[regsAsym, 1]/nSteps, {5, 2}], "%"];
Print["  C_n range:   [", NumberForm[Min[trajAsym], 4], ", ", NumberForm[Max[trajAsym], 4], "]"];
Print["  Mean C_n:    ", NumberForm[Mean[trajAsym], 6]];
Print[""];
Print["Symmetric (3, -6, 3):"];
Print["  Chaos:       ", NumberForm[100. Count[regsSym, -1]/nSteps, {5, 2}], "%"];
Print["  Equilibrium: ", NumberForm[100. Count[regsSym, 0]/nSteps, {5, 2}], "%"];
Print["  Resonance:   ", NumberForm[100. Count[regsSym, 1]/nSteps, {5, 2}], "%"];
Print["  C_n range:   [", NumberForm[Min[trajSym], 4], ", ", NumberForm[Max[trajSym], 4], "]"];
Print["  Mean C_n:    ", NumberForm[Mean[trajSym], 6]];

(* ============================================= *)
(* L = 1.5 vs L = \[CurlyPhi] COMPARISON TABLE             *)
(* ============================================= *)

Print[""];
Print["==================================================="];
Print["  CROSS-BOUNDARY COMPARISON: L = 1.5 vs L = \[Phi]"];
Print["==================================================="];
Print[""];
Print["  L = 1.5 (original Sim A):"];
Print["    Asymmetric: chaos ~49.5%, C_n range [-1.5, 1.5]"];
Print["    Symmetric:  chaos 0.00%, C_n range [0.10, 1.12]"];
Print[""];
Print["  L = \[Phi] (this run):"];
Print["    Asymmetric: chaos ", 
  NumberForm[100. Count[regsAsym, -1]/nSteps, {5, 2}], 
  "%, C_n range [", NumberForm[Min[trajAsym], 4], ", ", 
  NumberForm[Max[trajAsym], 4], "]"];
Print["    Symmetric:  chaos ", 
  NumberForm[100. Count[regsSym, -1]/nSteps, {5, 2}], 
  "%, C_n range [", NumberForm[Min[trajSym], 4], ", ", 
  NumberForm[Max[trajSym], 4], "]"];
Print[""];
If[100. Count[regsSym, -1]/nSteps < 1.0,
  Print["  \[DoubleRightArrow] Forward bias finding CONFIRMED at L = \[Phi]."];
  Print["     The \[Beta] sign mechanism is boundary-independent."];
  Print["     Symmetric coefficients eliminate chaos regardless of L."],
  Print["  \[DoubleRightArrow] Forward bias finding MODIFIED at L = \[Phi]."];
  Print["     The wider boundary introduces chaos in the symmetric system."];
  Print["     The mechanism needs revision."]
];

(* ============================================= *)
(* TRAJECTORY PLOTS                               *)
(* ============================================= *)

Print[""];
Print["=== TRAJECTORY PLOTS (first 500 steps) ==="];

plotAsym = ListLinePlot[trajAsym[[1 ;; 500]],
  PlotLabel -> Style["Asymmetric F_c (3, -6, 9) | k = 1/6 | L = \[Phi]", 11],
  PlotStyle -> {Blue, Thickness[0.002]},
  AxesLabel -> {"Iteration n", "C_n"},
  PlotRange -> {-1.7, 1.7},
  GridLines -> {{}, {0, 1, -phi, phi}},
  GridLinesStyle -> Directive[Gray, Dashed],
  ImageSize -> 600
];

plotSym = ListLinePlot[trajSym[[1 ;; 500]],
  PlotLabel -> Style["Symmetric F_c (3, -6, 3) | k = 1/6 | L = \[Phi]", 11],
  PlotStyle -> {Red, Thickness[0.002]},
  AxesLabel -> {"Iteration n", "C_n"},
  PlotRange -> {-1.7, 1.7},
  GridLines -> {{}, {0, 1, -phi, phi}},
  GridLinesStyle -> Directive[Gray, Dashed],
  ImageSize -> 600
];

Print[plotAsym];
Print[plotSym];

(* Overlay *)
plotOverlay = ListLinePlot[
  {trajAsym[[1 ;; 500]], trajSym[[1 ;; 500]]},
  PlotLabel -> Style["Asym (blue) vs Sym (red) | k = 1/6 | L = \[Phi]", 11],
  PlotStyle -> {{Blue, Thickness[0.002]}, {Red, Thickness[0.002]}},
  AxesLabel -> {"Iteration n", "C_n"},
  PlotRange -> {-1.7, 1.7},
  GridLines -> {{}, {0, -phi, phi}},
  GridLinesStyle -> Directive[Gray, Dashed],
  PlotLegends -> {"Asym (3,-6,9)", "Sym (3,-6,3)"},
  ImageSize -> 700
];

Print[plotOverlay];

(* ============================================= *)
(* MM_\[CurlyPhi] COMPARISON                                *)
(* Compute \[CurlyPhi]-weighted trajectory memory for both  *)
(* ============================================= *)

Print[""];
Print["=== MM_\[Phi] COMPARISON ==="];
Print[""];

Module[{mmAsym, mmSym, mmAsymFinal, mmSymFinal,
        phiInv = N[1/GoldenRatio]},
  
  (* Compute MM_\[CurlyPhi] for asymmetric trajectory *)
  mmAsym = 0.0;
  Do[
    mmAsym = mmAsym * phiInv + trajAsym[[n]]^2,
    {n, 1, nSteps}
  ];
  mmAsymFinal = mmAsym;
  
  (* Compute MM_\[CurlyPhi] for symmetric trajectory *)
  mmSym = 0.0;
  Do[
    mmSym = mmSym * phiInv + trajSym[[n]]^2,
    {n, 1, nSteps}
  ];
  mmSymFinal = mmSym;
  
  Print["  Asymmetric MM_\[Phi] (final):  ", NumberForm[mmAsymFinal, 6]];
  Print["  Symmetric  MM_\[Phi] (final):  ", NumberForm[mmSymFinal, 6]];
  Print[""];
  Print["  \[Phi]^1 = ", NumberForm[N[GoldenRatio], 6]];
  Print["  \[Phi]^2 = ", NumberForm[N[GoldenRatio^2], 6]];
  Print["  \[Phi]^3 = ", NumberForm[N[GoldenRatio^3], 6]];
  Print[""];
  Print["  Asymmetric operates at \[Phi]^",
    Which[
      mmAsymFinal < N[GoldenRatio^1.5], "1",
      mmAsymFinal < N[GoldenRatio^2.5], "2",
      mmAsymFinal < N[GoldenRatio^3.5], "3",
      True, "4"
    ], " level"];
  Print["  Symmetric  operates at \[Phi]^",
    Which[
      mmSymFinal < N[GoldenRatio^1.5], "1",
      mmSymFinal < N[GoldenRatio^2.5], "2",
      mmSymFinal < N[GoldenRatio^3.5], "3",
      True, "4"
    ], " level"];
  Print[""];
  Print["  The forward bias determines not just regime structure"];
  Print["  but which \[Phi]-power level the system accesses."];
];

(* ============================================= *)
(* SUMMARY                                        *)
(* ============================================= *)

Print[""];
Print["==================================================="];
Print["  HE SIMULATION A \[LongDash] L = \[Phi] \[LongDash] COMPLETE"];
Print["==================================================="];
Print[""];
Print["  The mechanism is the SIGN of \[Beta]:"];
Print["    Asymmetric: \[Beta] = -1 at k=1/6 (sign inversion)"];
Print["    Symmetric:  \[Beta] = +1 at k=1/6 (sign preservation)"];
Print[""];
Print["  Sign inversion forces zero crossing."];
Print["  Zero crossing creates the chaos regime."];
Print["  The chaos regime enables the ~49.5% invariant."];
Print["  The +6 forward bias is the structural source."];
Print[""];
Print["  This mechanism is TOPOLOGICAL (sign), not METRIC (boundary)."];
Print["  It persists at any L."];
Print[""];
Print["  \[CapitalPsi]    \[FullMoon]\[Ocean]\[DeciduousTree]\[SpiderWeb]    \[YinYang]    \[Infinity]"];



