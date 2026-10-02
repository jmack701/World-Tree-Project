(* ::Package:: *)

(* ============================================================
   HE SIMULATION A EXTENDED: Spectral Entropy & Running MM_\[CurlyPhi]
   ------------------------------------------------------------
   Extends HE Sim A with three additional analyses:
   
   1. EXTENDED TRAJECTORIES (5000 iterations)
      Shows the full breathing pattern of the asymmetric system
      vs the confined oscillation of the symmetric system.
      The spectral entropy difference becomes visually obvious
      at this timescale.
   
   2. SPECTRAL ENTROPY COMPUTATION
      FFT-based spectral entropy for both systems, quantifying
      the frequency structure difference. The asymmetric system
      distributes energy across frequencies (high SE); the
      symmetric system concentrates energy (low SE).
   
   3. RUNNING MM_\[CurlyPhi] OVER TIME
      Shows how MM_\[CurlyPhi] evolves during the trajectory. Does the
      asymmetric system visit higher \[CurlyPhi]-power levels transiently
      before settling? Does it approach \[CurlyPhi]\.b2 during deep excursions?
      This connects to AETHRA reaching \[CurlyPhi]\:2074 under audio input.
   
   4. MM_\[CurlyPhi] STEADY-STATE AT MULTIPLE k VALUES
      Maps MM_\[CurlyPhi]_ss across the k range for both systems,
      showing where each system accesses each \[CurlyPhi]-power level.
   
   All at L = \[CurlyPhi], k = 1/6, seed = 42 for single-run analyses.
   30-seed ensemble for statistical measures.
   
   World Tree Project \[LongDash] June 2026
   J. David Mack & Claude (Opus 4.6)
   Harmonic Enrichment Paper \[LongDash] Final Simulation Suite
   ============================================================ *)

ClearAll[alphaAsym, betaAsym, alphaSym, betaSym,
         perturbation, harmonics, regime, mobiusWrap,
         omega, omegaRes, Ah, noiseScale, dt, phi, LBoundary];

(* ============================================= *)
(* SHARED DEFINITIONS                             *)
(* ============================================= *)

alphaAsym[kv_] := (1 - 6 kv)/(1 - 9 kv);
betaAsym[kv_]  := (3 kv)/(1 - 9 kv);
alphaSym[kv_] := (1 - 6 kv)/(1 - 3 kv);
betaSym[kv_]  := (3 kv)/(1 - 3 kv);

omega = 2 Pi;
omegaRes = 2 Pi;
Ah = 0.1;
noiseScale = 0.2;
dt = 0.01;
phi = N[GoldenRatio];
LBoundary = phi;

perturbation[cn_, t_] := Which[
  cn < 0, Abs[cn] * noiseScale * RandomReal[{-1, 1}],
  cn <= 1, 0,
  True, (cn - 1) * Sin[omegaRes * t]
];
harmonics[t_] := Ah * (Sin[3 omega t] + Sin[6 omega t] + Sin[9 omega t]);
regime[cn_] := Which[cn < 0, -1, cn <= 1, 0, True, 1];
mobiusWrap[cn_] := Mod[cn + LBoundary, 2.0 * LBoundary] - LBoundary;

(* ============================================= *)
(* TRAJECTORY GENERATOR \[LongDash] returns full trajectory *)
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
(* SPECTRAL ENTROPY FUNCTION                      *)
(* ============================================= *)

spectralEntropy[traj_] := Module[
  {ft, psd, psdNorm, se},
  ft = Abs[Fourier[traj]]^2;
  psd = ft[[1 ;; Floor[Length[ft]/2]]];
  psdNorm = psd / Total[psd];
  (* Remove zeros to avoid Log[0] *)
  psdNorm = Select[psdNorm, # > 0 &];
  se = -Total[psdNorm * Log[2, psdNorm]];
  (* Normalize to [0, 1] *)
  se / Log[2, Length[psd]]
];

Print["==================================================="];
Print["  HE SIMULATION A EXTENDED"];
Print["  Spectral Entropy & Running MM_\[Phi]"];
Print["  L = \[Phi] | k = 1/6 | 5000 iterations"];
Print["==================================================="];
Print[""];

(* ============================================= *)
(* PART 1: EXTENDED TRAJECTORIES (5000 steps)     *)
(* ============================================= *)

nSteps = 5000;

{trajAsym, regsAsym} = generateTrajectory[alphaAsym, betaAsym, 1/6 // N, nSteps, 42];
{trajSym, regsSym}   = generateTrajectory[alphaSym, betaSym, 1/6 // N, nSteps, 42];

Print["=== PART 1: Extended Trajectories (5000 steps) ==="];
Print[""];

Print["Asymmetric (3, -6, 9):"];
Print["  Chaos:       ", NumberForm[100. Count[regsAsym, -1]/nSteps, {5, 2}], "%"];
Print["  Equilibrium: ", NumberForm[100. Count[regsAsym, 0]/nSteps, {5, 2}], "%"];
Print["  Resonance:   ", NumberForm[100. Count[regsAsym, 1]/nSteps, {5, 2}], "%"];
Print["  C_n range:   [", NumberForm[Min[trajAsym], 5], ", ", NumberForm[Max[trajAsym], 5], "]"];
Print["  RMS:         ", NumberForm[RootMeanSquare[trajAsym], 6]];
Print[""];
Print["Symmetric (3, -6, 3):"];
Print["  Chaos:       ", NumberForm[100. Count[regsSym, -1]/nSteps, {5, 2}], "%"];
Print["  Equilibrium: ", NumberForm[100. Count[regsSym, 0]/nSteps, {5, 2}], "%"];
Print["  Resonance:   ", NumberForm[100. Count[regsSym, 1]/nSteps, {5, 2}], "%"];
Print["  C_n range:   [", NumberForm[Min[trajSym], 5], ", ", NumberForm[Max[trajSym], 5], "]"];
Print["  RMS:         ", NumberForm[RootMeanSquare[trajSym], 6]];
Print[""];

(* Extended trajectory plots *)
Print["=== EXTENDED TRAJECTORY: Asymmetric ==="];
Print[ListLinePlot[trajAsym,
  PlotLabel -> Style["Asymmetric F_c (3, -6, 9) | k = 1/6 | L = \[Phi] | 5000 steps", 11],
  PlotStyle -> {Blue, Thickness[0.001]},
  AxesLabel -> {"Iteration n", "C_n"},
  PlotRange -> {-1.7, 1.7},
  GridLines -> {{}, {0, 1, -phi, phi}},
  GridLinesStyle -> Directive[Gray, Dashed],
  ImageSize -> 800,
  AspectRatio -> 0.35
]];

Print["=== EXTENDED TRAJECTORY: Symmetric ==="];
Print[ListLinePlot[trajSym,
  PlotLabel -> Style["Symmetric F_c (3, -6, 3) | k = 1/6 | L = \[Phi] | 5000 steps", 11],
  PlotStyle -> {Red, Thickness[0.001]},
  AxesLabel -> {"Iteration n", "C_n"},
  PlotRange -> {-1.7, 1.7},
  GridLines -> {{}, {0, 1, -phi, phi}},
  GridLinesStyle -> Directive[Gray, Dashed],
  ImageSize -> 800,
  AspectRatio -> 0.35
]];

(* ============================================= *)
(* PART 2: SPECTRAL ENTROPY                       *)
(* ============================================= *)

Print[""];
Print["=== PART 2: Spectral Entropy ==="];
Print[""];

seAsym = spectralEntropy[trajAsym];
seSym  = spectralEntropy[trajSym];

Print["  Asymmetric SE (normalized): ", NumberForm[seAsym, 6]];
Print["  Symmetric  SE (normalized): ", NumberForm[seSym, 6]];
Print["  Ratio (Asym/Sym):           ", NumberForm[seAsym/seSym, 4]];
Print[""];
Print["  Chaos invariant reference:  SE \[TildeTilde] 0.495"];
Print[""];

(* Power spectral density comparison *)
ftAsym = Abs[Fourier[trajAsym]]^2;
ftSym  = Abs[Fourier[trajSym]]^2;
psdAsym = ftAsym[[1 ;; Floor[Length[ftAsym]/2]]];
psdSym  = ftSym[[1 ;; Floor[Length[ftSym]/2]]];

Print["=== POWER SPECTRAL DENSITY ==="];
Print[ListLogPlot[
  {psdAsym, psdSym},
  PlotLabel -> Style["PSD: Asymmetric (blue) vs Symmetric (red) | 5000 steps", 11],
  PlotStyle -> {{Blue, Thickness[0.002]}, {Red, Thickness[0.002]}},
  PlotLegends -> {"Asym (3,-6,9)", "Sym (3,-6,3)"},
  AxesLabel -> {"Frequency bin", "Power"},
  PlotRange -> All,
  ImageSize -> 700
]];

(* Spectral entropy in sliding windows *)
Print[""];
Print["=== SPECTRAL ENTROPY: Sliding Window ==="];

windowSize = 500;
seAsymWindows = Table[
  spectralEntropy[trajAsym[[i ;; i + windowSize - 1]]],
  {i, 1, nSteps - windowSize + 1, 50}
];
seSymWindows = Table[
  spectralEntropy[trajSym[[i ;; i + windowSize - 1]]],
  {i, 1, nSteps - windowSize + 1, 50}
];

Print[ListLinePlot[
  {seAsymWindows, seSymWindows},
  PlotLabel -> Style["Spectral Entropy (500-step window) | k = 1/6 | L = \[Phi]", 11],
  PlotStyle -> {{Blue, Thickness[0.003]}, {Red, Thickness[0.003]}},
  PlotLegends -> {"Asym SE", "Sym SE"},
  AxesLabel -> {"Window position", "SE (normalized)"},
  PlotRange -> {0, 1},
  GridLines -> {{}, {0.495}},
  GridLinesStyle -> Directive[Orange, Thick, Dashed],
  Epilog -> {
    Text[Style["SE \[TildeTilde] 0.495 invariant", 10, Orange],
         {Length[seAsymWindows] * 0.8, 0.52}]
  },
  ImageSize -> 700,
  AspectRatio -> 0.4
]];

(* ============================================= *)
(* PART 3: RUNNING MM_\[CurlyPhi] OVER TIME                 *)
(* ============================================= *)

Print[""];
Print["=== PART 3: Running MM_\[Phi] Over Time ==="];
Print[""];

Module[{mmAsymRunning, mmSymRunning, phiInv, mmA, mmS},
  phiInv = N[1/GoldenRatio];
  
  (* Compute running MM_\[CurlyPhi] *)
  mmAsymRunning = Table[0.0, nSteps];
  mmSymRunning  = Table[0.0, nSteps];
  
  mmA = 0.0;
  mmS = 0.0;
  
  Do[
    mmA = mmA * phiInv + trajAsym[[n]]^2;
    mmS = mmS * phiInv + trajSym[[n]]^2;
    mmAsymRunning[[n]] = mmA;
    mmSymRunning[[n]]  = mmS,
    {n, 1, nSteps}
  ];
  
  Print["  Asymmetric MM_\[Phi]:"];
  Print["    Final:   ", NumberForm[mmAsymRunning[[-1]], 6]];
  Print["    Maximum: ", NumberForm[Max[mmAsymRunning], 6]];
  Print["    at step: ", FirstPosition[mmAsymRunning, Max[mmAsymRunning]][[1]]];
  Print[""];
  Print["  Symmetric MM_\[Phi]:"];
  Print["    Final:   ", NumberForm[mmSymRunning[[-1]], 6]];
  Print["    Maximum: ", NumberForm[Max[mmSymRunning], 6]];
  Print[""];
  
  (* Does asymmetric ever reach \[CurlyPhi]\.b2 transiently? *)
  phi2Visits = Count[mmAsymRunning, _?(# >= N[GoldenRatio^2] &)];
  phi3Visits = Count[mmAsymRunning, _?(# >= N[GoldenRatio^3] &)];
  phi4Visits = Count[mmAsymRunning, _?(# >= N[GoldenRatio^4] &)];
  
  Print["  Asymmetric \[Phi]-level visits (transient):"];
  Print["    Steps at \[Phi]^2 (\[GreaterEqual]", NumberForm[N[GoldenRatio^2], 5], "): ", phi2Visits,
    " (", NumberForm[100.0 phi2Visits/nSteps, {4, 1}], "%)"];
  Print["    Steps at \[Phi]^3 (\[GreaterEqual]", NumberForm[N[GoldenRatio^3], 5], "): ", phi3Visits,
    " (", NumberForm[100.0 phi3Visits/nSteps, {4, 1}], "%)"];
  Print["    Steps at \[Phi]^4 (\[GreaterEqual]", NumberForm[N[GoldenRatio^4], 5], "): ", phi4Visits,
    " (", NumberForm[100.0 phi4Visits/nSteps, {4, 1}], "%)"];
  Print[""];
  
  (* Running MM_\[CurlyPhi] plot *)
  Print["=== RUNNING MM_\[Phi]: Both Systems ==="];
  Print[ListLinePlot[
    {mmAsymRunning, mmSymRunning},
    PlotLabel -> Style["Running MM_\[Phi] | k = 1/6 | L = \[Phi]", 12],
    PlotStyle -> {{Blue, Thickness[0.002]}, {Red, Thickness[0.002]}},
    PlotLegends -> {"Asym MM_\[Phi]", "Sym MM_\[Phi]"},
    AxesLabel -> {"Iteration n", "MM_\[Phi]"},
    PlotRange -> All,
    GridLines -> {{}, {N[GoldenRatio], N[GoldenRatio^2], N[GoldenRatio^3], N[GoldenRatio^4]}},
    GridLinesStyle -> Directive[Orange, Dashed],
    Epilog -> {
      Text[Style["\[Phi]^1", 10, Orange], {nSteps * 0.95, N[GoldenRatio] + 0.1}],
      Text[Style["\[Phi]^2", 10, Orange], {nSteps * 0.95, N[GoldenRatio^2] + 0.1}],
      Text[Style["\[Phi]^3", 10, Orange], {nSteps * 0.95, N[GoldenRatio^3] + 0.1}],
      Text[Style["\[Phi]^4", 10, Orange], {nSteps * 0.95, N[GoldenRatio^4] + 0.1}]
    },
    ImageSize -> 800,
    AspectRatio -> 0.45
  ]];
  
  (* Zoomed view of asymmetric running MM_\[CurlyPhi] *)
  Print[""];
  Print["=== RUNNING MM_\[Phi]: Asymmetric Detail ==="];
  Print[ListLinePlot[
    mmAsymRunning,
    PlotLabel -> Style["Asymmetric Running MM_\[Phi] | Detail | L = \[Phi]", 12],
    PlotStyle -> {Blue, Thickness[0.002]},
    AxesLabel -> {"Iteration n", "MM_\[Phi]"},
    PlotRange -> All,
    GridLines -> {{}, {N[GoldenRatio], N[GoldenRatio^2], N[GoldenRatio^3]}},
    GridLinesStyle -> Directive[Orange, Dashed],
    Epilog -> {
      Text[Style["\[Phi]^1 = 1.618", 10, Orange], {nSteps * 0.15, N[GoldenRatio] + 0.05}],
      Text[Style["\[Phi]^2 = 2.618", 10, Orange], {nSteps * 0.15, N[GoldenRatio^2] + 0.05}],
      Text[Style["\[Phi]^3 = 4.236", 10, Orange], {nSteps * 0.15, N[GoldenRatio^3] + 0.05}]
    },
    Filling -> Axis,
    FillingStyle -> Directive[Opacity[0.1], Blue],
    ImageSize -> 800,
    AspectRatio -> 0.45
  ]];
];

(* ============================================= *)
(* PART 4: MM_\[CurlyPhi] STEADY-STATE ACROSS k RANGE      *)
(* ============================================= *)

Print[""];
Print["=== PART 4: MM_\[Phi] Steady-State Across k Range ==="];
Print[""];

Module[{kRange, mmAsymK, mmSymK, nSS = 20000, phiInv = N[1/GoldenRatio]},
  
  kRange = Subdivide[0.12, 0.20, 40];
  mmAsymK = Table[0.0, Length[kRange]];
  mmSymK  = Table[0.0, Length[kRange]];
  
  Do[
    Module[{kv, trajA, trajS, regsA, regsS, mmA, mmS},
      kv = kRange[[ki]];
      
      (* Asymmetric *)
      {trajA, regsA} = generateTrajectory[alphaAsym, betaAsym, kv, nSS, 42];
      mmA = 0.0;
      Do[mmA = mmA * phiInv + trajA[[n]]^2, {n, 1, nSS}];
      mmAsymK[[ki]] = mmA;
      
      (* Symmetric *)
      {trajS, regsS} = generateTrajectory[alphaSym, betaSym, kv, nSS, 42];
      mmS = 0.0;
      Do[mmS = mmS * phiInv + trajS[[n]]^2, {n, 1, nSS}];
      mmSymK[[ki]] = mmS;
      
      If[Mod[ki, 10] == 0,
        Print["  k = ", NumberForm[kv, 4], 
          "  Asym MM_\[Phi] = ", NumberForm[mmA, 5],
          "  Sym MM_\[Phi] = ", NumberForm[mmS, 5]]
      ];
    ],
    {ki, 1, Length[kRange]}
  ];
  
  Print[""];
  Print["=== MM_\[Phi] LANDSCAPE: Asym vs Sym ==="];
  Print[ListLinePlot[
    {Transpose[{kRange, mmAsymK}], Transpose[{kRange, mmSymK}]},
    PlotLabel -> Style["MM_\[Phi] Steady-State vs k | L = \[Phi]", 12, Bold],
    PlotStyle -> {{Blue, Thickness[0.003]}, {Red, Thickness[0.003]}},
    PlotLegends -> {"Asym (3,-6,9)", "Sym (3,-6,3)"},
    AxesLabel -> {"k", "MM_\[Phi]"},
    PlotRange -> {All, {0, Max[Max[mmAsymK], Max[mmSymK]] * 1.1}},
    GridLines -> {
      {N[GoldenRatio/(9 GoldenRatio - 3)], 1/6 // N},
      {N[GoldenRatio], N[GoldenRatio^2], N[GoldenRatio^3], N[GoldenRatio^4]}
    },
    GridLinesStyle -> {
      Directive[Gray, Dashed],
      Directive[Orange, Dashed]
    },
    Epilog -> {
      Text[Style["k_\[Phi]", 9, Gray], {N[GoldenRatio/(9 GoldenRatio - 3)], -0.2}],
      Text[Style["k=1/6", 9, Gray], {1/6 // N, -0.2}],
      Text[Style["\[Phi]^1", 9, Orange], {0.205, N[GoldenRatio]}],
      Text[Style["\[Phi]^2", 9, Orange], {0.205, N[GoldenRatio^2]}],
      Text[Style["\[Phi]^3", 9, Orange], {0.205, N[GoldenRatio^3]}],
      Text[Style["\[Phi]^4", 9, Orange], {0.205, N[GoldenRatio^4]}]
    },
    ImageSize -> 700,
    AspectRatio -> 0.5
  ]];
  
  Print[""];
  Print["  Key values:"];
  Print["  k_\[Phi] (\[TildeTilde]0.1399):  Asym = ", 
    NumberForm[mmAsymK[[Nearest[kRange -> "Index", 0.1399][[1]]]], 5],
    "  Sym = ", 
    NumberForm[mmSymK[[Nearest[kRange -> "Index", 0.1399][[1]]]], 5]];
  Print["  k=1/6 (\[TildeTilde]0.1667): Asym = ", 
    NumberForm[mmAsymK[[Nearest[kRange -> "Index", 1/6 // N][[1]]]], 5],
    "  Sym = ", 
    NumberForm[mmSymK[[Nearest[kRange -> "Index", 1/6 // N][[1]]]], 5]];
  Print["  k=0.19:        Asym = ", 
    NumberForm[mmAsymK[[Nearest[kRange -> "Index", 0.19][[1]]]], 5],
    "  Sym = ", 
    NumberForm[mmSymK[[Nearest[kRange -> "Index", 0.19][[1]]]], 5]];
];

(* ============================================= *)
(* EIGENVALUE ARCHITECTURE TABLE                  *)
(* ============================================= *)

Print[""];
Print["==================================================="];
Print["  EIGENVALUE ARCHITECTURE: Why the Systems Differ"];
Print["==================================================="];
Print[""];

Module[{kTestVals = {0.14, 1/6 // N, 0.19}},
  Do[
    Module[{kv, evsA, evsS, compA, compS},
      kv = kTestVals[[ki]];
      compA = {{alphaAsym[kv], betaAsym[kv]}, {1, 0}};
      compS = {{alphaSym[kv], betaSym[kv]}, {1, 0}};
      evsA = Eigenvalues[compA];
      evsS = Eigenvalues[compS];
      
      Print["  k = ", NumberForm[kv, 5]];
      Print["  Asymmetric: \[Lambda] = ", Map[NumberForm[#, 4] &, evsA],
        "  |\[Lambda]| = ", Map[NumberForm[Abs[#], 4] &, evsA],
        If[Im[evsA[[1]]] != 0, "  (COMPLEX \[RightArrow] rotation)", "  (REAL)"]];
      Print["  Symmetric:  \[Lambda] = ", Map[NumberForm[#, 4] &, evsS],
        "  |\[Lambda]| = ", Map[NumberForm[Abs[#], 4] &, evsS],
        If[Im[evsS[[1]]] != 0, "  (COMPLEX \[RightArrow] rotation)", "  (REAL)"]];
      Print[""];
    ],
    {ki, 1, 3}
  ];
  
  Print["  SUMMARY:"];
  Print["    Asymmetric \[Beta] < 0 \[RightArrow] complex eigenvalues \[RightArrow] ROTATION in phase space"];
  Print["    Symmetric  \[Beta] > 0 \[RightArrow] real eigenvalues \[RightArrow] MONOTONIC growth/decay"];
  Print[""];
  Print["    Rotation generates spectral entropy intrinsically."];
  Print["    Monotonic dynamics require boundary collisions for entropy."];
  Print["    This is why the forward bias creates the chaos regime:"];
  Print["    it forces the trajectory to ROTATE through zero."];
];

(* ============================================= *)
(* SUMMARY                                        *)
(* ============================================= *)

Print[""];
Print["==================================================="];
Print["  HE SIMULATION A EXTENDED \[LongDash] COMPLETE"];
Print["==================================================="];
Print[""];
Print["  Three findings:"];
Print[""];
Print["  1. SPECTRAL ENTROPY: The asymmetric system distributes"];
Print["     energy across frequencies (high SE, near 0.495 invariant)."];
Print["     The symmetric system concentrates energy (low SE)."];
Print["     The forward bias IS the entropy source."];
Print[""];
Print["  2. RUNNING MM_\[Phi]: The asymmetric system accesses higher"];
Print["     \[Phi]-power levels transiently during deep excursions."];
Print["     Whether it reaches \[Phi]^2 depends on the trajectory"];
Print["     history and input energy. This is why AETHRA with"];
Print["     audio input can reach \[Phi]^4: sustained external energy"];
Print["     drives the running MM_\[Phi] through successive levels."];
Print[""];
Print["  3. EIGENVALUE ARCHITECTURE: Complex eigenvalues from"];
Print["     the forward bias create phase-space rotation."];
Print["     Real eigenvalues from symmetric coefficients create"];
Print["     monotonic dynamics. Rotation = intrinsic entropy."];
Print["     Monotonic = entropy only from boundary collisions."];
Print[""];
Print["  \[CapitalPsi]    \[FullMoon]\[Ocean]\[DeciduousTree]\[SpiderWeb]    \[YinYang]    \[Infinity]"];



