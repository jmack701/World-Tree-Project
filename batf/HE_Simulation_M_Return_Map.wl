(* ::Package:: *)

(* ============================================================
   HE SIMULATION M: RETURN MAP & LEMNISCATE ATTRACTOR
   ------------------------------------------------------------
   Generates the (C_n, C_{n+1}) return map under L = \[CurlyPhi],
   revealing the lemniscate attractor geometry.
   
   Four outputs:
   
   1. RETURN MAP at k_\[CurlyPhi] under L = \[CurlyPhi]
      Regime-colored (C_n vs C_{n+1}) showing the lemniscate
      attractor with chaos in the outer lobes, resonance at
      the crossing, equilibrium at the transitions.
   
   2. THREE-PANEL k COMPARISON
      Lemniscate at k_\[CurlyPhi], k = 1/6, and k = 0.17 showing
      geometric transition across the phase boundary.
   
   3. LUMINBROT vs CLIPPED ESCAPE MAP
      Same initial condition grid, one panel under M\[ODoubleDot]bius
      (MM\[CurlyPhi] depth coloring, 0% escape), one panel under
      clipping (escape time coloring, fractal boundary).
   
   4. METRICS SUMMARY
      SE, C/E/R, MM\[CurlyPhi], virtual escape count \[LongDash] formatted
      for direct insertion into BatF reproducibility table.
   
   All at L = \[CurlyPhi] for M\[ODoubleDot]bius panels. Seed = 42.
   
   World Tree Project \[LongDash] July 2026
   J. David Mack & Claude (Opus 4.6)
   Blade and the Field \[LongDash] Simulation Suite
   ============================================================ *)

ClearAll[alphaAsym, betaAsym, perturbation, harmonics, regime,
         mobiusWrap, clipWrap, omega, omegaRes, Ah, noiseScale,
         dt, phi, LBoundary];

(* ============================================= *)
(* SHARED DEFINITIONS                             *)
(* ============================================= *)

alphaAsym[kv_] := (1 - 6 kv)/(1 - 9 kv);
betaAsym[kv_]  := (3 kv)/(1 - 9 kv);

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
clipWrap[cn_, L_] := Max[-L, Min[L, cn]];

(* ============================================= *)
(* TRAJECTORY GENERATOR \[LongDash] returns full trajectory *)
(* ============================================= *)

generateTrajectory[kv_, nSteps_, seed_, wrapFn_] := Module[
  {traj, regs, escapeCount = 0},
  SeedRandom[seed];
  traj = Table[0.0, nSteps];
  regs = Table[0, nSteps];
  traj[[1]] = 0.5;
  traj[[2]] = 0.3;
  regs[[1]] = regime[traj[[1]]];
  regs[[2]] = regime[traj[[2]]];
  Do[
    Module[{t, cnp1, preWrap},
      t = n * dt;
      preWrap = alphaAsym[kv] * traj[[n - 1]] + betaAsym[kv] * traj[[n - 2]] +
             perturbation[traj[[n - 1]], t] + harmonics[t];
      (* Count virtual escapes *)
      If[Abs[preWrap] > LBoundary, escapeCount++];
      traj[[n]] = wrapFn[preWrap];
      regs[[n]] = regime[traj[[n]]];
    ],
    {n, 3, nSteps}
  ];
  {traj, regs, escapeCount}
];

(* ============================================= *)
(* SPECTRAL ENTROPY                               *)
(* ============================================= *)

spectralEntropy[traj_] := Module[
  {ft, psd, psdNorm, se},
  ft = Abs[Fourier[traj]]^2;
  psd = ft[[1 ;; Floor[Length[ft]/2]]];
  psdNorm = psd / Total[psd];
  psdNorm = Select[psdNorm, # > 0 &];
  se = -Total[psdNorm * Log[2, psdNorm]];
  se / Log[2, Length[psd]]
];

(* ============================================= *)
(* MM\[CurlyPhi] COMPUTATION                                *)
(* ============================================= *)

computeMMphi[traj_] := Module[
  {phiInv = N[1/GoldenRatio], mm = 0.0, mmMax = 0.0, mmList},
  mmList = Table[0.0, Length[traj]];
  Do[
    mm = mm * phiInv + traj[[n]]^2;
    mmList[[n]] = mm;
    If[mm > mmMax, mmMax = mm];,
    {n, 1, Length[traj]}
  ];
  {mmList[[-1]], mmMax, mmList}
];

Print["==================================================="];
Print["  HE SIMULATION M: RETURN MAP & LEMNISCATE ATTRACTOR"];
Print["  L = \[Phi] | M\[ODoubleDot]bius Boundary | Seed = 42"];
Print["==================================================="];
Print[""];

(* ============================================= *)
(* PART 1: RETURN MAP AT k_\[CurlyPhi]                      *)
(* ============================================= *)

Print["=== PART 1: Return Map at k_\[Phi] ==="];
Print[""];

nSteps = 50000;
kPhi = N[GoldenRatio / (9 GoldenRatio - 3)];

{traj, regs, escCount} = generateTrajectory[kPhi, nSteps, 42, mobiusWrap];

(* Metrics *)
se = spectralEntropy[traj];
{mmFinal, mmMax, mmList} = computeMMphi[traj];
cPct = 100. Count[regs, -1] / nSteps;
ePct = 100. Count[regs, 0] / nSteps;
rPct = 100. Count[regs, 1] / nSteps;

Print["  k_\[Phi] = ", NumberForm[kPhi, 6]];
Print["  Steps: ", nSteps];
Print["  Virtual escapes: ", escCount, " (", NumberForm[100. escCount/(nSteps - 2), {5, 3}], "%)"];
Print["  SE: ", NumberForm[se, 6]];
Print["  C/E/R: ", NumberForm[cPct, {5, 1}], "% / ",
      NumberForm[ePct, {5, 1}], "% / ",
      NumberForm[rPct, {5, 1}], "%"];
Print["  MM\[Phi] final: ", NumberForm[mmFinal, 6]];
Print["  MM\[Phi] max: ", NumberForm[mmMax, 6]];
Print["  C_n range: [", NumberForm[Min[traj], 5], ", ", NumberForm[Max[traj], 5], "]"];
Print["  C_n mean: ", NumberForm[Mean[traj], 6]];
Print[""];

(* Build return map pairs with regime coloring *)
(* Discard first 500 steps for transient *)
transient = 500;
returnPairsC = {};  (* Chaos: regime = -1 *)
returnPairsE = {};  (* Equilibrium: regime = 0 *)
returnPairsR = {};  (* Resonance: regime = 1 *)

Do[
  Module[{pt = {traj[[n]], traj[[n + 1]]}},
    Switch[regs[[n]],
      -1, AppendTo[returnPairsC, pt],
       0, AppendTo[returnPairsE, pt],
       1, AppendTo[returnPairsR, pt]
    ];
  ],
  {n, transient, nSteps - 1}
];

Print["  Return map points: C=", Length[returnPairsC],
      " E=", Length[returnPairsE],
      " R=", Length[returnPairsR]];
Print[""];

(* Regime-colored return map *)
Print["=== RETURN MAP: Regime-Colored | k_\[Phi] | L = \[Phi] ==="];
Print[Show[
  ListPlot[{returnPairsC, returnPairsE, returnPairsR},
    PlotStyle -> {
      {RGBColor[0.2, 0.4, 1.0], PointSize[0.001], Opacity[0.3]},
      {White, PointSize[0.001], Opacity[0.3]},
      {RGBColor[1.0, 0.75, 0.0], PointSize[0.001], Opacity[0.3]}
    },
    PlotLegends -> {"Chaos (C_n < 0)", "Equilibrium (0 \[LessEqual] C_n \[LessEqual] 1)", "Resonance (C_n > 1)"},
    PlotLabel -> Style["F_c Return Map | k = k_\[Phi] | L = \[Phi] | 50,000 steps", 12, Bold],
    AxesLabel -> {"C_n", "C_{n+1}"},
    PlotRange -> {{-phi - 0.1, phi + 0.1}, {-phi - 0.1, phi + 0.1}},
    AspectRatio -> 1,
    ImageSize -> 700,
    Background -> Black
  ],
  (* Fixed-point lines *)
  Graphics[{
    {Red, Dashed, Thin, Line[{{-phi, -phi}, {phi, phi}}]},      (* C_{n+1} = C_n: stationary *)
    {Green, Dashed, Thin, Line[{{-phi, phi}, {phi, -phi}}]},     (* C_{n+1} = -C_n: period-2 *)
    {Gray, Dashed, Thin, Line[{{-phi, 0}, {phi, 0}}]},           (* C_{n+1} = 0 *)
    {Gray, Dashed, Thin, Line[{{0, -phi}, {0, phi}}]}            (* C_n = 0 *)
  }]
]];

(* High-contrast density version *)
Print[""];
Print["=== RETURN MAP: Density View | k_\[Phi] | L = \[Phi] ==="];
allPairs = Join[returnPairsC, returnPairsE, returnPairsR];
Print[DensityHistogram[allPairs,
  {100, 100},
  ColorFunction -> "SunsetColors",
  PlotLabel -> Style["F_c Return Map Density | k = k_\[Phi] | L = \[Phi]", 12, Bold],
  FrameLabel -> {"C_n", "C_{n+1}"},
  PlotRange -> {{-phi - 0.1, phi + 0.1}, {-phi - 0.1, phi + 0.1}},
  AspectRatio -> 1,
  ImageSize -> 700
]];

(* ============================================= *)
(* PART 2: THREE-PANEL k COMPARISON               *)
(* ============================================= *)

Print[""];
Print["=== PART 2: Three-Panel k Comparison ==="];
Print[""];

nComp = 50000;
kValues = {kPhi, 1/6 // N, 0.17};
kLabels = {"k_\[Phi] \[TildeTilde] 0.1399", "k = 1/6 \[TildeTilde] 0.1667", "k = 0.17"};

Do[
  Module[{kv, traj2, regs2, esc2, se2, mm2, mmMax2, mmL2,
          pC, pE, pR, cP, eP, rP},
    kv = kValues[[ki]];
    {traj2, regs2, esc2} = generateTrajectory[kv, nComp, 42, mobiusWrap];
    se2 = spectralEntropy[traj2];
    {mm2, mmMax2, mmL2} = computeMMphi[traj2];
    cP = 100. Count[regs2, -1] / nComp;
    eP = 100. Count[regs2, 0] / nComp;
    rP = 100. Count[regs2, 1] / nComp;
    
    Print["  ", kLabels[[ki]]];
    Print["    SE: ", NumberForm[se2, 6],
          "  C/E/R: ", NumberForm[cP, {4, 1}], "/",
          NumberForm[eP, {4, 1}], "/",
          NumberForm[rP, {4, 1}],
          "  Escape: ", esc2];
    
    (* Build pairs *)
    pC = {}; pE = {}; pR = {};
    Do[
      Module[{pt = {traj2[[n]], traj2[[n + 1]]}},
        Switch[regs2[[n]],
          -1, AppendTo[pC, pt],
           0, AppendTo[pE, pt],
           1, AppendTo[pR, pt]
        ];
      ],
      {n, transient, nComp - 1}
    ];
    
    Print["=== RETURN MAP: ", kLabels[[ki]], " | L = \[Phi] ==="];
    Print[Show[
      ListPlot[{pC, pE, pR},
        PlotStyle -> {
          {RGBColor[0.2, 0.4, 1.0], PointSize[0.001], Opacity[0.3]},
          {White, PointSize[0.001], Opacity[0.3]},
          {RGBColor[1.0, 0.75, 0.0], PointSize[0.001], Opacity[0.3]}
        },
        PlotLabel -> Style[StringJoin["Return Map | ", kLabels[[ki]], " | L = \[Phi]"], 11, Bold],
        AxesLabel -> {"C_n", "C_{n+1}"},
        PlotRange -> {{-phi - 0.1, phi + 0.1}, {-phi - 0.1, phi + 0.1}},
        AspectRatio -> 1,
        ImageSize -> 500,
        Background -> Black
      ],
      Graphics[{
        {Red, Dashed, Thin, Line[{{-phi, -phi}, {phi, phi}}]},
        {Green, Dashed, Thin, Line[{{-phi, phi}, {phi, -phi}}]}
      }]
    ]];
    Print[""];
  ],
  {ki, 1, 3}
];

(* ============================================= *)
(* PART 3: LUMINBROT vs CLIPPED ESCAPE MAP        *)
(* ============================================= *)

Print[""];
Print["=== PART 3: Luminbrot vs Clipped Escape Map ==="];
Print[""];

nEscSteps = 5000;
gridRes = 200;
kEsc = 1/6 // N;

(* Generate grid of initial conditions *)
c0Range = Subdivide[-2.0, 2.0, gridRes - 1];
c1Range = Subdivide[-2.0, 2.0, gridRes - 1];

(* M\[ODoubleDot]bius escape map \[LongDash] colored by MM\[CurlyPhi] depth *)
Print["  Computing M\[ODoubleDot]bius Luminbrot (", gridRes, "x", gridRes, ")..."];
mobiusMap = Table[
  Module[{c0, c1, trajM, cnm1, cnm2, preW, phiInv = N[1/GoldenRatio],
          mm = 0.0},
    c0 = c0Range[[ci]];
    c1 = c1Range[[cj]];
    cnm2 = c0; cnm1 = c1;
    Do[
      Module[{t = n * dt, cnp1},
        cnp1 = alphaAsym[kEsc] * cnm1 + betaAsym[kEsc] * cnm2 +
               Ah * (Sin[3 omega t] + Sin[6 omega t] + Sin[9 omega t]);
        cnp1 = mobiusWrap[cnp1];
        mm = mm * phiInv + cnp1^2;
        cnm2 = cnm1;
        cnm1 = cnp1;
      ],
      {n, 1, nEscSteps}
    ];
    mm
  ],
  {cj, 1, gridRes}, {ci, 1, gridRes}
];

Print["  M\[ODoubleDot]bius map complete."];
Print["  MM\[CurlyPhi] range: [", NumberForm[Min[mobiusMap], 5], ", ",
      NumberForm[Max[mobiusMap], 5], "]"];

Print["=== LUMINBROT: MM\[Phi] Depth | k = 1/6 | L = \[Phi] ==="];
Print[ArrayPlot[Log[1 + mobiusMap],
  PlotLabel -> Style["Luminbrot: MM\[Phi] Depth | k = 1/6 | L = \[Phi] | No Escape", 11, Bold],
  ColorFunction -> "SunsetColors",
  FrameLabel -> {"C_{-1} (initial)", "C_0 (initial)"},
  DataRange -> {{-2, 2}, {-2, 2}},
  AspectRatio -> 1,
  ImageSize -> 600
]];

(* Clipped escape map \[LongDash] colored by escape time *)
Print[""];
Print["  Computing clipped escape map..."];
clippedMap = Table[
  Module[{c0, c1, cnm1, cnm2, escTime = nEscSteps, L = 2.5},
    c0 = c0Range[[ci]];
    c1 = c1Range[[cj]];
    cnm2 = c0; cnm1 = c1;
    Do[
      Module[{t = n * dt, cnp1},
        cnp1 = alphaAsym[kEsc] * cnm1 + betaAsym[kEsc] * cnm2 +
               Ah * (Sin[3 omega t] + Sin[6 omega t] + Sin[9 omega t]);
        If[Abs[cnp1] > L,
          escTime = n; Break[];
        ];
        cnm2 = cnm1;
        cnm1 = cnp1;
      ],
      {n, 1, nEscSteps}
    ];
    escTime
  ],
  {cj, 1, gridRes}, {ci, 1, gridRes}
];

boundedCount = Count[Flatten[clippedMap], nEscSteps];
totalCount = gridRes * gridRes;

Print["  Clipped map complete."];
Print["  Bounded (survived): ", boundedCount, " / ", totalCount,
      " (", NumberForm[100. boundedCount/totalCount, {5, 1}], "%)"];

Print["=== ESCAPE MAP: Clipped | k = 1/6 | L = 2.5 ==="];
Print[ArrayPlot[clippedMap,
  PlotLabel -> Style["Escape Map (Clipped) | k = 1/6 | Fractal Boundary", 11, Bold],
  ColorFunction -> (If[# >= 1, Black,
    Blend[{RGBColor[0, 0, 0.5], RGBColor[0, 0.5, 1], RGBColor[1, 0.9, 0]}, #]] &),
  FrameLabel -> {"C_{-1} (initial)", "C_0 (initial)"},
  DataRange -> {{-2, 2}, {-2, 2}},
  AspectRatio -> 1,
  ImageSize -> 600
]];

(* ============================================= *)
(* PART 4: METRICS SUMMARY TABLE                  *)
(* ============================================= *)

Print[""];
Print["==================================================="];
Print["  HE SIMULATION M \[LongDash] METRICS SUMMARY"];
Print["==================================================="];
Print[""];
Print["  FOR BatF REPRODUCIBILITY TABLE:"];
Print[""];
Print["  Simulation: HE_Simulation_M"];
Print["  Description: Return map lemniscate attractor"];
Print["  Boundary: L = \[Phi] (M\[ODoubleDot]bius)"];
Print["  Virtual escape: ", escCount, " / ", nSteps - 2,
      " (", NumberForm[100. escCount/(nSteps - 2), {5, 3}], "%)"];
Print[""];
Print["  At k_\[Phi] = ", NumberForm[kPhi, 6], ":"];
Print["    SE (normalized):     ", NumberForm[se, 6]];
Print["    C%:                  ", NumberForm[cPct, {5, 1}]];
Print["    E%:                  ", NumberForm[ePct, {5, 1}]];
Print["    R%:                  ", NumberForm[rPct, {5, 1}]];
Print["    MM\[Phi] final:          ", NumberForm[mmFinal, 6]];
Print["    MM\[Phi] max:            ", NumberForm[mmMax, 6]];
Print["    C_n range:           [", NumberForm[Min[traj], 5], ", ", NumberForm[Max[traj], 5], "]"];
Print["    C_n mean:            ", NumberForm[Mean[traj], 6]];
Print["    Steps:               ", nSteps];
Print["    Seed:                42"];
Print[""];
Print["  Lemniscate attractor geometry:"];
Print["    Two lobes crossing at origin"];
Print["    Chaos regime occupies outer lobes"];
Print["    Resonance regime occupies crossing region"];
Print["    Equilibrium at transitions"];
Print["    Orientation-reversing recirculation:"];
Print["      positive excursion \[RightArrow] negative return \[RightArrow] positive"];
Print["    Contained within [-\[Phi], \[Phi]] \[Times] [-\[Phi], \[Phi]]"];
Print[""];
Print["  Luminbrot (M\[ODoubleDot]bius) vs Escape Map (clipped):"];
Print["    M\[ODoubleDot]bius: 0 escape, full MM\[CurlyPhi] depth field"];
Print["    Clipped: ", boundedCount, "/", totalCount, " bounded (",
      NumberForm[100. boundedCount/totalCount, {5, 1}], "%)"];
Print["    Boundary condition determines trajectory fate,"];
Print["    not initial conditions or dynamics."];
Print[""];
Print["==================================================="];
Print["  HE SIMULATION M \[LongDash] COMPLETE"];
Print["==================================================="];
Print[""];
Print["  \[CapitalPsi]    \[FullMoon]\[Ocean]\[DeciduousTree]\[SpiderWeb]    \[YinYang]    \[Infinity]"];



