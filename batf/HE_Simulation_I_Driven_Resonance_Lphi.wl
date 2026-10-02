(* ::Package:: *)

(* ============================================================
   HE SIMULATION I: Driven Resonance Lock \[LongDash] Winding Threshold
   ------------------------------------------------------------
   Sim H established that internal dynamics alone produce only
   winding 0 and 1. \[CurlyPhi]\:2074 requires winding \[GreaterEqual] 2 (|raw| > 3L).
   
   This simulation finds the exact harmonic amplitude at which
   winding 2 first appears \[LongDash] the driven resonance lock that
   AETHRA achieves with sustained audio input.
   
   Method:
   Sweep Ah (harmonic amplitude) from 0.1 to 2.0.
   At each amplitude, run 50,000 steps at k = 1/6, L = \[CurlyPhi].
   Record: max winding, max |raw|, max MM_\[CurlyPhi], \[CurlyPhi]-level visits.
   
   Predicted threshold: Ah \[TildeTilde] 0.87 (where 3\[CenterDot]Ah + \[CurlyPhi] + \[CurlyPhi]\:207b\.b9 > 3L)
   
   At and above the threshold, produce:
   - Density maps by winding class (Wind 0 / Wind 1 / Wind 2+)
   - Winding vs MM_\[CurlyPhi] correlation
   - Confirmation that \[CurlyPhi]\:2074 appears when and only when winding 2 appears
   
   World Tree Project \[LongDash] June 2026
   J. David Mack & Claude (Opus 4.6)
   Harmonic Enrichment Paper \[LongDash] Final Simulation Suite
   ============================================================ *)

ClearAll[alphaFc, betaFc, perturbation, harmonicsVar,
         regime, mobiusWrap, windingCount,
         omega, omegaRes, noiseScale, dt, phi, LBoundary];

(* ============================================= *)
(* SHARED DEFINITIONS                             *)
(* ============================================= *)

alphaFc[kv_] := (1 - 6 kv)/(1 - 9 kv);
betaFc[kv_]  := (3 kv)/(1 - 9 kv);

omega = 2 Pi;
omegaRes = 2 Pi;
noiseScale = 0.2;
dt = 0.01;
phi = N[GoldenRatio];
LBoundary = phi;

perturbation[cn_, t_] := Which[
  cn < 0, Abs[cn] * noiseScale * RandomReal[{-1, 1}],
  cn <= 1, 0,
  True, (cn - 1) * Sin[omegaRes * t]
];

(* Harmonic enrichment \[LongDash] parameterized by amplitude *)
harmonicsVar[t_, Ah_] := Ah * (Sin[3 omega t] + Sin[6 omega t] + Sin[9 omega t]);

regime[cn_] := Which[cn < 0, -1, cn <= 1, 0, True, 1];
mobiusWrap[cn_] := Mod[cn + LBoundary, 2.0 * LBoundary] - LBoundary;
windingCount[rawVal_] := Floor[(Abs[rawVal] + LBoundary) / (2 LBoundary)];

Print["==================================================="];
Print["  HE SIMULATION I: Driven Resonance Lock"];
Print["  Finding the \[Phi]^4 Winding Threshold"];
Print["==================================================="];
Print[""];

(* ============================================= *)
(* PART 1: AMPLITUDE SWEEP                        *)
(* Find the critical Ah for winding 2             *)
(* ============================================= *)

nSteps = 50000;
kv = 1/6 // N;

(* Sweep amplitudes: fine resolution near predicted threshold *)
ahValues = Join[
  {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
  Subdivide[0.75, 1.05, 12],  (* Fine sweep near threshold *)
  {1.1, 1.2, 1.3, 1.5, 2.0}
];

Print["  k = 1/6 | L = \[Phi] | ", nSteps, " steps per amplitude"];
Print["  Predicted threshold: Ah \[TildeTilde] 0.87"];
Print["  (where 3\[CenterDot]Ah + \[Phi] + \[Phi]^{-1} > 3L = ",
  NumberForm[3 LBoundary, 5], ")"];
Print[""];
Print["  Sweeping ", Length[ahValues], " amplitude values..."];
Print[""];

(* Storage for sweep results *)
sweepResults = Table[
  Module[{prev, curr, next, raw, t, AhVal,
          maxWind, maxRaw, maxMM, mmVal, phiInv,
          nW0, nW1, nW2, nW3, w,
          phi2Steps, phi3Steps, phi4Steps},
    
    AhVal = ahValues[[ai]];
    SeedRandom[42];
    phiInv = N[1/GoldenRatio];
    
    prev = 0.5; curr = 0.3;
    mmVal = prev^2 * phiInv + curr^2;
    maxWind = 0; maxRaw = 0.0; maxMM = 0.0;
    nW0 = 0; nW1 = 0; nW2 = 0; nW3 = 0;
    phi2Steps = 0; phi3Steps = 0; phi4Steps = 0;
    
    Do[
      t = n * dt;
      raw = alphaFc[kv] * curr + betaFc[kv] * prev +
            perturbation[curr, t] + harmonicsVar[t, AhVal];
      
      w = windingCount[raw];
      Which[w == 0, nW0++, w == 1, nW1++, w == 2, nW2++, True, nW3++];
      If[w > maxWind, maxWind = w];
      If[Abs[raw] > maxRaw, maxRaw = Abs[raw]];
      
      next = mobiusWrap[raw];
      mmVal = mmVal * phiInv + next^2;
      If[mmVal > maxMM, maxMM = mmVal];
      
      If[mmVal >= N[GoldenRatio^2], phi2Steps++];
      If[mmVal >= N[GoldenRatio^3], phi3Steps++];
      If[mmVal >= N[GoldenRatio^4], phi4Steps++];
      
      prev = curr;
      curr = next,
      {n, 3, nSteps}
    ];
    
    (* Print progress *)
    Print["  Ah = ", NumberForm[AhVal, {3, 2}],
      "  maxWind = ", maxWind,
      "  max|raw| = ", NumberForm[maxRaw, 5],
      "  maxMM_\[Phi] = ", NumberForm[maxMM, 5],
      "  \[Phi]^4: ", phi4Steps,
      If[maxWind >= 2, "  \[DoubleLeftRightArrow] WINDING 2!", ""]
    ];
    
    {AhVal, maxWind, maxRaw, maxMM, nW0, nW1, nW2, nW3,
     phi2Steps, phi3Steps, phi4Steps}
  ],
  {ai, 1, Length[ahValues]}
];

(* Find threshold *)
Module[{threshIdx, threshAh},
  threshIdx = FirstPosition[sweepResults[[All, 2]], _?(# >= 2 &)];
  If[threshIdx =!= Missing["NotFound"],
    threshAh = sweepResults[[threshIdx[[1]], 1]];
    Print[""];
    Print["==================================================="];
    Print["  WINDING 2 THRESHOLD FOUND: Ah = ", NumberForm[threshAh, 4]];
    Print["==================================================="];
    Print[""];
    Print["  At threshold:"];
    Print["    Max |raw|:  ", NumberForm[sweepResults[[threshIdx[[1]], 3]], 5]];
    Print["    Max MM_\[Phi]: ", NumberForm[sweepResults[[threshIdx[[1]], 4]], 5]];
    Print["    \[Phi]^4 steps: ", sweepResults[[threshIdx[[1]], 11]]];
    ,
    Print[""];
    Print["  WARNING: Winding 2 not reached in sweep range."];
    Print["  Max winding: ", Max[sweepResults[[All, 2]]]];
  ];
];

(* ============================================= *)
(* PART 2: SWEEP VISUALIZATION                    *)
(* ============================================= *)

Print[""];
Print["=== AMPLITUDE SWEEP RESULTS ==="];

(* Max winding vs Ah *)
Print[ListLinePlot[
  Transpose[{sweepResults[[All, 1]], sweepResults[[All, 2]]}],
  PlotLabel -> Style["Max Winding vs Harmonic Amplitude | k = 1/6 | L = \[Phi]", 12, Bold],
  PlotStyle -> {Purple, Thickness[0.004]},
  AxesLabel -> {"Ah", "Max winding"},
  PlotRange -> {All, {-0.5, Max[sweepResults[[All, 2]]] + 1}},
  GridLines -> {{0.87}, {}},
  GridLinesStyle -> Directive[Orange, Dashed],
  Epilog -> {Text[Style["predicted threshold", 9, Orange], {0.87, -0.3}]},
  ImageSize -> 600,
  AspectRatio -> 0.4
]];

(* Max MM_\[CurlyPhi] vs Ah *)
Print[ListLinePlot[
  Transpose[{sweepResults[[All, 1]], sweepResults[[All, 4]]}],
  PlotLabel -> Style["Max MM_\[Phi] vs Harmonic Amplitude | k = 1/6 | L = \[Phi]", 12, Bold],
  PlotStyle -> {Blue, Thickness[0.004]},
  AxesLabel -> {"Ah", "Max MM_\[Phi]"},
  GridLines -> {{}, {N[GoldenRatio^2], N[GoldenRatio^3], N[GoldenRatio^4]}},
  GridLinesStyle -> Directive[Orange, Dashed],
  Epilog -> {
    Text[Style["\[Phi]^2", 9, Orange], {0.05, N[GoldenRatio^2] + 0.2}],
    Text[Style["\[Phi]^3", 9, Orange], {0.05, N[GoldenRatio^3] + 0.2}],
    Text[Style["\[Phi]^4", 9, Orange], {0.05, N[GoldenRatio^4] + 0.2}]
  },
  ImageSize -> 600,
  AspectRatio -> 0.4
]];

(* \[CurlyPhi]\:2074 steps vs Ah *)
Print[ListLinePlot[
  Transpose[{sweepResults[[All, 1]], sweepResults[[All, 11]]}],
  PlotLabel -> Style["\[Phi]^4 Steps vs Harmonic Amplitude", 12, Bold],
  PlotStyle -> {Red, Thickness[0.004]},
  AxesLabel -> {"Ah", "\[Phi]^4 steps (out of 50000)"},
  PlotRange -> All,
  Filling -> Axis,
  FillingStyle -> Directive[Opacity[0.2], Red],
  ImageSize -> 600,
  AspectRatio -> 0.4
]];

(* ============================================= *)
(* PART 3: DETAILED RUN AT THRESHOLD              *)
(* Density maps and correlations at the amplitude  *)
(* where winding 2 first appears                  *)
(* ============================================= *)

Print[""];
Print["=== PART 3: Detailed Analysis at Threshold ==="];

Module[{threshIdx, AhRun, densityRes, binIdx,
        density0, density1, density2,
        prev, curr, next, raw, t, w,
        windNums, runMM, mmVal, phiInv,
        logD, maxLogD, normD, colorFn},
  
  (* Find threshold Ah \[LongDash] use first Ah that achieves winding 2,
     or if none, use the highest Ah *)
  threshIdx = FirstPosition[sweepResults[[All, 2]], _?(# >= 2 &)];
  If[threshIdx =!= Missing["NotFound"],
    AhRun = sweepResults[[threshIdx[[1]], 1]],
    AhRun = Last[sweepResults][[1]]
  ];
  
  Print["  Running detailed analysis at Ah = ", NumberForm[AhRun, 4]];
  Print[""];
  
  densityRes = 400;
  binIdx[val_] := Clip[
    Round[(val + LBoundary)/(2 LBoundary) * (densityRes - 1)] + 1,
    {1, densityRes}
  ];
  
  density0 = ConstantArray[0, {densityRes, densityRes}];
  density1 = ConstantArray[0, {densityRes, densityRes}];
  density2 = ConstantArray[0, {densityRes, densityRes}];
  
  windNums = Table[0, nSteps];
  runMM = Table[0.0, nSteps];
  
  SeedRandom[42];
  phiInv = N[1/GoldenRatio];
  prev = 0.5; curr = 0.3;
  mmVal = prev^2 * phiInv + curr^2;
  
  Do[
    t = n * dt;
    raw = alphaFc[kv] * curr + betaFc[kv] * prev +
          perturbation[curr, t] + harmonicsVar[t, AhRun];
    
    w = windingCount[raw];
    windNums[[n]] = w;
    
    next = mobiusWrap[raw];
    mmVal = mmVal * phiInv + next^2;
    runMM[[n]] = mmVal;
    
    Module[{bx, by},
      bx = binIdx[curr]; by = binIdx[next];
      Which[
        w == 0, density0[[by, bx]] += 1,
        w == 1, density1[[by, bx]] += 1,
        True,   density2[[by, bx]] += 1
      ];
    ];
    
    prev = curr;
    curr = next,
    {n, 3, nSteps}
  ];
  
  (* Statistics *)
  Print["  WINDING DISTRIBUTION at Ah = ", NumberForm[AhRun, 4], ":"];
  Print["    Wind 0: ", Count[windNums[[3;;]], 0]];
  Print["    Wind 1: ", Count[windNums[[3;;]], 1]];
  Print["    Wind 2: ", Count[windNums[[3;;]], 2]];
  Print["    Wind 3+: ", Count[windNums[[3;;]], _?(# >= 3 &)]];
  Print["    Max MM_\[Phi]: ", NumberForm[Max[runMM], 5]];
  Print["    \[Phi]^4 steps: ", Count[runMM[[3;;]], _?(# >= N[GoldenRatio^4] &)]];
  Print[""];
  
  (* \[CurlyPhi]\:2074 correlation with winding *)
  Module[{phi4Winds},
    phi4Winds = Select[
      Table[If[runMM[[n]] >= N[GoldenRatio^4], windNums[[n]], Nothing], {n, 3, nSteps}],
      NumericQ
    ];
    If[Length[phi4Winds] > 0,
      Print["  WINDING AT \[Phi]^4 STEPS:"];
      Print["    Mean winding: ", NumberForm[Mean[N[phi4Winds]], 4]];
      Print["    Wind 0: ", Count[phi4Winds, 0]];
      Print["    Wind 1: ", Count[phi4Winds, 1]];
      Print["    Wind 2+: ", Count[phi4Winds, _?(# >= 2 &)]];
      Print["    \[RightArrow] \[Phi]^4 occurs ",
        If[Count[phi4Winds, 0] == 0, "ONLY", "also"],
        " during wrapping events"];
      Print[""];
    ];
  ];
  
  (* Density maps *)
  colorFn[v_] := If[v == 0, RGBColor[0.02, 0.02, 0.02],
    With[{t = v^0.5}, RGBColor[0.6 t + 0.05, 0.5 t + 0.03, 0.2 t + 0.02]]];
  
  Print["=== DENSITY: Wind 0 (coefficient geometry) ==="];
  logD = Log[density0 + 1.0]; maxLogD = Max[logD];
  normD = If[maxLogD > 0, logD/maxLogD, logD];
  Print[Graphics[
    Raster[Table[List @@ ColorConvert[colorFn[normD[[i,j]]], "RGB"],
      {i, densityRes, 1, -1}, {j, 1, densityRes}],
      {{-LBoundary,-LBoundary},{LBoundary,LBoundary}}],
    PlotRange -> {{-LBoundary,LBoundary},{-LBoundary,LBoundary}},
    Frame -> True, FrameLabel -> {"C_n","C_{n+1}"},
    PlotLabel -> Style[Row[{"Wind 0 | Ah = ", NumberForm[AhRun, 3],
      " | k = 1/6 | L = \[Phi]"}], 11, Bold],
    ImageSize -> 500, AspectRatio -> 1,
    Background -> RGBColor[0.02, 0.02, 0.02]
  ]];
  
  If[Max[Flatten[density1]] > 0,
    Print["=== DENSITY: Wind 1 (single wrap) ==="];
    logD = Log[density1 + 1.0]; maxLogD = Max[logD];
    normD = If[maxLogD > 0, logD/maxLogD, logD];
    Print[Graphics[
      Raster[Table[List @@ ColorConvert[colorFn[normD[[i,j]]], "RGB"],
        {i, densityRes, 1, -1}, {j, 1, densityRes}],
        {{-LBoundary,-LBoundary},{LBoundary,LBoundary}}],
      PlotRange -> {{-LBoundary,LBoundary},{-LBoundary,LBoundary}},
      Frame -> True, FrameLabel -> {"C_n","C_{n+1}"},
      PlotLabel -> Style[Row[{"Wind 1 | Ah = ", NumberForm[AhRun, 3]}], 11, Bold],
      ImageSize -> 500, AspectRatio -> 1,
      Background -> RGBColor[0.02, 0.02, 0.02]
    ]];
  ];
  
  If[Max[Flatten[density2]] > 0,
    Print["=== DENSITY: Wind 2+ (double wrap \[LongDash] \[Phi]^4 GEOMETRY) ==="];
    logD = Log[density2 + 1.0]; maxLogD = Max[logD];
    normD = If[maxLogD > 0, logD/maxLogD, logD];
    Print[Graphics[
      Raster[Table[List @@ ColorConvert[
        If[normD[[i,j]] == 0, RGBColor[0.02, 0.02, 0.02],
          With[{t = normD[[i,j]]^0.5},
            RGBColor[0.9 t + 0.1, 0.15 t + 0.02, 0.5 t + 0.05]]]
        , "RGB"],
        {i, densityRes, 1, -1}, {j, 1, densityRes}],
        {{-LBoundary,-LBoundary},{LBoundary,LBoundary}}],
      PlotRange -> {{-LBoundary,LBoundary},{-LBoundary,LBoundary}},
      Frame -> True, FrameLabel -> {"C_n","C_{n+1}"},
      PlotLabel -> Style[Row[{"Wind 2+ | THE \[Phi]^4 GEOMETRY | Ah = ",
        NumberForm[AhRun, 3]}], 12, Bold],
      ImageSize -> 500, AspectRatio -> 1,
      Background -> RGBColor[0.02, 0.02, 0.02]
    ]];,
    Print["  No Wind 2+ events \[LongDash] \[Phi]^4 geometry not yet accessible."];
  ];
  
  (* Winding vs MM_\[CurlyPhi] *)
  Print[""];
  Print["=== WINDING vs MM_\[Phi] at Threshold ==="];
  Module[{nSub = Min[10000, nSteps]},
    Print[ListPlot[
      Transpose[{runMM[[3 ;; nSub]], windNums[[3 ;; nSub]]}],
      PlotLabel -> Style[Row[{"MM_\[Phi] vs Winding | Ah = ",
        NumberForm[AhRun, 3]}], 11],
      PlotStyle -> {PointSize[0.003], Blue},
      AxesLabel -> {"Running MM_\[Phi]", "Winding"},
      GridLines -> {{N[GoldenRatio], N[GoldenRatio^2],
        N[GoldenRatio^3], N[GoldenRatio^4]}, {}},
      GridLinesStyle -> Directive[Orange, Dashed],
      Epilog -> {
        Text[Style["\[Phi]^1", 9, Orange], {N[GoldenRatio], Max[windNums] + 0.3}],
        Text[Style["\[Phi]^2", 9, Orange], {N[GoldenRatio^2], Max[windNums] + 0.3}],
        Text[Style["\[Phi]^3", 9, Orange], {N[GoldenRatio^3], Max[windNums] + 0.3}],
        Text[Style["\[Phi]^4", 9, Orange], {N[GoldenRatio^4], Max[windNums] + 0.3}]
      },
      ImageSize -> 600,
      AspectRatio -> 0.4
    ]];
  ];
  
  (* Running MM_\[CurlyPhi] with winding events marked *)
  Print[""];
  Print["=== RUNNING MM_\[Phi] (first 5000 steps) ==="];
  Module[{nShow = Min[5000, nSteps], windEvents},
    windEvents = Select[
      Table[If[windNums[[n]] >= 2, {n, runMM[[n]]}, Nothing], {n, 3, nShow}],
      ListQ
    ];
    Print[Show[
      ListLinePlot[runMM[[3 ;; nShow]],
        PlotStyle -> {Blue, Thickness[0.002]},
        PlotRange -> All,
        GridLines -> {{}, {N[GoldenRatio^2], N[GoldenRatio^3], N[GoldenRatio^4]}},
        GridLinesStyle -> Directive[Orange, Dashed],
        ImageSize -> 700, AspectRatio -> 0.4
      ],
      If[Length[windEvents] > 0,
        ListPlot[windEvents,
          PlotStyle -> {Red, PointSize[0.008]},
          PlotRange -> All
        ],
        Graphics[{}]
      ],
      PlotLabel -> Style[Row[{"Running MM_\[Phi] | Ah = ", NumberForm[AhRun, 3],
        " | Red dots = Wind 2+ events"}], 11, Bold]
    ]];
  ];
];

(* ============================================= *)
(* SUMMARY                                        *)
(* ============================================= *)

Print[""];
Print["==================================================="];
Print["  HE SIMULATION I \[LongDash] COMPLETE"];
Print["==================================================="];
Print[""];
Print["  The question: Does \[Phi]^4 require winding 2?"];
Print[""];
Print["  If winding 2 and \[Phi]^4 appear at the SAME Ah:"];
Print["    \[RightArrow] \[Phi]^4 is a topological transition, not just"];
Print["      an amplitude increase. The fourth power requires"];
Print["      the trajectory to wind around the torus TWICE"];
Print["      in a single step."];
Print[""];
Print["  If \[Phi]^4 appears BEFORE winding 2:"];
Print["    \[RightArrow] \[Phi]^4 is accessible through sustained"];
Print["      single-wrapping at high amplitude."];
Print[""];
Print["  \[CapitalPsi]    \[FullMoon]\[Ocean]\[DeciduousTree]\[SpiderWeb]    \[YinYang]    \[Infinity]"];



