(* ::Package:: *)

(* ============================================================
   HE SIMULATION H: Winding Number Analysis
   ------------------------------------------------------------
   Question: What geometric structure lives in the WRAPPING
   EVENTS themselves \[LongDash] information we've been discarding?
   
   Standard Buddhabrot/density accumulation records WHERE the
   trajectory lands after M\[ODoubleDot]bius wrapping. It discards HOW it
   got there \[LongDash] specifically, how many times the raw pre-wrap
   value exceeded the boundary in a single iteration.
   
   This simulation tracks three quantities per step:
     1. The raw pre-wrap value (before Mod is applied)
     2. The post-wrap value (what we normally record)
     3. The integer winding count: how many full 2L periods
        the raw value spans beyond [-L, L]
   
   If \[CurlyPhi]\:2074 excursions produce a distinct winding signature \[LongDash]
   specific geometry that only appears at high winding counts \[LongDash]
   that reveals what the fourth power actually IS topologically.
   
   Three configurations tested:
     k_\[CurlyPhi] (blade regime), k = 1/6 (boundary), k = 0.165 (torus)
   All at L = \[CurlyPhi].
   
   World Tree Project \[LongDash] June 2026
   J. David Mack & Claude (Opus 4.6)
   Harmonic Enrichment Paper \[LongDash] Final Simulation Suite
   ============================================================ *)

ClearAll[alphaFc, betaFc, perturbation, harmonics, regime,
         omega, omegaRes, Ah, noiseScale, dt, phi, LBoundary];

(* ============================================= *)
(* SHARED DEFINITIONS                             *)
(* ============================================= *)

alphaFc[kv_] := (1 - 6 kv)/(1 - 9 kv);
betaFc[kv_]  := (3 kv)/(1 - 9 kv);

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

(* Winding count: how many full 2L periods the raw value spans *)
windingCount[rawVal_, L_] := Floor[(Abs[rawVal] + L) / (2 L)];

Print["==================================================="];
Print["  HE SIMULATION H: Winding Number Analysis"];
Print["  L = \[Phi] | Tracking wrapping topology"];
Print["==================================================="];
Print[""];

(* ============================================= *)
(* CORE ANALYSIS FUNCTION                         *)
(* Runs one configuration, returns all data       *)
(* ============================================= *)

analyzeWinding[kv_, label_, nSteps_, seed_] := Module[
  {prev, curr, next, raw, t,
   windNums, rawVals, wrappedVals, directions,
   runningMM, mmVal, phiInv,
   nWind0, nWind1, nWind2, nWind3plus,
   density0, density1, density2, density3,
   densityRes, binIdx,
   mmAtWind, windAtMM,
   maxRaw, maxWinding},
  
  SeedRandom[seed];
  phiInv = N[1/GoldenRatio];
  densityRes = 400;
  
  (* Storage *)
  windNums = Table[0, nSteps];
  rawVals = Table[0.0, nSteps];
  wrappedVals = Table[0.0, nSteps];
  directions = Table[0, nSteps];  (* +1 = positive overflow, -1 = negative, 0 = no wrap *)
  runningMM = Table[0.0, nSteps];
  
  (* Density maps by winding class *)
  density0 = ConstantArray[0, {densityRes, densityRes}];
  density1 = ConstantArray[0, {densityRes, densityRes}];
  density2 = ConstantArray[0, {densityRes, densityRes}];
  density3 = ConstantArray[0, {densityRes, densityRes}];
  
  binIdx[val_] := Clip[
    Round[(val + LBoundary)/(2 LBoundary) * (densityRes - 1)] + 1,
    {1, densityRes}
  ];
  
  (* Initial conditions *)
  prev = 0.5;
  curr = 0.3;
  wrappedVals[[1]] = prev;
  wrappedVals[[2]] = curr;
  mmVal = prev^2;
  mmVal = mmVal * phiInv + curr^2;
  runningMM[[1]] = prev^2;
  runningMM[[2]] = mmVal;
  
  (* Main loop *)
  Do[
    t = n * dt;
    raw = alphaFc[kv] * curr + betaFc[kv] * prev +
          perturbation[curr, t] + harmonics[t];
    
    rawVals[[n]] = raw;
    windNums[[n]] = windingCount[raw, LBoundary];
    
    (* Direction of overflow *)
    directions[[n]] = Which[
      raw > LBoundary, 1,
      raw < -LBoundary, -1,
      True, 0
    ];
    
    (* Apply M\[ODoubleDot]bius wrapping *)
    next = Mod[raw + LBoundary, 2.0 * LBoundary] - LBoundary;
    wrappedVals[[n]] = next;
    
    (* Running MM_\[CurlyPhi] *)
    mmVal = mmVal * phiInv + next^2;
    runningMM[[n]] = mmVal;
    
    (* Accumulate into winding-class density maps *)
    (* Using (C_n, C_{n+1}) = (curr, next) *)
    Module[{bx, by, w},
      bx = binIdx[curr];
      by = binIdx[next];
      w = windNums[[n]];
      Which[
        w == 0, density0[[by, bx]] += 1,
        w == 1, density1[[by, bx]] += 1,
        w == 2, density2[[by, bx]] += 1,
        True,   density3[[by, bx]] += 1
      ];
    ];
    
    prev = curr;
    curr = next,
    {n, 3, nSteps}
  ];
  
  (* === STATISTICS === *)
  nWind0 = Count[windNums[[3 ;;]], 0];
  nWind1 = Count[windNums[[3 ;;]], 1];
  nWind2 = Count[windNums[[3 ;;]], 2];
  nWind3plus = Count[windNums[[3 ;;]], _?(# >= 3 &)];
  maxRaw = Max[Abs[rawVals[[3 ;;]]]];
  maxWinding = Max[windNums[[3 ;;]]];
  
  Print[""];
  Print["==================================================="];
  Print["  ", label];
  Print["==================================================="];
  Print[""];
  Print["  k = ", NumberForm[kv, 6]];
  Print["  \[Alpha] = ", NumberForm[alphaFc[kv], 6],
        "  \[Beta] = ", NumberForm[betaFc[kv], 6]];
  Print["  |\[Lambda]|\[Squared] = ", NumberForm[Abs[betaFc[kv]], 6]];
  Print["  Steps: ", nSteps];
  Print[""];
  Print["  WINDING DISTRIBUTION:"];
  Print["    Wind 0 (no wrap):   ", nWind0, " (",
    NumberForm[100.0 nWind0/(nSteps - 2), {5, 2}], "%)"];
  Print["    Wind 1 (single):    ", nWind1, " (",
    NumberForm[100.0 nWind1/(nSteps - 2), {5, 2}], "%)"];
  Print["    Wind 2 (double):    ", nWind2, " (",
    NumberForm[100.0 nWind2/(nSteps - 2), {5, 2}], "%)"];
  Print["    Wind 3+ (triple+):  ", nWind3plus, " (",
    NumberForm[100.0 nWind3plus/(nSteps - 2), {5, 2}], "%)"];
  Print["    Max winding:        ", maxWinding];
  Print["    Max |raw|:          ", NumberForm[maxRaw, 6]];
  Print["    Max |raw|/L:        ", NumberForm[maxRaw/LBoundary, 4]];
  Print[""];
  
  (* Direction analysis *)
  Module[{nPos, nNeg, nNone},
    nPos = Count[directions[[3 ;;]], 1];
    nNeg = Count[directions[[3 ;;]], -1];
    nNone = Count[directions[[3 ;;]], 0];
    Print["  WRAPPING DIRECTION:"];
    Print["    Positive overflow (+): ", nPos, " (",
      NumberForm[100.0 nPos/(nSteps - 2), {4, 1}], "%)"];
    Print["    Negative overflow (-): ", nNeg, " (",
      NumberForm[100.0 nNeg/(nSteps - 2), {4, 1}], "%)"];
    Print["    No wrapping:           ", nNone, " (",
      NumberForm[100.0 nNone/(nSteps - 2), {4, 1}], "%)"];
    If[nPos + nNeg > 0,
      Print["    Direction ratio (+/-): ", NumberForm[N[nPos/(nPos + nNeg)], 4],
        " / ", NumberForm[N[nNeg/(nPos + nNeg)], 4]];
    ];
    Print[""];
  ];
  
  (* MM_\[CurlyPhi] at wrapping events *)
  Module[{mmWind0, mmWind1, mmWind2plus},
    mmWind0 = Select[
      Table[If[windNums[[n]] == 0, runningMM[[n]], Nothing], {n, 3, nSteps}],
      NumericQ
    ];
    mmWind1 = Select[
      Table[If[windNums[[n]] == 1, runningMM[[n]], Nothing], {n, 3, nSteps}],
      NumericQ
    ];
    mmWind2plus = Select[
      Table[If[windNums[[n]] >= 2, runningMM[[n]], Nothing], {n, 3, nSteps}],
      NumericQ
    ];
    
    Print["  MM_\[Phi] AT WRAPPING EVENTS:"];
    If[Length[mmWind0] > 0,
      Print["    Wind 0: mean MM_\[Phi] = ", NumberForm[Mean[mmWind0], 5],
        "  median = ", NumberForm[Median[mmWind0], 5]]];
    If[Length[mmWind1] > 0,
      Print["    Wind 1: mean MM_\[Phi] = ", NumberForm[Mean[mmWind1], 5],
        "  median = ", NumberForm[Median[mmWind1], 5]]];
    If[Length[mmWind2plus] > 0,
      Print["    Wind 2+: mean MM_\[Phi] = ", NumberForm[Mean[mmWind2plus], 5],
        "  median = ", NumberForm[Median[mmWind2plus], 5]]];
    Print[""];
  ];
  
  (* \[CurlyPhi]-level at wrapping events *)
  Module[{windAtPhi1, windAtPhi2, windAtPhi3, windAtPhi4},
    windAtPhi1 = Select[
      Table[If[runningMM[[n]] >= N[GoldenRatio] && runningMM[[n]] < N[GoldenRatio^2],
        windNums[[n]], Nothing], {n, 3, nSteps}], NumericQ];
    windAtPhi2 = Select[
      Table[If[runningMM[[n]] >= N[GoldenRatio^2] && runningMM[[n]] < N[GoldenRatio^3],
        windNums[[n]], Nothing], {n, 3, nSteps}], NumericQ];
    windAtPhi3 = Select[
      Table[If[runningMM[[n]] >= N[GoldenRatio^3] && runningMM[[n]] < N[GoldenRatio^4],
        windNums[[n]], Nothing], {n, 3, nSteps}], NumericQ];
    windAtPhi4 = Select[
      Table[If[runningMM[[n]] >= N[GoldenRatio^4],
        windNums[[n]], Nothing], {n, 3, nSteps}], NumericQ];
    
    Print["  WINDING AT EACH \[Phi]-LEVEL:"];
    If[Length[windAtPhi1] > 0,
      Print["    \[Phi]^1: mean winding = ", NumberForm[Mean[N[windAtPhi1]], 4],
        "  max = ", Max[windAtPhi1], "  (", Length[windAtPhi1], " steps)"]];
    If[Length[windAtPhi2] > 0,
      Print["    \[Phi]^2: mean winding = ", NumberForm[Mean[N[windAtPhi2]], 4],
        "  max = ", Max[windAtPhi2], "  (", Length[windAtPhi2], " steps)"]];
    If[Length[windAtPhi3] > 0,
      Print["    \[Phi]^3: mean winding = ", NumberForm[Mean[N[windAtPhi3]], 4],
        "  max = ", Max[windAtPhi3], "  (", Length[windAtPhi3], " steps)"]];
    If[Length[windAtPhi4] > 0,
      Print["    \[Phi]^4: mean winding = ", NumberForm[Mean[N[windAtPhi4]], 4],
        "  max = ", Max[windAtPhi4], "  (", Length[windAtPhi4], " steps)"],
      Print["    \[Phi]^4: not reached"]
    ];
    Print[""];
  ];
  
  (* === VISUALIZATIONS === *)
  
  (* 1. Winding count over time *)
  Print["=== WINDING COUNT OVER TIME ==="];
  Print[ListLinePlot[windNums[[3 ;; Min[5000, nSteps]]],
    PlotLabel -> Style[Row[{label, " | Winding per step"}], 11],
    PlotStyle -> {Purple, Thickness[0.001]},
    AxesLabel -> {"Iteration", "Winding count"},
    PlotRange -> {-0.5, Max[maxWinding, 3] + 0.5},
    Filling -> Axis,
    FillingStyle -> Directive[Opacity[0.3], Purple],
    ImageSize -> 700,
    AspectRatio -> 0.3
  ]];
  
  (* 2. Raw value vs wrapped value scatter *)
  Print[""];
  Print["=== RAW vs WRAPPED VALUES ==="];
  Module[{pairs, colors},
    pairs = Table[{rawVals[[n]], wrappedVals[[n]]}, {n, 3, Min[20000, nSteps]}];
    colors = Table[
      Which[
        windNums[[n]] == 0, RGBColor[0.2, 0.4, 0.8],
        windNums[[n]] == 1, RGBColor[0.8, 0.6, 0.1],
        windNums[[n]] == 2, RGBColor[0.9, 0.2, 0.2],
        True, RGBColor[1, 0, 0.5]
      ],
      {n, 3, Min[20000, nSteps]}
    ];
    Print[Graphics[{
      PointSize[0.002],
      MapThread[{#2, Point[#1]} &, {pairs, colors}],
      (* Reference lines *)
      {Gray, Dashed, Line[{{-LBoundary, -LBoundary}, {LBoundary, LBoundary}}]},
      {Orange, Dashed, 
        Line[{{-LBoundary, -10}, {-LBoundary, 10}}],
        Line[{{LBoundary, -10}, {LBoundary, 10}}]}
      },
      Frame -> True,
      FrameLabel -> {"Raw C_{n+1} (pre-wrap)", "Wrapped C_{n+1} (post-wrap)"},
      PlotLabel -> Style[Row[{label, " | Raw vs Wrapped"}], 12, Bold],
      PlotRange -> {{-Max[maxRaw, 3], Max[maxRaw, 3]}, {-LBoundary * 1.1, LBoundary * 1.1}},
      ImageSize -> 700,
      AspectRatio -> 0.5
    ]];
    Print["  Blue = no wrap, Gold = wind 1, Red = wind 2, Magenta = wind 3+"];
  ];
  
  (* 3. Density maps by winding class *)
  Print[""];
  Print["=== DENSITY BY WINDING CLASS ==="];
  
  Module[{colorFn, logD, maxLog, normD},
    colorFn[v_] := If[v == 0, RGBColor[0.02, 0.02, 0.02],
      With[{t = v^0.5}, RGBColor[0.6 t + 0.05, 0.5 t + 0.03, 0.2 t + 0.02]]];
    
    (* Wind 0 *)
    logD = Log[density0 + 1.0]; maxLog = Max[logD];
    normD = If[maxLog > 0, logD/maxLog, logD];
    If[Max[Flatten[density0]] > 0,
      Print[Graphics[
        Raster[
          Table[List @@ ColorConvert[colorFn[normD[[i, j]]], "RGB"],
            {i, densityRes, 1, -1}, {j, 1, densityRes}],
          {{-LBoundary, -LBoundary}, {LBoundary, LBoundary}}
        ],
        PlotRange -> {{-LBoundary, LBoundary}, {-LBoundary, LBoundary}},
        Frame -> True, FrameLabel -> {"C_n", "C_{n+1}"},
        PlotLabel -> Style["Wind 0 (no wrap) \[LongDash] coefficient geometry", 11, Bold],
        ImageSize -> 500, AspectRatio -> 1,
        Background -> RGBColor[0.02, 0.02, 0.02]
      ]];
    ];
    
    (* Wind 1 *)
    logD = Log[density1 + 1.0]; maxLog = Max[logD];
    normD = If[maxLog > 0, logD/maxLog, logD];
    If[Max[Flatten[density1]] > 0,
      Print[Graphics[
        Raster[
          Table[List @@ ColorConvert[colorFn[normD[[i, j]]], "RGB"],
            {i, densityRes, 1, -1}, {j, 1, densityRes}],
          {{-LBoundary, -LBoundary}, {LBoundary, LBoundary}}
        ],
        PlotRange -> {{-LBoundary, LBoundary}, {-LBoundary, LBoundary}},
        Frame -> True, FrameLabel -> {"C_n", "C_{n+1}"},
        PlotLabel -> Style["Wind 1 (single wrap) \[LongDash] boundary geometry", 11, Bold],
        ImageSize -> 500, AspectRatio -> 1,
        Background -> RGBColor[0.02, 0.02, 0.02]
      ]];
    ];
    
    (* Wind 2+ *)
    Module[{density2plus},
      density2plus = density2 + density3;
      logD = Log[density2plus + 1.0]; maxLog = Max[logD];
      normD = If[maxLog > 0, logD/maxLog, logD];
      If[Max[Flatten[density2plus]] > 0,
        Print[Graphics[
          Raster[
            Table[List @@ ColorConvert[
              If[normD[[i, j]] == 0, RGBColor[0.02, 0.02, 0.02],
                With[{t = normD[[i, j]]^0.5},
                  RGBColor[0.8 t + 0.1, 0.2 t + 0.02, 0.2 t + 0.02]]]
              , "RGB"],
              {i, densityRes, 1, -1}, {j, 1, densityRes}],
            {{-LBoundary, -LBoundary}, {LBoundary, LBoundary}}
          ],
          PlotRange -> {{-LBoundary, LBoundary}, {-LBoundary, LBoundary}},
          Frame -> True, FrameLabel -> {"C_n", "C_{n+1}"},
          PlotLabel -> Style["Wind 2+ (double+ wrap) \[LongDash] deep excursion geometry", 11, Bold],
          ImageSize -> 500, AspectRatio -> 1,
          Background -> RGBColor[0.02, 0.02, 0.02]
        ]];,
        Print["  No wind 2+ events recorded."];
      ];
    ];
  ];
  
  (* 4. Winding vs Running MM_\[CurlyPhi] correlation *)
  Print[""];
  Print["=== WINDING vs MM_\[Phi] CORRELATION ==="];
  Module[{mmSub, windSub, nSub},
    nSub = Min[5000, nSteps];
    mmSub = runningMM[[3 ;; nSub]];
    windSub = windNums[[3 ;; nSub]];
    
    Print[ListPlot[
      Transpose[{mmSub, windSub}],
      PlotLabel -> Style[Row[{label, " | MM_\[Phi] vs Winding"}], 11],
      PlotStyle -> {PointSize[0.003], Blue},
      AxesLabel -> {"Running MM_\[Phi]", "Winding count"},
      PlotRange -> {All, {-0.5, Max[maxWinding, 3] + 0.5}},
      GridLines -> {{N[GoldenRatio], N[GoldenRatio^2], N[GoldenRatio^3], N[GoldenRatio^4]}, {}},
      GridLinesStyle -> Directive[Orange, Dashed],
      Epilog -> {
        Text[Style["\[Phi]^1", 9, Orange], {N[GoldenRatio], Max[maxWinding, 3] + 0.3}],
        Text[Style["\[Phi]^2", 9, Orange], {N[GoldenRatio^2], Max[maxWinding, 3] + 0.3}],
        Text[Style["\[Phi]^3", 9, Orange], {N[GoldenRatio^3], Max[maxWinding, 3] + 0.3}],
        Text[Style["\[Phi]^4", 9, Orange], {N[GoldenRatio^4], Max[maxWinding, 3] + 0.3}]
      },
      ImageSize -> 600,
      AspectRatio -> 0.4
    ]];
  ];
  
  (* Return data for cross-configuration comparison *)
  {windNums, rawVals, wrappedVals, runningMM, directions,
   density0, density1, density2, density3}
];

(* ============================================= *)
(* RUN THREE CONFIGURATIONS                       *)
(* ============================================= *)

nSteps = 50000;

Print[""];
Print["Running three configurations at L = \[Phi]..."];
Print["Each: ", nSteps, " steps, seed = 42"];
Print[""];

(* Configuration 1: k_phi \[LongDash] blade regime *)
kPhi = N[GoldenRatio / (9 * GoldenRatio - 3)];
data1 = analyzeWinding[kPhi, "k = k_\[Phi] (blade regime)", nSteps, 42];

(* Configuration 2: k = 1/6 \[LongDash] stability boundary *)
data2 = analyzeWinding[1/6 // N, "k = 1/6 (stability boundary)", nSteps, 42];

(* Configuration 3: k = 0.165 \[LongDash] near-torus *)
data3 = analyzeWinding[0.165, "k = 0.165 (near-torus regime)", nSteps, 42];

(* ============================================= *)
(* CROSS-CONFIGURATION COMPARISON                 *)
(* ============================================= *)

Print[""];
Print["==================================================="];
Print["  CROSS-CONFIGURATION SUMMARY"];
Print["==================================================="];
Print[""];
Print["  The winding signature changes with k because"];
Print["  the eigenvalue magnitude determines how far"];
Print["  the raw pre-wrap value reaches:"];
Print[""];
Print["  k_\[Phi]:   |\[Lambda]|\[Squared] = \[Phi] \[TildeTilde] 1.618 (divergent, high winding)"];
Print["  k=1/6: |\[Lambda]|\[Squared] = 1.000 (neutral, moderate winding)"];
Print["  k=0.165: |\[Lambda]|\[Squared] < 1 (convergent, low winding)"];
Print[""];
Print["  If \[Phi]^4 has a distinct winding character,"];
Print["  it will appear as a geometric pattern in the"];
Print["  Wind 2+ density that is absent from Wind 0/1."];

(* ============================================= *)
(* SUMMARY                                        *)
(* ============================================= *)

Print[""];
Print["==================================================="];
Print["  HE SIMULATION H \[LongDash] COMPLETE"];
Print["==================================================="];
Print[""];
Print["  Three questions answered:"];
Print[""];
Print["  1. WINDING DISTRIBUTION: What fraction of steps"];
Print["     involve wrapping, and at what multiplicity?"];
Print[""];
Print["  2. WINDING-\[Phi] CORRELATION: Do higher \[Phi]-power"];
Print["     levels correspond to higher winding counts?"];
Print[""];
Print["  3. WINDING GEOMETRY: Does each winding class"];
Print["     produce a distinct density pattern, or do"];
Print["     they all look the same?"];
Print[""];
Print["  \[CapitalPsi]    \[FullMoon]\[Ocean]\[DeciduousTree]\[SpiderWeb]    \[YinYang]    \[Infinity]"];



