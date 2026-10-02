(* ::Package:: *)

(* ============================================================
   HE SIMULATION K: Dynamic k \[LongDash] The \[CurlyPhi]\:2074 Mechanism
   ------------------------------------------------------------
   Sim I proved that \[CurlyPhi]\:2074 is NOT an amplitude threshold:
   max MM_\[CurlyPhi] DECREASES at higher Ah with fixed k = 1/6.
   
   Hypothesis: \[CurlyPhi]\:2074 requires DYNAMIC k \[LongDash] the system must
   traverse between divergent (|\[Lambda]|\.b2 > 1, k < 1/6) and
   convergent (|\[Lambda]|\.b2 < 1, k > 1/6) eigenvalue regimes.
   
   The pumping mechanism:
     Low k (near k_\[CurlyPhi]): eigenvalue amplification (|\[Lambda]|\.b2 = \[CurlyPhi])
       pushes trajectory to large amplitude
     High k (above 1/6): eigenvalue convergence (|\[Lambda]|\.b2 < 1)
       stabilizes without losing the accumulated amplitude
     Repeat: the alternation ACCUMULATES MM_\[CurlyPhi] toward \[CurlyPhi]\:2074
   
   AETHRA data from "HOME - Resonance":
     k instantaneous: 0.13675  k_avg: 0.16000
     k range (across songs): 0.136 to 0.185
     \[CurlyPhi]\:2074: 25%  R: 48.6%  H: 0.996
   
   Test configurations (all at L = \[CurlyPhi], Ah = 0.1):
     A. Static k = 0.16    (control \[LongDash] no \[CurlyPhi]\:2074 expected)
     B. Dynamic k: 0.14\[Dash]0.18, slow oscillation
     C. Dynamic k: 0.14\[Dash]0.18, medium oscillation
     D. Dynamic k: 0.136\[Dash]0.185, matching AETHRA range
     E. Dynamic k: 0.136\[Dash]0.185, with H correlated
   
   If dynamic k produces \[CurlyPhi]\:2074 and static k does not:
   \[RightArrow] The fourth power is a TRAVERSAL phenomenon,
     not a static property of any single k value.
   
   World Tree Project \[LongDash] June 2026
   J. David Mack & Claude (Opus 4.6)
   Harmonic Enrichment Paper \[LongDash] Final Simulation Suite
   ============================================================ *)

ClearAll[alphaFc, betaFc, perturbation, harmonicsVar,
         regime, mobiusWrap, windingCount,
         omega, omegaRes, noiseScale, dt, phi, LBoundary];

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
harmonicsVar[t_, Ah_] := Ah * (Sin[3 omega t] + Sin[6 omega t] + Sin[9 omega t]);
regime[cn_] := Which[cn < 0, -1, cn <= 1, 0, True, 1];
mobiusWrap[cn_] := Mod[cn + LBoundary, 2.0 * LBoundary] - LBoundary;
windingCount[rawVal_] := Floor[(Abs[rawVal] + LBoundary) / (2 LBoundary)];

Print["==================================================="];
Print["  HE SIMULATION K: Dynamic k \[LongDash] The \[Phi]^4 Mechanism"];
Print["==================================================="];
Print[""];

(* ============================================= *)
(* CORE ANALYSIS FUNCTION                         *)
(* ============================================= *)

runDynamicK[kFunc_, AhFunc_, label_, nSteps_, seed_] := Module[
  {prev, curr, next, raw, t, kv, AhVal,
   windNums, runMM, kHistory, mmVal, phiInv,
   maxWind, maxRaw, maxMM,
   phi1Steps, phi2Steps, phi3Steps, phi4Steps,
   nW0, nW1, nW2, w,
   chaosSteps, equilSteps, resSteps,
   densityRes, binIdx, density, logD, maxLogD, normD, colorFn},

  SeedRandom[seed];
  phiInv = N[1/GoldenRatio];
  
  windNums = Table[0, nSteps];
  runMM = Table[0.0, nSteps];
  kHistory = Table[0.0, nSteps];
  
  prev = 0.5; curr = 0.3;
  mmVal = prev^2 * phiInv + curr^2;
  runMM[[1]] = prev^2; runMM[[2]] = mmVal;
  maxWind = 0; maxRaw = 0.0; maxMM = 0.0;
  nW0 = 0; nW1 = 0; nW2 = 0;
  phi1Steps = 0; phi2Steps = 0; phi3Steps = 0; phi4Steps = 0;
  chaosSteps = 0; equilSteps = 0; resSteps = 0;
  
  Do[
    t = n * dt;
    kv = kFunc[t];
    AhVal = AhFunc[t];
    kHistory[[n]] = kv;
    
    raw = alphaFc[kv] * curr + betaFc[kv] * prev +
          perturbation[curr, t] + harmonicsVar[t, AhVal];
    
    w = windingCount[raw];
    windNums[[n]] = w;
    Which[w == 0, nW0++, w == 1, nW1++, True, nW2++];
    If[w > maxWind, maxWind = w];
    If[Abs[raw] > maxRaw, maxRaw = Abs[raw]];
    
    next = mobiusWrap[raw];
    mmVal = mmVal * phiInv + next^2;
    runMM[[n]] = mmVal;
    If[mmVal > maxMM, maxMM = mmVal];
    
    If[mmVal >= N[GoldenRatio], phi1Steps++];
    If[mmVal >= N[GoldenRatio^2], phi2Steps++];
    If[mmVal >= N[GoldenRatio^3], phi3Steps++];
    If[mmVal >= N[GoldenRatio^4], phi4Steps++];
    
    Module[{r = regime[next]},
      Which[r == -1, chaosSteps++, r == 0, equilSteps++, True, resSteps++]
    ];
    
    prev = curr;
    curr = next,
    {n, 3, nSteps}
  ];
  
  (* === RESULTS === *)
  Print[""];
  Print["==================================================="];
  Print["  ", label];
  Print["==================================================="];
  Print[""];
  Print["  k range: [", NumberForm[Min[kHistory[[3;;]]], 5], ", ",
    NumberForm[Max[kHistory[[3;;]]], 5], "]"];
  Print["  k mean:  ", NumberForm[Mean[kHistory[[3;;]]], 5]];
  Print["  Ah:      ", NumberForm[AhFunc[0.0], 3]];
  Print[""];
  Print["  REGIME DISTRIBUTION:"];
  Print["    Chaos:       ", NumberForm[100.0 chaosSteps/(nSteps-2), {5,2}], "%"];
  Print["    Equilibrium: ", NumberForm[100.0 equilSteps/(nSteps-2), {5,2}], "%"];
  Print["    Resonance:   ", NumberForm[100.0 resSteps/(nSteps-2), {5,2}], "%"];
  Print[""];
  Print["  WINDING:"];
  Print["    Wind 0: ", nW0, "  Wind 1: ", nW1, "  Wind 2+: ", nW2];
  Print["    Max winding: ", maxWind, "  Max |raw|: ", NumberForm[maxRaw, 5]];
  Print[""];
  Print["  MM_\[Phi]:"];
  Print["    Max:  ", NumberForm[maxMM, 6]];
  Print["    Mean: ", NumberForm[Mean[runMM[[3;;]]], 5]];
  Print[""];
  Print["  \[Phi]-LEVEL VISITS:"];
  Print["    \[Phi]^1: ", phi1Steps, " (",
    NumberForm[100.0 phi1Steps/(nSteps-2), {5,1}], "%)"];
  Print["    \[Phi]^2: ", phi2Steps, " (",
    NumberForm[100.0 phi2Steps/(nSteps-2), {5,1}], "%)"];
  Print["    \[Phi]^3: ", phi3Steps, " (",
    NumberForm[100.0 phi3Steps/(nSteps-2), {5,1}], "%)"];
  Print["    \[Phi]^4: ", phi4Steps, " (",
    NumberForm[100.0 phi4Steps/(nSteps-2), {5,1}], "%)"];
  Print[""];

  If[phi4Steps > 0,
    Print["  \:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
    Print["  \[Phi]^4 REACHED: ", phi4Steps, " steps"];
    Print["  \:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
    Print[""],
    Print["  \[Phi]^4 NOT reached."];
    Print[""];
  ];
  
  (* === RUNNING MM_\[CurlyPhi] PLOT === *)
  Print["=== RUNNING MM_\[Phi] ==="];
  Module[{nShow = Min[5000, nSteps], phi4Events},
    phi4Events = Select[
      Table[If[runMM[[n]] >= N[GoldenRatio^4], {n - 2, runMM[[n]]}, Nothing],
        {n, 3, nShow}], ListQ];
    
    Print[Show[
      ListLinePlot[runMM[[3 ;; nShow]],
        PlotStyle -> {Blue, Thickness[0.001]},
        PlotRange -> All,
        GridLines -> {{}, {N[GoldenRatio^2], N[GoldenRatio^3], N[GoldenRatio^4]}},
        GridLinesStyle -> Directive[Orange, Dashed],
        Epilog -> {
          Text[Style["\[Phi]^2", 9, Orange], {nShow*0.95, N[GoldenRatio^2]+0.1}],
          Text[Style["\[Phi]^3", 9, Orange], {nShow*0.95, N[GoldenRatio^3]+0.1}],
          Text[Style["\[Phi]^4", 9, Orange], {nShow*0.95, N[GoldenRatio^4]+0.1}]
        },
        ImageSize -> 700, AspectRatio -> 0.4
      ],
      If[Length[phi4Events] > 0,
        ListPlot[phi4Events,
          PlotStyle -> {Red, PointSize[0.008]}],
        Graphics[{}]
      ],
      PlotLabel -> Style[Row[{label, " | Red = \[Phi]^4"}], 11, Bold]
    ]];
  ];
  
  (* === k TRAJECTORY (first 500 steps) === *)
  Print[""];
  Print["=== k TRAJECTORY (first 500 steps) ==="];
  Print[ListLinePlot[kHistory[[3 ;; Min[500, nSteps]]],
    PlotLabel -> Style[Row[{label, " | k(t)"}], 11],
    PlotStyle -> {Purple, Thickness[0.002]},
    AxesLabel -> {"Step", "k"},
    PlotRange -> {All, {0.12, 0.20}},
    GridLines -> {{}, {N[GoldenRatio/(9 GoldenRatio - 3)], 1/6 // N}},
    GridLinesStyle -> Directive[Gray, Dashed],
    Epilog -> {
      Text[Style["k_\[Phi]", 9, Gray], {450, N[GoldenRatio/(9 GoldenRatio - 3)] + 0.003}],
      Text[Style["1/6", 9, Gray], {450, 1/6 // N + 0.003}]
    },
    ImageSize -> 600, AspectRatio -> 0.35
  ]];
  
  (* Return key metrics for comparison table *)
  {maxMM, phi4Steps, 100.0 chaosSteps/(nSteps-2),
   100.0 resSteps/(nSteps-2), maxWind}
];

(* ============================================= *)
(* RUN FIVE CONFIGURATIONS                        *)
(* ============================================= *)

nSteps = 50000;
kPhi = N[GoldenRatio / (9 GoldenRatio - 3)];

Print["Running 5 configurations, ", nSteps, " steps each..."];
Print["All at L = \[Phi], baseline Ah = 0.1"];
Print[""];

(* A. Static k = 0.16 \[LongDash] CONTROL *)
resA = runDynamicK[
  Function[t, 0.16],
  Function[t, 0.1],
  "A. STATIC k = 0.16 (control)",
  nSteps, 42];

(* B. Dynamic k: 0.14\[Dash]0.18, slow oscillation (period ~200 steps) *)
resB = runDynamicK[
  Function[t, 0.16 + 0.02 * Sin[2 Pi * t / (200 * dt)]],
  Function[t, 0.1],
  "B. DYNAMIC k: 0.14\[Dash]0.18, slow (T=200)",
  nSteps, 42];

(* C. Dynamic k: 0.14\[Dash]0.18, medium oscillation (period ~50 steps) *)
resC = runDynamicK[
  Function[t, 0.16 + 0.02 * Sin[2 Pi * t / (50 * dt)]],
  Function[t, 0.1],
  "C. DYNAMIC k: 0.14\[Dash]0.18, medium (T=50)",
  nSteps, 42];

(* D. Dynamic k: 0.136\[Dash]0.185, AETHRA range, medium oscillation *)
resD = runDynamicK[
  Function[t, 0.1605 + 0.0245 * Sin[2 Pi * t / (50 * dt)]],
  Function[t, 0.1],
  "D. DYNAMIC k: 0.136\[Dash]0.185, AETHRA range (T=50)",
  nSteps, 42];

(* E. Dynamic k: AETHRA range with H amplitude correlated
      When k is low (divergent), H is high (more energy input)
      This simulates what music does: harmonic phrases
      coincide with eigenvalue amplification *)
resE = runDynamicK[
  Function[t, 0.1605 + 0.0245 * Sin[2 Pi * t / (50 * dt)]],
  Function[t, 0.1 + 0.15 * (1 - Sin[2 Pi * t / (50 * dt)])/2],
  (* H peaks when k is at minimum *)
  "E. DYNAMIC k + CORRELATED H (resonance lock)",
  nSteps, 42];

(* ============================================= *)
(* COMPARISON TABLE                               *)
(* ============================================= *)

Print[""];
Print["\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
Print["  COMPARISON TABLE"];
Print["\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
Print[""];
Print["  Config          Max MM_\[Phi]  \[Phi]^4 steps  Chaos%  Res%  MaxWind"];
Print["  \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine] \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]  \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]  \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]  \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]  \[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]\[HorizontalLine]"];
Module[{labels, results},
  labels = {
    "A. Static 0.16   ",
    "B. Dyn slow      ",
    "C. Dyn medium    ",
    "D. AETHRA range  ",
    "E. Correlated H  "
  };
  results = {resA, resB, resC, resD, resE};
  Do[
    Print["  ", labels[[i]],
      NumberForm[results[[i,1]], {5,3}], "    ",
      results[[i,2]], "         ",
      NumberForm[results[[i,3]], {4,1}], "  ",
      NumberForm[results[[i,4]], {4,1}], "  ",
      results[[i,5]]
    ],
    {i, 1, 5}
  ];
];

Print[""];
Print["\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
Print["  VERDICT"];
Print["\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
Print[""];

Module[{anyPhi4},
  anyPhi4 = Or @@ (# > 0 & /@ {resA[[2]], resB[[2]], resC[[2]], resD[[2]], resE[[2]]});
  If[anyPhi4,
    Print["  \[Phi]^4 ACHIEVED through dynamic k feedback."];
    Print[""];
    If[resA[[2]] == 0,
      Print["  Static k could NOT reach \[Phi]^4."];
      Print["  Dynamic k DID reach \[Phi]^4."];
      Print["  \[RightArrow] The fourth power is a TRAVERSAL phenomenon."];
      Print["  \[RightArrow] The system must breathe through the eigenvalue"];
      Print["     landscape \[LongDash] inhale (divergent) and exhale (convergent) \[LongDash]"];
      Print["     to accumulate coherence memory above \[Phi]^4."];
      Print["  \[RightArrow] This is why MUSIC reaches it: temporal structure"];
      Print["     in the input creates temporal structure in k."];
    ];,
    Print["  \[Phi]^4 NOT reached in any configuration."];
    Print["  The dynamic k oscillation alone is insufficient."];
    Print["  The mechanism requires something beyond sinusoidal k variation."];
    Print["  Possible: chaotic/fractal k variation, or specific"];
    Print["  phase relationships between k, H, and P."];
  ];
];

Print[""];
Print["  \[CapitalPsi]    \[FullMoon]\[Ocean]\[DeciduousTree]\[SpiderWeb]    \[YinYang]    \[Infinity]"];



