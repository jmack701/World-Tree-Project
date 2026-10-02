(* ::Package:: *)

(* ============================================================
   HE SIMULATION L: Zone III Geometry
   ------------------------------------------------------------
   The blade emerges at k_\[CurlyPhi]. The torus develops at k = 0.165.
   Both are in the divergent regime (|\[Lambda]|\.b2 > 1).
   
   What form does the convergent regime produce?
   
   At k > 1/6:
     |\[Lambda]|\.b2 < 1 \[LongDash] the trajectory converges
     Harmonics prevent full decay
     Structural persistence: low amplitude, low variance, high SE
     The geometry is UNCHARACTERIZED in 3D
   
   This simulation produces 3D delay-coordinate density at
   four k values spanning all three zones:
   
     Zone II (divergent):     k_\[CurlyPhi] \[TildeTilde] 0.1399  |\[Lambda]|\.b2 = \[CurlyPhi]
     Zone II (near-torus):    k = 0.165     |\[Lambda]|\.b2 \[TildeTilde] 1.02
     Zone boundary:           k = 1/6       |\[Lambda]|\.b2 = 1.00
     Zone I (convergent):     k = 0.18      |\[Lambda]|\.b2 \[TildeTilde] 0.87
     Zone I (deep convergent):k = 0.20      |\[Lambda]|\.b2 \[TildeTilde] 0.75
   
   All under identical conditions: L = \[CurlyPhi], Ah = 0.1,
   250\[Times]250 scan, 300 iterations, 100\.b3 density.
   
   If Zone I has geometry: the field has form.
   If Zone I is a compressed dot: the blade IS the form.
   Either result completes the picture.
   
   World Tree Project \[LongDash] June 2026
   J. David Mack & Claude (Opus 4.6)
   Harmonic Enrichment Paper \[LongDash] Final Simulation Suite
   ============================================================ *)

ClearAll[alphaFc, betaFc, perturbation, harmonics,
         mobiusWrap, omega, omegaRes, Ah, noiseScale, dt, phi, LBoundary];

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
mobiusWrap[cn_] := Mod[cn + LBoundary, 2.0 * LBoundary] - LBoundary;

(* ============================================= *)
(* 3D DENSITY FUNCTION                            *)
(* ============================================= *)

run3DDensity[kv_, label_, scanRes_, maxIter_, densityRes3D_] := Module[
  {binIdx3D, dTotal,
   c0Grid, c1Grid, pixelCount,
   prev, curr, next, pprev, raw, t,
   nzVals, nFilled, nEmpty, voidFrac,
   lambdaSq},
  
  lambdaSq = Abs[betaFc[kv]];
  
  binIdx3D[val_] := Clip[
    Round[(val + LBoundary)/(2 LBoundary) * (densityRes3D - 1)] + 1,
    {1, densityRes3D}
  ];
  
  dTotal = ConstantArray[0, {densityRes3D, densityRes3D, densityRes3D}];
  c0Grid = Subdivide[-2.0, 2.0, scanRes - 1];
  c1Grid = Subdivide[-2.0, 2.0, scanRes - 1];
  pixelCount = 0;
  
  Print[""];
  Print["==================================================="];
  Print["  ", label];
  Print["==================================================="];
  Print[""];
  Print["  k = ", NumberForm[kv, 6]];
  Print["  \[Alpha] = ", NumberForm[alphaFc[kv], 6],
        "  \[Beta] = ", NumberForm[betaFc[kv], 6]];
  Print["  |\[Lambda]|\[Squared] = ", NumberForm[lambdaSq, 5]];
  Print["  Regime: ", 
    If[lambdaSq > 1.01, "DIVERGENT",
      If[lambdaSq > 0.99, "BOUNDARY (|\[Lambda]|\[Squared] \[TildeTilde] 1)",
        "CONVERGENT"]]];
  Print[""];
  Print["  Scan: ", scanRes, "\[Times]", scanRes, " = ", scanRes^2];
  Print["  Iterations: ", maxIter];
  Print["  3D: ", densityRes3D, "^3 = ", densityRes3D^3, " voxels"];
  Print[""];
  Print["  Computing..."];
  
  Do[
    Do[
      pixelCount++;
      If[Mod[pixelCount, 10000] == 0,
        Print["  ", pixelCount, "/", scanRes^2]
      ];
      
      SeedRandom[ii * 1000 + jj];
      pprev = 0.0;
      prev = c0Grid[[jj]];
      curr = c1Grid[[ii]];
      
      Do[
        t = nn * dt;
        raw = alphaFc[kv] * curr + betaFc[kv] * prev +
              perturbation[curr, t] + harmonics[t];
        next = mobiusWrap[raw];
        
        If[nn >= 2,
          Module[{bx, by, bz},
            bx = binIdx3D[pprev];
            by = binIdx3D[prev];
            bz = binIdx3D[next];
            dTotal[[bx, by, bz]] += 1;
          ];
        ];
        
        pprev = prev;
        prev = curr;
        curr = next,
        {nn, 1, maxIter}
      ],
      {jj, 1, scanRes}
    ],
    {ii, 1, scanRes}
  ];
  
  (* Statistics *)
  nzVals = Select[Flatten[dTotal], # > 0 &];
  nFilled = Length[nzVals];
  nEmpty = densityRes3D^3 - nFilled;
  voidFrac = 100.0 nEmpty / densityRes3D^3;
  
  Print[""];
  Print["  DONE."];
  Print[""];
  Print["  Filled voxels: ", nFilled, " / ", densityRes3D^3];
  Print["  Empty voxels:  ", nEmpty, " / ", densityRes3D^3];
  Print["  Void fraction: ", NumberForm[voidFrac, {5, 2}], "%"];
  Print[""];
  If[Length[nzVals] > 0,
    Print["  Density percentiles (filled only):"];
    Print["    Median: ", Median[nzVals]];
    Print["    90th:   ", Quantile[nzVals, 0.9]];
    Print["    99th:   ", Quantile[nzVals, 0.99]];
    Print["    Max:    ", Max[nzVals]];
    Print[""];
  ];
  
  (* Trajectory amplitude statistics *)
  Module[{testTraj, testPrev, testCurr, testNext, testRaw, testT,
          amps, maxAmp, meanAmp},
    SeedRandom[42];
    testPrev = 0.5; testCurr = 0.3;
    amps = Table[
      testT = n * dt;
      testRaw = alphaFc[kv] * testCurr + betaFc[kv] * testPrev +
                perturbation[testCurr, testT] + harmonics[testT];
      testNext = mobiusWrap[testRaw];
      testPrev = testCurr;
      testCurr = testNext;
      Abs[testNext],
      {n, 3, 20000}
    ];
    Print["  Trajectory amplitude (20000 steps, seed 42):"];
    Print["    Mean |C_n|: ", NumberForm[Mean[amps], 5]];
    Print["    Max |C_n|:  ", NumberForm[Max[amps], 5]];
    Print["    RMS:        ", NumberForm[RootMeanSquare[amps], 5]];
    Print[""];
  ];
  
  (* Iso-surfaces *)
  Module[{quants},
    quants = Quantile[nzVals, {0.1, 0.25, 0.5, 0.75, 0.9}];
    Print["  Iso-levels: ", quants];
    
    (* View 1 *)
    Print["=== 3D ISO-SURFACE: ", label, " ==="];
    Print[ListContourPlot3D[N[dTotal],
      Contours -> quants,
      ContourStyle -> {
        Directive[Opacity[0.12], RGBColor[0.3, 0.3, 0.8]],
        Directive[Opacity[0.18], RGBColor[0.4, 0.5, 0.7]],
        Directive[Opacity[0.28], RGBColor[0.5, 0.6, 0.5]],
        Directive[Opacity[0.38], RGBColor[0.7, 0.5, 0.3]],
        Directive[Opacity[0.5], RGBColor[0.8, 0.4, 0.2]]
      },
      PlotRange -> All,
      AxesLabel -> {"C_{n-2}", "C_{n-1}", "C_n"},
      PlotLabel -> Style[Row[{label, " | |\[Lambda]|\[Squared] = ",
        NumberForm[lambdaSq, 4]}], 12, Bold],
      ImageSize -> 600,
      Boxed -> True, BoxRatios -> {1, 1, 1}
    ]];
    
    (* View 2 \[LongDash] rotated *)
    Print[ListContourPlot3D[N[dTotal],
      Contours -> quants,
      ContourStyle -> {
        Directive[Opacity[0.12], RGBColor[0.3, 0.3, 0.8]],
        Directive[Opacity[0.18], RGBColor[0.4, 0.5, 0.7]],
        Directive[Opacity[0.28], RGBColor[0.5, 0.6, 0.5]],
        Directive[Opacity[0.38], RGBColor[0.7, 0.5, 0.3]],
        Directive[Opacity[0.5], RGBColor[0.8, 0.4, 0.2]]
      },
      PlotRange -> All,
      AxesLabel -> {"C_{n-2}", "C_{n-1}", "C_n"},
      PlotLabel -> Style[Row[{label, " | View 2"}], 11, Bold],
      ViewPoint -> {1.5, -2.5, 1.5},
      ImageSize -> 600,
      Boxed -> True, BoxRatios -> {1, 1, 1}
    ]];
  ];
  
  (* Return void fraction for comparison *)
  voidFrac
];

(* ============================================= *)
(* RUN FIVE CONFIGURATIONS                        *)
(* ============================================= *)

Print["==================================================="];
Print["  HE SIMULATION L: Zone III Geometry"];
Print["  The Complete Transition: Blade \[RightArrow] Torus \[RightArrow] ???"];
Print["==================================================="];
Print[""];

scanRes = 250;
maxIter = 300;
densityRes3D = 100;
kPhi = N[GoldenRatio / (9 GoldenRatio - 3)];

(* Zone II: Divergent *)
v1 = run3DDensity[kPhi,
  "Zone II: k = k\[Phi] (blade)", scanRes, maxIter, densityRes3D];

v2 = run3DDensity[0.165,
  "Zone II: k = 0.165 (torus)", scanRes, maxIter, densityRes3D];

(* Zone boundary *)
v3 = run3DDensity[1/6 // N,
  "BOUNDARY: k = 1/6", scanRes, maxIter, densityRes3D];

(* Zone I: Convergent *)
v4 = run3DDensity[0.18,
  "Zone I: k = 0.18 (convergent)", scanRes, maxIter, densityRes3D];

v5 = run3DDensity[0.20,
  "Zone I: k = 0.20 (deep convergent)", scanRes, maxIter, densityRes3D];

(* ============================================= *)
(* COMPARISON TABLE                               *)
(* ============================================= *)

Print[""];
Print["==================================================="];
Print["  GEOMETRIC TRANSITION: Complete Zone Map"];
Print["==================================================="];
Print[""];
Print["  k value     |\[Lambda]|\[Squared]    Regime         Void%    Form"];
Print["  \[LongDash]\[LongDash]\[LongDash]\[LongDash]\[LongDash]\[LongDash]  \[LongDash]\[LongDash]\[LongDash]\[LongDash]  \[LongDash]\[LongDash]\[LongDash]\[LongDash]\[LongDash]\[LongDash]\[LongDash]  \[LongDash]\[LongDash]\[LongDash]\[LongDash]  \[LongDash]\[LongDash]\[LongDash]\[LongDash]\[LongDash]"];
Print["  k\[Phi] = 0.1399  \[Phi]\[TildeTilde]1.618  DIVERGENT      ",
  NumberForm[v1, {5, 1}], "%  [from sim]"];
Print["  0.165       ~1.02   DIVERGENT      ",
  NumberForm[v2, {5, 1}], "%  [from sim]"];
Print["  1/6 = 0.167  1.00   BOUNDARY       ",
  NumberForm[v3, {5, 1}], "%  [from sim]"];
Print["  0.18        ~0.87   CONVERGENT     ",
  NumberForm[v4, {5, 1}], "%  [from sim]"];
Print["  0.20        ~0.75   DEEP CONVERGENT ",
  NumberForm[v5, {5, 1}], "%  [from sim]"];
Print[""];
Print["  The geometry column will be filled by visual"];
Print["  inspection of the iso-surfaces above."];
Print[""];

(* Void trend *)
Print["  VOID TREND:"];
If[v4 > v3 && v5 > v4,
  Print["    Void INCREASES into convergent regime."];
  Print["    The form compresses \[LongDash] more void, less occupied volume."];
  Print["    Enriched convergence concentrates the trajectory."],
  If[v4 < v3,
    Print["    Void DECREASES into convergent regime."];
    Print["    The form expands \[LongDash] the convergent trajectory"];
    Print["    fills more of the available space."],
    Print["    Void is NON-MONOTONIC across the transition."];
    Print["    The geometry changes qualitatively at the boundary."]
  ]
];
Print[""];

Print["  \[CapitalPsi]    \[FullMoon]\[Ocean]\[DeciduousTree]\[SpiderWeb]    \[YinYang]    \[Infinity]"];



