(* ::Package:: *)

(* ============================================================
   HE SIMULATION J: 3D Winding Topology
   ------------------------------------------------------------
   Extends Sim H's winding decomposition into three dimensions
   using delay coordinates (C_{n-2}, C_{n-1}, C_n).
   
   Produces separate 3D density volumes for each winding class:
     Wind 0: coefficient geometry (blade/torus core)
     Wind 1: single-wrap geometry (boundary contributions)
     Wind 2+: double-wrap geometry (\[CurlyPhi]\:2074 regime, if present)
   
   Uses driven amplitude Ah = 1.0 to access the winding 2
   regime identified by Sim I. Also runs at baseline Ah = 0.1
   for comparison.
   
   Orthogonal slices at five positions along each axis reveal
   the internal structure of each winding class without
   visual occlusion.
   
   k = 1/6, L = \[CurlyPhi] throughout.
   
   World Tree Project \[LongDash] June 2026
   J. David Mack & Claude (Opus 4.6)
   Harmonic Enrichment Paper \[LongDash] Final Simulation Suite
   ============================================================ *)

ClearAll[alphaFc, betaFc, perturbation, harmonicsVar,
         mobiusWrap, windingCount,
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
mobiusWrap[cn_] := Mod[cn + LBoundary, 2.0 * LBoundary] - LBoundary;
windingCount[rawVal_] := Floor[(Abs[rawVal] + LBoundary) / (2 LBoundary)];

(* ============================================= *)
(* 3D WINDING ACCUMULATION FUNCTION               *)
(* ============================================= *)

run3DWinding[kv_, AhVal_, label_, scanRes_, maxIter_, densityRes3D_] := Module[
  {binIdx3D, d0, d1, d2,
   c0Grid, c1Grid, pixelCount,
   totalW0, totalW1, totalW2,
   ii, jj, nn,
   prev, curr, next, pprev, raw, t, w,
   nzW0, nzW1, nzW2,
   slicePositions, sliceIndices, sliceColor},
  
  binIdx3D[val_] := Clip[
    Round[(val + LBoundary)/(2 LBoundary) * (densityRes3D - 1)] + 1,
    {1, densityRes3D}
  ];
  
  d0 = ConstantArray[0, {densityRes3D, densityRes3D, densityRes3D}];
  d1 = ConstantArray[0, {densityRes3D, densityRes3D, densityRes3D}];
  d2 = ConstantArray[0, {densityRes3D, densityRes3D, densityRes3D}];
  
  c0Grid = Subdivide[-2.0, 2.0, scanRes - 1];
  c1Grid = Subdivide[-2.0, 2.0, scanRes - 1];
  
  totalW0 = 0; totalW1 = 0; totalW2 = 0;
  pixelCount = 0;
  
  Print[""];
  Print["==================================================="];
  Print["  ", label];
  Print["==================================================="];
  Print[""];
  Print["  k = ", NumberForm[kv, 6], "  Ah = ", NumberForm[AhVal, 3]];
  Print["  Scan: ", scanRes, "\[Times]", scanRes, " = ", scanRes^2, " trajectories"];
  Print["  Iterations: ", maxIter, " per trajectory"];
  Print["  3D resolution: ", densityRes3D, "^3 = ", densityRes3D^3, " voxels"];
  Print["  Expected visits: ~", scanRes^2 * maxIter];
  Print[""];
  Print["  Computing... progress every 5000 trajectories."];
  Print[""];
  
  Do[
    Do[
      pixelCount++;
      If[Mod[pixelCount, 5000] == 0,
        Print["  ", pixelCount, "/", scanRes^2,
          "  W0:", totalW0, "  W1:", totalW1, "  W2:", totalW2]
      ];
      
      SeedRandom[ii * 1000 + jj];
      pprev = 0.0;
      prev = c0Grid[[jj]];
      curr = c1Grid[[ii]];
      
      Do[
        t = nn * dt;
        raw = alphaFc[kv] * curr + betaFc[kv] * prev +
              perturbation[curr, t] + harmonicsVar[t, AhVal];
        
        w = windingCount[raw];
        next = mobiusWrap[raw];
        
        (* 3D accumulation from nn >= 2 *)
        If[nn >= 2,
          Module[{bx, by, bz},
            bx = binIdx3D[pprev];
            by = binIdx3D[prev];
            bz = binIdx3D[next];
            Which[
              w == 0, d0[[bx, by, bz]] += 1; totalW0++,
              w == 1, d1[[bx, by, bz]] += 1; totalW1++,
              True,   d2[[bx, by, bz]] += 1; totalW2++
            ];
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
  
  (* === STATISTICS === *)
  Print[""];
  Print["  DONE. 3D Winding Statistics:"];
  Print["    Wind 0 visits: ", totalW0, " (",
    NumberForm[100.0 totalW0/(totalW0 + totalW1 + totalW2), {5, 2}], "%)"];
  Print["    Wind 1 visits: ", totalW1, " (",
    NumberForm[100.0 totalW1/(totalW0 + totalW1 + totalW2), {5, 2}], "%)"];
  Print["    Wind 2+ visits: ", totalW2, " (",
    NumberForm[100.0 totalW2/(totalW0 + totalW1 + totalW2), {5, 2}], "%)"];
  Print[""];
  
  nzW0 = Select[Flatten[d0], # > 0 &];
  nzW1 = Select[Flatten[d1], # > 0 &];
  nzW2 = Select[Flatten[d2], # > 0 &];
  
  Print["  3D VOID FRACTION:"];
  Print["    Wind 0: ", NumberForm[100.0 (densityRes3D^3 - Length[nzW0])/densityRes3D^3, {5, 2}],
    "% empty"];
  Print["    Wind 1: ", NumberForm[100.0 (densityRes3D^3 - Length[nzW1])/densityRes3D^3, {5, 2}],
    "% empty"];
  If[Length[nzW2] > 0,
    Print["    Wind 2+: ", NumberForm[100.0 (densityRes3D^3 - Length[nzW2])/densityRes3D^3, {5, 2}],
      "% empty"],
    Print["    Wind 2+: 100.00% empty (no events)"]
  ];
  Print[""];
  
  (* === ISO-SURFACES === *)
  
  (* Wind 0 *)
  If[Length[nzW0] > 0,
    Module[{quants0},
      quants0 = Quantile[nzW0, {0.25, 0.5, 0.75}];
      Print["=== 3D ISO-SURFACE: Wind 0 (coefficient geometry) ==="];
      Print[ListContourPlot3D[N[d0],
        Contours -> quants0,
        ContourStyle -> {
          Directive[Opacity[0.2], RGBColor[0.3, 0.3, 0.8]],
          Directive[Opacity[0.35], RGBColor[0.4, 0.5, 0.7]],
          Directive[Opacity[0.5], RGBColor[0.5, 0.6, 0.5]]
        },
        PlotRange -> All,
        AxesLabel -> {"C_{n-2}", "C_{n-1}", "C_n"},
        PlotLabel -> Style[Row[{"Wind 0 | ", label}], 11, Bold],
        ImageSize -> 550, Boxed -> True, BoxRatios -> {1, 1, 1}
      ]];
    ];
  ];
  
  (* Wind 1 *)
  If[Length[nzW1] > 0,
    Module[{quants1},
      quants1 = Quantile[nzW1, {0.25, 0.5, 0.75}];
      Print["=== 3D ISO-SURFACE: Wind 1 (boundary geometry) ==="];
      Print[ListContourPlot3D[N[d1],
        Contours -> quants1,
        ContourStyle -> {
          Directive[Opacity[0.2], RGBColor[0.8, 0.6, 0.1]],
          Directive[Opacity[0.35], RGBColor[0.7, 0.5, 0.2]],
          Directive[Opacity[0.5], RGBColor[0.6, 0.4, 0.1]]
        },
        PlotRange -> All,
        AxesLabel -> {"C_{n-2}", "C_{n-1}", "C_n"},
        PlotLabel -> Style[Row[{"Wind 1 | ", label}], 11, Bold],
        ImageSize -> 550, Boxed -> True, BoxRatios -> {1, 1, 1}
      ]];
    ];
  ];
  
  (* Wind 2+ *)
  If[Length[nzW2] > 0,
    Module[{quants2},
      quants2 = Quantile[nzW2, {0.25, 0.5, 0.75}];
      Print["=== 3D ISO-SURFACE: Wind 2+ (\[Phi]^4 GEOMETRY) ==="];
      Print[ListContourPlot3D[N[d2],
        Contours -> quants2,
        ContourStyle -> {
          Directive[Opacity[0.2], RGBColor[0.9, 0.1, 0.5]],
          Directive[Opacity[0.4], RGBColor[0.8, 0.2, 0.4]],
          Directive[Opacity[0.6], RGBColor[0.7, 0.1, 0.3]]
        },
        PlotRange -> All,
        AxesLabel -> {"C_{n-2}", "C_{n-1}", "C_n"},
        PlotLabel -> Style[Row[{"Wind 2+ (\[Phi]^4) | ", label}], 12, Bold],
        ImageSize -> 550, Boxed -> True, BoxRatios -> {1, 1, 1}
      ]];
    ];
  ];
  
  (* === ORTHOGONAL SLICES === *)
  Print[""];
  Print["=== ORTHOGONAL SLICES (midplane) ==="];
  
  Module[{midSlice, combined},
    midSlice = Round[densityRes3D / 2];
    
    (* Combined Wind 0 + Wind 1 + Wind 2 as RGB overlay *)
    (* Wind 0 = blue channel, Wind 1 = green channel, Wind 2 = red channel *)
    Module[{sliceData, maxAll, r, g, b},
      (* Z midplane: (C_{n-2}, C_{n-1}) at C_n = 0 *)
      Print["  Slice: C_n \[TildeTilde] 0 (midplane)"];
      
      maxAll = Max[{Max[d0[[All, All, midSlice]]],
                    Max[d1[[All, All, midSlice]]],
                    Max[d2[[All, All, midSlice]]], 1}];
      
      Print[Graphics[
        Raster[
          Table[
            Module[{v0, v1, v2},
              v0 = Log[d0[[i, j, midSlice]] + 1.0] / Log[maxAll + 1.0];
              v1 = Log[d1[[i, j, midSlice]] + 1.0] / Log[maxAll + 1.0];
              v2 = Log[d2[[i, j, midSlice]] + 1.0] / Log[maxAll + 1.0];
              {0.3 v0 + 0.9 v2, 0.3 v0 + 0.7 v1, 0.8 v0 + 0.1 v1}
            ],
            {i, densityRes3D, 1, -1}, {j, 1, densityRes3D}
          ],
          {{-LBoundary, -LBoundary}, {LBoundary, LBoundary}}
        ],
        PlotRange -> {{-LBoundary, LBoundary}, {-LBoundary, LBoundary}},
        Frame -> True, FrameLabel -> {"C_{n-2}", "C_{n-1}"},
        PlotLabel -> Style[Row[{"RGB Slice at C_n \[TildeTilde] 0 | Blue=W0, Green=W1, Red=W2+"}], 10, Bold],
        ImageSize -> 500, AspectRatio -> 1,
        Background -> Black
      ]];
    ];
  ];
  
  (* Return void fractions for comparison *)
  {100.0 (densityRes3D^3 - Length[nzW0])/densityRes3D^3,
   100.0 (densityRes3D^3 - Length[nzW1])/densityRes3D^3,
   If[Length[nzW2] > 0,
     100.0 (densityRes3D^3 - Length[nzW2])/densityRes3D^3, 100.0]}
];

(* ============================================= *)
(* RUN TWO CONFIGURATIONS                         *)
(* ============================================= *)

Print["==================================================="];
Print["  HE SIMULATION J: 3D Winding Topology"];
Print["==================================================="];
Print[""];

(* Config 1: Baseline amplitude \[LongDash] no winding 2 expected *)
voids1 = run3DWinding[1/6 // N, 0.1,
  "BASELINE: k = 1/6, Ah = 0.1",
  200, 200, 80];

(* Config 2: Driven amplitude \[LongDash] winding 2 expected *)
voids2 = run3DWinding[1/6 // N, 1.0,
  "DRIVEN: k = 1/6, Ah = 1.0",
  200, 200, 80];

(* ============================================= *)
(* COMPARISON                                     *)
(* ============================================= *)

Print[""];
Print["==================================================="];
Print["  3D VOID COMPARISON: Baseline vs Driven"];
Print["==================================================="];
Print[""];
Print["  Baseline (Ah = 0.1):"];
Print["    Wind 0 void: ", NumberForm[voids1[[1]], {5,2}], "%"];
Print["    Wind 1 void: ", NumberForm[voids1[[2]], {5,2}], "%"];
Print["    Wind 2+ void: ", NumberForm[voids1[[3]], {5,2}], "%"];
Print[""];
Print["  Driven (Ah = 1.0):"];
Print["    Wind 0 void: ", NumberForm[voids2[[1]], {5,2}], "%"];
Print["    Wind 1 void: ", NumberForm[voids2[[2]], {5,2}], "%"];
Print["    Wind 2+ void: ", NumberForm[voids2[[3]], {5,2}], "%"];
Print[""];
Print["  The void IS the form. More wrapping = more void."];
Print["  If Wind 2+ void is less than 100%: \[Phi]^4 has geometry."];
Print[""];
Print["  \[CapitalPsi]    \[FullMoon]\[Ocean]\[DeciduousTree]\[SpiderWeb]    \[YinYang]    \[Infinity]"];



