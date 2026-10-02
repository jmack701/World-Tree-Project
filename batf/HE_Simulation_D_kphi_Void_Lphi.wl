(* ::Package:: *)

(* ============================================================
   HE SIMULATION D: k_phi Void at Higher Resolution \[LongDash] L = \[CurlyPhi]
   ------------------------------------------------------------
   Extension of The Field v2.0 Simulation D to the golden-ratio
   containment boundary L = \[CurlyPhi] \[TildeTilde] 1.618034.
   
   Tests whether the parallelogram void at k_phi EXPANDS when
   the M\[ODoubleDot]bius boundary matches the golden ratio. The Field
   established that void fraction increases monotonically with
   increasing form (68.5% \[RightArrow] 70.6% \[RightArrow] 74.4%). At L = \[CurlyPhi], the
   system is shaped by \[CurlyPhi] from BOTH directions: eigenvalue
   threshold (k_phi where |\[Lambda]|\.b2 = \[CurlyPhi]) AND containment boundary.
   
   Parameters identical to original Sim D except:
     L = 1.5  \[RightArrow]  L = \[CurlyPhi] (GoldenRatio)
     densityRange = 1.5  \[RightArrow]  densityRange = \[CurlyPhi]
   
   2D: 800\[Times]800 = 640,000 bins
   3D: 100\.b3 = 1,000,000 voxels
   Scan: 500\[Times]500 = 250,000 trajectories \[Times] 300 iterations
   Expected visits: ~75,000,000
   
   Comparison data (from original Sim D at L = 1.5):
     2D: 5,828 empty bins / 640,000 (0.91% empty)
     3D: 685,187 empty voxels / 1,000,000 (68.52% empty)
   
   If void expands at L = \[CurlyPhi]: the golden-ratio boundary
   increases the system's geometric discrimination \[LongDash] more form,
   more void, consistent with The Field's monotonic finding.
   
   World Tree Project \[LongDash] June 2026
   J. David Mack & Claude (Opus 4.6)
   Harmonic Enrichment Paper \[LongDash] Final Simulation Suite
   ============================================================ *)

Block[
  {alpha, beta, omega, omegaRes, Ah, noiseScale, dt, L, phi,
   pertFunc, harmFunc, kBdy, maxIter, escapeRadius, scanRes,
   densityRes, densityRange, densityRes3D,
   binIdx, binIdx3D,
   dTotal, dTotal3D,
   nVirtEsc, nNatBnd, totalVisits,
   c0Grid, c1Grid, pixelCount,
   ii, jj, nn, pB, cB, nxB, ppB, tB, rawB, bx, by, bx3, by3, bz3,
   virtEscTime, pA, cA, nxA, tA, rawA,
   logTotal, maxLogT, normTotal,
   totalColor, nzVals, nzVals3D,
   nEmpty2D, nEmpty3D, nFilled2D, nFilled3D},

  (* Golden ratio *)
  phi = N[GoldenRatio];

  (* F_c coefficients *)
  alpha[kv_] := (1 - 6 kv)/(1 - 9 kv);
  beta[kv_] := (3 kv)/(1 - 9 kv);

  (* Parameters \[LongDash] ALL UNCHANGED from Sim D except L and densityRange *)
  omega = 2 Pi;
  omegaRes = 2 Pi;
  Ah = 0.1;
  noiseScale = 0.2;
  dt = 0.01;

  (* === THE CHANGE: L = \[CurlyPhi] === *)
  L = phi;

  pertFunc[cnVal_, tVal_] := Which[
    cnVal < 0, Abs[cnVal] * noiseScale * RandomReal[{-1, 1}],
    cnVal <= 1, 0,
    True, (cnVal - 1) * Sin[omegaRes * tVal]
  ];
  harmFunc[tVal_] := Ah * (Sin[3 omega tVal] + Sin[6 omega tVal] + Sin[9 omega tVal]);

  (* k_phi = phi / (9*phi - 3) *)
  kBdy = N[GoldenRatio / (9 * GoldenRatio - 3)];

  (* === RESOLUTION PARAMETERS \[LongDash] UNCHANGED === *)
  maxIter = 300;
  escapeRadius = 2.5;
  scanRes = 500;       (* 500\[Times]500 = 250,000 trajectories *)
  densityRes = 800;    (* 800\[Times]800 = 640,000 bins *)
  densityRes3D = 100;  (* 100\.b3 = 1,000,000 voxels *)

  (* === DENSITY RANGE MATCHES L = \[CurlyPhi] === *)
  densityRange = phi;

  (* 2D bin indexing \[LongDash] maps [-\[CurlyPhi], \[CurlyPhi]] to [1, 800] *)
  binIdx[val_] := Clip[
    Round[(val + densityRange)/(2 densityRange) * (densityRes - 1)] + 1,
    {1, densityRes}
  ];

  (* 3D bin indexing \[LongDash] maps [-\[CurlyPhi], \[CurlyPhi]] to [1, 100] *)
  binIdx3D[val_] := Clip[
    Round[(val + densityRange)/(2 densityRange) * (densityRes3D - 1)] + 1,
    {1, densityRes3D}
  ];

  (* Density arrays *)
  dTotal = ConstantArray[0, {densityRes, densityRes}];
  dTotal3D = ConstantArray[0, {densityRes3D, densityRes3D, densityRes3D}];

  nVirtEsc = 0; nNatBnd = 0; totalVisits = 0;

  c0Grid = Subdivide[-2.0, 2.0, scanRes - 1];
  c1Grid = Subdivide[-2.0, 2.0, scanRes - 1];

  (* === HEADER === *)
  Print["==================================================="];
  Print["  HE SIMULATION D: k_\[Phi] Void \[LongDash] L = \[Phi]"];
  Print["==================================================="];
  Print[""];
  Print["  L = \[Phi] = ", NumberForm[L, 10]];
  Print["  k_\[CurlyPhi] = ", NumberForm[kBdy, 8]];
  Print["  \[Alpha](k_\[CurlyPhi]) = ", NumberForm[alpha[kBdy], 6]];
  Print["  \[Beta](k_\[CurlyPhi]) = ", NumberForm[beta[kBdy], 6]];
  Print["  |\[Lambda]|\[Squared] at k_\[CurlyPhi] = \[Phi] = GoldenRatio"];
  Print["  Max pre-wrap = ",
    NumberForm[Abs[alpha[kBdy]]*L + Abs[beta[kBdy]]*L + 0.5 + 0.3, 6]];
  Print["  (Note: P_max with L=\[Phi]: (C_n-1)\[CenterDot]sin at C_n=\[Phi] gives ",
    NumberForm[(phi - 1) * 1.0, 4]];
  Print["   H_max = ", NumberForm[Ah * 3, 2], ")"];
  Print[""];
  Print["  Corrected max pre-wrap = ",
    NumberForm[Abs[alpha[kBdy]]*phi + Abs[beta[kBdy]]*phi + 
      (phi - 1) * 1.0 + Ah * 3, 6]];
  Print[""];
  Print["  2D density: ", densityRes, "\[Times]", densityRes, " = ",
    densityRes^2, " bins"];
  Print["  3D density: ", densityRes3D, "^3 = ",
    densityRes3D^3, " voxels"];
  Print["  Scan grid:  ", scanRes, "\[Times]", scanRes, " = ",
    scanRes^2, " trajectories"];
  Print["  Iterations: ", maxIter, " per trajectory"];
  Print["  Expected visits: ~", scanRes^2 * maxIter];
  Print["  Expected visits/2D-bin: ~",
    NumberForm[N[scanRes^2 * maxIter / densityRes^2], {4, 1}]];
  Print["  Expected visits/3D-voxel: ~",
    NumberForm[N[scanRes^2 * maxIter / densityRes3D^3], {4, 1}]];
  Print[""];
  Print["  COMPARISON DATA (Sim D at L = 1.5):"];
  Print["    2D: 5,828 empty / 640,000 bins (0.91%)"];
  Print["    3D: 685,187 empty / 1,000,000 voxels (68.52%)"];
  Print["    Density: median 134, 90th 177, 99th 207"];
  Print[""];
  Print["  COMPARISON DATA (Sim E at L = \[Phi], 150^3 only):"];
  Print["    3D: 70.61% empty voxels"];
  Print["    Density: median 41, 90th 196, 99th 401"];
  Print[""];
  Print["Computing... progress every 10000 trajectories."];
  Print[""];

  pixelCount = 0;

  Do[
    Do[
      pixelCount++;
      If[Mod[pixelCount, 10000] == 0,
        Print["  ", pixelCount, "/", scanRes^2,
          "  VirtEsc:", nVirtEsc, "  NatBnd:", nNatBnd]
      ];

      (* Pass A: classify by virtual escape *)
      SeedRandom[ii * 1000 + jj];
      pA = c0Grid[[jj]];
      cA = c1Grid[[ii]];
      virtEscTime = 0;

      Do[
        tA = nn * dt;
        rawA = alpha[kBdy] * cA + beta[kBdy] * pA +
               pertFunc[cA, tA] + harmFunc[tA];
        If[Abs[rawA] > escapeRadius && virtEscTime == 0,
          virtEscTime = nn
        ];
        nxA = Mod[rawA + L, 2 L] - L;
        pA = cA;
        cA = nxA,
        {nn, 1, maxIter}
      ];

      (* Pass B: re-iterate with same seed, accumulate into 2D and 3D *)
      SeedRandom[ii * 1000 + jj];
      pB = c0Grid[[jj]];
      cB = c1Grid[[ii]];
      ppB = 0.0;  (* placeholder for first step *)

      If[virtEscTime > 0, nVirtEsc++, nNatBnd++];

      Do[
        tB = nn * dt;
        rawB = alpha[kBdy] * cB + beta[kBdy] * pB +
               pertFunc[cB, tB] + harmFunc[tB];
        nxB = Mod[rawB + L, 2 L] - L;

        (* 2D accumulation: (C_n, C_{n+1}) *)
        bx = binIdx[cB]; by = binIdx[nxB];
        dTotal[[by, bx]] += 1;

        (* 3D accumulation: (C_{n-2}, C_{n-1}, C_n) *)
        If[nn >= 2,
          bx3 = binIdx3D[ppB];
          by3 = binIdx3D[pB];
          bz3 = binIdx3D[nxB];
          dTotal3D[[bx3, by3, bz3]] += 1;
        ];

        totalVisits++;
        ppB = pB;
        pB = cB;
        cB = nxB,
        {nn, 1, maxIter}
      ],
      {jj, 1, scanRes}
    ],
    {ii, 1, scanRes}
  ];

  Print[""];
  Print["DONE. Computing statistics..."];
  Print[""];

  (* === 2D STATISTICS === *)
  nFilled2D = Count[Flatten[dTotal], _?(# > 0 &)];
  nEmpty2D = densityRes^2 - nFilled2D;
  nzVals = Select[Flatten[dTotal], # > 0 &];

  Print["==================================================="];
  Print["  2D DENSITY RESULTS (", densityRes, "\[Times]", densityRes, ")"];
  Print["  L = \[Phi] \[TildeTilde] ", NumberForm[phi, 8]];
  Print["==================================================="];
  Print[""];
  Print["  Total visits: ", totalVisits];
  Print["  Visits per bin (avg): ",
    NumberForm[N[totalVisits/densityRes^2], {5, 1}]];
  Print[""];
  Print["  Filled bins:  ", nFilled2D, " / ", densityRes^2];
  Print["  Empty bins:   ", nEmpty2D, " / ", densityRes^2,
    " (", NumberForm[100.0 nEmpty2D/densityRes^2, {4, 2}], "%)"];
  Print[""];
  Print["  Coverage: ", NumberForm[100.0 nFilled2D/densityRes^2, {5, 2}], "%"];
  Print[""];
  If[Length[nzVals] > 0,
    Print["  Density percentiles (filled bins only):"];
    Print["    Median: ", Median[nzVals]];
    Print["    90th:   ", Quantile[nzVals, 0.9]];
    Print["    99th:   ", Quantile[nzVals, 0.99]];
    Print["    Max:    ", Max[nzVals]];
  ];
  Print[""];
  Print["  \:2550\:2550\:2550 COMPARISON: L = 1.5 vs L = \[Phi] \:2550\:2550\:2550"];
  Print[""];
  Print["    L = 1.5:  5,828 empty / 640,000 (0.91%)"];
  Print["    L = \[Phi]:  ", nEmpty2D, " empty / ", densityRes^2,
    " (", NumberForm[100.0 nEmpty2D/densityRes^2, {4, 2}], "%)"];
  Print[""];
  If[nEmpty2D > 5828,
    Print["    \[DoubleRightArrow] VOID EXPANDED at L = \[Phi]"];
    Print["    Expansion ratio: ", NumberForm[N[nEmpty2D/5828], {4, 2}], "\[Times]"],
    If[nEmpty2D > 4000,
      Print["    \[DoubleRightArrow] Void present but contracted slightly"],
      Print["    \[DoubleRightArrow] Void significantly reduced at L = \[Phi]"]
    ]
  ];
  Print[""];

  (* === 3D STATISTICS === *)
  nFilled3D = Count[Flatten[dTotal3D], _?(# > 0 &)];
  nEmpty3D = densityRes3D^3 - nFilled3D;
  nzVals3D = Select[Flatten[dTotal3D], # > 0 &];

  Print["==================================================="];
  Print["  3D DENSITY RESULTS (", densityRes3D, "^3)"];
  Print["  L = \[Phi]"];
  Print["==================================================="];
  Print[""];
  Print["  Filled voxels: ", nFilled3D, " / ", densityRes3D^3];
  Print["  Empty voxels:  ", nEmpty3D, " / ", densityRes3D^3,
    " (", NumberForm[100.0 nEmpty3D/densityRes3D^3, {4, 2}], "%)"];
  Print[""];
  If[Length[nzVals3D] > 0,
    Print["  3D density percentiles (filled voxels only):"];
    Print["    Median: ", Median[nzVals3D]];
    Print["    90th:   ", Quantile[nzVals3D, 0.9]];
    Print["    99th:   ", Quantile[nzVals3D, 0.99]];
    Print["    Max:    ", Max[nzVals3D]];
  ];
  Print[""];
  Print["  \:2550\:2550\:2550 3D COMPARISON: L = 1.5 vs L = \[Phi] \:2550\:2550\:2550"];
  Print[""];
  Print["    L = 1.5 (Sim D):   68.52% empty"];
  Print["    L = \[Phi] (Sim E 150^3): 70.61% empty"];
  Print["    L = \[Phi] (this, 100^3): ",
    NumberForm[100.0 nEmpty3D/densityRes3D^3, {5, 2}], "% empty"];
  Print[""];

  (* === 2D VISUALIZATION === *)
  Print[""];
  Print["=== 2D Density Map \[LongDash] L = \[Phi] ==="];

  totalColor[v_] := If[v == 0, RGBColor[0.02, 0.02, 0.02],
    With[{t = v^0.5}, RGBColor[0.6 t + 0.05, 0.5 t + 0.03, 0.2 t + 0.02]]];

  logTotal = Log[dTotal + 1.0]; maxLogT = Max[logTotal];
  normTotal = If[maxLogT > 0, logTotal/maxLogT, logTotal];

  Print[Graphics[
    Raster[
      Table[
        List @@ ColorConvert[totalColor[normTotal[[ii, jj]]], "RGB"],
        {ii, densityRes, 1, -1}, {jj, 1, densityRes}
      ],
      {{-densityRange, -densityRange}, {densityRange, densityRange}}
    ],
    PlotRange -> {{-densityRange, densityRange},
                  {-densityRange, densityRange}},
    Frame -> True,
    FrameLabel -> {"C_n", "C_{n+1}"},
    PlotLabel -> Style[
      Row[{"HE Sim D: M\[ODoubleDot]bius Density at k_\[Phi] | L = \[Phi] | ",
           densityRes, "\[Times]", densityRes}],
      12, Bold],
    ImageSize -> 700, AspectRatio -> 1,
    Background -> RGBColor[0.02, 0.02, 0.02]
  ]];

  (* === 3D VISUALIZATION: ISO-SURFACES === *)
  Print[""];
  Print["=== 3D Void Structure \[LongDash] L = \[Phi] ==="];

  Module[{quantiles},
    quantiles = Quantile[nzVals3D, {0.1, 0.25, 0.5, 0.75, 0.9}];
    Print["  3D iso-levels (quantiles 10/25/50/75/90): ", quantiles];

    Print[ListContourPlot3D[
      N[dTotal3D],
      Contours -> quantiles,
      ContourStyle -> {
        Directive[Opacity[0.15], RGBColor[0.3, 0.3, 0.8]],
        Directive[Opacity[0.2], RGBColor[0.4, 0.5, 0.7]],
        Directive[Opacity[0.3], RGBColor[0.5, 0.6, 0.5]],
        Directive[Opacity[0.4], RGBColor[0.7, 0.5, 0.3]],
        Directive[Opacity[0.5], RGBColor[0.8, 0.4, 0.2]]
      },
      PlotRange -> All,
      AxesLabel -> {"C_{n-2}", "C_{n-1}", "C_n"},
      PlotLabel -> Style[
        Row[{"HE Sim D: 3D Void at k_\[Phi] | L = \[Phi] | ", densityRes3D, "^3"}],
        12, Bold],
      ImageSize -> 600,
      Boxed -> True,
      BoxRatios -> {1, 1, 1}
    ]];

    (* Second viewing angle *)
    Print[ListContourPlot3D[
      N[dTotal3D],
      Contours -> quantiles,
      ContourStyle -> {
        Directive[Opacity[0.15], RGBColor[0.3, 0.3, 0.8]],
        Directive[Opacity[0.2], RGBColor[0.4, 0.5, 0.7]],
        Directive[Opacity[0.3], RGBColor[0.5, 0.6, 0.5]],
        Directive[Opacity[0.4], RGBColor[0.7, 0.5, 0.3]],
        Directive[Opacity[0.5], RGBColor[0.8, 0.4, 0.2]]
      },
      PlotRange -> All,
      AxesLabel -> {"C_{n-2}", "C_{n-1}", "C_n"},
      PlotLabel -> Style[
        Row[{"HE Sim D: 3D Void at k_\[Phi] | L = \[Phi] | Angle 2"}],
        12, Bold],
      ViewPoint -> {1.5, -2.5, 1.5},
      ImageSize -> 600,
      Boxed -> True,
      BoxRatios -> {1, 1, 1}
    ]];
  ];

  (* === VOID LOCATION MAP === *)
  Print[""];
  Print["=== 2D Void Location Map \[LongDash] L = \[Phi] ==="];

  Module[{voidMap},
    voidMap = Map[If[# == 0, 1, 0] &, dTotal, {2}];
    Print["  Non-zero entries in void map: ", Total[Flatten[voidMap]]];
    If[Total[Flatten[voidMap]] > 0,
      Print[ArrayPlot[
        Reverse[voidMap],
        PlotRange -> {0, 1},
        ColorFunction -> (If[# > 0.5, RGBColor[1, 0.2, 0.2], RGBColor[0.02, 0.02, 0.02]] &),
        Frame -> True,
        FrameLabel -> {"C_{n+1}", "C_n"},
        PlotLabel -> Style[
          Row[{"HE Sim D: Void Locations | k_\[Phi] | L = \[Phi] | ",
               densityRes, "\[Times]", densityRes}],
          12, Bold],
        ImageSize -> 600,
        AspectRatio -> 1
      ]];
    ];
  ];

  (* === SUMMARY === *)
  Print[""];
  Print["==================================================="];
  Print["  HE SIMULATION D \[LongDash] L = \[Phi] \[LongDash] COMPLETE"];
  Print["==================================================="];
  Print[""];
  Print["  Final comparison:"];
  Print[""];
  Print["  2D Void:"];
  Print["    L = 1.5:  5,828 empty (0.91%)"];
  Print["    L = \[Phi]:  ", nEmpty2D, " empty (",
    NumberForm[100.0 nEmpty2D/densityRes^2, {4, 2}], "%)"];
  Print[""];
  Print["  3D Void:"];
  Print["    L = 1.5 (Sim D, 100^3):    68.52% empty"];
  Print["    L = \[Phi] (Sim E, 150^3):    70.61% empty"];
  Print["    L = \[Phi] (this run, 100^3): ",
    NumberForm[100.0 nEmpty3D/densityRes3D^3, {5, 2}], "% empty"];
  Print[""];
  Print["  The container equals the constant."];
  Print[""];
  Print["  \[CapitalPsi]    \[FullMoon]\[Ocean]\[DeciduousTree]\[SpiderWeb]    \[YinYang]    \[Infinity]"];
]



