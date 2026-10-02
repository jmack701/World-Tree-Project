(* ::Package:: *)

(* ============================================================
   HE SIMULATION M: LUMINBROT 3D
   ------------------------------------------------------------
   Two 3D visualizations:
   
   PART 1: HEIGHT MAP
   Takes the 2D Luminbrot (C_0 \[Times] C_{-1}) at k = 1/6, L = \[CurlyPhi]
   and renders it as a 3D surface where height = Log[1 + MM\[CurlyPhi]].
   Uses the data already computed from the boundary sweep.
   If lumResult is not in memory, recomputes at 200\[Times]200.
   
   PART 2: ISO-SURFACE
   3D grid over (C_0, C_{-1}, k) with k swept from 0.12 to 0.18.
   MM\[CurlyPhi] computed at each point. Iso-surfaces rendered at the
   four \[CurlyPhi]-power thresholds: \[CurlyPhi]\.b9, \[CurlyPhi]\.b2, \[CurlyPhi]\.b3, \[CurlyPhi]\:2074.
   Nested shells of equal phi-weighted memory in the
   (initial condition, gain) space.
   
   All under M\[ODoubleDot]bius boundary, L = \[CurlyPhi].
   
   World Tree Project \[LongDash] July 2026
   J. David Mack & Claude (Opus 4.6)
   ============================================================ *)

ClearAll["Global`*"];

phi = N[GoldenRatio];
omega = 2.0 Pi;
Ah = 0.1;
dtVal = 0.01;
phiInv = 1.0/phi;
LB = phi;

Print["==================================================="];
Print["  HE SIMULATION M: LUMINBROT 3D"];
Print["  Height Map + Iso-Surface"];
Print["==================================================="];
Print[""];

(* ============================================= *)
(* SHARED COMPUTATION FUNCTION                    *)
(* ============================================= *)

computeMMphiMobius[c0In_Real, c1In_Real, kv_Real, L_Real, nSt_Integer] :=
  Module[{cnm2, cnm1, cnp1, t, mm = 0.0},
    cnm2 = c0In;
    cnm1 = c1In;
    Do[
      t = n * dtVal;
      cnp1 = (1.0 - 6.0 kv)/(1.0 - 9.0 kv) * cnm1 +
             (3.0 kv)/(1.0 - 9.0 kv) * cnm2 +
             Ah * (Sin[3.0 omega t] + Sin[6.0 omega t] + Sin[9.0 omega t]);
      cnp1 = Mod[cnp1 + L, 2.0 L] - L;
      mm = mm * phiInv + cnp1 * cnp1;
      cnm2 = cnm1;
      cnm1 = cnp1;,
      {n, 1, nSt}
    ];
    mm
  ];

(* Quick test *)
Print["  Function test: ", 
  NumberForm[computeMMphiMobius[0.5, 0.3, 0.165, phi, 100], 6]];
Print["  Test passed."];
Print[""];

(* ============================================= *)
(* PART 1: HEIGHT MAP                             *)
(* ============================================= *)

Print["==================================================="];
Print["  PART 1: LUMINBROT HEIGHT MAP"];
Print["  k = 1/6 | L = phi | 200 x 200"];
Print["==================================================="];
Print[""];

gridRes2D = 200;
nSteps2D = 5000;
kHM = N[1/6];

c0Range2D = N[Subdivide[-2.0, 2.0, gridRes2D - 1]];
c1Range2D = N[Subdivide[-2.0, 2.0, gridRes2D - 1]];

Print["  Computing 2D Luminbrot for height map..."];

lumData2D = Table[
  computeMMphiMobius[c0Range2D[[ci]], c1Range2D[[cj]], kHM, LB, nSteps2D],
  {cj, 1, gridRes2D}, {ci, 1, gridRes2D}
];

Print["  Complete."];
Print["  MMphi range: [", NumberForm[Min[lumData2D], 6], ", ",
      NumberForm[Max[lumData2D], 6], "]"];
Print[""];

(* 3D Surface Plot *)
Print["=== LUMINBROT HEIGHT MAP: k = 1/6 | L = phi ==="];
Print[ListPlot3D[Log[1.0 + lumData2D],
  PlotLabel -> Style["Luminbrot Height Map | k = 1/6 | L = \[Phi]", 12, Bold],
  AxesLabel -> {"C_0", "C_{-1}", "Log(1+MM\[Phi])"},
  ColorFunction -> "SunsetColors",
  ColorFunctionScaling -> True,
  MeshNone,
  PlotRange -> All,
  BoxRatios -> {1, 1, 0.6},
  ImageSize -> 700,
  ViewPoint -> {-2.5, -2.0, 1.5}
]];
Print[""];

(* Additional angle \[LongDash] top-down for comparison with 2D *)
Print["=== LUMINBROT HEIGHT MAP: Top View ==="];
Print[ListPlot3D[Log[1.0 + lumData2D],
  PlotLabel -> Style["Luminbrot Height Map | Top View", 12, Bold],
  AxesLabel -> {"C_0", "C_{-1}", "Log(1+MM\[Phi])"},
  ColorFunction -> "SunsetColors",
  ColorFunctionScaling -> True,
  MeshNone,
  PlotRange -> All,
  BoxRatios -> {1, 1, 0.6},
  ImageSize -> 700,
  ViewPoint -> {0, 0, 3}
]];
Print[""];

(* Side view showing the valley depth *)
Print["=== LUMINBROT HEIGHT MAP: Side View ==="];
Print[ListPlot3D[Log[1.0 + lumData2D],
  PlotLabel -> Style["Luminbrot Height Map | Side View", 12, Bold],
  AxesLabel -> {"C_0", "C_{-1}", "Log(1+MM\[Phi])"},
  ColorFunction -> "SunsetColors",
  ColorFunctionScaling -> True,
  MeshNone,
  PlotRange -> All,
  BoxRatios -> {1, 1, 0.6},
  ImageSize -> 700,
  ViewPoint -> {0, -3, 0.5}
]];
Print[""];

(* ============================================= *)
(* PART 2: ISO-SURFACE                            *)
(* ============================================= *)

Print["==================================================="];
Print["  PART 2: LUMINBROT ISO-SURFACE"];
Print["  C_0 x C_{-1} x k | L = phi"];
Print["  k range: 0.12 to 0.18 (across bifurcation)"];
Print["==================================================="];
Print[""];

gridRes3D = 50;
nSteps3D = 3000;

c0Range3D = N[Subdivide[-1.8, 1.8, gridRes3D - 1]];
c1Range3D = N[Subdivide[-1.8, 1.8, gridRes3D - 1]];
kRange3D = N[Subdivide[0.12, 0.18, gridRes3D - 1]];

Print["  Grid: ", gridRes3D, " x ", gridRes3D, " x ", gridRes3D,
      " = ", gridRes3D^3, " points"];
Print["  Steps per trajectory: ", nSteps3D];
Print["  Total iterations: ", gridRes3D^3 * nSteps3D];
Print[""];
Print["  Computing 3D MMphi field..."];
Print["  This will take a while."];

mmField3D = Table[
  computeMMphiMobius[c0Range3D[[ci]], c1Range3D[[cj]], kRange3D[[ki]], LB, nSteps3D],
  {ki, 1, gridRes3D}, {cj, 1, gridRes3D}, {ci, 1, gridRes3D}
];

Print["  Complete."];
Print["  MMphi range: [", NumberForm[Min[mmField3D], 6], ", ",
      NumberForm[Max[mmField3D], 6], "]"];
Print[""];

(* Iso-levels at phi powers *)
isoLevels = N[{phi, phi^2, phi^3, phi^4}];
isoLabels = {"\[Phi]^1 = 1.618", "\[Phi]^2 = 2.618", 
             "\[Phi]^3 = 4.236", "\[Phi]^4 = 6.854"};
isoColors = {RGBColor[0.2, 0.4, 1.0], RGBColor[0.1, 0.8, 0.3],
             RGBColor[1.0, 0.75, 0.0], RGBColor[1.0, 0.2, 0.2]};

Print["  Iso-levels: ", isoLevels];

(* Check which levels are present in the data *)
Do[
  Print["    ", isoLabels[[i]], ": ",
    If[Max[mmField3D] >= isoLevels[[i]],
      "PRESENT in data",
      "ABOVE maximum \[LongDash] not reachable"
    ]
  ],
  {i, 1, 4}
];
Print[""];

(* Render iso-surfaces *)
Print["=== LUMINBROT ISO-SURFACE: Nested \[Phi]-Power Shells ==="];

(* Build interpolation function *)
mmInterp = ListInterpolation[mmField3D,
  {{0.12, 0.18}, {-1.8, 1.8}, {-1.8, 1.8}}
];

(* Find which iso-levels are actually achievable *)
achievableLevels = Select[isoLevels, # <= Max[mmField3D] &];
achievableColors = Take[isoColors, Length[achievableLevels]];
achievableLabels = Take[isoLabels, Length[achievableLevels]];

If[Length[achievableLevels] > 0,
  Print[Show[
    Table[
      ContourPlot3D[mmInterp[k, c1, c0] == achievableLevels[[i]],
        {k, 0.12, 0.18}, {c1, -1.8, 1.8}, {c0, -1.8, 1.8},
        ContourStyle -> {achievableColors[[i]], Opacity[0.3], Specularity[White, 20]},
        Mesh -> None
      ],
      {i, 1, Length[achievableLevels]}
    ],
    PlotLabel -> Style["Luminbrot Iso-Surface | \[Phi]-Power Shells | L = \[Phi]", 12, Bold],
    AxesLabel -> {"k (gain)", "C_{-1}", "C_0"},
    BoxRatios -> {1, 1, 1},
    ImageSize -> 750,
    ViewPoint -> {2.5, -2.0, 1.5},
    Lighting -> "Accent"
  ]];,
  Print["  No iso-levels achievable in data range."];
];
Print[""];

(* Individual shells for clarity *)
Do[
  If[Max[mmField3D] >= achievableLevels[[i]],
    Print["=== ISO-SURFACE: MM\[Phi] = ", achievableLabels[[i]], " ==="];
    Print[ContourPlot3D[mmInterp[k, c1, c0] == achievableLevels[[i]],
      {k, 0.12, 0.18}, {c1, -1.8, 1.8}, {c0, -1.8, 1.8},
      ContourStyle -> {achievableColors[[i]], Opacity[0.5], Specularity[White, 20]},
      Mesh -> None,
      PlotLabel -> Style[StringJoin["MM\[Phi] = ", achievableLabels[[i]], 
        " Shell | L = \[Phi]"], 11, Bold],
      AxesLabel -> {"k", "C_{-1}", "C_0"},
      BoxRatios -> {1, 1, 1},
      ImageSize -> 600,
      ViewPoint -> {2.5, -2.0, 1.5}
    ]];
    Print[""];
  ],
  {i, 1, Length[achievableLevels]}
];

(* ============================================= *)
(* SUMMARY                                        *)
(* ============================================= *)

Print["==================================================="];
Print["  HE SIMULATION M: LUMINBROT 3D \[LongDash] COMPLETE"];
Print["==================================================="];
Print[""];
Print["  Part 1: Height map from 200x200 grid (k = 1/6)"];
Print["    Three views: perspective, top, side"];
Print[""];
Print["  Part 2: Iso-surface from 50x50x50 grid"];
Print["    k range: 0.12 \[LongDash] 0.18 (across bifurcation)"];
Print["    Shells at phi-power levels"];
Print[""];
Print["  Psi    To preserve the harmonic field."];



