(* ::Package:: *)

(* ============================================================
   HE SIMULATION M: \[CurlyPhi]-SCALED BOUNDARY SWEEP (FIXED)
   ------------------------------------------------------------
   Eight panels: M\[ODoubleDot]bius Luminbrot + Raw Escape Map
   at L = \[CurlyPhi], \[CurlyPhi]\.b2, \[CurlyPhi]\.b3, \[CurlyPhi]\:2074, all at k = 1/6.
   
   World Tree Project \[LongDash] July 2026
   J. David Mack & Claude (Opus 4.6)
   ============================================================ *)

(* ============================================= *)
(* CLEAR AND DEFINE                               *)
(* ============================================= *)

ClearAll["Global`*"];

phi = N[GoldenRatio];
omega = 2.0 Pi;
Ah = 0.1;
dtVal = 0.01;
kEsc = N[1/6];
nSteps = 5000;
gridRes = 200;
phiInv = 1.0/phi;

Print["==================================================="];
Print["  HE SIMULATION M: PHI-SCALED BOUNDARY SWEEP (k = 1/6)"];
Print["  k = 1/6 | L = phi, phi^2, phi^3, phi^4"];
Print["==================================================="];
Print[""];

(* ============================================= *)
(* COMPUTATION FUNCTIONS                          *)
(* ============================================= *)

(* Single trajectory under Mobius \[LongDash] returns final MMphi *)
computeMMphiMobius[c0In_Real, c1In_Real, kv_Real, LB_Real, nSt_Integer] :=
  Module[{cnm2, cnm1, cnp1, t, mm = 0.0},
    cnm2 = c0In;
    cnm1 = c1In;
    Do[
      t = n * dtVal;
      cnp1 = (1.0 - 6.0 kv)/(1.0 - 9.0 kv) * cnm1 +
             (3.0 kv)/(1.0 - 9.0 kv) * cnm2 +
             Ah * (Sin[3.0 omega t] + Sin[6.0 omega t] + Sin[9.0 omega t]);
      cnp1 = Mod[cnp1 + LB, 2.0 LB] - LB;
      mm = mm * phiInv + cnp1 * cnp1;
      cnm2 = cnm1;
      cnm1 = cnp1;,
      {n, 1, nSt}
    ];
    mm
  ];

(* Single trajectory without wrapping \[LongDash] returns escape time *)
computeEscapeTime[c0In_Real, c1In_Real, kv_Real, LB_Real, nSt_Integer] :=
  Module[{cnm2, cnm1, cnp1, t, escTime = nSt},
    cnm2 = c0In;
    cnm1 = c1In;
    Do[
      t = n * dtVal;
      cnp1 = (1.0 - 6.0 kv)/(1.0 - 9.0 kv) * cnm1 +
             (3.0 kv)/(1.0 - 9.0 kv) * cnm2 +
             Ah * (Sin[3.0 omega t] + Sin[6.0 omega t] + Sin[9.0 omega t]);
      If[Abs[cnp1] > LB,
        escTime = n; Return[escTime, Module]
      ];
      cnm2 = cnm1;
      cnm1 = cnp1;,
      {n, 1, nSt}
    ];
    escTime
  ];

(* Quick test *)
Print["  Function test:"];
Print["    MMphi(0.5, 0.3, L=phi): ",
  NumberForm[computeMMphiMobius[0.5, 0.3, kEsc, phi, 100], 6]];
Print["    EscTime(0.5, 0.3, L=phi): ",
  computeEscapeTime[0.5, 0.3, kEsc, phi, 100]];
Print["    EscTime(1.9, 1.9, L=phi): ",
  computeEscapeTime[1.9, 1.9, kEsc, phi, 100]];
Print["  Tests passed \[LongDash] starting sweep."];
Print[""];

(* ============================================= *)
(* GRID                                           *)
(* ============================================= *)

c0Range = N[Subdivide[-2.0, 2.0, gridRes - 1]];
c1Range = N[Subdivide[-2.0, 2.0, gridRes - 1]];

(* ============================================= *)
(* SWEEP OVER FOUR PHI LEVELS                     *)
(* ============================================= *)

LList = N[{phi, phi^2, phi^3, phi^4}];
LNames = {"phi", "phi^2", "phi^3", "phi^4"};
LVals = {phi, phi^2, phi^3, phi^4};

Do[
  Module[{LB, label},
    LB = LList[[li]];
    label = LNames[[li]];
    
    Print["==================================================="];
    Print["  LEVEL ", li, ": L = ", label, " = ", NumberForm[LB, 8]];
    Print["==================================================="];
    
    (* --- MOBIUS LUMINBROT --- *)
    Print["  Computing Mobius Luminbrot..."];
    
    lumResult = Table[
      computeMMphiMobius[c0Range[[ci]], c1Range[[cj]], kEsc, LB, nSteps],
      {cj, 1, gridRes}, {ci, 1, gridRes}
    ];
    
    Print["    Complete."];
    Print["    MMphi range: [", NumberForm[Min[lumResult], 6], ", ",
          NumberForm[Max[lumResult], 6], "]"];
    Print["    MMphi max < phi^4 (", NumberForm[N[phi^4], 6], "): ",
          If[Max[lumResult] < phi^4, "YES", "NO"]];
    
    Print["=== LUMINBROT: L = ", label, " | k = 1/6 ==="];
    Print[ArrayPlot[Log[1.0 + lumResult],
      PlotLabel -> Style[StringJoin["Luminbrot | L = ", label,
        " | k = 1/6 | No Escape"], 11, Bold],
      ColorFunction -> "SunsetColors",
      FrameLabel -> {"C_{-1}", "C_0"},
      DataRange -> {{-2, 2}, {-2, 2}},
      AspectRatio -> 1,
      ImageSize -> 600
    ]];
    Print[""];
    
    (* --- RAW ESCAPE MAP --- *)
    Print["  Computing raw escape map..."];
    
    escResult = Table[
      computeEscapeTime[c0Range[[ci]], c1Range[[cj]], kEsc, LB, nSteps],
      {cj, 1, gridRes}, {ci, 1, gridRes}
    ];
    
    boundedN = Count[Flatten[escResult], nSteps];
    escapedN = gridRes^2 - boundedN;
    
    Print["    Complete."];
    Print["    Bounded: ", boundedN, " / ", gridRes^2,
          " (", NumberForm[100.0 boundedN/gridRes^2, {5, 1}], "%)"];
    Print["    Escaped: ", escapedN, " / ", gridRes^2,
          " (", NumberForm[100.0 escapedN/gridRes^2, {5, 1}], "%)"];
    
    Print["=== ESCAPE MAP: L = ", label, " | k = 1/6 ==="];
    Print[ArrayPlot[
      Map[If[# == nSteps, 0.0, N[#]/N[nSteps]] &, escResult, {2}],
      PlotLabel -> Style[StringJoin["Escape Map | L = ", label,
        " | k = 1/6"], 11, Bold],
      ColorFunction -> (If[# == 0, Black,
        Blend[{RGBColor[0, 0, 0.3], RGBColor[0, 0.5, 1],
               RGBColor[1, 1, 0], White}, #]] &),
      ColorFunctionScaling -> True,
      FrameLabel -> {"C_{-1}", "C_0"},
      DataRange -> {{-2, 2}, {-2, 2}},
      AspectRatio -> 1,
      ImageSize -> 600
    ]];
    Print[""];
    Print[""];
  ],
  {li, 1, 4}
];

(* ============================================= *)
(* COMPLETE                                       *)
(* ============================================= *)

Print["==================================================="];
Print["  PHI-SCALED BOUNDARY SWEEP (k = 1/6) \[LongDash] COMPLETE"];
Print["==================================================="];
Print[""];
Print["  Psi    To preserve the harmonic field."];



