(* ::Package:: *)

(* ============================================================
   SIMULATION G: MM_phi Power Level Tracking
   ------------------------------------------------------------
   Falsifiable prediction:
     At L = 1.5: phi^4 is IMPOSSIBLE (max MM_phi_ss < phi^4)
     At L = phi: phi^4 is ACHIEVABLE (max MM_phi_ss = phi^4)
   
   Mathematical basis:
     MM_phi_ss = <|C_n|^2> * phi^2
     Max <|C_n|^2> = L^2 (trajectory bounded by [-L, L])
     L = 1.5:  max MM_phi_ss = 2.25 * phi^2 = 5.89 < phi^4 = 6.854
     L = phi:  max MM_phi_ss = phi^2 * phi^2 = phi^4 = 6.854
   
   Method:
     Run trajectories at k_phi under both L values.
     Compute MM_phi at EVERY step.
     Track which phi-power levels are sustained.
     Report whether phi^4 is reached, and for how long.
   
   World Tree Project - June 2026
   J. David Mack & Claude (Opus 4.6)
   ============================================================ *)

Block[
  {alpha, beta, omega, omegaRes, Ah, noiseScale, dt,
   pertFunc, harmFunc, kBdy, maxIter, scanRes,
   phi, phi2, phi3, phi4,
   c0Grid, c1Grid,
   runConfig},

  (* F_c coefficients *)
  alpha[kv_] := (1 - 6 kv)/(1 - 9 kv);
  beta[kv_] := (3 kv)/(1 - 9 kv);

  (* Parameters *)
  omega = 2 Pi;
  omegaRes = 2 Pi;
  Ah = 0.1;
  noiseScale = 0.2;
  dt = 0.01;

  kBdy = N[GoldenRatio / (9 * GoldenRatio - 3)];

  (* Phi power thresholds *)
  phi = N[GoldenRatio];
  phi2 = phi^2;
  phi3 = phi^3;
  phi4 = phi^4;

  maxIter = 500;  (* Longer runs to see sustained behavior *)
  scanRes = 300;  (* 300x300 = 90,000 trajectories *)

  pertFunc[cnVal_, tVal_] := Which[
    cnVal < 0, Abs[cnVal] * noiseScale * RandomReal[{-1, 1}],
    cnVal <= 1, 0,
    True, (cnVal - 1) * Sin[omegaRes * tVal]
  ];
  harmFunc[tVal_] := Ah * (Sin[3 omega tVal] + Sin[6 omega tVal] + Sin[9 omega tVal]);

  c0Grid = Subdivide[-2.0, 2.0, scanRes - 1];
  c1Grid = Subdivide[-2.0, 2.0, scanRes - 1];

  Print["====================================================="];
  Print["  SIMULATION G: MM_\[Phi] Power Level Tracking"];
  Print["====================================================="];
  Print[""];
  Print["  PREDICTION:"];
  Print["    L = 1.5:  max MM_\[Phi]_ss = ", NumberForm[1.5^2 * phi2, 6],
    " < \[Phi]^4 = ", NumberForm[phi4, 6], "  -> \[Phi]^4 IMPOSSIBLE"];
  Print["    L = \[Phi]:   max MM_\[Phi]_ss = ", NumberForm[phi2 * phi2, 6],
    " = \[Phi]^4 = ", NumberForm[phi4, 6], "  -> \[Phi]^4 ACHIEVABLE"];
  Print[""];
  Print["  \[Phi]-power thresholds:"];
  Print["    \[Phi]^1 = ", NumberForm[phi, 6]];
  Print["    \[Phi]^2 = ", NumberForm[phi2, 6]];
  Print["    \[Phi]^3 = ", NumberForm[phi3, 6]];
  Print["    \[Phi]^4 = ", NumberForm[phi4, 6]];
  Print[""];

  (* === CONFIGURATION RUNNER === *)
  runConfig[LVal_, label_] := Module[
    {L, traj, mmPhi, mmPhiTimeSeries,
     pB, cB, nxB, tB, rawB,
     ii, jj, nn,
     allMaxMM, allMeanMM, allFinalMM,
     timePhi1, timePhi2, timePhi3, timePhi4,
     totalSteps, pixelCount,
     sampleMMtrace, sampleTraj,
     phiZoneHist},

    L = LVal;

    Print["====================================================="];
    Print["  ", label];
    Print["  k = k_\[Phi] = ", NumberForm[kBdy, 8]];
    Print["  L = ", NumberForm[L, 10]];
    Print["  Max possible <|C_n|^2> = L^2 = ", NumberForm[L^2, 6]];
    Print["  Max possible MM_\[Phi]_ss = L^2 * \[Phi]^2 = ", NumberForm[L^2 * phi2, 6]];
    Print["  Iterations: ", maxIter, " per trajectory"];
    Print["  Scan grid: ", scanRes, "x", scanRes, " = ", scanRes^2];
    Print["====================================================="];
    Print[""];

    allMaxMM = {};
    allMeanMM = {};
    allFinalMM = {};
    timePhi1 = 0; timePhi2 = 0; timePhi3 = 0; timePhi4 = 0;
    totalSteps = 0;
    pixelCount = 0;

    (* Store one sample trace for plotting *)
    sampleMMtrace = {};
    sampleTraj = {};

    Do[
      Do[
        pixelCount++;
        If[Mod[pixelCount, 10000] == 0,
          Print["  ", pixelCount, "/", scanRes^2]
        ];

        SeedRandom[ii * 1000 + jj];
        pB = c0Grid[[jj]];
        cB = c1Grid[[ii]];
        mmPhi = 0.0;
        mmPhiTimeSeries = Table[0.0, maxIter];

        Do[
          tB = nn * dt;
          rawB = alpha[kBdy] * cB + beta[kBdy] * pB +
                 pertFunc[cB, tB] + harmFunc[tB];
          nxB = Mod[rawB + L, 2 L] - L;

          (* Update MM_phi *)
          mmPhi = mmPhi * (1/phi) + Abs[nxB]^2;
          mmPhiTimeSeries[[nn]] = mmPhi;

          (* Count phi-power levels *)
          totalSteps++;
          If[mmPhi >= phi, timePhi1++];
          If[mmPhi >= phi2, timePhi2++];
          If[mmPhi >= phi3, timePhi3++];
          If[mmPhi >= phi4, timePhi4++];

          pB = cB;
          cB = nxB,
          {nn, 1, maxIter}
        ];

        AppendTo[allMaxMM, Max[mmPhiTimeSeries]];
        AppendTo[allMeanMM, Mean[mmPhiTimeSeries]];
        AppendTo[allFinalMM, mmPhi];

        (* Save first trajectory as sample *)
        If[pixelCount == 1,
          sampleMMtrace = mmPhiTimeSeries;
        ];

        (* Save the trajectory that achieves highest MM_phi *)
        If[pixelCount > 1 && Max[mmPhiTimeSeries] > Max[sampleMMtrace],
          sampleMMtrace = mmPhiTimeSeries;
        ],
        {jj, 1, scanRes}
      ],
      {ii, 1, scanRes}
    ];

    Print[""];
    Print["=== RESULTS: ", label, " ==="];
    Print[""];
    Print["  Total steps computed: ", totalSteps];
    Print[""];
    Print["  MM_\[Phi] statistics across all trajectories:"];
    Print["    Max MM_\[Phi] achieved:   ", NumberForm[Max[allMaxMM], 6]];
    Print["    Mean of max MM_\[Phi]:    ", NumberForm[Mean[allMaxMM], 6]];
    Print["    Mean of final MM_\[Phi]:  ", NumberForm[Mean[allFinalMM], 6]];
    Print["    Mean of mean MM_\[Phi]:   ", NumberForm[Mean[allMeanMM], 6]];
    Print[""];
    Print["  \[Phi]-power level occupancy (% of total steps):"];
    Print["    \[Phi]^1 (", NumberForm[phi, 5], "): ",
      NumberForm[100.0 timePhi1/totalSteps, {5, 2}], "%"];
    Print["    \[Phi]^2 (", NumberForm[phi2, 5], "): ",
      NumberForm[100.0 timePhi2/totalSteps, {5, 2}], "%"];
    Print["    \[Phi]^3 (", NumberForm[phi3, 5], "): ",
      NumberForm[100.0 timePhi3/totalSteps, {5, 2}], "%"];
    Print["    \[Phi]^4 (", NumberForm[phi4, 5], "): ",
      NumberForm[100.0 timePhi4/totalSteps, {5, 2}], "%"];
    Print[""];

    (* Did phi^4 occur? *)
    If[timePhi4 > 0,
      Print["  *** \[Phi]^4 REACHED ***"];
      Print["  Sustained for ", timePhi4, " steps (",
        NumberForm[100.0 timePhi4/totalSteps, {5, 4}], "% of total)"];
      Print["  Number of trajectories reaching \[Phi]^4: ",
        Count[allMaxMM, _?(# >= phi4 &)]],
      Print["  \[Phi]^4 NOT REACHED (max = ", NumberForm[Max[allMaxMM], 6],
        ", threshold = ", NumberForm[phi4, 6], ")"];
      Print["  Deficit: ", NumberForm[phi4 - Max[allMaxMM], 6]];
    ];
    Print[""];

    (* Histogram of max MM_phi values *)
    Print["=== Max MM_\[Phi] Distribution ==="];
    Print[Histogram[allMaxMM, 50,
      PlotLabel -> Style[Row[{label, " | Max MM_\[Phi] per trajectory"}], 12, Bold],
      AxesLabel -> {"Max MM_\[Phi]", "Count"},
      PlotRange -> All,
      GridLines -> {{phi, phi2, phi3, phi4}, None},
      GridLinesStyle -> Directive[Red, Dashed, Thick],
      Epilog -> {
        Text[Style["\[Phi]^1", 10, Red], {phi, -100}, {0, -1}],
        Text[Style["\[Phi]^2", 10, Red], {phi2, -100}, {0, -1}],
        Text[Style["\[Phi]^3", 10, Red], {phi3, -100}, {0, -1}],
        Text[Style["\[Phi]^4", 10, Red], {phi4, -100}, {0, -1}]
      },
      ImageSize -> 600
    ]];

    (* Best trajectory MM_phi over time *)
    Print[""];
    Print["=== Best Trajectory: MM_\[Phi] Over Time ==="];
    Print[ListLinePlot[sampleMMtrace,
      PlotLabel -> Style[
        Row[{label, " | Highest MM_\[Phi] trajectory"}], 12, Bold],
      AxesLabel -> {"Iteration", "MM_\[Phi]"},
      PlotStyle -> {Blue, Thickness[0.003]},
      PlotRange -> {0, Max[Max[sampleMMtrace] * 1.1, phi4 * 1.05]},
      GridLines -> {{}, {phi, phi2, phi3, phi4}},
      GridLinesStyle -> Directive[Red, Dashed],
      Epilog -> {
        Text[Style["\[Phi]^1", 9, Red], {maxIter * 0.95, phi}, {1, -1}],
        Text[Style["\[Phi]^2", 9, Red], {maxIter * 0.95, phi2}, {1, -1}],
        Text[Style["\[Phi]^3", 9, Red], {maxIter * 0.95, phi3}, {1, -1}],
        Text[Style["\[Phi]^4", 9, Red], {maxIter * 0.95, phi4}, {1, -1}]
      },
      ImageSize -> 700
    ]];

    (* Phi-zone pie chart *)
    Print[""];
    Print["=== \[Phi]-Zone Distribution ==="];
    phiZoneHist = {
      totalSteps - timePhi1,
      timePhi1 - timePhi2,
      timePhi2 - timePhi3,
      timePhi3 - timePhi4,
      timePhi4
    };
    Print["  Below \[Phi]^1: ", phiZoneHist[[1]], " (",
      NumberForm[100.0 phiZoneHist[[1]]/totalSteps, {5, 2}], "%)"];
    Print["  \[Phi]^1 to \[Phi]^2: ", phiZoneHist[[2]], " (",
      NumberForm[100.0 phiZoneHist[[2]]/totalSteps, {5, 2}], "%)"];
    Print["  \[Phi]^2 to \[Phi]^3: ", phiZoneHist[[3]], " (",
      NumberForm[100.0 phiZoneHist[[3]]/totalSteps, {5, 2}], "%)"];
    Print["  \[Phi]^3 to \[Phi]^4: ", phiZoneHist[[4]], " (",
      NumberForm[100.0 phiZoneHist[[4]]/totalSteps, {5, 2}], "%)"];
    Print["  At \[Phi]^4: ", phiZoneHist[[5]], " (",
      NumberForm[100.0 phiZoneHist[[5]]/totalSteps, {5, 2}], "%)"];

    Print[""];
    Print["  ", label, " COMPLETE"];
    Print[""];
  ];

  (* === RUN BOTH CONFIGURATIONS === *)

  runConfig[1.5, "CONFIG A: L = 1.5 (\[Phi]^4 predicted IMPOSSIBLE)"];

  runConfig[N[GoldenRatio], "CONFIG B: L = \[Phi] (\[Phi]^4 predicted ACHIEVABLE)"];

  Print["====================================================="];
  Print["  SIMULATION G COMPLETE"];
  Print["====================================================="];
  Print[""];
  Print["  PREDICTION VERIFICATION:"];
  Print["    L = 1.5: max possible MM_\[Phi]_ss = ", NumberForm[1.5^2 * phi2, 6],
    " < \[Phi]^4 = ", NumberForm[phi4, 6]];
  Print["    L = \[Phi]:   max possible MM_\[Phi]_ss = ", NumberForm[phi2 * phi2, 6],
    " = \[Phi]^4 = ", NumberForm[phi4, 6]];
  Print[""];
  Print["  If L = 1.5 never reaches \[Phi]^4 and L = \[Phi] does:"];
  Print["    -> L = \[Phi] is the MINIMUM boundary for the fourth level"];
  Print["    -> The containment boundary must be \[Phi]-scaled"];
  Print["       for the full power hierarchy to be accessible"];
  Print["    -> The impedance match IS the \[Phi]^4 condition"];
]



