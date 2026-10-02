(* ::Package:: *)

(* \:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550 *)
(* \[CapitalPsi] ENERGY GRID k-SWEEP                                     *)
(* Phi-spiral grid with F_c harmonic compensation             *)
(* Fine resolution near k = 2/9 bifurcation                   *)
(* For the \[Minus]6 paper: Chaos Invariant across the operating range *)
(* World Tree Project \[CenterDot] July 2026                              *)
(* \:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550 *)

(* === GRID PARAMETERS === *)
phi = GoldenRatio // N;
nNodes = 100;          (* Grid size \[LongDash] 100 nodes *)
nNeighbors = 6;        (* Each node couples to 6 nearest neighbors *)
coupling = 0.005;      (* Coupling strength *)
L = phi;               (* M\[ODoubleDot]bius boundary *)
harmonicFrac = 0.3;    (* Harmonic compensation fraction *)
nSteps = 10000;        (* Steps per k value *)
nTransient = 2000;     (* Discard transient *)
dt = 0.01;
omega = 2 Pi;

(* k values: coarse through operating range, fine near 2/9 *)
kCritical = 2/9 // N;  (* 0.22222... *)
kValues = Join[
  Range[0.200, 0.215, 0.005],     (* Below operating range *)
  Range[0.218, 0.226, 0.001],     (* Fine near operating point and bifurcation *)
  Range[0.228, 0.240, 0.002]      (* Past bifurcation *)
];

Print["\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
Print["ENERGY GRID k-SWEEP"];
Print["\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
Print["Nodes: ", nNodes, "  Neighbors: ", nNeighbors, "  Coupling: ", coupling];
Print["L = ", L, "  Harmonic fraction: ", harmonicFrac];
Print["Steps: ", nSteps, " (transient: ", nTransient, ")"];
Print["k values: ", Length[kValues], " from ", First[kValues], " to ", Last[kValues]];
Print["k critical (2/9): ", kCritical];
Print[""];

(* === BUILD PHI-SPIRAL GRID === *)
(* Nodes at golden angle increments *)
nodePositions = Table[
  Module[{r = Sqrt[n / nNodes], theta = n * GoldenAngle // N},
    {r Cos[theta], r Sin[theta]}
  ], {n, 1, nNodes}];

(* Neighbor list: k nearest by Euclidean distance *)
neighborList = Table[
  Module[{dists, sorted},
    dists = Table[
      If[i == j, Infinity, Norm[nodePositions[[i]] - nodePositions[[j]]]],
      {j, 1, nNodes}];
    sorted = Ordering[dists];
    Take[sorted, Min[nNeighbors, nNodes - 1]]
  ], {i, 1, nNodes}];

Print["Grid built: ", nNodes, " nodes, golden angle spacing"];
Print[""];

(* === SPECTRAL ENTROPY FUNCTION === *)
spectralEntropy[voltages_] := Module[
  {fft, power, pNorm, se},
  fft = Abs[Fourier[voltages]];
  power = fft^2;
  If[Total[power] < 10^-20, Return[0.0]];
  pNorm = power / Total[power];
  se = -Total[If[# > 10^-15, # Log[2, #], 0] & /@ pNorm];
  se / Log[2, Length[voltages]]
];

(* === VOLTAGE STABILITY === *)
voltageStability[voltages_] := Module[
  {dev = StandardDeviation[voltages]},
  If[dev < 10^-10, 1.0, Max[0, 1 - dev]]
];

(* === MAIN k-SWEEP === *)
Print["Running k-sweep..."];
Print[""];

results = {};

Do[
  SeedRandom[42];
  
  (* Initialize voltages *)
  v = ConstantArray[1.0, nNodes];
  vPrev = ConstantArray[1.0, nNodes];
  t = 0;
  
  (* Accumulators *)
  seValues = {};
  stabValues = {};
  chaosCount = 0;
  eqCount = 0;
  resCount = 0;
  totalCount = 0;
  maxAmplitude = 0;
  
  Do[
    t = step * dt;
    
    (* Compute deviations from neighbors *)
    deviations = Table[
      Module[{nbrs = neighborList[[i]], meanV},
        meanV = Mean[v[[nbrs]]];
        v[[i]] - meanV
      ], {i, 1, nNodes}];
    
    (* Harmonic enrichment at grid level *)
    H = harmonicFrac * (Sin[3 omega t] + Sin[6 omega t] + Sin[9 omega t]);
    
    (* Two-term causal correction: correction = k * (3*devPrev - 6*devCurr) *)
    corrections = Table[
      Module[{devCurr, devPrev, correction, vRaw},
        devCurr = deviations[[i]];
        devPrev = v[[i]] - If[Length[neighborList[[i]]] > 0, 
          Mean[vPrev[[neighborList[[i]]]]], vPrev[[i]]];
        
        correction = kVal * (3 * devPrev - 6 * devCurr);
        vRaw = v[[i]] + correction * coupling + H * coupling;
        
        (* M\[ODoubleDot]bius wrapping *)
        Mod[vRaw + L, 2 L] - L
      ], {i, 1, nNodes}];
    
    vPrev = v;
    v = corrections;
    
    (* Record after transient *)
    If[step > nTransient,
      se = spectralEntropy[v];
      stab = voltageStability[v];
      AppendTo[seValues, se];
      AppendTo[stabValues, stab];
      
      (* Regime classification per node *)
      Do[
        If[v[[i]] < 0, chaosCount++,
          If[v[[i]] <= 1, eqCount++, resCount++]];
        totalCount++;
      , {i, 1, nNodes}];
      
      maxAmplitude = Max[maxAmplitude, Max[Abs[v]]];
    ];
  , {step, 1, nSteps}];
  
  (* Compute averages *)
  meanSE = If[Length[seValues] > 0, Mean[seValues], 0];
  stdSE = If[Length[seValues] > 1, StandardDeviation[seValues], 0];
  meanStab = If[Length[stabValues] > 0, Mean[stabValues], 0];
  chaosP = 100.0 chaosCount / Max[1, totalCount];
  eqP = 100.0 eqCount / Max[1, totalCount];
  resP = 100.0 resCount / Max[1, totalCount];
  
  AppendTo[results, <|
    "k" -> kVal,
    "dist_from_2_9" -> Abs[kVal - kCritical],
    "SE_mean" -> meanSE,
    "SE_std" -> stdSE,
    "Stability" -> meanStab,
    "Chaos%" -> chaosP,
    "Eq%" -> eqP,
    "Res%" -> resP,
    "MaxAmp" -> maxAmplitude
  |>];
  
  Print[StringForm["k = ``  |k-2/9| = ``  SE = `` \[PlusMinus] ``  Stab = ``  C:``% E:``% R:``%",
    NumberForm[kVal, {5, 4}],
    NumberForm[Abs[kVal - kCritical], {5, 4}],
    NumberForm[meanSE, {5, 4}],
    NumberForm[stdSE, {5, 4}],
    NumberForm[meanStab, {5, 4}],
    NumberForm[chaosP, {4, 1}],
    NumberForm[eqP, {4, 1}],
    NumberForm[resP, {4, 1}]
  ]];
, {kVal, kValues}];

(* === RESULTS TABLE === *)
Print[""];
Print["\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
Print["COMPLETE RESULTS"];
Print["\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
Print[""];

(* === PLOTS === *)
Print["Generating plots..."];

sePlot = ListLinePlot[
  {#["k"], #["SE_mean"]} & /@ results,
  PlotLabel -> "Spectral Entropy vs k (Energy Grid, N=100)",
  AxesLabel -> {"k", "SE"},
  PlotStyle -> Blue,
  PlotRange -> {All, {0, 1}},
  GridLines -> {{kCritical}, {0.495}},
  GridLinesStyle -> {{Directive[Red, Dashed]}, {Directive[Green, Dashed]}},
  Epilog -> {
    Text["k = 2/9", {kCritical, 0.05}, {-1.5, 0}],
    Text["SE = 0.495", {First[kValues], 0.505}, {-1, 0}]
  },
  ImageSize -> 500
];
Print[sePlot];

stabPlot = ListLinePlot[
  {#["k"], #["Stability"]} & /@ results,
  PlotLabel -> "Voltage Stability vs k",
  AxesLabel -> {"k", "Stability"},
  PlotStyle -> Red,
  PlotRange -> {All, {0, 1.1}},
  GridLines -> {{kCritical}, {0.495}},
  GridLinesStyle -> {{Directive[Red, Dashed]}, {Directive[Green, Dashed]}},
  ImageSize -> 500
];
Print[stabPlot];

combinedPlot = Show[
  ListLinePlot[
    {{#["k"], #["SE_mean"]} & /@ results,
     {#["k"], #["Stability"]} & /@ results},
    PlotLabel -> "SE and Stability vs k \[LongDash] Energy Grid",
    AxesLabel -> {"k", ""},
    PlotStyle -> {Blue, Red},
    PlotLegends -> {"SE", "Stability"},
    PlotRange -> {All, {0, 1.1}},
    GridLines -> {{kCritical}, {0.495}},
    GridLinesStyle -> {{Directive[Orange, Dashed]}, {Directive[Green, Dashed]}},
    Epilog -> {Text["k = 2/9", {kCritical + 0.002, 0.05}]},
    ImageSize -> 600
  ]
];
Print[combinedPlot];

regimePlot = ListLinePlot[
  {{#["k"], #["Chaos%"]} & /@ results,
   {#["k"], #["Eq%"]} & /@ results,
   {#["k"], #["Res%"]} & /@ results},
  PlotLabel -> "Regime Distribution vs k \[LongDash] Energy Grid",
  AxesLabel -> {"k", "%"},
  PlotStyle -> {Red, Green, Blue},
  PlotLegends -> {"Chaos", "Equilibrium", "Resonance"},
  PlotRange -> {All, All},
  GridLines -> {{kCritical}, {49.5}},
  GridLinesStyle -> {{Directive[Orange, Dashed]}, {Directive[Gray, Dashed]}},
  ImageSize -> 600
];
Print[regimePlot];

(* === SUMMARY === *)
Print[""];
Print["\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];
Print["SUMMARY"];
Print["\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550\:2550"];

(* Find the operating point (SE closest to 0.495) *)
operatingIdx = Ordering[Abs[#["SE_mean"] - 0.495] & /@ results][[1]];
operatingK = results[[operatingIdx]]["k"];
operatingSE = results[[operatingIdx]]["SE_mean"];

(* Find the bifurcation point (stability drops below 0.5) *)
bifIdx = SelectFirst[Range[Length[results]], results[[#]]["Stability"] < 0.5 &];

Print["Operating point: k = ", operatingK, ", SE = ", operatingSE];
If[IntegerQ[bifIdx],
  Print["Bifurcation: k = ", results[[bifIdx]]["k"], 
    ", Stability = ", results[[bifIdx]]["Stability"]];
,
  Print["Bifurcation not reached in sweep range"];
];
Print["k = 2/9 = ", kCritical];
Print[""];
Print["SE at operating point vs invariant: ", 
  Abs[operatingSE - 0.495], " deviation"];
Print[""];

(* Check for quasi-periodic dip *)
seVals = #["SE_mean"] & /@ results;
minSEIdx = Ordering[seVals][[1]];
Print["Minimum SE: ", seVals[[minSEIdx]], " at k = ", results[[minSEIdx]]["k"]];
Print["Does a quasi-periodic dip exist? ", 
  If[seVals[[minSEIdx]] < 0.45, "YES \[LongDash] SE drops below 0.45", 
    If[seVals[[minSEIdx]] < 0.48, "POSSIBLE \[LongDash] SE drops below 0.48",
      "NO \[LongDash] SE stays in the invariant band"]]];

Print[""];
Print["\[CapitalPsi]"];
Print["\:262f To preserve the harmonic field."];



