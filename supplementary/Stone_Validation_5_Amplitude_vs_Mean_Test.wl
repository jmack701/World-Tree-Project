(* ============================================================
   Stone Validation 5  --  Amplitude vs. Mean Test
   ------------------------------------------------------------
   Question being settled:
   When the 3-6-9 harmonics (Sin[3wt]+Sin[6wt]+Sin[9wt]) are
   added to the Lumin Equation, do they reduce the AMPLITUDE
   (size of oscillation) of the signal, or only shift its MEAN
   (the DC / average level)?
   The Harmonic Enrichment draft cites Stone 5's -49.5% as an
   "amplitude reduction." But the original Stone 5 percentage was
   computed from Mean[Flatten[...]] -- a mean, not an amplitude.
   This script computes BOTH so we can see which one actually moved.
   Outcome rule we agreed on:
     - If Part B amplitudes drop by ~30-50%, there IS a real
       quantitative bridge to the draft's F_c result. Integrate it.
     - If only Part A (the mean) moves while Part B amplitudes
       barely change (or rise), Stone 5 measured a mean shift, not
       an amplitude reduction, and the two numbers should not be
       presented as the same phenomenon.
   ============================================================ *)
ClearAll[M, phi, omega, S, ph, feedback, psi, mc2, LuminWithout, LuminWith];
(* --- Core parameters: IDENTICAL to Mathematica Stone Validation 5 --- *)
M = 1;
phi = (1 + Sqrt[5])/2;        (* golden ratio *)
omega = 1;
S = Pi/4;                     (* symmetry constant *)
ph = Pi/6;                    (* the phase shift Stone 5 wrote as \[Phi] *)
feedback[i_] := 0.2 (i - 1);
psi[i_] := Sin[i];
mc2 = 1;
(* --- The two equations, exactly as Stone 5 defined them --- *)
LuminWithout[t_, i_] :=
  M phi^i (Sin[omega t + S] + Cos[omega t + ph]) + psi[i] + feedback[i] + mc2;
LuminWith[t_, i_] :=
  LuminWithout[t, i] + Sin[3 omega t] + Sin[6 omega t] + Sin[9 omega t];
(* ============================================================
   PART A -- Reproduce the ORIGINAL Stone 5 number (the MEAN).
   Coarse grid t=0..10 step 1, i=1..5, exactly as Stone 5 ran it.
   ============================================================ *)
gridT = Range[0, 10, 1];
gridI = Range[1, 5];
withoutCoarse = Flatten@Table[LuminWithout[t, i], {t, gridT}, {i, gridI}];
withCoarse    = Flatten@Table[LuminWith[t, i],    {t, gridT}, {i, gridI}];
avgWithout = Mean[withoutCoarse];
avgWith    = Mean[withCoarse];
meanPctChange = 100 (avgWith - avgWithout)/avgWithout;
Print["=== PART A: ORIGINAL Stone 5 metric (MEAN of pooled values) ==="];
Print["  Avg coherence WITHOUT harmonics: ", N[avgWithout]];
Print["  Avg coherence WITH harmonics:    ", N[avgWith]];
Print["  Mean % change (the original -49.5%-type number): ", N[meanPctChange], " %"];
(* ============================================================
   PART B -- The actual AMPLITUDE / VARIANCE test.
   Amplitude = spread of the signal about its own mean, NOT the mean.
   Done per i (the phi^i factor scales each row differently) on a
   FINE grid so Sin[9 omega t] is actually resolved -- the coarse
   11-point grid badly undersamples a 9 rad/s component.
   Metrics: StandardDeviation (RMS oscillation = sqrt of variance)
            and Peak-to-Peak (Max - Min).
   ============================================================ *)
fineT = Range[0, 10, 0.01];
ampTable = Table[
   Module[{wo, wi},
    wo = LuminWithout[#, i] & /@ fineT;
    wi = LuminWith[#, i] & /@ fineT;
    {
     i,
     StandardDeviation[wo], StandardDeviation[wi],
     100 (StandardDeviation[wi] - StandardDeviation[wo])/StandardDeviation[wo],
     (Max[wo] - Min[wo]), (Max[wi] - Min[wi]),
     100 ((Max[wi] - Min[wi]) - (Max[wo] - Min[wo]))/(Max[wo] - Min[wo])
     }],
   {i, gridI}];
Print["\n=== PART B: AMPLITUDE / VARIANCE per i (fine grid) ==="];
Print[Grid[
   Prepend[N[ampTable, 4],
    {"i", "SD_wo", "SD_wi", "SD %chg", "PtP_wo", "PtP_wi", "PtP %chg"}],
   Frame -> All, Alignment -> Right]];
(* Pooled amplitude across all i, each row mean-centered first so the
   phi^i DC offset cannot masquerade as oscillation. *)
woCenter = Flatten@Table[
    Module[{s = LuminWithout[#, i] & /@ fineT}, s - Mean[s]], {i, gridI}];
wiCenter = Flatten@Table[
    Module[{s = LuminWith[#, i] & /@ fineT}, s - Mean[s]], {i, gridI}];
sdWoPool = StandardDeviation[woCenter];
sdWiPool = StandardDeviation[wiCenter];
Print["\n=== Pooled (mean-centered) RMS amplitude ==="];
Print["  WITHOUT harmonics: ", N[sdWoPool]];
Print["  WITH harmonics:    ", N[sdWiPool]];
Print["  Amplitude % change: ", N[100 (sdWiPool - sdWoPool)/sdWoPool], " %"];
(* Variance versions of the same, for direct comparison to the draft *)
Print["\n=== Pooled VARIANCE (mean-centered) ==="];
Print["  WITHOUT: ", N[Variance[woCenter]],
      "   WITH: ", N[Variance[wiCenter]],
      "   % change: ", N[100 (Variance[wiCenter] - Variance[woCenter])/Variance[woCenter]], " %"];
(* ============================================================
   PART C -- Visual: representative row i=1, with vs without.
   If the orange (with) curve mostly shifts up/down -> mean effect.
   If it gets visibly tighter/looser -> amplitude effect.
   ============================================================ *)
Print["\n=== PART C: Visual, i = 1 ==="];
Plot[{LuminWithout[t, 1], LuminWith[t, 1]}, {t, 0, 10},
  PlotStyle -> {Blue, Orange},
  PlotLegends -> {"Without harmonics", "With harmonics"},
  AxesLabel -> {"t", "C(i=1)"}, PlotPoints -> 300, ImageSize -> 500]
