"""
make_figure_5A_v2.py — The Spine, Figure 5.A: regime placement (two-panel v2, rebuilt to the July spec)
(Deposited 2026-10-01 at the author's direction from the September 15, 2026 build
spiral's run record; the render ran there as an inline block — apart from this
header, the code is that block verbatim. Backbone: the §3.2 reference regeneration
of that session, phi_grid_k_sweep_refined.py; window rows: Table 5.14 verbatim.)
World Tree Project · J. David Mack & Claude
"""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ks = [0.1680,0.1714,0.1748,0.1783,0.1817,0.1851,0.1885,0.1919,0.1953,0.1988,0.2022,0.2056,0.2090,0.2124,0.2158,0.2193,0.2227,0.2261,0.2295,0.2329,0.2363,0.2398,0.2432,0.2466,0.2500]
se = [0.1723,0.1762,0.1846,0.1973,0.2142,0.2348,0.2586,0.2848,0.3124,0.3407,0.3687,0.3959,0.4221,0.4472,0.4720,0.4978,0.2950,0.6202,0.7424,0.7520,0.7674,0.8084,0.8391,0.8545,0.8666]
wpairs = [(91,0.2624),(137,0.2309),(182,0.2219),(274,0.2103),(365,0.2548),(548,0.1980),(730,0.1873),(1096,0.1760),(1461,0.1683),(2191,0.1595),(2922,0.1530),(4383,0.1462),(5844,0.1408),(7305,0.1382),(9131,0.1310),(13697,0.1260),(18262,0.1211),(27394,0.1160)]
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8.6,8.4), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios":[1.15,1]})
pts = [("Python grid (scales)",0.4993,0.0113/0.4993,"o","tab:blue"),
       ("Dimensional trio (topologies)",float(np.mean([.4932,.4989,.4996])),float(np.std([.4932,.4989,.4996])/np.mean([.4932,.4989,.4996])),"s","tab:green"),
       ("Boχ C%-correlate (inputs)",0.4902,0.0322,"D","tab:purple")]
for lab,x,y,mk,c in pts:
    ax1.scatter([x],[y],marker=mk,s=70,color=c,zorder=5)
    ax1.annotate(lab,(x,y),textcoords="offset points",xytext=(-6,7),fontsize=8,ha="right")
ax1.scatter([0.2548],[0.048],marker="*",s=240,color="firebrick",zorder=6)
ax1.annotate("Orbital (windows) — bounded-amplitude PASS (§5.6)",(0.2548,0.048),textcoords="offset points",xytext=(10,6),fontsize=8)
ax1.axvline(0.495,color="gray",lw=0.8,ls=":")
ax1.text(0.4955,0.082,"Mathematica F_c / grid tick 0.495",rotation=90,fontsize=7,color="gray",va="top")
for x0 in (0.15,0.30): ax1.axvline(x0,color="k",lw=0.8,ls="--")
ax1.axvspan(0.48,0.51,color="gold",alpha=0.25)
ax1.set_ylabel("entropy stationarity (CoV of SE)"); ax1.set_ylim(0,0.09); ax1.set_xlim(0.08,0.92)
ax1.set_title("Figure 5.A — Regime placement (two-panel v2, rebuilt to the July spec)")
sh1 = np.linspace(0.35,1.0,len(ks)); sh2 = np.linspace(0.35,1.0,len(wpairs))
ax2.scatter(se,[0.7]*len(se),c=sh1,cmap="Blues",s=30,label="grid SE(k) — §3.2 reference, regenerated this session")
ax2.scatter([p[1] for p in wpairs],[0.3]*len(wpairs),c=sh2,cmap="Reds",s=30,label="orbital SE(window) — Table 5.14, 18 rows")
for x0 in (0.15,0.30): ax2.axvline(x0,color="k",lw=0.8,ls="--")
ax2.axvspan(0.48,0.51,color="gold",alpha=0.25)
ax2.set_yticks([0.3,0.7]); ax2.set_yticklabels(["orbital traversal","grid traversal"]); ax2.set_ylim(0.1,0.9)
ax2.set_xlabel("windowed spectral entropy"); ax2.legend(fontsize=7,loc="upper left")
plt.savefig("fig_5A_regime_placement_v2.png",dpi=160)
print("written")
