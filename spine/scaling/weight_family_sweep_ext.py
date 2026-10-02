"""wf_engine.py - shared engine for the weight-family extension (gates + production)."""
import numpy as np, math, importlib.util, os
_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common", "fc_k_sweep_reference.py")  # repository layout; the original run resolved the deposited reference in the development environment
spec = importlib.util.spec_from_file_location("ref", _REF)
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
PHI=(1+5**0.5)/2; N=100_000; DT=0.01; OMEGA=2*math.pi; AH,NOISE,RES=0.1,0.2,1.0
SEEDS=list(range(20)); BAND=(0.48,0.51)

def cycle_scalar_conditional(alpha, beta, seed, n=N):
    """Bitwise-faithful to fc_k_sweep_reference: rng.random() drawn only when cc<0."""
    rng=np.random.RandomState(seed); cp,cc=0.5,0.3; out=np.empty(n)
    for i in range(1,n+1):
        t=i*DT; cr=alpha*cc+beta*cp
        if cc<0: cr+=NOISE*abs(cc)*(rng.random()*2-1)
        elif cc>RES: cr+=(cc-RES)*math.sin(OMEGA*t)
        cr+=AH*(math.sin(3*OMEGA*t)+math.sin(6*OMEGA*t)+math.sin(9*OMEGA*t))
        cn=((cr+PHI)%(2*PHI))-PHI; out[i-1]=cn; cp,cc=cc,cn
    return out

def cycle_scalar_predraw(alpha, beta, seed, n=N):
    """Same cycle, noise pre-drawn every step (the vectorizable stream)."""
    u=np.random.RandomState(seed).random_sample(n); cp,cc=0.5,0.3; out=np.empty(n)
    for i in range(1,n+1):
        t=i*DT; cr=alpha*cc+beta*cp
        if cc<0: cr+=NOISE*abs(cc)*(u[i-1]*2-1)
        elif cc>RES: cr+=(cc-RES)*math.sin(OMEGA*t)
        cr+=AH*(math.sin(3*OMEGA*t)+math.sin(6*OMEGA*t)+math.sin(9*OMEGA*t))
        cn=((cr+PHI)%(2*PHI))-PHI; out[i-1]=cn; cp,cc=cc,cn
    return out

def cycle_vec(alpha, beta, seeds, n=N):
    """Vectorized across seeds for one (alpha,beta); identical math to predraw scalar."""
    U=np.stack([np.random.RandomState(s).random_sample(n) for s in seeds])
    w=len(seeds); cp=np.full(w,0.5); cc=np.full(w,0.3); out=np.empty((w,n))
    for i in range(1,n+1):
        t=i*DT; cr=alpha*cc+beta*cp
        neg=cc<0; pos=cc>RES
        cr=np.where(neg, cr+NOISE*np.abs(cc)*(U[:,i-1]*2-1), cr)
        cr=np.where(pos, cr+(cc-RES)*math.sin(OMEGA*t), cr)
        cr+=AH*(math.sin(3*OMEGA*t)+math.sin(6*OMEGA*t)+math.sin(9*OMEGA*t))
        cn=((cr+PHI)%(2*PHI))-PHI; out[:,i-1]=cn; cp,cc=cc,cn
    return out

def se_of(trace):
    pw=np.abs(np.fft.rfft(trace)[1:])**2
    if pw.sum()<1e-12: return 0.0
    p=pw/pw.sum(); p=p[p>1e-12]
    return float(-(p*np.log(p)).sum()/np.log(len(p))) if len(p)>1 else 0.0

def edge_se(a,b,k):
    D=cycle_vec(1-b*k, a*k, SEEDS)
    v=np.array([se_of(D[i]) for i in range(len(SEEDS))])
    return float(v.mean()), float(v.std())

def growth(A,B,k,T=8192,chunk=512):
    """Renormalized noiseless linear growth exponent per config (vector across configs)."""
    dp=np.zeros(len(A)); dc=np.full(len(A),1.0); acc=np.zeros(len(A))
    for c0 in range(0,T,chunk):
        for _ in range(chunk):
            dn=(1-B*k)*dc+A*k*dp; dp,dc=dc,dn
        m=np.maximum(np.abs(dc),1e-300); acc+=np.log(m); dp/=m; dc/=m
    return acc/T

def bisect_flip(pairs, iters=48):
    A=np.array([p[0] for p in pairs],float); B=np.array([p[1] for p in pairs],float)
    S=A+B; lo,hi=0.6*2/S,1.4*2/S
    for _ in range(iters):
        mid=0.5*(lo+hi); g=np.empty(len(pairs))
        for j in range(len(pairs)):           # per-config k differs: run each row
            g[j]=growth(A[j:j+1],B[j:j+1],mid[j])[0]
        div=g>0; hi=np.where(div,mid,hi); lo=np.where(div,lo,mid)
    return 0.5*(lo+hi)

# ---------------- gates, extension, figure (single archival entry point) ----------------
if __name__ == "__main__":
    import json, time
    t0=time.time()
    k0=1/6; tr=cycle_scalar_conditional(ref.alpha(k0), ref.beta(k0), 42); r=ref.run_fc(k0)
    g0=(abs(100*(tr<0).sum()/N-r["C"])<1e-9 and abs(float(tr.mean())-r["c_avg"])<1e-12)
    k=0.98*2/9; s1=cycle_scalar_predraw(1-6*k,3*k,0); v1=cycle_vec(1-6*k,3*k,[0])[0]
    g0b=float(np.max(np.abs(s1-v1)))
    print(f"gate 0 identity vs reference: {'PASS' if g0 else 'FAIL'} | gate 0b vec-vs-scalar {g0b:.1e}: {'PASS' if g0b<1e-13 else 'FAIL'}")
    assert g0 and g0b<1e-13
    REG={1:0.1705,2:0.1617,3:0.1546,4:0.1465}
    kf=bisect_flip([(1,8),(2,7),(3,6),(4,5),(2,4),(4,8)]); ok=True; quartet=[]
    for i,(A_,B_) in enumerate([(1,8),(2,7),(3,6),(4,5)]):
        m,sd=edge_se(A_,B_,0.98*2/9); dk=abs(kf[i]-2/9); ok&=(dk<1e-4 and abs(m-REG[A_])<0.005)
        quartet.append(dict(a=A_,b=B_,se=m,se_sd=sd,reg=REG[A_],dk=float(dk)))
        print(f"  repro ({A_},-{B_}): flip d={dk:.1e} | SE {m:.4f}+-{sd:.4f} [{REG[A_]:.4f}]")
    for j,(A_,B_) in enumerate([(2,4),(4,8)]): ok&=abs(kf[4+j]-2/(A_+B_))<1e-4
    print(f"gate 1 §2.6 reproduction: {'PASS' if ok else 'FAIL'}"); assert ok
    res={}
    for S in [7,8,9,10,11,12]:
        pairs=[(a,S-a) for a in range(1,(S-1)//2+1)]; kfs=bisect_flip(pairs); fam=[]
        for i,(a,b) in enumerate(pairs):
            m,sd=edge_se(a,b,0.98*2/S)
            fam.append(dict(a=a,b=b,k_flip=float(kfs[i]),k_pred=2/S,dk=float(abs(kfs[i]-2/S)),
                            se=m,se_sd=sd,invariant=bool(BAND[0]<=m<=BAND[1])))
        aa=np.array([f["a"] for f in fam]); ss=np.array([f["se"] for f in fam])
        res[S]=dict(pairs=fam,gradient_slope=float(np.polyfit(aa,ss,1)[0]),
                    flip_law_max_dev=float(max(f["dk"] for f in fam)),
                    se_min=float(ss.min()),se_max=float(ss.max()),
                    any_invariant=bool(any(f["invariant"] for f in fam)))
        print(f"S={S:2d}: flip max dev {res[S]['flip_law_max_dev']:.1e} | slope {res[S]['gradient_slope']:+.5f}/a | SE [{res[S]['se_min']:.4f},{res[S]['se_max']:.4f}] | invariant {'YES' if res[S]['any_invariant'] else 'none'}")
    allpairs=[f for S in res for f in res[S]["pairs"]]
    maxdev=max(f["dk"] for f in allpairs)
    json.dump(dict(protocol=dict(engine="deposited fc_k_sweep_reference cycle; Step-1 alpha=1-b*k beta=a*k",
        steps=N,seeds=SEEDS,station="0.98*(2/S)",band=list(BAND),se_recipe="IEEE_Fc_Validation.metrics",
        bisection="noiseless linear growth exponent, renormalized, T=8192 x 48 iters",
        gates=dict(g0_identity=bool(g0),g0b_vec=g0b,g1_repro=bool(ok),quartet=quartet)),
        families={str(S):res[S] for S in res}, flip_law_max_dev_all=float(maxdev),
        stability_boundary="a>b divergent at all k>0 (positive companion root); a=b permanent unit root; stable family is a<b"),
        open("weight_family_ext_results.json","w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.4))
    Ss=np.linspace(6.6,12.4,200); ax1.plot(Ss,2/Ss,color="#888",lw=1.2,label="k = 2/(a+b)")
    cmap=plt.cm.viridis
    for S in res:
        col=cmap((S-7)/5.0)
        ax1.scatter([S]*len(res[S]["pairs"]),[f["k_flip"] for f in res[S]["pairs"]],s=26,color=col,zorder=3)
        ax2.plot([f["a"] for f in res[S]["pairs"]],[f["se"] for f in res[S]["pairs"]],"-o",ms=4.5,color=col,lw=1.3,label=f"a+b = {S}")
    ax1.set_xlabel("weight sum a+b"); ax1.set_ylabel("measured flip gain k_flip")
    ax1.set_title(f"The flip law across six families\nall 24 stable pairs on 2/(a+b); max |dev| = {maxdev:.1e}")
    ax1.legend(frameon=False,fontsize=9)
    ax2.axhspan(0.48,0.51,color="#cc3333",alpha=0.18)
    ax2.text(1.05,0.495,"chaos-invariant band 0.48-0.51 — unoccupied",fontsize=8.5,color="#992222",va="center")
    ax2.scatter([q["a"] for q in quartet],[q["reg"] for q in quartet],facecolors="none",edgecolors="#333",s=60,zorder=4,label="§2.6 registered (a+b=9)")
    ax2.set_ylim(0,0.56); ax2.set_xlabel("weight a  (pair (a, -(a+b-a)))"); ax2.set_ylabel("edge SE at k = 0.98·(2/S)")
    ax2.set_title("Edge spectral entropy, every family\nthe invariant stays home"); ax2.legend(frameon=False,fontsize=8,ncol=2)
    for ax in (ax1,ax2): ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); plt.savefig("fig_weight_family_ext.png",dpi=200)
    print(f"figure written  [{time.time()-t0:.0f}s total]")
