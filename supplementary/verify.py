"""Independent referee verification for Blade and the Field + The Spine v2.
Implemented from BatF §10 spec alone (fresh code), then compared to published tables.
"""
import numpy as np, math

PHI = (1 + 5**0.5) / 2
PHI_INV = 1/PHI

print("="*72); print("PART 1: ALGEBRA CHECKS"); print("="*72)

def alpha(k): return (1-6*k)/(1-9*k)
def beta(k):  return 3*k/(1-9*k)

# eigenvalues of companion matrix [[a,b],[1,0]]
def eigs(k):
    a,b = alpha(k), beta(k)
    disc = a*a + 4*b
    if disc < 0:
        re, im = a/2, math.sqrt(-disc)/2
        return complex(re,im), complex(re,-im)
    r = math.sqrt(disc)
    return (a+r)/2, (a-r)/2

kphi = PHI/(9*PHI-3)
print(f"k_phi = {kphi:.10f}  (paper 0.13994) ; closed form (7+sqrt5)/66 = {(7+5**0.5)/66:.10f}")
l1,l2 = eigs(kphi)
print(f"  eigs at k_phi: {l1:.6f}, {l2:.6f}; |lambda|^2 = {abs(l1)**2:.10f} (phi = {PHI:.10f})")
print(f"  alpha(k_phi) = {alpha(kphi):.10f}  (-1/phi = {-1/PHI:.10f})")
print(f"  beta(k_phi)  = {beta(kphi):.10f}  (-phi  = {-PHI:.10f})")
print(f"  alpha*beta = {alpha(kphi)*beta(kphi):.10f}; alpha+beta = {alpha(kphi)+beta(kphi):.10f} (-sqrt5={-5**0.5:.10f})")

print(f"k=1/6: alpha={alpha(1/6):.6f}, beta={beta(1/6):.6f}, eigs={eigs(1/6)}")
for k in [0.165,0.18,0.20]:
    l1,_ = eigs(k); print(f"k={k}: |lambda|^2={abs(l1)**2:.4f}")

kc = 1/(6*math.sqrt(2))
print(f"\nComplex/real boundary above pole: k* = 1/(6*sqrt2) = sqrt2/12 = {kc:.6f}")
for k in [0.112, 0.115, 0.1178, 0.118, 0.13]:
    l1,l2 = eigs(k)
    kind = "complex" if isinstance(l1,complex) else "REAL"
    print(f"  k={k}: {kind}  eigs=({l1:.4f}, {l2:.4f})")
print("  -> BatF §10 'Note on the k domain' claims complex conjugates for ALL k>1/9: check above.")

# two-term causal flip
print("\nTwo-term causal flip: d_{n+1}=(1-6k)d_n+3k d_{n-1}; lambda=-1 at 2-9k=0 -> k=2/9 =",2/9)
for (a_,b_) in [(3,6),(2,4),(4,8),(1,8),(5,4)]:
    # d_{n+1} = (1-b k) d_n + a k d_{n-1}; lambda=-1: 1+(1-bk)-ak = 2-(a+b)k
    print(f"  weights ({a_},-{b_}): flip k = 2/(a+b) = {2/(a_+b_):.6f} (algebraic, exact)")

# ceilings and gaps
print("\nCeilings L^2*phi^2:")
for L in [math.sqrt(2), 1.5, PHI]:
    print(f"  L={L:.6f}: ceiling={L*L*PHI*PHI:.6f}")
print(f"phi^4 = {PHI**4:.6f}; phi^3.5 = {PHI**3.5:.6f} (Conv A phi4-zone bound)")
for mx,tag in [(6.573,'prescribed suite'),(6.690,'stochastic 500k'),(6.143,'SimA Ext'),(6.52,'Sim G')]:
    print(f"  max {mx}: gap = {100*(1-mx/PHI**4):.2f}%  ({tag})")

# MM steady state identity
print(f"\n1-phi^{{-1}} = {1-PHI_INV:.10f}; phi^{{-2}} = {PHI**-2:.10f}  (identity used in MM_ss derivation)")

print("\n"+"="*72); print("PART 2: INDEPENDENT §10 REPLICATION (fresh implementation)"); print("="*72)

def run_fc(k, L=PHI, n_steps=100_000, seed=42, c0=0.3, cm1=0.5, mm0=None,
           Ah=0.1, noise_scale=0.2, res_ref=1.0):
    """Written from BatF §10 alone. Draws RNG only in chaos regime (sole-consumer spec)."""
    rng = np.random.RandomState(seed)
    dt, omega = 0.01, 2*math.pi
    a,b = alpha(k), beta(k)
    c_prev, c_curr = cm1, c0
    mm = (cm1*cm1)*PHI_INV + c0*c0 if mm0 is None else mm0
    cc=ec=rc=0; mm_sum=0.0; mm_max=0.0; c_sum=0.0; z4=0; wind2=0; maxwind=0
    for n in range(1, n_steps+1):
        t = n*dt
        c_raw = a*c_curr + b*c_prev                      # Step 1
        if c_curr < 0:                                   # Step 1b
            c_raw += noise_scale*abs(c_curr)*(rng.random()*2-1)
        elif c_curr > res_ref:
            c_raw += (c_curr-res_ref)*math.sin(omega*t)
        c_raw += Ah*(math.sin(3*omega*t)+math.sin(6*omega*t)+math.sin(9*omega*t))  # Step 2
        c_next = ((c_raw+L) % (2*L)) - L                 # Step 3
        w = 0 if abs(c_raw)<=L else (1 if abs(c_raw)<=3*L else 2)
        maxwind=max(maxwind,w); wind2 += (w>=2)
        mm = mm*PHI_INV + c_next*c_next                  # Step 4 (post-wrap)
        if c_next<0: cc+=1
        elif c_next<=1: ec+=1
        else: rc+=1
        mm_sum+=mm; mm_max=max(mm_max,mm); c_sum+=c_next
        z4 += (mm >= PHI**3.5)
        c_prev, c_curr = c_curr, c_next                  # Step 6
    N=n_steps
    return dict(C=100*cc/N, E=100*ec/N, R=100*rc/N, phi4A=100*z4/N,
                mm_avg=mm_sum/N, mm_max=mm_max, c_avg=c_sum/N, maxwind=maxwind)

pub = { # Spine v2 §2.2 published table (Python reference, 100k, seed 42)
 0.1500:(50.4,10.3,39.3, 3,4.037,6.366), 0.1560:(50.7, 4.2,45.1,18,4.781,6.639),
 0.1600:(44.6, 1.9,53.5,30,5.061,6.631), 0.1620:(42.2, 2.8,54.9,31,5.014,6.570),
 0.1650:(48.2,15.6,36.3,12,4.092,6.535), 1/6   :(50.0,30.2,19.8, 2,2.379,6.511),
 0.1680:(50.2,49.8, 0.0, 0,0.016,0.376), 0.1720:(49.7,50.3, 0.0, 0,0.014,0.309),
 0.1800:(49.9,50.3, 0.0, 0,0.016,0.244)}
print(f"{'k':>8} | {'C% me/pub':>13} | {'R% me/pub':>13} | {'phi4A me/pub':>13} | {'MMavg me/pub':>13} | {'MMmax me/pub':>13}")
for k,(pC,pE,pR,p4,pavg,pmax) in pub.items():
    r = run_fc(k)
    print(f"{k:8.4f} | {r['C']:5.1f}/{pC:5.1f} | {r['R']:5.1f}/{pR:5.1f} | {r['phi4A']:5.1f}/{p4:5.1f} | {r['mm_avg']:5.3f}/{pavg:5.3f} | {r['mm_max']:5.3f}/{pmax:5.3f}")
    assert r['mm_max'] < PHI**4, "Proposition violated!"

print("\n500k run at k=0.162 (paper: C=42.60 R=54.59 phi4A=30.87 MMavg=5.024 MMmax=6.690 Cavg=0.206):")
r = run_fc(0.162, n_steps=500_000)
print(f"  mine: C={r['C']:.2f} R={r['R']:.2f} phi4A={r['phi4A']:.2f} MMavg={r['mm_avg']:.3f} MMmax={r['mm_max']:.3f} Cavg={r['c_avg']:.3f}")

print("\nInit-independence check (BatF init 0.3/0.5 vs control init 0.5/0.1), k=0.162, 100k:")
rA = run_fc(0.162); rB = run_fc(0.162, c0=0.5, cm1=0.1, mm0=0.0)
print(f"  A: C={rA['C']:.1f} R={rA['R']:.1f} phi4A={rA['phi4A']:.1f} MMavg={rA['mm_avg']:.3f}")
print(f"  B: C={rB['C']:.1f} R={rB['R']:.1f} phi4A={rB['phi4A']:.1f} MMavg={rB['mm_avg']:.3f}")

print("\n"+"="*72); print("PART 3: CONTAINMENT ENSEMBLE (Spine §5.9 Table 5.9, absolute references)"); print("="*72)
for L,pubgap in [(PHI,3.54),(1.5,3.10),(math.sqrt(2),2.41)]:
    gaps=[]
    for s in range(1,21):
        r = run_fc(0.162, L=L, n_steps=100_000, seed=s, c0=0.5, cm1=0.1, mm0=0.0)
        gaps.append(100*(1 - r['mm_max']/(L*L*PHI*PHI)))
    g=np.array(gaps)
    print(f"  L={L:.4f}: gap = {g.mean():.2f}% +/- {g.std(ddof=1):.2f}   (published {pubgap}%)")

print("\n"+"="*72); print("PART 4: GRID CALIBRATION TEST (BatF §11 step 6 vs Spine §3.2 note)"); print("="*72)
import sys; sys.path.insert(0,'/mnt/project')
# re-implement grid per showcase source (verbatim mechanics) to avoid module-level exec
def grid_run(N, coupling, k_fc=0.221, steps=2000, seed=42):
    ga = 2*np.pi/PHI**2
    pos = np.array([[math.sqrt(i+1)*0.5*math.cos(i*ga), math.sqrt(i+1)*0.5*math.sin(i*ga)] for i in range(N)])
    adj = np.zeros((N,N))
    for i in range(N):
        d = np.linalg.norm(pos-pos[i],axis=1); d[i]=np.inf
        for j in np.argsort(d)[:min(5,N-1)]:
            w=1/(1+d[j]); adj[i,j]=w; adj[j,i]=w
    rs=adj.sum(1,keepdims=True); rs[rs==0]=1; adj/=rs
    np.random.seed(seed)
    OM=2*np.pi*60; DT=1/3600
    t=np.arange(steps)*DT
    loads=np.zeros((N,steps))
    for i in range(N):
        base=0.05*np.sin(2*np.pi*0.1*t+i*PHI*0.3)
        h3=0.05*np.sin(3*OM*t+np.random.uniform(0,2*np.pi))
        h5=0.03*np.sin(5*OM*t+np.random.uniform(0,2*np.pi))
        h7=0.02*np.sin(7*OM*t+np.random.uniform(0,2*np.pi))
        sp=np.zeros(steps)
        for _ in range(np.random.randint(5,15)):
            loc=np.random.randint(0,steps); w=np.random.randint(5,20)
            a=np.random.uniform(0.10,0.25)*np.random.choice([-1,1])
            s,e=max(0,loc-w//2),min(steps,loc+w//2); sp[s:e]=a
        loads[i]=base+h3+h5+h7+sp+0.01*np.random.randn(steps)
    vh=np.zeros((N,steps)); vc=np.ones(N); vp=np.ones(N)
    ph=np.arange(N)*ga
    for step in range(steps):
        tv=step*DT
        vn=vc+loads[:,step]*DT*10
        vn=vn+coupling*(adj@vn-vn)
        dp,dc=vp-1,vn-1
        vn=vn+k_fc*(3*dp-6*dc)
        amp=np.std(dc)*0.3
        vn=vn-amp*(np.sin(3*OM*tv+ph)+np.sin(6*OM*tv+ph)+np.sin(9*OM*tv+ph))
        dev=vn-1; wr=np.fmod(dev+PHI,2*PHI); wr=np.where(wr<0,wr+2*PHI,wr)
        vn=wr-PHI+1
        vp=vc.copy(); vc=vn.copy(); vh[:,step]=vc
    se=[]
    for i in range(N):
        pw=np.abs(np.fft.rfft(vh[i])[1:])**2
        if pw.sum()<1e-12: se.append(0); continue
        p=pw/pw.sum(); p=p[p>1e-12]
        se.append(-(p*np.log(p)).sum()/np.log(len(p)))
    return float(np.mean(se))

for c in [0.005, 0.05]:
    vals=[grid_run(N,c) for N in [25,50,100,200]]
    print(f"  coupling={c}: SE by N(25..200) = {['%.4f'%v for v in vals]}, mean={np.mean(vals):.4f}")
print("  BatF §11 step 6 instructs: run showcase (ships c=0.005), confirm SE=0.4993+/-0.0113")

print("\n"+"="*72); print("PART 5: SE ESTIMATOR SANITY"); print("="*72)
rng=np.random.RandomState(0)
for name,sig in [("white noise",rng.randn(4096)),("pure tone",np.sin(2*np.pi*50*np.arange(4096)/4096))]:
    pw=np.abs(np.fft.rfft(sig)[1:])**2; p=pw/pw.sum(); p=p[p>1e-12]
    print(f"  {name}: SE={-(p*np.log(p)).sum()/np.log(len(p)):.4f}")
