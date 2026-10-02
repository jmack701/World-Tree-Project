import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

d = np.load('a100_results.npz'); pc = np.load('phase_confirmation.npz')
fr, pw, pwse = d['fr'], d['pw_RME'], d['pw_RSE']
df = fr[1]-fr[0]; N = len(d['t'])
wl = d['window_lengths'].astype(float); wl[-1] = N

def local_floor(p, i, w=10):
    lo,hi = max(0,i-w), min(len(p), i+w+1)
    return np.median(np.concatenate([p[lo:i], p[i+1:hi]]))

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 150})

# ---------------- FIG 1: low-frequency spectrum + triplet ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
bins = np.arange(1, 26)
periods = 1/fr[bins]
a1.semilogy(periods, pw[bins], 'o-', ms=4, lw=0.8, color='#1a4a7a', label='R_ME power')
floor3 = np.array([3*local_floor(pw, i) for i in bins])
a1.semilogy(periods, floor3, '--', color='#aa3333', lw=1, label='3× local floor (registered)')
a1.axvline(6798.38, color='#888', ls=':', lw=1); a1.axvline(3231.50, color='#888', ls=':', lw=1)
a1.annotate('nodal predicted\n6798 d (bin 5.37)\nfeature: bins 5–6, 12.6×', xy=(6798, pw[6]), xytext=(12000, pw[6]*8),
            fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.7))
a1.annotate('apsidal predicted 3231.5 d\nbin 11: 3,136×\nrefined 3,234.4 d (0.09%)', xy=(3320, pw[11]), xytext=(1800, pw[11]*0.02),
            fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.7))
a1.set_xscale('log'); a1.set_xlabel('period (days)'); a1.set_ylabel('power')
a1.set_title('R_ME low-frequency region (bins 1–25), 100-yr span')
a1.legend(fontsize=7.5, loc='lower left'); a1.invert_xaxis()

m = (fr > 1/30.3) & (fr < 1/28.9)
a2.semilogy(1/fr[m], pw[m], '-', lw=1, color='#1a4a7a')
for P, lab, snr in [(29.268,'s+ap','26,949×'), (29.528,'s','44,705×'), (29.793,'s−ap ≡ l−y','51,828×')]:
    a2.axvline(P, color='#aa3333', ls=':', lw=0.9)
    a2.text(P, pw[m].max()*1.6, f'{lab}\n{P} d\n{snr}', ha='center', fontsize=7.2)
a2.set_ylim(pw[m].min()*0.5, pw[m].max()*30)
a2.set_xlabel('period (days)'); a2.set_ylabel('power')
a2.set_title('The 29.5-day apsidal triplet (splitting = f_apsidal)')
fig.tight_layout(); fig.savefig('fig_100yr_1_low_frequency_spectrum.png', bbox_inches='tight'); plt.close(fig)

# ---------------- FIG 2: entropy-scale curve ----------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
a1.semilogx(wl, d['curve_me_act'], 'o-', ms=4, color='#1a4a7a', label='R_ME actual')
a1.semilogx(wl, d['curve_me_kep'], 's--', ms=3.5, color='#3a7a3a', label='R_ME Keplerian')
a1.semilogx(wl, d['curve_me_syn'], '^:', ms=3.5, color='#aa8833', label='R_ME known-lines synthetic')
a1.semilogx(wl, d['curve_se_act'], 'o-', ms=3, color='#7a1a4a', alpha=0.7, label='R_SE actual')
a1.semilogx(wl, d['curve_se_kep'], 's--', ms=2.5, color='#7a1a4a', alpha=0.35, label='R_SE Keplerian')
a1.set_xlabel('window length (days)'); a1.set_ylabel('mean windowed SE')
a1.set_title('Spectral entropy vs observation scale (19 windows, 3 mo – 100 yr)')
a1.legend(fontsize=7.5)

dme = d['curve_me_act']-d['curve_me_kep']; dcb = d['curve_cb_act']-d['curve_cb_kep']
a2.semilogx(wl, dme, 'o-', ms=4, color='#1a4a7a', label='ΔSE Moon-Earth')
a2.semilogx(wl, dcb, 's--', ms=3.5, color='#3a7a3a', label='ΔSE combined')
a2.axhline(0, color='#999', lw=0.7)
a2.plot([365],[0.0328],'*', ms=13, color='#aa3333', label='phase-one control (+0.0328, 365 d)')
a2.annotate('peak +0.0405\nat 548 d', xy=(548, 0.0405), xytext=(1400, 0.041), fontsize=7.5,
            arrowprops=dict(arrowstyle='->', lw=0.7))
a2.set_xlabel('window length (days)'); a2.set_ylabel('ΔSE (actual − Keplerian)')
a2.set_title('The N-body organized excess is scale-stable and positive at all scales')
a2.legend(fontsize=7.5)
fig.tight_layout(); fig.savefig('fig_100yr_2_entropy_scale_curve.png', bbox_inches='tight'); plt.close(fig)

# ---------------- FIG 3: phase locking ----------------
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 4.0))
labels = ['anom', 'evection', 'half', 'harm.\nclosure', 'synodic', 'evec.\nclosure', 'a−s\ndiff']
cvs = [0.0002, 0.0007, 0.0011, 0.0003, 0.9684, 0.8409, 0.9682]
cols = ['#1a4a7a']*4 + ['#aa3333']*3
a1.bar(range(7), cvs, color=cols); a1.set_yscale('log'); a1.set_ylim(5e-5, 2)
a1.set_xticks(range(7)); a1.set_xticklabels(labels, fontsize=7.5)
a1.set_ylabel('circular variance'); a1.set_title('2-yr windows: locked vs triplet-mixed (red)')

W = pc['W']
a2.semilogy(W/365.25, pc['cv_syn'], 'o-', color='#aa3333', label='synodic')
a2.semilogy(W/365.25, pc['cv_cl_ev'], 's--', color='#aa8833', label='evection closure')
a2.semilogy(W/365.25, pc['cv_das'], '^:', color='#7a1a4a', label='anom−syn diff')
a2.semilogy(W/365.25, pc['cv_anom'], 'd-', color='#1a4a7a', label='anomalistic (ref)')
a2.axvline(3231.5/365.25, color='#555', ls='--', lw=1)
a2.text(3231.5/365.25*1.05, 0.3, 'window bandwidth =\napsidal splitting', fontsize=7.2)
a2.set_xlabel('demodulation window (years)'); a2.set_ylabel('circular variance')
a2.set_title('Confirmation: CV collapses when the\ntriplet resolves (0.9684 → 0.0002)')
a2.legend(fontsize=7)

Wp, step = 730, 365
x = d['R_ME'] - d['R_ME'].mean(); t = d['t']
w = signal.get_window('hann', Wp)
def refine(p, f0):
    i = int(round(f0/df)); lo,hi = max(1,i-3), min(len(p)-2,i+3)
    i = lo+int(np.argmax(p[lo:hi+1]))
    lp = np.log(p[i-1:i+2]+1e-300); den = lp[0]-2*lp[1]+lp[2]
    dl = np.clip(0.5*(lp[0]-lp[2])/den,-0.5,0.5) if den!=0 else 0
    return fr[i]+dl*df
f_anom = refine(pw, 1/27.55); f_syn = refine(pw, 1/29.5306)
comp = {'anom': f_anom, 'syn': f_syn, 'ev': 2*f_syn-f_anom, 'half': 2*f_anom}
ph = {k: [] for k in comp}; yrs=[]
for s0 in range(0, N-Wp, step):
    seg = x[s0:s0+Wp]*w; tt = t[s0:s0+Wp]; yrs.append(1925+(t[s0]+Wp/2)/365.25)
    for k,fc in comp.items(): ph[k].append(np.angle(np.sum(seg*np.exp(-1j*2*np.pi*fc*tt))))
ph = {k: np.array(v) for k,v in ph.items()}
cl_hf = np.degrees(np.angle(np.exp(1j*(ph['half']-2*ph['anom']))))
cl_ev = np.degrees(np.angle(np.exp(1j*(ph['ev']-(2*ph['syn']-ph['anom'])))))
a3.plot(yrs, cl_ev, '.', ms=3, color='#aa8833', label='evection closure (2-yr win)')
a3.plot(yrs, cl_hf, '.', ms=3, color='#1a4a7a', label='harmonic closure (2-yr win)')
a3.axhline(-180, color='#1a4a7a', lw=0.6, alpha=0.5)
a3.set_ylim(-200, 200); a3.set_xlabel('year'); a3.set_ylabel('closure phase (deg)')
a3.set_title('Harmonic closure locked at −180.0°;\nevection closure wanders at apsidal beat')
a3.legend(fontsize=7)
fig.tight_layout(); fig.savefig('fig_100yr_3_phase_locking.png', bbox_inches='tight'); plt.close(fig)

# ---------------- FIG 4: planetary ----------------
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
yrs_t = 1925 + d['t']/365.25
ax[0,0].plot(yrs_t[::5], d['R_EM'][::5], lw=0.4, color='#7a3a1a')
ax[0,0].set_title('R_EM = d(Earth–Mars)/D_Mars   (±67.6%)'); ax[0,0].set_ylabel('ratio')
ax[0,1].plot(yrs_t[::5], d['R_EJ'][::5], lw=0.4, color='#3a5a7a')
ax[0,1].set_title('R_EJ = d(Earth–Jupiter)/D_Jup   (±23.9%)')
for a, p, marks, ttl in [
    (ax[1,0], d['pw_EM'], [(777.1,'Mars synodic\n18,916×'), (388.6,'half-syn'), (365.3,'annual'), (689.2,'Mars anom.'), (29.53,'lunar 29.53 d\n4,816×')], 'Earth–Mars spectrum'),
    (ax[1,1], d['pw_EJ'], [(397.0,'Jup synodic\n1,901×'), (199.6,'half-syn'), (4565.8,'Jup anom.\n(±1 bin)'), (365.3,'annual'), (29.53,'lunar 29.53 d\n32,007×')], 'Earth–Jupiter spectrum')]:
    m = (fr > 1/9000) & (fr < 1/8)
    a.loglog(1/fr[m], p[m], lw=0.5, color='#333')
    for P, lab in marks:
        i = int(round((1/P)/df))
        a.plot(1/fr[i], p[i], 'v', ms=5, color='#aa3333')
        a.annotate(lab, xy=(1/fr[i], p[i]), xytext=(0, 10), textcoords='offset points',
                   fontsize=6.5, ha='center')
    a.set_xlabel('period (days)'); a.set_title(ttl); a.invert_xaxis()
ax[1,0].set_ylabel('power')
fig.tight_layout(); fig.savefig('fig_100yr_4_planetary.png', bbox_inches='tight'); plt.close(fig)
print("figures written:")
import os
for f in sorted(os.listdir('.')):
    if f.endswith('.png'): print(' ', f)

# ---------------- FIG 5: geocentric completion ----------------
# Data: geocentric_completion.py (two-ellipse baselines; geocentric_results.npz)
#       geocentric_phase_confirm.py (CV collapse values)
g = np.load('geocentric_results.npz')
kep_EM = g['kep_EM_asexec']
R_EM = d['R_EM']; t = d['t']
yrs = 1925 + t/365.25
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 4.0))
m = (yrs >= 1990) & (yrs <= 2005)
a1.plot(yrs[m], R_EM[m], lw=1.2, color='#1a4a7a', label='Earth–Mars actual (DE441)')
a1.plot(yrs[m], kep_EM[m], lw=1.0, ls='--', color='#aa3333',
        label='two-ellipse Kepler baseline\n(zero fitted parameters)')
a1.set_title('The geocentric geometry is pure Kepler\n(correlation 1.000000)')
a1.set_xlabel('year'); a1.set_ylabel('R_EM'); a1.legend(fontsize=7.2)
res = R_EM - kep_EM
a2.plot(yrs[::5], (res - res.mean())[::5], lw=0.4, color='#3a5a3a')
a2.set_title('Residual, mean-centered (std 20.1, 0.05% of mean;\nconstant −132 offset is the a-convention\'s\n(1+e²/2) artifact — spectrally null, see note)')
a2.set_xlabel('year'); a2.set_ylabel('R_EM residual (centered)')
W5 = [16, 24, 33]
series = {'EM annual': [0.1328, 0.0119, 0.0000],
          'EM Mars anomalistic': [0.8529, 0.3891, 0.0015],
          'EJ annual': [0.8510, 0.0014, 0.0095]}
for (nm, cv), (st, c) in zip(series.items(), [('o-', '#1a4a7a'), ('s--', '#aa3333'), ('^:', '#aa8833')]):
    a3.semilogy(W5, [max(v, 5e-5) for v in cv], st, color=c, label=nm)
a3.axhline(0.0009, color='#888', lw=0.7, ls=':')
a3.text(16.2, 0.0011, 'synodic lock level', fontsize=6.8, color='#666')
a3.set_xticks(W5); a3.set_xlabel('demodulation window (years)'); a3.set_ylabel('circular variance')
a3.set_title('Geocentric mixed components collapse\nin the predicted order (§5.14 mechanism)')
a3.legend(fontsize=7)
fig.tight_layout(); fig.savefig('fig_100yr_5_geocentric_completion.png', bbox_inches='tight'); plt.close(fig)
print('  fig_100yr_5_geocentric_completion.png')
