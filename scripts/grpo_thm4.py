"""
Theorem 4 check (CORRECTED after harness caught two errors in the problem-statement sketch).
Model (single prompt): c~Bern(p_a) correctness, f~Bern(p_b) format, independent.
    g = ac*(c-p_a) + af*(f-p_b)*c + gamma*(f-p_b)*(1-c) + N(0,tau^2)
  gamma = contamination: format drives gradient on WRONG answers (must not be rewarded).
Target theta* = wa*Cov(c,g) + wb*Cov(f c, g)   (format counts only when correct).

CORRECTED bias of unconditioned estimator (verified against sim):
    beta = Cov(f,g) - Cov(fc,g) = p_b(1-p_a)[ gamma(1-p_b) - ac*p_a ]   <-- can change sign.
Variance-penalty (1/p_a) claim: NOT supported; subgroup conditioning instead degrades at LOW p_a
    via subgroup degeneracy (too few correct samples to form the conditioned baseline).
"""
import numpy as np, matplotlib.pyplot as plt
rng = np.random.default_rng(11); FIG="."
p_b, ac, af, tau, wa, wb = 0.6, 1.0, 1.0, 0.5, 1.0, 1.0

def gen(n,m,p_a,gamma,rng):
    c=(rng.random((n,m))<p_a).astype(float); f=(rng.random((n,m))<p_b).astype(float)
    g=ac*(c-p_a)+af*(f-p_b)*c+gamma*(f-p_b)*(1-c)+tau*rng.standard_normal((n,m))
    return c,f,g
def theta_star(p_a,gamma,N=3_000_000):
    c,f,g=gen(1,N,p_a,gamma,rng); c,f,g=c[0],f[0],g[0]
    return wa*np.mean((c-p_a)*g)+wb*np.mean((f*c-p_a*p_b)*g)
def beta_theory(p_a,gamma): return p_b*(1-p_a)*(gamma*(1-p_b)-ac*p_a)
def estimators(c,f,g):
    cbar=c.mean(1,keepdims=True); fbar=f.mean(1,keepdims=True); fcbar=(f*c).mean(1,keepdims=True)
    unc=(wa*(c-cbar)+wb*(f-fbar))*g
    zf =(wa*(c-cbar)+wb*(f*c-fcbar))*g
    ncorr=c.sum(1,keepdims=True); fcm=np.divide((f*c).sum(1,keepdims=True),np.maximum(ncorr,1))
    fmt_sg=np.where(ncorr>=2, c*(f-fcm), 0.0)
    sg=(wa*(c-cbar)+wb*fmt_sg)*g
    return unc.mean(1),zf.mean(1),sg.mean(1)

# ---- T1: MSE vs p_a (crossover) + structural degeneracy overlay ----
m,gamma,n=16,1.5,60000
pas=np.linspace(0.1,0.95,18); Mu,Mz,Ms,bm,bt,Pdeg=[],[],[],[],[],[]
for p_a in pas:
    th=theta_star(p_a,gamma); c,f,g=gen(n,m,p_a,gamma,rng); eu,ez,es=estimators(c,f,g)
    Mu.append(((eu-th)**2).mean()); Mz.append(((ez-th)**2).mean()); Ms.append(((es-th)**2).mean())
    bm.append(eu.mean()-th); bt.append(beta_theory(p_a,gamma))
    Pdeg.append(float(np.mean(c.sum(1) < 2)))   # fraction of groups where subgroup baseline is undefined
Mu,Mz,Ms,bm,bt,Pdeg=map(np.array,(Mu,Mz,Ms,bm,bt,Pdeg))

fig,ax=plt.subplots(figsize=(7.6,5))
ax.plot(pas,Mu,'s-',color='C3',label="unconditioned (biased)")
ax.plot(pas,Ms,'o-',color='C0',label="subgroup conditioning")
ax.plot(pas,Mz,'^-',color='C2',label="zero-fill conditioning")
d=Mu-Ms; idx=np.where(np.diff(np.sign(d)))[0]
crossovers=[]
for k in idx:
    x0=pas[k]+(pas[k+1]-pas[k])*(-d[k])/(d[k+1]-d[k])
    crossovers.append(float(x0))
    ax.axvline(x0,ls=':',color='k',alpha=.45)
if crossovers:
    ax.plot([], [], ls=':', color='k', alpha=.6,
            label="unc/sg crossover(s)  $p_a^*$≈"+", ".join(f"{x:.2f}" for x in crossovers))
# right axis: structural degeneracy fraction
ax2 = ax.twinx()
ax2.plot(pas, Pdeg, 'x--', color='gray', alpha=.85,
         label="P(#correct < 2)  [subgroup baseline undefined]")
ax2.set_ylabel("degeneracy fraction (gray)", color='gray')
ax2.set_ylim(0, max(0.05, Pdeg.max()*1.15))
ax2.axvline(2.0/m, ls='-.', color='gray', alpha=.4)
ax2.text(2.0/m, ax2.get_ylim()[1]*0.92, f"  $p_a=2/m={2.0/m:.3f}$", color='gray', fontsize=8)
ax.set_yscale('log'); ax.set_xlabel("correctness rate $p_a$ (harder $\\to$ left)"); ax.set_ylabel("MSE vs $\\theta^*$")
ax.set_title(f"Thm 4: non-monotone subgroup MSE; zero-fill best on average (m={m}, $\\gamma$={gamma})")
# combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='lower center')
ax.grid(alpha=.3,which='both'); fig.tight_layout()
fig.savefig(f"{FIG}/figT1_crossover_pa.png",dpi=130); plt.close(fig)
x0 = crossovers[0] if crossovers else float('nan')

# ---- T2: corrected bias law, isolated with leave-one-out (removes (m-1)/m confound) ----
def loo_unc(c,f,g):
    mm=c.shape[1]; cloo=(c.sum(1,keepdims=True)-c)/(mm-1); floo=(f.sum(1,keepdims=True)-f)/(mm-1)
    return (wa*(c-cloo)+wb*(f-floo))*g
gam2=np.linspace(0,3,16); pa_fix=0.5; bm2=[]; bt2=[]
for gm in gam2:
    th=theta_star(pa_fix,gm); c,f,g=gen(150000,m,pa_fix,gm,rng)
    bm2.append(loo_unc(c,f,g).mean()-th); bt2.append(beta_theory(pa_fix,gm))
fig,ax=plt.subplots(figsize=(7.2,5)); ax.axhline(0,color='gray',lw=.8)
ax.plot(gam2,bm2,'o',label="contamination bias, LOO (sim)")
ax.plot(gam2,bt2,'-',label="theory $p_b(1-p_a)[\\gamma(1-p_b)-\\alpha_c p_a]$")
gz=ac*pa_fix/(1-p_b); ax.axvline(gz,ls=':',color='C3',label=f"sign change $\\gamma$={gz:.2f}")
ax.set_xlabel("contamination $\\gamma$"); ax.set_ylabel("bias removed by conditioning")
ax.set_title(f"Thm 4 (corrected & isolated): bias law verified, sign-changing  ($p_a$={pa_fix})")
ax.legend(fontsize=9); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(f"{FIG}/figT2_bias_law.png",dpi=130); plt.close(fig)

# ---- T3: phase diagram unc vs zero-fill, overlay indifference curve ----
pas_g=np.linspace(0.12,0.9,16); gam_g=np.linspace(0,3,16); n_pd=12000
win=np.zeros((len(gam_g),len(pas_g)))
for i,gm in enumerate(gam_g):
    for j,p_a in enumerate(pas_g):
        th=theta_star(p_a,gm,N=1_200_000); c,f,g=gen(n_pd,m,p_a,gm,rng); eu,ez,_=estimators(c,f,g)
        win[i,j]=((eu-th)**2).mean()-((ez-th)**2).mean()   # >0 => zero-fill wins
gz_line=ac*pas_g/(1-p_b)   # bias-zero / indifference curve
fig,ax=plt.subplots(figsize=(7.6,5.2))
ax.pcolormesh(pas_g,gam_g,np.sign(win),cmap='coolwarm',shading='auto',alpha=.55,vmin=-1,vmax=1)
ax.contour(pas_g,gam_g,win,levels=[0],colors='k',linewidths=1.6)
ax.plot(pas_g,gz_line,'w--',lw=2.4); ax.plot(pas_g,gz_line,'k--',lw=1.0,
        label="indifference $\\gamma=\\alpha_c p_a/(1-p_b)$ (bias$=0$)")
ax.set_ylim(0,3); ax.set_xlabel("correctness rate $p_a$"); ax.set_ylabel("contamination $\\gamma$")
ax.set_title("Thm 4 phase diagram: red = condition (zero-fill), blue = don't")
ax.legend(loc='upper left',fontsize=8); fig.tight_layout()
fig.savefig(f"{FIG}/figT3_phase_diagram.png",dpi=130); plt.close(fig)

print("bias law: at p_a=0.5,gamma=1.5  sim=%.4f  theory(m-1/m)=%.4f"%(bm[np.argmin(abs(pas-0.5))], (m-1)/m*beta_theory(0.5,1.5)))
print("bias sign change near gamma=%.2f (p_a=0.5): sim bias there=%.4f"%(ac*0.5/(1-p_b), bm2[np.argmin(abs(gam2-ac*0.5/(1-p_b)))]))
print("unc/sg crossovers at gamma=%.1f: %s"%(gamma, crossovers))
print("subgroup-baseline degeneracy P(#correct<2): low p_a=%.3f -> %.3f ; high p_a=%.3f -> %.3f"
      %(pas[0], Pdeg[0], pas[-1], Pdeg[-1]))
print("zero-fill mean-MSE vs unconditioned: %.3f x | vs subgroup: %.3f x"%(Mz.mean()/Mu.mean(), Mz.mean()/Ms.mean()))
print("fraction of (p_a,gamma) grid where conditioning (zero-fill) wins: %.2f"%( (win>0).mean()))
print("done")
