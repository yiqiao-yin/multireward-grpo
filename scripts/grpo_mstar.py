"""
Does optimal group size m* depend on reward correlation rho (i.e. on w^T C w)?
Budget-constrained: N rollouts -> P=N/m prompts, m rollouts each.
Per-prompt heterogeneity sigma_p is rho-INDEPENDENT (the signal).
Within-prompt noise carries w^T C w (Thm 3). Estimand = population mean gradient mu_theta.
"""
import numpy as np, matplotlib.pyplot as plt
rng=np.random.default_rng(9); FIG="."
w=np.array([1.0,1.0]); mu_th=0.5; sig_p=0.2; tau=1.0
def wCw(rho): return 2+2*rho

def mse(N,m,rho,n_runs=4000):
    P=max(N//m,1); L=np.linalg.cholesky(np.array([[1,rho],[rho,1.]]))
    out=np.empty(n_runs)
    for k in range(n_runs):
        th=rng.normal(mu_th,sig_p,P)                       # per-prompt true gradient (rho-indep)
        z=rng.standard_normal((P,m,2))@L.T                 # rewards, corr rho
        q=z@w                                              # Var=wCw
        g=th[:,None]*q/wCw(rho)+tau*rng.standard_normal((P,m))   # Cov(q,g)=th_i
        zbar=z.mean(1,keepdims=True); sd=np.sqrt(((z-zbar)**2).mean(1,keepdims=True))
        A=((z-zbar)/sd)@w                                  # self-normalized NA advantage
        thhat=(A*g).mean(1)                                # per-prompt estimate
        out[k]=thhat.mean()
    return ((out-mu_th)**2).mean()

N=3072; ms=np.array([4,8,16,24,32,48,64]); rhos=[-0.6,0.0,0.6,0.9]
fig,ax=plt.subplots(figsize=(7.2,5)); mstar={}
for rho in rhos:
    curve=np.array([mse(N,int(m),rho) for m in ms]); ms_star=ms[np.argmin(curve)]; mstar[rho]=ms_star
    l,=ax.plot(ms,curve,'o-',label=f"$\\rho$={rho} ($w^TCw$={wCw(rho):.1f}), $m^*$={ms_star}")
    ax.axvline(ms_star,ls=':',color=l.get_color(),alpha=.5)
ax.set_xscale('log',base=2); ax.set_yscale('log')
ax.set_xlabel("group size $m$ (budget $N$={} fixed; $P=N/m$ prompts)".format(N))
ax.set_ylabel("total MSE vs population gradient")
ax.set_title("Does $m^*$ move with reward correlation? (testing $m^*\\propto\\sqrt{w^TCw}$)")
ax.legend(fontsize=8); ax.grid(alpha=.3,which='both'); fig.tight_layout()
fig.savefig(f"{FIG}/figM1_mstar_vs_rho.png",dpi=130); plt.close(fig)

print("=== m* vs rho (N=%d) ==="%N)
for rho in rhos: print("  rho=%+.1f  wCw=%.1f  m*=%d"%(rho,wCw(rho),mstar[rho]))
print("prediction m* propto sqrt(wCw) would give m* ratio %.2f across rho range; observed flat? -> see above"
      %(np.sqrt(wCw(0.9)/wCw(-0.6))))
print("done")
