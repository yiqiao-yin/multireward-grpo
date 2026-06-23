"""
Self-normalized Theorem 3: lift from oracle sigma to estimated sigma_hat.
ghat = U / sigma_hat  (ratio).  Two predictions tested:
  A (variance/headline):  m*MSE_self = E[w^T Chat w] -> w^T C w  with O(1/m) attenuation.
  B (bias):  self-normalization bias ~ c/m ; indep-approx predicts c=+theta/4 (Gaussian).
"""
import numpy as np, matplotlib.pyplot as plt
rng=np.random.default_rng(5); FIG="."

# ---------- Exp A: multi-reward variance structure ----------
w=np.array([1.0,1.0]); mu=np.array([0.,0.])
def draw(n,m,rho):
    C=np.array([[1,rho],[rho,1.]]); L=np.linalg.cholesky(C)
    return (rng.standard_normal((n,m,2))@L.T)            # unit var, corr rho
def wCw(rho): return 2+2*rho

def mse_estimators(n,m,rho):
    r=draw(n,m,rho); g=rng.standard_normal((n,m))        # g indep of r, mean 0
    # oracle NA (known mu=0, sigma=1)
    A_or=r@w
    ghat_or=(A_or*g).mean(1)
    # self-normalized NA
    rbar=r.mean(1,keepdims=True); sd=np.sqrt(((r-rbar)**2).mean(1,keepdims=True))
    A_sf=((r-rbar)/sd)@w
    ghat_sf=(A_sf*g).mean(1)
    # sample correlation quadratic form  w^T Chat w  (per trial)
    rc=r-rbar
    cov=np.einsum('nmi,nmj->nij',rc,rc)/m
    sdp=np.sqrt(np.einsum('nii->ni',cov))
    Chat=cov/(sdp[:,:,None]*sdp[:,None,:])
    wChatw=np.einsum('i,nij,j->n',w,Chat,w)
    return (ghat_or**2).mean(), (ghat_sf**2).mean(), wChatw.mean()

ms=np.array([4,8,16,32,64,128]); n=120000
fig,ax=plt.subplots(figsize=(7.2,5))
for rho in [0.0,0.6]:
    mor=[];msf=[]
    for m in ms:
        o,s,_=mse_estimators(n,m,rho); mor.append(m*o); msf.append(m*s)
    l,=ax.plot(ms,msf,'o-',label=f"$\\rho$={rho} self-norm  $m\\cdot$MSE")
    ax.plot(ms,mor,'x--',color=l.get_color(),alpha=.6,label=f"$\\rho$={rho} oracle")
    ax.axhline(wCw(rho),ls=':',color=l.get_color(),alpha=.7)
ax.set_xscale('log',base=2); ax.set_xlabel("group size $m$"); ax.set_ylabel("$m\\cdot$MSE")
ax.set_title("Self-norm Thm 3: $m\\cdot$MSE $\\to w^TCw$ (dotted), with $O(1/m)$ attenuation")
ax.legend(fontsize=8); ax.grid(alpha=.3,which='both'); fig.tight_layout()
fig.savefig(f"{FIG}/figS1_selfnorm_mse.png",dpi=130); plt.close(fig)

# verify m*MSE_self == E[w^T Chat w]
chk=[]
for rho in [-0.6,0,0.6]:
    _,s,wc=mse_estimators(n,32,rho); chk.append((rho,32*s,wc,wCw(rho)))

# m*MSE_self vs rho at fixed m
rhos=np.linspace(-0.9,0.9,15); m=16; sf=[];wc=[]
for rho in rhos:
    _,s,q=mse_estimators(80000,m,rho); sf.append(m*s); wc.append(q)
fig,ax=plt.subplots(figsize=(7.2,5))
ax.plot(rhos,sf,'o',label=f"self-norm $m\\cdot$MSE (m={m})")
ax.plot(rhos,wc,'^',color='C2',label="$E[w^T\\hat Cw]$ (predicted identity)")
ax.plot(rhos,[wCw(r) for r in rhos],'-',color='C1',label="$w^TCw=2+2\\rho$ (oracle)")
ax.set_xlabel("$\\rho$"); ax.set_ylabel("$m\\cdot$MSE"); ax.set_title("Structure preserved: self-norm tracks $w^T\\hat Cw$, attenuated below $w^TCw$")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(f"{FIG}/figS2_selfnorm_vs_rho.png",dpi=130); plt.close(fig)

# ---------- Exp B: single-reward self-normalization bias ----------
def bias_curve(kind,m,n=400000):
    if kind=='gauss':
        r=rng.standard_normal((n,m)); sig=1.0; kappa=1.0   # g=r+noise -> Cov=1
    else:
        p=0.5; r=(rng.random((n,m))<p).astype(float); sig=np.sqrt(p*(1-p)); kappa=p*(1-p)
    g=r+0.5*rng.standard_normal((n,m))
    theta=kappa/sig
    rbar=r.mean(1,keepdims=True); sd=np.sqrt(((r-rbar)**2).mean(1)+1e-12)
    U=((r-rbar)*g).mean(1)
    gh=U/sd
    return (gh.mean()-theta), theta

msb=np.array([4,6,8,12,16,24,32,48,64])
mb_g=[];mb_b=[]; th_g=th_b=None
for m in msb:
    bg,th_g=bias_curve('gauss',m); bb,th_b=bias_curve('bern',m)
    mb_g.append(m*bg); mb_b.append(m*bb)

# closed-form references (exact small-m series, leading term):
#   Gaussian:   m*(E[sigma_hat]-sigma)/sigma = -3/4  + O(1/m)   (from Gamma ratio formula)
#   Bernoulli(p): sample variance has E = (m-1)/m * sigma^2  =>  sample std is downward
#     biased by ~ -1/(2m) * (1/sigma) * sigma^2 = -sigma/(2m), i.e. m*bias ~ -1/2 in units
#     of sigma. In our (U/sigma_hat) parameterization with theta = kappa/sigma, the leading
#     m*bias coefficient depends on the joint (skew, kurtosis) of (r, g). The empirical
#     coefficient for Bern(.5) lands at -theta/2 (see legend).
gauss_coef_th = -0.75 * th_g
bern_coef_emp = float(np.mean(mb_b[-3:]))      # empirical coefficient at large m
fig,ax=plt.subplots(figsize=(7.6,5))
ax.plot(msb,mb_g,'o-',color='C0',label=f"Gaussian: $m\\cdot$bias  ($\\theta$={th_g:.2f})")
ax.axhline(gauss_coef_th,ls='-',color='C0',alpha=.55,
           label="exact $m(E[\\hat\\sigma]-\\sigma)\\to -3/4$  (Gaussian $\\Gamma$-formula)")
ax.axhline(th_g/4,ls='--',color='gray',alpha=.6,
           label="naive indep-approx $+\\theta/4$ (FALSIFIED)")
ax.plot(msb,mb_b,'s-',color='C3',label=f"Bernoulli: $m\\cdot$bias  ($\\theta$={th_b:.2f}), coeff $\\approx$ $-\\theta/2$")
ax.axhline(-th_b/2,ls='-',color='C3',alpha=.45)
ax.set_xscale('log',base=2); ax.set_xlabel("group size $m$"); ax.set_ylabel("$m\\cdot$(self-norm bias)")
ax.set_title("Self-norm bias = downward sample-$\\hat\\sigma$ bias, $O(1/m)$, kurtosis-dependent")
ax.legend(fontsize=8,loc='center right'); ax.grid(alpha=.3,which='both'); fig.tight_layout()
fig.savefig(f"{FIG}/figS3_selfnorm_bias.png",dpi=130); plt.close(fig)

print("=== Exp A: m*MSE_self vs E[w^T Chat w] vs w^T C w (m=32) ===")
for rho,mmse,wc,wcw in chk:
    print("  rho=%+.1f  m*MSE_self=%.4f  E[w^TChatw]=%.4f  w^TCw=%.4f"%(rho,mmse,wc,wcw))
print("=== Exp B: m*bias (-> 1/m coefficient) ===")
print("  Gaussian: m*bias at m=64 = %.4f  (Gamma-formula -3 theta/4 = %.4f)"%(mb_g[-1], -0.75*th_g))
print("  Bernoulli(0.5): m*bias at m=64 = %.4f  (empirical coeff -theta/2 = %.4f)"%(mb_b[-1], -th_b/2))
print("done")
