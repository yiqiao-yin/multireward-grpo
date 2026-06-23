"""
Prop 2 (resolution): distinct advantage values, AN (scalarize-then-norm) vs NA (decoupled).
Advantage is affine in r, so #distinct values = #distinct affine-functional values over the
reward grid.  AN ~ w^T r ; NA ~ sum_l w_l r_l / sigma_l.
Tests whether NA reaches the product lattice L^R or collapses to AN's sum lattice.
"""
import numpy as np, itertools, matplotlib.pyplot as plt
FIG="."; R=2; w=np.array([1.0,1.0])
def counts(L, sig):
    grid=np.array(list(itertools.product(range(L), repeat=R)), float)   # all L^R combos
    an = grid@w                                       # AN advantage ~ w^T r
    na = grid@(w/np.array(sig))                       # NA advantage ~ sum w_l r_l/sig_l
    rnd=lambda x: len(np.unique(np.round(x,9)))
    return rnd(an), rnd(na)

Ls=np.arange(2,8)
an_c=[]; na_eq=[]; na_un=[]
for L in Ls:
    a,n_eq=counts(L,[1.0,1.0]); _,n_un=counts(L,[1.0,np.sqrt(2)])   # equal vs heterogeneous scale
    an_c.append(a); na_eq.append(n_eq); na_un.append(n_un)
fig,ax=plt.subplots(figsize=(7.2,5))
ax.plot(Ls, na_un,'o-',color='C0',label="NA, heterogeneous $\\sigma$ (decoupled)")
ax.plot(Ls, [L**R for L in Ls],'--',color='C0',alpha=.6,label="product lattice $L^R$")
ax.plot(Ls, an_c,'s-',color='C3',label="AN (scalarize-then-norm)")
ax.plot(Ls, na_eq,'^',color='C2',label="NA, equal $\\sigma$ (collapses to AN)")
ax.plot(Ls, [R*(L-1)+1 for L in Ls],':',color='C3',alpha=.7,label="sum lattice $R(L{-}1){+}1$")
ax.set_xlabel("reward levels per channel $L$ (R=2)"); ax.set_ylabel("# distinct advantage values")
ax.set_title("Prop 2 (qualified): NA reaches product lattice ONLY with heterogeneous scales")
ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
fig.savefig(f"{FIG}/figP2_resolution.png",dpi=130); plt.close(fig)

# binary special case (GDPO Fig 2 regime)
a2,_=counts(2,[1,1]); _,n2=counts(2,[1,np.sqrt(2)])
print("=== Prop 2 ===")
for L,a,ne,nu in zip(Ls,an_c,na_eq,na_un):
    print("  L=%d: AN=%d  NA(eq sig)=%d  NA(het sig)=%d  | sum-lattice=%d product=%d"%(L,a,ne,nu,R*(L-1)+1,L**R))
print("binary (L=2): AN=%d distinct, NA(het)=%d distinct  -> resolution gain needs heterogeneous scales"%(a2,n2))
print("done")
