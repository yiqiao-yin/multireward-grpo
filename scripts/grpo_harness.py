"""
Multi-reward GRPO advantage-estimator harness.
Step 1: oracle synthetic checks  -> w^T C w scaling + Prop-1 influence law
Step 2: single-reward estimated baseline -> U-statistic bias, variance, group-size law

Everything is checked against closed-form theory overlaid on each plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot

rng = np.random.default_rng(7)
FIG = "."

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def draw_rewards(n, m, mu, sigma, rho, rng):
    """Draw n trials of m bivariate-normal reward vectors. shape (n, m, 2)."""
    C = np.array([[1.0, rho], [rho, 1.0]])
    D = np.diag(sigma)
    Sigma = D @ C @ D
    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal((n, m, 2))
    return mu + z @ L.T  # (n,m,2)

def wCw(w, rho):
    return w[0]**2 + w[1]**2 + 2*rho*w[0]*w[1]   # unit-variance correlation form

# ============================================================================
# STEP 1a / 1c : oracle NA advantage variance == w^T C w, MSE of mean ~ wCw/m
# ============================================================================
w = np.array([1.0, 1.0])
mu = np.array([0.0, 0.0])
sigma = np.array([1.0, 1.0])           # unit variance => C is the covariance
n_trials = 40000

ms = np.array([2,4,8,16,32,64,128])
rhos_1a = [-0.6, 0.0, 0.6]

fig, ax = plt.subplots(figsize=(7,5))
for rho in rhos_1a:
    mses = []
    for m in ms:
        r = draw_rewards(n_trials, m, mu, sigma, rho, rng)            # (n,m,2)
        rt = (r - mu) / sigma                                         # standardized
        A_na = rt @ w                                                 # (n,m)  oracle NA advantage
        ghat = A_na.mean(axis=1)                                      # g_j = 1  -> ghat = mean advantage
        mses.append((ghat**2).mean())                                # MSE vs true 0
    mses = np.array(mses)
    line, = ax.plot(ms, mses, 'o', label=f"$\\rho={rho}$ (sim)")
    ax.plot(ms, wCw(w, rho)/ms, '--', color=line.get_color(),
            label=f"$\\rho={rho}$  $w^TCw/m$ (theory)")
ax.set_xscale('log', base=2); ax.set_yscale('log')
ax.set_xlabel("group size $m$"); ax.set_ylabel("gradient-noise MSE")
ax.set_title("Step 1a: NA estimator MSE = $w^{\\top}Cw\\,/\\,m$  (equal weights)")
ax.legend(fontsize=8); ax.grid(True, which='both', alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1a_wCw_scaling.png", dpi=130); plt.close(fig)

# 1c : MSE coefficient (m * MSE) vs rho should trace w^T C w exactly
fig, ax = plt.subplots(figsize=(7,5))
rhos = np.linspace(-0.9, 0.9, 19)
m_fix = 64
coef_sim = []
for rho in rhos:
    r = draw_rewards(n_trials, m_fix, mu, sigma, rho, rng)
    rt = (r - mu)/sigma
    A_na = rt @ w
    coef_sim.append(m_fix*((A_na.mean(axis=1))**2).mean())
ax.plot(rhos, coef_sim, 'o', label="$m\\cdot$MSE (sim)")
ax.plot(rhos, [wCw(w,rh) for rh in rhos], '-', label="$w^TCw=2+2\\rho$ (theory)")
ax.set_xlabel("reward correlation $\\rho$"); ax.set_ylabel("$m\\cdot$ MSE  coefficient")
ax.set_title("Step 1c: correlation-awareness — coefficient tracks $w^{\\top}Cw$")
ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1c_coeff_vs_rho.png", dpi=130); plt.close(fig)

# ============================================================================
# STEP 1b : Prop-1 influence law.  Heterogeneous variance, EQUAL weights.
#   measure realized influence I_l = Cov(A, standardized r_l).
#   NA: I2/I1 = 1 (flat) ; AN: I2/I1 ~ sigma2/sigma1 (contaminated)
# ============================================================================
ratios = np.linspace(0.25, 4.0, 16)     # sigma2/sigma1
rho_1b = 0.0
n_big = 200000
sim_an, sim_na, th_an, th_na = [], [], [], []
for ratio in ratios:
    sg = np.array([1.0, ratio])
    r = draw_rewards(n_big, 1, mu, sg, rho_1b, rng)[:,0,:]            # (n,2) iid draws
    rt = (r - mu)/sg                                                  # standardized channels
    # NA advantage
    A_na = rt @ w
    # AN advantage: standardize the weighted SUM by its own (known) std
    s = r @ w
    sig_s = np.sqrt(w @ (np.diag(sg) @ np.array([[1,rho_1b],[rho_1b,1]]) @ np.diag(sg)) @ w)
    A_an = (s - s.mean())/sig_s
    I_na = np.array([np.cov(A_na, rt[:,0])[0,1], np.cov(A_na, rt[:,1])[0,1]])
    I_an = np.array([np.cov(A_an, rt[:,0])[0,1], np.cov(A_an, rt[:,1])[0,1]])
    sim_na.append(I_na[1]/I_na[0]); sim_an.append(I_an[1]/I_an[0])
    # closed form (equal weights):
    th_na.append((w[1]+rho_1b*w[0])/(w[0]+rho_1b*w[1]))
    th_an.append((w[1]*sg[1]+w[0]*sg[0]*rho_1b)/(w[0]*sg[0]+w[1]*sg[1]*rho_1b))

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(ratios, sim_an, 's', color='C3', label="AN  scalarize-then-norm (sim)")
ax.plot(ratios, th_an, '-', color='C3', label="AN theory $\\propto\\sigma_2/\\sigma_1$")
ax.plot(ratios, sim_na, 'o', color='C0', label="NA  decoupled (sim)")
ax.plot(ratios, th_na, '-', color='C0', label="NA theory = 1")
ax.axhline(1, ls=':', color='gray')
ax.set_xlabel("variance ratio $\\sigma_2/\\sigma_1$ (harder objective $\\to$ right)")
ax.set_ylabel("realized influence ratio  $I_2/I_1$")
ax.set_title("Step 1b: why reweighting fails — equal weights, equal intended priority")
ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1b_influence_law.png", dpi=130); plt.close(fig)

# ============================================================================
# STEP 2 : single reward, ESTIMATED baseline. r ~ Bern(p), g = a*r + noise.
#   estimand theta = Cov(r,g) = a p(1-p)
#   self/GRPO baseline (incl. self): E = (m-1)/m * theta   -> O(1/m) bias
#   leave-one-out (U-statistic): unbiased
# ============================================================================
p, a, tau = 0.5, 1.0, 0.5
theta = a*p*(1-p)
n_tr = 200000
ms2 = np.array([2,3,4,6,8,12,16,24,32,48,64])

def sim_step2(m, n):
    r = (rng.random((n,m)) < p).astype(float)
    g = a*r + tau*rng.standard_normal((n,m))
    rbar = r.mean(axis=1, keepdims=True)
    # leave-one-out baseline
    rloo = (r.sum(axis=1, keepdims=True) - r)/(m-1)
    est_oracle = ((r - p)*g).mean(axis=1)
    est_self   = ((r - rbar)*g).mean(axis=1)
    est_loo    = ((r - rloo)*g).mean(axis=1)
    return est_oracle, est_self, est_loo

mean_self, mean_loo, mean_or = [], [], []
var_loo = []
for m in ms2:
    eo, es, el = sim_step2(m, n_tr)
    mean_or.append(eo.mean()); mean_self.append(es.mean()); mean_loo.append(el.mean())
    var_loo.append(el.var())
mean_self=np.array(mean_self); mean_loo=np.array(mean_loo); mean_or=np.array(mean_or)
var_loo=np.array(var_loo)

# 2a bias
fig, ax = plt.subplots(figsize=(7,5))
ax.axhline(theta, ls=':', color='gray', label="estimand $\\theta=Cov(r,g)$")
ax.plot(ms2, mean_self, 's', color='C3', label="self/GRPO baseline (sim)")
ax.plot(ms2, (ms2-1)/ms2*theta, '-', color='C3', label="theory $(m-1)/m\\,\\theta$")
ax.plot(ms2, mean_loo, 'o', color='C0', label="leave-one-out U-stat (sim)")
ax.set_xlabel("group size $m$"); ax.set_ylabel("estimator mean")
ax.set_title("Step 2a: self-normalization bias is exactly the $(m{-}1)/m$ U-statistic gap")
ax.legend(fontsize=9); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2a_bias.png", dpi=130); plt.close(fig)

# 2b variance: fit a/m + b/m^2  (inverse-variance weighted, so the fit is
#   geometrically faithful instead of dominated by the small-m points)
X = np.vstack([1/ms2, 1/ms2**2]).T
W = np.diag(1.0/var_loo**2)                      # weight by 1/sigma^2 with sigma ~ var_loo on log scale
coef = np.linalg.solve(X.T @ W @ X, X.T @ W @ var_loo)
fit = X @ coef
# pure 1/m fit (same weighting) for comparison
X1 = (1/ms2)[:,None]
c1 = float((X1.T @ W @ var_loo).squeeze() / (X1.T @ W @ X1).squeeze())
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(ms2, var_loo, 'o', label="Var(LOO) (sim)")
ax.plot(ms2, fit, '-', label=f"weighted fit $a/m+b/m^2$  (a={coef[0]:.3f}, b={coef[1]:.3f})")
ax.plot(ms2, c1/ms2, '--', color='gray', label=f"pure $a/m$ fit  (a={c1:.3f})")
ax.set_xscale('log',base=2); ax.set_yscale('log')
ax.set_xlabel("group size $m$"); ax.set_ylabel("estimator variance")
ax.set_title("Step 2b: variance follows two-term Hoeffding form (U-statistic signature)")
ax.legend(fontsize=9); ax.grid(True, which='both', alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2b_variance.png", dpi=130); plt.close(fig)

# ============================================================================
# STEP 2c : budget-constrained optimal group size.
#   fixed budget N rollouts; group size m -> P=N/m prompts, difficulties p_i~U.
#   use the (biased) self/GRPO estimator that practitioners actually use.
#   estimand = average per-prompt gradient over prompt distribution.
# ============================================================================
def budget_mse(N, m, n_runs=4000, rng=rng):
    P = max(N // m, 1)
    pdist_a = 1.0
    # true per-prompt gradient theta(p)=a p(1-p); population mean over U(0.1,0.9)
    # E_p[p(1-p)] for U(.1,.9):
    lo, hi = 0.1, 0.9
    # closed form mean of p(1-p) on [lo,hi]
    Epp = ( ( (hi**2-lo**2)/2 ) - ( (hi**3-lo**3)/3 ) )/(hi-lo)
    Theta = pdist_a*Epp
    ests = np.empty(n_runs)
    for k in range(n_runs):
        pis = rng.uniform(lo, hi, P)
        r = (rng.random((P,m)) < pis[:,None]).astype(float)
        g = pdist_a*r + tau*rng.standard_normal((P,m))
        rbar = r.mean(axis=1, keepdims=True)
        per = ((r - rbar)*g).mean(axis=1)         # self/GRPO biased estimator
        ests[k] = per.mean()
    bias2 = (ests.mean()-Theta)**2
    var = ests.var()
    return bias2+var, bias2, var

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
# finer m grid, especially near the typical optimum
m_grid = np.array([2,3,4,6,8,12,16,20,24,28,32,40,48,56,64,80,96,112,128,160,192,224,256,320,384])
budgets = [512, 2048, 8192, 32768, 131072, 524288]
mstars = {}
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(budgets)))
for N, c in zip(budgets, colors):
    mse_curve = np.array([budget_mse(N, int(m))[0] for m in m_grid])
    # find the broad minimum: log-quadratic interp around argmin for a refined m*
    j = int(np.argmin(mse_curve))
    j_lo, j_hi = max(j-1, 0), min(j+2, len(m_grid))
    xs = np.log2(m_grid[j_lo:j_hi].astype(float)); ys = np.log(mse_curve[j_lo:j_hi])
    if len(xs) == 3:
        a, b, _ = np.polyfit(xs, ys, 2)
        mstar = float(2 ** (-b / (2*a))) if a > 0 else float(m_grid[j])
    else:
        mstar = float(m_grid[j])
    mstars[N] = mstar
    ax.plot(m_grid, mse_curve, 'o-', color=c, alpha=.85,
            label=f"N={N}: $m^*$≈{mstar:.1f}")
    ax.axvline(mstar, ls=':', color=c, alpha=.5)
ax.set_xscale('log',base=2); ax.set_yscale('log')
ax.set_xlabel("group size $m$  (prompts $P=N/m$)"); ax.set_ylabel("total MSE of budget-averaged gradient")
ax.set_title("Step 2c: interior $m^*$, grows with budget")
ax.legend(fontsize=8); ax.grid(True, which='both', alpha=.3)

# log-log fit of m* vs N
Ns = np.array(sorted(mstars.keys()), float)
ms = np.array([mstars[int(N)] for N in Ns], float)
logN, logm = np.log(Ns), np.log(ms)
# OLS on log-log
xbar = logN.mean(); ybar = logm.mean()
slope = np.sum((logN-xbar)*(logm-ybar)) / np.sum((logN-xbar)**2)
intercept = ybar - slope*xbar
# standard error on the slope
resid = logm - (slope*logN + intercept)
se_slope = np.sqrt(np.sum(resid**2) / max(len(Ns)-2, 1)) / np.sqrt(np.sum((logN-xbar)**2))
ax2 = axes[1]
ax2.loglog(Ns, ms, 'o', ms=8, color='C3', label="empirical $m^*(N)$")
N_smooth = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 50)
ax2.loglog(N_smooth, np.exp(intercept) * N_smooth**slope, '-', color='C3',
           label=f"fit  $m^*\\propto N^{{{slope:.3f}\\pm{se_slope:.3f}}}$")
ax2.loglog(N_smooth, ms[0] * (N_smooth/Ns[0])**(1/3), '--', color='gray',
           label="reference $N^{1/3}$")
ax2.set_xlabel("budget $N$"); ax2.set_ylabel("optimal group size $m^*$")
ax2.set_title("Group-size law: fitted exponent vs $N^{1/3}$ reference")
ax2.legend(fontsize=9); ax2.grid(True, which='both', alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIG}/fig2c_groupsize_law.png", dpi=130); plt.close(fig)
print("2c  fitted exponent: m* ~ N^(%.3f +/- %.3f)" % (slope, se_slope))

# ----------------------------------------------------------------------------
print("=== STEP 1 ===")
print("1a  NA MSE vs theory wCw/m (rho=0.6, m=16): sim=%.5f  theory=%.5f"
      % ( ( (draw_rewards(n_trials,16,mu,sigma,0.6,rng)[:,:,:] - mu)/sigma @ w ).mean(1).var()*1,
          wCw(w,0.6)/16))
print("1b  influence ratio I2/I1 at sigma2/sigma1=4:  AN sim=%.3f (theory %.3f) | NA sim=%.3f (theory 1.0)"
      % (sim_an[-1], th_an[-1], sim_na[-1]))
print("=== STEP 2 ===")
print("2a  self/GRPO mean at m=4: sim=%.4f  theory(3/4*theta)=%.4f  | LOO sim=%.4f  theta=%.4f"
      % (mean_self[2], 0.75*theta, mean_loo[2], theta))
print("2b  Hoeffding fit: Var ≈ %.3f/m + %.3f/m^2" % (coef[0], coef[1]))
print("2c  optimal group size by budget:", {k:round(v,1) for k,v in mstars.items()})
print("done")
