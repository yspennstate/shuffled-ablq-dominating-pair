# Monte Carlo control for t3_dp.py: D_{e^eps}(P_S||Q_S) and the reverse at T = 3 (Chua et al.'s pair, no adaptivity).
import sys, numpy as np
s, eps, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]); e = np.exp(eps)
rng = np.random.default_rng(353)
def logmix(x, mu):   # log of mean_t N(mu e_t, s^2 I)
    l = -0.5 * ((x[:, None, :] - mu * np.eye(3)[None]) ** 2).sum(2) / s ** 2
    return np.log(np.exp(l).mean(1))
for name, mu_a, mu_b in (("P||Q", 2.0, 1.0), ("Q||P", 1.0, 2.0)):
    k = rng.integers(0, 3, n); x = rng.normal(0, s, (n, 3)); x[np.arange(n), k] += mu_a
    r = np.exp(logmix(x, mu_b) - logmix(x, mu_a)); z = np.maximum(1 - e * r, 0)
    print("T=3 MC %s s %.2f eps %.2f: %.6f +- %.6f (n %d)" % (name, s, eps, z.mean(), z.std() / np.sqrt(n), n), flush=True)
