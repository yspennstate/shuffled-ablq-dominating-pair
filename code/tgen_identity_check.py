# Control for tgen_mc.py: cert_t3_arb.diff(g1, g2) must equal beta * (I_SAME - I_ANTI)(A, G) with beta = phi(g1) phi(g2) / 3,
# x1 = exp(-g1 - 1/2) (coordinate 1 is sign-flipped in t3_dp.py), x2 = exp(g2 - 1/2), A = (x1^2 + x2^2)/e, G = x1 + x2.
import numpy as np
from scipy.stats import norm
from flint import arb
import cert_t3_arb as C, tgen_mc as M
E = np.exp(.5)
for g1, g2 in [(0.1, 2.2), (-0.5, 2.4), (0.0, 1.8), (0.1, 2.25), (0.3, 2.3), (-1.0, 0.5)]:
    x1, x2 = np.exp(-g1 - .5), np.exp(g2 - .5); A, G = (x1 ** 2 + x2 ** 2) / np.e, x1 + x2
    beta = norm.pdf(g1) * norm.pdf(g2) / 3
    mine = beta * (M.I_same(np.array([A]), np.array([G]), E) - M.I_anti(np.array([A]), np.array([G]), E))[0]
    arbv = float(C.diff(arb(str(g1)), arb(str(g2))).mid())
    print("(%5.2f, %4.2f)  eA/G^2 %.4f  arb %+.10e  reduced %+.10e  rel %.1e" % (g1, g2, np.e * A / G ** 2, arbv, mine, abs(mine - arbv) / abs(arbv)))
