# T=3 (19 Oct 2026). The policy that agrees with the conjectured pair at steps 1-2 and switches only
# step 3 (ANTI (1,-1) -> SAME (1,1)) on the set where that helps. Shifted model of t3_dp.py, step 1 (a1,v1) = (-1,1),
# step 2 ANTI. Gain = int over (g1,g2) of max(0, I3_SAME - I3_ANTI); this script maps where the difference is positive.
import sys, json, numpy as np
from scipy.stats import norm
s, e = 1.0, np.exp(0.5)
y = np.linspace(-12, 14, 5201); hy = y[1] - y[0]; f0 = norm.pdf(y)
def I3(g1, g2, a, v):
    P1 = norm.pdf(g1 + 2) * norm.pdf(g2); Q1 = norm.pdf(g1 + 1) * norm.pdf(g2)
    P2 = norm.pdf(g1) * norm.pdf(g2 - 2); Q2 = norm.pdf(g1) * norm.pdf(g2 - 1)
    P3 = Q3 = norm.pdf(g1) * norm.pdf(g2)
    p = (P1 + P2) * f0 + P3 * norm.pdf(y - (a - v)); q = (Q1 + Q2) * f0 + Q3 * norm.pdf(y + v)
    return np.maximum(p - e * q, 0).sum() * hy / 3
g = np.linspace(-6, 6, 121); h = g[1] - g[0]
D = np.array([[I3(a, b, 1, 1) - I3(a, b, 1, -1) for b in g] for a in g])
pos = D > 0
print("gain of the step-3-only switch: %.4e" % (np.maximum(D, 0).sum() * h * h))
print("max diff %.3e at g1 %.2f g2 %.2f" % (D.max(), g[D.argmax() // len(g)], g[D.argmax() % len(g)]))
ii = np.argwhere(pos)
print("positive region g1 in [%.2f, %.2f], g2 in [%.2f, %.2f], %d of %d points" % (g[ii[:,0]].min(), g[ii[:,0]].max(), g[ii[:,1]].min(), g[ii[:,1]].max(), len(ii), D.size))
np.save("t3_step3_D.npy", D)
