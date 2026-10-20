# Independent recomputation of the main T = 2 and T = 3 values (sigma = 1, eps = 1/2) with plain numpy/scipy.
# Coordinates as in the paper: z_t = g_t - b psi_t(w); the record's batch has mean 2 (x) / 1 (x') under ANTI and
# 0 (x) / -1 (x') under SAME; every other batch has mean 0.
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

E = np.exp(0.5)
phi = norm.pdf

# T = 2 on a grid: hockey-stick divergence of the pair, its reverse, and the best last-step switch.
z1 = np.linspace(-12, 12, 4801); z2 = np.linspace(-14, 14, 5601)
d1 = z1[1] - z1[0]; d2 = z2[1] - z2[0]
Z1, Z2 = np.meshgrid(z1, z2, indexing='ij')

def inner(a, c):
    p = 0.5 * (phi(Z1 - 2) * phi(Z2) + phi(Z1) * phi(Z2 - a))
    q = 0.5 * (phi(Z1 - 1) * phi(Z2) + phi(Z1) * phi(Z2 - c))
    return np.clip(p - E * q, 0, None).sum(1) * d2, np.clip(q - E * p, 0, None).sum(1) * d2

Ia, Ia_rev = inner(2, 1)
Is, _ = inner(0, -1)
print("T=2 pair       H(P_S||Q_S) =", Ia.sum() * d1)
print("T=2 pair       H(Q_S||P_S) =", Ia_rev.sum() * d1)
print("T=2 adaptive   H(M(x)||M(x')) =", np.maximum(Ia, Is).sum() * d1)
w = z1[Is > Ia]
print("SAME wins for z1 in about", w.min(), w.max())

# Point values with adaptive quadrature in z_T.
def I(al, be, ga, a, c):
    f = lambda s: max(al * phi(s) + be * phi(s - a) - E * (ga * phi(s) + be * phi(s - c)), 0.0)
    return quad(f, -30, 30, points=[-5, 0, 5], limit=400, epsabs=1e-15)[0]

t = 2.5
al, ga, be = phi(t - 2) / 2, phi(t - 1) / 2, phi(t) / 2
print("T=2 z1=2.5: I_ANTI =", I(al, be, ga, 2, 1), " I_SAME =", I(al, be, ga, 0, -1))

y1, y2 = -0.1, 2.25
al = (phi(y1 - 2) * phi(y2) + phi(y1) * phi(y2 - 2)) / 3
ga = (phi(y1 - 1) * phi(y2) + phi(y1) * phi(y2 - 1)) / 3
be = phi(y1) * phi(y2) / 3
print("T=3 y=(-0.1,2.25): I_SAME - I_ANTI =", I(al, be, ga, 0, -1) - I(al, be, ga, 2, 1))
