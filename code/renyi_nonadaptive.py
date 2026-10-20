# (19 Oct 2026). Non-adaptive control for renyi_alpha_grid.py: step 2 SAME for every g1 (a fixed
# query sequence, no adaptivity) against Chua et al.'s pair (ANTI everywhere), Renyi order alpha, both directions.
import sys, numpy as np
s = 1.0; g1 = np.linspace(-24, 24, 2401); y = np.linspace(-32, 32, 13001)
N = lambda x, m: np.exp(-(x - m) ** 2 / 2) / np.sqrt(2 * np.pi)
G, Y = g1[:, None], y[None, :]; wa, w0, wc = N(G, -1) / 2, N(G, 0) / 2, N(G, 1) / 2
for rev in (False, True):
    for a in (2, 3, 4, 5):
        r = []
        for a2, v2 in ((1, -1), (1, 1)):
            p, q = wa * N(Y, v2) + wc * N(Y, a2), w0 * N(Y, v2) + wc * N(Y, 0)
            if rev: p, q = q, p
            with np.errstate(all="ignore"): J = np.trapezoid(np.nan_to_num(p ** a * q ** (1 - a)), y, axis=1)
            r.append(np.log(np.trapezoid(J, g1)) / (a - 1))
        print("%s alpha %d  ANTI everywhere (the pair) %.7f   SAME everywhere (non-adaptive) %.7f   diff %+.4e" %
              ("Q||P" if rev else "P||Q", a, r[0], r[1], r[1] - r[0]), flush=True)
