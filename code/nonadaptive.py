# best NON-adaptive strategy at T=2 (any b, identical others), d=1: 9^4 grid over (a1, v1, a2, v2), both orders.
import sys, itertools, numpy as np
from t2b1_adaptive import phi
for s, eps in [(1.0, 0.5), (1.0, 1.0), (0.5, 1.0), (2.0, 0.5)]:
    g = np.linspace(-2 - 8 * s, 2 + 8 * s, 401); h = (g[1] - g[0]) ** 2; E = np.exp(eps)
    grid = np.linspace(-1, 1, 9)
    F = {m: phi(g - m, s) for m in np.round(np.linspace(-2, 2, 17), 6)}
    f = lambda m: F[round(float(m), 6)]
    best = (-1, None); conj = None
    for a1, v1, a2, v2 in itertools.product(grid, repeat=4):
        # shifted by (v1, v2): P: (a1-v1, 0) | (0, a2-v2);  Q: (-v1, 0) | (0, -v2)
        P = .5 * (np.outer(f(a1 - v1), f(0)) + np.outer(f(0), f(a2 - v2)))
        Q = .5 * (np.outer(f(-v1), f(0)) + np.outer(f(0), f(-v2)))
        d = max(np.maximum(P - E * Q, 0).sum() * h, np.maximum(Q - E * P, 0).sum() * h)
        if (a1, v1, a2, v2) == (1, -1, 1, -1): conj = d
        if d > best[0] + 1e-12: best = (d, (a1, v1, a2, v2))
    print("sigma %.1f eps %.1f conj %.6f best non-adaptive %.6f at %s" % (s, eps, conj, best[0], best[1]), flush=True)
