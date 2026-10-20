# -*- coding: utf-8 -*-
""" T = 2 batches of size b = 1 (n = 2), d = 1, zero-out adjacency (19 Oct 2026).

Chua et al. 2403.17673 Conjecture 3.2: P_S = mean_t N(2 e_t, s^2 I), Q_S = mean_t N(e_t, s^2 I) tightly dominates
shuffled ABLQ for every adaptive query method.

With n = 2 there is one other record j, so a non-adaptive strategy is four numbers in [-1,1]: the differing record's
answers (a1, a2) and record j's answers (v1, v2). The shuffle puts i in batch 1 or batch 2 with probability 1/2:
    x  (record i present): (g1, g2) ~ 1/2 N((a1, v2)) + 1/2 N((v1, a2))
    x' (record i = bottom): (g1, g2) ~ 1/2 N((0,  v2)) + 1/2 N((v1, 0 ))
Adaptive: (a2, v2) may depend on g1. The hockey-stick divergence is an integral over g1 of an inner integral that
depends on (a2, v2) only through that g1, so the adaptive optimum takes the inner maximum pointwise in g1. That is
exact for this n, d (up to the grids); a value above the conjectured pair's refutes Conjecture 3.2 at n = 2.
"""
import sys, json
import numpy as np

def phi(x, s):
    return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))

def conj_delta(s, eps, g):
    """max of the two hockey-sticks of the conjectured pair (2-D grid integral)."""
    G1, G2 = np.meshgrid(g, g, indexing="ij")
    p = 0.5 * (phi(G1 - 2, s) * phi(G2, s) + phi(G1, s) * phi(G2 - 2, s))
    q = 0.5 * (phi(G1 - 1, s) * phi(G2, s) + phi(G1, s) * phi(G2 - 1, s))
    h = (g[1] - g[0]) ** 2
    e = np.exp(eps)
    return max(np.maximum(p - e * q, 0).sum() * h, np.maximum(q - e * p, 0).sum() * h)

def adaptive_delta(s, eps, g, a1, v1, grid2, reverse=False):
    """sup over adaptive (a2, v2)(g1) in grid2 x grid2, step-1 answers (a1, v1) fixed."""
    dg = g[1] - g[0]
    e = np.exp(eps)
    A2, V2 = np.meshgrid(grid2, grid2, indexing="ij")
    A2 = A2.ravel()[:, None]; V2 = V2.ravel()[:, None]
    g2 = g[None, :]
    fv = phi(g2 - V2, s); fa = phi(g2 - A2, s); f0 = phi(g2, s)        # (K, N)
    tot = 0.0
    best = []
    for x1 in g:
        wa = 0.5 * phi(x1 - a1, s)      # i in batch 1, present
        w0 = 0.5 * phi(x1, s)           # i in batch 1, bottom
        wc = 0.5 * phi(x1 - v1, s)      # j in batch 1 (i in batch 2)
        p = wa * fv + wc * fa
        q = w0 * fv + wc * f0
        inner = (np.maximum(q - e * p, 0) if reverse else np.maximum(p - e * q, 0)).sum(1) * dg
        k = inner.argmax()
        tot += inner[k] * dg
        best.append(k)
    return tot, best

if __name__ == "__main__":
    out = {}
    g = np.linspace(-9, 11, 501)
    grid1 = np.linspace(-1, 1, 9)
    grid2 = np.linspace(-1, 1, 9)
    for s in [float(x) for x in sys.argv[1].split(",")]:
        gs = np.linspace(-2 - 8 * s, 4 + 8 * s, 401)
        for eps in [float(x) for x in sys.argv[2].split(",")]:
            dc = conj_delta(s, eps, gs)
            bestv = (-1, None)
            for rev in (False, True):
                for a1 in grid1:
                    for v1 in grid1:
                        d, _ = adaptive_delta(s, eps, gs, a1, v1, grid2, rev)
                        if d > bestv[0]:
                            bestv = (d, (float(a1), float(v1), rev))
            print("sigma %.3f eps %.2f conj %.6f adaptive %.6f at a1,v1,rev=%s ratio %.4f" %
                  (s, eps, dc, bestv[0], bestv[1], bestv[0] / max(dc, 1e-300)), flush=True)
            out["%g_%g" % (s, eps)] = {"conj": dc, "adaptive": bestv[0], "at": bestv[1]}
    json.dump(out, open(sys.argv[3], "w"), indent=1)
