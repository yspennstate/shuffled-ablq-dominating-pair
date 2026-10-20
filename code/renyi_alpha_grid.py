# T = 2, sigma s (19 Oct 2026). Renyi divergence of every order (and KL) for the adaptive step-2
# switch against Chua et al.'s pair. D_alpha = log(int dg1 J_alpha(g1))/(alpha-1), J_alpha(g1) = int p^alpha q^(1-alpha) dg2;
# KL = int dg1 J_KL(g1), J_KL = int p log(p/q) dg2. Only J depends on the step-2 answers at that g1, so the adaptive optimum
# takes the pointwise max of J. Vectorised trapezoid on a (g1, g2) grid; control: alpha = 2 must give chi2_adaptive.py's
# 1.589482 / 1.591212 (as e^{D_2} - 1).
import sys, json, numpy as np
s = float(sys.argv[1]); alphas = [float(a) for a in sys.argv[2].split(",")]
R1, R2, n1, n2 = (float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (10 * s + 2, 14 * s + 3, 1201, 8001)
g1 = np.linspace(-R1, R1, n1); y = np.linspace(-R2, R2, n2)
N = lambda x, m: np.exp(-(x - m) ** 2 / (2 * s * s)) / (s * np.sqrt(2 * np.pi))
G, Y = g1[:, None], y[None, :]
wa, w0, wc = N(G, -1) / 2, N(G, 0) / 2, N(G, 1) / 2
def pq(a2, v2): return wa * N(Y, v2) + wc * N(Y, a2), w0 * N(Y, v2) + wc * N(Y, 0)
out = {}
for rev in (False, True):
    for a in alphas:
        Js = []
        for a2, v2 in ((1, -1), (1, 1)):
            p, q = pq(a2, v2)
            if rev: p, q = q, p
            with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
                f = np.where(p > 0, p * np.log(p / q), 0.0) if a == 1 else p ** a * q ** (1 - a)
            Js.append(np.trapezoid(np.nan_to_num(f), y, axis=1))
        ja, js = Js
        best = np.maximum(ja, js)
        Ia, Ib = np.trapezoid(ja, g1), np.trapezoid(best, g1)
        if a == 1: dc, dd = Ia, Ib
        else: dc, dd = np.log(Ia) / (a - 1), np.log(Ib) / (a - 1)
        win = g1[js > ja]
        key = "%s_alpha%g" % ("QP" if rev else "PQ", a)
        out[key] = dict(conj=dc, adaptive=dd, gain=dd - dc, same_on=[float(win.min()), float(win.max())] if len(win) else None)
        print("s %.2f %s alpha %-4g conj %.7f adaptive %.7f gain %+.3e  SAME on %s" % (s, "Q||P" if rev else "P||Q", a, dc, dd, dd - dc,
              out[key]["same_on"]), flush=True)
json.dump(out, open("renyi_alpha_s%g_R%g.json" % (s, R1), "w"), indent=1)
