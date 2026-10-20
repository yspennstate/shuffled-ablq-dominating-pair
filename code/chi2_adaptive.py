# T=2 (19 Oct 2026). Chi-square (Renyi order 2) version of the adaptive switch.
# chi2(M(x)||M(x')) = int dg1 J(g1) - 1 with J(g1) = int p(g1,g2)^2 / q(g1,g2) dg2, and J depends on the step-2
# answers only through that g1, so the adaptive optimum is the pointwise max of J over the step-2 choice.
# Frame of t2b1_adaptive.py: a1 = -1, v1 = +1; ANTI (a2,v2) = (1,-1) is the conjectured pair, SAME (1,1).
# Control: van Dijk-Ertan arXiv:2605.06259 App. J give chi2(P_S||Q_S) = e - 1 - 1/e + e^{-5/4} E[sech W] = 1.5895
# at sigma = 1 (W ~ N(0,1/2)); the ANTI total must reproduce it.
import sys, json, numpy as np
from scipy import integrate
from scipy.stats import norm

def J(g1, s, a2, v2, rev, a1=-1.0, v1=1.0):
    wa, w0, wc = .5 * norm.pdf(g1, a1, s), .5 * norm.pdf(g1, 0, s), .5 * norm.pdf(g1, v1, s)
    p = lambda y: wa * norm.pdf(y, v2, s) + wc * norm.pdf(y, a2, s)
    q = lambda y: w0 * norm.pdf(y, v2, s) + wc * norm.pdf(y, 0, s)
    if rev: p, q = q, p
    return integrate.quad(lambda y: p(y) ** 2 / q(y), -12 * s - 3, 12 * s + 3, limit=200, epsabs=1e-15)[0]

def run(s, rev, n=2401):
    gs = np.linspace(-10 * s - 2, 10 * s + 2, n)
    ja = np.array([J(g, s, 1., -1., rev) for g in gs]); js = np.array([J(g, s, 1., 1., rev) for g in gs])
    w = np.ones(n); w[1:-1:2] = 4; w[2:-1:2] = 2; w *= (gs[1] - gs[0]) / 3
    best = np.maximum(ja, js)
    win = gs[js > ja]
    return float(w @ ja - 1), float(w @ best - 1), (float(win.min()), float(win.max())) if len(win) else None

if __name__ == "__main__":
    out = {}
    for s in [float(x) for x in sys.argv[1].split(",")]:
        for rev in (False, True):
            c, a, win = run(s, rev)
            out["%g_%s" % (s, "rev" if rev else "fwd")] = dict(conj=c, adaptive=a, gain=a - c, same_wins=win)
            print("sigma %.2f %s chi2 conj %.6f adaptive %.6f gain %+.3e SAME wins on %s" %
                  (s, "Q||P" if rev else "P||Q", c, a, a - c, win), flush=True)
            json.dump(out, open(sys.argv[2], "w"), indent=1)
    if 1.0 in [float(x) for x in sys.argv[1].split(",")]:
        Esech = integrate.quad(lambda w: norm.pdf(w, 0, np.sqrt(.5)) / np.cosh(w), -np.inf, np.inf)[0]
        print("control (van Dijk-Ertan closed form) chi2(P_S||Q_S) at sigma 1 = %.6f" % (np.e - 1 - 1 / np.e + np.exp(-1.25) * Esech))
