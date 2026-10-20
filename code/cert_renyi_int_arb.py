# Renyi divergence of integer order alpha >= 2, P||Q, T = 2, sigma = 1, every b (19 Oct 2026).
# Generalises cert_chi2_arb.py. D_alpha = log(int dg1 J(g1))/(alpha-1), J(g1) = int p^alpha q^(1-alpha) dg2, and only J
# depends on the step-2 answers, so J_SAME - J_ANTI > 0 on a cover of [a, b] proves the adaptive switch strictly beats
# Chua et al.'s pair at order alpha. With rho = wc/w0 = e^{g-1/2}, al = wa/w0 = e^{-g-1/2}:
#   J_SAME = w0 (al+rho)^alpha IS(rho),   IS(r) = int N_1^alpha / (N_1 + r N_0)^(alpha-1)
#   J_ANTI = w0 sum_k C(alpha,k) e^{(2k-alpha) g - alpha/2} A_k(rho),  A_k(r) = int N_-1^(alpha-k) N_1^k / (N_-1 + r N_0)^(alpha-1)
# (al^(alpha-k) rho^k = e^{(2k-alpha) g - alpha/2}). Every integral has a positive numerator and a denominator increasing
# in r, so it decreases in r; on a g ball [lo, hi], rho runs over [rho(lo), rho(hi)]:
#   J_SAME >= lower(w0 (al+rho)^alpha) IS(rho(hi)),  J_ANTI <= upper(w0 coef_k) A_k(rho(lo)).
# Integrals by acb.integral on [-L, L]; tails bounded by Gaussian tails (comments at TAILS).
import sys
from flint import arb, acb, ctx
ctx.prec = 128
SQ2PI = (2 * arb.pi()).sqrt()
def N(x, mu): return (-(x - mu) ** 2 / 2).exp() / SQ2PI
L = 30
def gtail(t):  # P(Z > t) <= phi(t)/t, t > 0
    t = arb(t); return N(t, 0) / t
def binom(n, k):
    from math import comb; return comb(n, k)
# TAILS. SAME integrand <= N_1 (denominator >= N_1^(alpha-1)): tail <= 2 gtail(L-1).
# A_k integrand, z > L: denominator >= (r N_0)^(alpha-1) gives N(z; 2k-alpha) e^{((alpha-2k)^2 - alpha)/2} / r^(alpha-1);
#                z < -L: denominator >= N_-1^(alpha-1) gives N(z; 2k-1) e^{((2k-1)^2 - 1)/2}.
def IS(r, al):
    v = acb.integral(lambda z, an: N(z, 1) ** al / (N(z, 1) + r * N(z, 0)) ** (al - 1), -L, L).real
    return v + arb(0, (2 * gtail(L - 1)).upper())
def A(r, al, k):
    v = acb.integral(lambda z, an: N(z, -1) ** (al - k) * N(z, 1) ** k / (N(z, -1) + r * N(z, 0)) ** (al - 1), -L, L).real
    tb = (arb((al - 2 * k) ** 2 - al) / 2).exp() / r ** (al - 1) * gtail(L - (2 * k - al)) + (arb((2 * k - 1) ** 2 - 1) / 2).exp() * gtail(L + 2 * k - 1)
    return v + arb(0, tb.upper())
def coef(g, al, k): return binom(al, k) * ((2 * k - al) * g - arb(al) / 2).exp()
def bounds(lo, hi, al):
    lo, hi = arb(lo), arb(hi); g = arb(lo).union(arb(hi))  # exact cover of [lo, hi]: a rounded midpoint ball can miss an endpoint by an ulp (99aebc7f, 19 Oct)
    rlo, rhi = (lo - arb(1) / 2).exp(), (hi - arb(1) / 2).exp()
    w0 = N(g, 0) / 2; aa = (-g - arb(1) / 2).exp(); rho = (g - arb(1) / 2).exp()
    js_lo = arb((w0 * (aa + rho) ** al).lower()) * IS(rhi, al)
    ja_up = sum(arb((w0 * coef(g, al, k)).upper()) * A(rlo, al, k) for k in range(al + 1))
    return js_lo.lower(), ja_up.upper()
def point(g, al):
    g = arb(g); r = (g - arb(1) / 2).exp(); w0 = N(g, 0) / 2; aa = (-g - arb(1) / 2).exp()
    js = w0 * (aa + r) ** al * IS(r, al)
    ja = w0 * sum(coef(g, al, k) * A(r, al, k) for k in range(al + 1))
    return js, ja
if __name__ == "__main__":
    al = int(sys.argv[1])
    if sys.argv[2] == "scan":
        for g in [float(x) for x in sys.argv[3].split(",")]:
            js, ja = point(g, al); print("alpha %d g1 %.3f J_SAME %s J_ANTI %s diff %s" % (al, g, js.str(8), ja.str(8), (js - ja).str(6)), flush=True)
        sys.exit()
    a, b, n = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    ok, gain, minlo = True, arb(0), None
    for i in range(n):
        lo, hi = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
        sl, au = bounds(lo, hi, al); d = arb(sl) - arb(au)
        if not d > 0: ok = False; print("ball %d [%.5f, %.5f] NOT certified, lower diff %s" % (i, lo, hi, d.str(5)), flush=True); break
        gain += (arb(hi) - arb(lo)) * arb(d.lower()); minlo = d.lower() if minlo is None else min(minlo, d.lower())
        if i % 25 == 0: print("ball %d [%.5f, %.5f] diff >= %s" % (i, lo, hi, arb(d.lower()).str(5)), flush=True)
    if ok:
        print("CERTIFIED alpha %d: J_SAME - J_ANTI > 0 on all of [%g, %g] (%d balls); min lower end %s" % (al, a, b, n, arb(minlo).str(5)))
        print("so int J(adaptive) - int J(P_S||Q_S) >= %s at sigma 1, T 2, order %d (rigorous)" % (gain.lower().str(6), al))
