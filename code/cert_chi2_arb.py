# chi-square (Renyi order 2), T = 2, sigma = 1, every b (19 Oct 2026). Rigorous certificate that
# the step-2 SAME switch on an interval of g1 raises chi2(P||Q) above Chua et al.'s pair. chi2 = int dg1 J(g1) - 1 and
# only J(g1) = int p^2/q dg2 depends on the step-2 answers, so J_SAME - J_ANTI > 0 on a cover of [a, b] suffices.
# With N_m = N(g2; m, 1), wa, w0, wc = N(g1; -1)/2, N(g1; 0)/2, N(g1; 1)/2, rho = wc/w0 = e^{g1-1/2}, al = wa/w0 = e^{-g1-1/2}:
#   J_SAME = (wa+wc)^2/w0 * IS(rho),            IS(r) = int N_1^2 / (N_1 + r N_0)
#   J_ANTI = w0 [al^2 I1(rho) + 2 e^{-1} I2(rho) + rho^2 I3(rho)],  Ik(r) = int num_k / (N_-1 + r N_0),
#            num = N_-1^2, N_-1 N_1, N_1^2   (al rho = e^{-1})
# Every integral is decreasing in r (positive numerator, denominator increasing in r). On a g1 ball [lo, hi],
# rho runs over [rho(lo), rho(hi)], so J_SAME >= pref_lo * IS(rho(hi)) and J_ANTI <= w0_up [al_up^2 I1(rho(lo)) + ...].
# Integrals: acb.integral (rigorous) on [-L, L] plus a Gaussian tail bound. No dependency blow-up.
import sys
from flint import arb, acb, ctx
ctx.prec = 128
SQ2PI = (2 * arb.pi()).sqrt(); E = arb(1).exp()
def N_r(x, mu): return (-(x - mu) ** 2 / 2).exp() / SQ2PI
def N_c(z, mu): return (-(z - mu) ** 2 / 2).exp() / SQ2PI
L = 16
TAIL = 2 * N_r(arb(L - 2), 0)          # mass of N(mu, 1), mu in [-1, 2], outside [-L, L] is below this
def IS(r):  # tail: integrand <= N_1^2/(r N_0) = e N_2 / r
    return acb.integral(lambda z, an: N_c(z, 1) ** 2 / (N_c(z, 1) + r * N_c(z, 0)), -L, L).real + arb(0, (E * TAIL / r).upper())
def Ik(r, k):  # tails: N_-1^2/N_-1 = N_-1; N_-1 N_1 / N_-1 = N_1; N_1^2/(r N_0) = e N_2 / r
    num = [lambda z: N_c(z, -1) ** 2, lambda z: N_c(z, -1) * N_c(z, 1), lambda z: N_c(z, 1) ** 2][k]
    tb = [TAIL, TAIL, E * TAIL / r][k]
    return acb.integral(lambda z, an: num(z) / (N_c(z, -1) + r * N_c(z, 0)), -L, L).real + arb(0, tb.upper())
def bounds(lo, hi):
    lo, hi = arb(lo), arb(hi); g = arb(lo).union(arb(hi))  # exact cover of [lo, hi]: a rounded midpoint ball can miss an endpoint by an ulp (99aebc7f, 19 Oct)
    rlo, rhi = (lo - arb(1) / 2).exp(), (hi - arb(1) / 2).exp()
    wa, w0, wc = N_r(g, -1) / 2, N_r(g, 0) / 2, N_r(g, 1) / 2
    pref = (wa + wc) ** 2 / w0; al = (-g - arb(1) / 2).exp(); rho = (g - arb(1) / 2).exp()
    js_lo = arb(pref.lower()) * IS(rhi)
    ja_up = arb(w0.upper()) * (arb((al ** 2).upper()) * Ik(rlo, 0) + 2 / E * Ik(rlo, 1) + arb((rho ** 2).upper()) * Ik(rlo, 2))
    return js_lo.lower(), ja_up.upper()
def point(g):
    g = arb(g); r = (g - arb(1) / 2).exp(); w0 = N_r(g, 0) / 2; al = (-g - arb(1) / 2).exp()
    js = (N_r(g, -1) / 2 + N_r(g, 1) / 2) ** 2 / w0 * IS(r)
    ja = w0 * (al ** 2 * Ik(r, 0) + 2 / E * Ik(r, 1) + r ** 2 * Ik(r, 2))
    return js, ja
if __name__ == "__main__":
    a, b, n = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    js, ja = point(-2); print("point g1 = -2: J_SAME %s J_ANTI %s diff %s (mpmath 1.80411576706e-3)" % (js.str(10), ja.str(10), (js - ja).str(10)), flush=True)
    ok, gain, minlo = True, arb(0), None
    for i in range(n):
        lo, hi = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
        sl, au = bounds(lo, hi); d = arb(sl) - arb(au)
        if not d > 0: ok = False; print("ball %d [%.5f, %.5f] NOT certified, lower diff %s" % (i, lo, hi, d.str(5)), flush=True); break
        gain += (arb(hi) - arb(lo)) * arb(d.lower()); minlo = d.lower() if minlo is None else min(minlo, d.lower())
        if i % 25 == 0: print("ball %d [%.5f, %.5f] diff >= %s" % (i, lo, hi, arb(d.lower()).str(5)), flush=True)
    if ok:
        print("CERTIFIED: J_SAME - J_ANTI > 0 on all of [%g, %g] (%d balls); min lower end %s" % (a, b, n, arb(minlo).str(5)))
        print("so chi2(adaptive) - chi2(P_S||Q_S) >= %s at sigma 1, T 2 (rigorous; numerical total 1.729e-3)" % gain.lower().str(6))
