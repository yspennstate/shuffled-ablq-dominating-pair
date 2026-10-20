# KL divergence, T = 2, sigma = 1, every b (19 Oct 2026). KL is additive over the step-1 output:
# D(P||Q) = int dg1 K(g1), K(g1) = int p log(p/q) dg2, and only K depends on the step-2 answers, so K_SAME - K_ANTI > 0 on
# a cover of [a, b] proves the adaptive switch strictly beats Chua et al.'s pair in KL. Densities at fixed g1 as in
# cert_renyi_qp_arb.py (wa, w0, wc = N(g1; -1)/2, N(g1; 0)/2, N(g1; 1)/2):
#   SAME: p = (wa + wc) N_1, q = w0 N_1 + wc N_0;   ANTI: p = wa N_-1 + wc N_1, q = w0 N_-1 + wc N_0.
# Direction QP swaps p and q. g1 enters as an arb ball; the log is taken with analytic=True, so a box that meets the branch
# cut returns an indeterminate value and acb.integral subdivides (the integrand is holomorphic near the real axis since p, q > 0).
# TAILS: p - q <= p log(p/q) <= p^2/q - p, so |p log(p/q)| <= p + q + p^2/q; every term is bounded on |z| > L by a Gaussian with
# centre in [-2, 2] times a coefficient (bounds in tail()).
import sys
from flint import arb, acb, ctx
ctx.prec = 128
SQ2PI = (2 * arb.pi()).sqrt(); E = arb(1).exp()
def N(x, mu): return (-(x - mu) ** 2 / 2).exp() / SQ2PI
L = 30
def gtail(t): t = arb(t); return N(t, 0) / t
T = 2 * gtail(L - 2)      # mass beyond |z| > L of N(mu, 1), |mu| <= 2
def dens(g, same, qp):
    wa, w0, wc = N(g, -1) / 2, N(g, 0) / 2, N(g, 1) / 2
    if same: p = lambda z: (wa + wc) * N(z, 1); q = lambda z: w0 * N(z, 1) + wc * N(z, 0)
    else:    p = lambda z: wa * N(z, -1) + wc * N(z, 1); q = lambda z: w0 * N(z, -1) + wc * N(z, 0)
    return (q, p) if qp else (p, q)
def tail(g, same, qp):
    wa, w0, wc = N(g, -1) / 2, N(g, 0) / 2, N(g, 1) / 2
    mass = (wa + w0 + 2 * wc)                      # p + q integrate to at most this on any set, either direction
    # p^2/q, by case (PQ: chi-square tails of cert_chi2_arb.py; QP: N_0^2/N_1 = e N_-1, N_0^2/N_-1 = e N_1)
    if not qp: sq = (wa + wc) ** 2 / w0 if same else 2 * (wa ** 2 / w0 + E * wc)
    else:      sq = 2 * (w0 ** 2 + E * wc ** 2 / wc) / (wa + wc) if same else 2 * (w0 ** 2 / wa + E * wc ** 2 * (1 / wc + 1 / wa))
    return (mass + sq) * T
def K(g, same, qp):
    p, q = dens(g, same, qp)
    v = acb.integral(lambda z, an: p(z) * (p(z) / q(z)).log(analytic=an), -L, L).real
    return v + arb(0, tail(g, same, qp).upper())
if __name__ == "__main__":
    qp = sys.argv[1] == "QP"
    if sys.argv[2] == "scan":
        for x in sys.argv[3].split(","):
            g = arb(float(x)); s, t = K(g, True, qp), K(g, False, qp)
            print("KL %s g1 %s K_SAME %s K_ANTI %s diff %s" % (sys.argv[1], x, s.str(8), t.str(8), (s - t).str(6)), flush=True)
        sys.exit()
    a, b, n = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    ok, gain, minlo = True, arb(0), None
    for i in range(n):
        lo, hi = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
        g = arb(lo).union(arb(hi))  # exact cover of [lo, hi]: a rounded midpoint ball can miss an endpoint by an ulp (99aebc7f, 19 Oct)
        d = K(g, True, qp) - K(g, False, qp)
        if not d > 0: ok = False; print("ball %d [%.5f, %.5f] NOT certified, diff %s" % (i, lo, hi, d.str(5)), flush=True); break
        gain += (arb(hi) - arb(lo)) * arb(d.lower()); minlo = d.lower() if minlo is None else min(minlo, d.lower())
        if i % 10 == 0: print("ball %d [%.5f, %.5f] diff in %s" % (i, lo, hi, d.str(5)), flush=True)
    if ok:
        print("CERTIFIED KL %s: K_SAME - K_ANTI > 0 on all of [%g, %g] (%d balls); min lower end %s" % (sys.argv[1], a, b, n, arb(minlo).str(5)))
        print("so KL(adaptive) - KL(pair) >= %s at sigma 1, T 2 (rigorous)" % gain.lower().str(6))
