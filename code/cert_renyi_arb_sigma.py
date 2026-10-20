# Renyi divergence of integer order alpha >= 2, T = 2, any sigma, either direction, every b (19 Oct 2026). Argv: SIGMA ALPHA PQ|QP (scan g,g,... | a b n). J(g1) = int p^alpha q^(1-alpha) dg2 with, at fixed g1
# (wa, w0, wc = N(g1; -1, s)/2, N(g1; 0, s)/2, N(g1; 1, s)/2):
#   SAME: p = (wa + wc) N_1,          q = w0 N_1 + wc N_0
#   ANTI: p = wa N_-1 + wc N_1,       q = w0 N_-1 + wc N_0     (Chua et al.'s pair, up to a sign flip)
# QP swaps p and q. D_alpha = log(int dg1 J)/(alpha - 1), and only J depends on the step-2 answers at that g1, so
# J_SAME - J_ANTI > 0 on a cover of [a, b] proves the switch strictly beats the pair at order alpha. g1 enters as an arb ball,
# as in cert_renyi_qp_arb.py and cert_kl_arb.py: acb.integral with a ball-valued integrand encloses the integral for every g1
# in the ball. This script uses the raw densities (no rho/al normalisation), so the same code serves both directions and
# every sigma.
# TAILS. Write p = sum_i a_i N_{m_i}, q = sum_j b_j N_{n_j} (positive coefficients, at most two terms each). Then
# p^alpha <= 2^(alpha-1) sum_i a_i^alpha N_{m_i}^alpha and q^(alpha-1) >= (b_j N_{n_j})^(alpha-1) for every j, and
# N_m^alpha N_n^(1-alpha) = N(z; c, s) exp(alpha(alpha-1)(m-n)^2/(2 s^2)), c = alpha m - (alpha-1) n, |c| <= 2 alpha - 1.
# So on |z| > L the integrand is below 2^(alpha-1) sum_i a_i^alpha min_j b_j^(1-alpha) e^{...} N(z; c_ij, s), whose mass there
# is at most 2 P(X > L - |c_ij|) <= 2 s^2 N(L - |c_ij|; 0, s)/(L - |c_ij|) (Mills), for L > |c_ij|.
import sys
from flint import arb, acb, ctx
ctx.prec = 128
SIG = float(sys.argv[1]); AL = int(sys.argv[2]); DIR = sys.argv[3]
S = arb(SIG); SQ2PI = (2 * arb.pi()).sqrt() * S
def N(x, mu): return (-(x - mu) ** 2 / (2 * S * S)).exp() / SQ2PI
L = 30 * SIG + 2 * AL
def gtail(t): t = arb(t); return S * S * N(t, 0) / t
def mix(g, same, qp):
    wa, w0, wc = N(g, -1) / 2, N(g, 0) / 2, N(g, 1) / 2
    if same: p, q = [(wa + wc, 1)], [(w0, 1), (wc, 0)]
    else:    p, q = [(wa, -1), (wc, 1)], [(w0, -1), (wc, 0)]
    return (q, p) if qp else (p, q)
def tail(p, q):
    tot = arb(0)
    for a, m in p:
        best = None
        for b, n in q:
            c = abs(AL * m - (AL - 1) * n)
            v = b ** (1 - AL) * (arb(AL * (AL - 1) * (m - n) ** 2) / (2 * S * S)).exp() * 2 * gtail(L - c)
            best = v if best is None else (v if v.upper() < best.upper() else best)
        tot += a ** AL * best
    return arb(2) ** (AL - 1) * tot
def J(g, same, qp):
    p, q = mix(g, same, qp)
    f = lambda z, an: sum((a * N(z, m) for a, m in p), acb(0)) ** AL / sum((b * N(z, n) for b, n in q), acb(0)) ** (AL - 1)
    v = acb.integral(f, -L, L).real
    return v + arb(0, tail(p, q).upper())
if __name__ == "__main__":
    qp = DIR == "QP"
    if sys.argv[4] == "scan":
        for x in sys.argv[5].split(","):
            g = arb(float(x)); s_, t_ = J(g, True, qp), J(g, False, qp)
            print("sigma %g alpha %d %s g1 %s J_SAME %s J_ANTI %s diff %s" % (SIG, AL, DIR, x, s_.str(8), t_.str(8), (s_ - t_).str(6)), flush=True)
        sys.exit()
    a, b, n = float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    ok, gain, minlo = True, arb(0), None
    for i in range(n):
        lo, hi = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
        g = arb(lo).union(arb(hi))  # exact cover of [lo, hi]: a rounded midpoint ball can miss an endpoint by an ulp (99aebc7f, 19 Oct)
        d = J(g, True, qp) - J(g, False, qp)
        if not d > 0: ok = False; print("ball %d [%.5f, %.5f] NOT certified, diff %s" % (i, lo, hi, d.str(5)), flush=True); break
        gain += (arb(hi) - arb(lo)) * arb(d.lower()); minlo = d.lower() if minlo is None else min(minlo, d.lower())
        if i % 10 == 0: print("ball %d [%.5f, %.5f] diff in %s" % (i, lo, hi, d.str(5)), flush=True)
    if ok:
        print("CERTIFIED sigma %g alpha %d %s: J_SAME - J_ANTI > 0 on all of [%g, %g] (%d balls); min lower end %s" % (SIG, AL, DIR, a, b, n, arb(minlo).str(5)))
        print("so int J(adaptive) - int J(pair) >= %s at sigma %g, T 2, order %d, %s (rigorous)" % (gain.lower().str(6), SIG, AL, DIR))
