# Renyi divergence of REAL order 0 < alpha < 1 (a rational string, e.g. 1/2), T = 2, any sigma, either direction, every b
# (20 Oct 2026; cert_renyi_real_arb_sigma.py with the inequality and the tail turned round). Here
# D_alpha = log(int dg1 J)/(alpha - 1) with alpha - 1 < 0, so the switch beats the pair where J is SMALLER: J_ANTI - J_SAME > 0
# on a cover of [a, b] proves the adaptive query strictly beats the pair at order alpha. Same p, q as before, at fixed g1
# (wa, w0, wc = N(g1; -1, s)/2, N(g1; 0, s)/2, N(g1; 1, s)/2):
#   SAME: p = (wa + wc) N_1,          q = w0 N_1 + wc N_0
#   ANTI: p = wa N_-1 + wc N_1,       q = w0 N_-1 + wc N_0     (Chua et al.'s pair, up to a sign flip)
# QP swaps p and q. Both exponents alpha and 1 - alpha lie in (0, 1), so x^alpha and x^(1-alpha) are subadditive:
# p^alpha <= sum_i a_i^alpha N_{m_i}^alpha and q^(1-alpha) <= sum_j b_j^(1-alpha) N_{n_j}^(1-alpha), and
# N_m^alpha N_n^(1-alpha) = N(z; c, s) exp(-alpha(1-alpha)(m-n)^2/(2 s^2)), c = alpha m + (1-alpha) n, |c| <= 1.
# So on |z| > L the integrand is below sum_ij a_i^alpha b_j^(1-alpha) e^{...} N(z; c_ij, s), whose mass there is at most
# 2 s^2 N(L - |c_ij|; 0, s)/(L - |c_ij|) (Mills) for L > |c_ij|.
import sys
from flint import arb, acb, ctx
ctx.prec = 128
SIG = float(sys.argv[1]); ALS = sys.argv[2]; AL = arb(ALS); DIR = sys.argv[3]
assert AL > 0 and AL < 1
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
        for b, n in q:
            c = abs(AL * m + (1 - AL) * n).upper()
            tot += a ** AL * b ** (1 - AL) * (-AL * (1 - AL) * (m - n) ** 2 / (2 * S * S)).exp() * 2 * gtail(L - c)
    return tot
def J(g, same, qp):
    p, q = mix(g, same, qp)
    A1, A2 = acb(AL), acb(1 - AL)
    f = lambda z, an: sum((a * N(z, m) for a, m in p), acb(0)).pow(A1, analytic=an) * sum((b * N(z, n) for b, n in q), acb(0)).pow(A2, analytic=an)
    v = acb.integral(f, -L, L).real
    return v + arb(0, tail(p, q).upper())
if __name__ == "__main__":
    qp = DIR == "QP"
    if sys.argv[4] == "scan":
        for x in sys.argv[5].split(","):
            g = arb(float(x)); s_, t_ = J(g, True, qp), J(g, False, qp)
            print("sigma %g alpha %s %s g1 %s J_SAME %s J_ANTI %s ANTI-SAME %s" % (SIG, ALS, DIR, x, s_.str(8), t_.str(8), (t_ - s_).str(6)), flush=True)
        sys.exit()
    a, b, n = float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    ok, gain, minlo = True, arb(0), None
    for i in range(n):
        lo, hi = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
        g = arb(lo).union(arb(hi))  # exact cover of [lo, hi]: a rounded midpoint ball can miss an endpoint by an ulp (99aebc7f, 19 Oct)
        d = J(g, False, qp) - J(g, True, qp)
        if not d > 0: ok = False; print("ball %d [%.5f, %.5f] NOT certified, diff %s" % (i, lo, hi, d.str(5)), flush=True); break
        gain += (arb(hi) - arb(lo)) * arb(d.lower()); minlo = d.lower() if minlo is None else min(minlo, d.lower())
        if i % 10 == 0: print("ball %d [%.5f, %.5f] diff in %s" % (i, lo, hi, d.str(5)), flush=True)
    if ok:
        print("CERTIFIED sigma %g alpha %s %s: J_ANTI - J_SAME > 0 on all of [%g, %g] (%d balls); min lower end %s" % (SIG, ALS, DIR, a, b, n, arb(minlo).str(5)))
        print("so int J(pair) - int J(adaptive) >= %s at sigma %g, T 2, order %s, %s (rigorous)" % (gain.lower().str(6), SIG, ALS, DIR))
