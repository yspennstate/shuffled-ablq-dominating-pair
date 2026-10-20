# Renyi divergence of integer order alpha >= 2 in the Q||P direction, T = 2, sigma = 1, every b (19 Oct 2026). J(g1) = int q^alpha p^(1-alpha) dg2 with, at fixed g1 (wa, w0, wc = N(g1; -1)/2, N(g1; 0)/2, N(g1; 1)/2),
#   SAME: p = (wa + wc) N_1,          q = w0 N_1 + wc N_0
#   ANTI: p = wa N_-1 + wc N_1,       q = w0 N_-1 + wc N_0     (Chua et al.'s pair, up to a sign flip)
# Divided through by w0 (rho = e^{g1-1/2}, al = e^{-g1-1/2}): J = w0 int (N_v + rho N_0)^alpha / (al N_v' + rho N_1)^(alpha-1).
# The denominator moves with al and rho in opposite directions, so the monotone endpoint form of cert_renyi_int_arb.py
# does not apply; instead g1 enters as an arb ball and every integrand evaluation encloses all g1 in the ball (acb.integral
# with a ball-valued integrand returns an enclosure of the integral for every parameter in the ball). Tails: for |z| > L the
# integrand is bounded by Gaussian tails (TAILS below), so J_SAME - J_ANTI > 0 on a cover of [a, b] proves the switch strictly
# beats the pair at order alpha in Q||P.
import sys
from flint import arb, acb, ctx
ctx.prec = 128
SQ2PI = (2 * arb.pi()).sqrt()
def N(x, mu): return (-(x - mu) ** 2 / 2).exp() / SQ2PI
L = 30
def gtail(t): t = arb(t); return N(t, 0) / t
# TAILS. SAME: (N_1 + r N_0)^alpha / ((al+r) N_1)^(alpha-1) <= 2^(alpha-1) (N_1 + r^alpha N_0^alpha N_1^(1-alpha)) / (al+r)^(alpha-1);
#   N_0^alpha N_1^(1-alpha) = N(z; 1-alpha) e^{alpha(alpha-1)/2}. ANTI: (N_-1 + r N_0)^alpha / (al N_-1 + r N_1)^(alpha-1)
#   <= 2^(alpha-1) [N_-1 / al^(alpha-1) + r^alpha N_0^alpha / (al N_-1 + r N_1)^(alpha-1)], and for z > 0 the second is
#   <= r N_0^alpha N_1^(1-alpha) (denominator >= r N_1), for z < 0 <= r^alpha al^(1-alpha) N_0^alpha N_-1^(1-alpha)
#   = r^alpha al^(1-alpha) N(z; alpha-1) e^{alpha(alpha-1)/2}. Each is a Gaussian centred in [-alpha, alpha]; mass beyond
#   |z| > L is below 2 gtail(L - alpha) times its coefficient.
def J(g, al_, same):
    r = (g - arb(1) / 2).exp(); a = (-g - arb(1) / 2).exp(); w0 = N(g, 0) / 2
    if same: f = lambda z, an: (N(z, 1) + r * N(z, 0)) ** al_ / ((a + r) * N(z, 1)) ** (al_ - 1)
    else:    f = lambda z, an: (N(z, -1) + r * N(z, 0)) ** al_ / (a * N(z, -1) + r * N(z, 1)) ** (al_ - 1)
    v = acb.integral(f, -L, L).real
    E = (arb(al_ * (al_ - 1)) / 2).exp(); T = 2 * gtail(L - al_); c = arb(2) ** (al_ - 1)
    if same: tb = c * (T + r ** al_ * E * T) / (a + r) ** (al_ - 1)
    else:    tb = c * (T / a ** (al_ - 1) + (r + r ** al_ * a ** (1 - al_)) * E * T)
    return w0 * (v + arb(0, tb.upper()))
if __name__ == "__main__":
    al_ = int(sys.argv[1])
    if sys.argv[2] == "scan":
        for g in [float(x) for x in sys.argv[3].split(",")]:
            s, t = J(arb(g), al_, True), J(arb(g), al_, False)
            print("alpha %d QP g1 %.3f J_SAME %s J_ANTI %s diff %s" % (al_, g, s.str(8), t.str(8), (s - t).str(6)), flush=True)
        sys.exit()
    a, b, n = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    ok, gain, minlo = True, arb(0), None
    for i in range(n):
        lo, hi = a + (b - a) * i / n, a + (b - a) * (i + 1) / n
        g = arb(lo).union(arb(hi))  # exact cover of [lo, hi]: a rounded midpoint ball can miss an endpoint by an ulp (99aebc7f, 19 Oct)
        d = J(g, al_, True) - J(g, al_, False)
        if not d > 0: ok = False; print("ball %d [%.5f, %.5f] NOT certified, diff %s" % (i, lo, hi, d.str(5)), flush=True); break
        gain += (arb(hi) - arb(lo)) * arb(d.lower()); minlo = d.lower() if minlo is None else min(minlo, d.lower())
        if i % 10 == 0: print("ball %d [%.5f, %.5f] diff in %s" % (i, lo, hi, d.str(5)), flush=True)
    if ok:
        print("CERTIFIED alpha %d Q||P: J_SAME - J_ANTI > 0 on all of [%g, %g] (%d balls); min lower end %s" % (al_, a, b, n, arb(minlo).str(5)))
        print("so int J(adaptive) - int J(Q_S||P_S) >= %s at sigma 1, T 2, order %d (rigorous)" % (gain.lower().str(6), al_))
