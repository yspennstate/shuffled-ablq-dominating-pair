# T=2 (19 Oct 2026). Rigorous certificate (Arb ball arithmetic, python-flint) that
# I_SAME(g1) > I_ANTI(g1) for every g1 in an interval S, at s = 1, eps = 1/2, n = 2, b = 1, d = 1.
# Unshifted model (t2b1_adaptive.py): step 1 answers a1 = psi_1(u) = -1, v1 = psi_1(w) = +1; step 2 (a2, v2).
#   x : (g1, g2) ~ 1/2 N(a1, s^2) x N(v2, s^2) + 1/2 N(v1, s^2) x N(a2, s^2)
#   x': (g1, g2) ~ 1/2 N(0,  s^2) x N(v2, s^2) + 1/2 N(v1, s^2) x N(0,  s^2)
# I(g1) = int (p - e q)_+ dg2 at fixed g1. h(y)/phi(y) * t = k1 t^2 + k0 t + k_{-1} with t = e^{y/s^2}, so {h > 0} is
# read off a quadratic's roots and I is a sum of Phi differences. Every quantity is an Arb ball; a g1 ball of radius r
# gives a ball enclosing I over all g1 in it, so a positive lower end of I_SAME - I_ANTI on every sub-ball proves the
# inequality on all of S. The adaptive method "SAME on S, ANTI elsewhere" then has
#   D_{e^eps}(M(x) || M(x')) = D_{e^eps}(P_S || Q_S) + int_S (I_SAME - I_ANTI) dg1 > D_{e^eps}(P_S || Q_S),
# because ANTI everywhere is Chua et al.'s pair up to a sign flip of coordinate 1. No continuity argument is needed.
import sys, json
from flint import arb, ctx
ctx.prec = 200

S2 = arb(2).sqrt()
def Phi(x):
    if x == "-inf": return arb(0)
    if x == "inf": return arb(1)
    return (1 + (x / S2).erf()) / 2
def npdf(x, m): return (-(x - m) ** 2 / 2).exp() / (2 * arb.pi()).sqrt()

def I(g1, a2, v2, e, a1=-1, v1=1):
    wa, w0, wc = npdf(g1, a1) / 2, npdf(g1, 0) / 2, npdf(g1, v1) / 2
    terms = [(wa, v2), (wc, a2), (-e * w0, v2), (-e * wc, 0)]
    k = {-1: arb(0), 0: arb(0), 1: arb(0)}
    for c, m in terms: k[m] += c * (arb(-m * m) / 2).exp()
    k0, k1, k2 = k[-1], k[0], k[1]
    if not (k2 > 0 or k2 < 0): raise ValueError("sign of k2 not certified")
    D = k1 * k1 - 4 * k2 * k0
    if k0 == 0:   # structurally zero (no mean -1 term): q/t = k2 t + k1, one root -k1/k2, sign of k1 as t -> 0+
        if not (k1 > 0 or k1 < 0): raise ValueError("sign of k1 not certified")
        t = -k1 / k2
        if t > 0: pts = ["-inf", t.log(), "inf"]
        elif t < 0: pts = ["-inf", "inf"]
        else: raise ValueError("root sign not certified")
        sgn = 1 if k1 > 0 else -1; cuts = []
        for seg in zip(pts[:-1], pts[1:]):
            if sgn > 0: cuts.append(seg)
            sgn = -sgn
    elif D < 0:   # no real root: sign of k2 everywhere on t > 0
        cuts = [("-inf", "inf")] if k2 > 0 else []
    else:
        # D > 0, or a ball holding 0 (two roots merging): r = sqrt(max(D, 0)) is continuous in the coefficients and the
        # segment formula below equals I on both sides of the tangency (a double root adds a segment of length 0), so
        # evaluating it on a ball r covering [0, sqrt(D_hi)] encloses I.
        if D > 0: r = D.sqrt()
        else:
            sh = arb(D.upper()).sqrt().upper(); r = arb(arb(sh).mid() / 2, (arb(sh) / 2 * (1 + arb(1) / 10 ** 6)).upper())
        t1, t2 = (-k1 - r) / (2 * k2), (-k1 + r) / (2 * k2)
        if k2 < 0: t1, t2 = t2, t1          # t1 < t2 as numbers
        pos = []
        for t in (t1, t2):
            if t > 0: pos.append(t)
            elif not t < 0: raise ValueError("root sign not certified: %s (k %s %s %s)" % (t.str(8), k0.str(8), k1.str(8), k2.str(8)))
        ys = [t.log() for t in pos]
        pts = ["-inf"] + ys + ["inf"]
        segs = list(zip(pts[:-1], pts[1:]))
        # sign of q on each segment: q -> sign(k0) as t -> 0+, alternates at each simple positive root
        if not (k0 > 0 or k0 < 0): raise ValueError("sign of k0 not certified")
        sgn = 1 if k0 > 0 else -1; cuts = []
        for seg in segs:
            if sgn > 0: cuts.append(seg)
            sgn = -sgn
    tot = arb(0)
    for lo, hi in cuts:
        for c, m in terms:
            tot += c * (Phi(hi if hi == "inf" else hi - m) - Phi(lo if lo == "-inf" else lo - m))
    return tot

if __name__ == "__main__":
    lo, hi, n = arb(sys.argv[1]), arb(sys.argv[2]), int(sys.argv[3])
    e = (arb(1) / 2).exp()
    w = (hi - lo) / n; worst = None; out = []
    for j in range(n):
        mid = lo + w * (2 * j + 1) / 2
        g = arb(mid.mid(), w.mid() / 2 * (1 + arb(1) / 1000).mid())   # ball covering [lo + j w, lo + (j+1) w]
        d = I(g, 1, 1, e) - I(g, 1, -1, e)
        lower = d.lower()
        worst = lower if worst is None or lower < worst else worst
        out.append(float(lower.mid()))
        if not (d > 0): print("NOT CERTIFIED on sub-ball", j, g.str(10), d.str(10)); sys.exit(1)
    print("CERTIFIED: I_SAME - I_ANTI > 0 on all g1 in [%s, %s] (%d balls); min lower end %s" %
          (sys.argv[1], sys.argv[2], n, worst.str(6)))
    gain = worst * (hi - lo)
    print("so D_adaptive - D(P_S||Q_S) >= %s (rigorous)" % gain.lower().str(6))
    # one-point values at raw g1 = -3/2 (shifted coordinate g1 - v1 = -5/2, as in REFUTATION_T2.md)
    g = arb(-3) / 2
    print("g1=-3/2: I_ANTI", I(g, 1, -1, e).str(15), " I_SAME", I(g, 1, 1, e).str(15))
    json.dump(dict(interval=[sys.argv[1], sys.argv[2]], balls=n, min_lower=worst.str(10), gain_lower=gain.lower().str(10),
                   lowers=out), open("cert_t2_arb.json", "w"), indent=1)
