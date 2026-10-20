# T=3 (19 Oct 2026). Rigorous certificate (Arb, python-flint) that the policy "conjectured answers
# at steps 1 and 2, step 3 SAME (1,1) instead of ANTI (1,-1) on a box B of (g1, g2)" beats Chua et al.'s pair at T = 3,
# s = 1, eps = 1/2, every b, d = 1. Shifted model of t3_dp.py: if record i sits in batch t, g_t ~ N(a_t - v_t) under x
# and N(-v_t) under x'; other batches N(0). Step 1 (a1,v1) = (-1,1), step 2 ANTI (1,-1). With all steps ANTI the
# output law is (P_S, Q_S) up to a sign flip of coordinate 1, and the hockey-stick integral is
#   int dg1 dg2 I3(g1, g2),  I3 = int (p - e q)_+ dg3,
# where only the g3 factor depends on the step-3 answers. So
#   D(switch on B) = D(P_S || Q_S) + int_B (I3_SAME - I3_ANTI) > D(P_S || Q_S)
# once I3_SAME - I3_ANTI > 0 on every ball of a cover of B. I3 is a sum of Phi differences between the roots of a
# quadratic in t = e^{z} (z = g3 shifted so the three means are -1, 0, 1), evaluated on (g1, g2) balls; balls that do
# not certify are split in four, to a depth limit.
import sys, json
from flint import arb, ctx
ctx.prec = 128

S2 = arb(2).sqrt(); SQ2PI = (2 * arb.pi()).sqrt()
def npdf(x, m): return (-(x - m) ** 2 / 2).exp() / SQ2PI
def Phi(x):
    if x == "-inf": return arb(0)
    if x == "inf": return arb(1)
    return (1 + (x / S2).erf()) / 2

class Uncertain(Exception): pass

def certified_sign(x):
    if x > 0: return 1
    if x < 0: return -1
    raise Uncertain()

def I_terms(terms):
    """int (sum_k c_k phi(z - m_k))_+ dz, means m_k in {-1, 0, 1}; terms whose coefficient is the exact zero
    ball are structurally absent. Returns an Arb ball enclosing the integral."""
    k = {-1: None, 0: None, 1: None}
    for c, m in terms:
        v = c * (arb(-m * m) / 2).exp()
        k[m] = v if k[m] is None else k[m] + v
    k0, k1, k2 = k[-1], k[0], k[1]
    # q(t) = k2 t^2 + k1 t + k0 on t > 0, where t * h(z) / phi(z) = q(t); absent coefficients are exact zeros
    if k2 is None and k0 is None:
        roots = []; s0 = certified_sign(k1)
    elif k2 is None:                         # linear: k1 t + k0
        s0 = certified_sign(k0)
        if k1 is None: roots = []
        else:
            t = -k0 / k1
            roots = [t] if t > 0 else ([] if t < 0 else None)
            if roots is None: raise Uncertain()
    elif k0 is None:                         # t (k2 t + k1)
        if k1 is None: roots = []; s0 = certified_sign(k2)
        else:
            s0 = certified_sign(k1); t = -k1 / k2
            roots = [t] if t > 0 else ([] if t < 0 else None)
            if roots is None: raise Uncertain()
    else:
        s0 = certified_sign(k0); k1v = k1 if k1 is not None else arb(0)
        D = k1v * k1v - 4 * k2 * k0
        if D < 0: roots = []
        else:
            if D > 0: r = D.sqrt()
            else:   # tangency: r = sqrt(max(D, 0)) on a ball over [0, sqrt(D_hi)] encloses both sides
                sh = arb(D.upper()).sqrt().upper(); r = arb(arb(sh).mid() / 2, (arb(sh) / 2 * (1 + arb(1) / 10 ** 6)).upper())
            t1, t2 = (-k1v - r) / (2 * k2), (-k1v + r) / (2 * k2)
            if k2 < 0: t1, t2 = t2, t1
            roots = []
            for t in (t1, t2):
                if t > 0: roots.append(t)
                elif not t < 0: raise Uncertain()
    pts = ["-inf"] + [t.log() for t in roots] + ["inf"]
    tot = arb(0); sgn = s0
    for lo, hi in zip(pts[:-1], pts[1:]):
        if sgn > 0:
            for c, m in terms:
                tot += c * (Phi(hi if hi == "inf" else hi - m) - Phi(lo if lo == "-inf" else lo - m))
        sgn = -sgn
    return tot

E = (arb(1) / 2).exp()
def weights(g1, g2):
    P12 = npdf(g1, -2) * npdf(g2, 0) + npdf(g1, 0) * npdf(g2, 2)
    Q12 = npdf(g1, -1) * npdf(g2, 0) + npdf(g1, 0) * npdf(g2, 1)
    P3 = npdf(g1, 0) * npdf(g2, 0)
    return P12 / 3, Q12 / 3, P3 / 3

def diff(g1, g2):
    P12, Q12, W3 = weights(g1, g2)
    # ANTI: P means 0 and 2, Q means 0 and 1; shift z = g3 - 1 -> P at -1 and 1, Q at -1 and 0
    anti = I_terms([(P12, -1), (W3, 1), (-E * Q12, -1), (-E * W3, 0)])
    # SAME: P means 0 and 0, Q means 0 and -1 (no shift)
    same = I_terms([(P12 + W3, 0), (-E * Q12, 0), (-E * W3, -1)])
    return same - anti

def ball(lo, hi):
    return arb(((lo + hi) / 2).mid(), (((hi - lo) / 2) * (1 + arb(1) / 1000)).upper())

def certify(a1, b1, a2, b2, depth, stats):
    try:
        d = diff(ball(a1, b1), ball(a2, b2))
        if d > 0:
            stats["boxes"] += 1
            area = (b1 - a1) * (b2 - a2)
            stats["gain"] += area * arb(d.lower()); stats["minlow"] = min(stats["minlow"], float(d.lower()))
            return True
    except Uncertain:
        pass
    if depth == 0:
        stats["fail"] = (float(a1.mid()), float(b1.mid()), float(a2.mid()), float(b2.mid())); return False
    m1, m2 = (a1 + b1) / 2, (a2 + b2) / 2
    return all(certify(x0, x1, y0, y1, depth - 1, stats) for x0, x1 in ((a1, m1), (m1, b1)) for y0, y1 in ((a2, m2), (m2, b2)))

if __name__ == "__main__":
    a1, b1, a2, b2 = (arb(x) for x in sys.argv[1:5]); n1, n2, depth = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
    stats = dict(boxes=0, gain=arb(0), minlow=1.0, fail=None)
    w1, w2 = (b1 - a1) / n1, (b2 - a2) / n2
    ok = all(certify(a1 + w1 * i, a1 + w1 * (i + 1), a2 + w2 * j, a2 + w2 * (j + 1), depth, stats)
             for i in range(n1) for j in range(n2))
    if ok:
        print("CERTIFIED: I3_SAME - I3_ANTI > 0 on all (g1, g2) in [%s, %s] x [%s, %s] (%d balls); min lower end %.4e" %
              (*sys.argv[1:5], stats["boxes"], stats["minlow"]))
        print("so D(step-3 switch on the box) - D(P_S||Q_S) >= %s (rigorous)" % stats["gain"].lower().str(6))
    else:
        print("NOT CERTIFIED; first failing box", stats["fail"])
    p0 = diff(arb("0.1"), arb("2.25"))
    print("point (0.1, 2.25): I3_SAME - I3_ANTI =", p0.str(10))
    json.dump(dict(box=sys.argv[1:5], ok=ok, balls=stats["boxes"], min_lower=stats["minlow"],
                   gain_lower=stats["gain"].lower().str(10) if ok else None, fail=stats["fail"]),
              open("cert_t3_arb.json", "w"), indent=1)
