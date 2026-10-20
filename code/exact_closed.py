# T=2 (19 Oct 2026). Closed form, no root search, no grid in g2.
# Frame of t2b1_adaptive.py: a1 = -1, v1 = +1; step 2 ANTI (a2,v2) = (1,-1) (conjectured), SAME (1,1).
# At fixed g1 the step-2 difference is h(y) = sum_k c_k phi(y - m_k), m_k in {-1,0,1}. Dividing by phi(y) > 0 and
# writing t = exp(y/s^2) > 0, phi(y-m)/phi(y) = exp(-m^2/(2 s^2)) t^m, so t*h/phi = k2 t^2 + k1 t + k0: the set
# {h > 0} is read off a quadratic's roots, and I(g1) = sum over its intervals of sum_k c_k [Phi((y-m_k)/s)].
# Outer integral by mpmath.quad at 30 digits, split at the points where I_SAME = I_ANTI.
import sys, json
import mpmath as mp
mp.mp.dps = 30

def I(g1, s, e, a2, v2, rev, a1=-1, v1=1):
    pdf = lambda x, m: mp.npdf(x, m, s)
    wa, w0, wc = pdf(g1, a1) / 2, pdf(g1, 0) / 2, pdf(g1, v1) / 2
    P = [(wa, v2), (wc, a2)]; Q = [(w0, v2), (wc, 0)]
    if rev: P, Q = Q, P
    terms = P + [(-e * c, m) for c, m in Q]
    k = {-1: mp.mpf(0), 0: mp.mpf(0), 1: mp.mpf(0)}
    for c, m in terms:
        k[m] += c * mp.e ** (-mp.mpf(m) ** 2 / (2 * s * s))
    k0, k1, k2 = k[-1], k[0], k[1]
    # sign of q(t) = k2 t^2 + k1 t + k0 on t > 0 -> intervals in y = s^2 ln t
    roots = []
    if k2 == 0:
        if k1 != 0: roots = [-k0 / k1]
    else:
        D = k1 * k1 - 4 * k2 * k0
        if D > 0:
            r = mp.sqrt(D); roots = sorted([(-k1 - r) / (2 * k2), (-k1 + r) / (2 * k2)])
    ys = [s * s * mp.log(t) for t in roots if t > 0]
    cuts = [mp.ninf] + ys + [mp.inf]
    qf = lambda t: k2 * t * t + k1 * t + k0
    tot = mp.mpf(0)
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if lo == mp.ninf and hi == mp.inf: tm = mp.mpf(1)
        elif lo == mp.ninf: tm = mp.e ** (hi / (s * s)) / 2
        elif hi == mp.inf: tm = mp.e ** (lo / (s * s)) * 2
        else: tm = mp.e ** ((lo + hi) / (2 * s * s))
        if qf(tm) > 0:
            tot += sum(c * (mp.ncdf(hi, m, s) - mp.ncdf(lo, m, s)) for c, m in terms)
    return tot

def run(s, eps, rev):
    s = mp.mpf(s); e = mp.e ** mp.mpf(eps)
    A = lambda g: I(g, s, e, 1, -1, rev); S = lambda g: I(g, s, e, 1, 1, rev)
    d = lambda g: S(g) - A(g)
    lo, hi = -2 - 12 * s, 2 + 12 * s
    n = 600
    gs = [lo + (hi - lo) * j / n for j in range(n + 1)]
    dv = [d(g) for g in gs]
    def root(a, b):
        # at large eps the difference is ~1e-17 and anderson's residual test (tol ~ eps_mach^2) can miss by a factor 2
        try: return mp.findroot(d, (a, b), solver='anderson')
        except ValueError: return mp.findroot(d, (a, b), solver='anderson', verify=False)
    sw = [root(gs[j], gs[j + 1]) for j in range(n) if dv[j] * dv[j + 1] < 0]
    pts = [mp.ninf] + sw + [mp.inf]
    conj = mp.quad(A, [mp.ninf, -3, -2, -1, 0, 1, 2, mp.inf])
    gain = mp.mpf(0); sets = []
    for a, b in zip(pts[:-1], pts[1:]):
        m = (b - 1 if a == mp.ninf else (a + 1 if b == mp.inf else (a + b) / 2))
        if d(m) > 0:
            seg = [a, b]
            if a == mp.ninf: seg = [mp.ninf, min(lo, b - 1), b]
            if b == mp.inf: seg = [a, max(hi, a + 1), mp.inf]
            gain += mp.quad(d, seg); sets.append((float(a), float(b)))
    return conj, gain, sets, float(max(dv))

if __name__ == "__main__":
    out = {}
    for s in sys.argv[1].split(","):
        for eps in sys.argv[2].split(","):
            r = {}
            for rev in (False, True):
                conj, gain, sets, dmax = run(s, eps, rev)
                r["rev" if rev else "fwd"] = dict(conj=mp.nstr(conj, 15), gain=mp.nstr(gain, 15), same_on=sets, max_pointwise=dmax)
            cf, cr = mp.mpf(r["fwd"]["conj"]), mp.mpf(r["rev"]["conj"])
            gap = max(cf + mp.mpf(r["fwd"]["gain"]), cr + mp.mpf(r["rev"]["gain"])) - max(cf, cr)
            r["gap_over_max"] = mp.nstr(gap, 10)
            out[s + "_" + eps] = r
            print("s %s eps %s | conj fwd %s rev %s | gain fwd %s rev %s | gap over max %s | SAME on %s" %
                  (s, eps, r["fwd"]["conj"][:10], r["rev"]["conj"][:10], r["fwd"]["gain"][:10], r["rev"]["gain"][:10],
                   r["gap_over_max"], r["fwd"]["same_on"]), flush=True)
            json.dump(out, open(sys.argv[3], "w"), indent=1)
