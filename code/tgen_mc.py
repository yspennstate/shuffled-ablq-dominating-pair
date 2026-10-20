# general T (19 Oct 2026). Last-step switch at any T, s = 1.
# Steps 1..T-1 answer as conjectured (ANTI); step T answers SAME on a set of prefixes. Given the prefix g_1..g_m
# (m = T-1, coordinates sign-normalised as in t3_dp.py), the last-step inner integral has weights
#   alpha = (1/T) sum_t phi(g_t - 2) prod_{s != t} phi(g_s),  gamma = (1/T) sum_t phi(g_t - 1) prod_{s != t} phi(g_s),
#   beta  = (1/T) prod_t phi(g_t)                                         (i in batch T: same under x and x'),
# and I3 is homogeneous of degree 1 in (alpha, beta, gamma). With beta = 1:  A = alpha/beta = e^{-1} sum x_t^2,
# G = gamma/beta = sum x_t, x_t = exp(g_t - 1/2) (sigma = 1; for general sigma x_t = exp((g_t - 1/2)/s^2), A = e^{-1/s^2} sum x_t^2). Hence
#   D(P_S || Q_S) = (1/T) E_{g ~ N(0, I_m)} I_ANTI(A, G),   gain of the switch = (1/T) E (I_SAME - I_ANTI)_+(A, G).
# For m = 1, (A, G) runs over the curve e A = G^2 (the T = 2 problem); for m >= 2 over the open region
# G^2/m < e A < G^2, which contains points just below that curve - so the T = 2 certificate transfers to every T.
import sys, json, numpy as np
from scipy.special import ndtr

def I_anti(A, G, E, sg=1.0):
    # z = g_T - 1: P = A phi(z+1) + phi(z-1), Q = E G phi(z+1) + E phi(z); {P > Q} <=> t^2 - E^{3/2}... see below
    # sum c phi(z - m)/phi(z) = c e^{-m^2/2} e^{m z}; with t = e^z: q(t) = e^{-1/2} t^2 - E t + (A - E G) e^{-1/2}
    # general sigma: phi_s(z - m)/phi_s(z) = e^{-m^2/(2 s^2)} t^m with t = e^{z/s^2}; y = s^2 log t; Phi((y - m)/s)
    h = np.exp(-.5 / sg ** 2)
    a, b, c = h, -E, (A - E * G) * h
    D = b * b - 4 * a * c
    sq = np.sqrt(np.maximum(D, 0))
    tlo, thi = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
    terms = [(A, -1.), (1., 1.), (-E * G, -1.), (-E, 0.)]
    def mass(lo, hi):   # sum c [Phi(hi - m) - Phi(lo - m)]
        return sum(cc * (ndtr((hi - m) / sg) - ndtr((lo - m) / sg)) for cc, m in terms)
    whole = mass(-np.inf, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        ylo = np.where(tlo > 0, sg ** 2 * np.log(np.where(tlo > 0, tlo, 1)), -np.inf)
        yhi = sg ** 2 * np.log(np.where(thi > 0, thi, 1))
    two = mass(-np.inf, ylo) + mass(yhi, np.inf)        # q > 0 outside [tlo, thi] (tlo may be <= 0)
    return np.where(D <= 0, whole, np.where(thi > 0, two, whole))

def I_same(A, G, E, sg=1.0):
    # P = (A + 1) phi(z), Q = E G phi(z) + E phi(z + 1): q(t) = (A + 1 - E G) t - E e^{-1/2}, positive iff t > t0
    k1 = A + 1 - E * G
    with np.errstate(divide="ignore", invalid="ignore"):
        y0 = sg ** 2 * np.log(E * np.exp(-.5 / sg ** 2) / np.where(k1 > 0, k1, 1))
    m = (A + 1 - E * G) * (1 - ndtr(y0 / sg)) - E * (1 - ndtr((y0 + 1) / sg))
    return np.where(k1 > 0, m, 0.0)

def run(T, eps, n, seed=1, chunk=2 * 10 ** 5, sg=1.0):
    E = np.exp(eps); rng = np.random.default_rng(seed + T)
    sa, sa2, sd, sd2, k = 0., 0., 0., 0., 0
    while k < n:
        g = sg * rng.standard_normal((min(chunk, n - k), T - 1)); x = np.exp((g - .5) / sg ** 2)
        A, G = (x * x).sum(1) * np.exp(-1 / sg ** 2), x.sum(1)
        ia = I_anti(A, G, E, sg); d = np.maximum(I_same(A, G, E, sg) - ia, 0)
        sa += ia.sum(); sa2 += (ia * ia).sum(); sd += d.sum(); sd2 += (d * d).sum(); k += len(g)
    ma, mg = sa / n, sd / n
    return dict(T=T, eps=eps, sigma=sg, n=n, conj=ma / T, conj_se=np.sqrt((sa2 / n - ma ** 2) / n) / T,
                gain=mg / T, gain_se=np.sqrt((sd2 / n - mg ** 2) / n) / T)

if __name__ == "__main__":
    Ts = [int(t) for t in sys.argv[1].split(",")]; eps = float(sys.argv[2]); n = int(float(sys.argv[3]))
    sg = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    out = []
    for T in Ts:
        r = run(T, eps, n, sg=sg); out.append(r)
        print("sigma %.2f T %2d eps %.2f  D(P_S||Q_S) %.6f +- %.6f   last-step switch gain %.4e +- %.1e" %
              (sg, T, eps, r["conj"], r["conj_se"], r["gain"], r["gain_se"]), flush=True)
        json.dump(out, open(sys.argv[4], "w"), indent=1)
