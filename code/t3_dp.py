# T=3 (19 Oct 2026). Exact nested maximum for adaptive ABLQ_S at T = 3, b any, d = 1.
# Shifted model (REFUTATION_T2.md): step t answers (a_t, v_t) = (psi_t(u), psi_t(w)); if record i sits in batch t,
# g_t ~ N(a_t - v_t, s^2) under x and N(-v_t, s^2) under x'; any other batch gives N(0, s^2). i's batch is uniform.
# The hockey-stick integral is int dg1 int dg2 int dg3 (p - e q)_+, and the step-t choice enters only the g_t
# factor, so the adaptive optimum is V3(g1,g2) = max_c3 int (.)_+ dg3, V2(g1) = max_c2 int V3 dg2, total int V2 dg1.
# Step 1 fixed at (a1, v1) = (-1, 1); the conjectured pair is (1, -1) at steps 2 and 3 (coordinate-1 sign flip).
import sys, json, itertools, numpy as np

def phi(x, s): return np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi))

def run(s, eps, N, M, choices, rev):
    e = np.exp(eps)
    g = np.linspace(-6 * s - 2, 6 * s + 2, N); h = g[1] - g[0]
    y = np.linspace(-8 * s - 2, 8 * s + 2, M); hy = y[1] - y[0]
    a1, v1 = -1.0, 1.0
    # coordinate-1 factors per position k of i (k = 0,1,2) under x (P) and x' (Q)
    P1 = [phi(g - (a1 - v1), s), phi(g, s), phi(g, s)]
    Q1 = [phi(g + v1, s), phi(g, s), phi(g, s)]
    f0y = phi(y, s)
    # step-3 inner tables: for each c3, columns of phi(y - m) needed
    def inner3(WP, WQ):            # WP, WQ: (3, n) position weights at fixed path; returns (K, n) inner integrals
        out = []
        for a, v in choices:
            fx = phi(y - (a - v), s); fq = phi(y + v, s)
            p = (WP[0] + WP[1])[:, None] * f0y + WP[2][:, None] * fx
            q = (WQ[0] + WQ[1])[:, None] * f0y + WQ[2][:, None] * fq
            if rev: p, q = q, p
            out.append(np.maximum(p - e * q, 0).sum(1) * hy / 3)
        return np.array(out)
    k_anti = choices.index((1.0, -1.0))
    tot_conj = tot_ad = 0.0; switch2 = 0
    for i1 in range(N):
        best_c2 = -1.0; conj_here = None
        for j2, (a, v) in enumerate(choices):
            # coordinate-2 factors over the g grid (as g2)
            P2 = [phi(g, s), phi(g - (a - v), s), phi(g, s)]
            Q2 = [phi(g, s), phi(g + v, s), phi(g, s)]
            WP = np.array([P1[k][i1] * P2[k] for k in range(3)]); WQ = np.array([Q1[k][i1] * Q2[k] for k in range(3)])
            I3 = inner3(WP, WQ)                       # (K, N)
            val_ad = I3.max(0).sum() * h
            if j2 == k_anti: conj_here = I3[k_anti].sum() * h
            if val_ad > best_c2: best_c2, bj = val_ad, j2
        tot_conj += conj_here * h; tot_ad += best_c2 * h; switch2 += (bj != k_anti)
    return tot_conj, tot_ad, switch2

if __name__ == "__main__":
    s, eps, N, M = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    grid = [float(x) for x in sys.argv[5].split(",")]
    choices = [(a, v) for a in grid for v in grid]
    out = {}
    for rev in (False, True):
        c, a, sw = run(s, eps, N, M, choices, rev)
        out["rev" if rev else "fwd"] = dict(conj=c, adaptive=a, gain=a - c, g1_points_switching_step2=sw)
        print("T=3 s %.2f eps %.2f N %d M %d %s conj %.6f adaptive %.6f gain %+.3e (step-2 switch at %d of %d g1)" %
              (s, eps, N, M, "Q||P" if rev else "P||Q", c, a, a - c, sw, N), flush=True)
        json.dump(out, open(sys.argv[6], "w"), indent=1)
