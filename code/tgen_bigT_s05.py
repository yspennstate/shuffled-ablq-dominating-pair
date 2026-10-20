# does the s = 0.5 last-step gain stay flat for T well past e^{1/s^2} ~ 55? (19 Oct 2026)
# Heuristic: x_t = exp((g_t - 1/2)/s^2) has log-sd 1/s = 2, E x = 1, E x^2 = e^{1/s^2}; the ratio sum x^2 / G^2 stays O(1)
# (one big term dominates) while m << e^{1/s^2}, so (A, G) sits in the same part of the region and the gain need not decay.
# Once m >> e^{1/s^2} the law of large numbers pushes sum x^2 / G^2 -> e^{1/s^2}/m and the gain should start to fall.
import json, sys
from tgen_mc import run
out = []
for T in [int(t) for t in sys.argv[1].split(",")]:
    r = run(T, 0.1, int(float(sys.argv[2])), chunk=2 * 10 ** 4, sg=0.5); out.append(r)
    print("T %4d conj %.5f +- %.5f gain %.4e +- %.1e  T*gain %.4e" % (T, r["conj"], r["conj_se"], r["gain"], r["gain_se"], T * r["gain"]), flush=True)
    json.dump(out, open("tgen_bigT_s05.json", "w"), indent=1)
