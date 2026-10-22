"""Independent review checks for pinned T353 paper; numerical checks are not the proof.

Run from this directory with python-flint 0.9.0 and scipy installed.
The analytic proof is in ANALYTIC_PROOF.md. This verifies source hashes,
T2 interval coverage, T3 interval coverage and reported thresholds, and
independent one-dimensional density quadrature at the paper's test points.
"""
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import flint
from flint import arb, ctx
from scipy.integrate import quad
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
PIN = "6d3a625a0f4cfeecb5dac3467ed7570bf7c7a597"
HASHES = {
    "paper/main.tex": "bf54d1eb16263af5386c54f4fe4e7eaf358040d6d9a596b285c21e92dce14eb2",
    "code/cert_t2_arb.py": "85bcd911b9a2061d7db4a27be8f9cecf8b179e83a3f4c82194e239b75be80bd4",
    "code/cert_t3_arb.py": "64d7b783be6f898cd6cb606278b0bb4664d371430eff0589078b9905e621beeb",
    "code/independent_check.py": "586c5020867cd2dec904fce4b18f4b70969324d51dfd2cb73d9121bac0197fc1",
}


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "source/code" / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("Python", platform.python_version(), "python-flint", flint.__version__, flush=True)
for path, expected in HASHES.items():
    actual = hashlib.sha256((ROOT / "source" / path).read_bytes()).hexdigest()
    assert actual == expected, (path, actual)
print("PASS: four source SHA256 hashes match published manifest at", PIN, flush=True)

t2 = module("cert_t2_arb")
lo, hi, n = arb("-1.6"), arb("-1.1"), 5000
w = (hi - lo) / n
worst = None
for j in range(n):
    mid = lo + w * (2 * j + 1) / 2
    g = arb(mid.mid(), w.mid() / 2 * (1 + arb(1) / 1000).mid())
    # Containment of the entire Arb endpoint enclosure implies containment
    # of the exact rational endpoint. Exact domain [-8/5,-11/10].
    left, right = arb(-8) / 5 + arb(j) / 10000, arb(-8) / 5 + arb(j + 1) / 10000
    assert g.contains(left) and g.contains(right), j
    d = t2.I(g, 1, 1, t2.arb("0.5").exp()) - t2.I(g, 1, -1, t2.arb("0.5").exp())
    assert d > 0, j
    worst = d.lower() if worst is None else min(worst, d.lower())
assert worst * (hi - lo) > arb("0.000104")
print("PASS: T2 exact endpoint coverage, all 5000 lower bounds positive; gain > 0.000104", flush=True)
control = t2.I(arb("-0.7"), 1, 1, arb("0.5").exp()) - t2.I(arb("-0.7"), 1, -1, arb("0.5").exp())
assert control < 0
print("PASS: negative control z1=1.7, same-minus-anti =", control.str(15), flush=True)

t3 = module("cert_t3_arb")
original_ball = t3.ball
covered = 0


def checked_ball(a, b):
    global covered
    result = original_ball(a, b)
    assert result.contains(a) and result.contains(b), (a, b, result)
    covered += 1
    return result


t3.ball = checked_ball
a1, b1, a2, b2 = map(arb, ("-0.1", "0.3", "2.2", "2.3"))
n1, n2, depth = 200, 50, 14
w1, w2 = (b1 - a1) / n1, (b2 - a2) / n2
stats = dict(boxes=0, gain=arb(0), minlow=1.0, fail=None)
for i in range(n1):
    for j in range(n2):
        assert t3.certify(a1 + i*w1, a1 + (i+1)*w1, a2 + j*w2, a2 + (j+1)*w2, depth, stats)
assert stats["gain"] > arb("0.0000225")
print("PASS: T3 endpoint coverage on", covered, "one-dimensional balls;", stats["boxes"], "leaf boxes", flush=True)
print("T3 summed gain lower endpoint:", stats["gain"].lower().str(20), flush=True)

# Independent density quadrature: no use of the shipped root or CDF routines.
from math import exp, log, sqrt
E = exp(0.5)


def direct_inner(al, be, ga, a, c):
    def f(s):
        return max(al*norm.pdf(s) + be*norm.pdf(s-a) - E*(ga*norm.pdf(s) + be*norm.pdf(s-c)), 0.0)
    return quad(f, -30, 30, points=[-5, 0, 5], limit=400, epsabs=1e-15, epsrel=1e-12)


y = 2.5
al, be, ga = norm.pdf(y-2)/2, norm.pdf(y)/2, norm.pdf(y-1)/2
aa, ss = direct_inner(al, be, ga, 2, 1), direct_inner(al, be, ga, 0, -1)
assert abs(aa[0] - 0.0635779652453018) < 1e-12
assert abs(ss[0] - 0.0641560879254130) < 1e-12
print("PASS: independent T2 point quadrature", aa, ss, flush=True)
y1, y2 = -0.1, 2.25
al = (norm.pdf(y1-2)*norm.pdf(y2) + norm.pdf(y1)*norm.pdf(y2-2))/3
ga = (norm.pdf(y1-1)*norm.pdf(y2) + norm.pdf(y1)*norm.pdf(y2-1))/3
be = norm.pdf(y1)*norm.pdf(y2)/3
aa, ss = direct_inner(al, be, ga, 2, 1), direct_inner(al, be, ga, 0, -1)
assert abs(ss[0] - aa[0] - 0.001535260930) < 1e-12
print("PASS: independent T3 point difference", ss[0] - aa[0], flush=True)

# This is a numerical check of corrected incidental statements, not a proof.
for y in (2.1, 2.6):
    r = exp(y)
    print("All-T ratio at y2-y1 =", y, ":", (1+r*r)/(1+r)**2)
print("C+1 at y=2:", exp(2*2-2)-exp(2)+1)
print("C+1 roots:", [log((exp(2)+sign*sqrt(exp(4)-4*exp(2)))/2) for sign in (-1, 1)])
print("ALL REVIEW CHECKS PASSED", flush=True)
