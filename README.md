# The conjectured dominating pair for DP-SGD with shuffled batches is not dominating

Paper and code for a counterexample to Conjecture 3.2 of Chua, Ghazi, Kamath, Kumar, Manurangsi, Sinha and Zhang,
*How private are DP-SGD implementations?* (ICML 2024, arXiv:2403.17673).

The conjecture says that the pair of Gaussian mixtures

    P_S = (1/T) sum_t N(2 e_t, sigma^2 I),   Q_S = (1/T) sum_t N(e_t, sigma^2 I)

tightly dominates the adaptive batch linear query mechanism with shuffled batches, for every adaptive query method.
It fails for every number of steps T >= 2 and every batch size. A query method that answers like the conjectured worst
case at every step but the last, and changes the last query when the earlier outputs suggest that the differing record
sat in an early batch, has a larger hockey-stick divergence. At T = 2, sigma = 1, eps = 1/2 it reaches 0.22323 against
0.22127 for the pair, and an Arb interval certificate bounds the excess below by 1.04e-4. The same switch also beats the
pair in KL divergence and in Renyi divergence of every order tested, from 1/4 to 5, in both directions.

## Independent review and corrected revision — 22 October 2026

The main hockey-stick refutation passed an independent Codex mathematical review and replay. The revised paper adds a short analytic proof for every T >= 2 at sigma = 1, epsilon = 1/2, and corrects the adaptive-coordinate explanation and several incidental statements. This resolves the proposed dominating-pair conjecture by refutation; exact/tight shuffled privacy accounting remains open.

- [Review and scope](reviews/2026-10-22/REVIEW.md)
- [Self-contained analytic proof](reviews/2026-10-22/ANALYTIC_PROOF.md)
- [Reproduction log](reviews/2026-10-22/REVIEW_CHECKS.txt) and [manifest](reviews/2026-10-22/MANIFEST.json)
- [Corrected paper](paper/main.pdf)

The review covers the main theorem and the T = 2 and T = 3 hockey-stick bounds. It does not constitute external human peer review and does not audit the KL/Renyi tables or all numerical optimization claims. The all-parameter extension in the analytic-proof note is a separate reviewer-derived result awaiting another independent review.

The original source files at commit `6d3a625a0f4cfeecb5dac3467ed7570bf7c7a597` are preserved under `reviews/2026-10-22/source/` for exact-hash audit reproduction. The corrected manuscript is `paper/main.tex`.

Run the independent checks from the repository root:

```sh
python -m pip install python-flint==0.9.0 scipy==1.17.0 numpy==2.3.5 mpmath==1.3.0
python reviews/2026-10-22/review_checks.py
```

The primary certificate commands are:

```sh
python code/cert_t2_arb.py -1.6 -1.1 5000
python code/cert_t3_arb.py -0.1 0.3 2.2 2.3 200 50 14
```

The second command gives a lower bound exceeding 2.8408e-5 on a fresh partition, which verifies the paper's stated 2.25e-5 threshold.

## Contents

- `paper/main.tex`, `paper/main.pdf` - the paper.
- `code/cert_*_arb.py` - interval-arithmetic certificates (python-flint / Arb). `cert_t2_arb.py` and `cert_t3_arb.py`
  prove the hockey-stick results at T = 2 and T = 3; the others prove the KL, chi-square and Renyi results.
- `code/exact_closed.py`, `t2b1_adaptive.py`, `nonadaptive.py`, `t3_dp.py`, `t3_step3.py`, `t3_mc.py`, `tgen_mc.py`,
  `tgen_bigT_s05.py`, `chi2_adaptive.py`, `renyi_alpha_grid.py`, `renyi_nonadaptive.py` - numerical values.
- `code/independent_check.py` - a short recomputation of the main T = 2 and T = 3 values with plain quadrature.
- `code/outputs/` - the output of each script as it was run.
- `SHA256SUMS.txt` - checksums of every file above.

Most scripts work in the raw b = 1 coordinate of `t2b1_adaptive.py`, in which the step-1 output is g_1 = 1 - z_1 in terms of
the paper's z_1 (so the certified interval z_1 in [2.1, 2.6] is g_1 in [-1.6, -1.1]); the header of each script states its coordinates.

## Requirements

Python 3 with numpy, scipy, mpmath and python-flint (tested with python-flint 0.9.0).

## Citation

Y. Shmalo, *The conjectured dominating pair for DP-SGD with shuffled batches is not dominating*, 2026.
