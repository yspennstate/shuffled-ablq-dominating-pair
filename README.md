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
