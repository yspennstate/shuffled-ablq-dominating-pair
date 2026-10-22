# T353 independent proof review — exact conjecture refuted

Date: 22 October 2026. Reviewer: the Codex session responding to the owner's request, independent of the Claude sessions reported as finding and writing the paper. This is an independent AI mathematical review and computational replay, not external human peer review or a journal acceptance.

**Verdict: PASS for the refutation of Chua et al., Conjecture 3.2, and for the manuscript's main theorem at sigma = 1, epsilon = 1/2, every T >= 2, every batch size b >= 1, and every dimension d >= 1.** Both explicit hockey-stick lower bounds were also verified. The manuscript needs the corrections below before a clean reviewed revision is advertised. This verdict does not certify every ancillary numerical, KL, or Renyi claim in the manuscript.

**Claim status: RESOLVED BY REFUTATION — independently reviewed.** This settles the proposed pair, while exact/tight shuffled privacy accounting remains open.

## Pinned sources

- Original conjecture: Chua, Ghazi, Kamath, Kumar, Manurangsi, Sinha, and Zhang, *How Private are DP-SGD Implementations?*, ICML 2024, [arXiv:2403.17673v2](https://arxiv.org/html/2403.17673v2), Definitions 2.1/2.2 and Conjecture 3.2.
- Reviewed manuscript: Yitzchak Shmalo, *The conjectured dominating pair for DP-SGD with shuffled batches is not dominating*, [public repository](https://github.com/yspennstate/shuffled-ablq-dominating-pair), commit **6d3a625a0f4cfeecb5dac3467ed7570bf7c7a597**.
- [Pinned paper PDF](https://github.com/yspennstate/shuffled-ablq-dominating-pair/blob/6d3a625a0f4cfeecb5dac3467ed7570bf7c7a597/paper/main.pdf) and [pinned TeX](https://github.com/yspennstate/shuffled-ablq-dominating-pair/blob/6d3a625a0f4cfeecb5dac3467ed7570bf7c7a597/paper/main.tex). The proof audit used the full TeX; this review did not independently typeset or visually inspect that PDF.
- Source hashes of the TeX and the three retrieved Python files match the repository's published SHA256 manifest exactly. See `MANIFEST.json` and `REVIEW_CHECKS.txt`.

## Why the main claim passes

The proposed dominating pair consists of uniform Gaussian mixtures with shifts 2 and 1 in the distinguished coordinate. The counterexample uses a valid directed zero-out neighbor, unit-norm queries, the stipulated uniform shuffle into fixed-size batches, independent Gaussian noise, and a final query chosen from previous outputs only. Thus it addresses the stated conjecture rather than a different sampling or adjacency model.

I checked the mixture weights and the exact last-step switching identity from the mechanism definition. The raw-to-centered transcript map needs to be described as a triangular Borel bijection, but the construction and the divergence comparison are valid. The query values also embed in any higher dimension, with all other coordinates independent ancillary noise.

The manuscript's two-step interval certificate passes. Its three-step certificate passes on the stated rectangle, and its two-variable image argument propagates positive gain to all larger T. The rounded ratio quoted inside that image argument is inaccurate; the exact inequality 1/2 < (1+r^2)/(1+r)^2 < 1 for r > 1 suffices and repairs it without changing the proof.

There is also a simpler analytic route. The accompanying `ANALYTIC_PROOF.md` shows that switching on the prefix cube [3,4]^(T-1) makes the ANTI signed density everywhere positive, while SAME has the same signed integral and a nonempty negative tail. Removing that negative part strictly increases the hockey-stick integral. This proves the main qualitative theorem without relying on software. Its last section derives a stronger, parameter-dependent counterexample for every positive sigma and every finite epsilon; that new strengthening is explicitly separate from the pinned manuscript and awaits another review before being promoted as an independently reviewed extension.

## Reproduction results

Environment: Python 3.12.14, python-flint 0.9.0; the shipped scripts set 200-bit precision for T=2 and 128-bit precision for T=3. The independent density quadrature uses SciPy and shares no root/CDF code with the shipped certificates.

| Check | Observed result | Assessment |
|---|---|---|
| T=2, raw interval [-1.6,-1.1], 5,000 balls | Gain lower endpoint approximately 0.000104142 | Exceeds the claimed 0.000104 |
| T=2 interval coverage | Every ball contains both exact rational endpoints of its assigned subinterval | Pass |
| T=2 negative control, paper coordinate z1=1.7 | SAME minus ANTI approximately -0.00995935560886641 | Correctly negative |
| T=3, raw box [-0.1,0.3] x [2.2,2.3], initial grid 200 x 50, depth limit 14 | 33,367 leaf boxes; summed lower endpoint 0.000028408182291354227666 | Exceeds the claimed 0.0000225 |
| T=3 interval coverage | 82,312 one-dimensional balls checked, including subdivided calls | Every evaluated ball contains its endpoint enclosures |
| Independent quadrature, T=2 z1=2.5 | ANTI 0.06357796524530177; SAME 0.06415608792541296 | Agrees with certificate |
| Independent quadrature, T=3 paper prefix (-0.1,2.25) | SAME minus ANTI 0.001535260930046847 | Agrees with certificate |

The raw-to-paper coordinates are z1=1-g1 for the T=2 script, and y1=-g1, y2=g2 for the T=3 script. Consequently the T=3 raw box is exactly the paper's [-0.3,0.1] x [2.2,2.3]. There is no sign or adjacency mismatch in the certified examples.

The first T=3 replay used a 10 x 10 initial grid and certified a lower bound of approximately 1.86382e-5. That is positive but too weak to reproduce the paper's stronger 2.25e-5 bound. Refining the initial grid to 200 x 50 gave the bound above. This is expected enclosure dependence on the partition, not a failed theorem. The original 21,539-box partition was not reconstructed; the independently checked finer partition proves the stated threshold. Record exact command-line parameters with future archived outputs.

Reproduce from a checkout of the pinned source placed under `source/` beside this review:

```sh
python -m pip install python-flint==0.9.0 scipy
python source/code/cert_t2_arb.py -1.6 -1.1 5000
python source/code/cert_t3_arb.py -0.1 0.3 2.2 2.3 200 50 14
python review_checks.py
```

`review_checks.py` checks the published source hashes before executing the certificate routines, checks interval coverage, asserts the two numerical thresholds, tests a negative control, and independently integrates the point densities. Exact runtime dependency versions are in `MANIFEST.json`. Its floating-point quadrature is a consistency check, not a rigorous enclosure or the basis of the verdict.

## Corrections for the reviewed public revision

1. **Adaptive coordinate map.** Replace the statement that g maps to z by a fixed translation with the triangular Borel-bijection argument in `ANALYTIC_PROOF.md`. A fixed translation is correct for all-ANTI, but not for the switched policy.
2. **No upper cutoff at 12.** Remove the claim that SAME wins exactly between about 2.005 and 12. The analytic argument proves strict gain for every prefix z1 >= 3 at T=2. Twelve is a numerical search/precision artifact, not an endpoint of the improvement set.
3. **Wrong vanishing coefficient.** At z1=2, alpha - sqrt(e) gamma = 0, whereas alpha - sqrt(e) gamma + beta = beta > 0. Correct the explanation of the undecided interval around 2. The latter coefficient instead vanishes at approximately 0.1759954606 and 1.8240045394.
4. **Rounded ratio enclosure.** The endpoint ratios are approximately 0.8056105903987492 and 0.8712834016484531, so [0.806,0.871] is not a valid enclosure. Replace the rounded claim with the exact inequality 1/2 < ratio < 1. It is all the all-T proof uses.
5. **Reproducibility.** Include the full certificate commands, dependency versions, and refreshed hashes. The 200 x 50 T=3 run in this package supplies a verified alternative to the original archived partition.
6. **Numerical labeling.** The rounded values 0.22127 and 1.966e-3 in the numerical table should be labeled closed-form quadrature estimates rather than exact numbers. The explicitly certified lower bounds have a different evidentiary status.

None of these changes removes the strict counterexample. The first correction supplies missing precision in the exposition; the next three repair false incidental statements. Keep the KL/Renyi claims outside this review's approval scope unless separately audited. This review also does not verify funding, affiliation, or competing-interest declarations.

## Revision and scope

The corrected 22 October manuscript implements the six corrections above and includes the analytic proof at sigma=1, epsilon=1/2. Its regenerated PDF was rendered and inspected separately after the baseline audit. The all-parameter extension in ANALYTIC_PROOF.md remains a reviewer-derived proposal awaiting another independent review.

This is a scoped independent AI review of the main hockey-stick results. It is not a certificate of every ancillary claim in the paper, external human peer review, or an exhaustive novelty search.
