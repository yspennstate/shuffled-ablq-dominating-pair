# An analytic last-step counterexample to the shuffled-ABLQ dominating-pair conjecture

Independent review derivation, 22 October 2026. This argument uses the construction in Yitzchak Shmalo's T353 manuscript, pinned at commit `6d3a625a0f4cfeecb5dac3467ed7570bf7c7a597`. It supplies a short analytic proof of the main qualitative result. Section 5 also gives a parameter extension derived in this review; that extension has not received a separate external review.

## 1. Exact claim and admissible mechanism

Chua et al., *How Private are DP-SGD Implementations?*, ICML 2024, [arXiv:2403.17673v2](https://arxiv.org/html/2403.17673v2), Conjecture 3.2 proposes the pair

\[
P_S=\frac1T\sum_{t=1}^T N(2e_t,\sigma^2I_T),\qquad
Q_S=\frac1T\sum_{t=1}^T N(e_t,\sigma^2I_T)
\]

as a dominating pair for all adaptive query methods in their shuffled adaptive batch linear query mechanism, under their directed zero-out adjacency.

Fix integers \(T\ge2\) and \(b\ge1\). Let \(n=bT\), \(x=(u,w,\ldots,w)\), and \(x'=(\bot,w,\ldots,w)\). The uniformly shuffled partition puts the distinguished record in each batch with probability \(1/T\). Use scalar queries, with \(\psi(\bot)=0\), and the following two choices:

| Query | \(\psi(u)\) | \(\psi(w)\) |
|---|---:|---:|
| ANTI | 1 | -1 |
| SAME | 1 | 1 |

Both obey the unit-norm query bound. Query values on unused domain elements may be set to zero. Ask ANTI for the first \(m=T-1\) steps. At the last step, ask SAME on a Borel set \(S\) of the earlier, transformed outputs and ANTI otherwise. This decision uses earlier outputs only; it does not use the hidden shuffle location or the current output.

For raw output \(g_t\), set \(z_t=g_t-b\psi_t(w)\). This is a common triangular Borel bijection of transcripts under both datasets: recursively recover \(g_t\) from earlier recovered outputs and \(z_t\). For the last-step policy, the prefix translation is fixed and the last coordinate is translated by a Borel function of that prefix. Fubini's theorem shows that this transformation preserves Lebesgue measure, hence also the hockey-stick divergence. It is not a single fixed translation for an adaptive policy.

In a batch without the distinguished record, \(z_t\) has mean zero. In its batch the means under \((x,x')\) are \((2,1)\) for ANTI and \((0,-1)\) for SAME. Independent Gaussian noise has variance \(\sigma^2\) at each step. ANTI at every step therefore yields exactly \((P_S,Q_S)\) in the transformed coordinates.

For any \(d>1\), place these scalar query values in the first coordinate. The remaining coordinates are independent, identically distributed Gaussian noise under both datasets, so they leave the divergence unchanged.

## 2. The two last-step densities

Write \(\phi_\sigma\) for the scalar \(N(0,\sigma^2)\) density. For a prefix \(y\in\mathbb R^m\), let

\[
\beta(y)=\frac1T\prod_{i=1}^m\phi_\sigma(y_i),\quad
\alpha(y)=\frac1T\sum_{i=1}^m\phi_\sigma(y_i-2)\prod_{j\ne i}\phi_\sigma(y_j),\quad
\gamma(y)=\frac1T\sum_{i=1}^m\phi_\sigma(y_i-1)\prod_{j\ne i}\phi_\sigma(y_j).
\]

These are joint-density weights, not conditional probabilities. In particular \(\beta>0\) everywhere. The final-coordinate joint densities at this prefix are

\[
\begin{array}{ll}
p_A=\alpha\phi_\sigma(s)+\beta\phi_\sigma(s-2),&
q_A=\gamma\phi_\sigma(s)+\beta\phi_\sigma(s-1),\\
p_B=(\alpha+\beta)\phi_\sigma(s),&
q_B=\gamma\phi_\sigma(s)+\beta\phi_\sigma(s+1).
\end{array}
\]

For \(k=e^\varepsilon\), put \(h_A=p_A-kq_A\), \(h_B=p_B-kq_B\), and \(I_A(y)=\int(h_A)_+ds\), \(I_B(y)=\int(h_B)_+ds\). Integrating in the last coordinate first gives the exact switching identity

\[
H_k(M(x)\|M(x'))-H_k(P_S\|Q_S)
=\int_S (I_B(y)-I_A(y))\,dy.
\]

All these integrals are finite. Each inner positive-part integral is bounded by the corresponding prefix marginal of \(p\), which is the same for both policies. The densities are measurable and the inner integrals are continuous in their finite, positive weights by the \(L^1\) Lipschitz property of the positive-part function.

## 3. A compact switch set proves the manuscript's theorem

Set \(\sigma=1\), \(\varepsilon=1/2\), and write \(\phi=\phi_1\). Define

\[
C(y)=\frac{\alpha-\sqrt e\,\gamma}{\beta}
=\sum_{i=1}^m\left(e^{2y_i-2}-e^{y_i}\right).
\]

Take simply \(S=[3,4]^m\). For every \(y\in S\),

\[
C(y)\ge m e^3(e-1)>\frac{e^2}{4}>0.
\]

The bound follows from \(e^y(e^{y-2}-1)\ge e^3(e-1)\) for \(y\ge3\), \(m\ge1\), and \(e>2\). For ANTI, with \(v=e^{s-1}>0\),

\[
\frac{h_A(y,s)}{\beta(y)\phi(s)}
=C(y)+e^{2s-2}-e^s
=C(y)+v^2-ev
\ge C(y)-\frac{e^2}{4}>0.
\]

Thus the ANTI signed density is positive for every final output. In contrast,

\[
\frac{h_B(y,s)}{\beta(y)\phi(s)}=C(y)+1-e^{-s}<0
\quad\text{whenever}\quad s<-\log(C(y)+1).
\]

Both signed densities have the same integral,

\[
\int h_A\,ds=\int h_B\,ds=\beta(y)(C(y)+1-\sqrt e).
\]

The identity \(\int h_+=\int h+\int(-h)_+\) now gives

\[
I_B(y)-I_A(y)=\int(-h_B(y,s))_+\,ds>0.
\]

The strict inequality holds because \(h_B\) is strictly negative on a nonempty interval and \(\beta\phi>0\). Since \(S\) has positive measure, the switching identity proves

\[
H_{\sqrt e}(M(x)\|M(x'))>H_{\sqrt e}(P_S\|Q_S)
\]

for every \(T\ge2\), \(b\ge1\), and \(d\ge1\). This refutes Conjecture 3.2. No numerical quadrature, interval-arithmetic implementation, Monte Carlo estimate, or asymptotic limit is needed for this qualitative conclusion.

## 4. What the proof does and does not settle

It resolves the specific proposed dominating-pair conjecture by a counterexample. It does not identify a replacement tight dominating pair or the exact worst-case privacy accountant, does not prove the paper's specific numerical gains, and does not by itself realize the adaptive queries as gradients of a single prescribed loss in a practical final-model-only DP-SGD implementation. The broader T353 row should stay open if it asks for tight shuffled privacy accounting.

It also shows that at \(T=2\) the region of strict improvement has no finite upper cutoff in \(y\): the same argument works at every \(y\ge3\). The manuscript's reported cutoff near 12 is a finite-grid or finite-precision observation, not a mathematical boundary.

## 5. Reviewer-derived extension: each positive noise level and each finite privacy parameter

The same reasoning actually works for every fixed \(\sigma>0\) and every finite \(\varepsilon\in\mathbb R\), in particular every \(\varepsilon\ge0\). This section is a new derivation in this independent review, not a claim that the pinned manuscript already states or separately audits this stronger theorem.

Let \(a=e^{-1/\sigma^2}\), \(k=e^\varepsilon>0\), \(t_i=e^{(y_i-1/2)/\sigma^2}\), and

\[
C=\frac{\alpha-k\gamma}{\beta}=\sum_{i=1}^m(a t_i^2-k t_i).
\]

Set

\[
L=\frac32+\sigma^2(\varepsilon+\log2),\qquad S=[L,L+1]^m.
\]

Then \(t_i\ge2k/a\). The function \(at^2-kt\) is increasing for \(t\ge2k/a\) and equals \(2k^2/a\) at that lower endpoint. Consequently

\[
C\ge\frac{2mk^2}{a}>\frac{k^2}{4a}>0.
\]

Writing \(v=e^{(s-1)/\sigma^2}\), the ANTI ratio is

\[
\frac{h_A}{\beta\phi_\sigma(s)}
=C+v^2-k e^{1/(2\sigma^2)}v
\ge C-\frac{k^2 e^{1/\sigma^2}}4>0.
\]

The SAME ratio is

\[
\frac{h_B}{\beta\phi_\sigma(s)}
=C+1-k e^{(-s-1/2)/\sigma^2},
\]

which is negative on the half-line

\[
s<\sigma^2\log\!\left(\frac{k}{C+1}\right)-\frac12.
\]

Again the signed integrals agree, both equal to \(\beta(C+1-k)\), and the SAME negative part has positive integral. The same positive-measure argument therefore proves a strict excess at the chosen \((\sigma,\varepsilon)\).

The quantifiers are: for each \((T,b,d,\sigma,\varepsilon)\) in the stated range there exists such a policy. Its switch set may depend on \(\sigma\) and \(\varepsilon\). This gives neither a uniform positive gap across parameters nor a single policy that wins simultaneously for every privacy parameter.
