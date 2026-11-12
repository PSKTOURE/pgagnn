# Author Response and Rebuttal (TMLR Submission)

We thank both reviewers for their insightful, constructive, and thorough reviews of our manuscript. In response to your feedback, we have carefully revised the manuscript (`docs/main_TMLR.tex` and compiled `docs/main_TMLR.pdf`), corrected the mathematical formulations and proofs, resolved all identified typos and diagram labels, and substantially refined the framing and empirical discussions.

Below is an overview of the key revisions, followed by detailed point-by-point responses to Reviewer L4x7 and Reviewer BYPr.

---

## Summary of Major Revisions to the Manuscript

1. **Rigorous Proof of Join Equivariance and Convention Consistency (Appendix A.2, Lemma 4 & Section 3.1, Eq. 8):**
   We completely replaced the proof of Lemma 4 and corrected Equation (8). We eliminated the invalid step asserting that the linear Poincaré complement operator commutes with translations. Following the algebraic decomposition of the PGA join established by Dorst (2020) and Brehmer et al. (2023), we:
   - **Reconciled Dual Conventions and Signs:** We explicitly derived the decomposition specialized to point trivectors under our canonical right-complement dual $\mathbf{e}_A \wedge J(\mathbf{e}_A) = I$ without operand swapping, clarifying its relation to Dorst's contraction dual and showing that the origin $X = \mathbf{e}_{123}$ and unit point $Y = \mathbf{e}_{123} - \mathbf{e}_{023}$ evaluate to $+\mathbf{e}_{23}$.
   - **Corrected Translation Identity and Explicit Cancellation:** We applied the corrected contraction-join identity $\mathbf{a} \vee_{\mathrm{Euc}} (\mathbf{v} \rfloor \mathbf{b}) = -(-1)^n \widehat{(\mathbf{v} \rfloor \mathbf{a})} \vee_{\mathrm{Euc}} \mathbf{b}$ (placing the grade involution correctly on $\mathbf{v} \rfloor \mathbf{a}$) and showed the explicit cancellation of the translation cross-terms to zero.
   - **Corrected Normalized-Point Join and Blade Coordinates:** In Equation (8) and Equation (54), we replaced improper duals on the moment with the correct bivector formula $\mathbf{X}_i \vee \mathbf{X}_j = (\mathbf{p}_j - \mathbf{p}_i)\mathbf{e}_{123} + \mathbf{e}_0 (\mathbf{p}_i \times \mathbf{p}_j)$, correcting the swapped blade assignments between Euclidean direction and ideal moment.
   - **Clarified Reference Scalar Transformation:** In both the Lemma 4 statement and derivation, we clarified that the projection coefficient $\langle \rho_V(\mathbf{Z}_{\text{ref}}) \rangle_{123}$ transforms as $\det(R)\langle \mathbf{Z}_{\text{ref}} \rangle_{123}$, canceling the parity sign under reflections.

2. **Symmetry-Specific Molecular Readouts (Section 4.5, Eq. 24):**
   We distinguish two centered molecular readouts: an $SE(3)$-invariant variant retaining the signed pseudoscalar coefficient and an $E(3)$-invariant variant taking its absolute value. Both use full coefficient grade norms, which are invariant under rotations and reflections in the centered molecular representation; centering supplies translation invariance. We distinguish these norms from the degenerate metric seminorms used for uncentered multivector representations.

3. **Symmetry Guarantees and Reported Benchmarks:**
   The reported benchmark results are unchanged from the original submission. The QM9 experiments in Table 2 use the $SE(3)$-invariant readout. The $E(3)$-invariant variant has been checked numerically for its symmetry properties but has not yet been retrained across all 12 targets; the reported errors therefore characterize the $SE(3)$-invariant variant.

4. **Positioning and Technical Comparison with GAFL and GATr (Sections 1, 2, and 3.2):**
   We added Wagner et al. (NeurIPS 2024; GAFL) to our references and expanded Sections 1, 2 (Related Work), and 3.2. We explicitly attribute the bilinear factorization of multivector aggregation to prior art (Clifford Frame Attention in GAFL), and provide an equation-level comparison delineating PGA-GNN's formulation: global projective algebra $\mathcal{G}_{3,0,1}$ on general geometric graphs (without requiring local residue frames), coupled with explicit geometric edge multivectors ($\mathbf{E}_{ij}$ via the join) and physical scalar edge attributes ($\mathbf{e}_{ij}$).

5. **Nuanced and Accurate Framing of Empirical Results (Abstract, Sections 1, 5.2, and 5.3):**
   We removed all overstatements ("across the board", "near-perfect", "fundamentally more efficient", "diluting sparse topological priors"). We clearly characterize the task-dependent nature of the results: standout gains on spring systems and in low-data regimes; lower average errors and improved force predictions across all ten molecules on rMD17 compared to 2-body baselines (EGNN, SEGNN), while acknowledging that specialized higher-order cluster expansion models (MACE, NequIP) achieve lower errors; and top performance on three properties on QM9 ($\mu, \alpha, C_v$), while EGNN attains the better aggregate GMRE score. We describe these as scalar molecular targets and limit the discussion to the observed performance differences.

6. **Direct Acknowledgment of Ablation Limitations (Section 6):**
   We explicitly state in the manuscript and response that component ablations are conducted solely on gravitational N-body systems. We acknowledge that they do not isolate the specific drivers of the large gains in spring dynamics or the performance on QM9, narrowing our conclusions accordingly.

7. **Grounded Systems and Hardware Discussion (Section 5.4 and Appendix E):**
   We qualified the Pareto efficiency claims strictly to the evaluated axes (FLOPs vs. error and parameters vs. error). In Appendix E and Section 5.4, we discuss measured inference latency and GPU memory as empirical trade-offs, and present architectural factors (dense matrix operations in set transformers vs. edge materialization and scatter/gather in GNNs) as plausible hypotheses rather than asserting unverified low-level kernel dispatches as facts.

8. **Typos and Diagram Labels Corrected:**
   - Section 5.2: "euclidean distances" $\to$ "Euclidean distances".
   - Figure 1 (`docs/images/article_draft_imgs/diagram.pdf`): "Classic operations" $\to$ "Classical operations".

---

## Detailed Response to Reviewer L4x7

> **Reviewer Summary:** *"This paper proposes PGA-GNN, an E(3)-equivariant graph neural network based on Projective Geometric Algebra... The method demonstrates strong empirical performance... The presentation is quite clear... Claims supported: Yes. Audience interest: Yes."*

We thank Reviewer L4x7 for their positive assessment of our theoretical formulation, empirical evaluation, presentation, and relevance to the TMLR audience.

### 1. Contribution Relative to Existing PGA Architectures
> **Weakness 1:** *"This is not the first work to introduce PGA into geometric data learning. Extending PGA from transformers to GNNs seems somewhat incremental."*

**Response:**
We fully agree that introducing PGA into geometric deep learning is not our contribution, and that GATr pioneered the use of PGA multivectors in deep learning. We have thoroughly revised the Introduction (Section 1) and Related Work (Section 2) to sharpen our specific contributions:
- **Architectural Formulation:** Rather than using set self-attention over node multivectors alone, PGA-GNN formulates edge-conditioned message passing on geometric graphs. It introduces explicit geometric edge multivectors ($\mathbf{E}_{ij}$ via the equivariant join) and scalar physical attributes ($\mathbf{e}_{ij}$, such as interaction stiffness or RBF-expanded distances) directly into both the geometric message stream and a coupled scalar channel.
- **Bilinear Factorization in Graph Message Passing:** We leverage the bilinear distributivity of the geometric product to factorize message aggregation, evaluating expensive multivector products once per node rather than once per edge. We also explicitly credit Wagner et al. (NeurIPS 2024; GAFL), who utilized this algebraic property in Clifford Frame Attention for protein backbones, while contrasting our general-graph, frame-free projective formulation.
- **Empirical Focus:** We establish that this edge-conditioned formulation provides concrete advantages in low-data regimes and physical interaction systems (such as the spring benchmark), where explicit graph connectivity and interaction attributes provide a decisive inductive prior.

We have tempered all contribution statements throughout the manuscript to reflect these distinctions accurately.

---

### 2. Computational Efficiency: FLOPs vs. Latency and Memory
> **Weakness 2:** *"Despite the reduction in FLOPs, PGA-GNN has higher inference latency and GPU memory usage than GATr (Table 6). This somewhat weakens the practical computational efficiency claims."*

**Response:**
We agree with the reviewer that lower theoretical arithmetic complexity (FLOPs) does not automatically translate to lower wall-clock inference latency or smaller GPU memory footprints on modern hardware. 

In the revised manuscript (Section 5.4 and Appendix E, Table 6), we have explicitly addressed this distinction:
- **Empirical Trade-off:** As measured on an NVIDIA A100 GPU for a batch of $B=64$ systems of $N=15$ bodies, PGA-GNN achieves an out-of-distribution MSE of $(2.02 \pm 0.21) \times 10^{-6}$ with $6.795\text{ GFLOPs}$, compared to $(2.33 \pm 0.08) \times 10^{-3}$ and $25.033\text{ GFLOPs}$ for GATr. In wall-clock terms, GATr executes faster ($62.03\text{ ms}$ vs. $86.27\text{ ms}$) and with less memory ($50.0\text{ MB}$ vs. $347.1\text{ MB}$).
- **Plausible Systems Factors:** We hypothesize that this difference is driven by two factors:
  1. GATr operates over node multivector sets ($C=16$) without edge attributes, allowing it to execute via regular dense batched matrix multiplications.
  2. PGA-GNN operates with a larger multivector channel capacity ($C=64$) and explicitly materializes geometric edge multivectors ($\mathbf{E}_{ij}$) and 64-dimensional RBF distance expansions across all directed edges ($B \times N^2$), which increases peak memory and introduces scatter-gather indexing operations that may incur kernel launch overhead on small graphs ($N=15$).
- **Qualified Pareto Claims:** In Section 5.4, we have restricted our Pareto efficiency claims strictly to the hardware-agnostic metrics evaluated: arithmetic complexity (FLOPs vs. OOD Error) and parameter footprint (Parameters vs. OOD Error). We explicitly note that on small, dense systems, dense set attention can execute faster in wall-clock time despite performing $3.7\times$ more arithmetic operations.

---

### 3. Requested Typos
> **Requested Change 1:** *Section 5.2: "euclidean distances" should be "Euclidean distances"*  
> **Requested Change 2:** *Figure 1: "Classic operations" should be "Classical operations"*

**Response:**
Both typos have been corrected in the revised manuscript:
- Section 5.2 now reads: *"encoding pairwise Euclidean distances via a Gaussian Radial Basis Function (RBF)..."*
- Figure 1 (`docs/images/article_draft_imgs/diagram.pdf`): The text in the vector diagram has been updated from "Classic operations" to "Classical operations".

---

## Detailed Response to Reviewer BYPr

> **Reviewer Summary:** *"The paper proposes PGA-GNN, a graph neural network that represents geometric features as multivectors in projective geometric algebra... The model provides a coherent formulation... mathematical development... strong practical benefits in several evaluated settings... However, architectural novelty is somewhat unclear... empirical advantages are task-dependent... explanations of gains provide limited evidence... inaccuracies in specific invariance statements and supporting proofs."*

We thank Reviewer BYPr for their constructive, rigorous, and technically precise critique. The reviewer correctly identified mathematical inaccuracies in Equation (24) and Lemma 4, and rightly challenged the breadth of empirical claims and the attribution of gains. We have addressed every weakness directly in both theory and text.

---

### 1. Architectural Novelty and Positioning Relative to GATr and GAFL
> **Weakness 1:** *"The architectural novelty is somewhat unclear. Several central ingredients already appear in existing geometric-algebra models. GATr provides equivariant PGA operations, while Clifford Frame Attention in GAFL [1] constructs pairwise PGA messages and exploits bilinearity to move product computation after aggregation. The contribution therefore appears to lie mainly in the particular formulation of equivariant message passing with explicit edge information. While this combination may be useful, its conceptual advance over these architectures is somewhat incremental."*

**Response:**
We agree with the reviewer's assessment. Pairwise geometric-algebra interactions, equivariant PGA primitives, and bilinear aggregation factorization have precedents in the literature, notably in GATr (Brehmer et al., 2023) and GAFL (Wagner et al., NeurIPS 2024). We apologize for not citing GAFL in the initial submission and for phrasing the bilinear factorization as entirely unique to our work.

In the revised manuscript, we have made the following updates:
1. **Direct Citation and Attribution:** We have added Wagner et al. (NeurIPS 2024) to our bibliography and explicitly cited GAFL in Sections 1, 2, and 3.2.
2. **Detailed Architectural Comparison:** In Section 2 (Related Work) and Section 3.2, we provide an equation-level comparison distinguishing PGA-GNN from GATr and GAFL:
   - **GAFL / Clifford Frame Attention (CFA):** GAFL is designed for protein backbone generation and operates on local residue reference frames (canonicalized coordinate frames constructed from backbone atoms). Multivectors are defined relative to these local frames, and interactions are evaluated by transforming features between frames. In contrast, PGA-GNN operates on general geometric graphs (not restricted to macromolecules with backbone frames) in a unified global projective space $\mathcal{G}_{3,0,1}$, avoiding frame construction or prediction steps.
   - **GATr:** GATr operates as a set transformer over bare node multivectors via self-attention and does not support explicit geometric or scalar edge features.
   - **PGA-GNN's Distinctive Formulation:** PGA-GNN formulates edge-conditioned message passing on geometric graphs: directed edges carry explicit geometric multivectors $\mathbf{E}_{ij}$ (constructed via the equivariant join) and scalar physical attributes $\mathbf{e}_{ij}$ (such as spring constants, interaction types, or radial basis expansions). These attributes modulate node messages alongside a parallel scalar message track.
   - **Bilinear Factorization:** In Section 3.2, we explicitly acknowledge that factoring the multivector product outside the neighborhood sum via bilinearity was also leveraged by Wagner et al. (2024). We clarify that in PGA-GNN, this property is integrated into an edge-conditioned graph message-passing architecture that combines node-level multivector bilinears with explicit edge multivectors ($\mathbf{E}_{ij}$) and scalar edge features. We also clarify the total layer complexity: while the geometric product evaluations scale as $\mathcal{O}(N \cdot C)$, edge scalar encoding, attention weights, and neighborhood scatter-reductions still scale with $|\mathcal{E}|$.

---

### 2. Scope and Task-Dependency of Empirical Claims
> **Weakness 2:** *"The empirical advantages are task-dependent, providing limited support for the breadth of the paper’s claims. The strongest improvements occur in the synthetic spring system, while the gravitational N-body gains are more modest. On MD17, PGA-GNN substantially improves over the reported EGNN and SEGNN baselines but remains behind MACE and NequIP on both energy and force prediction. On QM9, it achieves the best results on three of twelve properties, while EGNN obtains the better aggregate score. These findings demonstrate benefits in particular settings, but do not establish a consistent advantage across the physical and molecular tasks considered."*

**Response:**
We fully agree with the reviewer. The empirical evidence demonstrates substantial benefits in specific settings—most prominently on interacting spring dynamics and in low-data regimes—rather than a uniform, universal superiority across all tasks.

We have thoroughly revised the Abstract, Introduction, Section 5.2, Section 5.3, and Conclusion to provide a balanced and accurate account:
- **Spring Systems:** We highlight that the largest and most consistent advantages occur on the synthetic spring benchmark, especially under data scarcity and out-of-distribution system size generalization.
- **Gravitational N-Body:** We state that gains on gravitational N-body dynamics are modest compared to baselines.
- **Revised MD17:** We removed phrases like "across the board". We report that PGA-GNN lowers average energy and force prediction errors compared to our reported 2-body baselines (EGNN and SEGNN), and improves force prediction across all ten molecules relative to EGNN. At the same time, we explicitly state that PGA-GNN does not match specialized machine learning potentials like MACE and NequIP, which incorporate higher-order many-body cluster expansions. We also clarified that literature PaiNN results were averaged over a 7-molecule subset (excluding Azobenzene, Benzene, and Paracetamol), making direct 10-molecule mean comparisons non-equivalent.
- **QM9:** We explicitly report that EGNN achieves the lowest overall GMRE (1.119 vs. PGA-GNN's 1.253), driven by its lower errors on thermodynamic energies ($U_0, U, H, G$) and spatial extent ($R^2$). We report straightforwardly that among the compared 2-body architectures, PGA-GNN obtains the lowest error on three properties: dipole moment ($\mu = 0.016\text{ D}$), isotropic polarizability ($\alpha = 0.059\text{ bohr}^3$), and heat capacity ($C_v = 0.029\text{ cal/mol K}$). We describe these as scalar molecular targets without attributing the performance differences to mechanisms not isolated by the available ablations.
- **Data Splits:** We reconciled the text and captions on rMD17: the dataset uses the standard 1,000-sample split consisting of 950 training samples and 50 validation samples.

---

### 3. Explanations of Performance Gains and Baseline Comparison
> **Weakness 3:** *"The experiments provide limited evidence for the explanations of the performance gains. In particular, the molecular results are partly attributed to the advantages of local message passing over global attention, although the GATr baseline receives the same graph adjacency mask. This comparison therefore does not show locality as the source of the improvement. Furthermore, the component ablations are limited to gravitational N-body prediction, leaving their relevance to the strong spring results and mixed molecular results uncertain. Consequently, the evidence connecting the distinctive architectural choices to the observed benefits is less convincing than the performance discussion suggests."*

**Response:**
The reviewer makes an excellent point. Because the GATr baseline in our molecular experiments was evaluated with the identical sparse graph adjacency mask ($\mathbf{A}_{ij} > 0$), attributing the performance difference to "local message passing vs. global attention" was inaccurate.

We have revised Section 5.2, Section 5.3, Section 6, and the Conclusion as follows:
- **Reframed GATr Comparison:** We removed the claim that this comparison isolates locality. We now explain that because both models receive the same connectivity mask, the performance gap reflects other architectural differences: namely, GATr self-attention does not ingest explicit geometric edge multivectors ($\mathbf{E}_{ij}$) or scalar physical edge attributes (such as RBF distance expansions), whereas PGA-GNN explicitly conditions both geometric messages and scalar channels on edge attributes.
- **Force Optimization Stability as an Empirical Hypothesis:** We have rephrased our discussion of force optimization through deep attention maps. Rather than asserting it as an established causal fact, we explicitly frame it as an empirical hypothesis and open question for future study.
- **Direct Acknowledgment of Ablation Scope:** We explicitly state in Section 6 that our component ablations are conducted solely on gravitational N-body systems. We state directly that they do not isolate which specific architectural components drive the strong spring dynamics gains or the mixed molecular results, and narrow our conclusions accordingly.

---

### 4. Mathematical Inaccuracies in Invariance Statements and Proofs
> **Weakness 4:** *"There are inaccuracies in specific invariance statements and supporting proofs. Equation (24) includes the raw grade-four component among the claimed invariant features, although this pseudoscalar changes sign under reflections. Consequently, passing these features through an unrestricted MLP does not guarantee an E(3)-invariant QM9 readout. In addition, the duality-intertwining identity used in the proof of Lemma 4 in Appendix A.2 does not hold under translations, leaving an invalid intermediate step in the derivation. The join-equivariance conclusion itself should still hold, but the presented proof does not establish it."*

**Response:**
We are deeply grateful to Reviewer BYPr for pointing out these two mathematical errors. Both issues have been thoroughly resolved in the manuscript and codebase.

#### A. Correction to Invariant Readout (Section 4.5, Equation 24)
The reviewer is entirely correct: in Plane-based Projective Geometric Algebra $\mathcal{G}_{3,0,1}$, the pseudoscalar basis blade is $\mathbf{e}_{0123}$. Under an improper Euclidean transformation (a spatial reflection $R \in O(3)$ with $\det(R) = -1$), the pseudoscalar transforms as:
$$\langle \rho_R(\mathbf{X}) \rangle_4 = \det(R) \langle \mathbf{X} \rangle_4 = -\langle \mathbf{X} \rangle_4.$$
Because an unrestricted MLP is generally asymmetric ($f(-x) \neq f(x)$), passing the raw signed scalar $\langle \mathbf{X} \rangle_4$ into an MLP does not guarantee an $E(3)$-invariant output.

**Readout Definitions and Symmetry Guarantees:**
Section 4.5 (Equation 24) now distinguishes the two centered molecular readouts explicitly:
$$\mathrm{inv}_{SE(3)}(\mathbf{X})=\Big(\langle\mathbf{X}\rangle_0,\,s,\,\|\mathbf{X}_1\|_2,\,\|\mathbf{X}_2\|_2,\,\|\mathbf{X}_3\|_2\Big),$$
$$\mathrm{inv}_{E(3)}(\mathbf{X})=\Big(\langle\mathbf{X}\rangle_0,\,|s|,\,\|\mathbf{X}_1\|_2,\,\|\mathbf{X}_2\|_2,\,\|\mathbf{X}_3\|_2\Big),\qquad s=\langle\mathbf{X}\rangle_4.$$
Molecular centering supplies translation invariance, and the full coefficient grade norms are invariant under the remaining rotations and reflections. The signed pseudoscalar is invariant under proper rotations but changes sign under reflections. Consequently, the signed readout guarantees $SE(3)$ invariance, while the absolute-value variant additionally enforces reflection invariance and guarantees $E(3)$ invariance. The signed variant can learn reflection-invariant predictions but does not enforce them. We have removed the claim of a complete set of invariants.

**QM9 Experimental Configuration:**
The QM9 results in Table 2 were obtained using the $SE(3)$-invariant readout with the signed pseudoscalar coefficient. Our original submission incorrectly claimed reflection invariance for this readout. The manuscript now states its $SE(3)$ guarantee and identifies the readout used in the experimental setup and table caption. The $E(3)$-invariant variant has been implemented and checked numerically for its symmetry properties; it has not yet been retrained across all 12 targets, and the reported errors are not attributed to this variant.


#### B. Correction to Lemma 4 Proof (Appendix A.2)
The reviewer is completely right that the linear Poincaré complement operator $J$ does *not* commute or intertwine with translations:
$$J(\rho_V(\mathbf{A})) \neq \det(R) \rho_V(J(\mathbf{A})) \quad \text{when } \mathbf{t} \neq \mathbf{0}.$$
Specifically, a spatial translation acts as a shear on plane coordinates ($L$), but as the inverse-transpose ($L^{-\top}$) on the dual complement space, leaving an invalid intermediate step in our original derivation.

**Rigorous Replacement Proof for Point Trivectors:**
We completely replaced the proof of Lemma 4 in Appendix A.2 with an established algebraic decomposition from Dorst (2020, Section 3.2) and Brehmer et al. (2023, Appendix A.2, Proposition 7), specialized explicitly to point trivectors:
1. **Trivector Decomposition and Dual Conventions:** For any two point trivectors $\mathbf{X}, \mathbf{Y} \in \langle \mathcal{G}_{3,0,1} \rangle_3$, we decompose them uniquely into Euclidean and ideal parts: $\mathbf{X} = \mathbf{t}_X + \mathbf{e}_0 \mathbf{p}_X$ and $\mathbf{Y} = \mathbf{t}_Y + \mathbf{e}_0 \mathbf{p}_Y$, where $\mathbf{t}_X, \mathbf{t}_Y \in \operatorname{span}(\mathbf{e}_{123})$ and $\mathbf{p}_X, \mathbf{p}_Y \in \langle \mathcal{G}_{3,0,0} \rangle_2$. Under our right-complement dual convention ($\mathbf{e}_A \wedge J(\mathbf{e}_A) = I$) without operand swapping, the join expands in terms of the Euclidean join $\vee_{\mathrm{Euc}}$ as:
   $$(\mathbf{t}_X + \mathbf{e}_0 \mathbf{p}_X) \vee (\mathbf{t}_Y + \mathbf{e}_0 \mathbf{p}_Y) = -\Big[ \mathbf{t}_X \vee_{\mathrm{Euc}} \mathbf{p}_Y + (-1)^n \hat{\mathbf{p}}_X \vee_{\mathrm{Euc}} \mathbf{t}_Y + \mathbf{e}_0 (\mathbf{p}_X \vee_{\mathrm{Euc}} \mathbf{p}_Y) \Big].$$
   The overall minus sign accounts for our canonical operand order without swapping (as noted in Dorst, 2020, Eq. 36), and evaluating the origin $X = \mathbf{e}_{123}$ and unit point $Y = \mathbf{e}_{123} - \mathbf{e}_{023}$ confirms that $-(\mathbf{e}_{123} \vee_{\mathrm{Euc}} (-\mathbf{e}_{23})) = +\mathbf{e}_{23} = J^{-1}(J(X) \wedge J(Y))$.
2. **Translational Cancellation:** Substituting translated trivectors $\tau[\mathbf{X}] = \mathbf{t}_X + \mathbf{e}_0 (\mathbf{p}_X - \mathbf{v} \rfloor \mathbf{t}_X)$ into the decomposition, the Euclidean cross-terms produce:
   $$\mathbf{t}_X \vee_{\mathrm{Euc}} (\mathbf{v} \rfloor \mathbf{t}_Y) + (-1)^n \widehat{(\mathbf{v} \rfloor \mathbf{t}_X)} \vee_{\mathrm{Euc}} \mathbf{t}_Y.$$
   Applying the contraction-join identity $\mathbf{a} \vee_{\mathrm{Euc}} (\mathbf{v} \rfloor \mathbf{b}) = -(-1)^n \widehat{(\mathbf{v} \rfloor \mathbf{a})} \vee_{\mathrm{Euc}} \mathbf{b}$ (Brehmer et al., 2023, Lemma 6, Eq. 9), these cross-terms cancel identically to zero. The ideal cross-terms combine via $\mathbf{v} \rfloor (\mathbf{a} \vee_{\mathrm{Euc}} \mathbf{b}) = (\mathbf{v} \rfloor \mathbf{a}) \vee_{\mathrm{Euc}} \mathbf{b}$, establishing that $\tau[\mathbf{X}] \vee \tau[\mathbf{Y}] = \tau[\mathbf{X} \vee \mathbf{Y}]$.
3. **Normalized Point Formula (Main Text Eq. 8 and Appendix Eq. 54):** For two normalized points $\mathbf{X}_i, \mathbf{X}_j$, their join evaluates directly to:
   $$\mathbf{X}_i \vee \mathbf{X}_j = (\mathbf{p}_j - \mathbf{p}_i) \mathbf{e}_{123} + \mathbf{e}_0 (\mathbf{p}_i \times \mathbf{p}_j) \in \langle \mathcal{G}_{3,0,1} \rangle_2.$$
   Both terms are grade-2 bivectors: $(\mathbf{p}_j - \mathbf{p}_i)\mathbf{e}_{123}$ is the Euclidean direction bivector, while $\mathbf{e}_0 (\mathbf{p}_i \times \mathbf{p}_j)$ is the ideal moment bivector. We corrected Equation (8) in the main text, which previously swapped these assignments, and removed improper duals on the moment that produced trivectors.
4. **Reflection Parity Cancellation:** Under spatial reflections ($R \in O(3), \det(R) = -1$), the bare Euclidean join flips sign by $(-1)^n = -1$, yielding $\rho_V(\mathbf{X}_i) \vee \rho_V(\mathbf{X}_j) = \det(R) \rho_V(\mathbf{X}_i \vee \mathbf{X}_j)$. For the reference-modulated join $\mathrm{EquiJoin}(\mathbf{X}_i, \mathbf{X}_j; \mathbf{Z}_{\text{ref}}) = z_{\text{ref}}(\mathbf{X}_i \vee \mathbf{X}_j)$, the orientation coefficient transforms as $\langle \rho_V(\mathbf{Z}_{\text{ref}}) \rangle_{123} = \det(R) \langle \mathbf{Z}_{\text{ref}} \rangle_{123}$. This yields an overall factor of $\det(R)^2 = (-1)^2 = +1$, canceling the parity sign and establishing strict $E(3)$ equivariance across the entire Pin group.

We have verified this complete algebraic formulation across all 256 basis blade pairs under representative translations and reflections in our test suite.

---

## Conclusion

We believe these substantial revisions resolve the theoretical inaccuracies, appropriately contextualize our empirical findings, clarify our contributions relative to prior geometric algebra models (GATr and GAFL), and significantly improve the precision and rigor of the paper. We look forward to any further feedback from the reviewers and the Action Editor.
