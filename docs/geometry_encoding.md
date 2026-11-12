# Cartesian geometry encoding

The PGA-GNN experiment pipelines use per-graph centered positions and a Euclidean-dual encoding before message passing. The scalar and geometric message layers are unchanged. Baseline GATr, EGNN, SEGNN, and GGNN paths retain their existing encodings.

For a centered point trivector P and a canonical joined edge E, the features passed to PGA-GNN are P I3^{-1} and E I3^{-1}, with I3=e123 and I3^{-1}=-e123. The join is constructed before this conversion. The inverse exists in the Euclidean subalgebra; this does not invert the null PGA pseudoscalar e0123.

Under an orthogonal coordinate transform R, canonical point and join features transform as X(Rp)=det(R) rho_R(X(p)). Since rho_R(I3^{-1})=det(R) I3^{-1}, the converted feature transforms as rho_R(X I3^{-1}), as required by the message-passing layers. Centering supplies translation invariance; no claim is made that a fixed Euclidean dual commutes with uncentered PGA translations.

QM9 and MD17 already center coordinates. Their wrappers convert point and geometric edge features at the PGA-GNN model boundary. The QM9 readout keeps full coefficient grade norms and the absolute pseudoscalar coefficient. MD17 keeps its scalar atomic-energy readout and computes forces by differentiating the energy.

N-body centers each system before building node and edge features. Only its point component is dualized: the mass scalar and translation-motor velocity embedding already have the required O(3) transformation behavior. A centered point has the form 1-e0 p. The output decoder reads the ideal bivector slots as a vector, normalizes by the scalar slot, and adds the input centroid to obtain an absolute position. The denominator clamp acts on an O(3)-invariant scalar, so it preserves the orthogonal transformation behavior even near zero.

New run configurations record geometry_encoding=centered_euclidean_dual. This changes the interpretation of learned coefficients despite retaining parameter shapes. Historical checkpoints and unfinished runs must use their original source and encoding to reproduce or resume the historical experiment. Do not resume an old checkpoint in the new pipeline solely because its state_dict loads. The pre-change source and manuscript are saved in Git commit 3f234d7. Existing remote training snapshots were not modified.

Validation exercises the actual Cartesian pipelines with proper/improper rotations, translations, dense and packed graphs, noncoplanar point sets, deeper networks, amplified weights, and finite backward gradients. MD17 checks both energy invariance and force covariance, including force-loss backpropagation. Test helpers call the production N-body forward path rather than maintaining a copied embedding implementation.

Validation completed on 12 November 2026: 114 tests passed on CPU. The original six-layer MD17 counterexample now has energy discrepancy below 9e-15 and force discrepancy below 3.1e-13 in float64. QM9 and N-body reflection counterexamples likewise agree at float64 numerical precision. Local Quadro T2000 GPU float32 checks passed for all three QM9 head modes, Gaussian/Bessel MD17 energy and force-loss backward paths, and N-body position equivariance. These symmetry and gradient checks do not establish training accuracy for the new encoding.
