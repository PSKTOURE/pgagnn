# Cartesian geometry encoding

The PGA-GNN experiment pipelines use per-graph centered positions and a Euclidean-dual encoding before message passing. The scalar and geometric message layers are unchanged. Baseline GATr, EGNN, SEGNN, and GGNN paths retain their existing encodings.

For a centered point trivector P and a canonical joined edge E, the features passed to PGA-GNN are P I3^{-1} and E I3^{-1}, with I3=e123 and I3^{-1}=-e123. The join is constructed before this conversion. The inverse exists in the Euclidean subalgebra; this does not invert the null PGA pseudoscalar e0123.

Under an orthogonal coordinate transform R, canonical point and join features transform as X(Rp)=det(R) rho_R(X(p)). Since rho_R(I3^{-1})=det(R) I3^{-1}, the converted feature transforms as rho_R(X I3^{-1}), as required by the message-passing layers. Centering supplies translation invariance; no claim is made that a fixed Euclidean dual commutes with uncentered PGA translations.

QM9 and MD17 already center coordinates. Their wrappers convert point and geometric edge features at the PGA-GNN model boundary. The QM9 readout keeps full coefficient grade norms and the absolute pseudoscalar coefficient. MD17 keeps its scalar atomic-energy readout and computes forces by differentiating the energy.

N-body centers each system before building node and edge features. The point component is dualized and has the form 1-e0 p. Mass occupies the scalar slot and velocity is encoded as v1 e1 + v2 e2 + v3 e3. The combined node input is (m+1) + v - e0 p, so mass, centered position, and velocity remain separately recoverable in a single multivector channel. All three parts have the required O(3) transformation behavior. The earlier corrected N-body encoding added a translation-motor velocity to the dualized point, collapsing the vector information to p+v/2; passing symmetry tests did not establish that this encoding preserved the separate inputs. The output decoder reads the ideal bivector slots as a vector, normalizes by the scalar slot, and adds the input centroid to obtain an absolute position. The denominator clamp acts on an O(3)-invariant scalar, so it preserves the orthogonal transformation behavior even near zero.

QM9 and MD17 configurations record geometry_encoding=centered_euclidean_dual. N-body configurations now record geometry_encoding=centered_euclidean_dual_vector_velocity; its previous centered_euclidean_dual runs used the mixed position/velocity coefficients. This changes the interpretation of learned coefficients despite retaining parameter shapes. Historical checkpoints and unfinished runs must use their original source and encoding to reproduce or resume the historical experiment. Do not resume an old checkpoint in the new pipeline solely because its state_dict loads. The pre-change source and manuscript are saved in Git commit 3f234d7. Existing remote training snapshots were not modified.

Validation exercises the actual Cartesian pipelines with proper/improper rotations, translations, dense and packed graphs, noncoplanar point sets, deeper networks, amplified weights, and finite backward gradients. MD17 checks both energy invariance and force covariance, including force-loss backpropagation. Test helpers call the production N-body forward path rather than maintaining a copied embedding implementation.

Validation completed on 12 November 2026: 114 tests passed on CPU. The original six-layer MD17 counterexample now has energy discrepancy below 9e-15 and force discrepancy below 3.1e-13 in float64. QM9 and N-body reflection counterexamples likewise agree at float64 numerical precision. Local Quadro T2000 GPU float32 checks passed for all three QM9 head modes, Gaussian/Bessel MD17 energy and force-loss backward paths, and N-body position equivariance. These symmetry and gradient checks do not establish training accuracy for the new encoding.

The N-body velocity correction retains the scalar/ideal-bivector output decoder: it predicts a future position relative to the input system centroid, then restores that centroid. It does not add a displacement to each input particle position. Input/output channel counts and layer parameter shapes are unchanged. Regression checks cover separate recovery of all seven node inputs, the former position/velocity collision, full-rank input Jacobians, and preservation of the baseline embedding.

## N-body input channel comparison

A local single-seed pilot on 13 November 2026 retained the one-channel vector-velocity encoding. Each variant used 100 training systems (subsample 0.001), all 5,000 validation systems, seed 42, dropout seed 1042, batch size 64, and 2,000 optimizer steps. The two-layer model used 64 multivector and 128 scalar hidden channels, eight heads, Gaussian distance features, Adam with learning rate 3e-4 and weight decay 1e-6, 80 warmup steps, and cosine decay. Matching-shape initial weights and batch order were shared. Validation was evaluated every 200 steps. Test and OOD sets were not evaluated.

| Input encoding | Parameters | Best validation MSE | Step |
|---|---:|---:|---:|
| One channel, vector-grade velocity | 606,373 | 0.00462093 | 1,200 |
| Separate mass/point/velocity channels, motor velocity | 606,775 | 0.00474853 | 800 |
| Separate mass/point/velocity channels, vector-grade velocity | 606,775 | 0.00493432 | 400 |

Separate vector channels learned faster initially but overfit earlier; their best validation MSE was 6.8% higher than the single-channel variant. This limited pilot supports retaining the current encoding, without establishing the ranking under full training or across seeds. The multi-channel experiment options were removed from production. Local pilot outputs and source snapshots remain under runs/nbody/input_encoding_pilot and runs/nbody/separate_vector_pilot.
