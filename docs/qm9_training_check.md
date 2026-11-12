# QM9 training check — 12 November 2026

No training jobs were submitted. CRIANN was inspected read-only over `ssh criann`.

## Proposed training settings

The CRIANN launcher uses LR 5e-4, batch size 128, 1000 epochs, 15 warmup epochs, EMA 0.999, seed 42, three layers, 16 hidden multivector channels, 128 hidden scalar channels, eight heads, dropout 0.1, Gaussian RBF64, cutoff 5 Å, and sparse message passing. Weight decay resolves to the current Python default, 5e-7; make this explicit in the launcher before submission.

The scheduler starts at 5e-6, reaches 5e-4 after 15 completed epochs, and ends at 5e-7 after 1000 epochs. This schedule is internally consistent. Float32 matmul precision is highest and cuDNN TF32 is disabled. No mixed precision is enabled.

The saved CRIANN `qm9_run_02/U0/pgagnn/config.json` confirms LR 5e-4, batch 128, 1000 epochs, and weight decay 0.01. Thus a new 5e-7 run is not a readout-only ablation against those historical results. A controlled readout comparison would hold the training settings fixed for both variants.

Standard targets: mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv. Energy-like PyG targets and logged physical MAEs use eV; convert to meV when presenting those paper columns.

## Blocking symmetry check

`pytest tests/test_symmetry.py -q`: 29 passed. However, the QM9 end-to-end molecular tests currently exercise only proper rotations and translations. The centered-readout tests exercise reflections on already covariant multivectors, not the complete pipeline starting from reflected Cartesian positions.

Additional CPU float64 checks used the full launcher architecture, evaluation mode, the absolute-pseudoscalar readout, and the noncoplanar four-atom fixture in `tests/test_symmetry.py`. For seed 123 and the atomic-sum head:

| Matrix-weight scale | Proper rotation and translation: max prediction difference | Reflection and translation: max prediction difference |
|---|---:|---:|
| 1 | 3.89e-16 | 2.17e-6 |
| 2 | 9.33e-15 | 1.76e-3 |
| 3 | 7.11e-14 | 1.81e-1 |

Scales 2 and 3 multiply matrix-valued parameters of the same architecture; they are counterexamples to an architecture-wide guarantee, not estimates of trained-model errors. At ordinary initialization, all three head modes (atomic_sum, pooled_mlp, dipole_vector) showed reflection discrepancies over the three tested seeds. The three-atom fixture was insensitive in these checks; a planar example alone would miss the issue.

The primitive join covariance tests pass. The input convention needs particular attention: canonical point re-embedding under an improper transform gives `embed_point(Rp+t) = det(R) rho_V(embed_point(p))`. Recomputing the graph reference from canonical embeddings keeps its e123 coefficient at +1; this differs from transforming a covariant reference with rho_V. The current tests distinguish these actions. Taking an absolute pseudoscalar at the final readout does not by itself establish covariance of all preceding hidden features under canonical reflected inputs.

Reproduction from the repository root:

```
PYTHONPATH=. .venv/bin/python runs/qm9/preflight/check_reflections.py
PYTHONPATH=. .venv/bin/python runs/qm9/preflight/check_readout_modes.py
```

Outputs and audited source hashes are in `runs/qm9/preflight/`. Do not label a new full-pipeline run E(3)-invariant until the physical-coordinate reflection check passes.

## Restart issues in the current implementation

- The signal callback writes weights during an epoch but labels them with `last_completed_epoch`, which is updated only at periodic saves. Resuming can repeat updates with inconsistent scheduler/epoch metadata. Finish an epoch and save its complete state before exiting.
- The checkpoint omits the explicit training-loader generator state; resuming resets its shuffled sample ordering.
- The checkpoint omits best_state and the early-stopping counter. The Slurm script copies back only last_model.pt, not best_model.pth. If no new validation best is reached after a restart, final evaluation uses the wrong weights.
- In-memory best_state is built from state_dict tensors without cloning. Later model/EMA updates can change those tensors; select the saved best checkpoint or keep an independent snapshot.
- sync_back and scontrol requeue failures are downgraded to warnings, followed by a successful script exit. Failed persistence or requeue must be reported as failure rather than successful training.
- Requeues copy the current active source again. Freeze the submitted source so edits cannot silently change an interrupted experiment.

A short interrupted/resumed training comparison should verify model, EMA, optimizer, scheduler, completed epoch, shuffle ordering, and best-validation selection before long runs.

## CRIANN resources and environment

`hpda_mig` is UP with a 24-hour wall-time maximum. Configured slices: ten a100_1g.10gb, seventeen a100_2g.20gb, and four a100_3g.40gb. Availability is dynamic; configured counts do not imply all slices are free. The existing script requests one 10 GB slice, eight CPUs, and 23:50 hours, with USR1 1000 seconds before the limit and requeue enabled. A measured memory/time preflight is needed before choosing concurrency.

The old project is `/home/2026011/ptoura01/ga-gnn`. Its `.venv/bin/python` is missing. The QM9 processed dataset exists at `/home/2026011/ptoura01/datasets/qm9/processed/data_v3.pt` (314 MiB). uv exists at `~/.local/bin/uv`. Home storage has approximately 16 GB free within a 50 GB filesystem quota; choose environment/cache placement before installing the locked environment. No environment was installed and no existing remote source was overwritten during this audit.

## Launch criteria

Resolve full-pipeline reflection invariance; fix and verify checkpoint/requeue behavior; prepare a frozen source and runtime; run one MIG memory/timing/backward preflight; then submit the twelve target jobs with explicit settings and bounded concurrency. Do not change the RBF/cutoff or other model choices as part of diagnosing the symmetry failure.

## Follow-up localization

Tracing the seed-123, scale-2 model through the same physical reflection localizes the first readout-feature discrepancy to the first message layer's full bivector coefficient norm (max difference 0.01789198). Input-projection invariant features agree to 1.8e-15. The discrepancy subsequently reaches scalar features in layer 2 and the final prediction (0.00176172).

In this reproducer, zeroing only the geometric edge input in both evaluations removes the discrepancy: final reflected predictions agree to 1.6e-15. This is a diagnostic intervention, not a proposed architecture change. Supplying covariantly transformed node/edge/reference multivectors instead of canonically re-embedded reflected positions also restores agreement (1.9e-14).

The problematic combination is the addition of projected edge messages to geometric-product messages in `src/pgagnn.py:651`. Canonically re-embedded point and edge multivectors both transform as minus the stipulated rho action under reflections; the recomputed graph-reference e123 coefficient remains +1. The bivector parts generated by multiplying point-derived messages need not have the same parity as the joined geometric edge bivectors, so adding them can make their full coefficient norm reflection sensitive. The join primitive itself passes covariance tests when its inputs and reference are transformed by rho as required by its contract.

Reproduce the layer trace with `PYTHONPATH=. .venv/bin/python runs/qm9/preflight/trace_reflections.py`; recorded output is in `runs/qm9/preflight/reflection_trace.txt`. No model source was modified for these diagnostics.

## Candidate centered-input conversion (prototype only)

A tested candidate for centered molecular tasks is right multiplication of both point features and constructed join edge features by the inverse Euclidean pseudoscalar: `P_c -> P_c I3^{-1}` and `E_c -> E_c I3^{-1}`, where `I3=e123` and `I3^{-1}=-e123`. This is invertible because the Euclidean I3 squares to -1; it does not use the noninvertible PGA pseudoscalar e0123. The conversion is performed after constructing the canonical join edges and before message passing. Both canonical point and edge features have the extra det(R) factor under reflection, while `rho_R(I3^{-1})=det(R) I3^{-1}`, so the converted features transform with rho_R under the remaining O(3) action. Per-graph centering handles physical translations upstream.

The point feature becomes `1-e0 p_c`, equivalently a translation-motor representation `T(2 p_c)` under the current convention, instead of a pure point trivector. Joined edges become mixed grade-1/grade-3 features. The information is retained by an invertible conversion, but the learned architecture sees different grades; this is a method/encoding change and requires documentation and new training. It should not silently replace the original model or be described as only an absolute-pseudoscalar ablation.

Prototype `runs/qm9/preflight/check_centered_conversion.py` leaves production source unchanged. CPU float64 checks on the full three-layer launcher architecture with matrix weights scaled by 2 passed proper/improper rotations plus translations for atomic_sum, pooled_mlp, and dipole_vector heads in dense and sparse paths. Largest prediction discrepancy was 5.2e-15. Training-mode backward gradients were finite in all six configurations. This validates the prototype symmetry behavior on these cases, not accuracy or GPU memory/performance. It is intended for centered molecular inputs; no claim is made that multiplication by a fixed I3 commutes with uncentered PGA translations.

## MD17 and N-body follow-up

The same physical-coordinate reflection check also fails for the current MD17 pipeline. A seed-123 six-layer, hidden-MV64/scalar128 model with the smooth Gaussian c9 configuration (float64, evaluation mode, synthetic noncoplanar fixture) gives normalized energy difference 1.94e-8 and force covariance difference 7.98e-8 at ordinary initialization. Scaling matrix-valued weights by 2 gives energy difference 0.00225667 and force difference 0.4559119. These are architecture counterexamples on synthetic inputs; they are not measured errors on trained molecular checkpoints or benchmark physical-unit MAEs.

The existing N-body tests do reflect Cartesian positions and velocities, then re-embed, and therefore test the correct physical behavior. Their passing result is specific to the tested configuration/weights/tolerances. Reusing their two-layer dense model and fixture gives position discrepancy 4.38e-9 at original weights, which fits their atol=1e-8, rtol=1e-6 tolerance. Multiplying matrix-valued weights by 2 gives discrepancy 0.0172413. A six-layer counterpart gives 4.53e-6 at ordinary initialization and 0.246587 at scale 2. Thus the passing shallow test does not establish an architecture-wide reflection guarantee. Reproduction is in `runs/qm9/preflight/check_md17_nbody_reflections.py`.

No running MD17 training jobs were stopped, modified, or resubmitted during these checks. A failure of the guarantee does not alone measure the symmetry error of the actual trained checkpoints; those need separate evaluation.
