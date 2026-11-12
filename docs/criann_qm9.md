# QM9 on CRIANN MIG

The deployed project is `/home/2026011/ptoura01/pgagnn` on CRIANN. The historical `/home/2026011/ptoura01/ga-gnn` checkout was preserved. The launcher is `experiments/qm9/train_qm9.sl`; submit from the project root using the helper below.

The `.venv` is a small system-site-packages overlay on `aidl/pytorch/2.6.0-py3.12-cuda12.6`: Python 3.12.9, PyTorch 2.6.0+cu126, NumPy 2.2.4, SciPy 1.15.2, TensorBoard 2.19.0, PyG 2.7.0, and matching PyTorch 2.6 / CUDA 12.6 scatter/cluster wheels. The launcher reads `.venv/criann-module` and reloads the module on every start/requeue. It limits BLAS/OpenMP threads and disables user-site packages to avoid unrelated user installations.

The deployed code and overlay consume approximately 101 MB. PyTorch and CUDA are imported from `/soft/AIDL/conda_envs/pytorch2602`, not duplicated in home. The setup uses `experiments/qm9/requirements-criann.txt` and adds only eight packages. Use `setup_criann.sh` for this overlay; `uv sync` would select the separate Python 3.10 / PyTorch 2.8 locked stack and should not be run inside this module-based environment. The runtime versions differ from `uv.lock` and are recorded in the preflight reports.

To recreate the overlay, run `bash experiments/qm9/setup_criann.sh` in a short allocated compute job; login-node resource limits can interrupt installation. The script uses no package cache and does not install PyTorch or CUDA. The dataset is reused at `$HOME/datasets/qm9`, overridable with `QM9_DATA_DIR`. The Python executable can be overridden with `QM9_PYTHON`.

Submit all 12 standard targets, with at most four running concurrently:

```bash
bash experiments/qm9/submit_criann.sh qm9_e3 all 4
```

Submit one target:

```bash
bash experiments/qm9/submit_criann.sh qm9_e3 U0
```

The job requests `hpda_mig`, one `a100_1g.10gb` slice, eight CPUs, 16 GiB RAM, and 23:50 hours. Training uses batch 128, 1000 epochs, LR 5e-4, weight decay 5e-7, 15 warmup epochs, EMA 0.999, seed 42, and the existing three-layer sparse PGA-GNN configuration. `QM9_BATCH_SIZE`, `QM9_EPOCHS`, and `QM9_NUM_WORKERS` can override the corresponding values. GPU type, wall time, and resources can be overridden with standard sbatch options when invoking the Slurm script directly.

## Persistence and requeue

The first task freezes source into `runs/qm9/<run>/source` under a lock. All targets and restarts use this snapshot. Source and dataset are staged to `/dlocal/run/<job>/<task>/`; checkpoints, best weights, metrics, and TensorBoard outputs are written directly into persistent `runs/qm9/<run>/<target>/pgagnn`. No final data transfer is required to preserve a completed checkpoint.

At the pre-time-limit USR1 signal (1000 seconds before the limit), Python requests a stop, finishes training and validation for the current epoch, and atomically saves a consistent checkpoint. It exits with code 3. The batch script checks that the checkpoint exists and calls `scontrol requeue` for that specific job or array element. On restart, saved epoch N resumes at N+1. Model, optimizer, cosine/warmup scheduler, EMA, early-stopping counter, best validation weights, training sampler RNG, and Python/NumPy/Torch RNG states are restored. The sampler has a separate RNG from worker seeding so starting fresh workers does not change the batch permutation. QM9 preprocessing is deterministic; this is not a scheme for restoring arbitrary random augmentation state inside persistent workers.

A checkpoint is saved after every complete epoch and on a graceful stop. If a hard kill prevents finishing the epoch, the last already persisted epoch remains available; partial-epoch updates are not relabeled as complete. The time-limit signal must leave enough time for the current epoch and validation to finish; measure this in the first MIG preflight. Requeue permission or scheduler failures propagate as failures. Logs are appended across restarts.

The best model is an independent snapshot and is included in the restart checkpoint, so a requeued job does not forget a validation best achieved before interruption. Test evaluation is deferred on an interrupted run and performed after training completes.

Use a fresh run name for a different encoding, source version, or training recipe. Reusing a run name deliberately restores its frozen source and checkpoints. In particular, historical point-trivector checkpoints must not be resumed under the corrected geometry encoding.

## Verification

`tests/test_qm9_resume.py` compares uninterrupted training with a real USR1 interruption during the second epoch and a fresh model/loader resume. It checks epoch numbering, shuffled batch order, optimizer/scheduler/EMA state, random state, and best-model selection. `tests/test_criann_qm9.py` runs the batch script with Slurm stand-ins to check target mapping, persistent resume paths, frozen source, and failure propagation. These tests do not submit cluster jobs or replace a GPU memory/timing preflight.

## Deployed compatibility checks

On 12 November 2026, short MIG allocations 3354067, 3354070, and 3354071 installed the overlay and verified compatibility. No long training jobs were submitted. The initial batch check and the stress check on the 128 largest QM9 molecules both passed for atomic_sum (U0), pooled_mlp (alpha), and dipole_vector (mu) heads after three optimizer steps, with finite backward gradients and reflection agreement below 1e-6. The largest-molecule batch peaked at 3.58 GiB allocated on a 9.5 GiB A100 MIG device. Steady training-step times for that stress batch were approximately 0.88–0.92 seconds. These are short compatibility checks, not benchmark accuracy results or a measured full-epoch training time.

The module environment passed 26 focused symmetry, launcher, and interrupted-resume tests, including persistent-worker batch-order continuation. Reports are stored on CRIANN under `runs/qm9/criann_preflight/{report,largest_report}.json`, with copies in the local project. They record runtime versions, source hashes, memory use, and timing. To repeat the real-data check in a GPU allocation after loading the module:

```bash
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python experiments/qm9/check_criann.py --largest \
  --data_dir "$HOME/datasets/qm9" --output runs/qm9/criann_preflight/largest_report.json
```
