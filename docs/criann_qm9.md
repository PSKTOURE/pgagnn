# QM9 on CRIANN MIG

The launcher is `experiments/qm9/train_qm9.sl`; submit from the project root using the helper below. The project and Python environment must first be available on CRIANN. It uses `$PROJECT_DIR/.venv/bin/python` by default, or the executable specified by `QM9_PYTHON`. The dataset defaults to `$HOME/datasets/qm9`, overridable with `QM9_DATA_DIR`.

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
