#!/bin/bash
#SBATCH --job-name=htc_gpu_probe
#SBATCH --output=htc_gpu_probe-%j.out
#SBATCH --error=htc_gpu_probe-%j.err
#SBATCH --partition=htc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:03:00

# Submit once per node to inspect both driver/OS environments before installing.
# No packages are installed and no training is launched.
set -euo pipefail

srun bash <<'PROBE'
set -euo pipefail
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "Home: $HOME"
echo "Working directory: $PWD"
echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
uname -srmo
getconf GNU_LIBC_VERSION
test ! -f /etc/os-release || cat /etc/os-release
df -h "$HOME"
echo "Home readable: $(test -r "$HOME" && echo yes || echo no)"
echo "Home writable: $(test -w "$HOME" && echo yes || echo no)"
python3 --version
if [[ -x "$HOME/.local/bin/uv" ]]; then
    "$HOME/.local/bin/uv" --version
fi
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
PROBE
