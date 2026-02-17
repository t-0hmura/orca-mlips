#!/bin/sh
#PBS -N mlips4orca_smoke
#PBS -q default
#PBS -l nodes=1:ppn=32:gpus=1,mem=120GB,walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null

set -eu

test "${PBS_O_WORKDIR:-}" && cd "$PBS_O_WORKDIR"

# Keep a job-local log even when PBS stdout/stderr are disabled.
LOG_PATH="${PBS_O_WORKDIR:-$PWD}/run.${PBS_JOBID:-local}.log"
exec >"$LOG_PATH" 2>&1
echo "[INFO] log=${LOG_PATH}"

# Optional module initialization (path-independent).
if [ -n "${MODULESHOME:-}" ] && [ -f "${MODULESHOME}/init/profile.sh" ]; then
  . "${MODULESHOME}/init/profile.sh"
fi

if command -v module >/dev/null 2>&1; then
  module load gaussian16.C02 || true
  module load orca/6.1.1 || true
fi

# Optional conda activation:
# - Set MLIPS_CONDA_ENV to enable.
# - Optionally override CONDA_SH (default: $HOME/miniconda3/etc/profile.d/conda.sh).
CONDA_SH_PATH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "${CONDA_SH_PATH}" ]; then
  . "${CONDA_SH_PATH}"
fi
if [ -n "${MLIPS_CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  conda activate "${MLIPS_CONDA_ENV}"
fi

echo "[INFO] Python: $(command -v python3 || true)"

# Syntax/import smoke tests (do not execute heavy model inference)
python3 -m py_compile plugins/*.py

run_list_models() {
  pref_cmd="$1"
  short_cmd="$2"
  py_script="$3"

  if command -v "$pref_cmd" >/dev/null 2>&1; then
    echo "[INFO] using ${pref_cmd}"
    "$pref_cmd" --version || true
    "$pref_cmd" --list-models | head -n 20
    return 0
  fi

  if command -v "$short_cmd" >/dev/null 2>&1; then
    echo "[INFO] using ${short_cmd}"
    "$short_cmd" --version || true
    "$short_cmd" --list-models | head -n 20
    return 0
  fi

  echo "[INFO] using python script ${py_script}"
  python3 "$py_script" --version || true
  python3 "$py_script" --list-models | head -n 20
}

run_list_models "mlips4orca-uma" "uma" "plugins/uma_orca.py"
run_list_models "mlips4orca-orb" "orb" "plugins/orbmol_orca.py"
run_list_models "mlips4orca-mace" "mace" "plugins/mace_orca.py"
echo "[INFO] smoke test completed"

# Uncomment for full ORCA external execution after dependencies are installed.
# Example ORCA snippet:
# %method
#   ProgExt "uma"
#   Ext_Params ""
# end
