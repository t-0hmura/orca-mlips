#!/bin/sh
#PBS -N mlips4orca_smoke
#PBS -q default
#PBS -l nodes=1:ppn=4,mem=16GB,walltime=02:00:00
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
  backend_run "$pref_cmd" "$short_cmd" "$py_script" --version || true
  backend_run "$pref_cmd" "$short_cmd" "$py_script" --list-models | head -n 20
}

resolve_backend_runner() {
  pref_cmd="$1"
  short_cmd="$2"
  py_script="$3"

  BACKEND_RUNNER_TYPE=""
  BACKEND_RUNNER=""

  if command -v "$pref_cmd" >/dev/null 2>&1; then
    BACKEND_RUNNER_TYPE="cmd"
    BACKEND_RUNNER="$pref_cmd"
    echo "[INFO] using ${pref_cmd}"
    return 0
  fi

  if [ -f "$py_script" ]; then
    BACKEND_RUNNER_TYPE="py"
    BACKEND_RUNNER="$py_script"
    echo "[INFO] prefixed command not found; using python script ${py_script}"
    return 0
  fi

  if command -v "$short_cmd" >/dev/null 2>&1; then
    BACKEND_RUNNER_TYPE="cmd"
    BACKEND_RUNNER="$short_cmd"
    echo "[WARN] prefixed command missing; using short alias ${short_cmd} (may collide across packages)"
    return 0
  fi

  echo "[ERROR] no usable command found for ${pref_cmd}/${short_cmd}"
  return 1
}

backend_run() {
  pref_cmd="$1"
  short_cmd="$2"
  py_script="$3"
  shift 3

  resolve_backend_runner "$pref_cmd" "$short_cmd" "$py_script" || return 1
  if [ "$BACKEND_RUNNER_TYPE" = "py" ]; then
    python3 "$BACKEND_RUNNER" "$@"
  else
    "$BACKEND_RUNNER" "$@"
  fi
}

run_real_mlip_test() {
  backend="${MLIPS_BACKEND:-uma}"
  device="${MLIPS_DEVICE:-cuda}"
  test_dir="${PBS_O_WORKDIR:-$PWD}/tmp_real_mlip_${backend}_${PBS_JOBID:-local}"
  mkdir -p "$test_dir"

  cat >"${test_dir}/water.xyz" <<'EOF'
3
H2O real MLIP test
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
EOF

  cat >"${test_dir}/water_EXT.extinp.tmp" <<'EOF'
water.xyz
0
1
1
1
EOF

  python3 - <<'PY'
import os
import torch

device = os.environ.get("MLIPS_DEVICE", "cuda").strip().lower()
print("[INFO] torch={} cuda_build={} cuda_available={}".format(
    torch.__version__, torch.version.cuda, torch.cuda.is_available()
))
if device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA device requested but torch.cuda.is_available() is False")
if torch.cuda.is_available():
    print("[INFO] cuda_device_count={}".format(torch.cuda.device_count()))
    print("[INFO] cuda_device_name={}".format(torch.cuda.get_device_name(0)))
PY

  case "$backend" in
    uma)
      model="${MLIPS_MODEL:-uma-s-1p1}"
      task="${MLIPS_TASK:-omol}"
      backend_run "mlips4orca-uma" "uma" "plugins/uma_orca.py" \
        "${test_dir}/water_EXT.extinp.tmp" \
        --model "$model" --task "$task" --device "$device"
      ;;
    orb)
      model="${MLIPS_MODEL:-orb_v3_conservative_omol}"
      backend_run "mlips4orca-orb" "orb" "plugins/orbmol_orca.py" \
        "${test_dir}/water_EXT.extinp.tmp" \
        --model "$model" --device "$device"
      ;;
    mace)
      model="${MLIPS_MODEL:-MACE-OMOL-0}"
      backend_run "mlips4orca-mace" "mace" "plugins/mace_orca.py" \
        "${test_dir}/water_EXT.extinp.tmp" \
        --model "$model" --device "$device"
      ;;
    *)
      echo "[ERROR] unsupported MLIPS_BACKEND=${backend} (use uma|orb|mace)"
      return 2
      ;;
  esac

  if [ ! -s "${test_dir}/water.engrad" ]; then
    echo "[ERROR] expected output not found: ${test_dir}/water.engrad"
    return 3
  fi

  echo "[INFO] generated ${test_dir}/water.engrad"
  sed -n '1,18p' "${test_dir}/water.engrad"
  echo "[INFO] real MLIP test completed for backend=${backend}"
}

if [ "${MLIPS_REAL_MLIP_TEST:-0}" = "1" ]; then
  run_real_mlip_test
  exit $?
fi

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
