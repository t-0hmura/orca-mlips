#!/bin/sh
#PBS -N mlips4orca_smoke
#PBS -q default
#PBS -l nodes=1:ppn=32:gpus=1,mem=120GB,walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null

set -eu

test "${PBS_O_WORKDIR:-}" && cd "$PBS_O_WORKDIR"

. /home/apps/Modules/init/profile.sh
module load gaussian16.C02
module load orca/6.1.1

# Optional: activate your environment that has torch/ase and backend deps
# source /home/tohmura/miniconda3/etc/profile.d/conda.sh
# conda activate pdb2reaction

echo "[INFO] Python: $(command -v python3 || true)"

# Syntax/import smoke tests (do not execute heavy model inference)
python3 -m py_compile plugins/*.py

# Model alias listing (lightweight)
python3 plugins/uma_orca.py --list-models | head -n 20
python3 plugins/orbmol_orca.py --list-models | head -n 20
python3 plugins/mace_orca.py --list-models | head -n 20

# Uncomment for full ORCA external execution after dependencies are installed.
# Example ORCA snippet:
# %method
#   ProgExt "/data2/tohmura/pdb2reaction_workspace/mlips4orca/plugins/uma_orca.py"
#   Ext_Params "--model uma-s-1p1 --task omol --device auto"
# end
