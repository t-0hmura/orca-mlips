# mlips4orca

Path-independent MLIP plugins for ORCA `ProgExt` / ExtTool.

Included plugins:
- `plugins/uma_orca.py` (FAIR-Chem / UMA)
- `plugins/orbmol_orca.py` (orb-models / OrbMol)
- `plugins/mace_orca.py` (MACE)

Default models:
- UMA: `uma-s-1p1`
- OrbMol: `orb_v3_conservative_omol`
- MACE: `MACE-OMOL-0` (alias of `omol:extra_large`)

## Quick Start

1. Clone and enter this repository.
```bash
git clone https://github.com/t-0hmura/mlips4orca.git
cd mlips4orca
```

2. (Optional) Create a clean environment.
```bash
python3 -m venv .venv
. .venv/bin/activate
```

3. Install base requirements.
```bash
pip install -r requirements.txt
```

4. Install only the backend you need.
```bash
# UMA
pip install fairchem-core

# OrbMol
pip install orb-models

# MACE
pip install mace-torch
```

5. Verify model listing.
```bash
python3 plugins/uma_orca.py --list-models
python3 plugins/orbmol_orca.py --list-models
python3 plugins/mace_orca.py --list-models
```

## ORCA Input Example

Replace `/path/to/mlips4orca` with your local clone path.

```text
! SP

%pal
  nprocs 8
end

%method
  ProgExt "/path/to/mlips4orca/plugins/uma_orca.py"
  Ext_Params "--model uma-s-1p1 --task omol --device auto"
end

* xyz 0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
*
```

Switch backend by replacing `ProgExt` script:
- `.../plugins/orbmol_orca.py`
- `.../plugins/mace_orca.py`

## Model Selection

UMA:
```bash
python3 plugins/uma_orca.py --list-models
python3 plugins/uma_orca.py --list-tasks
```

OrbMol:
```bash
python3 plugins/orbmol_orca.py --list-models
```
Both dashed and underscored names are accepted, for example:
- `orb-v3-conservative-omol`
- `orb_v3_conservative_omol`

MACE:
```bash
python3 plugins/mace_orca.py --list-models
```
Accepted forms include:
- `MACE-OMOL-0`
- `mp:<alias>` or `<alias>` (for MP aliases)
- `off:<alias>` / `off-small|off-medium|off-large`
- `omol:extra_large`
- `anicc`
- local model path or model URL

## Hessian Mode

ORCA ExtTool normally expects `engrad` (energy + gradient).
This plugin can still compute Hessians internally:
- `--hessian-mode Analytical`
- `--hessian-mode Numerical`

Use `--dump-hessian <path>` to write Hessian (`eV/Angstrom^2`) for inspection.

## Notes

- ORCA ExtTool interface reference:
  - https://github.com/faccts/orca-external-tools
  - https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/externaloptimizer.html
- Upstream model sources:
  - https://github.com/facebookresearch/fairchem
  - https://github.com/orbital-materials/orb-models
  - https://github.com/ACEsuit/mace

## Cluster Smoke Test

Edit and submit:
```bash
qsub run.sh
```

`run.sh` is intentionally generic and only loads modules if your environment provides the module system.
