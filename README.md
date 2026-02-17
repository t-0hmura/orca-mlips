# mlips4orca

MLIP plugins for ORCA ExtTool with three model families:
- UMA (FAIR-Chem)
- OrbMol (orb-models)
- MACE

Default models:
- UMA: `uma-s-1p1`
- OrbMol: `orb_v3_conservative_omol`
- MACE: `MACE-OMOL-0`

## Quick Start (Default = UMA)

1. Install PyTorch (CUDA 12.9 build).
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

2. Install package with UMA profile.
```bash
pip install "mlips4orca[uma]"
```
This install creates the commands `uma`, `orb`, `mace` (and prefixed aliases).

3. Log in once to Hugging Face for UMA model access.
```bash
huggingface-cli login
```

4. Confirm commands and model list.
```bash
uma --list-models
uma --list-tasks
```
If `uma` alias conflicts in your environment, use `mlips4orca-uma`.

5. Confirm plugin version.
```bash
uma --version
```

6. Use in ORCA (`ProgExt`).
```text
! SP

%pal
  nprocs 8
end

%method
  ProgExt "uma"
  Ext_Params ""
end

* xyz 0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
*
```

Optional explicit options:
```text
%method
  ProgExt "uma"
  Ext_Params "--model uma-s-1p1 --task omol --hessian-mode Analytical"
end
```

Other backends with defaults:
```text
%method
  ProgExt "orb"
end
```
```text
%method
  ProgExt "mace"
end
```

Additional example inputs:
- `examples/cla_external.inp`
- `examples/sn2_external.inp`
- `examples/h2o_EXT.xyz`

## Install Model Families

PyPI install:
```bash
# Default profile (UMA)
pip install "mlips4orca[uma]"

# Add OrbMol
pip install "mlips4orca[orb]"

# Add MACE
pip install "mlips4orca[mace]"

# Add both OrbMol + MACE
pip install "mlips4orca[orb,mace]"

# Core package only (no backend dependencies)
pip install mlips4orca
```

Local source install:
```bash
git clone https://github.com/t-0hmura/mlips4orca.git
cd mlips4orca
pip install ".[uma]"
pip install ".[orb]"     # optional
pip install ".[mace]"    # optional
pip install .            # core only
```

Family-specific commands:
```bash
uma --list-models
orb --list-models
mace --list-models
```

Family notes:
- UMA: models are served from Hugging Face Hub. Run `huggingface-cli login` once.
- OrbMol: models are provided by `orb-models` and downloaded automatically on first use.
- MACE: models are provided by `mace-torch` and downloaded automatically on first use.

## Upstream Model Sources

- UMA / FAIR-Chem: https://github.com/facebookresearch/fairchem
- OrbMol / orb-models: https://github.com/orbital-materials/orb-models
- MACE: https://github.com/ACEsuit/mace

## Advanced Usage

### Backend Commands
- Short aliases: `uma`, `orb`, `mace`
- Prefixed aliases: `mlips4orca-uma`, `mlips4orca-orb`, `mlips4orca-mace`

Detailed and low-impact tuning options are documented in `OPTIONS.md`.

## Troubleshooting

- `ProgExt "uma"` runs the wrong plugin:
  Use prefixed aliases to avoid collisions, for example `ProgExt "mlips4orca-uma"`.
- `uma` command is not found after install:
  Activate the same environment where you installed the package, then reinstall with `python -m pip install "mlips4orca[uma]"`.
- UMA model download fails with 401/403:
  Run `huggingface-cli login`. Some UMA model repos are gated and require manual access approval on Hugging Face.
- Works interactively but fails in scheduler jobs:
  Job shells may have reduced `PATH`. Use an absolute command path in ORCA from `which uma`.

## Notes

- ORCA ExtTool references:
  - https://github.com/faccts/orca-external-tools
  - https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/externaloptimizer.html
- UMA and MACE profiles currently conflict at dependency level (`e3nn`); use separate environments.
- `run.sh` contains a PBS smoke test template (`qsub run.sh`).
