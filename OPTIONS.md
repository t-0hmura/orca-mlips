# orca-mlips Options

For most users, defaults are sufficient.

> **Note:** UMA and MACE currently have a dependency conflict (`e3nn`). Use separate environments.

## Common Options (all backends)

- `--model <name_or_alias_or_path>`
- `--device auto|cpu|cuda`
- `--solvent <name|none>` — xTB implicit-solvent correction (`none` disables correction).
- `--solvent-model <alpb|cpcmx>` — implicit solvent model (default: `alpb`).
  - `alpb` -> xTB `--alpb`
  - `cpcmx` -> xTB `--cpcmx`
- `--xtb-cmd <path_or_cmd>` — xTB executable for solvent correction (default: `xtb`).
- `--xtb-acc <float>` — xTB `--acc` value for solvent correction (default: `0.2`).
- `--xtb-workdir <tmp|path>` — xTB per-call scratch base directory (default: `tmp`).
- `--xtb-keep-files` — Keep xTB temporary files for debugging.
- `--dump-hessian <path>` — Dump analytical Hessian in ORCA `.hess` format (load with `InHessName`).
- `--list-models`
- `--version`

When solvent correction is enabled (`--solvent != none`), xTB must be available in the current environment/path.

## UMA Options (`uma` / `orca-mlips-uma`)

Available models (default: **`uma-s-1p1`**):

| Model | Description |
|-------|-------------|
| `uma-s-1p1` | Small model, fastest while still SOTA on most benchmarks (6.6M/150M active/total params) |
| `uma-m-1p1` | Best across all metrics, slower and more memory intensive (50M/1.4B active/total params) |

Run `uma --list-models` to see the full list including `esen-*` variants. Models are hosted on Hugging Face Hub (`huggingface-cli login` required).

- `--task <omol|omat|odac|oc20|oc25|omc>`
- `--list-tasks`
- `--workers <int>` — Predictor worker count.
- `--workers-per-node <int>` — Worker cap per node.
- `--max-neigh <int>` — Override graph neighbor cap.
- `--radius <float>` — Override graph cutoff radius (Angstrom).
- `--r-edges` — Enable distance edge attributes.
- `--otf-graph` / `--no-otf-graph` — Toggle on-the-fly (OTF) graph construction (default: on).

## ORB Options (`orb` / `orca-mlips-orb`)

Only conservative ORB models are supported. Underscores and dashes are interchangeable (e.g., `orb_v3_conservative_omol` = `orb-v3-conservative-omol`).

Available models (default: **`orb_v3_conservative_omol`**):

| Model | Dataset |
|-------|---------|
| `orb-v3-conservative-omol` | OMol25 (molecules) |
| `orb-v3-conservative-20-omat` | OMAT (materials, max 20 neighbors) |
| `orb-v3-conservative-inf-omat` | OMAT (materials, unlimited neighbors) |
| `orb-v3-conservative-20-mpa` | MPA (materials, max 20 neighbors) |
| `orb-v3-conservative-inf-mpa` | MPA (materials, unlimited neighbors) |

Run `orb --list-models` to see the full list. Models are downloaded automatically on first use.

- `--precision <str>` (default: `float32-high`)
- `--compile-model`
- `--loader-opt KEY=VALUE` (repeatable) — Extra kwargs for ORB loader.
- `--calc-opt KEY=VALUE` (repeatable) — Extra kwargs for `ORBCalculator`.

## MACE Options (`mace` / `orca-mlips-mace`)

Available models (default: **`MACE-OMOL-0`**):

| Model | Description |
|-------|-------------|
| `MACE-OMOL-0` | OMOL large model for molecules and transition metals |
| `mp:small`, `mp:medium`, `mp:large` | MACE-MP-0 (Materials Project, 89 elements) |
| `mp:medium-0b3` | MACE-MP-0b3, improved high-pressure stability |
| `mp:medium-mpa-0` | MACE-MPA-0, MPTrj + sAlex |
| `mp:small-omat-0`, `mp:medium-omat-0` | MACE-OMAT-0 |
| `mp:mace-matpes-pbe-0` | MACE-MATPES PBE functional |
| `mp:mace-matpes-r2scan-0` | MACE-MATPES r2SCAN functional |
| `mp:mh-0`, `mp:mh-1` | MACE-MH cross-domain (surfaces/bulk/molecules) |
| `off:small`, `off:medium`, `off:large` | MACE-OFF23 for organic molecules |
| `anicc` | ANI-CC model |

The `mp:` prefix selects Materials Project models, `off:` selects organic force field models. A local file path or URL can also be specified. Run `mace --list-models` to see the full list. Models are downloaded automatically on first use.

- `--dtype float32|float64` (default: `float64`)
- `--calc-opt KEY=VALUE` (repeatable) — Extra kwargs for MACE calculator.

## AIMNet2 Options (`aimnet2` / `orca-mlips-aimnet2`)

Available models (default: **`aimnet2`**):

| Model | Description |
|-------|-------------|
| `aimnet2` | AIMNet2 base model |
| `aimnet2_b973c` | AIMNet2 with B97-3c functional |
| `aimnet2_2025` | AIMNet2 B97-3c + improved intermolecular interactions |
| `aimnet2nse` | AIMNet2 open-shell model |
| `aimnet2pd` | AIMNet2 for Pd-containing systems |
| `<local_model_path>` | Local checkpoint file path |
| `<https://...model>` | Model URL |

- `--calc-opt KEY=VALUE` (repeatable) — Extra kwargs for AIMNet2 calculator.

## Server Options

The model server starts automatically on first use and stops after idle timeout. These options are for advanced use only.

- `--no-server` — Disable auto server; load model directly each time.
- `--server-socket <path>` — Manual socket path.
- `--stop-server` — Send shutdown to a running server.
- `--server-idle-timeout <int>` — Idle timeout in seconds (default: 600).

Auto-started servers are scoped per parent ORCA process and stop automatically when that parent exits.

## `KEY=VALUE` Parsing Rules

For `--loader-opt` / `--calc-opt`:

- `true` / `false` -> boolean
- `none` / `null` -> `None`
- integer/float strings -> numeric type
- otherwise -> string
