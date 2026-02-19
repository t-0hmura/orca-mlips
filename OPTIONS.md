# orca-mlips Options

For most users, defaults are sufficient.

> **Note:** UMA and MACE currently conflict at dependency level (`e3nn`). Use separate environments.

## Common Options (all backends)

- `--model <name_or_alias_or_path>`
- `--device auto|cpu|cuda`
- `--dump-hessian <path>` — Dump analytical Hessian in ORCA `.hess` format (load with `InHessName`).
- `--list-models`
- `--version`

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
- `--otf-graph` / `--no-otf-graph` — Toggle OTF graph collation (default: on).

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

The `mp:` prefix selects Materials Project models, `off:` selects organic force field models. A local file path or URL can also be passed. Run `mace --list-models` to see the full list. Models are downloaded automatically on first use.

- `--dtype float32|float64` (default: `float64`)
- `--calc-opt KEY=VALUE` (repeatable) — Extra kwargs for MACE calculator.

## Using Analytical Hessian with ORCA

ORCA's ExtTool protocol only returns energy and gradient. To use MLIP analytical Hessians as initial Hessian for TS searches, pass `--dump-hessian` via `Ext_Params` and load the file with `InHessName`:

**Step 1:** Generate `.hess` file with a 1-step optimization:
```text
! ExtOpt Opt

%geom
  MaxIter 1
end

%method
  ProgExt "uma"
  Ext_Params "--dump-hessian mlip.hess"
end

* xyz 0 1
...
*
```

**Step 2:** Load the Hessian for TS optimization:
```text
! ExtOpt OptTS

%method
  ProgExt "uma"
end

%geom
  InHessName "mlip.hess"
end

* xyz 0 1
...
*
```

`! ExtOpt` is required to make ORCA use the external tool. See `README.md` for the full two-step workflow rationale.

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
