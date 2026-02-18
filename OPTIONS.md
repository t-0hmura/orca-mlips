# OPTIONS.md (mlips4orca)

For most users, defaults are sufficient.

> **Note:** UMA and MACE currently conflict at dependency level (`e3nn`). Use separate environments.

## Common Options (all backends)

- `--model <name_or_alias_or_path>`
- `--device auto|cpu|cuda`
- `--dump-hessian <path>` — Dump analytical Hessian in ORCA `.hess` format (usable with `inhess Read`).
- `--list-models`
- `--version`

## Server Options

The model server starts automatically on first use and stops after idle timeout. These options are for advanced use only.

- `--no-server` — Disable auto server; load model directly each time.
- `--server-socket <path>` — Manual socket path.
- `--stop-server` — Send shutdown to a running server.
- `--server-idle-timeout <int>` — Idle timeout in seconds (default: 600).

## UMA Options (`uma` / `mlips4orca-uma`)

- `--task <omol|omat|odac|oc20|oc25|omc>`
- `--list-tasks`
- `--workers <int>` — Predictor worker count.
- `--workers-per-node <int>` — Worker cap per node.
- `--max-neigh <int>` — Override graph neighbor cap.
- `--radius <float>` — Override graph cutoff radius (Angstrom).
- `--r-edges` — Enable distance edge attributes.
- `--otf-graph` / `--no-otf-graph` — Toggle OTF graph collation (default: on).

## OrbMol Options (`orb` / `mlips4orca-orb`)

Only conservative Orb models are supported.

- `--precision <str>` (default: `float32-high`)
- `--compile-model`
- `--loader-opt KEY=VALUE` (repeatable) — Extra kwargs for Orb loader.
- `--calc-opt KEY=VALUE` (repeatable) — Extra kwargs for `ORBCalculator`.

## MACE Options (`mace` / `mlips4orca-mace`)

- `--dtype float32|float64` (default: `float64`)
- `--calc-opt KEY=VALUE` (repeatable) — Extra kwargs for MACE calculator.

## Using Analytical Hessian with ORCA

ORCA's ExtTool protocol only returns energy and gradient. To use MLIP analytical Hessians as initial Hessian for TS searches, pass `--dump-hessian` via `Ext_Params` and load the file with `inhess Read`:

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

## `KEY=VALUE` Parsing Rules

For `--loader-opt` / `--calc-opt`:

- `true` / `false` -> boolean
- `none` / `null` -> `None`
- integer/float strings -> numeric type
- otherwise -> string
