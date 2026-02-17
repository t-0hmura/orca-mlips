# OPTIONS.md (mlips4orca)

This file lists detailed plugin options. For most users, defaults are enough.

Compatibility note:
- UMA and MACE extras currently conflict at dependency level (`e3nn`).
- Use separate environments when switching between UMA and MACE.

## Common Options (all backends)

- `--model <name_or_alias_or_path>`
- `--device auto|cpu|cuda`
- `--hessian-mode Analytical|Numerical`
- `--hessian-step <float>`
- `--strict-hessian`
- `--list-models`
- `--version`
- `--dump-hessian <path>` (ORCA only; writes Hessian in `eV/Angstrom^2`)

## UMA Options (`uma` / `mlips4orca-uma`)

- `--task <omol|omat|odac|oc20|oc25|omc>`
- `--list-tasks`  
  Print available UMA task names and exit.
- `--workers <int>`  
  Predictor worker count.
- `--workers-per-node <int>`  
  Optional worker cap per node.
- `--max-neigh <int>`  
  Override graph neighbor cap. If omitted, model default is used.
- `--radius <float>`  
  Override graph cutoff radius in Angstrom. If omitted, model default is used.
- `--r-edges`  
  Enable distance edge attributes in graph construction.
- `--otf-graph` / `--no-otf-graph`  
  Enable or disable OTF graph collation (`--otf-graph` is default).

## OrbMol Options (`orb` / `mlips4orca-orb`)

- `--precision <str>` (default: `float32-high`)
- `--compile-model`
- `--loader-opt KEY=VALUE` (repeatable)  
  Extra kwargs forwarded to Orb pretrained loader.
- `--calc-opt KEY=VALUE` (repeatable)  
  Extra kwargs forwarded to `ORBCalculator`.

Examples:

```bash
orb --list-models --loader-opt compile=false
orb --list-models --calc-opt stress=true
```

## MACE Options (`mace` / `mlips4orca-mace`)

- `--dtype float32|float64` (default: `float64`)
- `--calc-opt KEY=VALUE` (repeatable)  
  Extra kwargs forwarded to MACE calculator builders (`mace_mp`, `mace_off`, `mace_omol`, `mace_anicc`, or `MACECalculator`).

Example:

```bash
mace --list-models --calc-opt default_dtype=float32
```

## `KEY=VALUE` Parsing Rules

For `--loader-opt` / `--calc-opt`:

- `true` / `false` -> boolean
- `none` / `null` -> `None`
- integer/float strings -> numeric type
- otherwise -> string

Example:

```bash
--calc-opt foo=true --calc-opt n=64 --calc-opt alpha=0.25 --calc-opt mode=fast
```
