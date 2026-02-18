# mlips4orca

MLIP (Machine Learning Interatomic Potential) plugins for ORCA `ExtTool` (`ProgExt`) interface.

Three model families are supported:
- **UMA** (FAIR-Chem) — default model: `uma-s-1p1`
- **OrbMol** (orb-models) — default model: `orb_v3_conservative_omol`
- **MACE** — default model: `MACE-OMOL-0`

All backends provide energy, gradient, and analytical Hessian. The model server starts automatically and stays resident, so repeated calls during optimization are fast.

## Quick Start (Default = UMA)

1. Install PyTorch (CUDA 12.9 build).
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

2. Install the package with UMA profile.
```bash
pip install "mlips4orca[uma]"
```

3. Log in to Hugging Face for UMA model access.
```bash
huggingface-cli login
```

4. Use in an ORCA input file.
```text
! ExtOpt Opt

%pal
  nprocs 8
end

%method
  ProgExt "uma"
end

* xyz 0 1
O  0.000000  0.000000  0.000000
H  0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
*
```

Other backends:
```text
%method
  ProgExt "orb"
end
```

> **Note:** Run `uma --list-models` to see available models. If the `uma` alias conflicts in your environment, use `mlips4orca-uma` instead.

Additional examples: `examples/cla_freq.inp` + `examples/cla_external.inp`, `examples/sn2_freq.inp` + `examples/sn2_external.inp`, `examples/h2o_freq.inp` + `examples/h2o_external.inp`

## Using Analytical Hessian (Recommended: two-step workflow)

> **Why two steps?** ORCA's ExtTool protocol only passes energy and gradient back to ORCA — the Hessian is never transmitted through the protocol. The only way to use the exact analytical Hessian from the MLIP is to dump it to a `.hess` file via `--dump-hessian`, then load it with `InHessName`. Without this, ORCA falls back to an approximate model Hessian or expensive numerical differentiation. The analytical Hessian leads to faster and more reliable convergence.

Generate a `.hess` file first, then load it via `InHessName`.

### TS Search

**Step 1: Generate analytical Hessian via `--dump-hessian`**
```text
! ExtOpt Opt

%geom
  MaxIter 1
end

%method
  ProgExt "uma"
  Ext_Params "--dump-hessian cla.hess"
end

* xyz 0 1
...
*
```
A 1-step optimization that triggers the ExtTool call and dumps the analytical Hessian in ORCA `.hess` format. `! ExtOpt` is required to make ORCA use the external tool instead of its own internal methods. The job may exit non-zero (not converged), but the `.hess` file is created.

**Step 2: TS optimization reading Hessian**
```text
! ExtOpt OptTS

%method
  ProgExt "uma"
end

%geom
  InHessName "cla.hess"
end

* xyz 0 1
...
*
```
ORCA reads the initial Hessian from the `.hess` file. The model server keeps the MLIP loaded so repeated calls during optimization are fast.

### Geometry Optimization (with analytical Hessian)

Same two-step workflow with `! ExtOpt Opt` instead of `! ExtOpt OptTS`:
```text
! ExtOpt Opt
%geom
  MaxIter 1
end
%method
  ProgExt "mace"
  Ext_Params "--dump-hessian h2o.hess"
end
```
then:
```text
! ExtOpt Opt
%method
  ProgExt "mace"
end
%geom
  InHessName "h2o.hess"
end
```

## Installing Model Families

```bash
pip install "mlips4orca[uma]"         # UMA (default)
pip install "mlips4orca[orb]"         # OrbMol
pip install "mlips4orca[mace]"        # MACE
pip install "mlips4orca[orb,mace]"    # OrbMol + MACE
pip install mlips4orca                # core only
```

> **Note:** UMA and MACE conflict at dependency level (`e3nn`). Use separate environments.

Local install:
```bash
git clone https://github.com/t-0hmura/mlips4orca.git
cd mlips4orca
pip install ".[uma]"
```

Model download notes:
- **UMA**: Hosted on Hugging Face Hub. Run `huggingface-cli login` once.
- **OrbMol / MACE**: Downloaded automatically on first use.

## Upstream Model Sources

- UMA / FAIR-Chem: https://github.com/facebookresearch/fairchem
- OrbMol / orb-models: https://github.com/orbital-materials/orb-models
- MACE: https://github.com/ACEsuit/mace

## Advanced Options

See `OPTIONS.md` for backend-specific tuning parameters.

Command aliases:
- Short: `uma`, `orb`, `mace`
- Prefixed: `mlips4orca-uma`, `mlips4orca-orb`, `mlips4orca-mace`

## Troubleshooting

- **`ProgExt "uma"` runs the wrong plugin** — Use `ProgExt "mlips4orca-uma"` to avoid alias conflicts.
- **`uma` command not found** — Activate the conda environment where the package is installed.
- **UMA model download fails (401/403)** — Run `huggingface-cli login`. Some models require access approval on Hugging Face.
- **Works interactively but fails in PBS jobs** — Use absolute path from `which uma` in the ORCA input.

## References

- ORCA ExtTool: https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/externaloptimizer.html
- ORCA external tools: https://github.com/faccts/orca-external-tools
