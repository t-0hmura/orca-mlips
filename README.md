# orca-mlips

MLIP (Machine Learning Interatomic Potential) plugins for ORCA `ExtTool` (`ProgExt`) interface.

Three model families are supported:
- **UMA** (FAIR-Chem) — default model: `uma-s-1p1`
- **OrbMol** (orb-models) — default model: `orb_v3_conservative_omol`
- **MACE** — default model: `MACE-OMOL-0`

All backends provide energy and gradient, and can output analytical Hessian to ORCA `.hess` files via `--dump-hessian`.
The model server starts automatically and stays resident, so repeated calls during optimization are fast.

## Quick Start (Default = UMA)

1. Install PyTorch (CUDA 12.9 build).
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

2. Install the package with UMA profile.
```bash
pip install "orca-mlips[uma]"
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

> **Note:** Run `uma --list-models` to see available models. If the `uma` alias conflicts in your environment, use `orca-mlips-uma` instead.

Additional examples: `examples/cla_hess.inp` + `examples/cla_external.inp`, `examples/sn2_hess.inp` + `examples/sn2_external.inp`, `examples/h2o_hess.inp` + `examples/h2o_external.inp`

## Using Analytical Hessian (Required two-step workflow in ORCA)

> **Why two steps?** ORCA has no API to receive Hessian data directly through `ExtTool`. The only supported path is:
> 1) dump Hessian with `--dump-hessian <file>` in step 1,  
> 2) read it in step 2 with `InHessName <file>`.

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
pip install "orca-mlips[uma]"         # UMA (default)
pip install "orca-mlips[orb]"         # OrbMol
pip install "orca-mlips[mace]"        # MACE
pip install "orca-mlips[orb,mace]"    # OrbMol + MACE
pip install orca-mlips                # core only
```

> **Note:** UMA and MACE conflict at dependency level (`e3nn`). Use separate environments.

Local install:
```bash
git clone https://github.com/t-0hmura/orca-mlips.git
cd orca-mlips
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
- Prefixed: `orca-mlips-uma`, `orca-mlips-orb`, `orca-mlips-mace`

## Troubleshooting

- **`ProgExt "uma"` runs the wrong plugin** — Use `ProgExt "orca-mlips-uma"` to avoid alias conflicts.
- **`uma` command not found** — Activate the conda environment where the package is installed.
- **UMA model download fails (401/403)** — Run `huggingface-cli login`. Some models require access approval on Hugging Face.
- **Works interactively but fails in PBS jobs** — Use absolute path from `which uma` in the ORCA input.

## References

- ORCA ExtTool: https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/externaloptimizer.html
- ORCA external tools: https://github.com/faccts/orca-external-tools
