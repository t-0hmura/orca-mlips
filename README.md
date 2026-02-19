# orca-mlips

MLIP (Machine Learning Interatomic Potential) plugins for ORCA `ExtTool` (`ProgExt`) interface.

Four model families are currently supported:
- **UMA** ([fairchem](https://github.com/facebookresearch/fairchem)) — default model: `uma-s-1p1`
- **ORB** ([orb-models](https://github.com/orbital-materials/orb-models)) — default model: `orb_v3_conservative_omol`
- **MACE** ([mace](https://github.com/ACEsuit/mace)) — default model: `MACE-OMOL-0`
- **AIMNet2** ([aimnetcentral](https://github.com/isayevlab/aimnetcentral)) — default model: `aimnet2`

All backends provide energy and gradient, and can output an **analytical Hessian** in **ORCA** `.hess` format via `--dump-hessian`.  

> The model server starts automatically and stays resident in memory, so repeated calls during optimization are fast.

Requires **Python 3.9** or later.

## Quick Start (Default = UMA)

1. Install PyTorch for your environment (CUDA/CPU).
```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
```

2. Install the package with UMA profile. If you need ORB/MACE/AIMNet2, use `orca-mlips[orb]`/`orca-mlips[mace]`/`orca-mlips[aimnet2]`.
```bash
pip install "orca-mlips[uma]"
```

3. Log in to Hugging Face for UMA model access. (Not required for ORB/MACE/AIMNet2)
```bash
huggingface-cli login
```

4. Use in an ORCA input file. If you use ORB/MACE/AIMNet2, use `ProgExt "orb"`/`ProgExt "mace"`/`ProgExt "aimnet2"`.
For detailed ORCA External Tool / `ExtOpt` usage, see https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html
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

%method
  ProgExt "mace"
end

%method
  ProgExt "aimnet2"
end
```

> **Note:** Run `uma --list-models` to see available models. If the `uma` alias conflicts in your environment, use `orca-mlips-uma` instead.

Additional examples: `examples/cla_hess_uma.inp` + `examples/cla_uma.inp`, `examples/sn2_hess_orb.inp` + `examples/sn2_orb.inp`, `examples/water_hess_mace.inp` + `examples/water_mace.inp`, `examples/cla_hess_aimnet2.inp` + `examples/cla_aimnet2.inp`

## Using Analytical Hessian (optional two-step workflow)

Optimization and TS searches can run without providing an initial Hessian — ORCA builds one internally. Providing an analytical Hessian from the MLIP via `--dump-hessian` + `InHessName` improves convergence, especially for TS searches.

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
A single-iteration optimization that triggers the ExtTool call and writes the analytical Hessian in ORCA `.hess` format. `! ExtOpt` is required to make ORCA use the external tool instead of its own internal methods. The job may exit non-zero (not converged), but the `.hess` file is created.

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
  Ext_Params "--dump-hessian water.hess"
end
* xyz 0 1
...
*
```
then:
```text
! ExtOpt Opt
%method
  ProgExt "mace"
end
%geom
  InHessName "water.hess"
end
* xyz 0 1
...
*
```

## Installing Model Families

```bash
pip install "orca-mlips[uma]"         # UMA (default)
pip install "orca-mlips[orb]"         # ORB
pip install "orca-mlips[mace]"        # MACE
pip install "orca-mlips[orb,mace]"    # ORB + MACE
pip install "orca-mlips[aimnet2]"     # AIMNet2
pip install "orca-mlips[orb,mace,aimnet2]"  # ORB + MACE + AIMNet2
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
- **ORB / MACE / AIMNet2**: Downloaded automatically on first use.

## Upstream Model Sources

- UMA / FAIR-Chem: https://github.com/facebookresearch/fairchem
- ORB / orb-models: https://github.com/orbital-materials/orb-models
- MACE: https://github.com/ACEsuit/mace
- AIMNet2: https://github.com/isayevlab/aimnetcentral

## Advanced Options

See `OPTIONS.md` for backend-specific tuning parameters.

Command aliases:
- Short: `uma`, `orb`, `mace`, `aimnet2`
- Prefixed: `orca-mlips-uma`, `orca-mlips-orb`, `orca-mlips-mace`, `orca-mlips-aimnet2`

## Troubleshooting

- **`ProgExt "uma"` runs the wrong plugin** — Use `ProgExt "orca-mlips-uma"` to avoid alias conflicts.
- **`ProgExt "aimnet2"` runs the wrong plugin** — Use `ProgExt "orca-mlips-aimnet2"` to avoid alias conflicts.
- **`uma` command not found** — Activate the conda environment where the package is installed.
- **UMA model download fails (401/403)** — Run `huggingface-cli login`. Some models require access approval on Hugging Face.
- **Works interactively but fails in PBS jobs** — Use absolute path from `which uma` in the ORCA input.

## Citation

If you use this package, please cite:

```bibtex
@software{ohmura2026orcamlips,
  author       = {Ohmura, Takuto},
  title        = {orca-mlips},
  year         = {2026},
  month        = {2},
  version      = {1.0.0},
  url          = {https://github.com/t-0hmura/orca-mlips},
  license      = {MIT},
  doi          = {10.5281/zenodo.18691881}
}
```

## References

- ORCA ExtTool official tutorial (ExtOpt workflow): https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html
- ORCA ExtTool: https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/externaloptimizer.html
- ORCA external tools: https://github.com/faccts/orca-external-tools
