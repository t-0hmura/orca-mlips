# Solvent Effects (xTB Implicit-Solvent Delta Correction)

`orca-mlips` supports an implicit-solvent correction that adds only solvent
contributions to MLIP vacuum predictions.

## Solvent Delta Terms Added to MLIP Outputs

For each geometry `R`, the plugin evaluates xTB twice:

- vacuum
- implicit solvent (`--solvent <name>`, model selected by `--solvent-model`)

The solvent delta terms are defined as:

- `dE(R) = E_xTB(solv) - E_xTB(vac)`
- `dF(R) = F_xTB(solv) - F_xTB(vac)`
- `dH(R) = H_xTB(solv) - H_xTB(vac)`

The plugin then returns:

- `E_total = E_MLIP(vac) + dE`
- `F_total = F_MLIP(vac) + dF`
- `H_total = H_MLIP(vac) + dH`

This keeps the MLIP model in vacuum mode and adds only solvent-induced correction terms.

## CLI Options

- `--solvent <name|none>`: enable implicit-solvent correction (`none` disables it)
- `--solvent-model <alpb|cpcmx>`: solvent model selector (default: `alpb`)
  - `alpb` uses xTB `--alpb`
  - `cpcmx` uses xTB `--cpcmx`
- `--xtb-cmd <path_or_cmd>`: xTB executable (default: `xtb`)
- `--xtb-acc <float>`: xTB accuracy value (default: `0.2`)
- `--xtb-workdir <tmp|path>`: per-call xTB scratch base (default: `tmp`)
- `--xtb-keep-files`: keep xTB temporary files

## CPCM-X Setup

CPCM-X requires xTB to be built from source with CPCM-X linked in.
The conda-forge `xtb` package does not include CPCM-X support.

**Step 1: Build xTB with `-DWITH_CPCMX=ON`**

CPCM-X is bundled in the xTB source tree (`subprojects/cpx.wrap`) and is fetched automatically during the CMake configure step. Requires GCC >= 10 (gfortran 8 causes internal compiler errors).

```bash
git clone --depth 1 https://github.com/grimme-lab/xtb.git
cd xtb
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_CPCMX=ON \
  -DBLAS_LIBRARIES=/path/to/libblas.so \
  -DLAPACK_LIBRARIES=/path/to/liblapack.so
make -C build tblite-lib -j8   # build tblite first to avoid a parallel build race
make -C build xtb-exe -j8
```

**Step 2: Use the custom xTB via `--xtb-cmd`**

```text
%method
  ProgExt "uma"
  Ext_Params "--solvent water --solvent-model cpcmx --xtb-cmd /path/to/xtb"
end
```

`CPXHOME` must be set at runtime to point to the CPCM-X source directory (containing `DB/`). When xTB fetches CPCM-X during build, the source is placed under `build/_deps/cpcmx-src/`.

For full details, see:
- https://github.com/grimme-lab/xtb
- https://github.com/grimme-lab/CPCM-X

## ORCA Usage

### Geometry optimization

```text
! ExtOpt Opt
%method
  ProgExt "uma"
  Ext_Params "--solvent water --solvent-model alpb"
end
```

### Dump solvent-corrected Hessian (`.hess`)

```text
! ExtOpt Opt
%geom
  MaxIter 1
end
%method
  ProgExt "uma"
  Ext_Params "--solvent water --solvent-model cpcmx --dump-hessian cla_solv.hess"
end
```

Then read in the next job:

```text
%geom
  InHessName "cla_solv.hess"
end
```

## Performance Notes

- Solvent correction runs two xTB calculations per geometry point (vacuum + solvated state).
- Hessian correction is expensive: each state needs an xTB Hessian.
- Use `--dump-hessian` only at required points in the workflow.

## Troubleshooting

- `xTB command not found`:
  - Install xTB in the active environment.
  - Or set `--xtb-cmd /full/path/to/xtb`.
- `xTB solvent correction failed`:
  - Verify the solvent spelling (`water`, `thf`, `toluene`, ...).
  - For `--solvent-model cpcmx`, use an xTB build with CPCM-X support
    (see https://github.com/grimme-lab/CPCM-X).
  - Rerun with `--xtb-keep-files --xtb-workdir <path>` and inspect the files.
