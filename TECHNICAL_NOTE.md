# Technical Note

## Solvent-Only Delta Injection

The plugin adds xTB implicit-solvent-vacuum deltas to backend MLIP outputs:

- `dE = E_xTB(solv) - E_xTB(vac)`
- `dF = F_xTB(solv) - F_xTB(vac)`
- `dH = H_xTB(solv) - H_xTB(vac)`

Injected quantities:

- `E_total = E_MLIP + dE`
- `F_total = F_MLIP + dF`
- `H_total = H_MLIP + dH`

This is implemented in `plugins/runner_orca.py` through
`plugins/xtb_alpb_correction.py`.

## Units

xTB parsed units:

- energy: `Eh`
- gradient: `Eh/Bohr`
- Hessian: `Eh/Bohr^2`

Converted to MLIP units before addition:

- energy: `eV`
- forces: `eV/Ang`
- Hessian: `eV/Ang^2`

Force conversion uses `F = -grad`.

## Hessian Path Per Backend

The shared backend implementation is in `plugins/mlip_backends.py`.

### UMA / ORB

UMA/ORB analytical Hessians are computed by autograd on energy with model-state
management to avoid graph/dropout issues:

1. `_prepare_model_for_autograd_hessian(...)`
2. compute Hessian with `torch.autograd.functional.hessian(...)`
3. `_restore_model_after_autograd_hessian(...)`

Key points:

- model is switched to train mode for reliable autograd graph construction
- dropout modules are effectively disabled (`p=0`, eval behavior)
- original training/dropout/`requires_grad` states are restored after Hessian

### MACE

Uses calculator-native Hessian path (`get_hessian`) when available.

### AIMNet2

Requests Hessian from AIMNet2 calculator outputs and reshapes to `(3N, 3N)`.

## ORCA ExtTool Integration

`runner_orca.py` flow:

1. read ORCA extinp (`symbols`, `coords`, charge/multiplicity, gradient flag)
2. evaluate MLIP backend
3. if `--solvent != none`, evaluate xTB delta and add `dE/dF/dH`
4. write `.engrad` and optional `.hess` (`--dump-hessian`)

Available solvent models:
- `--solvent-model alpb` -> xTB `--alpb`
- `--solvent-model cpcmx` -> xTB `--cpcmx`

For `--dump-hessian`, solvent-corrected Hessian is mandatory and enforced.
