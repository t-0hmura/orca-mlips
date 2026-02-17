#!/usr/bin/env python3
"""Shared ORCA plugin runner."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import numpy as np

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from mlip_backends import forces_ev_ang_to_gradient_ha_bohr, ev_to_ha
    from orca_extio import read_extinp, write_engrad
else:
    from .mlip_backends import forces_ev_ang_to_gradient_ha_bohr, ev_to_ha
    from .orca_extio import read_extinp, write_engrad


class RunnerError(RuntimeError):
    pass


def _write_hessian_dump(path, hessian):
    hessian = np.asarray(hessian, dtype=np.float64)
    dof = hessian.shape[0]
    with open(path, "w") as handle:
        handle.write("# Hessian in eV/Angstrom^2\n")
        handle.write("# shape: {} x {}\n".format(dof, dof))
        np.savetxt(handle, hessian, fmt="%.12e")


def run_orca_plugin(
    argv,
    plugin_name,
    make_evaluator,
    available_models,
    default_model,
    add_extra_args,
):
    parser = argparse.ArgumentParser(
        prog=plugin_name,
        description=(
            "ORCA external plugin for {}.\n"
            "Call signature from ORCA: {} <basename_EXT.extinp.tmp> [options]"
        ).format(plugin_name.replace("_", " "), plugin_name)
    )
    parser.add_argument("extinp", nargs="?", help="ORCA ExtTool input file (<basename_EXT.extinp.tmp>)")
    parser.add_argument("--model", default=default_model, help="Model name/alias/path")
    parser.add_argument("--device", default="auto", help="cpu|cuda|auto")
    parser.add_argument("--hessian-mode", choices=["Analytical", "Numerical"], default="Analytical")
    parser.add_argument("--hessian-step", type=float, default=1.0e-3, help="Finite-difference step in Angstrom")
    parser.add_argument(
        "--strict-hessian",
        action="store_true",
        help="Fail instead of falling back to numerical Hessian when analytical Hessian is unavailable.",
    )
    parser.add_argument(
        "--dump-hessian",
        default=None,
        help="Optional path to dump Hessian matrix (eV/Angstrom^2). Not used by ORCA itself.",
    )
    parser.add_argument("--list-models", action="store_true", help="Print available model aliases and exit")

    if add_extra_args is not None:
        add_extra_args(parser)

    args = parser.parse_args(argv)

    if args.list_models:
        models = list(available_models())
        for item in models:
            print(item)
        return 0

    if not args.extinp:
        parser.error("extinp path is required unless --list-models is used.")

    ext = read_extinp(args.extinp)

    evaluator = make_evaluator(args)

    need_grad = bool(ext["do_gradient"])
    need_hess = args.dump_hessian is not None

    energy_ev, forces_ev_ang, hessian_ev_ang2 = evaluator.evaluate(
        symbols=ext["symbols"],
        coords_ang=ext["coords_ang"],
        charge=ext["charge"],
        multiplicity=ext["multiplicity"],
        need_forces=need_grad,
        need_hessian=need_hess,
        hessian_mode=args.hessian_mode,
        hessian_step=float(args.hessian_step),
        strict_hessian=bool(args.strict_hessian),
    )

    grad_ha_bohr = None
    if need_grad:
        if forces_ev_ang is None:
            raise RunnerError("Backend returned no forces although ORCA requested gradient.")
        grad_ha_bohr = forces_ev_ang_to_gradient_ha_bohr(forces_ev_ang)

    write_engrad(
        path=ext["engrad_path"],
        natoms=len(ext["symbols"]),
        energy_ha=ev_to_ha(energy_ev),
        gradient_ha_bohr=grad_ha_bohr,
    )

    if args.dump_hessian:
        if hessian_ev_ang2 is None:
            raise RunnerError("Hessian dump was requested but backend returned no Hessian.")
        dump_path = args.dump_hessian
        if not os.path.isabs(dump_path):
            dump_path = os.path.join(ext["extinp_dir"], dump_path)
        _write_hessian_dump(dump_path, hessian_ev_ang2)

    return 0


def main_entry(entry_fn):
    try:
        code = int(entry_fn(sys.argv[1:]))
        raise SystemExit(code)
    except Exception as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
