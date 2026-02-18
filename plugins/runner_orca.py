#!/usr/bin/env python3
"""Shared ORCA plugin runner."""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import numpy as np
try:
    from importlib import metadata as _importlib_metadata
except Exception:
    try:
        import importlib_metadata as _importlib_metadata
    except Exception:
        _importlib_metadata = None

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from mlip_backends import (
        forces_ev_ang_to_gradient_ha_bohr,
        ev_to_ha,
        HARTREE_PER_EV,
        BOHR_PER_ANG,
    )
    from orca_extio import read_extinp, write_engrad
    from mlip_server import (
        MLIPServer,
        ServerError,
        auto_server_socket,
        client_evaluate,
        ensure_server,
        send_shutdown,
        server_is_alive,
    )
else:
    from .mlip_backends import (
        forces_ev_ang_to_gradient_ha_bohr,
        ev_to_ha,
        HARTREE_PER_EV,
        BOHR_PER_ANG,
    )
    from .orca_extio import read_extinp, write_engrad
    from .mlip_server import (
        MLIPServer,
        ServerError,
        auto_server_socket,
        client_evaluate,
        ensure_server,
        send_shutdown,
        server_is_alive,
    )


class RunnerError(RuntimeError):
    pass


def _backend_alias(plugin_name):
    low = os.path.basename(str(plugin_name or "")).lower()
    if "orb" in low:
        return "orb"
    if "mace" in low:
        return "mace"
    return "uma"


def _looks_like_gaussian_invocation(argv):
    if len(argv) < 6:
        return False
    tail = argv[-6:]
    layer = str(tail[0]).strip().upper()
    return layer in ("R", "M", "S")


def _version_text(plugin_name):
    version = "dev"
    if _importlib_metadata is not None:
        try:
            version = _importlib_metadata.version("mlips4orca")
        except Exception:
            pass
    return "{} (mlips4orca {})".format(plugin_name, version)


_STANDARD_ATOMIC_MASSES = {
    "H": 1.00794, "He": 4.00260, "Li": 6.941, "Be": 9.01218,
    "B": 10.811, "C": 12.0107, "N": 14.0067, "O": 15.9994,
    "F": 18.9984, "Ne": 20.1797, "Na": 22.9898, "Mg": 24.3050,
    "Al": 26.9815, "Si": 28.0855, "P": 30.9738, "S": 32.065,
    "Cl": 35.453, "Ar": 39.948, "K": 39.0983, "Ca": 40.078,
    "Sc": 44.9559, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961,
    "Mn": 54.9380, "Fe": 55.845, "Co": 58.9332, "Ni": 58.6934,
    "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.64,
    "As": 74.9216, "Se": 78.96, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.4678, "Sr": 87.62, "Y": 88.9059, "Zr": 91.224,
    "Nb": 92.9064, "Mo": 95.96, "Tc": 98.0, "Ru": 101.07,
    "Rh": 102.906, "Pd": 106.42, "Ag": 107.868, "Cd": 112.411,
    "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60,
    "I": 126.904, "Xe": 131.293, "Cs": 132.905, "Ba": 137.327,
}


def _write_orca_hessian(path, symbols, coords_ang, hessian_ev_ang2, energy_ev):
    """Write Hessian in ORCA .hess format (readable by inhess Read)."""
    hess = np.asarray(hessian_ev_ang2, dtype=np.float64)
    coords = np.asarray(coords_ang, dtype=np.float64)
    natoms = len(symbols)
    ndim = 3 * natoms

    # Convert: eV/Ang^2 -> Eh/Bohr^2
    hess_au = hess * (HARTREE_PER_EV / (BOHR_PER_ANG * BOHR_PER_ANG))
    # Convert: Ang -> Bohr
    coords_bohr = coords * BOHR_PER_ANG
    energy_ha = energy_ev * HARTREE_PER_EV

    block_size = 5

    with open(path, "w") as f:
        f.write("$orca_hessian_file\n\n")

        f.write("$act_energy\n")
        f.write("{:.12f}\n\n".format(energy_ha))

        f.write("$hessian\n")
        f.write("{}\n".format(ndim))
        for col_start in range(0, ndim, block_size):
            col_end = min(col_start + block_size, ndim)
            cols = list(range(col_start, col_end))
            f.write("".join("{:>12d}".format(c) for c in cols) + "\n")
            for row in range(ndim):
                f.write("{:>5d}".format(row))
                for c in cols:
                    f.write("  {:16.10f}".format(hess_au[row, c]))
                f.write("\n")
        f.write("\n")

        f.write("$atoms\n")
        f.write("{}\n".format(natoms))
        for i, sym in enumerate(symbols):
            mass = _STANDARD_ATOMIC_MASSES.get(sym, 1.0)
            x, y, z = coords_bohr[i]
            f.write(" {:<4s} {:10.4f} {:16.10f} {:16.10f} {:16.10f}\n".format(
                sym, mass, x, y, z))
        f.write("\n")


def _add_server_args(parser):
    """Add server-related arguments to the parser."""
    parser.add_argument(
        "--server-socket", default=None,
        help="Path to Unix domain socket for persistent model server.",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start as a persistent model server (internal use).",
    )
    parser.add_argument(
        "--stop-server", action="store_true",
        help="Send shutdown signal to a running server.",
    )
    parser.add_argument(
        "--no-server", action="store_true",
        help="Disable auto server mode; load model directly each time.",
    )
    parser.add_argument(
        "--server-idle-timeout", type=int, default=600,
        help="Server idle timeout in seconds (default: 600).",
    )
    parser.add_argument(
        "--server-parent-pid", type=int, default=None, help=argparse.SUPPRESS
    )


def _handle_serve(args, make_evaluator):
    if not args.server_socket:
        raise SystemExit("--serve requires --server-socket PATH")
    evaluator = make_evaluator(args)
    server = MLIPServer(
        evaluator=evaluator,
        socket_path=args.server_socket,
        idle_timeout=args.server_idle_timeout,
        parent_pid=args.server_parent_pid,
    )
    server.serve_forever()
    return 0


def _handle_stop_server(args):
    if not args.server_socket:
        raise SystemExit("--stop-server requires --server-socket PATH")
    if not server_is_alive(args.server_socket):
        print("No server running at {}".format(args.server_socket), file=sys.stderr)
        return 1
    resp = send_shutdown(args.server_socket)
    print("Server response: {}".format(resp), file=sys.stderr)
    return 0


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
    parser.add_argument(
        "--dump-hessian",
        default=None,
        help="Dump analytical Hessian in ORCA .hess format (usable with inhess Read).",
    )
    parser.add_argument("--list-models", action="store_true", help="Print available model aliases and exit")
    parser.add_argument(
        "--version",
        action="version",
        version=_version_text(plugin_name),
    )

    _add_server_args(parser)

    if add_extra_args is not None:
        add_extra_args(parser)

    # --serve / --stop-server: no ORCA extinp needed
    if "--serve" in argv or "--stop-server" in argv:
        args = parser.parse_args(argv)
        if args.stop_server:
            return _handle_stop_server(args)
        if args.serve:
            return _handle_serve(args, make_evaluator)
        return 0

    if _looks_like_gaussian_invocation(argv):
        backend = _backend_alias(plugin_name)
        parser.error(
            "Detected Gaussian-style invocation (LAYER + 5 generated files). "
            "This command appears to be an ORCA plugin, but it was called in Gaussian style. "
            "If short aliases are conflicting, set Gaussian External to '{}'.".format(
                "mlips4g16-" + backend
            )
        )

    args = parser.parse_args(argv)

    if args.list_models:
        models = list(available_models())
        for item in models:
            print(item)
        return 0

    if not args.extinp:
        parser.error("extinp path is required unless --list-models is used.")

    ext = read_extinp(args.extinp)

    need_grad = bool(ext["do_gradient"])
    need_hess = args.dump_hessian is not None

    # --- Evaluation: auto server mode (default) or direct mode ---
    if not args.no_server:
        parent_pid = os.getppid()
        socket_path = args.server_socket or auto_server_socket(args, parent_pid=parent_pid)

        # Build custom_args for server auto-start (exclude extinp positional arg)
        custom_args = [a for a in argv if a != args.extinp]
        for flag in ("--no-server",):
            while flag in custom_args:
                custom_args.remove(flag)

        server_ready = ensure_server(
            executable=sys.argv[0],
            custom_args=custom_args,
            socket_path=socket_path,
            idle_timeout=args.server_idle_timeout,
            parent_pid=parent_pid,
        )
        if server_ready:
            try:
                energy_ev, forces_ev_ang, hessian_ev_ang2 = client_evaluate(
                    socket_path=socket_path,
                    symbols=ext["symbols"],
                    coords_ang=ext["coords_ang"],
                    charge=ext["charge"],
                    multiplicity=ext["multiplicity"],
                    need_forces=need_grad,
                    need_hessian=need_hess,
                    hessian_mode="Analytical",
                    hessian_step=1.0e-3,
                )
            except ServerError as exc:
                print(
                    "[mlip-client] WARNING: Server error: {}. "
                    "Falling back to direct mode.".format(exc),
                    file=sys.stderr, flush=True,
                )
                evaluator = make_evaluator(args)
                energy_ev, forces_ev_ang, hessian_ev_ang2 = evaluator.evaluate(
                    symbols=ext["symbols"],
                    coords_ang=ext["coords_ang"],
                    charge=ext["charge"],
                    multiplicity=ext["multiplicity"],
                    need_forces=need_grad,
                    need_hessian=need_hess,
                    hessian_mode="Analytical",
                    hessian_step=1.0e-3,
                )
        else:
            print(
                "[mlip-client] WARNING: Server not available, "
                "loading model directly.",
                file=sys.stderr, flush=True,
            )
            evaluator = make_evaluator(args)
            energy_ev, forces_ev_ang, hessian_ev_ang2 = evaluator.evaluate(
                symbols=ext["symbols"],
                coords_ang=ext["coords_ang"],
                charge=ext["charge"],
                multiplicity=ext["multiplicity"],
                need_forces=need_grad,
                need_hessian=need_hess,
                hessian_mode="Analytical",
                hessian_step=1.0e-3,
            )
    else:
        evaluator = make_evaluator(args)
        energy_ev, forces_ev_ang, hessian_ev_ang2 = evaluator.evaluate(
            symbols=ext["symbols"],
            coords_ang=ext["coords_ang"],
            charge=ext["charge"],
            multiplicity=ext["multiplicity"],
            need_forces=need_grad,
            need_hessian=need_hess,
            hessian_mode="Analytical",
            hessian_step=1.0e-3,
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
        _write_orca_hessian(
            dump_path, ext["symbols"], ext["coords_ang"],
            hessian_ev_ang2, energy_ev,
        )

    return 0


def main_entry(entry_fn):
    try:
        code = int(entry_fn(sys.argv[1:]))
        raise SystemExit(code)
    except Exception as exc:
        print("[ERROR] {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
