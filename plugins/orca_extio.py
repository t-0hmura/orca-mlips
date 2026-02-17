#!/usr/bin/env python3
"""ORCA ExtTool input/output helpers."""

from __future__ import absolute_import, division, print_function

import os

import numpy as np


class OrcaIOError(RuntimeError):
    pass


def _strip_comment(line):
    return line.split("#", 1)[0].strip()


def read_xyz(xyz_path):
    xyz_path = os.path.abspath(xyz_path)
    if not os.path.isfile(xyz_path):
        raise OrcaIOError("XYZ file not found: {}".format(xyz_path))

    with open(xyz_path, "r") as handle:
        raw = [ln.rstrip("\n") for ln in handle]

    if len(raw) < 2:
        raise OrcaIOError("Malformed XYZ file: {}".format(xyz_path))

    try:
        nat = int(raw[0].strip())
    except Exception:
        raise OrcaIOError("First line of XYZ must be number of atoms: {}".format(xyz_path))

    rows = raw[2 : 2 + nat]
    if len(rows) != nat:
        raise OrcaIOError("XYZ atom row count mismatch in {}".format(xyz_path))

    symbols = []
    coords = []
    for row in rows:
        parts = row.split()
        if len(parts) < 4:
            raise OrcaIOError("Malformed XYZ atom line: '{}'".format(row))
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return symbols, np.asarray(coords, dtype=np.float64)


def read_extinp(extinp_path):
    extinp_path = os.path.abspath(extinp_path)
    if not os.path.isfile(extinp_path):
        raise OrcaIOError("extinp file not found: {}".format(extinp_path))

    with open(extinp_path, "r") as handle:
        entries = [_strip_comment(ln) for ln in handle]
    entries = [x for x in entries if x]

    if len(entries) < 5:
        raise OrcaIOError(
            "extinp requires at least 5 entries (xyz, charge, multiplicity, ncores, do_gradient)."
        )

    xyz_rel = entries[0]
    charge = int(entries[1])
    multiplicity = int(entries[2])
    ncores = int(entries[3])
    do_gradient = int(entries[4])

    if do_gradient not in (0, 1):
        raise OrcaIOError("do_gradient must be 0 or 1, got {}".format(do_gradient))

    ext_dir = os.path.dirname(extinp_path)
    xyz_path = xyz_rel if os.path.isabs(xyz_rel) else os.path.join(ext_dir, xyz_rel)

    pointcharges = None
    if len(entries) >= 6:
        pc_rel = entries[5]
        pointcharges = pc_rel if os.path.isabs(pc_rel) else os.path.join(ext_dir, pc_rel)

    symbols, coords = read_xyz(xyz_path)

    xyz_basename = os.path.basename(xyz_path)
    if xyz_basename.lower().endswith(".xyz"):
        base = xyz_basename[:-4]
    else:
        base = xyz_basename

    engrad_path = os.path.join(ext_dir, base + ".engrad")

    return {
        "extinp_path": extinp_path,
        "extinp_dir": ext_dir,
        "xyz_path": os.path.abspath(xyz_path),
        "engrad_path": os.path.abspath(engrad_path),
        "pointcharges_path": os.path.abspath(pointcharges) if pointcharges else None,
        "symbols": symbols,
        "coords_ang": coords,
        "charge": charge,
        "multiplicity": multiplicity,
        "ncores": ncores,
        "do_gradient": bool(do_gradient),
    }


def write_engrad(path, natoms, energy_ha, gradient_ha_bohr):
    """Write ORCA .engrad file compatible with ORCA 5/6."""
    out = []
    out.append("#")
    out.append("# Number of atoms")
    out.append("#")
    out.append(str(int(natoms)))
    out.append("#")
    out.append("# Total energy [Eh]")
    out.append("#")
    out.append("{:.12e}".format(float(energy_ha)))

    if gradient_ha_bohr is not None:
        grad = np.asarray(gradient_ha_bohr, dtype=np.float64).reshape(-1)
        out.append("#")
        out.append("# Gradient [Eh/Bohr] A1X, A1Y, A1Z, A2X, ...")
        out.append("#")
        out.extend(["{:.12e}".format(float(x)) for x in grad])

    with open(path, "w") as handle:
        handle.write("\n".join(out) + "\n")
