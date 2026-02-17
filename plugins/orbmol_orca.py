#!/usr/bin/env python3
"""ORCA ExtTool plugin for OrbMol (orb-models)."""

from __future__ import absolute_import, division, print_function

import os

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    import sys

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from cli_orca import main_backend
else:
    from .cli_orca import main_backend


if __name__ == "__main__":
    main_backend("orbmol", "orbmol_orca.py")
