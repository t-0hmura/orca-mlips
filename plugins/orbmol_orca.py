#!/usr/bin/env python3
"""ORCA ExtTool plugin for OrbMol (orb-models)."""

from __future__ import absolute_import, division, print_function

import os
import sys

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))

    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from cli_orca import main_backend
else:
    from .cli_orca import main_backend


def main():
    plugin_name = os.path.basename(sys.argv[0]) or "orbmol_orca.py"
    main_backend("orbmol", plugin_name)


if __name__ == "__main__":
    main()
