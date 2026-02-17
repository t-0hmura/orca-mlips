#!/usr/bin/env python3
"""ORCA ExtTool plugin for MACE."""

from __future__ import absolute_import, division, print_function

import os
import sys

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from mlip_backends import MACEEvaluator, get_available_mace_models
    from runner_orca import main_entry, run_orca_plugin
else:
    from .mlip_backends import MACEEvaluator, get_available_mace_models
    from .runner_orca import main_entry, run_orca_plugin


def _add_args(parser):
    parser.add_argument("--dtype", default="float64", help="MACE default dtype (float32 or float64)")


def _make_evaluator(args):
    return MACEEvaluator(
        model=args.model,
        device=args.device,
        default_dtype=args.dtype,
    )


def _available_models():
    return get_available_mace_models()


def main(argv=None):
    if argv is None:
        argv = []
    return run_orca_plugin(
        argv=argv,
        plugin_name="mace_orca.py",
        make_evaluator=_make_evaluator,
        available_models=_available_models,
        default_model="MACE-OMOL-0",
        add_extra_args=_add_args,
    )


if __name__ == "__main__":
    main_entry(main)
