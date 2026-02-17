#!/usr/bin/env python3
"""ORCA ExtTool plugin for OrbMol (orb-models)."""

from __future__ import absolute_import, division, print_function

import os
import sys

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from mlip_backends import OrbMolEvaluator, get_available_orb_models
    from runner_orca import main_entry, run_orca_plugin
else:
    from .mlip_backends import OrbMolEvaluator, get_available_orb_models
    from .runner_orca import main_entry, run_orca_plugin


def _add_args(parser):
    parser.add_argument("--precision", default="float32-high", help="orb-models precision")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable model compilation if supported by installed orb-models version.",
    )


def _make_evaluator(args):
    return OrbMolEvaluator(
        model=args.model,
        device=args.device,
        precision=args.precision,
        compile_model=bool(args.compile_model),
    )


def _available_models():
    return get_available_orb_models()


def main(argv=None):
    if argv is None:
        argv = []
    return run_orca_plugin(
        argv=argv,
        plugin_name="orbmol_orca.py",
        make_evaluator=_make_evaluator,
        available_models=_available_models,
        default_model="orb_v3_conservative_omol",
        add_extra_args=_add_args,
    )


if __name__ == "__main__":
    main_entry(main)
