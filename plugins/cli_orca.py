#!/usr/bin/env python3
"""Shared CLI wiring for ORCA backend plugins."""

from __future__ import absolute_import, division, print_function

import os
import sys

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from mlip_backends import (
        MACEEvaluator,
        OrbMolEvaluator,
        UMAEvaluator,
        get_available_mace_models,
        get_available_orb_models,
        get_available_uma_models,
        get_available_uma_tasks,
    )
    from runner_orca import main_entry, run_orca_plugin
else:
    from .mlip_backends import (
        MACEEvaluator,
        OrbMolEvaluator,
        UMAEvaluator,
        get_available_mace_models,
        get_available_orb_models,
        get_available_uma_models,
        get_available_uma_tasks,
    )
    from .runner_orca import main_entry, run_orca_plugin


def _uma_add_args(parser):
    parser.add_argument("--task", default="omol", help="UMA task (e.g. omol, omat, oc20, oc25, odac, omc)")
    parser.add_argument("--workers", type=int, default=1, help="fairchem predictor worker count")


def _orb_add_args(parser):
    parser.add_argument("--precision", default="float32-high", help="orb-models precision")
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Enable model compilation if supported by installed orb-models version.",
    )


def _mace_add_args(parser):
    parser.add_argument("--dtype", default="float64", help="MACE default dtype (float32 or float64)")


def _uma_make_evaluator(args):
    return UMAEvaluator(model=args.model, task=args.task, device=args.device, workers=args.workers)


def _orb_make_evaluator(args):
    return OrbMolEvaluator(
        model=args.model,
        device=args.device,
        precision=args.precision,
        compile_model=bool(args.compile_model),
    )


def _mace_make_evaluator(args):
    return MACEEvaluator(model=args.model, device=args.device, default_dtype=args.dtype)


_BACKEND_CONFIG = {
    "uma": {
        "default_model": "uma-s-1p1",
        "add_args": _uma_add_args,
        "make_evaluator": _uma_make_evaluator,
        "available_models": get_available_uma_models,
        "available_tasks": get_available_uma_tasks,
    },
    "orbmol": {
        "default_model": "orb_v3_conservative_omol",
        "add_args": _orb_add_args,
        "make_evaluator": _orb_make_evaluator,
        "available_models": get_available_orb_models,
        "available_tasks": None,
    },
    "mace": {
        "default_model": "MACE-OMOL-0",
        "add_args": _mace_add_args,
        "make_evaluator": _mace_make_evaluator,
        "available_models": get_available_mace_models,
        "available_tasks": None,
    },
}


def run_backend(backend_key, plugin_name, argv):
    cfg = _BACKEND_CONFIG[str(backend_key)]

    if "--list-tasks" in argv:
        list_tasks = cfg.get("available_tasks")
        if list_tasks is None:
            raise SystemExit("--list-tasks is only supported for UMA plugins.")
        for task in list_tasks():
            print(task)
        return 0

    return run_orca_plugin(
        argv=argv,
        plugin_name=plugin_name,
        make_evaluator=cfg["make_evaluator"],
        available_models=cfg["available_models"],
        default_model=cfg["default_model"],
        add_extra_args=cfg["add_args"],
    )


def main_backend(backend_key, plugin_name):
    def _entry(argv):
        return run_backend(backend_key=backend_key, plugin_name=plugin_name, argv=argv)

    main_entry(_entry)

