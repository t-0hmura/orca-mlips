#!/usr/bin/env python3
"""ORCA ExtTool plugin for UMA (fairchem)."""

from __future__ import absolute_import, division, print_function

import os
import sys

if __package__ in (None, ""):
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from mlip_backends import UMAEvaluator, get_available_uma_models, get_available_uma_tasks
    from runner_orca import main_entry, run_orca_plugin
else:
    from .mlip_backends import (
        UMAEvaluator,
        get_available_uma_models,
        get_available_uma_tasks,
    )
    from .runner_orca import main_entry, run_orca_plugin


def _add_args(parser):
    parser.add_argument("--task", default="omol", help="UMA task (e.g. omol, omat, oc20, oc25, odac, omc)")
    parser.add_argument("--workers", type=int, default=1, help="fairchem predictor worker count")


def _make_evaluator(args):
    return UMAEvaluator(
        model=args.model,
        task=args.task,
        device=args.device,
        workers=args.workers,
    )


def _available_models():
    return get_available_uma_models()


def main(argv=None):
    if argv is None:
        argv = []
    return run_orca_plugin(
        argv=argv,
        plugin_name="uma_orca.py",
        make_evaluator=_make_evaluator,
        available_models=_available_models,
        default_model="uma-s-1p1",
        add_extra_args=_add_args,
    )


if __name__ == "__main__":
    import sys

    # Friendly validation for UMA task list when explicitly requested.
    if "--list-tasks" in sys.argv:
        for task in get_available_uma_tasks():
            print(task)
        raise SystemExit(0)
    main_entry(main)
