#!/usr/bin/env python3
"""Common MLIP backend utilities for ORCA/Gaussian external plugins.

This module provides four backend classes:
- UMAEvaluator
- OrbMolEvaluator
- MACEEvaluator
- AIMNet2Evaluator

All backends expose the same high-level method:
    evaluate(..., need_hessian, hessian_mode, hessian_step)

Units returned by evaluators:
- energy: eV
- forces: eV/Angstrom
- hessian: eV/Angstrom^2 (Cartesian, 3N x 3N)
"""

from __future__ import absolute_import, division, print_function

import os
import tempfile
import urllib.request

import numpy as np

HARTREE_PER_EV = 1.0 / 27.211386245988
BOHR_PER_ANG = 1.8897261254578281

# Conversion: force (eV/A) -> gradient (Eh/Bohr)
GRAD_HA_BOHR_PER_FORCE_EV_ANG = -HARTREE_PER_EV / BOHR_PER_ANG
# Conversion: hessian (eV/A^2) -> (Eh/Bohr^2)
HESS_HA_BOHR2_PER_EV_ANG2 = HARTREE_PER_EV / (BOHR_PER_ANG * BOHR_PER_ANG)

FAIRCHEM_MODELS_FALLBACK = [
    "uma-s-1",
    "uma-s-1p1",
    "uma-m-1p1",
    "esen-md-direct-all-omol",
    "esen-sm-conserving-all-omol",
    "esen-sm-direct-all-omol",
    "esen-sm-conserving-all-oc25",
    "esen-md-direct-all-oc25",
    "esen-sm-filtered-odac25",
    "esen-sm-full-odac25",
]

FAIRCHEM_TASKS_FALLBACK = ["omol", "omat", "odac", "oc20", "oc25", "omc"]

ORB_MODELS_FALLBACK = [
    "orb-v3-conservative-omol",
    "orb-v3-conservative-20-omat",
    "orb-v3-conservative-inf-omat",
    "orb-v3-conservative-20-mpa",
    "orb-v3-conservative-inf-mpa",
]

ORB_DEPRECATED_MODEL_ALIASES = {
    "orb-v1": "orb-v2",
    "orb-d3-v1": "orb-d3-v2",
    "orb-d3-sm-v1": "orb-d3-sm-v2",
    "orb-d3-xs-v1": "orb-d3-xs-v2",
    "orb-v1-mptraj-only": "orb-mptraj-only-v2",
    "orb-mptraj-only-v1": "orb-mptraj-only-v2",
}


def _is_deprecated_orb_model(model_name):
    norm_dash = str(model_name).replace("_", "-").lower()
    return norm_dash in ORB_DEPRECATED_MODEL_ALIASES


def _is_conservative_orb_model(model_name):
    norm_dash = str(model_name).replace("_", "-").lower()
    return ("conservative" in norm_dash) and ("direct" not in norm_dash)


MACE_MP_ALIASES_FALLBACK = [
    "small",
    "medium",
    "large",
    "small-0b",
    "medium-0b",
    "small-0b2",
    "medium-0b2",
    "large-0b2",
    "medium-0b3",
    "medium-mpa-0",
    "small-omat-0",
    "medium-omat-0",
    "mace-matpes-pbe-0",
    "mace-matpes-r2scan-0",
    "mh-0",
    "mh-1",
]

AIMNET2_MODELS_FALLBACK = [
    "aimnet2",
    "aimnet2_b973c",
    "aimnet2_2025",
    "aimnet2nse",
    "aimnet2pd",
]


class BackendError(RuntimeError):
    """Raised for backend-specific runtime failures."""


def _is_hf_access_issue(exc):
    text = str(exc).lower()
    tags = (
        "gatedrepoerror",
        "forbidden",
        "401",
        "403",
        "cannot access gated repo",
        "huggingface",
        "hf_hub_download",
    )
    return any(tag in text for tag in tags)


def _with_uma_access_hint(prefix, exc):
    msg = "{}: {}".format(prefix, exc)
    if not _is_hf_access_issue(exc):
        return msg
    return (
        msg
        + "\nUMA model download failed due to Hugging Face access/auth."
        + "\nRun once: huggingface-cli login"
        + "\nIf the selected model repo is gated, request access on its Hugging Face page."
    )


def ev_to_ha(value_ev):
    return float(value_ev) * HARTREE_PER_EV


def forces_ev_ang_to_gradient_ha_bohr(forces_ev_ang):
    forces = np.asarray(forces_ev_ang, dtype=np.float64)
    return (forces.reshape(-1) * GRAD_HA_BOHR_PER_FORCE_EV_ANG).astype(np.float64)


def hessian_ev_ang2_to_ha_bohr2(hessian_ev_ang2):
    hess = np.asarray(hessian_ev_ang2, dtype=np.float64)
    return (hess * HESS_HA_BOHR2_PER_EV_ANG2).astype(np.float64)


def _as_square_hessian(hess_like, natoms):
    """Convert Hessian-like object to a (3N, 3N) float64 array in eV/A^2."""
    h = np.asarray(hess_like, dtype=np.float64)
    dof = int(natoms) * 3
    if h.ndim == 5:
        if h.shape[0] > 0:
            h = h[0]
    if h.ndim == 4:
        return h.reshape(dof, dof)
    if h.ndim == 2 and h.shape == (dof, dof):
        return h
    return h.reshape(dof, dof)


def _prepare_model_for_autograd_hessian(model_obj, torch_mod):
    state = {
        "was_training": bool(getattr(model_obj, "training", False)),
        "param_flags": [],
        "dropout_states": [],
    }

    if hasattr(model_obj, "parameters"):
        for param in model_obj.parameters():
            state["param_flags"].append((param, bool(param.requires_grad)))
            param.requires_grad_(False)

    if hasattr(model_obj, "train"):
        model_obj.train(True)

    dropout_types = []
    nn_mod = getattr(torch_mod, "nn", None)
    if nn_mod is not None:
        for name in (
            "Dropout",
            "Dropout1d",
            "Dropout2d",
            "Dropout3d",
            "AlphaDropout",
            "FeatureAlphaDropout",
        ):
            cls = getattr(nn_mod, name, None)
            if cls is not None:
                dropout_types.append(cls)

    if dropout_types and hasattr(model_obj, "modules"):
        dtypes = tuple(dropout_types)
        for module in model_obj.modules():
            if not isinstance(module, dtypes):
                continue
            old_p = getattr(module, "p", None)
            state["dropout_states"].append((module, bool(getattr(module, "training", False)), old_p))
            if old_p is not None:
                try:
                    module.p = 0.0
                except Exception:
                    pass
            module.train(False)

    return state


def _restore_model_after_autograd_hessian(model_obj, state):
    for module, was_training, old_p in state.get("dropout_states", []):
        if old_p is not None:
            try:
                module.p = old_p
            except Exception:
                pass
        module.train(was_training)

    if hasattr(model_obj, "train"):
        model_obj.train(state.get("was_training", False))

    for param, req_grad in state.get("param_flags", []):
        param.requires_grad_(req_grad)


def _numerical_hessian_from_forces(eval_energy_forces, coords_ang, step_ang):
    """Central-difference Hessian from force calls.

    Parameters
    ----------
    eval_energy_forces : callable
        Function(coords_ang) -> (energy_eV, forces_eV_per_ang)
    coords_ang : (N,3) ndarray
    step_ang : float

    Returns
    -------
    tuple
        (energy_eV, forces_eV_per_ang, hessian_eV_per_ang2)
    """
    coords0 = np.asarray(coords_ang, dtype=np.float64).reshape(-1, 3)
    nat = coords0.shape[0]
    dof = nat * 3

    e0, f0 = eval_energy_forces(coords0)
    f0 = np.asarray(f0, dtype=np.float64).reshape(-1)

    hess = np.zeros((dof, dof), dtype=np.float64)

    for k in range(dof):
        a = k // 3
        c = k % 3

        plus = coords0.copy()
        minus = coords0.copy()
        plus[a, c] += step_ang
        minus[a, c] -= step_ang

        _, f_plus = eval_energy_forces(plus)
        _, f_minus = eval_energy_forces(minus)

        g_plus = -np.asarray(f_plus, dtype=np.float64).reshape(-1)
        g_minus = -np.asarray(f_minus, dtype=np.float64).reshape(-1)

        hess[:, k] = (g_plus - g_minus) / (2.0 * step_ang)

    hess = 0.5 * (hess + hess.T)
    return float(e0), f0.reshape(nat, 3), hess


def get_available_uma_models():
    try:
        from fairchem.core.calculate.pretrained_mlip import available_models

        models = list(available_models)
        if models:
            return models
    except Exception:
        pass
    return list(FAIRCHEM_MODELS_FALLBACK)


def get_available_uma_tasks():
    try:
        from fairchem.core.units.mlip_unit.api.inference import UMATask

        vals = []
        for t in UMATask:
            val = getattr(t, "value", None)
            vals.append(str(val if val is not None else t))
        # preserve order and uniqueness
        seen = set()
        ordered = []
        for x in vals:
            if x not in seen:
                seen.add(x)
                ordered.append(x)
        if ordered:
            return ordered
    except Exception:
        pass
    return list(FAIRCHEM_TASKS_FALLBACK)


def get_available_orb_models():
    models = []
    try:
        from orb_models.forcefield import pretrained as orb_pretrained

        if hasattr(orb_pretrained, "ORB_PRETRAINED_MODELS"):
            models.extend(list(getattr(orb_pretrained, "ORB_PRETRAINED_MODELS").keys()))
    except Exception:
        pass
    if not models:
        models = list(ORB_MODELS_FALLBACK)

    out = []
    seen = set()
    for model in models:
        if _is_deprecated_orb_model(model):
            continue
        if not _is_conservative_orb_model(model):
            continue
        for cand in (model, model.replace("-", "_")):
            if _is_deprecated_orb_model(cand):
                continue
            if not _is_conservative_orb_model(cand):
                continue
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out


def get_available_mace_models():
    out = []
    mp_aliases = []
    try:
        from mace.calculators.foundations_models import mace_mp_urls

        mp_aliases = list(sorted(mace_mp_urls.keys()))
    except Exception:
        mp_aliases = list(MACE_MP_ALIASES_FALLBACK)

    out.extend(["mp:" + k for k in mp_aliases])
    out.extend(mp_aliases)

    out.extend(
        [
            "MACE-OMOL-0",
            "omol:extra_large",
            "extra_large",
            "off:small",
            "off:medium",
            "off:large",
            "off-small",
            "off-medium",
            "off-large",
            "anicc",
        ]
    )
    out.append("<local_model_path>")
    out.append("<https://...model>")

    # preserve order and uniqueness
    seen = set()
    uniq = []
    for item in out:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


def get_available_aimnet2_models():
    out = list(AIMNET2_MODELS_FALLBACK)
    out.append("<local_model_path>")
    out.append("<https://...model>")

    seen = set()
    uniq = []
    for item in out:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq


class _BackendBase(object):
    """Shared evaluator behavior."""

    def energy_forces(self, symbols, coords_ang, charge, multiplicity):
        raise NotImplementedError

    def analytical_hessian(self, symbols, coords_ang, charge, multiplicity):
        raise BackendError("Analytical Hessian is not available for this backend/model.")

    def evaluate(
        self,
        symbols,
        coords_ang,
        charge,
        multiplicity,
        need_forces,
        need_hessian,
        hessian_mode,
        hessian_step,
    ):
        coords_ang = np.asarray(coords_ang, dtype=np.float64).reshape(-1, 3)

        mode = str(hessian_mode or "Analytical").strip().lower()
        use_analytical = mode.startswith("ana")

        if need_hessian:
            if use_analytical:
                try:
                    e_ev, f_ev_ang = self.energy_forces(
                        symbols, coords_ang, charge, multiplicity
                    )
                    h_ev_ang2 = self.analytical_hessian(
                        symbols, coords_ang, charge, multiplicity
                    )
                    h_ev_ang2 = _as_square_hessian(h_ev_ang2, len(symbols))
                    return float(e_ev), np.asarray(f_ev_ang, dtype=np.float64), h_ev_ang2
                except Exception as exc:
                    raise BackendError(
                        "Analytical Hessian failed for {}: {}".format(
                            self.__class__.__name__,
                            exc,
                        )
                    )

            # Numerical mode
            e_ev, f_ev_ang, h_ev_ang2 = _numerical_hessian_from_forces(
                lambda x: self.energy_forces(symbols, x, charge, multiplicity),
                coords_ang,
                float(hessian_step),
            )
            return float(e_ev), np.asarray(f_ev_ang, dtype=np.float64), h_ev_ang2

        if need_forces:
            e_ev, f_ev_ang = self.energy_forces(symbols, coords_ang, charge, multiplicity)
            return float(e_ev), np.asarray(f_ev_ang, dtype=np.float64), None

        e_ev, _f = self.energy_forces(symbols, coords_ang, charge, multiplicity)
        return float(e_ev), None, None


class UMAEvaluator(_BackendBase):
    """UMA backend via fairchem."""

    def __init__(
        self,
        model,
        task,
        device,
        workers,
        workers_per_node=None,
        max_neigh=None,
        radius=None,
        r_edges=False,
        otf_graph=True,
    ):
        try:
            import torch
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            from fairchem.core.datasets import data_list_collater
            from fairchem.core.datasets.atomic_data import AtomicData
        except Exception as exc:
            raise BackendError(
                "UMA backend requires fairchem-core and torch. Install with: pip install \"orca-mlips[uma]\""
            ) from exc

        self._torch = torch
        if str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = str(device)
        self.model = str(model)
        self.task = str(task)
        self.workers = max(1, int(workers))
        self.workers_per_node = (
            None if workers_per_node is None else max(1, int(workers_per_node))
        )
        self.max_neigh = None if max_neigh is None else int(max_neigh)
        if self.max_neigh is not None and self.max_neigh <= 0:
            self.max_neigh = None
        self.radius = None if radius is None else float(radius)
        if self.radius is not None and self.radius <= 0:
            self.radius = None
        self.r_edges = bool(r_edges)
        self.otf_graph = bool(otf_graph)

        predictor_attempts = []
        predictor_attempts.append(
            {
                "device": self.device,
                "workers": self.workers,
                "workers_per_node": self.workers_per_node,
            }
        )
        predictor_attempts.append({"device": self.device, "workers": self.workers})
        predictor_attempts.append({"device": self.device})

        uniq_attempts = []
        seen = set()
        for kwargs in predictor_attempts:
            clean = {k: v for k, v in kwargs.items() if v is not None}
            key = tuple(sorted(clean.items()))
            if key in seen:
                continue
            seen.add(key)
            uniq_attempts.append(clean)

        last_exc = None
        self._predictor = None
        for kwargs in uniq_attempts:
            try:
                self._predictor = pretrained_mlip.get_predict_unit(self.model, **kwargs)
                break
            except Exception as exc:
                last_exc = exc
                continue
        if self._predictor is None:
            raise BackendError(
                _with_uma_access_hint(
                    "Failed to initialize UMA predictor with the requested worker settings",
                    last_exc,
                )
            )

        self._ase_calc = FAIRChemCalculator(self._predictor, task_name=self.task)
        self._AtomicData = AtomicData
        self._collater = data_list_collater
        self._has_torch_model = hasattr(self._predictor, "model")

    def energy_forces(self, symbols, coords_ang, charge, multiplicity):
        from ase import Atoms

        atoms = Atoms(symbols=symbols, positions=np.asarray(coords_ang, dtype=np.float64))
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = int(multiplicity)
        atoms.calc = self._ase_calc

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        return energy, forces

    def _make_batch(self, symbols, coords_ang, charge, multiplicity):
        from ase import Atoms

        atoms = Atoms(symbols=symbols, positions=np.asarray(coords_ang, dtype=np.float64))
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = int(multiplicity)

        backbone = None
        model = getattr(self._predictor, "model", None)
        if model is not None:
            model_inner = getattr(model, "module", model)
            backbone = getattr(model_inner, "backbone", None)

        max_neigh = getattr(backbone, "max_neighbors", None) if backbone is not None else None
        radius = getattr(backbone, "cutoff", 6.0) if backbone is not None else 6.0

        if self.max_neigh is not None:
            max_neigh = self.max_neigh
        if self.radius is not None:
            radius = self.radius

        data = self._AtomicData.from_ase(
            atoms,
            max_neigh=max_neigh,
            radius=radius,
            r_edges=self.r_edges,
        )
        data.dataset = self.task

        batch = self._collater([data], otf_graph=self.otf_graph)
        if hasattr(batch, "to"):
            batch = batch.to(self.device)
        return batch

    def analytical_hessian(self, symbols, coords_ang, charge, multiplicity):
        if not self._has_torch_model:
            raise BackendError("Current UMA predictor does not expose a torch model for analytical Hessian.")

        model = getattr(self._predictor, "model", None)
        if model is None:
            raise BackendError("Current UMA predictor model is not accessible.")

        batch = self._make_batch(symbols, coords_ang, charge, multiplicity)
        if not hasattr(batch, "pos"):
            raise BackendError("UMA batch object has no 'pos' attribute.")

        batch.pos.requires_grad_(True)
        model_state = _prepare_model_for_autograd_hessian(model, self._torch)

        try:

            def e_fn(flat):
                batch.pos = flat.view(-1, 3)
                pred = self._predictor.predict(batch)
                if "energy" not in pred:
                    raise BackendError("UMA predictor output does not contain 'energy'.")
                return pred["energy"].squeeze()

            hess = self._torch.autograd.functional.hessian(
                e_fn,
                batch.pos.view(-1),
                vectorize=False,
            )
            nat = len(symbols)
            hess = hess.view(nat * 3, nat * 3)
            return hess.detach().cpu().numpy().astype(np.float64)
        finally:
            _restore_model_after_autograd_hessian(model, model_state)
            if str(self.device).startswith("cuda"):
                try:
                    self._torch.cuda.empty_cache()
                except Exception:
                    pass


class OrbMolEvaluator(_BackendBase):
    """OrbMol backend via orb-models."""

    def __init__(self, model, device, precision, compile_model, loader_kwargs=None, calc_kwargs=None):
        try:
            import torch
            from orb_models.forcefield import pretrained as orb_pretrained
        except Exception as exc:
            raise BackendError(
                "OrbMol backend requires orb-models and torch. Install with: pip install \"orca-mlips[orb]\""
            ) from exc

        self._torch = torch
        self._pretrained = orb_pretrained

        if str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)

        self.model_name = str(model)
        self.precision = str(precision)
        self.compile_model = bool(compile_model)
        self.loader_kwargs = dict(loader_kwargs or {})
        self.calc_kwargs = dict(calc_kwargs or {})

        if not _is_conservative_orb_model(self.model_name):
            raise BackendError(
                "Only conservative Orb models are supported. Requested '{}'.".format(
                    self.model_name
                )
            )
        self._conservative = True

        self._loader = self._resolve_loader(self.model_name)
        self._model_obj, self._adapter = self._load_model()
        self._ase_calc = self._build_ase_calculator()

    def _resolve_loader(self, model_name):
        norm_dash = str(model_name).replace("_", "-").lower()
        if norm_dash in ORB_DEPRECATED_MODEL_ALIASES:
            model_name = ORB_DEPRECATED_MODEL_ALIASES[norm_dash]

        if not _is_conservative_orb_model(model_name):
            raise BackendError(
                "Only conservative Orb models are supported. Requested '{}'.".format(
                    model_name
                )
            )

        # 1) Prefer ORB_PRETRAINED_MODELS keys
        if hasattr(self._pretrained, "ORB_PRETRAINED_MODELS"):
            model_map = getattr(self._pretrained, "ORB_PRETRAINED_MODELS")
            cands = [
                model_name,
                model_name.replace("_", "-"),
                model_name.replace("-", "_"),
            ]
            cands.extend([x.lower() for x in cands])
            for cand in cands:
                if cand in model_map:
                    return model_map[cand]

            # Case-insensitive fallback for dictionary keys
            lower_map = {str(k).lower(): v for k, v in model_map.items()}
            for cand in cands:
                key = str(cand).lower()
                if key in lower_map:
                    return lower_map[key]

        # 2) Attribute style (underscore names)
        for cand in (model_name, model_name.replace("-", "_"), model_name.lower().replace("-", "_")):
            if hasattr(self._pretrained, cand):
                return getattr(self._pretrained, cand)

        raise BackendError(
            "Unknown Orb model '{}' . Use --list-models to see available options.".format(model_name)
        )

    def _load_model(self):
        # Handle API differences between orb-models releases.
        attempts = []
        for base in (
            {"device": self.device, "precision": self.precision, "compile": self.compile_model},
            {"device": self.device, "precision": self.precision},
            {"device": self.device},
            {},
        ):
            merged = dict(base)
            merged.update(self.loader_kwargs)
            attempts.append(merged)

        if self.loader_kwargs:
            attempts.append(dict(self.loader_kwargs))

        uniq_attempts = []
        seen = set()
        for kwargs in attempts:
            key = tuple(sorted(kwargs.items()))
            if key in seen:
                continue
            seen.add(key)
            uniq_attempts.append(kwargs)

        last_exc = None
        for kwargs in uniq_attempts:
            try:
                out = self._loader(**kwargs)
                if isinstance(out, tuple) and len(out) >= 2:
                    return out[0], out[1]
                return out, None
            except Exception as exc:
                last_exc = exc
                continue

        raise BackendError("Failed to load Orb model '{}': {}".format(self.model_name, last_exc))

    def _build_ase_calculator(self):
        calc_kwargs = dict(self.calc_kwargs)

        # New API: orb_models.forcefield.inference.calculator.ORBCalculator
        try:
            from orb_models.forcefield.inference.calculator import ORBCalculator

            if self._adapter is not None:
                try:
                    return ORBCalculator(self._model_obj, self._adapter, device=self.device, **calc_kwargs)
                except TypeError:
                    pass
                try:
                    return ORBCalculator(self._model_obj, self._adapter, **calc_kwargs)
                except TypeError:
                    pass

            try:
                return ORBCalculator(self._model_obj, device=self.device, **calc_kwargs)
            except TypeError:
                return ORBCalculator(self._model_obj, **calc_kwargs)
        except Exception:
            pass

        # Legacy API fallback
        try:
            from orb_models.forcefield.calculator import ORBCalculator

            if self._adapter is not None:
                try:
                    return ORBCalculator(self._model_obj, self._adapter, device=self.device, **calc_kwargs)
                except TypeError:
                    pass
                try:
                    return ORBCalculator(self._model_obj, self._adapter, **calc_kwargs)
                except TypeError:
                    pass
            try:
                return ORBCalculator(self._model_obj, device=self.device, **calc_kwargs)
            except TypeError:
                return ORBCalculator(self._model_obj, **calc_kwargs)
        except Exception as exc:
            raise BackendError("Failed to build ORBCalculator: {}".format(exc))

    def energy_forces(self, symbols, coords_ang, charge, multiplicity):
        from ase import Atoms

        atoms = Atoms(symbols=symbols, positions=np.asarray(coords_ang, dtype=np.float64))
        atoms.info["charge"] = float(charge)
        atoms.info["spin"] = float(multiplicity)
        atoms.calc = self._ase_calc

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        return energy, forces

    def _energy_key(self, out_dict):
        for key in ("energy", "free_energy", "total_energy", "E"):
            if key in out_dict:
                return key
        for key in out_dict.keys():
            if str(key).lower().startswith("energy"):
                return key
        raise BackendError("Could not find an energy key in Orb model output.")

    def analytical_hessian(self, symbols, coords_ang, charge, multiplicity):
        if not self._conservative:
            raise BackendError(
                "Analytical Hessian is typically meaningful only for conservative Orb models."
            )

        # New API path: model + adapter (autograd Hessian)
        if self._adapter is not None and hasattr(self._model_obj, "predict"):
            from ase import Atoms

            atoms = Atoms(symbols=symbols, positions=np.asarray(coords_ang, dtype=np.float64))
            atoms.info["charge"] = float(charge)
            atoms.info["spin"] = float(multiplicity)

            graph = self._adapter.from_ase_atoms(atoms=atoms, device=self.device)
            if not hasattr(graph, "node_features"):
                raise BackendError("Unexpected Orb graph format: missing node_features.")

            node_features = graph.node_features
            if "positions" not in node_features:
                raise BackendError("Unexpected Orb graph format: missing positions.")

            pos0 = node_features["positions"]
            flat0 = (
                pos0.reshape(-1)
                .detach()
                .clone()
                .to(self.device)
                .requires_grad_(True)
            )
            model_state = _prepare_model_for_autograd_hessian(self._model_obj, self._torch)

            try:

                def e_fn(flat):
                    node_features["positions"] = flat.view(-1, 3)
                    out = self._model_obj.predict(graph)
                    ek = self._energy_key(out)
                    return out[ek].squeeze()

                hess = self._torch.autograd.functional.hessian(e_fn, flat0, vectorize=False)
                nat = len(symbols)
                hess = hess.view(nat * 3, nat * 3)
                return hess.detach().cpu().numpy().astype(np.float64)
            finally:
                _restore_model_after_autograd_hessian(self._model_obj, model_state)
                if str(self.device).startswith("cuda"):
                    try:
                        self._torch.cuda.empty_cache()
                    except Exception:
                        pass

        # Legacy API fallback path
        try:
            from orb_models.forcefield import atomic_system
            from ase import Atoms

            atoms = Atoms(symbols=symbols, positions=np.asarray(coords_ang, dtype=np.float64))
            atoms.info["charge"] = float(charge)
            atoms.info["spin"] = float(multiplicity)

            graph = atomic_system.ase_atoms_to_atom_graphs(
                atoms,
                getattr(self._model_obj, "system_config", None),
                device=self.device,
            )
            pos_attr = None
            for cand in ("positions", "pos", "coords", "xyz"):
                if hasattr(graph, cand):
                    pos_attr = cand
                    break
            if pos_attr is None:
                raise BackendError("Could not locate position tensor for legacy Orb API.")

            pos0 = getattr(graph, pos_attr)
            flat0 = (
                self._torch.as_tensor(pos0, device=self.device)
                .reshape(-1)
                .detach()
                .clone()
                .requires_grad_(True)
            )
            model_state = _prepare_model_for_autograd_hessian(self._model_obj, self._torch)

            try:

                def e_fn(flat):
                    setattr(graph, pos_attr, flat.view(-1, 3))
                    out = self._model_obj.predict(graph, split=False)
                    ek = self._energy_key(out)
                    return out[ek].squeeze()

                hess = self._torch.autograd.functional.hessian(e_fn, flat0, vectorize=False)
                nat = len(symbols)
                hess = hess.view(nat * 3, nat * 3)
                return hess.detach().cpu().numpy().astype(np.float64)
            finally:
                _restore_model_after_autograd_hessian(self._model_obj, model_state)
                if str(self.device).startswith("cuda"):
                    try:
                        self._torch.cuda.empty_cache()
                    except Exception:
                        pass
        except Exception as exc:
            raise BackendError("Analytical Hessian failed for OrbMol: {}".format(exc))


class MACEEvaluator(_BackendBase):
    """MACE backend via mace.calculators."""

    def __init__(self, model, device, default_dtype, calc_kwargs=None):
        try:
            import torch
        except Exception as exc:
            raise BackendError(
                "MACE backend requires torch and mace-torch. Install with: pip install \"orca-mlips[mace]\""
            ) from exc

        self._torch = torch
        if str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)

        self.model_spec = str(model)
        self.default_dtype = str(default_dtype)
        self.calc_kwargs = dict(calc_kwargs or {})

        self._calc = self._build_calc(self.model_spec)

    def _download_to_tmp(self, url):
        target = os.path.join(
            tempfile.gettempdir(),
            os.path.basename(str(url)).split("?")[0] or "mace.model",
        )
        if not os.path.exists(target):
            urllib.request.urlretrieve(str(url), target)
        return target

    def _build_calc(self, model_spec):
        try:
            from mace.calculators import mace_anicc, mace_mp, mace_off, mace_omol
        except Exception as exc:
            raise BackendError(
                "Could not import mace.calculators helpers (mace_mp/mace_off/mace_omol/mace_anicc)."
            ) from exc

        mp_aliases = []
        try:
            from mace.calculators.foundations_models import mace_mp_urls

            mp_aliases = list(mace_mp_urls.keys())
        except Exception:
            mp_aliases = list(MACE_MP_ALIASES_FALLBACK)

        spec = str(model_spec).strip()
        spec_l = spec.lower()
        mp_alias_lookup = {str(x).lower(): x for x in mp_aliases}

        calc_kwargs = dict(self.calc_kwargs)

        def _mk_mace_calculator_from_path(path_or_url):
            from mace.calculators.mace import MACECalculator

            model_path = str(path_or_url)
            if model_path.startswith("http://") or model_path.startswith("https://"):
                model_path = self._download_to_tmp(model_path)
            return MACECalculator(
                model_paths=model_path,
                device=self.device,
                default_dtype=self.default_dtype,
                **calc_kwargs
            )

        def _safe_mace_anicc(kwargs):
            try:
                return mace_anicc(**kwargs)
            except Exception as exc:
                msg = str(exc)
                if ("no nvidia driver" in msg.lower()) or ("cuda" in msg.lower() and "backend" in msg.lower()):
                    raise BackendError(
                        "MACE ANICC could not be loaded on this host. "
                        "The upstream ANICC checkpoint appears to require CUDA-backed operators "
                        "in this environment. Try a CUDA-enabled node/driver, or use another MACE model."
                    ) from exc
                raise

        # Prefix forms
        if spec_l.startswith("mp:"):
            alias = spec.split(":", 1)[1].strip() or None
            if alias is not None:
                alias = mp_alias_lookup.get(str(alias).lower(), alias)
            return mace_mp(model=alias, device=self.device, default_dtype=self.default_dtype, **calc_kwargs)

        if spec_l.startswith("off:"):
            alias = spec.split(":", 1)[1].strip() or None
            if alias is not None:
                alias = str(alias).lower()
            return mace_off(model=alias, device=self.device, default_dtype=self.default_dtype, **calc_kwargs)

        if spec_l.startswith("omol:"):
            alias = spec.split(":", 1)[1].strip() or None
            if alias == "":
                alias = None
            if alias is not None:
                alias_l = str(alias).lower()
                if alias_l in ("mace-omol-0", "mace_omol_0", "maceomol0"):
                    alias = "extra_large"
                else:
                    alias = alias_l
            return mace_omol(model=alias, device=self.device, default_dtype=self.default_dtype, **calc_kwargs)

        if spec_l.startswith("anicc"):
            path = None
            if ":" in spec:
                path = spec.split(":", 1)[1].strip() or None
            kwargs = {"device": self.device}
            if path:
                kwargs["model_path"] = path
            kwargs.update(calc_kwargs)
            return _safe_mace_anicc(kwargs)

        # Alias forms
        if spec_l in mp_alias_lookup:
            return mace_mp(
                model=mp_alias_lookup[spec_l],
                device=self.device,
                default_dtype=self.default_dtype,
                **calc_kwargs
            )

        if spec_l in ("off-small", "off-medium", "off-large"):
            alias = spec_l.split("-", 1)[1]
            return mace_off(model=alias, device=self.device, default_dtype=self.default_dtype, **calc_kwargs)

        if spec_l in (
            "omol-extra_large",
            "extra_large",
            "mace-omol-0",
            "mace_omol_0",
            "maceomol0",
        ):
            return mace_omol(model="extra_large", device=self.device, default_dtype=self.default_dtype, **calc_kwargs)

        if spec_l in ("anicc", "ani", "ani500k"):
            kwargs = {"device": self.device}
            kwargs.update(calc_kwargs)
            return _safe_mace_anicc(kwargs)

        # Local file / URL
        if os.path.exists(spec) or spec.startswith("http://") or spec.startswith("https://"):
            return _mk_mace_calculator_from_path(spec)

        raise BackendError(
            "Unknown MACE model spec '{}'. Use --list-models for aliases, or pass a local path/URL.".format(
                spec
            )
        )

    def energy_forces(self, symbols, coords_ang, charge, multiplicity):
        from ase import Atoms

        atoms = Atoms(symbols=symbols, positions=np.asarray(coords_ang, dtype=np.float64))
        # Some backends may inspect these fields; harmless if ignored.
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = int(multiplicity)
        atoms.calc = self._calc

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        return energy, forces

    def analytical_hessian(self, symbols, coords_ang, charge, multiplicity):
        if not hasattr(self._calc, "get_hessian"):
            raise BackendError(
                "Installed MACE calculator does not expose get_hessian()."
            )

        from ase import Atoms

        atoms = Atoms(symbols=symbols, positions=np.asarray(coords_ang, dtype=np.float64))
        atoms.info["charge"] = int(charge)
        atoms.info["spin"] = int(multiplicity)
        hess = self._calc.get_hessian(atoms=atoms)
        return _as_square_hessian(hess, len(symbols))


class AIMNet2Evaluator(_BackendBase):
    """AIMNet2 backend via aimnet2calc / aimnetcentral."""

    def __init__(self, model, device, calc_kwargs=None):
        try:
            import torch
        except Exception as exc:
            raise BackendError(
                "AIMNet2 backend requires torch and an AIMNet2 package. "
                "Install with: pip install 'orca-mlips[aimnet2]'"
            ) from exc

        self._torch = torch
        if str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)
        self.model = str(model)
        self.calc_kwargs = dict(calc_kwargs or {})

        self._calculator = self._load_calculator(self.model)

    @staticmethod
    def _to_scalar(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "item"):
            return float(value.item())
        return float(np.asarray(value).reshape(-1)[0])

    @staticmethod
    def _extract_array(value, force_2d):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if force_2d:
            if arr.ndim == 1:
                return arr.reshape(-1, 3)
            if arr.ndim > 2:
                return arr.reshape(-1, 3)
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _pick_first_available(mapping, names):
        for name in names:
            if name in mapping:
                return mapping[name]
        lower = {str(k).lower(): v for k, v in mapping.items()}
        for name in names:
            val = lower.get(str(name).lower())
            if val is not None:
                return val
        return None

    def _load_calculator(self, model_name):
        candidates = []
        try:
            from aimnet2calc import AIMNet2Calculator

            candidates.append((AIMNet2Calculator, "aimnet2calc"))
        except Exception as exc:
            last_exc = exc
            try:
                from aimnetcentral import AIMNet2Calculator

                candidates.append((AIMNet2Calculator, "aimnetcentral"))
            except Exception as exc2:
                raise BackendError(
                    "AIMNet2 backend requires `aimnet2calc` or `aimnetcentral`. "
                    "Install with: pip install 'orca-mlips[aimnet2]'"
                ) from exc2
        else:
            last_exc = None

        init_kwargs_base = [
            {"device": self.device, **self.calc_kwargs},
            {"device": self.device},
            dict(self.calc_kwargs),
            {},
        ]

        for calc_factory, source in candidates:
            for init_kwargs in init_kwargs_base:
                try:
                    return calc_factory(model_name, **init_kwargs)
                except Exception as exc:
                    last_exc = exc
                    try:
                        return calc_factory(model=str(model_name), **init_kwargs)
                    except Exception as exc2:
                        last_exc = exc2
                        try:
                            return calc_factory(str(model_name))
                        except Exception as exc3:
                            last_exc = exc3
                            if init_kwargs:
                                filtered = {}
                                for key in init_kwargs:
                                    if key == "device":
                                        filtered[key] = init_kwargs[key]
                                if filtered:
                                    try:
                                        return calc_factory(model_name, **filtered)
                                    except Exception as exc4:
                                        last_exc = exc4
                                        continue
                                    else:
                                        continue
                            continue
        raise BackendError(
            "Failed to initialize AIMNet2 model '{}' from {}.".format(model_name, candidates[0][1])
        ) from last_exc

    def _call(self, symbols, coords_ang, charge, multiplicity, with_hessian):
        from ase import Atoms

        atoms = Atoms(symbols=symbols, positions=np.asarray(coords_ang, dtype=np.float64))
        numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)

        base = {
            "coord": np.asarray(coords_ang, dtype=np.float32)[None, :, :],
            "numbers": numbers[None, :],
            "charge": np.asarray([float(charge)], dtype=np.float32),
            "mult": np.asarray([float(multiplicity)], dtype=np.float32),
        }

        args = {
            "forces": True,
            "hessian": bool(with_hessian),
        }

        try:
            out = self._calculator(base, **args)
        except Exception:
            fallback = {
                "coords": np.asarray(coords_ang, dtype=np.float32)[None, :, :],
                "numbers": numbers[None, :],
                "charge": np.asarray([float(charge)], dtype=np.float32),
                "mult": np.asarray([float(multiplicity)], dtype=np.float32),
            }
            out = self._calculator(fallback, **args)

        if isinstance(out, tuple):
            out = list(out)

        if isinstance(out, (list, tuple)):
            if len(out) < 2:
                raise BackendError(
                    "Unexpected AIMNet2 output tuple length {}: {}".format(len(out), type(out))
                )
            energy = self._to_scalar(out[0])
            forces = self._extract_array(out[1], force_2d=True)
            hess = np.asarray(out[2], dtype=np.float64) if with_hessian and len(out) > 2 else None
            return energy, forces, hess

        if not isinstance(out, dict):
            raise BackendError("Unexpected AIMNet2 output type: {}".format(type(out)))

        energy = self._pick_first_available(
            out,
            ("energy", "E", "e", "total_energy", "free_energy"),
        )
        if energy is None:
            raise BackendError(
                "AIMNet2 output missing energy key. Keys: {}".format(sorted(out.keys()))
            )
        energy = self._to_scalar(energy)

        forces = self._pick_first_available(out, ("forces", "force", "gradient"))
        if forces is None:
            raise BackendError("AIMNet2 output missing forces key. Keys: {}".format(sorted(out.keys())))
        forces = self._extract_array(forces, force_2d=True)

        hess = self._pick_first_available(
            out,
            ("hessian", "Hessian", "hess", "hessians"),
        )
        if hess is not None:
            hess = np.asarray(hess, dtype=np.float64)

        if with_hessian and hess is None:
            raise BackendError(
                "AIMNet2 output missing hessian key for analytical request. Keys: {}".format(
                    sorted(out.keys())
                )
            )
        return energy, forces, hess

    def energy_forces(self, symbols, coords_ang, charge, multiplicity):
        energy, forces, _ = self._call(symbols, coords_ang, charge, multiplicity, with_hessian=False)
        return energy, np.asarray(forces, dtype=np.float64)

    def analytical_hessian(self, symbols, coords_ang, charge, multiplicity):
        _e, _f, h = self._call(symbols, coords_ang, charge, multiplicity, with_hessian=True)
        if h is None:
            raise BackendError("AIMNet2 did not return analytical Hessian.")
        return _as_square_hessian(h, len(symbols))
