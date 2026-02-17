#!/usr/bin/env python3
"""Common MLIP backend utilities for ORCA/Gaussian external plugins.

This module provides three backend classes:
- UMAEvaluator
- OrbMolEvaluator
- MACEEvaluator

All backends expose the same high-level method:
    evaluate(..., need_hessian, hessian_mode, hessian_step, strict_hessian)

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
    "orb-v3-direct-omol",
    "orb-v3-conservative-20-omat",
    "orb-v3-conservative-inf-omat",
    "orb-v3-direct-20-omat",
    "orb-v3-direct-inf-omat",
    "orb-v3-conservative-20-mpa",
    "orb-v3-conservative-inf-mpa",
    "orb-v3-direct-20-mpa",
    "orb-v3-direct-inf-mpa",
    "separate-d3-3layer",
    "separate-d3-5layer",
    "separate-d4-3layer",
    "separate-d4-5layer",
    "orb-v2",
    "orb-d3-v2",
    "orb-d3-sm-v2",
    "orb-d3-xs-v2",
    "orb-mptraj-only-v2",
    "orb-v1",
    "orb-d3-v1",
    "orb-d3-sm-v1",
    "orb-d3-xs-v1",
    "orb-v1-mptraj-only",
]

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


class BackendError(RuntimeError):
    """Raised for backend-specific runtime failures."""


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
    if h.ndim == 4:
        return h.reshape(dof, dof)
    if h.ndim == 2 and h.shape == (dof, dof):
        return h
    return h.reshape(dof, dof)


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
        for cand in (model, model.replace("-", "_")):
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
        strict_hessian,
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
                    if strict_hessian:
                        raise
                    # Fallback to numerical Hessian
                    e_ev, f_ev_ang, h_ev_ang2 = _numerical_hessian_from_forces(
                        lambda x: self.energy_forces(symbols, x, charge, multiplicity),
                        coords_ang,
                        float(hessian_step),
                    )
                    return float(e_ev), np.asarray(f_ev_ang, dtype=np.float64), h_ev_ang2

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

    def __init__(self, model, task, device, workers):
        try:
            import torch
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            from fairchem.core.datasets import data_list_collater
            from fairchem.core.datasets.atomic_data import AtomicData
        except Exception as exc:
            raise BackendError(
                "UMA backend requires fairchem-core, torch, and ase."
            ) from exc

        self._torch = torch
        if str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = str(device)
        self.model = str(model)
        self.task = str(task)
        self.workers = int(workers)

        self._predictor = pretrained_mlip.get_predict_unit(
            self.model,
            device=self.device,
            workers=max(1, self.workers),
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

        data = self._AtomicData.from_ase(
            atoms,
            max_neigh=max_neigh,
            radius=radius,
            r_edges=False,
        )
        data.dataset = self.task

        batch = self._collater([data], otf_graph=True)
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

        pflags = []
        for p in model.parameters():
            pflags.append(bool(p.requires_grad))
            p.requires_grad_(False)

        was_training = bool(getattr(model, "training", False))
        model.train(True)

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
            model.train(was_training)
            for p, flag in zip(model.parameters(), pflags):
                p.requires_grad_(flag)
            if str(self.device).startswith("cuda"):
                try:
                    self._torch.cuda.empty_cache()
                except Exception:
                    pass


class OrbMolEvaluator(_BackendBase):
    """OrbMol backend via orb-models."""

    def __init__(self, model, device, precision, compile_model):
        try:
            import torch
            from orb_models.forcefield import pretrained as orb_pretrained
        except Exception as exc:
            raise BackendError("OrbMol backend requires orb-models and torch.") from exc

        self._torch = torch
        self._pretrained = orb_pretrained

        if str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)

        self.model_name = str(model)
        self.precision = str(precision)
        self.compile_model = bool(compile_model)

        self._loader = self._resolve_loader(self.model_name)
        self._model_obj, self._adapter = self._load_model()
        self._ase_calc = self._build_ase_calculator()

        lname = self.model_name.lower()
        self._conservative = "conservative" in lname and "direct" not in lname

    def _resolve_loader(self, model_name):
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
        attempts = [
            {"device": self.device, "precision": self.precision, "compile": self.compile_model},
            {"device": self.device, "precision": self.precision},
            {"device": self.device},
            {},
        ]
        last_exc = None
        for kwargs in attempts:
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
        # New API: orb_models.forcefield.inference.calculator.ORBCalculator
        try:
            from orb_models.forcefield.inference.calculator import ORBCalculator

            if self._adapter is not None:
                try:
                    return ORBCalculator(self._model_obj, self._adapter, device=self.device)
                except TypeError:
                    pass

            try:
                return ORBCalculator(self._model_obj, device=self.device)
            except TypeError:
                return ORBCalculator(self._model_obj)
        except Exception:
            pass

        # Legacy API fallback
        try:
            from orb_models.forcefield.calculator import ORBCalculator

            if self._adapter is not None:
                try:
                    return ORBCalculator(self._model_obj, self._adapter, device=self.device)
                except TypeError:
                    pass
            try:
                return ORBCalculator(self._model_obj, device=self.device)
            except TypeError:
                return ORBCalculator(self._model_obj)
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

        # If ASE calculator exposes Hessian directly, prefer that.
        if hasattr(self._ase_calc, "get_hessian"):
            from ase import Atoms

            atoms = Atoms(symbols=symbols, positions=np.asarray(coords_ang, dtype=np.float64))
            atoms.info["charge"] = float(charge)
            atoms.info["spin"] = float(multiplicity)
            hess = self._ase_calc.get_hessian(atoms=atoms)
            return _as_square_hessian(hess, len(symbols))

        # New API path: model + adapter
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

            pflags = []
            if hasattr(self._model_obj, "parameters"):
                for p in self._model_obj.parameters():
                    pflags.append(bool(p.requires_grad))
                    p.requires_grad_(False)

            was_training = bool(getattr(self._model_obj, "training", False))
            if hasattr(self._model_obj, "train"):
                self._model_obj.train(True)

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
                if hasattr(self._model_obj, "train"):
                    self._model_obj.train(was_training)
                if hasattr(self._model_obj, "parameters"):
                    for p, flag in zip(self._model_obj.parameters(), pflags):
                        p.requires_grad_(flag)
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

            def e_fn(flat):
                setattr(graph, pos_attr, flat.view(-1, 3))
                out = self._model_obj.predict(graph, split=False)
                ek = self._energy_key(out)
                return out[ek].squeeze()

            hess = self._torch.autograd.functional.hessian(e_fn, flat0, vectorize=False)
            nat = len(symbols)
            hess = hess.view(nat * 3, nat * 3)
            return hess.detach().cpu().numpy().astype(np.float64)
        except Exception as exc:
            raise BackendError("Analytical Hessian failed for OrbMol: {}".format(exc))


class MACEEvaluator(_BackendBase):
    """MACE backend via mace.calculators."""

    def __init__(self, model, device, default_dtype):
        try:
            import torch
        except Exception as exc:
            raise BackendError("MACE backend requires torch.") from exc

        self._torch = torch
        if str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)

        self.model_spec = str(model)
        self.default_dtype = str(default_dtype)

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

        def _mk_mace_calculator_from_path(path_or_url):
            from mace.calculators.mace import MACECalculator

            model_path = str(path_or_url)
            if model_path.startswith("http://") or model_path.startswith("https://"):
                model_path = self._download_to_tmp(model_path)
            return MACECalculator(
                model_paths=model_path,
                device=self.device,
                default_dtype=self.default_dtype,
            )

        # Prefix forms
        if spec_l.startswith("mp:"):
            alias = spec.split(":", 1)[1].strip() or None
            if alias is not None:
                alias = mp_alias_lookup.get(str(alias).lower(), alias)
            return mace_mp(model=alias, device=self.device, default_dtype=self.default_dtype)

        if spec_l.startswith("off:"):
            alias = spec.split(":", 1)[1].strip() or None
            if alias is not None:
                alias = str(alias).lower()
            return mace_off(model=alias, device=self.device, default_dtype=self.default_dtype)

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
            return mace_omol(model=alias, device=self.device, default_dtype=self.default_dtype)

        if spec_l.startswith("anicc"):
            path = None
            if ":" in spec:
                path = spec.split(":", 1)[1].strip() or None
            kwargs = {"device": self.device}
            if path:
                kwargs["model_path"] = path
            return mace_anicc(**kwargs)

        # Alias forms
        if spec_l in mp_alias_lookup:
            return mace_mp(
                model=mp_alias_lookup[spec_l],
                device=self.device,
                default_dtype=self.default_dtype,
            )

        if spec_l in ("off-small", "off-medium", "off-large"):
            alias = spec_l.split("-", 1)[1]
            return mace_off(model=alias, device=self.device, default_dtype=self.default_dtype)

        if spec_l in (
            "omol-extra_large",
            "extra_large",
            "mace-omol-0",
            "mace_omol_0",
            "maceomol0",
        ):
            return mace_omol(model="extra_large", device=self.device, default_dtype=self.default_dtype)

        if spec_l in ("anicc", "ani", "ani500k"):
            return mace_anicc(device=self.device)

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
