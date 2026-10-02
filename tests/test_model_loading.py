import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch


@pytest.fixture
def modules():
    repo = Path(__file__).resolve().parents[1]
    package = repo.name.replace("-", "_") + "_test_plugins"
    spec = importlib.util.spec_from_file_location(
        package, repo / "plugins" / "__init__.py",
        submodule_search_locations=[str(repo / "plugins")],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    backend = importlib.import_module(package + ".mlip_backends")
    return repo, package, backend


@pytest.fixture
def weights(tmp_path):
    filename = tmp_path / "local weights.pt"
    filename.write_bytes(b"offline fixture")
    return filename


@pytest.fixture
def fairchem_stub(monkeypatch):
    import fairchem.core
    import fairchem.core.units.mlip_unit as unit
    from fairchem.core import pretrained_mlip

    calls = []
    def predictor(path, **kwargs):
        calls.append((str(path), kwargs))
        return SimpleNamespace(
            model=torch.nn.Linear(1, 1),
            move_to_device=lambda: None,
            predict=lambda batch: {
                "energy": batch.pos.square().sum().reshape(1),
                "forces": -2 * batch.pos,
            },
        )
    monkeypatch.setattr(unit, "load_predict_unit", predictor)
    def download(*args, **kwargs):
        raise AssertionError("Local weights must not access pretrained downloads")
    monkeypatch.setattr(pretrained_mlip, "get_predict_unit", download)
    monkeypatch.setattr(pretrained_mlip, "get_reference_energies", download)
    monkeypatch.setattr(pretrained_mlip, "pretrained_checkpoint_path_from_name", download)
    monkeypatch.setattr(fairchem.core, "FAIRChemCalculator", lambda *a, **k: SimpleNamespace())
    return calls


def test_uma_offline_energy_and_hessian(modules, weights, fairchem_stub):
    _, _, backends = modules
    calc = backends.UMAEvaluator(
        model="uma-s-1p1", task="omol", device="cpu", workers=1,
        weights_file=str(weights),
    )
    coords = np.array([[0., 0., 0.], [1., 0., 0.]])
    batch = calc._make_batch(["H", "H"], coords, 0, 1)
    result = calc._predictor.predict(batch)
    assert result["energy"].item() == 1
    np.testing.assert_array_equal(result["forces"].numpy(), -2 * coords)
    hessian = calc.analytical_hessian(["H", "H"], coords, 0, 1)
    np.testing.assert_array_equal(hessian, 2 * np.eye(6))
    assert all(path == str(weights) for path, _ in fairchem_stub)
    settings = fairchem_stub[-1][1]["inference_settings"]
    assert settings.compile is False
    assert settings.activation_checkpointing is False


def test_missing_uma_weights_fail_before_download(modules, fairchem_stub, tmp_path):
    _, _, backends = modules
    with pytest.raises(backends.BackendError, match="Weights file does not exist"):
        backends.UMAEvaluator("uma-s-1p1", "omol", "cpu", 1,
                             weights_file=str(tmp_path / "missing.pt"))
    assert fairchem_stub == []


@pytest.mark.parametrize("charge,spin", [(0, 1), (-1, 2)])
def test_uma_batch_preserves_charge_and_spin(modules, fairchem_stub, weights, charge, spin):
    _, _, backends = modules
    calc = backends.UMAEvaluator("uma-s-1p1", "omol", "cpu", 1, weights_file=str(weights))
    batch = calc._make_batch(["H", "H"], np.array([[0., 0., 0.], [1., 0., 0.]]), charge, spin)
    assert batch.charge.item() == charge
    assert batch.spin.item() == spin


@pytest.mark.parametrize("model", ["orb-v3-conservative-omol", "orbmol-v2"])
def test_orb_offline_loader(modules, weights, monkeypatch, model):
    from orb_models.forcefield import pretrained
    _, _, backends = modules
    calls = []
    def loader(*, weights_path, device, precision, compile):
        calls.append((weights_path, device, precision, compile))
        assert Path(weights_path).read_bytes() == b"offline fixture"
        return SimpleNamespace(), SimpleNamespace()
    monkeypatch.setattr(pretrained, "ORB_PRETRAINED_MODELS", {model: loader})
    monkeypatch.setattr(backends.OrbMolEvaluator, "_build_ase_calculator",
                        lambda self: SimpleNamespace())
    calc = backends.OrbMolEvaluator(model, "cpu", "float64", False,
                                    weights_file=str(weights))
    assert calls == [(str(weights), "cpu", "float64", False)]


def test_orb_weights_conflict(modules, weights, monkeypatch, tmp_path):
    _, _, backends = modules
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    with pytest.raises(backends.BackendError, match="conflicts"):
        backends.OrbMolEvaluator(
            "orb-v3-conservative-omol", "cpu", "float64", False,
            weights_file=str(weights), loader_kwargs={"weights_path": str(other)},
        )


@pytest.mark.parametrize("flag", [None, "-w", "--weights-file"])
@pytest.mark.parametrize("backend", ["uma", "orb", "mace", "aimnet2"])
def test_cli_routes_weights(modules, weights, monkeypatch, flag, backend):
    repo, package, backends = modules
    captured = []
    def fake(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(evaluate=lambda *a, **k: (1., np.zeros((2, 3)), None))
    name = {"uma": "UMAEvaluator", "orb": "OrbMolEvaluator",
            "mace": "MACEEvaluator", "aimnet2": "AIMNet2Evaluator"}[backend]
    if repo.name == "amber-mlips":
        cli = importlib.import_module(package + ".nonmpi_qc_shim")
        monkeypatch.setattr(cli, name, fake)
        keywords = "" if flag is None else flag + ' "' + str(weights) + '"'
        args = cli._parse_keywords(backend, keywords)
        cli._create_evaluator(args)
    else:
        program = "orca" if repo.name == "orca-mlips" else "g16"
        cli = importlib.import_module(package + ".cli_" + program)
        monkeypatch.setattr(cli, name, fake)
        if program == "orca":
            xyz = weights.parent / "input.xyz"
            xyz.write_text("2\nHydrogen\nH 0 0 0\nH 1 0 0\n")
            input_file = weights.parent / "input.extinp"
            input_file.write_text("input.xyz\n0\n1\n1\n1\n")
            argv = ["--no-server", str(input_file)]
        else:
            input_file = weights.parent / "input"
            input_file.write_text("2 1 0 1\n1 0 0 0 0\n1 1 0 0 0\n")
            argv = ["--no-server", "R", str(input_file)]
            argv += [str(weights.parent / name) for name in ("output", "message", "fchk", "mat")]
        if flag is not None:
            argv = [flag, str(weights), *argv]
        cli.run_backend("orbmol" if backend == "orb" else backend, backend, argv)
    assert len(captured) == 1
    if backend in ("uma", "orb"):
        assert captured[0]["weights_file"] == (None if flag is None else str(weights))
        default_model = "uma-s-1p1" if backend == "uma" else "orb_v3_conservative_omol"
        if backend == "orb" and repo.name == "amber-mlips":
            default_model = "orb-v3-conservative-omol"
        assert captured[0]["model"] == default_model
    else:
        default_model = {"mace": "MACE-OMOL-0", "aimnet2": "aimnet2"}[backend]
        assert captured[0]["model"] == (default_model if flag is None else str(weights))


def test_weight_files_separate_servers(modules, weights):
    repo, package, _ = modules
    if repo.name == "amber-mlips":
        pytest.skip("AMBER starts a dedicated server for each invocation.")
    server = importlib.import_module(package + ".mlip_server")
    args = SimpleNamespace(model="uma-s-1p1", weights_file=str(weights),
                           device="cpu", task="omol")
    first = server.auto_server_socket(args, parent_pid=12345)
    args.weights_file = str(weights.parent / "second.pt")
    assert server.auto_server_socket(args, parent_pid=12345) != first
