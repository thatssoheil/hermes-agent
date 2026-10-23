"""A sealed payload's recorded selection is re-resolved before activation (#142998).

A packaged updater replaces the payload in place, so the path-derived install key and the
``facts.json`` that names a generation survive the swap, and ``selected_venv()`` keeps
answering with the generation resolved against the previous release's lock. These tests
drive the boot seam -- ``venv_sync.prepare_launch`` (which runs before
``activate_dependencies`` in ``hermes_bootstrap``) and then the real selection readers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hermes_cli import venv_sync

PYPROJECT = "[project]\nname = 'payload'\nversion = '1'\n"


@pytest.fixture(autouse=True)
def _launchable(monkeypatch):
    """The suite disables launch-time installs globally; this is the launch path."""
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)


def _venv(path: Path, marker: str) -> Path:
    """A venv the selection readers accept: pyvenv.cfg plus a marker module."""
    version = f"{sys.version_info[0]}.{sys.version_info[1]}"
    (path / "pyvenv.cfg").parent.mkdir(parents=True, exist_ok=True)
    (path / "pyvenv.cfg").write_text(
        f"home = /usr/bin\nversion = {version}.0\n", encoding="utf-8")
    site = path / f"lib/python{version}/site-packages"
    site.mkdir(parents=True, exist_ok=True)
    (site / f"marker_{marker}.py").write_text(f'VALUE = "{marker}"\n', encoding="utf-8")
    return path


def _payload(tmp_path: Path, *, record: bool = True) -> tuple[Path, Path, Path | None]:
    """The bundled-app layout: a sealed tree, its manifest, its venv, and (optionally)
    the install record an in-place update leaves behind."""
    payloads = tmp_path / "payloads"
    root = payloads / "hermes-agent"
    root.mkdir(parents=True)
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (payloads / "manifest.json").write_text(json.dumps({
        "repo": "hermes-agent", "venv": "runtime/venv", "store": "runtime/store",
        "runtime": {"commands": {"hermes": "bin/hermes"}},
    }), encoding="utf-8")
    (payloads / "bin").mkdir()
    (payloads / "bin" / "hermes").write_text("#!/bin/sh\n", encoding="utf-8")
    payload = _venv(payloads / "runtime" / "venv", "new")
    stale = None
    if record:
        from pm.environments import install_state_dir, runtime_facts_path
        from pm.lock import Facts

        install_state_dir(root).mkdir(parents=True, exist_ok=True)
        stale = _venv(install_state_dir(root) / "environments" / "gen-old" / "venv", "old")
        Facts(runtime_facts_path(root)).record_state("venv", "release-N", [], environment=stale)
    return root, payload, stale


def _record_refresh(monkeypatch, outcome: str = "rebuilt") -> list[Path]:
    import pm.recovery

    seen: list[Path] = []

    def refresh(project_root):
        seen.append(Path(project_root))
        return outcome

    monkeypatch.setattr(pm.recovery, "refresh_dependencies", refresh)
    return seen


def test_a_payload_with_a_stale_record_refreshes_before_activation(tmp_path, monkeypatch):
    root, _payload_venv, _stale = _payload(tmp_path)
    seen = _record_refresh(monkeypatch)

    # No relaunch: this process has not activated yet, so a committed selection is picked up
    # by the activation that follows.
    assert venv_sync.prepare_launch(root, ["serve"]) is None
    assert seen == [root.resolve()]


@pytest.mark.parametrize("argv", [["--version"], ["pm", "install"], ["--help"]])
def test_commands_that_never_load_the_app_skip_the_refresh(tmp_path, monkeypatch, argv):
    root, _payload_venv, _stale = _payload(tmp_path)
    seen = _record_refresh(monkeypatch)

    assert venv_sync.prepare_launch(root, argv) is None
    assert seen == []


def test_no_recorded_selection_leaves_the_payload_alone(tmp_path, monkeypatch):
    root, _payload_venv, _stale = _payload(tmp_path, record=False)
    seen = _record_refresh(monkeypatch)

    assert venv_sync.prepare_launch(root, ["serve"]) is None
    assert seen == []


def test_a_tree_without_a_payload_manifest_is_not_a_payload(tmp_path, monkeypatch):
    root = tmp_path / "checkout-without-git"
    root.mkdir()
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    seen = _record_refresh(monkeypatch)

    assert venv_sync.prepare_launch(root, ["serve"]) is None
    assert seen == []


def test_a_checkout_keeps_its_own_update_owner(tmp_path, monkeypatch):
    root, _payload_venv, _stale = _payload(tmp_path)
    (root / ".git").mkdir()
    (root / "install-stamp.json").write_text(json.dumps({"updateMechanism": "external"}), encoding="utf-8")
    seen = _record_refresh(monkeypatch)

    assert venv_sync.prepare_launch(root, ["serve"]) is None
    assert seen == []


def test_a_refresh_failure_never_blocks_startup(tmp_path, monkeypatch, capsys):
    import pm.recovery

    root, _payload_venv, _stale = _payload(tmp_path)

    def broken(project_root):
        raise RuntimeError("uv is missing")

    monkeypatch.setattr(pm.recovery, "refresh_dependencies", broken)

    assert venv_sync.prepare_launch(root, ["serve"]) is None
    assert "dependency environment refresh skipped: uv is missing" in capsys.readouterr().err


def test_an_unexpected_refresh_failure_reaches_the_boot_boundary(tmp_path, monkeypatch):
    """Only the expected failure classes are swallowed; a defect must not vanish."""
    import pm.recovery

    root, _payload_venv, _stale = _payload(tmp_path)

    def broken(project_root):
        raise KeyError("a real defect")

    monkeypatch.setattr(pm.recovery, "refresh_dependencies", broken)

    with pytest.raises(KeyError):
        venv_sync.refresh_sealed_selection(root)


def test_a_current_record_is_not_rebuilt(tmp_path, monkeypatch):
    """The boot no-op: an up-to-date record keeps its generation and syncs nothing."""
    import pm.client
    import pm.install
    from pm import paths
    from pm.environments import committed_venv, runtime_facts_path
    from pm.lock import Facts

    root, payload, stale = _payload(tmp_path)
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: True)
    monkeypatch.setattr(pm.client, "sync_venv", lambda *a, **k: pytest.fail("no rebuild expected"))
    before = runtime_facts_path(root).read_text(encoding="utf-8")

    assert venv_sync.refresh_sealed_selection(root) == "current"
    assert committed_venv(root) == stale != payload
    assert runtime_facts_path(root).read_text(encoding="utf-8") == before
    fact = Facts(runtime_facts_path(root)).get("venv")
    assert fact is not None and fact.get("environment") == str(stale)


def test_a_rebuilt_selection_is_what_activation_uses(tmp_path, monkeypatch):
    """The rebuild is committed under the surviving install key, and the payload venv is
    not what boots."""
    import pm.client
    import pm.install
    from pm import paths
    from pm.environments import committed_venv, install_state_dir, runtime_facts_path
    from pm.lock import Facts

    root, payload, stale = _payload(tmp_path)
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: False)
    rebuilt = _venv(install_state_dir(root) / "environments" / "gen-new" / "venv", "newer")

    def sync(extras=None, **kwargs):
        assert extras is None
        Facts(runtime_facts_path(root)).record_state("venv", "release-N+1", [], environment=rebuilt)

    monkeypatch.setattr(pm.client, "sync_venv", sync)

    assert venv_sync.prepare_launch(root, ["serve"]) is None
    assert committed_venv(root) == rebuilt
    assert committed_venv(root) != payload
    assert committed_venv(root) != stale


def test_an_offline_refresh_boots_the_payload_environment(tmp_path, monkeypatch):
    """A rebuild that cannot finish drops the recorded environment, keeps the record for the
    next attempt, and boots on the payload's own venv."""
    import pm.client
    import pm.install
    from pm import paths
    from pm.environments import committed_venv, runtime_facts_path
    from pm.lock import Facts

    root, payload, _stale = _payload(tmp_path)
    monkeypatch.setattr(paths, "repo_root", lambda: root)
    monkeypatch.setattr(pm.install, "venv_is_current", lambda **kwargs: False)

    def offline(*args, **kwargs):
        raise RuntimeError("uv sync exited 1: no network")

    monkeypatch.setattr(pm.client, "sync_venv", offline)

    assert venv_sync.prepare_launch(root, ["serve"]) is None
    assert committed_venv(root) == payload
    fact = Facts(runtime_facts_path(root)).get("venv")
    assert fact is not None and fact.get("stamp") == "release-N"
    assert "environment" not in fact
