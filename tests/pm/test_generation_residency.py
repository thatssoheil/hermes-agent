"""A generation is kept while a config file pins it or a live process runs from it (#141448).

A lease proves that a process SELECTED a generation, not that anything still runs from it: the
lock belongs to the activating process, and the children it launches (stdio MCP helpers, cron
workers, subprocess tools) inherit the interpreter path without one. Configuration can pin a
generation even more durably, by absolute path. Both collectors therefore consult
``GenerationResidency`` before ``rmtree``. These tests drive the real collectors over sandbox
homes, with real child processes, real open descriptors and real config files as the inputs.
They live in ``tests/pm/`` because the probe is shared: both the app collector
(``hermes_cli/runtime_state``) and the PM runtime collector are exercised here, next to
``test_runtime_gc.py`` where the collectors' own contracts already live.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from contextlib import suppress
from types import SimpleNamespace
import time
from typing import NamedTuple

import pytest

from tests.waiting import wait_until

LINUX = pytest.mark.platforms("linux")


class _Home(NamedTuple):
    home: Path
    config: Path


def _app_generation(root: Path, name: str) -> Path:
    """An unselected, lease-managed, day-old app generation with an abandoned lease."""
    from pm.environments import install_state_dir

    generation = install_state_dir(root) / "environments" / name
    venv = generation / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    (generation / ".lease-managed").write_text("", encoding="utf-8")
    (generation / ".leases").mkdir()
    (generation / ".leases" / "abandoned").write_text("", encoding="utf-8")
    old = time.time() - 30 * 86400
    os.utime(generation / ".lease-managed", (old, old))
    return generation


def _select(root: Path, generation: Path) -> None:
    from pm.environments import runtime_facts_path

    runtime_facts_path(root).write_text(
        json.dumps({"packages": {"venv": {"environment": str(generation / "venv")}}, "schema": 1}),
        encoding="utf-8",
    )


def _pin(config: Path, generation: Path, *, windows_style: bool = False) -> None:
    """Write a config.yaml that names the generation's interpreter, as a real MCP pin does."""
    interpreter = str(generation / "venv" / "bin" / "python")
    if windows_style:
        interpreter = interpreter.replace("/", "\\")
    config.write_text("mcp_servers:\n  pin:\n    command: " + interpreter + "\n", encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch) -> _Home:
    """An isolated active home: the reference scan reads it through ``HERMES_HOME``."""
    import hermes_constants

    directory = tmp_path / "home"
    directory.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(directory))
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: directory)
    config = directory / "config.yaml"
    config.write_text("mcp_servers: {}\n", encoding="utf-8")
    return _Home(home=directory, config=config)


def _reason(generation: Path) -> str | None:
    """The signal keeping *generation*, from the same probe the collectors build per pass."""
    from hermes_cli.runtime_state import GenerationResidency

    return GenerationResidency().reason(generation)


def _proc_argv(pid: int) -> str:
    return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")


def _open_targets(pid: int) -> list[str]:
    targets = []
    for name in os.listdir(Path("/proc") / str(pid) / "fd"):
        try:
            targets.append(os.readlink(Path("/proc") / str(pid) / "fd" / name))
        except OSError:
            continue
    return targets


def _child(argv: list[str], *, executable: str | None = None, cwd: Path | None = None,
           env: dict[str, str] | None = None) -> subprocess.Popen:
    return subprocess.Popen(
        argv, executable=executable, cwd=cwd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _stop(child: subprocess.Popen) -> None:
    with suppress(ProcessLookupError, OSError):
        child.kill()
    child.wait(timeout=30)


def test_referenced_generation_survives_and_unreferenced_sibling_is_still_collected(tmp_path, home):
    from hermes_cli.runtime_state import collect_generations

    root = tmp_path / "hermes-agent"
    root.mkdir()
    selected = _app_generation(root, "selected0")
    _select(root, selected)
    pinned = _app_generation(root, "pinned00")
    _pin(home.config, pinned)

    assert collect_generations(root, min_age_seconds=0) == []
    assert pinned.is_dir(), "a generation a config file names must never be deleted"

    home.config.write_text("mcp_servers: {}\n", encoding="utf-8")
    assert collect_generations(root, min_age_seconds=0) == [pinned], "without the pin it is ordinary garbage"
    assert selected.is_dir()


def test_windows_style_config_spelling_pins_the_generation(tmp_path, home):
    """Config files carry native separators; PM records POSIX-style paths."""
    generation = _app_generation(tmp_path, "backslash0")
    _pin(home.config, generation, windows_style=True)

    assert _reason(generation) == f"referenced by {home.config}"


def test_a_siblings_longer_path_does_not_pin_the_shorter_generation(tmp_path, home):
    """``.../gen-ab/venv`` must not protect ``.../gen``: a reference is a whole path."""
    generation = _app_generation(tmp_path, "gen")
    sibling = generation.with_name("gen-ab")
    (sibling / "venv").mkdir(parents=True)
    _pin(home.config, sibling)

    assert _reason(generation) is None
    assert _reason(sibling) == f"referenced by {home.config}"


def test_profile_configs_pin_generations_too(tmp_path, home):
    """Every named profile's config can pin a generation, not only the active home's."""
    from hermes_cli.runtime_state import collect_generations

    root = tmp_path / "hermes-agent"
    root.mkdir()
    selected = _app_generation(root, "selected0")
    _select(root, selected)
    pinned = _app_generation(root, "profile000")
    profile = home.home / "profiles" / "ops" / "config.yaml"
    profile.parent.mkdir(parents=True)
    _pin(profile, pinned)

    assert collect_generations(root, min_age_seconds=0) == []
    assert pinned.is_dir(), "a profile's config pins the generation as firmly as the active home's"

    profile.unlink()
    assert collect_generations(root, min_age_seconds=0) == [pinned]


def test_an_unreadable_config_document_retains_the_generation(tmp_path, home):
    """An unreadable document is not an absent one: fail closed and name it."""
    from hermes_cli.runtime_state import collect_generations

    root = tmp_path / "hermes-agent"
    root.mkdir()
    selected = _app_generation(root, "selected0")
    _select(root, selected)
    unpinned = _app_generation(root, "unpinned0")
    home.config.unlink()
    home.config.mkdir()  # a directory where a file belongs: OSError on read

    assert collect_generations(root, min_age_seconds=0) == []
    assert unpinned.is_dir()


@LINUX
def test_child_executing_from_the_generation_blocks_collection(tmp_path):
    """The reporter's shape: an MCP helper runs from the tree, the activater is long gone."""
    from hermes_cli.runtime_state import collect_generations

    root = tmp_path / "hermes-agent"
    root.mkdir()
    selected = _app_generation(root, "selected0")
    _select(root, selected)
    in_use = _app_generation(root, "inuse000")
    script = in_use / "venv" / "bin" / "mcp_server.py"
    script.write_text("import time\ntime.sleep(120)\n", encoding="utf-8")

    child = _child([sys.executable, str(script)])
    try:
        wait_until(lambda: str(script) in _proc_argv(child.pid), what="child argv names the generation")
        assert collect_generations(root, min_age_seconds=0) == []
        assert in_use.is_dir(), "a running child must keep its generation alive"
    finally:
        _stop(child)

    assert collect_generations(root, min_age_seconds=0) == [in_use]


@LINUX
def test_child_whose_image_lives_in_the_generation_blocks_collection(tmp_path):
    """The interpreter itself inside the tree, with argv carrying no path: the exe branch."""
    from hermes_cli.runtime_state import collect_generations

    root = tmp_path / "hermes-agent"
    root.mkdir()
    selected = _app_generation(root, "selected0")
    _select(root, selected)
    in_use = _app_generation(root, "image000")
    interpreter = in_use / "tools" / "python3"  # outside venv/: a pyvenv.cfg would redirect the launch
    interpreter.parent.mkdir()
    try:
        os.link(os.path.realpath(sys.executable), interpreter)
    except OSError:
        pytest.skip("cannot hard-link the interpreter on this filesystem")

    def runs_the_generation_image() -> bool:
        try:
            return os.readlink(Path("/proc") / str(child.pid) / "exe") == str(interpreter)
        except OSError:
            return False

    child = _child(["python3", "-c", "import time\ntime.sleep(120)"], executable=str(interpreter))
    try:
        wait_until(runs_the_generation_image, what="child runs the generation's own interpreter")
        assert str(interpreter) not in _proc_argv(child.pid), "this probe must not rely on argv"
        assert collect_generations(root, min_age_seconds=0) == []
        assert in_use.is_dir()
    finally:
        _stop(child)

    assert collect_generations(root, min_age_seconds=0) == [in_use]


@LINUX
def test_child_working_inside_the_generation_blocks_collection(tmp_path):
    """A helper started elsewhere but working inside the tree is still a user of it."""
    from hermes_cli.runtime_state import collect_generations

    root = tmp_path / "hermes-agent"
    root.mkdir()
    selected = _app_generation(root, "selected0")
    _select(root, selected)
    in_use = _app_generation(root, "workspace0")

    child = _child(["python3", "-c", "import time\ntime.sleep(120)"], cwd=in_use)
    try:
        wait_until(
            lambda: os.readlink(Path("/proc") / str(child.pid) / "cwd") == str(in_use),
            what="child works inside the generation",
        )
        assert str(in_use) not in _proc_argv(child.pid)
        assert collect_generations(root, min_age_seconds=0) == []
        assert in_use.is_dir()
    finally:
        _stop(child)

    assert collect_generations(root, min_age_seconds=0) == [in_use]


@LINUX
def test_child_holding_an_open_file_without_path_in_argv_blocks_collection(tmp_path):
    """argv is not enough: a process that opened a file in the tree shows it only in its descriptors."""
    from hermes_cli.runtime_state import collect_generations

    root = tmp_path / "hermes-agent"
    root.mkdir()
    selected = _app_generation(root, "selected0")
    _select(root, selected)
    in_use = _app_generation(root, "inuse000")
    payload = in_use / "venv" / "lib" / "payload.bin"
    payload.parent.mkdir(parents=True)
    payload.write_text("x", encoding="utf-8")
    code = "import os, time\nhandle = open(os.environ['GEN_PAYLOAD'], 'rb')\ntime.sleep(120)\n"

    child = _child(["python3", "-c", code], env=dict(os.environ, GEN_PAYLOAD=str(payload)))
    try:
        wait_until(lambda: str(payload) in _open_targets(child.pid), what="child holds a descriptor in the tree")
        assert str(payload) not in _proc_argv(child.pid), "this probe must not rely on argv"
        assert collect_generations(root, min_age_seconds=0) == []
        assert in_use.is_dir()
    finally:
        _stop(child)

    assert collect_generations(root, min_age_seconds=0) == [in_use]


@LINUX
def test_scan_cap_retains_rather_than_deletes(tmp_path, monkeypatch):
    """An inconclusive scan keeps the tree: disk is cheaper than a broken tool."""
    from hermes_cli import runtime_state

    monkeypatch.setattr(runtime_state, "_RESIDENCY_PROCESS_LIMIT", 0)
    generation = _app_generation(tmp_path, "cap00000")

    assert "process cap" in (_reason(generation) or "")
    assert generation.is_dir()


@LINUX
def test_relative_argv_is_resolved_against_the_process_directory(tmp_path):
    """``../gen/venv/x`` reaches into the tree only once joined with the process's own cwd."""
    from hermes_cli.runtime_state import collect_generations

    root = tmp_path / "hermes-agent"
    root.mkdir()
    selected = _app_generation(root, "selected0")
    _select(root, selected)
    in_use = _app_generation(root, "relative0")
    payload = in_use / "venv" / "payload.bin"
    payload.write_text("x", encoding="utf-8")

    child = _child(["python3", "-c", "import time\ntime.sleep(120)", os.path.join(in_use.name, "venv", "payload.bin")],
                   cwd=in_use.parent)
    try:
        wait_until(lambda: "payload.bin" in _proc_argv(child.pid), what="child argv carries the relative path")
        assert not os.path.isabs(_proc_argv(child.pid).split()[-1]), "the probe must see a relative entry"
        assert collect_generations(root, min_age_seconds=0) == []
        assert in_use.is_dir()
    finally:
        _stop(child)


@LINUX
def test_descriptor_cap_retains_rather_than_deletes(tmp_path, monkeypatch):
    """A descriptor budget that trips is inconclusive, and inconclusive never deletes."""
    from hermes_cli import runtime_state

    monkeypatch.setattr(runtime_state, "_RESIDENCY_DESCRIPTOR_LIMIT", 0)
    generation = _app_generation(tmp_path, "fdcap000")

    assert "descriptor cap" in (_reason(generation) or "")
    assert generation.is_dir()


def _runtime_generation(root: Path, name: str) -> Path:
    generation = root / "generations" / name
    generation.mkdir(parents=True)
    (generation / ".lease-managed").write_text("", encoding="utf-8")
    (generation / "pm-runtime.json").write_text(json.dumps({"inputs": name}), encoding="utf-8")
    (generation / ".leases").mkdir()
    (generation / ".leases" / "abandoned").write_text("", encoding="utf-8")
    return generation


def _selected_runtime(root: Path) -> Path:
    selected = _runtime_generation(root, "selected0")
    (root / "selected.json").write_text(json.dumps({"generation": "generations/selected0"}), encoding="utf-8")
    return selected


def test_pm_runtime_collector_keeps_a_config_referenced_generation(tmp_path, home):
    """The PM runtime tree shares the seam: a superseded runtime a config names must survive."""
    from pm.runtime import collect_runtime_generations

    root = tmp_path / "pm-runtime"
    root.mkdir()
    _selected_runtime(root)
    pinned = _runtime_generation(root, "pinned00")
    _pin(home.config, pinned)

    assert collect_runtime_generations(root) == []
    assert pinned.is_dir()

    home.config.write_text("mcp_servers: {}\n", encoding="utf-8")
    assert collect_runtime_generations(root) == [pinned]


@LINUX
def test_pm_runtime_collector_keeps_a_resident_runtime_generation(tmp_path):
    from pm.runtime import collect_runtime_generations

    root = tmp_path / "pm-runtime"
    root.mkdir()
    _selected_runtime(root)
    in_use = _runtime_generation(root, "inuse000")
    script = in_use / "bin" / "serve.py"
    script.parent.mkdir(parents=True)
    script.write_text("import time\ntime.sleep(120)\n", encoding="utf-8")

    child = _child([sys.executable, str(script)])
    try:
        wait_until(lambda: str(script) in _proc_argv(child.pid), what="child argv names the runtime generation")
        assert collect_runtime_generations(root) == []
        assert in_use.is_dir()
    finally:
        _stop(child)

    assert collect_runtime_generations(root) == [in_use]


def test_an_aborted_stage_is_still_collected(tmp_path, home):
    """A stage that never published cannot be in use; the guard must not turn it into a leak."""
    from pm.runtime import collect_runtime_generations

    root = tmp_path / "pm-runtime"
    root.mkdir()
    _selected_runtime(root)
    aborted = root / "generations" / "aborted0"
    aborted.mkdir()
    _pin(home.config, aborted)  # even a config mention must not resurrect an unpublished stage

    assert collect_runtime_generations(root) == [aborted]


def test_collectors_report_what_they_retained(tmp_path, home):
    """``pm gc`` prints these pairs, so a survivor is explained rather than silently kept."""
    from hermes_cli.runtime_state import collect_generations

    root = tmp_path / "hermes-agent"
    root.mkdir()
    _select(root, _app_generation(root, "selected0"))
    pinned = _app_generation(root, "pinned00")
    _pin(home.config, pinned)
    retained: list[tuple[Path, str]] = []

    collect_generations(root, min_age_seconds=0, retained=retained)

    assert retained == [(pinned, f"referenced by {home.config}")]


class _FakeProcess:
    def __init__(self, pid: int, *, exe: str = "", cwd: str = "", cmdline: tuple[str, ...] = (),
                 files: tuple[str, ...] = (), uid: int | None = 0) -> None:
        self.pid = pid
        self.info = {"pid": pid, "cmdline": list(cmdline), "exe": exe, "cwd": cwd}
        self._uid = uid
        self._files = files

    def uids(self):
        if self._uid is None:
            raise NotImplementedError  # Windows: psutil cannot tell
        return SimpleNamespace(real=self._uid)

    def open_files(self):
        return [SimpleNamespace(path=path) for path in self._files]


def test_psutil_fallback_reads_the_hosts_without_proc(tmp_path, monkeypatch):
    """macOS has no /proc: the same signals come from psutil's process table."""
    from hermes_cli import runtime_state

    generation = _app_generation(tmp_path, "macos000")
    interpreter = str(generation / "venv" / "bin" / "python")
    processes = [
        _FakeProcess(1, cmdline=("launchd",), uid=0),
        _FakeProcess(2, exe=interpreter, uid=os.getuid()),
    ]
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(
        process_iter=lambda attrs: iter(processes), Error=Exception,
        AccessDenied=PermissionError, NoSuchProcess=ProcessLookupError, ZombieProcess=ProcessLookupError,
    ))

    assert runtime_state._resident_psutil(runtime_state._generation_roots(generation)) == "pid 2 is running from it"


def test_psutil_fallback_returns_none_when_the_module_is_absent(tmp_path, monkeypatch):
    from hermes_cli import runtime_state

    generation = _app_generation(tmp_path, "nopsutil0")
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert runtime_state._resident_psutil(runtime_state._generation_roots(generation)) is None
