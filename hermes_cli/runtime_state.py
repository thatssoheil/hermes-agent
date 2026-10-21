"""Stdlib-only recovery and lifetime protection for dependency generations."""
from __future__ import annotations

import atexit
import base64
from collections.abc import Callable
from contextlib import contextmanager, suppress
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from pm.environments import dependency_home_root, install_state_dir, runtime_facts_path
# Private aliases: this module calls them through its globals (tests patch ``_atomic_bytes``
# here) and updaters shipped before PM import them by these names mid-swap
# (tests/compat/old_updater_surface.json). New code imports the pm.filesystem names.
from pm.filesystem import (
    durable_write_bytes as _atomic_bytes,
    file_digest as _digest,
    lock_fd as _lock,
    read_bytes_or_none as _bytes,
)

LOG = logging.getLogger(__name__)

# How long a process that only needs to READ the install state waits for a writer. The holder can
# be another profile's backend rebuilding the whole dependency environment (measured: tens of
# seconds on a bundle, minutes when a sync falls back to an sdist build), and a backend that never
# binds its port is worse than one that binds against the previous generation. Losers skip — the
# same rule boot_bootstrap._RecordLock states for home maintenance.
INSTALL_LOCK_TIMEOUT_SECONDS = 10.0

# Residency scan bounds (see ``GenerationResidency``). A collection pass must stay cheap on a busy
# host, so the process table and the descriptor tables are walked only up to these caps. A trip is
# INCONCLUSIVE and retains the generation: keeping a dead tree costs disk a later pass reclaims,
# while deleting a live one breaks running tools permanently.
_CONFIG_SCAN_BYTES = 8 * 1024 * 1024
_RESIDENCY_PROCESS_LIMIT = 4096
_RESIDENCY_DESCRIPTOR_LIMIT = 100_000
_PATH_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


# A lease file younger than this is never a prune candidate: its creator may still sit between
# ``O_CREAT|O_EXCL`` and ``flock`` (two syscalls; the window is microseconds, the margin generous).
@contextmanager
def runtime_lock(project: Path, *, timeout: float | None = INSTALL_LOCK_TIMEOUT_SECONDS):
    """Hold the per-install dependency lock; yields True when held, False when the wait expired.

    Callers decide what a lost race means: readers skip the work the lock guards and carry on
    (``activate_dependencies`` still selects and leases the committed generation), writers that
    cannot be skipped pass ``timeout=None`` — an install the user asked for is theirs to wait on.
    """
    state = install_state_dir(project)
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if not _lock(fd, wait=True, timeout=timeout):
            LOG.warning(
                "dependency lock still held after %ss; continuing without it (%s)",
                timeout, state / ".install.lock")
            yield False
            return
        yield True
    finally:
        os.close(fd)


def _recover_plugin_publication(project: Path, row: dict, journal: Path) -> None:
    from hermes_cli.fs_utils import rmtree_force

    target, backup, metadata = (Path(row[key]) for key in ("target", "backup", "metadata"))
    home = dependency_home_root().resolve()
    if (not target.resolve().is_relative_to(home) or target.parent.name != "plugins"
            or backup.parent != target.parent or not backup.name.startswith(".previous-")
            or metadata != target.parent / ".install-metadata.json"):
        raise ValueError("plugin publication paths escape their home")
    committed = row.get("committed") or _digest(runtime_facts_path(project)) != row["facts_before"]
    if committed:
        if backup.exists():
            rmtree_force(backup)
    else:
        old = base64.b64decode(row["metadata_before"], validate=True) if row["metadata_before"] is not None else None
        current = _bytes(metadata)
        new = base64.b64decode(row["metadata_after"], validate=True)
        if current not in (old, new):
            raise ValueError("plugin metadata changed after publication; preserve it for manual recovery")
        if backup.exists():
            if target.exists():
                rmtree_force(target)
            os.replace(backup, target)
        elif not row["target_existed"] and target.exists():
            rmtree_force(target)
        if old is None:
            metadata.unlink(missing_ok=True)
        else:
            _atomic_bytes(metadata, old)
    journal.unlink()


def recover_publication(project: Path) -> None:
    """Recover while holding runtime_lock, before activation or another write."""
    journal = install_state_dir(project) / "publication.json"
    data = _bytes(journal)
    if data is None:
        return
    try:
        row = json.loads(data)
        if row.get("kind") == "plugin":
            _recover_plugin_publication(project, row, journal)
            return
        # One row may carry several configs (a plugin eviction edits every home that enables it).
        entries = row["configs"] if "configs" in row else [row]
        configs = []
        for entry in entries:
            config = Path(entry["config"])
            if config.name != "config.yaml" or not config.resolve().is_relative_to(dependency_home_root().resolve()):
                raise ValueError("config path is outside Hermes state")
            previous = base64.b64decode(entry["previous"], validate=True) if entry["previous"] is not None else None
            configs.append((config, previous, entry.get("config_after")))
        if not row.get("committed") and _digest(runtime_facts_path(project)) == row["facts_before"]:
            for config, previous, after in configs:
                prior = hashlib.sha256(previous).hexdigest() if previous is not None else None
                if _digest(config) not in (prior, after):
                    raise ValueError("config changed after publication began; preserve it for manual recovery")
            # No selection was published. Roll back before any plugin is loaded.
            for config, previous, _after in configs:
                if previous is None:
                    config.unlink(missing_ok=True)
                else:
                    _atomic_bytes(config, previous)
        journal.unlink()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise RuntimeError(f"cannot recover dependency publication: {journal}: {exc}") from exc


def finish_publication(project: Path) -> None:
    """Persist a commit even when code/config changed without a new generation."""
    journal = install_state_dir(project) / "publication.json"
    row = json.loads(journal.read_bytes())
    row["committed"] = True
    _atomic_bytes(journal, json.dumps(row).encode())
    recover_publication(project)


def lease_generation(environment: Path) -> Callable[[], None]:
    """Hold a kernel lock until process exit; the returned callable releases it early.

    Call under ``runtime_lock`` at boot. Without the lock (``runtime_lock`` timed out) the
    caller must re-read the selection after leasing: an installer may have moved it in between,
    and an unselected, unleased generation is exactly what the collector removes.
    """
    return lease_directory(environment.parent)


def lease_directory(generation: Path) -> Callable[[], None]:
    """Pin a lease-managed generation directory for this process's lifetime."""
    if not (generation / ".lease-managed").is_file():
        return lambda: None  # Generations produced before leases stay conservatively retained.
    leases = generation / ".leases"
    leases.mkdir(exist_ok=True)
    _prune_unlocked_leases(leases)
    while True:
        lease = leases / uuid.uuid4().hex
        fd = os.open(lease, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            _lock(fd, wait=True)
        except BaseException:
            os.close(fd)
            raise
        # Create and flock are two syscalls; a peer's prune (pm workers and launch run outside
        # ``runtime_lock``) can lock-and-unlink the file in between, and we would then hold an
        # unlinked inode invisible to every collector. The pruner only unlinks while holding the
        # lock, so once we hold it the path is either still ours or already gone for good.
        if lease.exists():
            break
        os.close(fd)

    def release() -> None:
        atexit.unregister(release)
        os.close(fd)
        # Best effort: the collector ignores unlocked lease files, but one per
        # invocation would otherwise accumulate for the life of the generation.
        with suppress(OSError):
            lease.unlink()

    atexit.register(release)
    return release


def collect_generations(
    project: Path, *, min_age_seconds: float = 86400, retained: list[tuple[Path, str]] | None = None
) -> list[Path]:
    """Remove unselected lease-managed generations once nothing references or runs from them.

    ``retained`` collects the (generation, reason) pairs the residency probe kept, so callers
    such as ``pm gc`` can report why a tree survived instead of only counting removals.

    Known limitation: a child spawned between the probe and the ``rmtree`` is not protected.
    Closing that window needs children to take a lease at their own boot (the issue's option 2),
    a larger change; this probe is the fail-closed backstop.
    """
    from pm.environments import selected_venv
    removed = []
    root = install_state_dir(project)
    if not root.exists():
        return removed
    with runtime_lock(project) as held:
        if not held:
            return removed  # maintenance: another process owns the install, skip rather than queue
        recover_publication(project)
        selected = selected_venv(project).parent.resolve()
        generations = root / "environments"
        if not generations.is_dir():
            return removed
        probe = GenerationResidency()
        for generation in generations.iterdir():
            if generation.is_symlink() or not generation.is_dir() or generation.resolve() == selected:
                continue
            marker = generation / ".lease-managed"
            if not marker.is_file() or time.time() - marker.stat().st_mtime < min_age_seconds:
                continue
            if not leases_held(generation):
                reason = probe.reason(generation)
                if reason:
                    LOG.info("retaining dependency generation %s: %s", generation, reason)
                    if retained is not None:
                        retained.append((generation, reason))
                    continue
                shutil.rmtree(generation)
                removed.append(generation)
    return removed


def leases_held(generation: Path) -> bool:
    """True while any process still holds a lease taken by ``lease_directory``."""
    return _prune_unlocked_leases(generation / ".leases")


_MATCH = "match"
_UNKNOWN = "unknown"


class GenerationResidency:
    """One "is anything still using this generation" snapshot, shared by a collection pass.

    A lease answers "did a process SELECT this generation", not "is anything still running from
    it": the lock belongs to the process that activated the environment, so every child it later
    launches (stdio MCP helpers, cron workers, subprocess tools) inherits the interpreter path
    without one, and a configured MCP server can pin the tree by absolute path (#141448).

    Build one probe per collection pass and ask it about every candidate: config documents are
    read once, a generation a config document names is answered without touching the process
    table, and the descriptor targets of each pid are read at most once. Every ambiguous answer
    RETAINS the generation - keeping a dead tree costs disk a later pass reclaims, while deleting
    a live one breaks running tools permanently.
    """

    def __init__(self, *, config_documents: list[Path] | None = None) -> None:
        self._documents = list(config_documents) if config_documents is not None else _config_documents()
        self._configs: list[tuple[Path, str]] | None = None
        self._unreadable_config: Path | None = None
        self._processes: list[Path] | None = None
        self._process_table_unreadable = False
        self._descriptors = 0
        self._fd_targets: dict[str, list[str]] = {}

    def reason(self, generation: Path) -> str | None:
        """The signal that keeps *generation*, or ``None`` when it is free to collect."""
        return self._reference(generation) or self._resident(generation)

    # ---- durable references -------------------------------------------------------------

    def _reference(self, generation: Path) -> str | None:
        configs = self._config_texts()
        if self._unreadable_config is not None:
            return f"cannot read {self._unreadable_config}"
        roots = _generation_roots(generation)
        for config, text in configs:
            if _mentions(text, roots):
                return f"referenced by {config}"
        return None

    def _config_texts(self) -> list[tuple[Path, str]]:
        if self._configs is None:
            self._configs = []
            for config in self._documents:
                try:
                    if config.stat().st_size > _CONFIG_SCAN_BYTES:
                        raise OSError(f"larger than {_CONFIG_SCAN_BYTES} bytes")
                    text = config.read_text(encoding="utf-8-sig", errors="replace")
                except FileNotFoundError:
                    continue  # no config: nothing pinned by this document
                except OSError:
                    # Unreadable is not unpinned. Fail closed and name the document.
                    self._unreadable_config = config
                    continue
                self._configs.append((config, _normalized_path(text)))
        return self._configs

    # ---- live residency -----------------------------------------------------------------

    def _resident(self, generation: Path) -> str | None:
        roots = _generation_roots(generation)
        if sys.platform.startswith("linux"):
            return self._resident_proc(roots)
        return _resident_psutil(roots)

    def _resident_proc(self, roots: tuple[str, ...]) -> str | None:
        """Walk ``/proc``: argv, executable, working directory and open descriptors."""
        if self._processes is None:
            try:
                entries = [entry for entry in Path("/proc").iterdir() if entry.name.isdigit()]
            except OSError:
                self._process_table_unreadable = True
                entries = []
            self._processes = sorted(entries, key=lambda entry: int(entry.name))
        if self._process_table_unreadable:
            return "the process table is unreadable"
        for inspected, pid_dir in enumerate(self._processes, start=1):
            if inspected > _RESIDENCY_PROCESS_LIMIT:
                return f"residency scan stopped at its {_RESIDENCY_PROCESS_LIMIT}-process cap"
            verdict = self._proc_pid(pid_dir, roots)
            if verdict == _MATCH:
                return f"pid {pid_dir.name} is running from it"
            if self._descriptors > _RESIDENCY_DESCRIPTOR_LIMIT:
                return f"residency scan stopped at its {_RESIDENCY_DESCRIPTOR_LIMIT}-descriptor cap"
            if verdict == _UNKNOWN:
                return f"pid {pid_dir.name} could not be inspected"
        return None

    def _proc_pid(self, pid_dir: Path, roots: tuple[str, ...]) -> str:
        """``_MATCH``, ``_UNKNOWN`` or nothing for one pid.

        ``cmdline`` is world-readable, so a process of another user that runs from this tree is
        still caught by its argv (the ``sudo hermes`` shape). Its executable and descriptors are
        not readable, and only the owner of a home can execute or open files under it, so an
        unreadable descriptor table there is skipped rather than treated as inconclusive -
        otherwise every root service on the host would pin every generation forever.
        """
        executable = _readlink(pid_dir / "exe")
        working_directory = _readlink(pid_dir / "cwd")
        for target in (executable, working_directory):
            if target and _under(target, roots):
                return _MATCH
        try:
            raw = (pid_dir / "cmdline").read_bytes()
        except PermissionError:
            return "" if _owned_by_user(pid_dir) is False else _UNKNOWN
        except OSError:
            raw = b""  # a kernel thread: no argv to run from
        for argument in raw.split(b"\0"):
            if not argument:
                continue
            text = argument.decode("utf-8", "replace")
            if _under(text, roots):
                return _MATCH
            if working_directory and not os.path.isabs(text) and _under(os.path.join(working_directory, text), roots):
                return _MATCH
        try:
            names = os.listdir(pid_dir / "fd")
        except PermissionError:
            return "" if _owned_by_user(pid_dir) is False else _UNKNOWN
        except OSError:
            return ""
        targets = self._fd_targets.get(str(pid_dir))
        if targets is None:
            targets = []
            for name in names:
                self._descriptors += 1
                if self._descriptors > _RESIDENCY_DESCRIPTOR_LIMIT:
                    return _UNKNOWN
                target = _readlink(pid_dir / "fd" / name)
                if target:
                    targets.append(target)
            self._fd_targets[str(pid_dir)] = targets
        return _MATCH if any(_under(target, roots) for target in targets) else ""


def _resident_psutil(roots: tuple[str, ...]) -> str | None:
    """Residency on hosts without ``/proc``, from psutil's cross-platform process table.

    Windows already refuses to unlink an executable a process has mapped, so ``rmtree`` fails
    safely there; macOS unlinks a running binary exactly like Linux does, which makes the scan
    the only protection on that host. psutil is imported late because this module is imported
    during boot, before the application's dependencies are guaranteed to be installed; without
    it the answer stays ``None`` and the config pin is the remaining guard.

    Ownership is judged the way the ``/proc`` path judges it: another user's process cannot be
    inspected, and cannot execute or hold files under a home we own, so it is skipped. On
    Windows ownership cannot be established at all, which leaves the unlink refusal in charge.
    """
    try:
        import psutil
    except ImportError:  # minimal hosts: the config reference remains the guard
        return None
    try:
        processes = list(psutil.process_iter(attrs=["pid", "cmdline", "exe", "cwd"]))
    except (OSError, psutil.Error):
        return None
    if len(processes) > _RESIDENCY_PROCESS_LIMIT:
        return f"residency scan stopped at its {_RESIDENCY_PROCESS_LIMIT}-process cap"
    descriptors = 0
    for process in processes:
        pid = getattr(process, "pid", None)
        try:
            info = process.info
            candidates = [*(info.get("cmdline") or []), info.get("exe") or "", info.get("cwd") or ""]
            if any(candidate and _under(candidate, roots) for candidate in candidates):
                return f"pid {pid} is running from it"
            if not _psutil_process_is_ours(process, psutil):
                continue
            for handle in process.open_files():
                descriptors += 1
                if descriptors > _RESIDENCY_DESCRIPTOR_LIMIT:
                    return f"residency scan stopped at its {_RESIDENCY_DESCRIPTOR_LIMIT}-descriptor cap"
                if _under(handle.path, roots):
                    return f"pid {pid} holds {handle.path}"
        except (OSError, psutil.Error):
            # psutil raises a family of errors per process (access denied, gone, zombie). Our own
            # process may be the one using the tree, so retain; another user's cannot be under a
            # home we own, and on Windows the unlink refusal covers it.
            if _psutil_process_is_ours(process, psutil):
                return f"pid {pid} could not be inspected"
            continue
    return None


def _psutil_process_is_ours(process: object, psutil: object) -> bool:
    """Whether a psutil process runs as this user; ``False`` where that cannot be told."""
    try:
        return process.uids().real == _current_uid()  # type: ignore[attr-defined]
    except (AttributeError, NotImplementedError, OSError, psutil.Error):  # type: ignore[attr-defined]
        return False


def _current_uid() -> int:
    """This user's uid, or -1 where the platform has none (Windows)."""
    return os.getuid() if hasattr(os, "getuid") else -1


def _readlink(path: Path) -> str | None:
    """A symlink's target, or ``None`` when it is absent or not ours to read."""
    try:
        return os.readlink(path)
    except OSError:
        return None


def _owned_by_user(pid_dir: Path) -> bool | None:
    """Whether this pid runs as our uid; ``None`` when even that is unreadable."""
    try:
        return pid_dir.stat().st_uid == os.getuid()
    except OSError:
        return None


def _normalized_path(path: str | Path) -> str:
    """A path comparable across the spellings PM and config documents use.

    Records are canonical POSIX paths while ``config.yaml`` carries native ones, so both
    separators and repeated separators are folded; Windows and default-APFS macOS compare
    case-insensitively, and a trailing separator never matters.
    """
    text = str(path).replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    text = text.rstrip("/")
    return text.lower() if os.name == "nt" or sys.platform == "darwin" else text


def _generation_roots(generation: Path) -> tuple[str, ...]:
    """Both spellings of *generation*: as recorded and after resolving symlinked homes."""
    roots = [_normalized_path(generation)]
    with suppress(OSError, RuntimeError):
        resolved = _normalized_path(generation.resolve())
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _config_documents() -> list[Path]:
    """``config.yaml`` files that can pin a generation: the active home and every profile."""
    homes: list[Path] = []
    with suppress(ImportError, OSError, RuntimeError):
        from hermes_constants import get_hermes_home_override

        override = get_hermes_home_override()
        if override:
            homes.append(Path(override))
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        homes.append(Path(os.path.expandvars(os.path.expanduser(env_home))))
    with suppress(ImportError, OSError, RuntimeError):
        from hermes_constants import get_default_hermes_root

        homes.append(get_default_hermes_root())
    documents: list[Path] = []
    for home in homes:
        documents.append(home / "config.yaml")
        profiles = home / "profiles"
        with suppress(OSError):
            if profiles.is_dir():
                documents.extend(sorted(profiles.glob("*/config.yaml")))
    return list(dict.fromkeys(documents))


def _mentions(text: str, roots: tuple[str, ...]) -> bool:
    """Whether *text* names a root as a whole path, not as the prefix of a longer one.

    A plain substring test would let ``.../gen-ab`` pin ``.../gen``; the named path must end
    there or continue with a separator, so a sibling's path never protects the wrong tree.
    """
    for root in roots:
        start = text.find(root)
        while start != -1:
            following = text[start + len(root):start + len(root) + 1]
            if following not in _PATH_NAME_CHARS:
                return True
            start = text.find(root, start + 1)
    return False


def _under(path: str, roots: tuple[str, ...]) -> bool:
    """Whether *path* is a root itself or sits inside one."""
    folded = _normalized_path(path.removesuffix(" (deleted)"))
    candidate = _normalized_path(os.path.normpath(folded))
    return any(candidate == root or candidate.startswith(root + "/") for root in roots)


def _prune_unlocked_leases(leases: Path) -> bool:
    """Remove abandoned lease files and report whether any live lock remains.

    Kernel locks disappear even when ``execv``, ``os._exit`` or a crash bypasses
    ``atexit``. Cleaning those unlocked files whenever a reader arrives bounds leaks in
    the selected generation too, which generation GC intentionally never visits.

    Fail closed: a lease this user cannot open (root-owned 0o600 from a ``sudo hermes`` on the
    same checkout) is held, never a boot failure.
    """
    held = False
    for lease in leases.glob("*"):
        try:
            fd = os.open(lease, os.O_RDWR)
        except FileNotFoundError:
            continue
        except OSError:
            held = True
            continue
        try:
            if not _lock(fd, wait=False):
                held = True
            else:
                with suppress(OSError):
                    lease.unlink()
        finally:
            os.close(fd)
    return held
