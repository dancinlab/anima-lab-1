#!/usr/bin/env python3
"""Deploy and inspect Anima on SSH-configured GPU hosts.

Host aliases and runtime settings live in ``deploy.targets.toml``.  SSH owns
addresses, ports, users, and identity files; this module never duplicates them.
Each deployment installs an immutable Git release, keeps runtime state in a
shared directory, and switches the ``current`` symlink only after setup.
"""
from __future__ import annotations

import argparse
import re
import shlex
import socket
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence


ANIMA_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = ANIMA_DIR / "deploy.targets.toml"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.@-]+$")


class DeployError(RuntimeError):
    """A deployment invariant or remote operation failed."""


@dataclass(frozen=True)
class Target:
    name: str
    ssh_alias: str
    role: str
    remote_root: PurePosixPath | None = None
    service: str | None = None
    python: str = "python3"
    requirements: str = "requirements-runtime.txt"
    port: int | None = None
    runtime_args: tuple[str, ...] = ()

    @property
    def deployable(self) -> bool:
        return all((self.remote_root, self.service, self.port, self.runtime_args))


def load_targets(path: Path = DEFAULT_CONFIG) -> dict[str, Target]:
    """Load the deployment SSOT and validate target invariants."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    targets: dict[str, Target] = {}
    for name, values in raw.get("targets", {}).items():
        remote_root = values.get("remote_root")
        target = Target(
            name=name,
            ssh_alias=values["ssh_alias"],
            role=values["role"],
            remote_root=PurePosixPath(remote_root) if remote_root else None,
            service=values.get("service"),
            python=values.get("python", "python3"),
            requirements=values.get("requirements", "requirements-runtime.txt"),
            port=values.get("port"),
            runtime_args=tuple(values.get("runtime_args", ())),
        )
        if target.remote_root and not target.remote_root.is_absolute():
            raise DeployError(f"{name}: remote_root must be absolute")
        if target.remote_root and len(target.remote_root.parts) < 4:
            raise DeployError(f"{name}: remote_root is too broad")
        for label, value in (("ssh_alias", target.ssh_alias), ("service", target.service)):
            if value and not SAFE_NAME.fullmatch(value):
                raise DeployError(f"{name}: invalid {label}")
        requirements_path = PurePosixPath(target.requirements)
        if requirements_path.is_absolute() or ".." in requirements_path.parts:
            raise DeployError(f"{name}: requirements must stay inside the release")
        if target.port is not None and not 1 <= target.port <= 65535:
            raise DeployError(f"{name}: invalid port {target.port}")
        targets[name] = target
    if not targets:
        raise DeployError(f"no targets configured in {path}")
    return targets


def _run(
    argv: Sequence[str],
    *,
    timeout: int = 30,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        list(argv), input=input_bytes, capture_output=True, timeout=timeout,
    )
    if check and result.returncode:
        stderr = result.stderr.decode(errors="replace").strip()
        raise DeployError(f"command failed ({result.returncode}): {stderr}")
    return result


def _ssh(target: Target, command: str, *, timeout: int = 30, check: bool = True):
    return _run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            target.ssh_alias, command,
        ],
        timeout=timeout,
        check=check,
    )


def _remote_script(target: Target, script: str, *, timeout: int = 30):
    quoted = shlex.quote(script)
    return _ssh(target, f"bash -ceu {quoted}", timeout=timeout)


def _git_output(*args: str) -> str:
    result = _run(["git", "-C", str(ANIMA_DIR), *args])
    return result.stdout.decode().strip()


def _assert_published_head() -> str:
    head = _git_output("rev-parse", "HEAD")
    remote = _git_output("rev-parse", "origin/main")
    if head != remote:
        raise DeployError("HEAD must equal origin/main before deployment")
    return head


def _release_paths(target: Target, revision: str):
    if not target.remote_root:
        raise DeployError(f"{target.name} is a research-only target")
    root = target.remote_root
    return root, root / "releases" / revision


def _upload_release(target: Target, revision: str) -> PurePosixPath:
    """Stream the committed Git tree directly into an immutable release."""
    root, release = _release_paths(target, revision)
    prepare = (
        f"mkdir -p {shlex.quote(str(root / 'releases'))} "
        f"{shlex.quote(str(root / 'shared'))}; "
        f"test ! -e {shlex.quote(str(release))} || exit 42; "
        f"mkdir {shlex.quote(str(release))}"
    )
    prepared = _ssh(target, prepare, check=False)
    if prepared.returncode == 42:
        return release
    if prepared.returncode:
        raise DeployError(prepared.stderr.decode(errors="replace").strip())

    archive = subprocess.Popen(
        ["git", "-C", str(ANIMA_DIR), "archive", "--format=tar.gz", revision],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    extract = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            target.ssh_alias,
            f"tar -xzf - -C {shlex.quote(str(release))}",
        ],
        stdin=archive.stdout,
        capture_output=True,
        timeout=300,
    )
    archive.stdout.close()
    archive_code = archive.wait(timeout=30)
    if archive_code or extract.returncode:
        _ssh(target, f"rm -rf -- {shlex.quote(str(release))}", check=False)
        message = extract.stderr.decode(errors="replace").strip()
        raise DeployError(f"release upload failed: {message}")
    return release


def _prepare_shared_state(target: Target, release: PurePosixPath) -> None:
    root, _ = _release_paths(target, release.name)
    shared = root / "shared"
    script = f"""
root={shlex.quote(str(root))}
release={shlex.quote(str(release))}
shared={shlex.quote(str(shared))}

if [ ! -d "$shared/data" ]; then
  if [ -d "$release/data" ]; then mv "$release/data" "$shared/data"; else mkdir -p "$shared/data"; fi
else
  rm -rf -- "$release/data"
fi
ln -s "$shared/data" "$release/data"

mkdir -p "$shared/checkpoints"
if [ -d "$release/checkpoints" ]; then cp -an "$release/checkpoints/." "$shared/checkpoints/"; rm -rf -- "$release/checkpoints"; fi
ln -s "$shared/checkpoints" "$release/checkpoints"

for name in memory_alive.json state_alive.pt growth_alive.json web_memories.json; do
  if [ -L "$release/$name" ]; then continue; fi
  if [ -e "$release/$name" ] && [ ! -e "$shared/$name" ]; then mv "$release/$name" "$shared/$name"; else rm -f -- "$release/$name"; fi
  ln -s "$shared/$name" "$release/$name"
done
"""
    _remote_script(target, script)


def render_service(target: Target) -> str:
    """Render the user service from the target SSOT."""
    if not target.deployable or not target.remote_root:
        raise DeployError(f"{target.name} has no runtime configuration")
    root = target.remote_root
    executable = root / "venv" / "bin" / "python"
    command = [str(executable), "-u", str(root / "current" / "anima_unified.py")]
    command.extend(target.runtime_args)
    command.extend(("--port", str(target.port)))
    exec_start = " ".join(shlex.quote(part) for part in command)
    return f"""[Unit]
Description=Anima Lab runtime ({target.name})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={root}/current
ExecStart={exec_start}
Restart=on-failure
RestartSec=3
TimeoutStopSec=30
KillSignal=SIGINT
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""


def _install_runtime(target: Target, release: PurePosixPath) -> None:
    if not target.remote_root or not target.service:
        raise DeployError(f"{target.name} has no runtime configuration")
    root = target.remote_root
    unit_path = root / f"{target.service}.service"
    unit = render_service(target)
    script = f"""
root={shlex.quote(str(root))}
release={shlex.quote(str(release))}
venv="$root/venv"
if [ ! -x "$venv/bin/python" ]; then {shlex.quote(target.python)} -m venv --system-site-packages "$venv"; fi
"$venv/bin/python" -m pip install --disable-pip-version-check -q -r "$release/{shlex.quote(target.requirements)}"
printf %s {shlex.quote(unit)} > {shlex.quote(str(unit_path))}
previous=$(readlink "$root/current" || true)
if [ -n "$previous" ]; then ln -sfn "$previous" "$root/previous"; fi
ln -sfn "$release" "$root/current"
systemctl --user link {shlex.quote(str(unit_path))} >/dev/null
systemctl --user daemon-reload
systemctl --user enable --now {shlex.quote(target.service)}.service >/dev/null
systemctl --user restart {shlex.quote(target.service)}.service
"""
    _remote_script(target, script, timeout=300)


def _tcp_health(target: Target, attempts: int = 20) -> bool:
    if target.port is None:
        return False
    probe = (
        f"import socket; s=socket.create_connection(('127.0.0.1',{target.port}),2); "
        "s.close()"
    )
    for _ in range(attempts):
        result = _ssh(
            target,
            f"{shlex.quote(target.python)} -c {shlex.quote(probe)}",
            timeout=8,
            check=False,
        )
        if result.returncode == 0:
            return True
        time.sleep(1)
    return False


def rollback(target: Target) -> None:
    if not target.remote_root or not target.service:
        raise DeployError(f"{target.name} has no runtime configuration")
    root = target.remote_root
    script = f"""
test -L {shlex.quote(str(root / 'previous'))}
old=$(readlink {shlex.quote(str(root / 'previous'))})
current=$(readlink {shlex.quote(str(root / 'current'))})
ln -sfn "$current" {shlex.quote(str(root / 'previous'))}
ln -sfn "$old" {shlex.quote(str(root / 'current'))}
systemctl --user restart {shlex.quote(target.service)}.service
"""
    _remote_script(target, script)
    if not _tcp_health(target):
        raise DeployError(f"{target.name}: rollback did not become healthy")


def deploy(target: Target, model_path: Path | None = None) -> str:
    if not target.deployable:
        raise DeployError(f"{target.name} is configured for {target.role}, not runtime deployment")
    revision = _assert_published_head()
    release = _upload_release(target, revision)
    _prepare_shared_state(target, release)
    if model_path:
        if not model_path.is_file():
            raise DeployError(f"model does not exist: {model_path}")
        assert target.remote_root
        destination = target.remote_root / "shared" / "checkpoints" / "clm_v2" / "final.pt"
        _ssh(target, f"mkdir -p {shlex.quote(str(destination.parent))}")
        _run(["scp", str(model_path), f"{target.ssh_alias}:{destination}"], timeout=600)
    assert target.remote_root and target.service
    had_previous = _ssh(
        target,
        f"test -L {shlex.quote(str(target.remote_root / 'current'))}",
        check=False,
    ).returncode == 0
    _install_runtime(target, release)
    if not _tcp_health(target):
        if had_previous:
            rollback(target)
            suffix = "; previous release restored"
        else:
            _ssh(
                target,
                f"systemctl --user stop {shlex.quote(target.service)}.service",
                check=False,
            )
            suffix = "; first deployment stopped"
        raise DeployError(f"{target.name}: health check failed{suffix}")
    return revision


def status(target: Target) -> tuple[bool, str]:
    gpu = _ssh(
        target,
        "timeout 8s nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu "
        "--format=csv,noheader",
        timeout=15,
        check=False,
    )
    gpu_text = gpu.stdout.decode(errors="replace").strip() or "GPU unavailable"
    healthy = gpu.returncode == 0
    lines = [f"{target.name} ({target.role}): {gpu_text}"]
    if target.deployable and target.service:
        service = _ssh(
            target,
            f"systemctl --user is-active {shlex.quote(target.service)}.service",
            check=False,
        )
        active = service.stdout.decode().strip() == "active"
        port_ok = _tcp_health(target, attempts=1)
        lines.append(f"runtime: {'active' if active else 'inactive'}, port: {'open' if port_ok else 'closed'}")
        healthy = healthy and active and port_ok
    return healthy, "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--gpu-status", action="store_true")
    parser.add_argument("--rollback", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        targets = load_targets(args.config)
        if args.gpu_status:
            results = [status(target) for target in targets.values()]
            for _, report in results:
                print(report)
            return 0 if all(ok for ok, _ in results) else 1
        if not args.target:
            raise DeployError("--target is required unless --gpu-status is used")
        if args.target not in targets:
            raise DeployError(f"unknown target {args.target!r}; choose from {', '.join(targets)}")
        target = targets[args.target]
        if args.status:
            ok, report = status(target)
            print(report)
            return 0 if ok else 1
        if args.rollback:
            rollback(target)
            print(f"rollback complete: {target.name}")
            return 0
        revision = deploy(target, args.model)
        print(f"deployed {revision[:12]} to {target.name}")
        return 0
    except (DeployError, OSError, subprocess.SubprocessError, socket.error) as error:
        print(f"deploy failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
