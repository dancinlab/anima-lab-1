#!/usr/bin/env python3
"""Deploy and inspect Anima on SSH-configured GPU hosts.

Host aliases and runtime settings live in ``deploy.targets.toml``.  SSH owns
addresses, ports, users, and identity files; this module never duplicates them.
Each deployment installs an immutable Git release, keeps runtime state in a
shared directory, and switches the ``current`` symlink only after setup.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

from training_runtime import TrainingRun, load_training_run


ANIMA_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = ANIMA_DIR / "deploy.targets.toml"
DEFAULT_TRAINING_CONFIG = ANIMA_DIR / "training.toml"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.@-]+$")
HOSTNAME = re.compile(r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$")


class DeployError(RuntimeError):
    """A deployment invariant or remote operation failed."""


@dataclass(frozen=True)
class Target:
    name: str
    ssh_alias: str
    role: str
    proxy_jump: str | None = None
    remote_root: PurePosixPath | None = None
    service: str | None = None
    python: str = "python3"
    requirements: str = "requirements-runtime.txt"
    port: int | None = None
    runtime_args: tuple[str, ...] = ()

    @property
    def deployable(self) -> bool:
        return all((self.remote_root, self.service, self.port, self.runtime_args))


@dataclass(frozen=True)
class PublicRoute:
    name: str
    target: str
    hostname: str
    zone: str
    tunnel_name: str
    service: str
    cloudflared_url: str
    ttl: int


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
            proxy_jump=values.get("proxy_jump"),
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
        for label, value in (("ssh_alias", target.ssh_alias),
                             ("proxy_jump", target.proxy_jump),
                             ("service", target.service)):
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


def load_public_routes(path: Path = DEFAULT_CONFIG) -> dict[str, PublicRoute]:
    """Load public routes from the deployment SSOT."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    routes = {}
    for name, values in raw.get("public_routes", {}).items():
        route = PublicRoute(name=name, **values)
        if not all(SAFE_NAME.fullmatch(value) for value in
                   (route.target, route.tunnel_name, route.service)):
            raise DeployError(f"{name}: invalid public route identifier")
        if not HOSTNAME.fullmatch(route.hostname) or not HOSTNAME.fullmatch(route.zone):
            raise DeployError(f"{name}: invalid hostname or zone")
        if not route.hostname.endswith(f".{route.zone}"):
            raise DeployError(f"{name}: hostname must belong to zone")
        if not route.cloudflared_url.startswith("https://"):
            raise DeployError(f"{name}: cloudflared_url must use HTTPS")
        if route.ttl != 1 and not 60 <= route.ttl <= 86400:
            raise DeployError(f"{name}: invalid DNS TTL {route.ttl}")
        routes[name] = route
    return routes


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


def _ssh_argv(target: Target) -> list[str]:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    if target.proxy_jump:
        argv.extend(("-J", target.proxy_jump))
    argv.append(target.ssh_alias)
    return argv


def _ssh(target: Target, command: str, *, timeout: int = 30, check: bool = True):
    return _run([*_ssh_argv(target), command], timeout=timeout, check=check)


def _remote_script(target: Target, script: str, *, timeout: int = 30):
    quoted = shlex.quote(script)
    return _ssh(target, f"bash -ceu {quoted}", timeout=timeout)


def _git_output(*args: str) -> str:
    result = _run(["git", "-C", str(ANIMA_DIR), *args])
    return result.stdout.decode().strip()


def _secret(name: str) -> str:
    """Read a credential through the repository's secret CLI."""
    result = _run(["secret", "get", name])
    value = result.stdout.decode().strip()
    if not value:
        raise DeployError(f"secret {name!r} is empty")
    return value


class CloudflareAPI:
    """Minimal authenticated client for the tunnel and DNS deployment path."""

    base_url = "https://api.cloudflare.com/client/v4"

    def __init__(self):
        self.account_id = _secret("cloudflare.account_id")
        self.headers = {
            "X-Auth-Email": _secret("cloudflare.email"),
            "X-Auth-Key": _secret("cloudflare.global_api_key"),
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=self.headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except (urllib.error.URLError, json.JSONDecodeError) as error:
            raise DeployError(f"Cloudflare API request failed: {error}") from error
        if not payload.get("success"):
            errors = ", ".join(item.get("message", "unknown error")
                               for item in payload.get("errors", []))
            raise DeployError(f"Cloudflare API rejected request: {errors}")
        return payload.get("result")


def _query(**values) -> str:
    return urllib.parse.urlencode(values)


def _ensure_cloudflare_route(route: PublicRoute, target: Target) -> str:
    """Idempotently provision a remotely-managed tunnel and proxied CNAME."""
    if not target.deployable or not target.remote_root or not target.port:
        raise DeployError(f"{route.target}: public route target is not deployable")
    api = CloudflareAPI()
    tunnels = api.request(
        "GET", f"/accounts/{api.account_id}/cfd_tunnel?" +
        _query(name=route.tunnel_name, is_deleted="false"),
    )
    if tunnels:
        tunnel = tunnels[0]
    else:
        tunnel = api.request(
            "POST", f"/accounts/{api.account_id}/cfd_tunnel",
            {"name": route.tunnel_name, "config_src": "cloudflare"},
        )
    tunnel_id = tunnel["id"]
    api.request(
        "PUT",
        f"/accounts/{api.account_id}/cfd_tunnel/{tunnel_id}/configurations",
        {"config": {"ingress": [
            {"hostname": route.hostname,
             "service": f"http://127.0.0.1:{target.port}"},
            {"service": "http_status:404"},
        ]}},
    )

    zones = api.request("GET", "/zones?" + _query(name=route.zone))
    if len(zones) != 1:
        raise DeployError(f"expected one Cloudflare zone for {route.zone}")
    zone_id = zones[0]["id"]
    record_body = {
        "type": "CNAME",
        "name": route.hostname,
        "content": f"{tunnel_id}.cfargotunnel.com",
        "proxied": True,
        "ttl": route.ttl,
    }
    records = api.request(
        "GET", f"/zones/{zone_id}/dns_records?" +
        _query(type="CNAME", name=route.hostname),
    )
    if records:
        api.request("PUT", f"/zones/{zone_id}/dns_records/{records[0]['id']}",
                    record_body)
    else:
        api.request("POST", f"/zones/{zone_id}/dns_records", record_body)

    token = api.request(
        "GET", f"/accounts/{api.account_id}/cfd_tunnel/{tunnel_id}/token")
    _install_tunnel_connector(target, route, token)
    return tunnel_id


def _install_tunnel_connector(target: Target, route: PublicRoute, token: str) -> None:
    """Install cloudflared and its token-file-backed user service."""
    assert target.remote_root
    root = target.remote_root
    binary = root / "bin" / "cloudflared"
    token_file = root / "shared" / "cloudflared" / "token"
    unit_path = root / f"{route.service}.service"
    prepare = f"""
mkdir -p {shlex.quote(str(binary.parent))} {shlex.quote(str(token_file.parent))}
if [ ! -x {shlex.quote(str(binary))} ]; then
  curl -fsSL {shlex.quote(route.cloudflared_url)} -o {shlex.quote(str(binary))}.tmp
  chmod 0755 {shlex.quote(str(binary))}.tmp
  mv {shlex.quote(str(binary))}.tmp {shlex.quote(str(binary))}
fi
umask 077
cat > {shlex.quote(str(token_file))}.tmp
mv {shlex.quote(str(token_file))}.tmp {shlex.quote(str(token_file))}
"""
    _run([*_ssh_argv(target), f"bash -ceu {shlex.quote(prepare)}"],
         timeout=180, input_bytes=token.encode())
    unit = render_tunnel_service(target, route)
    install = f"""
printf %s {shlex.quote(unit)} > {shlex.quote(str(unit_path))}
systemctl --user link {shlex.quote(str(unit_path))} >/dev/null
systemctl --user daemon-reload
systemctl --user enable --now {shlex.quote(route.service)}.service >/dev/null
systemctl --user restart {shlex.quote(route.service)}.service
"""
    _remote_script(target, install)


def render_tunnel_service(target: Target, route: PublicRoute) -> str:
    """Render the secret-safe cloudflared user service."""
    if not target.remote_root or not target.service:
        raise DeployError(f"{target.name}: public route target is not deployable")
    root = target.remote_root
    binary = root / "bin" / "cloudflared"
    token_file = root / "shared" / "cloudflared" / "token"
    return f"""[Unit]
Description=Cloudflare Tunnel for {route.hostname}
After=network-online.target {target.service}.service
Wants=network-online.target

[Service]
Type=simple
ExecStart={binary} tunnel run --token-file {token_file}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
"""


def _public_health(route: PublicRoute, attempts: int = 20) -> bool:
    for _ in range(attempts):
        request = urllib.request.Request(
            f"https://{route.hostname}/",
            headers={
                "User-Agent": "anima-deploy-health/1.0",
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return True
        except urllib.error.URLError:
            pass
        time.sleep(1)
    return False


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
        [*_ssh_argv(target),
         f"tar -xzf - -C {shlex.quote(str(release))}"],
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


def render_training_service(
    target: Target,
    run: TrainingRun,
    release: PurePosixPath,
) -> str:
    """Render a checkpoint-resuming GPU research service."""
    if not target.remote_root:
        raise DeployError(f"{target.name} has no research root")
    if run.target != target.name:
        raise DeployError(f"run {run.name} targets {run.target}, not {target.name}")
    executable = release / "training_runtime.py"
    # The trainer imports ``training_runtime`` from the research root, so both
    # the supervisor and child must read the same root-level configuration.
    config = target.remote_root / "training.toml"
    command = [
        target.python,
        "-u",
        str(executable),
        "--supervise",
        run.name,
        "--config",
        str(config),
        "--root",
        str(target.remote_root),
    ]
    exec_start = " ".join(shlex.quote(part) for part in command)
    return f"""[Unit]
Description=Anima GPU research ({run.name})
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory={target.remote_root}
ExecStart={exec_start}
Restart=on-failure
RestartSec={run.policy.restart_delay_seconds}
TimeoutStopSec={run.policy.terminate_grace_seconds + 5}
KillMode=control-group
KillSignal=SIGTERM
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
"""


def deploy_training(
    target: Target,
    run: TrainingRun,
    revision: str | None = None,
) -> str:
    """Install an immutable supervisor release and start one research run."""
    if not target.remote_root:
        raise DeployError(f"{target.name} has no research root")
    if run.target != target.name:
        raise DeployError(f"run {run.name} targets {run.target}, not {target.name}")
    revision = revision or _assert_published_head()
    root = target.remote_root
    supervisor_root = root / "supervisor"
    release = supervisor_root / "releases" / revision
    prepare = f"""
mkdir -p {shlex.quote(str(supervisor_root / 'releases'))}
if [ ! -d {shlex.quote(str(release))} ]; then
  mkdir {shlex.quote(str(release))}
fi
"""
    _remote_script(target, prepare)
    archive = subprocess.Popen(
        [
            "git", "-C", str(ANIMA_DIR), "archive", "--format=tar.gz",
            revision, "training_runtime.py", "training.toml",
        ],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    extract = subprocess.run(
        [*_ssh_argv(target), f"tar -xzf - -C {shlex.quote(str(release))}"],
        stdin=archive.stdout,
        capture_output=True,
        timeout=60,
    )
    archive.stdout.close()
    archive_code = archive.wait(timeout=30)
    if archive_code or extract.returncode:
        message = extract.stderr.decode(errors="replace").strip()
        raise DeployError(f"training supervisor upload failed: {message}")
    unit = render_training_service(target, run, release)
    unit_path = supervisor_root / f"{run.service}.service"
    install = f"""
cp {shlex.quote(str(release / 'training_runtime.py'))} {shlex.quote(str(root / 'training_runtime.py'))}.tmp
mv {shlex.quote(str(root / 'training_runtime.py'))}.tmp {shlex.quote(str(root / 'training_runtime.py'))}
cp {shlex.quote(str(release / 'training.toml'))} {shlex.quote(str(root / 'training.toml'))}.tmp
mv {shlex.quote(str(root / 'training.toml'))}.tmp {shlex.quote(str(root / 'training.toml'))}
printf %s {shlex.quote(unit)} > {shlex.quote(str(unit_path))}.tmp
mv {shlex.quote(str(unit_path))}.tmp {shlex.quote(str(unit_path))}
ln -sfn {shlex.quote(str(release))} {shlex.quote(str(supervisor_root / 'current'))}
systemctl --user link {shlex.quote(str(unit_path))} >/dev/null
systemctl --user daemon-reload
systemctl --user enable --now {shlex.quote(run.service)}.service >/dev/null
systemctl --user restart {shlex.quote(run.service)}.service
"""
    _remote_script(target, install)
    for _ in range(20):
        active = _ssh(
            target,
            f"systemctl --user is-active {shlex.quote(run.service)}.service",
            check=False,
        )
        if active.stdout.decode().strip() == "active":
            return revision
        time.sleep(1)
    raise DeployError(f"{run.service}: training service did not become active")


def training_status(target: Target, run: TrainingRun) -> tuple[bool, str]:
    """Return stable systemd state without invoking a potentially hung GPU API."""
    result = _ssh(
        target,
        "systemctl --user show "
        f"{shlex.quote(run.service)}.service "
        "--property=ActiveState,SubState,MainPID,NRestarts --no-pager",
        check=False,
    )
    report = result.stdout.decode(errors="replace").strip()
    active = result.returncode == 0 and "ActiveState=active" in report
    return active, report or f"{run.service}: unavailable"


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


def _runtime_health(target: Target, attempts: int = 20) -> bool:
    if target.port is None:
        return False
    probe = (
        "import http.client; "
        f"c=http.client.HTTPConnection('127.0.0.1',{target.port},timeout=2); "
        "c.request('GET','/'); r=c.getresponse(); r.read(1); "
        "raise SystemExit(0 if r.status == 200 else 1)"
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
    if not _runtime_health(target):
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
        scp = ["scp"]
        if target.proxy_jump:
            scp.extend(("-o", f"ProxyJump={target.proxy_jump}"))
        scp.extend((str(model_path), f"{target.ssh_alias}:{destination}"))
        _run(scp, timeout=600)
    assert target.remote_root and target.service
    had_previous = _ssh(
        target,
        f"test -L {shlex.quote(str(target.remote_root / 'current'))}",
        check=False,
    ).returncode == 0
    _install_runtime(target, release)
    if not _runtime_health(target):
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
        port_ok = _runtime_health(target, attempts=1)
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
    parser.add_argument("--public-route")
    parser.add_argument("--training-run")
    parser.add_argument("--training-status", action="store_true")
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        targets = load_targets(args.config)
        if args.training_run:
            run = load_training_run(args.training_run, args.training_config)
            if run.target not in targets:
                raise DeployError(f"unknown training target {run.target!r}")
            target = targets[run.target]
            if args.training_status:
                ok, report = training_status(target, run)
                print(report)
                return 0 if ok else 1
            revision = deploy_training(target, run)
            print(f"deployed training {run.name} {revision[:12]} to {target.name}")
            return 0
        if args.public_route:
            routes = load_public_routes(args.config)
            if args.public_route not in routes:
                raise DeployError(f"unknown public route {args.public_route!r}")
            route = routes[args.public_route]
            if route.target not in targets:
                raise DeployError(f"unknown route target {route.target!r}")
            tunnel_id = _ensure_cloudflare_route(route, targets[route.target])
            if not _public_health(route):
                raise DeployError(f"{route.hostname}: public health check failed")
            print(f"public route healthy: {route.hostname} ({tunnel_id})")
            return 0
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
    except (DeployError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"deploy failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
