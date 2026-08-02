from pathlib import Path, PurePosixPath

import pytest

import deploy


def test_repository_target_config_uses_ssh_aliases():
    targets = deploy.load_targets()

    assert set(targets) == {"aiden", "summer"}
    assert targets["aiden"].ssh_alias == "aiden"
    assert not targets["aiden"].deployable
    assert targets["summer"].ssh_alias == "summer"
    assert targets["summer"].proxy_jump == "aiden"
    assert targets["summer"].deployable
    assert targets["summer"].requirements == "requirements-runtime.txt"


def test_repository_public_route_uses_target_ssot():
    routes = deploy.load_public_routes()

    assert set(routes) == {"anima"}
    assert routes["anima"].target == "summer"
    assert routes["anima"].hostname == "anima.basedonapps.com"


def test_container_copies_runtime_requirements():
    dockerfile = (deploy.ANIMA_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY requirements.txt requirements-runtime.txt /tmp/" in dockerfile


def test_config_rejects_relative_remote_root(tmp_path: Path):
    config = tmp_path / "targets.toml"
    config.write_text(
        """[targets.bad]\nssh_alias='bad'\nrole='runtime'\nremote_root='relative'\n""",
        encoding="utf-8",
    )

    with pytest.raises(deploy.DeployError, match="must be absolute"):
        deploy.load_targets(config)


def test_config_rejects_broad_remote_root(tmp_path: Path):
    config = tmp_path / "targets.toml"
    config.write_text(
        """[targets.bad]\nssh_alias='bad'\nrole='runtime'\nremote_root='/srv'\n""",
        encoding="utf-8",
    )

    with pytest.raises(deploy.DeployError, match="too broad"):
        deploy.load_targets(config)


def test_service_is_rendered_from_target_config():
    target = deploy.Target(
        name="gpu",
        ssh_alias="gpu",
        role="runtime",
        remote_root=PurePosixPath("/srv/anima"),
        service="anima",
        port=9000,
        runtime_args=("--web", "--max-cells", "32"),
    )

    unit = deploy.render_service(target)

    assert "WorkingDirectory=/srv/anima/current" in unit
    assert "/srv/anima/venv/bin/python -u /srv/anima/current/anima_unified.py" in unit
    assert "--web --max-cells 32 --port 9000" in unit
    assert "KillSignal=SIGINT" in unit


def test_service_rejects_research_only_target():
    target = deploy.Target(name="gpu", ssh_alias="gpu", role="gpu-research")

    with pytest.raises(deploy.DeployError, match="no runtime configuration"):
        deploy.render_service(target)


def test_ssh_uses_configured_proxy_jump():
    target = deploy.Target(
        name="gpu", ssh_alias="gpu", proxy_jump="gateway",
        role="runtime",
    )

    assert deploy._ssh_argv(target) == [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-J", "gateway", "gpu",
    ]


def test_tunnel_service_reads_secret_from_token_file():
    target = deploy.load_targets()["summer"]
    route = deploy.load_public_routes()["anima"]

    unit = deploy.render_tunnel_service(target, route)

    assert "--token-file" in unit
    assert " tunnel run --token " not in unit
    assert "anima-lab-1.service" in unit


def test_public_route_reuses_tunnel_and_creates_canonical_dns_record(monkeypatch):
    route = deploy.load_public_routes()["anima"]
    target = deploy.load_targets()[route.target]
    calls = []
    installed = []

    class FakeAPI:
        account_id = "account"

        def request(self, method, path, body=None):
            calls.append((method, path, body))
            if path.startswith("/accounts/account/cfd_tunnel?"):
                return [{"id": "tunnel-id"}]
            if path == "/zones?name=basedonapps.com":
                return [{"id": "zone-id"}]
            if path.startswith("/zones/zone-id/dns_records?"):
                return []
            if path.endswith("/token"):
                return "tunnel-token"
            return {}

    monkeypatch.setattr(deploy, "CloudflareAPI", FakeAPI)
    monkeypatch.setattr(
        deploy, "_install_tunnel_connector",
        lambda route_target, public_route, token:
            installed.append((route_target, public_route, token)),
    )

    tunnel_id = deploy._ensure_cloudflare_route(route, target)

    assert tunnel_id == "tunnel-id"
    assert installed == [(target, route, "tunnel-token")]
    dns_create = next(
        call for call in calls
        if call[0:2] == ("POST", "/zones/zone-id/dns_records")
    )
    assert dns_create[2] == {
        "type": "CNAME",
        "name": "anima.basedonapps.com",
        "content": "tunnel-id.cfargotunnel.com",
        "proxied": True,
        "ttl": 1,
    }
    assert not any(
        method == "POST" and path == "/accounts/account/cfd_tunnel"
        for method, path, _body in calls
    )


def test_public_health_uses_explicit_probe_identity(monkeypatch):
    route = deploy.load_public_routes()["anima"]

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        assert timeout == 5
        assert request.get_header("User-agent") == "anima-deploy-health/1.0"
        assert request.get_header("Accept") == "text/html,application/xhtml+xml"
        return Response()

    monkeypatch.setattr(deploy.urllib.request, "urlopen", fake_urlopen)

    assert deploy._public_health(route, attempts=1)


def test_deploy_requires_published_head(monkeypatch):
    revisions = iter(("local", "remote"))
    monkeypatch.setattr(deploy, "_git_output", lambda *args: next(revisions))

    with pytest.raises(deploy.DeployError, match="origin/main"):
        deploy._assert_published_head()


def test_runtime_health_uses_valid_http_request(monkeypatch):
    commands = []
    target = deploy.Target(
        name="gpu",
        ssh_alias="gpu",
        role="runtime",
        port=9000,
    )

    class Result:
        returncode = 0

    def fake_ssh(_target, command, **_kwargs):
        commands.append(command)
        return Result()

    monkeypatch.setattr(deploy, "_ssh", fake_ssh)

    assert deploy._runtime_health(target, attempts=1)
    assert "HTTPConnection" in commands[0]
    assert "request" in commands[0]
