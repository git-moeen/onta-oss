"""Tests for the generic OMNIX_ROUTER_PLUGINS hook (COG-85).

Mirrors the enrichment/governance plugin-loading pattern: a dotted
"module.path:callable" is imported at create_app() time and invoked with the
FastAPI app so it can include_router(...). Failures are logged, not raised.
"""
from fastapi.testclient import TestClient


def test_router_plugin_loaded_and_mounts_router(monkeypatch):
    """register(app) runs during create_app() and the mounted route works."""
    from cograph_client.api import app as app_module
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "router_plugins", "tests.fake_router_plugin:register")
    try:
        app = app_module.create_app()

        from tests import fake_router_plugin

        # The callable received the app instance...
        assert fake_router_plugin.APP is app
        # ...and the router it mounted is reachable.
        with TestClient(app) as client:
            resp = client.get("/_fake_router_plugin/ping")
            assert resp.status_code == 200
            assert resp.json() == {"ok": True}
    finally:
        from tests import fake_router_plugin

        fake_router_plugin.APP = None


def test_router_plugins_supports_comma_separated_entries(monkeypatch):
    """A blank/duplicate-friendly comma-separated spec still loads each entry."""
    from cograph_client.api import app as app_module
    from cograph_client.config import settings

    monkeypatch.setattr(
        settings, "router_plugins", " tests.fake_router_plugin:register , "
    )
    try:
        app = app_module.create_app()
        from tests import fake_router_plugin

        assert fake_router_plugin.APP is app
    finally:
        from tests import fake_router_plugin

        fake_router_plugin.APP = None


def test_router_plugin_invalid_format_logged(monkeypatch):
    """Malformed entry is logged but does not raise."""
    from cograph_client.api import app as app_module
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "router_plugins", "no_colon_here")
    # Must not raise.
    app_module.create_app()


def test_router_plugin_import_failure_does_not_crash(monkeypatch):
    """A plugin that can't be imported is logged, app still starts."""
    from cograph_client.api import app as app_module
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "router_plugins", "tests.does_not_exist:register")
    # Must not raise.
    app_module.create_app()
