import importlib
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from cograph_client.api.middleware import RequestLoggingMiddleware
from cograph_client.api.rate_limit import limiter
from cograph_client.api.routes import actions, ask, enrich, explore, functions, health, ingest, jobs, knowledge_graphs, lambda_functions, ontology, query, tenants, triples
from cograph_client.config import settings
from cograph_client.graph.client import NeptuneClient
from cograph_client.logging import setup_logging

logger = structlog.stdlib.get_logger("cograph.app")


def _load_auth_plugin() -> None:
    """Import and invoke the configured auth plugin, if any.

    Format: "module.path:callable". The callable is invoked with no
    arguments and is expected to register an external verifier via
    omnix.auth.api_keys.register_external_verifier. Failures are logged
    but do not prevent the app from starting — the app will simply fall
    back to static API key auth.
    """
    spec = settings.auth_plugin.strip()
    if not spec:
        return
    if ":" not in spec:
        logger.warning("auth_plugin_invalid_format", spec=spec)
        return
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, attr)
        fn()
        logger.info("auth_plugin_loaded", plugin=spec)
    except Exception as exc:
        logger.error("auth_plugin_load_failed", plugin=spec, error=str(exc))


def _load_enrichment_plugin() -> None:
    """Import and invoke the configured enrichment plugin, if any.

    Format: "module.path:callable". The callable is invoked with no
    arguments and is expected to register paid source adapters via
    cograph_client.enrichment.sources.base.register_adapter and override
    tier→chain mappings via cograph_client.enrichment.tiers.register_tier.
    Failures are logged but do not prevent the app from starting — the
    app will simply fall back to the OSS defaults (lite tier, Wikidata).
    """
    spec = settings.enrichment_plugin.strip()
    if not spec:
        return
    if ":" not in spec:
        logger.warning("enrichment_plugin_invalid_format", spec=spec)
        return
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, attr)
        fn()
        logger.info("enrichment_plugin_loaded", plugin=spec)
    except Exception as exc:
        logger.error("enrichment_plugin_load_failed", plugin=spec, error=str(exc))


def _load_governance_plugin() -> None:
    """Import and invoke the configured governance plugin, if any (COG-56).

    Format: "module.path:callable". The callable is invoked with no
    arguments and is expected to register a mapping-shape judge panel via
    cograph_client.resolver.governance.register_governance_panel. Failures
    are logged but do not prevent the app from starting — the app simply
    falls back to the OSS default (proposals recorded pending,
    tenant-layer-only behavior).
    """
    spec = settings.governance_plugin.strip()
    if not spec:
        return
    if ":" not in spec:
        logger.warning("governance_plugin_invalid_format", spec=spec)
        return
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, attr)
        fn()
        logger.info("governance_plugin_loaded", plugin=spec)
    except Exception as exc:
        logger.error("governance_plugin_load_failed", plugin=spec, error=str(exc))


def _load_router_plugins(app: FastAPI) -> None:
    """Import and invoke the configured router plugins, if any.

    Format: comma-separated "module.path:callable" entries. Each callable is
    invoked with the FastAPI app instance and is expected to mount additional
    routers via app.include_router(...). Failures are logged per-entry but do
    not prevent the app from starting — the app simply runs with only the OSS
    routers. This is a generic plugin protocol (no proprietary coupling): it
    lets downstream deployments attach external routers (e.g. the premium
    ontology recommender).
    """
    spec = settings.router_plugins.strip()
    if not spec:
        return
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            logger.warning("router_plugin_invalid_format", spec=entry)
            continue
        module_name, attr = entry.split(":", 1)
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, attr)
            fn(app)
            logger.info("router_plugin_loaded", plugin=entry)
        except Exception as exc:
            logger.error("router_plugin_load_failed", plugin=entry, error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)
    logger.info("starting", neptune_endpoint=settings.neptune_endpoint)
    app.state.neptune_client = NeptuneClient(settings.neptune_endpoint, backend=settings.graph_backend)
    yield
    await app.state.neptune_client.close()
    logger.info("shutdown")


def create_app() -> FastAPI:
    _load_auth_plugin()
    _load_enrichment_plugin()
    _load_governance_plugin()
    app = FastAPI(
        title="Omnix",
        description="Living Knowledge Graph Platform",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(health.router, tags=["health"])
    app.include_router(triples.router, tags=["triples"])
    app.include_router(query.router, tags=["query"])
    app.include_router(functions.router, tags=["functions"])
    app.include_router(lambda_functions.router, tags=["lambda_functions"])
    app.include_router(ask.router, tags=["ask"])
    app.include_router(ontology.router, tags=["ontology"])
    app.include_router(ingest.router, tags=["ingest"])
    app.include_router(knowledge_graphs.router, tags=["knowledge_graphs"])
    app.include_router(enrich.router, tags=["enrich"])
    app.include_router(jobs.router, tags=["jobs"])
    app.include_router(actions.router, tags=["actions"])
    app.include_router(explore.router, tags=["explore"])
    app.include_router(tenants.router, tags=["tenants"])
    _load_router_plugins(app)
    return app


app = create_app()
