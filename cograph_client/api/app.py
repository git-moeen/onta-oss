import importlib
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from cograph_client.api.middleware import RequestLoggingMiddleware
from cograph_client.api.rate_limit import limiter
from cograph_client.api.routes import actions, agent, ask, conversations, enrich, explore, functions, health, ingest, jobs, knowledge_graphs, lambda_functions, normalize, ontology, query, schedules, tenants, triples
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


def _load_web_source_plugin() -> None:
    """Import and invoke the configured web-source plugin, if any.

    Format: "module.path:callable". The callable is invoked with no arguments
    and is expected to register a web-discovery provider via
    cograph_client.web_sources.base.register_web_source. Failures are logged but
    do not prevent startup — the "discover" intent simply stays dormant
    (plan() returns a "not enabled" message). The OSS dev stub registers via
    "cograph_client.web_sources.stub:register"; a downstream deployment points
    this at its paid provider with no OSS change.
    """
    spec = settings.web_source_plugin.strip()
    if not spec:
        return
    if ":" not in spec:
        logger.warning("web_source_plugin_invalid_format", spec=spec)
        return
    module_name, attr = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, attr)
        fn()
        logger.info("web_source_plugin_loaded", plugin=spec)
    except Exception as exc:
        logger.error("web_source_plugin_load_failed", plugin=spec, error=str(exc))


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
    # COG-136: start the in-process schedule firing loop. make_schedule_runner
    # returns None when scheduling is disabled (no database_url and not explicitly
    # enabled), so startup is unaffected when the feature is off. Failures here
    # are logged but never block the app from serving requests.
    app.state.schedule_runner = None
    try:
        from cograph_client.scheduling.runner import make_schedule_runner

        runner = make_schedule_runner(app.state)
        if runner is not None:
            runner.start()
            app.state.schedule_runner = runner
            logger.info("schedule_runner_enabled")
    except Exception as exc:  # noqa: BLE001 - scheduling must not break startup
        logger.error("schedule_runner_start_failed", error=str(exc))
    # ONTA-181: seed the semantic-maintenance schedule rows (the global
    # embed-fill sweep; per-KG reconcile rows are ensured by the write hook /
    # reindex route). Only meaningful when both the semantic index AND the
    # runner are enabled — rows without a runner would never fire, so we warn
    # instead of seeding. Best-effort: a seeding hiccup must not block startup.
    try:
        from cograph_client.semantic.reconciler import (
            ensure_embed_fill_schedule,
            semantic_index_enabled,
        )

        if semantic_index_enabled():
            if app.state.schedule_runner is not None:
                await ensure_embed_fill_schedule(app.state.schedule_store)
                logger.info("semantic_maintenance_schedules_seeded")
            else:
                logger.warning(
                    "semantic_index_enabled_without_scheduler",
                    hint=(
                        "COGRAPH_SEMANTIC_INDEX_ENABLED is set but the schedule "
                        "runner is disabled — embed-fill/reconcile will not run "
                        "(set OMNIX_DATABASE_URL or COGRAPH_SCHEDULER_ENABLED)."
                    ),
                )
    except Exception as exc:  # noqa: BLE001 - seeding must not break startup
        logger.error("semantic_schedule_seed_failed", error=str(exc))
    yield
    runner = getattr(app.state, "schedule_runner", None)
    if runner is not None:
        try:
            await runner.stop()
        except Exception as exc:  # noqa: BLE001 - shutdown best-effort
            logger.warning("schedule_runner_stop_failed", error=str(exc))
    await app.state.neptune_client.close()
    logger.info("shutdown")


def create_app() -> FastAPI:
    _load_auth_plugin()
    _load_enrichment_plugin()
    _load_governance_plugin()
    _load_web_source_plugin()
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
    app.include_router(schedules.router, tags=["schedules"])
    app.include_router(explore.router, tags=["explore"])
    app.include_router(normalize.router, tags=["normalize"])
    app.include_router(tenants.router, tags=["tenants"])
    app.include_router(agent.router, tags=["agent"])
    app.include_router(conversations.router, tags=["conversations"])
    _register_agent_capabilities()
    _load_router_plugins(app)
    return app


def _register_agent_capabilities() -> None:
    """Register the default OSS agent capabilities (query, normalize, enrich).

    The single agent endpoint dispatches through the capability registry, so
    capabilities must be registered for it to work. Import-safe + idempotent;
    a proprietary deployment registers additional capabilities the same way a
    router/enrichment plugin does, with no route change.
    """
    try:
        from cograph_client.agent.planner import register_default_capabilities

        register_default_capabilities()
        logger.info("agent_capabilities_registered")
    except Exception as exc:  # noqa: BLE001
        logger.error("agent_capability_registration_failed", error=str(exc))


app = create_app()
