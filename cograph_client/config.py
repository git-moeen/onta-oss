import json
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    neptune_endpoint: str = "http://localhost:8182"
    graph_backend: str = "neptune"  # "neptune" or "fuseki"
    api_keys: str = '{}'  # empty = open access, no auth required
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    cerebras_api_key: str = ""
    function_arns: str = "{}"
    log_level: str = "INFO"
    embeddings_s3_bucket: str = ""
    embeddings_s3_prefix: str = "omnix/embeddings"
    embeddings_top_k: int = 15

    # Optional Postgres DSN (env OMNIX_DATABASE_URL). When set, the durable
    # PostgresJobStore is used for tracked jobs; when empty, jobs are kept in
    # process memory. This is a GENERIC DSN — any Postgres (local, Aurora, Neon,
    # Supabase, ...) — and intentionally carries no cloud-provider identifiers.
    database_url: str = ""

    # Optional auth plugin: a dotted "module.path:callable" that will be
    # imported at app startup. The callable is invoked with no arguments
    # and is expected to register an external API key verifier via
    # omnix.auth.api_keys.register_external_verifier. Keeps omnix-oss
    # vendor-neutral while allowing downstream deployments to plug in
    # their own key verification backend (Clerk, WorkOS, custom, ...).
    auth_plugin: str = ""

    # Optional enrichment plugin: a dotted "module.path:callable" that will
    # be imported at app startup. The callable is invoked with no arguments
    # and is expected to register paid source adapters via
    # cograph_client.enrichment.sources.base.register_adapter and override
    # tier→chain mappings via cograph_client.enrichment.tiers.register_tier.
    # Keeps cograph-oss vendor-neutral while allowing downstream deployments
    # to plug in proprietary adapters (web search, LLM, GS1, ...).
    enrichment_plugin: str = ""

    # Optional governance plugin (COG-56): a dotted "module.path:callable"
    # imported at app startup. The callable is invoked with no arguments and
    # is expected to register a mapping-shape judge panel via
    # cograph_client.resolver.governance.register_governance_panel. Without
    # it, mapping-shape proposals are recorded pending (tenant-layer-only).
    governance_plugin: str = ""

    # Optional router plugins: a comma-separated list of dotted
    # "module.path:callable" entries imported at app startup. Each callable is
    # invoked with the FastAPI app instance so it can mount additional routers
    # via app.include_router(...). Keeps cograph-oss vendor-neutral while
    # letting downstream deployments attach proprietary endpoints (e.g. the
    # premium ontology recommender). Without it, only the OSS routers are
    # mounted.
    router_plugins: str = ""

    def get_api_keys_map(self) -> dict[str, str]:
        return json.loads(self.api_keys)

    def get_function_arns_map(self) -> dict[str, str]:
        return json.loads(self.function_arns)

    model_config = {"env_prefix": "OMNIX_"}


settings = Settings()
