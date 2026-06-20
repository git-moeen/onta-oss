from fastapi import Request

from cograph_client.enrichment.cache import get_enrichment_cache
from cograph_client.enrichment.executor import EnrichmentExecutor
from cograph_client.enrichment.job_store import make_job_store
from cograph_client.enrichment.sources.wikidata import WikidataAdapter
from cograph_client.graph.client import NeptuneClient


def get_neptune_client(request: Request) -> NeptuneClient:
    return request.app.state.neptune_client


def _ensure_enrichment_state(state) -> None:
    if getattr(state, "enrichment_job_store", None) is None:
        state.enrichment_job_store = make_job_store()
    if getattr(state, "enrichment_cache", None) is None:
        state.enrichment_cache = get_enrichment_cache()
    if getattr(state, "enrichment_wikidata", None) is None:
        state.enrichment_wikidata = WikidataAdapter()
    if getattr(state, "enrichment_executor", None) is None:
        state.enrichment_executor = EnrichmentExecutor(
            neptune_client=state.neptune_client,
            job_store=state.enrichment_job_store,
            cache=state.enrichment_cache,
            wikidata_adapter=state.enrichment_wikidata,
        )


def get_executor(request: Request) -> EnrichmentExecutor:
    """Lazily build and stash the enrichment executor on app.state."""
    _ensure_enrichment_state(request.app.state)
    return request.app.state.enrichment_executor


def get_enrichment_job_store(request: Request):
    _ensure_enrichment_state(request.app.state)
    return request.app.state.enrichment_job_store
