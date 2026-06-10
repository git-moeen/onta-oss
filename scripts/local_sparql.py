"""Local SPARQL store for development — no Docker, no Java.

Wraps an embedded pyoxigraph Store behind the three HTTP paths the
`fuseki` backend of NeptuneClient expects (/ds/query, /ds/update,
/$/ping), so the cograph API server runs against it unchanged:

    python scripts/local_sparql.py                 # in-memory
    python scripts/local_sparql.py --data ./graph  # persisted to disk

then start the API with:

    OMNIX_GRAPH_BACKEND=fuseki OMNIX_NEPTUNE_ENDPOINT=http://localhost:3030 \
        uvicorn cograph_client.api.app:app --port 8000

Queries run with the default graph as the union of all named graphs,
matching Neptune's semantics (the production backend).
"""

from __future__ import annotations

import argparse

import uvicorn
from fastapi import FastAPI, Form, Response
from pyoxigraph import QueryBoolean, QueryResultsFormat, RdfFormat, Store

app = FastAPI(title="cograph local SPARQL store")
store: Store


@app.get("/$/ping")
def ping() -> dict:
    return {"status": "ok"}


@app.post("/ds/query")
def query(query: str = Form(...)) -> Response:
    results = store.query(query, use_default_graph_as_union=True)
    if isinstance(results, QueryBoolean):
        payload = results.serialize(format=QueryResultsFormat.JSON)
        return Response(payload, media_type="application/sparql-results+json")
    if hasattr(results, "variables"):  # SELECT -> QuerySolutions
        payload = results.serialize(format=QueryResultsFormat.JSON)
        return Response(payload, media_type="application/sparql-results+json")
    # CONSTRUCT / DESCRIBE -> QueryTriples
    payload = results.serialize(format=RdfFormat.N_TRIPLES)
    return Response(payload, media_type="application/n-triples")


@app.post("/ds/update")
def update(update: str = Form(...)) -> dict:
    store.update(update)
    return {"status": "ok"}


def main() -> None:
    global store
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=3030)
    parser.add_argument(
        "--data", default=None,
        help="Directory to persist the store (default: in-memory)",
    )
    args = parser.parse_args()
    store = Store(args.data) if args.data else Store()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
