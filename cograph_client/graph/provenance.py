"""Per-fact provenance substrate (ADR 0002 §4).

Every attribute assertion can carry provenance — source, timestamp,
confidence — queryable later for conflict resolution, explainability,
and wholesale undo of a bad source.

Encoding decision (Neptune has NO RDF-star, so a triple cannot be
annotated in place): a dedicated **companion provenance named graph** per
data graph (``<data-graph>/provenance``) holding one statement-metadata
node per (fact, source) assertion. Chosen over per-source named graphs
because it composes with the existing single-data-graph layout — instance
triples stay exactly where they are, and "undo a source" / conflict
resolution become SELECTs over one graph instead of a graph-per-source
fan-out.

Keying:
- ``statement_id = sha1(s|p|o)`` identifies the *fact* (over the raw
  strings as written to Neptune, typed-literal convention included), so
  all assertions of the same fact group trivially.
- The metadata node is keyed by ``sha1(s|p|o|source)`` — one node per
  fact *per source* — so two sources asserting the same fact each carry
  their own (source, timestamp, confidence) without cross-products on
  read, and dropping a source is a single filtered DELETE.

For a fact (s, p, o) asserted by ``source`` the provenance graph holds::

    <https://cograph.tech/prov/stmt/{sha1(s|p|o|source)}>
        prov:subject    <s> ;
        prov:predicate  <p> ;
        prov:object     o ;                       # literal or URI, as written
        prov:statement  "{sha1(s|p|o)}" ;
        prov:source     "crm_export.csv" ;
        prov:confidence "1.0"^^xsd:float ;
        prov:timestamp  "2026-06-09T00:00:00+00:00"^^xsd:dateTime ;
        prov:graph      <data graph the fact lives in> .

Triples are idempotent on Neptune: re-ingesting the same fact from the
same source rewrites the same node (a refreshed timestamp accumulates as
an additional literal — last-write-wins policies resolve over max).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import _escape_value

PROV_NS = "https://cograph.tech/prov/"
PROV_SUBJECT = f"{PROV_NS}subject"
PROV_PREDICATE = f"{PROV_NS}predicate"
PROV_OBJECT = f"{PROV_NS}object"
PROV_STATEMENT = f"{PROV_NS}statement"
PROV_SOURCE = f"{PROV_NS}source"
PROV_CONFIDENCE = f"{PROV_NS}confidence"
PROV_TIMESTAMP = f"{PROV_NS}timestamp"
PROV_GRAPH = f"{PROV_NS}graph"

# Removal / rename events (ADR 0007). Assertions above record a fact ARRIVING;
# these record a fact LEAVING (``tombstone``) or a subject being RENAMED
# (``rewrite``), so governance/undo sees the full lifecycle — not just inserts.
# They live in the same companion provenance graph as assertions and are written
# by the ``delete_facts`` / ``rewrite_subject`` primitives (kg_writer.py), gated
# by ``COGRAPH_PROVENANCE_ENABLED`` exactly like assertion provenance.
PROV_EVENT = f"{PROV_NS}event"  # "tombstone" | "rewrite"
PROV_REASON = f"{PROV_NS}reason"
PROV_REWRITTEN_TO = f"{PROV_NS}rewrittenTo"  # rewrite event: old subject → new URI
PROV_AFFECTED_TYPE = f"{PROV_NS}affectedType"  # type(s) touched by the removal/rename

EVENT_TOMBSTONE = "tombstone"
EVENT_REWRITE = "rewrite"

_XSD = "http://www.w3.org/2001/XMLSchema"


def provenance_graph_uri(graph_uri: str) -> str:
    """Companion provenance graph for a data graph."""
    return f"{graph_uri}/provenance"


def statement_id(subject: str, predicate: str, obj: str) -> str:
    """Deterministic fact id: sha1 over the raw s|p|o strings as written."""
    return hashlib.sha1(f"{subject}|{predicate}|{obj}".encode("utf-8")).hexdigest()


def _assertion_uri(subject: str, predicate: str, obj: str, source: str) -> str:
    """Metadata node URI: one per (fact, source) — see module docstring."""
    aid = hashlib.sha1(f"{subject}|{predicate}|{obj}|{source}".encode("utf-8")).hexdigest()
    return f"{PROV_NS}stmt/{aid}"


def build_provenance_triples(
    subject: str,
    predicate: str,
    obj: str,
    source: str,
    confidence: float = 1.0,
    timestamp: datetime | str = "",
    graph_uri: str = "",
) -> list[tuple[str, str, str]]:
    """Build the statement-metadata triples for one fact assertion.

    Returned triples target the companion provenance graph
    (provenance_graph_uri of the data graph) — the caller inserts them
    there; the fact triple itself is untouched.

    Args:
        obj: the object exactly as written to Neptune (typed-literal
            convention included) so writer and reader agree on ids.
        confidence: 0.0-1.0; defaults to 1.0 for directly-ingested facts.
        timestamp: aware datetime or ISO-8601 string. Callers on the
            ingest path pass datetime.now(timezone.utc); tests inject
            fixed values.
        graph_uri: the DATA graph the fact lives in, recorded so a shared
            reader can scope records back to their graph.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    ts = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
    node = _assertion_uri(subject, predicate, obj, source)
    triples = [
        (node, PROV_SUBJECT, subject),
        (node, PROV_PREDICATE, predicate),
        (node, PROV_OBJECT, obj),
        (node, PROV_STATEMENT, statement_id(subject, predicate, obj)),
        (node, PROV_SOURCE, source),
        (node, PROV_CONFIDENCE, f"{confidence}^^{_XSD}#float"),
    ]
    if ts:
        triples.append((node, PROV_TIMESTAMP, f"{ts}^^{_XSD}#dateTime"))
    if graph_uri:
        triples.append((node, PROV_GRAPH, graph_uri))
    return triples


def _event_uri(event: str, subject: str, obj: str, ts: str) -> str:
    """Metadata node URI for one removal/rename event.

    Keyed by ``sha1(event|subject|obj|timestamp)`` so distinct removals of the
    same subject over time are distinct nodes (idempotent for a fixed timestamp,
    which is how tests pin them).
    """
    eid = hashlib.sha1(f"{event}|{subject}|{obj}|{ts}".encode("utf-8")).hexdigest()
    return f"{PROV_NS}event/{eid}"


def _event_common(
    node: str,
    event: str,
    subject: str,
    reason: str,
    ts: str,
    graph_uri: str,
    touched_types,
) -> list[tuple[str, str, str]]:
    triples = [
        (node, PROV_EVENT, event),
        (node, PROV_SUBJECT, subject),
    ]
    if reason:
        triples.append((node, PROV_REASON, reason))
    if ts:
        triples.append((node, PROV_TIMESTAMP, f"{ts}^^{_XSD}#dateTime"))
    if graph_uri:
        triples.append((node, PROV_GRAPH, graph_uri))
    for t in touched_types or ():
        if t:
            triples.append((node, PROV_AFFECTED_TYPE, t))
    return triples


def build_tombstone_triples(
    *,
    subjects=(),
    triples=(),
    graph_uri: str = "",
    reason: str = "",
    timestamp: datetime | str = "",
    touched_types=(),
) -> list[tuple[str, str, str]]:
    """Build the statement-metadata triples for a removal (``delete_facts``).

    One ``tombstone`` event node per removed **subject** (whole-subject delete)
    and per removed **triple** (concrete or predicate-scoped). Each records the
    subject (and predicate/object where applicable), the reason, a timestamp, the
    data graph, and any affected types — the mirror of
    :func:`build_provenance_triples`'s assertion node so an undo can see exactly
    what left the graph. ``o is None`` in a ``triples`` entry means a
    predicate-scoped removal (all objects of that ``(subject, predicate)``), so no
    ``prov:object`` is recorded. Returned triples target the companion provenance
    graph; the caller inserts them there.
    """
    ts = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
    out: list[tuple[str, str, str]] = []
    for s in subjects or ():
        if not s:
            continue
        node = _event_uri(EVENT_TOMBSTONE, s, "", ts)
        out.extend(_event_common(node, EVENT_TOMBSTONE, s, reason, ts, graph_uri, touched_types))
    for triple in triples or ():
        s, p, o = triple
        if not s:
            continue
        node = _event_uri(EVENT_TOMBSTONE, s, f"{p}|{'' if o is None else o}", ts)
        node_triples = _event_common(node, EVENT_TOMBSTONE, s, reason, ts, graph_uri, touched_types)
        if p:
            node_triples.append((node, PROV_PREDICATE, p))
        if o is not None:
            node_triples.append((node, PROV_OBJECT, o))
        out.extend(node_triples)
    return out


def build_rewrite_triples(
    old_uri: str,
    new_uri: str,
    *,
    graph_uri: str = "",
    reason: str = "",
    timestamp: datetime | str = "",
    touched_types=(),
) -> list[tuple[str, str, str]]:
    """Build the statement-metadata triples for a subject rename (``rewrite_subject``).

    One ``rewrite`` event node mapping ``old_uri → new_uri`` (``prov:rewrittenTo``)
    so governance/undo can follow an ER merge, and derived indexes have a record
    of the re-key. Returned triples target the companion provenance graph.
    """
    ts = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp
    node = _event_uri(EVENT_REWRITE, old_uri, new_uri, ts)
    out = _event_common(node, EVENT_REWRITE, old_uri, reason, ts, graph_uri, touched_types)
    out.append((node, PROV_REWRITTEN_TO, new_uri))
    return out


def provenance_query(graph_uri: str, subject: str, predicate: str | None = None, limit: int = 1000) -> str:
    """SELECT over the companion provenance graph for one subject
    (optionally narrowed to one predicate)."""
    pred_filter = f"  FILTER(?p = {_escape_value(predicate)})\n" if predicate else ""
    return (
        f"SELECT ?p ?o ?stmt ?source ?confidence ?timestamp ?graph "
        f"FROM <{provenance_graph_uri(graph_uri)}>\n"
        f"WHERE {{\n"
        f"  ?node <{PROV_SUBJECT}> {_escape_value(subject)} ;\n"
        f"        <{PROV_PREDICATE}> ?p ;\n"
        f"        <{PROV_OBJECT}> ?o ;\n"
        f"        <{PROV_STATEMENT}> ?stmt ;\n"
        f"        <{PROV_SOURCE}> ?source ;\n"
        f"        <{PROV_CONFIDENCE}> ?confidence .\n"
        f"  OPTIONAL {{ ?node <{PROV_TIMESTAMP}> ?timestamp }}\n"
        f"  OPTIONAL {{ ?node <{PROV_GRAPH}> ?graph }}\n"
        f"{pred_filter}}}\nLIMIT {limit}"
    )


@dataclass
class ProvenanceRecord:
    """One (fact, source) assertion read back from the provenance graph."""

    statement_id: str
    subject: str
    predicate: str
    obj: str
    source: str
    confidence: float
    timestamp: str
    graph: str = ""


async def fetch_provenance(
    neptune, graph_uri: str, subject: str, predicate: str | None = None,
) -> list[ProvenanceRecord]:
    """Read parsed provenance records for a subject (optionally one predicate).

    `graph_uri` is the DATA graph; the companion provenance graph is derived.
    Malformed confidence values degrade to 1.0 rather than failing the read.
    """
    raw = await neptune.query(provenance_query(graph_uri, subject, predicate))
    _, bindings = parse_sparql_results(raw)
    records: list[ProvenanceRecord] = []
    for row in bindings:
        try:
            confidence = float(row.get("confidence", "1.0"))
        except ValueError:
            confidence = 1.0
        records.append(
            ProvenanceRecord(
                statement_id=row.get("stmt", ""),
                subject=subject,
                predicate=row.get("p", ""),
                obj=row.get("o", ""),
                source=row.get("source", ""),
                confidence=confidence,
                timestamp=row.get("timestamp", ""),
                graph=row.get("graph", ""),
            )
        )
    return records
