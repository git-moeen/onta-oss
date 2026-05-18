"""SPARQL-backed block index for entity resolution candidate lookup.

Block keys are stored as literal-valued triples next to each entity:
    <entity_uri> cog:blockKey "email_local:johnsmith" .
    <entity_uri> cog:blockKey "lastname3_phone4:smi5506" .

Normalized signals are also persisted alongside so a candidate can be
scored without a second round-trip to fetch attributes:
    <entity_uri> cog:erSignal_email "john.smith@gmail.com" .
    <entity_uri> cog:erSignal_phone "+12005551234" .
    ...

This denormalization costs ~5 triples per ER-enabled entity. The payoff is
one SPARQL query per ingest row instead of N.
"""

from __future__ import annotations

from cograph_client.resolver.er.types import BlockKey, NormalizedSignals

ER_NS = "https://cograph.tech/er/"
BLOCK_KEY_PRED = f"<{ER_NS}blockKey>"
SIGNAL_PRED_PREFIX = f"<{ER_NS}erSignal_"

# Maximum candidates a single block lookup may return. Anything more than
# this is a sign of a degenerate block key (e.g., a phone number used by
# many fake records) — bail rather than spend the scoring budget.
MAX_CANDIDATES = 50


# ---------------------------------------------------------------------------
# Soundex — tiny stdlib implementation (American Soundex, 4-char output)
# ---------------------------------------------------------------------------


_SOUNDEX_MAP = str.maketrans({
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3",
    "l": "4",
    "m": "5", "n": "5",
    "r": "6",
})


def soundex(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    first = s[0]
    coded = s.translate(_SOUNDEX_MAP)
    # Drop adjacent duplicates
    dedup: list[str] = []
    prev = ""
    for c in coded[1:]:
        if c != prev and c.isdigit():
            dedup.append(c)
        prev = c if c.isdigit() else prev
    return (first.upper() + "".join(dedup) + "000")[:4]


# ---------------------------------------------------------------------------
# Block-key generation
# ---------------------------------------------------------------------------


def generate_block_keys(normalized: NormalizedSignals) -> list[BlockKey]:
    """Emit all blocking strategies for a normalized signal bundle.

    Multiple keys per entity; a candidate matches if it shares ANY key.
    """
    keys: list[BlockKey] = []

    # Strategy 1: email local part — strongest single signal
    if normalized.email_local:
        keys.append(BlockKey("email_local", normalized.email_local))

    # Strategy 2: last-name prefix + phone last 4 — handles missing email
    if normalized.name_tokens and normalized.phone_e164:
        last = normalized.name_tokens[-1] if normalized.name_tokens else ""
        if len(last) >= 3:
            phone_last4 = normalized.phone_e164[-4:]
            keys.append(BlockKey("lastname3_phone4", f"{last[:3]}{phone_last4}"))

    # Strategy 3: soundex(last) + first initial — handles name typos
    if normalized.name_tokens and len(normalized.name_tokens) >= 2:
        first = normalized.name_tokens[0]
        last = normalized.name_tokens[-1]
        if first and last:
            keys.append(BlockKey("soundex_finit", f"{soundex(last)}{first[0]}"))

    # Strategy 4: dob + last-name prefix — handles same-name siblings, etc.
    if normalized.dob_iso and normalized.name_tokens:
        last = normalized.name_tokens[-1]
        if len(last) >= 3:
            keys.append(BlockKey("dob_lname", f"{normalized.dob_iso}_{last[:3]}"))

    return keys


# ---------------------------------------------------------------------------
# SPARQL escape (very narrow — block-key values are alphanumeric+colon)
# ---------------------------------------------------------------------------


def _quote_literal(s: str) -> str:
    """Quote an arbitrary string as a SPARQL 1.1 string literal.

    Per the SPARQL grammar (STRING_LITERAL2), inside double-quoted literals
    we escape: backslash, double-quote, line feed, carriage return, tab.
    Everything else is allowed verbatim — including spaces, colons, dots,
    and Unicode letters.

    Earlier versions of this function stripped unsafe chars assuming all
    callers passed alphanumeric+colon block-key values. Signal values
    (names, emails, addresses) routinely contain spaces and other normal
    characters, so stripping silently mangled them at index time.
    """
    escaped = (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


# ---------------------------------------------------------------------------
# SparqlBlocker
# ---------------------------------------------------------------------------


class SparqlBlocker:
    """Concrete Blocker that uses the project's NeptuneClient."""

    def __init__(self, neptune):
        self._neptune = neptune

    @staticmethod
    def block_keys(normalized: NormalizedSignals) -> list[BlockKey]:
        return generate_block_keys(normalized)

    async def candidates_with_signals(
        self,
        instance_graph: str,
        type_uri: str,
        keys: list[BlockKey],
    ) -> dict[str, NormalizedSignals]:
        """Return candidate URIs that share at least one block key, along
        with their stored NormalizedSignals (denormalized for scoring).

        Empty list of keys → empty dict (no candidates).
        """
        if not keys:
            return {}

        key_values = ",".join(_quote_literal(f"{k.kind}:{k.value}") for k in keys)
        sparql = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT DISTINCT ?entity ?p ?o
FROM <{instance_graph}>
WHERE {{
  ?entity rdf:type <{type_uri}> ;
          {BLOCK_KEY_PRED} ?key .
  FILTER(?key IN ({key_values}))
  ?entity ?p ?o .
  FILTER(STRSTARTS(STR(?p), "{ER_NS}erSignal_"))
}}
LIMIT {MAX_CANDIDATES * 8}
"""
        data = await self._neptune.query(sparql)
        rows = data.get("results", {}).get("bindings", [])

        # Reassemble (entity_uri, signal_name, signal_value) into
        # NormalizedSignals. Accumulate values per signal: after a canonical
        # has been merge-expanded one or more times, it has multiple
        # erSignal_email / erSignal_email_local triples that represent its
        # accumulated aliases. Naive overwrite loses every alias except the
        # last-written one, which silently breaks transitive matching
        # (e.g., PMS+CRM merge contributes alt-email; later Loyalty ingest
        # can't find the canonical because the alt-email isn't visible).
        per_entity: dict[str, dict[str, list[str]]] = {}
        for row in rows:
            uri = row["entity"]["value"]
            pred = row["p"]["value"]
            val = row["o"]["value"]
            signal = pred.replace(f"{ER_NS}erSignal_", "")
            sig_lists = per_entity.setdefault(uri, {})
            sig_lists.setdefault(signal, [])
            if val not in sig_lists[signal]:
                sig_lists[signal].append(val)

        out: dict[str, NormalizedSignals] = {}
        for uri, sig_map in per_entity.items():
            emails = sig_map.get("email") or []
            email_locals = sig_map.get("email_local") or []
            # First-encountered values become the "primary"; the rest become aliases.
            primary_email = emails[0] if emails else None
            primary_local = email_locals[0] if email_locals else None
            aliases = tuple(emails[1:])
            local_aliases = tuple(email_locals[1:])
            names = sig_map.get("name") or []
            name = names[0] if names else None
            tokens = tuple(name.split()) if name else ()
            addresses = sig_map.get("address") or []
            address = addresses[0] if addresses else None
            addr_tokens = tuple(address.split()) if address else ()
            phones = sig_map.get("phone_e164") or []
            dobs = sig_map.get("dob_iso") or []
            out[uri] = NormalizedSignals(
                name=name,
                name_tokens=tokens,
                email=primary_email,
                email_local=primary_local,
                email_aliases=aliases,
                email_locals=(primary_local,) + local_aliases if primary_local else local_aliases,
                phone_e164=phones[0] if phones else None,
                address=address,
                address_tokens=addr_tokens,
                dob_iso=dobs[0] if dobs else None,
            )
        # Cap to MAX_CANDIDATES (defensive — degenerate block keys)
        if len(out) > MAX_CANDIDATES:
            return dict(list(out.items())[:MAX_CANDIDATES])
        return out

    @staticmethod
    def index_triples(
        entity_uri: str,
        normalized: NormalizedSignals,
        keys: list[BlockKey],
    ) -> list[tuple[str, str, str]]:
        """Return (subject, predicate, literal) triples that should be inserted
        into the instance graph to make this entity findable by future ER runs.

        The caller batches these into the existing batched_insert_triples flow
        in schema_resolver — no new SPARQL write path needed.
        """
        # IMPORTANT: do NOT pre-quote literal values here. The downstream
        # SPARQL serializer (graph.queries._escape_value) wraps any non-URI
        # string in "..." and escapes inner quotes. Passing a pre-quoted
        # value here produces a doubly-quoted stored literal like
        # `"\"lastname3_phone4:smi5506\""` (the inner quotes become part of
        # the value), which causes every ER candidate-lookup FILTER to miss.
        # Pass raw strings; the serializer handles quoting.
        triples: list[tuple[str, str, str]] = []
        s = f"<{entity_uri}>"
        # Block keys
        for k in keys:
            triples.append((s, BLOCK_KEY_PRED, f"{k.kind}:{k.value}"))
        # Denormalized signals (for fast scoring on future lookups)
        signal_fields = [
            ("name", normalized.name),
            ("email", normalized.email),
            ("email_local", normalized.email_local),
            ("phone_e164", normalized.phone_e164),
            ("address", normalized.address),
            ("dob_iso", normalized.dob_iso),
        ]
        for name, value in signal_fields:
            if value:
                pred = f"<{ER_NS}erSignal_{name}>"
                triples.append((s, pred, value))
        return triples
