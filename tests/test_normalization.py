"""Tests for the inferred data-normalization subsystem.

Three layers:
  1. Rule store roundtrip (save -> list/get -> update_status) against a fake
     Neptune that maintains a real quad store and evaluates the handful of SPARQL
     shapes the store/executor emit.
  2. Inference: stub openrouter_chat and assert a ranked suggested rule is built
     for a delimited sample, and NO rule for atomic samples.
  3. Execution (the important one): seed composite `speaks` edges, apply_rule, and
     assert canonical-IRI dedup, edge rewrite, composite drop, and idempotency.
"""

from __future__ import annotations

import re

import pytest

from cograph_client.normalization import execute as execute_mod
from cograph_client.normalization import inference as inference_mod
from cograph_client.normalization.execute import apply_rule
from cograph_client.normalization.inference import suggest_rules
from cograph_client.normalization.rules import (
    NormalizationRule,
    NormalizationRuleStore,
    make_rule_id,
)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
ENTITY = "https://cograph.tech/entities/"
TYPES = "https://cograph.tech/types/"
ONTO = "https://cograph.tech/onto/"

TENANT = "t1"
KG = "june-16"


# --------------------------------------------------------------------------- #
# Fake Neptune: a tiny in-memory quad store that understands the specific SPARQL
# shapes our store + executor emit. Not a general SPARQL engine — just enough.
# --------------------------------------------------------------------------- #
class FakeNeptune:
    def __init__(self):
        # quads keyed by graph -> set of (s, p, o). o is a string; for a literal
        # we just store the lexical value (the store/executor never round-trip a
        # datatype that matters to these tests).
        self.graphs: dict[str, set[tuple[str, str, str]]] = {}

    # -- helpers --
    def _g(self, uri: str) -> set:
        return self.graphs.setdefault(uri, set())

    @staticmethod
    def _graph_in(sparql: str) -> str:
        m = re.search(r"GRAPH <([^>]+)>", sparql) or re.search(r"FROM <([^>]+)>", sparql)
        return m.group(1) if m else ""

    # -- SPARQL execution --
    async def update(self, sparql: str) -> None:
        # Handle the multi-op store save (DELETE-by-subject then INSERT DATA) and
        # the executor's INSERT DATA / DELETE DATA / DELETE...WHERE forms.
        for op in _split_ops(sparql):
            op = op.strip()
            if not op:
                continue
            if op.startswith("INSERT DATA"):
                self._apply_insert_data(op)
            elif op.startswith("DELETE DATA"):
                self._apply_delete_data(op)
            elif op.startswith("DELETE"):
                self._apply_delete_where(op)

    async def query(self, sparql: str) -> dict:
        graph = self._graph_in(sparql)
        quads = self._g(graph)
        rows = self._eval_select(sparql, quads)
        variables: list[str] = []
        for r in rows:
            for k in r:
                if k not in variables:
                    variables.append(k)
        return {
            "head": {"vars": variables},
            "results": {
                "bindings": [{k: {"value": v} for k, v in r.items()} for r in rows]
            },
        }

    async def ask(self, sparql: str) -> bool:
        graph = self._graph_in(sparql)
        quads = self._g(graph)
        # The only ASK we emit: is <composite> still the object of an inbound
        # onto/<pred> (or …/attrs/<pred>) edge?
        m = re.search(r"<([^>]+)> }\s*\n\s*UNION", sparql) or re.search(
            r"\?s <([^>]+)> <([^>]+)>", sparql
        )
        # Pull the composite URI (object) and the onto predicate from the query.
        comp_m = re.search(r"\?s <([^>]+)> <([^>]+)> }", sparql)
        if not comp_m:
            return False
        onto_pred, composite = comp_m.group(1), comp_m.group(2)
        suffix_m = re.search(r'STRENDS\(STR\(\?p2\), "([^"]+)"\)', sparql)
        suffix = suffix_m.group(1) if suffix_m else None
        for (s, p, o) in quads:
            if o != composite:
                continue
            if p == onto_pred:
                return True
            if suffix and p.endswith(suffix):
                return True
        return False

    # -- update impl --
    def _apply_insert_data(self, op: str) -> None:
        graph = self._graph_in(op)
        for t in _parse_triples(op):
            self._g(graph).add(t)

    def _apply_delete_data(self, op: str) -> None:
        graph = self._graph_in(op)
        for t in _parse_triples(op):
            self._g(graph).discard(t)

    def _apply_delete_where(self, op: str) -> None:
        # Forms: DELETE { GRAPH <g> { <subj> ?p ?o } } WHERE { ... }
        graph = self._graph_in(op)
        subj_m = re.search(r"DELETE \{ GRAPH <[^>]+> \{ <([^>]+)> \?p \?o", op)
        if subj_m:
            subj = subj_m.group(1)
            for t in list(self._g(graph)):
                if t[0] == subj:
                    self._g(graph).discard(t)

    # -- select impl (only the shapes we emit) --
    def _eval_select(self, sparql: str, quads: set) -> list[dict]:
        # Store.get: SELECT ?p ?o WHERE { <uri> ?p ?o }
        m = re.search(r"SELECT \?p \?o FROM <[^>]+> WHERE \{\s*<([^>]+)> \?p \?o", sparql)
        if m:
            uri = m.group(1)
            return [{"p": p, "o": o} for (s, p, o) in quads if s == uri]

        # Store.list: SELECT ?s ?p ?o ... ?s rdf:type <RULE_TYPE> [filters] ?s ?p ?o
        if "SELECT ?s ?p ?o" in sparql and "NormalizationRule" in sparql:
            rule_type_uri = "https://cograph.tech/types/NormalizationRule"
            subjects = {s for (s, p, o) in quads if p == RDF_TYPE and o == rule_type_uri}
            # Apply literal-equality filters: ?s <P> "v" .
            for fp, fv in re.findall(r'\?s <([^>]+)> "([^"]*)" \.', sparql):
                subjects = {s for s in subjects if (s, fp, fv) in quads}
            out = []
            for (s, p, o) in quads:
                if s in subjects:
                    out.append({"s": s, "p": p, "o": o})
            return out

        # Inference._list_predicates: SELECT ?attr ?range ... ?attr a Property ; domain <t>
        if "SELECT ?attr ?range" in sparql:
            dom_m = re.search(r"<{0}#domain> <([^>]+)>".format(re.escape("http://www.w3.org/2000/01/rdf-schema")), sparql)
            domain = dom_m.group(1) if dom_m else None
            prop = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
            rng_pred = "http://www.w3.org/2000/01/rdf-schema#range"
            dom_pred = "http://www.w3.org/2000/01/rdf-schema#domain"
            attrs = {
                s for (s, p, o) in quads if p == RDF_TYPE and o == prop
            }
            out = []
            for a in attrs:
                if domain and (a, dom_pred, domain) not in quads:
                    continue
                rng = next((o for (s, p, o) in quads if s == a and p == rng_pred), "")
                out.append({"attr": a, "range": rng})
            return out

        # Inference._sample_values: SELECT DISTINCT ?v ...
        if "SELECT DISTINCT ?v" in sparql:
            return self._eval_sample(sparql, quads)

        # Executor _explode_relationship: SELECT ?s ?p ?composite ?clabel
        if "SELECT ?s ?p ?composite ?clabel" in sparql:
            return self._eval_explode_rel(sparql, quads)

        # Executor _strip_emoji: SELECT ?s ?p ?o ... isLiteral, NO CONTAINS filter.
        if "isLiteral(?o)" in sparql and "CONTAINS(STR(?o)" not in sparql:
            return self._eval_strip_emoji(sparql, quads)

        # Executor _explode_literal: SELECT ?s ?p ?o ... isLiteral + CONTAINS.
        if "isLiteral(?o)" in sparql:
            return self._eval_explode_lit(sparql, quads)

        return []

    def _eval_sample(self, sparql: str, quads: set) -> list[dict]:
        # match ?e rdf:type <t> and ?e <pred> ?o (onto or attr form via UNION)
        t_m = re.search(r"\?e <[^>]+#type> <([^>]+)>", sparql)
        t_uri = t_m.group(1) if t_m else None
        preds = re.findall(r"\?e <([^>]+)> \?o", sparql)
        ents = {s for (s, p, o) in quads if p == RDF_TYPE and o == t_uri}
        vals: list[str] = []
        seen = set()
        is_rel = "?o <" in sparql and "?lbl" in sparql
        for (s, p, o) in quads:
            if s in ents and p in preds:
                if is_rel:
                    lbl = next((oo for (ss, pp, oo) in quads if ss == o and pp == RDFS_LABEL), None)
                    v = lbl if lbl is not None else o.rstrip("/").split("/")[-1]
                else:
                    v = o
                if v and v not in seen:
                    seen.add(v)
                    vals.append(v)
        # respect a small LIMIT/OFFSET window so multiple draws don't all repeat
        off_m = re.search(r"OFFSET (\d+)", sparql)
        lim_m = re.search(r"LIMIT (\d+)", sparql)
        off = int(off_m.group(1)) if off_m else 0
        lim = int(lim_m.group(1)) if lim_m else len(vals)
        return [{"v": v} for v in vals[off : off + lim]]

    def _eval_explode_rel(self, sparql: str, quads: set) -> list[dict]:
        onto_m = re.search(r"\?p = <([^>]+)>", sparql)
        onto_pred = onto_m.group(1) if onto_m else None
        suffix_m = re.search(r'STRENDS\(STR\(\?p\), "([^"]+)"\)', sparql)
        suffix = suffix_m.group(1) if suffix_m else None
        delims = re.findall(r'CONTAINS\(\?cname, "([^"]+)"\)', sparql)
        out = []
        for (s, p, o) in quads:
            if not (p == onto_pred or (suffix and p.endswith(suffix))):
                continue
            if not o.startswith(ENTITY):
                continue
            clabel = next((oo for (ss, pp, oo) in quads if ss == o and pp == RDFS_LABEL), None)
            cname = clabel if clabel is not None else o.rstrip("/").split("/")[-1]
            if any(_unescape(d) in cname for d in delims):
                row = {"s": s, "p": p, "composite": o}
                if clabel is not None:
                    row["clabel"] = clabel
                out.append(row)
        return out

    def _eval_explode_lit(self, sparql: str, quads: set) -> list[dict]:
        onto_m = re.search(r"\?p = <([^>]+)>", sparql)
        onto_pred = onto_m.group(1) if onto_m else None
        suffix_m = re.search(r'STRENDS\(STR\(\?p\), "([^"]+)"\)', sparql)
        suffix = suffix_m.group(1) if suffix_m else None
        delims = re.findall(r'CONTAINS\(STR\(\?o\), "([^"]+)"\)', sparql)
        out = []
        for (s, p, o) in quads:
            if not (p == onto_pred or (suffix and p.endswith(suffix))):
                continue
            if o.startswith("http://") or o.startswith("https://"):
                continue  # treat URIs as non-literal
            if any(_unescape(d) in o for d in delims):
                out.append({"s": s, "p": p, "o": o})
        return out

    def _eval_strip_emoji(self, sparql: str, quads: set) -> list[dict]:
        # SELECT ?s ?p ?o for the predicate's LITERAL objects (no delimiter
        # filter — strip_emoji cleans in Python).
        onto_m = re.search(r"\?p = <([^>]+)>", sparql)
        onto_pred = onto_m.group(1) if onto_m else None
        suffix_m = re.search(r'STRENDS\(STR\(\?p\), "([^"]+)"\)', sparql)
        suffix = suffix_m.group(1) if suffix_m else None
        out = []
        for (s, p, o) in quads:
            if not (p == onto_pred or (suffix and p.endswith(suffix))):
                continue
            if o.startswith("http://") or o.startswith("https://"):
                continue  # treat URIs as non-literal
            out.append({"s": s, "p": p, "o": o})
        return out


# --------------------------------------------------------------------------- #
# Minimal SPARQL-string helpers for the fake.
# --------------------------------------------------------------------------- #
def _split_ops(sparql: str) -> list[str]:
    # Split a multi-op update on top-level ';' separating INSERT/DELETE blocks.
    # Our store emits "DELETE...WHERE" then (separately awaited) INSERT DATA, and
    # the multi-op upsert is joined by ';'. Splitting on ';\n' is sufficient here.
    parts = re.split(r";\s*\n", sparql)
    return parts if len(parts) > 1 else [sparql]


def _parse_triples(op: str) -> list[tuple[str, str, str]]:
    """Parse the body of an INSERT/DELETE DATA block into (s, p, o) triples."""
    body_m = re.search(r"GRAPH <[^>]+> \{(.*)\}\s*\}", op, re.DOTALL)
    if not body_m:
        return []
    body = body_m.group(1)
    triples = []
    for line in body.split("\n"):
        line = line.strip().rstrip(".").strip()
        if not line:
            continue
        t = _parse_triple_line(line)
        if t:
            triples.append(t)
    return triples


_TERM = r'(<[^>]+>|"(?:[^"\\]|\\.)*"(?:\^\^<[^>]+>)?)'


def _parse_triple_line(line: str):
    m = re.match(rf"^{_TERM}\s+{_TERM}\s+(.*)$", line)
    if not m:
        return None
    s = _term(m.group(1))
    p = _term(m.group(2))
    o = _term(m.group(3).strip())
    return (s, p, o)


def _term(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("<") and raw.endswith(">"):
        return raw[1:-1]
    if raw.startswith('"'):
        # strip optional ^^<...> datatype, then the quotes, then unescape.
        if "^^" in raw:
            raw = raw.rsplit("^^", 1)[0].strip()
        inner = raw[1:-1] if raw.endswith('"') else raw[1:]
        return inner.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")
    return raw


def _unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")


# --------------------------------------------------------------------------- #
# 1. Rule store roundtrip
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_rule_store_roundtrip():
    neptune = FakeNeptune()
    store = NormalizationRuleStore(neptune)
    rule = NormalizationRule(
        id=make_rule_id(KG, "Mentor", "speaks"),
        kg_name=KG,
        type_name="Mentor",
        predicate="speaks",
        target_kind="relationship",
        rule_type="list_explode",
        params={"delimiters": [", ", "__"], "target": "entity"},
        confidence=0.92,
        rationale="composite Language entities",
        sample_values=["English__Russian", "English__Persian"],
        status="suggested",
    )
    await store.save(TENANT, rule)

    got = await store.get(TENANT, rule.id)
    assert got is not None
    assert got.kg_name == KG
    assert got.predicate == "speaks"
    assert got.target_kind == "relationship"
    assert got.params == {"delimiters": [", ", "__"], "target": "entity"}
    assert got.confidence == pytest.approx(0.92)
    assert got.sample_values == ["English__Russian", "English__Persian"]
    assert got.status == "suggested"

    listed = await store.list(TENANT, kg=KG, status="suggested")
    assert len(listed) == 1
    assert listed[0].id == rule.id

    # Wrong-status filter returns nothing.
    assert await store.list(TENANT, status="confirmed") == []

    # update_status flips status (and is idempotent on re-save: no dup triples).
    updated = await store.update_status(TENANT, rule.id, "confirmed")
    assert updated is not None and updated.status == "confirmed"
    again = await store.get(TENANT, rule.id)
    assert again.status == "confirmed"
    # exactly one status triple after the flip (no stale "suggested" left behind)
    graph = neptune.graphs["https://cograph.tech/graphs/t1"]
    status_triples = [t for t in graph if t[0] == rule.uri and t[1].endswith("/status")]
    assert len(status_triples) == 1 and status_triples[0][2] == "confirmed"

    # update_status of a missing rule returns None.
    assert await store.update_status(TENANT, "nope", "applied") is None


# --------------------------------------------------------------------------- #
# 2. Inference
# --------------------------------------------------------------------------- #
def _seed_mentor_ontology(neptune: FakeNeptune, *, rng: str):
    onto = "https://cograph.tech/graphs/t1"
    prop = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
    dom = "http://www.w3.org/2000/01/rdf-schema#domain"
    rngp = "http://www.w3.org/2000/01/rdf-schema#range"
    attr = TYPES + "Mentor/attrs/speaks"
    neptune._g(onto).update(
        {
            (attr, RDF_TYPE, prop),
            (attr, dom, TYPES + "Mentor"),
            (attr, rngp, rng),
        }
    )


def _seed_mentor_instances(neptune: FakeNeptune, objects: list[str], *, pred: str):
    kg = "https://cograph.tech/graphs/t1/kg/june-16"
    for i, obj in enumerate(objects):
        e = ENTITY + f"Mentor/m{i}"
        neptune._g(kg).add((e, RDF_TYPE, TYPES + "Mentor"))
        neptune._g(kg).add((e, pred, obj))


@pytest.mark.asyncio
async def test_inference_suggests_for_delimited(monkeypatch):
    neptune = FakeNeptune()
    _seed_mentor_ontology(neptune, rng=TYPES + "Language")  # relationship
    # composite Language entities (the speaks case)
    for name in ["English__Russian", "English__Persian", "English__Ukrainian"]:
        kg = "https://cograph.tech/graphs/t1/kg/june-16"
        neptune._g(kg).add((ENTITY + f"Language/{name}", RDFS_LABEL, name.replace("__", ", ")))
    _seed_mentor_instances(
        neptune,
        [ENTITY + "Language/English__Russian", ENTITY + "Language/English__Persian"],
        pred=ONTO + "speaks",
    )

    async def fake_chat(api_key, system, user, **kw):
        return (
            '{"needs_normalization": true, "rule_type": "list_explode", '
            '"params": {"delimiters": [", ", "__"], "target": "entity"}, '
            '"confidence": 0.9, "rationale": "delimited composite Language values"}'
        )

    monkeypatch.setattr(inference_mod, "openrouter_chat", fake_chat)
    monkeypatch.setattr(inference_mod, "_openrouter_key", lambda: "key")

    rules = await suggest_rules(neptune, TENANT, KG, "Mentor")
    assert len(rules) == 1
    r = rules[0]
    assert r.rule_type == "list_explode"
    assert r.target_kind == "relationship"
    assert r.predicate == "speaks"
    assert r.status == "suggested"
    assert r.confidence == pytest.approx(0.9)
    assert "__" in r.params["delimiters"]


@pytest.mark.asyncio
async def test_inference_no_rule_for_atomic(monkeypatch):
    neptune = FakeNeptune()
    _seed_mentor_ontology(neptune, rng="http://www.w3.org/2001/XMLSchema#string")  # attribute
    _seed_mentor_instances(
        neptune,
        ["English", "Russian", "Mandarin Chinese", "Bahasa Indonesia"],
        pred=ONTO + "speaks",
    )

    async def fake_chat(api_key, system, user, **kw):
        # The LLM (correctly) declines: values are atomic / single multi-word.
        return (
            '{"needs_normalization": false, "rule_type": null, '
            '"params": {}, "confidence": 0.1, "rationale": "values already atomic"}'
        )

    monkeypatch.setattr(inference_mod, "openrouter_chat", fake_chat)
    monkeypatch.setattr(inference_mod, "_openrouter_key", lambda: "key")

    rules = await suggest_rules(neptune, TENANT, KG, "Mentor")
    assert rules == []


@pytest.mark.asyncio
async def test_inference_ranks_by_confidence(monkeypatch):
    """Two predicates with different confidence come back highest-first."""
    neptune = FakeNeptune()
    onto = "https://cograph.tech/graphs/t1"
    prop = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
    dom = "http://www.w3.org/2000/01/rdf-schema#domain"
    rngp = "http://www.w3.org/2000/01/rdf-schema#range"
    for leaf, rng in [("speaks", TYPES + "Language"), ("skills", "http://www.w3.org/2001/XMLSchema#string")]:
        a = TYPES + f"Mentor/attrs/{leaf}"
        neptune._g(onto).update({(a, RDF_TYPE, prop), (a, dom, TYPES + "Mentor"), (a, rngp, rng)})
    kg = "https://cograph.tech/graphs/t1/kg/june-16"
    neptune._g(kg).add((ENTITY + "Mentor/m0", RDF_TYPE, TYPES + "Mentor"))
    neptune._g(kg).add((ENTITY + "Mentor/m0", ONTO + "speaks", ENTITY + "Language/English__Russian"))
    neptune._g(kg).add((ENTITY + "Language/English__Russian", RDFS_LABEL, "English, Russian"))
    neptune._g(kg).add((ENTITY + "Mentor/m0", ONTO + "skills", "Python; SQL"))

    async def fake_chat(api_key, system, user, **kw):
        conf = 0.95 if "skills" in user else 0.6
        target = "literal" if "skills" in user else "entity"
        return (
            '{"needs_normalization": true, "rule_type": "list_explode", '
            f'"params": {{"delimiters": ["; ", ", ", "__"], "target": "{target}"}}, '
            f'"confidence": {conf}, "rationale": "delimited"}}'
        )

    monkeypatch.setattr(inference_mod, "openrouter_chat", fake_chat)
    monkeypatch.setattr(inference_mod, "_openrouter_key", lambda: "key")

    rules = await suggest_rules(neptune, TENANT, KG, "Mentor")
    assert [r.predicate for r in rules] == ["skills", "speaks"]
    assert rules[0].confidence > rules[1].confidence


# --------------------------------------------------------------------------- #
# 3. Execution — the important one.
# --------------------------------------------------------------------------- #
def _seed_speaks_composites(neptune: FakeNeptune):
    kg = "https://cograph.tech/graphs/t1/kg/june-16"
    speaks = ONTO + "speaks"
    # mentorA speaks English__Russian ; mentorB speaks English__Persian
    comp_ar = ENTITY + "Language/English__Russian"
    comp_ap = ENTITY + "Language/English__Persian"
    neptune._g(kg).update(
        {
            (ENTITY + "Mentor/A", RDF_TYPE, TYPES + "Mentor"),
            (ENTITY + "Mentor/B", RDF_TYPE, TYPES + "Mentor"),
            (ENTITY + "Mentor/A", speaks, comp_ar),
            (ENTITY + "Mentor/B", speaks, comp_ap),
            # composite Language nodes with a human label
            (comp_ar, RDF_TYPE, TYPES + "Language"),
            (comp_ar, RDFS_LABEL, "English, Russian"),
            (comp_ap, RDF_TYPE, TYPES + "Language"),
            (comp_ap, RDFS_LABEL, "English, Persian"),
        }
    )
    return kg


@pytest.mark.asyncio
async def test_execute_explode_relationship_and_idempotent():
    neptune = FakeNeptune()
    kg = _seed_speaks_composites(neptune)
    rule = NormalizationRule(
        id=make_rule_id(KG, "Mentor", "speaks"),
        kg_name=KG,
        type_name="Mentor",
        predicate="speaks",
        target_kind="relationship",
        rule_type="list_explode",
        params={"delimiters": [", ", "__"], "target": "entity"},
        confidence=0.9,
        status="confirmed",
    )

    summary = await apply_rule(neptune, TENANT, rule)

    speaks = ONTO + "speaks"
    quads = neptune.graphs[kg]
    eng = ENTITY + "Language/English"
    rus = ENTITY + "Language/Russian"
    per = ENTITY + "Language/Persian"
    comp_ar = ENTITY + "Language/English__Russian"
    comp_ap = ENTITY + "Language/English__Persian"

    # atomic entities exist, English is ONE shared node
    assert (eng, RDF_TYPE, TYPES + "Language") in quads
    assert (rus, RDF_TYPE, TYPES + "Language") in quads
    assert (per, RDF_TYPE, TYPES + "Language") in quads
    assert (eng, RDFS_LABEL, "English") in quads
    # exactly one English node (canonical-IRI dedup): only one rdf:type triple
    eng_types = [t for t in quads if t[0] == eng and t[1] == RDF_TYPE]
    assert len(eng_types) == 1

    # each mentor has the right atomic edges
    assert (ENTITY + "Mentor/A", speaks, eng) in quads
    assert (ENTITY + "Mentor/A", speaks, rus) in quads
    assert (ENTITY + "Mentor/B", speaks, eng) in quads
    assert (ENTITY + "Mentor/B", speaks, per) in quads

    # composite edges are gone
    assert (ENTITY + "Mentor/A", speaks, comp_ar) not in quads
    assert (ENTITY + "Mentor/B", speaks, comp_ap) not in quads

    # composite nodes dropped (no triples left for them)
    assert not [t for t in quads if t[0] == comp_ar]
    assert not [t for t in quads if t[0] == comp_ap]

    assert summary["edges_rewritten"] == 2
    assert summary["atomic_created"] == 3  # English, Russian, Persian
    assert summary["orphans_dropped"] == 2

    # idempotent: re-running finds nothing to split -> no-op
    before = set(neptune.graphs[kg])
    summary2 = await apply_rule(neptune, TENANT, rule)
    assert neptune.graphs[kg] == before
    assert summary2 == {"edges_rewritten": 0, "atomic_created": 0, "orphans_dropped": 0}


@pytest.mark.asyncio
async def test_execute_explode_literal():
    neptune = FakeNeptune()
    kg = "https://cograph.tech/graphs/t1/kg/june-16"
    skills = ONTO + "skills"
    neptune._g(kg).update(
        {
            (ENTITY + "Mentor/A", RDF_TYPE, TYPES + "Mentor"),
            (ENTITY + "Mentor/A", skills, "Python; SQL; Go"),
        }
    )
    rule = NormalizationRule(
        id=make_rule_id(KG, "Mentor", "skills"),
        kg_name=KG,
        type_name="Mentor",
        predicate="skills",
        target_kind="attribute",
        rule_type="list_explode",
        params={"delimiters": ["; "], "target": "literal"},
        confidence=0.9,
        status="confirmed",
    )
    summary = await apply_rule(neptune, TENANT, rule)
    quads = neptune.graphs[kg]
    assert (ENTITY + "Mentor/A", skills, "Python") in quads
    assert (ENTITY + "Mentor/A", skills, "SQL") in quads
    assert (ENTITY + "Mentor/A", skills, "Go") in quads
    assert (ENTITY + "Mentor/A", skills, "Python; SQL; Go") not in quads
    assert summary["edges_rewritten"] == 1
    assert summary["atomic_created"] == 3

    # idempotent
    before = set(neptune.graphs[kg])
    await apply_rule(neptune, TENANT, rule)
    assert neptune.graphs[kg] == before


# --------------------------------------------------------------------------- #
# 3b. Execution — strip_emoji.
# --------------------------------------------------------------------------- #
def _strip_emoji_rule() -> NormalizationRule:
    return NormalizationRule(
        id=make_rule_id(KG, "Mentor", "skills", "strip_emoji"),
        kg_name=KG,
        type_name="Mentor",
        predicate="skills",
        target_kind="attribute",
        rule_type="strip_emoji",
        params={"targets": ["attribute"]},
        confidence=0.9,
        status="confirmed",
    )


@pytest.mark.asyncio
async def test_execute_strip_emoji_cleans_and_drops_pure_emoji():
    neptune = FakeNeptune()
    kg = "https://cograph.tech/graphs/t1/kg/june-16"
    skills = ONTO + "skills"
    neptune._g(kg).update(
        {
            (ENTITY + "Mentor/A", RDF_TYPE, TYPES + "Mentor"),
            (ENTITY + "Mentor/A", skills, "🎨 design"),   # leading emoji + space
            (ENTITY + "Mentor/B", skills, "ai 🚀"),        # trailing emoji + space
            (ENTITY + "Mentor/C", skills, "growth"),       # no emoji — untouched
            (ENTITY + "Mentor/D", skills, "🔥🔥"),          # pure emoji — dropped
            (ENTITY + "Mentor/E", skills, "data  📊  viz"),  # interior + double space
        }
    )

    summary = await apply_rule(neptune, TENANT, _strip_emoji_rule())
    quads = neptune.graphs[kg]

    # emoji removed, whitespace collapsed/trimmed
    assert (ENTITY + "Mentor/A", skills, "design") in quads
    assert (ENTITY + "Mentor/A", skills, "🎨 design") not in quads
    assert (ENTITY + "Mentor/B", skills, "ai") in quads
    assert (ENTITY + "Mentor/B", skills, "ai 🚀") not in quads
    assert (ENTITY + "Mentor/E", skills, "data viz") in quads

    # non-emoji value untouched
    assert (ENTITY + "Mentor/C", skills, "growth") in quads

    # pure-emoji value dropped entirely (no replacement literal)
    assert not [t for t in quads if t[0] == ENTITY + "Mentor/D" and t[1] == skills]

    assert summary == {"literals_cleaned": 4, "triples_rewritten": 4}

    # idempotent: re-running finds nothing to clean -> no-op
    before = set(neptune.graphs[kg])
    summary2 = await apply_rule(neptune, TENANT, _strip_emoji_rule())
    assert neptune.graphs[kg] == before
    assert summary2 == {"literals_cleaned": 0, "triples_rewritten": 0}


@pytest.mark.asyncio
async def test_execute_strip_emoji_preserves_real_skill_names():
    """Accented letters, digits, and punctuation that belong to real skills
    (c++, C#, Node.js, R&D, café) must survive untouched."""
    neptune = FakeNeptune()
    kg = "https://cograph.tech/graphs/t1/kg/june-16"
    skills = ONTO + "skills"
    keepers = ["c++", "C#", "Node.js", "R&D", "café", "machine-learning", "A/B testing"]
    neptune._g(kg).add((ENTITY + "Mentor/A", RDF_TYPE, TYPES + "Mentor"))
    for v in keepers:
        neptune._g(kg).add((ENTITY + "Mentor/A", skills, v))

    summary = await apply_rule(neptune, TENANT, _strip_emoji_rule())
    quads = neptune.graphs[kg]
    for v in keepers:
        assert (ENTITY + "Mentor/A", skills, v) in quads
    assert summary == {"literals_cleaned": 0, "triples_rewritten": 0}


@pytest.mark.asyncio
async def test_execute_strip_emoji_works_on_exploded_atomic_literals():
    """strip_emoji is per-literal, so it cleans atomic literals the same way it
    would clean a still-packed one (works whether or not list_explode ran)."""
    neptune = FakeNeptune()
    kg = "https://cograph.tech/graphs/t1/kg/june-16"
    skills = ONTO + "skills"
    # already exploded into atomic literals, but each still carries emoji
    neptune._g(kg).update(
        {
            (ENTITY + "Mentor/A", RDF_TYPE, TYPES + "Mentor"),
            (ENTITY + "Mentor/A", skills, "🎨 design"),
            (ENTITY + "Mentor/A", skills, "🚀 growth"),
        }
    )
    summary = await apply_rule(neptune, TENANT, _strip_emoji_rule())
    quads = neptune.graphs[kg]
    assert (ENTITY + "Mentor/A", skills, "design") in quads
    assert (ENTITY + "Mentor/A", skills, "growth") in quads
    assert summary == {"literals_cleaned": 2, "triples_rewritten": 2}


# --------------------------------------------------------------------------- #
# 3c. Inference — multi-rule per predicate + id collision.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inference_emits_multiple_rules_per_predicate(monkeypatch):
    """skills warrants BOTH list_explode AND strip_emoji -> both emitted, with
    distinct ids, ranked by confidence (desc)."""
    neptune = FakeNeptune()
    _seed_mentor_ontology(neptune, rng="http://www.w3.org/2001/XMLSchema#string")  # attribute
    _seed_mentor_instances(
        neptune,
        ["🎨 design; ai; 🚀 growth", "design; data"],
        pred=ONTO + "speaks",  # _seed_mentor_ontology declares the 'speaks' attr
    )

    async def fake_chat(api_key, system, user, **kw):
        return (
            '{"rules": ['
            '{"rule_type": "strip_emoji", "params": {"targets": ["attribute"]}, '
            '"confidence": 0.95, "rationale": "values carry emoji junk"},'
            '{"rule_type": "list_explode", '
            '"params": {"delimiters": ["; "], "target": "literal"}, '
            '"confidence": 0.8, "rationale": "semicolon-delimited list"}'
            ']}'
        )

    monkeypatch.setattr(inference_mod, "openrouter_chat", fake_chat)
    monkeypatch.setattr(inference_mod, "_openrouter_key", lambda: "key")

    rules = await suggest_rules(neptune, TENANT, KG, "Mentor")
    assert len(rules) == 2
    # ranked by confidence desc: strip_emoji (0.95) before list_explode (0.8)
    assert [r.rule_type for r in rules] == ["strip_emoji", "list_explode"]
    assert rules[0].confidence > rules[1].confidence
    # distinct ids
    assert rules[0].id != rules[1].id
    # both target the same predicate
    assert {r.predicate for r in rules} == {"speaks"}
    # params carried through per rule type
    le = next(r for r in rules if r.rule_type == "list_explode")
    se = next(r for r in rules if r.rule_type == "strip_emoji")
    assert "; " in le.params["delimiters"]
    assert se.params["targets"] == ["attribute"]


@pytest.mark.asyncio
async def test_inference_dedupes_repeated_rule_type(monkeypatch):
    """Two recommendations of the SAME rule_type collapse to one (first wins)."""
    neptune = FakeNeptune()
    _seed_mentor_ontology(neptune, rng="http://www.w3.org/2001/XMLSchema#string")
    _seed_mentor_instances(neptune, ["🎨 design", "ai 🚀"], pred=ONTO + "speaks")

    async def fake_chat(api_key, system, user, **kw):
        return (
            '{"rules": ['
            '{"rule_type": "strip_emoji", "params": {}, "confidence": 0.9, "rationale": "a"},'
            '{"rule_type": "strip_emoji", "params": {}, "confidence": 0.4, "rationale": "b"}'
            ']}'
        )

    monkeypatch.setattr(inference_mod, "openrouter_chat", fake_chat)
    monkeypatch.setattr(inference_mod, "_openrouter_key", lambda: "key")

    rules = await suggest_rules(neptune, TENANT, KG, "Mentor")
    assert len(rules) == 1
    assert rules[0].rule_type == "strip_emoji"
    assert rules[0].confidence == pytest.approx(0.9)  # first recommendation wins


def test_make_rule_id_distinguishes_rule_type():
    """list_explode and strip_emoji on the SAME predicate get DISTINCT ids; the
    list_explode id stays byte-identical to the historical (3-arg) scheme."""
    le_id = make_rule_id(KG, "Mentor", "skills", "list_explode")
    se_id = make_rule_id(KG, "Mentor", "skills", "strip_emoji")
    assert le_id != se_id
    # backward compatible: list_explode id == legacy 3-arg id (no suffix)
    assert le_id == make_rule_id(KG, "Mentor", "skills")
    # strip_emoji carries the rule_type suffix
    assert se_id.endswith("__strip_emoji")


# --------------------------------------------------------------------------- #
# 4. Route surface — suggest persists + ranks; apply guards on status.
# --------------------------------------------------------------------------- #
@pytest.fixture
def route_client(monkeypatch):
    import os

    os.environ["OMNIX_API_KEYS"] = '{"test-key": "test-tenant"}'
    os.environ["OMNIX_NEPTUNE_ENDPOINT"] = "http://fake-neptune:8182"
    from fastapi.testclient import TestClient

    from cograph_client.api.app import create_app

    neptune = FakeNeptune()
    app = create_app()
    app.state.neptune_client = neptune
    return TestClient(app), neptune


def test_route_suggest_persists_and_lists(route_client, monkeypatch):
    client, neptune = route_client
    # Seed ontology + a composite instance for the fake under the test-tenant.
    onto = "https://cograph.tech/graphs/test-tenant"
    prop = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
    dom = "http://www.w3.org/2000/01/rdf-schema#domain"
    rngp = "http://www.w3.org/2000/01/rdf-schema#range"
    a = TYPES + "Mentor/attrs/speaks"
    neptune._g(onto).update({(a, RDF_TYPE, prop), (a, dom, TYPES + "Mentor"), (a, rngp, TYPES + "Language")})
    kg = "https://cograph.tech/graphs/test-tenant/kg/june-16"
    neptune._g(kg).add((ENTITY + "Mentor/A", RDF_TYPE, TYPES + "Mentor"))
    neptune._g(kg).add((ENTITY + "Mentor/A", ONTO + "speaks", ENTITY + "Language/English__Russian"))
    neptune._g(kg).add((ENTITY + "Language/English__Russian", RDFS_LABEL, "English, Russian"))

    async def fake_chat(api_key, system, user, **kw):
        return (
            '{"needs_normalization": true, "rule_type": "list_explode", '
            '"params": {"delimiters": [", ", "__"], "target": "entity"}, '
            '"confidence": 0.88, "rationale": "composite"}'
        )

    monkeypatch.setattr(inference_mod, "openrouter_chat", fake_chat)
    monkeypatch.setattr(inference_mod, "_openrouter_key", lambda: "key")

    h = {"X-API-Key": "test-key"}
    res = client.post("/graphs/test-tenant/normalize/suggest?kg=june-16&type=Mentor", headers=h)
    assert res.status_code == 200
    rules = res.json()
    assert len(rules) == 1 and rules[0]["status"] == "suggested"
    rule_id = rules[0]["id"]

    # Persisted: GET /rules returns it.
    listed = client.get("/graphs/test-tenant/normalize/rules?kg=june-16&status=suggested", headers=h)
    assert listed.status_code == 200
    assert [r["id"] for r in listed.json()] == [rule_id]

    # apply before confirm -> 409.
    blocked = client.post(f"/graphs/test-tenant/normalize/rules/{rule_id}/apply", headers=h)
    assert blocked.status_code == 409

    # confirm -> 200 confirmed; apply -> 202.
    confirmed = client.post(f"/graphs/test-tenant/normalize/rules/{rule_id}/confirm", headers=h)
    assert confirmed.status_code == 200 and confirmed.json()["status"] == "confirmed"
    accepted = client.post(f"/graphs/test-tenant/normalize/rules/{rule_id}/apply", headers=h)
    assert accepted.status_code == 202

    # reject path on a fresh suggested rule -> rejected.
    rej = client.post(f"/graphs/test-tenant/normalize/rules/{rule_id}/reject", headers=h)
    assert rej.status_code == 200 and rej.json()["status"] == "rejected"


def test_route_apply_missing_rule_404(route_client):
    client, _ = route_client
    h = {"X-API-Key": "test-key"}
    res = client.post("/graphs/test-tenant/normalize/rules/does-not-exist/apply", headers=h)
    assert res.status_code == 404
