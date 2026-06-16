from cograph_client.graph.ontology_queries import (
    insert_type,
    insert_attribute,
    insert_subtype,
    list_types_query,
    get_type_detail_query,
    get_type_attributes_query,
    get_subtypes_query,
    get_type_functions_query,
    get_full_ontology_query,
    mark_core_slot,
    retract_object_property,
    set_object_property_range,
    type_uri,
    attr_uri,
)

GRAPH = "https://cograph.tech/graphs/test"


def test_type_uri():
    assert type_uri("Place") == "https://cograph.tech/types/Place"


def test_attr_uri():
    assert attr_uri("Place", "name") == "https://cograph.tech/types/Place/attrs/name"


def test_insert_type_basic():
    sparql = insert_type(GRAPH, "Place")
    assert "INSERT DATA" in sparql
    assert "GRAPH <https://cograph.tech/graphs/test>" in sparql
    assert "cograph.tech/types/Place" in sparql
    assert "Class" in sparql
    assert '"Place"' in sparql


def test_insert_type_with_description():
    sparql = insert_type(GRAPH, "Place", description="A geographic location")
    assert "A geographic location" in sparql


def test_insert_type_with_parent():
    sparql = insert_type(GRAPH, "Park", parent_type="Place")
    assert "subClassOf" in sparql
    assert "cograph.tech/types/Place" in sparql


def test_insert_attribute():
    sparql = insert_attribute(GRAPH, "Place", "name", "The name", "string")
    assert "INSERT DATA" in sparql
    assert "cograph.tech/types/Place/attrs/name" in sparql
    assert "Property" in sparql
    assert "domain" in sparql
    assert "range" in sparql
    assert "string" in sparql


def test_insert_attribute_datetime():
    sparql = insert_attribute(GRAPH, "Event", "startDate", datatype="datetime")
    assert "dateTime" in sparql


def test_mark_core_slot():
    # ADR 0003 §3 / COG-52: a constitutive slot gets a coreSlot boolean
    # triple on its attribute URI so enrichment can query "instances with
    # empty core slots" as its work queue.
    sparql = mark_core_slot(GRAPH, "SKU", "issued_by")
    assert "INSERT DATA" in sparql
    assert "GRAPH <https://cograph.tech/graphs/test>" in sparql
    assert "cograph.tech/types/SKU/attrs/issued_by" in sparql
    assert "cograph.tech/onto/coreSlot" in sparql
    assert '"true"^^<http://www.w3.org/2001/XMLSchema#boolean>' in sparql


def test_mark_core_slot_targets_attr_uri():
    # The marker hangs off the same attribute URI insert_attribute declares,
    # so the two writes compose into one ontology entry.
    sparql = mark_core_slot(GRAPH, "Place", "name")
    assert f"<{attr_uri('Place', 'name')}>" in sparql


def test_insert_subtype():
    sparql = insert_subtype(GRAPH, "Place", "Park")
    assert "subClassOf" in sparql
    assert "cograph.tech/types/Park" in sparql
    assert "cograph.tech/types/Place" in sparql


def test_list_types_query():
    sparql = list_types_query(GRAPH)
    assert "SELECT" in sparql
    assert "Class" in sparql
    assert "FROM <https://cograph.tech/graphs/test>" in sparql


def test_get_type_detail_query():
    sparql = get_type_detail_query(GRAPH, "Place")
    assert "cograph.tech/types/Place" in sparql
    assert "label" in sparql


def test_get_type_attributes_query():
    sparql = get_type_attributes_query(GRAPH, "Place")
    assert "domain" in sparql
    assert "cograph.tech/types/Place" in sparql


def test_get_subtypes_query():
    sparql = get_subtypes_query(GRAPH, "Place")
    assert "subClassOf" in sparql


def test_get_type_functions_query():
    sparql = get_type_functions_query(GRAPH, "Place")
    assert "attachedTo" in sparql
    assert "cograph.tech/types/Place" in sparql


def test_get_full_ontology_query():
    sparql = get_full_ontology_query(GRAPH)
    assert "Class" in sparql
    assert "domain" in sparql
    assert "attachedTo" in sparql


def test_set_object_property_range():
    # Upgrading a predicate's range to a type URI must delete any existing range
    # (so the property keeps exactly one) and insert the types/ target.
    sparql = set_object_property_range(GRAPH, "RetailerSKU", "identifies", "Product")
    assert "DELETE" in sparql and "INSERT" in sparql
    assert f"<{GRAPH}>" in sparql
    assert attr_uri("RetailerSKU", "identifies") in sparql
    assert type_uri("Product") in sparql
    assert "range" in sparql
    # The old range is matched optionally so a predicate with no range yet still upgrades.
    assert "OPTIONAL" in sparql


def test_retract_object_property():
    # ADR 0004 §4 reconciliation: a quarantined relationship must stop being a
    # declared type-level edge, so its rdfs:range AND rdfs:domain are deleted.
    sparql = retract_object_property(GRAPH, "ManufacturerPartNumber", "issuedby")
    assert "DELETE" in sparql
    assert f"<{GRAPH}>" in sparql
    assert attr_uri("ManufacturerPartNumber", "issuedby") in sparql
    assert "range" in sparql
    assert "domain" in sparql
    # OPTIONAL match makes the retraction no-op-safe (missing range/domain is fine).
    assert "OPTIONAL" in sparql
