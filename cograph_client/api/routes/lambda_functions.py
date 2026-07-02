"""Lambda function endpoints — tier-2 HTTP lambdas and invoke/materialize.

Delivers two capabilities:
1. Concrete tier-2 lambda endpoints (e.g. SEC EDGAR latest-filing lookup)
2. A generic invoke endpoint that runs a registered function against an entity
   and materializes the output as triples on that entity in the KG.
"""

import datetime
import time

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from cograph_client.api.deps import get_neptune_client
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.functions.executor import FunctionExecutor
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.kg_writer import delete_facts, insert_facts, refresh_after_write
from cograph_client.graph.ontology_queries import insert_attribute
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import (
    kg_graph_uri,
    list_functions_query,
    tenant_graph_uri,
)
from cograph_client.models.function import FunctionRef, FunctionTier

logger = structlog.stdlib.get_logger("cograph.lambda_functions")

router = APIRouter()

# ---------------------------------------------------------------------------
# Tier-2 Lambda: SEC EDGAR latest-filing
# ---------------------------------------------------------------------------

SEC_USER_AGENT = "cograph-demo smoeenmh@gmail.com"


class SECFilingRequest(BaseModel):
    cik: str


class SECFilingResponse(BaseModel):
    latest_filing_date: str | None
    latest_filing_type: str | None
    days_since_last_filing: int | None
    source_url: str


@router.post("/functions/sec-latest-filing", response_model=SECFilingResponse)
async def sec_latest_filing(
    body: SECFilingRequest,
    _tenant: TenantContext = Depends(get_tenant),
):
    """Fetch a company's most recent SEC filing from EDGAR.

    Input: CIK (Central Index Key) as a string.
    Output: latest_filing_date, latest_filing_type, days_since_last_filing, source_url.
    """
    # Zero-pad CIK to 10 digits as required by SEC
    padded_cik = body.cik.lstrip("0").zfill(10)
    source_url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                source_url,
                headers={"User-Agent": SEC_USER_AGENT},
            )
        except httpx.RequestError as exc:
            logger.warning("sec_edgar_request_error", cik=padded_cik, error=str(exc))
            return SECFilingResponse(
                latest_filing_date=None,
                latest_filing_type=None,
                days_since_last_filing=None,
                source_url=source_url,
            )

    if resp.status_code == 404:
        return SECFilingResponse(
            latest_filing_date=None,
            latest_filing_type=None,
            days_since_last_filing=None,
            source_url=source_url,
        )

    resp.raise_for_status()
    data = resp.json()

    # Parse filings.recent — arrays of form, filingDate, etc.
    recent = data.get("filings", {}).get("recent", {})
    dates = recent.get("filingDate", [])
    forms = recent.get("form", [])

    if not dates:
        return SECFilingResponse(
            latest_filing_date=None,
            latest_filing_type=None,
            days_since_last_filing=None,
            source_url=source_url,
        )

    # The first entry is the most recent filing
    latest_date_str = dates[0]
    latest_form = forms[0] if forms else None

    try:
        latest_date = datetime.date.fromisoformat(latest_date_str)
        days_since = (datetime.date.today() - latest_date).days
    except ValueError:
        days_since = None

    return SECFilingResponse(
        latest_filing_date=latest_date_str,
        latest_filing_type=latest_form,
        days_since_last_filing=days_since,
        source_url=source_url,
    )


# ---------------------------------------------------------------------------
# Generic function invoke + materialize
# ---------------------------------------------------------------------------

class InvokeRequest(BaseModel):
    entity_uri: str
    kg_name: str


class DiscoveredEntity(BaseModel):
    uri: str
    type: str
    name: str
    skills: list[str]


class InvokeResponse(BaseModel):
    entity_uri: str
    function: str
    output: dict
    discovered_entities: list[DiscoveredEntity] = []
    duration_ms: float


# Hardcoded skill mapping per entity type (mirrors frontend TypeNode METHOD_MAP)
SKILLS_BY_TYPE: dict[str, list[str]] = {
    "Company": ["filings()", "patents()", "headcount()", "news()"],
    "Investor": ["portfolio()", "coInvestors()"],
    "Person": ["publications()", "bio()", "trajectory()"],
    "FundingRound": ["coInvestors()", "capTable()"],
}


# Shared executor instance
_executor: FunctionExecutor | None = None


def _get_executor() -> FunctionExecutor:
    global _executor
    if _executor is None:
        _executor = FunctionExecutor()
    return _executor


@router.post("/graphs/{tenant}/functions/{function_name}/invoke", response_model=InvokeResponse)
async def invoke_function(
    function_name: str,
    body: InvokeRequest,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Invoke a registered function for one entity and materialize the result as triples.

    Steps:
    1. Look up FunctionRef in the tenant ontology graph
    2. Resolve the entity's filing_cik attribute from the KG
    3. Invoke the function via FunctionExecutor
    4. Write result attributes back as triples on the entity
    """
    # Dispatch to specific invoke endpoints for functions that have their
    # own resolution logic (e.g., investor-portfolio resolves by name, not CIK)
    if function_name == "investor-portfolio":
        return await invoke_investor_portfolio(body, tenant, client)

    start = time.monotonic()
    ontology_graph = tenant_graph_uri(tenant.tenant_id)
    instance_graph = kg_graph_uri(tenant.tenant_id, body.kg_name)

    # --- Step 1: Look up the function definition ---
    sparql = list_functions_query(ontology_graph, entity_type=None)
    raw = await client.query(sparql)
    _, bindings = parse_sparql_results(raw)

    func_ref = None
    for row in bindings:
        if row.get("name") == function_name:
            func_ref = FunctionRef(
                name=row["name"],
                entity_type=row.get("type", "").split("/")[-1],
                endpoint_url=row.get("endpoint"),
                description=row.get("desc", ""),
                tier=FunctionTier.CUSTOM,
            )
            break

    if func_ref is None:
        raise HTTPException(status_code=404, detail=f"Function '{function_name}' not registered")

    # --- Step 2: Resolve the entity's filing_cik from the KG ---
    entity_type = func_ref.entity_type  # e.g. "Company"
    cik_attr_uri = f"https://cograph.tech/types/{entity_type}/attrs/filing_cik"

    # Try direct attribute on the entity
    cik_query = (
        f"SELECT ?cik FROM <{instance_graph}>\n"
        f"WHERE {{\n"
        f"  <{body.entity_uri}> <{cik_attr_uri}> ?cik .\n"
        f"}}"
    )
    raw_cik = await client.query(cik_query)
    _, cik_bindings = parse_sparql_results(raw_cik)

    cik_value = None
    if cik_bindings:
        cik_value = cik_bindings[0].get("cik")

    # Fallback 1: check linked FundingRound entities for a filing_cik attribute
    if not cik_value:
        fallback_query = (
            f"SELECT ?cik FROM <{instance_graph}>\n"
            f"WHERE {{\n"
            f"  ?round ?rel <{body.entity_uri}> .\n"
            f"  ?round <https://cograph.tech/types/FundingRound/attrs/filing_cik> ?cik .\n"
            f"}}"
        )
        raw_fallback = await client.query(fallback_query)
        _, fb_bindings = parse_sparql_results(raw_fallback)
        if fb_bindings:
            cik_value = fb_bindings[0].get("cik")

    # Fallback 2: FundingRound entity label IS the CIK (pear-backyard data pattern)
    if not cik_value:
        label_query = (
            f"SELECT ?label FROM <{instance_graph}>\n"
            f"WHERE {{\n"
            f"  ?round <https://cograph.tech/onto/company_name> <{body.entity_uri}> .\n"
            f"  ?round <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://cograph.tech/types/FundingRound> .\n"
            f"  ?round <http://www.w3.org/2000/01/rdf-schema#label> ?label .\n"
            f"}}\n"
            f"LIMIT 1"
        )
        raw_label = await client.query(label_query)
        _, label_bindings = parse_sparql_results(raw_label)
        if label_bindings:
            candidate = label_bindings[0].get("label", "")
            # Verify it looks like a CIK (all digits, possibly zero-padded)
            if candidate.lstrip("0").isdigit():
                cik_value = candidate

    if not cik_value:
        raise HTTPException(
            status_code=422,
            detail=f"Could not resolve filing_cik for entity {body.entity_uri}",
        )

    # --- Step 3: Invoke the function ---
    executor = _get_executor()
    payload = {"cik": cik_value}
    # Pass the caller's API key so the tier-2 endpoint can authenticate
    invoke_headers = {"X-API-Key": tenant.api_key}
    result = await executor.invoke(func_ref, payload, headers=invoke_headers)
    output = result.output

    # --- Step 4: Materialize result as triples on the entity ---
    new_triples: list[tuple[str, str, str]] = []
    replaced_preds: list[str] = []
    for key, value in output.items():
        if value is None:
            continue
        attr_pred = f"https://cograph.tech/types/{entity_type}/attrs/{key}"
        new_triples.append((body.entity_uri, attr_pred, str(value)))
        replaced_preds.append(attr_pred)

        # Ensure the attribute exists in the ontology (schema graph — unrelated to
        # the instance-fact write below; left as-is).
        datatype = "string"
        if isinstance(value, int):
            datatype = "integer"
        elif isinstance(value, float):
            datatype = "float"
        attr_sparql = insert_attribute(
            ontology_graph, entity_type, key,
            description=f"Lambda-computed by {function_name}",
            datatype=datatype,
        )
        try:
            await client.update(attr_sparql)
        except Exception:
            pass  # attribute may already exist

    # Add provenance triple
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    lambda_ts_pred = "https://cograph.tech/onto/lambda_refreshed_at"
    new_triples.append((body.entity_uri, lambda_ts_pred, now_iso))
    replaced_preds.append(lambda_ts_pred)

    # Persist via the shared write path (ADR 0007): an attribute update = clear the
    # old value + insert the new. delete_facts with object=None is a predicate-scoped
    # delete (drops any prior value of each replaced predicate, no-op when absent),
    # insert_facts writes the new values batched, and ONE refresh_after_write carries
    # the touched type — so this write fans out (index/cache/stats) like every other.
    if new_triples:
        await delete_facts(
            client,
            instance_graph,
            triples=[(body.entity_uri, pred, None) for pred in replaced_preds],
            touched_types=[entity_type],
            reason=f"lambda re-invoke: {function_name}",
        )
        await insert_facts(client, instance_graph, new_triples)
        await refresh_after_write(
            client,
            tenant_id=tenant.tenant_id,
            kg_name=body.kg_name,
            affected_types=[entity_type],
        )

    # --- Step 5: Discover linked entities for cascade ---
    discovered: list[DiscoveredEntity] = []

    if function_name == "sec-latest-filing":
        # Find Investor entities linked via FundingRound → lead_investor
        discover_query = (
            f"SELECT DISTINCT ?investor ?investorName FROM <{instance_graph}>\n"
            f"WHERE {{\n"
            f"  ?round <https://cograph.tech/onto/company_name> <{body.entity_uri}> .\n"
            f"  ?round <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://cograph.tech/types/FundingRound> .\n"
            f"  ?round <https://cograph.tech/onto/lead_investor> ?investor .\n"
            f"  ?investor <http://www.w3.org/2000/01/rdf-schema#label> ?investorName .\n"
            f"}}"
        )
        try:
            raw_discover = await client.query(discover_query)
            _, discover_bindings = parse_sparql_results(raw_discover)
            for row in discover_bindings:
                inv_uri = row.get("investor", "")
                inv_name = row.get("investorName", "")
                if inv_uri and inv_name:
                    inv_type = "Investor"
                    discovered.append(DiscoveredEntity(
                        uri=inv_uri,
                        type=inv_type,
                        name=inv_name,
                        skills=SKILLS_BY_TYPE.get(inv_type, []),
                    ))
        except Exception as exc:
            logger.warning("discover_entities_failed", error=str(exc))

    duration_ms = (time.monotonic() - start) * 1000

    logger.info(
        "lambda_invoked",
        function=function_name,
        entity=body.entity_uri,
        duration_ms=round(duration_ms, 1),
        output_keys=list(output.keys()),
        discovered_count=len(discovered),
    )

    return InvokeResponse(
        entity_uri=body.entity_uri,
        function=function_name,
        output=output,
        discovered_entities=discovered,
        duration_ms=round(duration_ms, 1),
    )


# ---------------------------------------------------------------------------
# Tier-2 Lambda: Investor Portfolio (SPARQL-based, no external API)
# ---------------------------------------------------------------------------

class PortfolioRequest(BaseModel):
    investor_name: str


class PortfolioResponse(BaseModel):
    portfolio_count: int
    companies: list[str]
    total_invested_usd: int | None


@router.post("/functions/investor-portfolio", response_model=PortfolioResponse)
async def investor_portfolio(
    body: PortfolioRequest,
    _tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Query the KG for all companies in an investor's portfolio.

    Looks up FundingRound entities where lead_investor matches this investor,
    then follows company_name relationships to get Company names and sums amounts.
    """
    tenant_id = _tenant.tenant_id

    # Search across all KGs in the tenant for this investor's portfolio
    # We query the instance graphs for FundingRound → company_name relationships
    # where lead_investor points to an entity with this investor's name.
    #
    # For demo purposes, we try the pear-backyard KG first.
    kg_names = ["pear-backyard"]
    companies: list[str] = []
    total_invested: int = 0

    for kg_name in kg_names:
        ig = kg_graph_uri(tenant_id, kg_name)
        portfolio_query = (
            f"SELECT ?companyName ?amount FROM <{ig}>\n"
            f"WHERE {{\n"
            f"  ?investor <https://cograph.tech/types/Investor/attrs/name> \"{body.investor_name}\" .\n"
            f"  ?investor <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://cograph.tech/types/Investor> .\n"
            f"  ?round <https://cograph.tech/onto/lead_investor> ?investor .\n"
            f"  ?round <https://cograph.tech/onto/company_name> ?company .\n"
            f"  ?company <https://cograph.tech/types/Company/attrs/name> ?companyName .\n"
            f"  OPTIONAL {{ ?round <https://cograph.tech/types/FundingRound/attrs/amount_usd> ?amount }}\n"
            f"}}"
        )
        try:
            raw_portfolio = await client.query(portfolio_query)
            _, portfolio_bindings = parse_sparql_results(raw_portfolio)
            for row in portfolio_bindings:
                cname = row.get("companyName", "")
                if cname and cname not in companies:
                    companies.append(cname)
                amt_str = row.get("amount", "")
                if amt_str:
                    try:
                        total_invested += int(float(amt_str))
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass

    return PortfolioResponse(
        portfolio_count=len(companies),
        companies=companies,
        total_invested_usd=total_invested if total_invested > 0 else None,
    )


# ---------------------------------------------------------------------------
# Generic invoke for investor-portfolio (reuses invoke pattern)
# ---------------------------------------------------------------------------

@router.post(
    "/graphs/{tenant}/functions/investor-portfolio/invoke",
    response_model=InvokeResponse,
)
async def invoke_investor_portfolio(
    body: InvokeRequest,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Invoke investor-portfolio for an Investor entity.

    Resolves the investor name from the entity URI, queries the KG for
    portfolio data, and materializes the results as triples.
    """
    start = time.monotonic()
    instance_graph = kg_graph_uri(tenant.tenant_id, body.kg_name)
    ontology_graph = tenant_graph_uri(tenant.tenant_id)

    # Resolve investor name from entity — prefer the Investor/attrs/name
    # attribute (which uses spaces) over rdfs:label (which uses underscores)
    name_query = (
        f"SELECT ?name FROM <{instance_graph}>\n"
        f"WHERE {{\n"
        f"  {{ <{body.entity_uri}> <https://cograph.tech/types/Investor/attrs/name> ?name }}\n"
        f"  UNION\n"
        f"  {{ <{body.entity_uri}> <http://www.w3.org/2000/01/rdf-schema#label> ?name }}\n"
        f"}}"
    )
    raw_name = await client.query(name_query)
    _, name_bindings = parse_sparql_results(raw_name)

    if not name_bindings:
        raise HTTPException(
            status_code=422,
            detail=f"Could not resolve name for entity {body.entity_uri}",
        )

    # Prefer the attrs/name value (with spaces) if both are present
    investor_name = name_bindings[0].get("name", "")

    # Query portfolio data inline — same SPARQL as the /functions/investor-portfolio
    # endpoint but executed directly rather than calling the endpoint function
    # (avoids FastAPI Depends() / connection-state issues when called internally)
    companies: list[str] = []
    total_invested: int = 0
    ig = instance_graph
    portfolio_query = (
        f"SELECT ?companyName ?amount FROM <{ig}>\n"
        f"WHERE {{\n"
        f"  ?investor <https://cograph.tech/types/Investor/attrs/name> \"{investor_name}\" .\n"
        f"  ?investor <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <https://cograph.tech/types/Investor> .\n"
        f"  ?round <https://cograph.tech/onto/lead_investor> ?investor .\n"
        f"  ?round <https://cograph.tech/onto/company_name> ?company .\n"
        f"  ?company <https://cograph.tech/types/Company/attrs/name> ?companyName .\n"
        f"  OPTIONAL {{ ?round <https://cograph.tech/types/FundingRound/attrs/amount_usd> ?amount }}\n"
        f"}}"
    )
    raw_portfolio = await client.query(portfolio_query)
    _, portfolio_bindings = parse_sparql_results(raw_portfolio)
    for row in portfolio_bindings:
        cname = row.get("companyName", "")
        if cname and cname not in companies:
            companies.append(cname)
        amt_str = row.get("amount", "")
        if amt_str:
            try:
                total_invested += int(float(amt_str))
            except (ValueError, TypeError):
                pass

    output = {
        "portfolio_count": len(companies),
        "companies": ", ".join(companies),
    }
    if total_invested > 0:
        output["total_invested_usd"] = str(total_invested)

    # Materialize results as triples on the Investor entity
    entity_type = "Investor"
    new_triples: list[tuple[str, str, str]] = []
    replaced_preds: list[str] = []
    for key, value in output.items():
        if value is None:
            continue
        attr_pred = f"https://cograph.tech/types/{entity_type}/attrs/{key}"
        new_triples.append((body.entity_uri, attr_pred, str(value)))
        replaced_preds.append(attr_pred)

        # Ensure attribute in ontology (schema graph — unrelated to the instance
        # write below; left as-is).
        datatype = "integer" if key == "portfolio_count" else "string"
        attr_sparql = insert_attribute(
            ontology_graph, entity_type, key,
            description=f"Lambda-computed by investor-portfolio",
            datatype=datatype,
        )
        try:
            await client.update(attr_sparql)
        except Exception:
            pass

    # Provenance timestamp
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    lambda_ts_pred = "https://cograph.tech/onto/lambda_refreshed_at"
    new_triples.append((body.entity_uri, lambda_ts_pred, now_iso))
    replaced_preds.append(lambda_ts_pred)

    # Persist via the shared write path (ADR 0007): clear each replaced predicate's
    # prior value, insert the new values, then ONE refresh carrying the touched type.
    if new_triples:
        await delete_facts(
            client,
            instance_graph,
            triples=[(body.entity_uri, pred, None) for pred in replaced_preds],
            touched_types=[entity_type],
            reason="lambda re-invoke: investor-portfolio",
        )
        await insert_facts(client, instance_graph, new_triples)
        await refresh_after_write(
            client,
            tenant_id=tenant.tenant_id,
            kg_name=body.kg_name,
            affected_types=[entity_type],
        )

    duration_ms = (time.monotonic() - start) * 1000

    logger.info(
        "lambda_invoked",
        function="investor-portfolio",
        entity=body.entity_uri,
        duration_ms=round(duration_ms, 1),
        portfolio_count=len(companies),
    )

    return InvokeResponse(
        entity_uri=body.entity_uri,
        function="investor-portfolio",
        output=output,
        discovered_entities=[],
        duration_ms=round(duration_ms, 1),
    )
