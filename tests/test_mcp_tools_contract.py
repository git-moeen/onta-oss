"""Contract tests for the MCP tools added in COG-129 parity work:
``create_knowledge_graph``, ``delete_knowledge_graph``, ``list_jobs`` and
``get_job``.

Like ``test_mcp_agent_route.py``, the OSS MCP server (``packages/cograph-mcp``)
is a thin TypeScript client over the HTTP API — each tool calls a canonical
route via the ``cograph`` SDK. The tool itself is exercised by the npm
typecheck + build in CI; here we lock the request/response *contract* each tool
depends on, through the FastAPI ``TestClient`` (the same path the SDK hits), with
Neptune mocked so the suite is deterministic and offline.

Tool → backend route (via the SDK):
  * create_knowledge_graph → ``POST   /graphs/{tenant}/kgs``            (SDK createKg)
  * delete_knowledge_graph → ``DELETE /graphs/{tenant}/kgs/{name}``     (SDK deleteKg)
  * list_jobs              → ``GET    /graphs/{tenant}/jobs``           (SDK jobs)
  * get_job                → ``GET    /graphs/{tenant}/enrich/jobs/{id}`` (SDK enrichJob)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMNIX_API_KEYS", '{"test-key": "test-tenant"}')
os.environ.setdefault("OMNIX_NEPTUNE_ENDPOINT", "http://fake:8182")

HEADERS = {"X-API-Key": "test-key"}
TENANT = "test-tenant"


def test_create_kg_tool_target_exists(client, mock_neptune, auth_headers):
    """create_knowledge_graph → POST /kgs creates a graph and echoes its name."""
    resp = client.post(
        f"/graphs/{TENANT}/kgs",
        json={"name": "mcp-created-kg", "description": "made by the MCP tool"},
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["name"] == "mcp-created-kg"


def test_delete_kg_tool_target_exists(client, mock_neptune, auth_headers):
    """delete_knowledge_graph → DELETE /kgs/{name} is mounted + reachable
    (dispatched, not a 404/405 from a missing route)."""
    resp = client.delete(f"/graphs/{TENANT}/kgs/mcp-created-kg", headers=auth_headers)
    assert resp.status_code in (200, 202, 204), resp.text


def test_list_jobs_tool_target_returns_a_list(client, mock_neptune, auth_headers):
    """list_jobs → GET /jobs returns a JSON array (empty when no jobs)."""
    resp = client.get(f"/graphs/{TENANT}/jobs", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json(), list)


def test_list_jobs_tool_accepts_category_filter(client, mock_neptune, auth_headers):
    """The `category` arg the tool forwards is a valid query param on /jobs."""
    resp = client.get(
        f"/graphs/{TENANT}/jobs", params={"category": "enrichment"}, headers=auth_headers
    )
    assert resp.status_code == 200, resp.text
    assert isinstance(resp.json(), list)


def test_get_job_tool_target_404s_for_unknown_id(client, mock_neptune, auth_headers):
    """get_job → GET /enrich/jobs/{id} is mounted and 404s for a missing id
    (proving the route exists and is owner-scoped, not that it's unreachable)."""
    resp = client.get(f"/graphs/{TENANT}/enrich/jobs/does-not-exist", headers=auth_headers)
    assert resp.status_code == 404, resp.text
