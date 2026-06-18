"""Test fixture — a router plugin registered by test_router_plugin.

`register(app)` receives the FastAPI app and mounts a dummy router, mirroring
how a downstream (e.g. premium recommender) package attaches its endpoints.
"""
from fastapi import APIRouter, FastAPI

APP = None

router = APIRouter()


@router.get("/_fake_router_plugin/ping")
def _ping():
    return {"ok": True}


def register(app: FastAPI):
    global APP
    APP = app
    app.include_router(router)
