#!/usr/bin/env python3
"""huggingface/conductor — local resilient proxy for HF Inference Providers.

Sits between Claude Code and https://router.huggingface.co. When the selected
model is `huggingface/conductor`, it transparently failovers across providers
for a base model, then across an equivalent model class, instead of letting
Claude Code spin on a single broken endpoint (the "attempt 10/10" hang that
happens when the router returns an HTML error page instead of JSON).

Routes the Anthropic Messages API (`/v1/messages`), since Claude Code is
configured with ANTHROPIC_BASE_URL pointing here. Other paths are proxied
through unchanged (HTML error pages are converted to clean Anthropic errors).
"""
import asyncio
import json
import logging
import os
import sys
import time

try:
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
except ImportError as e:  # pragma: no cover
    print(
        f"hf-conductor proxy: missing dependency ({e}).\n"
        "Install with:  pip install fastapi uvicorn httpx",
        file=sys.stderr,
    )
    sys.exit(1)

logging.basicConfig(
    level=os.environ.get("HF_CONDUCTOR_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("hf-conductor")

app = FastAPI()

HF_ROUTER = "https://router.huggingface.co"
CONDUCTOR_ID = "huggingface/conductor"
CONDUCTOR_BASE = os.environ.get("HF_CONDUCTOR_BASE_MODEL", "zai-org/GLM-5.2")
CONDUCTOR_FALLBACK = os.environ.get("HF_CONDUCTOR_FALLBACK_MODEL", "MiniMaxAI/MiniMax-M3")
PORT = int(os.environ.get("HF_CONDUCTOR_PORT", "8080"))
CATALOG_TTL = float(os.environ.get("HF_CONDUCTOR_CATALOG_TTL", "600"))
READ_TIMEOUT = float(os.environ.get("HF_CONDUCTOR_READ_TIMEOUT", "60"))

# model id -> equivalent frontier model to fall back to once all providers fail
MODEL_FAILOVER_MAP = {
    CONDUCTOR_BASE: CONDUCTOR_FALLBACK,
    CONDUCTOR_FALLBACK: CONDUCTOR_BASE,
}

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
    "content-length", "content-encoding",  # StreamingResponse sets its own
}

_catalog = {}            # model_id -> [provider ids]
_catalog_fetched = 0.0
_catalog_lock = asyncio.Lock()

REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=READ_TIMEOUT, write=60.0, pool=10.0)


def _clean_headers(headers):
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def _bearer(request):
    """Extract an Authorization: Bearer ... string from the incoming request."""
    auth = request.headers.get("authorization")
    if auth:
        return auth
    key = request.headers.get("x-api-key")
    if key:
        return f"Bearer {key}"
    return None


def _anthropic_error(message, status=502):
    # Anthropic-shaped error so Claude Code shows a clean message instead of
    # hanging on an unparseable HTML body.
    return JSONResponse(
        {"type": "error", "error": {"type": "api_error", "message": message}},
        status_code=status,
    )


def _stream(resp):
    async def gen():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        gen(),
        status_code=resp.status_code,
        headers=_clean_headers(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


async def _fetch_catalog(client, bearer):
    if not bearer:
        return {}
    try:
        r = await client.get(
            f"{HF_ROUTER}/v1/models",
            headers={"Authorization": bearer},
            timeout=20.0,
        )
        if r.status_code != 200:
            logger.warning("catalog fetch returned %s", r.status_code)
            return {}
        data = r.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("catalog fetch failed: %s", e)
        return {}
    out = {}
    for entry in data.get("data", []):
        mid = entry.get("id")
        provs = [p.get("provider") for p in entry.get("providers", []) if p.get("provider")]
        if mid:
            out[mid] = provs
    return out


async def _ensure_catalog(client, bearer):
    global _catalog, _catalog_fetched
    if _catalog and (time.monotonic() - _catalog_fetched) < CATALOG_TTL:
        return
    async with _catalog_lock:
        if _catalog and (time.monotonic() - _catalog_fetched) < CATALOG_TTL:
            return
        _catalog = await _fetch_catalog(client, bearer)
        _catalog_fetched = time.monotonic()
        logger.info("catalog loaded: %d models", len(_catalog))


def _build_chain(base):
    """[base(auto), base:p1, base:p2, ..., fallback(auto), fallback:p1, ...]"""
    chain = []
    for model in (base, MODEL_FAILOVER_MAP.get(base)):
        if not model:
            continue
        chain.append(model)  # no suffix == router "auto" routing
        chain.extend(f"{model}:{p}" for p in _catalog.get(model, []))
    # de-dup, preserve order
    seen, out = set(), []
    for c in chain:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


async def _forward_once(client, request, body):
    """One attempt. Returns an open streaming httpx.Response, or None on failure."""
    url = f"{HF_ROUTER}/v1/messages"
    if request.url.query:
        url += "?" + request.url.query
    try:
        req = client.build_request(
            "POST",
            url,
            headers=_clean_headers(request.headers),
            content=json.dumps(body).encode(),
            timeout=REQUEST_TIMEOUT,
        )
        resp = await client.send(req, stream=True)
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        logger.warning("  ✗ %s -> transport %s", body.get("model"), type(e).__name__)
        return None
    ct = resp.headers.get("content-type", "").lower()
    if resp.status_code != 200 or "text/html" in ct:
        logger.warning("  ✗ %s -> status=%s ct=%s", body.get("model"), resp.status_code, ct)
        await resp.aclose()
        return None
    return resp


async def _conductor(request, body):
    bearer = _bearer(request)
    async with httpx.AsyncClient() as client:
        await _ensure_catalog(client, bearer)
        chain = _build_chain(CONDUCTOR_BASE)
        logger.info("conductor chain (%d): %s", len(chain), chain)
        for model in chain:
            body["model"] = model
            resp = await _forward_once(client, request, body)
            if resp is not None:
                logger.info("  ✓ serving via %s", model)
                return _stream(resp)
        logger.error("conductor: all fallbacks exhausted for %s", CONDUCTOR_BASE)
        return _anthropic_error(
            f"huggingface/conductor: all providers and the equivalent fallback "
            f"({CONDUCTOR_BASE} -> {MODEL_FAILOVER_MAP.get(CONDUCTOR_BASE)}) failed."
        )


@app.get("/health")
async def health():
    return {"ok": True, "conductor": CONDUCTOR_BASE, "catalog": len(_catalog)}


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _anthropic_error("conductor: could not parse request body as JSON", 400)

    if body.get("model") == CONDUCTOR_ID:
        return await _conductor(request, body)

    # Non-conductor model reached the proxy: forward once, but still convert an
    # HTML error page into a clean Anthropic error so the client never hangs.
    async with httpx.AsyncClient() as client:
        resp = await _forward_once(client, request, body)
        if resp is not None:
            return _stream(resp)
        return _anthropic_error(
            f"conductor: upstream provider failed for {body.get('model')}."
        )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def catch_all(path: str, request: Request):
    """Generic passthrough for every other path Claude Code may hit
    (e.g. /v1/messages/count_tokens). HTML error pages become clean errors."""
    url = f"{HF_ROUTER}/{path}"
    if request.url.query:
        url += "?" + request.url.query
    async with httpx.AsyncClient() as client:
        body = await request.body()
        try:
            req = client.build_request(
                request.method,
                url,
                headers=_clean_headers(request.headers),
                content=body or None,
                timeout=REQUEST_TIMEOUT,
            )
            resp = await client.send(req, stream=True)
        except (httpx.TimeoutException, httpx.HTTPError) as e:
            return _anthropic_error(f"upstream transport error: {type(e).__name__}")
        ct = resp.headers.get("content-type", "").lower()
        if "text/html" in ct:
            await resp.aclose()
            return _anthropic_error("upstream returned an HTML error page instead of JSON.")
        return _stream(resp)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")