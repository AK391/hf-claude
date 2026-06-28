#!/usr/bin/env python3
"""huggingface/conductor — local OpenRouter-Fusion-style orchestrator.

When the selected model is `huggingface/conductor`, this proxy fans the prompt
out to a panel of cheap/fast models in parallel, then passes their raw answers
to a judge model that synthesizes one final streamed response — the OpenRouter
Fusion pattern: many cheap perspectives, one high-quality synthesis.

Speaks the Anthropic Messages API (`/v1/messages`), since Claude Code is
configured with ANTHROPIC_BASE_URL pointing here. Non-conductor requests are
proxied through unchanged, with HTML error pages converted to clean Anthropic
errors so the client never hangs on an unparseable body.
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
PORT = int(os.environ.get("HF_CONDUCTOR_PORT", "8080"))
READ_TIMEOUT = float(os.environ.get("HF_CONDUCTOR_READ_TIMEOUT", "120"))
# Streaming judges (reasoning models) can sit silent for a while during the
# thinking phase before emitting text. Give the relayed stream headroom so a
# long gap between SSE events doesn't trip the read timeout and abort mid-stream.
STREAM_READ_TIMEOUT = float(os.environ.get("HF_CONDUCTOR_STREAM_TIMEOUT", "600"))

# OpenRouter Fusion: a panel of cheap/fast models answer in parallel, then a
# frontier judge synthesizes their outputs into the single streamed response.
# The panel is deliberately small (~7-8B), non-reasoning models from different
# families so the fan-out is fast and cheap. The judge is a non-reasoning
# frontier model so its answer streams as visible text immediately (a reasoning
# judge like GLM-5.2 burns its token budget on hidden thinking blocks before
# emitting any text). Tune via env: HF_CONDUCTOR_PANEL / HF_CONDUCTOR_JUDGE.
DEFAULT_PANEL = "Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,allenai/Olmo-3-7B-Instruct"
DEFAULT_JUDGE = "MiniMaxAI/MiniMax-M3"
PANEL = [m.strip() for m in os.environ.get("HF_CONDUCTOR_PANEL", DEFAULT_PANEL).split(",") if m.strip()]
JUDGE = os.environ.get("HF_CONDUCTOR_JUDGE", DEFAULT_JUDGE)
PANEL_TIMEOUT = float(os.environ.get("HF_CONDUCTOR_PANEL_TIMEOUT", "45"))
# Reasoning judges (e.g. GLM-5.2) spend tokens on hidden thinking blocks before
# emitting the final text, so keep the budgets generous or the answer is empty.
PANEL_MAX_TOKENS = int(os.environ.get("HF_CONDUCTOR_PANEL_MAX_TOKENS", "4096"))
JUDGE_MAX_TOKENS = int(os.environ.get("HF_CONDUCTOR_JUDGE_MAX_TOKENS", "16384"))

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
    "content-length", "content-encoding",
    # The incoming Host header points at this local proxy (127.0.0.1:8080);
    # forwarding it to the router makes CloudFront return 403. httpx sets the
    # correct Host from the target URL, so drop it.
    "host",
    # Avoid advertising gzip to the upstream; we relay raw bytes either way, and
    # a decoded/encoded mismatch can corrupt streamed SSE.
    "accept-encoding",
}

REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=READ_TIMEOUT, write=120.0, pool=10.0)
STREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=STREAM_READ_TIMEOUT, write=120.0, pool=10.0)


def _clean_headers(headers):
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def _bearer(request):
    auth = request.headers.get("authorization")
    if auth:
        return auth
    key = request.headers.get("x-api-key")
    return f"Bearer {key}" if key else None


def _anthropic_error(message, status=502):
    return JSONResponse(
        {"type": "error", "error": {"type": "api_error", "message": message}},
        status_code=status,
    )


def _stream(resp, client=None, label="upstream"):
    """Relay a streaming httpx response to the client.

    The httpx.AsyncClient that owns `resp` may outlive the request handler
    (the body generator runs after the handler returns), so when a `client` is
    passed we close it in the generator's finally — otherwise the `async with`
    block would tear the connection down mid-stream and raise httpx.ReadError.
    """
    async def gen():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        except Exception as e:  # noqa: BLE001
            logger.warning("%s stream relay ended early: %s: %s", label, type(e).__name__, e)
        finally:
            await resp.aclose()
            if client is not None:
                await client.aclose()

    return StreamingResponse(
        gen(),
        status_code=resp.status_code,
        headers=_clean_headers(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


async def _stream_to_text(client, bearer, body, timeout):
    """POST /v1/messages (non-streaming) and return the assistant text, or None."""
    payload = {**body, "stream": False}
    try:
        r = await client.post(
            f"{HF_ROUTER}/v1/messages",
            headers={"Authorization": bearer, "Content-Type": "application/json"},
            content=json.dumps(payload).encode(),
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=60.0, pool=10.0),
        )
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        logger.warning("  panel %s: transport %s", body.get("model"), type(e).__name__)
        return None
    ct = r.headers.get("content-type", "").lower()
    if r.status_code != 200 or "text/html" in ct:
        logger.warning("  panel %s: status=%s", body.get("model"), r.status_code)
        return None
    try:
        data = r.json()
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "")
        # OpenAI-style fallback (some providers return chat-completions shape)
        if "choices" in data:
            return data["choices"][0].get("message", {}).get("content", "")
    except Exception as e:  # noqa: BLE001
        logger.warning("  panel %s: json parse %s", body.get("model"), e)
    return ""


async def _run_panel(client, bearer, messages, system, max_tokens):
    """Fan out to every panel model in parallel; collect (model, text)."""
    async def one(model):
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            body["system"] = system
        text = await _stream_to_text(client, bearer, body, timeout=PANEL_TIMEOUT)
        return model, text

    results = await asyncio.gather(*[one(m) for m in PANEL])
    return [(m, t) for m, t in results]


def _build_judge_messages(messages, system, panel_outputs):
    """Construct a synthesis prompt that fuses the panel's answers."""
    user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    user_text = ""
    if isinstance(user_msg, dict):
        c = user_msg.get("content")
        if isinstance(c, str):
            user_text = c
        elif isinstance(c, list):
            user_text = " ".join(
                b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
            )

    fused = "\n\n".join(
        f"### Perspective from {model}:\n{txt}"
        for model, txt in panel_outputs
        if txt
    ) or "(all panel models failed to respond)"

    instruction = (
        "You are a synthesis judge. Several AI models each answered the same user "
        "request independently. Review their perspectives for consensus, "
        "contradictions, and any missed details, then produce one optimal, unified "
        "response to the ORIGINAL request. Do not mention the panel or this process; "
        "just answer as a single assistant."
    )

    judge_system = instruction
    if system:
        if isinstance(system, str):
            judge_system = system + "\n\n" + instruction
        elif isinstance(system, list):
            judge_system = system + [{"type": "text", "text": instruction}]

    judge_messages = [
        {
            "role": "user",
            "content": (
                f"--- ORIGINAL USER REQUEST ---\n{user_text}\n\n"
                f"--- PANEL RESPONSES ---\n{fused}\n\n"
                f"--- YOUR TASK ---\nSynthesize the above into a single response."
            ),
        }
    ]
    return judge_messages, judge_system


async def _forward_once(client, request, body, timeout=None):
    url = f"{HF_ROUTER}/v1/messages"
    if request.url.query:
        url += "?" + request.url.query
    try:
        req = client.build_request(
            "POST", url,
            headers=_clean_headers(request.headers),
            content=json.dumps(body).encode(),
            timeout=timeout or REQUEST_TIMEOUT,
        )
        return await client.send(req, stream=True)
    except (httpx.TimeoutException, httpx.HTTPError):
        return None


async def _conductor(request, body):
    bearer = _bearer(request)
    messages = body.get("messages", [])
    system = body.get("system")
    want_stream = body.get("stream", True)
    max_tokens = body.get("max_tokens", JUDGE_MAX_TOKENS)

    # NOTE: we deliberately do NOT use `async with` for the streaming path's
    # client — the StreamingResponse body generator runs after this handler
    # returns, so the client must stay alive until the stream finishes. The
    # client is closed inside _stream()'s generator finally-block instead.
    if want_stream:
        client = httpx.AsyncClient()
        try:
            logger.info("conductor: fanning out to panel=%s", PANEL)
            panel_outputs = await _run_panel(
                client, bearer, messages, system, max_tokens=PANEL_MAX_TOKENS
            )
            ok = [m for m, t in panel_outputs if t]
            logger.info("conductor: panel responded %d/%d -> judge=%s", len(ok), len(PANEL), JUDGE)

            judge_messages, judge_system = _build_judge_messages(messages, system, panel_outputs)
            judge_body = {
                "model": JUDGE,
                "messages": judge_messages,
                "system": judge_system,
                "max_tokens": max_tokens,
                "stream": True,
            }
            for k in ("temperature", "top_p", "top_k", "stop_sequences"):
                if k in body:
                    judge_body[k] = body[k]

            resp = await _forward_once(client, request, judge_body, timeout=STREAM_TIMEOUT)
            if resp is not None and resp.status_code == 200 and "text/html" not in resp.headers.get("content-type", ""):
                logger.info("conductor: streaming judge synthesis")
                return _stream(resp, client=client, label="judge")
            if resp is not None:
                await resp.aclose()
            await client.aclose()
            return _anthropic_error(f"conductor: judge model {JUDGE} failed to stream.")
        except Exception as e:  # noqa: BLE001
            await client.aclose()
            return _anthropic_error(f"conductor: orchestration error: {type(e).__name__}: {e}")

    async with httpx.AsyncClient() as client:
        logger.info("conductor: fanning out to panel=%s", PANEL)
        panel_outputs = await _run_panel(
            client, bearer, messages, system, max_tokens=PANEL_MAX_TOKENS
        )
        ok = [m for m, t in panel_outputs if t]
        logger.info("conductor: panel responded %d/%d -> judge=%s", len(ok), len(PANEL), JUDGE)

        judge_messages, judge_system = _build_judge_messages(messages, system, panel_outputs)
        judge_body = {
            "model": JUDGE,
            "messages": judge_messages,
            "system": judge_system,
            "max_tokens": max_tokens,
            "stream": False,
        }
        for k in ("temperature", "top_p", "top_k", "stop_sequences"):
            if k in body:
                judge_body[k] = body[k]

        text = await _stream_to_text(client, bearer, judge_body, timeout=READ_TIMEOUT)
        if text is None:
            return _anthropic_error(f"conductor: judge model {JUDGE} failed to respond.")
        return JSONResponse(
            {
                "id": "msg_conductor",
                "type": "message",
                "role": "assistant",
                "model": CONDUCTOR_ID,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequences": judge_body.get("stop_sequences", []),
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
            status_code=200,
        )


@app.get("/health")
async def health():
    return {
        "ok": True,
        "conductor": "fusion",
        "panel": PANEL,
        "judge": JUDGE,
    }


@app.post("/v1/messages")
async def messages(request: Request):
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return _anthropic_error("conductor: could not parse request body as JSON", 400)

    if body.get("model") == CONDUCTOR_ID:
        return await _conductor(request, body)

    # Non-conductor model: forward once; convert HTML errors so client never hangs.
    # Client stays alive for the streamed response (closed by _stream's generator).
    client = httpx.AsyncClient()
    resp = await _forward_once(client, request, body)
    if resp is not None and resp.status_code == 200 and "text/html" not in resp.headers.get("content-type", ""):
        return _stream(resp, client=client, label=f"passthrough:{body.get('model')}")
    if resp is not None:
        await resp.aclose()
    await client.aclose()
    return _anthropic_error(f"conductor: upstream provider failed for {body.get('model')}.")


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def catch_all(path: str, request: Request):
    """Passthrough for other paths (e.g. /v1/messages/count_tokens)."""
    url = f"{HF_ROUTER}/{path}"
    if request.url.query:
        url += "?" + request.url.query
    client = httpx.AsyncClient()
    body = await request.body()
    try:
        req = client.build_request(
            request.method, url,
            headers=_clean_headers(request.headers),
            content=body or None,
            timeout=REQUEST_TIMEOUT,
        )
        resp = await client.send(req, stream=True)
    except (httpx.TimeoutException, httpx.HTTPError) as e:
        await client.aclose()
        return _anthropic_error(f"upstream transport error: {type(e).__name__}")
    ct = resp.headers.get("content-type", "").lower()
    if "text/html" in ct:
        await resp.aclose()
        await client.aclose()
        return _anthropic_error("upstream returned an HTML error page instead of JSON.")
    return _stream(resp, client=client, label=f"catchall:{path}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")