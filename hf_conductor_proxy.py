#!/usr/bin/env python3
"""huggingface/conductor — local OpenRouter-Fusion-style orchestrator.

When the selected model is `huggingface/conductor`, this proxy runs the
OpenRouter Fusion pipeline: it fans the prompt out to a panel of diverse models
in parallel, an analyst model reads every panel response and produces a
structured analysis (consensus, contradictions, partial coverage, unique
insights, blind spots), and a synthesizer streams one unified answer grounded
in that analysis. Two synthesis stages — not one — is what lets a panel of
budget models surpass a single frontier model.

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

# OpenRouter Fusion pipeline:
#   panel      -> diverse capable models answer the original request in parallel
#   analyst    -> reads every panel response, emits structured analysis
#                  (consensus / contradictions / coverage / insights / blind spots)
#   synthesizer -> streams the final unified answer grounded in that analysis
# The blog's two headline results both flow from this structure: (1) a diverse
# frontier panel (e.g. Fable 5 + GPT-5.5) synthesizes to ~69% on DRACO vs ~65%
# solo, and (2) a *budget* panel of different families (Gemini Flash + Kimi +
# DeepSeek) beats individual frontier models. So the panel is chosen for
# diversity and cost, NOT raw size. The analyst and synthesizer default to the
# same non-reasoning frontier model: OpenRouter found fusing Opus 4.8 with itself
# beat solo Opus by 6.7pts, so a lot of the lift is the synthesis step itself
# (two stages, one model). A non-reasoning judge streams visible text
# immediately; a reasoning judge burns its token budget on hidden thinking
# blocks. Split analyst/synth into two different models with
# HF_CONDUCTOR_ANALYST / HF_CONDUCTOR_SYNTH if you want cross-family checking.
#
# Three presets mirror the three Fusion presets on OpenRouter's Fusion page
# (general-high / general-budget / general-fast): Quality, Budget, Speed. Each
# is a 3-model diverse panel; the panel composition changes, the panel SIZE
# stays 3 (matches OpenRouter's three-member panels). A launcher-selectable
# preset writes the chosen panel into HF_CONDUCTOR_PANEL so the proxy needs no
# preset awareness of its own.
PRESETS = {
    # 🏆 Quality — strong, diverse frontier panel for best synthesis. Different
    # families (GLM / DeepSeek / Kimi) so they disagree productively and the
    # analyst has real consensus/contradiction signal to work with.
    "quality": "zai-org/GLM-5.2,deepseek-ai/DeepSeek-V3,moonshotai/Kimi-K2.6",
    # 💰 Budget — cheap-but-capable diverse panel, OpenRouter's "budget panel"
    # finding: this class beat individual frontier models at a fraction of cost.
    "budget": "deepreinforce-ai/Ornith-1.0-35B,Qwen/Qwen3.6-35B-A3B,google/gemma-4-31B-it",
    # ⚡ Speed — small fast models from different labs for lowest latency.
    "speed": "Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,allenai/Olmo-3-7B-Instruct",
}
DEFAULT_PANEL = PRESETS["speed"]  # Speed is the no-surprise default (fast + cheap).
DEFAULT_PRESET = "speed"
PRESET = os.environ.get("HF_CONDUCTOR_PRESET", "").strip().lower()
if PRESET and PRESET in PRESETS:
    DEFAULT_PANEL = PRESETS[PRESET]
DEFAULT_ANALYST = "MiniMaxAI/MiniMax-M3"
DEFAULT_SYNTH = "MiniMaxAI/MiniMax-M3"
PANEL = [m.strip() for m in os.environ.get("HF_CONDUCTOR_PANEL", DEFAULT_PANEL).split(",") if m.strip()]
# Legacy compat: a v0.3 HF_CONDUCTOR_JUDGE set both stages to one model. We keep
# that behavior — it's the blog's "fuse with itself" default — while letting the
# two new vars split analyst/synth into different models.
_LEGACY_JUDGE = os.environ.get("HF_CONDUCTOR_JUDGE", "")
ANALYST = os.environ.get("HF_CONDUCTOR_ANALYST", _LEGACY_JUDGE or DEFAULT_ANALYST)
SYNTH = os.environ.get("HF_CONDUCTOR_SYNTH", _LEGACY_JUDGE or DEFAULT_SYNTH)
PANEL_TIMEOUT = float(os.environ.get("HF_CONDUCTOR_PANEL_TIMEOUT", "45"))
# Reasoning judges (e.g. GLM-5.2) spend tokens on hidden thinking blocks before
# emitting the final text, so keep the budgets generous or the answer is empty.
PANEL_MAX_TOKENS = int(os.environ.get("HF_CONDUCTOR_PANEL_MAX_TOKENS", "4096"))
ANALYST_MAX_TOKENS = int(os.environ.get("HF_CONDUCTOR_ANALYST_MAX_TOKENS", "3072"))
SYNTH_MAX_TOKENS = int(os.environ.get("HF_CONDUCTOR_JUDGE_MAX_TOKENS", "16384"))
# Keep HF_CONDUCTOR_JUDGE_MAX_TOKENS as the documented knob (above). v0.3 users
# who set it keep getting the synthesizer budget they tuned.

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

# The conductor's panel can't see the panel/judge tool calls, so there's no
# contamination surface to block on this side. But the OpenRouter Fusion post
# flagged a real risk — panel models with web access found their eval rubric
# online. If you wire server-side tools onto the panel later (web_search /
# web_fetch), exclude the domains that host anything you don't want the panel
# reading (benchmark rubrics, the user's own answer keys, etc.). Parsed once at
# import; empty by default so it's a no-op until you need it.
_EXCLUDE_DOMAINS_RAW = os.environ.get("HF_CONDUCTOR_EXCLUDE_DOMAINS", "")
EXCLUDE_DOMAINS = [d.strip().lower() for d in _EXCLUDE_DOMAINS_RAW.split(",") if d.strip()]


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


def _user_text(messages):
    """Pull the last user message's text out of the message list."""
    user_msg = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not isinstance(user_msg, dict):
        return ""
    c = user_msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(
            b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _merge_system(system, instruction):
    """Append the fusion instruction to the original system prompt, preserving type."""
    if not system:
        return instruction
    if isinstance(system, str):
        return system + "\n\n" + instruction
    if isinstance(system, list):
        return system + [{"type": "text", "text": instruction}]
    return instruction


# Stage 1 — analyst. The blog describes this precisely: "A judge model reads every
# panel response and produces structured analysis: consensus points, contradictions,
# partial coverage, unique insights, blind spots." Keep it non-streaming and short —
# this is intermediate scaffolding the synthesizer reasons over, not the answer.
ANALYST_INSTRUCTION = (
    "You are an analysis judge. Several AI models answered the same request "
    "independently. Read every response and produce a concise structured analysis "
    "of how they relate. Cover exactly these sections, each as a short bullet list:\n"
    "  - Consensus: points most/all responses agree on\n"
    "  - Contradictions: where responses disagree, and which is better supported\n"
    "  - Partial coverage: points only some responses made that the others missed\n"
    "  - Unique insights: the strongest non-obvious points raised by any one model\n"
    "  - Blind spots: things the request asked for that no response fully addressed\n"
    "Be concise and factual. Do NOT answer the original request yourself — only "
    "analyze the panel's outputs. Do not mention the models by name beyond attributing a point to 'one response'."
)

# Stage 2 — synthesizer. OpenRouter's pipeline: "The calling model then writes the
# final answer grounded in that analysis." It streams the user-facing answer, so it
# must not narrate the panel/analysis process.
SYNTH_INSTRUCTION = (
    "You are the final synthesizer. You are given a user's original request, the "
    "raw perspectives from a panel of models, and a structured analysis of those "
    "perspectives. Write one optimal, unified response to the ORIGINAL request, "
    "grounded in the analysis: resolve the contradictions, fold in the unique "
    "insights, and fill the blind spots. Do not mention the panel, the analysis, or "
    "this process — just answer as a single assistant."
)


def _build_analyst_messages(messages, system, panel_outputs):
    """Stage 1 prompt: panel responses -> structured analysis."""
    user_text = _user_text(messages)
    fused = "\n\n".join(
        f"### Perspective {i + 1}:\n{txt}"
        for i, (_model, txt) in enumerate(panel_outputs)
        if txt
    ) or "(all panel models failed to respond)"
    analyst_system = _merge_system(system, ANALYST_INSTRUCTION)
    analyst_messages = [
        {
            "role": "user",
            "content": (
                f"--- ORIGINAL USER REQUEST ---\n{user_text}\n\n"
                f"--- PANEL RESPONSES ---\n{fused}\n\n"
                f"--- YOUR TASK ---\nProduce the structured analysis only."
            ),
        }
    ]
    return analyst_messages, analyst_system


def _build_synth_messages(messages, system, panel_outputs, analysis):
    """Stage 2 prompt: original request + panel + analysis -> final answer."""
    user_text = _user_text(messages)
    fused = "\n\n".join(
        f"### Perspective {i + 1}:\n{txt}"
        for i, (_model, txt) in enumerate(panel_outputs)
        if txt
    ) or "(all panel models failed to respond)"
    synth_system = _merge_system(system, SYNTH_INSTRUCTION)
    synth_messages = [
        {
            "role": "user",
            "content": (
                f"--- ORIGINAL USER REQUEST ---\n{user_text}\n\n"
                f"--- PANEL RESPONSES ---\n{fused}\n\n"
                f"--- STRUCTURED ANALYSIS ---\n{analysis or '(analysis unavailable)'}\n\n"
                f"--- YOUR TASK ---\nWrite the final unified response to the original request."
            ),
        }
    ]
    return synth_messages, synth_system


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


async def _run_analysis(client, bearer, messages, system, panel_outputs):
    """Stage 1: ask the analyst to produce structured analysis of the panel.

    Non-streaming and short — this is intermediate scaffolding, not the answer.
    Returns the analysis text, or None if the analyst failed (the synthesizer
    then degrades gracefully to synthesizing the raw panel without analysis).
    """
    analyst_messages, analyst_system = _build_analyst_messages(messages, system, panel_outputs)
    body = {
        "model": ANALYST,
        "messages": analyst_messages,
        "system": analyst_system,
        "max_tokens": ANALYST_MAX_TOKENS,
        "stream": False,
    }
    return await _stream_to_text(client, bearer, body, timeout=READ_TIMEOUT)


async def _conductor(request, body):
    bearer = _bearer(request)
    messages = body.get("messages", [])
    system = body.get("system")
    want_stream = body.get("stream", True)
    # The client controls the final-answer length budget; route it to the
    # synthesizer (stage 2), not the analyst. Stage 1 has its own smaller budget.
    max_tokens = body.get("max_tokens", SYNTH_MAX_TOKENS)

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
            logger.info("conductor: panel responded %d/%d", len(ok), len(PANEL))

            analysis = await _run_analysis(client, bearer, messages, system, panel_outputs)
            logger.info("conductor: analysis %s -> synth=%s", "ok" if analysis else "failed", SYNTH)

            synth_messages, synth_system = _build_synth_messages(
                messages, system, panel_outputs, analysis
            )
            synth_body = {
                "model": SYNTH,
                "messages": synth_messages,
                "system": synth_system,
                "max_tokens": max_tokens,
                "stream": True,
            }
            for k in ("temperature", "top_p", "top_k", "stop_sequences"):
                if k in body:
                    synth_body[k] = body[k]

            resp = await _forward_once(client, request, synth_body, timeout=STREAM_TIMEOUT)
            if resp is not None and resp.status_code == 200 and "text/html" not in resp.headers.get("content-type", ""):
                logger.info("conductor: streaming synthesizer")
                return _stream(resp, client=client, label="synth")
            if resp is not None:
                await resp.aclose()
            await client.aclose()
            return _anthropic_error(f"conductor: synthesizer {SYNTH} failed to stream.")
        except Exception as e:  # noqa: BLE001
            await client.aclose()
            return _anthropic_error(f"conductor: orchestration error: {type(e).__name__}: {e}")

    async with httpx.AsyncClient() as client:
        logger.info("conductor: fanning out to panel=%s", PANEL)
        panel_outputs = await _run_panel(
            client, bearer, messages, system, max_tokens=PANEL_MAX_TOKENS
        )
        ok = [m for m, t in panel_outputs if t]
        logger.info("conductor: panel responded %d/%d", len(ok), len(PANEL))

        analysis = await _run_analysis(client, bearer, messages, system, panel_outputs)
        logger.info("conductor: analysis %s -> synth=%s", "ok" if analysis else "failed", SYNTH)

        synth_messages, synth_system = _build_synth_messages(
            messages, system, panel_outputs, analysis
        )
        synth_body = {
            "model": SYNTH,
            "messages": synth_messages,
            "system": synth_system,
            "max_tokens": max_tokens,
            "stream": False,
        }
        for k in ("temperature", "top_p", "top_k", "stop_sequences"):
            if k in body:
                synth_body[k] = body[k]

        text = await _stream_to_text(client, bearer, synth_body, timeout=READ_TIMEOUT)
        if text is None:
            return _anthropic_error(f"conductor: synthesizer {SYNTH} failed to respond.")
        return JSONResponse(
            {
                "id": "msg_conductor",
                "type": "message",
                "role": "assistant",
                "model": CONDUCTOR_ID,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequences": synth_body.get("stop_sequences", []),
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
            status_code=200,
        )


@app.get("/health")
async def health():
    return {
        "ok": True,
        "conductor": "fusion",
        "preset": PRESET or DEFAULT_PRESET,
        "panel": PANEL,
        "analyst": ANALYST,
        "synthesizer": SYNTH,
        # Expose the full preset catalog so a client (the launcher's preset
        # picker) can show which 3 models each preset uses without hardcoding.
        "presets": PRESETS,
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