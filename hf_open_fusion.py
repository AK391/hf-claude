#!/usr/bin/env python3
"""huggingface/open-fusion — local OpenRouter-Fusion-style orchestrator.

When the selected model is `huggingface/open-fusion`, this proxy runs the
OpenRouter Fusion pipeline: it fans the prompt out to a panel of diverse models
in parallel, then a single judge model reads every panel response, produces a
structured analysis (consensus, contradictions, coverage, blind spots), and
writes the final answer grounded in it — in one inference pass, as the blog
describes ("A judge model reads every panel response and produces structured
analysis... The calling model then writes the final answer grounded in that
analysis"). The analysis preamble is stripped from the streamed output so the
client sees only the answer. A two-stage (separate analyst, then synthesizer)
fallback exists for operators who split the judge into two different models.

Speaks the Anthropic Messages API (`/v1/messages`), since Claude Code is
configured with ANTHROPIC_BASE_URL pointing here. Non-open-fusion requests are
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
        f"hf-open-fusion proxy: missing dependency ({e}).\n"
        "Install with:  pip install fastapi uvicorn httpx",
        file=sys.stderr,
    )
    sys.exit(1)

logging.basicConfig(
    level=os.environ.get("HF_OPEN_FUSION_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("hf-open-fusion")

app = FastAPI()

HF_ROUTER = "https://router.huggingface.co"
OPEN_FUSION_ID = "huggingface/open-fusion"
PORT = int(os.environ.get("HF_OPEN_FUSION_PORT", "8080"))
READ_TIMEOUT = float(os.environ.get("HF_OPEN_FUSION_READ_TIMEOUT", "120"))
# Streaming judges (reasoning models) can sit silent for a while during the
# thinking phase before emitting text. Give the relayed stream headroom so a
# long gap between SSE events doesn't trip the read timeout and abort mid-stream.
STREAM_READ_TIMEOUT = float(os.environ.get("HF_OPEN_FUSION_STREAM_TIMEOUT", "600"))

# OpenRouter Fusion pipeline:
#   panel -> diverse capable models answer the original request in parallel
#   judge -> reads every panel response, emits structured analysis
#             (consensus / contradictions / coverage / insights / blind spots),
#             then writes the final unified answer grounded in it — in ONE pass
# The blog's two headline results both flow from this structure: (1) a diverse
# frontier panel (e.g. Fable 5 + GPT-5.5) synthesizes to ~69% on DRACO vs ~65%
# solo, and (2) a *budget* panel of different families (Gemini Flash + Kimi +
# DeepSeek) beats individual frontier models. So the panel is chosen for
# diversity and cost, NOT raw size. The blog's judge is a single frontier model
# (Opus 4.8) that does analysis + answer in one turn — we follow that: one
# judge call, with a delimiter line separating the analysis preamble from the
# answer so the streaming transform can strip the preamble. OpenRouter also
# found fusing Opus 4.8 with itself beat solo Opus by 6.7pts, so a lot of the
# lift is the synthesis step itself. The default judge is GLM-5.2, a reasoning
# model — it spends tokens on hidden thinking blocks before emitting visible
# text, so keep the streaming budgets generous (STREAM_READ_TIMEOUT,
# SYNTH_MAX_TOKENS) or the answer can come back empty. To split the judge into
# two different models (analyst != synthesizer) for cross-family checking — a
# knob the blog doesn't have — set HF_OPEN_FUSION_ANALYST and
# HF_OPEN_FUSION_SYNTH to different models; that switches to two-stage mode.
#
# Three presets mirror the three Fusion presets on OpenRouter's Fusion page
# (general-high / general-budget / general-fast): Quality, Budget, Speed. Each
# is a 3-model diverse panel; the panel composition changes, the panel SIZE
# stays 3 (matches OpenRouter's three-member panels). A launcher-selectable
# preset writes the chosen panel into HF_OPEN_FUSION_PANEL so the proxy needs no
# preset awareness of its own.
PRESETS = {
    # 🏆 Quality — strong, diverse frontier panel for best synthesis. Different
    # families (MiniMax / DeepSeek / Kimi) so they disagree productively and the
    # analyst has real consensus/contradiction signal to work with.
    "quality": "MiniMaxAI/MiniMax-M3,deepseek-ai/DeepSeek-V3,moonshotai/Kimi-K2.6",
    # 💰 Budget — cheap-but-capable diverse panel, OpenRouter's "budget panel"
    # finding: this class beat individual frontier models at a fraction of cost.
    "budget": "deepreinforce-ai/Ornith-1.0-35B,Qwen/Qwen3.6-35B-A3B,google/gemma-4-31B-it",
    # ⚡ Speed — small fast models from different labs for lowest latency.
    "speed": "Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,allenai/Olmo-3-7B-Instruct",
}
DEFAULT_PANEL = PRESETS["speed"]  # Speed is the no-surprise default (fast + cheap).
DEFAULT_PRESET = "speed"
PRESET = os.environ.get("HF_OPEN_FUSION_PRESET", "").strip().lower()
if PRESET and PRESET in PRESETS:
    DEFAULT_PANEL = PRESETS[PRESET]
DEFAULT_ANALYST = "zai-org/GLM-5.2"
DEFAULT_SYNTH = "zai-org/GLM-5.2"
PANEL = [m.strip() for m in os.environ.get("HF_OPEN_FUSION_PANEL", DEFAULT_PANEL).split(",") if m.strip()]
# The judge. HF_OPEN_FUSION_JUDGE is the primary knob — it sets the single model
# that does analysis + answer in one pass (the blog's design). The two vars below
# override it per-stage for the two-stage cross-family fallback; when unset they
# inherit the judge. (HF_OPEN_FUSION_JUDGE was also the v0.3 name for this, hence
# the back-compat.)
_JUDGE_ENV = os.environ.get("HF_OPEN_FUSION_JUDGE", "")
ANALYST = os.environ.get("HF_OPEN_FUSION_ANALYST", _JUDGE_ENV or DEFAULT_ANALYST)
SYNTH = os.environ.get("HF_OPEN_FUSION_SYNTH", _JUDGE_ENV or DEFAULT_SYNTH)
# One-call vs two-stage. The blog's judge does analysis + answer in ONE inference
# pass, so the default is one-call (analyst == synth). Only when an operator splits
# analyst and synthesizer into DIFFERENT models — a cross-family checking knob the
# blog doesn't have — do we fall back to the two-stage (separate analyst, then synth)
# pipeline, since each stage needs its own model. HF_OPEN_FUSION_TWO_STAGE=1 forces
# two-stage even with a single model (e.g. to keep the non-streaming analyst).
TWO_STAGE = ANALYST != SYNTH or os.environ.get("HF_OPEN_FUSION_TWO_STAGE", "").strip().lower() in ("1", "true", "yes")
JUDGE = ANALYST  # the one-call judge model (== SYNTH in one-call mode)
PANEL_TIMEOUT = float(os.environ.get("HF_OPEN_FUSION_PANEL_TIMEOUT", "45"))
# Reasoning judges (e.g. GLM-5.2) spend tokens on hidden thinking blocks before
# emitting the final text, so keep the budgets generous or the answer is empty.
PANEL_MAX_TOKENS = int(os.environ.get("HF_OPEN_FUSION_PANEL_MAX_TOKENS", "4096"))
ANALYST_MAX_TOKENS = int(os.environ.get("HF_OPEN_FUSION_ANALYST_MAX_TOKENS", "3072"))
SYNTH_MAX_TOKENS = int(os.environ.get("HF_OPEN_FUSION_JUDGE_MAX_TOKENS", "16384"))
# The one-call judge streams analysis-then-answer in one turn; the answer begins
# after this marker line. The stream transform below strips everything up to and
# including it so the client sees only the final answer. Chose a sentinel that is
# unlikely to occur in prose; if the judge never emits it, we fall back to
# relaying the whole text so the user isn't left with an empty answer.
FINAL_ANSWER_MARKER = "=== FINAL ANSWER ==="
# Keep HF_OPEN_FUSION_JUDGE_MAX_TOKENS as the documented knob (above). v0.3 users
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

# The open-fusion panel can't see the panel/judge tool calls, so there's no
# contamination surface to block on this side. But the OpenRouter Fusion post
# flagged a real risk — panel models with web access found their eval rubric
# online. If you wire server-side tools onto the panel later (web_search /
# web_fetch), exclude the domains that host anything you don't want the panel
# reading (benchmark rubrics, the user's own answer keys, etc.). Parsed once at
# import; empty by default so it's a no-op until you need it.
_EXCLUDE_DOMAINS_RAW = os.environ.get("HF_OPEN_FUSION_EXCLUDE_DOMAINS", "")
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


# --- One-call judge: strip the analysis preamble from the streamed answer --------
#
# The judge streams analysis-then-answer in a single turn, with the answer starting
# after the FINAL_ANSWER_MARKER line. We can't relay that raw: the client (Claude
# Code) would render the analysis scaffold as the answer. So we parse the upstream
# Anthropic SSE on the fly, accumulate `text` deltas until the marker, then re-emit
# a clean Anthropic SSE stream containing only the post-marker text.
#
# Fragility notes (why this is conservative):
#   - We only act on `content_block_delta` events whose delta is a `text_delta`.
#     Reasoning models emit a separate `thinking` content block; we ignore it, so
#     hidden thinking never reaches the client regardless of the marker.
#   - The marker can land mid-delta or straddle two deltas, so we buffer across
#     deltas and search the running buffer; we emit nothing until we've found it.
#   - Fallback: if the stream ends and we never saw the marker, emit everything we
#     buffered as the answer. Better a leaked analysis than an empty response.
#   - Non-text SSE events we don't understand are dropped: in one-call mode the
#     judge's stream is text-only from our perspective, and reconstructing tool_use
#     or image blocks here is out of scope. The two-stage path doesn't use this.


def _sse_event(data: dict) -> bytes:
    """Serialize one Anthropic SSE event frame: `event: <t>\\ndata: <json>\\n\\n`."""
    return b"event: " + data.get("type", "unknown").encode() + b"\ndata: " + json.dumps(data).encode() + b"\n\n"


def _marker_split(buf: str):
    """Find the marker line in buf.

    Returns (before, after, found) where `after` is the answer text following the
    marker line (with the marker and its trailing newline removed). `found` is True
    only when the marker has fully arrived *and* we can see at least one byte past
    it — so a marker that lands as the very last bytes of a delta is held until the
    next delta confirms there's answer text (or the stream ends).
    """
    idx = buf.find(FINAL_ANSWER_MARKER)
    if idx == -1:
        return buf, "", False
    after_start = idx + len(FINAL_ANSWER_MARKER)
    # Swallow the marker line's trailing newline(s) so the answer doesn't start
    # with a blank line.
    rest = buf[after_start:]
    rest = rest.lstrip("\r\n")
    return buf[:idx], rest, True


async def _strip_preamble_stream(resp, client=None, label="judge"):
    """Re-emit the judge's SSE stream with the analysis preamble removed.

    Emits a valid Anthropic streaming sequence: message_start, a single text
    content_block (start/deltas/stop), message_delta (stop_reason), message_stop —
    containing only the final-answer text.
    """
    message_id = "msg_open_fusion"
    # Anthropic streaming ids are stable per-block; we mint one for our single
    # reconstructed text block.
    block_id = "block_open_fusion"

    async def gen():
        buf = ""
        marker_found = False
        started = False  # have we emitted the content_block_start + message_start?
        finished = False
        input_tokens = 0
        output_tokens = 0

        def begin_frames():
            """Emit the opening frames for the (re)constructed stream."""
            out = b""
            if not started:
                out += _sse_event({
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": OPEN_FUSION_ID,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequences": None,
                        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                    },
                })
                out += _sse_event({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": "", "citations": None},
                })
            return out

        def emit_text(text: str) -> bytes:
            nonlocal started, output_tokens
            out = begin_frames()
            started = True
            if text:
                output_tokens += len(text) // 4  # rough, only for usage echo
                out += _sse_event({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                })
            return out

        try:
            async for line in _aiter_sse_lines(resp):
                if not line or line.startswith(b":"):
                    continue
                if line.startswith(b"event:"):
                    continue  # we drive our own event names
                if line.startswith(b"data:"):
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        evt = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    etype = evt.get("type")
                    # Capture usage from the upstream message_start/message_delta.
                    if etype == "message_start":
                        try:
                            input_tokens = evt["message"]["usage"]["input_tokens"]
                        except (KeyError, TypeError):
                            pass
                        continue
                    if etype == "message_delta":
                        try:
                            output_tokens = max(output_tokens, evt.get("usage", {}).get("output_tokens", output_tokens))
                        except (KeyError, TypeError):
                            pass
                        continue
                    if etype == "content_block_delta":
                        delta = evt.get("delta", {})
                        if delta.get("type") != "text_delta":
                            continue  # ignore thinking_delta, input_json_delta, etc.
                        text = delta.get("text", "")
                        if not text:
                            continue
                        if marker_found:
                            yield emit_text(text)
                            continue
                        buf += text
                        before, after, found = _marker_split(buf)
                        if found:
                            marker_found = True
                            buf = after
                            # `before` is the analysis preamble — discarded. Only the
                            # text after the marker reaches the client.
                            yield emit_text(after)
                        # Until the marker is found we buffer rather than emit: the
                        # preamble is throwaway, so there's no latency win in streaming
                        # it (we'd only have to suppress it), and partial-emit logic
                        # across delta boundaries is error-prone. The preamble is short.
                    # We deliberately drop content_block_start/stop, ping, error
                    # etc. from the upstream and synthesize our own framing.
            # Stream ended.
            if not marker_found:
                # Fallback: never saw the marker — relay everything as the answer.
                logger.warning("%s stream: no '%s' marker found; relaying full judge text", label, FINAL_ANSWER_MARKER)
                yield emit_text(buf)
            # Close out the message. If we never emitted any text at all, still
            # send the framing so the client sees a well-formed (if empty) message.
            if not started:
                yield begin_frames()
                started = True
            out = b""
            out += _sse_event({"type": "content_block_stop", "index": 0})
            out += _sse_event({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequences": None},
                "usage": {"output_tokens": output_tokens},
            })
            out += _sse_event({"type": "message_stop"})
            yield out
            finished = True
        except Exception as e:  # noqa: BLE001
            logger.warning("%s preamble-strip stream ended early: %s: %s", label, type(e).__name__, e)
        finally:
            if not finished:
                try:
                    yield _sse_event({"type": "message_stop"})
                except Exception:  # noqa: BLE001
                    pass
            await resp.aclose()
            if client is not None:
                await client.aclose()

    return StreamingResponse(
        gen(),
        status_code=200,
        headers={"content-type": "text/event-stream", "cache-control": "no-cache"},
        media_type="text/event-stream",
    )


async def _aiter_sse_lines(resp):
    """Yield raw SSE line bytes from a streaming httpx response, splitting on newlines.

    SSE frames are separated by blank lines, but for our purposes we want the
    individual `data:` / `event:` / comment lines, so we split on any of \\n / \\r\\n
    / \\r and yield each non-frame-separator line. The frame separators (empty
    lines) are yielded as empty bytes so callers can ignore them.
    """
    pending = b""
    async for chunk in resp.aiter_bytes():
        pending += chunk
        # Split on any newline variant; keep handles incomplete trailing line.
        while True:
            # Find earliest of \r\n, \n, \r
            n_idx = pending.find(b"\n")
            r_idx = pending.find(b"\r")
            idx = -1
            length = 1
            if n_idx != -1 and (r_idx == -1 or n_idx <= r_idx):
                idx = n_idx
                length = 1
            elif r_idx != -1:
                idx = r_idx
                length = 2 if pending[r_idx:r_idx + 1] == b"\r" and pending[r_idx + 1:r_idx + 2] == b"\n" else 1
            if idx == -1:
                break
            line = pending[:idx]
            pending = pending[idx + length:]
            yield line
    if pending:
        yield pending


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


# The judge (one-call mode). The OpenRouter Fusion blog describes the judge as a
# single model that does BOTH steps in one turn: "A judge model reads every panel
# response and produces structured analysis: consensus points, contradictions,
# partial coverage, unique insights, blind spots. The calling model then writes the
# final answer grounded in that analysis." So the default is one inference pass —
# analyze, then answer — not two separate analyst/synthesizer calls. We ask the
# judge to emit a delimiter line before its final answer so the streaming transform
# below can strip the analysis preamble and relay only the answer to the client.
JUDGE_INSTRUCTION = (
    "You are a Fusion judge. Several AI models answered the same request "
    "independently. First, read every response and produce a concise structured "
    "analysis covering: Consensus (points most/all agree on), Contradictions (where "
    "they disagree and which is better supported), Partial coverage (points only some "
    "made), Unique insights (the strongest non-obvious points), and Blind spots "
    "(things the request asked for that none fully addressed). Then write one "
    "optimal, unified response to the ORIGINAL request, grounded in your analysis: "
    "resolve the contradictions, fold in the unique insights, fill the blind spots.\n"
    "\n"
    f"Format: write the analysis first, then a line containing exactly `{FINAL_ANSWER_MARKER}` "
    "on its own line, then the final answer. Only the text after that marker is shown "
    "to the user, so put nothing after it but the answer itself. Do not mention the "
    "panel, the analysis, or this process in the final answer — answer as a single "
    "assistant. Do not name the models beyond attributing a point to 'one response'."
)

# Two-stage fallback instructions, used only when the operator has split analyst and
# synthesizer into different models (HF_OPEN_FUSION_ANALYST != HF_OPEN_FUSION_SYNTH)
# for cross-family checking — a knob the blog doesn't have. Keeps the original
# non-streaming analyst + streaming synthesizer design.
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

SYNTH_INSTRUCTION = (
    "You are the final synthesizer. You are given a user's original request, the "
    "raw perspectives from a panel of models, and a structured analysis of those "
    "perspectives. Write one optimal, unified response to the ORIGINAL request, "
    "grounded in the analysis: resolve the contradictions, fold in the unique "
    "insights, and fill the blind spots. Do not mention the panel, the analysis, or "
    "this process — just answer as a single assistant."
)


def _fused_panel(panel_outputs):
    """Render panel outputs as numbered perspectives, or a failure sentinel."""
    return "\n\n".join(
        f"### Perspective {i + 1}:\n{txt}"
        for i, (_model, txt) in enumerate(panel_outputs)
        if txt
    ) or "(all panel models failed to respond)"


def _build_judge_messages(messages, system, panel_outputs):
    """One-call judge prompt: panel responses -> (analysis + final answer) in one turn."""
    user_text = _user_text(messages)
    fused = _fused_panel(panel_outputs)
    judge_system = _merge_system(system, JUDGE_INSTRUCTION)
    judge_messages = [
        {
            "role": "user",
            "content": (
                f"--- ORIGINAL USER REQUEST ---\n{user_text}\n\n"
                f"--- PANEL RESPONSES ---\n{fused}\n\n"
                f"--- YOUR TASK ---\nAnalyze the panel, then write the final unified answer."
            ),
        }
    ]
    return judge_messages, judge_system


def _build_analyst_messages(messages, system, panel_outputs):
    """Two-stage fallback, step 1: panel responses -> structured analysis."""
    user_text = _user_text(messages)
    fused = _fused_panel(panel_outputs)
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
    """Two-stage fallback, step 2: original request + panel + analysis -> final answer."""
    user_text = _user_text(messages)
    fused = _fused_panel(panel_outputs)
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


def _strip_preamble_text(text):
    """Non-streaming counterpart to _strip_preamble_stream: drop everything up to and
    including the FINAL_ANSWER_MARKER line. Falls back to the full text if the marker
    is absent, so a judge that ignored the format instruction still yields an answer.
    """
    if not text:
        return text
    idx = text.find(FINAL_ANSWER_MARKER)
    if idx == -1:
        return text
    return text[idx + len(FINAL_ANSWER_MARKER):].lstrip("\r\n")


async def _run_judge(client, bearer, request, body, messages, system, panel_outputs):
    """One-call judge: stream (or collect) the analysis-then-answer in a single turn.

    Returns a FastAPI Response — a StreamingResponse with the preamble stripped for
    streaming requests, or a JSONResponse for non-streaming. Raises on hard failure
    so the caller can convert to an Anthropic error.
    """
    want_stream = body.get("stream", True)
    max_tokens = body.get("max_tokens", SYNTH_MAX_TOKENS)
    judge_messages, judge_system = _build_judge_messages(messages, system, panel_outputs)
    judge_body = {
        "model": JUDGE,
        "messages": judge_messages,
        "system": judge_system,
        "max_tokens": max_tokens,
        "stream": want_stream,
    }
    for k in ("temperature", "top_p", "top_k", "stop_sequences"):
        if k in body:
            judge_body[k] = body[k]

    if want_stream:
        resp = await _forward_once(client, request, judge_body, timeout=STREAM_TIMEOUT)
        if resp is None or resp.status_code != 200 or "text/html" in resp.headers.get("content-type", ""):
            if resp is not None:
                await resp.aclose()
            raise RuntimeError(f"judge {JUDGE} failed to stream")
        logger.info("open-fusion: streaming one-call judge (preamble stripped)")
        return _strip_preamble_stream(resp, client=client, label="judge")

    text = await _stream_to_text(client, bearer, judge_body, timeout=READ_TIMEOUT)
    if text is None:
        raise RuntimeError(f"judge {JUDGE} failed to respond")
    answer = _strip_preamble_text(text)
    return JSONResponse(
        {
            "id": "msg_open_fusion",
            "type": "message",
            "role": "assistant",
            "model": OPEN_FUSION_ID,
            "content": [{"type": "text", "text": answer}],
            "stop_reason": "end_turn",
            "stop_sequences": judge_body.get("stop_sequences", []),
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
        status_code=200,
    )


async def _open_fusion(request, body):
    bearer = _bearer(request)
    messages = body.get("messages", [])
    system = body.get("system")
    want_stream = body.get("stream", True)
    # The client controls the final-answer length budget; route it to the judge's
    # answer step. (In two-stage mode the analyst has its own smaller budget.)
    max_tokens = body.get("max_tokens", SYNTH_MAX_TOKENS)

    # NOTE: we deliberately do NOT use `async with` for the streaming path's
    # client — the StreamingResponse body generator runs after this handler
    # returns, so the client must stay alive until the stream finishes. The
    # client is closed inside _stream()/_strip_preamble_stream()'s generator
    # finally-block instead.
    if want_stream:
        client = httpx.AsyncClient()
        try:
            logger.info("open-fusion: fanning out to panel=%s", PANEL)
            panel_outputs = await _run_panel(
                client, bearer, messages, system, max_tokens=PANEL_MAX_TOKENS
            )
            ok = [m for m, t in panel_outputs if t]
            logger.info("open-fusion: panel responded %d/%d", len(ok), len(PANEL))

            if TWO_STAGE:
                analysis = await _run_analysis(client, bearer, messages, system, panel_outputs)
                logger.info("open-fusion: analysis %s -> synth=%s", "ok" if analysis else "failed", SYNTH)
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
                    logger.info("open-fusion: streaming synthesizer (two-stage)")
                    return _stream(resp, client=client, label="synth")
                if resp is not None:
                    await resp.aclose()
                await client.aclose()
                return _anthropic_error(f"open-fusion: synthesizer {SYNTH} failed to stream.")

            # One-call judge (default, blog-faithful): analysis + answer in one stream.
            return await _run_judge(client, bearer, request, body, messages, system, panel_outputs)
        except Exception as e:  # noqa: BLE001
            await client.aclose()
            return _anthropic_error(f"open-fusion: orchestration error: {type(e).__name__}: {e}")

    async with httpx.AsyncClient() as client:
        logger.info("open-fusion: fanning out to panel=%s", PANEL)
        panel_outputs = await _run_panel(
            client, bearer, messages, system, max_tokens=PANEL_MAX_TOKENS
        )
        ok = [m for m, t in panel_outputs if t]
        logger.info("open-fusion: panel responded %d/%d", len(ok), len(PANEL))

        if TWO_STAGE:
            analysis = await _run_analysis(client, bearer, messages, system, panel_outputs)
            logger.info("open-fusion: analysis %s -> synth=%s", "ok" if analysis else "failed", SYNTH)
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
                return _anthropic_error(f"open-fusion: synthesizer {SYNTH} failed to respond.")
            return JSONResponse(
                {
                    "id": "msg_open_fusion",
                    "type": "message",
                    "role": "assistant",
                    "model": OPEN_FUSION_ID,
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "stop_sequences": synth_body.get("stop_sequences", []),
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
                status_code=200,
            )

        # One-call judge, non-streaming.
        return await _run_judge(client, bearer, request, body, messages, system, panel_outputs)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "open_fusion": "fusion",
        "preset": PRESET or DEFAULT_PRESET,
        "panel": PANEL,
        # one-call (default, blog-faithful): a single judge does analysis + answer.
        # two_stage: split analyst + synthesizer (used only when they differ, or
        # HF_OPEN_FUSION_TWO_STAGE=1). The launcher's explainer branches on this.
        "two_stage": TWO_STAGE,
        "judge": JUDGE,
        # Kept for back-comat with older clients that read analyst/synthesizer.
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
        return _anthropic_error("open-fusion: could not parse request body as JSON", 400)

    if body.get("model") == OPEN_FUSION_ID:
        return await _open_fusion(request, body)

    # Non-open-fusion model: forward once; convert HTML errors so client never hangs.
    # Client stays alive for the streamed response (closed by _stream's generator).
    client = httpx.AsyncClient()
    resp = await _forward_once(client, request, body)
    if resp is not None and resp.status_code == 200 and "text/html" not in resp.headers.get("content-type", ""):
        return _stream(resp, client=client, label=f"passthrough:{body.get('model')}")
    if resp is not None:
        await resp.aclose()
    await client.aclose()
    return _anthropic_error(f"open-fusion: upstream provider failed for {body.get('model')}.")


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