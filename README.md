# hf-claude

extension for `hf` to launch Claude Code with [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index)

It lets you pick:
- model — by default the curated list at `https://router.huggingface.co/v1/models` (~120 models), with a **Search the full Hub catalog…** entry that searches every text-generation model that has an inference provider (~17k models on `https://huggingface.co/api/models`)
- provider (`auto` or a concrete provider for the selected model; Hub-only models are routed with `auto` since the Hub list carries no per-provider info)

Then it runs `claude --model <model[:provider]>` with the required env vars preconfigured.

## Requirements

- Claude Code CLI installed
- `curl`, `jq`, `bash`
- [`fzf`](https://github.com/junegunn/fzf#installation) *(optional)* — enables fuzzy model/provider search; without it the launcher falls back to an arrow-key menu (↑/↓ to move, Enter to select)
- A Hugging Face token, via either:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login        
# or
export HF_TOKEN='hf_...'
```

## Install

```bash
hf extensions install hf-claude
# add --force to reinstall the extension
```

## Run

```bash
hf claude
# or
hf extensions exec claude
```

Forward extra args to Claude Code:

```bash
hf claude --help
hf extensions exec claude -- --help
```

## Searching the full Hub catalog

The default model menu shows the router's curated list (~120 models). To reach
the long tail, pick **🔍 Search the full Hub catalog…** at the top of that menu,
or open straight into the search prompt with `--all`:

```bash
hf claude --all
```

You'll be prompted for a search query (blank = whole catalog), which queries the
Hub's `api/models` endpoint filtered to text-generation models that have an
inference provider (~17,000 models). With `fzf` installed you fuzzy-match over
the results directly; without it, a type-to-filter prompt narrows the list
before an arrow-key menu.

Models found via Hub search are routed with `auto` provider selection (the Hub
listing doesn't expose which providers serve a model), so they launch as
`claude --model <model>` with no `:provider` suffix. Set
`HF_CLAUDE_ALL=1` to make search the default opening screen, and
`HF_CLAUDE_HUB_MAX_PAGES` (default `10`) to raise/lower the pagination cap on
broad queries.

## Billing to an organization

To bill inference usage to a Hugging Face organization instead of your personal
account, pass `--bill-to` (you must have Write privileges in the org):

```bash
hf claude --bill-to your-org-name
```

This sets the router's `X-HF-Bill-To` header (via `ANTHROPIC_CUSTOM_HEADERS`) on
every request. You can also set it via the `HF_BILL_TO` environment variable:

```bash
export HF_BILL_TO=your-org-name
hf claude
```

## `huggingface/conductor` — OpenRouter Fusion-style orchestrator

`huggingface/conductor` is an orchestrator option at the top of the model menu
that implements the [OpenRouter Fusion](https://openrouter.ai/blog/fusion)
pattern. It runs the full Fusion pipeline rather than a single synthesis pass:

1. The prompt is fanned out to a **panel** of diverse models **in parallel**
   (cheap, fast, different labs — they disagree productively).
2. An **analyst** model reads every panel response and produces *structured
   analysis*: consensus points, contradictions, partial coverage, unique
   insights, and blind spots.
3. A **synthesizer** writes the final streamed answer, **grounded in that
   analysis** — resolving contradictions, folding in unique insights, filling
   blind spots.

The two-stage analysis → synthesis is what OpenRouter found lets a panel of
budget models surpass a single frontier model on deep research. They also found
that fusing a model *with itself* (Opus 4.8 + Opus 4.8) beat solo Opus by 6.7
points — so a lot of the lift is the synthesis step itself. By default the
analyst and synthesizer are therefore the *same* frontier non-reasoning model;
the diversity that matters lives in the panel. Split analyst and synthesizer
into different models with env vars if you want cross-family checking.

It runs as a tiny local proxy bundled with this extension.

### How it works

1. Claude Code sends a request to `huggingface/conductor`.
2. The proxy fans the prompt out to every model in the panel **in parallel**
   (non-streaming, short max-tokens each).
3. The analyst model receives the panel answers plus the original request and
   emits a structured analysis (non-streaming).
4. The synthesizer model receives the original request, the panel answers, and
   the analysis, and streams the final unified response back to Claude Code.

### Requirements

- `python3`
- `pip install fastapi uvicorn httpx`

### Usage

Pick it from the top of the menu, or skip the menu entirely:

```bash
hf claude --conductor
```

The launcher starts the local proxy (default `127.0.0.1:8080`) — reusing one if
already running — and points Claude Code at it. Logs go to
`/tmp/hf-conductor.log`.

### Presets — Quality / Budget / Speed

The conductor offers three presets, mirroring OpenRouter Fusion's three
(`general-high` / `general-budget` / `general-fast`). Each is a **panel of 3**
diverse models from different labs — the composition changes, the size stays 3
(just like OpenRouter's three-member panels):

| Preset | Panel (3 models) | When to reach for it |
| --- | --- | --- |
| 🏆 **quality** | `deepseek-ai/DeepSeek-V4-Pro`, `zai-org/GLM-4.7`, `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Best synthesis — research, expert critique, where the cost of being wrong outweighs extra completions |
| 💰 **budget** | `Qwen/Qwen3.5-9B`, `openai/gpt-oss-20b`, `google/gemma-3-12b-it` | Cheap-but-capable diverse panel at a fraction of frontier cost |
| ⚡ **speed** | `Qwen/Qwen2.5-7B`, `meta-llama/Llama-3.1-8B-Instruct`, `allenai/Olmo-3-7B-Instruct` | Lowest latency; the default |

When you select the conductor interactively, a preset picker appears; the
explainer screen then shows the exact 3 panel models + analyst/synthesizer the
proxy will use this session. Pick a preset non-interactively with the
`HF_CONDUCTOR_PRESET` env var:

```bash
HF_CONDUCTOR_PRESET=quality hf claude --conductor
```

If a proxy is already running on a *different* preset, the launcher restarts it
so the chosen panel takes effect (the panel is resolved at proxy start).

When you pick the `🤗 huggingface/conductor` entry from the menu, an explainer
screen describes the Fusion pipeline and the resolved panel before Claude Code
starts; press Enter to continue or Ctrl-C to cancel. Skip it with
`HF_CONDUCTOR_SKIP_EXPLAINER=1` (or in any non-interactive run).

### Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_CONDUCTOR_PRESET` | `speed` | One of `quality` / `budget` / `speed`; sets the 3-model panel. Ignored if `HF_CONDUCTOR_PANEL` is set explicitly |
| `HF_CONDUCTOR_PANEL` | *(the preset's panel)* | Comma-separated panel models (overrides the preset; diverse, run in parallel) |
| `HF_CONDUCTOR_ANALYST` | `MiniMaxAI/MiniMax-M3` | Model that produces structured analysis of the panel outputs |
| `HF_CONDUCTOR_SYNTH` | `MiniMaxAI/MiniMax-M3` | Model that streams the final synthesized answer |
| `HF_CONDUCTOR_JUDGE` | *(unset)* | Legacy: if set, becomes the default for both `HF_CONDUCTOR_ANALYST` and `HF_CONDUCTOR_SYNTH` (one model for both stages, as in v0.3) |
| `HF_CONDUCTOR_PANEL_MAX_TOKENS` | `4096` | Max tokens per panel response |
| `HF_CONDUCTOR_ANALYST_MAX_TOKENS` | `3072` | Max tokens for the analyst's structured analysis |
| `HF_CONDUCTOR_JUDGE_MAX_TOKENS` | `16384` | Max tokens for the synthesizer's final answer |
| `HF_CONDUCTOR_PANEL_TIMEOUT` | `45` | Seconds before a panel model is treated as failed |
| `HF_CONDUCTOR_EXCLUDE_DOMAINS` | *(empty)* | Comma-separated domains to exclude from any future panel web tools (the Fusion blog's anti-contamination measure; no-op until you wire tools onto the panel) |
| `HF_CONDUCTOR_SKIP_EXPLAINER` | `0` | Set to `1` to skip the pre-launch 🤗 explainer screen when selecting the conductor |
| `HF_CONDUCTOR_PORT` | `8080` | Local proxy port |
| `HF_CONDUCTOR_PROXY` | *(auto-detected)* | Path to `hf_conductor_proxy.py` |
| `HF_CONDUCTOR_READ_TIMEOUT` | `120` | Seconds before the analyst / non-streaming synthesizer is treated as failed |
| `HF_CONDUCTOR_STREAM_TIMEOUT` | `600` | Read timeout for the relayed synthesizer stream (headroom for reasoning gaps) |
| `HF_CONDUCTOR_LOG_LEVEL` | `INFO` | Proxy log level |

> Note: this is a client-side orchestration layer. Requests are still
> authenticated with your HF token and metered by the router against the panel,
> analyst, and synthesizer models that actually serve each request, so a
> conductor turn costs roughly (sum of panel tokens) + (analyst tokens) +
> (synthesizer tokens).
