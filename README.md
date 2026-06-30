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

## `huggingface/open-fusion` — OpenRouter Fusion-style orchestrator

`huggingface/open-fusion` is an orchestrator option at the top of the model menu
that implements the [OpenRouter Fusion](https://openrouter.ai/blog/fusion)
pattern. It runs the full Fusion pipeline rather than a single synthesis pass:

1. The prompt is fanned out to a **panel** of diverse models **in parallel**
   (cheap, fast, different labs — they disagree productively).
2. A single **judge** model reads every panel response, produces *structured
   analysis* (consensus points, contradictions, partial coverage, unique
   insights, blind spots), and then writes the final streamed answer **grounded
   in that analysis** — resolving contradictions, folding in unique insights,
   filling blind spots. Both steps happen in **one inference pass**, as the blog
   describes; the analysis preamble is stripped from the stream so only the
   answer is returned.

The analysis → synthesis step is what OpenRouter found lets a panel of budget
models surpass a single frontier model on deep research. They also found that
fusing a model *with itself* (Opus 4.8 + Opus 4.8) beat solo Opus by 6.7 points —
so a lot of the lift is the synthesis step itself. The blog's judge is a single
frontier model (Opus 4.8), and we follow that: one judge (GLM-5.2 by default, a
reasoning model — it thinks before it streams, so keep the streaming budgets
generous); the diversity that matters lives in the panel. To split the judge into
two different models (a separate analyst and synthesizer) for cross-family
checking — a knob the blog doesn't have — set `HF_OPEN_FUSION_ANALYST` and
`HF_OPEN_FUSION_SYNTH` to different models; that switches to two-stage mode.

It runs as a tiny local proxy bundled with this extension.

### How it works

1. Claude Code sends a request to `huggingface/open-fusion`.
2. The proxy fans the prompt out to every model in the panel **in parallel**
   (non-streaming, short max-tokens each).
3. A single **judge** model receives the panel answers plus the original request,
   writes a structured analysis (consensus / contradictions / coverage / blind
   spots), and then streams the final unified answer grounded in it — **in one
   inference pass**, exactly as OpenRouter's Fusion blog describes ("A judge model
   reads every panel response and produces structured analysis… The calling model
   then writes the final answer grounded in that analysis"). The analysis preamble
   is stripped from the stream, so Claude Code sees only the final answer.

### Requirements

- `python3`
- `pip install fastapi uvicorn httpx`

### Usage

Pick it from the top of the menu, or skip the menu entirely:

```bash
hf claude --open-fusion
```

The launcher starts the local proxy (default `127.0.0.1:8080`) — reusing one if
already running — and points Claude Code at it. Logs go to
`/tmp/hf-open-fusion.log`.

### Presets — Quality / Budget / Speed

Open Fusion offers three presets, mirroring OpenRouter Fusion's three
(`general-high` / `general-budget` / `general-fast`). Each is a **panel of 3**
diverse models from different labs — the composition changes, the size stays 3
(just like OpenRouter's three-member panels):

| Preset | Panel (3 models) | When to reach for it |
| --- | --- | --- |
| 🏆 **quality** | `MiniMaxAI/MiniMax-M3`, `deepseek-ai/DeepSeek-V3`, `moonshotai/Kimi-K2.6` | Best synthesis — research, expert critique, where the cost of being wrong outweighs extra completions |
| 💰 **budget** | `deepreinforce-ai/Ornith-1.0-35B`, `Qwen/Qwen3.6-35B-A3B`, `google/gemma-4-31B-it` | Cheap-but-capable diverse panel at a fraction of frontier cost |
| ⚡ **speed** | `Qwen/Qwen2.5-7B`, `meta-llama/Llama-3.1-8B-Instruct`, `allenai/Olmo-3-7B-Instruct` | Lowest latency; the default |

When you select Open Fusion interactively, a preset picker appears; the
explainer screen then shows the exact 3 panel models + judge the proxy will use
this session. Pick a preset non-interactively with the
`HF_OPEN_FUSION_PRESET` env var:

```bash
HF_OPEN_FUSION_PRESET=quality hf claude --open-fusion
```

If a proxy is already running on a *different* preset, the launcher restarts it
so the chosen panel takes effect (the panel is resolved at proxy start).

When you pick the `🤗 huggingface/open-fusion` entry from the menu, an explainer
screen describes the Fusion pipeline and the resolved panel before Claude Code
starts; press Enter to continue or Ctrl-C to cancel. Skip it with
`HF_OPEN_FUSION_SKIP_EXPLAINER=1` (or in any non-interactive run).

### Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_OPEN_FUSION_PRESET` | `speed` | One of `quality` / `budget` / `speed`; sets the 3-model panel. Ignored if `HF_OPEN_FUSION_PANEL` is set explicitly |
| `HF_OPEN_FUSION_PANEL` | *(the preset's panel)* | Comma-separated panel models (overrides the preset; diverse, run in parallel) |
| `HF_OPEN_FUSION_JUDGE` | `zai-org/GLM-5.2` | The single judge model: reads the panel, writes structured analysis, then streams the final answer — in one pass (blog-faithful). Alias default for the two below |
| `HF_OPEN_FUSION_ANALYST` | `zai-org/GLM-5.2` | Analyst model (two-stage mode). Defaults to `HF_OPEN_FUSION_JUDGE` |
| `HF_OPEN_FUSION_SYNTH` | `zai-org/GLM-5.2` | Synthesizer model (two-stage mode). Defaults to `HF_OPEN_FUSION_JUDGE` |
| `HF_OPEN_FUSION_TWO_STAGE` | `0` | Force two-stage (separate analyst + synthesizer calls) even when they're the same model. Auto-enabled when analyst ≠ synthesizer |
| `HF_OPEN_FUSION_PANEL_MAX_TOKENS` | `4096` | Max tokens per panel response |
| `HF_OPEN_FUSION_ANALYST_MAX_TOKENS` | `3072` | Max tokens for the analyst's structured analysis (two-stage mode) |
| `HF_OPEN_FUSION_JUDGE_MAX_TOKENS` | `16384` | Max tokens for the judge's analysis + final answer (one-call) / synthesizer (two-stage) |
| `HF_OPEN_FUSION_PANEL_TIMEOUT` | `45` | Seconds before a panel model is treated as failed |
| `HF_OPEN_FUSION_EXCLUDE_DOMAINS` | *(empty)* | Comma-separated domains to exclude from any future panel web tools (the Fusion blog's anti-contamination measure; no-op until you wire tools onto the panel) |
| `HF_OPEN_FUSION_SKIP_EXPLAINER` | `0` | Set to `1` to skip the pre-launch 🤗 explainer screen when selecting Open Fusion |
| `HF_OPEN_FUSION_PORT` | `8080` | Local proxy port |
| `HF_OPEN_FUSION_PROXY` | *(auto-detected)* | Path to `hf_open_fusion.py` |
| `HF_OPEN_FUSION_READ_TIMEOUT` | `120` | Seconds before the judge / non-streaming response is treated as failed |
| `HF_OPEN_FUSION_STREAM_TIMEOUT` | `600` | Read timeout for the relayed judge stream (headroom for reasoning gaps) |
| `HF_OPEN_FUSION_LOG_LEVEL` | `INFO` | Proxy log level |

> Note: this is a client-side orchestration layer. Requests are still
> authenticated with your HF token and metered by the router against the panel
> and judge models that actually serve each request, so an Open Fusion turn costs
> roughly (sum of panel tokens) + (judge tokens). In two-stage mode the judge
> tokens are (analyst tokens) + (synthesizer tokens).
