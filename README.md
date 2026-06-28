# hf-claude

extension for `hf` to launch Claude Code with [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index)

It lets you pick:
- model (from `https://router.huggingface.co/v1/models`)
- provider (`auto` or a concrete provider for the selected model)

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
that implements the OpenRouter Fusion pattern: a prompt is fanned out to a
panel of cheap/fast models in parallel, and a frontier **judge** model
synthesizes their outputs into a single streamed response. The idea is many
inexpensive perspectives, one high-quality synthesis — often better than any
single model in the panel, at a fraction of a frontier-model's token cost.

It runs as a tiny local proxy bundled with this extension.

### How it works

1. Claude Code sends a request to `huggingface/conductor`.
2. The proxy fans the prompt out to every model in the panel **in parallel**
   (non-streaming, short max-tokens each).
3. Once the panel responds, a judge model receives all the answers plus the
   original request and synthesizes a single unified response.
4. That synthesized response is streamed back to Claude Code.

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

### Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_CONDUCTOR_PANEL` | `Qwen/Qwen2.5-7B-Instruct,meta-llama/Llama-3.1-8B-Instruct,allenai/Olmo-3-7B-Instruct` | Comma-separated panel models (cheap/fast, run in parallel) |
| `HF_CONDUCTOR_JUDGE` | `MiniMaxAI/MiniMax-M3` | Non-reasoning frontier model that synthesizes the panel outputs |
| `HF_CONDUCTOR_PANEL_MAX_TOKENS` | `4096` | Max tokens per panel response |
| `HF_CONDUCTOR_JUDGE_MAX_TOKENS` | `16384` | Max tokens for the judge's synthesized response |
| `HF_CONDUCTOR_PANEL_TIMEOUT` | `45` | Seconds before a panel model is treated as failed |
| `HF_CONDUCTOR_PORT` | `8080` | Local proxy port |
| `HF_CONDUCTOR_PROXY` | *(auto-detected)* | Path to `hf_conductor_proxy.py` |
| `HF_CONDUCTOR_READ_TIMEOUT` | `120` | Seconds before the judge stream is treated as failed |
| `HF_CONDUCTOR_STREAM_TIMEOUT` | `600` | Read timeout for the relayed judge stream (headroom for reasoning gaps) |
| `HF_CONDUCTOR_LOG_LEVEL` | `INFO` | Proxy log level |

> Note: this is a client-side orchestration layer. Requests are still
> authenticated with your HF token and metered by the router against the panel
> and judge models that actually serve each request, so a conductor turn costs
> roughly (sum of panel tokens) + (judge synthesis tokens).
