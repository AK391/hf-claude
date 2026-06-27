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

## Resilient orchestration with `huggingface/conductor`

When an underlying provider drops, rate-limits, or returns a gateway error,
Claude Code retries the same endpoint up to 10 times and then hangs — because
the router returns a generic HTML error page instead of a JSON error the client
can act on.

`huggingface/conductor` is a resilient orchestrator option that appears at the
top of the model menu. Selecting it transparently failovers across:

1. all providers for a base model (e.g. `zai-org/GLM-5.2`),
2. an equivalent model class (e.g. `MiniMaxAI/MiniMax-M3`) and *its* providers,

so a transient upstream failure is handled under the hood instead of hanging
your session. It runs as a tiny local proxy bundled with this extension.

### Requirements

- `python3`
- `pip install fastapi uvicorn httpx`

### Usage

Pick it from the top of the menu, or skip the menu entirely:

```bash
hf claude --conductor
```

The launcher starts the local proxy (default `127.0.0.1:8080`) — reusing one if
already running — and points Claude Code at it. Failover logs go to
`/tmp/hf-conductor.log`.

### Configuration (environment variables)

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_CONDUCTOR_BASE_MODEL` | `zai-org/GLM-5.2` | Primary model the conductor tries first |
| `HF_CONDUCTOR_FALLBACK_MODEL` | `MiniMaxAI/MiniMax-M3` | Equivalent model tried when all providers for the base fail |
| `HF_CONDUCTOR_PORT` | `8080` | Local proxy port |
| `HF_CONDUCTOR_PROXY` | *(auto-detected)* | Path to `hf_conductor_proxy.py` |
| `HF_CONDUCTOR_READ_TIMEOUT` | `60` | Seconds before a hung upstream counts as a failure |
| `HF_CONDUCTOR_LOG_LEVEL` | `INFO` | Proxy log level |

> Note: this is a client-side resilience layer. It does **not** change billing —
> requests are still authenticated with your HF token and metered by the router
> against whichever provider/model actually serves each request.
