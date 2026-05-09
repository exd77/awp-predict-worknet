#!/usr/bin/env bash
set -euo pipefail

# Local-only helper. Copy this file before editing:
#   cp run-predict-loop.example.sh run-predict-loop.local.sh
#
# Never commit real API keys, wallet tokens, OAuth files, or keystores.

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export AWP_WALLET_HOME="${AWP_WALLET_HOME:-$HOME/.hermes/awp-wallet}"

exec predict-agent loop --interval "${PREDICT_INTERVAL:-120}" --agent-id "${PREDICT_AGENT_ID:-predict-worker}" --notify
