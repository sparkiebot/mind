#!/usr/bin/env bash
# Run the local llama.cpp native-audio runtime and Sparkie Mind as one development stack.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

llama_binary="${MIND_LLAMA_BINARY:-llama-server}"
llama_host="${MIND_LLAMA_HOST:-127.0.0.1}"
llama_port="${MIND_LLAMA_PORT:-8080}"
llama_model="${MIND_LLAMA_MODEL_PATH:-models/llama.cpp/gemma4-e2b/google-gemma-4-E2B-it-Q4_K_M.gguf}"
llama_mmproj="${MIND_LLAMA_MMPROJ_PATH:-models/llama.cpp/gemma4-e2b/mmproj-BF16.gguf}"
llama_gpu_layers="${MIND_LLAMA_GPU_LAYERS:-99}"
llama_context_size="${MIND_LLAMA_CONTEXT_SIZE:-4096}"
python_binary="${MIND_PYTHON_BINARY:-.venv/bin/python}"
log_dir="${MIND_STACK_LOG_DIR:-logs}"
llama_log="$log_dir/llama-server.log"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Required command not found: $1" >&2
        exit 1
    fi
}

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

cleanup() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ -n "${mind_pid:-}" ]] && kill -0 "$mind_pid" >/dev/null 2>&1; then
        kill "$mind_pid" >/dev/null 2>&1 || true
        wait "$mind_pid" 2>/dev/null || true
    fi
    if [[ -n "${llama_pid:-}" ]] && kill -0 "$llama_pid" >/dev/null 2>&1; then
        kill "$llama_pid" >/dev/null 2>&1 || true
        wait "$llama_pid" 2>/dev/null || true
    fi
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

require_command "$llama_binary"
require_command curl
require_file "$llama_model"
require_file "$llama_mmproj"
require_file "$python_binary"
mkdir -p "$log_dir"

echo "Starting llama-server; logs: $llama_log"
"$llama_binary" \
    --model "$llama_model" \
    --mmproj "$llama_mmproj" \
    --jinja \
    --reasoning off \
    --gpu-layers "$llama_gpu_layers" \
    --ctx-size "$llama_context_size" \
    --no-warmup \
    --host "$llama_host" \
    --port "$llama_port" >"$llama_log" 2>&1 &
llama_pid=$!

llama_url="http://${llama_host}:${llama_port}"
for _ in $(seq 1 60); do
    if curl --fail --silent --max-time 1 "$llama_url/health" >/dev/null; then
        break
    fi
    if ! kill -0 "$llama_pid" >/dev/null 2>&1; then
        echo "llama-server exited during startup. See $llama_log" >&2
        exit 1
    fi
    sleep 1
done

if ! curl --fail --silent --max-time 1 "$llama_url/health" >/dev/null; then
    echo "llama-server did not become ready within 60 seconds. See $llama_log" >&2
    exit 1
fi

export MIND_RUNTIME=llama-server
export MIND_LLAMA_SERVER_URL="$llama_url"
export MIND_LLAMA_SERVER_MODEL="${MIND_LLAMA_SERVER_MODEL:-gemma4-e2b}"

echo "Starting Sparkie Mind with llama-server at $llama_url"
echo "Press Ctrl+C to stop both services."
"$python_binary" -m app &
mind_pid=$!
wait "$mind_pid"
