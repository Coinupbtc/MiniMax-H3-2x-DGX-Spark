#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env

HEAD_HOST="${HEAD_HOST:-spark-head}"
WORKER_HOST="${WORKER_HOST:-spark-peer}"
HEAD_IP="${HEAD_IP:-}"
WORKER_IP="${WORKER_IP:-}"
IMAGE="${H3_2X_IMAGE:-minimax-h3-2x-dgx-spark:experimental}"
MODEL_DIR="${MINIMAX_H3_MODEL_DIR:-}"
HF_CACHE="${HF_CACHE_DIR:-}"
API_PORT="${H3_API_PORT:-8000}"
RAY_PORT="${H3_RAY_PORT:-6379}"
MASTER_PORT="${H3_MASTER_PORT:-29500}"
IFACE="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"
HCA="${NCCL_IB_HCA:-rocep1s0f1}"
HEAD_GID="${H3_HEAD_GID_INDEX:-${NCCL_IB_GID_INDEX:-3}}"
WORKER_GID="${H3_WORKER_GID_INDEX:-${NCCL_IB_GID_INDEX:-3}}"
PROJECT_DIR="$H3_PROJECT_ROOT"
OUTPUT_DIR="${H3_OUTPUT_DIR:-$PROJECT_DIR/output}"
ATTENTION_BACKEND="${H3_DIFFUSION_ATTENTION_BACKEND:-CUDNN_ATTN}"
EXECUTION_MODE="${H3_EXECUTION_MODE:-compile}"
CACHE_BACKEND="${H3_CACHE_BACKEND:-none}"
CACHE_CONFIG="${H3_CACHE_CONFIG:-}"
ENABLE_CACHE_DIT_SUMMARY="${H3_ENABLE_CACHE_DIT_SUMMARY:-false}"
# Turbo/Acc LoRA (vllm-omni 0.28+): set H3_LORA_PATH to a .safetensors adapter
# (e.g. ~/models/lora/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors).
# Requires an image with --lora-path diffusion support (vllm-omni >= Aug 2026 main).
H3_LORA_PATH="${H3_LORA_PATH:-}"
H3_LORA_BACKEND="${H3_LORA_BACKEND:-peft}"
# OVERRIDE vars are not in .env. Use hyphen (not :-) so an empty/none
# H3_CACHE_PROFILE_OVERRIDE can disable Cache-DiT even when .env says balanced.
CACHE_PROFILE="${H3_CACHE_PROFILE_OVERRIDE-${H3_CACHE_PROFILE:-}}"
if [[ -n "${H3_EXECUTION_MODE_OVERRIDE:-}" ]]; then
  EXECUTION_MODE="$H3_EXECUTION_MODE_OVERRIDE"
fi
if [[ -n "${H3_DIFFUSION_ATTENTION_BACKEND_OVERRIDE:-}" ]]; then
  ATTENTION_BACKEND="$H3_DIFFUSION_ATTENTION_BACKEND_OVERRIDE"
fi
# Server-side ceiling on a single synchronous /v1/videos/sync generation.
# Was hardcoded 1800 s. A 1280×720 15 s request needs ~1800 s at 50 steps, so it
# landed exactly on the limit and returned HTTP 504 — that 504 was this timeout,
# NOT an OOM (2026-08-04). Raised so long clips are a wait, not a failure.
VIDEO_SYNC_TIMEOUT="${H3_VIDEO_SYNC_TIMEOUT:-5400}"
# More threads load the 89 GiB rank-0 weights faster but deepen the load-time
# memory dip (measured: MemAvailable bottoms ~10 GiB at 2 threads). Raise only
# with headroom to spare.
WEIGHT_LOAD_THREADS="${H3_WEIGHT_LOAD_THREADS:-2}"

if [[ "$CACHE_PROFILE" = balanced ]]; then
  CACHE_BACKEND=cache_dit
  ENABLE_CACHE_DIT_SUMMARY=true
  CACHE_CONFIG='{"Fn_compute_blocks":1,"Bn_compute_blocks":0,"max_warmup_steps":4,"max_cached_steps":-1,"residual_diff_threshold":0.15,"max_continuous_cached_steps":1,"enable_taylorseer":false}'
elif [[ "$CACHE_PROFILE" = none || -z "$CACHE_PROFILE" ]]; then
  CACHE_BACKEND=none
  ENABLE_CACHE_DIT_SUMMARY=false
  CACHE_CONFIG=
fi

for pair in \
  "HEAD_HOST:$HEAD_HOST" \
  "WORKER_HOST:$WORKER_HOST" \
  "H3_2X_IMAGE:$IMAGE" \
  "MINIMAX_H3_MODEL_DIR:$MODEL_DIR" \
  "HF_CACHE_DIR:$HF_CACHE" \
  "NCCL_SOCKET_IFNAME:$IFACE" \
  "NCCL_IB_HCA:$HCA" \
  "H3_HEAD_GID_INDEX:$HEAD_GID" \
  "H3_WORKER_GID_INDEX:$WORKER_GID" \
  "H3_OUTPUT_DIR:$OUTPUT_DIR"; do
  h3_require_safe_value "${pair%%:*}" "${pair#*:}"
done
h3_require_nonnegative_integer H3_HEAD_GID_INDEX "$HEAD_GID"
h3_require_nonnegative_integer H3_WORKER_GID_INDEX "$WORKER_GID"
case "$ATTENTION_BACKEND" in
  TORCH_SDPA|CUDNN_ATTN|FLASH_ATTN|FLASHINFER_ATTN) ;;
  *) h3_fail "H3_DIFFUSION_ATTENTION_BACKEND must be TORCH_SDPA, CUDNN_ATTN, FLASH_ATTN, or FLASHINFER_ATTN" ;;
esac
case "$EXECUTION_MODE" in
  eager|compile) ;;
  *) h3_fail "H3_EXECUTION_MODE must be eager or compile" ;;
esac
case "$CACHE_BACKEND" in
  none|cache_dit) ;;
  *) h3_fail "H3_CACHE_BACKEND must be none or cache_dit" ;;
esac
case "$CACHE_PROFILE" in
  ""|balanced|none) ;;
  *) h3_fail "H3_CACHE_PROFILE must be empty, none, or balanced" ;;
esac
case "$ENABLE_CACHE_DIT_SUMMARY" in
  true|false) ;;
  *) h3_fail "H3_ENABLE_CACHE_DIT_SUMMARY must be true or false" ;;
esac
if [[ -n "$CACHE_CONFIG" ]]; then
  h3_require_command python3
  python3 -c 'import json,sys; value=json.loads(sys.argv[1]); assert isinstance(value, dict)' "$CACHE_CONFIG" ||
    h3_fail "H3_CACHE_CONFIG must be a JSON object"
fi
if [[ "$CACHE_BACKEND" = none && ( -n "$CACHE_CONFIG" || "$ENABLE_CACHE_DIT_SUMMARY" = true ) ]]; then
  h3_fail "cache configuration requires H3_CACHE_BACKEND=cache_dit"
fi
h3_require_ipv4 HEAD_IP "$HEAD_IP"
h3_require_ipv4 WORKER_IP "$WORKER_IP"
h3_require_port H3_API_PORT "$API_PORT"
h3_require_port H3_RAY_PORT "$RAY_PORT"
h3_require_port H3_MASTER_PORT "$MASTER_PORT"

# NCCL transport: default IB/RoCE (RDMA). Set H3_NCCL_TRANSPORT=socket to force
# TCP over the fabric instead — workaround if container RDMA injection is broken
# (nvidia-container-toolkit 1.20.0 regression, H3 NCCL error 6 on first collective).
NCCL_NET_VAL="IB"
NCCL_IB_DISABLE_VAL="0"
_nccl_transport="${H3_NCCL_TRANSPORT_OVERRIDE:-${H3_NCCL_TRANSPORT:-}}"
if [[ "$_nccl_transport" = "socket" ]]; then
  NCCL_NET_VAL="Socket"
  NCCL_IB_DISABLE_VAL="1"
fi
COMMON_ENV=(
  -e NCCL_NET="$NCCL_NET_VAL"
  -e NCCL_IB_DISABLE="$NCCL_IB_DISABLE_VAL"
  -e NCCL_IB_HCA="$HCA"
  -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e NCCL_IB_ROCE_VERSION_NUM=2
  -e NCCL_IB_MTU=1024
  -e NCCL_IB_PCI_RELAXED_ORDERING=0
  -e NCCL_GIN_ENABLE=0
  -e NCCL_SOCKET_IFNAME="$IFACE"
  -e GLOO_SOCKET_IFNAME="$IFACE"
  -e NCCL_CROSS_NIC=0
  -e NCCL_CUMEM_ENABLE=0
  -e NCCL_NVLS_ENABLE=0
  -e NCCL_IGNORE_CPU_AFFINITY=1
  -e NCCL_DEBUG=WARN
  -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT="$VIDEO_SYNC_TIMEOUT"
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e RAY_memory_monitor_refresh_ms=0
)

docker_args_text() {
  printf ' %q' "$@"
}

"$(dirname "$0")/preflight.sh"
"$(dirname "$0")/stop-two-sparks.sh"

# Spark2 netdev was 9000 / mlx active_mtu 4096 while head was 1500 / 1024.
# That mismatch produced IBV_WC_REM_INV_REQ_ERR on the first GPU collective.
# Align both sides to 1500 (ioctl from a NET_ADMIN host-net container; no sudo).
if [[ "$_nccl_transport" != "socket" ]]; then
  _mtu_py="$PROJECT_DIR/scripts/align-fabric-mtu.py"
  scp -q "$_mtu_py" "$HEAD_HOST:/tmp/align-fabric-mtu.py"
  scp -q "$_mtu_py" "$WORKER_HOST:/tmp/align-fabric-mtu.py"
  for _host in "$HEAD_HOST" "$WORKER_HOST"; do
    ssh "$_host" "docker run --rm --network host --cap-add NET_ADMIN -v /tmp/align-fabric-mtu.py:/tmp/align-fabric-mtu.py:ro --entrypoint python3 '$IMAGE' /tmp/align-fabric-mtu.py '$IFACE' 1500"
  done
fi

head_common="$(docker_args_text "${COMMON_ENV[@]}" -e NCCL_IB_GID_INDEX="$HEAD_GID")"
worker_common="$(docker_args_text "${COMMON_ENV[@]}" -e NCCL_IB_GID_INDEX="$WORKER_GID")"
execution_args=()
if [[ "$EXECUTION_MODE" = eager ]]; then
  execution_args+=(--enforce-eager)
fi
execution_args_text=""
if (( ${#execution_args[@]} )); then
  execution_args_text="$(docker_args_text "${execution_args[@]}")"
fi
cache_args=()
if [[ "$CACHE_BACKEND" != none ]]; then
  cache_args+=(--cache-backend "$CACHE_BACKEND")
fi
if [[ -n "$CACHE_CONFIG" ]]; then
  cache_args+=(--cache-config "$CACHE_CONFIG")
fi
if [[ "$ENABLE_CACHE_DIT_SUMMARY" = true ]]; then
  cache_args+=(--enable-cache-dit-summary)
fi
cache_args_text=""
if (( ${#cache_args[@]} )); then
  cache_args_text="$(docker_args_text "${cache_args[@]}")"
fi
lora_args=()
lora_mount_text=""
if [[ -n "$H3_LORA_PATH" ]]; then
  [[ -f "$H3_LORA_PATH" ]] || h3_fail "H3_LORA_PATH file not found: $H3_LORA_PATH"
  LORA_HOST_DIR="$(dirname "$H3_LORA_PATH")"
  LORA_BASENAME="$(basename "$H3_LORA_PATH")"
  lora_args+=(--task-type fl2va --lora-backend "$H3_LORA_BACKEND" --lora-path "/lora/$LORA_BASENAME")
  lora_mount_text="$(docker_args_text -v "$LORA_HOST_DIR":/lora:ro)"
fi
lora_args_text=""
if (( ${#lora_args[@]} )); then
  lora_args_text="$(docker_args_text "${lora_args[@]}")"
fi

# Validated values are intentionally expanded on the client for remote Docker.
# shellcheck disable=SC2029
ssh "$HEAD_HOST" "docker run -d --name minimax-h3-2x-ray-head --network host --ipc host --gpus all --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 $head_common -v minimax-h3-2x-ray-tmp:/tmp/ray -v '$MODEL_DIR':'$MODEL_DIR':ro -v '$HF_CACHE':/root/.cache/huggingface --entrypoint ray '$IMAGE' start --head --node-ip-address='$HEAD_IP' --port='$RAY_PORT' --dashboard-host='$HEAD_IP' --dashboard-port=8265 --num-cpus=8 --num-gpus=1 --object-store-memory=2000000000 --disable-usage-stats --block >/dev/null"

for _ in $(seq 1 30); do
  if ssh "$HEAD_HOST" "docker exec minimax-h3-2x-ray-head ray status" >/dev/null 2>&1; then
    ray_head_ready=1
    break
  fi
  sleep 2
done
test "${ray_head_ready:-0}" = 1 || {
  echo "Ray head did not become ready" >&2
  exit 1
}

# shellcheck disable=SC2029
ssh "$WORKER_HOST" "docker run -d --name minimax-h3-2x-ray-worker --network host --ipc host --gpus all --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 $worker_common -v minimax-h3-2x-ray-worker-tmp:/tmp/ray -v '$MODEL_DIR':'$MODEL_DIR':ro -v '$HF_CACHE':/root/.cache/huggingface --entrypoint ray '$IMAGE' start --address='$HEAD_IP:$RAY_PORT' --node-ip-address='$WORKER_IP' --num-cpus=8 --num-gpus=1 --object-store-memory=2000000000 --disable-usage-stats --block >/dev/null"

for _ in $(seq 1 60); do
  nodes="$(ssh "$HEAD_HOST" "docker exec minimax-h3-2x-ray-head ray status 2>/dev/null" || true)"
  if grep -q '2 node' <<<"$nodes" || [ "$(grep -c 'node_' <<<"$nodes")" -ge 2 ]; then
    ray_pair_ready=1
    break
  fi
  sleep 2
done
test "${ray_pair_ready:-0}" = 1 || {
  echo "Ray cluster did not reach two active nodes" >&2
  exit 1
}

# shellcheck disable=SC2029
ssh "$HEAD_HOST" "docker run -d --name minimax-h3-2x-api --network host --ipc host --pid=container:minimax-h3-2x-ray-head --gpus all --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 $head_common $lora_mount_text -e H3_HEAD_IP='$HEAD_IP' -e H3_WORKER_IP='$WORKER_IP' -e H3_HEAD_GID_INDEX='$HEAD_GID' -e H3_WORKER_GID_INDEX='$WORKER_GID' -e H3_RAY_ADDRESS='$HEAD_IP:$RAY_PORT' -e H3_MASTER_PORT='$MASTER_PORT' -e H3_WORKER_START_TIMEOUT=2400 -e H3_API_PORT='$API_PORT' -v minimax-h3-2x-ray-tmp:/tmp/ray -v '$MODEL_DIR':'$MODEL_DIR':ro -v '$HF_CACHE':/root/.cache/huggingface -v '$OUTPUT_DIR':/output --entrypoint vllm '$IMAGE' serve '$MODEL_DIR' --omni --trust-remote-code --host '$HEAD_IP' --port '$API_PORT' --num-gpus 2 --usp 2 --ring 1 --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling --num-weight-load-threads '$WEIGHT_LOAD_THREADS' $execution_args_text $cache_args_text $lora_args_text --diffusion-attention-backend '$ATTENTION_BACKEND' --diffusion-quantization-config '{\"method\":\"fp8\",\"activation_scheme\":\"dynamic\",\"ignored_layers\":[\"video_patch_proj\",\"audio_patch_proj\",\"time_embedder.proj_in\",\"time_embedder.proj_out\",\"final_layer.video_out\",\"final_layer.audio_out\"]}' --force-cutlass-fp8 --distributed-executor-backend ray --stage-init-timeout 1800 --init-timeout 2400 >/dev/null"

echo "two-Spark H3 launch started: attention=$ATTENTION_BACKEND execution=$EXECUTION_MODE cache=$CACHE_BACKEND; API will appear at http://$HEAD_IP:$API_PORT/v1 after both ranks load"
