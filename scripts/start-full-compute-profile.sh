#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Quality path: no Cache-DiT. Optional attention backend:
#   H3_DIFFUSION_ATTENTION_BACKEND=CUDNN_ATTN|FLASHINFER_ATTN|FLASH_ATTN
# .env sets eager + Cache-DiT. These OVERRIDE vars are applied after load.
export H3_CACHE_PROFILE_OVERRIDE=none
export H3_EXECUTION_MODE_OVERRIDE="${H3_EXECUTION_MODE_OVERRIDE:-compile}"
export H3_DIFFUSION_ATTENTION_BACKEND_OVERRIDE="${H3_DIFFUSION_ATTENTION_BACKEND_OVERRIDE:-CUDNN_ATTN}"
exec "$SCRIPT_DIR/start-two-sparks.sh"
