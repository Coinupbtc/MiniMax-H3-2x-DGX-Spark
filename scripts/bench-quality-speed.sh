#!/usr/bin/env bash
# Same-seed 20-step quality clip for before/after timing + optional SSIM.
# Usage: bench-quality-speed.sh <label>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
h3_load_env
h3_require_license_acknowledgement

LABEL="${1:?label}"
HEAD_IP="${HEAD_IP:-}"
API_PORT="${H3_API_PORT:-8800}"
RESULT_DIR="${H3_RESULT_DIR:-$H3_PROJECT_ROOT/results/quality-speed}"
OUT="$RESULT_DIR/${LABEL}.mp4"
META="$RESULT_DIR/${LABEL}.txt"
API="http://${HEAD_IP}:${API_PORT}/v1/videos/sync"

mkdir -p "$RESULT_DIR"
[[ ! -e "$OUT" ]] || rm -f "$OUT" "$META"

tmp="$(mktemp "${OUT}.partial.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
start_ns="$(date +%s%N)"
code="$(curl -sS --max-time 3600 -w '%{http_code}' -o "$tmp" -X POST "$API" \
  -F 'prompt=Macro soldering a PCB under warm bench light, soft room tone.' \
  -F 'width=768' \
  -F 'height=448' \
  -F 'fps=24' \
  -F 'num_inference_steps=20' \
  -F 'flow_shift=12' \
  -F 'seed=42' \
  -F 'extra_params={"task":"t2va","duration":2.0,"audio_flow_shift":3.0}')"
end_ns="$(date +%s%N)"
[[ "$code" = 200 ]] || { echo "HTTP $code"; file "$tmp"; head -c 400 "$tmp"; exit 1; }
ffmpeg -v error -i "$tmp" -f null -
mv "$tmp" "$OUT"
trap - EXIT
elapsed_ms="$(( (end_ns - start_ns) / 1000000 ))"
{
  echo "label=$LABEL"
  echo "client_elapsed_ms=$elapsed_ms"
  echo "client_elapsed_s=$(python3 -c "print(f'{$elapsed_ms/1000:.3f}')")"
  sha256sum "$OUT"
  ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames -show_entries format=duration,size -of json "$OUT"
} | tee "$META"
echo "OK ${elapsed_ms}ms -> $OUT"
