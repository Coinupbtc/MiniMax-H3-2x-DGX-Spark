#!/usr/bin/env bash
# ffmpeg SSIM + PSNR of two mp4s (video stream). Usage: ssim-pair.sh a.mp4 b.mp4
set -euo pipefail
A="${1:?}"; B="${2:?}"
ffmpeg -hide_banner -i "$A" -i "$B" -lavfi "[0:v][1:v]ssim;[0:v][1:v]psnr" -f null - 2>&1 \
  | grep -E 'SSIM|PSNR' | tail -20
