# Quality vs speed (same clip)

Clip every time: seed 42, 768×448, 56 frames / 2 s, 20 steps, prompt
“Macro soldering a PCB under warm bench light, soft room tone.”

**Quality reference** = CUDNN attention, **eager**, **no Cache-DiT**.

SHA of the warm MP4:

`2d5e3d38e12f23b0cab480fcc28abdbbf4c7defbd36f90f41330ba3386888604`

Warmup (first request after load) SHA:

`81bfd70afb9b5cc4ac877b1cb5fe823f887927f16cf15162eba84759e9e78171`

Do not treat compile, Cache-DiT, Torch-SDPA, or sparse Sol-Attn as this picture
unless SHA matches or video SSIM vs this file is 1.0.

## Two Sparks (Ulysses SP=2) — 2026-09-04

| Profile | Warm client | vs Socket eager | vs quality SHA |
|---|---:|---:|---|
| Socket TCP, CUDNN eager, no cache | 90.5 s | 1.00× | reference (later matched IB) |
| **RoCE IB, CUDNN eager, no cache** | **55.5 s** (repeat 58.4 s) | **1.63×** | **SHA identical, SSIM 1.0** |
| CUDNN compile, no cache (Socket) | 84.8 s | 1.07× | SSIM **0.71** — different picture |
| CUDNN eager + Cache-DiT (Socket) | 57.8 s | 1.56× | SSIM **0.73** — different picture |
| FlashInfer compile, no cache | 93.0 s | 0.97× | SSIM **0.79** — different picture |

IB warmup on the same clip: 59.3 s. RoCE needed GID 3, matched MTU 1500,
`memlock=-1`, `NCCL_IB_MTU=1024`, `NCCL_DEBUG=WARN`. See [ROCE-GB10.md](ROCE-GB10.md).

## One Spark (SP=1) — 2026-09-05

Same checkpoint family (FP8 DiT), same clip. Companion launcher:
[Coinupbtc/MiniMax-H3-1x-DGX-Spark](https://github.com/Coinupbtc/MiniMax-H3-1x-DGX-Spark).

| Profile | Warm client | vs 2× IB quality SHA |
|---|---:|---|
| **CUDNN eager, no cache** (host `start-fp8.sh`) | **136.1 s** (warmup 139.6 s) | **SHA identical, SSIM 1.0** |
| Published `sm121-fp8` image (`TORCH_SDPA` hardcoded) | 162.4 s | SSIM **0.72** — different picture |

One-Spark CUDNN warmup SHA also matched the two-Spark warmup file (`81bfd70a…`).
Two CUDNN clips on one Spark were SSIM 1.0 with each other.

~2.45× slower than two-Spark IB (136 s vs 55.5 s) is Ulysses SP=2 going away,
not a quality trade.

### Joey 1× defaults vs this quality path

| Default | What it actually is | vs quality SHA |
|---|---|---|
| Compose `H3_EXECUTION_MODE` default | **compile** | SSIM **0.71** on the 2× Socket compile clip |
| Baked `minimax-h3-dgx-spark:sm121-fp8` entrypoint | **`TORCH_SDPA`**, ignores `H3_DIFFUSION_ATTENTION_BACKEND` | SSIM **0.72** on 1× |

The 1× quality launcher mounts host `start-fp8.sh` over that entrypoint and
sets CUDNN + eager + no Cache-DiT.

## Comfy Sol-Attn 1× (not vLLM-Omni) — 2026-09-04

INT8 DiT + NVFP4 text encoder. Not SHA-comparable to the Joey FP8 eager file.

| | Time | SSIM vs dense Comfy |
|---|---:|---:|
| Dense (no Sol, no FBC) | 93.81 s | — |
| Sol-Attn only (tau 1.3) | 80.10 s | **0.72** |
| Sol-Attn + FirstBlockCache | 63.72 s | **0.71** |
