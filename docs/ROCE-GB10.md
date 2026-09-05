# GB10 CX7 RoCE notes (this fork)

The original two-Spark recipe is [Joey Rodriguez / joeynyc](https://github.com/joeynyc/MiniMax-H3-2x-DGX-Spark).
These notes are for a later NVIDIA DGX Spark GB10 pair whose CX7 link was up
but NCCL was not actually using RoCE for Ulysses ALLTOALL.

## Symptoms

- Ray reports 2/2 GPU, but generation time matches a single Spark.
- NCCL logs `Using network Socket` / `via NET/Socket/0`.
- Or IB init succeeds, then the first collective dies with
  `IBV_WC_REM_INV_REQ_ERR(9)` and a garbage `reqSize`.
- Or denoise completes and the Ray actor is killed (`all references to the actor were removed`).

## Fixes applied in `scripts/start-two-sparks.sh`

| Knob | Why |
|---|---|
| IPv4-mapped RoCEv2 GID (often index **3**) | Index 1 is `fe80` link-local. `NCCL_IB_ADDR_FAMILY=AF_INET` plus GID 1 is an invalid pairing. |
| Match both netdev MTUs (1500 here) | 9000 vs 1500 → mlx `active_mtu` 4096 vs 1024. `scripts/align-fabric-mtu.py` ioctls from a `NET_ADMIN` host-net container (no host sudo). |
| `--ulimit memlock=-1` | Docker's 8 MiB lock is too small for GPU IB MRs. |
| `NCCL_IB_MTU=1024` | Cap QP MTU to the smaller mlx `active_mtu`. |
| `NCCL_GIN_ENABLE=0` | Avoid NCCL 2.28 `GIN_IB_GDAKI` on this path. |
| `NCCL_DEBUG=WARN` | `INFO` floods Ray's worker log pipe and the actor looks dead. |

`H3_NCCL_TRANSPORT=socket` remains the TCP fallback.

## Proof on this pair (2026-09-04)

- Host `ib_write_bw` GID 3 / 1K MTU: **109 Gb/s**.
- Container GPU `all_to_all` 256 MiB: **~140–200 Gb/s**.
- CUDNN eager, no Cache-DiT, seed 42, 768×448, 20 steps: **55.5 s** IB vs **90.5 s** Socket, SHA-identical MP4.

Joey's 2026-08-04 public-release warm compile (~46.6 s full-compute) is the
original acceptance on working RoCE. Do not treat Cache-DiT, sparse Sol-Attn,
or a compile graph that fails SSIM against eager as that quality path.

## One Spark, same SHA (2026-09-05)

The quality MP4 is not two-Spark-only. On one GB10, CUDNN eager / no Cache-DiT
produced **SHA `2d5e3d38…` identical** to the IB file (SSIM 1.0) in **136.1 s**.

The published `minimax-h3-dgx-spark:sm121-fp8` image hardcodes `TORCH_SDPA`
(SSIM 0.72 vs that file). Compose’s default **compile** path is SSIM 0.71.

Full table: [quality-speed.md](quality-speed.md). One-Spark launcher:
https://github.com/Coinupbtc/MiniMax-H3-1x-DGX-Spark
