# Eval

Held-out evaluation: 10 tasks in fresh domains never seen in `data/trajectories/`'s training
batches (sphere/cylinder volume, warehouse shipping, string case conversion, bracket nesting
depth, leaderboard, unit conversion, task priority queue, hex color normalization, matrix max,
decreasing streaks) — one task per file, matching the methodology that produced the cleanest
signal during trajectory collection (see `data/trajectories/batch-2026-08-08-c/README.md`).

## Results so far

| Model | Pass rate |
|---|---|
| Base (Qwen2.5-Coder-7B-Instruct) | **8/10 (80%)** — see `base-model-2026-08-08/` |
| Base + LoRA adapter (this project's SFT) | **Blocked** — see below |

Base model result is consistent with the 55-85% band established during trajectory collection
(batches f/g), which is a useful sanity check: this held-out set isn't systematically easier or
harder than the training-time task distribution.

## Why the LoRA-adapted run is blocked

Serving base model + adapter together needs vLLM's `--enable-lora` flag, which means installing
vLLM into the same pod that trained the adapter (to avoid transferring 154MB of weights out and
back in). That install hit three real environment conflicts in sequence:

1. `flash_attn`'s compiled `.so` broke (ABI mismatch) after installing vllm bumped torch —
   `VLLM_ATTENTION_BACKEND=XFORMERS` did **not** fix this; vLLM's RoPE code imports flash_attn
   unconditionally regardless of the selected backend.
2. Uninstalling the broken `flash_attn` fixed that, but surfaced a second conflict: two
   incompatible `libcudnn` versions installed by axolotl and vllm's separate dependency trees.
3. Isolating vLLM into its own venv (`python3 -m venv`) fixed both of the above at once. But the
   *next* attempt — identical commands, hours later — failed differently: `pip install -U axolotl`
   pulled a newer axolotl release whose torch dependency required a newer CUDA driver than two
   different rented hosts had, even though the exact same command had trained successfully earlier
   the same day. Confirmed on two separate physical machines, ruling out one bad host.

Update: pinning `axolotl==0.18.0` (the version from the successful runs' install log) was tried
and did **not** fix it — the same error recurred on yet another host with that exact pinned
version. Six total attempts across six different rented hosts: 2 succeeded (same host, reused via
image caching), 4 failed the same way on 4 different hosts. See `training/README.md` for the full
detail — this looks like real driver-version heterogeneity across the GPU fleet rather than a
software version issue this repo can fix from inside the container.

## Next step

Re-run the LoRA-adapted eval once a reliable way to land on (or request) a host with an adequate
driver is found — or investigate whether axolotl exposes a way to disable the specific fused LoRA
kernel optimization that's failing, since the main model load's own CUDA init succeeds fine on the
same failing hosts. Use the same 10 held-out tasks (reset the scratch files between runs — they
get modified in place) so the comparison stays apples-to-apples with the base model's 8/10.
