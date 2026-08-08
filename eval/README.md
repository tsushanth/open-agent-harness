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

Update: pinning `axolotl==0.18.0` did **not** fix it — same error, yet another host. Root cause
found by grepping axolotl's source for the actual config flags (`lora_mlp_kernel`,
`lora_qkv_kernel`, `lora_o_kernel`, `lora_embedding_kernel` — axolotl auto-enables a fused CUDA
kernel optimization for QLoRA unless told not to; its own lazy re-init, separate from the main
model load, was what crashed). **Disabling these in `training/qwen2.5-coder-7b-lora.yaml` fixed
training** — confirmed on 2 more hosts after the fix, both trained successfully (vs. 4/4 failures
before it). This is real, verified progress; see `training/README.md`.

Serving the resulting adapter for a live eval comparison hit a *different* wall after that: every
attempt at running vLLM with `--enable-lora` alongside the adapter — installing vLLM fresh in the
training pod (multiple dependency conflicts), via a RunPod Network Volume shared between a
training pod and a second pod using the already-proven `vllm/vllm-openai` image (the image's fixed
entrypoint can't run arbitrary shell, and attaching a network volume to it caused an immediate,
silent crash-loop even with plain non-LoRA args — sanity-checked and confirmed), and finally
installing an older pinned vLLM (`0.7.3`) inside the axolotl image again — all failed differently.
The last attempt got furthest (model loading started) before dying silently with no traceback
(pod uptime resets each time — consistent with an OOM-kill, not confirmed).

**Total for this specific sub-problem: 10 pod attempts, 4 distinct root causes found and fixed
along the way** (flash_attn ABI break, libcudnn conflict, fused-kernel CUDA crash — this one fully
resolved — and now this serving-specific crash, unresolved). Stopped here rather than continuing
to debug blind with no logs on the failing image.

## Next step

The training pipeline fix is solid and should be trusted going forward. The live LoRA-vs-base eval
comparison needs a fundamentally different approach rather than more ad hoc pod debugging —
candidates: (a) build a custom Docker image with axolotl + a compatible vLLM pre-installed and
tested once, rather than fresh `pip install`s per pod; (b) get proper log access on whatever image
serves the LoRA adapter (the `vllm/vllm-openai` image's fixed entrypoint blocks this); (c) skip
vLLM's `--enable-lora` entirely and eval via plain `transformers` + `peft` generation instead,
which is slower per-request but far simpler to get working and debug. Use the same 10 held-out
tasks (reset the scratch files between runs) so any future comparison stays apples-to-apples with
the base model's 8/10.
