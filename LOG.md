## 2026-05-12 — imatrix calibration alignment

User reported that swival + `ds4-server` + the IQ2XXS-imatrix GGUF runs away
into either an unbounded thinking section or `<｜begin▁of▁sentence｜>` spam,
while the non-imatrix GGUF and `./ds4 -p hello` both behave. Investigation
showed three classes of mismatch between the imatrix calibration corpus
(`gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py`) and the real
chat-with-tools rendering in `ds4_server.c`:

1. The `## Tools` system block is missing the `$TOOL_NAME2` chain example, the
   "pass JSON literals and set `string=\"false\"`" sentence, and the two
   thinking-mode paragraphs.
2. The tool-schema list is wrapped in a `[ ... ]` array of full
   `{"type":"function","function":{...}}` envelopes, while the server emits
   the bare inner function objects, one per line, no array. The schema region
   is therefore calibrated on tokens the runtime never produces.
3. The `<｜DSML｜tool_calls>` example uses a single leading newline, while the
   server emits two. Historic assistant turns also use `</think>` alone, while
   the server wraps them in `<think></think>` whenever tools are attached
   (`tool_context`).

On top of that, only 8 out of 1952 prompts (0.4 % by count, 0.23 % by bytes)
exercise the tool-attached path. At IQ2_XXS the routed-MoE experts that fire
on the tool preamble end up calibrated on roughly nothing.

Plan: rewrite the dataset builder so the templates match `ds4_server.c` byte
for byte, add a real population of agent/tool transcripts (multi-step, error
results, escape-heavy outputs, varied schemas, swival-shaped system prompts,
both think/nothink), regenerate, recollect the imatrix against the Q4 chat-v2
GGUF, requantize IQ2_XXS, and verify the swival smoke test no longer hangs.

## What landed

- `build_ds4_imatrix_dataset.py` rewritten. TOOLS prompt now matches
  `append_tools_prompt_text` exactly; tool schemas serialize as bare,
  newline-joined function objects mirroring `openai_function_schema_from_tool`;
  historic-assistant `<think></think>` wrap matches the server's
  `tool_context` rule; the DSML block uses the two leading newlines.
  Verified via `tmp/compare-render.py` and `tmp/compare-multiturn.py` against
  a live `--trace` capture: byte-identical for single-turn and multi-turn
  tool-attached prompts.
- Agent records expanded from 8 to 1176 (45.2% of corpus bytes, up from
  0.23%). Twenty-three named scenarios, replayed across six toolsets and
  five system prompts (swival short/long, claude-code, opencode, italian),
  plus short single-call transcripts and random-walk glue.
- Dataset regenerated: 3130 prompts, ~2.62M tokens.
- Imatrix collected at `gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-aligned.dat`.
  Reference model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
  (the Q4 reference at 153 GB does not fit safely on a 128 GB machine and
  crashed the system on a prior attempt; the IQ2 reference is noisier per
  weight but ranks columns consistently enough). 1500000 prompt tokens
  across 1674 prompts, ~387M routed-expert observations, 450 MB output.

## What landed (continued)

- User downloaded HF safetensors to
  `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/<sha>`.
- Found and fixed a parser bug in `gguf-tools/deepseek4-quantize.c`:
  `parse_expert_tensor` used `sscanf("blk.%d.ffn_%15[^_]_exps.weight", ...)`
  and trusted the count, but `sscanf` returns the number of successful
  conversions even when a trailing literal mismatches, so the i32 routing
  cache `blk.N.ffn_gate_tid2eid.weight` was getting mis-classified as a
  `ffn_gate_exps.weight` expert and the quantizer aborted with
  "unsupported expert target type". Switched to `%n` + length check so the
  full input must match.
- Re-quantized IQ2_XXS via `gguf-tools/deepseek4-quantize` against the new
  imatrix `.dat` to
  `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-aligned.gguf`
  (81 GB, 1328 tensors, ~88 min wall time).
- Verified the new GGUF against the three previously-broken paths:
  * tools + thinking (the swival default): 35 tokens, finish=stop, sensible
    reply with reasoning_content. Was 2350+ runaway tokens before.
  * tools + nothink (`deepseek-chat` alias): 34 tokens, finish=stop.
    Was infinite `<｜begin▁of▁sentence｜>` repeats before.
  * Real swival smoke (`swival --profile ds4 hello`): 53 tokens, finish=stop,
    one turn, 16 s wall. Was a forever-loop ending in
    `error="shutdown requested"` before.

## 2026-05-18 — merge antirez/ds4 PR #15 into m5opts

User asked to carefully merge PR #15 (ivanfioravanti's Metal 4 M5 Tensor
prefill work) into the local `m5opts` branch on the swival fork. The PR
adds the cooperative-tensor (MPP) prefill path for Q8_0 dense matmuls,
attention-output low projection, and routed MoE projections, gated by an
extensive set of environment switches. It targets the same files as the
m5opts simdgroup_matrix and private-scratch optimisations.

The auto-merge produced one structural conflict in `ds4_metal.m` and a
latent ABI collision at Metal function-constant slot 702: m5opts uses 702
for `FC_mul_mm_m5_sgmatrix` (declared in `dense.metal`), and the PR uses
the same slot for `FC_mul_mm_id_mpp` (declared in `moe.metal`). Both
declarations end up in the same concatenated shader library, so they
cannot share an index. Resolution:

- Moved `FC_MUL_MM_M5_SGMATRIX` to slot 703 in the embedded source string
  and updated both `ds4_gpu_get_mul_mm_pipeline` and
  `ds4_gpu_get_mul_mm_id_pipeline` to write `m5_sgmatrix` at the new slot.
- Reworked `ds4_gpu_get_mul_mm_id_pipeline` to carry both flags — added
  `use_mpp` (PR) and kept `m5_sgmatrix` (m5opts), with the pipeline cache
  key reflecting both. Callers from the PR pass `use_mpp`; the m5 flag is
  resolved internally from `ds4_gpu_use_m5_simdgroup_matrix()`.
- Kept both helper sets verbatim: the m5opts `ds4_gpu_use_m5_simdgroup_matrix`
  plus the PR's `ds4_gpu_mpp_f16_default_target`, `ds4_gpu_env_bool`,
  `ds4_gpu_use_indexed_attention_rb4`, and the global MPP policy enum and
  router that follow.
- Removed the duplicate `ds4_gpu_use_m5_private_scratch` and
  `ds4_gpu_scratch_needs_cpu_access` definitions that the auto-merge had
  emitted both above and below — the PR placed identical implementations
  at line 238/248, so the m5opts copies at 616/627 were redundant.
- Kept the m5opts wording for the hazard-tracking comment in
  `ds4_gpu_ensure_scratch_buffer`.

The auto-merged `metal/moe.metal` now contains both gates in the routed
matmul kernel: the MPP branch fires when `FC_mul_mm_id_mpp` is true, and
the simdgroup fallback elides the no-op barriers when
`FC_mul_mm_m5_sgmatrix` is true. They are orthogonal.

Tests on M5 Max after the merge:

- `make ds4 ds4_test`: clean build, no warnings related to the merge.
- `./ds4_test --metal-kernels`: OK.
- `./ds4_test --long-context`: OK (30 474 prefill tokens).
- `./ds4_test --logprob-vectors`: seven assertion failures on
  `long_memory_archive`, identical to the failure observed on clean
  `m5opts` at `910ba81` in a fresh worktree. The pure `pr-15-metal4`
  branch passes the same test, which means the regression is pre-existing
  on `m5opts` (the test vectors were captured against PR-15's older glu
  kernel, while `m5opts` inherits main's `5bc1e6d` SwiGLU clamping fix
  that was not yet in the PR). Not introduced by the merge.

PR #15 cannot be merged to antirez/ds4 directly (jedisct1 has no push
there, and antirez has stated in the PR thread that he cannot maintain it
without M5 hardware). Landing on swival's `m5opts` keeps both speedup
families available behind their respective env switches.
