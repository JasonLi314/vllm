# Issue #45099 — full investigation context

Working notes for vllm-project/vllm#45099. Written so a fresh session can verify
the root cause and the fix without re-deriving everything. Delete before merge
(along with PR_DESCRIPTION.md in this folder).

## The bug report

- Issue: https://github.com/vllm-project/vllm/issues/45099
- Reporter: baonudesifeizhai, 2026-06-09. Full crash log: https://paste.ubuntu.com/p/ZK55nz9d3r/
- Setup: DeepSeek-V4-Flash on 4× B200 (SM100), `--data-parallel-size 4
  --enable-expert-parallel --moe-backend deep_gemm_mega_moe --kv-cache-dtype fp8
  --max-num-batched-tokens 32768 --speculative-config '{"method":"mtp", ...}'`
- Symptom: server crashes during startup with `CUDA error: invalid argument`.
  Works fine without MTP. Crashes with both `num_speculative_tokens` 1 and 3.

## What the crash log shows (key evidence)

All four DP ranks crash with identical tracebacks:

```
gpu_worker.py:410 determine_available_memory
  -> gpu_model_runner.py:6228 profile_run -> _dummy_run
  -> gpu_model_runner.py:5939 drafter.dummy_run        # MTP draft, 32768 tokens
  -> llm_base_proposer.py:1538 self.model(**kwargs)
  -> models/deepseek_v4/nvidia/mtp.py:155
       hidden_states = self.h_proj(previous_hidden_states) + self.e_proj(...)
torch.AcceleratorError: CUDA error: invalid argument
```

Relevant log lines confirming the code path:
- `Selected DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod`
- `DeepGEMM E8M0 enabled on current platform.` (checkpoint has
  `quantization_config.scale_fmt=ue8m0`)
- The *target* model's dummy run at 32768 tokens completed before the drafter
  crashed — only the MTP draft path fails.

## Root cause (chain of facts, each verifiable in the repo)

1. **MTP draft is the only quantized linear consumer of 3D activations.**
   DeepSeek V4 uses mHC: `hc_mult` (= 4, see
   `tests/kernels/test_mhc_kernels.py` `@pytest.mark.parametrize("hc_mult", [4])`)
   parallel residual streams of shape `(T, hc_mult, H)`. In the target model,
   `mhc_pre_tilelang` mixes streams down to 2D `(T, H)` *before* every
   quantized GEMM (`vllm/models/deepseek_v4/nvidia/model.py:827,861`). The MTP
   layer instead applies `h_proj` (a `ReplicatedLinear` with fp8 quant) directly
   on the un-mixed 3D residual (`vllm/models/deepseek_v4/nvidia/mtp.py:142-155`).
   Effective GEMM row count: `mn = T × hc_mult`.

2. **profile_run drives the drafter at the full batch size.**
   `profile_run` → `_dummy_run(max_num_tokens=32768)` →
   `drafter.dummy_run(32768)` (`vllm/v1/worker/gpu_model_runner.py:6223,5934`).
   So `mn = 32768 × 4 = 131072` for `h_proj`'s input quant.

3. **The fp8 activation quant takes the UE8M0-packed CUDA kernel.**
   `QuantFP8.forward_cuda` (`vllm/model_executor/layers/quantization/input_quant_fp8.py:93-103`)
   routes to `per_token_group_quant_fp8_packed_for_deepgemm`
   (`vllm/model_executor/layers/quantization/utils/fp8_utils.py:639`) when
   group quant + UE8M0 + DeepGEMM. That wrapper computes
   `mn = x.numel() // hidden_dim` (3D input flattens to T×hc_mult) and calls
   `torch.ops._C.per_token_group_fp8_quant_packed`.

4. **The kernel mapped rows to grid.y; CUDA caps grid.y at 65535.**
   `per_token_group_quant_8bit_packed` in
   `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`
   launched `dim3 grid(blocks_x=padded_groups_per_row/kx, blocks_y=ceil(tma_aligned_mn/ry))`
   with `ry = 16/kx ∈ {1,2,4}` (`GetGroupsPerBlockX` returns 16/8/4).
   CUDA limits: gridDim.x ≤ 2^31−1, gridDim.y ≤ 65535, gridDim.z ≤ 65535.
   With `mn = 131072`: ry=1 → 131072 blocks, ry=2 → 65536 blocks — both over
   65535 → `cudaErrorInvalidValue` ("invalid argument") at launch.
   (ry=4 → 32768 would pass; whether a given model crashes depends on its
   hidden size via `kx`, plus `max_num_batched_tokens × hc_mult`.)

5. **The existing guard was wrong, so the bad launch wasn't caught.**
   The old check compared both grid dims against `INT32_MAX`, with a comment
   claiming "CUDA caps grid.x and grid.y at 2^31 - 1". That's only true for
   grid.x.

This explains every reported symptom:
- No MTP → all quantized GEMMs see post-mix 2D inputs, `mn ≤ 32768` → fine.
- `num_speculative_tokens` 1 vs 3 → both crash; profile size is unchanged.
- Crashes at startup, not during serving → runtime decode batches are small;
  only the profile-run worst case reaches the limit.
- DP4/EP/MegaMoE are **incidental**, not causal. Reproduces in principle with
  TP1/DP1 whenever `max_num_batched_tokens × hc_mult / ry > 65535`.

## The fix

One file, `csrc/libtorch_stable/quantization/w8a8/fp8/per_token_group_quant.cu`,
function `per_token_group_quant_8bit_packed` and its register kernel:

- Kernel: `mn_idx` now derives from `blockIdx.x`, `sf_k_idx` from `blockIdx.y`
  (previously swapped). Nothing else in the kernel changes — the within-block
  thread→(row, k-group) decomposition (`sf_k_local`, `row_local`) is untouched,
  so per-block memory access patterns are identical.
- Host: launch `dim3 grid(row_blocks, sf_k_blocks)` (renamed from
  `blocks_y`/`blocks_x` for clarity); guard now checks `row_blocks ≤ INT32_MAX`
  and `sf_k_blocks ≤ 65535`.
- `sf_k_blocks = padded_groups_per_row / kx ≤ ceil(hidden/128/4)·4/4`, single
  digits for all real models — never near 65535.

Regression test: `test_per_token_group_quant_fp8_packed_large_mn` in
`tests/kernels/quantization/test_per_token_group_quant.py`. Uses
`(65537, 2048)`: hidden 2048 → 16 groups/row → kx=16, ry=1 → one grid row per
mn row, so 65537 rows minimally exceeds the old limit while keeping memory
modest (~270 MB input). Verifies quantized output and packed scales against the
Triton reference; the scale check is vectorized because the O(mn×groups) Python
loop in the neighboring tests would take minutes at this size. The `<< 24` in
the scale packing is safe in int32 because UE8M0 exponents for `randn*8`
inputs stay < 128.

## Verification steps for a fresh session

1. Read the traceback section of the paste log (mirrors above) and confirm the
   crash site is `mtp.py:155` under `drafter.dummy_run`.
2. Confirm the grid mapping: before the fix, line ~305 of
   `per_token_group_quant.cu` read `sf_k_idx = blockIdx.x * ...` and
   `mn_idx = blockIdx.y * ...`, and the launch was `grid(blocks_x, blocks_y)`
   with `blocks_y` being the row blocks. `git log -p` on the file shows it.
3. On a CUDA machine, on `main` without this patch:
   `pytest tests/kernels/quantization/test_per_token_group_quant.py -k large_mn`
   must fail with `CUDA error: invalid argument`; with the patch it passes.
4. Sanity: all other tests in that file must pass unchanged (the fix does not
   alter any math, only block-index decomposition).

## Adjacent / potentially controversial points

- **Two open PRs touch `mtp.py` `e_proj`/`h_proj` but are unrelated**: #44821
  and #44837 add `prefix=` plumbing (quant-config selection at load time);
  #44847/#43319 handle BF16 draft weights. None addresses the kernel grid
  overflow. Checked 2026-06-09 via
  `gh pr list --search "deepseek v4 mtp"` and `--search "45099 in:body"`.
- **Alternative fix considered**: a grid-stride loop over rows inside the
  kernel. Rejected as more invasive for no benefit; grid.x's 2^31−1 cap is
  unreachable (would require ~2^31 tokens).
- **Another alternative**: cap/loop on the *caller* side (chunk the quant into
  ≤65535-row slices). Rejected: every caller of the op would need the
  workaround, and the kernel limit is an implementation detail.
- **Possible perf concern**: after the swap, adjacent blocks along grid.x read
  different rows at the same K offset instead of adjacent K-groups of the same
  row. Within-block coalescing is unchanged; L2 locality across blocks could
  differ marginally. If a reviewer asks, benchmark
  `per_token_group_fp8_quant_packed` before/after on representative shapes
  (e.g. (8192, 7168)); no measurable difference is expected since the kernel is
  HBM-bandwidth-bound (per the comment in the kernel itself).
- **The non-packed kernel** (`per_token_group_quant_8bit_kernel`, same file)
  uses a 1D grid on grid.x only — not affected.
- **The int8 packed dispatch** shares the fixed kernel
  (`LAUNCH_REG_KERNEL(scalar_t, int8_t)`), so it's fixed too; only fp8 had a
  known production caller hitting large mn.
- **Workaround for users on unpatched builds**: reduce
  `--max-num-batched-tokens` so `tokens × hc_mult / ry ≤ 65535` (for
  DeepSeek-V4-Flash: ≤ 16384).
- **Untested locally**: this machine is macOS (no CUDA); the kernel change
  compiles only in a CUDA build and the tests need a GPU. The submitting human
  must build (`uv pip install -e .` — C++ changes need full build, not
  `VLLM_USE_PRECOMPILED=1`) and run the test file before opening the PR.
