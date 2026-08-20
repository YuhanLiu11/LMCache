# Dual-stream KV transfer pipeline

Modules: `lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`
(planner), `csrc/cuda/mp_mem_kernels.{cuh,cu}` (executor),
`lmcache/v1/platform/cuda/cache_context.py` (staging slot ring + copy
stream).
Config: `MPServerConfig.transfer_pipeline_depth`
(`--transfer-pipeline-depth`, default 1 = disabled).

It answers: **why does a retrieve's copy engine sit idle, and how do we
keep it busy?** A retrieve is ~36 batch steps of "stage up to 4 chunks
host→device, scatter them into paged KV with
`multi_layer_block_transfer_kernel`". Profiling a co-located deployment
(vLLM workers sharing the GPUs, no MPS) showed each step's scatter kernel
stalling a median 199 µs / ceiling ≈ 2.3 ms waiting to get the SMs back —
the two processes time-slice the compute cores — while back-to-back
kernels show zero gap and copies start in ~3 µs. On the single in-order
context stream, every stalled kernel also dams the next step's copies:
~2.4 ms of idle DMA per stall, ~23 ms per retrieve, about 22% of the
retrieve span. The copy engine is exempt from SM arbitration, which is
the property this design exploits: put staging copies on a dedicated
**copy stream** and let them run during the stalls. Replay of the traced
retrieves projects the span dropping from ~106 ms to ~82 ms (−22%,
bounded at −33% by pure PCIe time).

## The plan shape

The planner (`_run_object_group_transfer_plan`) already resolves a whole
object group into `BatchStep`s executed natively in one GIL release. The
pipeline adds two fields, both defaulting to the legacy shape:

* `LaunchVar.slot_offset` — first staging slot the launch reads. Batch
  steps rotate through `pipeline_depth` sets of `max_batch_size` slots
  (the **slot ring**, allocated by `_TempGPUBuffer`), so step *i* uses
  slot set `i % depth` and its launches read
  `lmcache_objects_ptrs[slot_offset, slot_offset + num_objects)`.
* `BatchStep.wait_step` — the earlier step that last used this step's
  slot set (`i - depth`; −1 = none). **Emitted** steps drive the ring,
  not iteration indices, so batches skipped by `skip_first_n_tokens`
  never desynchronize `slot_offset` from `wait_step`.

Rotation and wait edges are emitted only when a copy stream is passed;
otherwise (depth 1, store direction, non-CUDA platforms) the plan
degenerates to slot 0 / no waits — byte-identical to the pre-pipeline
plan.

## The executor's event edges

`execute_object_group_transfer` gains `copy_stream` (raw `cudaStream_t`;
0 = legacy single-stream path, unchanged enqueue order — the A/B lever).
With a copy stream, staging runs there while kernels stay on the current
(compute) stream, fenced by four edges (H2D shown; D2H mirrors by
flipping data/slot):

| edge | when | orders |
| --- | --- | --- |
| entry | call start | first copies after everything already on compute (a previous plan may still read the slots) |
| data | per step | step *i*'s kernels after step *i*'s staging |
| slot | per step | step *i*'s staging after `wait_step`'s kernels (slot reuse) |
| exit join | call end | compute after the copy stream's tail |

The exit join is what preserves the external contract: **when the native
call returns, all of its work is ordered on `cache_context.stream`**, so
the completion event and the `finish_read_prefetched` /`finish_write`
callbacks recorded there cover every copy. It runs on *all* exit paths
(a `try/catch` around the loop): the validation `TORCH_CHECK`s sit
between a step's staging and its launches, and an exception there would
otherwise leave copies in flight that the caller's completion event does
not cover — the host chunk locks would be released while the DMA engine
is still reading the buffers.

Two deliberate conservatisms, accepted for v1:

* The entry edge is recorded after the producer-event wait already on the
  compute stream, so copies inherit that wait even though they only touch
  LMCache-owned staging memory. Legal head start forfeited; measure
  before optimizing.
* Consecutive object groups within one retrieve serialize at the group
  boundary (≤ one stall each). A follow-up that merges all groups into a
  single native call removes both.

## Sizing: why depth 2 with a doubled buffer

Replaying every traced retrieve through the variants: depth 2 with a
second slot-set allocation (+~64 MB/GPU for a 768-token chunk config)
recovers −22.5% with ~0.2 ms of residual guard stalls; depth 3 adds
nothing; halving the batch to 2 for a zero-memory ping-pong recovers only
−14.5%, because ~1 ms of copy per step cannot cover a 2.3 ms stall and
there are twice as many stall draws. `_TempGPUBuffer` therefore scales
its one flat allocation to `max_batch_size × pipeline_depth` slots; the
getter signatures are unchanged (only the valid index range grows) and
GDS registers the same single, larger buffer.

## Store (D2H)

The executor is direction-symmetric (kernels gather into slots, copies
drain them; edges flip), and store moves two thirds of the bytes — but
its stalls were never profiled, so the planner passes `copy_stream=0`
for D2H in v1. Store already uses `batch_size=1`, so the existing slots
form a free depth-`num_ring_slots` ring the day it is enabled.

## Rollout

`transfer_pipeline_depth=1` is a no-op merge (every new field defaults to
legacy behavior; `copy_stream=0` reproduces the old enqueue order byte
for byte). Enable 2 in a benchmark environment, re-trace with nsys, then
flip the default. Acceptance: copy-lane H2D→H2D gaps stay ~3 µs median,
copy→kernel stalls off the copy path, loaded-device retrieve wall −20%
or better (−22…−33% expected; the copy→copy gaps caused by host-side
pinning are out of scope). Orthogonal follow-ups that stack: MPS (shrinks
the stall itself), `cudaHostRegister` cost, `--max-workers` ≥ the TP
degree.

## Tests

* `tests/v1/test_execute_object_group_transfer.py` — bit-exact
  equivalence vs the single-stream path for H2D and D2H, a slot-hazard
  stress with the compute stream stalled by an injected sleep, the
  exception-path exit join, and `wait_step` validation.
* `tests/v1/multiprocess/test_transfer_plan_slot_ring.py` — the planner's
  rotation law (`slot_offset`/`wait_step` per emitted step), skip-batch
  desync protection, and the legacy plan shape for D2H / disabled
  pipeline.
* `tests/v1/platform/test_gpu_cache_context.py` — slot-ring allocation,
  disjointness, and the `copy_stream_handle` / `num_ring_slots` surface.
