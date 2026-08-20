# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the native ``execute_object_group_transfer`` executor,
focused on the dual-stream transfer pipeline.

The reference behavior is the single-stream path (``copy_stream=0``), whose
kernel-level correctness is covered by ``test_mp_mem_kernels.py``. Every
dual-stream test asserts bit-exact equivalence against that reference:

* H2D (retrieve) and D2H (store) equivalence with a real copy stream and a
  depth-2 slot ring.
* Slot-hazard stress: the compute stream is stalled with an injected sleep so
  the copy stream runs as far ahead as the ring allows; the per-step
  ``wait_step`` edges must prevent any overwrite-before-read.
* Exception path: a plan that fails validation *after* staging copies were
  enqueued must still join the copy stream into the compute stream (the exit
  join), so a completion event recorded on the compute stream covers the
  in-flight copies.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type

pytest.importorskip(
    "lmcache.cuda_ops",
    reason="Requires CUDA extension lmcache.cuda_ops",
)

# First Party
import lmcache.cuda_ops as cuda_ops
import lmcache.lmcache_native as lmcache_native

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch_dev.is_available() and torch_device_type == "cuda"),
        reason="Requires CUDA backend",
    ),
]

FMT_NORMAL = lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
H2D = lmcache_native.TransferDirection.H2D
D2H = lmcache_native.TransferDirection.D2H

# Geometry: one kernel group, [2, NB, BS, NH, HS] per layer.
NL = 4
NB = 128
BS = 16
NH = 8
HS = 64
DTYPE = torch.bfloat16
TOKENS_PER_OBJECT = 64
BLOCKS_PER_OBJECT = TOKENS_PER_OBJECT // BS
MAX_BATCH = 4
DEPTH = 2
NUM_OBJECTS = 24  # 6 batch steps of 4 -> 3 full trips around a depth-2 ring
# Big enough that no split happens inside lmcache_memcpy_async (power of two).
HOST_ALIGNMENT = 1 << 30
# ~tens of ms of GPU busy-wait; large against per-op latency, small for CI.
SLEEP_CYCLES = 200_000_000

_DEVICE = torch.device("cuda")


class _Plan:
    """A complete transfer plan plus the buffers it references."""

    def __init__(self, seed: int) -> None:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        obj_shape = (2, NL, TOKENS_PER_OBJECT, NH * HS)
        # Host-side chunk objects (pinned so cudaMemcpyAsync is truly async).
        self.host_objects = [
            torch.rand(obj_shape, dtype=torch.float32, generator=gen)
            .to(DTYPE)
            .pin_memory()
            for _ in range(NUM_OBJECTS)
        ]
        # Staging slot ring: DEPTH sets of MAX_BATCH slots.
        self.slots = [
            torch.zeros(obj_shape, dtype=DTYPE, device=_DEVICE)
            for _ in range(MAX_BATCH * DEPTH)
        ]
        # Paged engine KV buffers, one per layer.
        self.vllm_tensors = [
            torch.zeros(2, NB, BS, NH, HS, dtype=DTYPE, device=_DEVICE)
            for _ in range(NL)
        ]
        self.paged_ptrs = torch.tensor(
            [t.data_ptr() for t in self.vllm_tensors],
            dtype=torch.int64,
            device=_DEVICE,
        )
        # A shuffled block-id layout so scattering is non-trivial.
        total_blocks = NUM_OBJECTS * BLOCKS_PER_OBJECT
        perm = torch.randperm(NB, generator=gen)[:total_blocks]
        self.block_ids = perm.to(dtype=torch.int64, device=_DEVICE)

        shape_desc = cuda_ops.PageBufferShapeDesc()
        shape_desc.kv_size = 2
        shape_desc.nl = NL
        shape_desc.nb = NB
        shape_desc.bs = BS
        shape_desc.nh = NH
        shape_desc.hs = HS
        shape_desc.element_size = self.vllm_tensors[0].element_size()
        self.spec = cuda_ops.KernelGroupSpec(
            paged_buffer_ptrs=self.paged_ptrs.data_ptr(),
            lmcache_objects_ptrs=[s.data_ptr() for s in self.slots],
            shape_desc=shape_desc,
            lmcache_chunk_size=TOKENS_PER_OBJECT,
            engine_kv_format=int(FMT_NORMAL),
            block_ids_base=self.block_ids.data_ptr(),
            block_ids_capacity=self.block_ids.numel(),
        )

    def build_steps(self, direction, rotate: bool) -> list:
        """Build the batch steps; ``rotate=False`` pins everything to slot
        set 0 with no waits (the legacy plan shape)."""
        is_h2d = direction == H2D
        steps = []
        for step_idx, start in enumerate(range(0, NUM_OBJECTS, MAX_BATCH)):
            batch = self.host_objects[start : start + MAX_BATCH]
            slot_base = (step_idx % DEPTH) * MAX_BATCH if rotate else 0
            wait_step = step_idx - DEPTH if rotate and step_idx >= DEPTH else -1
            staging = []
            for j, obj in enumerate(batch):
                slot = self.slots[slot_base + j]
                dest, src = (
                    (slot.data_ptr(), obj.data_ptr())
                    if is_h2d
                    else (obj.data_ptr(), slot.data_ptr())
                )
                staging.append(
                    cuda_ops.StagingCopy(dest, src, obj.nbytes, host_offset=0)
                )
            launch = cuda_ops.LaunchVar(
                group_idx=0,
                block_ids_offset=start * BLOCKS_PER_OBJECT,
                total_blocks=len(batch) * BLOCKS_PER_OBJECT,
                num_objects=len(batch),
                skip_prefix_n_blocks=0,
                slot_offset=slot_base,
            )
            steps.append(cuda_ops.BatchStep(staging, [launch], wait_step=wait_step))
        return steps

    def execute(self, direction, steps, copy_stream: int) -> None:
        cuda_ops.execute_object_group_transfer(
            int(direction),
            _DEVICE,
            HOST_ALIGNMENT,
            [self.spec],
            steps,
            copy_stream=copy_stream,
        )

    def reset_gpu(self) -> None:
        for t in self.vllm_tensors:
            t.zero_()
        for s in self.slots:
            s.zero_()

    def reset_host(self) -> None:
        for o in self.host_objects:
            o.zero_()

    def vllm_snapshot(self) -> list[torch.Tensor]:
        torch.cuda.synchronize()
        return [t.clone() for t in self.vllm_tensors]

    def host_snapshot(self) -> list[torch.Tensor]:
        torch.cuda.synchronize()
        return [o.clone() for o in self.host_objects]


def _stall_stream(stream: torch.cuda.Stream) -> None:
    """Enqueue a GPU busy-wait so later work on *stream* starts late."""
    with torch.cuda.stream(stream):
        torch.cuda._sleep(SLEEP_CYCLES)


def test_h2d_dual_stream_matches_single_stream() -> None:
    """Retrieve with a real copy stream and a rotating depth-2 ring must be
    bit-exact vs the legacy single-stream plan."""
    plan = _Plan(seed=1234)

    plan.execute(H2D, plan.build_steps(H2D, rotate=False), copy_stream=0)
    reference = plan.vllm_snapshot()

    plan.reset_gpu()
    copy_s = torch.cuda.Stream()
    plan.execute(
        H2D, plan.build_steps(H2D, rotate=True), copy_stream=copy_s.cuda_stream
    )
    result = plan.vllm_snapshot()

    for ref, res in zip(reference, result, strict=True):
        assert torch.equal(ref, res)


def test_h2d_slot_hazard_under_compute_stall() -> None:
    """With the compute stream stalled, the copy stream runs as far ahead as
    the ring allows; the wait_step edges must prevent refilling a slot set the
    stalled kernels have not read yet."""
    plan = _Plan(seed=99)

    plan.execute(H2D, plan.build_steps(H2D, rotate=False), copy_stream=0)
    reference = plan.vllm_snapshot()

    plan.reset_gpu()
    copy_s = torch.cuda.Stream()
    _stall_stream(torch.cuda.current_stream())  # kernels start late
    plan.execute(
        H2D, plan.build_steps(H2D, rotate=True), copy_stream=copy_s.cuda_stream
    )
    result = plan.vllm_snapshot()

    for ref, res in zip(reference, result, strict=True):
        assert torch.equal(ref, res)


def test_d2h_dual_stream_matches_single_stream() -> None:
    """Store direction (kernels gather, then copies drain) with the mirrored
    edges must be bit-exact vs the single-stream plan."""
    plan = _Plan(seed=7)
    # Populate the paged buffers via the trusted single-stream H2D path.
    plan.execute(H2D, plan.build_steps(H2D, rotate=False), copy_stream=0)
    torch.cuda.synchronize()

    plan.reset_host()
    for s in plan.slots:
        s.zero_()
    plan.execute(D2H, plan.build_steps(D2H, rotate=False), copy_stream=0)
    reference = plan.host_snapshot()

    plan.reset_host()
    for s in plan.slots:
        s.zero_()
    copy_s = torch.cuda.Stream()
    _stall_stream(torch.cuda.current_stream())  # gather kernels start late
    plan.execute(
        D2H, plan.build_steps(D2H, rotate=True), copy_stream=copy_s.cuda_stream
    )
    result = plan.host_snapshot()

    for ref, res in zip(reference, result, strict=True):
        assert torch.equal(ref, res)


def test_exception_path_still_joins_copy_stream() -> None:
    """A validation failure after staging was enqueued must still order the
    copy stream's tail before subsequent compute-stream work (the exit join),
    or the caller's completion event would not cover the in-flight copies."""
    plan = _Plan(seed=5)
    steps = plan.build_steps(H2D, rotate=True)
    # Corrupt the SECOND step so the first step's staging is already enqueued
    # on the copy stream when validation throws.
    bad = cuda_ops.LaunchVar(
        group_idx=0,
        block_ids_offset=plan.block_ids.numel(),  # out of capacity
        total_blocks=BLOCKS_PER_OBJECT,
        num_objects=1,
        skip_prefix_n_blocks=0,
        slot_offset=0,
    )
    steps[1] = cuda_ops.BatchStep([], [bad], wait_step=-1)

    copy_s = torch.cuda.Stream()
    # Stall the COPY stream so its work is still pending when the executor
    # throws; only the exit join can order it behind the compute stream.
    _stall_stream(copy_s)

    with pytest.raises(RuntimeError, match="block_ids"):
        plan.execute(H2D, steps, copy_stream=copy_s.cuda_stream)

    # The exit join makes the compute stream wait for the copy stream's tail:
    # once the compute stream drains, the copy stream must be idle too.
    torch.cuda.current_stream().synchronize()
    assert copy_s.query(), (
        "copy stream still busy after compute stream drained - the exit join "
        "did not run on the exception path"
    )


def test_zero_steps_is_a_noop() -> None:
    plan = _Plan(seed=3)
    copy_s = torch.cuda.Stream()
    plan.execute(H2D, [], copy_stream=copy_s.cuda_stream)
    torch.cuda.synchronize()
    for t in plan.vllm_tensors:
        assert t.abs().sum().item() == 0


def test_bad_wait_step_rejected() -> None:
    """A wait_step that does not reference an earlier step is rejected."""
    plan = _Plan(seed=11)
    bad_first = cuda_ops.BatchStep([], [], wait_step=0)  # self-reference
    copy_s = torch.cuda.Stream()
    with pytest.raises(RuntimeError, match="wait_step"):
        plan.execute(H2D, [bad_first], copy_stream=copy_s.cuda_stream)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
