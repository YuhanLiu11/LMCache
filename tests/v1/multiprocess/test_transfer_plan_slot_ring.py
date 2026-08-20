# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the slot-ring rotation in ``_run_object_group_transfer_plan``.

The planner decides, per emitted batch step, which staging slot set the step
uses (``LaunchVar.slot_offset``) and which earlier step it must wait for
(``BatchStep.wait_step``), and which stream handle the native executor gets.
These tests pin that law with the native layer faked out:

* H2D with an enabled pipeline rotates slot sets and chains wait_step.
* Emitted steps (not iteration indices) drive the ring, so batches skipped by
  ``skip_first_n_tokens`` never desynchronize slot_offset from wait_step.
* D2H and a disabled pipeline (no copy stream) degenerate to the legacy plan
  shape: slot_offset 0, wait_step -1, copy_stream 0.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
import lmcache.lmcache_native as lmcache_native
import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as mod

H2D = lmcache_native.TransferDirection.H2D
D2H = lmcache_native.TransferDirection.D2H

CHUNK_TOKENS = 256
MAX_BATCH = 4
OBJ_SIZE = 1024


class _FakeLaunchVar:
    def __init__(
        self,
        group_idx,
        block_ids_offset,
        total_blocks,
        num_objects,
        skip_prefix_n_blocks,
        slot_offset=0,
    ):
        self.group_idx = group_idx
        self.block_ids_offset = block_ids_offset
        self.total_blocks = total_blocks
        self.num_objects = num_objects
        self.skip_prefix_n_blocks = skip_prefix_n_blocks
        self.slot_offset = slot_offset


class _FakeBatchStep:
    def __init__(self, staging, launches, wait_step=-1):
        self.staging = staging
        self.launches = launches
        self.wait_step = wait_step


class _FakeKernelGroupSpec:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _Recorder:
    """Captures the executor invocation the planner produces."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        direction,
        device,
        host_buffer_alignment,
        kernel_group_specs,
        batch_steps,
        copy_stream=0,
    ):
        self.calls.append(
            SimpleNamespace(
                direction=direction,
                device=device,
                kernel_group_specs=kernel_group_specs,
                batch_steps=batch_steps,
                copy_stream=copy_stream,
            )
        )


def _make_cache_context(pipeline_depth: int, copy_stream_handle: int) -> MagicMock:
    ctx = MagicMock()
    ctx.lmcache_tokens_per_chunk = CHUNK_TOKENS
    ctx.max_batch_size = MAX_BATCH
    ctx.transfer_pipeline_depth = pipeline_depth
    ctx.num_ring_slots = MAX_BATCH * pipeline_depth
    ctx.copy_stream_handle = copy_stream_handle
    ctx.device = "fake-device"
    ctx.calculate_num_blocks = MagicMock(side_effect=lambda tokens, kg: tokens // 16)
    ctx.get_slots_per_chunk_in_sw = MagicMock(return_value=CHUNK_TOKENS)
    ctx.get_shape_desc = MagicMock(return_value="shape-desc")
    ctx.get_engine_kv_format = MagicMock(return_value=0)

    kv_ptrs = MagicMock()
    kv_ptrs.data_ptr = MagicMock(return_value=1)
    ctx.get_kernel_group_kv_pointers = MagicMock(return_value=kv_ptrs)

    def temp_kernel_buffer(slot, kernel_group_idx):
        buf = MagicMock()
        buf.data_ptr = MagicMock(return_value=1000 + slot)
        return buf

    ctx.get_temp_kernel_group_buffer = MagicMock(side_effect=temp_kernel_buffer)
    ctx.get_temp_object_group_buffer = MagicMock(
        side_effect=lambda slot, og: f"objbuf-{slot}"
    )

    manager = ctx.kv_layer_groups_manager
    object_group = SimpleNamespace(kernel_group_indices=[0])
    manager.object_groups = [object_group]
    manager.get_subchunk_sw_size_tokens = MagicMock(return_value=CHUNK_TOKENS)
    attn_desc = MagicMock()
    attn_desc.is_full_attention = MagicMock(return_value=True)
    manager.get_attn_desc = MagicMock(return_value=attn_desc)
    return ctx


def _make_memory_objs(count: int) -> list:
    objs = []
    for i in range(count):
        obj = MagicMock()
        obj.raw_tensor = object()
        obj.get_size = MagicMock(return_value=OBJ_SIZE)
        obj.data_ptr = 5000 + i
        obj.meta = SimpleNamespace(address=0)
        objs.append(obj)
    return objs


@pytest.fixture
def fake_native(monkeypatch):
    """Fake device_ops + build_staging_copies; returns the call recorder and
    the per-step staged-buffer log."""
    recorder = _Recorder()
    staged_buffers = []

    def fake_build_staging_copies(memory_objs, gpu_buffers, is_h2d):
        staged_buffers.append(list(gpu_buffers))
        return [f"copy-{i}" for i in range(len(list(memory_objs)))]

    fake_ops = SimpleNamespace(
        LaunchVar=_FakeLaunchVar,
        BatchStep=_FakeBatchStep,
        KernelGroupSpec=_FakeKernelGroupSpec,
        execute_object_group_transfer=recorder,
    )
    monkeypatch.setattr(mod, "device_ops", fake_ops)
    monkeypatch.setattr(mod, "build_staging_copies", fake_build_staging_copies)
    return recorder, staged_buffers


def _block_ids_gpu() -> list:
    ids = MagicMock()
    ids.data_ptr = MagicMock(return_value=7)
    ids.numel = MagicMock(return_value=10_000)
    return [ids]


def _run(ctx, num_objects: int, direction, skip_first_n_tokens: int = 0) -> None:
    mod._run_object_group_transfer_plan(
        ctx,
        _block_ids_gpu(),
        _make_memory_objs(num_objects),
        object_group_id=0,
        batch_size=MAX_BATCH,
        skip_first_n_tokens=skip_first_n_tokens,
        direction=direction,
    )


def test_h2d_rotates_slot_sets_and_chains_wait_step(fake_native) -> None:
    recorder, staged_buffers = fake_native
    ctx = _make_cache_context(pipeline_depth=2, copy_stream_handle=12345)

    _run(ctx, num_objects=10, direction=H2D)  # steps of 4, 4, 2

    (call,) = recorder.calls
    assert call.copy_stream == 12345
    steps = call.batch_steps
    assert [s.launches[0].slot_offset for s in steps] == [0, 4, 0]
    assert [s.wait_step for s in steps] == [-1, -1, 0]
    # Staging targets follow the same rotation, sliced to the batch length.
    assert staged_buffers == [
        ["objbuf-0", "objbuf-1", "objbuf-2", "objbuf-3"],
        ["objbuf-4", "objbuf-5", "objbuf-6", "objbuf-7"],
        ["objbuf-0", "objbuf-1"],
    ]
    # The kernel-group spec carries the full slot ring.
    (spec,) = call.kernel_group_specs
    ring_ptrs = spec.args[1]
    assert ring_ptrs == [1000 + slot for slot in range(8)]


def test_skipped_batches_do_not_desync_the_ring(fake_native) -> None:
    """A batch fully consumed by skip_first_n_tokens emits no step; the ring
    indices must follow emitted steps, not iteration count."""
    recorder, staged_buffers = fake_native
    ctx = _make_cache_context(pipeline_depth=2, copy_stream_handle=12345)

    # Skip the entire first batch (4 chunks x 256 tokens).
    _run(
        ctx,
        num_objects=12,
        direction=H2D,
        skip_first_n_tokens=MAX_BATCH * CHUNK_TOKENS,
    )

    (call,) = recorder.calls
    steps = call.batch_steps
    assert len(steps) == 2
    assert [s.launches[0].slot_offset for s in steps] == [0, 4]
    assert [s.wait_step for s in steps] == [-1, -1]
    assert staged_buffers[0][0] == "objbuf-0"


def test_d2h_stays_single_stream_without_rotation(fake_native) -> None:
    recorder, _ = fake_native
    ctx = _make_cache_context(pipeline_depth=2, copy_stream_handle=12345)

    _run(ctx, num_objects=8, direction=D2H)

    (call,) = recorder.calls
    assert call.copy_stream == 0
    assert [s.launches[0].slot_offset for s in call.batch_steps] == [0, 0]
    assert [s.wait_step for s in call.batch_steps] == [-1, -1]


def test_disabled_pipeline_produces_legacy_plan(fake_native) -> None:
    recorder, staged_buffers = fake_native
    ctx = _make_cache_context(pipeline_depth=1, copy_stream_handle=0)

    _run(ctx, num_objects=10, direction=H2D)

    (call,) = recorder.calls
    assert call.copy_stream == 0
    assert [s.launches[0].slot_offset for s in call.batch_steps] == [0, 0, 0]
    assert [s.wait_step for s in call.batch_steps] == [-1, -1, -1]
    assert all(bufs[0] == "objbuf-0" for bufs in staged_buffers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
