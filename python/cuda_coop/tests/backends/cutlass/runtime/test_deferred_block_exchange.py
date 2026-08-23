# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

from ..support.runtime import (
    Float32,
    Float64,
    coop,
    cute,
    cutlass,
    runtime,
    runtime_pytestmark,
    torch,
)
from ..support.source import SOURCE_ROOT

provider_support = pytest.importorskip("cuda.coop.cutlass._dsl._provider")
provider_bundle = pytest.importorskip("cuda.coop.cutlass._dsl._provider_bundle")

REPOSITORY_ROOT = SOURCE_ROOT.parents[1]
pytestmark = [*runtime_pytestmark, pytest.mark.link]


def _make_deferred_exchange_runner(storage):
    @cute.kernel
    def kernel(
        input_f32: cute.Tensor,
        output_f32: cute.Tensor,
        input_f64: cute.Tensor,
        output_f64: cute.Tensor,
    ):
        block = coop.this_block()
        items_f32 = coop.ThreadData(4)
        coop._block.load(input_f32, items_f32)
        striped_f32 = coop._block.exchange_blocked_to_striped(
            items_f32,
            temp_storage=storage,
        )
        coop._block.store(output_f32, striped_f32, algorithm="striped")

        block.sync()

        items_f64 = coop.ThreadData(4)
        coop._block.load(input_f64, items_f64)
        striped_f64 = coop._block.exchange_blocked_to_striped(
            items_f64,
            temp_storage=storage,
        )
        coop._block.store(output_f64, striped_f64, algorithm="striped")

    @cute.jit
    def run(
        input_f32: cute.Tensor,
        output_f32: cute.Tensor,
        input_f64: cute.Tensor,
        output_f64: cute.Tensor,
    ):
        kernel(input_f32, output_f32, input_f64, output_f64).launch(
            grid=(1, 1, 1),
            block=(64, 1, 1),
        )

    return run


def test_failed_trace_session_is_discarded_before_next_compile(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        provider_bundle.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    monkeypatch.setenv(
        provider_bundle.CCCL_ROOT_ENV,
        str(REPOSITORY_ROOT),
    )
    provider_bundle.reset_compile_state()
    cutlass.cuda.initialize_cuda_context()

    storage = coop._block.TempStorage()
    run = _make_deferred_exchange_runner(storage)
    total_items = 64 * 4
    fake_f32 = runtime.make_fake_compact_tensor(Float32, (total_items,))
    fake_f64 = runtime.make_fake_compact_tensor(Float64, (total_items,))
    real_store = coop._block.store
    store_calls = 0

    def fail_first_post_exchange_store(*args, **kwargs):
        nonlocal store_calls
        store_calls += 1
        if store_calls == 1:
            raise RuntimeError("forced post-exchange trace failure")
        return real_store(*args, **kwargs)

    monkeypatch.setattr(
        coop._block,
        "store",
        fail_first_post_exchange_store,
    )
    with pytest.raises(RuntimeError, match="forced post-exchange trace failure"):
        cute.compile(run, fake_f32, fake_f32, fake_f64, fake_f64)

    # The compiler owns per-compile option identities; whichever way it keys
    # them, the failed trace's session must not feed the next compile. The
    # retry must trace from scratch and compile exactly one fresh bundle.
    monkeypatch.setattr(coop._block, "store", real_store)
    compiled = cute.compile(run, fake_f32, fake_f32, fake_f64, fake_f64)

    assert compiled is not None
    assert provider_bundle.get_nvrtc_compile_program_counter() == 1
    assert provider_support.lookup_bundle_session(cute.compile._compile_options) is None


def test_deferred_block_exchange_f32_f64_runtime_and_provider_layout(
    monkeypatch,
    request,
    tmp_path,
):
    monkeypatch.setenv(
        provider_bundle.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    monkeypatch.setenv(
        provider_bundle.CCCL_ROOT_ENV,
        str(REPOSITORY_ROOT),
    )
    provider_bundle.reset_compile_state()
    request.addfinalizer(provider_bundle.reset_compile_state)
    cutlass.cuda.initialize_cuda_context()

    storage = coop._block.TempStorage()
    run = _make_deferred_exchange_runner(storage)
    total_items = 64 * 4
    fake_f32 = runtime.make_fake_compact_tensor(Float32, (total_items,))
    fake_f64 = runtime.make_fake_compact_tensor(Float64, (total_items,))
    captured_layouts = []

    compile_with_layouts = provider_bundle.compile_bundle_source_with_layouts

    def capture_layouts(*args, **kwargs):
        compilation = compile_with_layouts(*args, **kwargs)
        captured_layouts.append(compilation.layouts)
        return compilation

    monkeypatch.setattr(
        provider_bundle,
        "compile_bundle_source_with_layouts",
        capture_layouts,
    )

    compiled = cute.compile(
        run,
        fake_f32,
        fake_f32,
        fake_f64,
        fake_f64,
    )

    assert provider_bundle.get_nvrtc_compile_program_counter() == 1
    assert len(captured_layouts) == 1
    assert set(captured_layouts[0].values()) == {
        provider_bundle.StorageLayout(1024, 16),
        provider_bundle.StorageLayout(2048, 16),
    }

    input_f32_host = torch.arange(total_items, dtype=torch.float32) + 0.25
    input_f64_host = torch.arange(total_items, dtype=torch.float64) + 0.5
    input_f32 = input_f32_host.cuda()
    input_f64 = input_f64_host.cuda()
    output_f32 = torch.zeros_like(input_f32)
    output_f64 = torch.zeros_like(input_f64)

    compiled(
        runtime.from_dlpack(input_f32),
        runtime.from_dlpack(output_f32),
        runtime.from_dlpack(input_f64),
        runtime.from_dlpack(output_f64),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(output_f32.cpu(), input_f32_host, atol=0, rtol=0)
    torch.testing.assert_close(output_f64.cpu(), input_f64_host, atol=0, rtol=0)
