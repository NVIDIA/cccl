# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
import subprocess

import pytest

from ....support.toolchains.cutlass import find_cuda_tool
from ..support.runtime import (
    Float32,
    Float64,
    Int32,
    coop,
    cute,
    cutlass,
    runtime,
    runtime_pytestmark,
    torch,
)
from ..support.source import SOURCE_ROOT

cutlass_compiler = pytest.importorskip("cutlass.base_dsl.compiler")
provider_bundle = pytest.importorskip("cuda.coop.cutlass._dsl._provider_bundle")

DumpDir = cutlass_compiler.DumpDir
KeepCUBIN = cutlass_compiler.KeepCUBIN
REPOSITORY_ROOT = SOURCE_ROOT.parents[1]
pytestmark = [*runtime_pytestmark, pytest.mark.link]


@cute.kernel
def _deferred_root_scan_kernel(
    input_f32: cute.Tensor,
    output_f32: cute.Tensor,
    input_f64: cute.Tensor,
    output_f64: cute.Tensor,
):
    storage = coop.TempStorage()
    group = coop.this_block()

    items_f32 = coop.ThreadData(4)
    coop.load(group, input_f32, items_f32)
    prefix_f32 = coop.scan(group, items_f32, temp_storage=storage)
    storage.sync()
    coop.store(group, output_f32, prefix_f32)

    items_f64 = coop.ThreadData(4)
    coop.load(group, input_f64, items_f64)
    prefix_f64 = coop.scan(group, items_f64, temp_storage=storage)
    coop.store(group, output_f64, prefix_f64)


@cute.jit
def _run_deferred_root_scan(
    input_f32: cute.Tensor,
    output_f32: cute.Tensor,
    input_f64: cute.Tensor,
    output_f64: cute.Tensor,
):
    _deferred_root_scan_kernel(
        input_f32,
        output_f32,
        input_f64,
        output_f64,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@cute.kernel
def _deferred_scoped_scalar_scan_kernel(
    values_in: cute.Tensor,
    initial_in: cute.Tensor,
    prefix_out: cute.Tensor,
    aggregate_out: cute.Tensor,
):
    storage = coop.TempStorage()
    tidx, _, _ = cute.arch.thread_idx()
    aggregate = coop.ThreadData(1, dtype=Int32)
    prefix_out[tidx] = coop._block.exclusive_scan(
        values_in[tidx],
        scan_op="max",
        initial_value=initial_in[0],
        block_aggregate=aggregate,
        temp_storage=storage,
    )
    aggregate_out[tidx] = aggregate[0]


@cute.jit
def _run_deferred_scoped_scalar_scan(
    values_in: cute.Tensor,
    initial_in: cute.Tensor,
    prefix_out: cute.Tensor,
    aggregate_out: cute.Tensor,
):
    _deferred_scoped_scalar_scan_kernel(
        values_in,
        initial_in,
        prefix_out,
        aggregate_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


def test_deferred_scoped_scalar_scan_initial_and_aggregate_runtime(
    monkeypatch,
    request,
    tmp_path,
):
    monkeypatch.setenv(
        provider_bundle.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    monkeypatch.setenv(provider_bundle.CCCL_ROOT_ENV, str(REPOSITORY_ROOT))
    provider_bundle.reset_compile_state()
    request.addfinalizer(provider_bundle.reset_compile_state)
    cutlass.cuda.initialize_cuda_context()

    fake_values = runtime.make_fake_compact_tensor(Int32, (64,))
    fake_initial = runtime.make_fake_compact_tensor(Int32, (1,))
    compiled = cute.compile(
        _run_deferred_scoped_scalar_scan,
        fake_values,
        fake_initial,
        fake_values,
        fake_values,
    )

    assert provider_bundle.get_nvrtc_compile_program_counter() == 1

    values_host = torch.arange(1, 65, dtype=torch.int32)
    initial_host = torch.tensor([1000], dtype=torch.int32)
    values_in = values_host.cuda()
    initial_in = initial_host.cuda()
    prefix_out = torch.zeros_like(values_in)
    aggregate_out = torch.zeros_like(values_in)
    compiled(
        runtime.from_dlpack(values_in),
        runtime.from_dlpack(initial_in),
        runtime.from_dlpack(prefix_out),
        runtime.from_dlpack(aggregate_out),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        prefix_out.cpu(),
        torch.full_like(values_host, initial_host.item()),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        aggregate_out.cpu(),
        torch.full_like(values_host, values_host.max()),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(
    find_cuda_tool("nvdisasm") is None,
    reason="requires nvdisasm to inspect the final cubin",
)
def test_deferred_root_block_scan_f32_f64_runtime_and_link(
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
    dsl_dump_dir = tmp_path / "dsl"
    dsl_dump_dir.mkdir()
    provider_bundle.reset_compile_state()
    request.addfinalizer(provider_bundle.reset_compile_state)
    cutlass.cuda.initialize_cuda_context()

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

    compiled = cute.compile[(KeepCUBIN, DumpDir(str(dsl_dump_dir)))](
        _run_deferred_root_scan,
        fake_f32,
        fake_f32,
        fake_f64,
        fake_f64,
    )

    assert provider_bundle.get_nvrtc_compile_program_counter() == 1
    assert len(captured_layouts) == 1
    assert set(captured_layouts[0].values()) == {
        provider_bundle.StorageLayout(288, 16),
        provider_bundle.StorageLayout(544, 16),
    }

    cubin_paths = sorted(dsl_dump_dir.rglob("*.cubin"))
    assert cubin_paths
    nvdisasm = find_cuda_tool("nvdisasm")
    assert nvdisasm is not None
    sass = subprocess.run(
        [str(nvdisasm), str(cubin_paths[-1])],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "cuda_coop_cutlass_cub_scan" not in sass
    assert re.search(r"\b(?:CALL|LDL|STL)(?:\.[A-Z0-9_]+)*\b", sass) is None

    input_f32_host = torch.ones(total_items, dtype=torch.float32)
    input_f64_host = torch.full((total_items,), 2.0, dtype=torch.float64)
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

    expected_f32 = torch.arange(total_items, dtype=torch.float32)
    expected_f64 = torch.arange(total_items, dtype=torch.float64) * 2.0
    torch.testing.assert_close(output_f32.cpu(), expected_f32, atol=0, rtol=0)
    torch.testing.assert_close(output_f64.cpu(), expected_f64, atol=0, rtol=0)
