# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
import subprocess

import pytest

from ....support.toolchains.cutlass import find_cuda_tool
from ..support.runtime import (
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
provider_support = pytest.importorskip("cuda.coop.cutlass._dsl._provider")
provider_bundle = pytest.importorskip("cuda.coop.cutlass._dsl._provider_bundle")

DumpDir = cutlass_compiler.DumpDir
KeepCUBIN = cutlass_compiler.KeepCUBIN
REPOSITORY_ROOT = SOURCE_ROOT.parents[1]
pytestmark = [*runtime_pytestmark, pytest.mark.link]


_ALL_TEMP_STORAGE_EXPECTED_LAYOUTS = (
    ("load", 1024, 16),
    ("adjacent_difference", 512, 4),
    ("discontinuity", 512, 4),
    ("scan", 288, 16),
    ("radix_sort", 2336, 16),
    ("merge_sort", 1028, 4),
    ("store", 1040, 16),
)


_ALL_TEMP_STORAGE_SHARED_OFFSETS = (0, 0, 0, 0, 0, 0, 0)


_ALL_TEMP_STORAGE_SHARED_SIZE = 2336


_ALL_TEMP_STORAGE_EXCLUSIVE_OFFSETS = (0, 1024, 1536, 2048, 2336, 4672, 5712)


_ALL_TEMP_STORAGE_EXCLUSIVE_SIZE = 6752


def _store_thread_data_blocked(destination, values):
    tidx, _, _ = cute.arch.thread_idx()
    base = tidx * values.items_per_thread
    for item_idx in range(values.items_per_thread):
        destination[base + item_idx] = values[item_idx]


@cute.kernel
def _deferred_all_temp_storage_kernel(
    values_in: cute.Tensor,
    loaded_out: cute.Tensor,
    adjacent_out: cute.Tensor,
    flags_out: cute.Tensor,
    scanned_out: cute.Tensor,
    radix_out: cute.Tensor,
    merge_out: cute.Tensor,
    values_out: cute.Tensor,
):
    """Manual-sync port of the OG seven-family TempStorage stress kernel."""

    storage = coop.TempStorage()
    group = coop.this_block()

    values = coop.ThreadData(4)
    coop.load(
        group,
        values_in,
        values,
        algorithm="transpose",
        temp_storage=storage,
    )
    _store_thread_data_blocked(loaded_out, values)
    storage.sync()

    values = coop.adjacent_difference(group, values, temp_storage=storage)
    _store_thread_data_blocked(adjacent_out, values)
    storage.sync()

    flags = coop.discontinuity(group, values, temp_storage=storage)
    _store_thread_data_blocked(flags_out, flags)
    storage.sync()

    values = coop.scan(group, values, temp_storage=storage)
    _store_thread_data_blocked(scanned_out, values)
    storage.sync()

    values = coop.radix_sort_keys(
        group,
        values,
        begin_bit=0,
        end_bit=8,
        temp_storage=storage,
    )
    _store_thread_data_blocked(radix_out, values)
    storage.sync()

    values = coop.merge_sort_keys(group, values, temp_storage=storage)
    _store_thread_data_blocked(merge_out, values)
    storage.sync()

    coop.store(
        group,
        values_out,
        values,
        algorithm="transpose",
        temp_storage=storage,
    )


@cute.jit
def _run_deferred_all_temp_storage(
    values_in: cute.Tensor,
    loaded_out: cute.Tensor,
    adjacent_out: cute.Tensor,
    flags_out: cute.Tensor,
    scanned_out: cute.Tensor,
    radix_out: cute.Tensor,
    merge_out: cute.Tensor,
    values_out: cute.Tensor,
):
    _deferred_all_temp_storage_kernel(
        values_in,
        loaded_out,
        adjacent_out,
        flags_out,
        scanned_out,
        radix_out,
        merge_out,
        values_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@cute.kernel
def _deferred_all_temp_storage_exclusive_kernel(
    values_in: cute.Tensor,
    loaded_out: cute.Tensor,
    adjacent_out: cute.Tensor,
    flags_out: cute.Tensor,
    scanned_out: cute.Tensor,
    radix_out: cute.Tensor,
    merge_out: cute.Tensor,
    values_out: cute.Tensor,
):
    """OG-style exclusive scratch uses one disjoint slice per call site."""

    storage = coop.TempStorage(sharing="exclusive")
    group = coop.this_block()

    values = coop.ThreadData(4)
    coop.load(
        group,
        values_in,
        values,
        algorithm="transpose",
        temp_storage=storage,
    )
    _store_thread_data_blocked(loaded_out, values)

    values = coop.adjacent_difference(group, values, temp_storage=storage)
    _store_thread_data_blocked(adjacent_out, values)

    flags = coop.discontinuity(group, values, temp_storage=storage)
    _store_thread_data_blocked(flags_out, flags)

    values = coop.scan(group, values, temp_storage=storage)
    _store_thread_data_blocked(scanned_out, values)

    values = coop.radix_sort_keys(
        group,
        values,
        begin_bit=0,
        end_bit=8,
        temp_storage=storage,
    )
    _store_thread_data_blocked(radix_out, values)

    values = coop.merge_sort_keys(group, values, temp_storage=storage)
    _store_thread_data_blocked(merge_out, values)

    coop.store(
        group,
        values_out,
        values,
        algorithm="transpose",
        temp_storage=storage,
    )


@cute.jit
def _run_deferred_all_temp_storage_exclusive(
    values_in: cute.Tensor,
    loaded_out: cute.Tensor,
    adjacent_out: cute.Tensor,
    flags_out: cute.Tensor,
    scanned_out: cute.Tensor,
    radix_out: cute.Tensor,
    merge_out: cute.Tensor,
    values_out: cute.Tensor,
):
    _deferred_all_temp_storage_exclusive_kernel(
        values_in,
        loaded_out,
        adjacent_out,
        flags_out,
        scanned_out,
        radix_out,
        merge_out,
        values_out,
    ).launch(grid=(1, 1, 1), block=(64, 1, 1))


@pytest.mark.skipif(
    find_cuda_tool("nvdisasm") is None,
    reason="requires nvdisasm to inspect the final cubin",
)
@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
def test_deferred_all_temp_storage_runtime_layout_and_link(
    monkeypatch,
    request,
    tmp_path,
    sharing,
):
    """Exercise one inferred identity across seven public-CUB block families.

    This is a direct CUTLASS-shaped port of the OG single-phase manual-sync
    stress kernel. Every stage writes an intermediate result so a later sort
    cannot hide corruption from an earlier collective.
    """

    monkeypatch.setenv(
        provider_bundle.CACHE_DIR_ENV,
        str(tmp_path / "provider-cache"),
    )
    monkeypatch.setenv(provider_bundle.CCCL_ROOT_ENV, str(REPOSITORY_ROOT))
    dsl_dump_dir = tmp_path / "dsl"
    dsl_dump_dir.mkdir()
    provider_bundle.reset_compile_state()
    request.addfinalizer(provider_bundle.reset_compile_state)
    cutlass.cuda.initialize_cuda_context()

    total_items = 64 * 4
    fake_values = runtime.make_fake_compact_tensor(Int32, (total_items,))
    captured_layouts = []
    captured_plans = []
    compile_with_layouts = provider_bundle.compile_bundle_source_with_layouts
    plan_events = provider_support.plan_deferred_temp_storage_events

    def capture_layouts(*args, **kwargs):
        compilation = compile_with_layouts(*args, **kwargs)
        captured_layouts.append(compilation.layouts)
        return compilation

    def capture_plans(events, layouts):
        plans = plan_events(events, layouts)
        captured_plans.append(plans)
        return plans

    monkeypatch.setattr(
        provider_bundle,
        "compile_bundle_source_with_layouts",
        capture_layouts,
    )
    monkeypatch.setattr(
        provider_support,
        "plan_deferred_temp_storage_events",
        capture_plans,
    )

    runner = (
        _run_deferred_all_temp_storage
        if sharing == "shared"
        else _run_deferred_all_temp_storage_exclusive
    )
    compiled = cute.compile[(KeepCUBIN, DumpDir(str(dsl_dump_dir)))](
        runner,
        fake_values,
        fake_values,
        fake_values,
        fake_values,
        fake_values,
        fake_values,
        fake_values,
        fake_values,
    )

    assert provider_bundle.get_nvrtc_compile_program_counter() == 1
    assert len(captured_layouts) == 1
    assert sorted(
        (layout.size_in_bytes, layout.alignment)
        for layout in captured_layouts[0].values()
    ) == sorted(
        (size_in_bytes, alignment)
        for _, size_in_bytes, alignment in _ALL_TEMP_STORAGE_EXPECTED_LAYOUTS
    )

    assert len(captured_plans) == 1
    assert len(captured_plans[0]) == 1
    plan = captured_plans[0][0]
    expected_size = (
        _ALL_TEMP_STORAGE_SHARED_SIZE
        if sharing == "shared"
        else _ALL_TEMP_STORAGE_EXCLUSIVE_SIZE
    )
    expected_offsets = (
        _ALL_TEMP_STORAGE_SHARED_OFFSETS
        if sharing == "shared"
        else _ALL_TEMP_STORAGE_EXCLUSIVE_OFFSETS
    )
    assert plan.size_in_bytes == expected_size
    assert plan.alignment == 16
    assert tuple(binding.event.primitive_name for binding in plan.bindings) == tuple(
        name for name, _, _ in _ALL_TEMP_STORAGE_EXPECTED_LAYOUTS
    )
    assert (
        tuple(binding.byte_offset_in_bytes for binding in plan.bindings)
        == expected_offsets
    )
    expected_binding_sizes = (
        (expected_size,) * len(_ALL_TEMP_STORAGE_EXPECTED_LAYOUTS)
        if sharing == "shared"
        else tuple(
            size_in_bytes for _, size_in_bytes, _ in _ALL_TEMP_STORAGE_EXPECTED_LAYOUTS
        )
    )
    assert tuple(binding.size_in_bytes for binding in plan.bindings) == (
        expected_binding_sizes
    )

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
    assert "cuda_coop_cutlass_cub_" not in sass
    assert re.search(r"\b(?:CALL|LDL|STL)(?:\.[A-Z0-9_]+)*\b", sass) is None

    values_host = (
        (torch.arange(total_items, dtype=torch.int32) * 37 + 11) % 29
    ).contiguous()
    values_in = values_host.cuda()
    loaded_out = torch.zeros_like(values_in)
    adjacent_out = torch.zeros_like(values_in)
    flags_out = torch.zeros_like(values_in)
    scanned_out = torch.zeros_like(values_in)
    radix_out = torch.zeros_like(values_in)
    merge_out = torch.zeros_like(values_in)
    values_out = torch.zeros_like(values_in)

    compiled(
        runtime.from_dlpack(values_in),
        runtime.from_dlpack(loaded_out),
        runtime.from_dlpack(adjacent_out),
        runtime.from_dlpack(flags_out),
        runtime.from_dlpack(scanned_out),
        runtime.from_dlpack(radix_out),
        runtime.from_dlpack(merge_out),
        runtime.from_dlpack(values_out),
    )
    torch.cuda.synchronize()

    expected_adjacent = values_host.clone()
    expected_adjacent[1:] = values_host[1:] - values_host[:-1]
    expected_flags = torch.zeros_like(values_host)
    expected_flags[0] = 1
    expected_flags[1:] = (expected_adjacent[1:] != expected_adjacent[:-1]).to(
        torch.int32
    )
    expected_scan = (
        torch.cumsum(expected_adjacent, dim=0, dtype=torch.int32) - expected_adjacent
    )
    radix_order = torch.argsort(expected_scan & 0xFF, stable=True)
    expected_radix = expected_scan[radix_order]
    expected_merge = torch.sort(expected_radix, stable=True).values

    torch.testing.assert_close(loaded_out.cpu(), values_host, atol=0, rtol=0)
    torch.testing.assert_close(adjacent_out.cpu(), expected_adjacent, atol=0, rtol=0)
    torch.testing.assert_close(flags_out.cpu(), expected_flags, atol=0, rtol=0)
    torch.testing.assert_close(scanned_out.cpu(), expected_scan, atol=0, rtol=0)
    torch.testing.assert_close(radix_out.cpu(), expected_radix, atol=0, rtol=0)
    torch.testing.assert_close(merge_out.cpu(), expected_merge, atol=0, rtol=0)
    torch.testing.assert_close(values_out.cpu(), expected_merge, atol=0, rtol=0)
