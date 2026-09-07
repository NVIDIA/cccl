# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def _provider_dependencies():
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")


def test_logical_warp_group_providers_use_exact_width_and_storage_partition():
    _provider_dependencies()

    from cutlass.base_dsl.typing import Int32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import (
        ArgumentBinding,
        GroupLoadStoreKind,
        LaunchFacts,
        ReduceValueKind,
        ScanValueKind,
    )
    from cuda.coop.cutlass import _group_exchange, _group_reduce, _group_scan
    from cuda.coop.cutlass._dsl import (
        _cub_exchange_provider,
        _cub_load_store_provider,
        _cub_scan_provider,
        _cudax_reduce_provider,
    )

    launch = LaunchFacts(exact_block_dim=64)
    group = coop.this_warp().group_by(8)

    exchange_plan = _group_exchange._make_group_exchange_plan(
        group=group,
        launch=launch,
        dtype=Int32,
        items_per_thread=2,
        mode="scatter_to_striped",
        rank_dtype=Int32,
    ).require_supported()
    exchange = _cub_exchange_provider._CubExchangeRequest(
        plan=exchange_plan,
        value_type=Int32,
        rank_type=Int32,
    )
    exchange_source = "\n".join(_cub_exchange_provider._render_cub_exchange(exchange))

    scan_plan = _group_scan._make_group_scan_plan(
        group=group,
        launch=launch,
        dtype=Int32,
        value_kind=ScanValueKind.SCALAR,
        items_per_thread=1,
        mode="inclusive",
        op="sum",
        aggregate=True,
        valid_items=5,
    ).require_supported()
    scan = _cub_scan_provider._CubScanRequest(
        plan=scan_plan,
        op="sum",
        value_type=Int32,
    )
    scan_source = "\n".join(_cub_scan_provider._render_cub_scan(scan))

    reduce_plan = _group_reduce._make_group_reduce_plan(
        group=group,
        launch=launch,
        dtype=Int32,
        value_kind=ReduceValueKind.SCALAR,
        items_per_thread=1,
        op="sum",
        broadcast=False,
        valid_items=ArgumentBinding.static(5),
    ).require_supported()
    reduce = _cudax_reduce_provider._CubReduceRequest(
        plan=reduce_plan,
        op="sum",
        value_type=Int32,
    )
    reduce_source = "\n".join(_cudax_reduce_provider._render_cub_reduce(reduce))

    load = _cub_load_store_provider._make_request(
        group=group,
        launch=launch,
        kind=GroupLoadStoreKind.LOAD,
        value_type=Int32,
        items_per_thread=2,
        algorithm="direct",
        valid_items_binding=ArgumentBinding.omitted(),
        oob_default_binding=ArgumentBinding.omitted(),
        offset_binding=ArgumentBinding.omitted(),
        external_scratch=False,
    )
    load_source = "\n".join(_cub_load_store_provider._render_cub_load_store(load))

    for source in (exchange_source, scan_source, reduce_source, load_source):
        assert "TempStorage storage[8]" in source
        assert "cuda_coop_cutlass_linear_tid() / 8u" in source
        assert "cuda_coop_cutlass_warp_sync()" in source
    assert "::cub::WarpExchange<int, 2, 8" in exchange_source
    assert ".ScatterToStriped(input_items, output_items, ranks)" in exchange_source
    assert "::cub::WarpScan<int, 8>" in scan_source
    assert (
        ".InclusiveScanPartial(value, result, ::cuda::std::plus<>{}, 5, aggregate)"
        in scan_source
    )
    assert "::cub::WarpReduce<int, 8>" in reduce_source
    assert ".Sum(item0, 5)" in reduce_source
    assert "::cub::WarpLoad<int, 2, ::cub::WARP_LOAD_DIRECT, 8>" in load_source
    assert "static_cast<long long>(storage_instance) * 16ll" in load_source


def test_block_group_exchange_renders_flagged_scatter_and_time_slicing():
    _provider_dependencies()

    from cutlass.base_dsl.typing import Int32, Uint32

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass import _group_exchange
    from cuda.coop.cutlass._dsl import _cub_exchange_provider

    plan = _group_exchange._make_group_exchange_plan(
        group=coop.this_block(),
        launch=LaunchFacts(exact_block_dim=64),
        dtype=Int32,
        items_per_thread=2,
        mode="scatter_to_striped_flagged",
        rank_dtype=Int32,
        valid_flag_dtype=Uint32,
        warp_time_slicing=True,
    ).require_supported()
    request = _cub_exchange_provider._CubExchangeRequest(
        plan=plan,
        value_type=Int32,
        rank_type=Int32,
        valid_flag_type=Uint32,
    )
    source = "\n".join(_cub_exchange_provider._render_cub_exchange(request))

    assert "::cub::BlockExchange<int, 64, 2, 1, 1, 1>" in source
    assert ".ScatterToStripedFlagged(" in source
    assert "input_items, output_items, ranks, valid_flags" in source
