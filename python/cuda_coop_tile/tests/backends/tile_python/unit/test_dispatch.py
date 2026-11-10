# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT


@pytest.fixture(autouse=True)
def _restore_cuda_modules():
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "cuda" or name.startswith("cuda.")
    }
    yield
    # Imports made with stub dependencies must not escape into later tests.
    for name in list(sys.modules):
        if name == "cuda" or name.startswith("cuda."):
            del sys.modules[name]
    sys.modules.update(saved)


def _stub_package(name: str, path: Path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = module.__path__
    module.__spec__ = spec
    return module


def _load_dispatch_with_cuda_tile_stubs(monkeypatch):
    monkeypatch.syspath_prepend(str(SOURCE_ROOT))
    monkeypatch.setattr(
        sys,
        "meta_path",
        [
            finder
            for finder in sys.meta_path
            if "cuda.coop.tile" not in getattr(finder, "known_source_files", {})
        ],
    )
    for loaded_module in list(sys.modules):
        if loaded_module == "cuda" or loaded_module.startswith("cuda.coop"):
            sys.modules.pop(loaded_module)

    cuda_pkg = _stub_package("cuda", SOURCE_ROOT / "cuda")
    cuda_coop = _stub_package("cuda.coop", SOURCE_ROOT / "cuda" / "coop")
    cuda_coop_tile = _stub_package(
        "cuda.coop.tile", SOURCE_ROOT / "cuda" / "coop" / "tile"
    )

    cuda_tile = types.ModuleType("cuda.tile")
    cuda_tile.__path__ = []
    cuda_pkg.coop = cuda_coop
    cuda_pkg.tile = cuda_tile
    cuda_coop.tile = cuda_coop_tile

    datatype = types.ModuleType("cuda.tile._datatype")

    class DType:
        pass

    datatype.DType = DType
    dtype_bits = {
        "bfloat16": 16,
        "float16": 16,
        "float32": 32,
        "float64": 64,
        "int8": 8,
        "uint8": 8,
        "int16": 16,
        "uint16": 16,
        "int32": 32,
        "uint32": 32,
        "int64": 64,
        "uint64": 64,
    }
    for name, bitwidth in dtype_bits.items():
        dtype = DType()
        dtype.bitwidth = bitwidth
        setattr(datatype, name, dtype)

    exception = types.ModuleType("cuda.tile._exception")

    class TileTypeError(TypeError):
        pass

    exception.TileTypeError = TileTypeError

    cext = types.ModuleType("cuda.tile._cext")
    cext.get_compute_capability = lambda: (9, 0)

    monkeypatch.setitem(sys.modules, "cuda", cuda_pkg)
    monkeypatch.setitem(sys.modules, "cuda.coop", cuda_coop)
    monkeypatch.setitem(sys.modules, "cuda.coop.tile", cuda_coop_tile)
    monkeypatch.setitem(sys.modules, "cuda.tile", cuda_tile)
    monkeypatch.setitem(sys.modules, "cuda.tile._datatype", datatype)
    monkeypatch.setitem(sys.modules, "cuda.tile._exception", exception)
    monkeypatch.setitem(sys.modules, "cuda.tile._cext", cext)

    return importlib.import_module("cuda.coop.tile._dispatch"), datatype


def _load_block_module_with_cuda_tile_stubs(monkeypatch, module_name):
    _, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)
    for loaded_module in list(sys.modules):
        if loaded_module.startswith("cuda.coop.tile._block"):
            sys.modules.pop(loaded_module)
    return importlib.import_module(
        f"cuda.coop.tile._block.{module_name}"
    ), datatype


def test_histogram_dispatch_selects_common_thread_divisor(monkeypatch):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)

    decision = dispatch.dispatch_histogram(
        numel=256,
        dtype=datatype.uint8,
        bins=64,
    )

    assert decision.block_threads == 64


def test_histogram_dispatch_accepts_small_bin_counts(monkeypatch):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)

    decision = dispatch.dispatch_histogram(
        numel=256,
        dtype=datatype.uint8,
        bins=32,
    )

    assert decision.block_threads == 32


def test_scan_and_reduce_dispatch_defaults(monkeypatch):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)

    scan = dispatch.dispatch_exclusive_sum(
        numel=64,
        dtype=datatype.float32,
        requested_algorithm=None,
    )
    reduce = dispatch.dispatch_sum(
        numel=256,
        dtype=datatype.float32,
        requested_algorithm=None,
    )

    assert (scan.backend_family, scan.block_threads, scan.algorithm) == (
        "warp_like",
        64,
        "raking",
    )
    assert (reduce.backend_family, reduce.block_threads, reduce.algorithm) == (
        "cta_block",
        128,
        "warp_reductions",
    )


def test_dispatch_sort_defaults_and_explicit_exchange(monkeypatch):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)

    merge_pairs = dispatch.dispatch_merge_sort_pairs(
        numel=128,
        key_dtype=datatype.int32,
        value_dtype=datatype.float32,
        descending=False,
        valid_items=None,
    )
    radix_keys = dispatch.dispatch_radix_sort_keys(
        numel=128,
        dtype=datatype.uint32,
        inner_scan_algorithm="warp_scans",
    )
    exchange = dispatch.dispatch_exchange_striped_to_blocked(
        numel=256,
        dtype=datatype.float32,
        block_threads=128,
    )

    assert merge_pairs.block_threads == 128
    assert radix_keys.algorithm == "warp_scans"
    assert radix_keys.backend_family == "warp_like"
    assert exchange.block_threads == 128


def test_merge_sort_dispatch_keeps_a_32_item_sort_warp_sized(monkeypatch):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)

    merge_keys = dispatch.dispatch_merge_sort_keys(
        numel=32,
        dtype=datatype.int32,
        descending=False,
        valid_items=None,
    )
    merge_pairs = dispatch.dispatch_merge_sort_pairs(
        numel=32,
        key_dtype=datatype.int32,
        value_dtype=datatype.float32,
        descending=False,
        valid_items=None,
    )

    assert (merge_keys.backend_family, merge_keys.block_threads) == (
        "warp_like",
        32,
    )
    assert (merge_pairs.backend_family, merge_pairs.block_threads) == (
        "warp_like",
        32,
    )


def test_sort_dispatch_preserves_one_item_per_thread_contract(monkeypatch):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)

    merge_keys = dispatch.dispatch_merge_sort_keys(
        numel=256,
        dtype=datatype.int32,
        descending=False,
        valid_items=None,
    )
    assert merge_keys.block_threads == 128

    with pytest.raises(
        dispatch.TileTypeError, match="does not support tile size"
    ):
        dispatch.dispatch_radix_sort_keys(
            numel=17,
            dtype=datatype.uint32,
            inner_scan_algorithm=None,
        )


def test_shuffle_and_radix_rank_dispatch(monkeypatch):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)

    shuffle = dispatch.dispatch_shuffle(
        primitive="shuffle_rotate",
        numel=64,
        dtype=datatype.int32,
        mode="rotate",
    )
    rank = dispatch.dispatch_radix_rank(numel=256, dtype=datatype.uint32)

    assert shuffle.block_threads == 64
    assert shuffle.algorithm == "rotate"
    assert rank.block_threads == 128


def test_shuffle_rotate_rejects_distance_at_block_size(monkeypatch):
    shuffle, datatype = _load_block_module_with_cuda_tile_stubs(
        monkeypatch, "shuffle"
    )

    with pytest.raises(
        shuffle.TileTypeError, match="distance must be < block_threads"
    ):
        shuffle.BlockShuffle(
            mode="rotate",
            items_per_thread=1,
            dtype=datatype.int32,
            block_threads=32,
            distance=32,
        )


def test_shuffle_offset_rejects_distance_at_block_size(monkeypatch):
    shuffle, datatype = _load_block_module_with_cuda_tile_stubs(
        monkeypatch, "shuffle"
    )

    with pytest.raises(
        shuffle.TileTypeError,
        match="-block_threads < distance < block_threads",
    ):
        shuffle.BlockShuffle(
            mode="offset",
            items_per_thread=1,
            dtype=datatype.int32,
            block_threads=32,
            distance=-32,
        )


def test_exclusive_scan_defaults_sum_to_zero_and_requires_non_sum_seed(
    monkeypatch,
):
    scan, datatype = _load_block_module_with_cuda_tile_stubs(
        monkeypatch, "scan"
    )

    options = scan.BlockScanExclusiveScanOptions.from_kwargs(
        scan_op="+",
        initial_value=None,
        prefix_value=None,
        algorithm=None,
        methods=None,
    )
    block = scan.BlockScanExclusiveScan(
        items_per_thread=1,
        dtype=datatype.int32,
        scan_op="+",
    )

    assert options.scan_op == "+"
    assert options.initial_value is None
    assert options.prefix_value is None
    assert block.scan_op_kind == "sum"
    assert block.initial_value is None
    assert block.prefix_value is None

    with pytest.raises(
        scan.TileTypeError,
        match="requires initial_value or prefix_value for a non-sum scan_op",
    ):
        scan.BlockScanExclusiveScanOptions.from_kwargs(
            scan_op="max",
            initial_value=None,
            prefix_value=None,
            algorithm=None,
            methods=None,
        )

    with pytest.raises(
        scan.TileTypeError,
        match="requires initial_value or prefix_value for a non-sum scan_op",
    ):
        scan.BlockScanExclusiveScan(
            items_per_thread=1,
            dtype=datatype.int32,
            scan_op="max",
        )

    inclusive = scan.BlockScanInclusiveScanOptions.from_kwargs(
        scan_op="+",
        initial_value=None,
        prefix_value=None,
        algorithm=None,
        methods=None,
    )

    assert inclusive.initial_value is None
    assert inclusive.prefix_value is None


def test_exclusive_scan_rejects_initial_and_prefix_value(monkeypatch):
    scan, datatype = _load_block_module_with_cuda_tile_stubs(
        monkeypatch, "scan"
    )

    with pytest.raises(
        scan.TileTypeError,
        match="does not support both initial_value and prefix_value",
    ):
        scan.BlockScanExclusiveScanOptions.from_kwargs(
            scan_op="+",
            initial_value=0,
            prefix_value=1,
            algorithm=None,
            methods=None,
        )

    with pytest.raises(
        scan.TileTypeError,
        match="does not support both initial_value and prefix_value",
    ):
        scan.BlockScanExclusiveScan(
            items_per_thread=1,
            dtype=datatype.int32,
            scan_op="+",
            initial_value=0,
            prefix_value=1,
        )


def test_inclusive_scan_rejects_initial_value(monkeypatch):
    scan, datatype = _load_block_module_with_cuda_tile_stubs(
        monkeypatch, "scan"
    )

    with pytest.raises(
        scan.TileTypeError,
        match="does not support initial_value",
    ):
        scan.BlockScanInclusiveScanOptions.from_kwargs(
            scan_op="+",
            initial_value=0,
            prefix_value=None,
            algorithm=None,
            methods=None,
        )

    with pytest.raises(
        scan.TileTypeError,
        match="does not support initial_value",
    ):
        scan.BlockScanInclusiveScan(
            items_per_thread=2,
            dtype=datatype.int32,
            scan_op="+",
            initial_value=0,
        )


def test_radix_sort_rejects_radix_bits_above_native_limit(monkeypatch):
    radix_sort, datatype = _load_block_module_with_cuda_tile_stubs(
        monkeypatch, "radix_sort"
    )
    base_options = {
        "descending": None,
        "begin_bit": None,
        "end_bit": None,
        "radix_bits": 9,
        "memoize_outer_scan": None,
        "inner_scan_algorithm": None,
        "smem_config": None,
        "methods": None,
    }

    with pytest.raises(
        radix_sort.TileTypeError,
        match=r"radix_bits must be in the range \[1, 8\]",
    ):
        radix_sort.BlockRadixSortKeysOptions.from_kwargs(**base_options)

    with pytest.raises(
        radix_sort.TileTypeError,
        match=r"radix_bits must be in the range \[1, 8\]",
    ):
        radix_sort.BlockRadixSortPairsOptions.from_kwargs(
            **(
                base_options
                | {"radix_bits": None, "methods": {"radix_bits": 9}}
            )
        )

    with pytest.raises(
        radix_sort.TileTypeError,
        match=r"radix_bits must be in the range \[1, 8\]",
    ):
        radix_sort.BlockRadixSortKeys(
            items_per_thread=1,
            dtype=datatype.int32,
            radix_bits=9,
        )

    with pytest.raises(
        radix_sort.TileTypeError,
        match=r"radix_bits must be in the range \[1, 8\]",
    ):
        radix_sort.BlockRadixSortPairs(
            items_per_thread=1,
            key_dtype=datatype.int32,
            value_dtype=datatype.int32,
            radix_bits=9,
        )


def test_methods_values_win_when_public_defaults_are_omitted(monkeypatch):
    reduce, _ = _load_block_module_with_cuda_tile_stubs(monkeypatch, "reduce")
    scan, _ = _load_block_module_with_cuda_tile_stubs(monkeypatch, "scan")
    merge_sort, _ = _load_block_module_with_cuda_tile_stubs(
        monkeypatch, "merge_sort"
    )
    radix_sort, _ = _load_block_module_with_cuda_tile_stubs(
        monkeypatch, "radix_sort"
    )

    reduce_options = reduce.BlockReduceOptions.from_kwargs(
        binary_op=None,
        algorithm=None,
        methods={"binary_op": "max"},
    )
    scan_options = scan.BlockScanInclusiveScanOptions.from_kwargs(
        scan_op=None,
        initial_value=None,
        prefix_value=None,
        algorithm=None,
        methods={"scan_op": "max", "prefix_value": 4},
    )
    explicit_scan_prefix = scan.BlockScanExclusiveSumOptions.from_kwargs(
        prefix_value=5,
        algorithm=None,
        methods={"prefix_value": 4},
    )
    merge_options = merge_sort.BlockMergeSortKeysOptions.from_kwargs(
        descending=None,
        valid_items=None,
        oob_default=None,
        methods={"descending": True},
    )
    radix_options = radix_sort.BlockRadixSortKeysOptions.from_kwargs(
        descending=None,
        begin_bit=None,
        end_bit=None,
        radix_bits=None,
        memoize_outer_scan=None,
        inner_scan_algorithm=None,
        smem_config=None,
        methods={"descending": True},
    )

    assert reduce_options.binary_op == "max"
    assert scan_options.scan_op == "max"
    assert scan_options.prefix_value == 4
    assert explicit_scan_prefix.prefix_value == 5
    assert merge_options.descending is True
    assert radix_options.descending is True


def test_provider_runtime_rejects_incomplete_explicit_cccl_root(
    monkeypatch, tmp_path
):
    _load_dispatch_with_cuda_tile_stubs(monkeypatch)
    runtime = importlib.import_module("cuda.coop.tile._provider_runtime")
    incomplete = tmp_path / "cccl"
    (incomplete / "cub/cub").mkdir(parents=True)
    (incomplete / "cub/cub/version.cuh").touch()
    monkeypatch.setenv("CUDA_COOP_CCCL_ROOT", str(incomplete))

    with pytest.raises(RuntimeError, match="missing source include roots"):
        runtime._include_paths()


def test_sort_options_accept_descending_in_methods(monkeypatch):
    merge_sort, _ = _load_block_module_with_cuda_tile_stubs(
        monkeypatch, "merge_sort"
    )
    radix_sort, _ = _load_block_module_with_cuda_tile_stubs(
        monkeypatch, "radix_sort"
    )

    merge_options = merge_sort.BlockMergeSortKeysOptions.from_kwargs(
        descending=None,
        valid_items=None,
        oob_default=None,
        methods={"descending": True},
    )
    radix_options = radix_sort.BlockRadixSortKeysOptions.from_kwargs(
        descending=False,
        begin_bit=None,
        end_bit=None,
        radix_bits=None,
        memoize_outer_scan=None,
        inner_scan_algorithm=None,
        smem_config=None,
        methods={"descending": True},
    )

    assert merge_options.descending is True
    assert radix_options.descending is False


def test_dispatch_rejection_branches(monkeypatch):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)

    with pytest.raises(dispatch.TileTypeError, match="bins must be <= 256"):
        dispatch.dispatch_histogram(numel=256, dtype=datatype.uint8, bins=512)

    with pytest.raises(dispatch.TileTypeError, match="one item per thread"):
        dispatch.dispatch_shuffle(
            primitive="shuffle_offset",
            numel=2048,
            dtype=datatype.int32,
            mode="offset",
        )


@pytest.mark.parametrize("threads", [32, 64, 128, 256])
def test_exchange_dispatch_preserves_explicit_topology(monkeypatch, threads):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)
    for operation in (
        dispatch.dispatch_exchange_striped_to_blocked,
        dispatch.dispatch_exchange_blocked_to_striped,
    ):
        decision = operation(
            numel=256, dtype=datatype.int32, block_threads=threads
        )
        assert decision.block_threads == threads


@pytest.mark.parametrize("threads", [None, True, 128.0, 0, 16, 33, 1056])
def test_exchange_dispatch_rejects_invalid_topology(monkeypatch, threads):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)
    with pytest.raises(
        dispatch.TileTypeError,
        match="block_threads must be a compile-time integer",
    ):
        dispatch.dispatch_exchange_striped_to_blocked(
            numel=256, dtype=datatype.int32, block_threads=threads
        )


def test_exchange_dispatch_rejects_nondivisible_extent(monkeypatch):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)
    with pytest.raises(
        dispatch.TileTypeError,
        match="tile extent must be positive and divisible",
    ):
        dispatch.dispatch_exchange_blocked_to_striped(
            numel=256, dtype=datatype.int32, block_threads=96
        )


@pytest.mark.parametrize("extent", [1, 2, 4, 8, 16, 20, 24])
def test_short_reduction_rows_must_divide_a_warp(monkeypatch, extent):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)
    for operation in (dispatch.dispatch_sum, dispatch.dispatch_reduce):
        if 32 % extent:
            with pytest.raises(dispatch.TileTypeError, match="must divide 32"):
                operation(
                    numel=extent,
                    dtype=datatype.int32,
                    requested_algorithm=None,
                )
        else:
            decision = operation(
                numel=extent,
                dtype=datatype.int32,
                requested_algorithm=None,
            )
            assert decision.block_threads == 32


@pytest.mark.parametrize("numel,bins", [(16, 32), (100, 64), (256, 8)])
def test_histogram_dispatch_rejects_unrealizable_worker_widths(
    monkeypatch, numel, bins
):
    dispatch, datatype = _load_dispatch_with_cuda_tile_stubs(monkeypatch)
    with pytest.raises(dispatch.TileTypeError, match="divides both"):
        dispatch.dispatch_histogram(
            numel=numel, dtype=datatype.uint8, bins=bins
        )
