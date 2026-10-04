# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check independent physical-warp tiles, layouts, and synchronization.

Two warps in a three-dimensional block use different counts, offsets, and
defaults. Plain CuTe accesses observe Load and supply Store independently.
Other cases exercise scratch reuse across runtime iterations, whole-warp
divergence, and the instructions in the final linked kernel.
"""

import importlib.util
import re
import shutil
import subprocess

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import (
    NUMPY_DTYPES,
    cutlass_dtype,
    device_array,
    values_for,
)
from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_APIS = (coop, cutlass_coop)
_ALGORITHMS = ("direct", "striped", "vectorize", "transpose")
_BLOCK = (8, 4, 2)
_THREADS = 64
_WIDTH = 32
_ITEMS = 4
_WARP_TILE = _WIDTH * _ITEMS
_BLOCK_TILE = _THREADS * _ITEMS


def _index(algorithm, lane, item):
    """Return an item's memory index relative to its warp's tile origin.

    Striped operations interleave lanes; other algorithms expose blocked
    items, including transpose after its internal redistribution.
    """

    return (
        lane + item * _WIDTH if algorithm == "striped" else lane * _ITEMS + item
    )


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize(
    "base_valid",
    (0, 100, _WARP_TILE),
    ids=("zero", "partial", "full-first-warp"),
)
def test_group_local_load_counts_offsets_and_defaults(
    api, algorithm, base_valid
):
    """Observe Load with different uniform controls in each physical warp.

    A warp's count, offset, and default differ from its neighbor's values but
    remain uniform across its own lanes. Plain CuTe output preserves each
    thread's payload, so the host can check its layout without using Store.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        valid: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        warp = thread // _WIDTH
        count = valid - warp * 7
        if count < 0:
            count = cutlass.Int32(0)
        payload = api.ThreadData(items_per_thread)
        returned = api.load(
            api.this_warp(),
            source,
            payload,
            algorithm=algorithm,
            valid_items=count,
            oob_default=cutlass.Int32(-29 - warp),
            offset=cutlass.Int64(3 + warp * 2),
        )
        assert returned is None
        outputs = cute.make_tensor(observed, cute.make_layout(_BLOCK_TILE))
        for item in cutlass.range_constexpr(items_per_thread):
            outputs[thread * items_per_thread + item] = payload[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        observed: cute.Pointer,
        valid: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, observed, valid, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    source = values_for(np.int32, _BLOCK_TILE + 7, shift=11)
    observed = np.full(_BLOCK_TILE, 71, dtype=np.int32)
    expected = np.empty_like(observed)
    for thread in range(_THREADS):
        warp, lane = divmod(thread, _WIDTH)
        count = max(base_valid - warp * 7, 0)
        origin = warp * _WARP_TILE + 3 + warp * 2
        for item in range(_ITEMS):
            index = _index(algorithm, lane, item)
            expected[thread * _ITEMS + item] = (
                source[origin + index] if index < count else -29 - warp
            )
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, base_valid, _ITEMS)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize(
    "base_valid",
    (0, 100, _WARP_TILE),
    ids=("zero", "partial", "full-first-warp"),
)
def test_group_local_store_counts_and_offsets(api, algorithm, base_valid):
    """Check Store with independently filled payloads and per-warp bounds.

    Plain CuTe reads fill the payloads. Host-side indexing predicts the stored
    positions for each warp, while sentinels expose writes into tile gaps or
    past the selected counts.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        valid: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        warp = thread // _WIDTH
        count = valid - warp * 7
        if count < 0:
            count = cutlass.Int32(0)
        inputs = cute.make_tensor(source, cute.make_layout(_BLOCK_TILE))
        payload = api.ThreadData(items_per_thread, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(items_per_thread):
            payload[item] = inputs[thread * items_per_thread + item]
        api.store(
            api.this_warp(),
            destination,
            payload,
            algorithm=algorithm,
            valid_items=count,
            offset=cutlass.Int64(5 + warp * 4),
        )

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        valid: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, valid, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    source = values_for(np.int32, _BLOCK_TILE, shift=19)
    destination = np.full(_BLOCK_TILE + 15, -101, dtype=np.int32)
    expected = destination.copy()
    for thread in range(_THREADS):
        warp, lane = divmod(thread, _WIDTH)
        count = max(base_valid - warp * 7, 0)
        origin = warp * _WARP_TILE + 5 + warp * 4
        for item in range(_ITEMS):
            index = _index(algorithm, lane, item)
            if index < count:
                expected[origin + index] = source[thread * _ITEMS + item]
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, base_valid, _ITEMS)
    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("operation", ("load", "store"))
@pytest.mark.parametrize("algorithm", ("direct", "transpose"))
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_warp_layout_for_every_dtype(
    api, dtype, operation, algorithm, items_per_thread
):
    """Check Load and Store separately for each supported scalar type.

    Each kernel uses only one collective direction. Plain CuTe supplies or
    observes the other side, so matching errors in Load and Store cannot hide
    an incorrect payload layout or scalar conversion.
    """

    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.recast_tensor(
            cute.make_tensor(
                source, cute.make_layout(_THREADS * items_per_thread)
            ),
            value_type,
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(
                destination, cute.make_layout(_THREADS * items_per_thread)
            ),
            value_type,
        )
        payload = api.ThreadData(items_per_thread, dtype=value_type)
        if cutlass.const_expr(operation == "load"):
            api.load(api.this_warp(), inputs, payload, algorithm=algorithm)
            for item in cutlass.range_constexpr(items_per_thread):
                outputs[thread * items_per_thread + item] = payload[item]
        else:
            for item in cutlass.range_constexpr(items_per_thread):
                payload[item] = inputs[thread * items_per_thread + item]
            api.store(api.this_warp(), outputs, payload, algorithm=algorithm)

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    source = values_for(dtype, (_THREADS * items_per_thread), shift=29)
    destination = np.zeros_like(source)
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, items_per_thread)
    np.testing.assert_array_equal(destination, source)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize(
    "static_count", (False, True), ids=("runtime", "static")
)
def test_partial_transpose_loads_each_warps_valid_items_without_default(
    api, static_count
):
    """Observe only defined entries after a partial transpose load.

    No default initializes the invalid tail. The kernel therefore writes only
    items inside its warp's count, leaving sentinels elsewhere. Runtime counts
    also differ between warps to check that one warp's bound stays local.
    """

    valid = _WARP_TILE - 19

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        count: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        outputs = cute.make_tensor(observed, cute.make_layout(_BLOCK_TILE))
        payload = api.ThreadData(items_per_thread, dtype=cutlass.Int32)
        if cutlass.const_expr(static_count):
            api.load(
                api.this_warp(),
                source,
                payload,
                algorithm="transpose",
                valid_items=valid,
            )
        else:
            api.load(
                api.this_warp(),
                source,
                payload,
                algorithm="transpose",
                valid_items=count - (thread // _WIDTH) * 13,
            )
        selected_count = (
            valid if static_count else count - (thread // _WIDTH) * 13
        )
        for item in cutlass.range_constexpr(items_per_thread):
            if (thread % _WIDTH) * items_per_thread + item < selected_count:
                outputs[thread * items_per_thread + item] = payload[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        observed: cute.Pointer,
        count: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, observed, count, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    source = values_for(np.int32, _BLOCK_TILE)
    observed = np.full(_BLOCK_TILE, 71, dtype=np.int32)
    expected = observed.copy()
    for warp in range(_THREADS // _WIDTH):
        start = warp * _WARP_TILE
        count = valid if static_count else valid - warp * 13
        expected[start : start + count] = source[start : start + count]
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, valid, _ITEMS)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize(
    "divergent", (False, True), ids=("all-warps", "second-warp-only")
)
def test_transpose_runtime_loop_and_whole_warp_divergence(api, divergent):
    """Reuse transpose scratch while a neighboring warp may skip every call.

    A reuse barrier prevents later calls from overwriting scratch that lanes
    still read. It must cover only the warp; a block barrier would require the
    skipped warp to participate. The branch selects whole warps, so all lanes
    in a participating warp take the same path. Eight runtime iterations add
    tile- and warp-based increments. The skipped warp's output must keep its
    sentinel values.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        tiles: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        warp = thread // _WIDTH
        if warp == 1 or not divergent:
            for tile in range(tiles):
                payload = api.ThreadData(items_per_thread)
                api.load(
                    api.this_warp(),
                    source,
                    payload,
                    algorithm="transpose",
                    offset=tile * _BLOCK_TILE,
                )
                for item in cutlass.range_constexpr(items_per_thread):
                    payload[item] = payload[item] + cutlass.Int32(
                        tile + warp + 1
                    )
                api.store(
                    api.this_warp(),
                    destination,
                    payload,
                    algorithm="transpose",
                    offset=tile * _BLOCK_TILE,
                )

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        tiles: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, tiles, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    tiles = 8
    source = values_for(np.int32, tiles * _BLOCK_TILE, shift=43)
    destination = np.full_like(source, -101)
    expected = destination.copy()
    for tile in range(tiles):
        for warp in range(_THREADS // _WIDTH):
            if divergent and warp != 1:
                continue
            start = tile * _BLOCK_TILE + warp * _WARP_TILE
            expected[start : start + _WARP_TILE] = (
                source[start : start + _WARP_TILE] + tile + warp + 1
            )
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, tiles, _ITEMS)
    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_final_warp_cubin_has_no_block_barrier(tmp_path, algorithm):
    """Inspect the final machine code after verifying the copy.

    The SASS must contain no Load or Store wrapper symbol and no CALL
    instruction, so the wrappers are fully inlined. It must contain no block
    barrier (BAR), which would require other warps to participate. Resource
    metadata must show shared memory only for transpose. Storage-free paths
    must omit the warp synchronization instruction WARPSYNC. Transpose permits
    warp synchronization without requiring a particular instruction or mask.
    """

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip(
            "cuobjdump is required to inspect final linked instructions"
        )

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        payload = cutlass_coop.ThreadData(items_per_thread)
        cutlass_coop.load(
            cutlass_coop.this_warp(), source, payload, algorithm=algorithm
        )
        cutlass_coop.store(
            cutlass_coop.this_warp(), destination, payload, algorithm=algorithm
        )

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    source = values_for(np.int32, _BLOCK_TILE, shift=59)
    destination = np.zeros_like(source)
    with device_array(source) as src, device_array(destination) as dst:
        compiled = cute.compile[(KeepCUBIN, DumpDir(str(tmp_path)))](
            launch, src, dst, _ITEMS
        )
        compiled(src, dst)
    np.testing.assert_array_equal(destination, source)
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output(
            [cuobjdump, "--dump-sass", str(cubin)], text=True
        )
        resources = subprocess.check_output(
            [cuobjdump, "--dump-resource-usage", str(cubin)], text=True
        )
        cubin.with_suffix(".sass").write_text(sass)
        cubin.with_suffix(".resources").write_text(resources)
        assert "cuda_coop_cutlass_load_" not in sass
        assert "cuda_coop_cutlass_store_" not in sass
        assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
        assert re.search(r"\bBAR(?:\.[A-Z0-9_]+)*\b", sass) is None
        shared = [
            int(size) for size in re.findall(r"\bSHARED:(\d+)", resources)
        ]
        assert shared
        assert any(shared) is (algorithm == "transpose")
        if algorithm != "transpose":
            assert "WARPSYNC" not in sass


@pytest.mark.parametrize("api", ("common", "qualified"))
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_warp_example(api, items_per_thread):
    path = PACKAGE_ROOT / "examples/cutlass/warp_load_store.py"
    spec = importlib.util.spec_from_file_location(
        "cutlass_warp_load_store_example", path
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(api, items_per_thread=items_per_thread)
