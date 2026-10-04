# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check block data layouts, tile bounds, and shared-memory use on the GPU.

Layout tests observe Load with plain CuTe writes and fill Store with plain
CuTe reads. This checks each direction independently, so matching mistakes
in Load and Store cannot make a round trip pass. Storage cases cover
repeated calls, caller allocations, and the linked kernel's instructions
and resource metadata.
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
from cutlass.memory import SmemAllocator

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array, values_for
from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_ALGORITHMS = (
    "direct",
    "striped",
    "vectorize",
    "transpose",
    "warp_transpose",
    "warp_transpose_timesliced",
)
_SCRATCH_ALGORITHMS = _ALGORITHMS[3:]
_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS = 4
_TILE = _THREADS * _ITEMS


def _tile_index(algorithm, thread, item):
    """Map one thread's item to its expected position in the tile.

    Striped operations interleave threads for each item. The other algorithms
    present blocked items to the caller, even when they redistribute data
    internally through shared memory.
    """

    return (
        thread + item * _THREADS
        if algorithm == "striped"
        else thread * _ITEMS + item
    )


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize(
    "valid", (0, _TILE - 19, _TILE), ids=("zero", "partial", "full")
)
def test_load_layout_and_runtime_bounds(api, algorithm, valid):
    """Observe each loaded item without passing it through Store.

    The kernel writes each thread's payload to a plain CuTe output. The host
    then applies the expected layout and runtime bound, including the default
    value for items beyond that bound.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        count: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        payload = api.ThreadData(items_per_thread)
        api.load(
            api.this_block(),
            source,
            payload,
            algorithm=algorithm,
            valid_items=count,
            oob_default=-29,
            offset=3,
        )
        outputs = cute.make_tensor(observed, cute.make_layout(_TILE))
        for item in cutlass.range_constexpr(items_per_thread):
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

    source = values_for(np.int32, _TILE + 3, shift=17)
    observed = np.full(_TILE, 71, dtype=np.int32)
    expected = np.full(_TILE, -29, dtype=np.int32)
    for thread in range(_THREADS):
        for item in range(_ITEMS):
            index = _tile_index(algorithm, thread, item)
            if index < valid:
                expected[thread * _ITEMS + item] = source[3 + index]
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, valid, _ITEMS)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize(
    "valid", (0, _TILE - 19, _TILE), ids=("zero", "partial", "full")
)
def test_store_layout_and_bounds(api, algorithm, valid):
    """Check Store against payloads filled independently of Load.

    Plain CuTe reads assign distinct values to each thread's payload. The host
    maps them to the expected tile positions. Sentinels reveal writes outside
    the requested interval, including the empty-tile case.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_TILE))
        payload = api.ThreadData(items_per_thread, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(items_per_thread):
            payload[item] = inputs[thread * items_per_thread + item]
        api.store(
            api.this_block(),
            destination,
            payload,
            algorithm=algorithm,
            valid_items=valid,
            offset=5,
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

    source = values_for(np.int32, _TILE, shift=37)
    destination = np.full(_TILE + 11, -41, dtype=np.int32)
    expected = destination.copy()
    for thread in range(_THREADS):
        for item in range(_ITEMS):
            index = _tile_index(algorithm, thread, item)
            if index < valid:
                expected[5 + index] = source[thread * _ITEMS + item]
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, _ITEMS)
    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize("algorithm", _SCRATCH_ALGORITHMS)
@pytest.mark.parametrize(
    "static_count", (False, True), ids=("runtime", "static")
)
def test_partial_transpose_loads_valid_items_without_default(
    algorithm, static_count
):
    """Read only the defined payload entries from a partial transpose load.

    Without a default value, entries beyond ``valid_items`` are undefined.
    The kernel therefore skips those entries when it writes the output. Their
    host-side sentinels must survive for both static and runtime bounds.
    """

    valid = _TILE - 19

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        count: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        outputs = cute.make_tensor(observed, cute.make_layout(_TILE))
        payload = cutlass_coop.ThreadData(items_per_thread, dtype=cutlass.Int32)
        if cutlass.const_expr(static_count):
            cutlass_coop.load(
                cutlass_coop.this_block(),
                source,
                payload,
                algorithm=algorithm,
                valid_items=valid,
            )
        else:
            cutlass_coop.load(
                cutlass_coop.this_block(),
                source,
                payload,
                algorithm=algorithm,
                valid_items=count,
            )
        for item in cutlass.range_constexpr(items_per_thread):
            if thread * items_per_thread + item < count:
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

    source = values_for(np.int32, _TILE)
    observed = np.full(_TILE, 71, dtype=np.int32)
    expected = observed.copy()
    expected[:valid] = source[:valid]
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, valid, _ITEMS)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    "offset", (0, 1), ids=("aligned", "misaligned-fallback")
)
@pytest.mark.parametrize(
    "algorithm", ("vectorize", "warp_transpose_timesliced")
)
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_unguarded_full_tiles_use_the_correct_layout(
    algorithm, offset, items_per_thread
):
    """Check full-tile Load and Store at aligned and shifted addresses.

    Neither call passes ``valid_items``. Plain CuTe stores observe Load, while
    plain CuTe reads fill the Store payload. These checks are independent.
    With ``vectorize`` and four items per thread, a one-item offset misaligns
    both source and destination, forcing the per-item fallback.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        observed: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(
            source, cute.make_layout((_THREADS * items_per_thread) + 1)
        )
        outputs = cute.make_tensor(
            observed, cute.make_layout(_THREADS * items_per_thread)
        )
        loaded = cutlass_coop.ThreadData(items_per_thread)
        cutlass_coop.load(
            cutlass_coop.this_block(),
            source,
            loaded,
            algorithm=algorithm,
            offset=offset,
        )
        stored = cutlass_coop.ThreadData(items_per_thread, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(items_per_thread):
            outputs[thread * items_per_thread + item] = loaded[item]
            stored[item] = inputs[thread * items_per_thread + item]
        cutlass_coop.store(
            cutlass_coop.this_block(),
            destination,
            stored,
            algorithm=algorithm,
            offset=offset,
        )

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        observed: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, observed, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    source = values_for(np.int32, (_THREADS * items_per_thread) + 1, shift=59)
    destination = np.full(
        (_THREADS * items_per_thread) + 1, -101, dtype=np.int32
    )
    observed = np.zeros((_THREADS * items_per_thread), dtype=np.int32)
    expected = destination.copy()
    expected[offset : offset + (_THREADS * items_per_thread)] = source[
        : (_THREADS * items_per_thread)
    ]
    with (
        device_array(source) as src,
        device_array(destination) as dst,
        device_array(observed) as out,
    ):
        launch(src, dst, out, items_per_thread)
    np.testing.assert_array_equal(
        observed, source[offset : offset + (_THREADS * items_per_thread)]
    )
    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize("algorithm", _SCRATCH_ALGORITHMS)
@pytest.mark.parametrize("capacity", (None, 16384), ids=("deferred", "fixed"))
@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("manual_sync", (False, True), ids=("auto", "manual"))
def test_storage_reuse_in_runtime_loop(
    algorithm, capacity, sharing, manual_sync
):
    """Reuse the same storage descriptor across eight runtime loop iterations.

    Shared storage gives Load and Store one slot, so Store reuses Load's
    scratch in the same iteration. Exclusive storage gives them separate
    slices, but each call site reuses its slice on the next iteration. Manual
    mode calls ``storage.sync()`` after each collective to cover both cases.
    In automatic mode the compiler adds these barriers. A different increment
    in each tile exposes stale data left by another iteration.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        tiles: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        storage = cutlass_coop.TempStorage(
            capacity,
            sharing=sharing,
            auto_sync=not manual_sync,
            alignment=1,
        )
        for tile in range(tiles):
            payload = cutlass_coop.ThreadData(items_per_thread)
            cutlass_coop.load(
                cutlass_coop.this_block(),
                source,
                payload,
                algorithm=algorithm,
                offset=tile * _TILE,
                temp_storage=storage,
            )
            if cutlass.const_expr(manual_sync):
                storage.sync()
            for item in cutlass.range_constexpr(items_per_thread):
                payload[item] = payload[item] + cutlass.Int32(tile + 1)
            cutlass_coop.store(
                cutlass_coop.this_block(),
                destination,
                payload,
                algorithm=algorithm,
                offset=tile * _TILE,
                temp_storage=storage,
            )
            if cutlass.const_expr(manual_sync):
                storage.sync()

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
    source = values_for(np.int32, tiles * _TILE, shift=61)
    destination = np.full_like(source, -101)
    expected = (
        source.reshape(tiles, _TILE)
        + np.arange(1, tiles + 1, dtype=np.int32)[:, None]
    )
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, tiles, _ITEMS)
    np.testing.assert_array_equal(destination, expected.reshape(-1))


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("alignment", (1, 32, 64))
@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
def test_requested_storage_alignment_is_a_minimum(api, alignment, sharing):
    """Use scratch even when the caller requests less alignment than it needs.

    Transpose operations on float64 data require aligned storage. Requests of
    one byte must still work because the provider's requirement raises the
    allocation alignment. Larger requests exercise the same copy path.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        storage = api.TempStorage(
            alignment=alignment, sharing=sharing, auto_sync=True
        )
        payload = api.ThreadData(items_per_thread)
        api.load(
            api.this_block(),
            source,
            payload,
            algorithm="transpose",
            temp_storage=storage,
        )
        api.store(
            api.this_block(),
            destination,
            payload,
            algorithm="transpose",
            temp_storage=storage,
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

    source = values_for(np.float64, _TILE, shift=43)
    destination = np.full_like(source, -101)
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, _ITEMS)
    np.testing.assert_array_equal(destination, source)


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("manual_sync", (False, True))
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_storage_example(sharing, manual_sync, items_per_thread):
    path = PACKAGE_ROOT / "examples/cutlass/block_storage.py"
    spec = importlib.util.spec_from_file_location(
        "cutlass_block_storage_example", path
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(
        sharing=sharing,
        manual_sync=manual_sync,
        items_per_thread=items_per_thread,
    )


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
def test_deferred_storage_preserves_user_shared_memory(api, sharing):
    """Keep deferred scratch separate from the caller's shared allocation.

    Each thread writes a canary in the caller's shared allocation before the
    collective calls. It reads another thread's canary only after Load, Store,
    and a final barrier. Incorrect output or canaries expose an overlap
    between deferred scratch and the caller's memory.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        preserved: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        allocator = SmemAllocator()
        canary = cute.make_tensor(
            allocator.allocate_array(
                cutlass.Int32, _THREADS, byte_alignment=16
            ),
            cute.make_layout(_THREADS),
        )
        canary[thread] = 101 + 7 * thread
        cute.arch.sync_threads()

        storage = api.TempStorage(sharing=sharing)
        payload = api.ThreadData(items_per_thread)
        api.load(
            api.this_block(),
            source,
            payload,
            algorithm="transpose",
            temp_storage=storage,
        )
        storage.sync()
        api.store(
            api.this_block(),
            destination,
            payload,
            algorithm="transpose",
            temp_storage=storage,
        )
        cute.arch.sync_threads()
        checks = cute.make_tensor(preserved, cute.make_layout(_THREADS))
        # Read another thread's live allocation after both scratch users.
        checks[thread] = canary[_THREADS - 1 - thread]

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        preserved: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, preserved, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    source = values_for(np.int32, _TILE, shift=79)
    destination = np.full_like(source, -101)
    preserved = np.zeros(_THREADS, dtype=np.int32)
    with (
        device_array(source) as src,
        device_array(destination) as dst,
        device_array(preserved) as check,
    ):
        launch(src, dst, check, _ITEMS)
    np.testing.assert_array_equal(destination, source)
    np.testing.assert_array_equal(
        preserved, 101 + 7 * np.arange(_THREADS - 1, -1, -1, dtype=np.int32)
    )


@pytest.mark.parametrize("algorithm", ("striped", "vectorize", "transpose"))
def test_final_cubin_storage_contract(tmp_path, algorithm):
    """Inspect storage and synchronization in the final linked kernel.

    First verify the copy result. Then check that provider calls have been
    inlined and that only transpose has shared-memory allocation and barriers.
    Generic load and store instructions can address shared memory, so resource
    metadata determines whether an allocation exists. This check does not
    require a particular shared-memory instruction or exact allocation size.
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
        storage = cutlass_coop.TempStorage(
            sharing="shared", alignment=64, auto_sync=True
        )
        payload = cutlass_coop.ThreadData(items_per_thread)
        cutlass_coop.load(
            cutlass_coop.this_block(),
            source,
            payload,
            algorithm=algorithm,
            temp_storage=storage,
        )
        cutlass_coop.store(
            cutlass_coop.this_block(),
            destination,
            payload,
            algorithm=algorithm,
            temp_storage=storage,
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

    source = values_for(np.int32, _TILE, shift=67)
    destination = np.full_like(source, -101)
    with device_array(source) as src, device_array(destination) as dst:
        compiled = cute.compile[(KeepCUBIN, DumpDir(str(tmp_path)))](
            launch, src, dst, _ITEMS
        )
        compiled(src, dst)
    np.testing.assert_array_equal(destination, source)
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins, "CUTLASS did not retain the final linked cubin"
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
        # Generic LD/ST instructions can address the shared-memory window;
        # the final cubin's allocation report is authoritative for capacity.
        shared_sizes = [
            int(size) for size in re.findall(r"\bSHARED:(\d+)", resources)
        ]
        assert shared_sizes
        has_shared = any(shared_sizes)
        has_barrier = re.search(r"\bBAR(?:\.[A-Z0-9_]+)*\b", sass) is not None
        assert has_shared is (algorithm == "transpose")
        assert has_barrier is (algorithm == "transpose")
