# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Debug scratch planning, reuse barriers, and automatic dynamic shared memory.

Start with --case dynamic; use --case all to verify every example.
See debugger_walkthrough_temp_storage.rst for the one-hour review route.
"""

import argparse
import os
from pathlib import Path

# Import the kernel DSL before cuda.coop so automatic registration sees it.
# isort: off
import numpy as np
from numba_cuda_mlir import cuda

from cuda import coop
# isort: on


BLOCK_THREADS = 128
BLOCKS = 2
TILES_PER_BLOCK = 2


@cuda.jit
def independent_primitives(source, ordered, prefixes, items_per_thread):
    block = coop.this_block()
    scan_scratch = coop.TempStorage(auto_sync=False)
    sort_scratch = coop.TempStorage(auto_sync=False)
    items = coop.ThreadData(items_per_thread)
    for tile in range(TILES_PER_BLOCK):
        offset = (
            (cuda.blockIdx.x * TILES_PER_BLOCK + tile)
            * cuda.blockDim.x
            * items_per_thread
        )
        coop.load(block, source, items, offset=offset, algorithm="direct")
        scanned = coop.exclusive_sum(block, items, temp_storage=scan_scratch)
        sorted_items = coop.merge_sort_keys(block, items, temp_storage=sort_scratch)
        coop.store(block, prefixes, scanned, offset=offset, algorithm="direct")
        coop.store(block, ordered, sorted_items, offset=offset, algorithm="direct")
        # Separate descriptors cannot alias each other, but each call site
        # reuses its own slice on the next loop iteration.
        block.sync()


@cuda.jit
def shared_auto(source, ordered, items_per_thread):
    block = coop.this_block()
    scratch = coop.TempStorage(auto_sync=True)
    items = coop.ThreadData(items_per_thread)
    for tile in range(TILES_PER_BLOCK):
        offset = (
            (cuda.blockIdx.x * TILES_PER_BLOCK + tile)
            * cuda.blockDim.x
            * items_per_thread
        )
        coop.load(
            block,
            source,
            items,
            offset=offset,
            algorithm="transpose",
            temp_storage=scratch,
        )
        sorted_items = coop.merge_sort_keys(block, items, temp_storage=scratch)
        coop.store(
            block,
            ordered,
            sorted_items,
            offset=offset,
            algorithm="transpose",
            temp_storage=scratch,
        )
        # Automatic trailing barriers protect all three transitions,
        # including Store -> Load on the next loop iteration.


@cuda.jit
def shared_manual(source, ordered, items_per_thread):
    block = coop.this_block()
    scratch = coop.TempStorage(auto_sync=False)
    items = coop.ThreadData(items_per_thread)
    for tile in range(TILES_PER_BLOCK):
        offset = (
            (cuda.blockIdx.x * TILES_PER_BLOCK + tile)
            * cuda.blockDim.x
            * items_per_thread
        )
        coop.load(
            block,
            source,
            items,
            offset=offset,
            algorithm="transpose",
            temp_storage=scratch,
        )
        block.sync()  # Load has finished using the shared slice.
        sorted_items = coop.merge_sort_keys(block, items, temp_storage=scratch)
        block.sync()  # Sort has finished using the same slice.
        coop.store(
            block,
            ordered,
            sorted_items,
            offset=offset,
            algorithm="transpose",
            temp_storage=scratch,
        )
        block.sync()  # Store -> next iteration's Load is also reuse.


@cuda.jit
def exclusive(source, ordered, items_per_thread):
    block = coop.this_block()
    scratch = coop.TempStorage(sharing="exclusive", auto_sync=False)
    items = coop.ThreadData(items_per_thread)
    for tile in range(TILES_PER_BLOCK):
        offset = (
            (cuda.blockIdx.x * TILES_PER_BLOCK + tile)
            * cuda.blockDim.x
            * items_per_thread
        )
        coop.load(
            block,
            source,
            items,
            offset=offset,
            algorithm="transpose",
            temp_storage=scratch,
        )
        sorted_items = coop.merge_sort_keys(block, items, temp_storage=scratch)
        coop.store(
            block,
            ordered,
            sorted_items,
            offset=offset,
            algorithm="transpose",
            temp_storage=scratch,
        )
        # Three distinct call-site slices remove cross-call reuse barriers.
        # Exclusive does not allocate fresh storage on each loop iteration.
        block.sync()


def run_case(name, items_per_thread):
    kernel = {
        "independent": independent_primitives,
        "shared-auto": shared_auto,
        "shared-manual": shared_manual,
        "exclusive": exclusive,
    }[name]
    tile_items = BLOCK_THREADS * items_per_thread
    values = np.random.default_rng(42).integers(
        -1000, 1000, size=BLOCKS * TILES_PER_BLOCK * tile_items, dtype=np.int64
    )
    source = cuda.to_device(values)
    ordered = cuda.device_array_like(source)
    args = [source, ordered]
    if name == "independent":
        prefixes = cuda.device_array_like(source)
        args.append(prefixes)
    args.append(items_per_thread)
    print(
        f"{name}: {BLOCK_THREADS} threads, {items_per_thread} items/thread",
        flush=True,
    )

    # First host breakpoint: step into planning on this ordinary launch.
    # No fourth launch argument supplies dynamic bytes; the compiler does it.
    kernel[BLOCKS, BLOCK_THREADS](*args)
    cuda.synchronize()
    tiles = values.reshape(BLOCKS * TILES_PER_BLOCK, tile_items)
    np.testing.assert_array_equal(
        ordered.copy_to_host(), np.sort(tiles, axis=1).ravel()
    )
    if name == "independent":
        np.testing.assert_array_equal(
            prefixes.copy_to_host(),
            (np.cumsum(tiles, axis=1) - tiles).ravel(),
        )

    # Debugger-only inspection of the specialization compiled by this launch.
    # These compiler internals are not part of the cuda.coop public API.
    compiled = next(
        result
        for (argtypes, _), result in kernel.launch_config_overloads.items()
        if argtypes[-1].literal_value == items_per_thread
    )
    dynamic_bytes = compiled.metadata.get("required_dynamic_shared_memory", 0)
    print(
        f"  verified all tiles; required dynamic shared memory: {dynamic_bytes} bytes"
    )
    return dynamic_bytes


def run_dynamic():
    device = cuda.get_current_device()
    default_limit = device.MAX_SHARED_MEMORY_PER_BLOCK
    optin_limit = device.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
    print(
        f"{device.name}: default shared memory {default_limit} bytes, "
        f"opt-in {optin_limit} bytes",
        flush=True,
    )

    # All three CUB primitives have roughly one tile of int64 scratch.
    # Pick a tile whose three exclusive slices exceed the default limit,
    # while the shared version fits. Actual CUB sizes include padding:
    # inspect the plan, and verify the compiler's decision below.
    items_per_thread = (
        default_limit // (3 * BLOCK_THREADS * np.dtype(np.int64).itemsize) + 1
    )
    if optin_limit <= default_limit:
        raise RuntimeError(
            "This device has no opt-in shared-memory headroom for the dynamic case."
        )
    print(
        "Dynamic comparison: exclusive first, then shared at the same tile size.",
        flush=True,
    )
    dynamic_bytes = run_case("exclusive", items_per_thread)
    assert default_limit < dynamic_bytes <= optin_limit, (
        f"Expected dynamic backing within device limits, got {dynamic_bytes} bytes"
    )
    shared_bytes = run_case("shared-auto", items_per_thread)
    assert shared_bytes == 0, "Expected the shared version to fit static memory"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        default="dynamic",
        choices=(
            "dynamic",
            "independent",
            "shared-auto",
            "shared-manual",
            "exclusive",
            "all",
        ),
    )
    parser.add_argument(
        "--items-per-thread",
        type=int,
        choices=(1, 4),
        default=4,
        help="Payload size for small cases; dynamic chooses its own device-sized tile.",
    )
    args = parser.parse_args()
    print(f"cuda.coop source: {coop.__file__}", flush=True)
    root = os.environ.get("CUDA_COOP_CCCL_ROOT")
    if root and not Path(coop.__file__).resolve().is_relative_to(
        Path(root).resolve() / "python/cuda_coop"
    ):
        raise RuntimeError(
            "cuda.coop was imported from another checkout. Select an interpreter "
            "without an editable cuda-coop install pointing at that checkout."
        )
    if args.case in ("dynamic", "all"):
        run_dynamic()
    if args.case == "all":
        for name in (
            "independent",
            "shared-auto",
            "shared-manual",
            "exclusive",
        ):
            run_case(name, args.items_per_thread)
    elif args.case != "dynamic":
        run_case(args.case, args.items_per_thread)


if __name__ == "__main__":
    main()
