# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for structured partitions (cute_partition), placement evaluation, and
geometry-aware (shaped) allocation on composite data places.
"""

import pytest

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402

MiB = 1024 * 1024


def _require_device():
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        pytest.skip("cuda-bindings is not available")
    err, count = cudart.cudaGetDeviceCount()
    if err != cudart.cudaError_t.cudaSuccess or count == 0:
        pytest.skip("no usable CUDA device")


def blocked_mapper_1d(data_coords, data_dims, grid_dims):
    """Blocked partition along first dimension (Python-callable form)."""
    n = data_dims[0]
    nplaces = grid_dims[0]
    part_size = max((n + nplaces - 1) // nplaces, 1)
    place_x = min(data_coords[0] // part_size, nplaces - 1)
    return (place_x, 0, 0, 0)


def test_cute_partition_from_spec():
    """Builder output: padding, leaves, offsets (no GPU needed)."""
    part = stf.cute_partition.from_spec((10,), (("blocked", 0),), (3,))
    assert part.true_dims == (10, 1, 1, 1)
    assert part.padded_dims == (12, 1, 1, 1)  # ceil(10/3) * 3
    assert part.grid_dims == (3, 1, 1, 1)
    assert part.place_leaves == [(3, 4, 0)]
    assert part.local_leaves == [(4, 1)]
    assert [part.place_offset(i) for i in range(3)] == [0, 4, 8]

    # 3-D tensor, dimension 1 blocked (dimension-0-fastest convention)
    part3 = stf.cute_partition.from_spec((8, 6, 4), (None, ("blocked", 0), None), (2,))
    assert part3.padded_dims == (8, 6, 4, 1)
    assert part3.place_leaves == [(2, 24, 0)]  # P=2, stride b*R1 = 3*8


def test_cute_partition_from_leaves_roundtrip():
    part = stf.cute_partition.from_spec((16,), (("block_cyclic", 0, 2),), (2,))
    rebuilt = stf.cute_partition.from_leaves(
        part.place_leaves,
        part.local_leaves,
        part.padded_dims,
        part.true_dims,
        part.grid_dims,
    )
    assert rebuilt.place_offset(1) == part.place_offset(1)

    # Non-exact leaves are rejected
    with pytest.raises(ValueError):
        stf.cute_partition.from_leaves([(2, 1, 0)], [(4, 1)], (8,), (8,), (2,))


def test_placement_evaluate_all_mapper_forms():
    """Native fn pointer, cute partition and Python callable must agree."""
    _require_device()
    stf.machine_init()
    grid = stf.exec_place_grid.from_devices([0, 0])

    n = 4 * MiB
    kwargs = dict(elemsize=1, block_size=2 * MiB)

    s_native = stf.placement_evaluate(grid, stf.partition_fn_blocked(0), (n,), **kwargs)
    assert s_native.nblocks == 2
    assert s_native.nallocs == 2
    assert s_native.accuracy == 1.0  # block-aligned split
    assert s_native.bytes_per_grid_index == [2 * MiB, 2 * MiB]

    part = stf.cute_partition.from_spec((n,), (("blocked", 0),), (2,))
    s_part = stf.placement_evaluate(grid, part, None, **kwargs)
    assert s_part.bytes_per_grid_index == s_native.bytes_per_grid_index

    s_callable = stf.placement_evaluate(grid, blocked_mapper_1d, (n,), **kwargs)
    assert s_callable.bytes_per_grid_index == s_native.bytes_per_grid_index


def test_placement_evaluate_majority_tie_breaking():
    """A block straddling two owners goes to the majority owner and the
    accuracy reflects the straddling (seeded: deterministic across calls)."""
    _require_device()
    stf.machine_init()
    grid = stf.exec_place_grid.from_devices([0, 0])

    # 2.5 MiB blocked over 2 places: block 0 straddles 62.5%/37.5%, so the
    # checks hold structurally for any uniform sampler (all-64-probes-majority
    # has probability ~1e-13)
    n = 5 * MiB // 2
    s1 = stf.placement_evaluate(
        grid, stf.partition_fn_blocked(0), (n,), 1, probes=64, block_size=2 * MiB
    )
    assert s1.nallocs == 2
    assert 0.7 < s1.accuracy < 1.0

    s2 = stf.placement_evaluate(
        grid, stf.partition_fn_blocked(0), (n,), 1, probes=64, block_size=2 * MiB
    )
    assert s1.matching_samples == s2.matching_samples
    assert s1.bytes_per_grid_index == s2.bytes_per_grid_index


def test_shaped_allocation_on_composite_places():
    _require_device()
    stf.machine_init()
    grid = stf.exec_place_grid.from_devices([0, 0])

    n = MiB  # ints

    dp = stf.data_place.composite(grid, stf.partition_fn_blocked(0))
    # A byte count alone cannot carry the tensor geometry
    with pytest.raises(MemoryError):
        dp.allocate(n * 4)
    ptr = dp.allocate((n,), elemsize=4)
    assert ptr != 0
    dp.deallocate(ptr, n * 4)

    part = stf.cute_partition.from_spec((n,), (("blocked", 0),), (2,))
    dpc = stf.data_place.composite_cute(grid, part)
    # Extents other than the partition's true extents are rejected
    with pytest.raises(MemoryError):
        dpc.allocate((n // 2,), elemsize=4)
    ptr2 = dpc.allocate((n,), elemsize=4)
    assert ptr2 != 0
    dpc.deallocate(ptr2, n * 4)


def test_multi_gpu_residency():
    """With 2+ devices, each half of a blocked allocation must be physically
    resident on its owner (the real check runs in multi-GPU CI)."""
    _require_device()
    from cuda.bindings import driver as cu
    from cuda.bindings import runtime as cudart

    err, count = cudart.cudaGetDeviceCount()
    if err != cudart.cudaError_t.cudaSuccess or count < 2:
        pytest.skip("requires 2+ CUDA devices")

    stf.machine_init()
    grid = stf.exec_place_grid.from_devices([0, 1])

    # One placement block per device, at whatever granularity this device uses
    prop = cu.CUmemAllocationProp()
    prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = 0
    err, granularity = cu.cuMemGetAllocationGranularity(
        prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    )
    assert err == cu.CUresult.CUDA_SUCCESS

    n = 2 * granularity // 4  # ints
    dp = stf.data_place.composite(grid, stf.partition_fn_blocked(0))
    ptr = dp.allocate((n,), elemsize=4)
    try:
        for half in range(2):
            err, ordinal = cu.cuPointerGetAttribute(
                cu.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                ptr + half * granularity,
            )
            assert err == cu.CUresult.CUDA_SUCCESS
            assert int(ordinal) == half, (
                "block is not resident on the place that owns it"
            )
    finally:
        dp.deallocate(ptr, n * 4)


def test_invalid_inputs_raise_cleanly():
    """Misuse must raise Python exceptions, never crash the interpreter."""
    _require_device()
    stf.machine_init()
    grid = stf.exec_place_grid.from_devices([0, 0])

    # Direct instantiation (NULL handle)
    with pytest.raises(TypeError):
        stf.cute_partition()

    # bool is not a function pointer
    with pytest.raises(TypeError):
        stf.placement_evaluate(grid, True, (1024,), 1)
    with pytest.raises(TypeError):
        stf.data_place.composite(grid, True)

    # elemsize 0 (would be a division by zero)
    with pytest.raises(RuntimeError):
        stf.placement_evaluate(grid, stf.partition_fn_blocked(0), (1024,), 0)

    # A raising mapper must surface, not silently yield wrong statistics
    def bad_mapper(coords, dims, gdims):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="mapper raised"):
        stf.placement_evaluate(grid, bad_mapper, (4 * MiB,), 1, block_size=2 * MiB)

    # Zero-extent grid axis
    with pytest.raises(ValueError):
        stf.cute_partition.from_spec((8,), (("blocked", 0),), (0,))

    # Partition grid must match the execution grid
    part = stf.cute_partition.from_spec((4 * MiB,), (("blocked", 0),), (3,))
    with pytest.raises(RuntimeError):
        stf.placement_evaluate(grid, part, None, 1)

    # elemsize is extents-form-only
    dp = stf.data_place.device(0)
    with pytest.raises(ValueError):
        dp.allocate(100, elemsize=4)
