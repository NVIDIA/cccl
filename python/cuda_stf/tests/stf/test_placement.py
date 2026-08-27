# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Tests for structured partitions (cute_partition), placement evaluation, and
geometry-aware (shaped) allocation on composite data places.

The Python contract is C order throughout: shapes, per-dimension
specifications, callback coordinates, and grid axes all use axis 0 as the
outermost dimension, and leaf lists are last-leaf-fastest. The C/C++ layers
remain dimension-0-fastest; the tests below use non-square shapes so an
accidental order reversal at the boundary is visible.
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


def _require_vmm():
    """Shaped composite allocations build a localized array, which needs CUDA
    virtual address management. Skip rather than fail where it is absent (the
    C test at c/experimental/stf/test/test_placement.cpp gates the same way)."""
    from cuda.bindings import driver as cu

    assert cu.cuInit(0)[0] == cu.CUresult.CUDA_SUCCESS
    err, dev = cu.cuDeviceGet(0)
    assert err == cu.CUresult.CUDA_SUCCESS
    err, supported = cu.cuDeviceGetAttribute(
        cu.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED,
        dev,
    )
    assert err == cu.CUresult.CUDA_SUCCESS
    if not supported:
        pytest.skip("device 0 does not support CUDA VMM (virtual address management)")


def blocked_mapper_1d(data_coords, data_dims, grid_dims):
    """Blocked partition along the outermost dimension (C-order contract)."""
    n = data_dims[0]
    nplaces = grid_dims[0]
    part_size = max((n + nplaces - 1) // nplaces, 1)
    return min(data_coords[0] // part_size, nplaces - 1)


def test_cute_partition_from_spec():
    """Builder output: rank-aware C-order dims, padding, leaves, offsets
    (no GPU needed)."""
    part = stf.cute_partition.from_spec((10,), (("blocked", 0),), (3,))
    assert part.rank == 1
    assert part.grid_rank == 1
    assert part.true_dims == (10,)
    assert part.padded_dims == (12,)  # ceil(10/3) * 3
    assert part.grid_dims == (3,)
    assert part.place_leaves == [(3, 4, 0)]
    assert part.local_leaves == [(4, 1)]
    assert [part.place_offset(i) for i in range(3)] == [0, 4, 8]
    assert [part.grid_place_offset(i) for i in range(3)] == [0, 4, 8]
    with pytest.raises(ValueError, match="out of range"):
        part.place_offset(-1)
    with pytest.raises(ValueError, match="out of range"):
        part.place_offset(3)
    with pytest.raises(TypeError, match="integer"):
        part.place_offset(0.5)

    # 3-D non-cubic tensor, middle dimension blocked. In C order, (8, 6, 4)
    # has the extent-4 axis contiguous: each place owns 3 slabs of 4 elements.
    part3 = stf.cute_partition.from_spec((8, 6, 4), (None, ("blocked", 0), None), (2,))
    assert part3.rank == 3
    assert part3.true_dims == (8, 6, 4)
    assert part3.padded_dims == (8, 6, 4)  # 6 divides evenly over 2 places
    assert part3.place_leaves == [(2, 12, 0)]  # stride = 3 slabs * 4 elements
    # Last leaf fastest: the contiguous extent-4 axis comes last
    assert part3.local_leaves == [(8, 24), (3, 4), (4, 1)]


def test_cute_partition_blocked_per_axis():
    """Public axis 0 and axis 1 of a non-square 2-D tensor blocked
    independently give visibly different layouts."""
    rows = stf.cute_partition.from_spec((6, 4), (("blocked", 0), None), (3,))
    assert rows.true_dims == (6, 4)
    assert rows.padded_dims == (6, 4)
    assert rows.place_leaves == [(3, 8, 0)]  # 2 rows of 4 per place
    assert rows.local_leaves == [(2, 4), (4, 1)]
    assert [rows.place_offset(i) for i in range(3)] == [0, 8, 16]

    cols = stf.cute_partition.from_spec((6, 4), (None, ("blocked", 0)), (2,))
    assert cols.true_dims == (6, 4)
    assert cols.padded_dims == (6, 4)
    assert cols.place_leaves == [(2, 2, 0)]  # 2 columns of each row
    assert cols.local_leaves == [(6, 4), (2, 1)]
    assert [cols.place_offset(i) for i in range(2)] == [0, 2]


def test_cute_partition_swapped_grid_axes():
    """A non-square 2-D grid with tensor dimensions mapped to grid axes in
    swapped order: the grid-linear offset differs from the place-mode offset,
    which pins the difference between grid_place_offset() and place_offset().
    """
    # Tensor (4, 6): axis 0 (extent 4) distributes over grid axis 1 (extent
    # 3, padded 4 -> 6), axis 1 (extent 6) over grid axis 0 (extent 2).
    part = stf.cute_partition.from_spec(
        (4, 6), (("blocked", 1), ("blocked", 0)), (2, 3)
    )
    assert part.rank == 2
    assert part.grid_rank == 2
    assert part.true_dims == (4, 6)
    assert part.padded_dims == (6, 6)  # 4 over 3 places pads to 6
    assert part.grid_dims == (2, 3)
    assert part.place_leaves == [(3, 12, 1), (2, 3, 0)]
    assert part.local_leaves == [(2, 6), (3, 1)]

    # C-order grid enumeration: index i -> coords (i // 3, i % 3), and the
    # owned block starts at axis0_coord * 3 + axis1_coord * 12.
    expected = [g0 * 3 + g1 * 12 for g0 in range(2) for g1 in range(3)]
    assert [part.grid_place_offset(i) for i in range(6)] == expected

    # Place-mode order enumerates the place leaves themselves (last leaf
    # fastest in the public reading), which differs from grid order here.
    assert part.place_offset(1) != part.grid_place_offset(1)


def test_cute_partition_owner():
    """Closed-form element ownership follows the C-order contract."""
    part = stf.cute_partition.from_spec((10,), (("blocked", 0),), (3,))
    assert [part.owner((i,)) for i in (0, 3, 4, 7, 8, 9)] == [
        (0,),
        (0,),
        (1,),
        (1,),
        (2,),
        (2,),
    ]

    # Swapped tensor/grid axes: axis 0 (extent 4, chunk 2) owns grid axis 1,
    # axis 1 (extent 6, chunk 3) owns grid axis 0.
    part2 = stf.cute_partition.from_spec(
        (4, 6), (("blocked", 1), ("blocked", 0)), (2, 3)
    )
    for i in range(4):
        for j in range(6):
            assert part2.owner((i, j)) == (j // 3, i // 2)

    with pytest.raises(ValueError):
        part2.owner((0,))  # rank mismatch
    with pytest.raises(ValueError, match="owner query failed"):
        part2.owner((6, 0))  # first coordinate is outside the padded extents


def test_tensor_of_tiles_owner_ignores_payload():
    """Ownership of a tensor-of-tiles element depends only on the tile
    coordinates: the payload dims are undistributed."""
    tiles, payload = (2, 3), (4, 8)
    tile_part = stf.cute_partition.from_spec(
        tiles, (("blocked", 0), ("blocked", 1)), (2, 3)
    )
    data_part = stf.cute_partition.from_spec(
        tiles + payload, (("blocked", 0), ("blocked", 1), None, None), (2, 3)
    )
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            expected = tile_part.owner((i, j))
            for y, x in ((0, 0), (payload[0] - 1, payload[1] - 1), (1, 5)):
                assert data_part.owner((i, j, y, x)) == expected


def test_cute_partition_rank_preserved_with_extent_one():
    """An active extent-1 dimension is legitimate and must not be trimmed."""
    part = stf.cute_partition.from_spec((5, 1), (("blocked", 0), None), (5,))
    assert part.rank == 2
    assert part.true_dims == (5, 1)
    assert part.padded_dims == (5, 1)
    assert part.grid_dims == (5,)


def test_cute_partition_from_leaves_roundtrip():
    """from_spec -> public leaves -> from_leaves reproduces the partition,
    including on a swapped-axis 2-D layout."""
    for build in (
        lambda: stf.cute_partition.from_spec((16,), (("block_cyclic", 0, 2),), (2,)),
        lambda: stf.cute_partition.from_spec(
            (4, 6), (("blocked", 1), ("blocked", 0)), (2, 3)
        ),
    ):
        part = build()
        rebuilt = stf.cute_partition.from_leaves(
            part.place_leaves,
            part.local_leaves,
            part.padded_dims,
            part.true_dims,
            part.grid_dims,
        )
        assert rebuilt.true_dims == part.true_dims
        assert rebuilt.padded_dims == part.padded_dims
        assert rebuilt.grid_dims == part.grid_dims
        assert rebuilt.place_leaves == part.place_leaves
        assert rebuilt.local_leaves == part.local_leaves
        nplaces = 1
        for e in part.grid_dims:
            nplaces *= e
        for i in range(nplaces):
            assert rebuilt.grid_place_offset(i) == part.grid_place_offset(i)

    # Non-exact leaves are rejected
    with pytest.raises(ValueError):
        stf.cute_partition.from_leaves([(2, 1, 0)], [(4, 1)], (8,), (8,), (2,))


def test_cute_partition_uneven_padding():
    """Uneven extents pad up to divisibility; padding is per-dimension."""
    part = stf.cute_partition.from_spec((10, 3), (("blocked", 0), None), (4,))
    assert part.true_dims == (10, 3)
    assert part.padded_dims == (12, 3)
    assert [part.place_offset(i) for i in range(4)] == [0, 9, 18, 27]


def test_tensor_of_tiles_from_spec():
    """The tensor-of-tiles data partition is the tile partition's spec plus
    trailing whole payload dimensions: ownership per tile is unchanged and
    the payload becomes dense local leaves (no new API needed)."""
    tiles = (2, 3)
    payload = (4, 8)
    tile_part = stf.cute_partition.from_spec(
        tiles, (("blocked", 0), ("blocked", 1)), (2, 3)
    )
    data_part = stf.cute_partition.from_spec(
        tiles + payload,
        (("blocked", 0), ("blocked", 1), None, None),
        (2, 3),
    )
    assert data_part.rank == 4
    assert data_part.true_dims == (2, 3, 4, 8)
    assert data_part.padded_dims == (2, 3, 4, 8)
    assert data_part.grid_dims == tile_part.grid_dims

    payload_size = payload[0] * payload[1]
    # Place leaves are the tile partition's, scaled by the payload size
    assert data_part.place_leaves == [
        (e, s * payload_size, a) for (e, s, a) in tile_part.place_leaves
    ]
    # Existing local leaves scale by the payload size (a tile-index step now
    # jumps a whole payload) and the payload appends compact row-major leaves
    assert data_part.local_leaves == [
        (e, s * payload_size) for (e, s) in tile_part.local_leaves
    ] + [
        (payload[0], payload[1]),
        (payload[1], 1),
    ]
    # Tile ownership is unchanged: same grid offsets, scaled by the payload
    nplaces = 6
    for i in range(nplaces):
        assert (
            data_part.grid_place_offset(i)
            == tile_part.grid_place_offset(i) * payload_size
        )


def test_placement_evaluate_all_mapper_forms():
    """Native fn pointer, cute partition and Python callable must agree."""
    _require_device()
    stf.machine_init()
    grid = stf.exec_place_grid.from_devices([0, 0])

    n = 4 * MiB
    kwargs = dict(elemsize=1, block_size=2 * MiB)

    s_native = stf.placement_evaluate(grid, stf.partition_fn_blocked(), (n,), **kwargs)
    assert s_native.nblocks == 2
    assert s_native.nallocs == 2
    assert s_native.accuracy == 1.0  # block-aligned split
    assert s_native.bytes_per_grid_index == [2 * MiB, 2 * MiB]

    part = stf.cute_partition.from_spec((n,), (("blocked", 0),), (2,))
    s_part = stf.placement_evaluate(grid, part, None, **kwargs)
    assert s_part.bytes_per_grid_index == s_native.bytes_per_grid_index

    s_callable = stf.placement_evaluate(grid, blocked_mapper_1d, (n,), **kwargs)
    assert s_callable.bytes_per_grid_index == s_native.bytes_per_grid_index


def test_placement_evaluate_misaligned_majority():
    """An ownership boundary inside a placement block exercises the sampled
    majority vote: the straddling block goes to the owner of most of it and
    accuracy drops below 1. The rank-2 row mapper also pins the C-order
    contract at rank > 1: with transposed coordinates the owner index would
    be a column index, far outside the grid, and the callback's range check
    would fail the evaluation loudly."""
    _require_device()
    stf.machine_init()
    grid = stf.exec_place_grid.from_devices([0, 0])

    rows, cols = 4, 768 * 1024  # 3 MiB of 1-byte elements
    block = 2 * MiB

    def row_blocked(data_coords, data_dims, grid_dims):
        # Rows 0-1 to place 0, rows 2-3 to place 1 (public axis 0 = rows)
        return (data_coords[0] // 2,)

    s = stf.placement_evaluate(
        grid, row_blocked, (rows, cols), elemsize=1, probes=100, block_size=block
    )
    # Two blocks: [0, 2 MiB) straddles the 1.5 MiB ownership boundary (3/4 of
    # it belongs to place 0), [2 MiB, 3 MiB) is entirely place 1. Blocks are
    # attributed at full block granularity.
    assert s.total_bytes == 3 * MiB
    assert s.vm_bytes == 4 * MiB
    assert s.nblocks == 2
    assert s.nallocs == 2
    assert s.bytes_per_grid_index == [2 * MiB, 2 * MiB]
    # The straddled block's majority holds at least half of its samples and
    # the aligned block matches fully, so accuracy lands in [0.75, 1). The
    # exact value depends on the standard library's distribution mapping, so
    # only the bounds are asserted; the upper bound is strict up to the
    # ~0.75**100 chance that every probe of the straddled block lands on one
    # side of the boundary.
    assert s.total_samples == 200
    assert 0.75 <= s.accuracy < 1.0


def test_partition_fn_returns_typed_wrapper():
    """Native partitioners are typed (not bare ints) so composite() can tell
    them apart from Python callables; int() still exposes the raw pointer."""
    fn = stf.partition_fn_blocked()
    assert isinstance(fn, stf.native_partition_fn)
    assert int(fn) != 0
    assert isinstance(stf.partition_fn_cyclic(), stf.native_partition_fn)
    assert int(stf.partition_fn_blocked(1, data_rank=2)) != 0


def test_shaped_allocation_on_composite_places():
    _require_device()
    _require_vmm()
    stf.machine_init()
    grid = stf.exec_place_grid.from_devices([0, 0])

    n = MiB  # ints

    dp = stf.data_place.composite(grid, stf.partition_fn_blocked())
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


def test_shaped_allocation_c_order_extents():
    """composite_cute allocation takes C-order extents: the partition's
    non-square public shape allocates, its transpose is rejected."""
    _require_device()
    _require_vmm()
    stf.machine_init()
    grid = stf.exec_place_grid.from_devices([0, 0])

    shape = (2, MiB // 2)  # non-square: rows of 512 KiB ints
    part = stf.cute_partition.from_spec(shape, (("blocked", 0), None), (2,))
    dpc = stf.data_place.composite_cute(grid, part)
    with pytest.raises(MemoryError):
        dpc.allocate(shape[::-1], elemsize=4)
    ptr = dpc.allocate(shape, elemsize=4)
    assert ptr != 0
    dpc.deallocate(ptr, shape[0] * shape[1] * 4)


def test_tensor_of_tiles_allocation():
    """A rank-4 tensor-of-tiles partition allocates through composite_cute
    with C-order extents (repeated device 0: functional, not residency)."""
    _require_device()
    _require_vmm()
    stf.machine_init()
    grid = stf.exec_place_grid.create([stf.exec_place.device(0)] * 4, grid_dims=(2, 2))

    tiles = (2, 2)
    tile = (512, 256)  # 512 KiB per tile at elemsize 4
    shape = tiles + tile
    part = stf.cute_partition.from_spec(
        shape, (("blocked", 0), ("blocked", 1), None, None), (2, 2)
    )
    dpc = stf.data_place.composite_cute(grid, part)
    nbytes = tiles[0] * tiles[1] * tile[0] * tile[1] * 4
    ptr = dpc.allocate(shape, elemsize=4)
    assert ptr != 0
    dpc.deallocate(ptr, nbytes)


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
    dp = stf.data_place.composite(grid, stf.partition_fn_blocked())
    ptr = dp.allocate((n,), elemsize=4)
    try:
        for half in range(2):
            # CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL resolves the whole VA
            # reservation, not the per-offset backing: query the retained
            # allocation handle instead, which names the physical owner.
            err, handle = cu.cuMemRetainAllocationHandle(ptr + half * granularity)
            assert err == cu.CUresult.CUDA_SUCCESS
            try:
                err, hprop = cu.cuMemGetAllocationPropertiesFromHandle(handle)
                assert err == cu.CUresult.CUDA_SUCCESS
                assert int(hprop.location.id) == half, (
                    "block is not resident on the place that owns it"
                )
            finally:
                assert cu.cuMemRelease(handle)[0] == cu.CUresult.CUDA_SUCCESS
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

    # neither is a bare int: native partitioners are typed (native_partition_fn)
    with pytest.raises(TypeError):
        stf.placement_evaluate(grid, 123, (1024,), 1)

    # bool is not a partition
    with pytest.raises(TypeError):
        stf.data_place.composite(grid, True)

    # A Python mapper is shape-free: its data rank must be explicit
    with pytest.raises(ValueError, match="data_rank"):
        stf.data_place.composite(grid, blocked_mapper_1d)
    with pytest.raises(ValueError, match="data_rank"):
        stf.partition_fn_blocked(1)

    # elemsize 0 (would be a division by zero)
    with pytest.raises(RuntimeError):
        stf.placement_evaluate(grid, stf.partition_fn_blocked(), (1024,), 0)

    # A raising mapper must surface, not silently yield wrong statistics
    def bad_mapper(coords, dims, gdims):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="mapper raised"):
        stf.placement_evaluate(grid, bad_mapper, (4 * MiB,), 1, block_size=2 * MiB)

    # A mapper returning the wrong number of grid coordinates must surface
    def wrong_rank_mapper(coords, dims, gdims):
        return (0, 0)

    with pytest.raises(RuntimeError, match="mapper raised"):
        stf.placement_evaluate(
            grid, wrong_rank_mapper, (4 * MiB,), 1, block_size=2 * MiB
        )

    # Conversely a native partitioner never takes one: rejecting it loudly
    # keeps data_rank's meaning unambiguous (Python-callable tuples only)
    with pytest.raises(ValueError, match="data_rank"):
        stf.data_place.composite(grid, stf.partition_fn_blocked(), data_rank=1)

    # Zero-extent grid axis
    with pytest.raises(ValueError):
        stf.cute_partition.from_spec((8,), (("blocked", 0),), (0,))

    # A grid axis bound to no tensor dimension leaves places unused
    with pytest.raises(ValueError):
        stf.cute_partition.from_spec((8,), (("blocked", 0),), (2, 3))

    # One spec entry per dimension, in the same C order
    with pytest.raises(ValueError, match="one entry per dimension"):
        stf.cute_partition.from_spec((8, 4), (("blocked", 0),), (2,))

    # Grid axes are validated against the grid rank
    with pytest.raises(ValueError, match="grid axis"):
        stf.cute_partition.from_spec((8,), (("blocked", 1),), (2,))

    # Rank overflow
    with pytest.raises(ValueError):
        stf.cute_partition.from_spec(
            (2, 2, 2, 2, 2), (None, None, None, None, None), (2,)
        )

    # Partition grid must match the execution grid
    part = stf.cute_partition.from_spec((4 * MiB,), (("blocked", 0),), (3,))
    with pytest.raises(RuntimeError):
        stf.placement_evaluate(grid, part, None, 1)

    # elemsize is extents-form-only
    dp = stf.data_place.device(0)
    with pytest.raises(ValueError):
        dp.allocate(100, elemsize=4)
