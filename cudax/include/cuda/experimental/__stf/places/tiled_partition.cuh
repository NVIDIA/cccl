//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Definition of the `tiled_partition` strategy
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/places/places.cuh>

namespace cuda::experimental::stf
{

namespace reserved
{

/*
 * Define a tiled transformation to a shape of mdspan
 */
template <size_t tile_size, typename mdspan_shape_t>
class tiled_mdspan_shape
{
public:
  // parts id, nparts ?
  tiled_mdspan_shape(const mdspan_shape_t& s, size_t part_id, size_t nparts)
      : original_shape(s)
      , part_id(part_id)
      , nparts(nparts)
  {}

  /* The number of elements in this part */
  size_t size() const
  {
    _CCCL_ASSERT(mdspan_shape_t::rank() == 1, "Tiled mdspan shape only implemented in 1D yet");

    const size_t n = original_shape.size();

    // 0000 1111 2222 0000 11xx xxxx xxxx
    // S=4, n=18
    //
    // nparts=3
    // ntiles = (18+3)/4 = 5
    // tile_per_part = (5+2)/3 = 2 (1 or 2 tiles per part)
    // cnt = 2*4 = 8
    // last_elem = {0 => (0+((2-1)*3)+1)*8=(1+3)*8=16} (n >= last_elem)        => cnt = 8
    //             {1 => (1+((2-1)*3)+1)*8=(2+3)*8=20} extra = min(4, 20-18)=2 => cnt = 6
    //             {2 => (2+((2-1)*3)+1)*8=(3+3)*8=24} extra = min(4, 24-18)=4 => cnt = 4

    // How many tiles if we round up to a multiple of tile_size ?
    const size_t ntiles = (n + tile_size - 1) / tile_size;

    const size_t tile_per_part = (ntiles + nparts - 1) / nparts;

    // If all parts are the same
    size_t cnt = tile_per_part * tile_size;

    // Assuming all parts have tile_per_part tiles, the last tile of the
    // part starts the last tile has index (part_id + ((tile_per_part-1)*
    // nparts)), and ends at the beginning of the next tile (hence +1).
    const size_t last_elem = (part_id + (tile_per_part - 1) * nparts + 1) * tile_size;

    // Remove extra elements (if any) by computing what would be the last
    // element for this part
    if (last_elem > n)
    {
      size_t extra_elems = min(tile_size, last_elem - n);
      cnt -= extra_elems;
    }

    return cnt;
  }

  using coords_t = typename mdspan_shape_t::coords_t;

  _CCCL_HOST_DEVICE coords_t index_to_coords(size_t index) const
  {
    // First transform from nparts and part_id to one coordinate
    const size_t remain  = index % tile_size;
    const size_t tile_id = index / tile_size;
    // Stage 2: apply original shape's transformation to stage1
    return original_shape.index_to_coords((part_id + tile_id * nparts) * tile_size + remain);
  }

private:
  mdspan_shape_t original_shape;
  size_t part_id;
  size_t nparts;
};

} // end namespace reserved

/**
 * @brief Tiled partition strategy applied on a shape of mdspan
 *
 * @tparam `tile_size` size of the tiles
 * @tparam `mdspan_shape_t` shape of a mdspan
 *
 * Since there is no partial template deduction on classes, we provide a
 * function to implement tiled<tile_size> and have the other type deduced.
 */
template <size_t tile_size, typename mdspan_shape_t>
auto tiled(const mdspan_shape_t s, size_t part_id, size_t nparts)
{
  return reserved::tiled_mdspan_shape<tile_size, mdspan_shape_t>(s, part_id, nparts);
}

template <size_t tile_size>
class tiled_partition
{
  static_assert(tile_size > 0);

public:
  tiled_partition() = default;

  template <typename mdspan_shape_t>
  static const reserved::tiled_mdspan_shape<tile_size, mdspan_shape_t>
  apply(const mdspan_shape_t& in, pos4 place_position, dim4 grid_dims)
  {
    // TODO assert 1D !
    assert(grid_dims.x > 0);
    return reserved::tiled_mdspan_shape<tile_size, mdspan_shape_t>(in, place_position.x, grid_dims.x);
  }

  _CCCL_HOST_DEVICE static pos4 get_executor(pos4 data_coords, dim4 /*unused*/, dim4 grid_dims)
  {
    assert(grid_dims.x > 0);
    return pos4((data_coords.x / tile_size) % grid_dims.x);
  }
};

#ifdef UNITTESTED_FILE
UNITTEST("Composite data place equality")
{
  auto all           = exec_place::all_devices();
  auto repeated_dev0 = exec_place::repeat(exec_place::device(0), 3);

  using P  = tiled_partition<128>;
  using P2 = tiled_partition<64>;

  /* Same partitioning operator, same execution place */
  EXPECT(data_place::composite(P(), all) == data_place::composite(P(), all));

  /* Make sure we do not have a false positive in the test below */
  EXPECT(all != repeated_dev0);

  /* Same partitioning operator, different execution place */
  EXPECT(data_place::composite(P(), repeated_dev0) != data_place::composite(P(), all));

  /* Different partitioning operator, same execution place */
  EXPECT(data_place::composite(P(), all) != data_place::composite(P2(), all));
};

UNITTEST("tiled partition with large 1D data")
{
  const size_t large_1d_size = (1ULL << 36);

  // Test coordinate that makes tiling calculation obvious
  const size_t test_coord = (1ULL << 34);
  pos4 large_coords(test_coord); // 1D coordinate
  dim4 data_dims(large_1d_size); // 1D data space
  dim4 grid_dims(32); // 32 places in grid

  constexpr size_t tile_size = 1000;

  pos4 tile_pos = tiled_partition<tile_size>::get_executor(large_coords, data_dims, grid_dims);

  EXPECT(tile_pos.x == (test_coord / tile_size) % grid_dims.x);
};

#endif // UNITTESTED_FILE

} // namespace cuda::experimental::stf
