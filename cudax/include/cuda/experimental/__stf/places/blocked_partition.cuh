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
 * @brief Definition of the `blocked_partition` strategy
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

#include <cuda/experimental/__stf/places/cyclic_shape.cuh>
#include <cuda/experimental/__stf/places/places.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>

namespace cuda::experimental::stf
{

template <::std::ptrdiff_t which_dim = -1>
class blocked_partition_custom
{
public:
  blocked_partition_custom() = default;

  template <size_t dimensions>
  _CCCL_HOST_DEVICE static auto apply(const box<dimensions>& in, pos4 place_position, dim4 grid_dims)
  {
    ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, dimensions> bounds;
    size_t target_dim = (which_dim == -1) ? dimensions - 1 : size_t(which_dim);
    if (target_dim > dimensions - 1)
    {
      target_dim = dimensions - 1;
    }

    //        if constexpr (dimensions > 1) {
    for (size_t d = 0; d < dimensions; d++)
    {
      // First position in this dimension (included)
      bounds[d].first = in.get_begin(d);
      // Last position in this dimension (excluded)
      bounds[d].second = in.get_end(d);
    }
    //        }

    size_t nplaces             = grid_dims.x;
    ::std::ptrdiff_t dim_beg   = bounds[target_dim].first;
    ::std::ptrdiff_t dim_end   = bounds[target_dim].second;
    size_t cnt                 = dim_end - dim_beg;
    ::std::ptrdiff_t part_size = (cnt + nplaces - 1) / nplaces;

    // If first = second, this means it's an empty shape. This may happen
    // when there are more entries in grid_dims than in the shape for
    // example
    bounds[target_dim].first  = ::std::min(dim_beg + part_size * place_position.x, dim_end);
    bounds[target_dim].second = ::std::min(dim_beg + part_size * (place_position.x + 1), dim_end);

    return box(bounds);
  }

  template <typename mdspan_shape_t>
  _CCCL_HOST_DEVICE static auto apply(const mdspan_shape_t& in, pos4 place_position, dim4 grid_dims)
  {
    constexpr size_t dimensions = mdspan_shape_t::rank();

    ::std::array<::std::pair<::std::ptrdiff_t, ::std::ptrdiff_t>, dimensions> bounds;
    for (size_t d = 0; d < dimensions; d++)
    {
      bounds[d].first  = 0;
      bounds[d].second = in.extent(d);
    }

    // Last position in this dimension (excluded)
    size_t target_dim = (which_dim == -1) ? dimensions - 1 : size_t(which_dim);
    if (target_dim > dimensions - 1)
    {
      target_dim = dimensions - 1;
    }

    ::std::ptrdiff_t dim_end = in.extent(target_dim);

    // The last dimension is split accross the different places
    size_t nplaces            = grid_dims.x;
    size_t part_size          = (in.extent(target_dim) + nplaces - 1) / nplaces;
    bounds[target_dim].first  = ::std::min((::std::ptrdiff_t) part_size * place_position.x, dim_end);
    bounds[target_dim].second = ::std::min((::std::ptrdiff_t) part_size * (place_position.x + 1), dim_end);

    return box<dimensions>(bounds);
  }

  _CCCL_HOST_DEVICE static pos4 get_executor(pos4 data_coords, dim4 data_dims, dim4 grid_dims)
  {
    // Find the largest dimension
    size_t rank       = data_dims.get_rank();
    size_t target_dim = (which_dim == -1) ? rank : size_t(which_dim);
    if (target_dim > rank)
    {
      target_dim = rank;
    }

    size_t extent = data_dims.get(target_dim);

    size_t nplaces   = grid_dims.x;
    size_t part_size = (extent + nplaces - 1) / nplaces;

    // Get the coordinate in the selected dimension
    size_t c = data_coords.get(target_dim);

    return pos4(c / part_size);
  }
};

using blocked_partition = blocked_partition_custom<>;

} // namespace cuda::experimental::stf
