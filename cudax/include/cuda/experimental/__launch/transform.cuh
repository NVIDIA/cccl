//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_TRANSFORM
#define _CUDAX__LAUNCH_TRANSFORM

#include <cuda/experimental/__hierarchy/hierarchy_dimensions.cuh>
#include <cuda/experimental/__launch/configuration.cuh>
#include <cuda/std/__exception/cuda_error.h>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

// Kind of a hack to fit with rest of the code (for now? ;)
template <typename Unit>
struct at_least // : dimensions<unsigned int, ::cuda::std::dynamic_extent, 1, 1>
{
  size_t cnt;
  using unit = Unit;
  _CCCL_HOST_DEVICE constexpr at_least(size_t count, const Unit& level = Unit())
      : cnt(count)
  //: dimensions<unsigned int, ::cuda::std::dynamic_extent, 1, 1>(count)
  {}
};

struct best_occupancy
{};

namespace detail
{

template <typename Dims>
struct meta_dimensions_handler
{
  static constexpr bool is_type_supported = true;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE static constexpr auto translate(const Dims& d) noexcept
  {
    return d;
  }
};

template <typename Level>
struct dimensions_handler<at_least<Level>> : meta_dimensions_handler<at_least<Level>>
{};

template <>
struct dimensions_handler<best_occupancy> : meta_dimensions_handler<best_occupancy>
{};

template <typename What, typename ByWhat>
auto ceil_div(What what, ByWhat by_what)
{
  return (what + by_what - 1) / by_what;
}

using meta_dims_transformed = dimensions<dimensions_index_type, ::cuda::std::dynamic_extent, 1, 1>;

// Assumes all meta dims are transformed into 1-d dynamic extent (seems like a safe assumption at least for now)
template <typename Level>
using transformed_level = ::cuda::std::conditional_t<
  detail::usable_for_queries<typename Level::dimensions_type>,
  decltype(::cuda::std::declval<Level>().transform(::cuda::std::declval<meta_dims_transformed>())),
  Level>;

template <typename Hierarchy>
struct transformed_hierarchy;

template <typename... Levels>
struct transformed_hierarchy<hierarchy_dimensions<Levels...>>
{
  using type = hierarchy_dimensions<transformed_level<Levels>...>;
};

/*
template <typename Level, typename RestTransformed>
_CCCL_NODISCARD constexpr auto level_transform(const Level& level, const RestTransformed& rest)
{}
*/

template <typename T>
struct check;

template <typename Dimensions>
struct level_transformer;

template <typename Unit>
struct level_transformer<at_least<Unit>>
{
  template <typename HierarchyBelow>
  _CCCL_NODISCARD meta_dims_transformed
  operator()(void* fn, unsigned int dynamic_smem_bytes, const at_least<Unit>& dims, const HierarchyBelow& rest)
  {
    auto tmp_hierarchy         = hierarchy_dimensions(rest);
    auto count_of_below_levels = tmp_hierarchy.template count<Unit>();
    return meta_dims_transformed(ceil_div(dims.cnt, count_of_below_levels));
  }
};

template <>
struct level_transformer<best_occupancy>
{
  template <typename HierarchyBelow>
  _CCCL_NODISCARD meta_dims_transformed
  operator()(void* fn, unsigned int dynamic_smem_bytes, const best_occupancy& dims, const HierarchyBelow& rest)
  {
    int block_size, dummy;

    cudaError_t status = cudaOccupancyMaxPotentialBlockSize(&dummy, &block_size, fn, dynamic_smem_bytes);
    if (status != cudaSuccess)
    {
      ::cuda::__throw_cuda_error(status, "Failed to query optimal block size");
    }
    return meta_dims_transformed(block_size);
  }
};

_CCCL_NODISCARD constexpr auto hierarchy_transform_impl(void* fn, unsigned int dynamic_smem_bytes)
{
  return ::cuda::std::make_tuple();
}

template <typename L1, typename... Rest>
_CCCL_NODISCARD constexpr auto
hierarchy_transform_impl(void* fn, unsigned int dynamic_smem_bytes, const L1& level, const Rest&... rest)
{
  auto rest_transformed = hierarchy_transform_impl(fn, dynamic_smem_bytes, rest...);

  using dims_type = ::cuda::std::decay_t<decltype(level.dims)>;

  if constexpr (!detail::usable_for_queries<dims_type>)
  {
    auto transformer = level_transformer<dims_type>();
    auto new_dims    = transformer(fn, dynamic_smem_bytes, level.dims, rest_transformed);
    return ::cuda::std::tuple_cat(::cuda::std::make_tuple(level.transform(new_dims)), rest_transformed);
  }
  else
  {
    return ::cuda::std::tuple_cat(::cuda::std::make_tuple(level), rest_transformed);
  }
}

template <typename... Levels>
_CCCL_NODISCARD constexpr auto
finalize_impl(void* fn, unsigned int dynamic_smem_bytes, const hierarchy_dimensions<Levels...>& hierarchy)
{
  return hierarchy_dimensions(::cuda::std::apply(
    [&](auto&... levels) {
      return detail::hierarchy_transform_impl(fn, dynamic_smem_bytes, levels...);
    },
    hierarchy.levels));
}
} // namespace detail

// Might consider making the fn optional depending on how many metadims can work without it, right now its only at_least
template <typename Fn, typename... Levels>
_CCCL_NODISCARD constexpr auto finalize(const hierarchy_dimensions<Levels...>& hierarchy, const Fn& fn)
{
  return detail::finalize_impl(reinterpret_cast<void*>(fn), 0, hierarchy);
}

template <typename Dimensions, typename... Options, typename Fn>
_CCCL_NODISCARD constexpr auto finalize(const kernel_config<Dimensions, Options...>& config, const Fn& fn)
{
  auto hierachy = detail::finalize_impl(reinterpret_cast<void*>(fn), 0, config.dims);
  return kernel_config(config, config.options);
}

} // namespace cuda::experimental
#endif
#endif
