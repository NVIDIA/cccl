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

#include <cuda/cmath>
#include <cuda/experimental/__hierarchy/hierarchy_dimensions.cuh>
#include <cuda/experimental/__launch/configuration.cuh>
#include <cuda/experimental/__launch/launch.cuh>
#include <cuda/std/__exception/cuda_error.h>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

template <typename Unit>
struct at_least
{
  size_t cnt;
  using unit = Unit;
  _CCCL_HOST_DEVICE constexpr at_least(size_t count, const Unit& level = Unit())
      : cnt(count)
  {}
};

struct best_occupancy
{};

struct max_coresident
{
  // Super janky until we get a way to query device id from a stream
  const unsigned int device_id;

  _CCCL_HOST_DEVICE constexpr max_coresident(unsigned int dev_id = 0)
      : device_id(dev_id)
  {}
};

namespace detail
{

struct all_levels_supported
{};

template <typename Dims, typename SupportedLevel = all_levels_supported>
struct meta_dimensions_handler : base_dimensions_handler
{
  template <typename Level>
  static constexpr bool is_level_supported =
    ::cuda::std::is_same_v<SupportedLevel, all_levels_supported> || ::cuda::std::is_same_v<SupportedLevel, Level>;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE static constexpr auto translate(const Dims& d) noexcept
  {
    return d;
  }
};

template <typename Level>
struct dimensions_handler<at_least<Level>> : meta_dimensions_handler<at_least<Level>>
{};

template <>
struct dimensions_handler<best_occupancy> : meta_dimensions_handler<best_occupancy, block_level>
{};

template <>
struct dimensions_handler<max_coresident> : meta_dimensions_handler<max_coresident, grid_level>
{};

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

template <typename Dimensions>
struct level_transformer;

template <typename Unit>
struct level_transformer<at_least<Unit>>
{
  template <typename HierarchyBelow>
  _CCCL_NODISCARD meta_dims_transformed
  operator()(void* fn, unsigned int dynamic_smem_bytes, const at_least<Unit>& dims, const HierarchyBelow& rest)
  {
    // Same us creating a hierarchy from rest and calling .count on it
    auto count_of_below_levels =
      detail::dims_to_count(::cuda::std::apply(detail::hierarchy_extents_helper<Unit>{}, rest));
    return meta_dims_transformed(::cuda::ceil_div<const size_t>(dims.cnt, count_of_below_levels));
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

// TODO take green context into account
template <>
struct level_transformer<max_coresident>
{
  template <typename HierarchyBelow>
  _CCCL_NODISCARD meta_dims_transformed
  operator()(void* fn, unsigned int dynamic_smem_bytes, const max_coresident& dims, const HierarchyBelow& rest)
  {
    int num_sms = 0, num_blocks_per_sm = 0;

    // Needs to be fragment for c++17, since hierarchy_dimensions is an alias and CTAD is not supported
    auto tmp_hierarchy = hierarchy_dimensions_fragment(thread, rest);
    auto block_size    = tmp_hierarchy.template count(thread, block);

    cudaError_t status = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dims.device_id);
    if (status != cudaSuccess)
    {
      ::cuda::__throw_cuda_error(status, "Failed to query device attributes");
    }

    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, fn, block_size, dynamic_smem_bytes);
    if (status != cudaSuccess)
    {
      ::cuda::__throw_cuda_error(status, "Failed to query functions maximal occupancy");
    }

    // TODO: should we throw when this is 0?
    return meta_dims_transformed(num_sms * num_blocks_per_sm);
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
  // Needs to be fragment for c++17, since hierarchy_dimensions is an alias and CTAD is not supported
  return hierarchy_dimensions_fragment(
    thread,
    ::cuda::std::apply(
      [&](auto&... levels) {
        return detail::hierarchy_transform_impl(fn, dynamic_smem_bytes, levels...);
      },
      hierarchy.levels));
}
} // namespace detail

// Might consider making the fn optional depending on how many metadims can work without it, right now its only at_least
// TODO not sure if we need hierarchy taking overload
template <typename... Args, typename... Levels>
_CCCL_NODISCARD constexpr auto finalize(const hierarchy_dimensions<Levels...>& hierarchy, void (*fn)(Args...))
{
  return detail::finalize_impl(reinterpret_cast<void*>(fn), 0, hierarchy);
}

template <typename... Args, typename Dimensions, typename... Options>
_CCCL_NODISCARD constexpr auto finalize(const kernel_config<Dimensions, Options...>& config, void (*fn)(Args...))
{
  auto finalized_hierarchy = detail::finalize_impl(reinterpret_cast<void*>(fn), 0, config.dims);
  return kernel_config(finalized_hierarchy, config.options);
}

// Functor overload needs the arguments types to correctly instantiate the launcher
template <typename... Args, typename Kernel, typename ConfOrDims>
_CCCL_NODISCARD constexpr auto finalize(const ConfOrDims& conf_or_dims, const Kernel& kernel)
{
  if constexpr (::cuda::std::is_invocable_v<Kernel, ConfOrDims, Args...>
                || __nv_is_extended_device_lambda_closure_type(Kernel))
  {
    auto launcher = detail::kernel_launcher<ConfOrDims, Kernel, Args...>;
    return finalize(conf_or_dims, launcher);
  }
  else
  {
    static_assert(::cuda::std::is_invocable_v<Kernel, Args...>);
    auto launcher = detail::kernel_launcher_no_config<Kernel, Args...>;
    return finalize(conf_or_dims, launcher);
  }
}

template <typename... Args, typename Kernel, typename ConfOrDims>
_CCCL_NODISCARD constexpr auto finalize(const ConfOrDims& conf_or_dims, const Kernel& kernel, const Args&...)
{
  return finalize<Args...>(conf_or_dims, kernel);
}

} // namespace cuda::experimental
#endif
#endif
