//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__LAUNCH_HIERARCHY_DRAFT
#define _CUDAX__LAUNCH_HIERARCHY_DRAFT

#include <cuda/cmath>
#include <cuda/std/__exception/cuda_error.h>
#include <cuda/stream_ref>

#include <cuda/experimental/__hierarchy/hierarchy_dimensions.cuh>
#include <cuda/experimental/__launch/configuration.cuh>
#include <cuda/experimental/__launch/kernel_launchers.cuh>
#include <cuda/experimental/__launch/launch_transform.cuh>
#include <cuda/experimental/__utility/ensure_current_device.cuh>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental
{

/**
 * @brief Describe dimensions of a level with a specific unit, that might be deeper down the hierarchy.
 *
 * Levels inside hierarchy_dimensions type have their dimensions described in terms of a unit
 * that is the next level below in the hierarchy. Sometimes, it would be more convinient
 * to describe dimensions of a level using a specific unit that appears in the hierarchy,
 * but it might now be the immediate level below. This type allows to request that the
 * count of the specified unit on this level will be equal or higher than the specified value.
 * The reason it sometimes might be higher is that it needs to be divisible by the count
 * of that unit of the levels below, so it sometimes need to be rounded up.
 * An common example where this is useful is scaling grid dimensions to the problem size.
 * In that case there is a specific number of threads requested, but the block
 * size needs to be configured to some specific value. It can be expressed like this:
 *
 * @par Snippet
 * @code
 * #include <cudax/launch.cuh>
 *
 * auto hierarchy = make_hierarchy(block_dims<256>(), grid_dims(at_least(problem_size, thread)));
 * assert(hierarchy.count(thread, grid) >= problem_size);
 * @endcode
 * @par
 *
 * @tparam Unit
 * The unit requested for the count of this level to be expressed in
 *
 * @param count
 * The count request for this level expressed in Unit
 *
 */
template <typename Unit>
struct at_least
{
  size_t cnt;
  using unit = Unit;
  _CCCL_HOST_DEVICE constexpr at_least(size_t count, const Unit& = Unit())
      : cnt(count)
  {}
};

/**
 * @brief Request the size of this level to be picked automatically in a way that tries to maximize the occupancy.
 *
 * When used to describe a level CUDA will calculate the projected occupancy for a given function and pick
 * a largest size that still allows full occupancy.
 * This type is usable only to describe dimensions at block level
 */
struct max_occupancy
{};

/**
 * @brief Request the size of this level to be the maximal number that still allows parallel execution.
 *
 * When used to descrive a level CUDA will calculate the projected occupancy for a given function and pick
 * a largest size that still allows all below levels to run concurrently
 * This type is usable only to describe dimensions at grid level.
 *
 * The most common use case for this type is a cooperative grid that occupies the entire device:
 * @par Snippet
 * @code
 * #include <cudax/launch.cuh>
 *
 * auto hierarchy = make_hierarchy(block_dims<256>(), grid_dims(max_coresident()));
 * auto config = make_config(hierarchy, launch_cooperative);
 * // Configuration will launch a cooperative grid that occupies the entire device
 * @endcode
 * @par
 */
struct max_coresident
{};

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
struct dimensions_handler<max_occupancy> : meta_dimensions_handler<max_occupancy, block_level>
{};

template <>
struct dimensions_handler<max_coresident> : meta_dimensions_handler<max_coresident, grid_level>
{};

using meta_dims_finalized = dimensions<dimensions_index_type, ::cuda::std::dynamic_extent, 1, 1>;

template <typename Dimensions>
struct level_finalizer;

template <typename Unit>
struct level_finalizer<at_least<Unit>>
{
  template <typename HierarchyBelow>
  _CCCL_NODISCARD meta_dims_finalized
  operator()(void* fn, unsigned int dynamic_smem_bytes, const at_least<Unit>& dims, const HierarchyBelow& rest)
  {
    auto tmp_hierarchy         = hierarchy_dimensions_fragment(thread, rest);
    auto count_of_below_levels = tmp_hierarchy.count(Unit());
    return meta_dims_finalized(::cuda::ceil_div<const size_t>(dims.cnt, count_of_below_levels));
  }
};

template <>
struct level_finalizer<max_occupancy>
{
  template <typename HierarchyBelow>
  _CCCL_NODISCARD meta_dims_finalized
  operator()(void* fn, unsigned int dynamic_smem_bytes, const max_occupancy& dims, const HierarchyBelow& rest)
  {
    int block_size, dummy;

    cudaError_t status = cudaOccupancyMaxPotentialBlockSize(&dummy, &block_size, fn, dynamic_smem_bytes);
    if (status != cudaSuccess)
    {
      ::cuda::__throw_cuda_error(status, "Failed to query optimal block size");
    }

    // TODO if there ever is a level below block_level, we need to divide here (and round down?)
    return meta_dims_finalized(block_size);
  }
};

// TODO take green context into account
template <>
struct level_finalizer<max_coresident>
{
  template <typename HierarchyBelow>
  _CCCL_NODISCARD meta_dims_finalized
  operator()(void* fn, unsigned int dynamic_smem_bytes, const max_coresident& dims, const HierarchyBelow& rest)
  {
    int num_sms = 0, num_blocks_per_sm = 0;
    int device;

    // Needs to be fragment for c++17, since hierarchy_dimensions is an alias and CTAD is not supported
    auto tmp_hierarchy = hierarchy_dimensions_fragment(thread, rest);
    auto block_size    = tmp_hierarchy.count(thread, block);

    // We might have cluster or other levels below the grid, needs to count them to properly divide this levels count
    // TODO: there might be some consideration of clusters fitting on the device?
    auto blocks_multiplier = tmp_hierarchy.count(block);

    // Device will be properly set outside of this function
    _CCCL_TRY_CUDA_API(cudaGetDevice, "Could not get device", &device);
    cudaError_t status = cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
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
    return meta_dims_finalized(num_sms * num_blocks_per_sm / blocks_multiplier);
  }
};

_CCCL_NODISCARD constexpr auto hierarchy_finalize_impl(void*, unsigned int)
{
  return ::cuda::std::make_tuple();
}

template <typename L1, typename... Rest>
_CCCL_NODISCARD constexpr auto
hierarchy_finalize_impl(void* fn, unsigned int dynamic_smem_bytes, const L1& level, const Rest&... rest)
{
  auto rest_finalized = hierarchy_finalize_impl(fn, dynamic_smem_bytes, rest...);

  using dims_type = ::cuda::std::decay_t<decltype(level.dims)>;

  if constexpr (!detail::usable_for_queries<dims_type>)
  {
    auto finalizer = level_finalizer<dims_type>();
    auto new_dims  = finalizer(fn, dynamic_smem_bytes, level.dims, rest_finalized);
    return ::cuda::std::tuple_cat(::cuda::std::make_tuple(level.finalize(new_dims)), rest_finalized);
  }
  else
  {
    return ::cuda::std::tuple_cat(::cuda::std::make_tuple(level), rest_finalized);
  }
}

template <typename... Levels>
_CCCL_NODISCARD constexpr auto
finalize_impl(void* fn, unsigned int dynamic_smem_bytes, const hierarchy_dimensions<Levels...>& hierarchy)
{
  // Needs to be a fragment for c++17, since hierarchy_dimensions is an alias and CTAD is not supported
  return hierarchy_dimensions_fragment(
    thread,
    ::cuda::std::apply(
      [&](auto&... levels) {
        return detail::hierarchy_finalize_impl(fn, dynamic_smem_bytes, levels...);
      },
      hierarchy.levels));
}

template <typename... Args, typename... Levels>
_CCCL_NODISCARD constexpr auto finalize_no_device_set(
  ::cuda::stream_ref stream, const hierarchy_dimensions<Levels...>& hierarchy, void (*kernel)(Args...))
{
  return detail::finalize_impl(reinterpret_cast<void*>(kernel), 0, hierarchy);
}

template <typename... Args, typename Dimensions, typename... Options>
_CCCL_NODISCARD constexpr auto finalize_no_device_set(
  ::cuda::stream_ref stream, const kernel_config<Dimensions, Options...>& config, void (*kernel)(Args...))
{
  size_t smem_size = 0;
  auto dyn_smem    = detail::find_option_in_tuple<detail::launch_option_kind::dynamic_shared_memory>(config.options);
  if constexpr (!::cuda::std::is_same_v<decltype(dyn_smem), detail::option_not_found>)
  {
    smem_size = dyn_smem.size_bytes();
  }

  auto finalized_hierarchy =
    detail::finalize_impl(reinterpret_cast<void*>(kernel), static_cast<unsigned int>(smem_size), config.dims);
  return kernel_config(finalized_hierarchy, config.options);
}

// Assumes all meta dims are finalized into 1-d dynamic extent (seems like a safe assumption at least for now)
// TODO should this be in the main namespace
template <typename Level>
using finalized_level = ::cuda::std::conditional_t<
  detail::usable_for_queries<typename Level::dimensions_type>,
  Level,
  decltype(::cuda::std::declval<Level>().finalize(::cuda::std::declval<detail::meta_dims_finalized>()))>;
} // namespace detail

template <typename>
struct finalized;

template <typename... Levels>
struct finalized<hierarchy_dimensions<Levels...>>
{
  using type = hierarchy_dimensions<detail::finalized_level<Levels>...>;
};

template <typename Dimensions, typename... Options>
struct finalized<kernel_config<Dimensions, Options...>>
{
  using type = kernel_config<typename finalized<Dimensions>::type, Options...>;
};

/**
 * @brief Transform a hierarchy or configuration type into one that finalize would return
 *
 * @tparam T
 * Either hierarchy_dimensions or kernel_config type
 */
template <typename T>
using finalized_t = typename finalized<T>::type;

// Might consider making the kernel optional depending on how many metadims can work without it, right now its only
// at_least
// TODO not sure if we need hierarchy taking overload
// TODO should finalize test if finalization is needed and be just identitiy if not?
// TODO figure out how to add checks that if a hierarchy was finalized and later launched, then the same function was
// used in both places
/**
 * @brief Returns a hierarchy updated to replace meta dimensions with concrete dimensions
 *
 * This function will create a new hierarchy_dimensions with the same levels as the input hierarchy.
 * For each level, its going to replace the dimenions with concrete values if they are meta dimensions or just
 * pass them through otherwise. This function needs to take the stream and the function, because the finalization
 * of some meta dimensions require informations about the device or the kernel.
 *
 * @param stream
 * Stream that is going to be used in the launch function to launch kernel with these dimensions
 *
 * @param hierarchy
 * Hierarchy that is going to be finalized
 *
 * @param kernel
 * Kernel function that the dimensions are intended for
 */
template <typename... Args, typename... Levels>
_CCCL_NODISCARD constexpr auto
finalize(::cuda::stream_ref stream, const hierarchy_dimensions<Levels...>& hierarchy, void (*kernel)(Args...))
{
  __ensure_current_device __dev_setter(stream);
  return detail::finalize_no_device_set(stream, hierarchy, kernel);
}

/**
 * @brief Returns a configuration with the hierarchy updated to replace meta dimensions with concrete dimensions
 *
 * This function will create a new kernel_configuration with the same options and the hierarchy as the input
 * configuration. For each level in the hierarchy in the config, finalize is going to replace the dimenions with
 * concrete values if they are meta dimensions or just pass them through otherwise. This function needs to take the
 * stream and the function, because the finalization of some meta dimensions require informations about the device or
 * the kernel.
 *
 * @param stream
 * Stream that is going to be used in the launch function to launch kernel with this configuration
 *
 * @param config
 * Configuration that is going to be finalized
 *
 * @param kernel
 * Kernel function that the configuration are intended for
 */
template <typename... Args, typename Dimensions, typename... Options>
_CCCL_NODISCARD constexpr auto
finalize(::cuda::stream_ref stream, const kernel_config<Dimensions, Options...>& config, void (*kernel)(Args...))
{
  __ensure_current_device __dev_setter(stream);
  return detail::finalize_no_device_set(stream, config, kernel);
}

// Functor overload needs the arguments types to correctly instantiate the launcher
/**
 * @brief Returns a hierarchy or a configuration with the hierarchy updated to replace meta dimsnions with concrete
 * dimensions
 *
 * This function will create a new kernel_configuration or hierarchy_dimensions based on what is passed in. It will be
 * finalized for the passed in kernel functor. Each level in the hierarchy or the hierarchy contained in the
 * configuration will be replaced with concrete dimensions if they are meta dimensions or passed through otherwise. This
 * function needs to take the stream and the function, because the finalization of some meta dimensions require
 * informations about the device or the kernel. Compared to the function pointer overloads, this version also needs to
 * know the types of arguments of the kernel functor, which are taken in the first template argument pack.
 *
 * @tparam Args
 * Types of arguments of the kernel functor
 *
 * @param stream
 * Stream that is going to be used in the launch function to launch kernel with this configuration or dimensions
 *
 * @param conf_or_dims
 * Configuration or dimensions that are going to be finalized
 *
 * @param kernel
 * Kernel functor that the configuration are intended for
 */

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4180) // qualifier applied to function type has no meaning; ignored
template <typename... Args,
          typename Kernel,
          typename ConfOrDims,
          typename = ::cuda::std::enable_if_t<!::cuda::std::is_function_v<std::remove_pointer_t<Kernel>>>>
_CCCL_NODISCARD constexpr auto finalize(::cuda::stream_ref stream, const ConfOrDims& conf_or_dims, const Kernel&)
{
  return finalize(
    stream, conf_or_dims, detail::get_kernel_launcher<Kernel, finalized_t<ConfOrDims>, as_kernel_arg_t<Args>...>());
}
_CCCL_DIAG_POP

/**
 * @brief Returns a hierarchy or a configuration with the hierarchy updated to replace meta dimsnions with concrete
 * dimensions
 *
 * This function will create a new kernel_configuration or hierarchy_dimensions based on what is passed in. It will be
 * finalized for the passed in kernel functor. Each level in the hierarchy or the hierarchy contained in the
 * configuration will be replaced with concrete dimensions if they are meta dimensions or passed through otherwise. This
 * function needs to take the stream and the function, because the finalization of some meta dimensions require
 * informations about the device or the kernel. Compared to the function pointer overloads, this version also needs to
 * know the types of arguments of the kernel functor, which are taken as the last arguments pack
 *
 * @param stream
 * Stream that is going to be used in the launch function to launch kernel with this configuration or dimensions
 *
 * @param conf_or_dims
 * Configuration or dimensions that are going to be finalized
 *
 * @param kernel
 * Kernel functor that the configuration are intended for
 *
 * @param args
 * Arguments that the kernel functor will be launched with
 */
template <typename... Args, typename Kernel, typename ConfOrDims>
_CCCL_NODISCARD constexpr auto finalize(
  ::cuda::stream_ref stream, const ConfOrDims& conf_or_dims, const Kernel& kernel, [[maybe_unused]] const Args&... args)
{
  return finalize<Args...>(stream, conf_or_dims, kernel);
}

} // namespace cuda::experimental
#endif
#endif
