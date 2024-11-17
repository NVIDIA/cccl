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

//! @brief This type can be used to opt-in to rounding up the dimensions of a __level
template <typename _CntType>
struct at_least
{
  at_least(_CntType __cnt)
      : __value(__cnt)
  {}

  _CntType __value;
};

template <typename _Unit>
struct __target_count
{
  size_t __count;
  bool __ceil_div;

  template <typename _CntType>
  __target_count(_CntType __cnt)
      : __count(__cnt)
      , __ceil_div(false)
  {
    static_assert(::cuda::std::is_integral_v<_CntType>);
  }

  template <typename _CntType>
  __target_count(at_least<_CntType> __cnt)
      : __count(__cnt.__value)
      , __ceil_div(true)
  {
    static_assert(::cuda::std::is_integral_v<_CntType>);
  }
};

struct __max_occupancy
{};

struct __device_fill
{
  float __fill_coef;
};

// All finalized levels should have the same type, 1d dynamic extent
using __meta_dims_finalized = dimensions<dimensions_index_type, ::cuda::std::dynamic_extent, 1, 1>;

template <typename _Unit, typename _HierarchyBelow>
_CCCL_NODISCARD __meta_dims_finalized __level_finalize(
  [[maybe_unused]] void* __fn,
  [[maybe_unused]] unsigned int __dynamic_smem_bytes,
  const __target_count<_Unit>& __dims,
  const _HierarchyBelow& __rest)
{
  auto __tmp_hierarchy         = hierarchy_dimensions_fragment(thread, __rest);
  auto __count_of_levels_below = __tmp_hierarchy.count(_Unit());
  if (__dims.__ceil_div)
  {
    return __meta_dims_finalized(::cuda::ceil_div<const size_t>(__dims.__count, __count_of_levels_below));
  }
  else
  {
    if (__dims.__count % __count_of_levels_below != 0)
    {
      ::cuda::std::__throw_invalid_argument(
        "Count for this __level is not a multiple of below __level, use at_least "
        "type to allow rounding up");
    }
    return __meta_dims_finalized(__dims.__count / __count_of_levels_below);
  }
}

template <typename _HierarchyBelow>
_CCCL_NODISCARD __meta_dims_finalized __level_finalize(
  void* __fn,
  unsigned int __dynamic_smem_bytes,
  [[maybe_unused]] const __max_occupancy& __dims,
  [[maybe_unused]] const _HierarchyBelow& __rest)
{
  int __block_size, __dummy;

  _CCCL_TRY_CUDA_API(
    cudaOccupancyMaxPotentialBlockSize,
    "Failed to query optimal block size",
    &__dummy,
    &__block_size,
    __fn,
    __dynamic_smem_bytes);

  // TODO if there ever is a level below block_level, we need to divide here (and round down?)
  // for now just static assert to make it not supported
  static_assert(::cuda::std::tuple_size_v<_HierarchyBelow> == 0);

  return __meta_dims_finalized(__block_size);
}

template <typename _HierarchyBelow>
_CCCL_NODISCARD __meta_dims_finalized __level_finalize(
  void* __fn,
  unsigned int __dynamic_smem_bytes,
  [[maybe_unused]] const __device_fill& __dims,
  const _HierarchyBelow& __rest)
{
  int __num_sms = 0, __num_blocks_per_sm = 0;
  int __device;

  // Needs to be fragment for c++17, since hierarchy_dimensions is an alias and CTAD is not supported
  auto __tmp_hierarchy = hierarchy_dimensions_fragment(thread, __rest);
  auto __block_size    = __tmp_hierarchy.count(thread, block);

  // We might have cluster or other levels below the grid, needs to count them to properly divide this levels count
  // TODO: there might be some consideration of clusters fitting on the device?
  auto __block_multiplier = __tmp_hierarchy.count(block);

  // Device will be properly set outside of this function
  _CCCL_TRY_CUDA_API(cudaGetDevice, "Could not get __device", &__device);
  _CCCL_TRY_CUDA_API(
    cudaDeviceGetAttribute, "Failed to query __device attributes", &__num_sms, cudaDevAttrMultiProcessorCount, __device);

  _CCCL_TRY_CUDA_API(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    "Failed to query functions maximal occupancy",
    &__num_blocks_per_sm,
    __fn,
    __block_size,
    __dynamic_smem_bytes);

  int __cnt = static_cast<int>((__num_sms * __num_blocks_per_sm / __block_multiplier * __dims.__fill_coef));
  if (__cnt == 0)
  {
    _CUDA_VSTD::__throw_invalid_argument("Not able to run any blocks with this configuration");
  }
  return __meta_dims_finalized(__cnt);
}

_CCCL_NODISCARD constexpr auto __hierarchy_finalize_impl(void*, unsigned int)
{
  return ::cuda::std::make_tuple();
}

template <typename _L1, typename... _Rest>
_CCCL_NODISCARD auto
__hierarchy_finalize_impl(void* __fn, unsigned int __dynamic_smem_bytes, const _L1& __level, const _Rest&... __rest)
{
  auto __rest_finalized = __hierarchy_finalize_impl(__fn, __dynamic_smem_bytes, __rest...);

  using __dims_type = ::cuda::std::decay_t<decltype(__level.dims)>;

  if constexpr (!detail::usable_for_queries<__dims_type>)
  {
    auto __new_dims = __level_finalize(__fn, __dynamic_smem_bytes, __level.dims, __rest_finalized);
    return ::cuda::std::tuple_cat(::cuda::std::make_tuple(__level.finalize(__new_dims)), __rest_finalized);
  }
  else
  {
    return ::cuda::std::tuple_cat(::cuda::std::make_tuple(__level), __rest_finalized);
  }
}

template <typename _Unit, typename... _Levels>
_CCCL_NODISCARD auto __finalize_impl(
  void* __fn, unsigned int __dynamic_smem_bytes, const hierarchy_dimensions_fragment<_Unit, _Levels...>& __hierarchy)
{
  // Needs to be a fragment for c++17, since hierarchy_dimensions is an alias and CTAD is not supported
  return hierarchy_dimensions_fragment(
    _Unit(),
    ::cuda::std::apply(
      [&](auto&... __levels) {
        return __hierarchy_finalize_impl(__fn, __dynamic_smem_bytes, __levels...);
      },
      __hierarchy.levels));
}

template <typename... _Args, typename _Unit, typename... _Levels>
_CCCL_NODISCARD auto __finalize_no_device_set(const hierarchy_dimensions_fragment<_Unit, _Levels...>& __hierarchy,
                                              void (*__kernel)(_Args...))
{
  return __finalize_impl(reinterpret_cast<void*>(__kernel), 0, __hierarchy);
}

template <typename... _Args, typename _Dimensions, typename... _Options>
_CCCL_NODISCARD auto
__finalize_no_device_set(const kernel_config<_Dimensions, _Options...>& __config, void (*__kernel)(_Args...))
{
  size_t __smem_size = 0;
  auto __dyn_smem = detail::find_option_in_tuple<detail::launch_option_kind::dynamic_shared_memory>(__config.options);
  if constexpr (!::cuda::std::is_same_v<decltype(__dyn_smem), detail::option_not_found>)
  {
    __smem_size = __dyn_smem.size_bytes();
  }

  auto __finalized_hierarchy =
    __finalize_impl(reinterpret_cast<void*>(__kernel), static_cast<unsigned int>(__smem_size), __config.dims);
  return kernel_config(__finalized_hierarchy, __config.options);
}

// Assumes all meta dims are finalized into 1-d dynamic extent (seems like a safe assumption at least for now)
template <typename _Level>
using __finalized_level = ::cuda::std::conditional_t<
  detail::usable_for_queries<typename _Level::dimensions_type>,
  _Level,
  decltype(::cuda::std::declval<_Level>().finalize(::cuda::std::declval<__meta_dims_finalized>()))>;

/**
 * @brief Block level with size automatically picked to maximize occupancy
 *
 * When used to launch a kernel or finalized the kernel and device will be
 * used to determine block size that maximizes the occupancy.
 * Selected block size is not guaranteed to achieve best performance,
 * but it is a good estimate without in-depth analysis or profiling.
 */
_CCCL_NODISCARD inline constexpr auto auto_block_dims()
{
  return level_dimensions<block_level, __max_occupancy>{};
}

/**
 * @brief Grid level that contains enough blocks to fill the device up to the supplied coefficient
 *
 * When used to launch a kernel or finalized the supplied kernel and device will be used to determine
 * how many blocks can fit on the entire __device. Then the supplied coefficient will be used to scale
 * that size. For example 0.5 will fill half of the device, 1.0 will fill entire device. Values above
 * 1.0 are supported if the launch is done without `cooperative_launch()` option. In that case some blocks
 * are not going to start execution until some of the previous blocks finish to free execution resources.
 *
 * The most common use case for this type is a cooperative grid that occupies the entire device:
 * @par Snippet
 * @code
 * #include <cudax/launch.cuh>
 *
 * // Configuration will launch a cooperative grid that occupies the entire device
 * auto cofig = make_config(block_dims<256>, fill_device(1.0), launch_cooperative());
 * @endcode
 * @par
 *
 * @param __coef Coefficient used to scale the number of blocks relative to full __device utilisation
 */
_CCCL_NODISCARD inline constexpr auto fill_device(float __coef = 1.0f)
{
  return level_dimensions<grid_level, __device_fill>(__device_fill{__coef});
}

/**
 * @brief Describe grid dimensions with a specific unit, that might be deeper down the hierarchy.
 *
 * Levels inside hierarchy_dimensions type have their dimensions described in terms of the level immediately below
 * being its unit. Sometimes, it would be more convinient to describe dimensions of a level using a specific unit that
 * appears deeper in the hierarchy. This function will create a grid level described with the supplied unit,
 * regardless of position of that unit in the hierarchy. Provided size for this level needs to be divisible by the
 * size of levels below it expressed in the same unit unless its wrapped in at_least type, in which case it will be
 * rounded up if needed.
 *
 * @par Snippet
 * @code
 * #include <cudax/launch.cuh>
 *
 * // Request the grid size to be specified using threads as a unit
 * auto hierarchy = make_hierarchy(block_dims<256>(), grid_dims(at_least(problem_size), thread));
 * // Resulting hierarchy will have at least `problem_size` threads
 * auto finalized = hierarchy.finalize(device, kernel);
 * assert(finalized.count(thread, grid) >= problem_size);
 * @endcode
 * @par
 *
 * @tparam _Unit
 * The unit requested for the count of this level to be expressed in
 *
 * @param count
 * The count request for this level expressed in _Unit
 */
template <typename _Unit, typename _CntType>
_CCCL_NODISCARD constexpr auto grid_dims(_CntType __count, [[maybe_unused]] _Unit __unit = _Unit())
{
  return level_dimensions<grid_level, __target_count<_Unit>>(__target_count<_Unit>(__count));
}

/**
 * @brief Describe cluster dimensions with a specific unit, that might be deeper down the hierarchy.
 *
 * Levels inside hierarchy_dimensions type have their dimensions described in terms of the level immediately below
 * being its unit. Sometimes, it would be more convinient to describe dimensions of a level using a specific unit that
 * appears deeper in the hierarchy. This function will create a cluster level described with the supplied unit,
 * regardless of position of that unit in the hierarchy. Provided size for this level needs to be divisible by the
 * size of levels below it expressed in the same unit unless its wrapped in at_least type, in which case it will be
 * rounded up if needed.
 *
 * @par Snippet
 * @code
 * #include <cudax/launch.cuh>
 *
 * // Request the cluster size to be specified using threads as a unit
 * auto hierarchy = make_hierarchy(block_dims<256>(), cluster_dims(1024, thread));
 * // Resulting hierarchy will have 1024 / 256 blocks in the cluster
 * auto finalized = hierarchy.finalize(device, kernel);
 * assert(finalized.count(thread, cluster) == 4);
 * @endcode
 * @par
 *
 * @tparam _Unit
 * The unit requested for the count of this level to be expressed in
 *
 * @param count
 * The count request for this level expressed in _Unit
 */
template <typename _Unit, typename _CntType>
_CCCL_NODISCARD constexpr auto cluster_dims(_CntType __count, [[maybe_unused]] _Unit __unit = _Unit())
{
  return level_dimensions<cluster_level, __target_count<_Unit>>(__target_count<_Unit>(__count));
}

/**
 * @brief Describe block dimensions with a specific unit, that might be deeper down the hierarchy.
 *
 * Levels inside hierarchy_dimensions type have their dimensions described in terms of the level immediately below
 * being its unit. Sometimes, it would be more convinient to describe dimensions of a level using a specific unit that
 * appears deeper in the hierarchy. This function will create a block level described with the supplied unit,
 * regardless of position of that unit in the hierarchy. Provided size for this level needs to be divisible by the
 * size of levels below it expressed in the same unit unless its wrapped in at_least type, in which case it will be
 * rounded up if needed.
 *
 * @tparam _Unit
 * The unit requested for the count of this level to be expressed in
 *
 * @param count
 * The count request for this level expressed in _Unit
 */
template <typename _Unit, typename _CntType>
_CCCL_NODISCARD constexpr auto block_dims(_CntType __count, [[maybe_unused]] _Unit __unit = _Unit())
{
  return level_dimensions<block_level, __target_count<_Unit>>(__target_count<_Unit>(__count));
}

template <typename>
struct finalized;

template <typename _BottomUnit, typename... _Levels>
struct finalized<hierarchy_dimensions_fragment<_BottomUnit, _Levels...>>
{
  using type = hierarchy_dimensions_fragment<_BottomUnit, __finalized_level<_Levels>...>;
};

template <typename _Dimensions, typename... _Options>
struct finalized<kernel_config<_Dimensions, _Options...>>
{
  using type = kernel_config<typename finalized<_Dimensions>::type, _Options...>;
};

/**
 * @brief Transform a hierarchy or configuration type into one that finalize would return
 *
 * @tparam T
 * Either hierarchy_dimensions or kernel_config type
 */
template <typename T>
using finalized_t = typename finalized<T>::type;

template <typename... _Args, typename _T, typename _Kernel>
_CCCL_NODISCARD auto __finalize_impl(const _T& __to_finalize, const _Kernel& __kernel)
{
  if constexpr (::cuda::std::is_function_v<std::remove_pointer_t<_Kernel>>)
  {
    return __finalize_no_device_set(__to_finalize, __kernel);
  }
  else
  {
    return __finalize_no_device_set(
      __to_finalize, __get_kernel_launcher<_Kernel, finalized_t<_T>, as_kernel_arg_t<_Args>...>());
  }
}

// TODO figure out how to add checks that if a hierarchy was finalized and later launched, then the same function was
// used in both places
template <typename _BottomUnit, typename... _Levels>
template <typename... _Args, typename _Kernel>
_CCCL_NODISCARD auto
hierarchy_dimensions_fragment<_BottomUnit, _Levels...>::finalize(device_ref __device, const _Kernel& __kernel)
{
  __ensure_current_device __dev_setter(__device);
  return __finalize_impl<_Args...>(*this, __kernel);
}

template <typename _Dimensions, typename... _Options>
template <typename... _Args, typename _Kernel>
_CCCL_NODISCARD auto kernel_config<_Dimensions, _Options...>::finalize(device_ref __device, const _Kernel& __kernel)
{
  __ensure_current_device __dev_setter(__device);
  return __finalize_impl<_Args...>(*this, __kernel);
}

/**
 * @brief A shorthand for creating a hierarchy with at least the requested count of CUDA threads
 *
 * This function will return a hierarchy with at least the requested number of threads,
 * The rest of the hierarchy is not specified, but it will be a valid hierarchy that
 * can be used to launch a kernel.
 * Requires finalization before any queries.
 *
 * @par Snippet
 * @code
 * #include <cudax/hierarchy_dimensions.cuh>
 * using namespace cuda::experimental;
 *
 * // When used to launch a kernel the resulting grid will have at least numElements threads
 * auto dims = distribute(numElements);
 * launch(stream, dims, kernel, kernel_arg);
 * @endcode
 */
_CCCL_NODISCARD inline auto distribute(int __num_elements) noexcept
{
  return make_hierarchy(auto_block_dims(), grid_dims(at_least(__num_elements), thread));
}

} // namespace cuda::experimental
#endif // _CCCL_STD_VER >= 2017
#endif // !_CUDAX__LAUNCH_HIERARCHY_DRAFT
