//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___LAUNCH_FINALIZE_H
#define _CUDA___LAUNCH_FINALIZE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__cmath/ceil_div.h>
#  include <cuda/__driver/driver_api.h>
#  include <cuda/__hierarchy/hierarchy_dimensions.h>
#  include <cuda/__hierarchy/meta_level_dimensions.h>
#  include <cuda/__launch/configuration.h>
#  include <cuda/__launch/get_cufunction.h>
#  include <cuda/__runtime/ensure_current_context.h>
#  include <cuda/__stream/launch_transform.h>
#  include <cuda/std/__cmath/isfinite.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__limits/numeric_limits.h>
#  include <cuda/std/__type_traits/decay.h>
#  include <cuda/std/__type_traits/is_function.h>
#  include <cuda/std/__type_traits/is_pointer.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/tuple>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#  if _CCCL_CUDA_COMPILATION()

template <class _Kernel, class _Config, class... _Args>
[[nodiscard]] _CCCL_API constexpr const void* __get_kernel_launcher() noexcept;

#  endif // _CCCL_CUDA_COMPILATION()

namespace __detail
{
using __meta_dims_finalized = ::cuda::std::extents<dimensions_index_type, ::cuda::std::dynamic_extent, 1, 1>;
} // namespace __detail

template <typename _Tp>
struct finalized
{
  using type = _Tp;
};

template <class _Level, class _Meta>
struct finalized<hierarchy_level_desc_meta<_Level, _Meta>>
{
  using type = hierarchy_level_desc<_Level, __detail::__meta_dims_finalized>;
};

template <class _BottomUnit, class... _Levels>
struct finalized<hierarchy<_BottomUnit, _Levels...>>
{
  using type = hierarchy<_BottomUnit, typename finalized<_Levels>::type...>;
};

template <typename _Dimensions, typename... _Options>
struct finalized<kernel_config<_Dimensions, _Options...>>
{
  using type = kernel_config<typename finalized<_Dimensions>::type, _Options...>;
};

template <typename _Tp>
using finalized_t = typename finalized<_Tp>::type;

namespace __detail
{
struct __finalize_context
{
  ::CUfunction __kernel_{};
  unsigned int __dynamic_smem_size_{};
};

[[nodiscard]] _CCCL_HOST_API inline __detail::__meta_dims_finalized
__make_finalized_meta_extents(::cuda::std::size_t __x_extent)
{
  return __detail::__meta_dims_finalized{static_cast<dimensions_index_type>(__x_extent)};
}

template <class _Unit, class _BottomUnit>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::size_t __count_of_units_below(const ::cuda::std::tuple<>&) noexcept
{
  static_assert(::cuda::std::is_same_v<_Unit, _BottomUnit>, "The requested unit is not available below this level");
  return 1;
}

template <class _Unit, class _BottomUnit, class _TopLevelDesc, class... _LevelDescs>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::size_t
__count_of_units_below(const ::cuda::std::tuple<_TopLevelDesc, _LevelDescs...>& __levels) noexcept
{
  using _TopLevel    = typename _TopLevelDesc::level_type;
  const auto __below = hierarchy<_BottomUnit, _TopLevelDesc, _LevelDescs...>{_BottomUnit{}, __levels};
  return _Unit{}.template count_as<::cuda::std::size_t>(_TopLevel{}, __below);
}

template <class _Unit, class _Level, class _BottomUnit, class... _LevelDescs>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::size_t
__count_of_units_at_level(const ::cuda::std::tuple<_LevelDescs...>& __levels) noexcept
{
  const auto __below = hierarchy<_BottomUnit, _LevelDescs...>{_BottomUnit{}, __levels};
  return _Unit{}.template count_as<::cuda::std::size_t>(_Level{}, __below);
}

template <class _BottomUnit, class _LevelDesc, class _FinalizedRest>
[[nodiscard]] _CCCL_HOST_API constexpr _LevelDesc
__finalize_level(const __finalize_context&, const _LevelDesc& __level, const _FinalizedRest&) noexcept
{
  return __level;
}

template <class _BottomUnit, class _Level, class _Unit, class... _FinalizedRest>
[[nodiscard]] _CCCL_HOST_API hierarchy_level_desc<_Level, __meta_dims_finalized> __finalize_level(
  const __finalize_context&,
  const hierarchy_level_desc_meta<_Level, __target_count<_Unit>>& __level,
  const ::cuda::std::tuple<_FinalizedRest...>& __rest)
{
  const auto __below_count = __count_of_units_below<_Unit, _BottomUnit>(__rest);
  if (__below_count == 0)
  {
    _CCCL_THROW(::std::invalid_argument, "Cannot finalize hierarchy level using zero lower-level units");
  }

  const auto __target = __level.meta().__count_;
  ::cuda::std::size_t __final_count{};

  if (__level.meta().__ceil_div_)
  {
    __final_count = ::cuda::ceil_div(__target, __below_count);
  }
  else
  {
    if (__target % __below_count != 0)
    {
      _CCCL_THROW(::std::invalid_argument,
                  "Count for this hierarchy level is not divisible by lower levels; use cuda::at_least to round up");
    }
    __final_count = __target / __below_count;
  }

  return hierarchy_level_desc<_Level, __meta_dims_finalized>{__make_finalized_meta_extents(__final_count)};
}

template <class _BottomUnit, class _Level, class... _FinalizedRest>
[[nodiscard]] _CCCL_HOST_API hierarchy_level_desc<_Level, __meta_dims_finalized> __finalize_level(
  const __finalize_context& __ctx,
  const hierarchy_level_desc_meta<_Level, __max_occupancy>&,
  const ::cuda::std::tuple<_FinalizedRest...>&)
{
  static_assert(::cuda::std::is_same_v<_Level, block_level>, "auto_block_dims can only describe block dimensions");
  static_assert(sizeof...(_FinalizedRest) == 0, "auto_block_dims requires the block level to be described in threads");

  int __minimum_grid_size{};
  int __block_size{};
  ::cuda::__driver::__occupancyMaxPotentialBlockSize(
    &__minimum_grid_size, &__block_size, __ctx.__kernel_, __ctx.__dynamic_smem_size_);

  return hierarchy_level_desc<_Level, __meta_dims_finalized>{
    __make_finalized_meta_extents(static_cast<::cuda::std::size_t>(__block_size))};
}

template <class _BottomUnit, class _Level, class... _FinalizedRest>
[[nodiscard]] _CCCL_HOST_API hierarchy_level_desc<_Level, __meta_dims_finalized> __finalize_level(
  const __finalize_context& __ctx,
  const hierarchy_level_desc_meta<_Level, __device_fill>& __level,
  const ::cuda::std::tuple<_FinalizedRest...>& __rest)
{
  static_assert(::cuda::std::is_same_v<_Level, grid_level>, "fill_device can only describe grid dimensions");

  if (!(__level.meta().__fill_coeff_ > 0.0f))
  {
    _CCCL_THROW(::std::invalid_argument, "fill_device requires a positive fill coefficient");
  }

  const auto __block_size       = __count_of_units_at_level<thread_level, block_level, _BottomUnit>(__rest);
  const auto __block_multiplier = __count_of_units_below<block_level, _BottomUnit>(__rest);
  if (__block_size > static_cast<::cuda::std::size_t>((::cuda::std::numeric_limits<int>::max)()))
  {
    _CCCL_THROW(::std::invalid_argument, "fill_device block size exceeds CUDA occupancy API limits");
  }
  if (__block_multiplier == 0)
  {
    _CCCL_THROW(::std::invalid_argument, "Cannot finalize fill_device using zero lower-level blocks");
  }

  const auto __device  = ::cuda::__driver::__ctxGetDevice();
  const auto __num_sms = ::cuda::__driver::__deviceGetAttribute(::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, __device);
  const auto __num_blocks_per_sm = ::cuda::__driver::__occupancyMaxActiveBlocksPerMultiprocessor(
    __ctx.__kernel_, static_cast<int>(__block_size), __ctx.__dynamic_smem_size_);

  const auto __final_count_raw =
    static_cast<double>(__num_sms) * static_cast<double>(__num_blocks_per_sm) / static_cast<double>(__block_multiplier)
    * static_cast<double>(__level.meta().__fill_coeff_);
  if (!::cuda::std::isfinite(__final_count_raw)
      || __final_count_raw >= static_cast<double>((::cuda::std::numeric_limits<::cuda::std::size_t>::max)()))
  {
    _CCCL_THROW(::std::invalid_argument, "fill_device produces too many hierarchy levels");
  }

  const auto __final_count = static_cast<::cuda::std::size_t>(__final_count_raw);
  if (__final_count == 0)
  {
    _CCCL_THROW(::std::invalid_argument, "Not able to run any blocks with this configuration");
  }

  return hierarchy_level_desc<_Level, __meta_dims_finalized>{__make_finalized_meta_extents(__final_count)};
}

template <class _BottomUnit>
[[nodiscard]] _CCCL_HOST_API constexpr auto __finalize_hierarchy_levels(const __finalize_context&) noexcept
{
  return ::cuda::std::make_tuple();
}

template <class _BottomUnit, class _LevelDesc, class... _LevelDescs>
[[nodiscard]] _CCCL_HOST_API auto
__finalize_hierarchy_levels(const __finalize_context& __ctx, const _LevelDesc& __level, const _LevelDescs&... __rest)
{
  auto __rest_finalized  = __finalize_hierarchy_levels<_BottomUnit>(__ctx, __rest...);
  auto __level_finalized = __finalize_level<_BottomUnit>(__ctx, __level, __rest_finalized);
  return ::cuda::std::tuple_cat(::cuda::std::make_tuple(__level_finalized), __rest_finalized);
}

template <class _BottomUnit, class... _LevelDescs>
[[nodiscard]] _CCCL_HOST_API auto
__finalize_hierarchy(const __finalize_context& __ctx, const hierarchy<_BottomUnit, _LevelDescs...>& __hierarchy)
{
  auto __finalized_levels = ::cuda::std::apply(
    [&](const auto&... __levels) {
      return __finalize_hierarchy_levels<_BottomUnit>(__ctx, __levels...);
    },
    __hierarchy.__levels());
  return finalized_t<hierarchy<_BottomUnit, _LevelDescs...>>{_BottomUnit{}, __finalized_levels};
}

template <class _Option>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::size_t __dynamic_smem_size_from_option(const _Option& __option)
{
  if constexpr (_Option::kind == __detail::launch_option_kind::dynamic_shared_memory)
  {
    return __option.size_bytes();
  }
  else
  {
    return 0;
  }
}

template <typename... _Options>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::size_t
__dynamic_smem_size_from_options(const ::cuda::std::tuple<_Options...>& __options)
{
  ::cuda::std::size_t __ret{};
  ::cuda::std::apply(
    [&](const auto&... __opts) {
      ((void) (__ret += __dynamic_smem_size_from_option(__opts)), ...);
    },
    __options);
  return __ret;
}

template <class _BottomUnit, class... _LevelDescs>
[[nodiscard]] _CCCL_HOST_API auto __finalize_no_device_set(
  const hierarchy<_BottomUnit, _LevelDescs...>& __hierarchy, [[maybe_unused]] ::CUfunction __kernel)
{
  using _Hierarchy = hierarchy<_BottomUnit, _LevelDescs...>;
  if constexpr (::cuda::std::is_same_v<finalized_t<_Hierarchy>, _Hierarchy>)
  {
    return __hierarchy;
  }
  else
  {
    return __finalize_hierarchy(__finalize_context{__kernel, 0}, __hierarchy);
  }
}

template <class _BottomUnit, class... _LevelDescs>
[[nodiscard]] _CCCL_HOST_API auto __finalize_no_device_set(
  const hierarchy<_BottomUnit, _LevelDescs...>& __hierarchy, [[maybe_unused]] const void* __kernel)
{
  using _Hierarchy = hierarchy<_BottomUnit, _LevelDescs...>;
  if constexpr (::cuda::std::is_same_v<finalized_t<_Hierarchy>, _Hierarchy>)
  {
    return __hierarchy;
  }
  else
  {
    return __finalize_hierarchy(__finalize_context{::cuda::__get_cufunction_of(__kernel), 0}, __hierarchy);
  }
}

template <typename _Dimensions, typename... _Options>
[[nodiscard]] _CCCL_HOST_API auto __finalize_no_device_set(
  const kernel_config<_Dimensions, _Options...>& __config, [[maybe_unused]] ::CUfunction __kernel)
{
  using _Config = kernel_config<_Dimensions, _Options...>;
  if constexpr (::cuda::std::is_same_v<finalized_t<_Config>, _Config>)
  {
    return __config;
  }
  else
  {
    const auto __ctx =
      __finalize_context{__kernel, static_cast<unsigned int>(__dynamic_smem_size_from_options(__config.options()))};
    return kernel_config<finalized_t<_Dimensions>, _Options...>{
      __finalize_hierarchy(__ctx, __config.hierarchy()), __config.options()};
  }
}

template <typename _Dimensions, typename... _Options>
[[nodiscard]] _CCCL_HOST_API auto
__finalize_no_device_set(const kernel_config<_Dimensions, _Options...>& __config, [[maybe_unused]] const void* __kernel)
{
  using _Config = kernel_config<_Dimensions, _Options...>;
  if constexpr (::cuda::std::is_same_v<finalized_t<_Config>, _Config>)
  {
    return __config;
  }
  else
  {
    const auto __ctx =
      __finalize_context{::cuda::__get_cufunction_of(__kernel),
                         static_cast<unsigned int>(__dynamic_smem_size_from_options(__config.options()))};
    return kernel_config<finalized_t<_Dimensions>, _Options...>{
      __finalize_hierarchy(__ctx, __config.hierarchy()), __config.options()};
  }
}
} // namespace __detail

template <class _BottomUnit, class... _LevelDescs>
[[nodiscard]] _CCCL_HOST_API auto
finalize(const hierarchy<_BottomUnit, _LevelDescs...>& __hierarchy, const void* __kernel)
{
  return ::cuda::__detail::__finalize_no_device_set(__hierarchy, __kernel);
}

template <typename _Dimensions, typename... _Options>
[[nodiscard]] _CCCL_HOST_API auto finalize(const kernel_config<_Dimensions, _Options...>& __config, const void* __kernel)
{
  return ::cuda::__detail::__finalize_no_device_set(__config, __kernel);
}

template <class _BottomUnit, class... _LevelDescs, class... _Args>
[[nodiscard]] _CCCL_HOST_API auto
finalize(const hierarchy<_BottomUnit, _LevelDescs...>& __hierarchy, void (*__kernel)(_Args...))
{
  return ::cuda::__detail::__finalize_no_device_set(__hierarchy, reinterpret_cast<const void*>(__kernel));
}

template <typename _Dimensions, typename... _Options, class... _Args>
[[nodiscard]] _CCCL_HOST_API auto
finalize(const kernel_config<_Dimensions, _Options...>& __config, void (*__kernel)(_Args...))
{
  return ::cuda::__detail::__finalize_no_device_set(__config, reinterpret_cast<const void*>(__kernel));
}

template <class _ToFinalize>
[[nodiscard]] _CCCL_HOST_API auto finalize(device_ref __device, const _ToFinalize& __to_finalize, const void* __kernel)
{
  __ensure_current_context __dev_setter{__device};
  return ::cuda::__detail::__finalize_no_device_set(__to_finalize, __kernel);
}

template <class _ToFinalize, class... _Args>
[[nodiscard]] _CCCL_HOST_API auto
finalize(device_ref __device, const _ToFinalize& __to_finalize, void (*__kernel)(_Args...))
{
  __ensure_current_context __dev_setter{__device};
  return ::cuda::__detail::__finalize_no_device_set(__to_finalize, reinterpret_cast<const void*>(__kernel));
}

#  if _CCCL_CUDA_COMPILATION()

_CCCL_TEMPLATE(typename... _Args, typename _Dimensions, typename... _Config, typename _Kernel)
_CCCL_REQUIRES((!::cuda::std::is_pointer_v<_Kernel>) _CCCL_AND(!::cuda::std::is_function_v<_Kernel>))
[[nodiscard]] _CCCL_HOST_API auto finalize(const kernel_config<_Dimensions, _Config...>& __conf, const _Kernel&)
{
  using _FinalizedConfig = finalized_t<kernel_config<_Dimensions, _Config...>>;
  auto __launcher        = ::cuda::
    __get_kernel_launcher<_Kernel, _FinalizedConfig, ::cuda::std::decay_t<transformed_device_argument_t<_Args>>...>();
  return ::cuda::__detail::__finalize_no_device_set(__conf, __launcher);
}

_CCCL_TEMPLATE(typename... _Args, typename _Dimensions, typename... _Config, typename _Kernel)
_CCCL_REQUIRES((!::cuda::std::is_pointer_v<_Kernel>) _CCCL_AND(!::cuda::std::is_function_v<_Kernel>))
[[nodiscard]] _CCCL_HOST_API auto
finalize(device_ref __device, const kernel_config<_Dimensions, _Config...>& __conf, const _Kernel& __kernel)
{
  __ensure_current_context __dev_setter{__device};
  return ::cuda::finalize<_Args...>(__conf, __kernel);
}

#  endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___LAUNCH_FINALIZE_H
