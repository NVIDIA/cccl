//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DATA_MOVEMENT_PROPERTIES_H
#define _CUDA___DATA_MOVEMENT_PROPERTIES_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__annotated_ptr/access_property.h>
#include <cuda/__annotated_ptr/associate_access_property.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

/***********************************************************************************************************************
 * Load Behavior
 **********************************************************************************************************************/

enum class _LdStPropertyEnum
{
  _ReadOnly,
  _ReadWrite,
  _LowReuse,
  _HighReuse,
  _NoReuse,
  _EnablePrefetch,
  _EnableL2Hint
};

template <_LdStPropertyEnum _Value>
using __ld_st_property_t = _CUDA_VSTD::integral_constant<_LdStPropertyEnum, _Value>;

using __read_only_t        = __ld_st_property_t<_LdStPropertyEnum::_ReadOnly>;
using __read_write_t       = __ld_st_property_t<_LdStPropertyEnum::_ReadWrite>;
using __cache_reuse_low_t  = __ld_st_property_t<_LdStPropertyEnum::_LowReuse>;
using __cache_reuse_high_t = __ld_st_property_t<_LdStPropertyEnum::_HighReuse>;
using __cache_no_reuse_t   = __ld_st_property_t<_LdStPropertyEnum::_NoReuse>;
using __enable_prefetch    = __ld_st_property_t<_LdStPropertyEnum::_EnablePrefetch>;
using __enable_l2_hint     = __ld_st_property_t<_LdStPropertyEnum::_EnableL2Hint>;

inline constexpr auto read_only        = __read_only_t{};
inline constexpr auto read_write       = __read_write_t{};
inline constexpr auto cache_reuse_low  = __cache_reuse_low_t{};
inline constexpr auto cache_reuse_high = __cache_reuse_high_t{};
inline constexpr auto cache_no_reuse   = __cache_no_reuse_t{};
inline constexpr auto enable_prefetch  = __enable_prefetch{};
inline constexpr auto enable_l2_hint   = __enable_l2_hint{};

/***********************************************************************************************************************
 * Cache Hint
 **********************************************************************************************************************/

template <_LdStPropertyEnum... _Args>
struct __ld_st_list
{};

using __st_allowed =
  __ld_st_list<_LdStPropertyEnum::_LowReuse,
               _LdStPropertyEnum::_HighReuse,
               _LdStPropertyEnum::_NoReuse,
               _LdStPropertyEnum::_EnableL2Hint>;

template <_LdStPropertyEnum _Property, _LdStPropertyEnum... _Args>
[[nodiscard]] _CCCL_DEVICE_API constexpr bool __has_property(__ld_st_list<_Args...>) noexcept
{
  if constexpr (sizeof...(_Args) > 0)
  {
    constexpr _LdStPropertyEnum __array[] = {_Args...};
    for (auto __v : __array)
    {
      if (__v == _Property)
      {
        return true;
      }
    }
  }
  return false;
}

/***********************************************************************************************************************
 * Load Properties
 **********************************************************************************************************************/

template <_LdStPropertyEnum... _Args>
struct _LoadProperties
{
  access_property __l2_hint;

  _LoadProperties() noexcept = default;

  _CCCL_DEVICE_API constexpr _LoadProperties(access_property __l2_hint1) noexcept
      : __l2_hint{__l2_hint1}
  {}
};

_LoadProperties(access_property) -> _LoadProperties<_LdStPropertyEnum::_EnableL2Hint>;

//----------------------------------------------------------------------------------------------------------------------
// operator|

template <_LdStPropertyEnum _Property, _LdStPropertyEnum... _Args>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto operator|(_LoadProperties<_Args...> __props, __ld_st_property_t<_Property>)
{
  static_assert(!_CUDA_DEVICE::__has_property<_Property>(__ld_st_list<_Args...>{}), "property already set");
  return _LoadProperties<_Args..., _Property>{__props.__l2_hint};
}

template <_LdStPropertyEnum _Property, _LdStPropertyEnum... _Args>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto
operator|(__ld_st_property_t<_Property> __prop, _LoadProperties<_Args...> __props)
{
  return __prop | __props;
}

template <_LdStPropertyEnum... _Args>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto operator|(_LoadProperties<_Args...> __props, access_property __l2_hint)
{
  static_assert(!_CUDA_DEVICE::__has_property<_LdStPropertyEnum::_EnableL2Hint, _Args...>(__props),
                "property already set");
  return _LoadProperties<_Args..., _LdStPropertyEnum::_EnableL2Hint>{__l2_hint};
}

template <_LdStPropertyEnum... _Args>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto operator|(access_property __l2_hint, _LoadProperties<_Args...> __props)
{
  return __l2_hint | __props;
}

/***********************************************************************************************************************
 * Store Properties
 **********************************************************************************************************************/

template <_LdStPropertyEnum... _Args>
struct _StoreProperties
{
  access_property __l2_hint;

  _StoreProperties() noexcept = default;

  _CCCL_DEVICE_API constexpr _StoreProperties(access_property __l2_hint1) noexcept
      : __l2_hint{__l2_hint1}
  {}
};

_StoreProperties(access_property) -> _StoreProperties<_LdStPropertyEnum::_EnableL2Hint>;

//----------------------------------------------------------------------------------------------------------------------
// operator|

template <_LdStPropertyEnum _Property, _LdStPropertyEnum... _Args>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto
operator|(_StoreProperties<_Args...> __props, __ld_st_property_t<_Property>)
{
  static_assert(!_CUDA_DEVICE::__has_property<_Property>(__ld_st_list<_Args...>{}), "property already set");
  static_assert(!_CUDA_DEVICE::__has_property<_Property>(__st_allowed{}), "property not allowed for store");
  return _StoreProperties<_Args..., _Property>{__props.__l2_hint};
}

template <_LdStPropertyEnum _Property, _LdStPropertyEnum... _Args>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto
operator|(__ld_st_property_t<_Property>, _StoreProperties<_Args...> __props)
{
  return __props | _Property;
}

template <_LdStPropertyEnum... _Args>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto operator|(_StoreProperties<_Args...> __props, access_property __l2_hint)
{
  static_assert(!_CUDA_DEVICE::__has_property<_LdStPropertyEnum::_EnableL2Hint, _Args...>(__props),
                "property already set");
  return _StoreProperties<_Args..., _LdStPropertyEnum::_EnableL2Hint>{__l2_hint};
}

template <_LdStPropertyEnum... _Args>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto operator|(access_property __l2_hint, _StoreProperties<_Args...> __props)
{
  return __props | __l2_hint;
}

/***********************************************************************************************************************
 * PTX Maximum Access Size
 **********************************************************************************************************************/

#if __cccl_ptx_isa >= 830
inline constexpr size_t __max_ptx_access_size = 16;
#else
inline constexpr size_t __max_ptx_access_size = 8;
#endif

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#endif // _CUDA___DATA_MOVEMENT_PROPERTIES_H
