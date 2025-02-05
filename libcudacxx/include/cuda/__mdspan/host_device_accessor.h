//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_HOST_DEVICE_ACCESSOR
#define _CUDA___MDSPAN_HOST_DEVICE_ACCESSOR

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Accessor>
struct host_accessor;

template <typename _Accessor>
struct device_accessor;

template <typename _Accessor>
struct managed_accessor;

/***********************************************************************************************************************
 * Host/Device/Managed Accessor Traits
 **********************************************************************************************************************/

template <typename T>
inline constexpr bool is_host_accessor_v = false;

template <typename T>
inline constexpr bool is_device_accessor_v = false;

template <typename T>
inline constexpr bool is_managed_accessor_v = false;

template <typename _Accessor>
inline constexpr bool is_host_accessor_v<host_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_device_accessor_v<device_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_managed_accessor_v<managed_accessor<_Accessor>> = true;

template <typename T>
inline constexpr bool is_host_device_managed_accessor_v =
  is_host_accessor_v<T> || is_device_accessor_v<T> || is_managed_accessor_v<T>;

/***********************************************************************************************************************
 * Host Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
struct host_accessor : public _Accessor
{
private:
  static constexpr bool __is_ctor_noexcept      = noexcept(_Accessor{});
  static constexpr bool __is_copy_ctor_noexcept = noexcept(_Accessor{});

  static_assert(!is_host_device_managed_accessor_v<_Accessor>, "Host/Device/Managed accessor cannot be nested");

public:
  using offset_policy = host_accessor;

  constexpr host_accessor() noexcept(__is_ctor_noexcept) = default;

  constexpr host_accessor(const host_accessor&) noexcept(__is_copy_ctor_noexcept) = default;
};

/***********************************************************************************************************************
 * Device Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
struct device_accessor : public _Accessor
{
private:
  static constexpr bool __is_ctor_noexcept      = noexcept(_Accessor{});
  static constexpr bool __is_copy_ctor_noexcept = noexcept(_Accessor{});

  static_assert(!is_host_device_managed_accessor_v<_Accessor>, "Host/Device/Managed accessor cannot be nested");

public:
  using offset_policy = device_accessor;

  constexpr device_accessor() noexcept(__is_ctor_noexcept) = default;

  constexpr device_accessor(const device_accessor&) noexcept(__is_copy_ctor_noexcept) = default;
};

/***********************************************************************************************************************
 * Managed Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
struct managed_accessor : public _Accessor
{
private:
  static constexpr bool __is_ctor_noexcept      = noexcept(_Accessor{});
  static constexpr bool __is_copy_ctor_noexcept = noexcept(_Accessor{});

  static_assert(!is_host_device_managed_accessor_v<_Accessor>, "Host/Device/Managed accessor cannot be nested");

public:
  using offset_policy = managed_accessor;

  constexpr managed_accessor() noexcept(__is_ctor_noexcept) = default;

  constexpr managed_accessor(const managed_accessor&) noexcept(__is_copy_ctor_noexcept) = default;
};

/***********************************************************************************************************************
 * Accessibility Traits
 **********************************************************************************************************************/

template <typename T>
inline constexpr bool is_host_accessible_v = false;

template <typename T>
inline constexpr bool is_device_accessible_v = false;

template <typename _Accessor>
inline constexpr bool is_host_accessible_v<host_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_host_accessible_v<managed_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_device_accessible_v<device_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_device_accessible_v<managed_accessor<_Accessor>> = true;

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___MDSPAN_HOST_DEVICE_ACCESSOR
