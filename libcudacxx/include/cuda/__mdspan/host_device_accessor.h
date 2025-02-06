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

#include <cuda_runtime_api.h>

#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/cassert>

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
  using __data_handle_type = typename _Accessor::data_handle_type;
  using __reference        = typename _Accessor::reference;

  static constexpr bool __is_ctor_noexcept      = noexcept(_Accessor{});
  static constexpr bool __is_copy_ctor_noexcept = noexcept(_Accessor{});
  static constexpr bool __is_access_noexcept    = noexcept(_Accessor{}.access(__data_handle_type{}, 0));
  static constexpr bool __is_offset_noexcept    = noexcept(_Accessor{}.offset(__data_handle_type{}, 0));

  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::host_accessor/cuda::device_accessor/cuda::managed_accessor accessor cannot be nested");

  template <typename __data_handle_type>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __is_host_accessible_pointer(__data_handle_type __p)
  {
    if constexpr (::cuda::std::is_pointer_v<__data_handle_type>)
    {
      cudaPointerAttributes __attrib;
      _CCCL_VERIFY(::cudaPointerGetAttributes(&__attrib, __p) == ::cudaSuccess, "cudaPointerGetAttributes failed");
      return __attrib.hostPointer != nullptr || __attrib.type == ::cudaMemoryTypeUnregistered;
    }
    else
    {
      return true; // cannot be verified
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __check_host_pointer(__data_handle_type __p)
  {
    _CCCL_ASSERT(__is_host_accessible_pointer(__p), "cuda::host_accessor data handle is not a HOST pointer");
  }

public:
  using offset_policy = host_accessor;

  _CCCL_HIDE_FROM_ABI host_accessor() noexcept(__is_ctor_noexcept) = default;

  _CCCL_HIDE_FROM_ABI host_accessor(const host_accessor&) noexcept(__is_copy_ctor_noexcept) = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __reference access(__data_handle_type __p, size_t __i) const
    noexcept(__is_access_noexcept)
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (__check_host_pointer(__p);), //
                      (static_assert(false, "cuda::host_accessor cannot be used in DEVICE code");))
    return _Accessor::access(__p, __i);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __data_handle_type offset(__data_handle_type __p, size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (__check_host_pointer(__p);), //
                      (static_assert(false, "cuda::host_accessor cannot be used in DEVICE code");))
    return _Accessor::offset(__p, __i);
  }
};

/***********************************************************************************************************************
 * Device Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
struct device_accessor : public _Accessor
{
private:
  using __data_handle_type = typename _Accessor::data_handle_type;
  using __reference        = typename _Accessor::reference;

  static constexpr bool __is_ctor_noexcept      = noexcept(_Accessor{});
  static constexpr bool __is_copy_ctor_noexcept = noexcept(_Accessor{});
  static constexpr bool __is_access_noexcept    = noexcept(_Accessor{}.access(__data_handle_type{}, 0));
  static constexpr bool __is_offset_noexcept    = noexcept(_Accessor{}.offset(__data_handle_type{}, 0));

  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::host_accessor/cuda::device_accessor/cuda::managed_accessor accessor cannot be nested");

  template <typename _Sp = bool> // lazy evaluation
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __prevent_host_instantiation() noexcept
  {
    static_assert(sizeof(_Sp) != sizeof(_Sp), "cuda::device_accessor cannot be used in HOST code");
  }

public:
  using offset_policy = device_accessor;

  _CCCL_HIDE_FROM_ABI device_accessor() noexcept(__is_ctor_noexcept) = default;

  _CCCL_HIDE_FROM_ABI device_accessor(const device_accessor&) noexcept(__is_copy_ctor_noexcept) = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __reference access(__data_handle_type __p, size_t __i) const
    noexcept(__is_access_noexcept)
  {
    NV_IF_TARGET(NV_IS_HOST, (__prevent_host_instantiation();))
    return _Accessor::access(__p, __i);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __data_handle_type offset(__data_handle_type __p, size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    NV_IF_TARGET(NV_IS_HOST, (__prevent_host_instantiation();))
    return _Accessor::offset(__p, __i);
  }
};

/***********************************************************************************************************************
 * Managed Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
struct managed_accessor : public _Accessor
{
private:
  using __data_handle_type = typename _Accessor::data_handle_type;
  using __reference        = typename _Accessor::reference;

  static constexpr bool __is_ctor_noexcept      = noexcept(_Accessor{});
  static constexpr bool __is_copy_ctor_noexcept = noexcept(_Accessor{});
  static constexpr bool __is_access_noexcept    = noexcept(_Accessor{}.access(__data_handle_type{}, 0));
  static constexpr bool __is_offset_noexcept    = noexcept(_Accessor{}.offset(__data_handle_type{}, 0));

  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::host_accessor/cuda::device_accessor/cuda::managed_accessor accessor cannot be nested");

  template <typename __data_handle_type>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __is_managed_pointer(__data_handle_type __p)
  {
    if constexpr (::cuda::std::is_pointer_v<__data_handle_type>)
    {
      cudaPointerAttributes __attrib;
      _CCCL_VERIFY(::cudaPointerGetAttributes(&__attrib, __p) == ::cudaSuccess, "cudaPointerGetAttributes failed");
      return __attrib.devicePointer != nullptr && __attrib.hostPointer == __attrib.devicePointer;
    }
    else
    {
      return true; // cannot be verified
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __check_managed_pointer(__data_handle_type __p)
  {
    _CCCL_ASSERT(__is_managed_pointer(__p), "cuda::managed_accessor data handle is not a MANAGED pointer");
  }

public:
  using offset_policy = managed_accessor;

  _CCCL_HIDE_FROM_ABI managed_accessor() noexcept(__is_ctor_noexcept) = default;

  _CCCL_HIDE_FROM_ABI managed_accessor(const managed_accessor&) noexcept(__is_copy_ctor_noexcept) = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __reference access(__data_handle_type __p, size_t __i) const
    noexcept(__is_access_noexcept)
  {
    NV_IF_TARGET(NV_IS_HOST, (__check_managed_pointer(__p);))
    return _Accessor::access(__p, __i);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __data_handle_type offset(__data_handle_type __p, size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    NV_IF_TARGET(NV_IS_HOST, (__check_managed_pointer(__p);))
    return _Accessor::offset(__p, __i);
  }
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
