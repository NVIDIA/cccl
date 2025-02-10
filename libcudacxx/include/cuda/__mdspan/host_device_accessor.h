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

#include <cuda_runtime_api.h> // TODO: the code should compile even outside nvcc

#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
//
#include <cuda/std/__concepts/__concept_macros.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Accessor>
struct __host_accessor;

template <typename _Accessor>
struct __device_accessor;

template <typename _Accessor>
struct __managed_accessor;

template <typename _Accessor>
using host_accessor = __host_accessor<_Accessor>;

template <typename _Accessor>
using device_accessor = __device_accessor<_Accessor>;

template <typename _Accessor>
using managed_accessor = __managed_accessor<_Accessor>;

/***********************************************************************************************************************
 * Host/Device/Managed Accessor Traits
 **********************************************************************************************************************/

template <typename>
inline constexpr bool is_host_accessor_v = false;

template <typename>
inline constexpr bool is_device_accessor_v = false;

template <typename>
inline constexpr bool is_managed_accessor_v = false;

template <typename _Accessor>
inline constexpr bool is_host_accessor_v<__host_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_device_accessor_v<__device_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_managed_accessor_v<__managed_accessor<_Accessor>> = true;

template <typename _Tp>
inline constexpr bool is_host_device_managed_accessor_v =
  is_host_accessor_v<_Tp> || is_device_accessor_v<_Tp> || is_managed_accessor_v<_Tp>;

/***********************************************************************************************************************
 * Host Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
struct __host_accessor : public _Accessor
{
  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::__host_accessor/cuda::__device_accessor/cuda::__managed_accessor cannot be nested");

  using offset_policy    = __host_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = typename _Accessor::data_handle_type;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

private:
  using __self = __host_accessor<_Accessor>;

  static constexpr bool __is_ctor_noexcept      = noexcept(_Accessor{});
  static constexpr bool __is_copy_ctor_noexcept = noexcept(_Accessor{_Accessor{}});
  static constexpr bool __is_access_noexcept    = noexcept(_Accessor{}.access(data_handle_type{}, 0));
  static constexpr bool __is_offset_noexcept    = noexcept(_Accessor{}.offset(data_handle_type{}, 0));

  template <typename data_handle_type>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __is_host_accessible_pointer(data_handle_type __p) noexcept
  {
    if constexpr (_CUDA_VSTD::is_pointer_v<data_handle_type>)
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

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __check_host_pointer(data_handle_type __p) noexcept
  {
    _CCCL_ASSERT(__self::__is_host_accessible_pointer(__p), "cuda::__host_accessor data handle is not a HOST pointer");
  }

public:
  _CCCL_TEMPLATE(typename _NotUsed = void)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_default_constructible, _Accessor))
  _LIBCUDACXX_HIDE_FROM_ABI __host_accessor() noexcept(__is_ctor_noexcept)
      : _Accessor{}
  {}

  _CCCL_TEMPLATE(typename _OtherElementType)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _OtherElementType (*)[], element_type (*)[]))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __host_accessor(__host_accessor<_OtherElementType>) noexcept(
    __is_copy_ctor_noexcept)
  {}

  _CCCL_TEMPLATE(typename _OtherElementType)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _OtherElementType (*)[], element_type (*)[]))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __host_accessor(__managed_accessor<_OtherElementType>) noexcept(
    __is_copy_ctor_noexcept)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference access(data_handle_type __p, _CUDA_VSTD::size_t __i) const
    noexcept(__is_access_noexcept)
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (__self::__check_host_pointer(__p);), //
                      (static_assert(false, "cuda::__host_accessor cannot be used in DEVICE code");))
    return _Accessor::access(__p, __i);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr data_handle_type offset(data_handle_type __p, _CUDA_VSTD::size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (__self::__check_host_pointer(__p);), //
                      (static_assert(false, "cuda::__host_accessor cannot be used in DEVICE code");))
    return _Accessor::offset(__p, __i);
  }
};

/***********************************************************************************************************************
 * Device Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
struct __device_accessor : public _Accessor
{
  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::__host_accessor/cuda::__device_accessor/cuda::__managed_accessor cannot be nested");

  using offset_policy    = __device_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = typename _Accessor::data_handle_type;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

private:
  using __self = __device_accessor<_Accessor>;

  static constexpr bool __is_ctor_noexcept      = noexcept(_Accessor{});
  static constexpr bool __is_copy_ctor_noexcept = noexcept(_Accessor{_Accessor{}});
  static constexpr bool __is_access_noexcept    = noexcept(_Accessor{}.access(data_handle_type{}, 0));
  static constexpr bool __is_offset_noexcept    = noexcept(_Accessor{}.offset(data_handle_type{}, 0));

  template <typename _Sp = bool> // lazy evaluation
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __prevent_host_instantiation() noexcept
  {
    static_assert(sizeof(_Sp) != sizeof(_Sp), "cuda::__device_accessor cannot be used in HOST code");
  }

public:
  _CCCL_TEMPLATE(typename _NotUsed = void)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_default_constructible, _Accessor))
  _LIBCUDACXX_HIDE_FROM_ABI __device_accessor() noexcept(__is_ctor_noexcept)
      : _Accessor{}
  {}

  _CCCL_TEMPLATE(typename _OtherElementType)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _OtherElementType (*)[], element_type (*)[]))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __device_accessor(__device_accessor<_OtherElementType>) noexcept(
    __is_copy_ctor_noexcept)
  {}

  _CCCL_TEMPLATE(typename _OtherElementType)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _OtherElementType (*)[], element_type (*)[]))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __device_accessor(__managed_accessor<_OtherElementType>) noexcept(
    __is_copy_ctor_noexcept)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference access(data_handle_type __p, _CUDA_VSTD::size_t __i) const
    noexcept(__is_access_noexcept)
  {
    NV_IF_TARGET(NV_IS_HOST, (__self::__prevent_host_instantiation();))
    return _Accessor::access(__p, __i);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr data_handle_type offset(data_handle_type __p, _CUDA_VSTD::size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    NV_IF_TARGET(NV_IS_HOST, (__self::__prevent_host_instantiation();))
    return _Accessor::offset(__p, __i);
  }
};

/***********************************************************************************************************************
 * Managed Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
struct __managed_accessor : public _Accessor
{
  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::__host_accessor/cuda::__device_accessor/cuda::__managed_accessor cannot be nested");

  using offset_policy    = __managed_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = typename _Accessor::data_handle_type;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

private:
  using __self = __managed_accessor<_Accessor>;

  static constexpr bool __is_ctor_noexcept      = noexcept(_Accessor{});
  static constexpr bool __is_copy_ctor_noexcept = noexcept(_Accessor{_Accessor{}});
  static constexpr bool __is_access_noexcept    = noexcept(_Accessor{}.access(data_handle_type{}, 0));
  static constexpr bool __is_offset_noexcept    = noexcept(_Accessor{}.offset(data_handle_type{}, 0));

  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::__host_accessor/cuda::__device_accessor/cuda::__managed_accessor cannot be nested");

  template <typename data_handle_type>
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __is_managed_pointer(data_handle_type __p) noexcept
  {
    if constexpr (_CUDA_VSTD::is_pointer_v<data_handle_type>)
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

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __check_managed_pointer(data_handle_type __p) noexcept
  {
    _CCCL_ASSERT(__self::__is_managed_pointer(__p), "cuda::__managed_accessor data handle is not a MANAGED pointer");
  }

public:
  _CCCL_TEMPLATE(typename _NotUsed = void)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_default_constructible, _Accessor))
  _LIBCUDACXX_HIDE_FROM_ABI __managed_accessor() noexcept(__is_ctor_noexcept)
      : _Accessor{}
  {}

  _CCCL_TEMPLATE(typename _OtherElementType)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _OtherElementType (*)[], element_type (*)[]))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __managed_accessor(__managed_accessor<_OtherElementType>) noexcept(
    __is_copy_ctor_noexcept)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference access(data_handle_type __p, _CUDA_VSTD::size_t __i) const
    noexcept(__is_access_noexcept)
  {
    NV_IF_TARGET(NV_IS_HOST, (__self::__check_managed_pointer(__p);))
    return _Accessor::access(__p, __i);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr data_handle_type offset(data_handle_type __p, _CUDA_VSTD::size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    NV_IF_TARGET(NV_IS_HOST, (__self::__check_managed_pointer(__p);))
    return _Accessor::offset(__p, __i);
  }
};

/***********************************************************************************************************************
 * Accessibility Traits
 **********************************************************************************************************************/

template <typename>
inline constexpr bool is_host_accessible_v = false;

template <typename>
inline constexpr bool is_device_accessible_v = false;

template <typename _Accessor>
inline constexpr bool is_host_accessible_v<__host_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_host_accessible_v<__managed_accessor<_Accessor>> = true;

template <template <typename> class _TClass, typename _Accessor>
inline constexpr bool is_host_accessible_v<_TClass<_Accessor>> = is_host_accessible_v<_Accessor>;

template <typename _Accessor>
inline constexpr bool is_device_accessible_v<__device_accessor<_Accessor>> = true;

template <typename _Accessor>
inline constexpr bool is_device_accessible_v<__managed_accessor<_Accessor>> = true;

template <template <typename> class _TClass, typename _Accessor>
inline constexpr bool is_device_accessible_v<_TClass<_Accessor>> = is_device_accessible_v<_Accessor>;

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___MDSPAN_HOST_DEVICE_ACCESSOR
