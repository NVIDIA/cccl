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

#if !_CCCL_HAS_CUDA_COMPILER || _CCCL_CUDA_COMPILER(CLANG)
#  include <cuda_runtime_api.h>
#endif // _CCCL_CUDA_COMPILER(CLANG)

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Accessor>
class __host_accessor;

template <typename _Accessor>
class __device_accessor;

template <typename _Accessor>
class __managed_accessor;

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
class __host_accessor : public _Accessor
{
  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::__host_accessor/cuda::__device_accessor/cuda::__managed_accessor cannot be nested");

  using __data_handle_type = typename _Accessor::data_handle_type;

  static constexpr bool __is_access_noexcept =
    noexcept(_CUDA_VSTD::declval<_Accessor>().access(_CUDA_VSTD::declval<__data_handle_type>(), 0));

  static constexpr bool __is_offset_noexcept =
    noexcept(_CUDA_VSTD::declval<_Accessor>().offset(_CUDA_VSTD::declval<__data_handle_type>(), 0));

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool
  __is_host_accessible_pointer([[maybe_unused]] __data_handle_type __p) noexcept
  {
    if constexpr (_CUDA_VSTD::contiguous_iterator<__data_handle_type>)
    {
      ::cudaPointerAttributes __ptr_attrib{};
      auto __p1 = _CUDA_VSTD::to_address(__p);
      _CCCL_ASSERT_CUDA_API(::cudaPointerGetAttributes, "cudaPointerGetAttributes failed", &__ptr_attrib, __p1);
      return __ptr_attrib.hostPointer != nullptr || __ptr_attrib.type == ::cudaMemoryTypeUnregistered;
    }
    else
    {
      return true; // cannot be verified
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __check_host_pointer([[maybe_unused]] __data_handle_type __p) noexcept
  {
    _CCCL_ASSERT(__is_host_accessible_pointer(__p), "cuda::__host_accessor data handle is not a HOST pointer");
  }

public:
  using offset_policy    = __host_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = __data_handle_type;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

  _CCCL_TEMPLATE(class _Accessor2 = _Accessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_default_constructible, _Accessor2))
  _LIBCUDACXX_HIDE_FROM_ABI __host_accessor() noexcept(_CUDA_VSTD::is_nothrow_default_constructible_v<_Accessor2>)
      : _Accessor{}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __host_accessor(const _Accessor& __acc) noexcept(
    _CUDA_VSTD::is_nothrow_copy_constructible_v<_Accessor>)
      : _Accessor{__acc}
  {}

  template <typename _OtherAccessor>
  __host_accessor(const __device_accessor<_OtherAccessor>&) = delete;

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __host_accessor(const __host_accessor<_OtherAccessor>& __acc) noexcept(
    noexcept(_Accessor{_CUDA_VSTD::declval<_OtherAccessor>()}))
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(!_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __host_accessor(const __host_accessor<_OtherAccessor>& __acc) noexcept(
    noexcept(_Accessor{_CUDA_VSTD::declval<_OtherAccessor>()}))
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __host_accessor(const __managed_accessor<_OtherAccessor>& __acc) noexcept(
    noexcept(_Accessor{_CUDA_VSTD::declval<_OtherAccessor>()}))
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(!_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __host_accessor(
    const __managed_accessor<_OtherAccessor>& __acc) noexcept(noexcept(_Accessor{_CUDA_VSTD::declval<_OtherAccessor>()}))
      : _Accessor{__acc}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference access(data_handle_type __p, size_t __i) const
    noexcept(__is_access_noexcept)
  {
#if _CCCL_HOST_COMPILATION()
    __check_host_pointer(__p);
#else // ^^^ _CCCL_HOST_COMPILATION() ^^^ // vvv !_CCCL_HOST_COMPILATION() vvv
    static_assert(false, "cuda::__host_accessor cannot be used in DEVICE code");
#endif // !_CCCL_HOST_COMPILATION()
    return _Accessor::access(__p, __i);
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr data_handle_type offset(data_handle_type __p, size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    return _Accessor::offset(__p, __i);
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  __detectably_invalid([[maybe_unused]] data_handle_type __p, size_t) const noexcept
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __is_host_accessible_pointer(__p);), (return false;))
  }
};

/***********************************************************************************************************************
 * Device Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
class __device_accessor : public _Accessor
{
  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::__host_accessor/cuda::__device_accessor/cuda::__managed_accessor cannot be nested");

  using __data_handle_type = typename _Accessor::data_handle_type;

  static constexpr bool __is_access_noexcept =
    noexcept(_CUDA_VSTD::declval<_Accessor>().access(_CUDA_VSTD::declval<__data_handle_type>(), 0));

  static constexpr bool __is_offset_noexcept =
    noexcept(_CUDA_VSTD::declval<_Accessor>().offset(_CUDA_VSTD::declval<__data_handle_type>(), 0));

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool
  __is_device_accessible_pointer([[maybe_unused]] __data_handle_type __p) noexcept
  {
    if constexpr (_CUDA_VSTD::contiguous_iterator<__data_handle_type>)
    {
      ::cudaPointerAttributes __ptr_attrib{};
      auto __p1 = _CUDA_VSTD::to_address(__p);
      _CCCL_ASSERT_CUDA_API(::cudaPointerGetAttributes, "cudaPointerGetAttributes failed", &__ptr_attrib, __p1);
      return __ptr_attrib.devicePointer != nullptr || __ptr_attrib.type == ::cudaMemoryTypeUnregistered;
    }
    else
    {
      return true; // cannot be verified
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void
  __check_device_pointer([[maybe_unused]] __data_handle_type __p) noexcept
  {
    _CCCL_ASSERT(__is_device_accessible_pointer(__p), "cuda::__host_accessor data handle is not a HOST pointer");
  }

  template <typename _Sp = bool> // lazy evaluation
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void __prevent_host_instantiation() noexcept
  {
    static_assert(_CUDA_VSTD::__always_false_v<_Sp>, "cuda::__device_accessor cannot be used in HOST code");
  }

public:
  using offset_policy    = __device_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = __data_handle_type;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

  _CCCL_TEMPLATE(typename _NotUsed = void)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_default_constructible, _Accessor))
  _LIBCUDACXX_HIDE_FROM_ABI __device_accessor() noexcept(_CUDA_VSTD::is_nothrow_default_constructible_v<_Accessor>)
      : _Accessor{}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __device_accessor(const _Accessor& __acc) noexcept(
    _CUDA_VSTD::is_nothrow_copy_constructible_v<_Accessor>)
      : _Accessor{__acc}
  {}

  template <typename _OtherAccessor>
  __device_accessor(const __host_accessor<_OtherAccessor>&) = delete;

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __device_accessor(const __device_accessor<_OtherAccessor>& __acc) noexcept(
    _CUDA_VSTD::is_nothrow_copy_constructible_v<_Accessor>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(!_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __device_accessor(
    const __device_accessor<_OtherAccessor>& __acc) noexcept(_CUDA_VSTD::is_nothrow_copy_constructible_v<_Accessor>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __device_accessor(const __managed_accessor<_OtherAccessor>& __acc) noexcept(
    noexcept(_Accessor{_CUDA_VSTD::declval<_OtherAccessor>()}))
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(!_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __device_accessor(
    const __managed_accessor<_OtherAccessor>& __acc) noexcept(noexcept(_Accessor{_CUDA_VSTD::declval<_OtherAccessor>()}))
      : _Accessor{__acc}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference access(data_handle_type __p, size_t __i) const
    noexcept(__is_access_noexcept)
  {
#if _CCCL_HOST_COMPILATION()
    __prevent_host_instantiation();
#endif // _CCCL_HOST_COMPILATION()
    return _Accessor::access(__p, __i);
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr data_handle_type offset(data_handle_type __p, size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    return _Accessor::offset(__p, __i);
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  __detectably_invalid(data_handle_type __p, size_t) const noexcept
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __is_device_accessible_pointer(__p);), (return false;))
  }
};

/***********************************************************************************************************************
 * Managed Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
class __managed_accessor : public _Accessor
{
  static_assert(!is_host_device_managed_accessor_v<_Accessor>,
                "cuda::__host_accessor/cuda::__device_accessor/cuda::__managed_accessor cannot be nested");

  using __data_handle_type = typename _Accessor::data_handle_type;

  static constexpr bool __is_access_noexcept =
    noexcept(_CUDA_VSTD::declval<_Accessor>().access(_CUDA_VSTD::declval<__data_handle_type>(), 0));

  static constexpr bool __is_offset_noexcept =
    noexcept(_CUDA_VSTD::declval<_Accessor>().offset(_CUDA_VSTD::declval<__data_handle_type>(), 0));

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool
  __is_managed_pointer([[maybe_unused]] __data_handle_type __p) noexcept
  {
    if constexpr (_CUDA_VSTD::contiguous_iterator<__data_handle_type>)
    {
      ::cudaPointerAttributes __ptr_attrib{};
      auto __p1 = _CUDA_VSTD::to_address(__p);
      _CCCL_ASSERT_CUDA_API(::cudaPointerGetAttributes, "cudaPointerGetAttributes failed", &__ptr_attrib, __p1);
      return __ptr_attrib.devicePointer != nullptr && __ptr_attrib.hostPointer == __ptr_attrib.devicePointer;
    }
    else
    {
      return true; // cannot be verified
    }
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr void
  __check_managed_pointer([[maybe_unused]] __data_handle_type __p) noexcept
  {
    _CCCL_ASSERT(__is_managed_pointer(__p), "cuda::__managed_accessor data handle is not a MANAGED pointer");
  }

public:
  using offset_policy    = __managed_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = __data_handle_type;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

  _CCCL_TEMPLATE(typename _NotUsed = void)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_default_constructible, _Accessor))
  _LIBCUDACXX_HIDE_FROM_ABI __managed_accessor() noexcept(_CUDA_VSTD::is_nothrow_default_constructible_v<_Accessor>)
      : _Accessor{}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __managed_accessor(const _Accessor& __acc) noexcept(
    _CUDA_VSTD::is_nothrow_copy_constructible_v<_Accessor>)
      : _Accessor{__acc}
  {}

  template <typename _OtherAccessor>
  __managed_accessor(const __host_accessor<_OtherAccessor>&) = delete;

  template <typename _OtherAccessor>
  __managed_accessor(const __device_accessor<_OtherAccessor>&) = delete;

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __managed_accessor(const __managed_accessor<_OtherAccessor>& __acc) noexcept(
    noexcept(_Accessor{_CUDA_VSTD::declval<_OtherAccessor>()}))
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(!_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __managed_accessor(
    const __managed_accessor<_OtherAccessor>& __acc) noexcept(noexcept(_Accessor{_CUDA_VSTD::declval<_OtherAccessor>()}))
      : _Accessor{__acc}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference access(data_handle_type __p, size_t __i) const
    noexcept(__is_access_noexcept)
  {
    NV_IF_TARGET(NV_IS_HOST, (__check_managed_pointer(__p);))
    return _Accessor::access(__p, __i);
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr data_handle_type offset(data_handle_type __p, size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    return _Accessor::offset(__p, __i);
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
  __detectably_invalid([[maybe_unused]] data_handle_type __p, size_t) const noexcept
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __is_managed_pointer(__p);), (return false;))
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
