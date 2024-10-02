//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MDSPAN_CACHED_ACCESSORS_H
#define _CUDA__MDSPAN_CACHED_ACCESSORS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2014

#  include <cub/thread/thread_load.cuh>
#  include <cub/thread/thread_store.cuh>

#  include <cuda/std/__concepts/__concept_macros.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/is_abstract.h>
#  include <cuda/std/__type_traits/is_array.h>
#  include <cuda/std/__type_traits/is_convertible.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

enum class EvictionPolicy
{
  Default,
  First,
  Normal,
  Last,
  LastUse,
  NoAllocation
};

enum class PrefetchSize
{
  NoPrefetch,
  Bytes64,
  Bytes128,
  Bytes256
};

template <class _ElementType,
          EvictionPolicy _Eviction = EvictionPolicy::Default,
          PrefetchSize _Prefetch   = PrefetchSize::NoPrefetch,
          typename _Enable         = void>
struct cache_policy_accessor;

/***********************************************************************************************************************
 * accessor_reference
 **********************************************************************************************************************/

template <class _ElementType, EvictionPolicy _Eviction, PrefetchSize _Prefetch>
class accessor_reference
{
  using pointer_type = _ElementType*;

  pointer_type __p;

  friend class cache_policy_accessor<_ElementType, _Eviction, _Prefetch>;

public:
  accessor_reference() = delete;

  accessor_reference(accessor_reference&&) = delete;

  accessor_reference& operator=(accessor_reference&&) = delete;

  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CCCL_FORCEINLINE accessor_reference(const accessor_reference&) = default;

  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CCCL_FORCEINLINE accessor_reference&
  operator=(const accessor_reference& __x) noexcept
  {
    return operator=(static_cast<_ElementType>(__x));
  }

  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CCCL_FORCEINLINE accessor_reference& operator=(_ElementType __x) noexcept
  {
    return cub::ThreadStore<_Eviction>(__p, __x);
  }

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE operator _ElementType() const noexcept
  {
    return cub::ThreadLoad<_Eviction, _Prefetch>(__p);
  }

private:
  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CCCL_FORCEINLINE explicit accessor_reference(pointer_type __p_) noexcept
      : __p{__p_}
  {}
};

/***********************************************************************************************************************
 * load/store cache_policy_accessor
 **********************************************************************************************************************/

template <class _ElementType, EvictionPolicy _Eviction, PrefetchSize _Prefetch>
struct cache_policy_accessor<_ElementType,
                             _Eviction,
                             _Prefetch,
                             _CUDA_VSTD::__enable_if_t<!_CUDA_VSTD::is_const<_ElementType>::value>>
{
  static_assert(!_CUDA_VSTD::is_array<_ElementType>::value,
                "cache_policy_accessor: template argument may not be an array type");
  static_assert(!_CUDA_VSTD::is_abstract<_ElementType>::value,
                "cache_policy_accessor: template argument may not be an abstract class");

  using offset_policy    = cache_policy_accessor;
  using element_type     = _ElementType;
  using reference        = ::cuda::accessor_reference<_ElementType, _Eviction, _Prefetch>;
  using data_handle_type = _ElementType*;

  constexpr cache_policy_accessor() noexcept = default;

  _LIBCUDACXX_TEMPLATE(class _OtherElementType)
  _LIBCUDACXX_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _OtherElementType (*)[], element_type (*)[]))
  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE
  _CCCL_FORCEINLINE constexpr cache_policy_accessor(cache_policy_accessor<_OtherElementType>) noexcept
  {}

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE reference access(data_handle_type __p, size_t __i) const noexcept
  {
    return reference{__p + __i};
  }

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE data_handle_type offset(data_handle_type __p, size_t __i) const noexcept
  {
    return __p + __i;
  }
};

/***********************************************************************************************************************
 * load-only cache_policy_accessor
 **********************************************************************************************************************/

template <class _ElementType, EvictionPolicy _Eviction, PrefetchSize _Prefetch>
struct cache_policy_accessor<_ElementType,
                             _Eviction,
                             _Prefetch,
                             _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::is_const<_ElementType>::value>>
{
  static_assert(!_CUDA_VSTD::is_array<_ElementType>::value,
                "cache_policy_accessor: template argument may not be an array type");
  static_assert(!_CUDA_VSTD::is_abstract<_ElementType>::value,
                "cache_policy_accessor: template argument may not be an abstract class");

  using offset_policy    = cache_policy_accessor;
  using element_type     = _ElementType;
  using reference        = _ElementType&;
  using data_handle_type = _ElementType*;

  explicit cache_policy_accessor() noexcept = default;

  template <typename _OtherElementType,
            _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::is_convertible<_OtherElementType (*)[], _ElementType (*)[]>::value>>
  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE
  _CCCL_FORCEINLINE constexpr cache_policy_accessor(cache_policy_accessor<_OtherElementType>) noexcept
  {}

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE element_type access(data_handle_type __p, size_t __i) const noexcept
  {
    return cub::ThreadLoad<_Eviction, _Prefetch>(__p + __i);
  }

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE data_handle_type offset(data_handle_type __p, size_t __i) const noexcept
  {
    return __p + __i;
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2014
#endif // _CUDA__MDSPAN_CACHED_ACCESSORS_H
