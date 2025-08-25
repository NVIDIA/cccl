//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ANNOTATED_PTR_ANNOTATED_PTR_H
#define _CUDA___ANNOTATED_PTR_ANNOTATED_PTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__annotated_ptr/access_property.h>
#include <cuda/__annotated_ptr/annotated_ptr_base.h>
#include <cuda/__memcpy_async/memcpy_async.h>
#include <cuda/__memory/address_space.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp, typename _Property>
class annotated_ptr : private ::cuda::__annotated_ptr_base<_Property>
{
public:
  using value_type      = _Tp;
  using size_type       = size_t;
  using reference       = value_type&;
  using pointer         = value_type*;
  using const_pointer   = const value_type*;
  using difference_type = ptrdiff_t;

private:
  static_assert(__is_access_property_v<_Property>);

  static constexpr bool __is_smem = ::cuda::std::is_same_v<_Property, access_property::shared>;

  // Converting from a 64-bit to 32-bit shared pointer and maybe back just for storage might or might not be profitable.
  pointer __repr = nullptr;

  [[nodiscard]] _CCCL_API inline pointer __get(difference_type __n = 0) const noexcept
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (auto __repr1 = const_cast<void*>(static_cast<const volatile void*>(__repr + __n));
                  return static_cast<pointer>(this->__apply_prop(__repr1));))
    return __repr + __n;
  }

  [[nodiscard]] _CCCL_API inline pointer __offset(difference_type __n) const noexcept
  {
    return __get(__n);
  }

public:
  _CCCL_HIDE_FROM_ABI annotated_ptr() noexcept = default;

  _CCCL_API explicit constexpr annotated_ptr(pointer __p) noexcept
      : __repr{__p}
  {
    NV_IF_TARGET(NV_IS_HOST, (_CCCL_ASSERT(!__is_smem, "shared memory pointer is not supported on the host");))
    if constexpr (__is_smem)
    {
      if (!::cuda::std::__cccl_default_is_constant_evaluated())
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (_CCCL_ASSERT(::cuda::device::is_address_from(__p, ::cuda::device::address_space::shared),
                                   "__p must be shared");))
      }
    }
    else
    {
      _CCCL_ASSERT(__p != nullptr, "__p must not be null");
      if (!::cuda::std::__cccl_default_is_constant_evaluated())
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (_CCCL_ASSERT(::cuda::device::is_address_from(__p, ::cuda::device::address_space::global),
                                   "__p must be global");))
      }
    }
  }

  template <typename _RuntimeProperty>
  _CCCL_API inline annotated_ptr(pointer __p, _RuntimeProperty __prop) noexcept
      : ::cuda::__annotated_ptr_base<_Property>{access_property{__prop}}
      , __repr{__p}
  {
    static_assert(::cuda::std::is_same_v<_Property, access_property>,
                  "This method requires annotated_ptr<T, cuda::access_property>");
    static_assert(__is_global_access_property_v<_RuntimeProperty>,
                  "This method requires RuntimeProperty=global|normal|streaming|persisting|access_property");
    _CCCL_ASSERT(__p != nullptr, "__p must not be null");
    NV_IF_TARGET(NV_IS_DEVICE,
                 (_CCCL_ASSERT(::cuda::device::is_address_from(__p, ::cuda::device::address_space::global),
                               "__p must be global");))
  }

  // cannot be constexpr because of get()
  template <typename _OtherType, class _OtherProperty>
  _CCCL_API inline annotated_ptr(const annotated_ptr<_OtherType, _OtherProperty>& __other) noexcept
      : ::cuda::__annotated_ptr_base<_Property>{__other.__property()}
      , __repr{__other.get()}
  {
    using namespace ::cuda::std;
    static_assert(is_assignable_v<pointer&, _OtherType*>, "pointer must be assignable from other pointer");
    static_assert(is_same_v<_Property, _OtherProperty>
                    || (is_same_v<_Property, access_property> && !is_same_v<_OtherProperty, access_property::shared>),
                  "Both properties must have same address space, or current property is access_property and "
                  "OtherProperty is not shared");
  }

  // cannot be constexpr because is_constant_evaluated is not supported by clang-14, gcc-8.
  // when the method is called in these platforms, it needs to be called at run-time.
  [[nodiscard]] _CCCL_API inline pointer operator->() const noexcept
  {
    return __get();
  }

  [[nodiscard]] _CCCL_API inline reference operator*() const noexcept
  {
    _CCCL_ASSERT(__get() != nullptr, "dereference of null annotated_ptr");
    return *__get();
  }

  [[nodiscard]] _CCCL_API inline reference operator[](difference_type __n) const noexcept
  {
    _CCCL_ASSERT(__offset(__n) != nullptr, "dereference of null annotated_ptr");
    return *__offset(__n);
  }

  [[nodiscard]] _CCCL_API constexpr difference_type operator-(annotated_ptr __other) const noexcept
  {
    _CCCL_ASSERT(__repr >= __other.__repr, "underflow");
    return __repr - __other.__repr;
  }

  [[nodiscard]] _CCCL_API constexpr explicit operator bool() const noexcept
  {
    return (__repr != nullptr);
  }

  // cannot be constexpr because of operator->()
  [[nodiscard]] _CCCL_API inline pointer get() const noexcept
  {
    return (__is_smem || __repr == nullptr)
           ? __repr
           : annotated_ptr<value_type, access_property::global>{__repr}.operator->();
  }

  [[nodiscard]] _CCCL_API constexpr _Property __property() const noexcept
  {
    return this->__get_property();
  }
};

//----------------------------------------------------------------------------------------------------------------------
// memcpy_async

template <typename _Dst, typename _Src, typename _SrcProperty, typename _Shape, typename _Sync>
_CCCL_API inline void
memcpy_async(_Dst* __dst, annotated_ptr<_Src, _SrcProperty> __src, _Shape __shape, _Sync& __sync) noexcept
{
  ::cuda::memcpy_async(__dst, __src.operator->(), __shape, __sync);
}

template <typename _Dst, typename _DstProperty, typename _Src, typename _SrcProperty, typename _Shape, typename _Sync>
_CCCL_API inline void memcpy_async(
  annotated_ptr<_Dst, _DstProperty> __dst,
  annotated_ptr<_Src, _SrcProperty> __src,
  _Shape __shape,
  _Sync& __sync) noexcept
{
  ::cuda::memcpy_async(__dst.operator->(), __src.operator->(), __shape, __sync);
}

template <typename _Group, typename _Dst, typename _Src, typename _SrcProperty, typename _Shape, typename _Sync>
_CCCL_API inline void memcpy_async(
  const _Group& __group, _Dst* __dst, annotated_ptr<_Src, _SrcProperty> __src, _Shape __shape, _Sync& __sync) noexcept
{
  ::cuda::memcpy_async(__group, __dst, __src.operator->(), __shape, __sync);
}

template <typename _Group,
          typename _Dst,
          typename _DstProperty,
          typename _Src,
          typename _SrcProperty,
          typename _Shape,
          typename _Sync>
_CCCL_API inline void memcpy_async(
  const _Group& __group,
  annotated_ptr<_Dst, _DstProperty> __dst,
  annotated_ptr<_Src, _SrcProperty> __src,
  _Shape __shape,
  _Sync& __sync) noexcept
{
  ::cuda::memcpy_async(__group, __dst.operator->(), __src.operator->(), __shape, __sync);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ANNOTATED_PTR_ANNOTATED_PTR_H
