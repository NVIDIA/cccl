//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_SEMIREGULAR_H
#define __CUDAX_DETAIL_BASIC_ANY_SEMIREGULAR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/movable.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/experimental/__utility/basic_any/access.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_from.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>
#include <cuda/experimental/__utility/basic_any/storage.cuh>
#include <cuda/experimental/__utility/basic_any/virtcall.cuh>

_CCCL_PUSH_MACROS
#undef interface

#if defined(_CCCL_CUDA_COMPILER_NVCC) || defined(_CCCL_CUDA_COMPILER_NVHPC)
// WAR for NVBUG #4924416
#  define _CUDAX_FNPTR_CONSTANT_WAR(...) ::cuda::experimental::__constant_war(__VA_ARGS__)
namespace cuda::experimental
{
template <class _Tp>
_CCCL_NODISCARD _CUDAX_HOST_API constexpr _Tp __constant_war(_Tp __val) noexcept
{
  return __val;
}
} // namespace cuda::experimental
#else // ^^^ defined(_CCCL_CUDA_COMPILER_NVCC) || defined(_CCCL_CUDA_COMPILER_NVHPC) ^^^ /
      // vvv !defined(_CCCL_CUDA_COMPILER_NVCC) && !defined(_CCCL_CUDA_COMPILER_NVHPC) vvv
#  define _CUDAX_FNPTR_CONSTANT_WAR(...) __VA_ARGS__
#endif // !defined(_CCCL_CUDA_COMPILER_NVCC) && !defined(_CCCL_CUDA_COMPILER_NVHPC)

namespace cuda::experimental
{
//!
//! semi-regular overrides
//!

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CUDA_VSTD::movable<_Tp>)
_CUDAX_PUBLIC_API auto __move_fn(_Tp& __src, void* __dst) noexcept -> void
{
  ::new (__dst) _Tp(static_cast<_Tp&&>(__src));
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CUDA_VSTD::movable<_Tp>)
_CCCL_NODISCARD _CUDAX_PUBLIC_API auto __try_move_fn(_Tp& __src, void* __dst, size_t __size, size_t __align) -> bool
{
  if (__is_small<_Tp>(__size, __align))
  {
    ::new (__dst) _Tp(static_cast<_Tp&&>(__src));
    return true;
  }
  else
  {
    ::new (__dst) __identity_t<_Tp*>(new _Tp(static_cast<_Tp&&>(__src)));
    return false;
  }
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CUDA_VSTD::copyable<_Tp>)
_CCCL_NODISCARD _CUDAX_PUBLIC_API auto __copy_fn(_Tp const& __src, void* __dst, size_t __size, size_t __align) -> bool
{
  if (__is_small<_Tp>(__size, __align))
  {
    ::new (__dst) _Tp(__src);
    return true;
  }
  else
  {
    ::new (__dst) __identity_t<_Tp*>(new _Tp(__src));
    return false;
  }
}

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(_CUDA_VSTD::equality_comparable<_Tp>)
_CCCL_NODISCARD _CUDAX_PUBLIC_API auto
__equal_fn(_Tp const& __self, _CUDA_VSTD::__type_info_ref __type, void const* __other) -> bool
{
  if (_CCCL_TYPEID(_Tp) == __type)
  {
    return __self == *static_cast<_Tp const*>(__other);
  }
  return false;
}

//!
//! semi-regular interfaces
//!
template <class...>
struct imovable : interface<imovable>
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CUDA_VSTD::movable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS =
    overrides_for<_Tp, _CUDAX_FNPTR_CONSTANT_WAR(&__try_move_fn<_Tp>), _CUDAX_FNPTR_CONSTANT_WAR(&__move_fn<_Tp>)>;

  _CUDAX_HOST_API auto __move_to(void* __pv) noexcept -> void
  {
    return __cudax::virtcall<&__move_fn<imovable>>(this, __pv);
  }

  _CCCL_NODISCARD _CUDAX_HOST_API auto __move_to(void* __pv, size_t __size, size_t __align) -> bool
  {
    return __cudax::virtcall<&__try_move_fn<imovable>>(this, __pv, __size, __align);
  }
};

template <class...>
struct icopyable : interface<icopyable, extends<imovable<>>>
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CUDA_VSTD::copyable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Tp, _CUDAX_FNPTR_CONSTANT_WAR(&__copy_fn<_Tp>)>;

  _CCCL_NODISCARD _CUDAX_HOST_API auto __copy_to(void* __pv, size_t __size, size_t __align) const -> bool
  {
    return virtcall<&__copy_fn<icopyable>>(this, __pv, __size, __align);
  }
};

template <class...>
struct iequality_comparable : interface<iequality_comparable>
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Tp, _CUDAX_FNPTR_CONSTANT_WAR(&__equal_fn<_Tp>)>;

#if !defined(_CCCL_NO_THREE_WAY_COMPARISON)
  _CCCL_NODISCARD _CUDAX_HOST_API auto operator==(iequality_comparable const& __other) const -> bool
  {
    auto const& __rhs = __cudax::basic_any_from(__other);
    void const* __obj = __basic_any_access::__get_optr(__rhs);
    return __cudax::virtcall<&__equal_fn<iequality_comparable>>(this, __rhs.type(), __obj);
  }
#else // ^^^ !_CCCL_NO_THREE_WAY_COMPARISON ^^^ / vvv _CCCL_NO_THREE_WAY_COMPARISON vvv
  _CCCL_NODISCARD_FRIEND _CUDAX_HOST_API auto
  operator==(iequality_comparable const& __left, iequality_comparable const& __right) -> bool
  {
    auto const& __rhs = __cudax::basic_any_from(__right);
    void const* __obj = __basic_any_access::__get_optr(__rhs);
    return __cudax::virtcall<&__equal_fn<iequality_comparable>>(&__left, __rhs.type(), __obj);
  }

  _CCCL_NODISCARD_FRIEND _CUDAX_TRIVIAL_HOST_API auto
  operator!=(iequality_comparable const& __left, iequality_comparable const& __right) -> bool
  {
    return !(__left == __right);
  }
#endif // _CCCL_NO_THREE_WAY_COMPARISON
};

} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif // __CUDAX_DETAIL_BASIC_ANY_SEMIREGULAR_H
