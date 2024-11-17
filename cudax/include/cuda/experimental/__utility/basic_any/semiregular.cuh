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

#include <cuda/std/__concepts/__concept_macros.h>
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

#if defined(_CCCL_CUDA_COMPILER_NVCC) || defined(_CCCL_CUDA_COMPILER_NVHPC)
// WAR for NVBUG #4924416
#  define _CUDAX_FNPTR_CONSTANT_WAR(...) ::cuda::experimental::__constant_war(__VA_ARGS__)
namespace cuda::experimental
{
template <class _Tp>
_CCCL_NODISCARD _CUDAX_API constexpr _Tp __constant_war(_Tp __val) noexcept
{
  return __val;
}
} // namespace cuda::experimental
#else
#  define _CUDAX_FNPTR_CONSTANT_WAR(...) __VA_ARGS__
#endif

namespace cuda::experimental
{
///
/// semi-regular overrides
///

_LIBCUDACXX_TEMPLATE(class _Tp)
_LIBCUDACXX_REQUIRES(_CUDA_VSTD::movable<_Tp>)
_CUDAX_PUBLIC_API void __move_fn(_Tp& __src, void* __dst) noexcept
{
  ::new (__dst) _Tp(static_cast<_Tp&&>(__src));
}

_LIBCUDACXX_TEMPLATE(class _Tp)
_LIBCUDACXX_REQUIRES(_CUDA_VSTD::movable<_Tp>)
_CCCL_NODISCARD _CUDAX_PUBLIC_API bool __try_move_fn(_Tp& __src, void* __dst, size_t __size, size_t __align)
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

_LIBCUDACXX_TEMPLATE(class _Tp)
_LIBCUDACXX_REQUIRES(_CUDA_VSTD::copyable<_Tp>)
_CCCL_NODISCARD _CUDAX_PUBLIC_API bool __copy_fn(_Tp const& __src, void* __dst, size_t __size, size_t __align)
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

_LIBCUDACXX_TEMPLATE(class _Tp)
_LIBCUDACXX_REQUIRES(_CUDA_VSTD::equality_comparable<_Tp>)
_CCCL_NODISCARD _CUDAX_PUBLIC_API bool
__equal_fn(_Tp const& __self, _CUDA_VSTD::__type_info_ref __type, void const* __other)
{
  if (_CCCL_TYPEID(_Tp) == __type)
  {
    return __self == *static_cast<_Tp const*>(__other);
  }
  return false;
}

///
/// semi-regular interfaces
///
template <class...>
struct imovable : interface<imovable>
{
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::movable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS =
    overrides_for<_Tp, _CUDAX_FNPTR_CONSTANT_WAR(&__try_move_fn<_Tp>), _CUDAX_FNPTR_CONSTANT_WAR(&__move_fn<_Tp>)>;

  _CUDAX_API void __move_to(void* __pv) noexcept
  {
    return __cudax::virtcall<&__move_fn<imovable>>(this, __pv);
  }

  _CCCL_NODISCARD _CUDAX_API bool __move_to(void* __pv, size_t __size, size_t __align)
  {
    return __cudax::virtcall<&__try_move_fn<imovable>>(this, __pv, __size, __align);
  }
};

template <class...>
struct icopyable : interface<icopyable, extends<imovable<>>>
{
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::copyable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Tp, _CUDAX_FNPTR_CONSTANT_WAR(&__copy_fn<_Tp>)>;

  _CCCL_NODISCARD _CUDAX_API bool __copy_to(void* __pv, size_t __size, size_t __align) const
  {
    return virtcall<&__copy_fn<icopyable>>(this, __pv, __size, __align);
  }
};

template <class...>
struct iequality_comparable : interface<iequality_comparable>
{
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::equality_comparable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Tp, _CUDAX_FNPTR_CONSTANT_WAR(&__equal_fn<_Tp>)>;

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
  _CCCL_NODISCARD _CUDAX_API bool operator==(iequality_comparable const& __other) const
  {
    auto const& __other = __cudax::basic_any_from(__other);
    void const* __obj   = __basic_any_access::__get_optr(__other);
    return __cudax::virtcall<&__equal_fn<iequality_comparable>>(this, __other.type(), __obj);
  }
#else
  _CCCL_NODISCARD_FRIEND _CUDAX_API bool
  operator==(iequality_comparable const& __left, iequality_comparable const& __right)
  {
    auto const& __rhs = __cudax::basic_any_from(__right);
    void const* __obj = __basic_any_access::__get_optr(__rhs);
    return __cudax::virtcall<&__equal_fn<iequality_comparable>>(&__left, __rhs.type(), __obj);
  }

  _CCCL_NODISCARD_FRIEND _CUDAX_TRIVIAL_API bool
  operator!=(iequality_comparable const& __left, iequality_comparable const& __right)
  {
    return !(__left == __right);
  }
#endif
};

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_SEMIREGULAR_H
