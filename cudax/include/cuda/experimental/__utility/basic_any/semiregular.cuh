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
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/movable.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/typeid.h>
#include <cuda/std/__utility/unreachable.h>

#include <cuda/experimental/__utility/basic_any/access.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_base.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_from.cuh>
#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/conversions.cuh>
#include <cuda/experimental/__utility/basic_any/interfaces.cuh>
#include <cuda/experimental/__utility/basic_any/storage.cuh>
#include <cuda/experimental/__utility/basic_any/virtcall.cuh>

_CCCL_PUSH_MACROS
#undef interface

#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_COMPILER(NVHPC)
// WAR for NVBUG #4924416
#  define _CUDAX_FNPTR_CONSTANT_WAR(...) ::cuda::experimental::__constant_war(__VA_ARGS__)
namespace cuda::experimental
{
template <class _Tp>
[[nodiscard]] _CUDAX_HOST_API constexpr _Tp __constant_war(_Tp __val) noexcept
{
  return __val;
}
} // namespace cuda::experimental
#else // ^^^ _CCCL_CUDA_COMPILER(NVCC) || _CCCL_COMPILER(NVHPC) ^^^ /
      // vvv !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVHPC) vvv
#  define _CUDAX_FNPTR_CONSTANT_WAR(...) __VA_ARGS__
#endif // !_CCCL_CUDA_COMPILER(NVCC) && !_CCCL_COMPILER(NVHPC)

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
[[nodiscard]] _CUDAX_PUBLIC_API auto __try_move_fn(_Tp& __src, void* __dst, size_t __size, size_t __align) -> bool
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
[[nodiscard]] _CUDAX_PUBLIC_API auto __copy_fn(_Tp const& __src, void* __dst, size_t __size, size_t __align) -> bool
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
[[nodiscard]] _CUDAX_PUBLIC_API auto
__equal_fn(_Tp const& __self, _CUDA_VSTD::__type_info_ref __type, void const* __other) -> bool
{
  if (_CCCL_TYPEID(_Tp) == __type)
  {
    return __self == *static_cast<_Tp const*>(__other);
  }
  return false;
}

_CCCL_TEMPLATE(class _From, class _To)
_CCCL_REQUIRES(_CUDA_VSTD::convertible_to<_From, _To>)
[[nodiscard]] _CUDAX_PUBLIC_API _To __conversion_fn(_CUDA_VSTD::type_identity_t<_From> __self)
{
  return static_cast<_To>(static_cast<_From&&>(__self));
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

  [[nodiscard]] _CUDAX_HOST_API auto __move_to(void* __pv, size_t __size, size_t __align) -> bool
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

  [[nodiscard]] _CUDAX_HOST_API auto __copy_to(void* __pv, size_t __size, size_t __align) const -> bool
  {
    return virtcall<&__copy_fn<icopyable>>(this, __pv, __size, __align);
  }
};

template <class _Object>
_CCCL_CONCEPT __non_polymorphic = (!__is_basic_any<_Object>) && (!__is_interface<_Object>);

template <class... _Super>
struct iequality_comparable;

struct iequality_comparable_base : interface<iequality_comparable>
{
  // These overloads are only necessary so that iequality_comparable<> itself
  // satisfies the std::equality_comparable constraint that is used by the
  // `iequality_comparable<>::overloads` alias template below.
  [[noreturn]] friend _CUDAX_TRIVIAL_HOST_API auto
  operator==(iequality_comparable<> const&, iequality_comparable<> const&) noexcept -> bool
  {
    _CUDA_VSTD::unreachable();
  }

  [[noreturn]] friend _CUDAX_TRIVIAL_HOST_API auto
  operator!=(iequality_comparable<> const&, iequality_comparable<> const&) noexcept -> bool
  {
    _CUDA_VSTD::unreachable();
  }

  // These are the overloads that get used when testing two `basic_any` objects
  // for equality.
  _CCCL_TEMPLATE(class _ILeft, class _IRight)
  _CCCL_REQUIRES(__any_convertible_to<basic_any<_ILeft> const&, basic_any<_IRight> const&>
                 || __any_convertible_to<basic_any<_IRight> const&, basic_any<_ILeft> const&>)
  _CCCL_NODISCARD_FRIEND _CUDAX_HOST_API auto
  operator==(iequality_comparable<_ILeft> const& __lhs, iequality_comparable<_IRight> const& __rhs) noexcept -> bool
  {
    auto const& __other = __cudax::basic_any_from(__rhs);
    constexpr auto __eq = &__equal_fn<iequality_comparable<_ILeft>>;
    return __cudax::virtcall<__eq>(&__lhs, __other.type(), __basic_any_access::__get_optr(__other));
  }

  _CCCL_TEMPLATE(class _ILeft, class _IRight)
  _CCCL_REQUIRES(__any_convertible_to<basic_any<_ILeft> const&, basic_any<_IRight> const&>
                 || __any_convertible_to<basic_any<_IRight> const&, basic_any<_ILeft> const&>)
  _CCCL_NODISCARD_FRIEND _CUDAX_TRIVIAL_HOST_API auto
  operator!=(iequality_comparable<_ILeft> const& __lhs, iequality_comparable<_IRight> const& __rhs) noexcept -> bool
  {
    return !(__lhs == __rhs);
  }

  // These are the overloads that get used when testing a `basic_any` object
  // against a non-type-erased object.
  //
  // Q: Why require that the basic_any wrapper is not convertible to _Object?
  //
  // A: If there is a user-defined conversion from the basic_any type to _Object
  // such that we can use _Object's symmetric equality comparison operator, that
  // should be preferred. _Object may be another kind of wrapper object, in which
  // case using the address of the **wrapper** object for the comparison (as
  // opposed to the address of the wrapped object) is probably wrong.
  _CCCL_TEMPLATE(class _Interface, class _Object, class _Self = basic_any_from_t<iequality_comparable<_Interface>>)
  _CCCL_REQUIRES(__non_polymorphic<_Object> _CCCL_AND(!_CUDA_VSTD::convertible_to<_Self, _Object>)
                   _CCCL_AND __satisfies<_Object, _Interface>)
  _CCCL_NODISCARD_FRIEND _CUDAX_HOST_API auto
  operator==(iequality_comparable<_Interface> const& __lhs, _Object const& __rhs) -> bool
  {
    constexpr auto __eq = &__equal_fn<iequality_comparable<_Interface>>;
    return __cudax::virtcall<__eq>(&__lhs, _CCCL_TYPEID(_Object), _CUDA_VSTD::addressof(__rhs));
  }

  _CCCL_TEMPLATE(class _Interface, class _Object, class _Self = basic_any_from_t<iequality_comparable<_Interface>>)
  _CCCL_REQUIRES(__non_polymorphic<_Object> _CCCL_AND(!_CUDA_VSTD::convertible_to<_Self, _Object>)
                   _CCCL_AND __satisfies<_Object, _Interface>)
  _CCCL_NODISCARD_FRIEND _CUDAX_HOST_API auto
  operator==(_Object const& __lhs, iequality_comparable<_Interface> const& __rhs) noexcept -> bool
  {
    constexpr auto __eq = &__equal_fn<iequality_comparable<_Interface>>;
    return __cudax::virtcall<__eq>(&__rhs, _CCCL_TYPEID(_Object), _CUDA_VSTD::addressof(__lhs));
  }

  _CCCL_TEMPLATE(class _Interface, class _Object, class _Self = basic_any_from_t<iequality_comparable<_Interface>>)
  _CCCL_REQUIRES(__non_polymorphic<_Object> _CCCL_AND(!_CUDA_VSTD::convertible_to<_Self, _Object>)
                   _CCCL_AND __satisfies<_Object, _Interface>)
  _CCCL_NODISCARD_FRIEND _CUDAX_TRIVIAL_HOST_API auto
  operator!=(iequality_comparable<_Interface> const& __lhs, _Object const& __rhs) noexcept -> bool
  {
    return !(__lhs == __rhs);
  }

  _CCCL_TEMPLATE(class _Interface, class _Object, class _Self = basic_any_from_t<iequality_comparable<_Interface>>)
  _CCCL_REQUIRES(__non_polymorphic<_Object> _CCCL_AND(!_CUDA_VSTD::convertible_to<_Self, _Object>)
                   _CCCL_AND __satisfies<_Object, _Interface>)
  _CCCL_NODISCARD_FRIEND _CUDAX_TRIVIAL_HOST_API auto
  operator!=(_Object const& __lhs, iequality_comparable<_Interface> const& __rhs) noexcept -> bool
  {
    return !(__lhs == __rhs);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CUDA_VSTD::equality_comparable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Tp, _CUDAX_FNPTR_CONSTANT_WAR(&__equal_fn<_Tp>)>;
};

template <class... _Super>
struct iequality_comparable : iequality_comparable_base
{};

struct self; // a nice placeholder type

template <class _CvSelf, class _To>
struct __iconvertible_to
{
  static_assert(_CUDA_VSTD::is_same_v<_CUDA_VSTD::decay_t<_CvSelf>, self>,
                "The first template parameter to iconvertible_to must be the placeholder type "
                "cuda::experimental::self, possibly with cv- and/or ref-qualifiers");
};

template <class _To>
struct __iconvertible_to<self&&, _To>
{
  static_assert(_CUDA_VSTD::__always_false_v<_To>, "rvalue-qualified conversion operations are not yet supported");
};

template <class _To>
struct __iconvertible_to<self, _To>
{
  template <class...>
  struct __interface_ : interface<__interface_>
  {
    [[nodiscard]] _CUDAX_HOST_API operator _To()
    {
      return __cudax::virtcall<__conversion_fn<__interface_, _To>>(this);
    }

    template <class _From>
    using overrides = overrides_for<_From, _CUDAX_FNPTR_CONSTANT_WAR(&__conversion_fn<_From, _To>)>;
  };
};

template <class _To>
struct __iconvertible_to<self&, _To>
{
  template <class...>
  struct __interface_ : interface<__interface_>
  {
    [[nodiscard]] _CUDAX_HOST_API operator _To() &
    {
      return __cudax::virtcall<&__conversion_fn<__interface_&, _To>>(this);
    }

    template <class _From>
    using overrides = overrides_for<_From, _CUDAX_FNPTR_CONSTANT_WAR(&__conversion_fn<_From&, _To>)>;
  };
};

template <class _To>
struct __iconvertible_to<self const&, _To>
{
  template <class...>
  struct __interface_ : interface<__interface_>
  {
    [[nodiscard]] _CUDAX_HOST_API operator _To() const&
    {
      return __cudax::virtcall<&__conversion_fn<__interface_ const&, _To>>(this);
    }

    template <class _From>
    using overrides = overrides_for<_From, _CUDAX_FNPTR_CONSTANT_WAR(&__conversion_fn<_From const&, _To>)>;
  };
};

template <class _From, class _To>
using iconvertible_to _CCCL_NODEBUG_ALIAS = typename __iconvertible_to<_From, _To>::template __interface_<>;
} // namespace cuda::experimental

_CCCL_POP_MACROS

#endif // __CUDAX_DETAIL_BASIC_ANY_SEMIREGULAR_H
