//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_SEMIREGULAR_H
#define _CUDA___UTILITY_BASIC_ANY_SEMIREGULAR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/access.h>
#include <cuda/__utility/__basic_any/basic_any_base.h>
#include <cuda/__utility/__basic_any/basic_any_from.h>
#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/__utility/__basic_any/conversions.h>
#include <cuda/__utility/__basic_any/interfaces.h>
#include <cuda/__utility/__basic_any/storage.h>
#include <cuda/__utility/__basic_any/virtcall.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/movable.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/typeid.h>
#include <cuda/std/__utility/unreachable.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//!
//! semi-regular overrides
//!

_CCCL_EXEC_CHECK_DISABLE
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::movable<_Tp>)
_CCCL_PUBLIC_API auto __move_fn(_Tp& __src, void* __dst) noexcept -> void
{
  ::new (__dst) _Tp(static_cast<_Tp&&>(__src));
}

_CCCL_EXEC_CHECK_DISABLE
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::movable<_Tp>)
[[nodiscard]] _CCCL_PUBLIC_API auto __try_move_fn(_Tp& __src, void* __dst, size_t __size, size_t __align) -> bool
{
  if (::cuda::__is_small<_Tp>(__size, __align))
  {
    ::new (__dst) _Tp(static_cast<_Tp&&>(__src));
    return true;
  }
  else
  {
    ::new (__dst)::cuda::std::type_identity_t<_Tp*>(new _Tp(static_cast<_Tp&&>(__src)));
    return false;
  }
}

_CCCL_EXEC_CHECK_DISABLE
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::copyable<_Tp>)
[[nodiscard]] _CCCL_PUBLIC_API auto __copy_fn(_Tp const& __src, void* __dst, size_t __size, size_t __align) -> bool
{
  if (::cuda::__is_small<_Tp>(__size, __align))
  {
    ::new (__dst) _Tp(__src);
    return true;
  }
  else
  {
    ::new (__dst)::cuda::std::type_identity_t<_Tp*>(new _Tp(__src));
    return false;
  }
}

_CCCL_EXEC_CHECK_DISABLE
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::equality_comparable<_Tp>)
[[nodiscard]] _CCCL_PUBLIC_API auto
__equal_fn(_Tp const& __self, ::cuda::std::__type_info_ref __type, void const* __other) -> bool
{
  if (_CCCL_TYPEID(_Tp) == __type)
  {
    return __self == *static_cast<_Tp const*>(__other);
  }
  return false;
}

_CCCL_EXEC_CHECK_DISABLE
_CCCL_TEMPLATE(class _From, class _To)
_CCCL_REQUIRES(::cuda::std::convertible_to<_From, _To>)
[[nodiscard]] _CCCL_PUBLIC_API _To __conversion_fn(::cuda::std::type_identity_t<_From> __self)
{
  return static_cast<_To>(static_cast<_From&&>(__self));
}

//!
//! semi-regular interfaces
//!
template <class...>
struct __imovable : __basic_interface<__imovable>
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::movable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS = __overrides_for<_Tp, &::cuda::__try_move_fn<_Tp>, &::cuda::__move_fn<_Tp>>;

  _CCCL_API auto __move_to(void* __pv) noexcept -> void
  {
    return ::cuda::__virtcall<&::cuda::__move_fn<__imovable>>(this, __pv);
  }

  [[nodiscard]] _CCCL_API auto __move_to(void* __pv, size_t __size, size_t __align) -> bool
  {
    return ::cuda::__virtcall<&::cuda::__try_move_fn<__imovable>>(this, __pv, __size, __align);
  }
};

template <class...>
struct __icopyable : __basic_interface<__icopyable, __extends<__imovable<>>>
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::copyable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS = __overrides_for<_Tp, &::cuda::__copy_fn<_Tp>>;

  [[nodiscard]] _CCCL_API auto __copy_to(void* __pv, size_t __size, size_t __align) const -> bool
  {
    return ::cuda::__virtcall<&::cuda::__copy_fn<__icopyable>>(this, __pv, __size, __align);
  }
};

template <class _Object>
_CCCL_CONCEPT __non_polymorphic = (!__is_basic_any<_Object>) && (!__is_interface<_Object>);

template <class... _Super>
struct __iequality_comparable;

struct iequality_comparable_base : __basic_interface<__iequality_comparable>
{
  // These overloads are only necessary so that __iequality_comparable<> itself
  // satisfies the std::equality_comparable constraint that is used by the
  // `__iequality_comparable<>::overloads` alias template below.
  [[noreturn]] friend _CCCL_NODEBUG_API auto
  operator==(__iequality_comparable<> const&, __iequality_comparable<> const&) noexcept -> bool
  {
    ::cuda::std::unreachable();
  }

  [[noreturn]] friend _CCCL_NODEBUG_API auto
  operator!=(__iequality_comparable<> const&, __iequality_comparable<> const&) noexcept -> bool
  {
    ::cuda::std::unreachable();
  }

  // These are the overloads that get used when testing two `__basic_any` objects
  // for equality.
  _CCCL_TEMPLATE(class _ILeft, class _IRight)
  _CCCL_REQUIRES(__any_convertible_to<__basic_any<_ILeft> const&, __basic_any<_IRight> const&>
                 || __any_convertible_to<__basic_any<_IRight> const&, __basic_any<_ILeft> const&>)
  [[nodiscard]] _CCCL_API friend auto
  operator==(__iequality_comparable<_ILeft> const& __lhs, __iequality_comparable<_IRight> const& __rhs) noexcept -> bool
  {
    auto const& __other = ::cuda::__basic_any_from(__rhs);
    constexpr auto __eq = &::cuda::__equal_fn<__iequality_comparable<_ILeft>>;
    return ::cuda::__virtcall<__eq>(&__lhs, __other.type(), __basic_any_access::__get_optr(__other));
  }

  _CCCL_TEMPLATE(class _ILeft, class _IRight)
  _CCCL_REQUIRES(__any_convertible_to<__basic_any<_ILeft> const&, __basic_any<_IRight> const&>
                 || __any_convertible_to<__basic_any<_IRight> const&, __basic_any<_ILeft> const&>)
  [[nodiscard]] _CCCL_NODEBUG_API friend auto
  operator!=(__iequality_comparable<_ILeft> const& __lhs, __iequality_comparable<_IRight> const& __rhs) noexcept -> bool
  {
    return !(__lhs == __rhs);
  }

  // These are the overloads that get used when testing a `__basic_any` object
  // against a non-type-erased object.
  //
  // Q: Why require that the __basic_any wrapper is not convertible to _Object?
  //
  // A: If there is a user-defined conversion from the __basic_any type to _Object
  // such that we can use _Object's symmetric equality comparison operator, that
  // should be preferred. _Object may be another kind of wrapper object, in which
  // case using the address of the **wrapper** object for the comparison (as
  // opposed to the address of the wrapped object) is probably wrong.
  _CCCL_TEMPLATE(class _Interface, class _Object, class _Self = __basic_any_from_t<__iequality_comparable<_Interface>>)
  _CCCL_REQUIRES(__non_polymorphic<_Object> _CCCL_AND(!::cuda::std::convertible_to<_Self, _Object>)
                   _CCCL_AND __satisfies<_Object, _Interface>)
  [[nodiscard]] _CCCL_API friend auto operator==(__iequality_comparable<_Interface> const& __lhs, _Object const& __rhs)
    -> bool
  {
    constexpr auto __eq = &::cuda::__equal_fn<__iequality_comparable<_Interface>>;
    return ::cuda::__virtcall<__eq>(&__lhs, _CCCL_TYPEID(_Object), ::cuda::std::addressof(__rhs));
  }

  _CCCL_TEMPLATE(class _Interface, class _Object, class _Self = __basic_any_from_t<__iequality_comparable<_Interface>>)
  _CCCL_REQUIRES(__non_polymorphic<_Object> _CCCL_AND(!::cuda::std::convertible_to<_Self, _Object>)
                   _CCCL_AND __satisfies<_Object, _Interface>)
  [[nodiscard]] _CCCL_API friend auto
  operator==(_Object const& __lhs, __iequality_comparable<_Interface> const& __rhs) noexcept -> bool
  {
    constexpr auto __eq = &::cuda::__equal_fn<__iequality_comparable<_Interface>>;
    return ::cuda::__virtcall<__eq>(&__rhs, _CCCL_TYPEID(_Object), ::cuda::std::addressof(__lhs));
  }

  _CCCL_TEMPLATE(class _Interface, class _Object, class _Self = __basic_any_from_t<__iequality_comparable<_Interface>>)
  _CCCL_REQUIRES(__non_polymorphic<_Object> _CCCL_AND(!::cuda::std::convertible_to<_Self, _Object>)
                   _CCCL_AND __satisfies<_Object, _Interface>)
  [[nodiscard]] _CCCL_NODEBUG_API friend auto
  operator!=(__iequality_comparable<_Interface> const& __lhs, _Object const& __rhs) noexcept -> bool
  {
    return !(__lhs == __rhs);
  }

  _CCCL_TEMPLATE(class _Interface, class _Object, class _Self = __basic_any_from_t<__iequality_comparable<_Interface>>)
  _CCCL_REQUIRES(__non_polymorphic<_Object> _CCCL_AND(!::cuda::std::convertible_to<_Self, _Object>)
                   _CCCL_AND __satisfies<_Object, _Interface>)
  [[nodiscard]] _CCCL_NODEBUG_API friend auto
  operator!=(_Object const& __lhs, __iequality_comparable<_Interface> const& __rhs) noexcept -> bool
  {
    return !(__lhs == __rhs);
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::equality_comparable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS = __overrides_for<_Tp, &::cuda::__equal_fn<_Tp>>;
};

template <class... _Super>
struct __iequality_comparable : iequality_comparable_base
{};

struct __self; // a nice placeholder type

template <class _CvSelf, class _To>
struct __iconvertible_to_
{
  static_assert(::cuda::std::is_same_v<::cuda::std::decay_t<_CvSelf>, __self>,
                "The first template parameter to __iconvertible_to must be the placeholder type "
                "cuda::__self, possibly with cv- and/or ref-qualifiers");
};

template <class _To>
struct __iconvertible_to_<__self&&, _To>
{
  static_assert(::cuda::std::__always_false_v<_To>, "rvalue-qualified conversion operations are not yet supported");
};

template <class _To>
struct __iconvertible_to_<__self, _To>
{
  template <class...>
  struct __interface_ : __basic_interface<__interface_>
  {
    [[nodiscard]] _CCCL_API operator _To()
    {
      return ::cuda::__virtcall<::cuda::__conversion_fn<__interface_, _To>>(this);
    }

    template <class _From>
    using overrides = __overrides_for<_From, &::cuda::__conversion_fn<_From, _To>>;
  };
};

template <class _To>
struct __iconvertible_to_<__self&, _To>
{
  template <class...>
  struct __interface_ : __basic_interface<__interface_>
  {
    [[nodiscard]] _CCCL_API operator _To() &
    {
      return ::cuda::__virtcall<&::cuda::__conversion_fn<__interface_&, _To>>(this);
    }

    template <class _From>
    using overrides = __overrides_for<_From, &::cuda::__conversion_fn<_From&, _To>>;
  };
};

template <class _To>
struct __iconvertible_to_<__self const&, _To>
{
  template <class...>
  struct __interface_ : __basic_interface<__interface_>
  {
    [[nodiscard]] _CCCL_API operator _To() const&
    {
      return ::cuda::__virtcall<&::cuda::__conversion_fn<__interface_ const&, _To>>(this);
    }

    template <class _From>
    using overrides = __overrides_for<_From, &::cuda::__conversion_fn<_From const&, _To>>;
  };
};

template <class _From, class _To>
using __iconvertible_to _CCCL_NODEBUG_ALIAS = typename __iconvertible_to_<_From, _To>::template __interface_<>;
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_SEMIREGULAR_H
