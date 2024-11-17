//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_INTERFACES_H
#define __CUDAX_DETAIL_BASIC_ANY_INTERFACES_H

#include <cuda/std/detail/__config>

#include "cuda/std/__cccl/attributes.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/find.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/type_set.h>

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>
#include <cuda/experimental/__utility/basic_any/overrides.cuh>

namespace cuda::experimental
{
///
/// __ireference
///
/// Note: a `basic_any<__ireference<_Interface>>&&` is an rvalue reference, whereas
/// a `basic_any<_Interface&>&&` is an lvalue reference.
template <class _Interface>
struct __ireference : _Interface
{
  static_assert(_CUDA_VSTD::is_class_v<_Interface>, "expected a class type");
  static constexpr size_t __size_      = sizeof(void*);
  static constexpr size_t __align_     = alignof(void*);
  static constexpr bool __is_const_ref = _CUDA_VSTD::is_const_v<_Interface>;

  using interface _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::remove_const_t<_Interface>;
};

///
/// Interface type traits
///
template <class _Interface>
extern _Interface __remove_ireference_v<_Interface const>;

template <class _Interface>
extern _Interface __remove_ireference_v<__ireference<_Interface>>;

template <class _Interface>
extern _Interface __remove_ireference_v<__ireference<_Interface const>>;

template <class _Interface>
inline constexpr bool __is_value_v = _CUDA_VSTD::is_class_v<_Interface>;

template <class _Interface>
inline constexpr bool __is_value_v<__ireference<_Interface>> = false;

template <class _Interface>
inline constexpr bool __is_lvalue_reference_v = false;

template <class _Interface>
inline constexpr bool __is_lvalue_reference_v<__ireference<_Interface const>> = true;

template <class _Interface>
inline constexpr bool __is_lvalue_reference_v<_Interface&> = true;

///
/// __bases_of: get the list of base interface for an interface, including itself
///             and iunknown.
///
template <class _Interface, class _Fn>
using __bases_of _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_call<
  _CUDA_VSTD::__type_concat<_CUDA_VSTD::__type_list<iunknown, _CUDA_VSTD::remove_const_t<_Interface>>,
                            typename _Interface::template __ibases<__make_type_list>>,
  _Fn>;

///
/// interface subsumption
///
template <class _Interface1, class _Interface2>
inline constexpr bool __subsumes = false;

template <class _Interface>
inline constexpr bool __subsumes<_Interface, _Interface> = true;

template <class... _Set>
inline constexpr bool __subsumes<__iset<_Set...>, __iset<_Set...>> = true;

template <class... _Subset, class... _Superset>
inline constexpr bool __subsumes<__iset<_Subset...>, __iset<_Superset...>> =
  _CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_Superset...>, _Subset...>;

///
/// extension_of: Checks if one interface is an extension of another.
///
/// An interface \c A is considered an extension of another \c B if \c B
/// can be found by recursively searching the base interfaces of \c A.
///
/// \note An interface is considered an extension of itself.
///
template <class _Base>
struct __has_base_fn
{
  template <class... _Interfaces>
  using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::bool_constant<_CUDA_VSTD::__is_included_in_v<_Base, _Interfaces...>>;
};

template <class... _Bases>
struct __has_base_fn<__iset<_Bases...>>
{
  using __bases_set _CCCL_NODEBUG_ALIAS = __iset<_Bases...>;

  template <class... _Interfaces>
  using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::bool_constant<(__subsumes<__bases_set, _Interfaces> || ...)>;
};

template <class _Derived, class _Base, class = void>
inline constexpr bool __extension_of = false;

template <class _Derived, class _Base>
inline constexpr bool
  __extension_of<_Derived,
                 _Base,
                 _CUDA_VSTD::enable_if_t<_CUDA_VSTD::is_class_v<_Derived> && _CUDA_VSTD::is_class_v<_Base>>> =
    __bases_of<_Derived, __has_base_fn<_CUDA_VSTD::remove_const_t<_Base>>>::value;

template <class _Derived, class _Base>
_LIBCUDACXX_CONCEPT extension_of = __extension_of<_Derived, _Base>;

///
/// interface
///
template <template <class...> class _Interface, class... _Bases, size_t Size, size_t Align>
struct interface<_Interface, extends<_Bases...>, Size, Align>
{
  static constexpr size_t size  = (_CUDA_VSTD::max)({Size, _Bases::size...});
  static constexpr size_t align = (_CUDA_VSTD::max)({Align, _Bases::align...});

  template <class... _Super>
  using __rebind _CCCL_NODEBUG_ALIAS = _Interface<_Super...>;

  template <class _Fn>
  using __ibases _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_call<
    _CUDA_VSTD::__type_concat<__bases_of<_Bases, _CUDA_VSTD::__type_quote<_CUDA_VSTD::__type_list>>...>,
    _Fn>;

  template <class _Tp>
  using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Tp>;
};

///
/// __is_interface
///
template <template <class...> class _Interface, class Extends, size_t Size, size_t Align>
_CUDAX_API auto __is_interface_test(interface<_Interface, Extends, Size, Align> const&) -> void;

// clang-format off
template <class _Tp>
_LIBCUDACXX_CONCEPT __is_interface =
  _LIBCUDACXX_REQUIRES_EXPR((_Tp), _Tp& __value)
  (
    __is_interface_test(__value)
  );
// clang-format on

///
/// __unique_interfaces
///
/// Given an interface, return a list that contains the interface and all its
/// bases, but with duplicates removed.
///
template <class _Interface, class _Fn = _CUDA_VSTD::__type_quote<_CUDA_VSTD::__type_list>>
using __unique_interfaces _CCCL_NODEBUG_ALIAS =
  _CUDA_VSTD::__type_apply<_Fn, __bases_of<_Interface, _CUDA_VSTD::__type_quote<_CUDA_VSTD::__make_type_set>>>;

///
/// __index_of: find the index of an interface in a list of unique interfaces
///
_CCCL_NODISCARD _CUDAX_API constexpr size_t __find_index(_CUDA_VSTD::initializer_list<bool> __il)
{
  auto __it = _CUDA_VSTD::find(__il.begin(), __il.end(), true);
  return static_cast<size_t>(__it - __il.begin());
}

template <class _Interface>
struct __find_index_of
{
  template <class... _Interfaces>
  static constexpr size_t __index = __find_index({__subsumes<_Interface, _Interfaces>...});

  template <class... _Interfaces>
  using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::integral_constant<size_t, __index<_Interfaces...>>;
};

template <class _Interface, class _Super>
using __index_of _CCCL_NODEBUG_ALIAS =
  _CUDA_VSTD::__type_apply<__find_index_of<_Interface>, __unique_interfaces<_Super>>;

///
/// __iunknown: Logically, the root of all interfaces.
///
struct iunknown : interface<_CUDA_VSTD::__type_always<iunknown>::__call>
{};

template <class...>
struct __iempty : interface<__iempty>
{};

#if _CCCL_COMPILER(NVHPC)

template <class _Interface>
struct __vptr_for_impl
{
  using type = typename _Interface::template overrides<__remove_ireference_t<_Interface>>::__vptr_t;
};

template <class _Interface>
using __vptr_for _CCCL_NODEBUG_ALIAS = typename __vptr_for_impl<_Interface>::type;

#else

template <class _Interface>
using __vptr_for _CCCL_NODEBUG_ALIAS = typename __overrides_for<_Interface>::__vptr_t;

#endif

///
/// interface satisfaction
///
template <class _Tp>
struct __satisfaction_fn
{
  template <class _Interface>
  using __does_not_satisfy _CCCL_NODEBUG_ALIAS =
    _CUDA_VSTD::_Not<_CUDA_VSTD::_IsValidExpansion<__overrides_for, _Interface, _Tp>>;

  // Try to find an unsatisfied interface. If we find one, we return it (it's at
  // the front of the list returned from __type_find_if). If we don't find one
  // (that is, if the returned list is empty), we return __iempty<>.
  template <class... _Interfaces>
  using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_front< // take the front of the list
    _CUDA_VSTD::__type_push_back< // add __iempty<> to the end of the list
      _CUDA_VSTD::__type_find_if< // find the first unsatisfied interface if any, returns a list
        _CUDA_VSTD::__type_list<_Interfaces...>,
        _CUDA_VSTD::__type_quote1<__does_not_satisfy>>,
      __iempty<>>>;
};

template <class _Interface, class _Tp, class = void>
struct __unsatisfied_interface
{};

template <class _Interface, class _Tp>
struct __unsatisfied_interface<_Interface, _Tp, _CUDA_VSTD::enable_if_t<_CUDA_VSTD::is_class_v<_Interface>>>
{
  using type _CCCL_NODEBUG_ALIAS = __unique_interfaces<_Interface, __satisfaction_fn<_Tp>>;
};

template <class _Interface, class _Tp>
struct __unsatisfied_interface<_Interface*, _Tp*> : __unsatisfied_interface<_Interface, _Tp>
{};

template <class _Interface, class _Tp>
struct __unsatisfied_interface<_Interface const*, _Tp const*> : __unsatisfied_interface<_Interface, _Tp>
{};

template <class _Interface, class _Tp>
struct __unsatisfied_interface<_Interface&, _Tp> : __unsatisfied_interface<_Interface, _Tp>
{};

template <class _Tp, class _Interface>
_LIBCUDACXX_CONCEPT __has_overrides = _CUDA_VSTD::_IsValidExpansion<__overrides_for, _Interface, _Tp>::value;

/// The \c __satisfies concept checks if a type \c _Tp satisfies an interface
/// \c _Interface. It does this by trying to instantiate
/// `__overrides_for<_X, _Tp>` for all \c _X, where \c _X is \c _Interface or
/// one of its bases. If any of the \c __overrides_for instantiations are ill-
/// formed, then \c _Tp does not satisfy \c _Interface.
///
/// \c __satisfies is implemented by searching through the list of interfaces for
/// one that \c _Tp does not satisfy. If such an interface is found, the concept
/// check fails in such a way as to hopefully tell the user which interface is
/// not satisfied and why.
template <class _Tp,
          class _Interface,
          class UnsatisfiedInterface = _CUDA_VSTD::__type<__unsatisfied_interface<_Interface, _Tp>>>
_LIBCUDACXX_CONCEPT __satisfies = __has_overrides<_Tp, UnsatisfiedInterface>;

///
/// __interface_of
///
template <class _Super>
struct __make_interface_fn
{
  static_assert(_CUDA_VSTD::is_class_v<_Super>, "expected a class type");
  template <class... _Interfaces>
  using __call _CCCL_NODEBUG_ALIAS = detail::__inherit<__rebind_interface<_Interfaces, _Super>...>;
};

// Given an interface `_I<>`, let `_Bs<>...` be the list of types consisting
// of all of `_I<>`'s unique bases. Then `__interface_of<_I<>>` is the
// type `__inherit<_I<_I<>>, _Bs<_I<>>...>`. That is, it transforms the
// unspecialized interfaces into ones specialized for `_I<>` and then
// makes a type that inherits publicly from all of them.
template <class _Interface>
using __interface_of _CCCL_NODEBUG_ALIAS = __unique_interfaces<_Interface, __make_interface_fn<_Interface>>;

///
/// interface_cast
///
/// given a `basic_any<X<>>` object `o`, `interface_cast<Y>(o)` return a
/// reference to the (empty) sub-object of type `Y<X<>>`, from which
/// `basic_any<X<>>` inherits, where `Y<>` is an interface that `X<>` extends.
///
template <class _Interface>
struct __interface_cast_fn;

template <template <class...> class _Interface>
struct __interface_cast_fn<_Interface<>>
{
  template <class _Super>
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API _Interface<_Super>&& operator()(_Interface<_Super>&& __self) const noexcept
  {
    return _CUDA_VSTD::move(__self);
  }

  template <class _Super>
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API _Interface<_Super>& operator()(_Interface<_Super>& __self) const noexcept
  {
    return __self;
  }

  template <class _Super>
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API _Interface<_Super> const& operator()(_Interface<_Super> const& __self) noexcept
  {
    return __self;
  }
};

_LIBCUDACXX_TEMPLATE(template <class...> class _Interface, class Object)
_LIBCUDACXX_REQUIRES(
  __is_interface<_Interface<>> _LIBCUDACXX_AND _CUDA_VSTD::__is_callable_v<__interface_cast_fn<_Interface<>>, Object>)
_CCCL_NODISCARD _CUDAX_TRIVIAL_API decltype(auto) __interface_cast(Object&& __obj) noexcept
{
  return __interface_cast_fn<_Interface<>>{}(_CUDA_VSTD::forward<Object>(__obj));
}

_LIBCUDACXX_TEMPLATE(class _Interface, class Object)
_LIBCUDACXX_REQUIRES(
  __is_interface<_Interface> _LIBCUDACXX_AND _CUDA_VSTD::__is_callable_v<__interface_cast_fn<_Interface>, Object>)
_CCCL_NODISCARD _CUDAX_TRIVIAL_API decltype(auto) __interface_cast(Object&& __obj) noexcept
{
  return __interface_cast_fn<_Interface>{}(_CUDA_VSTD::forward<Object>(__obj));
}
} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_INTERFACES_H
