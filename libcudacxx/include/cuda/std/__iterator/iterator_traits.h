// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ITERATOR_ITERATOR_TRAITS_H
#define _CUDA_STD___ITERATOR_ITERATOR_TRAITS_H

#include <cuda/std/detail/__config>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/referenceable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__fwd/iterator.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/iterator_tag.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/add_lvalue_reference.h>
#include <cuda/std/__type_traits/common_reference.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_primary_template.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/priority_tag.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// [iterator.traits]
#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __dereferenceable = requires(_Tp& __t) {
  { *__t } -> __referenceable; // not required to be equality-preserving
};

template <__dereferenceable _Tp>
using iter_reference_t = decltype(*::cuda::std::declval<_Tp&>());

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ // vvv _CCCL_HAS_CONCEPTS() vvv

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wvoid-ptr-dereference")

template <class _Tp>
_CCCL_CONCEPT __dereferenceable = _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)(requires(__referenceable<decltype(*__t)>));

_CCCL_DIAG_POP

template <class _Tp>
using iter_reference_t = enable_if_t<__dereferenceable<_Tp>, decltype(*::cuda::std::declval<_Tp&>())>;
#endif // _CCCL_HAS_CONCEPTS()

// The `cpp17-*-iterator` exposition-only concepts have very similar names to the `Cpp17*Iterator` named requirements
// from `[iterator.cpp17]`. To avoid confusion between the two, the exposition-only concepts have been banished to
// a "detail" namespace indicating they have a niche use-case.
namespace __iterator_traits_detail
{
// [iterator.traits#concept:cpp17-iterator]
template <class _Iter>
_CCCL_CONCEPT __cpp17_iterator = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i)(
  requires(copyable<_Iter>),
  _Satisfies(__referenceable)(*__i),
  _Same_as(_Iter&)(++__i),
  _Satisfies(__referenceable)(*__i++));

// [iterator.traits#concept:cpp17-input-iterator]
template <class _Iter>
_CCCL_CONCEPT __cpp17_input_iterator = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i)(
  requires(__cpp17_iterator<_Iter>),
  requires(equality_comparable<_Iter>),
  typename(typename incrementable_traits<_Iter>::difference_type),
  typename(typename indirectly_readable_traits<_Iter>::value_type),
  typename(common_reference_t<iter_reference_t<_Iter>&&, typename indirectly_readable_traits<_Iter>::value_type&>),
  typename(common_reference_t<decltype(*__i++)&&, typename indirectly_readable_traits<_Iter>::value_type&>),
  requires(signed_integral<typename incrementable_traits<_Iter>::difference_type>));

// [iterator.traits#concept:cpp17-forward-iterator]
template <class _Iter>
_CCCL_CONCEPT __cpp17_forward_iterator = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i)(
  requires(__cpp17_input_iterator<_Iter>),
  requires(constructible_from<_Iter>),
  requires(is_lvalue_reference_v<iter_reference_t<_Iter>>),
  requires(same_as<remove_cvref_t<iter_reference_t<_Iter>>, typename indirectly_readable_traits<_Iter>::value_type>),
  requires(convertible_to<decltype(__i++), _Iter const&>),
  _Same_as(iter_reference_t<_Iter>)(*__i++));

// [iterator.traits#concept:cpp17-bidirectional-iterator]
template <class _Iter>
_CCCL_CONCEPT __cpp17_bidirectional_iterator = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i)(
  requires(__cpp17_forward_iterator<_Iter>),
  _Same_as(_Iter&)(--__i),
  requires(convertible_to<decltype(__i--), _Iter const&>),
  _Same_as(iter_reference_t<_Iter>)(*__i--));

// [iterator.traits#concept:cpp17-random-access-iterator]
// Needs to be its own concept, because we need `typename incrementable_traits<_Iter>::difference_type` to be valid
template <class _Iter>
_CCCL_CONCEPT __cpp17_random_access_iterator_operations =
  _CCCL_REQUIRES_EXPR((_Iter), _Iter __i, typename incrementable_traits<_Iter>::difference_type __n)(
    _Same_as(_Iter&)(__i += __n),
    _Same_as(_Iter&)(__i -= __n),
    _Same_as(_Iter)(__i + __n),
    _Same_as(_Iter)(__n + __i),
    _Same_as(_Iter)(__i - __n),
    _Same_as(decltype(__n))(__i - __i),
    requires(convertible_to<decltype(__i[__n]), iter_reference_t<_Iter>>));

template <class _Iter>
_CCCL_CONCEPT __cpp17_random_access_iterator = _CCCL_REQUIRES_EXPR((_Iter))(
  requires(__cpp17_bidirectional_iterator<_Iter>),
  requires(totally_ordered<_Iter>),
  requires(__cpp17_random_access_iterator_operations<_Iter>));
} // namespace __iterator_traits_detail

// [iterator.traits]#3.1
// If the qualified-id I::pointer is valid and denotes a type, then pointer names that type;
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_pointer_or_void(int) -> typename _Iter::pointer;
// Otherwise, it names void.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_pointer_or_void(...) -> void;

template <class _Iter>
using __iterator_traits_member_pointer_or_void _CCCL_NODEBUG =
  decltype(::cuda::std::__iterator_traits_deduce_member_pointer_or_void<_Iter>(0));

// [iterator.traits]#3.2
// [iterator.traits]#3.2.1
// If the qualified-id I::pointer is valid and denotes a type, then pointer names that type.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_pointer_or_arrow_or_void(int, __priority_tag<1>) ->
  typename _Iter::pointer;

// Otherwise, if decltype(declval<I&>().operator->()) is well-formed, then pointer names that type.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_pointer_or_arrow_or_void(int, __priority_tag<0>)
  -> decltype(::cuda::std::declval<_Iter&>().operator->());

// Otherwise, pointer names void.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_pointer_or_arrow_or_void(...) -> void;

template <class _Iter>
using __iterator_traits_member_pointer_or_arrow_or_void _CCCL_NODEBUG =
  decltype(::cuda::std::__iterator_traits_deduce_member_pointer_or_arrow_or_void<_Iter>(0, __priority_tag<1>{}));

// [iterator.traits]#3.2.2
// If the qualified-id `I::reference` is valid and denotes a type, `reference` names that type.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_reference(int) -> typename _Iter::reference;
// Otherwise, `reference` names `iter-reference-t<I>`.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_reference(...) -> iter_reference_t<_Iter>;

template <class _Iter>
using __iterator_traits_member_reference _CCCL_NODEBUG =
  decltype(::cuda::std::__iterator_traits_deduce_member_reference<_Iter>(0));

// [iterator.traits]#3.2.3
template <class _Iter>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __iterator_traits_deduce_iterator_category() noexcept
{
  if constexpr (__has_member_iterator_category<_Iter>)
  { // If the qualified-id `I::iterator-category` is valid and denotes a type, `iterator-category` names that type.
    return typename _Iter::iterator_category{};
  }
  else if constexpr (__iterator_traits_detail::__cpp17_random_access_iterator<_Iter>)
  { // Otherwise `random_access_iterator_tag` if `I` satisfies `cpp17-random-access-iterator`,
    return random_access_iterator_tag{};
  }
  else if constexpr (__iterator_traits_detail::__cpp17_bidirectional_iterator<_Iter>)
  { // or otherwise `bidirectional_iterator_tag` if `I` satisfies `cpp17-bidirectional-iterator`,
    return bidirectional_iterator_tag{};
  }
  else if constexpr (__iterator_traits_detail::__cpp17_forward_iterator<_Iter>)
  { // or otherwise `forward_iterator_tag` if `I` satisfies `cpp17-forward-iterator`,
    return forward_iterator_tag{};
  }
  else
  { // or otherwise input_iterator_tag
    return input_iterator_tag{};
  }
}

template <class _Iter>
using __iterator_traits_iterator_category _CCCL_NODEBUG =
  decltype(::cuda::std::__iterator_traits_deduce_iterator_category<_Iter>());

// [iterator.traits]#3.3
// If the qualified-id `incrementable_traits<I>::difference_type` is valid and denotes a type, then
// `difference_type` names that type;
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_difference(int) -> typename incrementable_traits<_Iter>::difference_type;
// Otherwise, it names void.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_difference(...) -> void;

template <class _Iter>
using __iterator_traits_difference_type _CCCL_NODEBUG =
  decltype(::cuda::std::__iterator_traits_deduce_member_difference<_Iter>(0));

enum class __iterator_traits_selection
{
  __specialized_from_std,
  __specifies_members,
  __cpp17_input_iterator,
  __cpp17_iterator,
  __no_members,
};

// We need to consider if a user has specialized std::iterator_traits
template <class _Iter>
_CCCL_CONCEPT __specialized_from_std = !__is_primary_std_template<remove_cvref_t<_Iter>>::value;

// If I has valid member types difference_type, value_type, reference, and iterator_category,
template <class _Iter>
_CCCL_CONCEPT __specifies_members = _CCCL_REQUIRES_EXPR((_Iter))(
  typename(typename _Iter::value_type),
  typename(typename _Iter::difference_type),
  typename(typename _Iter::reference),
  typename(typename _Iter::iterator_category));

// [iterator.traits]#3.2.3
template <class _Iter>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL __iterator_traits_selection __select_iterator_traits_specialization() noexcept
{
  if constexpr (__specialized_from_std<_Iter>)
  { // We need to consider if a user has specialized std::iterator_traits
    return __iterator_traits_selection::__specialized_from_std;
  }
  if constexpr (__specifies_members<_Iter>)
  { // If I has valid member types difference_type, value_type, reference, and iterator_category,
    return __iterator_traits_selection::__specifies_members;
  }
  else if constexpr (__iterator_traits_detail::__cpp17_input_iterator<_Iter>)
  { // Otherwise, if I satisfies the exposition-only concept cpp17-input-iterator,
    return __iterator_traits_selection::__cpp17_input_iterator;
  }
  else if constexpr (__iterator_traits_detail::__cpp17_iterator<_Iter>)
  { // Otherwise, if I satisfies the exposition-only concept cpp17-iterator,
    return __iterator_traits_selection::__cpp17_iterator;
  }
  else
  { // Otherwise, iterator_traits<I> has no members by any of the above names.
    return __iterator_traits_selection::__no_members;
  }
}

// [iterator.traits]#3
template <class _Iter, __iterator_traits_selection = ::cuda::std::__select_iterator_traits_specialization<_Iter>()>
struct __iterator_traits;

#if _CCCL_HOSTED()
// We need to properly accept specializations of `std::iterator_traits`
template <class _Iter>
struct __iterator_traits<_Iter, __iterator_traits_selection::__specialized_from_std>
    : public ::std::iterator_traits<_Iter>
{};
#endif // _CCCL_HOSTED()

// [iterator.traits]#3.1
// If `I` has valid member types `difference-type`, `value-type`, `reference`, and
// `iterator-category`, then `iterator-traits<I>` has the following publicly accessible members:
template <class _Iter>
struct __iterator_traits<_Iter, __iterator_traits_selection::__specifies_members>
{
  using iterator_category = typename _Iter::iterator_category;
  using value_type        = typename _Iter::value_type;
  using difference_type   = typename _Iter::difference_type;
  using pointer           = __iterator_traits_member_pointer_or_void<_Iter>;
  using reference         = typename _Iter::reference;
};

// [iterator.traits]#3.2
// Otherwise, if `I` satisfies the exposition-only concept `cpp17-input-iterator`,
// `iterator-traits<I>` has the following publicly accessible members:
template <class _Iter>
struct __iterator_traits<_Iter, __iterator_traits_selection::__cpp17_input_iterator>
{
  using iterator_category = __iterator_traits_iterator_category<_Iter>;
  using value_type        = typename indirectly_readable_traits<_Iter>::value_type;
  using difference_type   = typename incrementable_traits<_Iter>::difference_type;
  using pointer           = __iterator_traits_member_pointer_or_arrow_or_void<_Iter>;
  using reference         = __iterator_traits_member_reference<_Iter>;
};

// [iterator.traits]#3.3
// Otherwise, if `I` satisfies the exposition-only concept `cpp17-iterator`, then
// `iterator_traits<I>` has the following publicly accessible members:
template <class _Iter>
struct __iterator_traits<_Iter, __iterator_traits_selection::__cpp17_iterator>
{
  using iterator_category = output_iterator_tag;
  using value_type        = void;
  using difference_type   = __iterator_traits_difference_type<_Iter>;
  using pointer           = void;
  using reference         = void;
};

// [iterator.traits]#3.4
// Otherwise, `iterator_traits<I>` has no members by any of the above names.
template <class _Iter>
struct __iterator_traits<_Iter, __iterator_traits_selection::__no_members>
{};

template <class _Iter, class>
struct _CCCL_TYPE_VISIBILITY_DEFAULT iterator_traits : __iterator_traits<_Iter>
{
  using __cccl_primary_template _CCCL_NODEBUG = iterator_traits;
};

// [iterator.traits]#5
template <class _Tp>
#if _CCCL_HAS_CONCEPTS()
  requires is_object_v<_Tp>
#endif // _CCCL_HAS_CONCEPTS()
struct _CCCL_TYPE_VISIBILITY_DEFAULT iterator_traits<_Tp*>
{
  using difference_type   = ptrdiff_t;
  using value_type        = remove_cv_t<_Tp>;
  using pointer           = _Tp*;
  using reference         = add_lvalue_reference_t<_Tp>;
  using iterator_category = random_access_iterator_tag;
  using iterator_concept  = contiguous_iterator_tag;
};

template <class _Iter, class _Tag>
_CCCL_CONCEPT __has_iterator_category_convertible_to = _CCCL_REQUIRES_EXPR((_Iter, _Tag)) //
  (typename(typename iterator_traits<_Iter>::iterator_category),
   requires(is_convertible_v<typename iterator_traits<_Iter>::iterator_category, _Tag>));

template <class _Iter, class _Tag>
_CCCL_CONCEPT __has_iterator_concept_convertible_to = _CCCL_REQUIRES_EXPR((_Iter, _Tag)) //
  (typename(typename _Iter::iterator_concept), requires(is_convertible_v<typename _Iter::iterator_concept, _Tag>));

template <class _Iter>
inline constexpr bool __has_input_traversal =
  __has_iterator_category_convertible_to<_Iter, input_iterator_tag>
  || __has_iterator_concept_convertible_to<_Iter, input_iterator_tag>;

template <class _Iter>
inline constexpr bool __has_forward_traversal =
  __has_iterator_category_convertible_to<_Iter, forward_iterator_tag>
  || __has_iterator_concept_convertible_to<_Iter, forward_iterator_tag>;

template <class _Iter>
inline constexpr bool __has_bidirectional_traversal =
  __has_iterator_category_convertible_to<_Iter, bidirectional_iterator_tag>
  || __has_iterator_concept_convertible_to<_Iter, bidirectional_iterator_tag>;

template <class _Iter>
inline constexpr bool __has_random_access_traversal =
  __has_iterator_category_convertible_to<_Iter, random_access_iterator_tag>
  || __has_iterator_concept_convertible_to<_Iter, random_access_iterator_tag>;

// in C++17 the standard does not yet know about the contiguous iterator concept, so we need to check for it explicitly
template <class _Iter>
_CCCL_CONCEPT __is_cpp17_contiguous_iterator = _CCCL_REQUIRES_EXPR((_Iter)) //
  (typename(_ITER_CONCEPT<_Iter>), requires(is_convertible_v<_ITER_CONCEPT<_Iter>, contiguous_iterator_tag>));

// __has_contiguous_traversal determines if an iterator is known by
// libc++ to be contiguous, either because it advertises itself as such
// (in C++20) or because it is a pointer type or a known trivial wrapper
// around a (possibly fancy) pointer type, such as __wrap_iter<T*>.
// Such iterators receive special "contiguous" optimizations in
// std::copy and std::sort.
//
template <class _Iter>
inline constexpr bool __has_contiguous_traversal =
  __has_iterator_category_convertible_to<_Iter, contiguous_iterator_tag>
  || __has_iterator_concept_convertible_to<_Iter, contiguous_iterator_tag> || __is_cpp17_contiguous_iterator<_Iter>;

// Any native pointer which is an iterator is also a contiguous iterator.
template <class _Tp>
inline constexpr bool __has_contiguous_traversal<_Tp*> = true;

template <class _Iter>
using __iter_value_type _CCCL_NODEBUG = typename iterator_traits<_Iter>::value_type;

template <class _Iter>
using __iterator_category_type _CCCL_NODEBUG = typename iterator_traits<_Iter>::iterator_category;

template <class _Iter>
using __iterator_pointer_type _CCCL_NODEBUG = typename iterator_traits<_Iter>::pointer;

template <class _Iter>
using __iter_diff_t _CCCL_NODEBUG = typename iterator_traits<_Iter>::difference_type;

// C++20 iterators do not play nicely with C++17 interfaces, because they commmonly use `iterator_concept` to
// communicate their iterator category. Deduce the actual iterator category from the maximum of `iterator_concept` and
// `iterator_category`
template <class _Iter>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __iterator_traits_category_or_concept() noexcept
{
  if constexpr (!__has_member_iterator_concept<_Iter>)
  {
    return __iterator_category_type<_Iter>{};
  }
  else if constexpr (__has_contiguous_traversal<_Iter>)
  {
    return contiguous_iterator_tag{};
  }
  else if constexpr (__has_random_access_traversal<_Iter>)
  {
    return random_access_iterator_tag{};
  }
  else if constexpr (__has_bidirectional_traversal<_Iter>)
  {
    return bidirectional_iterator_tag{};
  }
  else if constexpr (__has_forward_traversal<_Iter>)
  {
    return forward_iterator_tag{};
  }
  else if constexpr (__has_input_traversal<_Iter>)
  {
    return input_iterator_tag{};
  }
  else // if constexpr (__has_member_iterator_category<_Iter>)
  {
    return typename _Iter::iterator_category{};
  }
}

template <class _Iter>
using __iterator_traits_category_or_concept_t _CCCL_NODEBUG =
  decltype(::cuda::std::__iterator_traits_category_or_concept<_Iter>());

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_ITERATOR_TRAITS_H
