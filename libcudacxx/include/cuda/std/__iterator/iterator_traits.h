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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
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
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__concepts/totally_ordered.h>
#include <cuda/std/__fwd/iterator.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__type_traits/add_const.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_primary_template.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/priority_tag.h>
#include <cuda/std/cstddef>

#if _CCCL_HOSTED()
#  if _CCCL_COMPILER(MSVC)
#    include <xutility> // for ::std::input_iterator_tag
#  else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
#    include <iterator> // for ::std::input_iterator_tag
#  endif // !_CCCL_COMPILER(MSVC)

#  ifdef _GLIBCXX_DEBUG
#    include <debug/safe_iterator.h>
#  endif // _GLIBCXX_DEBUG

#  if _CCCL_STD_VER >= 2020
#    include <cuda/std/__cccl/prologue.h>
template <class _Tp, class = void>
struct __cccl_type_is_defined : ::cuda::std::false_type
{};

template <class _Tp>
struct __cccl_type_is_defined<_Tp, ::cuda::std::void_t<decltype(sizeof(_Tp))>> : ::cuda::std::true_type
{};

// detect whether the used STL has contiguous_iterator_tag defined
namespace std
{
struct __cccl_std_contiguous_iterator_tag_exists : __cccl_type_is_defined<struct contiguous_iterator_tag>
{};
} // namespace std

#    include <cuda/std/__cccl/epilogue.h>
#  endif // _CCCL_STD_VER >= 2020

#endif // _CCCL_HOSTED()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
using __with_reference = _Tp&;

template <class _Tp>
_CCCL_CONCEPT __can_reference = _CCCL_REQUIRES_EXPR((_Tp))(typename(__with_reference<_Tp>));

// [iterator.traits]
#if _CCCL_HAS_CONCEPTS()
template <class _Tp>
concept __dereferenceable = requires(_Tp& __t) {
  { *__t } -> __can_reference; // not required to be equality-preserving
};

template <__dereferenceable _Tp>
using iter_reference_t = decltype(*::cuda::std::declval<_Tp&>());

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ // vvv _CCCL_HAS_CONCEPTS() vvv

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wvoid-ptr-dereference")

template <class _Tp>
_CCCL_CONCEPT __dereferenceable = _CCCL_REQUIRES_EXPR((_Tp), _Tp& __t)(requires(__can_reference<decltype(*__t)>));

_CCCL_DIAG_POP

template <class _Tp>
using iter_reference_t = enable_if_t<__dereferenceable<_Tp>, decltype(*::cuda::std::declval<_Tp&>())>;
#endif // _CCCL_HAS_CONCEPTS()

#if _CCCL_FREESTANDING()

struct _CCCL_TYPE_VISIBILITY_DEFAULT input_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT output_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT forward_iterator_tag : public input_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT bidirectional_iterator_tag : public forward_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT random_access_iterator_tag : public bidirectional_iterator_tag
{};
struct _CCCL_TYPE_VISIBILITY_DEFAULT contiguous_iterator_tag : public random_access_iterator_tag
{};

#else // ^^^ _CCCL_FREESTANDING() ^^^ / vvv _CCCL_HOSTED() vvv

using input_iterator_tag         = ::std::input_iterator_tag;
using output_iterator_tag        = ::std::output_iterator_tag;
using forward_iterator_tag       = ::std::forward_iterator_tag;
using bidirectional_iterator_tag = ::std::bidirectional_iterator_tag;
using random_access_iterator_tag = ::std::random_access_iterator_tag;

#  if _CCCL_STD_VER >= 2020
struct _CCCL_TYPE_VISIBILITY_DEFAULT __contiguous_iterator_tag_backfill : public ::std::random_access_iterator_tag
{};
using contiguous_iterator_tag =
  _If<::std::__cccl_std_contiguous_iterator_tag_exists::value,
      ::std::contiguous_iterator_tag,
      __contiguous_iterator_tag_backfill>;
#  else // ^^^ C++20 ^^^ / vvv C++17 vvv
struct _CCCL_TYPE_VISIBILITY_DEFAULT contiguous_iterator_tag : public random_access_iterator_tag
{};
#  endif // _CCCL_STD_VER <= 2017

#endif // _CCCL_HOSTED()

template <class _Iter>
struct __iter_traits_cache
{
  using type = __select_traits<remove_cvref_t<_Iter>, remove_cvref_t<_Iter>>;
};
template <class _Iter>
using _ITER_TRAITS = typename __iter_traits_cache<_Iter>::type;

#if defined(_GLIBCXX_DEBUG)
_CCCL_TEMPLATE(class _Iter, class _Ty, class _Range)
_CCCL_REQUIRES(_IsSame<_Iter, ::__gnu_debug::_Safe_iterator<_Ty*, _Range>>::value)
_CCCL_API inline auto __iter_concept_fn(::__gnu_debug::_Safe_iterator<_Ty*, _Range>, __priority_tag<3>)
  -> contiguous_iterator_tag;
#endif // _GLIBCXX_DEBUG
#if _CCCL_HOST_STD_LIB(LIBSTDCXX)
_CCCL_TEMPLATE(class _Iter, class _Ty, class _Range)
_CCCL_REQUIRES(_IsSame<_Iter, ::__gnu_cxx::__normal_iterator<_Ty*, _Range>>::value)
_CCCL_API inline auto __iter_concept_fn(::__gnu_cxx::__normal_iterator<_Ty*, _Range>, __priority_tag<3>)
  -> contiguous_iterator_tag;
#endif // _CCCL_HOST_STD_LIB(LIBSTDCXX)
#if _CCCL_HOST_STD_LIB(LIBCXX)
_CCCL_TEMPLATE(class _Iter, class _Ty)
_CCCL_REQUIRES(_IsSame<_Iter, ::std::__wrap_iter<_Ty*>>::value)
_CCCL_API inline auto __iter_concept_fn(::std::__wrap_iter<_Ty*>, __priority_tag<3>) -> contiguous_iterator_tag;
#elif _CCCL_HOST_STD_LIB(STL)
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_Array_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_Array_const_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_Vector_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_Vector_const_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_String_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_String_const_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_String_view_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
_CCCL_TEMPLATE(class _Iter)
_CCCL_REQUIRES(_IsSame<_Iter, class _Iter::_Span_iterator>::value)
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<3>) -> contiguous_iterator_tag;
#endif // _CCCL_HOST_STD_LIB(STL)
_CCCL_TEMPLATE(class _Iter, class _Ty)
_CCCL_REQUIRES(_IsSame<_Iter, _Ty*>::value)
_CCCL_API inline auto __iter_concept_fn(_Ty*, __priority_tag<3>) -> contiguous_iterator_tag;

template <class _Iter>
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<2>) -> typename _ITER_TRAITS<_Iter>::iterator_concept;
template <class _Iter>
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<1>) -> typename _ITER_TRAITS<_Iter>::iterator_category;
template <class _Iter>
_CCCL_API inline auto __iter_concept_fn(_Iter, __priority_tag<0>)
  -> enable_if_t<__is_primary_cccl_template<_Iter>::value && __is_primary_std_template<_Iter>::value,
                 random_access_iterator_tag>;

template <class _Iter>
using __iter_concept_t =
  decltype(::cuda::std::__iter_concept_fn<_Iter>(::cuda::std::declval<_Iter>(), __priority_tag<3>{}));

template <class _Iter, class = void>
struct __iter_concept_cache
{};

template <class _Iter>
struct __iter_concept_cache<_Iter, void_t<__iter_concept_t<_Iter>>>
{
  using type = __iter_concept_t<_Iter>;
};

template <class _Iter>
using _ITER_CONCEPT = typename __iter_concept_cache<_Iter>::type;

template <class _Tp>
_CCCL_CONCEPT __has_member_reference = _CCCL_REQUIRES_EXPR((_Tp))(typename(typename _Tp::reference));

template <class _Tp>
_CCCL_CONCEPT __has_member_pointer = _CCCL_REQUIRES_EXPR((_Tp))(typename(typename _Tp::pointer));

template <class _Tp>
_CCCL_CONCEPT __has_member_iterator_category = _CCCL_REQUIRES_EXPR((_Tp))(typename(typename _Tp::iterator_category));

// The `cpp17-*-iterator` exposition-only concepts have very similar names to the `Cpp17*Iterator` named requirements
// from `[iterator.cpp17]`. To avoid confusion between the two, the exposition-only concepts have been banished to
// a "detail" namespace indicating they have a niche use-case.
namespace __iterator_traits_detail
{
// [iterator.traits#concept:cpp17-iterator]
template <class _Iter>
_CCCL_CONCEPT __cpp17_iterator = _CCCL_REQUIRES_EXPR((_Iter), _Iter __i)(
  requires(copyable<_Iter>),
  _Satisfies(__can_reference)(*__i),
  _Same_as(_Iter&)(++__i),
  _Satisfies(__can_reference)(*__i++));

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
// If the qualified-id I​::​pointer is valid and denotes a type, then ​pointer names that type;
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_pointer_or_void(int) -> typename _Iter::pointer;
// Otherwise, it names void.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_pointer_or_void(...) -> void;

template <class _Iter>
using __iterator_traits_member_pointer_or_void =
  decltype(::cuda::std::__iterator_traits_deduce_member_pointer_or_void<_Iter>(0));

// [iterator.traits]#3.2
// [iterator.traits]#3.2.1
// If the qualified-id I​::​pointer is valid and denotes a type, then pointer names that type.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_pointer_or_arrow_or_void(int, __priority_tag<1>) ->
  typename _Iter::pointer;

// Otherwise, if decltype(​declval<I&>().operator->()) is well-formed, then pointer names that type.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_pointer_or_arrow_or_void(int, __priority_tag<0>)
  -> decltype(::cuda::std::declval<_Iter&>().operator->());

// Otherwise, pointer names void.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_pointer_or_arrow_or_void(...) -> void;

template <class _Iter>
using __iterator_traits_member_pointer_or_arrow_or_void =
  decltype(::cuda::std::__iterator_traits_deduce_member_pointer_or_arrow_or_void<_Iter>(0, __priority_tag<1>{}));

// [iterator.traits]#3.2.2
// If the qualified-id `I::reference` is valid and denotes a type, `reference` names that type.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_reference(int) -> typename _Iter::reference;
// Otherwise, `reference` names `iter-reference-t<I>`.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_reference(...) -> iter_reference_t<_Iter>;

template <class _Iter>
using __iterator_traits_member_reference = decltype(::cuda::std::__iterator_traits_deduce_member_reference<_Iter>(0));

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
using __iterator_traits_iterator_category = decltype(::cuda::std::__iterator_traits_deduce_iterator_category<_Iter>());

// [iterator.traits]#3.3
// If the qualified-id `incrementable_traits<I>::difference_type` is valid and denotes a type, then
// `difference_type` names that type;
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_difference(int) -> typename incrementable_traits<_Iter>::difference_type;
// Otherwise, it names void.
template <class _Iter>
_CCCL_API auto __iterator_traits_deduce_member_difference(...) -> void;

template <class _Iter>
using __iterator_traits_difference_type = decltype(::cuda::std::__iterator_traits_deduce_member_difference<_Iter>(0));

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
  using __cccl_primary_template = iterator_traits;
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
  || __has_iterator_concept_convertible_to<_Iter, contiguous_iterator_tag>;

// Any native pointer which is an iterator is also a contiguous iterator.
template <class _Tp>
inline constexpr bool __has_contiguous_traversal<_Tp*> = true;

template <class _Iter>
using __iter_value_type = typename iterator_traits<_Iter>::value_type;

template <class _Iter>
using __iterator_category_type = typename iterator_traits<_Iter>::iterator_category;

template <class _Iter>
using __iterator_pointer_type = typename iterator_traits<_Iter>::pointer;

template <class _Iter>
using __iter_diff_t = typename iterator_traits<_Iter>::difference_type;

template <class _Iter>
using __iter_value_type = typename iterator_traits<_Iter>::value_type;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ITERATOR_ITERATOR_TRAITS_H
