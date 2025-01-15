//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MDSPAN_ACCESSOR_WITH_PROPERTIES_H
#define _CUDAX__MDSPAN_ACCESSOR_WITH_PROPERTIES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_abstract.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/mdspan>

#include <cuda/experimental/__mdspan/load_store.cuh>
#include <cuda/experimental/__mdspan/properties.cuh>
#include <cuda/experimental/__mdspan/property_traits.cuh>

namespace cuda::experimental
{

namespace detail
{

/***********************************************************************************************************************
 * Find Duplicate Utilities
 **********************************************************************************************************************/

template <template <typename> class _Predicate>
struct type_count_if
{
  template <class... Ts>
  using __call = ::cuda::std::integral_constant<int, (_Predicate<Ts>::value + ... + 0)>;
};

template <class... Ts>
using count_eviction = ::cuda::std::__type_call<type_count_if<is_eviction_policy>, Ts...>;

template <class... Ts>
using count_prefetch = ::cuda::std::__type_call<type_count_if<is_prefetch>, Ts...>;

template <class... Ts>
using count_alignment = ::cuda::std::__type_call<type_count_if<is_alignment>, Ts...>;

template <class... Ts>
using count_aliasing = ::cuda::std::__type_call<type_count_if<is_ptr_aliasing_policy>, Ts...>;

/***********************************************************************************************************************
 * Find Property Utilities
 **********************************************************************************************************************/

template <template <typename> class _Predicate>
struct predicate_call
{
  template <typename _Property>
  using __call = ::cuda::std::bool_constant<_Predicate<_Property>::value>;
};

template <template <typename> class _Predicate, typename _DefaultValue, typename... UserProperties>
struct find_property
{
  using ret = ::cuda::std::__type_find_if<::cuda::std::__type_list<UserProperties...>, predicate_call<_Predicate>>;

  using type = ::cuda::std::__type_front<::cuda::std::__type_concat<ret, ::cuda::std::__type_list<_DefaultValue>>>;
};

template <typename... UserProperties>
using find_eviction_policy =
  find_property<::cuda::experimental::is_eviction_policy, eviction_none_t, UserProperties...>;

template <typename... UserProperties>
using find_prefetch_size = find_property<::cuda::experimental::is_prefetch, no_prefetch_t, UserProperties...>;

template <size_t _DefaultAlignment, typename... UserProperties>
using find_alignment = find_property<is_alignment, aligned_size_t<_DefaultAlignment>, UserProperties...>;

template <typename... UserProperties>
using find_aliasing_policy = find_property<is_ptr_aliasing_policy, ptr_no_aliasing_t, UserProperties...>;

} // namespace detail

/***********************************************************************************************************************
 * accessor_with_properties Forward Declaration
 **********************************************************************************************************************/

template <typename ElementType, typename Restrict, typename Alignment, typename Eviction, typename Prefetch>
struct accessor_with_properties;

/***********************************************************************************************************************
 * accessor_reference
 **********************************************************************************************************************/

template <typename ElementType, typename Restrict, typename Alignment, typename Eviction, typename Prefetch>
class accessor_reference
{
  static constexpr bool _is_restrict = ::cuda::std::is_same_v<Restrict, ptr_no_aliasing_t>;

  using pointer_type = ::cuda::std::conditional_t<_is_restrict, ElementType * _CCCL_RESTRICT, ElementType*>;

  pointer_type _ptr;

  friend class accessor_with_properties<ElementType, Restrict, Alignment, Eviction, Prefetch>;

public:
  explicit accessor_reference() noexcept = default;

  accessor_reference(const accessor_reference&) noexcept = default;

  accessor_reference(accessor_reference&&) noexcept = default;

  accessor_reference& operator=(accessor_reference&&) noexcept = default;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE accessor_reference& operator=(const accessor_reference& value) noexcept
  {
    return operator=(static_cast<ElementType>(value));
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE accessor_reference& operator=(ElementType value) noexcept
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (static_assert(
         ::cuda::std::is_same_v<Eviction, eviction_none_t> && ::cuda::std::is_same_v<Prefetch, no_prefetch_t>);
       return * _ptr = value;),
      (::cuda::experimental::store(value, _ptr, Eviction{})));
    return *this;
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE operator ElementType() const noexcept
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (static_assert(
         ::cuda::std::is_same_v<Eviction, eviction_none_t> && ::cuda::std::is_same_v<Prefetch, no_prefetch_t>);
       return *_ptr;),
      (return ::cuda::experimental::load(_ptr, read_write, Eviction{}, Prefetch{});));
  }

private:
  _CCCL_HOST_DEVICE explicit accessor_reference(pointer_type ptr) noexcept
      : _ptr{ptr}
  {}
};

/***********************************************************************************************************************
 * accessor_with_properties Definition
 **********************************************************************************************************************/

template <typename ElementType, typename Restrict, typename Alignment, typename Eviction, typename Prefetch>
class accessor_with_properties
{
  static_assert(!::cuda::std::is_array_v<ElementType>,
                "accessor_with_properties: template argument may not be an array type");
  static_assert(!::cuda::std::is_abstract_v<ElementType>,
                "accessor_with_properties: template argument may not be an abstract class");

  static_assert(is_ptr_aliasing_policy_v<Restrict>, "Restrict must be a pointer aliasing policy");
  static_assert(is_eviction_policy_v<Eviction>, "Eviction must be an eviction policy");
  static_assert(is_prefetch_v<Prefetch>, "Restrict must be a prefetch policy");
  static_assert(is_alignment_v<Alignment>, "Alignment must be an alignment policy");

  static constexpr bool _is_const_elem = ::cuda::std::is_const_v<ElementType>;
  static constexpr bool _is_restrict   = ::cuda::std::is_same_v<Restrict, ptr_no_aliasing_t>;

public:
  using offset_policy = accessor_with_properties;
  using element_type  = ElementType;
  using reference     = ::cuda::std::
    conditional_t<_is_const_elem, ElementType, accessor_reference<ElementType, Restrict, Alignment, Eviction, Prefetch>>;
  using data_handle_type = ::cuda::std::conditional_t<_is_restrict, ElementType * _CCCL_RESTRICT, ElementType*>;

  explicit accessor_with_properties() noexcept = default;

  // template <typename _OtherElementType,
  //           typename... _OtherProperties,
  //           ::cuda::std::enable_if_t<::cuda::std::is_convertible_v<_OtherElementType (*)[], ElementType (*)[]>>>
  //_CCCL_HOST_DEVICE constexpr accessor_with_properties(
  //   accessor_with_properties<_OtherElementType, _OtherProperties...>) noexcept
  //{}

  accessor_with_properties(const accessor_with_properties&) noexcept = default;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference access(data_handle_type ptr, size_t i) const noexcept
  {
    auto ptr1 = ::cuda::std::assume_aligned<Alignment::align>(ptr);
    if constexpr (_is_const_elem)
    {
      NV_IF_ELSE_TARGET(
        NV_IS_HOST,
        (static_assert(
           ::cuda::std::is_same_v<Eviction, eviction_none_t> && ::cuda::std::is_same_v<Prefetch, no_prefetch_t>);
         return ptr1[i];),
        (return ::cuda::experimental::load(ptr1 + i, read_only, Eviction{}, Prefetch{});));
    }
    else
    {
      return reference{ptr1 + i};
    }
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE data_handle_type
  offset(data_handle_type ptr, size_t i) const noexcept
  {
    return ::cuda::std::assume_aligned<Alignment::align>(ptr) + i;
  }
};

/***********************************************************************************************************************
 * make_accessor_with_properties() / add_properties()
 **********************************************************************************************************************/

template <typename ElementType, typename... UserProperties>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto make_accessor_with_properties(UserProperties...) noexcept
{
  using namespace detail;
  using Restrict  = typename find_aliasing_policy<UserProperties...>::type;
  using Alignment = typename find_alignment<alignof(ElementType), UserProperties...>::type;
  using Eviction  = typename find_eviction_policy<UserProperties...>::type;
  using Prefetch  = typename find_prefetch_size<UserProperties...>::type;
  static_assert(count_aliasing<UserProperties...>::value <= 1, "Duplicate aliasing policy found");
  static_assert(count_alignment<UserProperties...>::value <= 1, "Duplicate alignment found");
  static_assert(count_eviction<UserProperties...>::value <= 1, "Duplicate eviction policy found");
  static_assert(count_prefetch<UserProperties...>::value <= 1, "Duplicate prefetch policy found");
  return accessor_with_properties<ElementType, Restrict, Alignment, Eviction, Prefetch>();
}

template <typename E, typename T, typename L, typename A, typename... UserProperties>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
add_properties(::cuda::std::mdspan<T, E, L, A> mdspan, UserProperties... properties) noexcept
{
  static_assert(::cuda::std::is_same_v<A, ::cuda::std::default_accessor<T>>, "requires default_accessor");
  auto accessor = ::cuda::experimental::make_accessor_with_properties<T>(properties...);
  return ::cuda::std::mdspan{mdspan.data_handle(), mdspan.mapping(), accessor};
}

/***********************************************************************************************************************
 * Predefined Accessors with Properties
 **********************************************************************************************************************/

template <typename T>
using streaming_accessor = decltype(make_accessor_with_properties<const T>(::cuda::experimental::eviction_no_alloc));

template <typename T>
using cache_all_accessor = decltype(make_accessor_with_properties<T>(::cuda::experimental::eviction_last));

} // namespace cuda::experimental

#endif // _CUDAX__MDSPAN_ACCESSOR_WITH_PROPERTIES_H
