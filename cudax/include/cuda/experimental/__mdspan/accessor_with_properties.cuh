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
constexpr auto count_memory_behavior_v = ::cuda::std::__type_call<type_count_if<is_memory_behavior>, Ts...>::value;

template <class... Ts>
constexpr auto count_eviction_v = ::cuda::std::__type_call<type_count_if<is_eviction_policy>, Ts...>::value;

template <class... Ts>
constexpr auto count_prefetch_v = ::cuda::std::__type_call<type_count_if<is_prefetch_spatial>, Ts...>::value;

template <class... Ts>
constexpr auto count_alignment_v = ::cuda::std::__type_call<type_count_if<is_alignment>, Ts...>::value;

template <class... Ts>
constexpr auto count_aliasing_v = ::cuda::std::__type_call<type_count_if<is_ptr_aliasing_policy>, Ts...>::value;

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
using find_memory_behavior_t =
  typename find_property<::cuda::experimental::is_memory_behavior, eviction_none_t, UserProperties...>::type;

template <typename... UserProperties>
using find_eviction_policy_t =
  typename find_property<::cuda::experimental::is_eviction_policy, eviction_none_t, UserProperties...>::type;

template <typename... UserProperties>
using find_prefetch_size_t =
  typename find_property<::cuda::experimental::is_prefetch_spatial, no_prefetch_spatial_t, UserProperties...>::type;

template <size_t _DefaultAlignment, typename... UserProperties>
using find_alignment_t = typename find_property<is_alignment, alignment_t<_DefaultAlignment>, UserProperties...>::type;

template <typename... UserProperties>
using find_aliasing_policy_t =
  typename find_property<is_ptr_aliasing_policy, ptr_no_aliasing_t, UserProperties...>::type;

} // namespace detail

/***********************************************************************************************************************
 * accessor_with_properties Forward Declaration
 **********************************************************************************************************************/

template <typename ElementType,
          typename MemoryBehavior,
          typename Restrict,
          typename Alignment,
          typename Eviction,
          typename Prefetch>
struct accessor_with_properties;

/***********************************************************************************************************************
 * accessor_reference
 **********************************************************************************************************************/

template <typename ElementType,
          typename MemoryBehavior,
          typename Restrict,
          typename Alignment,
          typename Eviction,
          typename Prefetch>
class accessor_reference
{
  friend class accessor_with_properties<ElementType, MemoryBehavior, Restrict, Alignment, Eviction, Prefetch>;

  static constexpr bool _is_restrict = ::cuda::std::is_same_v<Restrict, ptr_no_aliasing_t>;

  using pointer_type = ::cuda::std::conditional_t<_is_restrict, ElementType * _CCCL_RESTRICT, ElementType*>;

  _CCCL_HOST_DEVICE explicit accessor_reference(pointer_type ptr) noexcept
      : _ptr{ptr}
  {}

  pointer_type _ptr;

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
         ::cuda::std::is_same_v<Eviction, eviction_none_t> && ::cuda::std::is_same_v<Prefetch, no_prefetch_spatial_t>);
       *_ptr = value;),
      (::cuda::experimental::store(value, _ptr, Eviction{});));
    return *this;
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE operator ElementType() const noexcept
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (static_assert(
         ::cuda::std::is_same_v<Eviction, eviction_none_t> && ::cuda::std::is_same_v<Prefetch, no_prefetch_spatial_t>);
       return *_ptr;),
      (return ::cuda::experimental::load(_ptr, MemoryBehavior{}, Eviction{}, Prefetch{});));
  }
};

/***********************************************************************************************************************
 * accessor_with_properties Definition
 **********************************************************************************************************************/

template <typename ElementType,
          typename MemoryBehavior,
          typename Restrict,
          typename Alignment,
          typename Eviction,
          typename Prefetch>
class accessor_with_properties
{
  static_assert(!::cuda::std::is_array_v<ElementType>,
                "accessor_with_properties: template argument may not be an array type");
  static_assert(!::cuda::std::is_abstract_v<ElementType>,
                "accessor_with_properties: template argument may not be an abstract class");

  static_assert(is_memory_behavior_v<MemoryBehavior>, "Restrict must be a memory behavior");
  static_assert(is_ptr_aliasing_policy_v<Restrict>, "Restrict must be a pointer aliasing policy");
  static_assert(is_eviction_policy_v<Eviction>, "Eviction must be an eviction policy");
  static_assert(is_prefetch_spatial_v<Prefetch>, "Restrict must be a prefetch policy");
  static_assert(is_alignment_v<Alignment>, "Alignment must be an alignment policy");

  static constexpr bool _is_const     = ::cuda::std::is_const_v<ElementType>;
  static constexpr bool _is_read_only = ::cuda::std::is_same_v<MemoryBehavior, read_only_t>;
  static constexpr bool _is_restrict  = ::cuda::std::is_same_v<Restrict, ptr_no_aliasing_t>;

public:
  using offset_policy = accessor_with_properties;
  using element_type  = ElementType;
  using reference =
    ::cuda::std::conditional_t<_is_const,
                               ElementType, //
                               accessor_reference<ElementType, MemoryBehavior, Restrict, Alignment, Eviction, Prefetch>>;
  using data_handle_type = ::cuda::std::conditional_t<_is_restrict, ElementType * _CCCL_RESTRICT, ElementType*>;

  explicit accessor_with_properties() noexcept = default;

  accessor_with_properties(const accessor_with_properties&) noexcept = default;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE reference access(data_handle_type ptr, size_t i) const noexcept
  {
    auto ptr1 = ::cuda::std::assume_aligned<Alignment{}>(ptr);
    if constexpr (_is_const)
    {
      NV_IF_ELSE_TARGET(
        NV_IS_HOST,
        (static_assert(::cuda::std::is_same_v<Eviction, eviction_none_t>
                       && ::cuda::std::is_same_v<Prefetch, no_prefetch_spatial_t>);
         return ptr1[i];),
        (return ::cuda::experimental::load(ptr1 + i, MemoryBehavior{}, Eviction{}, Prefetch{});));
    }
    else
    {
      return reference{ptr1 + i};
    }
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE data_handle_type
  offset(data_handle_type ptr, size_t i) const noexcept
  {
    return ::cuda::std::assume_aligned<Alignment{}>(ptr) + i;
  }
};

/***********************************************************************************************************************
 * make_accessor_with_properties() / add_properties()
 **********************************************************************************************************************/

template <typename ElementType, typename... UserProperties>
_CCCL_NODISCARD _CCCL_HOST_DEVICE auto make_accessor_with_properties(UserProperties...) noexcept
{
  using namespace detail;
  using MemoryBehavior = find_memory_behavior_t<UserProperties...>;
  using Restrict       = find_aliasing_policy_t<UserProperties...>;
  using Alignment      = find_alignment_t<alignof(ElementType), UserProperties...>;
  using Eviction       = find_eviction_policy_t<UserProperties...>;
  using Prefetch       = find_prefetch_size_t<UserProperties...>;
  static_assert(count_memory_behavior_v<UserProperties...> <= 1, "Duplicate memory behavior found");
  static_assert(count_aliasing_v<UserProperties...> <= 1, "Duplicate memory attribute found");
  static_assert(count_aliasing_v<UserProperties...> <= 1, "Duplicate aliasing policy found");
  static_assert(count_alignment_v<UserProperties...> <= 1, "Duplicate alignment found");
  static_assert(count_eviction_v<UserProperties...> <= 1, "Duplicate eviction policy found");
  static_assert(count_prefetch_v<UserProperties...> <= 1, "Duplicate prefetch policy found");
  return accessor_with_properties<ElementType, MemoryBehavior, Restrict, Alignment, Eviction, Prefetch>();
}

template <typename E, typename T, typename L, typename A, typename... UserProperties>
_CCCL_NODISCARD _CCCL_HOST_DEVICE auto
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
