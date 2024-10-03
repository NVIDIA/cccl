//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MDSPAN_ACCESSOR_WITH_PROPERTIES_H
#define _CUDA__MDSPAN_ACCESSOR_WITH_PROPERTIES_H

#include <cuda/std/detail/__config>

#include "cuda/std/__type_traits/integral_constant.h"
#include "cuda/std/__type_traits/is_same.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2014

#  include <cub/thread/thread_load.cuh>
#  include <cub/thread/thread_store.cuh>

#  include <cuda/annotated_ptr>
#  include <cuda/std/__concepts/__concept_macros.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/is_abstract.h>
#  include <cuda/std/__type_traits/is_array.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/type_list.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//#  if 0
//template <class _ElementType,
//          EvictionPolicy _Eviction = EvictionPolicy::Default,
//          PrefetchSize _Prefetch   = PrefetchSize::NoPrefetch,
//          typename _CacheHint      = ::cuda::access_property::normal,
//          typename _Enable         = void>
//struct accessor_with_properties;
//
///***********************************************************************************************************************
// * accessor_reference
// **********************************************************************************************************************/
//
//template <class _ElementType, EvictionPolicy _Eviction, PrefetchSize _Prefetch, typename _CacheHint>
//class accessor_reference
//{
//  using __pointer_type = _ElementType*;
//
//  __pointer_type __p;
//
//  friend class accessor_with_properties<_ElementType, _Eviction, _Prefetch, _CacheHint>;
//
//public:
//  explicit constexpr accessor_reference() noexcept = default;
//
//  accessor_reference(accessor_reference&&) = delete;
//
//  accessor_reference& operator=(accessor_reference&&) = delete;
//
//  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr accessor_reference(const accessor_reference&) = default;
//
//  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CCCL_FORCEINLINE accessor_reference&
//  operator=(const accessor_reference& __x) noexcept
//  {
//    return operator=(static_cast<_ElementType>(__x));
//  }
//
//  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE _CCCL_FORCEINLINE accessor_reference& operator=(_ElementType __x) noexcept
//  {
//    return cub::ThreadStore<_Eviction>(__p, __x);
//  }
//
//  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE operator _ElementType() const noexcept
//  {
//    return cub::ThreadLoad<_Eviction>(__p);
//  }
//
//private:
//  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE explicit accessor_reference(__pointer_type __p_) noexcept
//      : __p{__p_}
//  {}
//};
//
///***********************************************************************************************************************
// * load/store accessor_with_properties
// **********************************************************************************************************************/
//
//template <class _ElementType, EvictionPolicy _Eviction, PrefetchSize _Prefetch, typename _CacheHint>
//struct accessor_with_properties<_ElementType,
//                             _Eviction,
//                             _Prefetch,
//                             _CacheHint,
//                             _CUDA_VSTD::__enable_if_t<!_CUDA_VSTD::is_const<_ElementType>::value>>
//{
//  static_assert(!_CUDA_VSTD::is_array<_ElementType>::value,
//                "accessor_with_properties: template argument may not be an array type");
//  static_assert(!_CUDA_VSTD::is_abstract<_ElementType>::value,
//                "accessor_with_properties: template argument may not be an abstract class");
//
//  using offset_policy = accessor_with_properties;
//  using element_type  = _ElementType;
//  using reference     = ::cuda::accessor_reference<_ElementType, _Eviction, _Prefetch, _CacheHint>;
//  using data_handle_type =
//    typename _CUDA_VSTD::conditional<RestrictProperty, _ElementType*, _ElementType * _CCCL_RESTRICT>::type;
//
//  explicit constexpr accessor_with_properties() noexcept = default;
//
//  _LIBCUDACXX_TEMPLATE(class _OtherElementType)
//  _LIBCUDACXX_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _OtherElementType (*)[], element_type (*)[]))
//  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr accessor_with_properties(accessor_with_properties<_OtherElementType>) noexcept {}
//
//  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE reference access(data_handle_type __p, size_t __i) const noexcept
//  {
//    return reference{__p + __i};
//  }
//
//  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE data_handle_type offset(data_handle_type __p, size_t __i) const noexcept
//  {
//    return __p + __i;
//  }
//};
//#  endif
/***********************************************************************************************************************
 * load-only accessor_with_properties
 **********************************************************************************************************************/

#define _CCCL_RESTRICT

template <typename _ElementType, typename... P>
class accessor_reference {};

struct EvictionPolicy
{
  struct Default
  {};
  struct First
  {};
  struct Normal
  {};
  struct Last
  {};
  struct LastUse
  {};
  struct NoAllocation
  {};
};

struct PrefetchSize
{
  struct NoPrefetch
  {};
  struct Bytes64
  {};
  struct Bytes128
  {};
  struct Bytes25
  {};
};

struct AliasingPolicy
{
  struct Restrict
  {};
  struct MayAlias
  {};
};


template <typename _TypeList>
struct __find_type;

template <typename... _Types>
struct __find_type<_CUDA_VSTD::__type_list<_Types...>>
{
  template <typename _TypeToSeach>
  using __call = _CUDA_VSTD::bool_constant<(_CUDA_VSTD::is_same_v<_TypeToSeach, _Types> || ...)>;
};

template <typename>
struct __is_alignment_impl : std::false_type
{};

template <size_t align_bytes>
struct __is_alignment_impl<cuda::aligned_size_t<align_bytes>> : std::true_type
{};

struct __is_alignment
{
  template <typename _T>
  using __call = _CUDA_VSTD::bool_constant<__is_alignment_impl<_T>::value>;
};

template <typename _TypeList, typename... _Properties>
struct __find_property;

template <typename... _Types, typename... _Properties>
struct __find_property<_CUDA_VSTD::__type_list<_Types...>, _Properties...>
{
  using type = _CUDA_VSTD::__type_front<_CUDA_VSTD::__type_find_if<_CUDA_VSTD::__type_list<_Properties...>,
                                                                   __find_type<_CUDA_VSTD::__type_list<_Types...>>>>;
};

template <typename _T>
struct __is_type_list_empty;

template <typename _T, typename... _TArgs>
struct __is_type_list_empty<_CUDA_VSTD::__type_list<_T, _TArgs...>> : _CUDA_VSTD::false_type
{};

template <>
struct __is_type_list_empty<_CUDA_VSTD::__type_list<>> : _CUDA_VSTD::true_type
{};

template <typename T>
constexpr bool __is_type_list_empty_v = __is_type_list_empty<T>::value;

template <typename... _Properties>
struct __find_eviction_policy
{
  using __find_result =
    __find_property<_CUDA_VSTD::__type_list<EvictionPolicy::Default,
                                            EvictionPolicy::First,
                                            EvictionPolicy::Normal,
                                            EvictionPolicy::Last,
                                            EvictionPolicy::LastUse,
                                            EvictionPolicy::NoAllocation>,
                    _Properties...>;

  using type = _CUDA_VSTD::_If<__is_type_list_empty_v<__find_result>, EvictionPolicy::Default, __find_result>;
};

template <typename... _Properties>
struct __find_prefetch_size
{
  using __find_result = __find_property<
    _CUDA_VSTD::
      __type_list<PrefetchSize::NoPrefetch, PrefetchSize::Bytes64, PrefetchSize::Bytes128, PrefetchSize::Bytes25>,
    _Properties...>;

  using type = _CUDA_VSTD::_If<__is_type_list_empty_v<__find_result>, PrefetchSize::NoPrefetch, __find_result>;
};

template <typename... _Properties>
struct __find_cache_hint_policy
{
  using __find_result =
    __find_property<_CUDA_VSTD::__type_list<access_property::shared,
                                            access_property::global,
                                            access_property::persisting,
                                            access_property::streaming,
                                            access_property::normal>,
                    _Properties...>;

  using type = _CUDA_VSTD::_If<__is_type_list_empty_v<__find_result>, access_property::global, __find_result>;
};

template <typename... _Properties>
struct __find_aliasing_policy
{
  using __find_result =
    __find_property<_CUDA_VSTD::__type_list<AliasingPolicy::Restrict,
                                            AliasingPolicy::MayAlias>,
                    _Properties...>;

  using type = _CUDA_VSTD::_If<__is_type_list_empty_v<__find_result>, AliasingPolicy::Restrict, __find_result>;
};

template <typename DefaultAlignment, typename... _Properties>
struct __find_aligment
{
  using __find_result =
    _CUDA_VSTD::__type_front<_CUDA_VSTD::__type_find_if<_CUDA_VSTD::__type_list<_Properties...>, __is_alignment>>;

  using type = _CUDA_VSTD::_If<__is_type_list_empty_v<__find_result>, DefaultAlignment, __find_result>;
};

template <typename _ElementType, typename... _Properties>
struct accessor_with_properties
{
  static_assert(!_CUDA_VSTD::is_array<_ElementType>::value,
                "accessor_with_properties: template argument may not be an array type");
  static_assert(!_CUDA_VSTD::is_abstract<_ElementType>::value,
                "accessor_with_properties: template argument may not be an abstract class");

  using _Eviction  = __find_eviction_policy<_Properties...>;
  using _Prefetch  = __find_prefetch_size<_Properties...>;
  using _CacheHint = __find_cache_hint_policy<_Properties...>;
  using _Alignment = __find_aligment<cuda::aligned_size_t<alignof(_ElementType)>, _Properties...>;
  using _Restrict  = __find_aliasing_policy<_Properties...>;

  static constexpr bool _IsConst    = _CUDA_VSTD::is_const_v<_ElementType>;
  static constexpr bool _IsRestrict = _CUDA_VSTD::is_same_v<_Restrict, AliasingPolicy::Restrict>;

  using offset_policy    = accessor_with_properties;
  using element_type     = _ElementType;
  using reference        = _CUDA_VSTD::_If<_IsConst, _ElementType,
                                           accessor_reference<_ElementType, _Eviction, _Prefetch>>;
  using data_handle_type = _CUDA_VSTD::_If<_IsRestrict, _ElementType* _CCCL_RESTRICT, _ElementType*>;

  explicit accessor_with_properties() noexcept = default;

  template <typename _OtherElementType,
            _CUDA_VSTD::__enable_if_t<_CUDA_VSTD::is_convertible<_OtherElementType (*)[], _ElementType (*)[]>::value>>
  _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr accessor_with_properties(accessor_with_properties<_OtherElementType>) noexcept
  {}

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE reference access(data_handle_type __p, size_t __i) const noexcept
  {
    auto __p1 = _CUDA_VSTD::assume_aligned(__p, _Alignment::value);
    return cub::ThreadLoad<_Eviction>(__p + __i);
  }

  _CCCL_NODISCARD _CCCL_DEVICE _CCCL_FORCEINLINE data_handle_type offset(data_handle_type __p, size_t __i) const noexcept
  {
    return __p + __i;
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2014
#endif // _CUDA__MDSPAN_ACCESSOR_WITH_PROPERTIES_H
