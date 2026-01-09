//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_HOST_DEVICE_MDSPAN_H
#define _CUDA___MDSPAN_HOST_DEVICE_MDSPAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__mdspan/host_device_accessor.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/extent.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/rank.h>
#include <cuda/std/__type_traits/remove_all_extents.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/delegate_constructors.h>
#include <cuda/std/mdspan>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy   = ::cuda::std::layout_right,
          typename _AccessorPolicy = ::cuda::std::default_accessor<_ElementType>>
class host_mdspan : public ::cuda::std::mdspan<_ElementType, _Extents, _LayoutPolicy, host_accessor<_AccessorPolicy>>
{
public:
  _CCCL_DELEGATE_CONSTRUCTORS(
    host_mdspan, ::cuda::std::mdspan, _ElementType, _Extents, _LayoutPolicy, host_accessor<_AccessorPolicy>);

  _CCCL_API friend constexpr void swap(host_mdspan& __x, host_mdspan& __y) noexcept
  {
    swap(static_cast<__base&>(__x), static_cast<__base&>(__y));
  }
};

_CCCL_TEMPLATE(class _ElementType, class... _OtherIndexTypes)
_CCCL_REQUIRES((sizeof...(_OtherIndexTypes) > 0)
                 _CCCL_AND(::cuda::std::is_convertible_v<_OtherIndexTypes, size_t>&&...))
_CCCL_HOST_DEVICE explicit host_mdspan(_ElementType*, _OtherIndexTypes...)
  -> host_mdspan<_ElementType, ::cuda::std::extents<size_t, ::cuda::std::__maybe_static_ext<_OtherIndexTypes>...>>;

_CCCL_TEMPLATE(class _Pointer)
_CCCL_REQUIRES(::cuda::std::is_pointer_v<::cuda::std::remove_reference_t<_Pointer>>)
_CCCL_HOST_DEVICE host_mdspan(_Pointer&&)
  -> host_mdspan<::cuda::std::remove_pointer_t<::cuda::std::remove_reference_t<_Pointer>>, ::cuda::std::extents<size_t>>;

_CCCL_TEMPLATE(class _CArray)
_CCCL_REQUIRES(::cuda::std::is_array_v<_CArray> _CCCL_AND(::cuda::std::rank_v<_CArray> == 1))
_CCCL_HOST_DEVICE host_mdspan(_CArray&)
  -> host_mdspan<::cuda::std::remove_all_extents_t<_CArray>,
                 ::cuda::std::extents<size_t, ::cuda::std::extent_v<_CArray, 0>>>;

template <class _ElementType, class _OtherIndexType, size_t _Size>
_CCCL_HOST_DEVICE host_mdspan(_ElementType*, const ::cuda::std::array<_OtherIndexType, _Size>&)
  -> host_mdspan<_ElementType, ::cuda::std::dextents<size_t, _Size>>;

template <class _ElementType, class _OtherIndexType, size_t _Size>
_CCCL_HOST_DEVICE host_mdspan(_ElementType*, ::cuda::std::span<_OtherIndexType, _Size>)
  -> host_mdspan<_ElementType, ::cuda::std::dextents<size_t, _Size>>;

// This one is necessary because all the constructors take `data_handle_type`s, not
// `_ElementType*`s, and `data_handle_type` is taken from `accessor_type::data_handle_type`, which
// seems to throw off automatic deduction guides.
template <class _ElementType, class _OtherIndexType, size_t... _ExtentsPack>
_CCCL_HOST_DEVICE host_mdspan(_ElementType*, const ::cuda::std::extents<_OtherIndexType, _ExtentsPack...>&)
  -> host_mdspan<_ElementType, ::cuda::std::extents<_OtherIndexType, _ExtentsPack...>>;

template <class _ElementType, class _MappingType>
_CCCL_HOST_DEVICE host_mdspan(_ElementType*, const _MappingType&)
  -> host_mdspan<_ElementType, typename _MappingType::extents_type, typename _MappingType::layout_type>;

template <class _MappingType, class _AccessorType>
_CCCL_HOST_DEVICE host_mdspan(const typename _AccessorType::data_handle_type, const _MappingType&, const _AccessorType&)
  -> host_mdspan<typename _AccessorType::element_type,
                 typename _MappingType::extents_type,
                 typename _MappingType::layout_type,
                 _AccessorType>;

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy   = ::cuda::std::layout_right,
          typename _AccessorPolicy = ::cuda::std::default_accessor<_ElementType>>
class device_mdspan
    : public ::cuda::std::mdspan<_ElementType, _Extents, _LayoutPolicy, device_accessor<_AccessorPolicy>>
{
public:
  _CCCL_DELEGATE_CONSTRUCTORS(
    device_mdspan, ::cuda::std::mdspan, _ElementType, _Extents, _LayoutPolicy, device_accessor<_AccessorPolicy>);

  _CCCL_API friend constexpr void swap(device_mdspan& __x, device_mdspan& __y) noexcept
  {
    swap(static_cast<__base&>(__x), static_cast<__base&>(__y));
  }
};

_CCCL_TEMPLATE(class _ElementType, class... _OtherIndexTypes)
_CCCL_REQUIRES((sizeof...(_OtherIndexTypes) > 0)
                 _CCCL_AND(::cuda::std::is_convertible_v<_OtherIndexTypes, size_t>&&... && true))
_CCCL_HOST_DEVICE explicit device_mdspan(_ElementType*, _OtherIndexTypes...)
  -> device_mdspan<_ElementType, ::cuda::std::extents<size_t, ::cuda::std::__maybe_static_ext<_OtherIndexTypes>...>>;

_CCCL_TEMPLATE(class _Pointer)
_CCCL_REQUIRES(::cuda::std::is_pointer_v<::cuda::std::remove_reference_t<_Pointer>>)
_CCCL_HOST_DEVICE device_mdspan(_Pointer&&)
  -> device_mdspan<::cuda::std::remove_pointer_t<::cuda::std::remove_reference_t<_Pointer>>,
                   ::cuda::std::extents<size_t>>;

_CCCL_TEMPLATE(class _CArray)
_CCCL_REQUIRES(::cuda::std::is_array_v<_CArray> _CCCL_AND(::cuda::std::rank_v<_CArray> == 1))
_CCCL_HOST_DEVICE device_mdspan(_CArray&)
  -> device_mdspan<::cuda::std::remove_all_extents_t<_CArray>,
                   ::cuda::std::extents<size_t, ::cuda::std::extent_v<_CArray, 0>>>;

template <class _ElementType, class _OtherIndexType, size_t _Size>
_CCCL_HOST_DEVICE device_mdspan(_ElementType*, const ::cuda::std::array<_OtherIndexType, _Size>&)
  -> device_mdspan<_ElementType, ::cuda::std::dextents<size_t, _Size>>;

template <class _ElementType, class _OtherIndexType, size_t _Size>
_CCCL_HOST_DEVICE device_mdspan(_ElementType*, ::cuda::std::span<_OtherIndexType, _Size>)
  -> device_mdspan<_ElementType, ::cuda::std::dextents<size_t, _Size>>;

// This one is necessary because all the constructors take `data_handle_type`s, not
// `_ElementType*`s, and `data_handle_type` is taken from `accessor_type::data_handle_type`, which
// seems to throw off automatic deduction guides.
template <class _ElementType, class _OtherIndexType, size_t... _ExtentsPack>
_CCCL_HOST_DEVICE device_mdspan(_ElementType*, const ::cuda::std::extents<_OtherIndexType, _ExtentsPack...>&)
  -> device_mdspan<_ElementType, ::cuda::std::extents<_OtherIndexType, _ExtentsPack...>>;

template <class _ElementType, class _MappingType>
_CCCL_HOST_DEVICE device_mdspan(_ElementType*, const _MappingType&)
  -> device_mdspan<_ElementType, typename _MappingType::extents_type, typename _MappingType::layout_type>;

template <class _MappingType, class _AccessorType>
_CCCL_HOST_DEVICE
device_mdspan(const typename _AccessorType::data_handle_type, const _MappingType&, const _AccessorType&)
  -> device_mdspan<typename _AccessorType::element_type,
                   typename _MappingType::extents_type,
                   typename _MappingType::layout_type,
                   _AccessorType>;

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy   = ::cuda::std::layout_right,
          typename _AccessorPolicy = ::cuda::std::default_accessor<_ElementType>>
class managed_mdspan
    : public ::cuda::std::mdspan<_ElementType, _Extents, _LayoutPolicy, managed_accessor<_AccessorPolicy>>
{
public:
  _CCCL_DELEGATE_CONSTRUCTORS(
    managed_mdspan, ::cuda::std::mdspan, _ElementType, _Extents, _LayoutPolicy, managed_accessor<_AccessorPolicy>);

  _CCCL_API friend constexpr void swap(managed_mdspan& __x, managed_mdspan& __y) noexcept
  {
    swap(static_cast<__base&>(__x), static_cast<__base&>(__y));
  }
};

_CCCL_TEMPLATE(class _ElementType, class... _OtherIndexTypes)
_CCCL_REQUIRES((sizeof...(_OtherIndexTypes) > 0)
                 _CCCL_AND(::cuda::std::is_convertible_v<_OtherIndexTypes, size_t>&&... && true))
_CCCL_HOST_DEVICE explicit managed_mdspan(_ElementType*, _OtherIndexTypes...)
  -> managed_mdspan<_ElementType, ::cuda::std::extents<size_t, ::cuda::std::__maybe_static_ext<_OtherIndexTypes>...>>;

_CCCL_TEMPLATE(class _Pointer)
_CCCL_REQUIRES(::cuda::std::is_pointer_v<::cuda::std::remove_reference_t<_Pointer>>)
_CCCL_HOST_DEVICE managed_mdspan(_Pointer&&)
  -> managed_mdspan<::cuda::std::remove_pointer_t<::cuda::std::remove_reference_t<_Pointer>>,
                    ::cuda::std::extents<size_t>>;

_CCCL_TEMPLATE(class _CArray)
_CCCL_REQUIRES(::cuda::std::is_array_v<_CArray> _CCCL_AND(::cuda::std::rank_v<_CArray> == 1))
_CCCL_HOST_DEVICE managed_mdspan(_CArray&)
  -> managed_mdspan<::cuda::std::remove_all_extents_t<_CArray>,
                    ::cuda::std::extents<size_t, ::cuda::std::extent_v<_CArray, 0>>>;

template <class _ElementType, class _OtherIndexType, size_t _Size>
_CCCL_HOST_DEVICE managed_mdspan(_ElementType*, const ::cuda::std::array<_OtherIndexType, _Size>&)
  -> managed_mdspan<_ElementType, ::cuda::std::dextents<size_t, _Size>>;

template <class _ElementType, class _OtherIndexType, size_t _Size>
_CCCL_HOST_DEVICE managed_mdspan(_ElementType*, ::cuda::std::span<_OtherIndexType, _Size>)
  -> managed_mdspan<_ElementType, ::cuda::std::dextents<size_t, _Size>>;

// This one is necessary because all the constructors take `data_handle_type`s, not
// `_ElementType*`s, and `data_handle_type` is taken from `accessor_type::data_handle_type`, which
// seems to throw off automatic deduction guides.
template <class _ElementType, class _OtherIndexType, size_t... _ExtentsPack>
_CCCL_HOST_DEVICE managed_mdspan(_ElementType*, const ::cuda::std::extents<_OtherIndexType, _ExtentsPack...>&)
  -> managed_mdspan<_ElementType, ::cuda::std::extents<_OtherIndexType, _ExtentsPack...>>;

template <class _ElementType, class _MappingType>
_CCCL_HOST_DEVICE managed_mdspan(_ElementType*, const _MappingType&)
  -> managed_mdspan<_ElementType, typename _MappingType::extents_type, typename _MappingType::layout_type>;

template <class _MappingType, class _AccessorType>
_CCCL_HOST_DEVICE
managed_mdspan(const typename _AccessorType::data_handle_type, const _MappingType&, const _AccessorType&)
  -> managed_mdspan<typename _AccessorType::element_type,
                    typename _MappingType::extents_type,
                    typename _MappingType::layout_type,
                    _AccessorType>;

/***********************************************************************************************************************
 * Accessibility Traits
 **********************************************************************************************************************/

template <typename _Tp, typename _Ep, typename _Lp, typename _Ap>
inline constexpr bool is_host_accessible_v<::cuda::std::mdspan<_Tp, _Ep, _Lp, _Ap>> = is_host_accessible_v<_Ap>;

template <typename _Tp, typename _Ep, typename _Lp, typename _Ap>
inline constexpr bool is_device_accessible_v<::cuda::std::mdspan<_Tp, _Ep, _Lp, _Ap>> = is_device_accessible_v<_Ap>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_HOST_DEVICE_MDSPAN_H
