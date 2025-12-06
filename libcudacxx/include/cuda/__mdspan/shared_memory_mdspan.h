//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_SHARED_MEMORY_MDSPAN_H
#define _CUDA___MDSPAN_SHARED_MEMORY_MDSPAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__mdspan/shared_memory_accessor.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/array.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__mdspan/mdspan.h>
#include <cuda/std/__type_traits/extent.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/rank.h>
#include <cuda/std/__type_traits/remove_all_extents.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/delegate_constructors.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/span> // __maybe_static_ext

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy   = ::cuda::std::layout_right,
          typename _AccessorPolicy = ::cuda::std::default_accessor<_ElementType>>
class shared_memory_mdspan
    : public ::cuda::std::mdspan<_ElementType, _Extents, _LayoutPolicy, shared_memory_accessor<_AccessorPolicy>>
{
public:
  _CCCL_DELEGATE_CONSTRUCTORS(
    shared_memory_mdspan,
    ::cuda::std::mdspan,
    _ElementType,
    _Extents,
    _LayoutPolicy,
    shared_memory_accessor<_AccessorPolicy>);

  _CCCL_API friend constexpr void swap(shared_memory_mdspan& __x, shared_memory_mdspan& __y) noexcept
  {
    swap(static_cast<__base&>(__x), static_cast<__base&>(__y));
  }
};

_CCCL_TEMPLATE(typename _ElementType, typename... _OtherIndexTypes)
_CCCL_REQUIRES((sizeof...(_OtherIndexTypes) > 0)
                 _CCCL_AND(::cuda::std::is_convertible_v<_OtherIndexTypes, ::cuda::std::size_t>&&... && true))
_CCCL_HOST_DEVICE explicit shared_memory_mdspan(_ElementType*, _OtherIndexTypes...) -> shared_memory_mdspan<
  _ElementType,
  ::cuda::std::extents<::cuda::std::size_t, ::cuda::std::__maybe_static_ext<_OtherIndexTypes>...>>;

_CCCL_TEMPLATE(typename _Pointer)
_CCCL_REQUIRES(::cuda::std::is_pointer_v<::cuda::std::remove_reference_t<_Pointer>>)
_CCCL_HOST_DEVICE shared_memory_mdspan(_Pointer&&)
  -> shared_memory_mdspan<::cuda::std::remove_pointer_t<::cuda::std::remove_reference_t<_Pointer>>,
                          ::cuda::std::extents<::cuda::std::size_t>>;

_CCCL_TEMPLATE(typename _CArray)
_CCCL_REQUIRES(::cuda::std::is_array_v<_CArray> _CCCL_AND(::cuda::std::rank_v<_CArray> == 1))
_CCCL_HOST_DEVICE shared_memory_mdspan(_CArray&)
  -> shared_memory_mdspan<::cuda::std::remove_all_extents_t<_CArray>,
                          ::cuda::std::extents<::cuda::std::size_t, ::cuda::std::extent_v<_CArray, 0>>>;

template <typename _ElementType, typename _OtherIndexType, ::cuda::std::size_t _Size>
_CCCL_HOST_DEVICE shared_memory_mdspan(_ElementType*, const ::cuda::std::array<_OtherIndexType, _Size>&)
  -> shared_memory_mdspan<_ElementType, ::cuda::std::dextents<::cuda::std::size_t, _Size>>;

template <typename _ElementType, typename _OtherIndexType, ::cuda::std::size_t _Size>
_CCCL_HOST_DEVICE shared_memory_mdspan(_ElementType*, ::cuda::std::span<_OtherIndexType, _Size>)
  -> shared_memory_mdspan<_ElementType, ::cuda::std::dextents<::cuda::std::size_t, _Size>>;

// This one is necessary because all the constructors take `data_handle_type`s, not
// `_ElementType*`s, and `data_handle_type` is taken from `accessor_type::data_handle_type`, which
// seems to throw off automatic deduction guides.
template <typename _ElementType, typename _OtherIndexType, ::cuda::std::size_t... _ExtentsPack>
_CCCL_HOST_DEVICE shared_memory_mdspan(_ElementType*, const ::cuda::std::extents<_OtherIndexType, _ExtentsPack...>&)
  -> shared_memory_mdspan<_ElementType, ::cuda::std::extents<_OtherIndexType, _ExtentsPack...>>;

template <typename _ElementType, typename _MappingType>
_CCCL_HOST_DEVICE shared_memory_mdspan(_ElementType*, const _MappingType&)
  -> shared_memory_mdspan<_ElementType, typename _MappingType::extents_type, typename _MappingType::layout_type>;

template <typename _MappingType, typename _AccessorType>
_CCCL_HOST_DEVICE
shared_memory_mdspan(const typename _AccessorType::data_handle_type, const _MappingType&, const _AccessorType&)
  -> shared_memory_mdspan<typename _AccessorType::element_type,
                          typename _MappingType::extents_type,
                          typename _MappingType::layout_type,
                          _AccessorType>;

/***********************************************************************************************************************
 * Accessibility Traits
 **********************************************************************************************************************/

template <typename>
inline constexpr bool is_shared_memory_mdspan_v = false;

template <typename _Tp, typename _Ep, typename _Lp, typename _Ap>
inline constexpr bool is_shared_memory_mdspan_v<shared_memory_mdspan<_Tp, _Ep, _Lp, _Ap>> = true;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_SHARED_MEMORY_MDSPAN_H
