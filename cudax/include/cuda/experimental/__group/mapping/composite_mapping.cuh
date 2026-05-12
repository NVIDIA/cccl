//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_MAPPING_COMPOSITE_MAPPING_CUH
#define _CUDA_EXPERIMENTAL___GROUP_MAPPING_COMPOSITE_MAPPING_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/tuple>

#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/queries.cuh>
#include <cuda/experimental/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): do we want to always use uint32_t for all counts/ranks?

namespace cuda::experimental
{
template <class... _Mappings>
class composite_mapping
{
  ::cuda::std::tuple<_Mappings...> __mappings_;

  template <::cuda::std::size_t _Ip = 0, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  __map_impl(const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    const auto __result = ::cuda::std::get<_Ip>(__mappings_).map(__parent, __prev_mapping_result);
    if constexpr (_Ip + 1 < sizeof...(_Mappings))
    {
      return __map_impl<_Ip + 1>(__parent, __result);
    }
    else
    {
      return __result;
    }
  }

public:
  _CCCL_DEVICE_API constexpr composite_mapping(const _Mappings&... __mappings) noexcept(
    ::cuda::std::__fold_and_v<::cuda::std::is_nothrow_copy_constructible_v<_Mappings>...>)
      : __mappings_{__mappings...}
  {}

  [[nodiscard]] _CCCL_DEVICE_API constexpr const ::cuda::std::tuple<_Mappings...>& get() const noexcept
  {
    return __mappings_;
  }

  template <class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    return __map_impl(__parent, __prev_mapping_result);
  }
};

template <class... _Mappings>
_CCCL_DEVICE composite_mapping(const _Mappings&...) -> composite_mapping<_Mappings...>;

_CCCL_TEMPLATE(class _Lhs, class _Rhs)
_CCCL_REQUIRES(__is_group_mapping_v<_Lhs> _CCCL_AND __is_group_mapping_v<_Rhs>)
[[nodiscard]] _CCCL_DEVICE_API constexpr composite_mapping<_Lhs, _Rhs>
operator|(const _Lhs& __lhs, const _Rhs& __rhs) noexcept(
  ::cuda::std::is_nothrow_constructible_v<composite_mapping<_Lhs, _Rhs>, const _Lhs&, const _Rhs&>)
{
  return {__lhs, __rhs};
}

_CCCL_TEMPLATE(class... _LhsMappings, class _Rhs)
_CCCL_REQUIRES(__is_group_mapping_v<_Rhs>)
[[nodiscard]] _CCCL_DEVICE_API constexpr composite_mapping<_LhsMappings..., _Rhs>
operator|(const composite_mapping<_LhsMappings...>& __lhs, const _Rhs& __rhs) noexcept(
  ::cuda::std::is_nothrow_constructible_v<composite_mapping<_LhsMappings..., _Rhs>, const _LhsMappings&..., const _Rhs&>)
{
  return ::cuda::std::apply(
    [&](const auto&... __lhs_mappings) {
      return composite_mapping{__lhs_mappings..., __rhs};
    },
    __lhs.get());
}

_CCCL_TEMPLATE(class _Lhs, class... _RhsMappings)
_CCCL_REQUIRES(__is_group_mapping_v<_Lhs>)
[[nodiscard]] _CCCL_DEVICE_API constexpr composite_mapping<_Lhs, _RhsMappings...>
operator|(const _Lhs& __lhs, const composite_mapping<_RhsMappings...>& __rhs) noexcept(
  ::cuda::std::is_nothrow_constructible_v<composite_mapping<_Lhs, _RhsMappings...>, const _Lhs&, const _RhsMappings&...>)
{
  return ::cuda::std::apply(
    [&](const auto&... __rhs_mappings) {
      return composite_mapping{__lhs, __rhs_mappings...};
    },
    __rhs.get());
}

template <class... _LhsMappings, class... _RhsMappings>
[[nodiscard]] _CCCL_DEVICE_API constexpr composite_mapping<_LhsMappings..., _RhsMappings...>
operator|(const composite_mapping<_LhsMappings...>& __lhs, const composite_mapping<_RhsMappings...>& __rhs) noexcept(
  ::cuda::std::is_nothrow_constructible_v<composite_mapping<_LhsMappings..., _RhsMappings...>,
                                          const _LhsMappings&...,
                                          const _RhsMappings&...>)
{
  return ::cuda::std::apply(
    [&](const auto&... __lhs_mappings) {
      return ::cuda::std::apply(
        [&](const auto&... __rhs_mappings) {
          return composite_mapping{__lhs_mappings..., __rhs_mappings...};
        },
        __rhs.get());
    },
    __lhs.get());
}
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_MAPPING_COMPOSITE_MAPPING_CUH
