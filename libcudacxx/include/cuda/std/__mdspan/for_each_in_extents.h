//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MDSPAN_FOR_EACH_IN_EXTENTS_H
#define _CUDA_STD___MDSPAN_FOR_EACH_IN_EXTENTS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/extents.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Exts, size_t... _Is>
[[nodiscard]] _CCCL_API constexpr auto
__drop_highest_rank_from_extents_impl(const _Exts& __exts, integer_sequence<size_t, _Is...>) noexcept
{
  return extents<typename _Exts::index_type, _Exts::static_extent(_Is + 1)...>{__exts.extent(_Is + 1)...};
}

template <class _Exts>
[[nodiscard]] _CCCL_API constexpr auto __drop_highest_rank_from_extents(const _Exts& __exts) noexcept
{
  return ::cuda::std::__drop_highest_rank_from_extents_impl(__exts, make_index_sequence<_Exts::rank() - 1>{});
}

template <class _Callable, class _Exts, class... _Is>
_CCCL_API constexpr void __for_each_in_extents_impl(_Callable& __callable, const _Exts& __exts, _Is... __is)
{
  for (typename _Exts::index_type __i = 0; __i < __exts.extent(0); ++__i)
  {
    if constexpr (_Exts::rank() == 1)
    {
      __callable(__is..., __i);
    }
    else
    {
      const auto __new_exts = ::cuda::std::__drop_highest_rank_from_extents(__exts);
      ::cuda::std::__for_each_in_extents_impl(__callable, __new_exts, __is..., __i);
    }
  }
}

template <class _Callable, class _Exts>
_CCCL_API constexpr void __for_each_in_extents(_Callable&& __callable, const _Exts& __exts)
{
  if constexpr (_Exts::rank() == 0)
  {
    __callable();
  }
  else
  {
    ::cuda::std::__for_each_in_extents_impl(__callable, __exts);
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MDSPAN_FOR_EACH_IN_EXTENTS_H
