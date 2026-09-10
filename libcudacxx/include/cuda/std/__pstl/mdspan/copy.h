//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_MDSPAN_COPY_H
#define _CUDA_STD___PSTL_MDSPAN_COPY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HOSTED()

#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__fwd/mdspan.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/is_assignable.h>
#  include <cuda/std/__type_traits/is_constructible.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/remove_cvref.h>
#  include <cuda/std/__utility/forward.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _ExecutionPolicy, class _Src, class _Dst>
_CCCL_CONCEPT __mdspan_copyable =
  is_execution_policy_v<remove_cvref_t<_ExecutionPolicy>> && __is_cuda_std_mdspan_v<_Src>
  && __is_cuda_std_mdspan_v<_Dst> && is_assignable_v<typename _Dst::reference, typename _Src::reference>
  && is_constructible_v<typename _Src::extents_type, typename _Dst::extents_type>;

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _ExecutionPolicy, class _Src, class _Dst)
_CCCL_REQUIRES(__mdspan_copyable<_ExecutionPolicy, _Src, _Dst>)
_CCCL_HOST_API void copy(_ExecutionPolicy&& __policy, const _Src& __src, const _Dst& __dst)
{
  _CCCL_ASSERT(__src.extents() == __dst.extents(), "__src and __dst extents must match");
  _CCCL_ASSERT(__dst.is_unique(), "__dst's mapping must be unique");

  using __policy_t _CCCL_NODEBUG = remove_cvref_t<_ExecutionPolicy>;
  constexpr auto __tag           = ::cuda::std::execution::__pstl_algorithm::__mdspan_copy;
  constexpr auto __dispatch      = ::cuda::std::execution::__pstl_select_dispatch<__tag, __policy_t>();
  constexpr bool __can_dispatch  = ::cuda::std::execution::__pstl_can_dispatch<remove_cvref_t<decltype(__dispatch)>>;
  static_assert(__can_dispatch, "Parallel cuda::std::copy for mdspan requires at least one selected backend");

  _CCCL_NVTX_RANGE_SCOPE("cuda::std::copy");
  __dispatch(::cuda::std::forward<_ExecutionPolicy>(__policy), __src, __dst);
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HOSTED()
#endif // _CUDA_STD___PSTL_MDSPAN_COPY_H
