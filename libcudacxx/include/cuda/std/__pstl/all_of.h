//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_ALL_OF_H
#define _CUDA_STD___PSTL_ALL_OF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/all_of.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/not_fn.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/is_nothrow_convertible.h>
#  include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/find_if.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _Iter, class _UnaryPred)
_CCCL_REQUIRES(__has_forward_traversal<_Iter> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API bool
all_of([[maybe_unused]] const _Policy& __policy, _Iter __first, _Iter __last, _UnaryPred __pred)
{
  static_assert(indirect_unary_predicate<_UnaryPred, _Iter>,
                "cuda::std::all_of: UnaryOp must satisfy indirect_unary_predicate<Iter>");
  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__find_if, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::all_of");

    if (__first == __last)
    {
      return true;
    }
    auto __res =
      __dispatch(__policy, ::cuda::std::move(__first), __last, ::cuda::std::not_fn(::cuda::std::move(__pred)));
    return __res == __last;
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::all_of requires at least one selected backend");
    return ::cuda::std::all_of(::cuda::std::move(__first), ::cuda::std::move(__last));
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_ALL_OF_H
