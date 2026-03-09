//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_ADJACENT_FIND_H
#define _CUDA_STD___PSTL_ADJACENT_FIND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__iterator/zip_function.h>
#  include <cuda/__iterator/zip_iterator.h>
#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/adjacent_find.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/next.h>
#  include <cuda/std/__iterator/prev.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/find_if.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy,
               class _InputIterator,
               class _BinaryPredicate = ::cuda::std::equal_to<iter_value_t<_InputIterator>>)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API _InputIterator adjacent_find(
  [[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, _BinaryPredicate __pred = {})
{
  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__find_if, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::adjacent_find");

    if (__first == __last)
    {
      return __first;
    }

    auto __zipped_ret = __dispatch(
      __policy,
      ::cuda::zip_iterator{__first, ::cuda::std::next(__first)},
      ::cuda::zip_iterator{::cuda::std::prev(__last), __last},
      ::cuda::zip_function{::cuda::std::move(__pred)});
    return ::cuda::std::get<0>(__zipped_ret.__iterators());
  }
  else
  {
    static_assert(__always_false_v<_Policy>,
                  "Parallel cuda::std::adjacent_find requires at least one selected backend");
    return ::cuda::std::adjacent_find(::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_ADJACENT_FIND_H
