//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_FIND_H
#define _CUDA_STD___PSTL_FIND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__functional/equal_to_value.h>
#  include <cuda/__nvtx/nvtx.h>
#  include <cuda/std/__algorithm/find.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_comparable.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/find_if.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _Iter, class _Tp)
_CCCL_REQUIRES(__has_forward_traversal<_Iter> _CCCL_AND is_execution_policy_v<_Policy>)
[[nodiscard]] _CCCL_HOST_API _Iter
find([[maybe_unused]] const _Policy& __policy, _Iter __first, _Iter __last, const _Tp& __val)
{
  static_assert(__is_cpp17_equality_comparable_v<_Tp, iter_value_t<_Iter>>,
                "Parallel cuda::std::find requires that T is equality comparable with iter_value_t<Iter>");

  if (__first == __last)
  {
    return __first;
  }

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__find_if, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    _CCCL_NVTX_RANGE_SCOPE("cuda::std::find");
    return __dispatch(
      __policy, ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::equal_to_value<_Tp>{__val});
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::find requires at least one selected backend");
    return ::cuda::std::find(::cuda::std::move(__first), ::cuda::std::move(__last), __val);
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_FIND_IF_H
