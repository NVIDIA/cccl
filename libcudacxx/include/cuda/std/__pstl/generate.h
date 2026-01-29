//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_GENERATE_H
#define _CUDA_STD___PSTL_GENERATE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/std/__algorithm/generate.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__iterator/concepts.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/incrementable_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__utility/move.h>

#  if _CCCL_HAS_BACKEND_CUDA()
#    include <cuda/std/__pstl/cuda/transform.h>
#  endif // _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Generator>
struct __generate_wrap_generator
{
  _Generator __gen_;

  _CCCL_HOST_API constexpr __generate_wrap_generator(_Generator __gen) noexcept(
    is_nothrow_move_constructible_v<_Generator>)
      : __gen_(::cuda::std::move(__gen))
  {}

  template <class _Tp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE invoke_result_t<const _Generator&> constexpr
  operator()(const _Tp&) const noexcept(is_nothrow_invocable_v<const _Generator&>)
  {
    return __gen_();
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(class _Policy, class _InputIterator, class _Generator)
_CCCL_REQUIRES(__has_forward_traversal<_InputIterator> _CCCL_AND is_execution_policy_v<_Policy>)
_CCCL_HOST_API void
generate([[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, _Generator __gen)
{
  static_assert(indirectly_writable<_InputIterator, invoke_result_t<_Generator>>,
                "cuda::std::generate requires InputIterator to be indirectly writable with the return value of "
                "Generator");

  if (__first == __last)
  {
    return;
  }

  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__transform, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    const auto __count = ::cuda::std::distance(__first, __last);
    (void) __dispatch(
      __policy,
      ::cuda::counting_iterator{iter_difference_t<_InputIterator>{0}},
      ::cuda::counting_iterator{__count},
      ::cuda::std::move(__first),
      __generate_wrap_generator{__gen});
  }
  else
  {
    static_assert(__always_false_v<_Policy>, "Parallel cuda::std::generate requires at least one selected backend");
    ::cuda::std::generate(::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__gen));
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___PSTL_GENERATE_H
