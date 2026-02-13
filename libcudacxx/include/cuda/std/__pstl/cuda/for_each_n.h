//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_FOR_EACH_N_H
#define _CUDA_STD___PSTL_CUDA_FOR_EACH_N_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_BACKEND_CUDA()

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wshadow")
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

#  include <cub/device/device_for.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/for_each_n.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/terminate.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/convert_to_integral.h>
#  include <cuda/std/__utility/move.h>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__for_each_n, __execution_backend::__cuda>
{
  template <class _Policy, class _Iter, class _Size, class _Fn>
  [[nodiscard]] _CCCL_HOST_API static _Iter
  __par_impl([[maybe_unused]] const _Policy& __policy, _Iter __first, _Size __orig_n, _Fn __func) noexcept
  {
    const auto __count = ::cuda::std::__convert_to_integral(__orig_n);

    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);

    _CCCL_TRY_CUDA_API(
      ::cub::DeviceFor::ForEachN,
      "__pstl_dispatch: kernel launch failed",
      __first,
      __count,
      ::cuda::std::move(__func),
      __stream.get());

    __stream.sync();
    return __first + __count;
  }

  template <class _Policy, class _Iter, class _Size, class _Fn>
  [[nodiscard]] _CCCL_HOST_API _CCCL_FORCEINLINE _Iter
  operator()(const _Policy& __policy, _Iter __first, _Size __orig_n, _Fn __func) const noexcept
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_Iter>)
    {
      return __par_impl(__policy, ::cuda::std::move(__first), __orig_n, ::cuda::std::move(__func));
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "__pstl_dispatch: CUDA backend of cuda::std::for_each_n requires at least random access iterators");
      return ::cuda::std::for_each_n(::cuda::std::move(__first), __orig_n, ::cuda::std::move(__func));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_FOR_EACH_N_H
