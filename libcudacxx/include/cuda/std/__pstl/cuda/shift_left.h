//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_SHIFT_LEFT_H
#define _CUDA_STD___PSTL_CUDA_SHIFT_LEFT_H

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
_CCCL_DIAG_SUPPRESS_CLANG("-Wunused-local-typedef")
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

#  include <cub/device/device_select.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__iterator/transform_iterator.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/shift_left.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/incrementable_traits.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

struct __shift_left_predicate
{
  size_t __start_;

  [[nodiscard]] _CCCL_API constexpr bool operator()(size_t __index) const noexcept
  {
    return __index >= __start_;
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__shift_left, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator>
  [[nodiscard]] _CCCL_HOST_API static _InputIterator __par_impl(
    const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    iter_difference_t<_InputIterator> __num_shifted)
  {
    using _OffsetType   = iter_difference_t<_InputIterator>;
    const auto __count  = ::cuda::std::distance(__first, __last);
    const auto __result = __first + static_cast<_OffsetType>(__count - __num_shifted);
    auto __flag_iter    = ::cuda::transform_iterator{
      ::cuda::counting_iterator<size_t>{0}, __shift_left_predicate{static_cast<size_t>(__num_shifted)}};

    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);

    // Determine temporary device storage requirements for DeviceSelect::Flagged
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceSelect::Flagged,
      "__pstl_cuda_shift_left: determination of device storage for cub::DeviceSelect::Flagged failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      __first,
      __flag_iter,
      static_cast<_OffsetType*>(nullptr),
      __count,
      __stream.get());

    {
      __temporary_storage<_OffsetType> __storage{__policy, __num_bytes, 1};

      // Run the kernel
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DeviceSelect::Flagged,
        "__pstl_cuda_shift_left: kernel launch of cub::DeviceSelect::Flagged failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        ::cuda::std::move(__first),
        ::cuda::std::move(__flag_iter),
        __storage.template __get_ptr<0>(),
        __count,
        __stream.get());
    }

    __stream.sync();
    return __result;
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIterator)
  _CCCL_REQUIRES(__has_forward_traversal<_InputIterator>)
  [[nodiscard]] _CCCL_HOST_API _InputIterator operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    iter_difference_t<_InputIterator> __num_shifted) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>)
    {
      try
      {
        return __par_impl(__policy, ::cuda::std::move(__first), ::cuda::std::move(__last), __num_shifted);
      }
      catch (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == cudaErrorMemoryAllocation)
        {
          _CCCL_THROW(::std::bad_alloc);
        }
        else
        {
          throw __err;
        }
      }
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "CUDA backend of cuda::std::shift_left requires random access iterators");
      return ::cuda::std::shift_left(::cuda::std::move(__first), ::cuda::std::move(__last), __num_shifted);
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_SHIFT_LEFT_H
