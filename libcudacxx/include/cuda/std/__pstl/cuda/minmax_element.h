//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_MINMAX_ELEMENT_H
#define _CUDA_STD___PSTL_CUDA_MINMAX_ELEMENT_H

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

#  include <cub/device/device_reduce.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__iterator/discard_iterator.h>
#  include <cuda/__runtime/api_wrapper.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/minmax_element.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__pstl/cuda/ensure_current_context.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__minmax_element, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator, class _BinaryPred>
  [[nodiscard]] _CCCL_HOST_API static ::cuda::std::pair<_InputIterator, _InputIterator> __par_impl(
    [[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, _BinaryPred __pred)
  {
    const auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, __policy);
    const auto __ctx    = ::cuda::std::execution::__pstl_ensure_current_ctx_for(__policy);

    // cub::DeviceReduce::ArgMinLastMax reports the first minimum and the last maximum in a single traversal, which is
    // exactly what cuda::std::minmax_element requires. Both offsets are written as int64_t.
    int64_t __ret[2]{};
    const auto __count = static_cast<int64_t>(::cuda::std::distance(__first, __last));

    // Determine temporary device storage requirements for minmax_element
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceReduce::ArgMinLastMax,
      "__pstl_cuda_minmax_element: determination of device storage for cub::DeviceReduce::ArgMinLastMax failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      __first,
      ::cuda::discard_iterator{},
      static_cast<int64_t*>(nullptr),
      ::cuda::discard_iterator{},
      static_cast<int64_t*>(nullptr),
      __count,
      __pred,
      __policy);

    {
      // Both offsets share one allocation, so they also travel back to the host in a single copy
      __temporary_storage<int64_t> __storage{__policy, __num_bytes, 2};

      // Run the reduction
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DeviceReduce::ArgMinLastMax,
        "__pstl_cuda_minmax_element: kernel launch of cub::DeviceReduce::ArgMinLastMax failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        __first,
        ::cuda::discard_iterator{},
        __storage.template __get_raw_ptr<0>(),
        ::cuda::discard_iterator{},
        __storage.template __get_raw_ptr<0>() + 1,
        __count,
        ::cuda::std::move(__pred),
        __policy);

      // Copy the result back from storage
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "__pstl_cuda_minmax_element: copy of result from device to host failed",
        ::cuda::std::addressof(__ret[0]),
        __storage.template __get_raw_ptr<0>(),
        sizeof(__ret),
        ::cudaMemcpyDefault,
        __stream.get());
    }

    __stream.sync();

    using __diff_t = iter_difference_t<_InputIterator>;
    return ::cuda::std::pair<_InputIterator, _InputIterator>{
      __first + static_cast<__diff_t>(__ret[0]), __first + static_cast<__diff_t>(__ret[1])};
  }

  template <class _Policy, class _InputIterator, class _BinaryPred>
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::pair<_InputIterator, _InputIterator> _CCCL_STATIC_CALL_OPERATOR(
    [[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, _BinaryPred __pred)
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>)
    {
      _CCCL_TRY
      {
        return __par_impl(__policy, ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
      }
      _CCCL_CATCH (const ::cuda::cuda_error& __err)
      {
        if (__err.status() == cudaErrorMemoryAllocation)
        {
          _CCCL_THROW(::std::bad_alloc);
        }
        else
        {
          _CCCL_RETHROW;
        }
      }
      _CCCL_CATCH_FALLTHROUGH
    }
    else
    {
      static_assert(__always_false_v<_Policy>,
                    "__pstl_dispatch: CUDA backend of cuda::std::minmax_element requires at least random access "
                    "iterators");
      return ::cuda::std::minmax_element(
        ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_MINMAX_ELEMENT_H
