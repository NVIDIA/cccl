//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_STABLE_PARTITION_H
#define _CUDA_STD___PSTL_CUDA_STABLE_PARTITION_H

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

#  include <cub/device/device_partition.cuh>
#  include <cub/device/device_transform.cuh>

_CCCL_DIAG_POP

#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/stable_partition.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/identity.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__pstl/reverse.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__stable_partition, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator, class _UnaryPred>
  [[nodiscard]] _CCCL_HOST_API static _InputIterator
  __par_impl(const _Policy& __policy, _InputIterator __first, _InputIterator __last, _UnaryPred __pred)
  {
    using _OffsetType = size_t;
    using value_type  = iter_value_t<_InputIterator>;

    _OffsetType __num_selected;
    const auto __count = static_cast<_OffsetType>(::cuda::std::distance(__first, __last));
    auto __stream      = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);

    // Determine temporary device storage requirements for device_stable_partition
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DevicePartition::If,
      "__pstl_cuda_stable_partition: determination of device storage for cub::DevicePartition::If failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      static_cast<value_type*>(nullptr),
      __first,
      static_cast<_OffsetType*>(nullptr),
      __count,
      __pred,
      __stream.get());

    {
      __temporary_storage<_OffsetType, value_type> __storage{__policy, __num_bytes, 1, __count};

      // Partition cannot run inplace, so we need to first copy the input into temporary storage
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DeviceTransform::TransformIf,
        "__pstl_cuda_stable_partition: kernel launch of cub::DeviceTransform::TransformIf failed",
        tuple<_InputIterator>{__first},
        __storage.template __get_ptr<1>(),
        __count,
        CUB_NS_QUALIFIER::detail::transform::always_true_predicate{},
        identity{},
        __stream.get());

      // Run the kernel, the standard requires that the input and output range do not overlap
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DevicePartition::If,
        "__pstl_cuda_stable_partition: kernel launch of cub::DevicePartition::If failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        __storage.template __get_raw_ptr<1>(),
        __first,
        __storage.template __get_ptr<0>(),
        __count,
        ::cuda::std::move(__pred),
        __stream.get());

      // Copy the result back from storage
      _CCCL_TRY_CUDA_API(
        ::cudaMemcpyAsync,
        "__pstl_cuda_stable_partition: copy of result from device to host failed",
        ::cuda::std::addressof(__num_selected),
        __storage.template __get_ptr<0>(),
        sizeof(_OffsetType),
        ::cudaMemcpyDefault,
        __stream.get());
    }

    __stream.sync();

    // Need to reverse the elements in the second partition
    const auto __mid = __first + static_cast<iter_difference_t<_InputIterator>>(__num_selected);
    ::cuda::std::reverse(__policy, __mid, __last);
    return __mid;
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIterator, class _UnaryPred)
  _CCCL_REQUIRES(__has_forward_traversal<_InputIterator>)
  [[nodiscard]] _CCCL_HOST_API _InputIterator operator()(
    [[maybe_unused]] const _Policy& __policy, _InputIterator __first, _InputIterator __last, _UnaryPred __pred) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>)
    {
      try
      {
        return __par_impl(__policy, ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
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
                    "CUDA backend of cuda::std::stable_partition requires random access iterators");
      return ::cuda::std::stable_partition(
        ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_STABLE_PARTITION_H
