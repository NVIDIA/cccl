//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_ROTATE_H
#define _CUDA_STD___PSTL_CUDA_ROTATE_H

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
#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__iterator/transform_iterator.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/rotate.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/identity.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

struct __rotate_fn
{
  size_t __middle_;

  [[nodiscard]] _CCCL_API constexpr bool operator()(size_t __index) const noexcept
  {
    return __index >= __middle_;
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__rotate, __execution_backend::__cuda>
{
  template <class _Policy, class _InputIterator>
  [[nodiscard]] _CCCL_HOST_API static _InputIterator
  __par_impl(const _Policy& __policy, _InputIterator __first, _InputIterator __middle, _InputIterator __last)
  {
    using _OffsetType = size_t;
    using value_type  = iter_value_t<_InputIterator>;

    const auto __count  = static_cast<_OffsetType>(::cuda::std::distance(__first, __last));
    const auto __count1 = static_cast<_OffsetType>(::cuda::std::distance(__first, __middle));
    auto __result       = __first + static_cast<iter_difference_t<_InputIterator>>(__count - __count1);

    // Knowing the sizes of the partitions, we can directly write into them
    using __output_wrapper_t =
      CUB_NS_QUALIFIER::detail::select::partition_distinct_output_t<_InputIterator, _InputIterator>;
    __output_wrapper_t __output_wrapper{__first, __result};

    // Determine temporary device storage requirements for cub::DevicePartition::Flagged
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DevicePartition::Flagged,
      "__pstl_cuda_rotate: determination of device storage for cub::DevicePartition::Flagged failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      static_cast<value_type*>(nullptr),
      ::cuda::transform_iterator{::cuda::counting_iterator<size_t>{0}, __rotate_fn{__count1}},
      __output_wrapper,
      static_cast<_OffsetType*>(nullptr),
      __count,
      nullptr);

    auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, __policy);

    {
      // Allocate memory for result
      __temporary_storage<_OffsetType, value_type> __storage{__policy, __num_bytes, 1, __count};

      // Partition cannot run inplace, so we need to first copy the input into temporary storage
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DeviceTransform::TransformIf,
        "__pstl_cuda_rotate: kernel launch of cub::DeviceTransform::TransformIf failed",
        tuple<_InputIterator>{::cuda::std::move(__first)},
        __storage.template __get_ptr<1>(),
        __count,
        CUB_NS_QUALIFIER::detail::transform::always_true_predicate{},
        identity{},
        __policy);

      // Run the kernel, we use the flagged kernel because we know the exact ordering we want
      _CCCL_TRY_CUDA_API(
        CUB_NS_QUALIFIER::DevicePartition::Flagged,
        "__pstl_cuda_rotate: kernel launch of cub::DevicePartition::Flagged failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        __storage.template __get_raw_ptr<1>(),
        ::cuda::transform_iterator{::cuda::counting_iterator<size_t>{0}, __rotate_fn{__count1}},
        ::cuda::std::move(__output_wrapper),
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
    _InputIterator __middle,
    _InputIterator __last) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>)
    {
      try
      {
        return __par_impl(__policy, ::cuda::std::move(__first), ::cuda::std::move(__middle), ::cuda::std::move(__last));
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
      static_assert(__always_false_v<_Policy>, "CUDA backend of cuda::std::rotate requires random access iterators");
      return ::cuda::std::rotate(::cuda::std::move(__first), ::cuda::std::move(__middle), ::cuda::std::move(__last));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_ROTATE_H
