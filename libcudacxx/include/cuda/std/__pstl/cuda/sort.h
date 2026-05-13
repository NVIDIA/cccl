//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_SORT_H
#define _CUDA_STD___PSTL_CUDA_SORT_H

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
_CCCL_DIAG_SUPPRESS_CLANG("-Wignored-attributes")
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")
_CCCL_DIAG_SUPPRESS_NVHPC(attribute_requires_external_linkage)

#  include <cub/device/device_merge_sort.cuh>
#  include <cub/device/device_radix_sort.cuh>
#  include <cub/device/device_transform.cuh>

_CCCL_DIAG_POP

#  include <cuda/__cmath/round_up.h>
#  include <cuda/__execution/policy.h>
#  include <cuda/__functional/always_true_false.h>
#  include <cuda/__functional/call_or.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/sort.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__execution/env.h>
#  include <cuda/std/__execution/policy.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iterator_traits.h>
#  include <cuda/std/__pstl/cuda/temporary_storage.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_one_of.h>
#  include <cuda/std/__type_traits/remove_cvref.h>
#  include <cuda/std/__utility/move.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__sort, __execution_backend::__cuda>
{
  template <class _Tp>
  using _DeviceRadixSort =
    cudaError_t (*)(void*, size_t&, CUB_NS_QUALIFIER::DoubleBuffer<_Tp>&, size_t, int, int, cudaStream_t);

  template <class _Tp, class _BinaryPredicate>
  [[nodiscard]] static _CCCL_CONSTEVAL _DeviceRadixSort<_Tp> __select_radix_impl() noexcept
  {
    if constexpr (__is_one_of_v<remove_cvref_t<_BinaryPredicate>, less<>, less<_Tp>>)
    {
      return CUB_NS_QUALIFIER::DeviceRadixSort::SortKeys;
    }
    else
    {
      return CUB_NS_QUALIFIER::DeviceRadixSort::SortKeysDescending;
    }
  }

  template <class _Policy, class _Tp, class _BinaryPredicate>
  _CCCL_HOST_API static void __radix_sort_impl(const _Policy& __policy, _Tp* __first, _Tp* __last, _BinaryPredicate)
  {
    const auto __count = static_cast<size_t>(::cuda::std::distance(__first, __last));
    auto __stream      = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);

    CUB_NS_QUALIFIER::DoubleBuffer<_Tp> __buffer{__first, nullptr};

    constexpr _DeviceRadixSort<_Tp> __device_radix_sort = __select_radix_impl<_Tp, _BinaryPredicate>();

    // Determine temporary device storage requirements for device_sort
    size_t __num_bytes = 0;
    _CCCL_TRY_CUDA_API(
      __device_radix_sort,
      "__pstl_cuda_sort: determination of device storage for cub::DeviceRadixSort::SortKeys failed",
      static_cast<void*>(nullptr),
      __num_bytes,
      __buffer,
      __count,
      0,
      static_cast<int>(sizeof(_Tp) * CHAR_BIT),
      __stream.get());

    {
      __temporary_storage<_Tp> __storage{__policy, __num_bytes, ::cuda::round_up(__count, 128)};
      __buffer.d_buffers[1] = __storage.template __get_raw_ptr<0>();

      // Run the kernel
      _CCCL_TRY_CUDA_API(
        __device_radix_sort,
        "__pstl_cuda_sort: kernel launch of cub::DeviceRadixSort::SortKeys failed",
        __storage.__get_temp_storage(),
        __num_bytes,
        __buffer,
        __count,
        0,
        static_cast<int>(sizeof(_Tp) * CHAR_BIT),
        __stream.get());

      // Need to copy the memory back
      if (__buffer.selector != 0)
      {
        _CCCL_TRY_CUDA_API(
          CUB_NS_QUALIFIER::DeviceTransform::TransformIf,
          "__pstl_cuda_sort: kernel launch of cub::DeviceTransform::TransformIf failed",
          tuple{__storage.template __get_raw_ptr<0>()},
          __first,
          __count,
          ::cuda::always_true{},
          identity{},
          __stream.get());
      }
    }

    __stream.sync();
  }

  template <class _Policy, class _InputIterator, class _BinaryPredicate>
  _CCCL_HOST_API static void
  __merge_sort_impl(const _Policy& __policy, _InputIterator __first, _InputIterator __last, _BinaryPredicate __pred)
  {
    const auto __count = ::cuda::std::distance(__first, __last);
    auto __stream      = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStreamPerThread}, __policy);

    // Run the kernel
    _CCCL_TRY_CUDA_API(
      CUB_NS_QUALIFIER::DeviceMergeSort::SortKeys,
      "__pstl_cuda_sort: kernel launch of cub::DeviceMergeSort::SortKeys failed",
      ::cuda::std::move(__first),
      __count,
      ::cuda::std::move(__pred),
      __policy);

    __stream.sync();
  }

  _CCCL_TEMPLATE(class _Policy, class _InputIterator, class _BinaryPredicate)
  _CCCL_REQUIRES(__has_forward_traversal<_InputIterator>)
  _CCCL_HOST_API void operator()(
    [[maybe_unused]] const _Policy& __policy,
    _InputIterator __first,
    _InputIterator __last,
    _BinaryPredicate __pred) const
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_InputIterator>)
    {
      try
      {
        if constexpr (CUB_NS_QUALIFIER::__can_use_radix_sort<_InputIterator, _BinaryPredicate> //
                      && __can_to_address<_InputIterator>)
        {
          __radix_sort_impl(
            __policy, ::cuda::std::to_address(__first), ::cuda::std::to_address(__last), ::cuda::std::move(__pred));
        }
        else
        {
          __merge_sort_impl(__policy, ::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
        }
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
      static_assert(__always_false_v<_Policy>, "CUDA backend of cuda::std::sort requires random access iterators");
      // TODO(miscco) Implement a GPU friendly serial sort
      // ::cuda::std::sort(::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__pred));
    }
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_SORT_H
