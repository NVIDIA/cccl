//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_TEMPORARY_STORAGE_H
#define _CUDA_STD___PSTL_CUDA_TEMPORARY_STORAGE_H

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

#  include <cub/device/device_find.cuh>

_CCCL_DIAG_POP

#  include <cuda/__iterator/tabulate_output_iterator.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__memory/construct_at.h>
#  include <cuda/std/__type_traits/is_nothrow_constructible.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION

template <class _ResultType>
struct __temporary_storage_construct_result
{
  _ResultType* __res_;

  _CCCL_HOST_API __temporary_storage_construct_result(_ResultType* __res = nullptr) noexcept
      : __res_(__res)
  {}

  template <class _Index, class _Up>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void
  operator()(_Index, _Up&& __value) noexcept(is_nothrow_constructible_v<_ResultType, _Up>)
  {
    ::cuda::std::__construct_at(__res_, ::cuda::std::forward<_Up>(__value));
  }
};

template <class _ResultType, class _Resource>
struct __temporary_storage
{
  ::cuda::stream_ref __stream_;
  _Resource& __resource_;
  size_t __num_bytes_allocated_;
  _ResultType* __res_;

  [[nodiscard]] _CCCL_HOST_API static constexpr size_t __get_min_alignment() noexcept
  {
    return alignof(_ResultType) < ::cuda::mr::default_cuda_malloc_alignment
           ? ::cuda::mr::default_cuda_malloc_alignment
           : alignof(_ResultType);
  }

  [[nodiscard]] _CCCL_HOST_API static constexpr size_t __get_bytes_allocated(const size_t __num_bytes_storage) noexcept
  {
    // We want to combine the allocation of the return value and the temporary storage into a single allocation
    // However, we also want that the temporary storage is properly aligned to allow efficient vectorized access
    // This might waste some space, e.g 254 bytes for short, but given the memory available on modern devices this is
    // fine
    constexpr size_t __padding = sizeof(_ResultType) % __get_min_alignment();
    return sizeof(_ResultType) + __padding + __num_bytes_storage;
  }

  _CCCL_HOST_API __temporary_storage(::cuda::stream_ref __stream, _Resource& __resource, size_t __num_bytes)
      : __stream_(__stream)
      , __resource_(__resource)
      , __num_bytes_allocated_(__get_bytes_allocated(__num_bytes))
      , __res_(
          static_cast<_ResultType*>(__resource_.allocate(__stream_, __num_bytes_allocated_, __get_min_alignment())))
  {}

  _CCCL_HOST_API ~__temporary_storage()
  {
    __resource_.deallocate(__stream_, __res_, __num_bytes_allocated_, __get_min_alignment());
  }

  template <class _AccumT = _ResultType>
  [[nodiscard]] _CCCL_HOST_API auto __get_result_iter() noexcept
  {
    if constexpr (::cuda::std::__detail::__can_optimize_construct_at<_ResultType, _AccumT>)
    {
      return __res_;
    }
    else
    {
      return ::cuda::tabulate_output_iterator{__temporary_storage_construct_result<_ResultType>{__res_}};
    }
  }

  [[nodiscard]] _CCCL_HOST_API void* __get_temp_storage() noexcept
  {
    constexpr size_t __padding = sizeof(_ResultType) % __get_min_alignment();
    constexpr size_t __offset  = sizeof(_ResultType) + __padding;
    return static_cast<void*>(static_cast<unsigned char*>(static_cast<void*>(__res_)) + __offset);
  }
};

_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

#  include <cuda/std/__cccl/epilogue.h>

#endif /// _CCCL_HAS_BACKEND_CUDA()

#endif // _CUDA_STD___PSTL_CUDA_TEMPORARY_STORAGE_H
