//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___HYPERLOGLOG_KERNELS_CUH
#define _CUDAX___CUCO___HYPERLOGLOG_KERNELS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/span>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

namespace cuda::experimental::cuco::__hyperloglog_ns
{
//! @brief Returns the global thread ID in a 1D grid
//!
//! @return The global thread ID
[[nodiscard]] _CCCL_DEVICE inline int64_t __global_thread_id() noexcept
{
  return static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
}

//! @brief Returns the grid stride of a 1D grid
//!
//! @return The grid stride
[[nodiscard]] _CCCL_DEVICE inline int64_t __grid_stride() noexcept
{
  return static_cast<int64_t>(gridDim.x) * blockDim.x;
}

template <class _RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __clear(_RefType __ref)
{
  const auto __block = ::cooperative_groups::this_thread_block();
  if (__block.group_index().x == 0)
  {
    __ref.__clear(__block);
  }
}

template <int _VectorSize, class _RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void
__add_shmem_vectorized(const typename _RefType::value_type* __first, int64_t __n, _RefType __ref)
{
  using __value_type     = typename _RefType::value_type;
  using __vector_type    = ::cuda::std::array<__value_type, _VectorSize>;
  using __local_ref_type = typename _RefType::template with_scope<::cuda::std::thread_scope_block>;

  // Base address of dynamic shared memory is guaranteed to be aligned to at least 16 bytes which is
  // sufficient for this purpose
  extern __shared__ ::cuda::std::byte __local_sketch[];

  const auto __loop_stride = __grid_stride();
  auto __idx               = __global_thread_id();
  const auto __grid        = ::cooperative_groups::this_grid();
  const auto __block       = ::cooperative_groups::this_thread_block();

  __local_ref_type __local_ref(::cuda::std::span{__local_sketch, __ref.__sketch_bytes()}, {});
  __local_ref.__clear(__block);
  __block.sync();

  // each thread processes VectorSize-many items per iteration
  __vector_type __vec;
  while (__idx < __n / _VectorSize)
  {
    __vec = *static_cast<const __vector_type*>(
      ::cuda::std::assume_aligned<sizeof(__vector_type)>(__first + __idx * _VectorSize));
    for (int i = 0; i < _VectorSize; ++i)
    {
      __local_ref.__add(__vec[i]);
    }

    __idx += __loop_stride;
  }
  // a single thread processes the remaining items
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12010)
  ::cooperative_groups::invoke_one(__grid, [&]() {
    const auto __remainder = __n % _VectorSize;
    for (int __i = 0; __i < __remainder; ++__i)
    {
      __local_ref.__add(*(__first + __n - __i - 1));
    }
  });
#else
  if (__grid.thread_rank() == 0)
  {
    const auto __remainder = __n % _VectorSize;
    for (int __i = 0; __i < __remainder; ++__i)
    {
      __local_ref.__add(*(__first + __n - __i - 1));
    }
  }
#endif
  __block.sync();

  __ref.__merge(__block, __local_ref);
}

template <class _InputIt, class _RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __add_shmem(_InputIt __first, int64_t __n, _RefType __ref)
{
  using __local_ref_type = typename _RefType::template with_scope<::cuda::std::thread_scope_block>;

  // TODO assert alignment
  extern __shared__ ::cuda::std::byte __local_sketch[];

  const auto __loop_stride = __grid_stride();
  auto __idx               = __global_thread_id();
  const auto __block       = ::cooperative_groups::this_thread_block();

  __local_ref_type __local_ref(::cuda::std::span{__local_sketch, __ref.__sketch_bytes()}, {});
  __local_ref.__clear(__block);
  __block.sync();

  while (__idx < __n)
  {
    __local_ref.__add(*(__first + __idx));
    __idx += __loop_stride;
  }
  __block.sync();

  __ref.__merge(__block, __local_ref);
}

template <class _InputIt, class _RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __add_gmem(_InputIt __first, int64_t __n, _RefType __ref)
{
  const auto __loop_stride = __grid_stride();
  auto __idx               = __global_thread_id();

  while (__idx < __n)
  {
    __ref.__add(*(__first + __idx));
    __idx += __loop_stride;
  }
}

template <class _OtherRefType, class _RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __merge(_OtherRefType __other_ref, _RefType __ref)
{
  const auto __block = ::cooperative_groups::this_thread_block();
  if (__block.group_index().x == 0)
  {
    __ref.__merge(__block, __other_ref);
  }
}
} // namespace cuda::experimental::cuco::__hyperloglog_ns

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___HYPERLOGLOG_KERNELS_CUH
