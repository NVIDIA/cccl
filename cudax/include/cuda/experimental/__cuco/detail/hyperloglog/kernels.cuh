//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO_DETAIL_HYPERLOGLOG_KERNELS_CUH
#define _CUDAX__CUCO_DETAIL_HYPERLOGLOG_KERNELS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/array>
#include <cuda/std/span>

#include <cstddef>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

namespace cuda::experimental::cuco::detail::hyperloglog_ns
{
//! @brief Returns the global thread ID in a 1D grid
//!
//! @return The global thread ID
[[nodiscard]] _CCCL_DEVICE inline int64_t __global_thread_id() noexcept
{
  return int64_t{threadIdx.x} + int64_t{blockDim.x} * int64_t{blockIdx.x};
}

//! @brief Returns the grid stride of a 1D grid
//!
//! @return The grid stride
[[nodiscard]] _CCCL_DEVICE inline int64_t __grid_stride() noexcept
{
  return int64_t{gridDim.x} * int64_t{blockDim.x};
}

template <class RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __clear(RefType __ref)
{
  auto const __block = cooperative_groups::this_thread_block();
  if (__block.group_index().x == 0)
  {
    __ref.__clear(__block);
  }
}

template <int32_t VectorSize, class RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void
__add_shmem_vectorized(typename RefType::value_type const* __first, int64_t __n, RefType __ref)
{
  using value_type     = typename RefType::value_type;
  using vector_type    = ::cuda::std::array<value_type, VectorSize>;
  using local_ref_type = typename RefType::template with_scope<::cuda::thread_scope_block>;

  // Base address of dynamic shared memory is guaranteed to be aligned to at least 16 bytes which is
  // sufficient for this purpose
  extern __shared__ ::cuda::std::byte __local_sketch[];

  auto const __loop_stride = __grid_stride();
  auto __idx               = __global_thread_id();
  auto const __grid        = cooperative_groups::this_grid();
  auto const __block       = cooperative_groups::this_thread_block();

  local_ref_type __local_ref(::cuda::std::span{__local_sketch, __ref.__sketch_bytes()}, {});
  __local_ref.__clear(__block);
  __block.sync();

  // each thread processes VectorSize-many items per iteration
  vector_type __vec;
  while (__idx < __n / VectorSize)
  {
    __vec =
      *reinterpret_cast<vector_type*>(__builtin_assume_aligned(__first + __idx * VectorSize, sizeof(vector_type)));
    for (auto const& __i : __vec)
    {
      __local_ref.__add(__i);
    }
    __idx += __loop_stride;
  }
  // a single thread processes the remaining items
#if defined(CUCO_HAS_CG_INVOKE_ONE)
  cooperative_groups::invoke_one(__grid, [&]() {
    auto const __remainder = __n % VectorSize;
    for (int __i = 0; __i < __remainder; ++__i)
    {
      __local_ref.__add(*(__first + __n - __i - 1));
    }
  });
#else
  if (__grid.thread_rank() == 0)
  {
    auto const __remainder = __n % VectorSize;
    for (int __i = 0; __i < __remainder; ++__i)
    {
      __local_ref.__add(*(__first + __n - __i - 1));
    }
  }
#endif
  __block.sync();

  __ref.__merge(__block, __local_ref);
}

template <class InputIt, class RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __add_shmem(InputIt __first, int64_t __n, RefType __ref)
{
  using local_ref_type = typename RefType::template with_scope<::cuda::thread_scope_block>;

  // TODO assert alignment
  extern __shared__ ::cuda::std::byte __local_sketch[];

  auto const __loop_stride = __grid_stride();
  auto __idx               = __global_thread_id();
  auto const __block       = cooperative_groups::this_thread_block();

  local_ref_type __local_ref(::cuda::std::span{__local_sketch, __ref.__sketch_bytes()}, {});
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

template <class InputIt, class RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __add_gmem(InputIt __first, int64_t __n, RefType __ref)
{
  auto const __loop_stride = __grid_stride();
  auto __idx               = __global_thread_id();

  while (__idx < __n)
  {
    __ref.__add(*(__first + __idx));
    __idx += __loop_stride;
  }
}

template <class OtherRefType, class RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __merge(OtherRefType __other_ref, RefType __ref)
{
  auto const __block = cooperative_groups::this_thread_block();
  if (__block.group_index().x == 0)
  {
    __ref.__merge(__block, __other_ref);
  }
}

// TODO this kernel currently isn't being used
template <class RefType>
CCCL_DETAIL_KERNEL_ATTRIBUTES void __estimate(std::size_t* __cardinality, RefType __ref)
{
  auto const __block = cooperative_groups::this_thread_block();
  if (__block.group_index().x == 0)
  {
    auto const __estimate = __ref.__estimate(__block);
    if (__block.thread_rank() == 0)
    {
      *__cardinality = __estimate;
    }
  }
}
} // namespace cuda::experimental::cuco::detail::hyperloglog_ns

_CCCL_DIAG_POP

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_DETAIL_HYPERLOGLOG_IMPL_CUH
