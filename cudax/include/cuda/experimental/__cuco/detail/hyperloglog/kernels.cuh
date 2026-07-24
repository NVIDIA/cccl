//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_HYPERLOGLOG_KERNELS_CUH
#define _CUDAX___CUCO_DETAIL_HYPERLOGLOG_KERNELS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/group.cuh>

#include <cooperative_groups.h>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_CUDA_COMPILATION()

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes")

namespace cuda::experimental::cuco::__hyperloglog_ns
{
//! @brief Returns the global thread ID in a 1D grid
//!
//! @return The global thread ID
[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int64_t __global_thread_id() noexcept
{
  return static_cast<::cuda::std::int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
}

//! @brief Returns the grid stride of a 1D grid
//!
//! @return The grid stride
[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::std::int64_t __grid_stride() noexcept
{
  return static_cast<::cuda::std::int64_t>(gridDim.x) * blockDim.x;
}

struct __clear_kernel
{
  template <class _Config, class _RefType>
  _CCCL_DEVICE_API void operator()(_Config __config, _RefType __ref) const
  {
    // This kernel is expected to be ran with only 1 block.
    static_assert(block.static_count(grid, __config) == 1);
    __ref.__clear(this_block{__config});
  }
};

// todo(dabayer): Make this a kernel functor once cuda::launch supports occupancy calculation.
template <int _VectorSize, class _Config, class _RefType>
_CCCL_KERNEL_ATTRIBUTES void __add_shmem_vectorized(
  _Config __config, const typename _RefType::__value_type* __first, ::cuda::std::int64_t __n, _RefType __ref)
{
  using __value_type = typename _RefType::__value_type;
  // TODO: replace with ::cuda::__vector_type
  using __vector_type    = ::cuda::std::array<__value_type, _VectorSize>;
  using __local_ref_type = typename _RefType::template __rebind_scope<::cuda::std::thread_scope_block>;

  // Base address of dynamic shared memory is guaranteed to be aligned to at least 16 bytes which is
  // sufficient for this purpose
  extern __shared__ ::cuda::std::byte __local_sketch[];

  const this_block __block{__config};
  const this_grid __grid{__config};

  __local_ref_type __local_ref(::cuda::std::span{__local_sketch, __ref.__sketch_bytes()}, {});
  __local_ref.__clear(__block);
  __block.sync_aligned();

  // each thread processes VectorSize-many items per iteration
  for (auto __i = gpu_thread.rank(__grid); __i < __n / _VectorSize; __i += gpu_thread.count(__grid))
  {
    const __vector_type __vec = *reinterpret_cast<const __vector_type*>(
      ::cuda::std::assume_aligned<sizeof(__vector_type)>(__first + __i * _VectorSize));

    for (const auto& __v : __vec)
    {
      __local_ref.__add(__v);
    }
  }

  // a single thread processes the remaining items
  ::cuda::experimental::invoke_one(__grid, [&]() {
    const auto __remainder = __n % _VectorSize;
    for (int __i = 0; __i < __remainder; ++__i)
    {
      __local_ref.__add(*(__first + __n - __i - 1));
    }
  });

  __block.sync_aligned();
  __ref.__merge(__block, __local_ref);
}

// todo(dabayer): Make this a kernel functor once cuda::launch supports occupancy calculation.
template <class _Config, class _InputIt, class _RefType>
_CCCL_KERNEL_ATTRIBUTES void __add_shmem(_Config __config, _InputIt __first, ::cuda::std::int64_t __n, _RefType __ref)
{
  using __local_ref_type = typename _RefType::template __rebind_scope<::cuda::std::thread_scope_block>;

  // TODO assert alignment
  extern __shared__ ::cuda::std::byte __local_sketch[];

  const this_block __block{__config};
  const this_grid __grid{__config};

  __local_ref_type __local_ref(::cuda::std::span{__local_sketch, __ref.__sketch_bytes()}, {});
  __local_ref.__clear(__block);
  __block.sync_aligned();

  for (auto __i = gpu_thread.rank(__grid); __i < __n; __i += gpu_thread.count(__grid))
  {
    __local_ref.__add(*(__first + __i));
  }

  __block.sync_aligned();
  __ref.__merge(__block, __local_ref);
}

// todo(dabayer): Make this a kernel functor once cuda::launch supports occupancy calculation.
template <class _Config, class _InputIt, class _RefType>
_CCCL_KERNEL_ATTRIBUTES void __add_gmem(_Config __config, _InputIt __first, ::cuda::std::int64_t __n, _RefType __ref)
{
  const this_grid __grid{__config};

  for (auto __i = gpu_thread.rank(__grid); __i < __n; __i += gpu_thread.count(__grid))
  {
    __ref.__add(*(__first + __i));
  }
}

struct __merge_kernel
{
  template <class Config, class _OtherRefType, class _RefType>
  _CCCL_DEVICE_API void operator()(Config __config, _OtherRefType __other_ref, _RefType __ref) const
  {
    // This kernel is expected to be ran with only 1 block.
    static_assert(block.static_count(grid, __config) == 1);
    __ref.__merge(this_block{__config}, __other_ref);
  }
};
} // namespace cuda::experimental::cuco::__hyperloglog_ns

_CCCL_DIAG_POP

#endif // _CCCL_CUDA_COMPILATION()

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_HYPERLOGLOG_KERNELS_CUH
