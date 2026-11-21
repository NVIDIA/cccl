// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_KERNELS_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_KERNELS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILER(NVCC)
#  include <cuda/experimental/mgmn/__algorithm/common.h>
#  include <cuda/experimental/mgmn/__algorithm/sort/multiway/agent.h>
#  include <cuda/experimental/mgmn/__algorithm/sort/multiway/partition.h>
#  include <cuda/experimental/mgmn/__algorithm/sort/multiway/policy.h>

#  include <cstddef>
#  include <cstdint>

#  include <cuda_runtime.h>

#  include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT
namespace __multiway
{
inline constexpr int mway_partition_threads_per_block = 256; // 8 warps/block, 8 boundaries/block

//! One warp per diagonal (tile boundary); lane k owns input array k (k < input.k).
template <typename KeyT, typename Offset, typename CompareOp>
_CCCL_KERNEL_ATTRIBUTES void mway_partition_kernel(
  mway_input<KeyT, Offset> input,
  Offset items_per_tile,
  Offset num_diagonals,
  Offset total,
  Offset* boundaries,
  CompareOp compare_op)
{
  const int warps_per_block = static_cast<int>(blockDim.x / 32);
  const Offset warp_id =
    static_cast<Offset>(blockIdx.x) * static_cast<Offset>(warps_per_block) + static_cast<Offset>(threadIdx.x / 32);
  if (warp_id >= num_diagonals)
  {
    return;
  }
  const int lane    = static_cast<int>(threadIdx.x & 31u);
  const Offset diag = (::cuda::std::min) (warp_id * items_per_tile, total);

  const KeyT* array  = (lane < input.k) ? input.ptrs[lane] : nullptr;
  const Offset count = (lane < input.k) ? input.counts[lane] : Offset{0};
  const Offset cut   = mway_select(array, count, input.k, diag, compare_op);

  boundaries[warp_id * mway_max_k_limit + lane] = cut;
}

template <typename Agent, typename KeyT, typename Offset, typename CompareOp>
_CCCL_KERNEL_ATTRIBUTES void __launch_bounds__(Agent::threads_per_block, Agent::min_blocks_per_sm)
  mway_merge_kernel(mway_input<KeyT, Offset> input, KeyT* keys_out, CompareOp compare_op, const Offset* boundaries)
{
  typename Agent::temp_storage_t* storage_ptr; // static up to 48 KiB, dynamic above (see mway_sort_merge_kernel)
  if constexpr (Agent::dynamic_smem)
  {
    extern __shared__ __align__(16) unsigned char mway_merge_smem[];
    storage_ptr = reinterpret_cast<typename Agent::temp_storage_t*>(mway_merge_smem);
  }
  else
  {
    __shared__ typename Agent::TempStorage temp_storage_static;
    storage_ptr = &temp_storage_static.Alias();
  }
  Agent{*storage_ptr, input, keys_out, nullptr, compare_op, boundaries, nullptr, 1, false}();
}
} // namespace __multiway
_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace __detail::__sort
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILER(NVCC)

#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SORT_MULTIWAY_KERNELS_H
