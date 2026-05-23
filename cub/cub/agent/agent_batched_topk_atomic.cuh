// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_topk.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_topk.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/atomic>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cooperative_groups.h>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk_atomic
{
struct atomic_topk_policy
{
  int cluster_size;
  int threads_per_block;
};

template <int ThreadsPerBlock,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
struct agent_batched_topk_atomic
{
  using key_it_t = it_value_t<KeyInputItItT>;
  using key_t    = it_value_t<key_it_t>;

  using segment_size_val_t = typename SegmentSizeParameterT::value_type;
  using num_segments_val_t = typename NumSegmentsParameterT::value_type;

  using offset_t     = ::cuda::std::uint32_t;
  using out_offset_t = ::cuda::std::uint32_t;

  // TODO(gevtushenko): direction
  using twiddle_t     = cub::RadixSortTwiddle<false, key_t>;
  using bit_ordered_t = typename cub::Traits<key_t>::UnsignedBits;

  static constexpr auto threads_per_block = ThreadsPerBlock;
  static constexpr auto max_k             = params::static_max_value_v<KParameterT>;

  using decomposer_t = detail::identity_decomposer_t;
  using block_scan_t = BlockScan<offset_t, threads_per_block, BLOCK_SCAN_WARP_SCANS>;

  struct _TempStorage
  {
    bit_ordered_t top_k[max_k];
  };

  struct TempStorage : Uninitialized<_TempStorage>
  {};

  _TempStorage& temp_storage;
  KeyInputItItT d_key_segments_it;
  KeyOutputItItT d_key_segments_out_it;
  SegmentSizeParameterT segment_sizes;
  KParameterT k_param;
  SelectDirectionParameterT select_directions;
  NumSegmentsParameterT num_segments;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE agent_batched_topk_atomic(
    TempStorage& temp_storage_,
    KeyInputItItT d_key_segments_it_,
    KeyOutputItItT d_key_segments_out_it_,
    SegmentSizeParameterT segment_sizes_,
    KParameterT k_param_,
    SelectDirectionParameterT select_directions_,
    NumSegmentsParameterT num_segments_)
      : temp_storage(temp_storage_.Alias())
      , d_key_segments_it(d_key_segments_it_)
      , d_key_segments_out_it(d_key_segments_out_it_)
      , segment_sizes(segment_sizes_)
      , k_param(k_param_)
      , select_directions(select_directions_)
      , num_segments(num_segments_)
  {}

  _CCCL_DEVICE_API _CCCL_FORCEINLINE void Process()
  {
    const auto segment_id   = static_cast<num_segments_val_t>(blockIdx.x);
    const auto direction    = select_directions.get_param(segment_id);
    const auto segment_size = static_cast<segment_size_val_t>(segment_sizes.get_param(segment_id));
    const auto k_requested  = static_cast<out_offset_t>(k_param.get_param(segment_id));
    const auto k =
      static_cast<out_offset_t>((::cuda::std::min) (static_cast<segment_size_val_t>(k_requested), segment_size));

    if (k == 0)
    {
      return;
    }

    // TODO(gevtushenk); direction
    (void) direction;
    bit_ordered_t sentinel{0};

    init_smem(k, sentinel);
    deposit(segment_id, segment_size, k, sentinel);
    merge(segment_id, k);
    store(segment_id, k);
  }

private:
  __device__ void init_smem(unsigned k, bit_ordered_t sentinel)
  {
    for (unsigned i = threadIdx.x; i < k; i += threads_per_block)
    {
      temp_storage.top_k[i] = sentinel;
    }
    __syncthreads();
  }

  __device__ void deposit(unsigned segment_id, unsigned segment_size, unsigned k, bit_ordered_t sentinel)
  {
    auto min           = sentinel;
    auto block_keys_in = d_key_segments_it[segment_id];
    for (unsigned i = threadIdx.x; i < segment_size; i += threads_per_block)
    {
      auto bit_ordered_key = twiddle_t::In(reinterpret_cast<const bit_ordered_t&>(block_keys_in[i]));
      min                  = block_top_k(bit_ordered_key, temp_storage.top_k, k);
    }
    __syncthreads();
  }

  __device__ void merge(unsigned segment_id, unsigned k)
  {
    // TODO(gevtushenko): cluster
  }

  __device__ void store(unsigned segment_id, unsigned k)
  {
    auto block_keys_out = d_key_segments_out_it[segment_id];

    for (unsigned i = threadIdx.x; i < k; i += threads_per_block)
    {
      auto key          = twiddle_t::Out(temp_storage.top_k[i]);
      block_keys_out[i] = reinterpret_cast<const key_t&>(key);
    }
  }

  __device__ bit_ordered_t block_top_k(bit_ordered_t key, bit_ordered_t* top, unsigned k)
  {
    for (unsigned j = 0; j < k; ++j)
    {
      const bit_ordered_t old_max_j = atomicMax_block(top + j, key);
      if (old_max_j < key)
      {
        key = old_max_j;
      }
    }
    return key;
  }
};
} // namespace detail::batched_topk_atomic

CUB_NAMESPACE_END
