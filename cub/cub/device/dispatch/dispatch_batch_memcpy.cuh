// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! \file
//! cub::detail::batch_memcpy::dispatch provides device-wide, parallel operations for copying data from a number of
//! given source buffers to their corresponding destination buffer.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_batch_memcpy.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/detail/temporary_storage.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/tuning/tuning_batch_memcpy.cuh>
#include <cub/thread/thread_search.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_ptx.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

enum class CopyAlg
{
  Memcpy,
  Copy
};

namespace detail::batch_memcpy
{
// Type used to specialize the kernel templates for indexing the buffers processed within a single kernel invocation
using per_invocation_buffer_offset_t = ::cuda::std::uint32_t;

/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <typename BufferOffsetScanTileStateT, typename BlockOffsetScanTileStateT, typename TileOffsetT>
_CCCL_KERNEL_ATTRIBUTES void InitTileStateKernel(
  BufferOffsetScanTileStateT buffer_offset_scan_tile_state,
  BlockOffsetScanTileStateT block_offset_scan_tile_state,
  _CCCL_GRID_CONSTANT const TileOffsetT num_tiles)
{
  // Initialize tile status
  buffer_offset_scan_tile_state.InitializeStatus(num_tiles);
  block_offset_scan_tile_state.InitializeStatus(num_tiles);
}

/**
 * Kernel that copies buffers that need to be copied by at least one (and potentially many) thread
 * blocks.
 */
template <typename PolicySelector,
          typename BufferOffsetT,
          typename InputBufferIt,
          typename OutputBufferIt,
          typename BufferSizeIteratorT,
          typename BufferTileOffsetItT,
          typename TileT,
          typename TileOffsetT,
          CopyAlg MemcpyOpt>
#if _CCCL_HAS_CONCEPTS()
  requires batch_memcpy_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().large_buffer.block_threads))
  _CCCL_KERNEL_ATTRIBUTES void MultiBlockBatchMemcpyKernel(
    _CCCL_GRID_CONSTANT const InputBufferIt input_buffer_it,
    _CCCL_GRID_CONSTANT const OutputBufferIt output_buffer_it,
    _CCCL_GRID_CONSTANT const BufferSizeIteratorT buffer_sizes,
    _CCCL_GRID_CONSTANT const BufferTileOffsetItT buffer_tile_offsets,
    TileT buffer_offset_tile,
    _CCCL_GRID_CONSTANT const TileOffsetT last_tile_offset)
{
  static constexpr large_buffer_policy policy = current_policy<PolicySelector>().large_buffer;
  using StatusWord                            = typename TileT::StatusWord;
  using BufferSizeT                           = it_value_t<BufferSizeIteratorT>;
  /// Internal load/store type. For byte-wise memcpy, a single-byte type
  using AliasT = typename ::cuda::std::conditional_t<MemcpyOpt == CopyAlg::Memcpy,
                                                     ::cuda::std::type_identity<char>,
                                                     lazy_trait<it_value_t, it_value_t<InputBufferIt>>>::type;
  /// Types of the input and output buffers
  using InputBufferT  = it_value_t<InputBufferIt>;
  using OutputBufferT = it_value_t<OutputBufferIt>;

  constexpr uint32_t BLOCK_THREADS    = static_cast<uint32_t>(policy.block_threads);
  constexpr uint32_t ITEMS_PER_THREAD = static_cast<uint32_t>(policy.bytes_per_thread);
  constexpr BufferSizeT TILE_SIZE     = static_cast<BufferSizeT>(BLOCK_THREADS * ITEMS_PER_THREAD);

  BufferOffsetT num_blev_buffers = buffer_offset_tile.LoadValid(last_tile_offset);

  uint32_t tile_id = blockIdx.x;

  // No block-level buffers => we're done here
  if (num_blev_buffers == 0)
  {
    return;
  }

  // While there's still tiles of bytes from block-level buffers to copied
  do
  {
    __shared__ BufferOffsetT block_buffer_id;

    // Make sure thread 0 does not overwrite the buffer id before other threads have finished with
    // the prior iteration of the loop
    __syncthreads();

    // Binary search the buffer that this tile belongs to
    if (threadIdx.x == 0)
    {
      block_buffer_id = UpperBound(buffer_tile_offsets, num_blev_buffers, tile_id) - 1;
    }

    // Make sure thread 0 has written the buffer this thread block is assigned to
    __syncthreads();

    const BufferOffsetT buffer_id = block_buffer_id;

    // The relative offset of this tile within the buffer it's assigned to
    BufferSizeT tile_offset_within_buffer =
      static_cast<BufferSizeT>(tile_id - buffer_tile_offsets[buffer_id]) * TILE_SIZE;

    // If the tile has already reached beyond the work of the end of the last buffer
    if (buffer_id >= num_blev_buffers - 1 && tile_offset_within_buffer > buffer_sizes[buffer_id])
    {
      return;
    }

    // Tiny remainders are copied without vectorizing loads
    if (buffer_sizes[buffer_id] - tile_offset_within_buffer <= 32)
    {
      BufferSizeT thread_offset = tile_offset_within_buffer + threadIdx.x;
      for (int i = 0; i < ITEMS_PER_THREAD; i++)
      {
        if (thread_offset < buffer_sizes[buffer_id])
        {
          const auto value = read_item < MemcpyOpt == CopyAlg::Memcpy, AliasT,
                     InputBufferT > (input_buffer_it[buffer_id], thread_offset);
          write_item<MemcpyOpt == CopyAlg::Memcpy, AliasT, OutputBufferT>(
            output_buffer_it[buffer_id], thread_offset, value);
        }
        thread_offset += BLOCK_THREADS;
      }
    }
    else
    {
      copy_items<MemcpyOpt == CopyAlg::Memcpy, BLOCK_THREADS, InputBufferT, OutputBufferT, BufferSizeT>(
        input_buffer_it[buffer_id],
        output_buffer_it[buffer_id],
        (::cuda::std::min) (buffer_sizes[buffer_id] - tile_offset_within_buffer, TILE_SIZE),
        tile_offset_within_buffer);
    }

    tile_id += gridDim.x;
  } while (true);
}

/**
 * @brief Kernel that copies data from a batch of given source buffers to their corresponding
 * destination buffer. If a buffer's size is too large to be copied by a single thread block, that
 * buffer is put into a queue of buffers that will get picked up later on, where multiple blocks
 * collaborate on each of these buffers. All other buffers get copied straight away.o
 *
 * @param input_buffer_it [in] Iterator providing the pointers to the source memory buffers
 * @param output_buffer_it [in] Iterator providing the pointers to the destination memory buffers
 * @param buffer_sizes [in] Iterator providing the number of bytes to be copied for each pair of
 * buffers
 * @param num_buffers [in] The total number of buffer pairs
 * @param blev_buffer_srcs [out] The source pointers of buffers that require block-level
 * collaboration
 * @param blev_buffer_dsts [out] The destination pointers of buffers that require block-level
 * collaboration
 * @param blev_buffer_sizes [out] The sizes of buffers that require block-level collaboration
 * @param blev_buffer_scan_state [in,out] Tile states for the prefix sum over the count of buffers
 * requiring block-level collaboration (to "stream compact" (aka "select") BLEV-buffers)
 * @param blev_block_scan_state [in,out] Tile states for the prefix sum over the number of thread
 * blocks getting assigned to each buffer that requires block-level collaboration
 */
template <typename PolicySelector,
          typename InputBufferIt,
          typename OutputBufferIt,
          typename BufferSizeIteratorT,
          typename BufferOffsetT,
          typename BlevBufferSrcsOutItT,
          typename BlevBufferDstsOutItT,
          typename BlevBufferSizesOutItT,
          typename BlevBufferTileOffsetsOutItT,
          typename BlockOffsetT,
          typename BLevBufferOffsetTileState,
          typename BLevBlockOffsetTileState,
          CopyAlg MemcpyOpt>
#if _CCCL_HAS_CONCEPTS()
  requires batch_memcpy_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().small_buffer.block_threads))
  _CCCL_KERNEL_ATTRIBUTES void BatchMemcpyKernel(
    _CCCL_GRID_CONSTANT const InputBufferIt input_buffer_it,
    _CCCL_GRID_CONSTANT const OutputBufferIt output_buffer_it,
    _CCCL_GRID_CONSTANT const BufferSizeIteratorT buffer_sizes,
    _CCCL_GRID_CONSTANT const BufferOffsetT num_buffers,
    _CCCL_GRID_CONSTANT const BlevBufferSrcsOutItT blev_buffer_srcs,
    _CCCL_GRID_CONSTANT const BlevBufferDstsOutItT blev_buffer_dsts,
    _CCCL_GRID_CONSTANT const BlevBufferSizesOutItT blev_buffer_sizes,
    _CCCL_GRID_CONSTANT const BlevBufferTileOffsetsOutItT blev_buffer_tile_offsets,
    _CCCL_GRID_CONSTANT const BLevBufferOffsetTileState blev_buffer_scan_state,
    _CCCL_GRID_CONSTANT const BLevBlockOffsetTileState blev_block_scan_state)
{
  static constexpr small_buffer_policy policy = current_policy<PolicySelector>().small_buffer;
  // Internal type used for storing a buffer's size
  using BufferSizeT = it_value_t<BufferSizeIteratorT>;

  // TODO(bgruber): refactor this in C++20, when we can pass policy as NTTP
  using AgentBatchMemcpyPolicyT = AgentBatchMemcpyPolicy<
    policy.block_threads,
    policy.buffers_per_thread,
    policy.tlev_bytes_per_thread,
    policy.prefer_pow2_bits,
    policy.block_level_tile_size,
    policy.warp_level_threshold,
    policy.block_level_threshold,
    delay_constructor_t<policy.buff_delay_constructor.kind,
                        policy.buff_delay_constructor.delay,
                        policy.buff_delay_constructor.l2_write_latency>,
    delay_constructor_t<policy.block_delay_constructor.kind,
                        policy.block_delay_constructor.delay,
                        policy.block_delay_constructor.l2_write_latency>>;

  // Block-level specialization
  using AgentBatchMemcpyT = AgentBatchMemcpy<
    AgentBatchMemcpyPolicyT,
    InputBufferIt,
    OutputBufferIt,
    BufferSizeIteratorT,
    BufferOffsetT,
    BlevBufferSrcsOutItT,
    BlevBufferDstsOutItT,
    BlevBufferSizesOutItT,
    BlevBufferTileOffsetsOutItT,
    BlockOffsetT,
    BLevBufferOffsetTileState,
    BLevBlockOffsetTileState,
    MemcpyOpt == CopyAlg::Memcpy>;

  // Shared memory for AgentBatchMemcpy
  __shared__ typename AgentBatchMemcpyT::TempStorage temp_storage;

  // Process this block's tile of input&output buffer pairs
  AgentBatchMemcpyT(
    temp_storage,
    input_buffer_it,
    output_buffer_it,
    buffer_sizes,
    num_buffers,
    blev_buffer_srcs,
    blev_buffer_dsts,
    blev_buffer_sizes,
    blev_buffer_tile_offsets,
    blev_buffer_scan_state,
    blev_block_scan_state)
    .ConsumeTile(blockIdx.x);
}

//! @tparam BlockOffsetT Integer type large enough to hold any offset in [0, num_thread_blocks_launched)
//! @tparam InputBufferIt **[inferred]** Random-access input iterator type providing the pointers to the source memory
//! buffers
//! @tparam OutputBufferIt **[inferred]** Random-access input iterator type providing the pointers to the destination
//! memory buffers
//! @tparam BufferSizeIteratorT **[inferred]** Random-access input iterator type providing the number of bytes to be
//! copied for each pair of buffers
template <CopyAlg MemcpyOpt = CopyAlg::Memcpy,
          typename BlockOffsetT,
          typename InputBufferIt,
          typename OutputBufferIt,
          typename BufferSizeIteratorT,
          typename PolicySelectorT = policy_selector>
#if _CCCL_HAS_CONCEPTS()
  requires batch_memcpy_policy_selector<PolicySelectorT>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputBufferIt input_buffer_it,
  OutputBufferIt output_buffer_it,
  BufferSizeIteratorT buffer_sizes,
  ::cuda::std::int64_t num_buffers,
  cudaStream_t stream,
  PolicySelectorT policy_selector = {})
{
  using per_invocation_buffer_offset_t = detail::batch_memcpy::per_invocation_buffer_offset_t;
  using BufferSizeT                    = cub::detail::it_value_t<BufferSizeIteratorT>;
  using BLevBufferOffsetTileState      = cub::ScanTileState<per_invocation_buffer_offset_t>;
  using BLevBlockOffsetTileState       = cub::ScanTileState<BlockOffsetT>;

  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(ptx_arch_id(arch_id)))
  {
    return error;
  }
  const batch_memcpy_policy active_policy = policy_selector(arch_id);

#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
  NV_IF_TARGET(NV_IS_HOST, ({
                 ::std::stringstream ss;
                 ss << active_policy;
                 _CubLog("Dispatching DeviceBatchMemcpy to arch %d with tuning: %s\n",
                         static_cast<int>(arch_id),
                         ss.str().c_str());
               }))
#endif

  enum : uint32_t
  {
    // Memory for the source pointers of the buffers that require block-level collaboration
    MEM_BLEV_BUFFER_SRCS = 0,
    // Memory for the destination pointers of the buffers that require block-level collaboration
    MEM_BLEV_BUFFER_DSTS,
    // Memory for the block-level buffers' sizes
    MEM_BLEV_BUFFER_SIZES,
    // Memory to keep track of the assignment of thread blocks to block-level buffers
    MEM_BLEV_BUFFER_TBLOCK,
    // Memory for the tile states of the prefix sum over the number of buffers that require
    // block-level collaboration
    MEM_BLEV_BUFFER_SCAN_STATE,
    // Memory for the scan tile states of the prefix sum over the number of thread block's
    // assigned up to and including a certain block-level buffer
    MEM_BLEV_BLOCK_SCAN_STATE,
    MEM_NUM_ALLOCATIONS
  };

  constexpr BlockOffsetT init_kernel_threads = 128U;
  const auto tile_size                       = static_cast<uint32_t>(active_policy.small_buffer.block_threads)
                       * static_cast<uint32_t>(active_policy.small_buffer.buffers_per_thread);

  constexpr auto max_num_buffers_per_invocation = ::cuda::std::int64_t{512 * 1024 * 1024};
  static_assert(max_num_buffers_per_invocation <= ::cuda::std::numeric_limits<per_invocation_buffer_offset_t>::max());
  const auto max_num_buffers = ::cuda::std::min(max_num_buffers_per_invocation, num_buffers);
  const auto max_num_tiles   = static_cast<BlockOffsetT>(::cuda::ceil_div(max_num_buffers, tile_size));

  using BlevBufferSrcsOutT =
    ::cuda::std::_If<MemcpyOpt == CopyAlg::Memcpy, const void*, cub::detail::it_value_t<InputBufferIt>>;
  using BlevBufferDstOutT =
    ::cuda::std::_If<MemcpyOpt == CopyAlg::Memcpy, void*, cub::detail::it_value_t<OutputBufferIt>>;
  using BlevBufferSrcsOutItT        = BlevBufferSrcsOutT*;
  using BlevBufferDstsOutItT        = BlevBufferDstOutT*;
  using BlevBufferSizesOutItT       = BufferSizeT*;
  using BlevBufferTileOffsetsOutItT = BlockOffsetT*;

  temporary_storage::layout<MEM_NUM_ALLOCATIONS> temporary_storage_layout;

  auto blev_buffer_srcs_slot       = temporary_storage_layout.get_slot(MEM_BLEV_BUFFER_SRCS);
  auto blev_buffer_dsts_slot       = temporary_storage_layout.get_slot(MEM_BLEV_BUFFER_DSTS);
  auto blev_buffer_sizes_slot      = temporary_storage_layout.get_slot(MEM_BLEV_BUFFER_SIZES);
  auto blev_buffer_block_slot      = temporary_storage_layout.get_slot(MEM_BLEV_BUFFER_TBLOCK);
  auto blev_buffer_scan_slot       = temporary_storage_layout.get_slot(MEM_BLEV_BUFFER_SCAN_STATE);
  auto blev_buffer_block_scan_slot = temporary_storage_layout.get_slot(MEM_BLEV_BLOCK_SCAN_STATE);

  auto blev_buffer_srcs_alloc  = blev_buffer_srcs_slot->template create_alias<BlevBufferSrcsOutT>();
  auto blev_buffer_dsts_alloc  = blev_buffer_dsts_slot->template create_alias<BlevBufferDstOutT>();
  auto blev_buffer_sizes_alloc = blev_buffer_sizes_slot->template create_alias<BufferSizeT>();
  auto blev_buffer_block_alloc = blev_buffer_block_slot->template create_alias<BlockOffsetT>();
  auto blev_buffer_scan_alloc  = blev_buffer_scan_slot->template create_alias<uint8_t>();
  auto blev_block_scan_alloc   = blev_buffer_block_scan_slot->template create_alias<uint8_t>();

  size_t buffer_offset_scan_storage = 0;
  size_t blev_block_scan_storage    = 0;
  if (const auto error = CubDebug(
        BLevBufferOffsetTileState::AllocationSize(static_cast<int32_t>(max_num_tiles), buffer_offset_scan_storage)))
  {
    return error;
  }
  if (const auto error = CubDebug(
        BLevBlockOffsetTileState::AllocationSize(static_cast<int32_t>(max_num_tiles), blev_block_scan_storage)))
  {
    return error;
  }

  blev_buffer_srcs_alloc.grow(max_num_buffers);
  blev_buffer_dsts_alloc.grow(max_num_buffers);
  blev_buffer_sizes_alloc.grow(max_num_buffers);
  blev_buffer_block_alloc.grow(max_num_buffers);
  blev_buffer_scan_alloc.grow(buffer_offset_scan_storage);
  blev_block_scan_alloc.grow(blev_block_scan_storage);

  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = temporary_storage_layout.get_size();
    return cudaSuccess;
  }
  if (num_buffers == 0)
  {
    return cudaSuccess;
  }

  if (const auto error = CubDebug(temporary_storage_layout.map_to_buffer(d_temp_storage, temp_storage_bytes)))
  {
    return error;
  }

  BlevBufferSrcsOutItT d_blev_src_buffers          = blev_buffer_srcs_alloc.get();
  BlevBufferDstsOutItT d_blev_dst_buffers          = blev_buffer_dsts_alloc.get();
  BlevBufferSizesOutItT d_blev_buffer_sizes        = blev_buffer_sizes_alloc.get();
  BlevBufferTileOffsetsOutItT d_blev_block_offsets = blev_buffer_block_alloc.get();

  auto init_scan_states_kernel =
    detail::batch_memcpy::InitTileStateKernel<BLevBufferOffsetTileState, BLevBlockOffsetTileState, BlockOffsetT>;
  auto batch_memcpy_non_blev_kernel = detail::batch_memcpy::BatchMemcpyKernel<
    PolicySelectorT,
    InputBufferIt,
    OutputBufferIt,
    BufferSizeIteratorT,
    per_invocation_buffer_offset_t,
    BlevBufferSrcsOutItT,
    BlevBufferDstsOutItT,
    BlevBufferSizesOutItT,
    BlevBufferTileOffsetsOutItT,
    BlockOffsetT,
    BLevBufferOffsetTileState,
    BLevBlockOffsetTileState,
    MemcpyOpt>;
  auto multi_block_memcpy_kernel = detail::batch_memcpy::MultiBlockBatchMemcpyKernel<
    PolicySelectorT,
    per_invocation_buffer_offset_t,
    BlevBufferSrcsOutItT,
    BlevBufferDstsOutItT,
    BlevBufferSizesOutItT,
    BlevBufferTileOffsetsOutItT,
    BLevBufferOffsetTileState,
    BlockOffsetT,
    MemcpyOpt>;

  const auto blev_block_threads = static_cast<uint32_t>(active_policy.large_buffer.block_threads);

  int device_ordinal;
  if (const auto error = CubDebug(cudaGetDevice(&device_ordinal)))
  {
    return error;
  }
  int sm_count;
  if (const auto error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal)))
  {
    return error;
  }

  int batch_memcpy_blev_occupancy;
  if (const auto error =
        CubDebug(MaxSmOccupancy(batch_memcpy_blev_occupancy, multi_block_memcpy_kernel, blev_block_threads)))
  {
    return error;
  }
  const int batch_memcpy_blev_grid_size =
    static_cast<int>(sm_count * batch_memcpy_blev_occupancy * subscription_factor);

  const ::cuda::std::int64_t num_invocations = ::cuda::ceil_div(num_buffers, max_num_buffers_per_invocation);

  for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
  {
    const auto current_buffer_offset = invocation_index * max_num_buffers_per_invocation;
    const auto num_current_buffers =
      ::cuda::std::min(max_num_buffers_per_invocation, num_buffers - current_buffer_offset);
    const auto num_current_tiles = static_cast<BlockOffsetT>(::cuda::ceil_div(num_current_buffers, tile_size));
    const auto init_grid_size    = static_cast<BlockOffsetT>(::cuda::ceil_div(num_current_tiles, init_kernel_threads));
    const auto batch_memcpy_grid_size = num_current_tiles;

    BLevBufferOffsetTileState buffer_scan_tile_state;
    if (const auto error = CubDebug(buffer_scan_tile_state.Init(
          static_cast<int32_t>(num_current_tiles), blev_buffer_scan_alloc.get(), buffer_offset_scan_storage)))
    {
      return error;
    }

    BLevBlockOffsetTileState block_scan_tile_state;
    if (const auto error = CubDebug(block_scan_tile_state.Init(
          static_cast<int32_t>(num_current_tiles), blev_block_scan_alloc.get(), blev_block_scan_storage)))
    {
      return error;
    }

    if (const auto error = CubDebug(
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(init_grid_size, init_kernel_threads, 0, stream)
            .doit(init_scan_states_kernel, buffer_scan_tile_state, block_scan_tile_state, num_current_tiles)))
    {
      return error;
    }
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    if (const auto error = CubDebug(
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
            batch_memcpy_grid_size, active_policy.small_buffer.block_threads, 0, stream)
            .doit(batch_memcpy_non_blev_kernel,
                  input_buffer_it + current_buffer_offset,
                  output_buffer_it + current_buffer_offset,
                  buffer_sizes + current_buffer_offset,
                  static_cast<per_invocation_buffer_offset_t>(num_current_buffers),
                  d_blev_src_buffers,
                  d_blev_dst_buffers,
                  d_blev_buffer_sizes,
                  d_blev_block_offsets,
                  buffer_scan_tile_state,
                  block_scan_tile_state)))
    {
      return error;
    }
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    if (const auto error = CubDebug(
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(
            batch_memcpy_blev_grid_size, blev_block_threads, 0, stream)
            .doit(multi_block_memcpy_kernel,
                  d_blev_src_buffers,
                  d_blev_dst_buffers,
                  d_blev_buffer_sizes,
                  d_blev_block_offsets,
                  buffer_scan_tile_state,
                  batch_memcpy_grid_size - 1)))
    {
      return error;
    }
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }
  }

  return cudaSuccess;
}
} // namespace detail::batch_memcpy

CUB_NAMESPACE_END
