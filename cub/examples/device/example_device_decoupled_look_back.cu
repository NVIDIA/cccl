// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>

#include <iostream>

template <class ScanTileStateT>
__global__ void init_kernel(ScanTileStateT tile_state, int blocks_in_grid)
{
  tile_state.InitializeStatus(blocks_in_grid);
}

template <class MessageT>
__global__ void decoupled_look_back_kernel(cub::ScanTileState<MessageT> tile_state)
{
  using scan_op_t         = ::cuda::std::plus<>;
  using scan_tile_state_t = cub::ScanTileState<MessageT>;
  using tile_prefix_op    = cub::TilePrefixCallbackOp<MessageT, scan_op_t, scan_tile_state_t>;
  using temp_storage_t    = typename tile_prefix_op::TempStorage;

  // Allocate temp storage in shared memory
  __shared__ temp_storage_t temp_storage;

  scan_op_t scan_op{};
  constexpr unsigned int threads_in_warp = 32;
  const unsigned int tid                 = threadIdx.x;

  // Construct prefix op
  tile_prefix_op prefix(tile_state, temp_storage, scan_op);
  const unsigned int tile_idx = prefix.GetTileIdx();

  // Compute block aggregate
  MessageT block_aggregate = blockIdx.x;

  if (tile_idx == 0)
  {
    // There are no blocks to look back to, immediately set the inclusive state
    if (tid == 0)
    {
      tile_state.SetInclusive(tile_idx, block_aggregate);
      printf("tile %d: inclusive = %d\n", tile_idx, block_aggregate);
    }
  }
  else
  {
    // Only the first warp in the block can perform the look back
    const unsigned int warp_id = tid / threads_in_warp;

    if (warp_id == 0)
    {
      // Perform the decoupled look-back
      // Invocation of the prefix will block until the look-back is complete.
      MessageT exclusive_prefix = prefix(block_aggregate);

      if (tid == 0)
      {
        MessageT inclusive_prefix = scan_op(exclusive_prefix, block_aggregate);
        printf("tile %d: exclusive = %d inclusive = %d\n", tile_idx, exclusive_prefix, inclusive_prefix);
      }
    }
  }
}

template <class MessageT>
void decoupled_look_back_example(int blocks_in_grid)
{
  using scan_tile_state_t = cub::ScanTileState<MessageT>;

  // Query temporary storage requirements
  std::size_t temp_storage_bytes{};
  scan_tile_state_t::AllocationSize(blocks_in_grid, temp_storage_bytes);

  // Allocate temporary storage
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  std::uint8_t* d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Initialize temporary storage
  scan_tile_state_t tile_status;
  tile_status.Init(blocks_in_grid, d_temp_storage, temp_storage_bytes);
  constexpr unsigned int threads_in_init_block = 256;
  const unsigned int blocks_in_init_grid       = ::cuda::ceil_div(blocks_in_grid, threads_in_init_block);
  init_kernel<<<blocks_in_init_grid, threads_in_init_block>>>(tile_status, blocks_in_grid);

  // Launch decoupled look-back
  constexpr unsigned int threads_in_block = 256;
  decoupled_look_back_kernel<<<blocks_in_grid, threads_in_block>>>(tile_status);

  // Wait for kernel to finish
  cudaDeviceSynchronize();
}

int main()
{
  decoupled_look_back_example<int>(14);
}
