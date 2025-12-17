// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/******************************************************************************
 * Simple demonstration of cub::BlockScan
 *
 * To compile using the command line:
 *   nvcc -arch=sm_XX example_block_scan.cu -I../.. -lcudart -O3
 *
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console (define before including cub.h)
#define CUB_STDERR

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include <iostream>

#include <stdio.h>

#include "../../test/test_util.h"

using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and aliases
//---------------------------------------------------------------------

/// Verbose output
bool g_verbose = false;

/// Timing iterations
int g_timing_iterations = 100;

/// Default grid size
int g_grid_size = 1;

//---------------------------------------------------------------------
// Kernels
//---------------------------------------------------------------------

/**
 * Simple kernel for performing a block-wide exclusive prefix sum over integers
 */
template <int BLOCK_THREADS,
          int ITEMS_PER_THREAD,
          BlockScanAlgorithm ALGORITHM>
__global__ void BlockPrefixSumKernel(int* d_in, // Tile of input
                                     int* d_out, // Tile of output
                                     clock_t* d_elapsed) // Elapsed cycle count of block scan
{
  // Specialize BlockLoad type for our thread block (uses warp-striped loads for coalescing, then transposes in shared
  // memory to a blocked arrangement)
  using BlockLoadT = BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE>;

  // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared
  // memory to a blocked arrangement)
  using BlockStoreT = BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_STORE_WARP_TRANSPOSE>;

  // Specialize BlockScan type for our thread block
  using BlockScanT = BlockScan<int, BLOCK_THREADS, ALGORITHM>;

  // Shared memory
  __shared__ union TempStorage
  {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
  } temp_storage;

  // Per-thread tile data
  int data[ITEMS_PER_THREAD];

  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in, data);

  // Barrier for smem reuse
  __syncthreads();

  // Start cycle timer
  clock_t start = clock();

  // Compute exclusive prefix sum
  int aggregate;
  BlockScanT(temp_storage.scan).ExclusiveSum(data, data, aggregate);

  // Stop cycle timer
  clock_t stop = clock();

  // Barrier for smem reuse
  __syncthreads();

  // Store items from a blocked arrangement
  BlockStoreT(temp_storage.store).Store(d_out, data);

  // Store aggregate and elapsed clocks
  if (threadIdx.x == 0)
  {
    *d_elapsed                              = (start > stop) ? start - stop : stop - start;
    d_out[BLOCK_THREADS * ITEMS_PER_THREAD] = aggregate;
  }
}

//---------------------------------------------------------------------
// Host utilities
//---------------------------------------------------------------------

/**
 * Initialize exclusive prefix sum problem (and solution).
 * Returns the aggregate
 */
int Initialize(int* h_in, int* h_reference, int num_items)
{
  int inclusive = 0;

  for (int i = 0; i < num_items; ++i)
  {
    h_in[i] = i % 17;

    h_reference[i] = inclusive;
    inclusive += h_in[i];
  }

  return inclusive;
}

/**
 * Test thread block scan
 */
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
void Test()
{
  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  // Allocate host arrays
  int* h_in        = new int[TILE_SIZE];
  int* h_reference = new int[TILE_SIZE];
  int* h_gpu       = new int[TILE_SIZE + 1];

  // Initialize problem and reference output on host
  int h_aggregate = Initialize(h_in, h_reference, TILE_SIZE);

  // Initialize device arrays
  int* d_in          = nullptr;
  int* d_out         = nullptr;
  clock_t* d_elapsed = nullptr;
  cudaMalloc((void**) &d_in, sizeof(int) * TILE_SIZE);
  cudaMalloc((void**) &d_out, sizeof(int) * (TILE_SIZE + 1));
  cudaMalloc((void**) &d_elapsed, sizeof(clock_t));

  // Display input problem data
  if (g_verbose)
  {
    printf("Input data: ");
    for (int i = 0; i < TILE_SIZE; i++)
    {
      printf("%d, ", h_in[i]);
    }
    printf("\n\n");
  }

  // Kernel props
  int max_sm_occupancy;
  CubDebugExit(
    MaxSmOccupancy(max_sm_occupancy, BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>, BLOCK_THREADS));

  // Copy problem to device
  cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);

  printf(
    "BlockScan algorithm %s on %d items (%d timing iterations, %d blocks, %d threads, %d items per thread, %d SM "
    "occupancy):\n",
    (ALGORITHM == BLOCK_SCAN_RAKING) ? "BLOCK_SCAN_RAKING"
    : (ALGORITHM == BLOCK_SCAN_RAKING_MEMOIZE)
      ? "BLOCK_SCAN_RAKING_MEMOIZE"
      : "BLOCK_SCAN_WARP_SCANS",
    TILE_SIZE,
    g_timing_iterations,
    g_grid_size,
    BLOCK_THREADS,
    ITEMS_PER_THREAD,
    max_sm_occupancy);

  // Run aggregate/prefix kernel
  BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>
    <<<g_grid_size, BLOCK_THREADS>>>(d_in, d_out, d_elapsed);

  // Check results
  printf("\tOutput items: ");
  int compare = CompareDeviceResults(h_reference, d_out, TILE_SIZE, g_verbose, g_verbose);
  printf("%s\n", compare ? "FAIL" : "PASS");
  AssertEquals(0, compare);

  // Check total aggregate
  printf("\tAggregate: ");
  compare = CompareDeviceResults(&h_aggregate, d_out + TILE_SIZE, 1, g_verbose, g_verbose);
  printf("%s\n", compare ? "FAIL" : "PASS");
  AssertEquals(0, compare);

  // Run this several times and average the performance results
  GpuTimer timer;
  float elapsed_millis   = 0.0;
  clock_t elapsed_clocks = 0;

  for (int i = 0; i < g_timing_iterations; ++i)
  {
    // Copy problem to device
    cudaMemcpy(d_in, h_in, sizeof(int) * TILE_SIZE, cudaMemcpyHostToDevice);

    timer.Start();

    // Run aggregate/prefix kernel
    BlockPrefixSumKernel<BLOCK_THREADS, ITEMS_PER_THREAD, ALGORITHM>
      <<<g_grid_size, BLOCK_THREADS>>>(d_in, d_out, d_elapsed);

    timer.Stop();
    elapsed_millis += timer.ElapsedMillis();

    // Copy clocks from device
    clock_t clocks;
    CubDebugExit(cudaMemcpy(&clocks, d_elapsed, sizeof(clock_t), cudaMemcpyDeviceToHost));
    elapsed_clocks += clocks;
  }

  // Check for kernel errors and STDIO from the kernel, if any
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  // Display timing results
  float avg_millis          = elapsed_millis / g_timing_iterations;
  float avg_items_per_sec   = float(TILE_SIZE * g_grid_size) / avg_millis / 1000.0f;
  float avg_clocks          = float(elapsed_clocks) / g_timing_iterations;
  float avg_clocks_per_item = avg_clocks / TILE_SIZE;

  printf("\tAverage BlockScan::Sum clocks: %.3f\n", avg_clocks);
  printf("\tAverage BlockScan::Sum clocks per item: %.3f\n", avg_clocks_per_item);
  printf("\tAverage kernel millis: %.4f\n", avg_millis);
  printf("\tAverage million items / sec: %.4f\n", avg_items_per_sec);

  // Cleanup
  if (h_in)
  {
    delete[] h_in;
  }
  if (h_reference)
  {
    delete[] h_reference;
  }
  if (h_gpu)
  {
    delete[] h_gpu;
  }
  if (d_in)
  {
    cudaFree(d_in);
  }
  if (d_out)
  {
    cudaFree(d_out);
  }
  if (d_elapsed)
  {
    cudaFree(d_elapsed);
  }
}

//---------------------------------------------------------------------
// Documentation examples
//---------------------------------------------------------------------

// example-begin exclusive-sum-array
__global__ void ExclusiveSumArrayKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  int thread_data[4];
  for (int i = 0; i < 4; i++)
  {
    thread_data[i] = d_data[threadIdx.x * 4 + i];
  }

  // Collectively compute the block-wide exclusive prefix sum
  BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);

  // Store results
  for (int i = 0; i < 4; i++)
  {
    d_data[threadIdx.x * 4 + i] = thread_data[i];
  }
}
// example-end exclusive-sum-array

// example-begin exclusive-sum-single
__global__ void ExclusiveSumSingleKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain input item for each thread
  int thread_data = d_data[threadIdx.x];

  // Collectively compute the block-wide exclusive prefix sum
  BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);

  // Store result
  d_data[threadIdx.x] = thread_data;
}
// example-end exclusive-sum-single

// example-begin exclusive-sum-aggregate
__global__ void ExclusiveSumAggregateKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain input item for each thread
  int thread_data = d_data[threadIdx.x];

  // Collectively compute the block-wide exclusive prefix sum
  int block_aggregate;
  BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);

  // Store result
  d_data[threadIdx.x] = thread_data;
}
// example-end exclusive-sum-aggregate

// example-begin block-prefix-callback-op
// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackOp
{
  // Running prefix
  int running_total;

  // Constructor
  __device__ BlockPrefixCallbackOp(int running_total)
      : running_total(running_total)
  {}

  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  __device__ int operator()(int block_aggregate)
  {
    int old_prefix = running_total;
    running_total += block_aggregate;
    return old_prefix;
  }
};
// example-end block-prefix-callback-op

// example-begin exclusive-sum-prefix-callback
__global__ void ExclusiveSumPrefixCallbackKernel(int* d_data, int num_items)
{
  // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
  using BlockLoadT  = BlockLoad<int, 128, 4, BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreT = BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE>;
  using BlockScanT  = BlockScan<int, 128>;

  // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  __shared__ union
  {
    typename BlockLoadT::TempStorage load;
    typename BlockScanT::TempStorage scan;
    typename BlockStoreT::TempStorage store;
  } temp_storage;

  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  // Have the block iterate over segments of items
  for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
  {
    // Load a segment of consecutive items that are blocked across threads
    int thread_data[4];
    BlockLoadT(temp_storage.load).Load(d_data + block_offset, thread_data);
    __syncthreads();

    // Collectively compute the block-wide exclusive prefix sum
    BlockScanT(temp_storage.scan).ExclusiveSum(thread_data, thread_data, prefix_op);
    __syncthreads();

    // Store scanned items to output segment
    BlockStoreT(temp_storage.store).Store(d_data + block_offset, thread_data);
    __syncthreads();
  }
}
// example-end exclusive-sum-prefix-callback

// example-begin exclusive-scan-single
__global__ void ExclusiveScanSingleKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain input item for each thread
  int thread_data = d_data[threadIdx.x];

  // Collectively compute the block-wide exclusive prefix max scan
  BlockScan(temp_storage).ExclusiveScan(thread_data, thread_data, INT_MIN, cuda::maximum<>{});

  // Store result
  d_data[threadIdx.x] = thread_data;
}
// example-end exclusive-scan-single

// example-begin inclusive-scan-prefix-callback
__global__ void InclusiveSumPrefixCallbackKernel(int* d_data, int num_items)
{
  // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
  using BlockLoadT  = BlockLoad<int, 128, 4, BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreT = BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE>;
  using BlockScanT  = BlockScan<int, 128>;

  // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  __shared__ union
  {
    typename BlockLoadT::TempStorage load;
    typename BlockScanT::TempStorage scan;
    typename BlockStoreT::TempStorage store;
  } temp_storage;

  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  // Have the block iterate over segments of items
  for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
  {
    // Load a segment of consecutive items that are blocked across threads
    int thread_data[4];
    BlockLoadT(temp_storage.load).Load(d_data + block_offset, thread_data);
    __syncthreads();

    // Collectively compute the block-wide inclusive prefix sum
    BlockScanT(temp_storage.scan).InclusiveSum(thread_data, thread_data, prefix_op);
    __syncthreads();

    // Store scanned items to output segment
    BlockStoreT(temp_storage.store).Store(d_data + block_offset, thread_data);
    __syncthreads();
  }
}
// example-end inclusive-scan-prefix-callback

// example-begin inclusive-sum-array
__global__ void InclusiveSumArrayKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  int thread_data[4];
  for (int i = 0; i < 4; i++)
  {
    thread_data[i] = d_data[threadIdx.x * 4 + i];
  }

  // Collectively compute the block-wide inclusive prefix sum
  BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);

  // Store results
  for (int i = 0; i < 4; i++)
  {
    d_data[threadIdx.x * 4 + i] = thread_data[i];
  }
}
// example-end inclusive-sum-array

// example-begin inclusive-sum-single
__global__ void InclusiveSumSingleKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain input item for each thread
  int thread_data = d_data[threadIdx.x];

  // Collectively compute the block-wide inclusive prefix sum
  BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);

  // Store result
  d_data[threadIdx.x] = thread_data;
}
// example-end inclusive-sum-single

// example-begin exclusive-scan-array
__global__ void ExclusiveScanArrayKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  int thread_data[4];
  for (int i = 0; i < 4; i++)
  {
    thread_data[i] = d_data[threadIdx.x * 4 + i];
  }

  // Collectively compute the block-wide exclusive prefix max scan
  BlockScan(temp_storage).ExclusiveScan(thread_data, thread_data, INT_MIN, cuda::maximum<>{});

  // Store results
  for (int i = 0; i < 4; i++)
  {
    d_data[threadIdx.x * 4 + i] = thread_data[i];
  }
}
// example-end exclusive-scan-array

// example-begin exclusive-scan-aggregate
__global__ void ExclusiveScanAggregateKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain input item for each thread
  int thread_data = d_data[threadIdx.x];

  // Collectively compute the block-wide exclusive prefix max scan
  int block_aggregate;
  BlockScan(temp_storage).ExclusiveScan(thread_data, thread_data, INT_MIN, cuda::maximum<>{}, block_aggregate);

  // Store result
  d_data[threadIdx.x] = thread_data;
}
// example-end exclusive-scan-aggregate

// example-begin block-prefix-callback-max-op
// A stateful callback functor that maintains a running prefix to be applied
// during consecutive scan operations.
struct BlockPrefixCallbackMaxOp
{
  // Running prefix
  int running_total;

  // Constructor
  __device__ BlockPrefixCallbackMaxOp(int running_total)
      : running_total(running_total)
  {}

  // Callback operator to be entered by the first warp of threads in the block.
  // Thread-0 is responsible for returning a value for seeding the block-wide scan.
  __device__ int operator()(int block_aggregate)
  {
    int old_prefix = running_total;
    running_total  = (block_aggregate > old_prefix) ? block_aggregate : old_prefix;
    return old_prefix;
  }
};
// example-end block-prefix-callback-max-op

// example-begin exclusive-scan-prefix-callback
__global__ void ExclusiveScanPrefixCallbackKernel(int* d_data, int num_items)
{
  // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
  using BlockLoadT  = BlockLoad<int, 128, 4, BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreT = BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE>;
  using BlockScanT  = BlockScan<int, 128>;

  // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  __shared__ union
  {
    typename BlockLoadT::TempStorage load;
    typename BlockScanT::TempStorage scan;
    typename BlockStoreT::TempStorage store;
  } temp_storage;

  // Initialize running total
  BlockPrefixCallbackMaxOp prefix_op(INT_MIN);

  // Have the block iterate over segments of items
  for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
  {
    // Load a segment of consecutive items that are blocked across threads
    int thread_data[4];
    BlockLoadT(temp_storage.load).Load(d_data + block_offset, thread_data);
    __syncthreads();

    // Collectively compute the block-wide exclusive prefix max scan
    BlockScanT(temp_storage.scan).ExclusiveScan(thread_data, thread_data, cuda::maximum<>{}, prefix_op);
    __syncthreads();

    // Store scanned items to output segment
    BlockStoreT(temp_storage.store).Store(d_data + block_offset, thread_data);
    __syncthreads();
  }
}
// example-end exclusive-scan-prefix-callback

// example-begin exclusive-sum-single-prefix-callback
__global__ void ExclusiveSumSinglePrefixCallbackKernel(int* d_data, int num_items)
{
  // Specialize BlockScan for a 1D block of 128 threads
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Initialize running total
  BlockPrefixCallbackOp prefix_op(0);

  // Have the block iterate over segments of items
  for (int block_offset = 0; block_offset < num_items; block_offset += 128)
  {
    // Load a segment of consecutive items that are blocked across threads
    int thread_data = d_data[block_offset + threadIdx.x];

    // Collectively compute the block-wide exclusive prefix sum
    BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, prefix_op);
    __syncthreads();

    // Store scanned items to output segment
    d_data[block_offset + threadIdx.x] = thread_data;
  }
}
// example-end exclusive-sum-single-prefix-callback

// example-begin exclusive-sum-array-aggregate
__global__ void ExclusiveSumArrayAggregateKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  int thread_data[4];
  for (int i = 0; i < 4; i++)
  {
    thread_data[i] = d_data[threadIdx.x * 4 + i];
  }

  // Collectively compute the block-wide exclusive prefix sum
  int block_aggregate;
  BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data, block_aggregate);

  // Store results
  for (int i = 0; i < 4; i++)
  {
    d_data[threadIdx.x * 4 + i] = thread_data[i];
  }
}
// example-end exclusive-sum-array-aggregate

// example-begin inclusive-scan-array
__global__ void InclusiveScanArrayKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  int thread_data[4];
  for (int i = 0; i < 4; i++)
  {
    thread_data[i] = d_data[threadIdx.x * 4 + i];
  }

  // Collectively compute the block-wide inclusive prefix max scan
  BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, ::cuda::maximum<>{});

  // Store results
  for (int i = 0; i < 4; i++)
  {
    d_data[threadIdx.x * 4 + i] = thread_data[i];
  }
}
// example-end inclusive-scan-array

// example-begin inclusive-scan-single
__global__ void InclusiveScanSingleKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain input item for each thread
  int thread_data = d_data[threadIdx.x];

  // Collectively compute the block-wide inclusive prefix max scan
  BlockScan(temp_storage).InclusiveScan(thread_data, thread_data, cuda::maximum<>{});

  // Store result
  d_data[threadIdx.x] = thread_data;
}
// example-end inclusive-scan-single

// example-begin inclusive-scan-prefix-callback-max
__global__ void InclusiveScanPrefixCallbackKernel(int* d_data, int num_items)
{
  // Specialize BlockLoad, BlockStore, and BlockScan for a 1D block of 128 threads, 4 ints per thread
  using BlockLoadT  = BlockLoad<int, 128, 4, BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreT = BlockStore<int, 128, 4, BLOCK_STORE_TRANSPOSE>;
  using BlockScanT  = BlockScan<int, 128>;

  // Allocate aliased shared memory for BlockLoad, BlockStore, and BlockScan
  __shared__ union
  {
    typename BlockLoadT::TempStorage load;
    typename BlockScanT::TempStorage scan;
    typename BlockStoreT::TempStorage store;
  } temp_storage;

  // Initialize running total
  BlockPrefixCallbackMaxOp prefix_op(INT_MIN);

  // Have the block iterate over segments of items
  for (int block_offset = 0; block_offset < num_items; block_offset += 128 * 4)
  {
    // Load a segment of consecutive items that are blocked across threads
    int thread_data[4];
    BlockLoadT(temp_storage.load).Load(d_data + block_offset, thread_data);
    __syncthreads();

    // Collectively compute the block-wide inclusive prefix max
    BlockScanT(temp_storage.scan).InclusiveScan(thread_data, thread_data, cuda::maximum<>{}, prefix_op);
    __syncthreads();

    // Store scanned items to output segment
    BlockStoreT(temp_storage.store).Store(d_data + block_offset, thread_data);
    __syncthreads();
  }
}
// example-end inclusive-scan-prefix-callback-max

// example-begin inclusive-sum-array-aggregate
__global__ void InclusiveSumArrayAggregateKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain a segment of consecutive items that are blocked across threads
  int thread_data[4];
  for (int i = 0; i < 4; i++)
  {
    thread_data[i] = d_data[threadIdx.x * 4 + i];
  }

  // Collectively compute the block-wide inclusive prefix sum
  int block_aggregate;
  BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);

  // Store results
  for (int i = 0; i < 4; i++)
  {
    d_data[threadIdx.x * 4 + i] = thread_data[i];
  }
}
// example-end inclusive-sum-array-aggregate

// example-begin inclusive-sum-single-aggregate
__global__ void InclusiveSumSingleAggregateKernel(int* d_data)
{
  // Specialize BlockScan for a 1D block of 128 threads of type int
  using BlockScan = cub::BlockScan<int, 128>;

  // Allocate shared memory for BlockScan
  __shared__ typename BlockScan::TempStorage temp_storage;

  // Obtain input item for each thread
  int thread_data = d_data[threadIdx.x];

  // Collectively compute the block-wide inclusive prefix sum
  int block_aggregate;
  BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, block_aggregate);

  // Store result
  d_data[threadIdx.x] = thread_data;
}
// example-end inclusive-sum-single-aggregate

/**
 * Test documentation example kernels
 */
void TestDocumentationExamples()
{
  printf("Testing documentation example kernels...\n");

  const int num_items = 128 * 4; // 512 items for array examples, 128 for single-item examples
  int* d_data;
  int* h_data = new int[num_items];
  int running_max;
  bool all_passed = true;

  cudaMalloc(&d_data, num_items * sizeof(int));

  // Test ExclusiveSumArrayKernel
  if (all_passed)
  {
    printf("  Testing ExclusiveSumArrayKernel... ");
    for (int i = 0; i < num_items; i++)
    {
      h_data[i] = 1;
    }
    cudaMemcpy(d_data, h_data, num_items * sizeof(int), cudaMemcpyHostToDevice);
    ExclusiveSumArrayKernel<<<1, 128>>>(d_data);
    cudaMemcpy(h_data, d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < 128 && passed; i++)
    {
      for (int j = 0; j < 4 && passed; j++)
      {
        int expected = i * 4 + j;
        if (h_data[i * 4 + j] != expected)
        {
          printf("FAILED at [%d]: expected %d, got %d\n", i * 4 + j, expected, h_data[i * 4 + j]);
          passed     = false;
          all_passed = false;
        }
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  // Test ExclusiveSumSingleKernel
  if (all_passed)
  {
    printf("  Testing ExclusiveSumSingleKernel... ");
    for (int i = 0; i < 128; i++)
    {
      h_data[i] = 1;
    }
    cudaMemcpy(d_data, h_data, 128 * sizeof(int), cudaMemcpyHostToDevice);
    ExclusiveSumSingleKernel<<<1, 128>>>(d_data);
    cudaMemcpy(h_data, d_data, 128 * sizeof(int), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < 128 && passed; i++)
    {
      if (h_data[i] != i)
      {
        printf("FAILED at [%d]: expected %d, got %d\n", i, i, h_data[i]);
        passed     = false;
        all_passed = false;
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  // Test ExclusiveSumAggregateKernel
  if (all_passed)
  {
    printf("  Testing ExclusiveSumAggregateKernel... ");
    for (int i = 0; i < 128; i++)
    {
      h_data[i] = 1;
    }
    cudaMemcpy(d_data, h_data, 128 * sizeof(int), cudaMemcpyHostToDevice);
    ExclusiveSumAggregateKernel<<<1, 128>>>(d_data);
    cudaMemcpy(h_data, d_data, 128 * sizeof(int), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < 128 && passed; i++)
    {
      if (h_data[i] != i)
      {
        printf("FAILED at [%d]: expected %d, got %d\n", i, i, h_data[i]);
        passed     = false;
        all_passed = false;
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("  Testing ExclusiveSumPrefixCallbackKernel... ");
    for (int i = 0; i < num_items; i++)
    {
      h_data[i] = 1;
    }
    cudaMemcpy(d_data, h_data, num_items * sizeof(int), cudaMemcpyHostToDevice);
    ExclusiveSumPrefixCallbackKernel<<<1, 128>>>(d_data, num_items);
    cudaMemcpy(h_data, d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < num_items && passed; i++)
    {
      if (h_data[i] != i)
      {
        printf("FAILED at [%d]: expected %d, got %d\n", i, i, h_data[i]);
        passed     = false;
        all_passed = false;
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("  Testing InclusiveSumPrefixCallbackKernel... ");
    for (int i = 0; i < num_items; i++)
    {
      h_data[i] = 1;
    }
    cudaMemcpy(d_data, h_data, num_items * sizeof(int), cudaMemcpyHostToDevice);
    InclusiveSumPrefixCallbackKernel<<<1, 128>>>(d_data, num_items);
    cudaMemcpy(h_data, d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < num_items && passed; i++)
    {
      if (h_data[i] != i + 1)
      {
        printf("FAILED at [%d]: expected %d, got %d\n", i, i + 1, h_data[i]);
        passed     = false;
        all_passed = false;
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("  Testing InclusiveSumArrayKernel... ");
    for (int i = 0; i < num_items; i++)
    {
      h_data[i] = 1;
    }
    cudaMemcpy(d_data, h_data, num_items * sizeof(int), cudaMemcpyHostToDevice);
    InclusiveSumArrayKernel<<<1, 128>>>(d_data);
    cudaMemcpy(h_data, d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < 128 && passed; i++)
    {
      for (int j = 0; j < 4 && passed; j++)
      {
        int expected = i * 4 + j + 1;
        if (h_data[i * 4 + j] != expected)
        {
          printf("FAILED at [%d]: expected %d, got %d\n", i * 4 + j, expected, h_data[i * 4 + j]);
          passed     = false;
          all_passed = false;
        }
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("  Testing InclusiveSumSingleKernel... ");
    for (int i = 0; i < 128; i++)
    {
      h_data[i] = 1;
    }
    cudaMemcpy(d_data, h_data, 128 * sizeof(int), cudaMemcpyHostToDevice);
    InclusiveSumSingleKernel<<<1, 128>>>(d_data);
    cudaMemcpy(h_data, d_data, 128 * sizeof(int), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < 128 && passed; i++)
    {
      if (h_data[i] != i + 1)
      {
        printf("FAILED at [%d]: expected %d, got %d\n", i, i + 1, h_data[i]);
        passed     = false;
        all_passed = false;
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("  Testing ExclusiveScanArrayKernel... ");
    for (int i = 0; i < num_items; i++)
    {
      h_data[i] = i;
    }
    cudaMemcpy(d_data, h_data, num_items * sizeof(int), cudaMemcpyHostToDevice);
    ExclusiveScanArrayKernel<<<1, 128>>>(d_data);
    cudaMemcpy(h_data, d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    running_max = INT_MIN;
    bool passed = true;
    for (int i = 0; i < 128 && passed; i++)
    {
      for (int j = 0; j < 4 && passed; j++)
      {
        int idx      = i * 4 + j;
        int expected = running_max;
        if (h_data[idx] != expected)
        {
          printf("FAILED at [%d]: expected %d, got %d\n", idx, expected, h_data[idx]);
          passed     = false;
          all_passed = false;
        }
        running_max = (idx > running_max) ? idx : running_max;
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("  Testing ExclusiveScanAggregateKernel... ");
    for (int i = 0; i < 128; i++)
    {
      h_data[i] = i;
    }
    cudaMemcpy(d_data, h_data, 128 * sizeof(int), cudaMemcpyHostToDevice);
    ExclusiveScanAggregateKernel<<<1, 128>>>(d_data);
    cudaMemcpy(h_data, d_data, 128 * sizeof(int), cudaMemcpyDeviceToHost);
    running_max = INT_MIN;
    bool passed = true;
    for (int i = 0; i < 128 && passed; i++)
    {
      int expected = running_max;
      if (h_data[i] != expected)
      {
        printf("FAILED at [%d]: expected %d, got %d\n", i, expected, h_data[i]);
        passed     = false;
        all_passed = false;
      }
      running_max = (i > running_max) ? i : running_max;
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("  Testing ExclusiveScanPrefixCallbackKernel... ");
    for (int i = 0; i < num_items; i++)
    {
      h_data[i] = (i % 2 == 0) ? i : -i;
    }
    cudaMemcpy(d_data, h_data, num_items * sizeof(int), cudaMemcpyHostToDevice);
    ExclusiveScanPrefixCallbackKernel<<<1, 128>>>(d_data, num_items);
    cudaMemcpy(h_data, d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    running_max = INT_MIN;
    bool passed = true;
    for (int i = 0; i < num_items && passed; i++)
    {
      int input_val = (i % 2 == 0) ? i : -i;
      int expected  = running_max;
      if (h_data[i] != expected)
      {
        printf("FAILED at [%d]: expected %d, got %d\n", i, expected, h_data[i]);
        passed     = false;
        all_passed = false;
      }
      running_max = (input_val > running_max) ? input_val : running_max;
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("  Testing InclusiveScanArrayKernel... ");
    for (int i = 0; i < num_items; i++)
    {
      h_data[i] = i;
    }
    cudaMemcpy(d_data, h_data, num_items * sizeof(int), cudaMemcpyHostToDevice);
    InclusiveScanArrayKernel<<<1, 128>>>(d_data);
    cudaMemcpy(h_data, d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    running_max = INT_MIN;
    bool passed = true;
    for (int i = 0; i < 128 && passed; i++)
    {
      for (int j = 0; j < 4 && passed; j++)
      {
        int idx     = i * 4 + j;
        running_max = (idx > running_max) ? idx : running_max;
        if (h_data[idx] != running_max)
        {
          printf("FAILED at [%d]: expected %d, got %d\n", idx, running_max, h_data[idx]);
          passed     = false;
          all_passed = false;
        }
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("  Testing InclusiveScanSingleKernel... ");
    for (int i = 0; i < 128; i++)
    {
      h_data[i] = i;
    }
    cudaMemcpy(d_data, h_data, 128 * sizeof(int), cudaMemcpyHostToDevice);
    InclusiveScanSingleKernel<<<1, 128>>>(d_data);
    cudaMemcpy(h_data, d_data, 128 * sizeof(int), cudaMemcpyDeviceToHost);
    running_max = INT_MIN;
    bool passed = true;
    for (int i = 0; i < 128 && passed; i++)
    {
      running_max = (i > running_max) ? i : running_max;
      if (h_data[i] != running_max)
      {
        printf("FAILED at [%d]: expected %d, got %d\n", i, running_max, h_data[i]);
        passed     = false;
        all_passed = false;
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("  Testing InclusiveScanPrefixCallbackKernel... ");
    for (int i = 0; i < num_items; i++)
    {
      h_data[i] = (i % 2 == 0) ? i : -i;
    }
    cudaMemcpy(d_data, h_data, num_items * sizeof(int), cudaMemcpyHostToDevice);
    InclusiveScanPrefixCallbackKernel<<<1, 128>>>(d_data, num_items);
    cudaMemcpy(h_data, d_data, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    running_max = INT_MIN;
    bool passed = true;
    for (int i = 0; i < num_items && passed; i++)
    {
      int input_val = (i % 2 == 0) ? i : -i;
      running_max   = (input_val > running_max) ? input_val : running_max;
      if (h_data[i] != running_max)
      {
        printf("FAILED at [%d]: expected %d, got %d\n", i, running_max, h_data[i]);
        passed     = false;
        all_passed = false;
      }
    }
    if (passed)
    {
      printf("PASS\n");
    }
  }

  if (all_passed)
  {
    printf("All documentation example tests PASSED!\n\n");
  }

  delete[] h_data;
  cudaFree(d_data);
}

/**
 * Main
 */
int main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");
  args.GetCmdLineArgument("i", g_timing_iterations);
  args.GetCmdLineArgument("grid-size", g_grid_size);

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--device=<device-id>] "
           "[--i=<timing iterations (default:%d)>]"
           "[--grid-size=<grid size (default:%d)>]"
           "[--v] "
           "\n",
           argv[0],
           g_timing_iterations,
           g_grid_size);
    exit(0);
  }

  // Initialize device
  CubDebugExit(args.DeviceInit());

  // Test documentation example kernels first
  TestDocumentationExamples();

  // Run tests
  Test<1024, 1, BLOCK_SCAN_RAKING>();
  Test<512, 2, BLOCK_SCAN_RAKING>();
  Test<256, 4, BLOCK_SCAN_RAKING>();
  Test<128, 8, BLOCK_SCAN_RAKING>();
  Test<64, 16, BLOCK_SCAN_RAKING>();
  Test<32, 32, BLOCK_SCAN_RAKING>();

  printf("-------------\n");

  Test<1024, 1, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<512, 2, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<256, 4, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<128, 8, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<64, 16, BLOCK_SCAN_RAKING_MEMOIZE>();
  Test<32, 32, BLOCK_SCAN_RAKING_MEMOIZE>();

  printf("-------------\n");

  Test<1024, 1, BLOCK_SCAN_WARP_SCANS>();
  Test<512, 2, BLOCK_SCAN_WARP_SCANS>();
  Test<256, 4, BLOCK_SCAN_WARP_SCANS>();
  Test<128, 8, BLOCK_SCAN_WARP_SCANS>();
  Test<64, 16, BLOCK_SCAN_WARP_SCANS>();
  Test<32, 32, BLOCK_SCAN_WARP_SCANS>();

  return 0;
}
