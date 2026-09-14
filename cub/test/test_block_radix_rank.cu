// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Ensure printing of CUDA runtime errors to console
#include "cub/util_type.cuh"
#define CUB_STDERR

#include <cub/block/block_load.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>
#include <cub/util_vsmem.cuh>

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/stream>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>

#include "cub_non_catch2_test_memory.h"
#include "test_util.h"

CUB_TEST_MEMORY_CLASS(CUB_SMALL);

bool g_verbose = false;

template <cub::RadixRankAlgorithm RankAlgorithm,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          int Descending,
          typename Key>
__launch_bounds__(ThreadsPerBlock, 1) __global__ void kernel(Key* d_keys, int* d_ranks)
{
  using block_radix_rank =
    cub::detail::block_radix_rank_t<RankAlgorithm, ThreadsPerBlock, RadixBits, Descending, ScanAlgorithm>;

  using storage_t = typename block_radix_rank::TempStorage;

  // Allocate temp storage in shared memory
  __shared__ storage_t temp_storage;

  // Items per thread
  Key keys[ItemsPerThread];
  int ranks[ItemsPerThread];

  constexpr bool uses_warp_striped_arrangement =
    RankAlgorithm == cub::RadixRankAlgorithm::RADIX_RANK_MATCH
    || RankAlgorithm == cub::RadixRankAlgorithm::RADIX_RANK_MATCH_EARLY_COUNTS_ANY
    || RankAlgorithm == cub::RadixRankAlgorithm::RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR;

  if (uses_warp_striped_arrangement)
  {
    cub::LoadDirectWarpStriped(threadIdx.x, d_keys, keys);
  }
  else
  {
    cub::LoadDirectBlocked(threadIdx.x, d_keys, keys);
  }

  cub::BFEDigitExtractor<Key> extractor(0, RadixBits); // NOLINT(misc-const-correctness)
  block_radix_rank(temp_storage).RankKeys(keys, ranks, extractor);

  if (uses_warp_striped_arrangement)
  {
    cub::StoreDirectWarpStriped(threadIdx.x, d_ranks, ranks);
  }
  else
  {
    cub::StoreDirectBlocked(threadIdx.x, d_ranks, ranks);
  }
}

//---------------------------------------------------------------------
// Host testing subroutines
//---------------------------------------------------------------------

/**
 * Simple key-value pairing
 */
template <typename Key>
struct pair_t
{
  Key key;
  int value;

  bool operator<(const pair_t& b) const
  {
    return (key < b.key);
  }
};

template <bool DESCENDING, typename Key>
void Initialize(GenMode gen_mode, Key* h_keys, int* h_reference_ranks, int num_items, int num_bits)
{
  const std::unique_ptr<pair_t<Key>[]> h_pairs_storage(new pair_t<Key>[num_items]);
  pair_t<Key>* h_pairs = h_pairs_storage.get();

  for (int i = 0; i < num_items; ++i)
  {
    InitValue(gen_mode, h_keys[i], i);

    // Mask off unwanted portions
    std::uint64_t base = 0;
    memcpy(&base, &h_keys[i], sizeof(Key));
    base &= (1ull << num_bits) - 1;
    memcpy(&h_keys[i], &base, sizeof(Key));

    h_pairs[i].key   = h_keys[i];
    h_pairs[i].value = i;
  }

  if (DESCENDING)
  {
    std::reverse(h_pairs, h_pairs + num_items);
  }

  std::stable_sort(h_pairs, h_pairs + num_items);

  if (DESCENDING)
  {
    std::reverse(h_pairs, h_pairs + num_items);
  }

  for (int i = 0; i < num_items; ++i)
  {
    h_reference_ranks[h_pairs[i].value] = i;
  }
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          int Descending,
          typename Key>
void TestDriver(GenMode gen_mode)
{
  constexpr int tile_size = ThreadsPerBlock * ItemsPerThread;

  // Allocate host arrays
  const std::unique_ptr<Key[]> h_keys(new Key[tile_size]);
  const std::unique_ptr<int[]> h_reference_ranks(new int[tile_size]);

  // Initialize problem and solution on host
  Initialize<Descending>(gen_mode, h_keys.get(), h_reference_ranks.get(), tile_size, RadixBits);

  // Allocate device arrays and copy the problem to the device
  const auto device = cuda::devices[0];
  const auto stream = cuda::stream{device};
  auto d_keys       = cuda::make_device_buffer<Key>(stream, device, h_keys.get(), h_keys.get() + tile_size);
  auto d_ranks      = cuda::make_device_buffer<int>(stream, device, tile_size, cuda::no_init);

  // Run kernel
  kernel<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, RadixBits, ScanAlgorithm, Descending, Key>
    <<<1, ThreadsPerBlock, 0, stream.get()>>>(d_keys.data(), d_ranks.data());

  // Flush kernel output / errors
  CubDebugExit(cudaPeekAtLastError());
  CubDebugExit(cudaDeviceSynchronize());

  // Check keys results
  const bool compare = CompareDeviceResults(h_reference_ranks.get(), d_ranks.data(), tile_size, g_verbose, g_verbose);
  AssertEquals(0, compare);
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          int Descending,
          typename Key>
void TestValid(cuda::std::true_type /*fits_smem_capacity*/)
{
  TestDriver<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, RadixBits, ScanAlgorithm, Descending, Key>(UNIFORM);

  TestDriver<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, RadixBits, ScanAlgorithm, Descending, Key>(INTEGER_SEED);
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          int Descending,
          typename Key>
void TestValid(cuda::std::false_type fits_smem_capacity)
{}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          bool Descending,
          typename Key>
void Test()
{
  // Check size of smem storage for the target arch to make sure it will fit
  using block_radix_rank =
    cub::detail::block_radix_rank_t<RankAlgorithm, ThreadsPerBlock, RadixBits, Descending, ScanAlgorithm>;
  using storage_t = typename block_radix_rank::TempStorage;

  const cuda::std::bool_constant<(sizeof(storage_t) <= cub::detail::max_smem_per_block)> fits_smem_capacity;

  TestValid<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, RadixBits, ScanAlgorithm, Descending, Key>(
    fits_smem_capacity);
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm,
          typename Key>
void Test()
{
  Test<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, RadixBits, ScanAlgorithm, true, Key>();
  Test<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, RadixBits, ScanAlgorithm, false, Key>();
}

template <cub::RadixRankAlgorithm RankAlgorithm,
          int ThreadsPerBlock,
          int ItemsPerThread,
          int RadixBits,
          cub::BlockScanAlgorithm ScanAlgorithm>
void Test()
{
  Test<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, RadixBits, ScanAlgorithm, std::uint8_t>();
  Test<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, RadixBits, ScanAlgorithm, std::uint16_t>();
}

template <cub::RadixRankAlgorithm RankAlgorithm, int ThreadsPerBlock, int ItemsPerThread, int RadixBits>
void Test()
{
  Test<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, RadixBits, cub::BLOCK_SCAN_RAKING>();
  Test<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, RadixBits, cub::BLOCK_SCAN_WARP_SCANS>();
}

template <cub::RadixRankAlgorithm RankAlgorithm, int ThreadsPerBlock, int ItemsPerThread>
void Test()
{
  Test<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, 1>();
  Test<RankAlgorithm, ThreadsPerBlock, ItemsPerThread, 5>();
}

template <cub::RadixRankAlgorithm RankAlgorithm, int ThreadsPerBlock>
void Test()
{
  Test<RankAlgorithm, ThreadsPerBlock, 1>();
  Test<RankAlgorithm, ThreadsPerBlock, 4>();
}

template <int ThreadsPerBlock>
void Test(cuda::std::true_type /* multiple of hw warp */)
{
  Test<cub::RadixRankAlgorithm::RADIX_RANK_MATCH, ThreadsPerBlock>();

  // TODO(senior-zero):
  // - RADIX_RANK_MATCH_EARLY_COUNTS_ANY
  // - RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR
}

template <int ThreadsPerBlock>
void Test(cuda::std::false_type /* multiple of hw warp */)
{}

template <int ThreadsPerBlock>
void Test()
{
  Test<cub::RadixRankAlgorithm::RADIX_RANK_BASIC, ThreadsPerBlock>();
  Test<cub::RadixRankAlgorithm::RADIX_RANK_MEMOIZE, ThreadsPerBlock>();

  Test<ThreadsPerBlock>(cuda::std::bool_constant<(ThreadsPerBlock % 32) == 0>{});
}

int main(int argc, char** argv)
{
  // Initialize command line
  CommandLineArgs args(argc, argv);
  g_verbose = args.CheckCmdLineFlag("v");

  // Print usage
  if (args.CheckCmdLineFlag("help"))
  {
    printf("%s "
           "[--v] "
           "\n",
           argv[0]);
    exit(0);
  }

  Test<16>();
  Test<32>();
  Test<128>();
  Test<130>();

  return 0;
}
