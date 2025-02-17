/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/device/device_memcpy.cuh>
#include <cub/util_ptx.cuh>

#include <thrust/fill.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "test_util.h"
#include <c2h/device_policy.h>
#include <c2h/vector.h>

/**
 * @brief Host-side random data generation
 */
template <typename T>
void GenerateRandomData(
  T* rand_out,
  const std::size_t num_items,
  const T min_rand_val                                         = std::numeric_limits<T>::min(),
  const T max_rand_val                                         = std::numeric_limits<T>::max(),
  const std::uint_fast32_t seed                                = 320981U,
  std::enable_if_t<std::is_integral_v<T> && (sizeof(T) >= 2)>* = nullptr)
{
  // initialize random number generator
  std::mt19937 rng(seed);
  std::uniform_int_distribution<T> uni_dist(min_rand_val, max_rand_val);

  // generate random numbers
  for (std::size_t i = 0; i < num_items; ++i)
  {
    rand_out[i] = uni_dist(rng);
  }
}

template <typename InputBufferIt, typename OutputBufferIt, typename BufferSizeIteratorT, typename BufferOffsetT>
void __global__ BaselineBatchMemCpyKernel(
  InputBufferIt input_buffer_it,
  OutputBufferIt output_buffer_it,
  BufferSizeIteratorT buffer_sizes,
  BufferOffsetT num_buffers)
{
  BufferOffsetT gtid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gtid >= num_buffers)
  {
    return;
  }
  for (BufferOffsetT i = 0; i < buffer_sizes[gtid]; i++)
  {
    reinterpret_cast<uint8_t*>(output_buffer_it[gtid])[i] = reinterpret_cast<uint8_t*>(input_buffer_it[gtid])[i];
  }
}

template <typename InputBufferIt, typename OutputBufferIt, typename BufferSizeIteratorT>
void InvokeBaselineBatchMemcpy(
  InputBufferIt input_buffer_it, OutputBufferIt output_buffer_it, BufferSizeIteratorT buffer_sizes, uint32_t num_buffers)
{
  constexpr uint32_t block_threads = 128U;
  uint32_t num_blocks              = (num_buffers + block_threads - 1) / block_threads;
  BaselineBatchMemCpyKernel<<<num_blocks, block_threads>>>(input_buffer_it, output_buffer_it, buffer_sizes, num_buffers);
}

template <typename InputBufferIt, typename OutputBufferIt, typename BufferSizeIteratorT, typename BufferOffsetT>
void __global__ BaselineBatchMemCpyPerBlockKernel(
  InputBufferIt input_buffer_it,
  OutputBufferIt output_buffer_it,
  BufferSizeIteratorT buffer_sizes,
  BufferOffsetT num_buffers)
{
  BufferOffsetT gbid = blockIdx.x;
  if (gbid >= num_buffers)
  {
    return;
  }
  for (BufferOffsetT i = threadIdx.x; i < buffer_sizes[gbid] / 8; i += blockDim.x)
  {
    reinterpret_cast<uint64_t*>(output_buffer_it[gbid])[i] = reinterpret_cast<uint64_t*>(input_buffer_it[gbid])[i];
  }
}

/**
 * @brief Used for generating a shuffled but cohesive sequence of output-buffer offsets for the
 * sequence of input-buffers.
 */
template <typename BufferOffsetT, typename ByteOffsetT, typename BufferSizeT>
c2h::host_vector<ByteOffsetT>
GetShuffledBufferOffsets(const c2h::host_vector<BufferSizeT>& buffer_sizes, const std::uint_fast32_t seed = 320981U)
{
  BufferOffsetT num_buffers = static_cast<BufferOffsetT>(buffer_sizes.size());

  // We're remapping the i-th buffer to pmt_idxs[i]
  std::mt19937 rng(seed);
  c2h::host_vector<BufferOffsetT> pmt_idxs(num_buffers);
  std::iota(pmt_idxs.begin(), pmt_idxs.end(), static_cast<BufferOffsetT>(0));
  std::shuffle(std::begin(pmt_idxs), std::end(pmt_idxs), rng);

  // Compute the offsets using the new mapping
  ByteOffsetT running_offset = {};
  c2h::host_vector<ByteOffsetT> permuted_offsets;
  permuted_offsets.reserve(num_buffers);
  for (auto permuted_buffer_idx : pmt_idxs)
  {
    permuted_offsets.push_back(running_offset);
    running_offset += buffer_sizes[permuted_buffer_idx];
  }

  // Generate the scatter indexes that identify where each buffer was mapped to
  c2h::host_vector<BufferOffsetT> scatter_idxs(num_buffers);
  for (BufferOffsetT i = 0; i < num_buffers; i++)
  {
    scatter_idxs[pmt_idxs[i]] = i;
  }

  c2h::host_vector<ByteOffsetT> new_offsets(num_buffers);
  for (BufferOffsetT i = 0; i < num_buffers; i++)
  {
    new_offsets[i] = permuted_offsets[scatter_idxs[i]];
  }

  return new_offsets;
}

/**
 * @brief Function object class template that takes an offset and returns an iterator at the given
 * offset relative to a fixed base iterator.
 *
 * @tparam IteratorT The random-access iterator type to be returned
 */
template <typename IteratorT>
struct OffsetToPtrOp
{
  template <typename T>
  __host__ __device__ __forceinline__ IteratorT operator()(T offset) const
  {
    return base_it + offset;
  }
  IteratorT base_it;
};

enum class TestDataGen
{
  // Random offsets into a data segment
  RANDOM,

  // Buffers cohesively reside next to each other
  CONSECUTIVE
};

std::string TestDataGenToString(TestDataGen gen)
{
  switch (gen)
  {
    case TestDataGen::RANDOM:
      return "TestDataGen::RANDOM";
    case TestDataGen::CONSECUTIVE:
      return "TestDataGen::CONSECUTIVE";
    default:
      return "Unknown";
  }
}

/**
 * @brief
 *
 * @tparam AtomicT The most granular type being copied. All source and destination pointers will be
 * aligned based on this type, the number of bytes being copied will be an integer multiple of this
 * type's size
 * @tparam BufferOffsetT Type used for indexing into the array of buffers
 * @tparam BufferSizeT Type used for indexing into individual bytes of a buffer (large enough to
 * cover the max buffer size)
 * @tparam ByteOffsetT Type used for indexing into bytes over *all* the buffers' sizes
 */
template <typename AtomicT, typename BufferOffsetT, typename BufferSizeT, typename ByteOffsetT>
void RunTest(BufferOffsetT num_buffers,
             BufferSizeT min_buffer_size,
             BufferSizeT max_buffer_size,
             TestDataGen input_gen,
             TestDataGen output_gen)
{
  try
  {
    using SrcPtrT = const uint8_t*;
    using DstPtrT = uint8_t*;

    // Buffer segment data (their offsets and sizes)
    c2h::host_vector<BufferSizeT> h_buffer_sizes(num_buffers);
    c2h::host_vector<ByteOffsetT> h_buffer_src_offsets(num_buffers);
    c2h::host_vector<ByteOffsetT> h_buffer_dst_offsets(num_buffers);

    // Generate the buffer sizes
    GenerateRandomData(h_buffer_sizes.data(), h_buffer_sizes.size(), min_buffer_size, max_buffer_size);

    // Make sure buffer sizes are a multiple of the most granular unit (one AtomicT) being copied
    // (round down)
    for (BufferOffsetT i = 0; i < num_buffers; i++)
    {
      h_buffer_sizes[i] = (h_buffer_sizes[i] / sizeof(AtomicT)) * sizeof(AtomicT);
    }

    // Compute the total bytes to be copied
    ByteOffsetT num_total_bytes = 0;
    for (BufferOffsetT i = 0; i < num_buffers; i++)
    {
      if (input_gen == TestDataGen::CONSECUTIVE)
      {
        h_buffer_src_offsets[i] = num_total_bytes;
      }
      if (output_gen == TestDataGen::CONSECUTIVE)
      {
        h_buffer_dst_offsets[i] = num_total_bytes;
      }
      num_total_bytes += h_buffer_sizes[i];
    }

    // Shuffle input buffer source-offsets
    std::uint_fast32_t shuffle_seed = 320981U;
    if (input_gen == TestDataGen::RANDOM)
    {
      h_buffer_src_offsets = GetShuffledBufferOffsets<BufferOffsetT, ByteOffsetT>(h_buffer_sizes, shuffle_seed);
      shuffle_seed += 42;
    }

    // Shuffle input buffer source-offsets
    if (output_gen == TestDataGen::RANDOM)
    {
      h_buffer_dst_offsets = GetShuffledBufferOffsets<BufferOffsetT, ByteOffsetT>(h_buffer_sizes, shuffle_seed);
    }

    // Populate the data source with random data
    using RandomInitAliasT         = uint16_t;
    std::size_t num_aliased_factor = sizeof(RandomInitAliasT) / sizeof(uint8_t);
    std::size_t num_aliased_units  = CUB_QUOTIENT_CEILING(num_total_bytes, num_aliased_factor);

    c2h::host_vector<std::uint8_t> h_in(num_aliased_units * num_aliased_factor);
    c2h::host_vector<std::uint8_t> h_out(num_total_bytes);
    c2h::host_vector<std::uint8_t> h_gpu_results(num_total_bytes);

    // Generate random offsets into the random-bits data buffer
    GenerateRandomData(reinterpret_cast<RandomInitAliasT*>(thrust::raw_pointer_cast(h_in.data())), num_aliased_units);

    c2h::device_vector<std::uint8_t> d_in(h_in);
    c2h::device_vector<std::uint8_t> d_out(num_total_bytes);
    c2h::device_vector<ByteOffsetT> d_buffer_src_offsets(h_buffer_src_offsets);
    c2h::device_vector<ByteOffsetT> d_buffer_dst_offsets(h_buffer_dst_offsets);
    c2h::device_vector<BufferSizeT> d_buffer_sizes(h_buffer_sizes);

    // Prepare d_buffer_srcs
    OffsetToPtrOp<SrcPtrT> src_transform_op{static_cast<SrcPtrT>(thrust::raw_pointer_cast(d_in.data()))};
    thrust::transform_iterator<OffsetToPtrOp<SrcPtrT>, ByteOffsetT*> d_buffer_srcs(
      thrust::raw_pointer_cast(d_buffer_src_offsets.data()), src_transform_op);

    // Prepare d_buffer_dsts
    OffsetToPtrOp<DstPtrT> dst_transform_op{static_cast<DstPtrT>(thrust::raw_pointer_cast(d_out.data()))};
    thrust::transform_iterator<OffsetToPtrOp<DstPtrT>, ByteOffsetT*> d_buffer_dsts(
      thrust::raw_pointer_cast(d_buffer_dst_offsets.data()), dst_transform_op);

    // Get temporary storage requirements
    std::size_t temp_storage_bytes = 0;
    CubDebugExit(cub::DeviceMemcpy::Batched(
      nullptr, temp_storage_bytes, d_buffer_srcs, d_buffer_dsts, d_buffer_sizes.cbegin(), num_buffers));

    c2h::device_vector<std::uint8_t> d_temp_storage(temp_storage_bytes);

    // Invoke device-side algorithm being under test
    CubDebugExit(cub::DeviceMemcpy::Batched(
      thrust::raw_pointer_cast(d_temp_storage.data()),
      temp_storage_bytes,
      d_buffer_srcs,
      d_buffer_dsts,
      d_buffer_sizes.begin(),
      num_buffers));

    // Copy back the output buffer
    CubDebugExit(cudaDeviceSynchronize());
    h_gpu_results = d_out;

    // CPU-side result generation for verification
    for (BufferOffsetT i = 0; i < num_buffers; i++)
    {
      std::memcpy(thrust::raw_pointer_cast(h_out.data()) + h_buffer_dst_offsets[i],
                  thrust::raw_pointer_cast(h_in.data()) + h_buffer_src_offsets[i],
                  h_buffer_sizes[i]);
    }

    for (ByteOffsetT i = 0; i < num_total_bytes; i++)
    {
      if (h_gpu_results[i] != h_out[i])
      {
        std::cout << "Mismatch at index " << i << ", CPU vs. GPU: " << static_cast<uint16_t>(h_gpu_results[i]) << ", "
                  << static_cast<uint16_t>(h_out[i]) << "\n";
      }
      AssertEquals(h_out[i], h_gpu_results[i]);
    }
  }
  catch (std::bad_alloc& e)
  {
    (void) e;
#ifdef DEBUG_CHECKED_ALLOC_FAILURE
    std::cout
      << "Skipping test 'RunTest(" //
      << num_buffers << ", " //
      << min_buffer_size << ", " //
      << max_buffer_size << ", " //
      << TestDataGenToString(input_gen) << ", " //
      << TestDataGenToString(output_gen) << ")" //
      << "' due to insufficient memory: " << e.what() << "\n";
#endif // DEBUG_CHECKED_ALLOC_FAILURE
  }
}

int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  // Initialize device
  CubDebugExit(args.DeviceInit());

  //---------------------------------------------------------------------
  // DeviceMemcpy::Batched tests
  //---------------------------------------------------------------------
  // The most granular type being copied. Buffer's will be aligned and their size be an integer
  // multiple of this type
  using AtomicCopyT = uint8_t;

  // Type used for indexing into the array of buffers
  using BufferOffsetT = uint32_t;

  // Type used for indexing into individual bytes of a buffer (large enough to cover the max buffer
  using BufferSizeT = uint32_t;

  // Type used for indexing into bytes over *all* the buffers' sizes
  using ByteOffsetT = uint32_t;

  // Total number of bytes that are targeted to be copied on each run
  constexpr BufferOffsetT target_copy_size = 64U << 20;

  // The number of randomly
  constexpr std::size_t num_rnd_buffer_range_tests = 32;

  // Each buffer's size will be random within this interval
  c2h::host_vector<std::pair<std::size_t, std::size_t>> buffer_size_ranges = {
    {0, 1},
    {1, 2},
    {0, 16},
    {1, 32},
    {1, 1024},
    {1, 32 * 1024},
    {128 * 1024, 256 * 1024},
    {target_copy_size, target_copy_size}};

  std::mt19937 rng(0);
  std::uniform_int_distribution<std::size_t> size_dist(1, 1000000);
  for (std::size_t i = 0; i < num_rnd_buffer_range_tests; i++)
  {
    auto range_begin = size_dist(rng);
    auto range_end   = size_dist(rng);
    if (range_begin > range_end)
    {
      std::swap(range_begin, range_end);
    }
    buffer_size_ranges.push_back({range_begin, range_end});
  }

  for (const auto& buffer_size_range : buffer_size_ranges)
  {
    BufferSizeT min_buffer_size =
      static_cast<BufferSizeT>(CUB_ROUND_UP_NEAREST(buffer_size_range.first, sizeof(AtomicCopyT)));
    BufferSizeT max_buffer_size = static_cast<BufferSizeT>(
      CUB_ROUND_UP_NEAREST(buffer_size_range.second, static_cast<BufferSizeT>(sizeof(AtomicCopyT))));
    double average_buffer_size       = (min_buffer_size + max_buffer_size) / 2.0;
    BufferOffsetT target_num_buffers = static_cast<BufferOffsetT>(target_copy_size / average_buffer_size);

    // Run tests with input buffer being consecutive and output buffers being consecutive
    RunTest<AtomicCopyT, BufferOffsetT, BufferSizeT, ByteOffsetT>(
      target_num_buffers, min_buffer_size, max_buffer_size, TestDataGen::CONSECUTIVE, TestDataGen::CONSECUTIVE);

    // Run tests with input buffer being randomly shuffled and output buffers being randomly
    // shuffled
    RunTest<AtomicCopyT, BufferOffsetT, BufferSizeT, ByteOffsetT>(
      target_num_buffers, min_buffer_size, max_buffer_size, TestDataGen::RANDOM, TestDataGen::RANDOM);
  }

  //---------------------------------------------------------------------
  // DeviceMemcpy::Batched test with 64-bit offsets
  //---------------------------------------------------------------------
  using ByteOffset64T = uint64_t;
  using BufferSize64T = uint64_t;
  ByteOffset64T large_target_copy_size =
    static_cast<ByteOffset64T>(std::numeric_limits<uint32_t>::max()) + (128ULL * 1024ULL * 1024ULL);
  // Make sure min_buffer_size is in fact smaller than max buffer size
  constexpr BufferOffsetT single_buffer = 1;

  // Run tests with input buffer being consecutive and output buffers being consecutive
  RunTest<AtomicCopyT, BufferOffsetT, BufferSize64T, ByteOffset64T>(
    single_buffer, large_target_copy_size, large_target_copy_size, TestDataGen::CONSECUTIVE, TestDataGen::CONSECUTIVE);
}
