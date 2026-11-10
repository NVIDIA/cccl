//===----------------------------------------------------------------------===//
//
// Part of the CUDA Core Compute Libraries, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/coop/tile.cuh>
#include <cuda/std/cstdint>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

namespace ct = ::cuda::tiles;

inline constexpr size_t tile_size           = 256;
inline constexpr size_t histogram_bins      = 128;
inline constexpr size_t default_tile_blocks = 257;
inline constexpr size_t maximum_tile_blocks = 4096;
inline constexpr size_t maximum_iterations  = 1000;

struct options
{
  size_t tile_blocks = default_tile_blocks;
  size_t iterations  = 1;
  bool help          = false;
};

void print_usage(std::string_view program)
{
  std::cout
    << "Usage: " << program << " [--tile-blocks N] [--iterations N]\n"
    << "  --tile-blocks N  logical Tile Grid blocks (1-" << maximum_tile_blocks << ")\n"
    << "  --iterations N   repeated launches (1-" << maximum_iterations << ")\n";
}

bool parse_bounded_positive(std::string_view text, size_t upper_bound, size_t& value)
{
  size_t parsed        = 0;
  const auto* begin    = text.data();
  const auto* end      = begin + text.size();
  const auto [ptr, ec] = std::from_chars(begin, end, parsed);
  if (text.empty() || ec != std::errc{} || ptr != end || parsed == 0 || parsed > upper_bound)
  {
    return false;
  }
  value = parsed;
  return true;
}

bool parse_options(int argc, char** argv, options& parsed)
{
  for (int index = 1; index < argc; ++index)
  {
    const std::string_view argument{argv[index]};
    if (argument == "--help")
    {
      parsed.help = true;
      return true;
    }
    if (argument != "--tile-blocks" && argument != "--iterations")
    {
      std::cerr << "unknown argument: " << argument << '\n';
      return false;
    }
    if (++index == argc)
    {
      std::cerr << "missing value for " << argument << '\n';
      return false;
    }

    size_t value             = 0;
    const size_t upper_bound = argument == "--tile-blocks" ? maximum_tile_blocks : maximum_iterations;
    if (!parse_bounded_positive(argv[index], upper_bound, value))
    {
      std::cerr << "invalid value for " << argument << ": " << argv[index] << '\n';
      return false;
    }
    if (argument == "--tile-blocks")
    {
      parsed.tile_blocks = value;
    }
    else
    {
      parsed.iterations = value;
    }
  }
  return true;
}

bool check_cuda(cudaError_t status, std::string_view operation)
{
  if (status == cudaSuccess)
  {
    return true;
  }
  std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
  return false;
}

template <typename T>
bool allocate_device(T** pointer, size_t bytes, std::string_view name)
{
  return check_cuda(cudaMalloc(reinterpret_cast<void**>(pointer), bytes), name);
}

__tile_global__ void mixed_operations_kernel(
  const int* signed_input,
  const unsigned int* unsigned_input,
  const unsigned long long* values,
  const unsigned char* histogram_input,
  const unsigned char* histogram_256_input,
  const int* group_input,
  int* seeded_a_output,
  int* seeded_b_output,
  unsigned int* scan_output,
  unsigned int* sum_output,
  unsigned int* xor_output,
  unsigned int* top_keys_output,
  unsigned long long* top_values_output,
  unsigned int* merge_keys_output,
  unsigned int* merge_pair_keys_output,
  unsigned long long* merge_pair_values_output,
  unsigned int* partial_merge_keys_output,
  int* histogram_output,
  int* histogram_256_output,
  int* heads_output,
  int* tails_output,
  int* pair_heads_output,
  int* pair_tails_output,
  unsigned int* exchange_s2b_output,
  unsigned int* exchange_original_output,
  unsigned int* exchange_b2s_output,
  ::cuda::std::int64_t total_tile_items,
  ::cuda::std::int64_t total_histogram_items,
  int initial_a,
  int initial_b,
  int runtime_k,
  int runtime_valid)
{
  const auto logical_block    = ct::bid().x;
  const auto tile_extent      = ct::extents<::cuda::std::int64_t, ct::dynamic_extent>{total_tile_items};
  const auto histogram_extent = ct::extents<::cuda::std::int64_t, ct::dynamic_extent>{total_histogram_items};

  auto signed_span                   = ct::tensor_span{signed_input, tile_extent};
  auto unsigned_span                 = ct::tensor_span{unsigned_input, tile_extent};
  auto values_span                   = ct::tensor_span{values, tile_extent};
  auto histogram_span                = ct::tensor_span{histogram_input, tile_extent};
  auto histogram_256_span            = ct::tensor_span{histogram_256_input, histogram_extent};
  auto group_span                    = ct::tensor_span{group_input, tile_extent};
  auto seeded_a_span                 = ct::tensor_span{seeded_a_output, tile_extent};
  auto seeded_b_span                 = ct::tensor_span{seeded_b_output, tile_extent};
  auto scan_span                     = ct::tensor_span{scan_output, tile_extent};
  auto sum_span                      = ct::tensor_span{sum_output, tile_extent};
  auto xor_span                      = ct::tensor_span{xor_output, tile_extent};
  auto top_keys_span                 = ct::tensor_span{top_keys_output, tile_extent};
  auto top_values_span               = ct::tensor_span{top_values_output, tile_extent};
  auto merge_keys_span               = ct::tensor_span{merge_keys_output, tile_extent};
  auto merge_pair_keys_span          = ct::tensor_span{merge_pair_keys_output, tile_extent};
  auto merge_pair_values_span        = ct::tensor_span{merge_pair_values_output, tile_extent};
  auto partial_merge_keys_span       = ct::tensor_span{partial_merge_keys_output, tile_extent};
  auto histogram_output_span         = ct::tensor_span{histogram_output, histogram_extent};
  auto histogram_256_output_span     = ct::tensor_span{histogram_256_output, tile_extent};
  auto heads_output_span             = ct::tensor_span{heads_output, tile_extent};
  auto tails_output_span             = ct::tensor_span{tails_output, tile_extent};
  auto pair_heads_output_span        = ct::tensor_span{pair_heads_output, tile_extent};
  auto pair_tails_output_span        = ct::tensor_span{pair_tails_output, tile_extent};
  auto exchange_s2b_output_span      = ct::tensor_span{exchange_s2b_output, tile_extent};
  auto exchange_original_output_span = ct::tensor_span{exchange_original_output, tile_extent};
  auto exchange_b2s_output_span      = ct::tensor_span{exchange_b2s_output, tile_extent};

  auto signed_view                   = ct::partition_view{signed_span, ct::shape<tile_size>{}};
  auto unsigned_view                 = ct::partition_view{unsigned_span, ct::shape<tile_size>{}};
  auto values_view                   = ct::partition_view{values_span, ct::shape<tile_size>{}};
  auto histogram_view                = ct::partition_view{histogram_span, ct::shape<tile_size>{}};
  auto histogram_256_view            = ct::partition_view{histogram_256_span, ct::shape<histogram_bins>{}};
  auto group_view                    = ct::partition_view{group_span, ct::shape<tile_size>{}};
  auto seeded_a_view                 = ct::partition_view{seeded_a_span, ct::shape<tile_size>{}};
  auto seeded_b_view                 = ct::partition_view{seeded_b_span, ct::shape<tile_size>{}};
  auto scan_view                     = ct::partition_view{scan_span, ct::shape<tile_size>{}};
  auto sum_view                      = ct::partition_view{sum_span, ct::shape<tile_size>{}};
  auto xor_view                      = ct::partition_view{xor_span, ct::shape<tile_size>{}};
  auto top_keys_view                 = ct::partition_view{top_keys_span, ct::shape<tile_size>{}};
  auto top_values_view               = ct::partition_view{top_values_span, ct::shape<tile_size>{}};
  auto merge_keys_view               = ct::partition_view{merge_keys_span, ct::shape<tile_size>{}};
  auto merge_pair_keys_view          = ct::partition_view{merge_pair_keys_span, ct::shape<tile_size>{}};
  auto merge_pair_values_view        = ct::partition_view{merge_pair_values_span, ct::shape<tile_size>{}};
  auto partial_merge_keys_view       = ct::partition_view{partial_merge_keys_span, ct::shape<tile_size>{}};
  auto histogram_output_view         = ct::partition_view{histogram_output_span, ct::shape<histogram_bins>{}};
  auto histogram_256_output_view     = ct::partition_view{histogram_256_output_span, ct::shape<tile_size>{}};
  auto heads_output_view             = ct::partition_view{heads_output_span, ct::shape<tile_size>{}};
  auto tails_output_view             = ct::partition_view{tails_output_span, ct::shape<tile_size>{}};
  auto pair_heads_output_view        = ct::partition_view{pair_heads_output_span, ct::shape<tile_size>{}};
  auto pair_tails_output_view        = ct::partition_view{pair_tails_output_span, ct::shape<tile_size>{}};
  auto exchange_s2b_output_view      = ct::partition_view{exchange_s2b_output_span, ct::shape<tile_size>{}};
  auto exchange_original_output_view = ct::partition_view{exchange_original_output_span, ct::shape<tile_size>{}};
  auto exchange_b2s_output_view      = ct::partition_view{exchange_b2s_output_span, ct::shape<tile_size>{}};

  auto signed_values        = signed_view.load(logical_block);
  auto unsigned_values      = unsigned_view.load(logical_block);
  auto pair_values          = values_view.load(logical_block);
  auto histogram_values     = histogram_view.load(logical_block);
  auto histogram_256_values = histogram_256_view.load(logical_block);
  auto groups               = group_view.load(logical_block);
  auto initial_b_tile       = ct::reshape(ct::tile{initial_b}, ct::shape<1>{});
  auto runtime_valid_tile   = ct::reshape(ct::tile{runtime_valid}, ct::shape<1>{});
  short begin_bit           = 0;

  auto seeded_a = ::cuda::coop::tile::exclusive_scan(signed_values, initial_a);
  auto seeded_b = ::cuda::coop::tile::scan<::cuda::coop::tile::scan_mode::exclusive>(
    signed_values, ::cuda::std::plus<>{}, initial_b_tile);
  auto scanned =
    ::cuda::coop::tile::scan<::cuda::coop::tile::scan_mode::exclusive>(unsigned_values, ::cuda::std::plus<>{});
  auto summed_raw = ::cuda::coop::tile::sum(unsigned_values);
  auto summed = ::cuda::coop::tile::exchange_striped_to_blocked(summed_raw, ::cuda::coop::tile::exchange_policy<128>{});
  auto xored_raw = ::cuda::coop::tile::reduce(unsigned_values, ::cuda::std::bit_xor<>{});
  auto xored = ::cuda::coop::tile::exchange_blocked_to_striped(xored_raw, ::cuda::coop::tile::exchange_policy<128>{});
  auto [top_keys,
        top_values] = ::cuda::coop::tile::topk_max_pairs(unsigned_values, pair_values, runtime_k, begin_bit, 32);
  auto merge_keys   = ::cuda::coop::tile::merge_sort_keys(unsigned_values);
  auto [merge_pair_keys, merge_pair_values] =
    ::cuda::coop::tile::merge_sort_pairs(unsigned_values, pair_values, ::cuda::std::greater<>{});
  auto partial_merge_keys       = ::cuda::coop::tile::merge_sort_keys(unsigned_values, runtime_valid_tile, -1);
  auto histogram                = ::cuda::coop::tile::histogram<histogram_bins>(histogram_values);
  auto histogram_256            = ::cuda::coop::tile::histogram_256(histogram_256_values);
  auto heads                    = ::cuda::coop::tile::discontinuity_flag_heads(groups);
  auto tails                    = ::cuda::coop::tile::discontinuity_flag_tails(groups);
  auto [pair_heads, pair_tails] = ::cuda::coop::tile::discontinuity_flag_heads_and_tails(groups);
  auto exchange_live_values     = unsigned_view.load(logical_block);
  auto exchange_s2b =
    ::cuda::coop::tile::exchange_striped_to_blocked(exchange_live_values, ::cuda::coop::tile::exchange_policy<128>{});
  exchange_s2b_output_view.store(exchange_s2b, logical_block);
  // This use after the inout call is the live-original preservation gate.
  exchange_original_output_view.store(exchange_live_values, logical_block);
  auto exchange_dead_values = unsigned_view.load(logical_block);
  auto exchange_b2s =
    ::cuda::coop::tile::exchange_blocked_to_striped(exchange_dead_values, ::cuda::coop::tile::exchange_policy<128>{});
  // exchange_dead_values is intentionally dead after this call, allowing the
  // compiler to reuse its storage for the returned Tile.
  exchange_b2s_output_view.store(exchange_b2s, logical_block);

  seeded_a_view.store(seeded_a, logical_block);
  seeded_b_view.store(seeded_b, logical_block);
  scan_view.store(scanned, logical_block);
  sum_view.store(summed, logical_block);
  xor_view.store(xored, logical_block);
  top_keys_view.store(top_keys, logical_block);
  top_values_view.store(top_values, logical_block);
  merge_keys_view.store(merge_keys, logical_block);
  merge_pair_keys_view.store(merge_pair_keys, logical_block);
  merge_pair_values_view.store(merge_pair_values, logical_block);
  partial_merge_keys_view.store(partial_merge_keys, logical_block);
  histogram_output_view.store(histogram, logical_block);
  histogram_256_output_view.store(histogram_256, logical_block);
  heads_output_view.store(heads, logical_block);
  tails_output_view.store(tails, logical_block);
  pair_heads_output_view.store(pair_heads, logical_block);
  pair_tails_output_view.store(pair_tails, logical_block);
}

int main(int argc, char** argv)
{
  options run;
  if (!parse_options(argc, argv, run))
  {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }
  if (run.help)
  {
    print_usage(argv[0]);
    return EXIT_SUCCESS;
  }
  if (run.tile_blocks > (std::numeric_limits<size_t>::max)() / tile_size
      || run.tile_blocks > (std::numeric_limits<size_t>::max)() / histogram_bins)
  {
    std::cerr << "requested logical Tile Grid overflows host allocation sizes\n";
    return EXIT_FAILURE;
  }

  const size_t total_tile_items          = run.tile_blocks * tile_size;
  const size_t total_histogram_items     = run.tile_blocks * histogram_bins;
  const size_t int_bytes                 = total_tile_items * sizeof(int);
  const size_t uint_bytes                = total_tile_items * sizeof(unsigned int);
  const size_t value_bytes               = total_tile_items * sizeof(unsigned long long);
  const size_t histogram_input_bytes     = total_tile_items * sizeof(unsigned char);
  const size_t histogram_256_input_bytes = total_histogram_items * sizeof(unsigned char);
  const size_t histogram_output_bytes    = total_histogram_items * sizeof(int);
  constexpr int initial_a                = 3;
  constexpr int initial_b                = 9;
  constexpr int topk                     = 11;
  constexpr int merge_valid              = 193;

  std::vector<int> signed_input(total_tile_items);
  std::vector<unsigned int> unsigned_input(total_tile_items);
  std::vector<unsigned long long> values(total_tile_items);
  std::vector<unsigned char> histogram_input(total_tile_items);
  std::vector<unsigned char> histogram_256_input(total_histogram_items);
  std::vector<int> group_input(total_tile_items);
  for (size_t block = 0; block < run.tile_blocks; ++block)
  {
    const int signed_step = 1 + static_cast<int>(block % 3);
    for (size_t item = 0; item < tile_size; ++item)
    {
      const size_t global_index  = block * tile_size + item;
      signed_input[global_index] = signed_step;
      unsigned_input[global_index] =
        item % 2 == 0 ? 0x80000000u + static_cast<unsigned int>(global_index) : static_cast<unsigned int>(global_index);
      values[global_index]          = static_cast<unsigned long long>(global_index);
      histogram_input[global_index] = static_cast<unsigned char>((item + block) % histogram_bins);
      group_input[global_index]     = static_cast<int>(1000 * block + item / 4);
    }
    for (size_t item = 0; item < histogram_bins; ++item)
    {
      histogram_256_input[block * histogram_bins + item] = static_cast<unsigned char>((item + 17 * block) % tile_size);
    }
  }

  // The inputs do not change between launches. Build the sorting references
  // once so --iterations stresses the device path instead of repeating
  // identical O0 host-side sorts.
  std::vector<unsigned int> expected_merge_keys(total_tile_items);
  std::vector<unsigned int> expected_partial_keys(total_tile_items, ~0u);
  for (size_t block = 0; block < run.tile_blocks; ++block)
  {
    const size_t block_offset = block * tile_size;
    std::copy(unsigned_input.begin() + block_offset,
              unsigned_input.begin() + block_offset + tile_size,
              expected_merge_keys.begin() + block_offset);
    std::sort(expected_merge_keys.begin() + block_offset, expected_merge_keys.begin() + block_offset + tile_size);

    std::copy(unsigned_input.begin() + block_offset,
              unsigned_input.begin() + block_offset + merge_valid,
              expected_partial_keys.begin() + block_offset);
    std::sort(expected_partial_keys.begin() + block_offset, expected_partial_keys.begin() + block_offset + merge_valid);
  }

  std::vector<int> seeded_a_output(total_tile_items);
  std::vector<int> seeded_b_output(total_tile_items);
  std::vector<unsigned int> scan_output(total_tile_items);
  std::vector<unsigned int> sum_output(total_tile_items);
  std::vector<unsigned int> xor_output(total_tile_items);
  std::vector<unsigned int> top_keys_output(total_tile_items);
  std::vector<unsigned long long> top_values_output(total_tile_items);
  std::vector<unsigned int> merge_keys_output(total_tile_items);
  std::vector<unsigned int> merge_pair_keys_output(total_tile_items);
  std::vector<unsigned long long> merge_pair_values_output(total_tile_items);
  std::vector<unsigned int> partial_merge_keys_output(total_tile_items);
  std::vector<int> histogram_output(total_histogram_items);
  std::vector<int> histogram_256_output(total_tile_items);
  std::vector<int> heads_output(total_tile_items);
  std::vector<int> tails_output(total_tile_items);
  std::vector<int> pair_heads_output(total_tile_items);
  std::vector<int> pair_tails_output(total_tile_items);
  std::vector<unsigned int> exchange_s2b_output(total_tile_items);
  std::vector<unsigned int> exchange_original_output(total_tile_items);
  std::vector<unsigned int> exchange_b2s_output(total_tile_items);

  int* device_signed_input                            = nullptr;
  unsigned int* device_unsigned_input                 = nullptr;
  unsigned long long* device_values                   = nullptr;
  unsigned char* device_histogram_input               = nullptr;
  unsigned char* device_histogram_256_input           = nullptr;
  int* device_group_input                             = nullptr;
  int* device_seeded_a_output                         = nullptr;
  int* device_seeded_b_output                         = nullptr;
  unsigned int* device_scan_output                    = nullptr;
  unsigned int* device_sum_output                     = nullptr;
  unsigned int* device_xor_output                     = nullptr;
  unsigned int* device_top_keys_output                = nullptr;
  unsigned long long* device_top_values_output        = nullptr;
  unsigned int* device_merge_keys_output              = nullptr;
  unsigned int* device_merge_pair_keys_output         = nullptr;
  unsigned long long* device_merge_pair_values_output = nullptr;
  unsigned int* device_partial_merge_keys_output      = nullptr;
  int* device_histogram_output                        = nullptr;
  int* device_histogram_256_output                    = nullptr;
  int* device_heads_output                            = nullptr;
  int* device_tails_output                            = nullptr;
  int* device_pair_heads_output                       = nullptr;
  int* device_pair_tails_output                       = nullptr;
  unsigned int* device_exchange_s2b_output            = nullptr;
  unsigned int* device_exchange_original_output       = nullptr;
  unsigned int* device_exchange_b2s_output            = nullptr;

  if (!allocate_device(&device_signed_input, int_bytes, "cudaMalloc(signed_input)")
      || !allocate_device(&device_unsigned_input, uint_bytes, "cudaMalloc(unsigned_input)")
      || !allocate_device(&device_values, value_bytes, "cudaMalloc(values)")
      || !allocate_device(&device_histogram_input, histogram_input_bytes, "cudaMalloc(histogram_input)")
      || !allocate_device(&device_histogram_256_input, histogram_256_input_bytes, "cudaMalloc(histogram_256_input)")
      || !allocate_device(&device_group_input, int_bytes, "cudaMalloc(group_input)")
      || !allocate_device(&device_seeded_a_output, int_bytes, "cudaMalloc(seeded_a_output)")
      || !allocate_device(&device_seeded_b_output, int_bytes, "cudaMalloc(seeded_b_output)")
      || !allocate_device(&device_scan_output, uint_bytes, "cudaMalloc(scan_output)")
      || !allocate_device(&device_sum_output, uint_bytes, "cudaMalloc(sum_output)")
      || !allocate_device(&device_xor_output, uint_bytes, "cudaMalloc(xor_output)")
      || !allocate_device(&device_top_keys_output, uint_bytes, "cudaMalloc(top_keys_output)")
      || !allocate_device(&device_top_values_output, value_bytes, "cudaMalloc(top_values_output)")
      || !allocate_device(&device_merge_keys_output, uint_bytes, "cudaMalloc(merge_keys_output)")
      || !allocate_device(&device_merge_pair_keys_output, uint_bytes, "cudaMalloc(merge_pair_keys_output)")
      || !allocate_device(&device_merge_pair_values_output, value_bytes, "cudaMalloc(merge_pair_values_output)")
      || !allocate_device(&device_partial_merge_keys_output, uint_bytes, "cudaMalloc(partial_merge_keys_output)")
      || !allocate_device(&device_histogram_output, histogram_output_bytes, "cudaMalloc(histogram_output)")
      || !allocate_device(&device_histogram_256_output, int_bytes, "cudaMalloc(histogram_256_output)")
      || !allocate_device(&device_heads_output, int_bytes, "cudaMalloc(heads_output)")
      || !allocate_device(&device_tails_output, int_bytes, "cudaMalloc(tails_output)")
      || !allocate_device(&device_pair_heads_output, int_bytes, "cudaMalloc(pair_heads_output)")
      || !allocate_device(&device_pair_tails_output, int_bytes, "cudaMalloc(pair_tails_output)")
      || !allocate_device(&device_exchange_s2b_output, uint_bytes, "cudaMalloc(exchange_s2b_output)")
      || !allocate_device(&device_exchange_original_output, uint_bytes, "cudaMalloc(exchange_original_output)")
      || !allocate_device(&device_exchange_b2s_output, uint_bytes, "cudaMalloc(exchange_b2s_output)"))
  {
    return EXIT_FAILURE;
  }

  if (!check_cuda(cudaMemcpy(device_signed_input, signed_input.data(), int_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy(signed_input)")
      || !check_cuda(cudaMemcpy(device_unsigned_input, unsigned_input.data(), uint_bytes, cudaMemcpyHostToDevice),
                     "cudaMemcpy(unsigned_input)")
      || !check_cuda(cudaMemcpy(device_values, values.data(), value_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(values)")
      || !check_cuda(
        cudaMemcpy(device_histogram_input, histogram_input.data(), histogram_input_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(histogram_input)")
      || !check_cuda(
        cudaMemcpy(
          device_histogram_256_input, histogram_256_input.data(), histogram_256_input_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy(histogram_256_input)")
      || !check_cuda(cudaMemcpy(device_group_input, group_input.data(), int_bytes, cudaMemcpyHostToDevice),
                     "cudaMemcpy(group_input)"))
  {
    return EXIT_FAILURE;
  }

  for (size_t iteration = 0; iteration < run.iterations; ++iteration)
  {
    if (!check_cuda(cudaMemset(device_seeded_a_output, 0xa5, int_bytes), "cudaMemset(seeded_a_output)")
        || !check_cuda(cudaMemset(device_seeded_b_output, 0xa5, int_bytes), "cudaMemset(seeded_b_output)")
        || !check_cuda(cudaMemset(device_scan_output, 0xa5, uint_bytes), "cudaMemset(scan_output)")
        || !check_cuda(cudaMemset(device_sum_output, 0xa5, uint_bytes), "cudaMemset(sum_output)")
        || !check_cuda(cudaMemset(device_xor_output, 0xa5, uint_bytes), "cudaMemset(xor_output)")
        || !check_cuda(cudaMemset(device_top_keys_output, 0xa5, uint_bytes), "cudaMemset(top_keys_output)")
        || !check_cuda(cudaMemset(device_top_values_output, 0xa5, value_bytes), "cudaMemset(top_values_output)")
        || !check_cuda(cudaMemset(device_merge_keys_output, 0xa5, uint_bytes), "cudaMemset(merge_keys_output)")
        || !check_cuda(cudaMemset(device_merge_pair_keys_output, 0xa5, uint_bytes), "cudaMemset(merge_pair_keys_output)")
        || !check_cuda(cudaMemset(device_merge_pair_values_output, 0xa5, value_bytes),
                       "cudaMemset(merge_pair_values_output)")
        || !check_cuda(cudaMemset(device_partial_merge_keys_output, 0xa5, uint_bytes),
                       "cudaMemset(partial_merge_keys_output)")
        || !check_cuda(cudaMemset(device_histogram_output, 0xa5, histogram_output_bytes), "cudaMemset(histogram_output)")
        || !check_cuda(cudaMemset(device_histogram_256_output, 0xa5, int_bytes), "cudaMemset(histogram_256_output)")
        || !check_cuda(cudaMemset(device_heads_output, 0xa5, int_bytes), "cudaMemset(heads_output)")
        || !check_cuda(cudaMemset(device_tails_output, 0xa5, int_bytes), "cudaMemset(tails_output)")
        || !check_cuda(cudaMemset(device_pair_heads_output, 0xa5, int_bytes), "cudaMemset(pair_heads_output)")
        || !check_cuda(cudaMemset(device_pair_tails_output, 0xa5, int_bytes), "cudaMemset(pair_tails_output)")
        || !check_cuda(cudaMemset(device_exchange_s2b_output, 0xa5, uint_bytes), "cudaMemset(exchange_s2b_output)")
        || !check_cuda(cudaMemset(device_exchange_original_output, 0xa5, uint_bytes),
                       "cudaMemset(exchange_original_output)")
        || !check_cuda(cudaMemset(device_exchange_b2s_output, 0xa5, uint_bytes), "cudaMemset(exchange_b2s_output)"))
    {
      return EXIT_FAILURE;
    }

    mixed_operations_kernel<<<static_cast<unsigned int>(run.tile_blocks), 1>>>(
      device_signed_input,
      device_unsigned_input,
      device_values,
      device_histogram_input,
      device_histogram_256_input,
      device_group_input,
      device_seeded_a_output,
      device_seeded_b_output,
      device_scan_output,
      device_sum_output,
      device_xor_output,
      device_top_keys_output,
      device_top_values_output,
      device_merge_keys_output,
      device_merge_pair_keys_output,
      device_merge_pair_values_output,
      device_partial_merge_keys_output,
      device_histogram_output,
      device_histogram_256_output,
      device_heads_output,
      device_tails_output,
      device_pair_heads_output,
      device_pair_tails_output,
      device_exchange_s2b_output,
      device_exchange_original_output,
      device_exchange_b2s_output,
      static_cast<::cuda::std::int64_t>(total_tile_items),
      static_cast<::cuda::std::int64_t>(total_histogram_items),
      initial_a,
      initial_b,
      topk,
      merge_valid);
    if (!check_cuda(cudaGetLastError(), "mixed_operations_kernel launch")
        || !check_cuda(cudaDeviceSynchronize(), "mixed_operations_kernel execution"))
    {
      std::cerr << "iteration " << iteration << " of " << run.iterations << " failed\n";
      return EXIT_FAILURE;
    }

    if (!check_cuda(cudaMemcpy(seeded_a_output.data(), device_seeded_a_output, int_bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy(seeded_a_output)")
        || !check_cuda(cudaMemcpy(seeded_b_output.data(), device_seeded_b_output, int_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(seeded_b_output)")
        || !check_cuda(cudaMemcpy(scan_output.data(), device_scan_output, uint_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(scan_output)")
        || !check_cuda(cudaMemcpy(sum_output.data(), device_sum_output, uint_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(sum_output)")
        || !check_cuda(cudaMemcpy(xor_output.data(), device_xor_output, uint_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(xor_output)")
        || !check_cuda(cudaMemcpy(top_keys_output.data(), device_top_keys_output, uint_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(top_keys_output)")
        || !check_cuda(
          cudaMemcpy(top_values_output.data(), device_top_values_output, value_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(top_values_output)")
        || !check_cuda(
          cudaMemcpy(merge_keys_output.data(), device_merge_keys_output, uint_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(merge_keys_output)")
        || !check_cuda(
          cudaMemcpy(merge_pair_keys_output.data(), device_merge_pair_keys_output, uint_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(merge_pair_keys_output)")
        || !check_cuda(
          cudaMemcpy(
            merge_pair_values_output.data(), device_merge_pair_values_output, value_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(merge_pair_values_output)")
        || !check_cuda(
          cudaMemcpy(
            partial_merge_keys_output.data(), device_partial_merge_keys_output, uint_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(partial_merge_keys_output)")
        || !check_cuda(
          cudaMemcpy(histogram_output.data(), device_histogram_output, histogram_output_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(histogram_output)")
        || !check_cuda(
          cudaMemcpy(histogram_256_output.data(), device_histogram_256_output, int_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(histogram_256_output)")
        || !check_cuda(cudaMemcpy(heads_output.data(), device_heads_output, int_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(heads_output)")
        || !check_cuda(cudaMemcpy(tails_output.data(), device_tails_output, int_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(tails_output)")
        || !check_cuda(cudaMemcpy(pair_heads_output.data(), device_pair_heads_output, int_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(pair_heads_output)")
        || !check_cuda(cudaMemcpy(pair_tails_output.data(), device_pair_tails_output, int_bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy(pair_tails_output)")
        || !check_cuda(
          cudaMemcpy(exchange_s2b_output.data(), device_exchange_s2b_output, uint_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(exchange_s2b_output)")
        || !check_cuda(
          cudaMemcpy(
            exchange_original_output.data(), device_exchange_original_output, uint_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(exchange_original_output)")
        || !check_cuda(
          cudaMemcpy(exchange_b2s_output.data(), device_exchange_b2s_output, uint_bytes, cudaMemcpyDeviceToHost),
          "cudaMemcpy(exchange_b2s_output)"))
    {
      return EXIT_FAILURE;
    }

    constexpr size_t block_threads    = 128;
    constexpr size_t items_per_thread = tile_size / block_threads;
    for (size_t block = 0; block < run.tile_blocks; ++block)
    {
      const size_t block_offset     = block * tile_size;
      const size_t histogram_offset = block * histogram_bins;
      const int signed_step         = 1 + static_cast<int>(block % 3);
      unsigned int expected_prefix  = 0;
      unsigned int expected_xor     = 0;

      for (size_t item = 0; item < tile_size; ++item)
      {
        const size_t index          = block_offset + item;
        const int expected_seeded_a = initial_a + static_cast<int>(item) * signed_step;
        const int expected_seeded_b = initial_b + static_cast<int>(item) * signed_step;
        if (seeded_a_output[index] != expected_seeded_a || seeded_b_output[index] != expected_seeded_b
            || scan_output[index] != expected_prefix)
        {
          std::cerr
            << "scan mismatch at block=" << block << " item=" << item << ": seeded_a=" << seeded_a_output[index]
            << " seeded_b=" << seeded_b_output[index] << " scan=" << scan_output[index]
            << " expected_scan=" << expected_prefix << '\n';
          return EXIT_FAILURE;
        }
        expected_prefix += unsigned_input[index];
        expected_xor ^= unsigned_input[index];
      }
      for (size_t item = 0; item < tile_size; ++item)
      {
        const size_t index = block_offset + item;
        if (sum_output[index] != expected_prefix || xor_output[index] != expected_xor)
        {
          std::cerr
            << "reduce mismatch at block=" << block << " item=" << item << ": sum=" << sum_output[index] << " xor="
            << xor_output[index] << " expected_sum=" << expected_prefix << " expected_xor=" << expected_xor << '\n';
          return EXIT_FAILURE;
        }
      }

      std::vector<unsigned int> actual_top_keys(
        top_keys_output.begin() + block_offset, top_keys_output.begin() + block_offset + topk);
      std::sort(actual_top_keys.begin(), actual_top_keys.end());
      for (int item = 0; item < topk; ++item)
      {
        const size_t index = block_offset + static_cast<size_t>(item);
        const unsigned int expected_key =
          0x80000000u + static_cast<unsigned int>(block_offset + tile_size - 2 - 2 * item);
        if (actual_top_keys[topk - 1 - item] != expected_key || (top_keys_output[index] & 0x80000000u) == 0
            || top_keys_output[index] != 0x80000000u + static_cast<unsigned int>(top_values_output[index]))
        {
          std::cerr
            << "TopK mismatch at block=" << block << " item=" << item << ": key=" << top_keys_output[index]
            << " value=" << top_values_output[index] << " sorted_key=" << actual_top_keys[topk - 1 - item]
            << " expected_key=" << expected_key << '\n';
          return EXIT_FAILURE;
        }
      }

      for (size_t item = 0; item < tile_size; ++item)
      {
        const size_t index         = block_offset + item;
        const auto source_index    = merge_pair_values_output[index];
        const bool source_in_block = source_index >= block_offset && source_index < block_offset + tile_size;
        if (merge_keys_output[index] != expected_merge_keys[index]
            || merge_pair_keys_output[index] != expected_merge_keys[block_offset + tile_size - 1 - item]
            || !source_in_block || merge_pair_keys_output[index] != unsigned_input[static_cast<size_t>(source_index)])
        {
          std::cerr
            << "Merge Sort mismatch at block=" << block << " item=" << item << ": key=" << merge_keys_output[index]
            << " pair_key=" << merge_pair_keys_output[index] << " source_index=" << source_index
            << " expected_key=" << expected_merge_keys[index]
            << " expected_pair_key=" << expected_merge_keys[block_offset + tile_size - 1 - item] << '\n';
          return EXIT_FAILURE;
        }
      }

      for (size_t item = 0; item < tile_size; ++item)
      {
        const size_t index = block_offset + item;
        if (partial_merge_keys_output[index] != expected_partial_keys[index])
        {
          std::cerr
            << "partial Merge Sort mismatch at block=" << block << " item=" << item
            << ": key=" << partial_merge_keys_output[index] << " expected_key=" << expected_partial_keys[index] << '\n';
          return EXIT_FAILURE;
        }
      }

      const size_t histogram_256_shift = (17 * block) % tile_size;
      for (size_t bin = 0; bin < tile_size; ++bin)
      {
        const int expected_256 = (bin + tile_size - histogram_256_shift) % tile_size < histogram_bins ? 1 : 0;
        if ((bin < histogram_bins && histogram_output[histogram_offset + bin] != 2)
            || histogram_256_output[block_offset + bin] != expected_256)
        {
          std::cerr
            << "histogram mismatch at block=" << block << " bin=" << bin
            << ": histogram=" << (bin < histogram_bins ? histogram_output[histogram_offset + bin] : -1)
            << " histogram_256=" << histogram_256_output[block_offset + bin]
            << " expected_histogram_256=" << expected_256 << '\n';
          return EXIT_FAILURE;
        }
      }

      for (size_t item = 0; item < tile_size; ++item)
      {
        const size_t index      = block_offset + item;
        const int expected_head = item == 0 || group_input[index] != group_input[index - 1];
        const int expected_tail = item + 1 == tile_size || group_input[index] != group_input[index + 1];
        if (heads_output[index] != expected_head || tails_output[index] != expected_tail
            || pair_heads_output[index] != expected_head || pair_tails_output[index] != expected_tail)
        {
          std::cerr
            << "discontinuity mismatch at block=" << block << " item=" << item << ": head=" << heads_output[index]
            << " tail=" << tails_output[index] << " pair_head=" << pair_heads_output[index]
            << " pair_tail=" << pair_tails_output[index] << " expected_head=" << expected_head
            << " expected_tail=" << expected_tail << '\n';
          return EXIT_FAILURE;
        }
      }

      for (size_t thread = 0; thread < block_threads; ++thread)
      {
        for (size_t item = 0; item < items_per_thread; ++item)
        {
          const size_t blocked_index = block_offset + thread * items_per_thread + item;
          const size_t striped_index = block_offset + item * block_threads + thread;
          if (exchange_s2b_output[blocked_index] != unsigned_input[striped_index]
              || exchange_original_output[blocked_index] != unsigned_input[blocked_index]
              || exchange_b2s_output[striped_index] != unsigned_input[blocked_index])
          {
            std::cerr
              << "Exchange mismatch at block=" << block << " thread=" << thread << " item=" << item
              << ": striped_to_blocked=" << exchange_s2b_output[blocked_index] << " expected_striped_to_blocked="
              << unsigned_input[striped_index] << " original_after_inout=" << exchange_original_output[blocked_index]
              << " expected_original=" << unsigned_input[blocked_index]
              << " blocked_to_striped=" << exchange_b2s_output[striped_index]
              << " expected_blocked_to_striped=" << unsigned_input[blocked_index] << '\n';
            return EXIT_FAILURE;
          }
        }
      }
    }
  }

  bool released = true;
  released &= check_cuda(cudaFree(device_exchange_b2s_output), "cudaFree(exchange_b2s_output)");
  released &= check_cuda(cudaFree(device_exchange_original_output), "cudaFree(exchange_original_output)");
  released &= check_cuda(cudaFree(device_exchange_s2b_output), "cudaFree(exchange_s2b_output)");
  released &= check_cuda(cudaFree(device_pair_tails_output), "cudaFree(pair_tails_output)");
  released &= check_cuda(cudaFree(device_pair_heads_output), "cudaFree(pair_heads_output)");
  released &= check_cuda(cudaFree(device_tails_output), "cudaFree(tails_output)");
  released &= check_cuda(cudaFree(device_heads_output), "cudaFree(heads_output)");
  released &= check_cuda(cudaFree(device_histogram_256_output), "cudaFree(histogram_256_output)");
  released &= check_cuda(cudaFree(device_histogram_output), "cudaFree(histogram_output)");
  released &= check_cuda(cudaFree(device_partial_merge_keys_output), "cudaFree(partial_merge_keys_output)");
  released &= check_cuda(cudaFree(device_merge_pair_values_output), "cudaFree(merge_pair_values_output)");
  released &= check_cuda(cudaFree(device_merge_pair_keys_output), "cudaFree(merge_pair_keys_output)");
  released &= check_cuda(cudaFree(device_merge_keys_output), "cudaFree(merge_keys_output)");
  released &= check_cuda(cudaFree(device_top_values_output), "cudaFree(top_values_output)");
  released &= check_cuda(cudaFree(device_top_keys_output), "cudaFree(top_keys_output)");
  released &= check_cuda(cudaFree(device_xor_output), "cudaFree(xor_output)");
  released &= check_cuda(cudaFree(device_sum_output), "cudaFree(sum_output)");
  released &= check_cuda(cudaFree(device_scan_output), "cudaFree(scan_output)");
  released &= check_cuda(cudaFree(device_seeded_b_output), "cudaFree(seeded_b_output)");
  released &= check_cuda(cudaFree(device_seeded_a_output), "cudaFree(seeded_a_output)");
  released &= check_cuda(cudaFree(device_values), "cudaFree(values)");
  released &= check_cuda(cudaFree(device_group_input), "cudaFree(group_input)");
  released &= check_cuda(cudaFree(device_histogram_256_input), "cudaFree(histogram_256_input)");
  released &= check_cuda(cudaFree(device_histogram_input), "cudaFree(histogram_input)");
  released &= check_cuda(cudaFree(device_unsigned_input), "cudaFree(unsigned_input)");
  released &= check_cuda(cudaFree(device_signed_input), "cudaFree(signed_input)");
  if (!released)
  {
    return EXIT_FAILURE;
  }

  std::cout << "PASS: C++ Tile API Scan/Reduce/TopK/MergeSort/Histogram/Discontinuity/Exchange"
            << " blocks=" << run.tile_blocks << " iterations=" << run.iterations << '\n';
  return EXIT_SUCCESS;
}
