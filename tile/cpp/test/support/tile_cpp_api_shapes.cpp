// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/coop/tile.cuh>

#include <type_traits>

namespace coop = cuda::coop::tile;
namespace ct   = cuda::tiles;

// Public constraints must support detection without instantiating function bodies.
template <typename T>
concept can_sum = requires(ct::tile<T, ct::shape<64>> input) { coop::sum(input); };

template <typename T, typename Op>
concept can_reduce = requires(ct::tile<T, ct::shape<64>> input, Op op) { coop::reduce(input, op); };

static_assert(can_sum<int> && can_sum<long> && can_sum<unsigned long>);
static_assert(can_sum<__half> && can_sum<__nv_bfloat16>);
static_assert(!can_sum<bool>);
static_assert(can_reduce<int, cuda::std::bit_xor<>>);
static_assert(can_reduce<float, cuda::std::plus<>>);
static_assert(!can_reduce<float, cuda::std::bit_xor<>>);
static_assert(!can_reduce<int, cuda::std::plus<float>>);
static_assert(!can_reduce<int, cuda::maximum<float>>);
static_assert(!can_reduce<bool, cuda::std::plus<>>);

template <size_t Axis, typename Tile>
concept can_sum_axis = requires(Tile input) { coop::sum<Axis>(input); };

static_assert(can_sum_axis<0, ct::tile<double, ct::shape<128>>>);
static_assert(can_sum_axis<1, ct::tile<int, ct::shape<8, 128>>>);
static_assert(!can_sum_axis<2, ct::tile<int, ct::shape<8, 128>>>);
static_assert(!can_sum_axis<0, ct::tile<int, ct::shape<>>>);
static_assert(!can_sum_axis<0, ct::tile<bool, ct::shape<128>>>);

void compile_reduction_axes(ct::tile<double, ct::shape<128>> values,
                            ct::tile<float, ct::shape<8, 128>> matrix,
                            ct::tile<int, ct::shape<4, 8, 16>> cube)
{
  const auto scalar  = coop::sum<0>(values);
  const auto rows    = coop::sum<1>(matrix);
  const auto columns = coop::sum<0>(matrix, coop::reduce_policy<coop::reduce_algorithm::raking>{});
  const auto total   = coop::reduce<0>(rows);
  const auto plus    = coop::reduce<1>(matrix, cuda::std::plus<>{});
  const auto minimum = coop::reduce<1>(matrix, cuda::minimum<>{});
  const auto maximum = coop::reduce<2>(cube, cuda::maximum<int>{});
  const auto product = coop::reduce<0>(cube, cuda::std::multiplies<>{});
  const auto bits    = coop::reduce<1>(cube, cuda::std::bit_xor<>{});
  static_assert(std::is_same_v<decltype(scalar), const ct::tile<double, ct::shape<1>>>);
  static_assert(std::is_same_v<decltype(rows), const ct::tile<float, ct::shape<8, 1>>>);
  static_assert(std::is_same_v<decltype(columns), const ct::tile<float, ct::shape<1, 128>>>);
  static_assert(std::is_same_v<decltype(total), const ct::tile<float, ct::shape<1, 1>>>);
  static_assert(std::is_same_v<decltype(plus), decltype(rows)>);
  static_assert(std::is_same_v<decltype(minimum), decltype(rows)>);
  static_assert(std::is_same_v<decltype(maximum), const ct::tile<int, ct::shape<4, 8, 1>>>);
  static_assert(std::is_same_v<decltype(product), const ct::tile<int, ct::shape<1, 8, 16>>>);
  static_assert(std::is_same_v<decltype(bits), const ct::tile<int, ct::shape<4, 1, 16>>>);
}

void compile_shapes(
  ct::tile<int, ct::shape<64>> keys,
  ct::tile<long long, ct::shape<64>> values,
  ct::tile<float, ct::shape<64>> float_values,
  ct::tile<__half, ct::shape<64>> half_values,
  ct::tile<__nv_bfloat16, ct::shape<64>> bfloat_values,
  ct::tile<unsigned int, ct::shape<64>> unsigned_keys,
  ct::tile<unsigned long long, ct::shape<64>> unsigned_values,
  ct::tile<long, ct::shape<64>> long_values,
  ct::tile<unsigned long, ct::shape<64>> unsigned_long_values,
  ct::tile<unsigned char, ct::shape<256>> histogram_input,
  ct::tile<unsigned char, ct::shape<128>> histogram_128_input,
  ct::tile<unsigned int, ct::shape<64>> run_lengths_u32,
  ct::tile<unsigned long long, ct::shape<64>> run_lengths_u64,
  ct::tile<unsigned int, ct::shape<1>> window_offset_u32,
  ct::tile<unsigned long long, ct::shape<1>> window_offset_u64,
  ct::tile<int, ct::shape<1>> tile_k,
  ct::tile<int, ct::shape<1>> tile_initial,
  int runtime_k,
  int runtime_initial)
{
  short short_control = 4;
  auto exclusive      = coop::exclusive_sum(keys);
  auto inclusive =
    coop::inclusive_scan(keys, cuda::maximum<int>{}, coop::scan_policy<coop::scan_algorithm::raking_memoize>{});
  auto multiplied                 = coop::exclusive_scan(keys, cuda::std::multiplies<>{}, runtime_initial);
  auto tile_initial_scan          = coop::exclusive_scan(keys, cuda::std::plus<>{}, tile_initial);
  auto default_scalar_initial     = coop::exclusive_scan(keys, runtime_initial);
  auto default_tile_initial       = coop::exclusive_scan(keys, tile_initial);
  auto float_initial_scan         = coop::exclusive_scan(float_values, cuda::std::plus<>{}, 0);
  auto generic                    = coop::scan<coop::scan_mode::inclusive>(keys);
  auto generic_inclusive_multiply = coop::scan<coop::scan_mode::inclusive>(keys, cuda::std::multiplies<>{});
  auto generic_exclusive_plus =
    coop::scan<coop::scan_mode::exclusive>(keys, cuda::std::plus<>{}, coop::scan_policy<>{});
  auto generic_exclusive_initial =
    coop::scan<coop::scan_mode::exclusive>(keys, cuda::std::bit_xor<>{}, runtime_initial);
  auto generic_exclusive_tile_initial =
    coop::scan<coop::scan_mode::exclusive>(keys, cuda::std::plus<>{}, tile_initial, coop::scan_policy<>{});
  auto unsigned_scan     = coop::inclusive_scan(unsigned_keys, cuda::maximum<>{});
  auto summed            = coop::sum(keys);
  auto half_sum          = coop::sum(half_values);
  auto bfloat_sum        = coop::sum(bfloat_values);
  auto unsigned_sum      = coop::sum(unsigned_keys);
  auto long_sum          = coop::sum(long_values);
  auto unsigned_long_sum = coop::sum(unsigned_long_values);
  auto reduced_plus      = coop::reduce(keys, cuda::std::plus<>{});
  auto reduced_multiply  = coop::reduce(keys, cuda::std::multiplies<>{});
  auto reduced_minimum   = coop::reduce(keys, cuda::minimum<>{}, coop::reduce_policy<coop::reduce_algorithm::raking>{});
  auto reduced_maximum   = coop::reduce(keys, cuda::maximum<>{});
  auto reduced_bits      = coop::reduce(keys, cuda::std::bit_xor<>{});
  auto unsigned_bits     = coop::reduce(unsigned_keys, cuda::std::bit_xor<>{});
  auto histogram_default = coop::histogram(histogram_input);
  auto histogram_128     = coop::histogram<128>(histogram_input, coop::histogram_policy<>{});
  auto fixed_histogram   = coop::histogram_256(histogram_128_input, coop::histogram_policy<>{});
  auto decoded_default   = coop::run_length_decode<128>(keys, run_lengths_u32);
  auto decoded_ulong     = coop::run_length_decode<128>(keys, unsigned_long_values);
  auto decoded_policy    = coop::run_length_decode<128>(keys, run_lengths_u32, coop::run_length_decode_policy<>{});
  auto decoded_scalar_offset =
    coop::run_length_decode<128>(unsigned_keys, run_lengths_u64, short_control, coop::run_length_decode_policy<>{});
  auto decoded_tile_offset = coop::run_length_decode<128>(keys, run_lengths_u32, window_offset_u32);
  auto [decoded_with_offsets, relative_offsets] = coop::run_length_decode_with_offsets<128>(
    keys, run_lengths_u32, window_offset_u32, coop::run_length_decode_policy<>{});
  auto [decoded_u64, relative_offsets_u64] =
    coop::run_length_decode_with_offsets<128>(unsigned_keys, run_lengths_u64, window_offset_u64);
  auto heads                     = coop::discontinuity_flag_heads(keys);
  auto tails                     = coop::discontinuity_flag_tails(keys, coop::discontinuity_policy<>{});
  auto [pair_heads, pair_tails]  = coop::discontinuity_flag_heads_and_tails(keys);
  auto adjacent_left             = coop::adjacent_difference_subtract_left(keys);
  auto adjacent_right            = coop::adjacent_difference_subtract_right(keys, coop::adjacent_difference_policy<>{});
  auto shuffled_up               = coop::shuffle_up(keys);
  auto shuffled_down             = coop::shuffle_down(keys, coop::shuffle_policy<>{});
  auto shuffled_offset           = coop::shuffle_offset(keys, 2, coop::shuffle_policy<>{});
  auto shuffled_offset_tile      = coop::shuffle_offset(keys, tile_k);
  auto shuffled_rotate           = coop::shuffle_rotate(keys, runtime_k);
  auto shuffled_rotate_tile      = coop::shuffle_rotate(keys, tile_k, coop::shuffle_policy<>{});
  auto exchange_s2b              = coop::exchange_striped_to_blocked(unsigned_keys, coop::exchange_policy<32>{});
  auto exchange_b2s              = coop::exchange_blocked_to_striped(unsigned_keys, coop::exchange_policy<32>{});
  auto explicit_s2b              = coop::exchange_striped_to_blocked<unsigned int, 64, 32>(unsigned_keys);
  auto explicit_b2s              = coop::exchange_blocked_to_striped<unsigned int, 64, 32>(unsigned_keys);
  auto ranks                     = coop::radix_rank_keys(unsigned_keys);
  auto signed_ranks              = coop::radix_rank_keys(keys);
  auto shifted_ranks             = coop::radix_rank_keys(unsigned_keys, 4);
  auto bounded_ranks             = coop::radix_rank_keys(unsigned_keys, 4, 8, coop::radix_rank_policy<>{});
  auto tile_begin_ranks          = coop::radix_rank_keys(unsigned_keys, tile_k);
  auto mixed_bit_ranks           = coop::radix_rank_keys(unsigned_keys, 0, tile_k);
  auto converted_mixed_bit_ranks = coop::radix_rank_keys(unsigned_keys, tile_k, short_control);
  auto descending_ranks          = coop::radix_rank_keys(
    unsigned_keys,
    8,
    16,
    coop::radix_rank_policy<8, true, false, coop::scan_algorithm::raking, coop::radix_smem_config::eight_byte>{});
  auto merge_keys         = coop::merge_sort_keys(keys);
  auto merge_keys_less    = coop::merge_sort_keys(keys, cuda::std::less<>{});
  auto merge_keys_greater = coop::merge_sort_keys(keys, cuda::std::greater<int>{}, coop::merge_sort_policy<>{});
  auto merge_keys_partial = coop::merge_sort_keys(keys, 48, -1);
  auto merge_keys_partial_greater =
    coop::merge_sort_keys(keys, cuda::std::greater<>{}, 48, -1, coop::merge_sort_policy<>{});
  auto merge_keys_tile_controls             = coop::merge_sort_keys(keys, tile_k, tile_initial);
  auto merge_keys_mixed_valid               = coop::merge_sort_keys(keys, tile_k, -1);
  auto merge_keys_converted_mixed_valid     = coop::merge_sort_keys(keys, tile_k, short_control);
  auto merge_keys_mixed_oob                 = coop::merge_sort_keys(keys, 48, tile_initial);
  auto merge_keys_tile_compare              = coop::merge_sort_keys(keys, cuda::std::less<>{}, tile_k, tile_initial);
  auto merge_keys_mixed_compare             = coop::merge_sort_keys(keys, cuda::std::greater<>{}, tile_k, -1);
  auto [merge_pair_keys, merge_pair_values] = coop::merge_sort_pairs(keys, values);
  auto [merge_pair_desc_keys, merge_pair_desc_values] =
    coop::merge_sort_pairs(keys, values, cuda::std::greater<>{}, coop::merge_sort_policy<>{});
  auto merge_pairs_partial         = coop::merge_sort_pairs(keys, values, 48, -1);
  auto merge_pairs_partial_compare = coop::merge_sort_pairs(keys, values, cuda::std::less<int>{}, 48, -1);
  auto merge_pairs_tile_controls   = coop::merge_sort_pairs(keys, values, tile_k, tile_initial);
  auto merge_pairs_mixed_valid     = coop::merge_sort_pairs(keys, values, tile_k, -1);
  auto merge_pairs_mixed_oob       = coop::merge_sort_pairs(keys, values, 48, tile_initial);
  auto merge_pairs_tile_compare    = coop::merge_sort_pairs(keys, values, cuda::std::greater<>{}, tile_k, tile_initial);
  auto merge_pairs_mixed_compare   = coop::merge_sort_pairs(keys, values, cuda::std::less<>{}, 48, tile_initial);
  auto radix_keys                  = coop::radix_sort_keys(unsigned_keys);
  auto radix_descending            = coop::radix_sort_keys(
    unsigned_keys,
    cuda::std::greater<>{},
    4,
    28,
    coop::radix_sort_policy<5, false, coop::scan_algorithm::raking, coop::radix_smem_config::eight_byte>{});
  auto radix_tile_controls                            = coop::radix_sort_keys(unsigned_keys, tile_k, 32);
  auto radix_mixed_controls                           = coop::radix_sort_keys(unsigned_keys, 4, tile_k);
  auto radix_converted_mixed_controls                 = coop::radix_sort_keys(unsigned_keys, tile_k, short_control);
  auto [radix_pair_keys, radix_pair_values]           = coop::radix_sort_pairs(unsigned_keys, unsigned_values);
  auto [radix_desc_pair_keys, radix_desc_pair_values] = coop::radix_sort_pairs(
    unsigned_keys, unsigned_values, cuda::std::greater<unsigned int>{}, tile_k, 28, coop::radix_sort_policy<4>{});
  auto radix_mixed_pairs        = coop::radix_sort_pairs(unsigned_keys, unsigned_values, 4, tile_k);
  auto max_keys                 = coop::topk_max_keys(keys, runtime_k);
  auto unsigned_top_keys        = coop::topk_max_keys(unsigned_keys, runtime_k);
  auto min_keys                 = coop::topk_min_keys(keys, runtime_k, 48, 0, 32, coop::topk_policy<>{});
  auto full_bit_range           = coop::topk_max_keys(keys, runtime_k, 4, 28);
  auto [pair_keys, pair_values] = coop::topk_max_pairs(keys, values, runtime_k);
  auto [unsigned_pair_keys, unsigned_pair_values] = coop::topk_max_pairs(unsigned_keys, unsigned_values, runtime_k);
  auto full_bit_range_pairs                       = coop::topk_max_pairs(keys, values, runtime_k, 4, 28);
  static_assert(std::is_same_v<decltype(summed), ct::tile<int, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(reduced_multiply), ct::tile<int, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(unsigned_pair_values), ct::tile<unsigned long long, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(histogram_128), ct::tile<int, ct::shape<128>>>);
  static_assert(std::is_same_v<decltype(fixed_histogram), ct::tile<int, ct::shape<256>>>);
  static_assert(std::is_same_v<decltype(decoded_default), ct::tile<int, ct::shape<128>>>);
  static_assert(std::is_same_v<decltype(decoded_scalar_offset), ct::tile<unsigned int, ct::shape<128>>>);
  static_assert(std::is_same_v<decltype(decoded_with_offsets), ct::tile<int, ct::shape<128>>>);
  static_assert(std::is_same_v<decltype(relative_offsets), ct::tile<unsigned int, ct::shape<128>>>);
  static_assert(std::is_same_v<decltype(relative_offsets_u64), ct::tile<unsigned long long, ct::shape<128>>>);
  static_assert(std::is_same_v<decltype(heads), ct::tile<int, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(pair_tails), ct::tile<int, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(adjacent_left), ct::tile<int, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(shuffled_rotate_tile), ct::tile<int, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(exchange_s2b), ct::tile<unsigned int, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(descending_ranks), ct::tile<int, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(merge_keys_partial), ct::tile<int, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(merge_pair_values), ct::tile<long long, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(radix_descending), ct::tile<unsigned int, ct::shape<64>>>);
  static_assert(std::is_same_v<decltype(radix_pair_values), ct::tile<unsigned long long, ct::shape<64>>>);
  (void) exclusive;
  (void) inclusive;
  (void) multiplied;
  (void) tile_initial_scan;
  (void) default_scalar_initial;
  (void) default_tile_initial;
  (void) float_initial_scan;
  (void) generic;
  (void) generic_inclusive_multiply;
  (void) generic_exclusive_plus;
  (void) generic_exclusive_initial;
  (void) generic_exclusive_tile_initial;
  (void) unsigned_scan;
  (void) summed;
  (void) half_sum;
  (void) bfloat_sum;
  (void) unsigned_sum;
  (void) long_sum;
  (void) unsigned_long_sum;
  (void) reduced_plus;
  (void) reduced_multiply;
  (void) reduced_minimum;
  (void) reduced_maximum;
  (void) reduced_bits;
  (void) unsigned_bits;
  (void) histogram_default;
  (void) histogram_128;
  (void) fixed_histogram;
  (void) decoded_default;
  (void) decoded_ulong;
  (void) decoded_policy;
  (void) decoded_scalar_offset;
  (void) decoded_tile_offset;
  (void) decoded_with_offsets;
  (void) relative_offsets;
  (void) decoded_u64;
  (void) relative_offsets_u64;
  (void) heads;
  (void) tails;
  (void) pair_heads;
  (void) pair_tails;
  (void) adjacent_left;
  (void) adjacent_right;
  (void) shuffled_up;
  (void) shuffled_down;
  (void) shuffled_offset;
  (void) shuffled_offset_tile;
  (void) shuffled_rotate;
  (void) shuffled_rotate_tile;
  (void) exchange_s2b;
  (void) exchange_b2s;
  (void) explicit_s2b;
  (void) explicit_b2s;
  (void) ranks;
  (void) signed_ranks;
  (void) shifted_ranks;
  (void) bounded_ranks;
  (void) tile_begin_ranks;
  (void) mixed_bit_ranks;
  (void) converted_mixed_bit_ranks;
  (void) descending_ranks;
  (void) merge_keys;
  (void) merge_keys_less;
  (void) merge_keys_greater;
  (void) merge_keys_partial;
  (void) merge_keys_partial_greater;
  (void) merge_keys_tile_controls;
  (void) merge_keys_mixed_valid;
  (void) merge_keys_converted_mixed_valid;
  (void) merge_keys_mixed_oob;
  (void) merge_keys_tile_compare;
  (void) merge_keys_mixed_compare;
  (void) merge_pair_keys;
  (void) merge_pair_values;
  (void) merge_pair_desc_keys;
  (void) merge_pair_desc_values;
  (void) merge_pairs_partial;
  (void) merge_pairs_partial_compare;
  (void) merge_pairs_tile_controls;
  (void) merge_pairs_mixed_valid;
  (void) merge_pairs_mixed_oob;
  (void) merge_pairs_tile_compare;
  (void) merge_pairs_mixed_compare;
  (void) radix_keys;
  (void) radix_descending;
  (void) radix_tile_controls;
  (void) radix_mixed_controls;
  (void) radix_converted_mixed_controls;
  (void) radix_pair_keys;
  (void) radix_pair_values;
  (void) radix_desc_pair_keys;
  (void) radix_desc_pair_values;
  (void) radix_mixed_pairs;
  (void) max_keys;
  (void) unsigned_top_keys;
  (void) min_keys;
  (void) full_bit_range;
  (void) pair_keys;
  (void) pair_values;
  (void) unsigned_pair_keys;
  (void) unsigned_pair_values;
  (void) full_bit_range_pairs;
}
