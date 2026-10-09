// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items
//! from sequences of unordered data items residing within device-accessible memory.

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
#include <cub/detail/cc_dispatch.cuh>
#include <cub/detail/logging.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/tuning/tuning_topk.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::topk
{
// Used in the bin ID calculation to exclude bits unrelated to the current pass
template <typename T, int BitsPerPass>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr unsigned calc_mask(const int pass)
{
  const int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
  return (1 << num_bits) - 1;
}

// Bit-ordering used by the top-k radix passes. Without `NormalizeMinusZero` this is `Traits<T>::TwiddleIn`. With it,
// floating-point keys use a variant that negates (rather than complements) keys with the sign bit set. `TwiddleIn`
// complements negative keys, which leaves -0.0 at `0x7FFF...`, one below +0.0's `0x8000...`. Negation is that
// complement plus one, so it shifts the whole negative half up by one and -0.0 lands exactly on +0.0 with no separate
// fix-up. Every sign-clear pattern is unchanged and every sign-set pattern moves by exactly +1, so relative order is
// preserved everywhere (-NaN stays below -inf, +NaN above +inf) and the only new equivalence is the two zeros.
// The spelling matters for codegen: this form compiles to the standard twiddle plus one integer add, whereas
// `TwiddleIn(bits) + sign_bit` does not fold and measured about one percent slower.
template <typename T, bool NormalizeMinusZero>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE typename Traits<T>::UnsignedBits
twiddle_in_bits(typename Traits<T>::UnsignedBits bits)
{
  using bits_t = typename Traits<T>::UnsignedBits;
  // Every FLOATING_POINT specialization of Traits asserts `is_floating_point_v<T>`, so this selects exactly the keys
  // whose `TwiddleIn` is the floating-point one.
  if constexpr (NormalizeMinusZero && ::cuda::is_floating_point_v<T>)
  {
    // 0 for sign-clear keys, ~0 for sign-set keys. Unsigned arithmetic only, so well-defined in every dialect.
    const bits_t neg_mask = static_cast<bits_t>(bits_t{0} - static_cast<bits_t>(bits >> (8 * sizeof(bits_t) - 1)));
    // `neg_mask | HIGH_BIT` is HIGH_BIT for sign-clear keys and ~0 for sign-set keys, which is exactly the mask that
    // `Traits<T>::TwiddleIn` xors with. So `twiddled` is the standard twiddle: top bit set for non-negative keys,
    // complement for negative keys.
    const bits_t twiddled = static_cast<bits_t>(bits ^ static_cast<bits_t>(neg_mask | Traits<T>::HIGH_BIT));
    // Subtracting ~0 adds one. Negative keys become `~bits + 1 == -bits`, non-negative keys are untouched. That +1 is
    // what moves -0.0 (`0x7FFF...` after the complement) onto +0.0 (`0x8000...`).
    return static_cast<bits_t>(twiddled - neg_mask);
  }
  else
  {
    return Traits<T>::TwiddleIn(bits);
  }
}

// Get the bin ID from the value of element
template <typename T,
          select SelectDirection,
          int BitsPerPass,
          typename DecomposerT,
          bool NormalizeMinusZero,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct extract_bin_op_t;

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT, bool NormalizeMinusZero>
struct extract_bin_op_t<T, SelectDirection, BitsPerPass, DecomposerT, NormalizeMinusZero, true>
{
  static constexpr bool is_descending = SelectDirection != select::min;
  using bit_ordered_type              = typename Traits<T>::UnsignedBits;

  int pass{};
  int start_bit{};
  unsigned mask{};

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE extract_bin_op_t(int pass, int /*total_bits*/, DecomposerT /*decomposer*/)
      : pass(pass)
      , start_bit(calc_start_bit<T, BitsPerPass>(pass))
      , mask(calc_mask<T, BitsPerPass>(pass))
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    auto bits = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    bits      = twiddle_in_bits<T, NormalizeMinusZero>(bits);
    if constexpr (SelectDirection != select::min)
    {
      bits = ~bits;
    }
    const int bucket = (bits >> start_bit) & mask;
    return bucket;
  }
};

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT, bool NormalizeMinusZero>
struct extract_bin_op_t<T, SelectDirection, BitsPerPass, DecomposerT, NormalizeMinusZero, false>
{
  static constexpr bool is_descending = SelectDirection != select::min;
  using radix_traits_t                = detail::radix::traits_t<T>;
  using bit_ordered_type              = typename radix_traits_t::bit_ordered_type;
  using digit_extractor_t = typename radix_traits_t::template digit_extractor_t<ShiftDigitExtractor<T>, DecomposerT>;

  DecomposerT decomposer{};
  digit_extractor_t digit_extractor;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE extract_bin_op_t(int pass, int total_bits, DecomposerT decomposer)
      : decomposer(decomposer)
      , digit_extractor(radix_traits_t::template digit_extractor<ShiftDigitExtractor<T>>(
          calc_start_bit<BitsPerPass>(total_bits, pass),
          calc_start_bit<BitsPerPass>(total_bits, pass - 1) - calc_start_bit<BitsPerPass>(total_bits, pass),
          decomposer))
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    bit_ordered_type ordered = key;
    ordered                  = RadixSortTwiddle<is_descending, T>::In(ordered, decomposer);
    return static_cast<int>(digit_extractor.Digit(ordered));
  }
};

// Check if the input element is still a candidate for the target pass.
template <typename T,
          select SelectDirection,
          int BitsPerPass,
          typename DecomposerT,
          bool NormalizeMinusZero,
          bool CanTwiddle = detail::radix::can_twiddle<T>>
struct identify_candidates_op_t;

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT, bool NormalizeMinusZero>
struct identify_candidates_op_t<T, SelectDirection, BitsPerPass, DecomposerT, NormalizeMinusZero, true>
{
  using unsigned_bits_t = typename Traits<T>::UnsignedBits;
  using key_prefix_t    = key_prefix_storage_t<T>;
  unsigned_bits_t* kth_key_bits;
  int start_bit;
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  identify_candidates_op_t(key_prefix_t* kth_key_bits, int pass, int /*total_bits*/, DecomposerT /*decomposer*/)
      : kth_key_bits(&kth_key_bits->bits)
      , start_bit(calc_start_bit<T, BitsPerPass>(pass - 1))
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE candidate_class operator()(T key) const
  {
    auto bits = reinterpret_cast<unsigned_bits_t&>(key);
    bits      = twiddle_in_bits<T, NormalizeMinusZero>(bits); // must match extract_bin_op_t, which built kth_key_bits

    if constexpr (SelectDirection != select::min)
    {
      bits = ~bits;
    }

    bits = (bits >> start_bit) << start_bit;

    return (bits < *kth_key_bits) ? candidate_class::selected
         : (bits == *kth_key_bits)
           ? candidate_class::candidate
           : candidate_class::rejected;
  }
};

template <typename T, select SelectDirection, int BitsPerPass, typename DecomposerT, bool NormalizeMinusZero>
struct identify_candidates_op_t<T, SelectDirection, BitsPerPass, DecomposerT, NormalizeMinusZero, false>
{
  static constexpr bool is_descending = SelectDirection != select::min;
  using radix_traits_t                = detail::radix::traits_t<T>;
  using bit_ordered_type              = typename radix_traits_t::bit_ordered_type;
  using key_prefix_t                  = key_prefix_storage_t<T>;

  key_prefix_t* kth_key_bits{};
  int pass{};
  int total_bits{};
  DecomposerT decomposer{};

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE
  identify_candidates_op_t(key_prefix_t* kth_key_bits, int pass, int total_bits, DecomposerT decomposer)
      : kth_key_bits(kth_key_bits)
      , pass(pass)
      , total_bits(total_bits)
      , decomposer(decomposer)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE candidate_class operator()(T key) const
  {
    if (pass <= 0)
    {
      return candidate_class::candidate;
    }

    bit_ordered_type ordered = key;
    ordered                  = RadixSortTwiddle<is_descending, T>::In(ordered, decomposer);

    // Build the key's prefix using the same funnel shift as set_kth_key_bits
    key_prefix_t key_prefix{};
    for (int prefix_pass = 0; prefix_pass < pass; ++prefix_pass)
    {
      const int start_bit = calc_start_bit<BitsPerPass>(total_bits, prefix_pass);
      const int num_bits =
        calc_start_bit<BitsPerPass>(total_bits, prefix_pass - 1) - calc_start_bit<BitsPerPass>(total_bits, prefix_pass);
      auto extractor =
        radix_traits_t::template digit_extractor<ShiftDigitExtractor<T>>(start_bit, num_bits, decomposer);
      key_prefix.shift_or(BitsPerPass, static_cast<unsigned int>(extractor.Digit(ordered)));
    }

    // Compare word-by-word from MSB to LSB
    const int total_prefix_bits = pass * BitsPerPass;
    const int top_word_idx      = (total_prefix_bits - 1) / 32;
    const int bits_in_top_word  = ((total_prefix_bits - 1) % 32) + 1;

    // Top word may be partially filled
    {
      unsigned int key_w = key_prefix.words[top_word_idx];
      unsigned int kth_w = kth_key_bits->words[top_word_idx];
      if (bits_in_top_word < 32)
      {
        const unsigned int mask = (1u << bits_in_top_word) - 1u;
        key_w &= mask;
        kth_w &= mask;
      }
      if (key_w < kth_w)
      {
        return candidate_class::selected;
      }
      if (key_w > kth_w)
      {
        return candidate_class::rejected;
      }
    }

    // Remaining words are fully populated
    for (int w = top_word_idx - 1; w >= 0; --w)
    {
      if (key_prefix.words[w] < kth_key_bits->words[w])
      {
        return candidate_class::selected;
      }
      if (key_prefix.words[w] > kth_key_bits->words[w])
      {
        return candidate_class::rejected;
      }
    }

    return candidate_class::candidate;
  }
};

template <typename PolicySelector,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyInT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceTopKKernel(
    const KeyInputIteratorT d_keys_in,
    const KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    const ValueOutputIteratorT d_values_out,
    KeyInT* const in_buf,
    OffsetT* const in_idx_buf,
    KeyInT* const out_buf,
    OffsetT* const out_idx_buf,
    Counter<it_value_t<KeyInputIteratorT>, OffsetT, OutOffsetT>* counter,
    OffsetT* const histogram,
    const OffsetT num_items,
    const OutOffsetT k,
    const OffsetT buffer_length,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op,
    const int pass,
    const bool is_last_pass)
{
  static constexpr topk_policy policy = current_policy<PolicySelector>();
  using agent_topk_policy_t =
    agent_topk_policy<policy.threads_per_block,
                      policy.items_per_thread,
                      policy.bits_per_pass,
                      policy.load_algorithm,
                      policy.scan_algorithm>;
  using agent_topk_t =
    AgentTopK<agent_topk_policy_t,
              KeyInputIteratorT,
              KeyOutputIteratorT,
              ValueInputIteratorT,
              ValueOutputIteratorT,
              ExtractBinOpT,
              IdentifyCandidatesOpT,
              OffsetT,
              OutOffsetT>;

  __shared__ typename agent_topk_t::TempStorage temp_storage;
  agent_topk_t(
    temp_storage,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    num_items,
    k,
    buffer_length,
    extract_bin_op,
    identify_candidates_op)
    .invoke_filter_and_histogram(in_buf, in_idx_buf, out_buf, out_idx_buf, counter, histogram, pass, is_last_pass);
}

template <typename PolicySelector,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyInT,
          typename ExtractBinOpT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceTopKHistogramKernel(
    const KeyInputIteratorT d_keys_in,
    const KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    const ValueOutputIteratorT d_values_out,
    Counter<it_value_t<KeyInputIteratorT>, OffsetT, OutOffsetT>* counter,
    OffsetT* const histogram,
    const OffsetT num_items,
    const OutOffsetT k,
    const OffsetT buffer_length,
    ExtractBinOpT extract_bin_op,
    const int pass,
    const bool is_last_pass)
{
  static constexpr topk_policy policy = current_policy<PolicySelector>();
  using agent_topk_policy_t =
    agent_topk_policy<policy.threads_per_block,
                      policy.items_per_thread,
                      policy.bits_per_pass,
                      policy.load_algorithm,
                      policy.scan_algorithm>;
  using identify_candidates_op_t = NullType;
  using agent_topk_t =
    AgentTopK<agent_topk_policy_t,
              KeyInputIteratorT,
              KeyOutputIteratorT,
              ValueInputIteratorT,
              ValueOutputIteratorT,
              ExtractBinOpT,
              identify_candidates_op_t,
              OffsetT,
              OutOffsetT>;

  __shared__ typename agent_topk_t::TempStorage temp_storage;
  agent_topk_t(
    temp_storage,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    num_items,
    k,
    buffer_length,
    extract_bin_op,
    identify_candidates_op_t{})
    .invoke_histogram_only(counter, histogram, pass, is_last_pass);
}

template <typename PolicySelector,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename OffsetT,
          typename OutOffsetT,
          typename KeyInT,
          typename IdentifyCandidatesOpT>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(int(current_policy<PolicySelector>().threads_per_block))
  _CCCL_KERNEL_ATTRIBUTES void DeviceTopKLastFilterKernel(
    const KeyInputIteratorT d_keys_in,
    const KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    const ValueOutputIteratorT d_values_out,
    KeyInT* const in_buf,
    OffsetT* const in_idx_buf,
    Counter<it_value_t<KeyInputIteratorT>, OffsetT, OutOffsetT>* counter,
    const OffsetT num_items,
    const OutOffsetT k,
    const OffsetT buffer_length,
    IdentifyCandidatesOpT identify_candidates_op,
    const int pass)
{
  static constexpr topk_policy policy = current_policy<PolicySelector>();
  using agent_topk_policy_t =
    agent_topk_policy<policy.threads_per_block,
                      policy.items_per_thread,
                      policy.bits_per_pass,
                      policy.load_algorithm,
                      policy.scan_algorithm>;
  using extract_bin_op_t = NullType;
  using agent_topk_t =
    AgentTopK<agent_topk_policy_t,
              KeyInputIteratorT,
              KeyOutputIteratorT,
              ValueInputIteratorT,
              ValueOutputIteratorT,
              extract_bin_op_t, // ExtractBinOp operator (not used)
              IdentifyCandidatesOpT,
              OffsetT,
              OutOffsetT>;

  __shared__ typename agent_topk_t::TempStorage temp_storage;
  agent_topk_t(
    temp_storage,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    num_items,
    k,
    buffer_length,
    extract_bin_op_t{},
    identify_candidates_op)
    .invoke_last_filter(in_buf, in_idx_buf, counter, k, pass);
}

//! @tparam SelectDirection
//!   Determines whether to select the smallest or largest K elements.
//!
//! @tparam KeyInputIteratorT
//!   **[inferred]** Random-access input iterator type for reading input keys @iterator
//!
//! @tparam KeyOutputIteratorT
//!   **[inferred]** Random-access output iterator type for writing output keys @iterator
//!
//! @tparam ValueInputIteratorT
//!   **[inferred]** Random-access input iterator type for reading input values @iterator
//!
//! @tparam ValueOutputIteratorT
//!   **[inferred]** Random-access input iterator type for writing output values @iterator
//!
//! @tparam OffsetT
//!  Data Type for variables: num_items
//!
//! @tparam OutOffsetT
//!  Data Type for variables: k
//!
//! @tparam DecomposerT
//!   Implementation detail, do not specify directly, requirements on the content of this type are subject to breaking
//!   change.
template <
  select SelectDirection,
  typename KeyInputIteratorT,
  typename KeyOutputIteratorT,
  typename ValueInputIteratorT,
  typename ValueOutputIteratorT,
  typename OffsetT,
  typename OutOffsetT,
  typename DecomposerT = detail::identity_decomposer_t,
  typename PolicySelector =
    policy_selector_from_types<it_value_t<KeyInputIteratorT>, it_value_t<ValueInputIteratorT>, OffsetT, OutOffsetT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  const KeyInputIteratorT d_keys_in,
  KeyOutputIteratorT d_keys_out,
  const ValueInputIteratorT d_values_in,
  ValueOutputIteratorT d_values_out,
  OffsetT num_items,
  OutOffsetT k,
  DecomposerT decomposer,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelLauncherFactory launcher_factory = {})
{
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  return dispatch_compute_cap(policy_selector, cc, [&](auto policy_getter) {
    static constexpr topk_policy active_policy = policy_getter();
    detail::log_dispatch("DeviceTopK", cc, active_policy);
    using key_in_t                  = it_value_t<KeyInputIteratorT>;
    using value_in_t                = it_value_t<ValueInputIteratorT>;
    static constexpr bool keys_only = ::cuda::std::is_same_v<value_in_t, NullType>;

    // atomicAdd does not implement overloads for all integer types, so we limit OffsetT to uint32_t or unsigned long
    // long
    static_assert(
      ::cuda::std::is_same_v<OffsetT, ::cuda::std::uint32_t> || ::cuda::std::is_same_v<OffsetT, unsigned long long>,
      "The top-k algorithm is limited to unsigned offset types retrieved from choose_offset_t<T>.");

    // atomicAdd does not implement overloads for all integer types, so we limit OffsetT to uint32_t or unsigned long
    // long
    static_assert(::cuda::std::is_same_v<OutOffsetT, ::cuda::std::uint32_t>
                    || ::cuda::std::is_same_v<OutOffsetT, unsigned long long>,
                  "The top-k algorithm is limited to unsigned offset types retrieved from choose_offset_t<T>.");

    // TODO (elstehle): consider making this part of the env-based API
    // The algorithm allocates a double-buffer for intermediate results of size
    // num_items/coefficient_for_candidate_buffer
    static constexpr OffsetT coefficient_for_candidate_buffer = 128;
    constexpr int threads_per_block                           = active_policy.threads_per_block;
    constexpr int items_per_thread                            = active_policy.items_per_thread;
    constexpr int bits_per_pass                               = active_policy.bits_per_pass;
    constexpr int tile_size                                   = threads_per_block * items_per_thread;
    const auto num_tiles      = static_cast<unsigned int>(::cuda::ceil_div(num_items, tile_size));
    const int total_bits      = detail::radix::traits_t<key_in_t>::default_end_bit(decomposer);
    const int num_passes      = calc_num_passes<bits_per_pass>(total_bits);
    constexpr int num_buckets = 1 << bits_per_pass;

    // Define operators
    // The trailing `false` skips the -0.0 normalization: DeviceTopK has no index tie-break, so any set of tied keys is
    // valid and -0.0 may rank below +0.0
    using identify_candidates_op =
      identify_candidates_op_t<key_in_t, SelectDirection, bits_per_pass, DecomposerT, false>;
    using extract_bin_op = extract_bin_op_t<key_in_t, SelectDirection, bits_per_pass, DecomposerT, false>;

    // We are capping k at a maximum of num_items
    using common_offset_t = ::cuda::std::common_type_t<OffsetT, OutOffsetT>;
    k = static_cast<OutOffsetT>((::cuda::std::min) (common_offset_t{k}, static_cast<common_offset_t>(num_items)));

    // Specify temporary storage allocation requirements
    using counter_t             = Counter<key_in_t, OffsetT, OutOffsetT>;
    const size_t size_counter   = sizeof(counter_t);
    const size_t size_histogram = num_buckets * sizeof(OffsetT);
    const OffsetT candidate_buffer_length =
      (::cuda::std::max) (OffsetT{1}, num_items / coefficient_for_candidate_buffer);

    constexpr int allocations_array_size = keys_only ? 4 : 6;
    // NOLINTNEXTLINE(misc-const-correctness)
    size_t allocation_sizes[allocations_array_size] = {
      size_counter,
      size_histogram,
      candidate_buffer_length * sizeof(key_in_t),
      candidate_buffer_length * sizeof(key_in_t)};
    if constexpr (!keys_only)
    {
      allocation_sizes[4] = candidate_buffer_length * sizeof(OffsetT);
      allocation_sizes[5] = candidate_buffer_length * sizeof(OffsetT);
    }

    // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
    void* allocations[allocations_array_size] = {};
    if (const auto error =
          CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
    {
      return error;
    }

    if (d_temp_storage == nullptr)
    {
      // Return if the caller is simply requesting the size of the storage allocation
      return cudaSuccess;
    }

    // Init the buffer for descriptor and histogram
    if (const auto error = CubDebug(launcher_factory.MemsetAsync(
          allocations[0], 0, static_cast<char*>(allocations[2]) - static_cast<char*>(allocations[0]), stream)))
    {
      return error;
    }

    // Get grid size for scanning tiles
    int num_sms = 0;
    if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(num_sms)))
    {
      return error;
    }

    auto topk_kernel =
      DeviceTopKKernel<PolicySelector,
                       KeyInputIteratorT,
                       KeyOutputIteratorT,
                       ValueInputIteratorT,
                       ValueOutputIteratorT,
                       OffsetT,
                       OutOffsetT,
                       key_in_t,
                       extract_bin_op,
                       identify_candidates_op>;

    int main_kernel_blocks_per_sm = 0;
    if (const auto error =
          CubDebug(launcher_factory.MaxSmOccupancy(main_kernel_blocks_per_sm, topk_kernel, threads_per_block)))
    {
      return error;
    }
    const auto main_kernel_max_occupancy = static_cast<unsigned int>(main_kernel_blocks_per_sm * num_sms);
    const auto topk_grid_size            = (::cuda::std::min) (main_kernel_max_occupancy, num_tiles);

    _CUB_LOG_KERNEL_LAUNCH(
      "topk_kernel", topk_grid_size, 1, 1, threads_per_block, 0, stream, ", SM occupancy: %d", main_kernel_blocks_per_sm);

    // Initialize address variables
    counter_t* counter = static_cast<counter_t*>(allocations[0]);
    OffsetT* histogram = static_cast<decltype(histogram)>(allocations[1]); // NOLINT(misc-const-correctness)

    // Pass 0: dedicated histogram-only kernel over the full input
    {
      auto histogram_kernel = DeviceTopKHistogramKernel<
        PolicySelector,
        KeyInputIteratorT,
        KeyOutputIteratorT,
        ValueInputIteratorT,
        ValueOutputIteratorT,
        OffsetT,
        OutOffsetT,
        key_in_t,
        extract_bin_op>;

      int histogram_kernel_blocks_per_sm = 0;
      if (const auto error = CubDebug(
            launcher_factory.MaxSmOccupancy(histogram_kernel_blocks_per_sm, histogram_kernel, threads_per_block)))
      {
        return error;
      }
      const auto histogram_kernel_max_occupancy = static_cast<unsigned int>(histogram_kernel_blocks_per_sm * num_sms);
      const auto histogram_grid_size            = (::cuda::std::min) (histogram_kernel_max_occupancy, num_tiles);

      const extract_bin_op extract_op(0, total_bits, decomposer);
      if (const auto error = CubDebug(
            launcher_factory(histogram_grid_size, threads_per_block, 0, stream)
              .doit(histogram_kernel,
                    d_keys_in,
                    d_keys_out,
                    d_values_in,
                    d_values_out,
                    counter,
                    histogram,
                    num_items,
                    k,
                    candidate_buffer_length,
                    extract_op,
                    0,
                    num_passes == 1)))
      {
        return error;
      }
    }

    // Passes 1..num_passes-1: fused filter + histogram kernel
    // Current() = input buffer (read), Alternate() = output buffer (write)
    DoubleBuffer<key_in_t> key_bufs(static_cast<key_in_t*>(allocations[3]), static_cast<key_in_t*>(allocations[2]));
    DoubleBuffer<OffsetT> idx_bufs;
    if constexpr (!keys_only)
    {
      idx_bufs = DoubleBuffer<OffsetT>(static_cast<OffsetT*>(allocations[5]), static_cast<OffsetT*>(allocations[4]));
    }

    int pass = 1;
    for (; pass < num_passes; pass++)
    {
      const extract_bin_op extract_op(pass, total_bits, decomposer);
      const identify_candidates_op identify_op(&counter->kth_key_bits, pass, total_bits, decomposer);

      if (const auto error = CubDebug(
            launcher_factory(topk_grid_size, threads_per_block, 0, stream)
              .doit(topk_kernel,
                    d_keys_in,
                    d_keys_out,
                    d_values_in,
                    d_values_out,
                    key_bufs.Current(),
                    idx_bufs.Current(),
                    key_bufs.Alternate(),
                    idx_bufs.Alternate(),
                    counter,
                    histogram,
                    num_items,
                    k,
                    candidate_buffer_length,
                    extract_op,
                    identify_op,
                    pass,
                    pass == num_passes - 1)))
      {
        return error;
      }

      key_bufs.selector ^= 1;
      if constexpr (!keys_only)
      {
        idx_bufs.selector ^= 1;
      }
    }

    auto topk_last_filter_kernel = DeviceTopKLastFilterKernel<
      PolicySelector,
      KeyInputIteratorT,
      KeyOutputIteratorT,
      ValueInputIteratorT,
      ValueOutputIteratorT,
      OffsetT,
      OutOffsetT,
      key_in_t,
      identify_candidates_op>;

    const identify_candidates_op identify_op(&counter->kth_key_bits, pass, total_bits, decomposer);
    int last_filter_kernel_blocks_per_sm = 0;
    if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
          last_filter_kernel_blocks_per_sm, topk_last_filter_kernel, threads_per_block)))
    {
      return error;
    }
    const auto last_filter_kernel_max_occupancy = static_cast<unsigned int>(last_filter_kernel_blocks_per_sm * num_sms);
    const auto last_filter_grid_size            = (::cuda::std::min) (last_filter_kernel_max_occupancy, num_tiles);
    if (const auto error = CubDebug(
          launcher_factory(last_filter_grid_size, threads_per_block, 0, stream)
            .doit(topk_last_filter_kernel,
                  d_keys_in,
                  d_keys_out,
                  d_values_in,
                  d_values_out,
                  key_bufs.Current(),
                  idx_bufs.Current(),
                  counter,
                  num_items,
                  k,
                  candidate_buffer_length,
                  identify_op,
                  pass)))
    {
      return error;
    }

    return cudaSuccess;
  });
}
} // namespace detail::topk

CUB_NAMESPACE_END
