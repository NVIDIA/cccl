// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__cmath/round_up.h>
#include <cuda/__device/arch_id.h>
#include <cuda/__functional/address_stability.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/array>
#include <cuda/std/tuple>

#if _CCCL_HAS_CONCEPTS()
#  include <cuda/std/concepts>
#endif // _CCCL_HAS_CONCEPTS()

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN
namespace detail::transform
{
struct always_true_predicate
{
  template <typename... Ts>
  _CCCL_HOST_DEVICE constexpr bool operator()(Ts&&...) const
  {
    return true;
  }
};
} // namespace detail::transform
CUB_NAMESPACE_END

template <>
struct ::cuda::proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::transform::always_true_predicate>
    : ::cuda::std::true_type
{};

CUB_NAMESPACE_BEGIN
namespace detail::transform
{
enum class Algorithm
{
  // We previously had a fallback algorithm that would use cub::DeviceFor. Benchmarks showed that the prefetch algorithm
  // is always superior to that fallback, so it was removed.
  prefetch,
  vectorized,
  memcpy_async,
  ublkcp
};

#if !_CCCL_COMPILER(NVRTC)
inline ::std::ostream& operator<<(::std::ostream& os, const Algorithm& algorithm)
{
  switch (algorithm)
  {
    case Algorithm::prefetch:
      return os << "Algorithm::prefetch";
    case Algorithm::vectorized:
      return os << "Algorithm::vectorized";
    case Algorithm::memcpy_async:
      return os << "Algorithm::memcpy_async";
    case Algorithm::ublkcp:
      return os << "Algorithm::ublkcp";
    default:
      return os << "Algorithm::<unknown>";
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

struct prefetch_policy
{
  int block_threads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  int items_per_thread_no_input = 2; // when there are no input iterators, the kernel is just filling
  int min_items_per_thread      = 1;
  int max_items_per_thread      = 32;

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const prefetch_policy& lhs, const prefetch_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread_no_input == rhs.items_per_thread_no_input
        && lhs.min_items_per_thread == rhs.min_items_per_thread && lhs.max_items_per_thread == rhs.max_items_per_thread;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const prefetch_policy& lhs, const prefetch_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const prefetch_policy& policy)
  {
    return os << "prefetch_policy { .block_threads = " << policy.block_threads << ", .items_per_thread_no_input = "
              << policy.items_per_thread_no_input << ", .min_items_per_thread = " << policy.min_items_per_thread
              << ", .max_items_per_thread = " << policy.max_items_per_thread << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct vectorized_policy
{
  int block_threads;
  int items_per_thread;
  int vec_size;
  // if we have to fall back to prefetching, use these values:
  int prefetch_items_per_thread_no_input = 2;
  int prefetch_min_items_per_thread      = 1;
  int prefetch_max_items_per_thread      = 32;

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const vectorized_policy& lhs, const vectorized_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.vec_size == rhs.vec_size
        && lhs.prefetch_items_per_thread_no_input == rhs.prefetch_items_per_thread_no_input
        && lhs.prefetch_min_items_per_thread == rhs.prefetch_min_items_per_thread
        && lhs.prefetch_max_items_per_thread == rhs.prefetch_max_items_per_thread;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const vectorized_policy& lhs, const vectorized_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const vectorized_policy& policy)
  {
    return os
        << "vectorized_policy { .block_threads = " << policy.block_threads
        << ", .items_per_thread = " << policy.items_per_thread << ", .vec_size = " << policy.vec_size
        << ", .prefetch_items_per_thread_no_input = " << policy.prefetch_items_per_thread_no_input
        << ", .prefetch_min_items_per_thread = " << policy.prefetch_min_items_per_thread
        << ", .prefetch_max_items_per_thread = " << policy.prefetch_max_items_per_thread << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct async_copy_policy
{
  int block_threads;
  int bulk_copy_alignment; // TODO(bgruber): this should probably be removed from the tuning policy
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  int min_items_per_thread = 1;
  int max_items_per_thread = 32;

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const async_copy_policy& lhs, const async_copy_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.bulk_copy_alignment == rhs.bulk_copy_alignment
        && lhs.min_items_per_thread == rhs.min_items_per_thread && lhs.max_items_per_thread == rhs.max_items_per_thread;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const async_copy_policy& lhs, const async_copy_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const async_copy_policy& policy)
  {
    return os << "async_copy_policy { .block_threads = " << policy.block_threads << ", .bulk_copy_alignment = "
              << policy.bulk_copy_alignment << ", .min_items_per_thread = " << policy.min_items_per_thread
              << ", .max_items_per_thread = " << policy.max_items_per_thread << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct transform_policy
{
  int min_bytes_in_flight;
  Algorithm algorithm;
  prefetch_policy prefetch;
  vectorized_policy vectorized;
  async_copy_policy async_copy;

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const transform_policy& lhs, const transform_policy& rhs)
  {
    return lhs.min_bytes_in_flight == rhs.min_bytes_in_flight && lhs.algorithm == rhs.algorithm
        && lhs.prefetch == rhs.prefetch && lhs.vectorized == rhs.vectorized && lhs.async_copy == rhs.async_copy;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const transform_policy& lhs, const transform_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const transform_policy& policy)
  {
    return os << "transform_policy { .min_bytes_in_flight = " << policy.min_bytes_in_flight
              << ", .algorithm = " << policy.algorithm << ", .prefetch = " << policy.prefetch
              << ", .vectorized = " << policy.vectorized << ", .async_copy = " << policy.async_copy << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept transform_policy_selector = policy_selector<T, transform_policy>;
#endif // _CCCL_HAS_CONCEPTS()

template <typename... Its>
_CCCL_HOST_DEVICE constexpr auto loaded_bytes_per_iteration() -> int
{
  return (int{sizeof(it_value_t<Its>)} + ... + 0);
}

constexpr int ldgsts_size_and_align = 16;

template <int InputCount>
_CCCL_HOST_DEVICE constexpr auto memcpy_async_dyn_smem_for_tile_size(
  const ::cuda::std::array<iterator_info, InputCount>& inputs,
  int tile_size,
  int copy_alignment = ldgsts_size_and_align) -> int
{
  int smem_size = 0;
  for (const auto& input : inputs)
  {
    smem_size =
      static_cast<int>(::cuda::round_up(smem_size, ::cuda::std::max(input.value_type_alignment, copy_alignment)));
    // max head/tail padding is copy_alignment - sizeof(T) each
    const int max_bytes_to_copy =
      input.value_type_size * tile_size + ::cuda::std::max(copy_alignment - input.value_type_size, 0) * 2;
    smem_size += max_bytes_to_copy;
  };
  return smem_size;
}

constexpr int bulk_copy_size_multiple = 16;

_CCCL_HOST_DEVICE constexpr auto bulk_copy_alignment(::cuda::arch_id arch) -> int
{
  return arch < ::cuda::arch_id::sm_100 ? 128 : 16;
}

template <int InputCount>
_CCCL_HOST_DEVICE constexpr auto bulk_copy_dyn_smem_for_tile_size(
  const ::cuda::std::array<iterator_info, InputCount>& inputs, int tile_size, int bulk_copy_align) -> int
{
  // we rely on the tile_size being a multiple of alignments, so shifting offsets/pointers by it retains alignments
  _CCCL_ASSERT(tile_size % bulk_copy_align == 0, "");
  _CCCL_ASSERT(tile_size % bulk_copy_size_multiple == 0, "");

  int tile_padding = bulk_copy_align;
  for (const auto& input : inputs)
  {
    tile_padding = ::cuda::std::max(tile_padding, input.value_type_alignment);
  }

  int smem_size = tile_padding; // for the barrier and padding
  for (const auto& input : inputs)
  {
    smem_size += tile_padding + input.value_type_size * tile_size;
  }
  return smem_size;
}

[[nodiscard]] _CCCL_API constexpr int arch_to_min_bytes_in_flight(::cuda::arch_id arch)
{
  if (arch >= ::cuda::arch_id::sm_100)
  {
    return 64 * 1024; // B200
  }
  if (arch >= ::cuda::arch_id::sm_90)
  {
    return 48 * 1024; // 32 for H100, 48 for H200
  }
  if (arch >= ::cuda::arch_id::sm_80)
  {
    return 16 * 1024; // A100
  }
  return 12 * 1024; // V100 and below
}

[[nodiscard]] _CCCL_API constexpr auto tuned_vectorized_policy(::cuda::arch_id arch, int store_size, bool filling)
{
  if (filling)
  {
    // manually tuned fill on RTX 5090
    // TODO(bgruber): re-enable this later! It's disabled to avoid SASS changes in PR #6914
    // if (arch >= ::cuda::arch_id::sm_120)
    // {
    //   return vectorized_policy{256, 8, 4};
    // }
    // manually tuned fill on B200, same as H200
    if (arch >= ::cuda::arch_id::sm_90)
    {
      return vectorized_policy{
        store_size > 4 ? 128 : 256, 16, ::cuda::std::max(8 / store_size, 1) /* 64-bit instructions */};
    }
    // manually tuned fill on A100
    if (arch >= ::cuda::arch_id::sm_80)
    {
      return vectorized_policy{256, 8, ::cuda::std::max(8 / store_size, 1) /* 64-bit instructions */};
    }
  }
  else
  {
    // manually tuned triad on A100
    if (arch == ::cuda::arch_id::sm_80)
    {
      return vectorized_policy{128, 16, 4};
    }
  }

  // defaults from fill on RTX 5090, but can be changed
  return vectorized_policy{256, 8, 4};
}

template <int InputCount>
struct policy_selector
{
  bool requires_stable_address;
  bool dense_output;
  ::cuda::std::array<iterator_info, InputCount> inputs;
  iterator_info output;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> transform_policy
  {
    const bool no_input_streams = InputCount == 0;

    bool all_inputs_contiguous                  = true;
    bool all_input_values_trivially_reloc       = true;
    bool can_memcpy_contiguous_inputs           = true;
    bool all_value_types_have_power_of_two_size = ::cuda::is_power_of_two(output.value_type_size);
    for (const auto& input : inputs)
    {
      all_inputs_contiguous &= input.is_contiguous;
      all_input_values_trivially_reloc &= input.value_type_is_trivially_relocatable;
      // the vectorized kernel supports mixing contiguous and non-contiguous iterators
      can_memcpy_contiguous_inputs &= !input.is_contiguous || input.value_type_is_trivially_relocatable;
      all_value_types_have_power_of_two_size &= ::cuda::is_power_of_two(input.value_type_size);
    }
    const bool can_memcpy_all_inputs = all_inputs_contiguous && all_input_values_trivially_reloc;
    const bool fallback_to_prefetch  = requires_stable_address || !can_memcpy_contiguous_inputs || !dense_output;
    const int min_bytes_in_flight    = arch_to_min_bytes_in_flight(arch);

    if (arch >= ::cuda::arch_id::sm_90) // handles sm_100 as well
    {
      const int async_block_size = arch < ::cuda::arch_id::sm_100 ? 256 : 128;
      const int alignment        = bulk_copy_alignment(arch);

      const auto prefetch = prefetch_policy{256};
      const auto vectorized =
        tuned_vectorized_policy(arch, ::cuda::std::max(1, output.value_type_size), no_input_streams);
      const auto async = async_copy_policy{async_block_size, alignment};

      // We cannot use the architecture-specific amount of SMEM here instead of max_smem_per_block, because this is not
      // forward compatible. If a user compiled for sm_xxx and we assume the available SMEM for that architecture, but
      // then runs on the next architecture after that, which may have a smaller available SMEM, we get a crash.
      const bool exhaust_smem =
        bulk_copy_dyn_smem_for_tile_size<InputCount>(inputs, async_block_size * async.min_items_per_thread, alignment)
        > int{max_smem_per_block};

      // if each tile size is a multiple of the bulk copy and maximum value type alignments, the alignment is retained
      // if the base pointer is sufficiently aligned (the correct check would be if it's a multiple of all value types
      // following the current tile). we would otherwise need to realign every SMEM tile individually, which is costly
      // and complex, so let's fall back in this case.
      int max_alignment = alignment;
      for (const auto& input : inputs)
      {
        max_alignment = ::cuda::std::max({max_alignment, input.value_type_alignment});
      }

      bool tile_sizes_retain_alignment = true;
      for (const auto& input : inputs)
      {
        tile_sizes_retain_alignment &= (input.value_type_size * async_block_size) % max_alignment == 0;
      }

      // on Hopper, the vectorized kernel performs better for 1 and 2 byte values, except for BabelStream mul (1 input)
      bool vector_kernel_is_faster = arch == ::cuda::arch_id::sm_90 && output.value_type_size < 4 && InputCount > 1;
      for (const auto& input : inputs)
      {
        vector_kernel_is_faster &= input.value_type_size < 4;
      }

      const bool enough_threads_for_peeling = async_block_size >= alignment; // head and tail bytes
      const bool fallback_to_vectorized = exhaust_smem || !tile_sizes_retain_alignment || !enough_threads_for_peeling
                                       || no_input_streams || !can_memcpy_all_inputs || vector_kernel_is_faster;

      const auto algorithm =
        fallback_to_prefetch ? Algorithm::prefetch
        : fallback_to_vectorized
          ? (all_value_types_have_power_of_two_size ? Algorithm::vectorized : Algorithm::prefetch)
          : Algorithm::ublkcp;

      return transform_policy{
        min_bytes_in_flight,
        algorithm,
        prefetch,
        vectorized,
        async,
      };
    }
    else if (arch >= ::cuda::arch_id::sm_80)
    {
      const int block_threads = 256;
      const auto prefetch     = prefetch_policy{block_threads};
      const auto vectorized =
        tuned_vectorized_policy(arch, ::cuda::std::max(1, output.value_type_size), no_input_streams);
      const auto async = async_copy_policy{block_threads, ldgsts_size_and_align};

      // We cannot use the architecture-specific amount of SMEM here instead of max_smem_per_block, because this is not
      // forward compatible. If a user compiled for sm_xxx and we assume the available SMEM for that architecture, but
      // then runs on the next architecture after that, which may have a smaller available SMEM, we get a crash.
      const bool exhaust_smem =
        memcpy_async_dyn_smem_for_tile_size<InputCount>(
          inputs, block_threads * async.min_items_per_thread, ldgsts_size_and_align)
        > int{max_smem_per_block};

      // on Ampere, the vectorized kernel performs better for 1 and 2 byte values
      bool use_vector_kernel_on_ampere = output.value_type_size < 4 && inputs.size() > 1;
      for (const auto& input : inputs)
      {
        use_vector_kernel_on_ampere &= input.value_type_size < 4;
      }
      const bool fallback_to_vectorized =
        exhaust_smem || no_input_streams || !can_memcpy_all_inputs || use_vector_kernel_on_ampere;

      const auto algorithm =
        fallback_to_prefetch ? Algorithm::prefetch
        : fallback_to_vectorized
          ? (all_value_types_have_power_of_two_size ? Algorithm::vectorized : Algorithm::prefetch)
          : Algorithm::memcpy_async;

      return transform_policy{
        min_bytes_in_flight,
        algorithm,
        prefetch,
        vectorized,
        async,
      };
    }

    // fallback
    return transform_policy{
      min_bytes_in_flight,
      (fallback_to_prefetch || !all_value_types_have_power_of_two_size) ? Algorithm::prefetch : Algorithm::vectorized,
      prefetch_policy{256},
      tuned_vectorized_policy(::cuda::arch_id::sm_60, ::cuda::std::max(1, output.value_type_size), no_input_streams),
      async_copy_policy{}, // never used
    };
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(transform_policy_selector<policy_selector<1>>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <bool RequiresStableAddress,
          bool DenseOutput,
          typename RandomAccessIteratorTupleIn,
          typename RandomAccessIteratorOut>
struct policy_selector_from_types
{
  static_assert(sizeof(RandomAccessIteratorTupleIn) == 0, "Second parameter must be a tuple");
};

template <bool RequiresStableAddress,
          bool DenseOutput,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut>
struct policy_selector_from_types<RequiresStableAddress,
                                  DenseOutput,
                                  ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                                  RandomAccessIteratorOut>
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> transform_policy
  {
    constexpr auto policies = policy_selector<sizeof...(RandomAccessIteratorsIn)>{
      RequiresStableAddress,
      DenseOutput,
      {make_iterator_info<RandomAccessIteratorsIn>()...},
      make_iterator_info<RandomAccessIteratorOut>()};
    return policies(arch);
  }
};
} // namespace detail::transform

CUB_NAMESPACE_END
