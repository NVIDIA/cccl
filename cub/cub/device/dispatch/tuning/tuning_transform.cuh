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

#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

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

struct prefetch_policy
{
  int block_threads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  int items_per_thread_no_input = 2; // when there are no input iterators, the kernel is just filling
  int min_items_per_thread      = 1;
  int max_items_per_thread      = 32;

  // TODO: operator==, !=, <<
};

struct vectorized_policy : prefetch_policy
{
  int items_per_thread_vectorized;
  int vec_size;

  // TODO: operator==, !=, <<
};

struct async_copy_policy
{
  int block_threads;
  int bulk_copy_alignment;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  int min_items_per_thread = 1;
  int max_items_per_thread = 32;

  // TODO: operator==, !=, <<
};

struct transform_arch_policy
{
  int min_bif;
  Algorithm algorithm;
  prefetch_policy prefetch_policy;
  vectorized_policy vectorized_policy;
  async_copy_policy async_copy_policy;

  // TODO: operator==, !=, <<
};

#if _CCCL_HAS_CONCEPTS()
_CCCL_API consteval void __needs_a_constexpr_value(auto) {}

// TODO(bgruber): bikeshed name before we make the tuning API public
template <typename T>
concept transform_policy_hub = requires(T hub, ::cuda::arch_id arch) {
  { hub(arch) } -> _CCCL_CONCEPT_VSTD::same_as<transform_arch_policy>;
  { __needs_a_constexpr_value(hub(arch)) };
};
#endif // _CCCL_HAS_CONCEPTS()

struct iterator_info
{
  int value_type_size;
  int value_type_alignment;
  bool value_type_is_trivially_relocatable;
  bool is_contiguous;
};

template <typename T>
inline constexpr size_t size_of = sizeof(T);

template <>
inline constexpr size_t size_of<void> = 0;

template <typename T>
inline constexpr size_t align_of = alignof(T);

template <>
inline constexpr size_t align_of<void> = 0;

template <typename It>
_CCCL_API constexpr auto make_iterator_info() -> iterator_info
{
  using vt = it_value_t<It>;
  return iterator_info{
    static_cast<int>(size_of<vt>),
    static_cast<int>(align_of<vt>),
    THRUST_NS_QUALIFIER::is_trivially_relocatable_v<vt>,
    THRUST_NS_QUALIFIER::is_contiguous_iterator_v<It>};
}

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

_CCCL_HOST_DEVICE constexpr int arch_to_min_bytes_in_flight(::cuda::arch_id arch)
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

[[nodiscard]] _CCCL_API constexpr auto vectorized_policy_for_filling(::cuda::arch_id arch, int store_size)
{
  // manually tuned fill on RTX 5090
  if (arch >= ::cuda::arch_id::sm_120)
  {
    return vectorized_policy{{256}, 8, 4};
  }
  // manually tuned fill on B200, same as H200
  if (arch >= ::cuda::arch_id::sm_90)
  {
    return vectorized_policy{
      {store_size > 4 ? 128 : 256}, 16, ::cuda::std::max(8 / store_size, 1) /* 64-bit instructions */};
  }
  // manually tuned fill on A100
  if (arch >= ::cuda::arch_id::sm_90)
  {
    return vectorized_policy{{256}, 8, ::cuda::std::max(8 / store_size, 1) /* 64-bit instructions */};
  }
  // defaults from fill on RTX 5090, but can be changed
  return vectorized_policy{{256}, 8, 4};
}

template <int InputCount>
struct arch_policies
{
  bool requires_stable_address;
  bool dense_output;
  ::cuda::std::array<iterator_info, InputCount> inputs;
  iterator_info output;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> transform_arch_policy
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
    const bool fallback_to_prefetch  = requires_stable_address || !can_memcpy_contiguous_inputs
                                   || !all_value_types_have_power_of_two_size || !dense_output;
    const int min_bif = arch_to_min_bytes_in_flight(arch);

    if (arch >= ::cuda::arch_id::sm_90) // handles sm_100 as well
    {
      const int async_block_size = arch < ::cuda::arch_id::sm_100 ? 256 : 128;
      const int alignment        = bulk_copy_alignment(arch);

      const auto prefetch   = prefetch_policy{256};
      const auto vectorized = vectorized_policy_for_filling(arch, output.value_type_size);
      const auto async      = async_copy_policy{async_block_size, alignment};

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
          ? Algorithm::vectorized
          : Algorithm::ublkcp;

      return transform_arch_policy{
        min_bif,
        algorithm,
        prefetch,
        vectorized,
        async,
      };
    }

    if (arch >= ::cuda::arch_id::sm_80)
    {
      const int block_threads = 256;
      const auto prefetch     = prefetch_policy{block_threads};
      const auto vectorized   = vectorized_policy_for_filling(arch, output.value_type_size);
      const auto async        = async_copy_policy{block_threads, ldgsts_size_and_align};

      // We cannot use the architecture-specific amount of SMEM here instead of max_smem_per_block, because this is not
      // forward compatible. If a user compiled for sm_xxx and we assume the available SMEM for that architecture, but
      // then runs on the next architecture after that, which may have a smaller available SMEM, we get a crash.
      const bool exhaust_smem =
        memcpy_async_dyn_smem_for_tile_size<InputCount>(
          inputs, block_threads * async.min_items_per_thread, ldgsts_size_and_align)
        > int{max_smem_per_block};
      const bool fallback_to_vectorized = exhaust_smem || no_input_streams || !can_memcpy_all_inputs;

      const auto algorithm =
        fallback_to_prefetch ? Algorithm::prefetch
        : fallback_to_vectorized
          ? Algorithm::vectorized
          : Algorithm::memcpy_async;

      return transform_arch_policy{
        min_bif,
        algorithm,
        prefetch,
        vectorized,
        async,
      };
    }

    // fallback
    return transform_arch_policy{
      min_bif,
      fallback_to_prefetch ? Algorithm::prefetch : Algorithm::vectorized,
      prefetch_policy{256},
      vectorized_policy_for_filling(::cuda::arch_id::sm_60, output.value_type_size),
      async_copy_policy{}, // never used
    };
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(transform_policy_hub<arch_policies>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <bool RequiresStableAddress,
          bool DenseOutput,
          typename RandomAccessIteratorTupleIn,
          typename RandomAccessIteratorOut>
struct arch_policies_from_types
{
  static_assert(sizeof(RandomAccessIteratorTupleIn) == 0, "Second parameter must be a tuple");
};

template <bool RequiresStableAddress,
          bool DenseOutput,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut>
struct arch_policies_from_types<RequiresStableAddress,
                                DenseOutput,
                                ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                                RandomAccessIteratorOut>
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> transform_arch_policy
  {
    constexpr auto policies = arch_policies<sizeof...(RandomAccessIteratorsIn)>{
      RequiresStableAddress,
      DenseOutput,
      {make_iterator_info<RandomAccessIteratorsIn>()...},
      make_iterator_info<RandomAccessIteratorOut>()};
    return policies(arch);
  }
};
} // namespace detail::transform

CUB_NAMESPACE_END
