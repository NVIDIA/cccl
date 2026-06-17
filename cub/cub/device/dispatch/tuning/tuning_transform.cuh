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
#include <cuda/__device/compute_capability.h>
#include <cuda/__functional/address_stability.h>
#include <cuda/__functional/always_true_false.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/array>
#include <cuda/std/concepts>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

//! Backend algorithms for @ref DeviceTransform.
enum class TransformAlgorithm
{
  // We previously had a fallback algorithm that would use cub::DeviceFor. Benchmarks showed that the prefetch algorithm
  // is always superior to that fallback, so it was removed.
  prefetch, //!< Uses a transform kernel that relies on prefetching the memory of all contiguous iterators.
  vectorized, //!< Uses a transform kernel that uses load and store vectorization.
  ldgsts, //!< Uses a transform kernel that relies on cp.async/LDGSTS to stage data into shared memory.
  ublkcp //!< Uses a transform kernel that relies on cp.async.bulk/UBLKCP to stage data into shared memory.
};

#if _CCCL_HOSTED()
inline ::std::ostream& operator<<(::std::ostream& os, const TransformAlgorithm& algorithm)
{
  switch (algorithm)
  {
    case TransformAlgorithm::prefetch:
      return os << "TransformAlgorithm::prefetch";
    case TransformAlgorithm::vectorized:
      return os << "TransformAlgorithm::vectorized";
    case TransformAlgorithm::ldgsts:
      return os << "TransformAlgorithm::ldgsts";
    case TransformAlgorithm::ublkcp:
      return os << "TransformAlgorithm::ublkcp";
    default:
      return os << "TransformAlgorithm::<unknown>";
  }
}
#endif // _CCCL_HOSTED()

//! The prefetch sub-policy for @ref TransformPolicy.
struct TransformPrefetchPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  int items_per_thread_no_input = 2; //!< When there are no iterators as inputs, the kernel is just filling. This is the
                                     //!< number of items written per thread in this case.
  int min_items_per_thread = 1; //!< Minimum number of items per thread (inclusive)
  int max_items_per_thread = 32; //!< Maximum number of items per thread (inclusive)
  int prefetch_byte_stride = 128; //!< The stride in bytes to issue prefetch requests to memory. Corresponds somewhat to
                                  //!< the size of a cache line.
  // ahendriksen: various unrolling yields less <1% gains at much higher compile-time cost, so prevent unrolling
  // bgruber: but A6000 and H100 show small gains without pragma, so omitting pragma
  int unroll_factor = 0; //!< For any value >1, the unroll factor for the transformation loop in the kernel. The value 0
                         //!< retains the compiler's default unrolling by specifying no unroll pragma. 1 prevents
                         //!< unrolling.

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const TransformPrefetchPolicy& lhs, const TransformPrefetchPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block
        && lhs.items_per_thread_no_input == rhs.items_per_thread_no_input
        && lhs.min_items_per_thread == rhs.min_items_per_thread && lhs.max_items_per_thread == rhs.max_items_per_thread
        && lhs.prefetch_byte_stride == rhs.prefetch_byte_stride && lhs.unroll_factor == rhs.unroll_factor;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const TransformPrefetchPolicy& lhs, const TransformPrefetchPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const TransformPrefetchPolicy& policy)
  {
    return os
        << "TransformPrefetchPolicy { .threads_per_block = " << policy.threads_per_block
        << ", .items_per_thread_no_input = " << policy.items_per_thread_no_input << ", .min_items_per_thread = "
        << policy.min_items_per_thread << ", .max_items_per_thread = " << policy.max_items_per_thread
        << ", .prefetch_byte_stride = " << policy.prefetch_byte_stride << ", .unroll_factor = " << policy.unroll_factor
        << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The vectorized sub-policy for @ref TransformPolicy.
struct TransformVectorizedPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread. Must be a multiple of vec_size.
  int vec_size; //!< Number of elements loaded/stored per vectorized access. Must evenly divide items_per_thread.

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const TransformVectorizedPolicy& lhs, const TransformVectorizedPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.vec_size == rhs.vec_size;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const TransformVectorizedPolicy& lhs, const TransformVectorizedPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const TransformVectorizedPolicy& policy)
  {
    return os << "TransformVectorizedPolicy { .threads_per_block = " << policy.threads_per_block
              << ", .items_per_thread = " << policy.items_per_thread << ", .vec_size = " << policy.vec_size << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The async copy sub-policy for @ref TransformPolicy.
struct TransformAsyncCopyPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  int min_items_per_thread = 1; //!< Minimum number of items per thread (inclusive)
  int max_items_per_thread = 32; //!< Maximum number of items per thread (inclusive)
  // Unroll 1 tends to improve performance, especially for smaller data types (confirmed by benchmark)
  int unroll_factor = 1; //!< The unroll factor for the transformation loop in the kernel. The value 0 retains the
                         //!< compiler's default unrolling (specifying no unroll pragma), 1 means no unrolling.

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const TransformAsyncCopyPolicy& lhs, const TransformAsyncCopyPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.min_items_per_thread == rhs.min_items_per_thread
        && lhs.max_items_per_thread == rhs.max_items_per_thread && lhs.unroll_factor == rhs.unroll_factor;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const TransformAsyncCopyPolicy& lhs, const TransformAsyncCopyPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const TransformAsyncCopyPolicy& policy)
  {
    return os << "TransformAsyncCopyPolicy { .threads_per_block = " << policy.threads_per_block
              << ", .min_items_per_thread = " << policy.min_items_per_thread << ", .max_items_per_thread = "
              << policy.max_items_per_thread << ", .unroll_factor = " << policy.unroll_factor << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for all algorithms in @ref DeviceTransform.
struct TransformPolicy
{
  int min_bytes_in_flight; //!< Minimum number of bytes in flight per SM to reach by scaling the items per thread. Has
                           //!< no effect if algorithm is @p vectorized.
  TransformAlgorithm algorithm; //!< The transform algorithm to use by the kernel
  TransformPrefetchPolicy prefetch; //!< Sub-policy for the prefetch algorithm. Only used when @p algorithm is @p
                                    //!< prefetch or when @p algorithm is @p vectorized and the input pointers are not
                                    //!< sufficiently aligned.
  TransformVectorizedPolicy vectorized; //!< Sub-policy for the vectorized algorithm. Only used when @p algorithm is @p
                                        //!< vectorized.
  TransformAsyncCopyPolicy async_copy; //!< Sub-policy for the async copy algorithms. Only used when @p algorithm is @p
                                       //!< ldgsts or @p ublkcp.

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const TransformPolicy& lhs, const TransformPolicy& rhs) noexcept
  {
    return lhs.min_bytes_in_flight == rhs.min_bytes_in_flight && lhs.algorithm == rhs.algorithm
        && lhs.prefetch == rhs.prefetch && lhs.vectorized == rhs.vectorized && lhs.async_copy == rhs.async_copy;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const TransformPolicy& lhs, const TransformPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const TransformPolicy& policy)
  {
    return os << "TransformPolicy { .min_bytes_in_flight = " << policy.min_bytes_in_flight
              << ", .algorithm = " << policy.algorithm << ", .prefetch = " << policy.prefetch
              << ", .vectorized = " << policy.vectorized << ", .async_copy = " << policy.async_copy << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::transform
{
#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept transform_policy_selector = policy_selector<T, TransformPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

template <typename... Its>
_CCCL_HOST_DEVICE constexpr auto loaded_bytes_per_iteration() -> int
{
  return (int{sizeof(it_value_t<Its>)} + ... + 0);
}

inline constexpr int ldgsts_size_and_align = 16;

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

_CCCL_HOST_DEVICE constexpr auto bulk_copy_alignment(::cuda::compute_capability cc) -> int
{
  return (cc < ::cuda::compute_capability{10, 0}) ? 128 : 16;
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

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int cc_to_min_bytes_in_flight(::cuda::compute_capability cc)
{
  if (cc >= ::cuda::compute_capability{10, 0})
  {
    return 64 * 1024; // B200
  }
  if (cc >= ::cuda::compute_capability{9, 0})
  {
    return 48 * 1024; // 32 for H100, 48 for H200
  }
  if (cc >= ::cuda::compute_capability{8, 0})
  {
    return 16 * 1024; // A100
  }
  return 12 * 1024; // V100 and below
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
tuned_vectorized_policy(::cuda::compute_capability cc, int store_size, bool filling)
{
  if (filling)
  {
    // manually tuned fill on RTX 5090
    // TODO(bgruber): re-enable this later! It's disabled to avoid SASS changes in PR #6914
    // if (cc >= ::cuda::compute_capability{12, 0})
    // {
    //   return TransformVectorizedPolicy{256, 8, 4};
    // }
    // manually tuned fill on B200, same as H200
    if (cc >= ::cuda::compute_capability{9, 0})
    {
      return TransformVectorizedPolicy{
        store_size > 4 ? 128 : 256, 16, ::cuda::std::max(8 / store_size, 1) /* 64-bit instructions */};
    }
    // manually tuned fill on A100
    if (cc >= ::cuda::compute_capability{8, 0})
    {
      return TransformVectorizedPolicy{256, 8, ::cuda::std::max(8 / store_size, 1) /* 64-bit instructions */};
    }
  }
  else
  {
    // manually tuned triad on A100
    if (cc == ::cuda::compute_capability{8, 0})
    {
      return TransformVectorizedPolicy{128, 16, 4};
    }
  }

  // defaults from fill on RTX 5090, but can be changed
  return TransformVectorizedPolicy{256, 8, 4};
}

template <int InputCount>
struct policy_selector
{
  bool requires_stable_address;
  bool dense_output;
  ::cuda::std::array<iterator_info, InputCount> inputs;
  iterator_info output;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> TransformPolicy
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
      _CCCL_ASSERT(input.value_type_size > 0, "Iterators to inputs must not have a value_type of zero size");
      all_value_types_have_power_of_two_size &= ::cuda::is_power_of_two(input.value_type_size);
    }
    const bool can_memcpy_all_inputs = all_inputs_contiguous && all_input_values_trivially_reloc;
    const bool fallback_to_prefetch  = requires_stable_address || !can_memcpy_contiguous_inputs || !dense_output;
    const int min_bytes_in_flight    = cc_to_min_bytes_in_flight(cc);

    if (cc >= ::cuda::compute_capability{9, 0}) // handles sm_100 as well
    {
      const int async_block_size = (cc < ::cuda::compute_capability{10, 0}) ? 256 : 128;
      const int alignment        = bulk_copy_alignment(cc);

      const auto prefetch = TransformPrefetchPolicy{256};
      const auto vectorized =
        tuned_vectorized_policy(cc, ::cuda::std::max(1, output.value_type_size), no_input_streams);
      const auto async = TransformAsyncCopyPolicy{async_block_size};

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
      bool vector_kernel_is_faster =
        (cc == ::cuda::compute_capability{9, 0} && output.value_type_size < 4 && InputCount > 1);
      for (const auto& input : inputs)
      {
        vector_kernel_is_faster &= input.value_type_size < 4;
      }

      const bool enough_threads_for_peeling = async_block_size >= alignment; // head and tail bytes
      const bool fallback_to_vectorized = exhaust_smem || !tile_sizes_retain_alignment || !enough_threads_for_peeling
                                       || no_input_streams || !can_memcpy_all_inputs || vector_kernel_is_faster;

      const auto algorithm =
        fallback_to_prefetch ? TransformAlgorithm::prefetch
        : fallback_to_vectorized
          ? (all_value_types_have_power_of_two_size ? TransformAlgorithm::vectorized : TransformAlgorithm::prefetch)
          : TransformAlgorithm::ublkcp;

      return TransformPolicy{
        min_bytes_in_flight,
        algorithm,
        prefetch,
        vectorized,
        async,
      };
    }
    else if (cc >= ::cuda::compute_capability{8, 0})
    {
      const int threads_per_block = 256;
      const auto prefetch         = TransformPrefetchPolicy{threads_per_block};
      const auto vectorized =
        tuned_vectorized_policy(cc, ::cuda::std::max(1, output.value_type_size), no_input_streams);
      const auto async = TransformAsyncCopyPolicy{threads_per_block};

      // We cannot use the architecture-specific amount of SMEM here instead of max_smem_per_block, because this is not
      // forward compatible. If a user compiled for sm_xxx and we assume the available SMEM for that architecture, but
      // then runs on the next architecture after that, which may have a smaller available SMEM, we get a crash.
      const bool exhaust_smem =
        memcpy_async_dyn_smem_for_tile_size<InputCount>(
          inputs, threads_per_block * async.min_items_per_thread, ldgsts_size_and_align)
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
        fallback_to_prefetch ? TransformAlgorithm::prefetch
        : fallback_to_vectorized
          ? (all_value_types_have_power_of_two_size ? TransformAlgorithm::vectorized : TransformAlgorithm::prefetch)
          : TransformAlgorithm::ldgsts;

      return TransformPolicy{
        min_bytes_in_flight,
        algorithm,
        prefetch,
        vectorized,
        async,
      };
    }

    // fallback
    return TransformPolicy{
      min_bytes_in_flight,
      (fallback_to_prefetch || !all_value_types_have_power_of_two_size)
        ? TransformAlgorithm::prefetch
        : TransformAlgorithm::vectorized,
      TransformPrefetchPolicy{256},
      tuned_vectorized_policy(
        ::cuda::compute_capability{6, 0}, ::cuda::std::max(1, output.value_type_size), no_input_streams),
      TransformAsyncCopyPolicy{}, // never used
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
  static_assert((!::cuda::std::is_void_v<it_value_t<RandomAccessIteratorsIn>> && ...),
                "Iterators for inputs must not have a value_type of void. This can happen for multiple reasons. You "
                "could pass an output iterator by accident, but it could also be a transform_iterator with a "
                "__device__ callable and a deduced return type (which is void in host code).");

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> TransformPolicy
  {
    constexpr auto policies = policy_selector<sizeof...(RandomAccessIteratorsIn)>{
      RequiresStableAddress,
      DenseOutput,
      {make_iterator_info<RandomAccessIteratorsIn>()...},
      make_iterator_info<RandomAccessIteratorOut>()};
    return policies(cc);
  }
};
} // namespace detail::transform
CUB_NAMESPACE_END
