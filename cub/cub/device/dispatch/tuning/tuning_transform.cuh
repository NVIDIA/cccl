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

#include <cub/detail/detect_cuda_runtime.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/__cmath/pow2.h>
#include <cuda/__cmath/round_up.h>
#include <cuda/__functional/address_stability.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/__numeric/reduce.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>

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

template <int BlockThreads>
struct prefetch_policy_t
{
  static constexpr int block_threads = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int items_per_thread_no_input = 2; // when there are no input iterators, the kernel is just filling
  static constexpr int min_items_per_thread      = 1;
  static constexpr int max_items_per_thread      = 32;

  // TODO: remove with C++20
  // The value of the below does not matter.
  static constexpr int not_a_vectorized_policy = 0;
};

CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  TransformAgentPrefetchPolicy,
  (always_true),
  (block_threads, BlockThreads, int),
  (items_per_thread_no_input, ItemsPerThreadNoInput, int),
  (min_items_per_thread, MinItemsPerThread, int),
  (max_items_per_thread, MaxItemsPerThread, int),
  (not_a_vectorized_policy, NotAVectorizedPolicy, int) ) // TODO: remove with C++20

template <typename Tuning>
struct vectorized_policy_t : prefetch_policy_t<Tuning::block_threads>
{
  static constexpr int items_per_thread_vectorized = Tuning::items_per_thread;
  static constexpr int vec_size                    = Tuning::vec_size;

  using not_a_vectorized_policy = void; // TODO: remove with C++20, shadows the variable in prefetch_policy_t
};

CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  TransformAgentVectorizedPolicy,
  (always_true), // TODO: restore with C++20: (TransformAgentPrefetchPolicy),
  (block_threads, BlockThreads, int),
  (items_per_thread_no_input, ItemsPerThreadNoInput, int),
  (min_items_per_thread, MinItemsPerThread, int),
  (max_items_per_thread, MaxItemsPerThread, int),
  (items_per_thread_vectorized, ItemsPerThreadVectorized, int),
  (vec_size, VecSize, int) )

template <int BlockThreads, int BulkCopyAlignment>
struct async_copy_policy_t
{
  static constexpr int block_threads = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int min_items_per_thread = 1;
  static constexpr int max_items_per_thread = 32;

  static constexpr int bulk_copy_alignment = BulkCopyAlignment;
};

CUB_DETAIL_POLICY_WRAPPER_DEFINE(
  TransformAgentAsyncPolicy,
  (always_true),
  (block_threads, BlockThreads, int),
  (min_items_per_thread, MinItemsPerThread, int),
  (max_items_per_thread, MaxItemsPerThread, int),
  (bulk_copy_alignment, BulkCopyAlignment, int) )

_CCCL_TEMPLATE(typename PolicyT)
_CCCL_REQUIRES((!TransformAgentPrefetchPolicy<PolicyT> && !TransformAgentAsyncPolicy<PolicyT>
                && !TransformAgentVectorizedPolicy<PolicyT>) )
__host__ __device__ constexpr PolicyT MakePolicyWrapper(PolicyT policy)
{
  return policy;
}

template <typename PolicyT, typename = void>
struct TransformPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE TransformPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct TransformPolicyWrapper<StaticPolicyT,
                              ::cuda::std::void_t<decltype(StaticPolicyT::algorithm),
                                                  decltype(StaticPolicyT::min_bif),
                                                  typename StaticPolicyT::prefetch_policy,
                                                  typename StaticPolicyT::vectorized_policy,
                                                  typename StaticPolicyT::async_policy>> : StaticPolicyT
{
  _CCCL_HOST_DEVICE TransformPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  _CCCL_HOST_DEVICE static constexpr Algorithm Algorithm()
  {
    return StaticPolicyT::algorithm;
  }

  _CCCL_HOST_DEVICE static constexpr int MinBif()
  {
    return StaticPolicyT::min_bif;
  }

  _CCCL_HOST_DEVICE static constexpr auto PrefetchPolicy()
  {
    return MakePolicyWrapper(typename StaticPolicyT::prefetch_policy());
  }

  _CCCL_HOST_DEVICE static constexpr auto VectorizedPolicy()
  {
    return MakePolicyWrapper(typename StaticPolicyT::vectorized_policy());
  }

  _CCCL_HOST_DEVICE static constexpr auto AsyncPolicy()
  {
    return MakePolicyWrapper(typename StaticPolicyT::async_policy());
  }

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"min_bif">()           = value<StaticPolicyT::min_bif>(),
                  key<"algorithm">()         = value<static_cast<int>(StaticPolicyT::algorithm)>(),
                  key<"prefetch_policy">()   = PrefetchPolicy().EncodedPolicy(),
                  key<"vectorized_policy">() = VectorizedPolicy().EncodedPolicy(),
                  key<"async_policy">()      = AsyncPolicy().EncodedPolicy()>();
  }
#endif // CUB_ENABLE_POLICY_PTX_JSON
};

template <typename PolicyT>
_CCCL_HOST_DEVICE TransformPolicyWrapper<PolicyT> MakeTransformPolicyWrapper(PolicyT base)
{
  return TransformPolicyWrapper<PolicyT>(base);
}

template <typename... Its>
_CCCL_HOST_DEVICE constexpr auto loaded_bytes_per_iteration() -> int
{
  return (int{sizeof(it_value_t<Its>)} + ... + 0);
}

constexpr int ldgsts_size_and_align = 16;

template <typename ItValueSizesAlignments>
_CCCL_HOST_DEVICE constexpr auto memcpy_async_dyn_smem_for_tile_size(
  ItValueSizesAlignments it_value_sizes_alignments, int tile_size, int copy_alignment = ldgsts_size_and_align) -> int
{
  int smem_size = 0;
  for (auto&& [vt_size, vt_alignment] : it_value_sizes_alignments)
  {
    smem_size =
      static_cast<int>(::cuda::round_up(smem_size, ::cuda::std::max(static_cast<int>(vt_alignment), copy_alignment)));
    // max head/tail padding is copy_alignment - sizeof(T) each
    const int max_bytes_to_copy =
      static_cast<int>(vt_size) * tile_size + ::cuda::std::max(copy_alignment - static_cast<int>(vt_size), 0) * 2;
    smem_size += max_bytes_to_copy;
  };
  return smem_size;
}

constexpr int bulk_copy_size_multiple = 16;

_CCCL_HOST_DEVICE constexpr auto bulk_copy_alignment(int sm_arch) -> int
{
  return sm_arch < 1000 ? 128 : 16;
}

template <typename ItValueSizesAlignments>
_CCCL_HOST_DEVICE constexpr auto
bulk_copy_dyn_smem_for_tile_size(ItValueSizesAlignments it_value_sizes_alignments, int tile_size, int bulk_copy_align)
  -> int
{
  // we rely on the tile_size being a multiple of alignments, so shifting offsets/pointers by it retains alignments
  _CCCL_ASSERT(tile_size % bulk_copy_align == 0, "");
  _CCCL_ASSERT(tile_size % bulk_copy_size_multiple == 0, "");

  int tile_padding = bulk_copy_align;
  for (auto&& [_, vt_alignment] : it_value_sizes_alignments)
  {
    tile_padding = ::cuda::std::max(tile_padding, static_cast<int>(vt_alignment));
  }

  int smem_size = tile_padding; // for the barrier and padding
  for (auto&& [vt_size, _] : it_value_sizes_alignments)
  {
    smem_size += tile_padding + static_cast<int>(vt_size) * tile_size;
  }
  return smem_size;
}

_CCCL_HOST_DEVICE constexpr int arch_to_min_bytes_in_flight(int sm_arch)
{
  if (sm_arch >= 1000)
  {
    return 64 * 1024; // B200
  }
  if (sm_arch >= 900)
  {
    return 48 * 1024; // 32 for H100, 48 for H200
  }
  if (sm_arch >= 800)
  {
    return 16 * 1024; // A100
  }
  return 12 * 1024; // V100 and below
}

template <typename T>
inline constexpr size_t size_of = sizeof(T);

template <>
inline constexpr size_t size_of<void> = 0;

template <typename... RandomAccessIteratorsIn>
_CCCL_HOST_DEVICE static constexpr auto make_sizes_alignments()
{
  return ::cuda::std::array<::cuda::std::pair<::cuda::std::size_t, ::cuda::std::size_t>,
                            sizeof...(RandomAccessIteratorsIn)>{
    {{sizeof(it_value_t<RandomAccessIteratorsIn>), alignof(it_value_t<RandomAccessIteratorsIn>)}...}};
}

template <int PtxVersion, int StoreSize, int... LoadSizes>
struct tuning_vec
{
  // defaults from fill on RTX 5090, but can be changed
  static constexpr int block_threads    = 256;
  static constexpr int vec_size         = 4;
  static constexpr int items_per_thread = 8;
};

// manually tuned fill on A100
template <int StoreSize>
struct tuning_vec<800, StoreSize>
{
  static constexpr int block_threads    = 256;
  static constexpr int vec_size         = ::cuda::std::max(8 / StoreSize, 1); // 64-bit instructions
  static constexpr int items_per_thread = 8;
};

// manually tuned fill on H200
template <int StoreSize>
struct tuning_vec<900, StoreSize>
{
  static constexpr int block_threads    = StoreSize > 4 ? 128 : 256;
  static constexpr int vec_size         = ::cuda::std::max(8 / StoreSize, 1); // 64-bit instructions
  static constexpr int items_per_thread = 16;
};

// manually tuned fill on B200, same as H200
template <int StoreSize>
struct tuning_vec<1000, StoreSize> : tuning_vec<900, StoreSize>
{};

// manually tuned fill on RTX 5090
template <int StoreSize>
struct tuning_vec<1200, StoreSize>
{
  static constexpr int block_threads    = 256;
  static constexpr int vec_size         = 4;
  static constexpr int items_per_thread = 8;
};

// manually tuned triad on A100
template <int StoreSize, int LoadSize0, int... LoadSizes>
struct tuning_vec<800, StoreSize, LoadSize0, LoadSizes...>
{
  static constexpr int block_threads    = 128;
  static constexpr int vec_size         = 4;
  static constexpr int items_per_thread = 16;
};

template <bool RequiresStableAddress,
          bool DenseOutput,
          typename RandomAccessIteratorTupleIn,
          typename RandomAccessIteratorOut>
struct policy_hub
{
  static_assert(sizeof(RandomAccessIteratorTupleIn) == 0, "Second parameter must be a tuple");
};

template <bool RequiresStableAddress,
          bool DenseOutput,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut>
struct policy_hub<RequiresStableAddress,
                  DenseOutput,
                  ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                  RandomAccessIteratorOut>
{
  static constexpr bool no_input_streams = sizeof...(RandomAccessIteratorsIn) == 0;
  static constexpr bool all_inputs_contiguous =
    (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIteratorsIn> && ...);
  static constexpr bool all_input_values_trivially_reloc =
    (THRUST_NS_QUALIFIER::is_trivially_relocatable_v<it_value_t<RandomAccessIteratorsIn>> && ...);
  static constexpr bool can_memcpy_all_inputs = all_inputs_contiguous && all_input_values_trivially_reloc;
  // the vectorized kernel supports mixing contiguous and non-contiguous iterators
  static constexpr bool can_memcpy_contiguous_inputs =
    ((!THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIteratorsIn>
      || THRUST_NS_QUALIFIER::is_trivially_relocatable_v<it_value_t<RandomAccessIteratorsIn>>)
     && ...);

  static constexpr bool all_value_types_have_power_of_two_size =
    (::cuda::is_power_of_two(sizeof(it_value_t<RandomAccessIteratorsIn>)) && ...)
    && ::cuda::is_power_of_two(size_of<it_value_t<RandomAccessIteratorOut>>);

  static constexpr bool fallback_to_prefetch = RequiresStableAddress || !can_memcpy_contiguous_inputs || !DenseOutput;

  // TODO(bgruber): consider a separate kernel for just filling

  struct policy300 : ChainedPolicy<300, policy300, policy300>
  {
    static constexpr int min_bif = arch_to_min_bytes_in_flight(300);
    using prefetch_policy        = prefetch_policy_t<256>;
    using vectorized_policy      = vectorized_policy_t<
           tuning_vec<500, size_of<it_value_t<RandomAccessIteratorOut>>, sizeof(it_value_t<RandomAccessIteratorsIn>)...>>;
    using async_policy = async_copy_policy_t<256, 16>; // dummy policy, never used
    static constexpr auto algorithm =
      (fallback_to_prefetch || !all_value_types_have_power_of_two_size) ? Algorithm::prefetch : Algorithm::vectorized;
  };

  struct policy800 : ChainedPolicy<800, policy800, policy300>
  {
  private:
    static constexpr int block_threads = 256;

  public:
    static constexpr int min_bif = arch_to_min_bytes_in_flight(800);
    using prefetch_policy        = prefetch_policy_t<block_threads>;
    using vectorized_policy      = vectorized_policy_t<
           tuning_vec<800, size_of<it_value_t<RandomAccessIteratorOut>>, sizeof(it_value_t<RandomAccessIteratorsIn>)...>>;
    using async_policy = async_copy_policy_t<block_threads, ldgsts_size_and_align>;

  private:
    // We cannot use the architecture-specific amount of SMEM here instead of max_smem_per_block, because this is not
    // forward compatible. If a user compiled for sm_xxx and we assume the available SMEM for that architecture, but
    // then runs on the next architecture after that, which may have a smaller available SMEM, we get a crash.
    static constexpr bool exhaust_smem =
      memcpy_async_dyn_smem_for_tile_size(
        make_sizes_alignments<RandomAccessIteratorsIn...>(),
        block_threads* async_policy::min_items_per_thread,
        ldgsts_size_and_align)
      > int{max_smem_per_block};

    // on Ampere, the vectorized kernel performs better for 1 and 2 byte values
    static constexpr bool use_vector_kernel_on_ampere =
      ((size_of<it_value_t<RandomAccessIteratorsIn>> < 4) && ...) && sizeof...(RandomAccessIteratorsIn) > 1
      && size_of<it_value_t<RandomAccessIteratorOut>> < 4;

    static constexpr bool fallback_to_vectorized =
      exhaust_smem || no_input_streams || !can_memcpy_all_inputs || use_vector_kernel_on_ampere;

  public:
    static constexpr auto algorithm =
      fallback_to_prefetch ? Algorithm::prefetch
      : fallback_to_vectorized
        ? (all_value_types_have_power_of_two_size ? Algorithm::vectorized : Algorithm::prefetch)
        : Algorithm::memcpy_async;
  };

  template <int AsyncBlockSize, int PtxVersion>
  struct bulk_copy_policy_base
  {
  private:
    static constexpr int alignment = bulk_copy_alignment(PtxVersion);

  public:
    static constexpr int min_bif = arch_to_min_bytes_in_flight(PtxVersion);
    using prefetch_policy        = prefetch_policy_t<256>;
    using vectorized_policy =
      vectorized_policy_t<tuning_vec<PtxVersion,
                                     size_of<it_value_t<RandomAccessIteratorOut>>,
                                     sizeof(it_value_t<RandomAccessIteratorsIn>)...>>;
    using async_policy = async_copy_policy_t<AsyncBlockSize, alignment>;

  private:
    // We cannot use the architecture-specific amount of SMEM here instead of max_smem_per_block, because this is not
    // forward compatible. If a user compiled for sm_xxx and we assume the available SMEM for that architecture, but
    // then runs on the next architecture after that, which may have a smaller available SMEM, we get a crash.
    static constexpr bool exhaust_smem =
      bulk_copy_dyn_smem_for_tile_size(
        make_sizes_alignments<RandomAccessIteratorsIn...>(),
        AsyncBlockSize* async_policy::min_items_per_thread,
        alignment)
      > int{max_smem_per_block};

    // on Hopper, the vectorized kernel performs better for 1 and 2 byte values
    static constexpr bool use_vector_kernel_on_hopper =
      ((size_of<it_value_t<RandomAccessIteratorsIn>> < 4) && ...) && sizeof...(RandomAccessIteratorsIn) > 1
      && size_of<it_value_t<RandomAccessIteratorOut>> < 4;

    // if each tile size is a multiple of the bulk copy and maximum value type alignments, the alignment is retained if
    // the base pointer is sufficiently aligned (the correct check would be if it's a multiple of all value types
    // following the current tile). we would otherwise need to realign every SMEM tile individually, which is costly and
    // complex, so let's fall back in this case.
    static constexpr int max_alignment =
      ::cuda::std::max({alignment, int{alignof(it_value_t<RandomAccessIteratorsIn>)}...});
    static constexpr bool tile_sizes_retain_alignment =
      (((int{sizeof(it_value_t<RandomAccessIteratorsIn>)} * AsyncBlockSize) % max_alignment == 0) && ...);
    static constexpr bool enough_threads_for_peeling = AsyncBlockSize >= alignment; // head and tail bytes
    static constexpr bool fallback_to_vectorized =
      exhaust_smem || !tile_sizes_retain_alignment || !enough_threads_for_peeling || no_input_streams
      || !can_memcpy_all_inputs || (PtxVersion == 900 && use_vector_kernel_on_hopper);

  public:
    static constexpr auto algorithm =
      fallback_to_prefetch ? Algorithm::prefetch
      : fallback_to_vectorized
        ? (all_value_types_have_power_of_two_size ? Algorithm::vectorized : Algorithm::prefetch)
        : Algorithm::ublkcp;
  };

  struct policy900
      : bulk_copy_policy_base<256, 900>
      , ChainedPolicy<900, policy900, policy800>
  {};

  struct policy1000
      : bulk_copy_policy_base<128, 1000>
      , ChainedPolicy<1000, policy1000, policy900>
  {};

  using max_policy = policy1000;
};
} // namespace detail::transform

CUB_NAMESPACE_END
