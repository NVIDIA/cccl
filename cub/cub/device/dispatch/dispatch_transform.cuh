// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/uninitialized_copy.cuh>
#include <cub/device/dispatch/kernels/transform.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/type_traits/is_trivially_relocatable.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/cmath>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/array>
#include <cuda/std/expected>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cassert>

// cooperative groups do not support NVHPC yet
#if !_CCCL_CUDA_COMPILER(NVHPC)
#  include <cooperative_groups.h>
#  include <cooperative_groups/memcpy_async.h>
#endif

CUB_NAMESPACE_BEGIN

namespace detail::transform
{

enum class requires_stable_address
{
  no,
  yes
};

template <typename T>
using cuda_expected = ::cuda::std::expected<T, cudaError_t>;

// TODO(bgruber): this is very similar to thrust::cuda_cub::core::get_max_shared_memory_per_block. We should unify this.
_CCCL_HOST_DEVICE inline cuda_expected<int> get_max_shared_memory()
{
  //  gevtushenko promised me that I can assume that the stream passed to the CUB API entry point (where the kernels
  //  will later be launched on) belongs to the currently active device. So we can just query the active device here.
  int device = 0;
  auto error = CubDebug(cudaGetDevice(&device));
  if (error != cudaSuccess)
  {
    return error;
  }

  int max_smem = 0;
  error        = CubDebug(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device));
  if (error != cudaSuccess)
  {
    return error;
  }

  return max_smem;
}

_CCCL_HOST_DEVICE inline cuda_expected<int> get_sm_count()
{
  int device = 0;
  auto error = CubDebug(cudaGetDevice(&device));
  if (error != cudaSuccess)
  {
    return error;
  }

  int sm_count = 0;
  error        = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  if (error != cudaSuccess)
  {
    return error;
  }

  return sm_count;
}

struct elem_counts
{
  int elem_per_thread;
  int tile_size;
  int smem_size;
};

struct prefetch_config
{
  int max_occupancy;
  int sm_count;
};

template <requires_stable_address StableAddress,
          typename Offset,
          typename RandomAccessIteratorTupleIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub = policy_hub<StableAddress == requires_stable_address::yes, RandomAccessIteratorTupleIn>>
struct dispatch_t;

template <requires_stable_address StableAddress,
          typename Offset,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub>
struct dispatch_t<StableAddress,
                  Offset,
                  ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                  RandomAccessIteratorOut,
                  TransformOp,
                  PolicyHub>
{
  static_assert(::cuda::std::is_same_v<Offset, ::cuda::std::int32_t>
                  || ::cuda::std::is_same_v<Offset, ::cuda::std::int64_t>,
                "cub::DeviceTransform is only tested and tuned for 32-bit or 64-bit signed offset types");

  ::cuda::std::tuple<RandomAccessIteratorsIn...> in;
  RandomAccessIteratorOut out;
  Offset num_items;
  TransformOp op;
  cudaStream_t stream;

#define CUB_DETAIL_TRANSFORM_KERNEL_PTR             \
  &transform_kernel<typename PolicyHub::max_policy, \
                    Offset,                         \
                    TransformOp,                    \
                    RandomAccessIteratorOut,        \
                    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>...>

  static constexpr int loaded_bytes_per_iter = loaded_bytes_per_iteration<RandomAccessIteratorsIn...>();

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  // TODO(bgruber): I want to write tests for this but those are highly depending on the architecture we are running
  // on?
  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE auto configure_ublkcp_kernel() -> cuda_expected<
    ::cuda::std::
      tuple<THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron, decltype(CUB_DETAIL_TRANSFORM_KERNEL_PTR), int>>
  {
    using policy_t          = typename ActivePolicy::algo_policy;
    constexpr int block_dim = policy_t::block_threads;
    static_assert(block_dim % bulk_copy_alignment == 0,
                  "block_threads needs to be a multiple of bulk_copy_alignment (128)"); // then tile_size is a multiple
                                                                                        // of 128-byte

    auto determine_element_counts = [&]() -> cuda_expected<elem_counts> {
      const auto max_smem = get_max_shared_memory();
      if (!max_smem)
      {
        return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(max_smem.error());
      }

      elem_counts last_counts{};
      // Increase the number of output elements per thread until we reach the required bytes in flight.
      static_assert(policy_t::min_items_per_thread <= policy_t::max_items_per_thread, ""); // ensures the loop below
      // runs at least once
      for (int elem_per_thread = +policy_t::min_items_per_thread; elem_per_thread < +policy_t::max_items_per_thread;
           ++elem_per_thread)
      {
        const int tile_size = block_dim * elem_per_thread;
        const int smem_size = bulk_copy_smem_for_tile_size<RandomAccessIteratorsIn...>(tile_size);
        if (smem_size > *max_smem)
        {
          // assert should be prevented by smem check in policy
          _CCCL_ASSERT_HOST(last_counts.elem_per_thread > 0, "min_items_per_thread exceeds available shared memory");
          return last_counts;
        }

        if (tile_size >= num_items)
        {
          return elem_counts{elem_per_thread, tile_size, smem_size};
        }

        int max_occupancy = 0;
        const auto error =
          CubDebug(MaxSmOccupancy(max_occupancy, CUB_DETAIL_TRANSFORM_KERNEL_PTR, block_dim, smem_size));
        if (error != cudaSuccess)
        {
          return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 with GCC 7 fails CTAD here */>(error);
        }

        const int bytes_in_flight_SM = max_occupancy * tile_size * loaded_bytes_per_iter;
        if (ActivePolicy::min_bif <= bytes_in_flight_SM)
        {
          return elem_counts{elem_per_thread, tile_size, smem_size};
        }

        last_counts = elem_counts{elem_per_thread, tile_size, smem_size};
      }
      return last_counts;
    };
    cuda_expected<elem_counts> config = [&]() {
      NV_IF_TARGET(NV_IS_HOST,
                   (static auto cached_config = determine_element_counts(); return cached_config;),
                   (
                     // we cannot cache the determined element count in device code
                     return determine_element_counts();));
    }();
    if (!config)
    {
      return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 with GCC 7 fails CTAD here */>(config.error());
    }
    _CCCL_ASSERT_HOST(config->elem_per_thread > 0, "");
    _CCCL_ASSERT_HOST(config->tile_size > 0, "");
    _CCCL_ASSERT_HOST(config->tile_size % bulk_copy_alignment == 0, "");
    _CCCL_ASSERT_HOST((sizeof...(RandomAccessIteratorsIn) == 0) != (config->smem_size != 0), ""); // logical xor

    const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{config->tile_size}));
    return ::cuda::std::make_tuple(
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(grid_dim, block_dim, config->smem_size, stream),
      CUB_DETAIL_TRANSFORM_KERNEL_PTR,
      config->elem_per_thread);
  }

  template <typename ActivePolicy, size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
    invoke_algorithm(::cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::ublkcp>)
  {
    auto ret = configure_ublkcp_kernel<ActivePolicy>();
    if (!ret)
    {
      return ret.error();
    }
    auto [launcher, kernel, elem_per_thread] = *ret;
    return launcher.doit(
      kernel,
      num_items,
      elem_per_thread,
      op,
      out,
      make_aligned_base_ptr_kernel_arg(
        THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(::cuda::std::get<Is>(in)), bulk_copy_alignment)...);
  }
#endif // _CUB_HAS_TRANSFORM_UBLKCP

  template <typename ActivePolicy, size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
    invoke_algorithm(::cuda::std::index_sequence<Is...>, ::cuda::std::integral_constant<Algorithm, Algorithm::prefetch>)
  {
    using policy_t          = typename ActivePolicy::algo_policy;
    constexpr int block_dim = policy_t::block_threads;

    auto determine_config = [&]() -> cuda_expected<prefetch_config> {
      int max_occupancy = 0;
      const auto error  = CubDebug(MaxSmOccupancy(max_occupancy, CUB_DETAIL_TRANSFORM_KERNEL_PTR, block_dim, 0));
      if (error != cudaSuccess)
      {
        return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
      }
      const auto sm_count = get_sm_count();
      if (!sm_count)
      {
        return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(sm_count.error());
      }
      return prefetch_config{max_occupancy, *sm_count};
    };

    cuda_expected<prefetch_config> config = [&]() {
      NV_IF_TARGET(
        NV_IS_HOST,
        (
          // this static variable exists for each template instantiation of the surrounding function and class, on which
          // the chosen element count solely depends (assuming max SMEM is constant during a program execution)
          static auto cached_config = determine_config(); return cached_config;),
        (
          // we cannot cache the determined element count in device code
          return determine_config();));
    }();
    if (!config)
    {
      return config.error();
    }

    // choose items per thread to reach minimum bytes in flight
    const int items_per_thread =
      loaded_bytes_per_iter == 0
        ? +policy_t::items_per_thread_no_input
        : ::cuda::ceil_div(ActivePolicy::min_bif, config->max_occupancy * block_dim * loaded_bytes_per_iter);

    // but also generate enough blocks for full occupancy to optimize small problem sizes, e.g., 2^16 or 2^20 elements
    const int items_per_thread_evenly_spread = static_cast<int>(
      (::cuda::std::min)(Offset{items_per_thread}, num_items / (config->sm_count * block_dim * config->max_occupancy)));

    const int items_per_thread_clamped = ::cuda::std::clamp(
      items_per_thread_evenly_spread, +policy_t::min_items_per_thread, +policy_t::max_items_per_thread);
    const int tile_size = block_dim * items_per_thread_clamped;
    const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{tile_size}));
    return CubDebug(
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(grid_dim, block_dim, 0, stream)
        .doit(
          CUB_DETAIL_TRANSFORM_KERNEL_PTR,
          num_items,
          items_per_thread_clamped,
          op,
          out,
          make_iterator_kernel_arg(THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(::cuda::std::get<Is>(in)))...));
  }

  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    // // TODO(bgruber): replace the overload set by if constexpr in C++17
    return invoke_algorithm<ActivePolicy>(::cuda::std::index_sequence_for<RandomAccessIteratorsIn...>{},
                                          ::cuda::std::integral_constant<Algorithm, ActivePolicy::algorithm>{});
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> in,
    RandomAccessIteratorOut out,
    Offset num_items,
    TransformOp op,
    cudaStream_t stream)
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    int ptx_version = 0;
    auto error      = CubDebug(PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    dispatch_t dispatch{::cuda::std::move(in), ::cuda::std::move(out), num_items, ::cuda::std::move(op), stream};
    return CubDebug(PolicyHub::max_policy::Invoke(ptx_version, dispatch));
  }

#undef CUB_DETAIL_TRANSFORM_KERNEL_PTR
};
} // namespace detail::transform
CUB_NAMESPACE_END
