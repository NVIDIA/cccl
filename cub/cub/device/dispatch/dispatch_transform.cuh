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

#include <cub/detail/detect_cuda_runtime.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/uninitialized_copy.cuh>
#include <cub/device/dispatch/kernels/transform.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/detail/util/align.h>
#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/type_traits/is_trivially_relocatable.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/cmath>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

// cooperative groups do not support NVHPC yet
#if !_CCCL_CUDA_COMPILER(NVHPC)
#  include <cooperative_groups.h>

#  include <cooperative_groups/memcpy_async.h>
#endif // !_CCCL_CUDA_COMPILER(NVHPC)

CUB_NAMESPACE_BEGIN

namespace detail::transform
{

template <typename Offset,
          typename RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub>
struct TransformKernelSource;
;

template <typename Offset,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub>
struct TransformKernelSource<Offset,
                             ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                             RandomAccessIteratorOut,
                             TransformOp,
                             PolicyHub>
{
  CUB_DEFINE_KERNEL_GETTER(
    TransformKernel,
    transform_kernel<typename PolicyHub::max_policy,
                     Offset,
                     TransformOp,
                     THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorOut>,
                     THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>...>);

  CUB_RUNTIME_FUNCTION static constexpr int LoadedBytesPerIteration()
  {
    return loaded_bytes_per_iteration<RandomAccessIteratorsIn...>();
  };

  template <typename It>
  CUB_RUNTIME_FUNCTION constexpr kernel_arg<It> MakeIteratorKernelArg(It it)
  {
    return detail::transform::make_iterator_kernel_arg(it);
  }
};

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

struct async_config
{
  int items_per_thread;
  int max_occupancy;
  int sm_count;
};

struct prefetch_config
{
  int max_occupancy;
  int sm_count;
};

template <
  requires_stable_address StableAddress,
  typename Offset,
  typename RandomAccessIteratorTupleIn,
  typename RandomAccessIteratorOut,
  typename TransformOp,
  typename PolicyHub =
    policy_hub<StableAddress == requires_stable_address::yes, RandomAccessIteratorTupleIn, RandomAccessIteratorOut>,
  typename KernelSource =
    TransformKernelSource<Offset, RandomAccessIteratorTupleIn, RandomAccessIteratorOut, TransformOp, PolicyHub>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct dispatch_t;

template <requires_stable_address StableAddress,
          typename Offset,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename TransformOp,
          typename PolicyHub,
          typename KernelSource,
          typename KernelLauncherFactory>
struct dispatch_t<StableAddress,
                  Offset,
                  ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                  RandomAccessIteratorOut,
                  TransformOp,
                  PolicyHub,
                  KernelSource,
                  KernelLauncherFactory>
{
  static_assert(::cuda::std::is_same_v<Offset, ::cuda::std::int32_t>
                  || ::cuda::std::is_same_v<Offset, ::cuda::std::int64_t>,
                "cub::DeviceTransform is only tested and tuned for 32-bit or 64-bit signed offset types");

  ::cuda::std::tuple<RandomAccessIteratorsIn...> in;
  RandomAccessIteratorOut out;
  Offset num_items;
  TransformOp op;
  int bulk_copy_align;
  cudaStream_t stream;
  KernelSource kernel_source             = {};
  KernelLauncherFactory launcher_factory = {};

  // Reduces the items_per_thread when necessary to generate enough blocks to reach the maximum occupancy.
  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE int
  spread_out_items_per_thread(ActivePolicy active_policy, int items_per_thread, int sm_count, int max_occupancy)
  {
    auto wrapped_policy     = detail::transform::MakeTransformPolicyWrapper(active_policy);
    const int block_threads = wrapped_policy.BlockThreads();

    const int items_per_thread_evenly_spread = static_cast<int>(
      (::cuda::std::min) (Offset{items_per_thread},
                          ::cuda::ceil_div(num_items, sm_count * block_threads * max_occupancy)));
    const int items_per_thread_clamped = ::cuda::std::clamp(
      items_per_thread_evenly_spread, +wrapped_policy.MinItemsPerThread(), +wrapped_policy.MaxItemsPerThread());
    return items_per_thread_clamped;
  }

  template <typename ActivePolicy, typename SMemFunc>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE auto
  configure_async_kernel(int alignment, SMemFunc smem_for_tile_size) -> cuda_expected<
    ::cuda::std::
      tuple<THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron, decltype(kernel_source.TransformKernel()), int>>
  {
    // Benchmarking shows that even for a few iteration, this loop takes around 4-7 us, so should not be a concern.
    using policy_t              = typename ActivePolicy::algo_policy;
    constexpr int block_threads = policy_t::block_threads;
    _CCCL_ASSERT(block_threads % alignment == 0,
                 "block_threads needs to be a multiple of the copy alignment"); // then tile_size is a multiple
    auto determine_element_counts = [&]() -> cuda_expected<async_config> {
      const auto max_smem = get_max_shared_memory();
      if (!max_smem)
      {
        return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(max_smem.error());
      }

      int sm_count = 0;
      auto error   = CubDebug(launcher_factory.MultiProcessorCount(sm_count));
      if (error != cudaSuccess)
      {
        return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
      }

      // Increase the number of output elements per thread until we reach the required bytes in flight. This computation
      // MUST NOT depend on any runtime state of the current API invocation (like num_items), since the result will be
      // cached.
      async_config last_config{};
      for (int items_per_thread = +policy_t::min_items_per_thread; items_per_thread <= +policy_t::max_items_per_thread;
           ++items_per_thread)
      {
        // ensures the loop below runs at least once
        static_assert(policy_t::min_items_per_thread <= policy_t::max_items_per_thread);

        const int tile_size = block_threads * items_per_thread;
        const int smem_size = smem_for_tile_size(tile_size, block_threads, alignment);
        if (smem_size > *max_smem)
        {
          // assert should be prevented by smem check in policy
          _CCCL_ASSERT(last_config.items_per_thread > 0, "min_items_per_thread exceeds available shared memory");
          return last_config;
        }

        int max_occupancy = 0;
        error             = CubDebug(
          launcher_factory.MaxSmOccupancy(max_occupancy, kernel_source.TransformKernel(), block_threads, smem_size));
        if (error != cudaSuccess)
        {
          return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
        }

        const auto config = async_config{items_per_thread, max_occupancy, sm_count};

        const int bytes_in_flight_SM = max_occupancy * tile_size * kernel_source.LoadedBytesPerIteration();
        if (ActivePolicy::min_bif <= bytes_in_flight_SM)
        {
          return config;
        }

        last_config = config;
      }
      return last_config;
    };
    cuda_expected<async_config> config = [&]() {
      NV_IF_TARGET(NV_IS_HOST,
                   (static auto cached_config = determine_element_counts(); return cached_config;),
                   (
                     // we cannot cache the determined element count in device code
                     return determine_element_counts();));
    }();
    if (!config)
    {
      return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(config.error());
    }
    _CCCL_ASSERT(config->items_per_thread > 0, "");
    _CCCL_ASSERT(config->items_per_thread > 0, "");
    _CCCL_ASSERT((config->items_per_thread * block_threads) % alignment == 0, "");

    const int ipt =
      spread_out_items_per_thread(ActivePolicy{}, config->items_per_thread, config->sm_count, config->max_occupancy);
    const int tile_size = block_threads * ipt;
    const int smem_size = smem_for_tile_size(tile_size, block_threads, alignment);
    _CCCL_ASSERT((sizeof...(RandomAccessIteratorsIn) == 0) != (smem_size != 0), ""); // logical xor

    const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{tile_size}));
    // config->smem_size is 16 bytes larger than needed for UBLKCP because it's the total SMEM size, but 16 bytes are
    // occupied by static shared memory and padding. But let's not complicate things.
    return ::cuda::std::make_tuple(
      launcher_factory(grid_dim, block_threads, smem_size, stream, true), kernel_source.TransformKernel(), ipt);
  }

  template <typename ActivePolicy, typename SMemFunc, std::size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_async_algorithm(int alignment, SMemFunc smem_for_tile_size, cuda::std::index_sequence<Is...>)
  {
    auto ret = configure_async_kernel<ActivePolicy>(alignment, smem_for_tile_size);
    if (!ret)
    {
      return ret.error();
    }
    auto [launcher, kernel, items_per_thread] = *ret;
    return launcher.doit(
      kernel,
      num_items,
      items_per_thread,
      false,
      op,
      THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(out),
      make_aligned_base_ptr_kernel_arg(
        THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(::cuda::std::get<Is>(in)), alignment)...);
  }

  template <typename ActivePolicy, size_t... Is>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  invoke_prefetch_or_vectorized_algorithm(::cuda::std::index_sequence<Is...>, ActivePolicy active_policy)
  {
    auto wrapped_policy     = detail::transform::MakeTransformPolicyWrapper(active_policy);
    const int block_threads = wrapped_policy.BlockThreads();

    auto determine_config = [&]() -> cuda_expected<prefetch_config> {
      int max_occupancy = 0;
      auto error =
        CubDebug(launcher_factory.MaxSmOccupancy(max_occupancy, kernel_source.TransformKernel(), block_threads, 0));
      if (error != cudaSuccess)
      {
        return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
      }
      int sm_count = 0;
      error        = CubDebug(launcher_factory.MultiProcessorCount(sm_count));
      if (error != cudaSuccess)
      {
        return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
      }
      return prefetch_config{max_occupancy, sm_count};
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

    auto can_vectorize = false;
    // the policy already handles the compile-time checks if we can vectorize. Do the remaining alignment check here
#ifdef CCCL_C_EXPERIMENTAL
    if (Algorithm::vectorized == wrapped_policy.GetAlgorithm())
#else // CCCL_C_EXPERIMENTAL
    if constexpr (Algorithm::vectorized == ActivePolicy::algorithm)
#endif // CCCL_C_EXPERIMENTAL
    {
      const int alignment     = wrapped_policy.LoadStoreWordSize();
      auto is_pointer_aligned = [&](auto it) {
        if constexpr (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<decltype(it)>)
        {
          return THRUST_NS_QUALIFIER::detail::util::is_aligned(
            THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(it), alignment);
        }
        else
        {
          return true; // fancy iterators are aligned, since the vectorized kernel chooses a different code path
        }
      };
      can_vectorize = (is_pointer_aligned(::cuda::std::get<Is>(in)) && ...) && is_pointer_aligned(out);
    }

    const int ipt = [&] {
#ifdef CCCL_C_EXPERIMENTAL
      if (Algorithm::vectorized == wrapped_policy.GetAlgorithm())
#else // CCCL_C_EXPERIMENTAL
      if constexpr (Algorithm::vectorized == ActivePolicy::algorithm)
#endif // CCCL_C_EXPERIMENTAL
      {
        if (can_vectorize)
        {
          return wrapped_policy.ItemsPerThreadForVectorizedPath();
        }
      }
      // otherwise, set up the prefetch kernel

      auto loaded_bytes_per_iter = kernel_source.LoadedBytesPerIteration();
      // choose items per thread to reach minimum bytes in flight
      const int items_per_thread =
        loaded_bytes_per_iter == 0
          ? wrapped_policy.ItemsPerThreadNoInput()
          : ::cuda::ceil_div(wrapped_policy.MinBif(), config->max_occupancy * block_threads * loaded_bytes_per_iter);

      // but also generate enough blocks for full occupancy to optimize small problem sizes, e.g., 2^16/2^20 elements
      return spread_out_items_per_thread(active_policy, items_per_thread, config->sm_count, config->max_occupancy);
    }();
    const int tile_size = block_threads * ipt;
    const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{tile_size}));
    return CubDebug(
      launcher_factory(grid_dim, block_threads, 0, stream, true)
        .doit(kernel_source.TransformKernel(),
              num_items,
              ipt,
              can_vectorize,
              op,
              THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(out),
              kernel_source.MakeIteratorKernelArg(
                THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(::cuda::std::get<Is>(in)))...));
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT active_policy = {})
  {
    auto wrapped_policy = detail::transform::MakeTransformPolicyWrapper(active_policy);
    const auto seq      = ::cuda::std::index_sequence_for<RandomAccessIteratorsIn...>{};
    if constexpr (Algorithm::ublkcp == wrapped_policy.GetAlgorithm())
    {
      return invoke_async_algorithm<ActivePolicyT>(
        bulk_copy_align, &bulk_copy_smem_for_tile_size<RandomAccessIteratorsIn...>, seq);
    }
    else if constexpr (Algorithm::memcpy_async == wrapped_policy.GetAlgorithm())
    {
      return invoke_async_algorithm<ActivePolicyT>(
        ldgsts_size_and_align, &memcpy_async_smem_for_tile_size<RandomAccessIteratorsIn...>, seq);
    }
    else
    {
      return invoke_prefetch_or_vectorized_algorithm(seq, active_policy);
    }
  }

  template <typename MaxPolicyT = typename PolicyHub::max_policy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(
    ::cuda::std::tuple<RandomAccessIteratorsIn...> in,
    RandomAccessIteratorOut out,
    Offset num_items,
    TransformOp op,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    if (num_items == 0)
    {
      return cudaSuccess;
    }

    int ptx_version = 0;
    auto error      = CubDebug(launcher_factory.PtxVersion(ptx_version));
    if (cudaSuccess != error)
    {
      return error;
    }

    dispatch_t dispatch{
      ::cuda::std::move(in),
      ::cuda::std::move(out),
      num_items,
      ::cuda::std::move(op),
      bulk_copy_alignment(ptx_version),
      stream,
      kernel_source,
      launcher_factory};
    return CubDebug(max_policy.Invoke(ptx_version, dispatch));
  }
};
} // namespace detail::transform
CUB_NAMESPACE_END
