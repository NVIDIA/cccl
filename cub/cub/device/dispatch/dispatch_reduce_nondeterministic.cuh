// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//! @file
//! cub::DeviceReduceNondeterministic provides device-wide, parallel operations for computing a reduction
//! across a sequence of data items residing within device-accessible memory. The reduction is not guaranteed
//! to be deterministic.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/type_traits.cuh> // for cub::detail::invoke_result_t
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/grid/grid_even_share.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh> // for cub::detail::non_void_value_t, cub::detail::value_t

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__iterator/iterator_traits.h>

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{
template <typename MaxPolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT,
          typename AccumT,
          typename TransformOpT>
struct DeviceReduceNondeterministicKernelSource
{
  CUB_DEFINE_KERNEL_GETTER(
    AtomicKernel,
    NondeterministicDeviceReduceAtomicKernel<
      MaxPolicyT,
      InputIteratorT,
      OutputIteratorT,
      OffsetT,
      ReductionOpT,
      AccumT,
      InitT,
      TransformOpT>);

  CUB_RUNTIME_FUNCTION static constexpr size_t InitSize()
  {
    return sizeof(InitT);
  }
};

//! @brief Utility class for dispatching the appropriately-tuned kernels for device-wide reduction
//!
//! @tparam InputIteratorT
//!   Random-access input iterator type for reading input items @iterator
//!
//! @tparam OutputIteratorT
//!   Output iterator type for recording the reduced aggregate @iterator
//!
//! @tparam OffsetT
//!   Signed integer type for global offsets
//!
//! @tparam ReductionOpT
//!   Binary reduction functor type having member `auto operator()(const T &a, const U &b)`
//!
//! @tparam InitT
//!   Initial value type
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitT = non_void_value_t<OutputIteratorT, ::cuda::std::iter_value_t<InputIteratorT>>,
          typename AccumT = ::cuda::std::__accumulator_t<ReductionOpT, ::cuda::std::iter_value_t<InputIteratorT>, InitT>,
          typename TransformOpT = ::cuda::std::identity,
          typename ArchPolicies = arch_policies_from_types<AccumT, OffsetT, ReductionOpT>,
          typename KernelSource = DeviceReduceNondeterministicKernelSource<
            ArchPolicies,
            InputIteratorT,
            OutputIteratorT,
            OffsetT,
            ReductionOpT,
            InitT,
            AccumT,
            TransformOpT>,
          typename KernelLauncherFactory = TripleChevronFactory>
#if _CCCL_HAS_CONCEPTS()
  requires detail::reduce::reduce_policy_hub<ArchPolicies>
#endif // _CCCL_HAS_CONCEPTS()
struct dispatch_nondeterministic_t
{
  static_assert(detail::is_cuda_std_plus_v<ReductionOpT>,
                "Only plus is currently supported in nondeterministic reduce");
  //---------------------------------------------------------------------------
  // Problem state
  //---------------------------------------------------------------------------

  //! Device-accessible allocation of temporary storage. When `nullptr`, the required allocation
  //! size is written to `temp_storage_bytes` and no work is done.
  void* d_temp_storage;

  //! Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  //! Pointer to the input sequence of data items
  InputIteratorT d_in;

  //! Pointer to the output aggregate
  OutputIteratorT d_out;

  //! Total number of input items (i.e., length of `d_in`)
  OffsetT num_items;

  //! Binary reduction functor
  ReductionOpT reduction_op;

  //! The initial value of the reduction
  InitT init;

  //! CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  TransformOpT transform_op;

  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  //! @brief Invoke a single block block to reduce in-core
  //!
  //! @tparam AtomicKernelT
  //!   Function type of cub::DeviceReduceAtomicKernel
  //!
  //! @param[in] last_block_kernel
  //!   Kernel function pointer to parameterization of cub::DeviceReduceLastBlockKernel
  template <typename AtomicKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokeAtomicKernel(AtomicKernelT atomic_kernel, detail::reduce::reduce_arch_policy active_policy)
  {
    // This memory is not actually needed but we keep it to make sure the API is consistent
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    // Init regular kernel configuration
    const int tile_size = active_policy.reduce_nondeterministic_policy.block_threads
                        * active_policy.reduce_nondeterministic_policy.items_per_thread;
    int sm_occupancy;
    if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
          sm_occupancy, atomic_kernel, active_policy.reduce_nondeterministic_policy.block_threads)))
    {
      return error;
    }
    int sm_count;
    if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(sm_count)))
    {
      return error;
    }
    const int reduce_device_occupancy = sm_occupancy * sm_count;

    // Even-share work distribution
    const int max_blocks = reduce_device_occupancy * subscription_factor;
    GridEvenShare<OffsetT> even_share;
    even_share.DispatchInit(num_items, max_blocks, tile_size);
    // Get grid size for nondeterministic_device_reduce_atomic_kernel
    const int reduce_grid_size = ::cuda::std::max(1, even_share.grid_size);

    if (const auto error = CubDebug(launcher_factory.MemsetAsync(d_out, 0, kernel_source.InitSize(), stream)))
    {
      return error;
    }

// Log nondeterministic_device_reduce_atomic_kernel configuration
#ifdef CUB_DEBUG_LOG
    _CubLog("Invoking NondeterministicDeviceReduceAtomicKernel<<<%llu, %d, 0, %p>>>(), %d items "
            "per thread, %d SM occupancy\n",
            (unsigned long long) reduce_grid_size,
            active_policy.reduce_nondeterministic_policy.block_threads,
            (long long) stream,
            active_policy.reduce_nondeterministic_policy.items_per_thread,
            sm_occupancy);
#endif // CUB_DEBUG_LOG

    // Invoke NondeterministicDeviceReduceAtomicKernel
    launcher_factory(reduce_grid_size, active_policy.reduce_nondeterministic_policy.block_threads, 0, stream)
      .doit(atomic_kernel, d_in, d_out, num_items, even_share, reduction_op, init, transform_op);

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }
    // Sync the stream if specified to flush runtime errors
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }
    return cudaSuccess;
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  //! @brief Internal dispatch routine for computing a device-wide reduction
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the required allocation
  //!   size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] init
  //!   The initial value of the reduction
  //!
  //! @param[in] stream
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    OffsetT num_items,
    ReductionOpT reduction_op,
    InitT init                             = {},
    cudaStream_t stream                    = {},
    TransformOpT transform_op              = {},
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
  {
    // Get PTX version
    ::cuda::arch_id arch_id{};
    if (const auto error = CubDebug(launcher_factory.PtxArchId(arch_id)))
    {
      return error;
    }

    const detail::reduce::reduce_arch_policy active_policy = ArchPolicies{}(arch_id);
#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
    NV_IF_TARGET(
      NV_IS_HOST,
      (std::stringstream ss; ss << active_policy; _CubLog(
         "Dispatching DeviceReduceNondeterministic to arch %d with tuning: %s\n", (int) arch_id, ss.str().c_str());))
#endif // !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)

    // Create dispatch functor
    dispatch_nondeterministic_t dispatch{
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_items,
      reduction_op,
      init,
      stream,
      transform_op,
      kernel_source,
      launcher_factory};

    return dispatch.InvokeAtomicKernel(kernel_source.AtomicKernel(), active_policy);
  }
};
} // namespace detail::reduce

CUB_NAMESPACE_END
