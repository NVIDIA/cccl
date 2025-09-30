// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//! @file
//! cub::DeviceCopy provides device-wide, parallel operations for copying data.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/dispatch_batch_memcpy.cuh>
#include <cub/device/dispatch/dispatch_copy_mdspan.cuh>
#include <cub/device/dispatch/tuning/tuning_batch_memcpy.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/cstdint>
#include <cuda/std/mdspan>

CUB_NAMESPACE_BEGIN

//! @brief cub::DeviceCopy provides device-wide, parallel operations for copying data.
struct DeviceCopy
{
  //! @rst
  //! Copies data from a batch of given source ranges to their corresponding destination ranges.
  //!
  //! .. note::
  //!
  //!    If any input range aliases any output range the behavior is undefined.
  //!    If any output range aliases another output range the behavior is undefined.
  //!    Input ranges can alias one another.
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates usage of DeviceCopy::Batched to perform a DeviceRunLength Decode operation.
  //!
  //! .. code-block:: c++
  //!
  //!    struct GetIteratorToRange
  //!    {
  //!      __host__ __device__ __forceinline__ auto operator()(uint32_t index)
  //!      {
  //!        return thrust::make_constant_iterator(d_data_in[index]);
  //!      }
  //!      int32_t *d_data_in;
  //!    };
  //!
  //!    struct GetPtrToRange
  //!    {
  //!      __host__ __device__ __forceinline__ auto operator()(uint32_t index)
  //!      {
  //!        return d_data_out + d_offsets[index];
  //!      }
  //!      int32_t *d_data_out;
  //!      uint32_t *d_offsets;
  //!    };
  //!
  //!    struct GetRunLength
  //!    {
  //!      __host__ __device__ __forceinline__ uint32_t operator()(uint32_t index)
  //!      {
  //!        return d_offsets[index + 1] - d_offsets[index];
  //!      }
  //!      uint32_t *d_offsets;
  //!    };
  //!
  //!    uint32_t num_ranges = 5;
  //!    int32_t *d_data_in;           // e.g., [4, 2, 7, 3, 1]
  //!    int32_t *d_data_out;          // e.g., [0,                ...               ]
  //!    uint32_t *d_offsets;          // e.g., [0, 2, 5, 6, 9, 14]
  //!
  //!    // Returns a constant iterator to the element of the i-th run
  //!    thrust::counting_iterator<uint32_t> iota(0);
  //!    auto iterators_in = thrust::make_transform_iterator(iota, GetIteratorToRange{d_data_in});
  //!
  //!    // Returns the run length of the i-th run
  //!    auto sizes = thrust::make_transform_iterator(iota, GetRunLength{d_offsets});
  //!
  //!    // Returns pointers to the output range for each run
  //!    auto ptrs_out = thrust::make_transform_iterator(iota, GetPtrToRange{d_data_out, d_offsets});
  //!
  //!    // Determine temporary device storage requirements
  //!    void *d_temp_storage      = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, iterators_in, ptrs_out, sizes,
  //!    num_ranges);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run batched copy algorithm (used to perform runlength decoding)
  //!    cub::DeviceCopy::Batched(d_temp_storage, temp_storage_bytes, iterators_in, ptrs_out, sizes,
  //!    num_ranges);
  //!
  //!    // d_data_out       <-- [4, 4, 2, 2, 2, 7, 3, 3, 3, 1, 1, 1, 1, 1]
  //!
  //! @endrst
  //!
  //! @tparam InputIt
  //!   **[inferred]** Device-accessible random-access input iterator type providing the iterators to the source ranges
  //!
  //! @tparam OutputIt
  //!  **[inferred]** Device-accessible random-access input iterator type providing the iterators to
  //!  the destination ranges
  //!
  //! @tparam SizeIteratorT
  //!   **[inferred]** Device-accessible random-access input iterator type providing the number of items to be
  //!   copied for each pair of ranges
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] input_it
  //!   Device-accessible iterator providing the iterators to the source ranges
  //!
  //! @param[in] output_it
  //!   Device-accessible iterator providing the iterators to the destination ranges
  //!
  //! @param[in] sizes
  //!   Device-accessible iterator providing the number of elements to be copied for each pair of ranges
  //!
  //! @param[in] num_ranges
  //!   The total number of range pairs
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIt, typename OutputIt, typename SizeIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t Batched(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIt input_it,
    OutputIt output_it,
    SizeIteratorT sizes,
    ::cuda::std::int64_t num_ranges,
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceCopy::Batched");

    // Integer type large enough to hold any offset in [0, num_thread_blocks_launched), where a safe
    // upper bound on num_thread_blocks_launched can be assumed to be given by
    // IDIV_CEIL(num_ranges, 64)
    using BlockOffsetT = uint32_t;

    return detail::DispatchBatchMemcpy<InputIt, OutputIt, SizeIteratorT, BlockOffsetT, CopyAlg::Copy>::Dispatch(
      d_temp_storage, temp_storage_bytes, input_it, output_it, sizes, num_ranges, stream);
  }

  //! @rst
  //! Copies data from a multidimensional source mdspan to a destination mdspan.
  //!
  //! This function performs a parallel copy operation between two mdspan objects with potentially different layouts but
  //! identical extents. The copy operation handles arbitrary-dimensional arrays and automatically manages layout
  //! transformations.
  //!
  //! Preconditions
  //! +++++++++++++
  //!
  //!    * The source and destination mdspans must have identical extents (same ranks and sizes).
  //!    * The source and destination mdspans data handle must not be nullptr if the size is not 0.
  //!    * The underlying memory of the source and destination must not overlap.
  //!    * Both mdspans must point to device memory.
  //!
  //! Snippet
  //! +++++++
  //!
  //! The code snippet below illustrates usage of DeviceCopy::Copy to copy between mdspans.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_copy_mdspan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin copy-mdspan-example-op
  //!     :end-before: example-end copy-mdspan-example-op
  //!
  //! @endrst
  //!
  //! @tparam T_In
  //!   **[inferred]** The element type of the source mdspan
  //!
  //! @tparam Extents_In
  //!   **[inferred]** The extents type of the source mdspan
  //!
  //! @tparam Layout_In
  //!   **[inferred]** The layout type of the source mdspan
  //!
  //! @tparam Accessor_In
  //!   **[inferred]** The accessor type of the source mdspan
  //!
  //! @tparam T_Out
  //!   **[inferred]** The element type of the destination mdspan
  //!
  //! @tparam Extents_Out
  //!   **[inferred]** The extents type of the destination mdspan
  //!
  //! @tparam Layout_Out
  //!   **[inferred]** The layout type of the destination mdspan
  //!
  //! @tparam Accessor_Out
  //!   **[inferred]** The accessor type of the destination mdspan
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] mdspan_in
  //!   Source mdspan containing the data to be copied
  //!
  //! @param[in] mdspan_out
  //!   Destination mdspan where the data will be copied
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  //!
  //! @returns
  //!   @rst
  //!   **cudaSuccess** on success, **cudaErrorInvalidValue** if mdspan extents don't match, or error code on failure
  //!   @endrst
  template <typename T_In,
            typename Extents_In,
            typename Layout_In,
            typename Accessor_In,
            typename T_Out,
            typename Extents_Out,
            typename Layout_Out,
            typename Accessor_Out>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  Copy(void* d_temp_storage,
       size_t& temp_storage_bytes,
       ::cuda::std::mdspan<T_In, Extents_In, Layout_In, Accessor_In> mdspan_in,
       ::cuda::std::mdspan<T_Out, Extents_Out, Layout_Out, Accessor_Out> mdspan_out,
       ::cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceCopy::Copy");
    _CCCL_ASSERT(mdspan_in.extents() == mdspan_out.extents(), "mdspan extents must be equal");
    _CCCL_ASSERT((mdspan_in.data_handle() != nullptr && mdspan_out.data_handle() != nullptr) || mdspan_in.size() == 0,
                 "mdspan data handle must not be nullptr if the size is not 0");
    // Check for memory overlap between input and output mdspans
    if (mdspan_in.size() != 0)
    {
      auto in_start  = mdspan_in.data_handle();
      auto in_end    = in_start + mdspan_in.mapping().required_span_size();
      auto out_start = mdspan_out.data_handle();
      auto out_end   = out_start + mdspan_out.mapping().required_span_size();
      // TODO(fbusato): replace with __are_ptrs_overlapping
      _CCCL_ASSERT(!(in_end >= out_start && out_end >= in_start), "mdspan memory ranges must not overlap");
    }
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return ::cudaSuccess;
    }
    return detail::copy_mdspan::copy(mdspan_in, mdspan_out, stream);
  }
};

CUB_NAMESPACE_END
