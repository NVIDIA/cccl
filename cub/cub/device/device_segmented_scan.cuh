// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/dispatch_segmented_scan.cuh>

#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

struct DeviceSegmentedScan
{
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename ScanOpT,
            typename InitValueT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSegmentedScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ScanOpT scan_op,
    InitValueT init_value,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::ExclusiveSegmentedScan");

    using OffsetT =
      detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>;
    using integral_offset_check = ::cuda::std::is_integral<OffsetT>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return cub::DispatchSegmentedScan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      ScanOpT,
      detail::InputValue<InitValueT>>::
      Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_in_begin_offsets,
        d_in_end_offsets,
        d_out_begin_offsets,
        scan_op,
        detail::InputValue<InitValueT>(init_value),
        stream);
  }

  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename ScanOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ScanOpT scan_op,
    cudaStream_t stream = 0)
  {
    // defined in cub/config.cuh
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedScan");

    using OffsetT =
      detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>;
    using integral_offset_check = ::cuda::std::is_integral<OffsetT>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");

    return cub::DispatchSegmentedScan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      ScanOpT,
      NullType>::Dispatch(d_temp_storage,
                          temp_storage_bytes,
                          d_in,
                          d_out,
                          num_segments,
                          d_in_begin_offsets,
                          d_in_end_offsets,
                          d_out_begin_offsets,
                          scan_op,
                          NullType(),
                          stream);
  }

  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorInputT,
            typename EndOffsetIteratorInputT,
            typename BeginOffsetIteratorOutputT,
            typename ScanOpT,
            typename InitValueT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSegmentedScanInit(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorInputT d_in_begin_offsets,
    EndOffsetIteratorInputT d_in_end_offsets,
    BeginOffsetIteratorOutputT d_out_begin_offsets,
    ScanOpT scan_op,
    InitValueT init_value,
    cudaStream_t stream = 0)
  {
    // defined in cub/config.cuh
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedScan::InclusiveSegmentedScanInit");

    using OffsetT =
      detail::common_iterator_value_t<BeginOffsetIteratorInputT, EndOffsetIteratorInputT, BeginOffsetIteratorOutputT>;
    using integral_offset_check = ::cuda::std::is_integral<OffsetT>;

    static_assert(integral_offset_check::value, "Offset iterator value type should be integral.");
    static_assert(!::cuda::std::is_same_v<InitValueT, NullType>);

    using AccumT = ::cuda::std::__accumulator_t<ScanOpT, cub::detail::it_value_t<InputIteratorT>, InitValueT>;

    return cub::DispatchSegmentedScan<
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorInputT,
      EndOffsetIteratorInputT,
      BeginOffsetIteratorOutputT,
      ScanOpT,
      detail::InputValue<InitValueT>,
      AccumT,
      ForceInclusive::Yes>::Dispatch(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     num_segments,
                                     d_in_begin_offsets,
                                     d_in_end_offsets,
                                     d_out_begin_offsets,
                                     scan_op,
                                     detail::InputValue<InitValueT>(init_value),
                                     stream);
  }
};

CUB_NAMESPACE_END
