// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data
//! items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_find.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/__nvtx/nvtx.h>

#include <cassert>

CUB_NAMESPACE_BEGIN

struct DeviceFind
{
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t FindIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFind::FindIf");

    // Signed integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchFind<InputIteratorT, OutputIteratorT, OffsetT, ScanOpT>::Dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, static_cast<OffsetT>(num_items), scan_op, stream);
  }
};

CUB_NAMESPACE_END
