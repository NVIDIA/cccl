//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Utility to get the id of the CUDA device associated to a CUDA stream
 */

#pragma once

#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda::experimental::stf {

/**
 * @brief This computes the CUDA device in which the stream was created
 */
inline int get_device_from_stream(cudaStream_t stream) {
    // Convert the runtime API stream to a driver API structure
    auto stream_driver = CUstream(stream);

    CUcontext ctx;
    cuda_safe_call(cuStreamGetCtx(stream_driver, &ctx));

    // Query the context associated with a stream by using the underlying driver API
    CUdevice stream_dev;
    cuda_safe_call(cuCtxPushCurrent(ctx));
    cuda_safe_call(cuCtxGetDevice(&stream_dev));
    cuda_safe_call(cuCtxPopCurrent(&ctx));

    return int(stream_dev);
}

}  // end namespace cuda::experimental::stf
