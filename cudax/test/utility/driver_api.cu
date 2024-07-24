//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#define LIBCUDACXX_ENABLE_EXCEPTIONS

#include <cuda/experimental/__utility/driver_api.cuh>

#include "../hierarchy/testing_common.cuh"

TEST_CASE("Call each one", "[driver api]") {
    cudaStream_t stream;
    // Assumes the ctx stack was empty or had one ctx, should be the case unless some other
    // test leaves 2+ ctxs on the stack

    // Pushes the primary context if the stack is empty
    CUDART(cudaStreamCreate(&stream));

    auto ctx = cuda::experimental::detail::driver::ctxGetCurrent();
    CUDAX_REQUIRE(ctx != nullptr);

    cuda::experimental::detail::driver::ctxPop();
    CUDAX_REQUIRE(cuda::experimental::detail::driver::ctxGetCurrent() == nullptr);

    cuda::experimental::detail::driver::ctxPush(ctx);
    CUDAX_REQUIRE(cuda::experimental::detail::driver::ctxGetCurrent() == ctx);

    cuda::experimental::detail::driver::ctxPush(ctx);
    CUDAX_REQUIRE(cuda::experimental::detail::driver::ctxGetCurrent() == ctx);

    cuda::experimental::detail::driver::ctxPop();
    CUDAX_REQUIRE(cuda::experimental::detail::driver::ctxGetCurrent() == ctx);


    auto stream_ctx = cuda::experimental::detail::driver::streamGetCtx(stream);
    CUDAX_REQUIRE(ctx == stream_ctx);

    CUDART(cudaStreamDestroy(stream));
}