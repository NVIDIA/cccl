//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/stream/interfaces/slice_reduction_ops.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

struct OR_op
{
  static void init_host(bool& out)
  {
    out = false;
  };
  static __device__ void init_gpu(bool& out)
  {
    out = false;
  };
  static void op_host(const bool& in, bool& inout)
  {
    inout |= in;
  };
  static __device__ void op_gpu(const bool& in, bool& inout)
  {
    inout |= in;
  };
};

int main()
{
  stream_ctx ctx;

  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  bool A[4] = {false, false, true, false};
  bool B[4] = {false, false, false, false};
  bool C[4] = {true, false, false, true};

  auto lA = ctx.logical_data(make_slice(&A[0], 4));
  auto lB = ctx.logical_data(make_slice(&B[0], 4));
  auto lC = ctx.logical_data(make_slice(&C[0], 4));

  auto op = std::make_shared<slice_reduction_op<bool, 1, OR_op>>();

  // C |= A
  ctx.task(lC.relaxed(op), lA.read())->*[](auto stream, auto sC, auto sA) {
    cudaMemcpyAsync(sC.data_handle(), sA.data_handle(), sA.extent(0) * sizeof(bool), cudaMemcpyDeviceToDevice, stream);
  };

  // C |= B
  ctx.task(lC.relaxed(op), lB.read())->*[](auto stream, auto sC, auto sB) {
    cudaMemcpyAsync(sC.data_handle(), sB.data_handle(), sB.extent(0) * sizeof(bool), cudaMemcpyDeviceToDevice, stream);
  };

  ctx.task(exec_place::host, lC.read())->*[](auto stream, auto sC) {
    cuda_safe_call(cudaStreamSynchronize(stream));
    for (size_t i = 0; i < sC.extent(0); i++)
    {
      // fprintf(stderr, "RESULT C[i] = %d\n", sC(i));
    }
  };

  ctx.finalize();
}
