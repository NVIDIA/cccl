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
 * @brief Example of reduction implementing using CUB
 */

#include <cub/cub.cuh>

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

template <typename BinaryOp>
struct OpWrapper
{
  OpWrapper(BinaryOp _op)
      : op(mv(_op)) {};

  template <typename T>
  __device__ __forceinline__ T operator()(const T& a, const T& b) const
  {
    return op(a, b);
  }

  BinaryOp op;
};

template <typename Ctx, typename InT, typename OutT, typename BinaryOp>
void exclusive_scan(
  Ctx& ctx, logical_data<slice<InT>> in_data, logical_data<slice<OutT>> out_data, BinaryOp&& op, OutT init_val)
{
  size_t nitems = in_data.shape().size();

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveScan(
    d_temp_storage,
    temp_storage_bytes,
    (InT*) nullptr,
    (OutT*) nullptr,
    OpWrapper<BinaryOp>(op),
    init_val,
    in_data.shape().size(),
    0);

  auto ltemp = ctx.logical_data(shape_of<slice<char>>(temp_storage_bytes));

  ctx.task(in_data.read(), out_data.write(), ltemp.write())
      ->*[&op, init_val, nitems, temp_storage_bytes](cudaStream_t stream, auto d_in, auto d_out, auto d_temp) {
            size_t d_temp_size = shape(d_temp).size();
            cub::DeviceScan::ExclusiveScan(
              (void*) d_temp.data_handle(),
              d_temp_size,
              (InT*) d_in.data_handle(),
              (OutT*) d_out.data_handle(),
              OpWrapper<BinaryOp>(op),
              init_val,
              nitems,
              stream);
          };
}

template <typename Ctx>
void run()
{
  Ctx ctx;

  const size_t N = 1024 * 16;

  ::std::vector<int> X(N);
  ::std::vector<int> out(N);

  ::std::vector<int> ref_out(N);

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = rand() % N;

    // compute the exclusive sum of X
    ref_out[ind] = (ind == 0) ? 0 : (X[ind - 1] + ref_out[ind - 1]);
  }

  auto lX   = ctx.logical_data(X.data(), {N});
  auto lout = ctx.logical_data(out.data(), {N});

  exclusive_scan(
    ctx,
    lX,
    lout,
    [] __device__(const int& a, const int& b) {
      return a + b;
    },
    0);

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    _CCCL_ASSERT(ref_out[i] == out[i], "Incorrect result");
  }
}

int main()
{
  run<stream_ctx>();
  // run<graph_ctx>();
}
