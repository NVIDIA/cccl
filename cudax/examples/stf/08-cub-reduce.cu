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

template <typename T, typename Ctx, typename BinaryOp>
auto reduce(Ctx& ctx, logical_data<slice<T>> data, BinaryOp&& op, T init_val)
{
  auto result = ctx.logical_data(shape_of<scalar_view<T>>());

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(
    d_temp_storage,
    temp_storage_bytes,
    (T*) nullptr,
    (T*) nullptr,
    data.shape().size(),
    OpWrapper<BinaryOp>(op),
    init_val,
    0);

  auto ltemp = ctx.logical_data(shape_of<slice<char>>(temp_storage_bytes));

  ctx.task(data.read(), result.write(), ltemp.write())
      ->*[&op, init_val, temp_storage_bytes](cudaStream_t stream, auto d_data, auto d_result, auto d_temp) {
            size_t d_temp_size = shape(d_temp).size();
            cub::DeviceReduce::Reduce(
              (void*) d_temp.data_handle(),
              d_temp_size,
              (T*) d_data.data_handle(),
              (T*) d_result.addr,
              shape(d_data).size(),
              OpWrapper<BinaryOp>(op),
              init_val,
              stream);
          };

  return result;
}

template <typename Ctx>
void run()
{
  Ctx ctx;

  const size_t N = 1024 * 16;

  int *X, ref_tot;

  X       = new int[N];
  ref_tot = 0;

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = rand() % N;
    ref_tot += X[ind];
  }

  auto values = ctx.logical_data(X, {N});

  // int should be deduced from "values"...
  auto lresult = reduce(
    ctx,
    values,
    [] __device__(const int& a, const int& b) {
      return a + b;
    },
    0);

  int result = ctx.wait(lresult);
  _CCCL_ASSERT(result == ref_tot, "Incorrect result");

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  // run<graph_ctx>();
}
