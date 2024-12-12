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

template <typename OutT, typename Ctx, typename T, typename BinaryOp>
auto reduce(Ctx& ctx, logical_data<T> data, BinaryOp&& op, OutT init_val)
{
  auto result = ctx.logical_data(shape_of<scalar_view<OutT>>());

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Reduce(
    d_temp_storage,
    temp_storage_bytes,
    (OutT*) nullptr,
    (OutT*) nullptr,
    data.shape().size(),
    OpWrapper<BinaryOp>(op),
    init_val,
    0);

  auto ltemp = ctx.logical_data(shape_of<slice<char>>(temp_storage_bytes));

  ctx.task(data.read(), result.write(), ltemp.write())
      ->*[&op, init_val](cudaStream_t stream, auto d_data, auto d_result, auto d_temp) {
            size_t tmp_size = d_temp.size();
            cub::DeviceReduce::Reduce(
              (void*) d_temp.data_handle(),
              tmp_size,
              (OutT*) d_data.data_handle(),
              (OutT*) d_result.addr,
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
  auto lresult = reduce<int>(
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
