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
 *
 * @brief This test illustrates how we can use multiple reserved::launch in a single task on different pieces of data
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int X0(int i)
{
  return i * i + 12;
}

int main()
{
  stream_ctx ctx;

  const int N = 16;
  int X[N], Y[N], Z[N];

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = X0(ind);
    Y[ind] = 0;
    Z[ind] = 0;
  }

  auto handle_X = ctx.logical_data(X, {N});
  auto handle_Y = ctx.logical_data(Y, {N});
  auto handle_Z = ctx.logical_data(Z, {N});

  ctx.task(handle_X.read(), handle_Y.write(), handle_Z.write())
      ->*[](cudaStream_t s, slice<int> x, slice<int> y, slice<int> z) {
            std::vector<cudaStream_t> streams;
            streams.push_back(s);
            auto spec = par(1024);
            reserved::launch(spec, exec_place::current_device(), streams, std::tuple{x, y})
                ->*[] _CCCL_DEVICE(auto t, slice<int> x, slice<int> y) {
                      size_t tid      = t.rank();
                      size_t nthreads = t.size();
                      for (size_t ind = tid; ind < N; ind += nthreads)
                      {
                        y(ind) = 2 * x(ind);
                      }
                    };

            reserved::launch(spec, exec_place::current_device(), streams, std::tuple{y, z})
                ->*[] _CCCL_DEVICE(auto t, slice<int> y, slice<int> z) {
                      size_t tid      = t.rank();
                      size_t nthreads = t.size();
                      for (size_t ind = tid; ind < N; ind += nthreads)
                      {
                        z(ind) = 3 * y(ind);
                      }
                    };
          };

  ctx.finalize();

  for (size_t ind = 0; ind < N; ind++)
  {
    assert(Y[ind] == 2 * X[ind]);
    assert(Z[ind] == 3 * Y[ind]);
  }
}
