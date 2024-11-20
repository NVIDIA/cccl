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
 * @brief An AXPY kernel implemented using the parallel_for construct
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

double X0(int i)
{
  return sin((double) i);
}

double Y0(int i)
{
  return cos((double) i);
}

template <typename T>
class sum {
public:
    static __host__ __device__ void init_op(T &dst) {
        dst = static_cast<T>(0);
    }

    static __host__ __device__ void apply_op(T &dst, const T &src) {
        dst += src;
    }
};

template <typename T>
class maxval {
public:
    static __host__ __device__ void init_op(T &dst) {
        dst = ::std::numeric_limits<T>::lowest();
    }

    static __host__ __device__ void apply_op(T &dst, const T &src) {
        dst = ::std::max(dst, src);
    }
};



int main()
{
  context ctx;
  const size_t N = 16;
  double X[N], Y[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  double alpha = 3.14;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  auto lsum = ctx.logical_data(shape_of<slice<double>>(1));
  auto lsum2 = ctx.logical_data(shape_of<scalar<double>>());
  auto lmax = ctx.logical_data(shape_of<scalar<double>>());

  /* Compute Y = Y + alpha X */
  ctx.parallel_for(lY.shape(), lX.read(), lY.rw(), lsum.write(), lsum2.reduce(sum<double>{}), lmax.reduce(maxval<double>{}))->*[alpha] __device__(size_t i, auto dX, auto dY, auto sum, double &sum2, double &maxval) {
    dY(i) += alpha * dX(i);
    sum2 += dY(i);
    maxval = ::std::max(dY(i), maxval);
    atomicAdd(sum.data_handle(), dY(i));
  };

  ctx.host_launch(lsum.read(), lsum2.read(), lmax.read())->*[](auto sum, scalar<double> sum2, scalar<double> max) {
      fprintf(stderr, "REF SUM ... %lf\n", sum(0));
      fprintf(stderr, "REDUX SUM2 ... %lf\n", *(sum2.addr));
      fprintf(stderr, "MAX VAL ... %lf\n", *(max.addr));
      assert(fabs(sum(0) - *(sum2.addr)) < 0.0001);
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(Y[i] - (Y0(i) + alpha * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }

}
