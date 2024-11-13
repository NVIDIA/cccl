//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

static __global__ void scalar_div(const double* a, const double* b, double* c)
{
  *c = (*a) / (*b);
}

static __global__ void scalar_minus(const double* a, double* res)
{
  *res = -(*a);
}

/**
 * This class is an example of class to issue tasks when accessing a scalar
 * value.
 *
 * This is not meant to be the most efficient approach, but this is
 * supposedly convenient.
 *
 */
template <typename Ctx>
class scalar
{
public:
  scalar(Ctx* ctx, bool is_tmp = false)
      : ctx(ctx)
  {
    size_t s = sizeof(double);

    if (is_tmp)
    {
      // There is no physical backing for this temporary vector
      h_addr = NULL;
    }
    else
    {
      h_addr = (double*) malloc(s);
      cuda_safe_call(cudaHostRegister(h_addr, s, cudaHostRegisterPortable));
    }

    data_place d = is_tmp ? data_place::invalid : data_place::host;
    handle       = ctx->logical_data(make_slice(h_addr), d);
  }

  // Copy constructor
  scalar(const scalar& a)
      : ctx(a.ctx)
  {
    h_addr = NULL;
    handle = ctx->logical_data(make_slice((double*) nullptr));

    ctx->task(handle.write(), a.handle.read())->*[](cudaStream_t stream, auto dst, auto src) {
      // There are likely much more efficient ways.
      cuda_safe_call(
        cudaMemcpyAsync(dst.data_handle(), src.data_handle(), sizeof(double), cudaMemcpyDeviceToDevice, stream));
    };
  }

  scalar operator/(scalar const& rhs) const
  {
    // Submit a task that computes this/rhs
    scalar res(ctx);

    ctx->task(handle.read(), rhs.handle.read(), res.handle.write())
        ->*[](cudaStream_t stream, auto x, auto y1, auto result) {
              scalar_div<<<1, 1, 0, stream>>>(x.data_handle(), y1.data_handle(), result.data_handle());
            };

    return res;
  }

  scalar operator-() const
  {
    // Submit a task that computes -s
    scalar res(ctx);
    ctx->task(handle.read(), res.handle.write())->*[](cudaStream_t stream, auto x, auto result) {
      scalar_minus<<<1, 1, 0, stream>>>(x.data_handle(), result.data_handle());
    };

    return res;
  }

  Ctx* ctx;
  mutable logical_data<slice<double, 0>> handle;
  double* h_addr;
};

template <typename Ctx>
void run()
{
  Ctx ctx;
  scalar a(&ctx);
  scalar b(&ctx);

  *a.h_addr = 42.0;
  *b.h_addr = 12.3;

  scalar c = (-a) / b;

  ctx.host_launch(c.handle.read())->*[](auto x) {
    EXPECT(fabs(*x.data_handle() - (-42.0) / 12.3) < 0.001);
  };

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
}
