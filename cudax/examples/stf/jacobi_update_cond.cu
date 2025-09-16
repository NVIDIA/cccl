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
 * @brief Jacobi method using the update_cond helper for clean condition management
 */

#include <cuda/experimental/stf.cuh>
#include <cuda/experimental/__stf/utility/stackable_ctx.cuh>

#include <iostream>

using namespace cuda::experimental::stf;

template <typename ...Deps>
class condition_update_scope {
public:
    condition_update_scope(stackable_ctx& ctx, cudaGraphConditionalHandle handle, Deps... deps) : ctx_(ctx), handle_(handle), tdeps(mv(deps)...) {
    }

    // Helper to extract data_t from dependency types
    template <typename T>
    using data_t_of = typename T::data_t;

      template <typename CondFunc>
      void operator->*(CondFunc&& cond_func)
      {
	   /* Build a cuda kernel from the deps then pass it a lambda that will
 * configure a call to the condition_update_kernel kernel */
           ::std::apply([this](auto&&... deps){
               return this->ctx_.cuda_kernel(deps...);
           }, tdeps)->*[cond_func=mv(cond_func), h=handle_](data_t_of<Deps>... args) {
                 return cuda_kernel_desc{condition_update_kernel<CondFunc, data_t_of<Deps>...>, 1, 1, 0, h, cond_func, args...};
           };

      }


private:
    stackable_ctx& ctx_;
    cudaGraphConditionalHandle handle_;
    ::std::tuple<::std::decay_t<Deps>...> tdeps;
};

class bla {
public:
    bla() = default;
private:
    __host__ __device__ void func() {}
    __host__ void hfunc() {}
    ::std::shared_ptr<int> ptr;
};

int main(int argc, char** argv)
{
  stackable_ctx ctx;

  size_t n   = 4096;
  size_t m   = 4096;
  double tol = 0.1;

  if (argc > 2)
  {
    n = atol(argv[1]);
    m = atol(argv[2]);
  }

  if (argc > 3)
  {
    tol = atof(argv[3]);
  }

  auto lA        = ctx.logical_data(shape_of<slice<double, 2>>(m, n));
  auto lAnew     = ctx.logical_data(lA.shape());
  auto lresidual = ctx.logical_data(shape_of<scalar_view<double>>());
  auto liter     = ctx.logical_data(shape_of<scalar_view<int>>());

  ctx.parallel_for(lA.shape(), lA.write(), lAnew.write()).set_symbol("init")->*
    [=] __device__(size_t i, size_t j, auto A, auto Anew) {
      A(i, j) = (i == j) ? 1.0 : -1.0;
      Anew(i, j) = A(i, j);
    };

  // Initialize iteration counter
  ctx.parallel_for(box(1), liter.write())->*[] __device__(size_t, auto iter) {
    *iter = 0;
  };

  cudaEvent_t start, stop;
  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventCreate(&stop));
  cuda_safe_call(cudaEventRecord(start, ctx.fence()));
  {
    auto while_guard = ctx.while_graph_scope(1, cudaGraphCondAssignDefault);

    ctx.parallel_for(inner<1>(lA.shape()), lA.read(), lAnew.write(), lresidual.reduce(reducer::maxval<double>()))
        ->*[tol] __device__(size_t i, size_t j, auto A, auto Anew, auto& residual) {
              Anew(i, j)   = 0.25 * (A(i - 1, j) + A(i + 1, j) + A(i, j - 1) + A(i, j + 1));
              double error = fabs(A(i, j) - Anew(i, j));
              residual     = error; // Max reduction will find the maximum error
            };

    ctx.parallel_for(inner<1>(lA.shape()), lA.rw(), lAnew.read())->*[] __device__(size_t i, size_t j, auto A, auto Anew) {
      A(i, j) = Anew(i, j);
    };

    auto h = while_guard.cond_handle();
    condition_update_scope(ctx, h, lresidual.read(), liter.rw())->*[tol] __device__(auto residual, auto iter) {
        bool converged = (*residual < tol);
        bool max_reached = ((*iter)++ >= 1000); // Maximum iteration limit
        return !converged && !max_reached; // Continue if not converged and under limit
    };
  }

  cuda_safe_call(cudaEventRecord(stop, ctx.fence()));

  // Print final iteration count
  ctx.host_launch(liter.read())->*[] (auto iter) {
    printf("Converged after %d iterations\n", *iter);
  };

  ctx.finalize();

  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Elapsed time: %f ms\n", elapsedTime);

  return 0;
}
