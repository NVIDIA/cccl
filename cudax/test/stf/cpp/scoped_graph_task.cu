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
 * @brief Show how we can create tasks in the CUDA graph backend by using the
 *        actual CUDA graph API in tasks (instead of relying on graph capture
 *        implicitly)
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

int main(int argc, char** argv)
{
  const size_t n = 12;

  double X[n];
  double Y[n];

  for (size_t ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind + 42;
    Y[ind] = 0.0;
  }

  // We here do not assume there is a valid copy on the host and only provide
  // constant parameters
  graph_ctx ctx;
  auto handle_X = ctx.logical_data(X);
  handle_X.set_symbol("x");
  auto handle_Y = ctx.logical_data(Y);
  handle_Y.set_symbol("y");
  auto handle_TMP = ctx.logical_data<double>(n);
  handle_TMP.set_symbol("tmp");

  int NITER = 4;
  for (int iter = 0; iter < NITER; iter++)
  {
    // We swap X and Y using TMP as temporary buffer
    // TMP = X
    // X = Y
    // Y = TMP
    ctx.task(exec_place::current_device(), handle_X.rw(), handle_Y.rw(), handle_TMP.write())
        ->*[&](cudaGraph_t child_graph, auto d_x, auto d_y, auto d_tmp) {
              // TMP = X
              cudaGraphNode_t cpy_tmp_to_x;
              cuda_try(cudaGraphAddMemcpyNode1D(
                &cpy_tmp_to_x,
                child_graph,
                nullptr,
                0,
                d_tmp.data_handle(),
                d_x.data_handle(),
                n * sizeof(double),
                cudaMemcpyDeviceToDevice));

              // X = Y
              cudaGraphNode_t cpy_x_to_y;
              cuda_try(cudaGraphAddMemcpyNode1D(
                &cpy_x_to_y,
                child_graph,
                &cpy_tmp_to_x,
                1,
                d_x.data_handle(),
                d_y.data_handle(),
                n * sizeof(double),
                cudaMemcpyDeviceToDevice));

              // Y = TMP
              cudaGraphNode_t cpy_tmp_to_y;
              cuda_try(cudaGraphAddMemcpyNode1D(
                &cpy_tmp_to_y,
                child_graph,
                &cpy_x_to_y,
                1,
                d_y.data_handle(),
                d_tmp.data_handle(),
                n * sizeof(double),
                cudaMemcpyDeviceToDevice));
            };
  }

  ctx.submit();

  if (argc > 1)
  {
    std::cout << "Generating DOT output in " << argv[1] << std::endl;
    ctx.print_to_dot(argv[1]);
  }

  ctx.finalize();
}
