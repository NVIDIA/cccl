//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Test that update_cond accepts a mix of token and typed dependencies
 *
 * cuda_kernel drops void_interface (token) instances from the arguments it
 * applies to a wrapper functor, so update_cond's condition functor must
 * receive only the non-void instances even when some of its declared
 * dependencies are tokens. This reproduces that arity mismatch: the
 * condition depends on both a data-less token (ordering only, exercised by a
 * preceding token-only task) and a real scalar (the iteration counter), and
 * the condition functor's signature matches the filtered (token-free)
 * argument list.
 */

#include <cuda/experimental/stf.cuh>

#include <cstdio>
#include <vector>

using namespace cuda::experimental::stf;

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving test: while_graph_scope is only available since CUDA 12.4.\n");
  return 0;
#else
  stackable_ctx ctx;

  size_t sz = 1024;
  ::std::vector<int> data(sz);
  for (size_t i = 0; i < sz; i++)
  {
    data[i] = static_cast<int>(i);
  }

  auto ldata  = ctx.logical_data(make_slice(data.data(), sz));
  auto liter  = ctx.logical_data(shape_of<scalar_view<int>>());
  auto ltoken = ctx.token();

  int max_iter = 4;

  ctx.parallel_for(box(1), liter.write())->*[] __device__(size_t, auto iter) {
    *iter = 0;
  };

  // The while-loop scope is a block so its RAII pop runs before
  // ctx.finalize() below.
  {
    auto while_guard = ctx.while_graph_scope();

    // A token-only task: no payload, just orders the loop body relative to
    // the condition update below through ltoken.
    ctx.task(ltoken.write())->*[](cudaStream_t) {};

    ctx.parallel_for(ldata.shape(), ldata.rw())->*[] __device__(size_t i, auto d) {
      d(i)++;
    };

    // Mixed dependency list: the token is filtered out and the typed scalar
    // is passed to the condition functor.
    while_guard.update_cond(ltoken.read(), liter.rw())->*[max_iter] __device__(auto iter) {
      (*iter)++;
      return (*iter < max_iter);
    };
  }

  ctx.finalize();

  for (size_t i = 0; i < sz; i++)
  {
    int expected = static_cast<int>(i) + max_iter;
    _CCCL_ASSERT(data[i] == expected, "invalid result at index");
  }

  return 0;
#endif // !_CCCL_CTK_BELOW(12, 4)
}
