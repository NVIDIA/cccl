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
 * @brief Ensure an error is reported dynamically if we access a data instance
 *        with the wrong interface type
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace cuda::experimental::stf;

template <typename Ctx, size_t n>
bool run(double (&X)[n])
{
  Ctx ctx;
  // This creates an untyped logical data that is implicitly a vector of size
  // n. Had the code used `auto` instead of `logical_data_untyped`, errors
  // would have been rejected statically. We want to disable static checking
  // for the purposes of this test.
  logical_data_untyped handle_X = ctx.logical_data(X);

  // Here we create a dynamically-typed task, again to go around static typechecking.
  auto t = ctx.task();
  t.add_deps(handle_X.rw());

  bool caught = false;
  try
  {
    t->*[&](auto&) {
      // Programming error: a vector of `double` accessed as a vector of `float`.
      handle_X.instance<slice<float>>(t);
    };
  }
  catch (const ::std::invalid_argument& e)
  {
    caught = true;
    fprintf(stderr, "Caught expected error: %s\n", e.what());
  }

  // The failed task was ended on the way out; the context is still usable.
  ctx.finalize();

  return caught;
}

int main()
{
  const int n = 12;
  double X[n];

  for (int ind = 0; ind < n; ind++)
  {
    X[ind] = 1.0 * ind;
  }

  const bool ok = run<stream_ctx>(X) && run<graph_ctx>(X);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
