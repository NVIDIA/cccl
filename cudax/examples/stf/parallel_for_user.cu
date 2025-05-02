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
 * @brief An AXPY kernel implemented with a task of the CUDA stream backend
 * where the task accesses managed memory from the device. This tests
 * explicitly created managed memory, and passes it to a logical data.
 */

#include <cuda/experimental/stf.cuh>

#include <iostream>

using namespace cuda::experimental::stf;

/**
 * @brief A lightweight synchronization primitive for structuring task dependencies across phases.
 *
 * The `epoch` class provides a simple and expressive mechanism to sequence asynchronous
 * operations in task-based execution models. Each `epoch` instance tracks a logical phase,
 * represented internally by a task dependency token. Code may insert the current epoch as
 * a dependency when launching tasks or parallel loops (e.g., via `context::parallel_for` or
 * `context::task`).
 *
 * Multiple operations that depend on the same epoch may execute concurrently. When user code
 * increments the epoch (via `operator++()`), a no-op task is inserted that depends on all prior
 * operations within that epoch, and the epoch is updated to refer to this new task. This ensures
 * that any tasks inserted after the increment will occur only after all tasks from the previous
 * epoch complete.
 *
 * This sequencing is reminiscent of fork-join parallelism, but with two key generalizations:
 * 1. Multiple independent `epoch` objects may coexist, allowing more flexible and fine-grained
 *    dependency patterns.
 * 2. Advancing an epoch does not imply synchronization with the host (e.g., no `cudaDeviceSynchronize`),
 *    but instead introduces purely logical ordering between asynchronous tasks.
 *
 * Epochs are useful for expressing barriers, task phases, or staged execution pipelines in
 * a way that naturally fits into asynchronous task graphs.
 */
class epoch : public task_dep<void_interface, ::std::monostate, false>
{
public:
  epoch(epoch&) = default;
  epoch(const epoch&) = default;
  epoch(epoch&&) = default;

  /**
   * @brief Constructs an epoch from a given context.
   *
   * Initializes the internal dependency token by reading from the context's token.
   *
   * @param ctx Reference to the task execution context.
   */
  template <typename Ctx>
  epoch(Ctx& ctx)
      : task_dep<void_interface, ::std::monostate, false>(ctx.token().read())
      , increment([&]() {
        ctx.task(this->as_mode(access_mode::rw))->*[](cudaStream_t, auto) {};
      })
  {}

  /**
   * @brief Prefix increment operator.
   *
   * Advances the epoch. New tasks depending on this epoch will wait for the completion
   * of existing tasks that depend on this epoch.
   *
   * @return Reference to the updated `epoch` object.
   */
  epoch& operator++()
  {
    increment();
    return *this;
  }

  /**
   * @brief Postfix increment operator (disabled).
   *
   * This operator is intentionally disabled to enforce use of the prefix version.
   * Attempting to use it results in a compile-time error.
   */
  epoch operator++(int) = delete;

private:
  ::std::function<void()> increment;
};

double X0(size_t i)
{
  return sin((double) i);
}

double Y0(size_t i)
{
  return cos((double) i);
}

int main()
{
  context ctx = graph_ctx();
  const size_t N = 16384;

  double *X, *Y, *Z;
  cuda_safe_call(cudaMallocManaged(&X, N * sizeof(double)));
  cuda_safe_call(cudaMallocManaged(&Y, N * sizeof(double)));
  cuda_safe_call(cudaMallocManaged(&Z, N * sizeof(double)));

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
    Z[i] = Y0(i);
  }

  double alpha = 3.14;
  double beta  = 1664.0;

  auto e = epoch(ctx);

  ctx.parallel_for(box(N), e)->*[alpha, X, Y] __device__(size_t i) {
    Y[i] += alpha * X[i];
  };

  ctx.parallel_for(box(N), e)->*[beta, X, Z] __device__(size_t i) {
    Z[i] += beta * X[i];
  };

  ++e;

  ctx.parallel_for(box(N), e)->*[Y, Z] __device__(size_t i) {
    Z[i] += Y[i];
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(X[i] - X0(i)) < 0.0001);
    assert(fabs(Z[i] - (Y0(i) + Y0(i) + alpha * X0(i) + beta * X0(i))) < 0.0001);
  }
}
