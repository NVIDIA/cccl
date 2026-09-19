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
 * @brief Cross-thread error handling with exception policies
 *
 * A user-provided f(x) = 1/x is evaluated over a set of inputs by worker
 * threads. IEEE division by zero does not raise a signal by default, so the
 * floating-point status word plays the role of the trap: each worker checks
 * std::fetestexcept / std::isfinite after evaluating, and turns a poisoned
 * result into an exception. The policy algebra plays the role of the signal:
 * on_throw(defer) turns the worker's exception into a value, a handler
 * thread rethrows and interprets it, and the whole hand-off needs no shared
 * try/catch choreography.
 */

#include <cuda/experimental/stf.cuh>

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace cuda::experimental::stf;
namespace pol = cuda::experimental::stf::exception_policies;

double f(double x)
{
  return 1.0 / x;
}

int main()
{
  const ::std::vector<double> xs = {2.0, 4.0, 0.0, 8.0, 0.5, -0.0};
  ::std::vector<double> ys(xs.size());

  constexpr size_t n_workers = 2;
  const size_t chunk         = (xs.size() + n_workers - 1) / n_workers;

  // One report slot per worker: an empty exception_ptr means the chunk was clean.
  ::std::vector<::std::exception_ptr> reports(n_workers);
  ::std::vector<::std::thread> workers;

  for (size_t w = 0; w < n_workers; ++w)
  {
    workers.emplace_back([&, w] {
      reports[w] = ON_THROW(defer)
      {
        const size_t lo = w * chunk;
        const size_t hi = ::std::min(xs.size(), lo + chunk);
        for (size_t i = lo; i < hi; ++i)
        {
          ::std::feclearexcept(FE_ALL_EXCEPT);
          ys[i] = f(xs[i]);
          // The FP status check is the trap: division by zero raises no signal.
          if (::std::fetestexcept(FE_DIVBYZERO | FE_INVALID) || !::std::isfinite(ys[i]))
          {
            throw ::std::domain_error("f(x) is not finite at index " + ::std::to_string(i));
          }
        }
        return ::std::exception_ptr{};
      };
    });
  }
  for (auto& t : workers)
  {
    t.join();
  }

  // The handler thread interprets the reports through the algebra: domain errors are
  // reported and execution resumes; anything else would decline and terminate loudly.
  ::std::thread handler([&reports] {
    for (auto& report : reports)
    {
      if (report)
      {
        ON_THROW(catch_only<::std::domain_error>(notify))
        {
          ::std::rethrow_exception(report);
        };
      }
    }
  });
  handler.join();

  for (size_t i = 0; i < xs.size(); ++i)
  {
    printf("f(%g) = %g\n", xs[i], ys[i]);
  }
  return 0;
}
