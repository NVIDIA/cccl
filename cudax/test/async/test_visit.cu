//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__algorithm/max.h>

#include <cuda/experimental/__async/sender.cuh>

#include "testing.cuh" // IWYU pragma: keep

namespace
{
template <class Fn>
struct recursive_lambda
{
  Fn fn;

  template <class... Args>
  __host__ __device__ auto operator()(Args&&... args)
  {
    return fn(*this, cuda::std::forward<Args>(args)...);
  }
};

template <class Fn>
recursive_lambda(Fn) -> recursive_lambda<Fn>;

C2H_TEST("sender visitation API works", "[visit]")
{
  int leaves = 0;
  int depth  = 0;

  auto snd = cudax_async::when_all(
    cudax_async::just(3), //
    cudax_async::just(0.1415),
    cudax_async::then(cudax_async::just(0.1415), [](double f) {
      return f;
    }));
  auto snd1 = std::move(snd) | cudax_async::then([](int x, double y, double z) {
                return x + y + z;
              });

  auto count_leaves = recursive_lambda{[](auto& self, int& leaves, auto, auto&, auto&... child) {
    leaves += (sizeof...(child) == 0);
    ((cudax_async::visit(self, child, leaves)), ...);
  }};

  cudax_async::visit(count_leaves, snd1, leaves);
  CHECK(leaves == 3);

  auto max_depth = recursive_lambda{[i = 0](auto& self, int& depth, auto, auto&, auto&... child) mutable {
    ++i;
    depth = cuda::std::max(depth, i);
    ((cudax_async::visit(self, child, depth)), ...);
    --i;
  }};

  cudax_async::visit(max_depth, snd1, depth);
  CHECK(depth == 4);
}
} // namespace
