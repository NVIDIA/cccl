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
 * @brief This test ensures that a task can access a logical data with a const qualifier.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

struct foo
{
  // Intentionally choose an odd (actually prime) size
  foo(context& ctx)
  {
    l = ctx.logical_data(shape_of<slice<int>>(50867));
  }

  void set(context& ctx, int val)
  {
    ctx.parallel_for(l.shape(), l.write())->*[=] _CCCL_DEVICE(size_t i, auto dl) {
      dl(i) = val;
    };
  }

  void copy_from(context& ctx, const foo& other)
  {
    ctx.parallel_for(l.shape(), l.write(), other.l.read())->*[=] _CCCL_DEVICE(size_t i, auto dl, auto dotherl) {
      dl(i) = dotherl(i);
    };
  }

  void ensure(context& ctx, int val)
  {
    std::ignore = val;
    ctx.parallel_for(l.shape(), l.read())->*[=] _CCCL_DEVICE(size_t i, auto dl) {
      assert(dl(i) == val);
    };
  }

  auto& get_l() const
  {
    return l;
  }

  logical_data<slice<int>> l;
};

void read_only_access(context& ctx, const foo& f)
{
  ctx.parallel_for(f.l.shape(), f.l.read())->*[] _CCCL_DEVICE(size_t i, auto dl) {
    // no-op
  };

  ctx.parallel_for(f.get_l().shape(), f.get_l().read())->*[] _CCCL_DEVICE(size_t i, auto dl) {
    // no-op
  };
}

int main()
{
  context ctx;

  foo A(ctx);
  A.set(ctx, 42);
  A.ensure(ctx, 42);
  foo B(ctx);
  B.copy_from(ctx, A);
  B.ensure(ctx, 42);

  read_only_access(ctx, A);

  ctx.finalize();
}
