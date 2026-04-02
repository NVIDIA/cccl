//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Tests for the host_launch untyped dispatch path (host_launch_deps)

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

// ---------------------------------------------------------------------------
// stream_ctx tests
// ---------------------------------------------------------------------------

void test_stream_basic()
{
  const size_t N = 64;
  double X[N];
  for (size_t i = 0; i < N; i++)
  {
    X[i] = static_cast<double>(i);
  }

  stream_ctx ctx;
  auto lX = ctx.logical_data(X);

  auto scope = ctx.host_launch();
  scope.add_deps(task_dep_untyped(lX, access_mode::read));
  scope->*[](host_launch_deps& deps) {
    EXPECT(deps.size() == 1);
    auto sX = deps.get<slice<double>>(0);
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(sX(i) == static_cast<double>(i));
    }
  };

  ctx.finalize();
}

void test_stream_multiple_deps()
{
  const size_t N = 32;
  double X[N], Y[N];
  for (size_t i = 0; i < N; i++)
  {
    X[i] = static_cast<double>(i);
    Y[i] = static_cast<double>(i * 10);
  }

  stream_ctx ctx;
  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  auto scope = ctx.host_launch();
  scope.add_deps(task_dep_untyped(lX, access_mode::read));
  scope.add_deps(task_dep_untyped(lY, access_mode::read));
  scope->*[](host_launch_deps& deps) {
    EXPECT(deps.size() == 2);
    auto sX = deps.get<slice<double>>(0);
    auto sY = deps.get<slice<double>>(1);
    for (size_t i = 0; i < 32; i++)
    {
      EXPECT(sX(i) == static_cast<double>(i));
      EXPECT(sY(i) == static_cast<double>(i * 10));
    }
  };

  ctx.finalize();
}

void test_stream_user_data()
{
  stream_ctx ctx;

  struct my_ctx
  {
    int magic;
    double value;
  };

  my_ctx uctx{42, 3.14};

  auto scope = ctx.host_launch();
  scope.set_user_data(&uctx, sizeof(uctx));
  scope->*[](host_launch_deps& deps) {
    EXPECT(deps.size() == 0);
    EXPECT(deps.user_data() != nullptr);
    EXPECT(deps.user_data_size() == sizeof(my_ctx));
    auto* u = static_cast<my_ctx*>(deps.user_data());
    EXPECT(u->magic == 42);
    EXPECT(u->value == 3.14);
  };

  ctx.finalize();
}

void test_stream_write_back()
{
  const size_t N = 64;
  double X[N];
  for (size_t i = 0; i < N; i++)
  {
    X[i] = 0.0;
  }

  stream_ctx ctx;
  auto lX = ctx.logical_data(X);

  auto scope = ctx.host_launch();
  scope.add_deps(task_dep_untyped(lX, access_mode::rw));
  scope->*[](host_launch_deps& deps) {
    auto sX = deps.get<slice<double>>(0);
    for (size_t i = 0; i < 64; i++)
    {
      sX(i) = static_cast<double>(i * 2);
    }
  };

  ctx.host_launch(lX.read())->*[](auto sX) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(sX(i) == static_cast<double>(i * 2));
    }
  };

  ctx.finalize();
}

void test_stream_no_user_data()
{
  stream_ctx ctx;

  auto scope = ctx.host_launch();
  scope->*[](host_launch_deps& deps) {
    EXPECT(deps.size() == 0);
    EXPECT(deps.user_data() == nullptr);
    EXPECT(deps.user_data_size() == 0);
  };

  ctx.finalize();
}

void test_stream_chained()
{
  const size_t N = 64;
  double X[N];
  for (size_t i = 0; i < N; i++)
  {
    X[i] = 1.0;
  }

  stream_ctx ctx;
  auto lX = ctx.logical_data(X);

  auto s1 = ctx.host_launch();
  s1.add_deps(task_dep_untyped(lX, access_mode::rw));
  s1->*[](host_launch_deps& deps) {
    auto sX = deps.get<slice<double>>(0);
    for (size_t i = 0; i < 64; i++)
    {
      sX(i) *= 2.0;
    }
  };

  auto s2 = ctx.host_launch();
  s2.add_deps(task_dep_untyped(lX, access_mode::rw));
  s2->*[](host_launch_deps& deps) {
    auto sX = deps.get<slice<double>>(0);
    for (size_t i = 0; i < 64; i++)
    {
      sX(i) += 10.0;
    }
  };

  ctx.host_launch(lX.read())->*[](auto sX) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(sX(i) == 12.0);
    }
  };

  ctx.finalize();
}

// ---------------------------------------------------------------------------
// graph_ctx tests
// ---------------------------------------------------------------------------

void test_graph_basic()
{
  const size_t N = 64;
  double X[N];
  for (size_t i = 0; i < N; i++)
  {
    X[i] = static_cast<double>(i);
  }

  graph_ctx ctx;
  auto lX = ctx.logical_data(X);

  auto scope = ctx.host_launch();
  scope.add_deps(task_dep_untyped(lX, access_mode::read));
  scope->*[](host_launch_deps& deps) {
    EXPECT(deps.size() == 1);
    auto sX = deps.get<slice<double>>(0);
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(sX(i) == static_cast<double>(i));
    }
  };

  ctx.finalize();
}

void test_graph_multiple_deps()
{
  const size_t N = 32;
  double X[N], Y[N];
  for (size_t i = 0; i < N; i++)
  {
    X[i] = static_cast<double>(i);
    Y[i] = static_cast<double>(i * 10);
  }

  graph_ctx ctx;
  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  auto scope = ctx.host_launch();
  scope.add_deps(task_dep_untyped(lX, access_mode::read));
  scope.add_deps(task_dep_untyped(lY, access_mode::read));
  scope->*[](host_launch_deps& deps) {
    EXPECT(deps.size() == 2);
    auto sX = deps.get<slice<double>>(0);
    auto sY = deps.get<slice<double>>(1);
    for (size_t i = 0; i < 32; i++)
    {
      EXPECT(sX(i) == static_cast<double>(i));
      EXPECT(sY(i) == static_cast<double>(i * 10));
    }
  };

  ctx.finalize();
}

void test_graph_user_data()
{
  graph_ctx ctx;

  struct my_ctx
  {
    int magic;
    double value;
  };

  my_ctx uctx{42, 3.14};

  auto scope = ctx.host_launch();
  scope.set_user_data(&uctx, sizeof(uctx));
  scope->*[](host_launch_deps& deps) {
    EXPECT(deps.size() == 0);
    EXPECT(deps.user_data() != nullptr);
    EXPECT(deps.user_data_size() == sizeof(my_ctx));
    auto* u = static_cast<my_ctx*>(deps.user_data());
    EXPECT(u->magic == 42);
    EXPECT(u->value == 3.14);
  };

  ctx.finalize();
}

void test_graph_write_back()
{
  const size_t N = 64;
  double X[N];
  for (size_t i = 0; i < N; i++)
  {
    X[i] = 0.0;
  }

  graph_ctx ctx;
  auto lX = ctx.logical_data(X);

  auto scope = ctx.host_launch();
  scope.add_deps(task_dep_untyped(lX, access_mode::rw));
  scope->*[](host_launch_deps& deps) {
    auto sX = deps.get<slice<double>>(0);
    for (size_t i = 0; i < 64; i++)
    {
      sX(i) = static_cast<double>(i * 2);
    }
  };

  ctx.host_launch(lX.read())->*[](auto sX) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(sX(i) == static_cast<double>(i * 2));
    }
  };

  ctx.finalize();
}

void test_graph_no_user_data()
{
  graph_ctx ctx;

  auto scope = ctx.host_launch();
  scope->*[](host_launch_deps& deps) {
    EXPECT(deps.size() == 0);
    EXPECT(deps.user_data() == nullptr);
    EXPECT(deps.user_data_size() == 0);
  };

  ctx.finalize();
}

void test_graph_chained()
{
  const size_t N = 64;
  double X[N];
  for (size_t i = 0; i < N; i++)
  {
    X[i] = 1.0;
  }

  graph_ctx ctx;
  auto lX = ctx.logical_data(X);

  auto s1 = ctx.host_launch();
  s1.add_deps(task_dep_untyped(lX, access_mode::rw));
  s1->*[](host_launch_deps& deps) {
    auto sX = deps.get<slice<double>>(0);
    for (size_t i = 0; i < 64; i++)
    {
      sX(i) *= 2.0;
    }
  };

  auto s2 = ctx.host_launch();
  s2.add_deps(task_dep_untyped(lX, access_mode::rw));
  s2->*[](host_launch_deps& deps) {
    auto sX = deps.get<slice<double>>(0);
    for (size_t i = 0; i < 64; i++)
    {
      sX(i) += 10.0;
    }
  };

  ctx.host_launch(lX.read())->*[](auto sX) {
    for (size_t i = 0; i < 64; i++)
    {
      EXPECT(sX(i) == 12.0);
    }
  };

  ctx.finalize();
}

int main()
{
  // stream_ctx
  test_stream_basic();
  test_stream_multiple_deps();
  test_stream_user_data();
  test_stream_write_back();
  test_stream_no_user_data();
  test_stream_chained();

  // graph_ctx
  test_graph_basic();
  test_graph_multiple_deps();
  test_graph_user_data();
  test_graph_write_back();
  test_graph_no_user_data();
  test_graph_chained();

  return 0;
}
