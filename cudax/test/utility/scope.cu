//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__utility/scope.cuh>

#include <utility>

#include <c2h/catch2_test_helper.h>

namespace cudax = cuda::experimental;

TEST_CASE("scope_exit runs on normal exit", "[utility][scope]")
{
  int count = 0;
  {
    const cudax::scope_exit guard{[&]() noexcept {
      ++count;
    }};
    CHECK(count == 0);
  }
  CHECK(count == 1);
}

TEST_CASE("scope_exit runs on exceptional exit", "[utility][scope]")
{
  int count   = 0;
  bool caught = false;
  try
  {
    const cudax::scope_exit guard{[&]() noexcept {
      ++count;
    }};
    throw 42;
  }
  catch (...)
  {
    caught = true;
  }
  CHECK(caught);
  CHECK(count == 1);
}

TEST_CASE("scope_exit release disables the guard", "[utility][scope]")
{
  int count = 0;
  {
    cudax::scope_exit guard{[&]() noexcept {
      ++count;
    }};
    guard.release();
  }
  CHECK(count == 0);
}

TEST_CASE("scope_exit move releases the source", "[utility][scope]")
{
  int count = 0;
  {
    cudax::scope_exit first{[&]() noexcept {
      ++count;
    }};
    auto second = std::move(first);
    CHECK(count == 0);
  }
  CHECK(count == 1);
}

TEST_CASE("scope_fail runs only on exceptional exit", "[utility][scope]")
{
  int count = 0;
  {
    const cudax::scope_fail guard{[&]() noexcept {
      ++count;
    }};
  }
  CHECK(count == 0);

  bool caught = false;
  try
  {
    const cudax::scope_fail guard{[&]() noexcept {
      ++count;
    }};
    throw 42;
  }
  catch (...)
  {
    caught = true;
  }
  CHECK(caught);
  CHECK(count == 1);
}

TEST_CASE("scope_fail release disables the guard", "[utility][scope]")
{
  int count   = 0;
  bool caught = false;
  try
  {
    cudax::scope_fail guard{[&]() noexcept {
      ++count;
    }};
    guard.release();
    throw 42;
  }
  catch (...)
  {
    caught = true;
  }
  CHECK(caught);
  CHECK(count == 0);
}

TEST_CASE("scope_fail move releases the source", "[utility][scope]")
{
  int count   = 0;
  bool caught = false;
  try
  {
    cudax::scope_fail first{[&]() noexcept {
      ++count;
    }};
    auto second = std::move(first);
    throw 42;
  }
  catch (...)
  {
    caught = true;
  }
  CHECK(caught);
  CHECK(count == 1);
}

TEST_CASE("scope_success runs only on normal exit", "[utility][scope]")
{
  int count = 0;
  {
    const cudax::scope_success guard{[&] {
      ++count;
    }};
  }
  CHECK(count == 1);

  count       = 0;
  bool caught = false;
  try
  {
    const cudax::scope_success guard{[&] {
      ++count;
    }};
    throw 42;
  }
  catch (...)
  {
    caught = true;
  }
  CHECK(caught);
  CHECK(count == 0);
}

TEST_CASE("scope_success release disables the guard", "[utility][scope]")
{
  int count = 0;
  {
    cudax::scope_success guard{[&] {
      ++count;
    }};
    guard.release();
  }
  CHECK(count == 0);
}

TEST_CASE("scope_success move releases the source", "[utility][scope]")
{
  int count = 0;
  {
    cudax::scope_success first{[&] {
      ++count;
    }};
    auto second = std::move(first);
    CHECK(count == 0);
  }
  CHECK(count == 1);
}
