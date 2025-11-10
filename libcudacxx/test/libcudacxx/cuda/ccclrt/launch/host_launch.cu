//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/__launch/host_launch.h>
#include <cuda/__stream/stream.h>
#include <cuda/atomic>
#include <cuda/memory>

#include <cooperative_groups.h>
#include <testing.cuh>

void block_stream(cuda::stream_ref stream, cuda::atomic<int>& atomic)
{
  auto block_lambda = [&]() {
    while (atomic != 1)
      ;
  };
  cuda::host_launch(stream, block_lambda);
}

void unblock_and_wait_stream(cuda::stream_ref stream, cuda::atomic<int>& atomic)
{
  CCCLRT_REQUIRE(!stream.is_done());
  atomic = 1;
  stream.sync();
  atomic = 0;
}

void launch_local_lambda(cuda::stream_ref stream, int& set, int set_to)
{
  auto lambda = [&set, set_to]() {
    set = set_to;
  };
  cuda::host_launch(stream, lambda);
}

template <typename Lambda>
struct lambda_wrapper
{
  Lambda lambda;

  lambda_wrapper(const Lambda& lambda)
      : lambda(lambda)
  {}

  lambda_wrapper(lambda_wrapper&&)      = default;
  lambda_wrapper(const lambda_wrapper&) = default;

  void operator()()
  {
    if constexpr (cuda::std::is_same_v<cuda::std::invoke_result_t<Lambda>, void*>)
    {
      // If lambda returns the address it captured, confirm this object wasn't moved
      CCCLRT_REQUIRE(lambda() == this);
    }
    else
    {
      lambda();
    }
  }

  // Make sure we fail if const is added to this wrapper anywhere
  void operator()() const
  {
    CCCLRT_REQUIRE(false);
  }
};

struct MoveOnlyArg
{
  static MoveOnlyArg make()
  {
    return MoveOnlyArg{};
  }

  MoveOnlyArg(const MoveOnlyArg& other)      = delete;
  MoveOnlyArg(MoveOnlyArg&&)                 = default;
  MoveOnlyArg& operator=(const MoveOnlyArg&) = delete;
  MoveOnlyArg& operator=(MoveOnlyArg&&)      = delete;

private:
  MoveOnlyArg() = default;
};

struct MoveOnlyCallable
{
  static MoveOnlyCallable make()
  {
    return MoveOnlyCallable{};
  }

  MoveOnlyCallable(const MoveOnlyCallable&)            = delete;
  MoveOnlyCallable(MoveOnlyCallable&&)                 = default;
  MoveOnlyCallable& operator=(const MoveOnlyCallable&) = delete;
  MoveOnlyCallable& operator=(MoveOnlyCallable&&)      = delete;

  void operator()(MoveOnlyArg) {}

private:
  MoveOnlyCallable() = default;
};

C2H_CCCLRT_TEST("Host launch", "")
{
  cuda::device_ref device{0};
  device.init();

  cuda::atomic<int> atomic = 0;
  cuda::stream stream{device};
  int i = 0;

  auto set_lambda = [&](int set) {
    i = set;
  };

  SECTION("Can do a host launch")
  {
    block_stream(stream, atomic);

    cuda::host_launch(stream, set_lambda, 2);

    unblock_and_wait_stream(stream, atomic);
    CCCLRT_REQUIRE(i == 2);
  }

  SECTION("Can launch multiple functions")
  {
    block_stream(stream, atomic);
    auto check_lambda = [&]() {
      CCCLRT_REQUIRE(i == 4);
    };

    cuda::host_launch(stream, set_lambda, 3);
    cuda::host_launch(stream, set_lambda, 4);
    cuda::host_launch(stream, check_lambda);
    cuda::host_launch(stream, set_lambda, 5);
    unblock_and_wait_stream(stream, atomic);
    CCCLRT_REQUIRE(i == 5);
  }

  SECTION("Non trivially copyable")
  {
    std::string s = "hello";

    cuda::host_launch(
      stream,
      [&](auto str_arg) {
        CCCLRT_REQUIRE(s == str_arg);
      },
      s);
    stream.sync();
  }

  SECTION("Confirm no const added to the callable")
  {
    lambda_wrapper wrapped_lambda([&]() {
      i = 21;
    });

    cuda::host_launch(stream, wrapped_lambda);
    stream.sync();
    CCCLRT_REQUIRE(i == 21);
  }

  SECTION("Can launch a local function and return")
  {
    block_stream(stream, atomic);
    launch_local_lambda(stream, i, 42);
    unblock_and_wait_stream(stream, atomic);
    CCCLRT_REQUIRE(i == 42);
  }

  SECTION("Launch by reference")
  {
    // Grab the pointer to confirm callable was not moved
    void* wrapper_ptr = nullptr;
    lambda_wrapper another_lambda_setter([&]() {
      i = 84;
      return wrapper_ptr;
    });
    wrapper_ptr = static_cast<void*>(&another_lambda_setter);

    block_stream(stream, atomic);
    cuda::host_launch(stream, cuda::std::ref(another_lambda_setter));
    unblock_and_wait_stream(stream, atomic);
    CCCLRT_REQUIRE(i == 84);
  }

  SECTION("Launch by reference with arguments")
  {
    i           = 10;
    int result  = 0;
    auto lambda = [&result](int j) {
      result = j;
    };
    block_stream(stream, atomic);
    cuda::host_launch(stream, cuda::std::ref(lambda), i);
    unblock_and_wait_stream(stream, atomic);
    CCCLRT_REQUIRE(result == 10);
  }

  SECTION("Launch by reference with arguments captured by reference")
  {
    i           = 0;
    auto lambda = [](int& j) {
      j = 10;
    };
    block_stream(stream, atomic);
    cuda::host_launch(stream, cuda::std::ref(lambda), cuda::std::ref(i));
    unblock_and_wait_stream(stream, atomic);
    CCCLRT_REQUIRE(i == 10);
  }

  SECTION("Check that host_launch works with move only callables and arguments")
  {
    cuda::host_launch(stream, MoveOnlyCallable::make(), MoveOnlyArg::make());
    stream.sync();
  }
}
