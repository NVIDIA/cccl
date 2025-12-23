//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/execution.cuh>

#include "./common/retry.cuh"
#include "testing.cuh"

namespace ex = ::cuda::experimental::execution;

#if !_CCCL_DEVICE_COMPILATION()

namespace
{
struct try_again
{};

class fails_alot
{
  template <class Receiver>
  struct operation;

public:
  using sender_concept = ex::sender_t;

  fails_alot() = default;

  __host__ fails_alot(fails_alot&& other) noexcept
      : counter_(std::move(other.counter_))
  {}

  __host__ fails_alot(const fails_alot& other) noexcept
      : counter_(other.counter_)
  {}

  template <class Receiver>
  [[nodiscard]] auto connect(Receiver rcvr) const noexcept -> operation<Receiver>
  {
    return operation<Receiver>{static_cast<Receiver&&>(rcvr), --*counter_};
  }

  template <class...>
  static _CCCL_CONSTEVAL auto get_completion_signatures() noexcept
  {
    return ex::completion_signatures<ex::set_value_t(), ex::set_error_t(try_again)>{};
  }

private:
  template <class Receiver>
  struct operation
  {
    void start() & noexcept
    {
      if (counter_ == 0)
      {
        ex::set_value(static_cast<Receiver&&>(rcvr_));
      }
      else
      {
        ex::set_error(static_cast<Receiver&&>(rcvr_), try_again{});
      }
    }

    Receiver rcvr_;
    int counter_;
  };

  std::shared_ptr<int> counter_ = std::make_shared<int>(1'000'000);
};

// #if defined(REQUIRE_TERMINATE)
// // For some reason, when compiling with nvc++, the forked process dies with SIGSEGV
// // but the error code returned from ::wait reports success, so this test fails.
// TEST_CASE("running deeply recursing algo blows the stack", "[schedulers][trampoline_scheduler]") {

//   auto recurse_deeply = retry(fails_alot{});
//   REQUIRE_TERMINATE([&] { sync_wait(std::move(recurse_deeply)); });
// }
// #endif

TEST_CASE("running deeply recursing algo on trampoline_scheduler doesn't blow the stack",
          "[scheduler][trampoline_scheduler]")
{
  ex::trampoline_scheduler sched;
  auto recurse_deeply = retry(ex::on(sched, fails_alot{}));
  ex::sync_wait(std::move(recurse_deeply));
}
} // namespace

#endif // !_CCCL_DEVICE_COMPILATION()
