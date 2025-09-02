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

#include "common/dummy_scheduler.cuh"
#include "testing.cuh" // IWYU pragma: keep

namespace async = ::cuda::experimental::execution;

struct not_a_receiver
{};

struct a_receiver
{
  using receiver_concept   = async::receiver_t;
  a_receiver(a_receiver&&) = default;

  void set_value(int) && noexcept {}
  void set_stopped() && noexcept {}
};

C2H_TEST("tests for the receiver concepts", "[concepts]")
{
  static_assert(!async::receiver<not_a_receiver>);
  static_assert(async::receiver<a_receiver>);

  using yes_completions = async::completion_signatures<async::set_value_t(int), async::set_stopped_t()>;
  static_assert(async::receiver_of<a_receiver, yes_completions>);

  using no_completions = async::
    completion_signatures<async::set_value_t(int), async::set_stopped_t(), async::set_error_t(::std::exception_ptr)>;
  static_assert(!async::receiver_of<a_receiver, no_completions>);
}

struct not_a_sender
{};

struct a_sender
{
  using sender_concept = async::sender_t;

  template <class _Self>
  static constexpr auto get_completion_signatures()
  {
    return async::completion_signatures<async::set_value_t(int), async::set_stopped_t()>{};
  }
};

struct non_constexpr_complsigs
{
  using sender_concept = async::sender_t;

  template <class _Self, class...>
  _CCCL_HOST_DEVICE static auto get_completion_signatures()
  {
    return async::completion_signatures<async::set_value_t(int), async::set_stopped_t()>{};
  }
};

C2H_TEST("tests for the sender concepts", "[concepts]")
{
  static_assert(!async::sender<not_a_sender>);
  static_assert(async::sender<a_sender>);

  static_assert(async::sender_in<a_sender>);
  static_assert(async::sender_in<a_sender, async::env<>>);

  static_assert(async::sender<non_constexpr_complsigs>);
  static_assert(!async::sender_in<non_constexpr_complsigs>);
  static_assert(!async::sender_in<non_constexpr_complsigs, async::env<>>);

  [[maybe_unused]] auto read_env = async::read_env(async::get_scheduler);
  using read_env_t               = decltype(read_env);
  static_assert(async::sender<read_env_t>);
  static_assert(!async::sender_in<read_env_t>);
  static_assert(!async::sender_in<read_env_t, async::env<>>);
  static_assert(async::sender_in<read_env_t, async::prop<async::get_scheduler_t, dummy_scheduler<>>>);
}
