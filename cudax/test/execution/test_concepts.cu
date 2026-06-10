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

namespace ex = ::cuda::experimental::execution;

struct not_a_receiver
{};

struct a_receiver
{
  using receiver_concept   = ex::receiver_t;
  a_receiver(a_receiver&&) = default;

  void set_value(int) && noexcept {}
  void set_stopped() && noexcept {}
};

C2H_TEST("tests for the receiver concepts", "[concepts]")
{
  static_assert(!ex::receiver<not_a_receiver>);
  static_assert(ex::receiver<a_receiver>);

  using yes_completions = ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>;
  static_assert(ex::receiver_of<a_receiver, yes_completions>);

  using no_completions =
    ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t(), ex::set_error_t(ex::exception_ptr)>;
  static_assert(!ex::receiver_of<a_receiver, no_completions>);
}

struct not_a_sender
{};

struct a_sender
{
  using sender_concept = ex::sender_t;

  template <class _Self>
  static constexpr auto get_completion_signatures()
  {
    return ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>{};
  }
};

struct non_constexpr_complsigs
{
  using sender_concept = ex::sender_t;

  template <class _Self, class...>
  _CCCL_HOST_DEVICE static auto get_completion_signatures()
  {
    return ex::completion_signatures<ex::set_value_t(int), ex::set_stopped_t()>{};
  }
};

C2H_TEST("tests for the sender concepts", "[concepts]")
{
  static_assert(!ex::sender<not_a_sender>);
  static_assert(ex::sender<a_sender>);

  static_assert(ex::sender_in<a_sender>);
  static_assert(ex::sender_in<a_sender, ex::env<>>);

  static_assert(ex::sender<non_constexpr_complsigs>);
  static_assert(!ex::sender_in<non_constexpr_complsigs>);
  static_assert(!ex::sender_in<non_constexpr_complsigs, ex::env<>>);

  [[maybe_unused]] auto read_env = ex::read_env(ex::get_scheduler);
  using read_env_t               = decltype(read_env);
  static_assert(ex::sender<read_env_t>);
  static_assert(!ex::sender_in<read_env_t>);
  static_assert(!ex::sender_in<read_env_t, ex::env<>>);
  static_assert(ex::sender_in<read_env_t, ex::prop<ex::get_scheduler_t, dummy_scheduler<>>>);
}
