//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_MULTI_GPU_ALGORITHM_COMMON_H
#define CUDAX_TEST_MULTI_GPU_ALGORITHM_COMMON_H

#include <cuda/functional>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/random>
#include <cuda/std/type_traits>

#include <chrono>
#include <exception>
#include <future>
#include <type_traits>
#include <vector>

#include <nccl_test_common.h>

#include <c2h/catch2_test_helper.h>

[[nodiscard]] inline cuda::std::minstd_rand make_rng(const c2h::seed_t& seed)
{
  return cuda::std::minstd_rand(static_cast<cuda::std::minstd_rand::result_type>(seed.get()));
}

template <class Fn>
void run_threaded(cuda::std::size_t num_ranks, Fn fn)
{
  // Every rank must be launched before any is waited on: the single-communicator `reduce`
  // blocks on a collective, so calling `get()` on rank 0's future before rank 1 is even
  // started would deadlock. Launch all futures into the vector first, then drain them.
  std::vector<std::future<void>> futures;

  futures.reserve(num_ranks);
  for (cuda::std::size_t i = 0; i < num_ranks; ++i)
  {
    futures.push_back(std::async(std::launch::async, fn, i));
  }

  // `std::async` stashes any exception thrown by `fn` in the future and `get()` rethrows it on
  // the main thread, where Catch2 can report it as a normal failure. Any not-yet-drained
  // future still joins its thread in its destructor, so a throw here never leaves a peer
  // waiting on an unposted collective. Drain every future so a failure on rank 0 does not mask
  // one on a peer.
  std::exception_ptr error = nullptr;

  for (cuda::std::size_t i = 0; i < futures.size(); ++i)
  {
    auto& f = futures[i];

    REQUIRE(f.valid());

    {
      using namespace std::chrono_literals;
      // Doing the timeout detection this way gives a much nicer error message than ctest or catch2.
      switch (constexpr auto timeout = 15s; f.wait_for(timeout))
      {
        case std::future_status::deferred:
          FAIL("Test should not use deferred execution, only async");
          break;
        case std::future_status::timeout: {
          // The standard gives no mechanism to abandon or cancel a task launched with
          // std::async, and futures block in their destructor for task completion. So to make
          // this error message actually useful, we need to toss them all into the oubliette.
          [[maybe_unused]] auto* _ = new std::vector<std::future<void>>{std::move(futures)};
          static_assert(std::is_same_v<std::remove_cv_t<decltype(timeout)>, std::chrono::seconds>);
          FAIL("Future for rank " << i << " timed out after " << timeout.count() << "s, likely a deadlock");
          break;
        }
        case std::future_status::ready:
          break;
      }
    }

    try
    {
      f.get();
    }
    catch (...)
    {
      INFO("rank " << i << " exception");
      if (!error)
      {
        error = std::current_exception();
      }
    }
  }

  if (error)
  {
    std::rethrow_exception(error);
  }
}

// A user-defined operator, to make sure nothing along the way assumes the operator is one of the
// library-provided ones.
struct custom_plus
{
  template <class T>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr T operator()(const T& lhs, const T& rhs) const
  {
    return lhs + rhs;
  }
};

using custom_value = c2h::custom_type_t<c2h::accumulateable_t, c2h::less_comparable_t, c2h::equal_comparable_t>;
using value_types  = c2h::type_list<cuda::std::int32_t, float, custom_value>;
using operators    = c2h::type_list<::cuda::std::plus<>, ::cuda::maximum<>, custom_plus>;

static_assert(cudax::nccl_transportable<custom_value>);

template <class T>
[[nodiscard]] inline T make_value(int i)
{
  return static_cast<T>(i);
}

template <>
[[nodiscard]] inline custom_value make_value<custom_value>(int i)
{
  custom_value ret{};

  ret.key = static_cast<cuda::std::size_t>(i);
  ret.val = static_cast<cuda::std::size_t>(i);
  return ret;
}

// `custom_value` orders lexicographically on (key, val), so the caller controls the tiebreak by
// picking distinct `value`s for equal `key`s.
template <class T>
[[nodiscard]] inline T make_value(const cuda::std::int64_t key, const cuda::std::int64_t)
{
  return static_cast<T>(key);
}

template <>
[[nodiscard]] inline custom_value make_value<custom_value>(const cuda::std::int64_t key, const cuda::std::int64_t value)
{
  custom_value result{};

  result.key = static_cast<cuda::std::size_t>(key);
  result.val = static_cast<cuda::std::size_t>(value);
  return result;
}

// Must cover every operator in `operators`.
template <class T, class Op>
[[nodiscard]] T get_identity()
{
  if constexpr (cuda::std::is_same_v<Op, cuda::std::plus<>> || cuda::std::is_same_v<Op, custom_plus>)
  {
    return make_value<T>(0);
  }
  else if constexpr (cuda::std::is_same_v<Op, cuda::maximum<>>)
  {
    return cuda::std::numeric_limits<T>::lowest();
  }
  else
  {
    static_assert(cuda::std::__always_false_v<T, Op>, "Add handling");
  }
}

#endif // CUDAX_TEST_MULTI_GPU_ALGORITHM_COMMON_H
