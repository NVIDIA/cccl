//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//                         Copyright (c) 2022 Lucian Radu Teodorescu
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__algorithm/fill_n.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__numeric/iota.h>
#include <cuda/std/array>

#include <cuda/experimental/execution.cuh>

#include "common/checked_receiver.cuh"
#include "common/dummy_scheduler.cuh"
#include "common/error_scheduler.cuh"
#include "common/utility.cuh"
#include "testing.cuh" // IWYU pragma: keep

namespace ex = cuda::experimental::execution;

#if !defined(__CUDA_ARCH__)
using _exception_ptr = ::std::exception_ptr;
#else
struct _exception_ptr
{};
#endif

namespace
{
template <class Shape, int N>
_CCCL_HOST_DEVICE void function(Shape i, int (*counter)[N])
{
  (*counter)[i]++;
}

template <class Shape, int N>
_CCCL_HOST_DEVICE void function_range(Shape b, Shape e, int (*counter)[N])
{
  while (b != e)
  {
    (*counter)[b++]++;
  }
}

template <class Shape>
struct function_object_t
{
  int* counter_;

  _CCCL_HOST_DEVICE void operator()(Shape i)
  {
    counter_[i]++;
  }
};

template <class Shape>
struct function_object_range_t
{
  int* counter_;

  _CCCL_HOST_DEVICE void operator()(Shape b, Shape e)
  {
    while (b != e)
    {
      counter_[b++]++;
    }
  }
};

struct ignore_lvalue_ref
{
  template <class T>
  _CCCL_HOST_DEVICE ignore_lvalue_ref(T&) noexcept
  {
    // Do nothing, just ignore the value
  }
};
} // anonymous namespace

void bulk_returns_a_sender()
{
  auto sndr = ex::bulk(ex::just(19), ex::par, 8, [] _CCCL_HOST_DEVICE(int, int) {});
  STATIC_REQUIRE(ex::sender<decltype(sndr)>);
  (void) sndr;
}

void bulk_chunked_returns_a_sender()
{
  auto sndr = ex::bulk_chunked(ex::just(19), ex::par, 8, [] _CCCL_HOST_DEVICE(int, int, int) {});
  STATIC_REQUIRE(ex::sender<decltype(sndr)>);
  (void) sndr;
}

void bulk_unchunked_returns_a_sender()
{
  auto sndr = ex::bulk_unchunked(ex::just(19), ex::par, 8, [] _CCCL_HOST_DEVICE(int, int) {});
  STATIC_REQUIRE(ex::sender<decltype(sndr)>);
  (void) sndr;
}

void bulk_with_environment_returns_a_sender()
{
  auto sndr = ex::bulk(ex::just(19), ex::par, 8, [] _CCCL_HOST_DEVICE(int, int) {});
  STATIC_REQUIRE(ex::sender_in<decltype(sndr), ex::env<>>);
  (void) sndr;
}

void bulk_chunked_with_environment_returns_a_sender()
{
  auto sndr = ex::bulk_chunked(ex::just(19), ex::par, 8, [] _CCCL_HOST_DEVICE(int, int, int) {});
  STATIC_REQUIRE(ex::sender_in<decltype(sndr), ex::env<>>);
  (void) sndr;
}

void bulk_unchunked_with_environment_returns_a_sender()
{
  auto sndr = ex::bulk_unchunked(ex::just(19), ex::par, 8, [] _CCCL_HOST_DEVICE(int, int) {});
  STATIC_REQUIRE(ex::sender_in<decltype(sndr), ex::env<>>);
  (void) sndr;
}

void bulk_can_be_piped()
{
  auto sndr = ex::just() //
            | ex::bulk(ex::par, 42, [] _CCCL_HOST_DEVICE(int) {});
  (void) sndr;
}

void bulk_chunked_can_be_piped()
{
  auto sndr = ex::just() //
            | ex::bulk_chunked(ex::par, 42, [] _CCCL_HOST_DEVICE(int, int) {});
  (void) sndr;
}

void bulk_unchunked_can_be_piped()
{
  auto sndr = ex::just() //
            | ex::bulk_unchunked(ex::par, 42, [] _CCCL_HOST_DEVICE(int) {});
  (void) sndr;
}

void bulk_keeps_values_type_from_input_sender()
{
  constexpr int n = 42;
  check_value_types<types<>>(ex::just() //
                             | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) {}));
  check_value_types<types<double>>(ex::just(4.2) //
                                   | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int, double) {}));
  check_value_types<types<double, string>>(ex::just(4.2, string{}) //
                                           | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int, double, string) {}));
}

void bulk_chunked_keeps_values_type_from_input_sender()
{
  constexpr int n = 42;
  check_value_types<types<>>(ex::just() //
                             | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) {}));
  check_value_types<types<double>>(ex::just(4.2) //
                                   | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int, double) {}));
  check_value_types<types<double, string>>(
    ex::just(4.2, string{}) | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int, double, string) {}));
}

void bulk_unchunked_keeps_values_type_from_input_sender()
{
  constexpr int n = 42;
  check_value_types<types<>>(ex::just() //
                             | ex::bulk_unchunked(ex::par, n, [] _CCCL_HOST_DEVICE(int) {}));
  check_value_types<types<double>>(ex::just(4.2) //
                                   | ex::bulk_unchunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, double) {}));
  check_value_types<types<double, string>>(
    ex::just(4.2, string{}) //
    | ex::bulk_unchunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, double, string) {}));
}

void bulk_keeps_error_types_from_input_sender()
{
#if !_CCCL_COMPILER(MSVC)
  constexpr int n = 42;
  dummy_scheduler sched1{};
  error_scheduler<_exception_ptr> sched2{};
  error_scheduler<int> sched3{43};

  // MSVCBUG https://developercommunity.visualstudio.com/t/noexcept-expression-in-lambda-template-n/10718680
  check_error_types<>(ex::just() //
                      | ex::continues_on(sched1) //
                      | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<_exception_ptr>(
    ex::just() //
    | ex::continues_on(sched2) //
    | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<int>(ex::just_error(n) //
                         | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
#  if !defined(__CUDA_ARCH__)
  check_error_types<::std::exception_ptr, int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk(ex::par, n, [](int) {
        throw std::logic_error{"err"};
      }));
#  else
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int) {
        cuda::std::__cccl_terminate();
      }));
#  endif
#endif
}

void bulk_chunked_keeps_error_types_from_input_sender()
{
  constexpr int n = 42;
  dummy_scheduler sched1{};
  error_scheduler<_exception_ptr> sched2{};
  error_scheduler<int> sched3{43};

  check_error_types<>(ex::just() //
                      | ex::continues_on(sched1) //
                      | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) noexcept {}));
  check_error_types<_exception_ptr>(
    ex::just() //
    | ex::continues_on(sched2) //
    | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) noexcept {}));
  check_error_types<int>(ex::just_error(n) //
                         | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) noexcept {}));
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) noexcept {}));
#if !defined(__CUDA_ARCH__)
  check_error_types<::std::exception_ptr, int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_chunked(ex::par, n, [](int, int) {
        throw std::logic_error{"err"};
      }));
#else
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) {
        cuda::std::__cccl_terminate();
      }));
#endif
}

void bulk_unchunked_keeps_error_types_from_input_sender()
{
  constexpr int n = 42;
  dummy_scheduler sched1{};
  error_scheduler<_exception_ptr> sched2{};
  error_scheduler<int> sched3{43};

  check_error_types<>(ex::just() //
                      | ex::continues_on(sched1) //
                      | ex::bulk_unchunked(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<_exception_ptr>(
    ex::just() //
    | ex::continues_on(sched2) //
    | ex::bulk_unchunked(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<int>(ex::just_error(n) //
                         | ex::bulk_unchunked(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_unchunked(ex::par, n, [] _CCCL_HOST_DEVICE(int) noexcept {}));
#if !defined(__CUDA_ARCH__)
  check_error_types<::std::exception_ptr, int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_unchunked(ex::par, n, [](int) {
        throw std::logic_error{"err"};
      }));
#else
  check_error_types<int>(
    ex::just() //
    | ex::continues_on(sched3) //
    | ex::bulk_unchunked(ex::par, n, [] _CCCL_HOST_DEVICE(int) {
        cuda::std::__cccl_terminate();
      }));
#endif
}

void bulk_can_be_used_with_a_function()
{
  constexpr int n = 9;
  int counter1[n]{};
  ::cuda::std::fill_n(counter1, n, 0);

  auto sndr = ex::just(&counter1) //
            | ex::bulk(ex::par, n, function<int, n>);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{&counter1});
  ex::start(op);

  for (int i : counter1)
  {
    CHECK(i == 1);
  }
}

void bulk_chunked_can_be_used_with_a_function()
{
  constexpr int n = 9;
  int counter2[n]{};
  ::cuda::std::fill_n(counter2, n, 0);

  auto sndr = ex::just(&counter2) //
            | ex::bulk_chunked(ex::par, n, function_range<int, n>);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{&counter2});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter2[i] == 1);
  }
}

void bulk_unchunked_can_be_used_with_a_function()
{
  constexpr int n = 9;
  int counter3[n]{};
  ::cuda::std::fill_n(counter3, n, 0);

  auto sndr = ex::just(&counter3) //
            | ex::bulk_unchunked(ex::par, n, function<int, n>);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{&counter3});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter3[i] == 1);
  }
}

void bulk_can_be_used_with_a_function_object()
{
  constexpr int n = 9;
  int counter[n]{0};
  function_object_t<int> fn{counter};

  auto sndr = ex::just() //
            | ex::bulk(ex::par, n, fn);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i : counter)
  {
    CHECK(i == 1);
  }
}

void bulk_chunked_can_be_used_with_a_function_object()
{
  constexpr int n = 9;
  int counter[n]{0};
  function_object_range_t<int> fn{counter};

  auto sndr = ex::just() //
            | ex::bulk_chunked(ex::par, n, fn);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}

void bulk_unchunked_can_be_used_with_a_function_object()
{
  constexpr int n = 9;
  int counter[n]{0};
  function_object_t<int> fn{counter};

  auto sndr = ex::just() //
            | ex::bulk_unchunked(ex::par, n, fn);
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}

#if !defined(__CUDA_ARCH__)
void bulk_can_be_used_with_a_lambda()
{
  constexpr int n = 9;
  int counter[n]{0};

  auto sndr = ex::just() //
            | ex::bulk(ex::par, n, [&](int i) {
                counter[i]++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i : counter)
  {
    CHECK(i == 1);
  }
}

void bulk_chunked_can_be_used_with_a_lambda()
{
  constexpr int n = 9;
  int counter[n]{0};

  auto sndr = ex::just() //
            | ex::bulk_chunked(ex::par, n, [&](int b, int e) {
                while (b < e)
                {
                  counter[b++]++;
                }
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}

void bulk_unchunked_can_be_used_with_a_lambda()
{
  constexpr int n = 9;
  int counter[n]{0};

  auto sndr = ex::just() //
            | ex::bulk_unchunked(ex::par, n, [&](int i) {
                counter[i]++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}
#endif // !defined(__CUDA_ARCH__)

void bulk_works_with_all_standard_execution_policies()
{
  auto snd1 = ex::just() //
            | ex::bulk(ex::seq, 9, [] _CCCL_HOST_DEVICE(int) {});
  auto snd2 = ex::just() //
            | ex::bulk(ex::par, 9, [] _CCCL_HOST_DEVICE(int) {});
  auto snd3 = ex::just() //
            | ex::bulk(ex::par_unseq, 9, [] _CCCL_HOST_DEVICE(int) {});
  auto snd4 = ex::just() //
            | ex::bulk(ex::unseq, 9, [] _CCCL_HOST_DEVICE(int) {});

  STATIC_REQUIRE(ex::sender<decltype(snd1)>);
  STATIC_REQUIRE(ex::sender<decltype(snd2)>);
  STATIC_REQUIRE(ex::sender<decltype(snd3)>);
  STATIC_REQUIRE(ex::sender<decltype(snd4)>);
  (void) snd1;
  (void) snd2;
  (void) snd3;
  (void) snd4;
}

void bulk_chunked_works_with_all_standard_execution_policies()
{
  auto snd1 = ex::just() //
            | ex::bulk_chunked(ex::seq, 9, [] _CCCL_HOST_DEVICE(int, int) {});
  auto snd2 = ex::just() //
            | ex::bulk_chunked(ex::par, 9, [] _CCCL_HOST_DEVICE(int, int) {});
  auto snd3 = ex::just() //
            | ex::bulk_chunked(ex::par_unseq, 9, [] _CCCL_HOST_DEVICE(int, int) {});
  auto snd4 = ex::just() //
            | ex::bulk_chunked(ex::unseq, 9, [] _CCCL_HOST_DEVICE(int, int) {});

  STATIC_REQUIRE(ex::sender<decltype(snd1)>);
  STATIC_REQUIRE(ex::sender<decltype(snd2)>);
  STATIC_REQUIRE(ex::sender<decltype(snd3)>);
  STATIC_REQUIRE(ex::sender<decltype(snd4)>);
  (void) snd1;
  (void) snd2;
  (void) snd3;
  (void) snd4;
}

void bulk_forwards_values()
{
  constexpr int n            = 9;
  constexpr int magic_number = 42;
  int counter[n]{0};

  auto sndr = ex::just(magic_number, &counter) //
            | ex::bulk(ex::par, n, [](int i, int val, int (*counter)[n]) {
                if (val == magic_number)
                {
                  (*counter)[i]++;
                }
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number, &counter});
  ex::start(op);

  for (int i : counter)
  {
    CHECK(i == 1);
  }
}

void bulk_chunked_forwards_values()
{
  constexpr int n            = 9;
  constexpr int magic_number = 42;
  int counter[n]{0};

  auto sndr = ex::just(magic_number) //
            | ex::bulk_chunked(ex::par, n, [&](int b, int e, int val) {
                if (val == magic_number)
                {
                  while (b < e)
                  {
                    counter[b++]++;
                  }
                }
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}

void bulk_unchunked_forwards_values()
{
  constexpr int n            = 9;
  constexpr int magic_number = 42;
  int counter[n]{0};

  auto sndr = ex::just(magic_number) //
            | ex::bulk_unchunked(ex::par, n, [&](int i, int val) {
                if (val == magic_number)
                {
                  counter[i]++;
                }
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number});
  ex::start(op);

  for (int i = 0; i < n; i++)
  {
    CHECK(counter[i] == 1);
  }
}

constexpr std::size_t n = 9;

void bulk_forwards_values_that_can_be_taken_by_reference()
{
  ::cuda::std::array<int, n> vals{};
  ::cuda::std::array<int, n> vals_expected{};
  ::cuda::std::iota(vals_expected.begin(), vals_expected.end(), 0);

  auto sndr = ex::just(cuda::std::move(vals)) //
            | ex::bulk(ex::par, n, [&](std::size_t i, ::cuda::std::array<int, n>& vals) {
                vals[i] = static_cast<int>(i);
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{vals_expected});
  ex::start(op);
}

void bulk_chunked_forwards_values_that_can_be_taken_by_reference()
{
  ::cuda::std::array<int, n> vals{};
  ::cuda::std::array<int, n> vals_expected{};
  ::cuda::std::iota(vals_expected.begin(), vals_expected.end(), 0);

  auto sndr = ex::just(cuda::std::move(vals)) //
            | ex::bulk_chunked(ex::par, n, [&](std::size_t b, std::size_t e, ::cuda::std::array<int, n>& vals) {
                for (; b != e; ++b)
                {
                  vals[b] = static_cast<int>(b);
                }
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{vals_expected});
  ex::start(op);
}

void bulk_unchunked_forwards_values_that_can_be_taken_by_reference()
{
  ::cuda::std::array<int, n> vals{};
  ::cuda::std::array<int, n> vals_expected{};
  ::cuda::std::iota(vals_expected.begin(), vals_expected.end(), 0);

  auto sndr = ex::just(cuda::std::move(vals)) //
            | ex::bulk_unchunked(ex::par, n, [&](std::size_t i, ::cuda::std::array<int, n>& vals) {
                vals[i] = static_cast<int>(i);
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{vals_expected});
  ex::start(op);
}

void bulk_cannot_be_used_to_change_the_value_type()
{
  constexpr int magic_number = 42;
  constexpr int n            = 2;

  auto sndr = ex::just(magic_number) //
            | ex::bulk(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) {
                return function_object_t<int>{nullptr};
              });

  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number});
  ex::start(op);
}

void bulk_chunked_cannot_be_used_to_change_the_value_type()
{
  constexpr int magic_number = 42;
  constexpr int n            = 2;

  auto sndr = ex::just(magic_number) //
            | ex::bulk_chunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int, int) {
                return function_object_range_t<int>{nullptr};
              });

  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number});
  ex::start(op);
}

void bulk_unchunked_cannot_be_used_to_change_the_value_type()
{
  constexpr int magic_number = 42;
  constexpr int n            = 2;

  auto sndr = ex::just(magic_number) //
            | ex::bulk_unchunked(ex::par, n, [] _CCCL_HOST_DEVICE(int, int) {
                return function_object_t<int>{nullptr};
              });

  auto op = ex::connect(cuda::std::move(sndr), checked_value_receiver{magic_number});
  ex::start(op);
}

#if _CCCL_HAS_EXCEPTIONS() && !defined(__CUDA_ARCH__)
void bulk_can_throw_and_set_error_will_be_called()
{
  constexpr int n = 2;

  auto sndr = ex::just() //
            | ex::bulk(ex::par, n, [](int) -> int {
                throw std::logic_error{"err"};
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{std::logic_error{"err"}});
  ex::start(op);
}

void bulk_chunked_can_throw_and_set_error_will_be_called()
{
  constexpr int n = 2;

  auto sndr = ex::just() //
            | ex::bulk_chunked(ex::par, n, [](int, int) -> int {
                throw std::logic_error{"err"};
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{std::logic_error{"err"}});
  ex::start(op);
}

void bulk_unchunked_can_throw_and_set_error_will_be_called()
{
  constexpr int n = 2;

  auto sndr = ex::just() //
            | ex::bulk_unchunked(ex::par, n, [](int) -> int {
                throw std::logic_error{"err"};
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{std::logic_error{"err"}});
  ex::start(op);
}
#endif // _CCCL_HAS_EXCEPTIONS() && !defined(__CUDA_ARCH__)

void bulk_function_is_not_called_on_error()
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_error(string{"err"}) //
            | ex::bulk(ex::par, n, [&called](int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{string{"err"}});
  ex::start(op);
}

void bulk_chunked_function_is_not_called_on_error()
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_error(string{"err"}) //
            | ex::bulk_chunked(ex::par, n, [&called](int, int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{string{"err"}});
  ex::start(op);
}

void bulk_unchunked_function_is_not_called_on_error()
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_error(string{"err"}) //
            | ex::bulk_unchunked(ex::par, n, [&called](int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_error_receiver{string{"err"}});
  ex::start(op);
}

void bulk_function_in_not_called_on_stop()
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_stopped() //
            | ex::bulk(ex::par, n, [&called](int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_stopped_receiver{});
  ex::start(op);
}

void bulk_chunked_function_in_not_called_on_stop()
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_stopped() //
            | ex::bulk_chunked(ex::par, n, [&called](int, int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_stopped_receiver{});
  ex::start(op);
}

void bulk_unchunked_function_in_not_called_on_stop()
{
  constexpr int n = 2;
  int called{};

  auto sndr = ex::just_stopped() //
            | ex::bulk_unchunked(ex::par, n, [&called](int) {
                called++;
              });
  auto op = ex::connect(cuda::std::move(sndr), checked_stopped_receiver{});
  ex::start(op);
}

void default_bulk_works_with_non_default_constructible_types()
{
  auto s = ex::just(non_default_constructible{42}) //
         | ex::bulk(ex::par, 1, [] _CCCL_HOST_DEVICE(int, ignore_lvalue_ref) {});
  ex::sync_wait(cuda::std::move(s));
}

void default_bulk_chunked_works_with_non_default_constructible_types()
{
  auto s = ex::just(non_default_constructible{42}) //
         | ex::bulk_chunked(ex::par, 1, [] _CCCL_HOST_DEVICE(int, int, ignore_lvalue_ref) {});
  ex::sync_wait(cuda::std::move(s));
}

void default_bulk_unchunked_works_with_non_default_constructible_types()
{
  auto s = ex::just(non_default_constructible{42}) //
         | ex::bulk_unchunked(ex::par, 1, [] _CCCL_HOST_DEVICE(int, ignore_lvalue_ref) {});
  ex::sync_wait(cuda::std::move(s));
}

#if !defined(__CUDA_ARCH__)
// TODO: modify these tests to work on device as well
struct my_domain
{
  _CCCL_TEMPLATE(class Sender, class... Env)
  _CCCL_REQUIRES(ex::sender_for<Sender, ex::bulk_chunked_t>)
  static auto transform_sender(Sender, const Env&...)
  {
    return ex::just(string{"hijacked"});
  }
};

void late_customizing_bulk_chunked_also_changes_the_behavior_of_bulk()
{
  bool called{false};
  // The customization will return a different value
  dummy_scheduler<my_domain> sched;
  auto sndr = ex::just(string{"hello"}) //
            | ex::continues_on(sched) //
            | ex::bulk(ex::par, 1, [&called](int, string) {
                called = true;
              });
  wait_for_value(cuda::std::move(sndr), string{"hijacked"});
  REQUIRE_FALSE(called);
}

struct my_domain2 : ex::default_domain
{
  _CCCL_TEMPLATE(class Sender, class... Env)
  _CCCL_REQUIRES(ex::sender_for<Sender, ex::bulk_t>)
  static auto transform_sender(Sender, const Env&...)
  {
    return ex::just(string{"hijacked"});
  }

private:
  using ex::default_domain::apply_sender;
};

void bulk_can_be_customized_independently_of_bulk_chunked()
{
  bool called{false};
  // The customization will return a different value
  dummy_scheduler<my_domain2> sched;
  auto sndr = ex::just(string{"hello"}) //
            | ex::continues_on(sched) //
            | ex::bulk(ex::par, 1, [&called](int, string) {
                called = true;
              });
  wait_for_value(cuda::std::move(sndr), string{"hijacked"});
  REQUIRE_FALSE(called);

  // bulk_chunked will still use the default implementation
  auto snd2 = ex::just(string{"hello"}) //
            | ex::continues_on(sched) | ex::bulk_chunked(ex::par, 1, [&called](int, int, string) {
                called = true;
              });
  wait_for_value(cuda::std::move(snd2), string{"hello"});
  REQUIRE(called);
}
#endif // !defined(__CUDA_ARCH__)

namespace
{
C2H_TEST("bulk returns a sender", "[adaptors][bulk]")
{
  bulk_returns_a_sender();
}

C2H_TEST("bulk_chunked returns a sender", "[adaptors][bulk]")
{
  bulk_chunked_returns_a_sender();
}

C2H_TEST("bulk_unchunked returns a sender", "[adaptors][bulk]")
{
  bulk_unchunked_returns_a_sender();
}

C2H_TEST("bulk with environment returns a sender", "[adaptors][bulk]")
{
  bulk_with_environment_returns_a_sender();
}

C2H_TEST("bulk_chunked with environment returns a sender", "[adaptors][bulk]")
{
  bulk_chunked_with_environment_returns_a_sender();
}

C2H_TEST("bulk_unchunked with environment returns a sender", "[adaptors][bulk]")
{
  bulk_unchunked_with_environment_returns_a_sender();
}

C2H_TEST("bulk can be piped", "[adaptors][bulk]")
{
  bulk_can_be_piped();
}

C2H_TEST("bulk_chunked can be piped", "[adaptors][bulk]")
{
  bulk_chunked_can_be_piped();
}

C2H_TEST("bulk_unchunked can be piped", "[adaptors][bulk]")
{
  bulk_unchunked_can_be_piped();
}

C2H_TEST("bulk keeps values_type from input sender", "[adaptors][bulk]")
{
  bulk_keeps_values_type_from_input_sender();
}

C2H_TEST("bulk_chunked keeps values_type from input sender", "[adaptors][bulk]")
{
  bulk_chunked_keeps_values_type_from_input_sender();
}

C2H_TEST("bulk_unchunked keeps values_type from input sender", "[adaptors][bulk]")
{
  bulk_unchunked_keeps_values_type_from_input_sender();
}

C2H_TEST("bulk keeps error_types from input sender", "[adaptors][bulk]")
{
  bulk_keeps_error_types_from_input_sender();
}

C2H_TEST("bulk_chunked keeps error_types from input sender", "[adaptors][bulk]")
{
  bulk_chunked_keeps_error_types_from_input_sender();
}

C2H_TEST("bulk_unchunked keeps error_types from input sender", "[adaptors][bulk]")
{
  bulk_unchunked_keeps_error_types_from_input_sender();
}

C2H_TEST("bulk can be used with a function", "[adaptors][bulk]")
{
  bulk_can_be_used_with_a_function();
}

C2H_TEST("bulk_chunked can be used with a function", "[adaptors][bulk]")
{
  bulk_chunked_can_be_used_with_a_function();
}

C2H_TEST("bulk_unchunked can be used with a function", "[adaptors][bulk]")
{
  bulk_unchunked_can_be_used_with_a_function();
}

C2H_TEST("bulk can be used with a function object", "[adaptors][bulk]")
{
  bulk_can_be_used_with_a_function_object();
}

C2H_TEST("bulk_chunked can be used with a function object", "[adaptors][bulk]")
{
  bulk_chunked_can_be_used_with_a_function_object();
}

C2H_TEST("bulk_unchunked can be used with a function object", "[adaptors][bulk]")
{
  bulk_unchunked_can_be_used_with_a_function_object();
}

#if !defined(__CUDA_ARCH__)
C2H_TEST("bulk can be used with a lambda", "[adaptors][bulk]")
{
  bulk_can_be_used_with_a_lambda();
}

C2H_TEST("bulk_chunked can be used with a lambda", "[adaptors][bulk]")
{
  bulk_chunked_can_be_used_with_a_lambda();
}

C2H_TEST("bulk_unchunked can be used with a lambda", "[adaptors][bulk]")
{
  bulk_unchunked_can_be_used_with_a_lambda();
}
#endif // !defined(__CUDA_ARCH__)

C2H_TEST("bulk works with all standard execution policies", "[adaptors][bulk]")
{
  bulk_works_with_all_standard_execution_policies();
}

C2H_TEST("bulk_chunked works with all standard execution policies", "[adaptors][bulk]")
{
  bulk_chunked_works_with_all_standard_execution_policies();
}

C2H_TEST("bulk forwards values", "[adaptors][bulk]")
{
  bulk_forwards_values();
}

C2H_TEST("bulk_chunked forwards values", "[adaptors][bulk]")
{
  bulk_chunked_forwards_values();
}

C2H_TEST("bulk_unchunked forwards values", "[adaptors][bulk]")
{
  bulk_unchunked_forwards_values();
}

C2H_TEST("bulk forwards values that can be taken by reference", "[adaptors][bulk]")
{
  bulk_forwards_values_that_can_be_taken_by_reference();
}

C2H_TEST("bulk_chunked forwards values that can be taken by reference", "[adaptors][bulk]")
{
  bulk_chunked_forwards_values_that_can_be_taken_by_reference();
}

C2H_TEST("bulk_unchunked forwards values that can be taken by reference", "[adaptors][bulk]")
{
  bulk_unchunked_forwards_values_that_can_be_taken_by_reference();
}

C2H_TEST("bulk cannot be used to change the value type", "[adaptors][bulk]")
{
  bulk_cannot_be_used_to_change_the_value_type();
}

C2H_TEST("bulk_chunked cannot be used to change the value type", "[adaptors][bulk]")
{
  bulk_chunked_cannot_be_used_to_change_the_value_type();
}

C2H_TEST("bulk_unchunked cannot be used to change the value type", "[adaptors][bulk]")
{
  bulk_unchunked_cannot_be_used_to_change_the_value_type();
}

#if _CCCL_HAS_EXCEPTIONS() && !defined(__CUDA_ARCH__)
C2H_TEST("bulk can throw, and set_error will be called", "[adaptors][bulk]")
{
  bulk_can_throw_and_set_error_will_be_called();
}

C2H_TEST("bulk_chunked can throw, and set_error will be called", "[adaptors][bulk]")
{
  bulk_chunked_can_throw_and_set_error_will_be_called();
}

C2H_TEST("bulk_unchunked can throw, and set_error will be called", "[adaptors][bulk]")
{
  bulk_unchunked_can_throw_and_set_error_will_be_called();
}
#endif // _CCCL_HAS_EXCEPTIONS() && !defined(__CUDA_ARCH__)

C2H_TEST("bulk function is not called on error", "[adaptors][bulk]")
{
  bulk_function_is_not_called_on_error();
}

C2H_TEST("bulk_chunked function is not called on error", "[adaptors][bulk]")
{
  bulk_chunked_function_is_not_called_on_error();
}

C2H_TEST("bulk_unchunked function is not called on error", "[adaptors][bulk]")
{
  bulk_unchunked_function_is_not_called_on_error();
}

C2H_TEST("bulk function in not called on stop", "[adaptors][bulk]")
{
  bulk_function_in_not_called_on_stop();
}

C2H_TEST("bulk_chunked function in not called on stop", "[adaptors][bulk]")
{
  bulk_chunked_function_in_not_called_on_stop();
}

C2H_TEST("bulk_unchunked function in not called on stop", "[adaptors][bulk]")
{
  bulk_unchunked_function_in_not_called_on_stop();
}

C2H_TEST("default bulk works with non_default constructible types", "[adaptors][bulk]")
{
  default_bulk_works_with_non_default_constructible_types();
}

C2H_TEST("default bulk_chunked works with non_default constructible types", "[adaptors][bulk]")
{
  default_bulk_chunked_works_with_non_default_constructible_types();
}

C2H_TEST("default bulk_unchunked works with non_default constructible types", "[adaptors][bulk]")
{
  default_bulk_unchunked_works_with_non_default_constructible_types();
}

#if !defined(__CUDA_ARCH__)
// TODO: modify these tests to work on device as well
struct my_domain
{};

C2H_TEST("late customizing bulk_chunked also changes the behavior of bulk", "[adaptors][then]")
{
  late_customizing_bulk_chunked_also_changes_the_behavior_of_bulk();
}

C2H_TEST("bulk can be customized, independently of bulk_chunked", "[adaptors][then]")
{
  bulk_can_be_customized_independently_of_bulk_chunked();
}
#endif // !defined(__CUDA_ARCH__)

} // namespace
