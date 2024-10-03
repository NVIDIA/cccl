//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr const T& value() const &;
// constexpr T& value() &;
// constexpr T&& value() &&;
// constexpr const T&& value() const &&;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // non-const &
  {
    cuda::std::expected<int, int> e(5);
    decltype(auto) x = e.value();
    static_assert(cuda::std::same_as<decltype(x), int&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  // const &
  {
    const cuda::std::expected<int, int> e(5);
    decltype(auto) x = e.value();
    static_assert(cuda::std::same_as<decltype(x), const int&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  // non-const &&
  {
    cuda::std::expected<int, int> e(5);
    decltype(auto) x = cuda::std::move(e).value();
    static_assert(cuda::std::same_as<decltype(x), int&&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  // const &&
  {
    const cuda::std::expected<int, int> e(5);
    decltype(auto) x = cuda::std::move(e).value();
    static_assert(cuda::std::same_as<decltype(x), const int&&>, "");
    assert(&x == &(*e));
    assert(x == 5);
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
struct Error
{
  enum
  {
    Default,
    MutableRefCalled,
    ConstRefCalled,
    MutableRvalueCalled,
    ConstRvalueCalled
  } From  = Default;
  Error() = default;
  Error(const Error& e)
      : From(e.From)
  {
    if (e.From == Default)
    {
      From = ConstRefCalled;
    }
  }
  Error(Error& e)
      : From(e.From)
  {
    if (e.From == Default)
    {
      From = MutableRefCalled;
    }
  }
  Error(const Error&& e)
      : From(e.From)
  {
    if (e.From == Default)
    {
      From = ConstRvalueCalled;
    }
  }
  Error(Error&& e)
      : From(e.From)
  {
    if (e.From == Default)
    {
      From = MutableRvalueCalled;
    }
  }
};

void test_exceptions()
{
  try
  {
    const cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    (void) e.value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<int>& ex)
  {
    assert(ex.error() == 5);
  }

  // Test & overload
  try
  {
    cuda::std::expected<int, Error> e(cuda::std::unexpect);
    (void) e.value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<Error>& ex)
  {
    assert(ex.error().From == Error::ConstRefCalled);
  }

  // Test const& overload
  try
  {
    const cuda::std::expected<int, Error> e(cuda::std::unexpect);
    (void) e.value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<Error>& ex)
  {
    assert(ex.error().From == Error::ConstRefCalled);
  }

  // Test && overload
  try
  {
    cuda::std::expected<int, Error> e(cuda::std::unexpect);
    (void) cuda::std::move(e).value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<Error>& ex)
  {
    assert(ex.error().From == Error::MutableRvalueCalled);
  }

  // Test const&& overload
  try
  {
    const cuda::std::expected<int, Error> e(cuda::std::unexpect);
    (void) cuda::std::move(e).value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<Error>& ex)
  {
    assert(ex.error().From == Error::ConstRvalueCalled);
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
