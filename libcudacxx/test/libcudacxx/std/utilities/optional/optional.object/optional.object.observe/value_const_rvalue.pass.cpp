//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// constexpr const T& optional<T>::value() const &&;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::in_place;
using cuda::std::in_place_t;
using cuda::std::optional;
#if TEST_HAS_EXCEPTIONS()
using cuda::std::bad_optional_access;
#endif

struct X
{
  X()         = default;
  X(const X&) = delete;
  __host__ __device__ constexpr int test() const&
  {
    return 3;
  }
  __host__ __device__ constexpr int test() &
  {
    return 4;
  }
  __host__ __device__ constexpr int test() const&&
  {
    return 5;
  }
  __host__ __device__ constexpr int test() &&
  {
    return 6;
  }
};

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  const optional<X> opt{};
  try
  {
    (void) cuda::std::move(opt).value();
    assert(false);
  }
  catch (const bad_optional_access&)
  {}
}
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ constexpr bool test()
{
  {
    const optional<X> opt{};
    unused(opt);
    static_assert(!noexcept(cuda::std::move(opt).value()));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(opt).value()), const X&&>);

    const optional<X&> optref;
    unused(optref);
    static_assert(noexcept(cuda::std::move(optref).value()));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(optref).value()), X&>);
  }

  {
    const optional<X> opt{cuda::std::in_place};
    assert(cuda::std::move(opt).value().test() == 5);
  }

  {
    X val{};
    const optional<X&> opt{val};
    assert(cuda::std::move(opt).value().test() == 4);
    assert(cuda::std::addressof(val) == cuda::std::addressof(cuda::std::move(opt).value()));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
