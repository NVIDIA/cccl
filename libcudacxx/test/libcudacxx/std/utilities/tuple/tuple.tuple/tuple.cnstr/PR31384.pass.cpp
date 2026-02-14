// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Some early versions (cl.exe 14.16 / VC141) do not identify correct constructors
// UNSUPPORTED: msvc

// <cuda/std/tuple>

// template <class TupleLike> tuple(TupleLike&&); // libc++ extension

// See llvm.org/PR31384
#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE int count = 0;

struct Explicit
{
  Explicit() = default;
  __host__ __device__ explicit Explicit(int) {}
};

struct Implicit
{
  Implicit() = default;
  __host__ __device__ Implicit(int) {}
};

template <class T>
struct Derived : cuda::std::tuple<T>
{
  using cuda::std::tuple<T>::tuple;
  template <class U>
  __host__ __device__ operator cuda::std::tuple<U>() &&
  {
    ++count;
    return {};
  }
};

template <class T>
struct ExplicitDerived : cuda::std::tuple<T>
{
  using cuda::std::tuple<T>::tuple;
  template <class U>
  __host__ __device__ explicit operator cuda::std::tuple<U>() &&
  {
    ++count;
    return {};
  }
};

int main(int, char**)
{
  {
    [[maybe_unused]] cuda::std::tuple<Explicit> foo = Derived<int>{42};
    assert(count == 1);
    [[maybe_unused]] cuda::std::tuple<Explicit> bar(Derived<int>{42});
    assert(count == 2);
  }
  count = 0;
  {
    [[maybe_unused]] cuda::std::tuple<Implicit> foo = Derived<int>{42};
    assert(count == 1);
    [[maybe_unused]] cuda::std::tuple<Implicit> bar(Derived<int>{42});
    assert(count == 2);
  }
  count = 0;
  {
    static_assert(!cuda::std::is_convertible<ExplicitDerived<int>, cuda::std::tuple<Explicit>>::value, "");
    [[maybe_unused]] cuda::std::tuple<Explicit> bar(ExplicitDerived<int>{42});
    assert(count == 1);
  }
  count = 0;
  {
    // FIXME: Libc++ incorrectly rejects this code.
#ifndef _CUDA_STD_VERSION
    cuda::std::tuple<Implicit> foo = ExplicitDerived<int>{42};
    static_assert(cuda::std::is_convertible<ExplicitDerived<int>, cuda::std::tuple<Implicit>>::value,
                  "correct STLs accept this");
#else
    static_assert(!cuda::std::is_convertible<ExplicitDerived<int>, cuda::std::tuple<Implicit>>::value,
                  "libc++ incorrectly rejects this");
#endif
    assert(count == 0);
    [[maybe_unused]] cuda::std::tuple<Implicit> bar(ExplicitDerived<int>{42});
    assert(count == 1);
  }
  count = 0;

  return 0;
}
