//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03

// <memory>

// unique_ptr

// Test unique_ptr converting move ctor

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "deleter_types.h"
#include "test_macros.h"
#include "unique_ptr_test_helper.h"

template <int ID = 0>
struct GenericDeleter
{
  __host__ __device__ void operator()(void*) const {}
};

template <int ID = 0>
struct GenericConvertingDeleter
{
  template <int OID>
  __host__ __device__ GenericConvertingDeleter(GenericConvertingDeleter<OID>)
  {}
  __host__ __device__ void operator()(void*) const {}
};

__host__ __device__ TEST_CONSTEXPR_CXX23 void test_sfinae()
{
  { // Disallow copying
    using U1 = cuda::std::unique_ptr<A[], GenericConvertingDeleter<0>>;
    using U2 = cuda::std::unique_ptr<A[], GenericConvertingDeleter<1>>;
    static_assert(cuda::std::is_constructible<U1, U2&&>::value, "");
    static_assert(!cuda::std::is_constructible<U1, U2&>::value, "");
    static_assert(!cuda::std::is_constructible<U1, const U2&>::value, "");
    static_assert(!cuda::std::is_constructible<U1, const U2&&>::value, "");
  }
  { // Disallow illegal qualified conversions
    using U1 = cuda::std::unique_ptr<const A[]>;
    using U2 = cuda::std::unique_ptr<A[]>;
    static_assert(cuda::std::is_constructible<U1, U2&&>::value, "");
    static_assert(!cuda::std::is_constructible<U2, U1&&>::value, "");
  }
  { // Disallow base-to-derived conversions.
    using UA = cuda::std::unique_ptr<A[]>;
    using UB = cuda::std::unique_ptr<B[]>;
    static_assert(!cuda::std::is_constructible<UA, UB&&>::value, "");
  }
  { // Disallow base-to-derived conversions.
    using UA = cuda::std::unique_ptr<A[], GenericConvertingDeleter<0>>;
    using UB = cuda::std::unique_ptr<B[], GenericConvertingDeleter<1>>;
    static_assert(!cuda::std::is_constructible<UA, UB&&>::value, "");
  }
  { // Disallow invalid deleter initialization
    using U1 = cuda::std::unique_ptr<A[], GenericDeleter<0>>;
    using U2 = cuda::std::unique_ptr<A[], GenericDeleter<1>>;
    static_assert(!cuda::std::is_constructible<U1, U2&&>::value, "");
  }
  { // Disallow reference deleters with different qualifiers
    using U1 = cuda::std::unique_ptr<A[], Deleter<A[]>&>;
    using U2 = cuda::std::unique_ptr<A[], const Deleter<A[]>&>;
    static_assert(!cuda::std::is_constructible<U1, U2&&>::value, "");
    static_assert(!cuda::std::is_constructible<U2, U1&&>::value, "");
  }
  {
    using U1 = cuda::std::unique_ptr<A[]>;
    using U2 = cuda::std::unique_ptr<A>;
    static_assert(!cuda::std::is_constructible<U1, U2&&>::value, "");
    static_assert(!cuda::std::is_constructible<U2, U1&&>::value, "");
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  test_sfinae();

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
