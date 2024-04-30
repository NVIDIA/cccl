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

// Test unique_ptr converting move assignment

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "test_macros.h"
#include "unique_ptr_test_helper.h"

template <class APtr, class BPtr>
__host__ __device__ void testAssign(APtr& aptr, BPtr& bptr)
{
  A* p = bptr.get();
  assert(A_count == 2);
  aptr = cuda::std::move(bptr);
  assert(aptr.get() == p);
  assert(bptr.get() == 0);
  assert(A_count == 1);
  assert(B_count == 1);
}

template <class LHS, class RHS>
__host__ __device__ void checkDeleter(LHS& lhs, RHS& rhs, int LHSState, int RHSState)
{
  assert(lhs.get_deleter().state() == LHSState);
  assert(rhs.get_deleter().state() == RHSState);
}

template <class T>
struct NCConvertingDeleter
{
  NCConvertingDeleter()                           = default;
  NCConvertingDeleter(NCConvertingDeleter const&) = delete;
  NCConvertingDeleter(NCConvertingDeleter&&)      = default;

  template <class U>
  __host__ __device__ NCConvertingDeleter(NCConvertingDeleter<U>&&)
  {}

  __host__ __device__ void operator()(T*) const {}
};

template <class T>
struct NCConvertingDeleter<T[]>
{
  NCConvertingDeleter()                           = default;
  NCConvertingDeleter(NCConvertingDeleter const&) = delete;
  NCConvertingDeleter(NCConvertingDeleter&&)      = default;

  template <class U>
  __host__ __device__ NCConvertingDeleter(NCConvertingDeleter<U>&&)
  {}

  __host__ __device__ void operator()(T*) const {}
};

struct GenericDeleter
{
  __host__ __device__ void operator()(void*) const;
};

struct NCGenericDeleter
{
  NCGenericDeleter()                        = default;
  NCGenericDeleter(NCGenericDeleter const&) = delete;
  NCGenericDeleter(NCGenericDeleter&&)      = default;

  __host__ __device__ void operator()(void*) const {}
};

__host__ __device__ void test_sfinae()
{
  using DA  = NCConvertingDeleter<A[]>; // non-copyable deleters
  using DAC = NCConvertingDeleter<const A[]>; // non-copyable deleters

  using UA   = cuda::std::unique_ptr<A[]>;
  using UAC  = cuda::std::unique_ptr<const A[]>;
  using UAD  = cuda::std::unique_ptr<A[], DA>;
  using UACD = cuda::std::unique_ptr<const A[], DAC>;

  { // cannot move from an lvalue
    static_assert(cuda::std::is_assignable<UAC, UA&&>::value, "");
    static_assert(!cuda::std::is_assignable<UAC, UA&>::value, "");
    static_assert(!cuda::std::is_assignable<UAC, const UA&>::value, "");
  }
  { // cannot move if the deleter-types cannot convert
    static_assert(cuda::std::is_assignable<UACD, UAD&&>::value, "");
    static_assert(!cuda::std::is_assignable<UACD, UAC&&>::value, "");
    static_assert(!cuda::std::is_assignable<UAC, UACD&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = cuda::std::unique_ptr<A[], DA&>;
    using UA2 = cuda::std::unique_ptr<A[], DAC&>;
    static_assert(!cuda::std::is_assignable<UA1, UA2&&>::value, "");
  }
  { // cannot move-convert with reference deleters of different types
    using UA1 = cuda::std::unique_ptr<A[], const DA&>;
    using UA2 = cuda::std::unique_ptr<A[], const DAC&>;
    static_assert(!cuda::std::is_assignable<UA1, UA2&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Single>
    using UA1 = cuda::std::unique_ptr<A[]>;
    using UA2 = cuda::std::unique_ptr<A>;
    static_assert(!cuda::std::is_assignable<UA1, UA2&&>::value, "");
  }
  { // cannot move-convert from unique_ptr<Array[]>
    using UA1 = cuda::std::unique_ptr<A[], NCGenericDeleter>;
    using UA2 = cuda::std::unique_ptr<A, NCGenericDeleter>;
    static_assert(!cuda::std::is_assignable<UA1, UA2&&>::value, "");
  }
}

int main(int, char**)
{
  test_sfinae();
  // FIXME: add tests

  return 0;
}
