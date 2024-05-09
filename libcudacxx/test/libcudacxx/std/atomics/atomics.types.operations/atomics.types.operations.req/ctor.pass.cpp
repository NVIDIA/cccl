//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
// UNSUPPORTED: c++98, c++03

// NOTE: atomic<> of a TriviallyCopyable class is wrongly rejected by older
// clang versions. It was fixed right before the llvm 3.5 release. See PR18097.
// XFAIL: apple-clang-6.0, clang-3.4, clang-3.3

// <cuda/std/atomic>

// constexpr atomic<T>::atomic(T value)

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

struct UserType
{
  int i;

  __host__ __device__ UserType() noexcept {}
  __host__ __device__ constexpr explicit UserType(int d) noexcept
      : i(d)
  {}

  __host__ __device__ friend bool operator==(const UserType& x, const UserType& y)
  {
    return x.i == y.i;
  }
};

template <class Tp, template <typename, typename> class, cuda::thread_scope Scope>
struct TestFunc
{
  __host__ __device__ void operator()() const
  {
    typedef cuda::atomic<Tp, Scope> Atomic;
    static_assert(cuda::std::is_literal_type<Atomic>::value, "");
    constexpr Tp t(42);
    {
      constexpr Atomic a(t);
      assert(a == t);
    }
    {
      constexpr Atomic a{t};
      assert(a == t);
    }
#if !defined(_GNUC_VER) || _GNUC_VER >= 409
    // TODO: Figure out why this is failing with GCC 4.8.2 on CentOS 7 only.
    {
      constexpr Atomic a = LIBCUDACXX_ATOMIC_VAR_INIT(t);
      assert(a == t);
    }
#endif
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (TestFunc<UserType, local_memory_selector, cuda::thread_scope_system>()();
     TestEachIntegralType<TestFunc, local_memory_selector, cuda::thread_scope_system>()();),
    NV_PROVIDES_SM_60,
    (TestFunc<UserType, local_memory_selector, cuda::thread_scope_system>()();
     TestEachIntegralType<TestFunc, local_memory_selector, cuda::thread_scope_system>()();))

  TestFunc<UserType, local_memory_selector, cuda::thread_scope_device>()();
  TestEachIntegralType<TestFunc, local_memory_selector, cuda::thread_scope_device>()();
  TestFunc<UserType, local_memory_selector, cuda::thread_scope_block>()();
  TestEachIntegralType<TestFunc, local_memory_selector, cuda::thread_scope_block>()();

  return 0;
}
