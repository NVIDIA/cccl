//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <functional>
//
// reference_wrapper
//
// template <class U>
//   reference_wrapper(U&&) noexcept(see below);

// #include <cuda/std/functional>
#include <cuda/std/utility>
#include <cuda/std/cassert>

#include "test_macros.h"

struct convertible_to_int_ref {
  int val = 0;
  __host__ __device__ operator int&() { return val; }
  __host__ __device__ operator int const&() const { return val; }
};

template <bool IsNothrow>
struct nothrow_convertible {
  int val = 0;
  __host__ __device__ operator int&() TEST_NOEXCEPT_COND(IsNothrow) { return val; }
};

struct convertible_from_int {
  __host__ __device__ convertible_from_int(int) {}
};

__host__ __device__ void meow(cuda::std::reference_wrapper<int>) {}
__host__ __device__ void meow(convertible_from_int) {}

int main(int, char**)
{
  {
    convertible_to_int_ref t;
    cuda::std::reference_wrapper<convertible_to_int_ref> r(t);
    assert(&r.get() == &t);
  }
  {
    const convertible_to_int_ref t{};
    cuda::std::reference_wrapper<const convertible_to_int_ref> r(t);
    assert(&r.get() == &t);
  }
  {
    using Ref = cuda::std::reference_wrapper<int>;
    ASSERT_NOEXCEPT(Ref(nothrow_convertible<true>()));
#if !defined(TEST_COMPILER_NVHPC) && !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
    ASSERT_NOT_NOEXCEPT(Ref(nothrow_convertible<false>()));
#endif // !defined(TEST_COMPILER_NVHPC) && !defined(__INTEL_COMPILER) && !defined(__INTEL_LLVM_COMPILER)
  }
  {
    meow(0);
  }
#if !defined(TEST_COMPILER_C1XX)
  {
    extern cuda::std::reference_wrapper<int> purr();
    ASSERT_SAME_TYPE(decltype(true ? purr() : 0), int);
  }
#endif // !defined(TEST_COMPILER_C1XX)
#if TEST_STD_VER > 14
#if (!defined(__GNUC__) || __GNUC__ >= 8) // gcc-7 is broken wrt ctad
  {
    int i = 0;
    cuda::std::reference_wrapper ri(i);
    static_assert((cuda::std::is_same<decltype(ri), cuda::std::reference_wrapper<int>>::value), "" );
    const int j = 0;
    cuda::std::reference_wrapper rj(j);
    static_assert((cuda::std::is_same<decltype(rj), cuda::std::reference_wrapper<const int>>::value), "" );
  }
#endif
#endif

  return 0;
}
