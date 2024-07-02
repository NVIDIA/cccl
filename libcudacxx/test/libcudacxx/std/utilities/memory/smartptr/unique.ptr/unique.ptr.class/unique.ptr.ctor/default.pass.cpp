//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// unique_ptr

//=============================================================================
// TESTING cuda::std::unique_ptr::unique_ptr()
//
// Concerns:
//   1 The default constructor works for any default constructible deleter types.
//   2 The stored type 'T' is allowed to be incomplete.
//
// Plan
//  1 Default construct unique_ptr's with various deleter types (C-1)
//  2 Default construct a unique_ptr with an incomplete element_type and
//    various deleter types (C-1,2)

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "deleter_types.h"
#include "test_macros.h"
#include "unique_ptr_test_helper.h"

#ifndef TEST_COMPILER_NVRTC // no dynamic initialization
_LIBCUDACXX_SAFE_STATIC cuda::std::unique_ptr<int> global_static_unique_ptr_single;
_LIBCUDACXX_SAFE_STATIC cuda::std::unique_ptr<int[]> global_static_unique_ptr_runtime;
#endif // TEST_COMPILER_NVRTC

struct NonDefaultDeleter
{
  NonDefaultDeleter() = delete;
  __host__ __device__ TEST_CONSTEXPR_CXX23 void operator()(void*) const {}
};

template <class ElemType>
__host__ __device__ TEST_CONSTEXPR_CXX23 void test_sfinae()
{
  { // the constructor does not participate in overload resolution when
    // the deleter is a pointer type
    using U = cuda::std::unique_ptr<ElemType, void (*)(void*)>;
    static_assert(!cuda::std::is_default_constructible<U>::value, "");
  }
  { // the constructor does not participate in overload resolution when
    // the deleter is not default constructible
    using Del = CDeleter<ElemType>;
    using U1  = cuda::std::unique_ptr<ElemType, NonDefaultDeleter>;
    using U2  = cuda::std::unique_ptr<ElemType, Del&>;
    using U3  = cuda::std::unique_ptr<ElemType, Del const&>;
    static_assert(!cuda::std::is_default_constructible<U1>::value, "");
    static_assert(!cuda::std::is_default_constructible<U2>::value, "");
    static_assert(!cuda::std::is_default_constructible<U3>::value, "");
  }
}

template <class ElemType>
__host__ __device__ TEST_CONSTEXPR_CXX23 bool test_basic()
{
  {
    using U1 = cuda::std::unique_ptr<ElemType>;
    using U2 = cuda::std::unique_ptr<ElemType, Deleter<ElemType>>;
    static_assert(cuda::std::is_nothrow_default_constructible<U1>::value, "");
    static_assert(cuda::std::is_nothrow_default_constructible<U2>::value, "");
  }
  {
    cuda::std::unique_ptr<ElemType> p;
    assert(p.get() == 0);
  }
  {
    cuda::std::unique_ptr<ElemType, NCDeleter<ElemType>> p;
    assert(p.get() == 0);
    assert(p.get_deleter().state() == 0);
    p.get_deleter().set_state(5);
    assert(p.get_deleter().state() == 5);
  }
  {
    cuda::std::unique_ptr<ElemType, DefaultCtorDeleter<ElemType>> p;
    assert(p.get() == 0);
    assert(p.get_deleter().state() == 0);
  }

  return true;
}

#ifndef __CUDACC__
DEFINE_AND_RUN_IS_INCOMPLETE_TEST(
  {
    doIncompleteTypeTest(0);
    doIncompleteTypeTest<IncompleteType, Deleter<IncompleteType>>(0);
  } {
    doIncompleteTypeTest<IncompleteType[]>(0);
    doIncompleteTypeTest<IncompleteType[], Deleter<IncompleteType[]>>(0);
  })
#endif // __CUDACC__

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  {
    test_sfinae<int>();
    test_basic<int>();
  }
  {
    test_sfinae<int[]>();
    test_basic<int[]>();
  }

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
