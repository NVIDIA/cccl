//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// pointer operator->() const; // constexpr in C++17

// Be sure to respect LWG 198:
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#198
// LWG 198 was superseded by LWG 2360
//    http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2360

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#  include <cuda/std/list>
#endif
#include <cuda/std/cassert>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(2734) // invalid arithmetic on non-array pointer
TEST_NV_DIAG_SUPPRESS(2693)

class A
{
  int data_;

public:
  __host__ __device__ A()
      : data_(1)
  {}
  __host__ __device__ ~A()
  {
    data_ = -1;
  }

  __host__ __device__ int get() const
  {
    return data_;
  }

  __host__ __device__ friend bool operator==(const A& x, const A& y)
  {
    return x.data_ == y.data_;
  }
};

template <class It>
__host__ __device__ void test(It i, typename cuda::std::iterator_traits<It>::value_type x)
{
  cuda::std::reverse_iterator<It> r(i);
  assert(r->get() == x.get());
}

class B
{
  int data_;

public:
  __host__ __device__ B(int d = 1)
      : data_(d)
  {}
  __host__ __device__ ~B()
  {
    data_ = -1;
  }

  __host__ __device__ int get() const
  {
    return data_;
  }

  __host__ __device__ friend bool operator==(const B& x, const B& y)
  {
    return x.data_ == y.data_;
  }
  __host__ __device__ const B* operator&() const
  {
    return nullptr;
  }
  __host__ __device__ B* operator&()
  {
    return nullptr;
  }
};

class C
{
  int data_;

public:
  __host__ __device__ TEST_CONSTEXPR C()
      : data_(1)
  {}

  __host__ __device__ TEST_CONSTEXPR int get() const
  {
    return data_;
  }

  __host__ __device__ friend TEST_CONSTEXPR bool operator==(const C& x, const C& y)
  {
    return x.data_ == y.data_;
  }
};

STATIC_TEST_GLOBAL_VAR TEST_CONSTEXPR_GLOBAL C gC[1];

int main(int, char**)
{
  A a;
  test(&a + 1, A());

#if defined(_LIBCUDACXX_HAS_LIST)
  {
    cuda::std::list<B> l;
    l.push_back(B(0));
    l.push_back(B(1));
    l.push_back(B(2));

    {
      cuda::std::list<B>::const_iterator i = l.begin();
      assert(i->get() == 0);
      ++i;
      assert(i->get() == 1);
      ++i;
      assert(i->get() == 2);
      ++i;
      assert(i == l.end());
    }

    {
      cuda::std::list<B>::const_reverse_iterator ri = l.rbegin();
      assert(ri->get() == 2);
      ++ri;
      assert(ri->get() == 1);
      ++ri;
      assert(ri->get() == 0);
      ++ri;
      assert(ri == l.rend());
    }
  }
#endif // defined(_LIBCUDACXX_HAS_LIST)

#if TEST_STD_VER > 2011 && !defined(TEST_COMPILER_NVRTC) && !defined(TEST_COMPILER_CUDACC_BELOW_11_3) \
  && defined(_CCCL_BUILTIN_ADDRESSOF)
  {
    typedef cuda::std::reverse_iterator<const C*> RI;
    constexpr RI it1 = cuda::std::make_reverse_iterator(gC + 1);

    static_assert(it1->get() == gC[0].get(), "");
  }
#endif // TEST_STD_VER > 2011 && !TEST_COMPILER_NVRTC && !TEST_COMPILER_CUDACC_BELOW_11_3 && _CCCL_BUILTIN_ADDRESSOF
  {
    unused(gC);
  }

  return 0;
}
