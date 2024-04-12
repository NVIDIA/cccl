//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class I>
// unspecified iter_swap;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "../unqualified_lookup_wrapper.h"
#include "test_iterators.h"
#include "test_macros.h"

using IterSwapT = decltype(cuda::std::ranges::iter_swap);

struct HasIterSwap
{
  int& value_;
  __host__ __device__ constexpr explicit HasIterSwap(int& value)
      : value_(value)
  {
    assert(value == 0);
  }

  __host__ __device__ friend constexpr void iter_swap(HasIterSwap& a, HasIterSwap& b)
  {
    a.value_ = 1;
    b.value_ = 1;
  }
  __host__ __device__ friend constexpr void iter_swap(HasIterSwap& a, int& b)
  {
    a.value_ = 2;
    b        = 2;
  }
};

#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_MSVC_2017) // nvcc segfaults here
static_assert(cuda::std::is_invocable_v<IterSwapT, HasIterSwap&, HasIterSwap&>);
static_assert(cuda::std::is_invocable_v<IterSwapT, HasIterSwap&, int&>);
static_assert(!cuda::std::is_invocable_v<IterSwapT, int&, HasIterSwap&>);

static_assert(cuda::std::is_invocable_v<IterSwapT&, HasIterSwap&, HasIterSwap&>);
static_assert(cuda::std::is_invocable_v<IterSwapT&, HasIterSwap&, int&>);
static_assert(!cuda::std::is_invocable_v<IterSwapT&, int&, HasIterSwap&>);

static_assert(cuda::std::is_invocable_v<IterSwapT&&, HasIterSwap&, HasIterSwap&>);
static_assert(cuda::std::is_invocable_v<IterSwapT&&, HasIterSwap&, int&>);
static_assert(!cuda::std::is_invocable_v<IterSwapT&&, int&, HasIterSwap&>);
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_MSVC_2017

#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_MSVC_2017)
struct NodiscardIterSwap
{
  __host__ __device__ _CCCL_NODISCARD_FRIEND int iter_swap(NodiscardIterSwap&, NodiscardIterSwap&)
  {
    return 0;
  }
};

__host__ __device__ void ensureVoidCast(NodiscardIterSwap& a, NodiscardIterSwap& b)
{
  cuda::std::ranges::iter_swap(a, b);
}
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_MSVC_2017

struct HasRangesSwap
{
  int& value_;
  __host__ __device__ constexpr explicit HasRangesSwap(int& value)
      : value_(value)
  {
    assert(value == 0);
  }

  __host__ __device__ friend constexpr void swap(HasRangesSwap& a, HasRangesSwap& b)
  {
    a.value_ = 1;
    b.value_ = 1;
  }
  __host__ __device__ friend constexpr void swap(HasRangesSwap& a, int& b)
  {
    a.value_ = 2;
    b        = 2;
  }
};

struct HasRangesSwapWrapper
{
  using value_type = HasRangesSwap;

  HasRangesSwap& value_;
  __host__ __device__ constexpr explicit HasRangesSwapWrapper(HasRangesSwap& value)
      : value_(value)
  {}

  __host__ __device__ constexpr HasRangesSwap& operator*() const
  {
    return value_;
  }
};

#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_MSVC_2017) // nvcc segfaults here
static_assert(cuda::std::is_invocable_v<IterSwapT, HasRangesSwapWrapper&, HasRangesSwapWrapper&>);
// Does not satisfy swappable_with, even though swap(X, Y) is valid.
static_assert(!cuda::std::is_invocable_v<IterSwapT, HasRangesSwapWrapper&, int&>);
static_assert(!cuda::std::is_invocable_v<IterSwapT, int&, HasRangesSwapWrapper&>);
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_MSVC_2017

struct B;

struct A
{
  bool value = false;
  __host__ __device__ constexpr A& operator=(const B&)
  {
    value = true;
    return *this;
  };
};

struct B
{
  bool value = false;
  __host__ __device__ constexpr B& operator=(const A&)
  {
    value = true;
    return *this;
  };
};

struct MoveOnly2;

struct MoveOnly1
{
  bool value = false;

  MoveOnly1()                            = default;
  MoveOnly1(MoveOnly1&&)                 = default;
  MoveOnly1& operator=(MoveOnly1&&)      = default;
  MoveOnly1(const MoveOnly1&)            = delete;
  MoveOnly1& operator=(const MoveOnly1&) = delete;

  __host__ __device__ constexpr MoveOnly1& operator=(MoveOnly2&&)
  {
    value = true;
    return *this;
  };
};

struct MoveOnly2
{
  bool value = false;

  MoveOnly2()                            = default;
  MoveOnly2(MoveOnly2&&)                 = default;
  MoveOnly2& operator=(MoveOnly2&&)      = default;
  MoveOnly2(const MoveOnly2&)            = delete;
  MoveOnly2& operator=(const MoveOnly2&) = delete;

  __host__ __device__ constexpr MoveOnly2& operator=(MoveOnly1&&)
  {
    value = true;
    return *this;
  };
};

__host__ __device__ constexpr bool test()
{
#if !defined(TEST_COMPILER_CUDACC_BELOW_11_3) && !defined(TEST_COMPILER_MSVC_2017)
  {
    int value1 = 0;
    int value2 = 0;
    HasIterSwap a(value1), b(value2);
    cuda::std::ranges::iter_swap(a, b);
    assert(value1 == 1 && value2 == 1);
  }
  {
    int value1 = 0;
    int value2 = 0;
    HasRangesSwap c(value1), d(value2);
    HasRangesSwapWrapper cWrapper(c), dWrapper(d);
    cuda::std::ranges::iter_swap(cWrapper, dWrapper);
    assert(value1 == 1 && value2 == 1);
  }
  {
    int value1 = 0;
    int value2 = 0;
    HasRangesSwap c(value1), d(value2);
    cuda::std::ranges::iter_swap(HasRangesSwapWrapper(c), HasRangesSwapWrapper(d));
    assert(value1 == 1 && value2 == 1);
  }
  {
    A e;
    B f;
    A* ePtr = &e;
    B* fPtr = &f;
    cuda::std::ranges::iter_swap(ePtr, fPtr);
    assert(e.value && f.value);
  }

  {
    MoveOnly1 g;
    MoveOnly2 h;
    cuda::std::ranges::iter_swap(&g, &h);
    assert(g.value && h.value);
  }
#endif // !TEST_COMPILER_CUDACC_BELOW_11_3 && !TEST_COMPILER_MSVC_2017
#if TEST_HAS_BUILTIN(__builtin_is_constant_evaluated)
  {
    move_tracker arr[2];
    cuda::std::ranges::iter_swap(cuda::std::begin(arr), cuda::std::begin(arr) + 1);
    if (__builtin_is_constant_evaluated())
    {
      assert(arr[0].moves() == 1 && arr[1].moves() == 3);
    }
    else
    {
      assert(arr[0].moves() == 1 && arr[1].moves() == 2);
    }
  }
#endif
  {
    int buff[2] = {1, 2};
    cuda::std::ranges::iter_swap(buff + 0, buff + 1);
    assert(buff[0] == 2 && buff[1] == 1);

    cuda::std::ranges::iter_swap(cpp20_input_iterator<int*>(buff), cpp20_input_iterator<int*>(buff + 1));
    assert(buff[0] == 1 && buff[1] == 2);

    cuda::std::ranges::iter_swap(cpp17_input_iterator<int*>(buff), cpp17_input_iterator<int*>(buff + 1));
    assert(buff[0] == 2 && buff[1] == 1);

    cuda::std::ranges::iter_swap(forward_iterator<int*>(buff), forward_iterator<int*>(buff + 1));
    assert(buff[0] == 1 && buff[1] == 2);

    cuda::std::ranges::iter_swap(bidirectional_iterator<int*>(buff), bidirectional_iterator<int*>(buff + 1));
    assert(buff[0] == 2 && buff[1] == 1);

    cuda::std::ranges::iter_swap(random_access_iterator<int*>(buff), random_access_iterator<int*>(buff + 1));
    assert(buff[0] == 1 && buff[1] == 2);

    cuda::std::ranges::iter_swap(contiguous_iterator<int*>(buff), contiguous_iterator<int*>(buff + 1));
    assert(buff[0] == 2 && buff[1] == 1);
  }
  return true;
}

#ifndef TEST_COMPILER_CUDACC_BELOW_11_3 // nvcc segfaults here
static_assert(!cuda::std::is_invocable_v<IterSwapT, int*>); // too few arguments
static_assert(!cuda::std::is_invocable_v<IterSwapT, int*, int*, int*>); // too many arguments
static_assert(!cuda::std::is_invocable_v<IterSwapT, int, int*>);
static_assert(!cuda::std::is_invocable_v<IterSwapT, int*, int>);
static_assert(!cuda::std::is_invocable_v<IterSwapT, void*, void*>);
#endif // TEST_COMPILER_CUDACC_BELOW_11_3

#if TEST_STD_VER > 2017
// Test ADL-proofing.
struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
static_assert(cuda::std::is_invocable_v<IterSwapT, Holder<Incomplete>**, Holder<Incomplete>**>);
static_assert(cuda::std::is_invocable_v<IterSwapT, Holder<Incomplete>**, Holder<Incomplete>**&>);
static_assert(cuda::std::is_invocable_v<IterSwapT, Holder<Incomplete>**&, Holder<Incomplete>**>);
static_assert(cuda::std::is_invocable_v<IterSwapT, Holder<Incomplete>**&, Holder<Incomplete>**&>);
#endif

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
