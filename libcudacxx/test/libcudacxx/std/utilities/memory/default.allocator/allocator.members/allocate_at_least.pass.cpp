//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// allocation_result<T*> allocate_at_least(size_t n)

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>

#include "count_new.h"

#ifdef TEST_HAS_NO_ALIGNED_ALLOCATION
static const bool UsingAlignedNew = false;
#else
static const bool UsingAlignedNew = true;
#endif

#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
static const cuda::std::size_t MaxAligned = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
#else
static const cuda::std::size_t MaxAligned = cuda::std::alignment_of<cuda::std::max_align_t>::value;
#endif

static const cuda::std::size_t OverAligned = MaxAligned * 2;

template <cuda::std::size_t Align>
struct alignas(Align) AlignedType
{
  char data;
  static int constructed;
  __host__ __device__ AlignedType()
  {
    ++constructed;
  }
  __host__ __device__ AlignedType(AlignedType const&)
  {
    ++constructed;
  }
  __host__ __device__ ~AlignedType()
  {
    --constructed;
  }
};
template <cuda::std::size_t Align>
int AlignedType<Align>::constructed = 0;

template <cuda::std::size_t Align>
__host__ __device__ void test_aligned()
{
  typedef AlignedType<Align> T;
  T::constructed = 0;
  globalMemCounter.reset();
  cuda::std::allocator<T> a;
  const bool IsOverAlignedType = Align > MaxAligned;
  const bool ExpectAligned     = IsOverAlignedType && UsingAlignedNew;
  {
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(T::constructed == 0);
    globalMemCounter.last_new_size                                         = 0;
    globalMemCounter.last_new_align                                        = 0;
    cuda::std::same_as<cuda::std::allocation_result<T*>> decltype(auto) ap = a.allocate_at_least(3);
    assert(ap.count >= 3);
    DoNotOptimize(ap);
    assert(globalMemCounter.checkOutstandingNewEq(1));
    assert(globalMemCounter.checkNewCalledEq(1));
    assert(globalMemCounter.checkAlignedNewCalledEq(ExpectAligned));
    assert(globalMemCounter.checkLastNewSizeEq(3 * sizeof(T)));
    assert(globalMemCounter.checkLastNewAlignEq(ExpectAligned ? Align : 0));
    assert(T::constructed == 0);
    globalMemCounter.last_delete_align = 0;
    a.deallocate(ap.ptr, 3);
    assert(globalMemCounter.checkOutstandingNewEq(0));
    assert(globalMemCounter.checkDeleteCalledEq(1));
    assert(globalMemCounter.checkAlignedDeleteCalledEq(ExpectAligned));
    assert(globalMemCounter.checkLastDeleteAlignEq(ExpectAligned ? Align : 0));
    assert(T::constructed == 0);
  }
}

template <cuda::std::size_t Align>
__host__ __device__ constexpr bool test_aligned_constexpr()
{
  typedef AlignedType<Align> T;
  cuda::std::allocator<T> a;
  cuda::std::same_as<cuda::std::allocation_result<T*>> decltype(auto) ap = a.allocate_at_least(3);
  assert(ap.count >= 3);
  a.deallocate(ap.ptr, 3);

  return true;
}

int main(int, char**)
{
  test_aligned<1>();
  test_aligned<2>();
  test_aligned<4>();
  test_aligned<8>();
  test_aligned<16>();
  test_aligned<MaxAligned>();
  test_aligned<OverAligned>();
  test_aligned<OverAligned * 2>();

  static_assert(test_aligned_constexpr<1>());
  static_assert(test_aligned_constexpr<2>());
  static_assert(test_aligned_constexpr<4>());
  static_assert(test_aligned_constexpr<8>());
  static_assert(test_aligned_constexpr<16>());
  static_assert(test_aligned_constexpr<MaxAligned>());
  static_assert(test_aligned_constexpr<OverAligned>());
  static_assert(test_aligned_constexpr<OverAligned * 2>());
  return 0;
}
