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

// NOTE: atomic<> of a TriviallyCopyable class is wrongly rejected by older
// clang versions. It was fixed right before the llvm 3.5 release. See PR18097.
// XFAIL: apple-clang-6.0, clang-3.4, clang-3.3

// <cuda/std/atomic>

#include <cuda/atomic>
#include <cuda/std/atomic>
#include <cuda/std/cassert>
#include <cuda/std/utility>
// #include <cuda/std/thread> // for thread_id
// #include <cuda/std/chrono> // for nanoseconds

#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test_not_copy_constructible()
{
  static_assert(!cuda::std::is_constructible<T, T&&>(), "");
  static_assert(!cuda::std::is_constructible<T, const T&>(), "");
  static_assert(!cuda::std::is_assignable<T, T&&>(), "");
  static_assert(!cuda::std::is_assignable<T, const T&>(), "");
}

template <class T>
__host__ __device__ void test_copy_constructible()
{
  static_assert(cuda::std::is_constructible<T, T&&>(), "");
  static_assert(cuda::std::is_constructible<T, const T&>(), "");
  static_assert(!cuda::std::is_assignable<T, T&&>(), "");
  static_assert(!cuda::std::is_assignable<T, const T&>(), "");
}

template <class T, class A>
__host__ __device__ void test_atomic_ref_copy_ctor()
{
  SHARED A val;
  val = 0;

  T t0(val);
  T t1(t0);

  t0++;
  t1++;

  assert(t1.load() == 2);
}

template <class T, class A>
__host__ __device__ void test_atomic_ref_move_ctor()
{
  SHARED A val;
  val = 0;

  T t0(val);
  t0++;

  T t1(cuda::std::move(t0));
  t1++;

  assert(t1.load() == 2);
}

int main(int, char**)
{
  test_not_copy_constructible<cuda::std::atomic<int>>();
  test_not_copy_constructible<cuda::atomic<int>>();

  test_copy_constructible<cuda::std::atomic_ref<int>>();
  test_copy_constructible<cuda::atomic_ref<int>>();

  test_atomic_ref_copy_ctor<cuda::std::atomic_ref<int>, int>();
  test_atomic_ref_copy_ctor<cuda::atomic_ref<int>, int>();
  test_atomic_ref_copy_ctor<const cuda::std::atomic_ref<int>, int>();
  test_atomic_ref_copy_ctor<const cuda::atomic_ref<int>, int>();

  test_atomic_ref_move_ctor<cuda::std::atomic_ref<int>, int>();
  test_atomic_ref_move_ctor<cuda::atomic_ref<int>, int>();
  test_atomic_ref_move_ctor<const cuda::std::atomic_ref<int>, int>();
  test_atomic_ref_move_ctor<const cuda::atomic_ref<int>, int>();
  // test(cuda::std::this_thread::get_id());
  // test(cuda::std::chrono::nanoseconds(2));

  return 0;
}
