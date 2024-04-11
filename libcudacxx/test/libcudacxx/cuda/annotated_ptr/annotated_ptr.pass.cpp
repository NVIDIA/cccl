//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc
// UNSUPPORTED: nvrtc
// UNSUPPORTED: c++98, c++03

#include "utils.h"

static_assert(sizeof(cuda::annotated_ptr<int, cuda::access_property::global>) == sizeof(uintptr_t),
              "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<char, cuda::access_property::global>) == sizeof(uintptr_t),
              "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::global>) == sizeof(uintptr_t),
              "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::persisting>) == sizeof(uintptr_t),
              "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::normal>) == sizeof(uintptr_t),
              "annotated_ptr<T, global> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::streaming>) == sizeof(uintptr_t),
              "annotated_ptr<T, global> must be pointer size");

static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property>) == 2 * sizeof(uintptr_t),
              "annotated_ptr<T,access_property> must be 2 * pointer size");

// NOTE: we could make these smaller in the future (e.g. 32-bit) but that would be an ABI breaking change:
static_assert(sizeof(cuda::annotated_ptr<int, cuda::access_property::shared>) == sizeof(uintptr_t),
              "annotated_ptr<T, shared> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<char, cuda::access_property::shared>) == sizeof(uintptr_t),
              "annotated_ptr<T, shared> must be pointer size");
static_assert(sizeof(cuda::annotated_ptr<uintptr_t, cuda::access_property::shared>) == sizeof(uintptr_t),
              "annotated_ptr<T, shared> must be pointer size");

static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::global>) == alignof(int*),
              "annotated_ptr must align with int*");
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::persisting>) == alignof(int*),
              "annotated_ptr must align with int*");
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::normal>) == alignof(int*),
              "annotated_ptr must align with int*");
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::streaming>) == alignof(int*),
              "annotated_ptr must align with int*");
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property>) == alignof(int*),
              "annotated_ptr must align with int*");

// NOTE: we could lower the alignment in the future but that would be an ABI breaking change:
static_assert(alignof(cuda::annotated_ptr<int, cuda::access_property::shared>) == alignof(int*),
              "annotated_ptr must align with int*");

#define N 128

struct S
{
  int x;
  __host__ __device__ S& operator=(int o)
  {
    this->x = o;
    return *this;
  }
};

template <typename In, typename T>
__device__ __host__ __noinline__ void test_read_access(In i, T* r)
{
  assert(i);
  assert(i - i == 0);
  assert((bool) i);
  const In o = i;

  // assert(i->x == 0);  // FAILS with shmem
  // assert(o->x == 0);  // FAILS with shmem
  for (int n = 0; n < N; ++n)
  {
    assert(i[n].x == n);
    assert(&i[n] == &i[n]);
    assert(&i[n] == &r[n]);
    assert(o[n].x == n);
    assert(&o[n] == &o[n]);
    assert(&o[n] == &r[n]);
  }
}

template <typename In>
__device__ __host__ __noinline__ void test_write_access(In i)
{
  assert(i);
  assert((bool) i);
  const In o = i;

  for (int n = 0; n < N; ++n)
  {
    i[n].x = 2 * n;
    assert(i[n].x == 2 * n);
    assert(i[n].x == 2 * n);
    i[n].x = n;

    o[n].x = 2 * n;
    assert(o[n].x == 2 * n);
    assert(o[n].x == 2 * n);
    o[n].x = n;
  }
}

__device__ __host__ __noinline__ void all_tests()
{
  S* arr = alloc<S, N>(false);

  test_read_access(cuda::annotated_ptr<S, cuda::access_property::normal>(arr), arr);
  test_read_access(cuda::annotated_ptr<S, cuda::access_property::streaming>(arr), arr);
  test_read_access(cuda::annotated_ptr<S, cuda::access_property::persisting>(arr), arr);
  test_read_access(cuda::annotated_ptr<S, cuda::access_property::global>(arr), arr);
  test_read_access(cuda::annotated_ptr<S, cuda::access_property>(arr), arr);

  test_read_access(cuda::annotated_ptr<const S, cuda::access_property::normal>(arr), arr);
  test_read_access(cuda::annotated_ptr<const S, cuda::access_property::streaming>(arr), arr);
  test_read_access(cuda::annotated_ptr<const S, cuda::access_property::persisting>(arr), arr);
  test_read_access(cuda::annotated_ptr<const S, cuda::access_property::global>(arr), arr);
  test_read_access(cuda::annotated_ptr<const S, cuda::access_property>(arr), arr);

  test_read_access(cuda::annotated_ptr<volatile S, cuda::access_property::normal>(arr), arr);
  test_read_access(cuda::annotated_ptr<volatile S, cuda::access_property::streaming>(arr), arr);
  test_read_access(cuda::annotated_ptr<volatile S, cuda::access_property::persisting>(arr), arr);
  test_read_access(cuda::annotated_ptr<volatile S, cuda::access_property::global>(arr), arr);
  test_read_access(cuda::annotated_ptr<volatile S, cuda::access_property>(arr), arr);

  test_read_access(cuda::annotated_ptr<const volatile S, cuda::access_property::normal>(arr), arr);
  test_read_access(cuda::annotated_ptr<const volatile S, cuda::access_property::streaming>(arr), arr);
  test_read_access(cuda::annotated_ptr<const volatile S, cuda::access_property::persisting>(arr), arr);
  test_read_access(cuda::annotated_ptr<const volatile S, cuda::access_property::global>(arr), arr);
  test_read_access(cuda::annotated_ptr<const volatile S, cuda::access_property>(arr), arr);

  test_write_access(cuda::annotated_ptr<S, cuda::access_property::normal>(arr));
  test_write_access(cuda::annotated_ptr<S, cuda::access_property::streaming>(arr));
  test_write_access(cuda::annotated_ptr<S, cuda::access_property::persisting>(arr));
  test_write_access(cuda::annotated_ptr<S, cuda::access_property::global>(arr));
  test_write_access(cuda::annotated_ptr<S, cuda::access_property>(arr));

  test_write_access(cuda::annotated_ptr<volatile S, cuda::access_property::normal>(arr));
  test_write_access(cuda::annotated_ptr<volatile S, cuda::access_property::streaming>(arr));
  test_write_access(cuda::annotated_ptr<volatile S, cuda::access_property::persisting>(arr));
  test_write_access(cuda::annotated_ptr<volatile S, cuda::access_property::global>(arr));
  test_write_access(cuda::annotated_ptr<volatile S, cuda::access_property>(arr));

  NV_IF_TARGET(
    NV_IS_DEVICE,
    (S* sarr = alloc<S, N>(true); test_read_access(cuda::annotated_ptr<S, cuda::access_property::shared>(sarr), sarr);
     test_read_access(cuda::annotated_ptr<const S, cuda::access_property::shared>(sarr), sarr);
     test_read_access(cuda::annotated_ptr<volatile S, cuda::access_property::shared>(sarr), sarr);
     test_read_access(cuda::annotated_ptr<const volatile S, cuda::access_property::shared>(sarr), sarr);
     test_write_access(cuda::annotated_ptr<S, cuda::access_property::shared>(sarr));
     test_write_access(cuda::annotated_ptr<volatile S, cuda::access_property::shared>(sarr));))
}

int main(int argc, char** argv)
{
  all_tests();
  return 0;
}
