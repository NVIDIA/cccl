//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: asm statement is unsupported in tile code

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/atomic>

// _Tp cuda::atomic_ref<_Tp>::fetch_max(_Tp, memory_order = memory_order_seq_cst) const;

#include <cuda/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class Selector>
TEST_FUNC void test_integral()
{
  using A = cuda::atomic_ref<T>;
  {
    Selector<T, constructor_initializer> sel;
    A a(*sel.construct(T(1)));
    assert(a.fetch_max(T(3)) == T(1));
    assert(a.load() == T(3));
  }
  {
    Selector<T, constructor_initializer> sel;
    A a(*sel.construct(T(3)));
    assert(a.fetch_max(T(1), cuda::std::memory_order_relaxed) == T(3));
    assert(a.load() == T(3));
  }
}

template <class T, template <typename, typename> class Selector>
TEST_FUNC void test_pointer()
{
  using X = cuda::std::remove_pointer_t<T>;
  using A = cuda::atomic_ref<T>;
  // Pointers into the same array have a well-defined ordering.
  X arr[8] = {};
  {
    Selector<T, constructor_initializer> sel;
    A a(*sel.construct(arr + 2));
    assert(a.fetch_max(arr + 5) == arr + 2);
    assert(a.load() == arr + 5);
  }
  {
    Selector<T, constructor_initializer> sel;
    A a(*sel.construct(arr + 5));
    assert(a.fetch_max(arr + 2) == arr + 5);
    assert(a.load() == arr + 5);
  }
}

template <template <typename, typename> class Selector>
TEST_FUNC void test()
{
  test_integral<int, Selector>();
  test_integral<unsigned int, Selector>();
  test_integral<long long, Selector>();
  test_integral<unsigned long long, Selector>();
  test_pointer<int*, Selector>();
  test_pointer<const int*, Selector>();
}

int main(int, char**)
{
  NV_DISPATCH_TARGET(NV_IS_HOST, (test<local_memory_selector>();), NV_PROVIDES_SM_70, (test<local_memory_selector>();))

  NV_IF_TARGET(NV_IS_DEVICE, (test<shared_memory_selector>(); test<global_memory_selector>();))

  return 0;
}
