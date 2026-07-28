//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// If the invocation of any non-const member function of `iterator` exits via an
// exception, the iterator acquires a singular value.

#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"
#include "test_macros.h"

struct ThrowOnIncrementIterator
{
  int* it_;

  using value_type       = int;
  using difference_type  = cuda::std::intptr_t;
  using iterator_concept = cuda::std::input_iterator_tag;

  ThrowOnIncrementIterator() = default;
  TEST_FUNC explicit ThrowOnIncrementIterator(int* it)
      : it_(it)
  {}

  TEST_FUNC ThrowOnIncrementIterator& operator++()
  {
    ++it_;
#if !TEST_CUDA_COMPILATION()
    TEST_THROW(5);
#endif
    return *this;
  }
  TEST_FUNC void operator++(int)
  {
    ++it_;
  }

  TEST_FUNC int& operator*() const
  {
    return *it_;
  }

  // nvrtc complains that = default is a host function. But if we add TEST_FUNC then nvcc will
  // complain that it ignores TEST_FUNC because it's = default
#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)
  friend bool operator==(ThrowOnIncrementIterator const&, ThrowOnIncrementIterator const&) = default;
#else // ^^^ C++20 && !nvrtc ^^^ / vvv C++17 || nvrtc vvv
  TEST_FUNC friend bool operator==(ThrowOnIncrementIterator const&, ThrowOnIncrementIterator const&);
  TEST_FUNC friend bool operator!=(ThrowOnIncrementIterator const&, ThrowOnIncrementIterator const&);
#endif // TEST_STD_VER <= 2017 || nvrtc
};

struct ThrowOnIncrementView : IntBufferView
{
  TEST_FUNC ThrowOnIncrementIterator begin() const
  {
    return ThrowOnIncrementIterator{buffer_};
  }
  TEST_FUNC ThrowOnIncrementIterator end() const
  {
    return ThrowOnIncrementIterator{buffer_ + size_};
  }
};

// Cannot run the test at compile time because it is not allowed to throw exceptions
TEST_FUNC void test()
{
#if TEST_HAS_EXCEPTIONS() && !TEST_CUDA_COMPILATION()
  int buffer[] = {1, 2, 3};
  {
    // zip iterator should be able to be destroyed after member function throws
    cuda::std::ranges::zip_view v{ThrowOnIncrementView{buffer}};
    auto it = v.begin();
    try
    {
      ++it;
      assert(false); // should not be reached as the above expression should throw.
    }
    catch (int e)
    {
      assert(e == 5);
    }
    catch (...)
    {
      assert(false); // wrong exception thrown
    }
  }

  {
    // zip iterator should be able to be assigned after member function throws
    cuda::std::ranges::zip_view v{ThrowOnIncrementView{buffer}};
    auto it = v.begin();
    try
    {
      ++it;
      assert(false); // should not be reached as the above expression should throw.
    }
    catch (int e)
    {
      assert(e == 5);
    }
    catch (...)
    {
      assert(false); // wrong exception thrown
    }

    it       = v.begin();
    auto [x] = *it;
    assert(x == 1);
  }
#endif
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
