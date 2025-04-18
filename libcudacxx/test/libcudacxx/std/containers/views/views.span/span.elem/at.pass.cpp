//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/span>

// constexpr reference at(size_type idx) const;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"

#if TEST_HAS_EXCEPTIONS()
#  include <stdexcept>
#endif // TEST_HAS_EXCEPTIONS()

template <typename ReferenceT, typename SpanT>
__host__ __device__ constexpr void testSpanAt(SpanT&& anySpan, int index, int expectedValue)
{
  // non-const
  {
    auto elem = anySpan.at(index);
    static_assert(cuda::std::is_same_v<ReferenceT, decltype(anySpan.at(index))>);
    assert(elem == expectedValue);
  }

  // const
  {
    auto elem = cuda::std::as_const(anySpan).at(index);
    static_assert(cuda::std::is_same_v<ReferenceT, decltype(cuda::std::as_const(anySpan).at(index))>);
    assert(elem == expectedValue);
  }
}

__host__ __device__ constexpr bool test()
{
  // With static extent
  {
    cuda::std::array<int, 7> arr{0, 1, 2, 3, 4, 5, 9084};
    cuda::std::span<int, 7> arrSpan{arr};

    assert(cuda::std::dynamic_extent != arrSpan.extent);

    using ReferenceT = typename decltype(arrSpan)::reference;

    testSpanAt<ReferenceT>(arrSpan, 0, 0);
    testSpanAt<ReferenceT>(arrSpan, 1, 1);
    testSpanAt<ReferenceT>(arrSpan, 6, 9084);
  }

  // With dynamic extent
  {
    cuda::std::array<int, 7> arr{0, 1, 2, 3, 4, 5, 9084};
    cuda::std::span<int> dynSpan{arr};

    assert(cuda::std::dynamic_extent == dynSpan.extent);

    using ReferenceT = typename decltype(dynSpan)::reference;

    testSpanAt<ReferenceT>(dynSpan, 0, 0);
    testSpanAt<ReferenceT>(dynSpan, 1, 1);
    testSpanAt<ReferenceT>(dynSpan, 6, 9084);
  }

  return true;
}

#if TEST_HAS_EXCEPTIONS()
void test_exceptions()
{
  // With static extent
  {
    cuda::std::array<int, 8> arr{0, 1, 2, 3, 4, 5, 9084, cuda::std::numeric_limits<int>::max()};
    const cuda::std::span<int, 8> arrSpan{arr};

    try
    {
      using SizeT       = typename decltype(arrSpan)::size_type;
      cuda::std::ignore = arrSpan.at(cuda::std::numeric_limits<SizeT>::max());
      assert(false);
    }
    catch (const std::out_of_range&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }

    try
    {
      cuda::std::ignore = arrSpan.at(arr.size());
      assert(false);
    }
    catch (const std::out_of_range&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }

    try
    {
      cuda::std::ignore = arrSpan.at(arr.size() - 1);
      // pass
      assert(arrSpan.at(arr.size() - 1) == cuda::std::numeric_limits<int>::max());
    }
    catch (...)
    {
      assert(false);
    }
  }

  {
    cuda::std::array<int, 0> arr{};
    const cuda::std::span<int, 0> arrSpan{arr};

    try
    {
      cuda::std::ignore = arrSpan.at(0);
      assert(false);
    }
    catch (const std::out_of_range&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }
  }

  // With dynamic extent

  {
    cuda::std::array<int, 8> arr{0, 1, 2, 3, 4, 5, 9084, cuda::std::numeric_limits<int>::max()};
    const cuda::std::span<int> dynSpan{arr};

    try
    {
      using SizeT       = typename decltype(dynSpan)::size_type;
      cuda::std::ignore = dynSpan.at(cuda::std::numeric_limits<SizeT>::max());
      assert(false);
    }
    catch (const std::out_of_range&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }

    try
    {
      cuda::std::ignore = dynSpan.at(arr.size());
      assert(false);
    }
    catch (const std::out_of_range&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }

    try
    {
      cuda::std::ignore = dynSpan.at(arr.size() - 1);
      assert(dynSpan.at(arr.size() - 1) == cuda::std::numeric_limits<int>::max());
    }
    catch (...)
    {
      assert(false);
    }
  }

  {
    cuda::std::array<int, 0> arr{};
    const cuda::std::span<int> dynSpan{arr};

    try
    {
      cuda::std::ignore = dynSpan.at(0);
      assert(false);
    }
    catch (const std::out_of_range&)
    {
      // pass
    }
    catch (...)
    {
      assert(false);
    }
  }
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test(), "");

#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()

  return 0;
}
