//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>
#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cuda/experimental/container.cuh>

#include <stdexcept>

#include "helper.h"
#include "types.h"

TEMPLATE_TEST_CASE("cudax::synchronous_buffer assign",
                   "[container][synchronous_buffer]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Resource = typename extract_properties<TestType>::resource;
  using Buffer   = typename extract_properties<TestType>::synchronous_buffer;
  using T        = typename Buffer::value_type;

  Resource resource{};

  SECTION("cudax::synchronous_buffer::assign_range uncommon range")
  {
    { // cudax::synchronous_buffer::assign_range with an empty input
      Buffer buf{resource};
      buf.assign_range(uncommon_range<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // cudax::synchronous_buffer::assign_range with an empty input, shrinking
      Buffer buf{resource, 10, T(-2)};
      buf.assign_range(uncommon_range<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() != nullptr);
    }

    { // cudax::synchronous_buffer::assign_range with a non-empty input, shrinking
      Buffer buf{resource, 10, T(-2)};
      buf.assign_range(uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // cudax::synchronous_buffer::assign_range with a non-empty input, growing
      Buffer buf{resource, 4, T(-2)};
      buf.assign_range(uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("cudax::synchronous_buffer::assign_range sized uncommon range")
  {
    { // cudax::synchronous_buffer::assign_range with an empty input
      Buffer buf{resource};
      buf.assign_range(sized_uncommon_range<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // cudax::synchronous_buffer::assign_range with an empty input, shrinking
      Buffer buf{resource, 10, T(-2)};
      buf.assign_range(sized_uncommon_range<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() != nullptr);
    }

    { // cudax::synchronous_buffer::assign_range with a non-empty input, shrinking
      Buffer buf{resource, 10, T(-2)};
      buf.assign_range(sized_uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // cudax::synchronous_buffer::assign_range with a non-empty input, growing
      Buffer buf{resource, 4, T(-2)};
      buf.assign_range(sized_uncommon_range<T, 6>{{T(1), T(42), T(1337), T(0), T(12), T(-1)}});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("cudax::synchronous_buffer::assign_range random access range")
  {
    { // cudax::synchronous_buffer::assign_range with an empty input
      Buffer buf{resource};
      buf.assign_range(cuda::std::array<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // cudax::synchronous_buffer::assign_range with an empty input, shrinking
      Buffer buf{resource, 10, T(-2)};
      buf.assign_range(cuda::std::array<T, 0>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() != nullptr);
    }

    { // cudax::synchronous_buffer::assign_range with a non-empty input, shrinking
      Buffer buf{resource, 10, T(-2)};
      buf.assign_range(cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // cudax::synchronous_buffer::assign_range with a non-empty input, growing
      Buffer buf{resource, 4, T(-2)};
      buf.assign_range(cuda::std::array<T, 6>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("cudax::synchronous_buffer::assign(count, const T&)")
  {
    { // cudax::synchronous_buffer::assign(count, const T&), zero count from empty
      Buffer buf{resource};
      buf.assign(0, T(42));
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // cudax::synchronous_buffer::assign(count, const T&), shrinking to empty
      Buffer buf{resource, 10, T(-2)};
      buf.assign(0, T(42));
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() != nullptr);
    }

    { // cudax::synchronous_buffer::assign(count, const T&), shrinking
      Buffer buf{resource, 10, T(-2)};
      buf.assign(2, T(42));
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_size_value(buf, 2, T(42)));
    }

    { // cudax::synchronous_buffer::assign(count, const T&), growing
      Buffer buf{resource, 4, T(-2)};
      buf.assign(6, T(42));
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_size_value(buf, 6, T{42}));
    }
  }

  SECTION("cudax::synchronous_buffer::assign(iter, iter) forward iterators")
  {
    const cuda::std::array<T, 6> input = {T(1), T(42), T(1337), T(0), T(12), T(-1)};
    { // cudax::synchronous_buffer::assign(iter, iter), with forward iterators empty range
      Buffer buf{resource};
      buf.assign(input.begin(), input.begin());
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // cudax::synchronous_buffer::assign(iter, iter), with forward iterators shrinking to empty
      Buffer buf{resource, 10, T(-2)};
      buf.assign(input.begin(), input.begin());
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() != nullptr);
    }

    { // cudax::synchronous_buffer::assign(iter, iter), with forward iterators shrinking
      Buffer buf{resource, 10, T(-2)};
      buf.assign(input.begin(), input.end());
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // cudax::synchronous_buffer::assign(iter, iter), with forward iterators growing
      Buffer buf{resource, 4, T(-2)};
      buf.assign(input.begin(), input.end());
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("cudax::synchronous_buffer::assign(initializer_list)")
  {
    { // cudax::synchronous_buffer::assign(initializer_list), empty range
      Buffer buf{resource};
      buf.assign(cuda::std::initializer_list<T>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // cudax::synchronous_buffer::assign(initializer_list), shrinking to empty
      Buffer buf{resource, 10, T(-2)};
      buf.assign(cuda::std::initializer_list<T>{});
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() != nullptr);
    }

    { // cudax::synchronous_buffer::assign(initializer_list), shrinking
      Buffer buf{resource, 10, T(-2)};
      buf.assign(cuda::std::initializer_list<T>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // cudax::synchronous_buffer::assign(initializer_list), growing
      Buffer buf{resource, 4, T(-2)};
      buf.assign(cuda::std::initializer_list<T>{T(1), T(42), T(1337), T(0), T(12), T(-1)});
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

#if 0 // Implement exceptions
#  ifndef TEST_HAS_NO_EXCEPTIONS
  SECTION("cudax::synchronous_buffer::assign exception handling")
  {
    try
    {
      too_small.assign(2 * capacity, 42);
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      too_small.assign(input.begin(), input.end());
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }

    try
    {
      too_small.assign(cuda::std::initializer_list<int>{0, 1, 2, 3, 4, 5, 6});
    }
    catch (const std::bad_alloc&)
    {}
    catch (...)
    {
      CUDAX_CHECK(false);
    }
  }
#  endif // TEST_HAS_NO_EXCEPTIONS
#endif // Implement exceptions
}
