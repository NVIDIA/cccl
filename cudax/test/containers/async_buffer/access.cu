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

TEMPLATE_TEST_CASE("cudax::async_buffer access",
                   "[container][async_buffer]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Env             = typename extract_properties<TestType>::env;
  using Resource        = typename extract_properties<TestType>::resource;
  using Buffer          = typename extract_properties<TestType>::async_buffer;
  using T               = typename Buffer::value_type;
  using reference       = typename Buffer::reference;
  using const_reference = typename Buffer::const_reference;
  using pointer         = typename Buffer::pointer;
  using const_pointer   = typename Buffer::const_pointer;

  cudax::stream stream{};
  Env env{Resource{}, stream};

  SECTION("cudax::async_buffer::data")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().data()), pointer>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().data()), const_pointer>);

    { // Works without allocation
      Buffer buf{env};
      CHECK(buf.data() == nullptr);
      CHECK(cuda::std::as_const(buf).data() == nullptr);
    }

    { // Works with allocation
      Buffer buf{env, {T(1), T(42), T(1337), T(0)}};
      CHECK(buf.data() != nullptr);
      CHECK(cuda::std::as_const(buf).data() != nullptr);
      CHECK(cuda::std::as_const(buf).data() == buf.data());
    }
  }
}
