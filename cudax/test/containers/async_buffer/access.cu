//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
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

#if _CCCL_CUDACC_AT_LEAST(12, 6)
using test_types = c2h::type_list<cuda::std::tuple<cuda::mr::host_accessible>,
                                  cuda::std::tuple<cuda::mr::device_accessible>,
                                  cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>>;
#else
using test_types = c2h::type_list<cuda::std::tuple<cuda::mr::device_accessible>>;
#endif

C2H_CCCLRT_TEST("cudax::async_buffer access and stream", "[container][async_buffer]", test_types)
{
  using TestT           = c2h::get<0, TestType>;
  using Env             = typename extract_properties<TestT>::env;
  using Resource        = typename extract_properties<TestT>::resource;
  using Buffer          = typename extract_properties<TestT>::async_buffer;
  using T               = typename Buffer::value_type;
  using reference       = typename Buffer::reference;
  using const_reference = typename Buffer::const_reference;
  using pointer         = typename Buffer::pointer;
  using const_pointer   = typename Buffer::const_pointer;

  cudax::stream stream{cuda::device_ref{0}};
  Env env{Resource{}, stream};

  SECTION("cudax::async_buffer::get_unsynchronized")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().get_unsynchronized(1ull)), reference>);
    static_assert(
      cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().get_unsynchronized(1ull)), const_reference>);

    {
      Buffer buf{env, {T(1), T(42), T(1337), T(0)}};
      buf.stream().sync();
      auto& res = buf.get_unsynchronized(2);
      CUDAX_CHECK(compare_value<Buffer::__is_host_only>(res, T(1337)));
      CUDAX_CHECK(static_cast<size_t>(cuda::std::addressof(res) - buf.data()) == 2);
      assign_value<Buffer::__is_host_only>(res, T(4));

      auto& const_res = cuda::std::as_const(buf).get_unsynchronized(2);
      CUDAX_CHECK(compare_value<Buffer::__is_host_only>(const_res, T(4)));
      CUDAX_CHECK(static_cast<size_t>(cuda::std::addressof(const_res) - buf.data()) == 2);
    }
  }

  SECTION("cudax::async_buffer::data")
  {
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<Buffer&>().data()), pointer>);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const Buffer&>().data()), const_pointer>);

    { // Works without allocation
      Buffer buf{env};
      buf.stream().sync();
      CUDAX_CHECK(buf.data() == nullptr);
      CUDAX_CHECK(cuda::std::as_const(buf).data() == nullptr);
    }

    { // Works with allocation
      Buffer buf{env, {T(1), T(42), T(1337), T(0)}};
      buf.stream().sync();
      CUDAX_CHECK(buf.data() != nullptr);
      CUDAX_CHECK(cuda::std::as_const(buf).data() != nullptr);
      CUDAX_CHECK(cuda::std::as_const(buf).data() == buf.data());
    }
  }

  SECTION("cudax::async_buffer::stream")
  {
    Buffer buf{env, {T(1), T(42), T(1337), T(0)}};
    CUDAX_CHECK(buf.stream() == stream);

    {
      cudax::stream other_stream{cuda::device_ref{0}};
      buf.set_stream_unsynchronized(other_stream);
      CUDAX_CHECK(buf.stream() == other_stream);
      // TODO swap to synchronized setter one cudax::stream_ref is moved to cuda namespace
      buf.set_stream_unsynchronized(stream);
    }

    CUDAX_CHECK(buf.stream() == stream);
  }
}
