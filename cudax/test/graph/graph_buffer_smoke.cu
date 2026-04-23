//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/cuda_toolkit.h>

#if _CCCL_CTK_AT_LEAST(12, 2)

#  include <cuda/memory_resource>

#  include <cuda/experimental/container.cuh>
#  include <cuda/experimental/graph.cuh>
#  include <cuda/experimental/launch.cuh>
#  include <cuda/experimental/stream.cuh>

#  include <testing.cuh>
#  include <utility.cuh>

namespace
{
namespace test
{
// RAII wrapper around a pinned-memory allocation.
template <typename T>
struct pinned_array
{
  cuda::mr::legacy_pinned_memory_resource __mr{};
  T* __ptr;
  std::size_t __n;

  explicit pinned_array(std::size_t __count, T __init = T{})
      : __ptr(static_cast<T*>(__mr.allocate_sync(__count * sizeof(T))))
      , __n(__count)
  {
    for (std::size_t i = 0; i < __n; ++i)
    {
      __ptr[i] = __init;
    }
  }
  ~pinned_array()
  {
    __mr.deallocate_sync(__ptr, __n * sizeof(T));
  }
  pinned_array(const pinned_array&)            = delete;
  pinned_array& operator=(const pinned_array&) = delete;

  T* get() const noexcept
  {
    return __ptr;
  }
  T& operator[](std::size_t i) const noexcept
  {
    return __ptr[i];
  }
};
} // namespace test

struct write_iota
{
  __device__ void operator()(cuda::std::span<int> buf) const noexcept
  {
    for (int i = 0; i < static_cast<int>(buf.size()); ++i)
    {
      buf[i] = i;
    }
  }
};

struct verify_iota
{
  __device__ void operator()(cuda::std::span<const int> buf) const noexcept
  {
    for (int i = 0; i < static_cast<int>(buf.size()); ++i)
    {
      CUDAX_REQUIRE(buf[i] == i);
    }
  }
};

struct verify_all_zero
{
  __device__ void operator()(cuda::std::span<const int> buf) const noexcept
  {
    for (const auto& val : buf)
    {
      CUDAX_REQUIRE(val == 0);
    }
  }
};

struct sum_to_ptr
{
  __device__ void operator()(cuda::std::span<const int> buf, int* out) const noexcept
  {
    int s = 0;
    for (const auto& val : buf)
    {
      s += val;
    }
    *out = s;
  }
};
} // namespace

C2H_TEST("graph_buffer with no_init allocates and can be written/read", "[graph][graph_buffer]")
{
  cudax::stream s{cuda::device_ref{0}};
  constexpr int N = 10;

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  cudax::graph_memory_resource mr{cuda::device_ref{0}};
  cudax::graph_buffer<int> buf(pb, mr, N, cuda::no_init);

  CUDAX_REQUIRE(buf.size() == N);
  CUDAX_REQUIRE(buf.data() != nullptr);
  STATIC_CHECK(decltype(buf)::properties_list::has_property(cuda::mr::device_accessible{}));

  // Write iota pattern to buffer
  cudax::launch(pb, test::one_thread_dims, write_iota{}, buf);

  // Read back and verify
  cudax::launch(pb, test::one_thread_dims, verify_iota{}, buf);

  // Free the buffer
  buf.destroy(pb);

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();
}

C2H_TEST("graph_buffer with zero-fill initializes to zero", "[graph][graph_buffer]")
{
  cudax::stream s{cuda::device_ref{0}};
  constexpr int N = 16;

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  cudax::graph_memory_resource mr{cuda::device_ref{0}};
  int zero = 0;
  cudax::graph_buffer<int> buf(pb, mr, N, zero);

  // Verify all zeros
  cudax::launch(pb, test::one_thread_dims, verify_all_zero{}, buf);

  buf.destroy(pb);

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();
}

C2H_TEST("graph_buffer from span", "[graph][graph_buffer]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> host_data{6};
  for (int i = 0; i < 6; ++i)
  {
    host_data[i] = i + 1;
  }

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  cudax::graph_memory_resource mr{cuda::device_ref{0}};
  cudax::graph_buffer<int> buf(pb, mr, cuda::std::span<const int>{host_data.get(), 6});

  CUDAX_REQUIRE(buf.size() == 6);

  test::pinned_array<int> result{6};
  cudax::copy_bytes(pb, buf, cuda::std::span<int>{result.get(), 6});

  buf.destroy(pb);

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  for (int i = 0; i < 6; ++i)
  {
    CUDAX_REQUIRE(result[i] == i + 1);
  }
}

C2H_TEST("graph_buffer from initializer_list", "[graph][graph_buffer]")
{
  cudax::stream s{cuda::device_ref{0}};

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  cudax::graph_memory_resource mr{cuda::device_ref{0}};
  cudax::graph_buffer<int> buf(pb, mr, {10, 20, 30, 40});

  CUDAX_REQUIRE(buf.size() == 4);

  test::pinned_array<int> result{4};
  cudax::copy_bytes(pb, buf, cuda::std::span<int>{result.get(), 4});

  buf.destroy(pb);

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  CUDAX_REQUIRE(result[0] == 10);
  CUDAX_REQUIRE(result[1] == 20);
  CUDAX_REQUIRE(result[2] == 30);
  CUDAX_REQUIRE(result[3] == 40);
}

C2H_TEST("make_buffer factory with no_init", "[graph][graph_buffer]")
{
  cudax::stream s{cuda::device_ref{0}};
  constexpr int N = 8;

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  cudax::graph_memory_resource mr{cuda::device_ref{0}};
  auto buf = cudax::make_buffer<int>(pb, mr, N, cuda::no_init);

  CUDAX_REQUIRE(buf.size() == N);

  cudax::launch(pb, test::one_thread_dims, write_iota{}, buf);
  cudax::launch(pb, test::one_thread_dims, verify_iota{}, buf);

  buf.destroy(pb);

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();
}

C2H_TEST("graph_buffer on forked paths", "[graph][graph_buffer]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> result_mem{1};
  int* result = result_mem.get();

  constexpr int N = 10;

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  cudax::graph_memory_resource mr{cuda::device_ref{0}};
  cudax::graph_buffer<int> buf(pb, mr, N, cuda::no_init);

  // Write on one path
  cudax::launch(pb, test::one_thread_dims, write_iota{}, buf);

  // Fork: read from the buffer on a second path
  auto read_path = cudax::start_path(g, pb);
  cudax::launch(read_path, test::one_thread_dims, sum_to_ptr{}, buf, result);

  // Join and free
  pb.wait(read_path);
  buf.destroy(pb);

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  // sum of 0..9 = 45
  CUDAX_REQUIRE(*result == 45);
}

C2H_TEST("graph_buffer move semantics", "[graph][graph_buffer]")
{
  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);
  cudax::stream s{cuda::device_ref{0}};

  cudax::graph_memory_resource mr{cuda::device_ref{0}};
  cudax::graph_buffer<int> buf1(pb, mr, 4, cuda::no_init);

  auto* original_data = buf1.data();
  auto original_size  = buf1.size();

  // Move construct
  cudax::graph_buffer<int> buf2(::cuda::std::move(buf1));
  CUDAX_REQUIRE(buf2.data() == original_data);
  CUDAX_REQUIRE(buf2.size() == original_size);
  CUDAX_REQUIRE(buf1.data() == nullptr);
  CUDAX_REQUIRE(buf1.size() == 0);

  // Move assign — create a second buffer and move-assign over it
  cudax::graph_buffer<int> buf3(pb, mr, 1, cuda::no_init);
  buf3.destroy(pb); // free the dummy allocation before overwriting
  buf3 = ::cuda::std::move(buf2);
  CUDAX_REQUIRE(buf3.data() == original_data);
  CUDAX_REQUIRE(buf3.size() == original_size);

  buf3.destroy(pb);

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();
}

C2H_TEST("graph_buffer empty buffer", "[graph][graph_buffer]")
{
  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);
  cudax::stream s{cuda::device_ref{0}};

  cudax::graph_memory_resource mr{cuda::device_ref{0}};
  cudax::graph_buffer<int> buf(pb, mr, 0, cuda::no_init);

  CUDAX_REQUIRE(buf.data() == nullptr);
  CUDAX_REQUIRE(buf.size() == 0);
  CUDAX_REQUIRE(buf.empty());

  // destroy on empty buffer should be a no-op
  buf.destroy(pb);

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();
}

#endif // _CCCL_CTK_AT_LEAST(12, 2)
