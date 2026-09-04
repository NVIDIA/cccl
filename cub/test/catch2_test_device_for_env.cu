// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/__execution/policy.h>
#include <cuda/__execution/tune.h>
#include <cuda/devices>
#include <cuda/std/execution>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cstdint>
#include <sstream>

#include "cub_test_macros.h"

struct square_ref_op
{
  __device__ void operator()(int& i)
  {
    i *= i;
  }
};

struct square_idx_op
{
  int* d_ptr;

  __device__ void operator()(int i)
  {
    d_ptr[i] *= d_ptr[i];
  }
};

struct odd_count_op
{
  int* d_count;

  __device__ void operator()(int i)
  {
    if (i % 2 == 1)
    {
      atomicAdd(d_count, 1);
    }
  }
};

// c2h selects the device via -d/--device, so the stream must be created on the current device;
// c2h::device_vector allocates there, and a device 0 stream would cross devices.
[[nodiscard]] cuda::stream make_current_device_stream()
{
  int device_id{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device_id));
  return cuda::stream{cuda::devices[device_id]};
}

// -----------------------------------------------------------------------
// Bulk
// -----------------------------------------------------------------------

CUB_TEST("DeviceFor::Bulk env uses custom stream", "[for][env]", CUB_SMALL)
{
  auto vec = c2h::device_vector<int>{1, 2, 3, 4};
  const square_idx_op op{thrust::raw_pointer_cast(vec.data())};

  cuda::stream stream = make_current_device_stream();
  const auto env      = cuda::std::execution::env{cuda::stream_ref{stream}};

  const auto error = cub::DeviceFor::Bulk(4, op, env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  const c2h::device_vector<int> expected{1, 4, 9, 16};
  REQUIRE(vec == expected);
}

// -----------------------------------------------------------------------
// ForEachN
// -----------------------------------------------------------------------

CUB_TEST("DeviceFor::ForEachN env uses custom stream", "[for][env]", CUB_SMALL)
{
  auto vec = c2h::device_vector<int>{1, 2, 3, 4};
  const square_ref_op op{};

  cuda::stream stream = make_current_device_stream();
  const auto env      = cuda::std::execution::env{cuda::stream_ref{stream}};

  const auto error = cub::DeviceFor::ForEachN(vec.begin(), static_cast<int>(vec.size()), op, env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  const c2h::device_vector<int> expected{1, 4, 9, 16};
  REQUIRE(vec == expected);
}

// __for_each_n is the internal two-phase entry point other device algorithms compose with
CUB_TEST("DeviceFor::__for_each_n two-phase overload takes an environment", "[for][env]", CUB_SMALL)
{
  auto vec = c2h::device_vector<int>{1, 2, 3, 4};
  const square_ref_op op{};
  const auto num_items = static_cast<int>(vec.size());

  cuda::stream stream = make_current_device_stream();
  const auto env      = cuda::std::execution::env{cuda::stream_ref{stream}};

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == cub::DeviceFor::__for_each_n(nullptr, temp_storage_bytes, vec.begin(), num_items, op, env));
  REQUIRE(temp_storage_bytes > 0);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  REQUIRE(
    cudaSuccess == cub::DeviceFor::__for_each_n(d_temp_storage, temp_storage_bytes, vec.begin(), num_items, op, env));
  REQUIRE(cudaSuccess == cudaStreamSynchronize(stream.get()));

  const c2h::device_vector<int> expected{1, 4, 9, 16};
  REQUIRE(vec == expected);
}

// -----------------------------------------------------------------------
// ForEach
// -----------------------------------------------------------------------

CUB_TEST("DeviceFor::ForEach env uses custom stream", "[for][env]", CUB_SMALL)
{
  auto vec = c2h::device_vector<int>{1, 2, 3, 4};
  const square_ref_op op{};

  cuda::stream stream = make_current_device_stream();
  const auto env      = cuda::std::execution::env{cuda::stream_ref{stream}};

  const auto error = cub::DeviceFor::ForEach(vec.begin(), vec.end(), op, env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  const c2h::device_vector<int> expected{1, 4, 9, 16};
  REQUIRE(vec == expected);
}

// -----------------------------------------------------------------------
// ForEachCopyN
// -----------------------------------------------------------------------

CUB_TEST("DeviceFor::ForEachCopyN env uses custom stream", "[for][env]", CUB_SMALL)
{
  auto vec   = c2h::device_vector<int>{1, 2, 3, 4};
  auto count = c2h::device_vector<int>(1);
  const odd_count_op op{thrust::raw_pointer_cast(count.data())};

  cuda::stream stream = make_current_device_stream();
  const auto env      = cuda::std::execution::env{cuda::stream_ref{stream}};

  const auto error = cub::DeviceFor::ForEachCopyN(vec.begin(), static_cast<int>(vec.size()), op, env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  const c2h::device_vector<int> expected_count{2};
  REQUIRE(count == expected_count);
}

// -----------------------------------------------------------------------
// ForEachCopy
// -----------------------------------------------------------------------

CUB_TEST("DeviceFor::ForEachCopy env uses custom stream", "[for][env]", CUB_SMALL)
{
  auto vec   = c2h::device_vector<int>{1, 2, 3, 4};
  auto count = c2h::device_vector<int>(1);
  const odd_count_op op{thrust::raw_pointer_cast(count.data())};

  cuda::stream stream = make_current_device_stream();
  const auto env      = cuda::std::execution::env{cuda::stream_ref{stream}};

  const auto error = cub::DeviceFor::ForEachCopy(vec.begin(), vec.end(), op, env);
  REQUIRE(error == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream.get()) == cudaSuccess);

  const c2h::device_vector<int> expected_count{2};
  REQUIRE(count == expected_count);
}

// -----------------------------------------------------------------------
// Environment kinds accepted by the single-phase and two-phase APIs
// -----------------------------------------------------------------------

// Regression guard for stream wrappers (PR #7798): non-copyable types implicitly convertible to
// cudaStream_t must bind to the env APIs by const& without being copied. The conversion operator
// must be const-qualified, since the env is queried through a const reference.
struct non_copyable_stream_wrapper
{
  cudaStream_t stream;

  explicit non_copyable_stream_wrapper(cudaStream_t stream_in)
      : stream(stream_in)
  {}
  non_copyable_stream_wrapper(const non_copyable_stream_wrapper&)            = delete;
  non_copyable_stream_wrapper& operator=(const non_copyable_stream_wrapper&) = delete;

  operator cudaStream_t() const
  {
    return stream;
  }
};

struct with_stream_method
{
  cudaStream_t str;

  cudaStream_t stream() const
  {
    return str;
  }
};

struct with_get_stream_method
{
  cudaStream_t stream;

  cudaStream_t get_stream() const
  {
    return stream;
  }
};

// A non-const conversion operator is unreachable through the const& the env APIs take, so the type
// provides no stream to the env query and work runs on the default stream. The static_asserts pin
// the trait pair behind that fallback.
struct mutable_stream_wrapper
{
  cudaStream_t stream;

  operator cudaStream_t()
  {
    return stream;
  }
};
static_assert(cuda::std::is_convertible_v<mutable_stream_wrapper, cudaStream_t>);
static_assert(!cuda::std::__is_callable_v<cuda::get_stream_t, const mutable_stream_wrapper&>);
static_assert(cuda::std::__is_callable_v<cuda::get_stream_t, const non_copyable_stream_wrapper&>);

struct bulk_single_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto vec = c2h::device_vector<int>{1, 2, 3, 4};
    const square_idx_op op{thrust::raw_pointer_cast(vec.data())};

    REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(4, op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));

    const c2h::device_vector<int> expected{1, 4, 9, 16};
    REQUIRE(vec == expected);
  }
};

struct bulk_two_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto vec = c2h::device_vector<int>{1, 2, 3, 4};
    const square_idx_op op{thrust::raw_pointer_cast(vec.data())};

    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(nullptr, temp_storage_bytes, 4, op, env));
    REQUIRE(temp_storage_bytes > 0);

    c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
    REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, 4, op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));

    const c2h::device_vector<int> expected{1, 4, 9, 16};
    REQUIRE(vec == expected);
  }
};

struct for_each_n_single_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto vec = c2h::device_vector<int>{1, 2, 3, 4};
    const square_ref_op op{};

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachN(vec.begin(), static_cast<int>(vec.size()), op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));

    const c2h::device_vector<int> expected{1, 4, 9, 16};
    REQUIRE(vec == expected);
  }
};

struct for_each_n_two_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto vec = c2h::device_vector<int>{1, 2, 3, 4};
    const square_ref_op op{};
    const auto num_items = static_cast<int>(vec.size());

    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachN(nullptr, temp_storage_bytes, vec.begin(), num_items, op, env));
    REQUIRE(temp_storage_bytes > 0);

    c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
    REQUIRE(
      cudaSuccess == cub::DeviceFor::ForEachN(d_temp_storage, temp_storage_bytes, vec.begin(), num_items, op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));

    const c2h::device_vector<int> expected{1, 4, 9, 16};
    REQUIRE(vec == expected);
  }
};

struct for_each_single_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto vec = c2h::device_vector<int>{1, 2, 3, 4};
    const square_ref_op op{};

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEach(vec.begin(), vec.end(), op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));

    const c2h::device_vector<int> expected{1, 4, 9, 16};
    REQUIRE(vec == expected);
  }
};

struct for_each_two_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto vec = c2h::device_vector<int>{1, 2, 3, 4};
    const square_ref_op op{};

    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEach(nullptr, temp_storage_bytes, vec.begin(), vec.end(), op, env));
    REQUIRE(temp_storage_bytes > 0);

    c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
    REQUIRE(
      cudaSuccess == cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, vec.begin(), vec.end(), op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));

    const c2h::device_vector<int> expected{1, 4, 9, 16};
    REQUIRE(vec == expected);
  }
};

struct for_each_copy_n_single_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto vec   = c2h::device_vector<int>{1, 2, 3, 4};
    auto count = c2h::device_vector<int>(1);
    const odd_count_op op{thrust::raw_pointer_cast(count.data())};

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopyN(vec.begin(), static_cast<int>(vec.size()), op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));
    const c2h::device_vector<int> expected{2};
    REQUIRE(count == expected);
  }
};

struct for_each_copy_n_two_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto vec             = c2h::device_vector<int>{1, 2, 3, 4};
    auto count           = c2h::device_vector<int>(1);
    const auto num_items = static_cast<int>(vec.size());
    const odd_count_op op{thrust::raw_pointer_cast(count.data())};

    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopyN(nullptr, temp_storage_bytes, vec.begin(), num_items, op, env));
    REQUIRE(temp_storage_bytes > 0);

    c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    REQUIRE(cudaSuccess
            == cub::DeviceFor::ForEachCopyN(
              thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, vec.begin(), num_items, op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));
    const c2h::device_vector<int> expected{2};
    REQUIRE(count == expected);
  }
};

struct for_each_copy_single_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto vec   = c2h::device_vector<int>{1, 2, 3, 4};
    auto count = c2h::device_vector<int>(1);
    const odd_count_op op{thrust::raw_pointer_cast(count.data())};

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopy(vec.begin(), vec.end(), op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));
    const c2h::device_vector<int> expected{2};
    REQUIRE(count == expected);
  }
};

struct for_each_copy_two_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto vec   = c2h::device_vector<int>{1, 2, 3, 4};
    auto count = c2h::device_vector<int>(1);
    const odd_count_op op{thrust::raw_pointer_cast(count.data())};

    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopy(nullptr, temp_storage_bytes, vec.begin(), vec.end(), op, env));
    REQUIRE(temp_storage_bytes > 0);

    c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    REQUIRE(cudaSuccess
            == cub::DeviceFor::ForEachCopy(
              thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, vec.begin(), vec.end(), op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));
    const c2h::device_vector<int> expected{2};
    REQUIRE(count == expected);
  }
};

// Rank 2 so that layout_left and layout_right disagree, and keyed on the coordinates so that the
// mapping is actually observed rather than just the linear index.
struct store_coords_op
{
  int* output;

  __device__ void operator()(int index, int x, int y) const
  {
    output[index] = x * 10 + y;
  }
};

using coords_extents = cuda::std::extents<int, 2, 3>;

// layout_right varies the rightmost coordinate fastest, layout_left the leftmost
inline c2h::host_vector<int> layout_right_coords()
{
  return c2h::host_vector<int>{0, 1, 2, 10, 11, 12};
}

inline c2h::host_vector<int> layout_left_coords()
{
  return c2h::host_vector<int>{0, 10, 1, 11, 2, 12};
}

struct for_each_in_extents_single_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto output = c2h::device_vector<int>(coords_extents{}.extent(0) * coords_extents{}.extent(1));
    const store_coords_op op{thrust::raw_pointer_cast(output.data())};

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInExtents(coords_extents{}, op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));
    REQUIRE(output == layout_right_coords());
  }
};

struct for_each_in_extents_two_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto output = c2h::device_vector<int>(coords_extents{}.extent(0) * coords_extents{}.extent(1));
    const store_coords_op op{thrust::raw_pointer_cast(output.data())};

    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInExtents(nullptr, temp_storage_bytes, coords_extents{}, op, env));
    REQUIRE(temp_storage_bytes > 0);

    c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    REQUIRE(cudaSuccess
            == cub::DeviceFor::ForEachInExtents(
              thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, coords_extents{}, op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));
    REQUIRE(output == layout_right_coords());
  }
};

struct for_each_in_layout_single_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto output        = c2h::device_vector<int>(coords_extents{}.extent(0) * coords_extents{}.extent(1));
    const auto mapping = cuda::std::layout_left::mapping<coords_extents>{};
    const store_coords_op op{thrust::raw_pointer_cast(output.data())};

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInLayout(mapping, op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));
    REQUIRE(output == layout_left_coords());
  }
};

struct for_each_in_layout_two_phase_test
{
  template <class EnvT>
  void operator()(const EnvT& env, cuda::stream_ref sync_stream) const
  {
    auto output        = c2h::device_vector<int>(coords_extents{}.extent(0) * coords_extents{}.extent(1));
    const auto mapping = cuda::std::layout_left::mapping<coords_extents>{};
    const store_coords_op op{thrust::raw_pointer_cast(output.data())};

    size_t temp_storage_bytes = 0;
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInLayout(nullptr, temp_storage_bytes, mapping, op, env));
    REQUIRE(temp_storage_bytes > 0);

    c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    REQUIRE(cudaSuccess
            == cub::DeviceFor::ForEachInLayout(
              thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes, mapping, op, env));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(sync_stream.get()));
    REQUIRE(output == layout_left_coords());
  }
};

template <class TestFn>
void test_env_kinds(TestFn test_fn)
{
  cuda::stream stream       = make_current_device_stream();
  const auto default_stream = cuda::stream_ref{cudaStream_t{}};

  SECTION("default environment")
  {
    test_fn(cuda::std::execution::env<>{}, default_stream);
  }

  SECTION("raw cudaStream_t")
  {
    test_fn(stream.get(), cuda::stream_ref{stream});
  }

  SECTION("cuda::stream_ref")
  {
    test_fn(cuda::stream_ref{stream}, cuda::stream_ref{stream});
  }

  SECTION("cuda::stream")
  {
    test_fn(stream, cuda::stream_ref{stream});
  }

  SECTION("environment with stream")
  {
    test_fn(cuda::std::execution::env{cuda::stream_ref{stream}}, cuda::stream_ref{stream});
  }

  SECTION("non-copyable wrapper convertible to cudaStream_t")
  {
    non_copyable_stream_wrapper wrapper{stream.get()};
    test_fn(wrapper, cuda::stream_ref{stream});
  }

  SECTION("wrapper with a non-const conversion to cudaStream_t")
  {
    // the stream is unreachable through the const& env query, so work runs on the default stream
    test_fn(mutable_stream_wrapper{stream.get()}, default_stream);
  }

  SECTION("type with a stream() member")
  {
    test_fn(with_stream_method{stream.get()}, cuda::stream_ref{stream});
  }

  SECTION("type with a get_stream() member")
  {
    test_fn(with_get_stream_method{stream.get()}, cuda::stream_ref{stream});
  }

  SECTION("environment with a get_stream prop")
  {
    // MSVC has trouble nesting two aggregate initializations with CTAD
    auto stream_prop = cuda::std::execution::prop{cuda::get_stream, cuda::stream_ref{stream}};
    test_fn(cuda::std::execution::env{cuda::std::move(stream_prop)}, cuda::stream_ref{stream});
  }

  SECTION("cuda::execution::gpu")
  {
    test_fn(cuda::execution::gpu, default_stream);
  }

  SECTION("cuda::execution::gpu with stream")
  {
    test_fn(cuda::execution::gpu.with(cuda::get_stream, cuda::stream_ref{stream}), cuda::stream_ref{stream});
  }
}

CUB_TEST("DeviceFor::Bulk single-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(bulk_single_phase_test{});
}

CUB_TEST("DeviceFor::Bulk two-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(bulk_two_phase_test{});
}

CUB_TEST("DeviceFor::ForEachN single-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_n_single_phase_test{});
}

CUB_TEST("DeviceFor::ForEachN two-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_n_two_phase_test{});
}

CUB_TEST("DeviceFor::ForEach single-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_single_phase_test{});
}

CUB_TEST("DeviceFor::ForEach two-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_two_phase_test{});
}

CUB_TEST("DeviceFor::ForEachCopyN single-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_copy_n_single_phase_test{});
}

CUB_TEST("DeviceFor::ForEachCopyN two-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_copy_n_two_phase_test{});
}

CUB_TEST("DeviceFor::ForEachCopy single-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_copy_single_phase_test{});
}

CUB_TEST("DeviceFor::ForEachCopy two-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_copy_two_phase_test{});
}

CUB_TEST("DeviceFor::ForEachInExtents single-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_in_extents_single_phase_test{});
}

CUB_TEST("DeviceFor::ForEachInExtents two-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_in_extents_two_phase_test{});
}

CUB_TEST("DeviceFor::ForEachInLayout single-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_in_layout_single_phase_test{});
}

CUB_TEST("DeviceFor::ForEachInLayout two-phase API accepts environments", "[for][env]", CUB_SMALL)
{
  test_env_kinds(for_each_in_layout_two_phase_test{});
}

// -----------------------------------------------------------------------
// Stream routing
// -----------------------------------------------------------------------

// The value checks above cannot see which stream the work ran on: they compare on the default
// stream, which is ordered after a misplaced kernel either way. Capturing the environment's stream
// makes the routing observable -- a launch into any other stream fails the capture.
struct capture_args
{
  int* data;
  int* count;
  int num_items;
  void* d_temp_storage;
  size_t temp_storage_bytes;
};

// A failed REQUIRE unwinds past EndCapture, and an abandoned capture poisons every later
// default-stream CUDA call in the process, so the destructor closes the capture on unwind.
struct stream_capture_guard
{
  cudaStream_t stream;
  bool active = true;

  explicit stream_capture_guard(cudaStream_t stream_in)
      : stream(stream_in)
  {
    REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  }

  stream_capture_guard(const stream_capture_guard&)            = delete;
  stream_capture_guard& operator=(const stream_capture_guard&) = delete;

  ~stream_capture_guard()
  {
    if (active)
    {
      cudaGraph_t graph{};
      cudaStreamEndCapture(stream, &graph);
      cudaGraphDestroy(graph);
      cudaGetLastError(); // reset the sticky error state left by the aborted capture
    }
  }

  size_t finish_and_count_nodes()
  {
    active = false;
    cudaGraph_t graph{};
    REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));
    size_t num_nodes = 0;
    REQUIRE(cudaSuccess == cudaGraphGetNodes(graph, nullptr, &num_nodes));
    REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
    return num_nodes;
  }
};

struct bulk_stream_routing
{
  template <class EnvT>
  void operator()(const capture_args& args, const EnvT& env) const
  {
    size_t bytes = args.temp_storage_bytes;
    const square_idx_op op{args.data};

    REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(args.num_items, op, env));
    REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(args.d_temp_storage, bytes, args.num_items, op, env));
  }
};

struct for_each_n_stream_routing
{
  template <class EnvT>
  void operator()(const capture_args& args, const EnvT& env) const
  {
    size_t bytes = args.temp_storage_bytes;
    const square_ref_op op{};

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachN(args.data, args.num_items, op, env));
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachN(args.d_temp_storage, bytes, args.data, args.num_items, op, env));
  }
};

struct for_each_stream_routing
{
  template <class EnvT>
  void operator()(const capture_args& args, const EnvT& env) const
  {
    size_t bytes = args.temp_storage_bytes;
    const square_ref_op op{};
    int* last = args.data + args.num_items;

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEach(args.data, last, op, env));
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEach(args.d_temp_storage, bytes, args.data, last, op, env));
  }
};

struct for_each_copy_n_stream_routing
{
  template <class EnvT>
  void operator()(const capture_args& args, const EnvT& env) const
  {
    size_t bytes = args.temp_storage_bytes;
    const odd_count_op op{args.count};

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopyN(args.data, args.num_items, op, env));
    REQUIRE(
      cudaSuccess == cub::DeviceFor::ForEachCopyN(args.d_temp_storage, bytes, args.data, args.num_items, op, env));
  }
};

struct for_each_copy_stream_routing
{
  template <class EnvT>
  void operator()(const capture_args& args, const EnvT& env) const
  {
    size_t bytes = args.temp_storage_bytes;
    const odd_count_op op{args.count};
    int* last = args.data + args.num_items;

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopy(args.data, last, op, env));
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopy(args.d_temp_storage, bytes, args.data, last, op, env));
  }
};

struct for_each_in_extents_stream_routing
{
  template <class EnvT>
  void operator()(const capture_args& args, const EnvT& env) const
  {
    size_t bytes = args.temp_storage_bytes;
    const store_coords_op op{args.data};

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInExtents(coords_extents{}, op, env));
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInExtents(args.d_temp_storage, bytes, coords_extents{}, op, env));
  }
};

struct for_each_in_layout_stream_routing
{
  template <class EnvT>
  void operator()(const capture_args& args, const EnvT& env) const
  {
    size_t bytes = args.temp_storage_bytes;
    const store_coords_op op{args.data};
    const auto mapping = cuda::std::layout_left::mapping<coords_extents>{};

    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInLayout(mapping, op, env));
    REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInLayout(args.d_temp_storage, bytes, mapping, op, env));
  }
};

template <class LaunchFn>
void test_env_stream_routing(LaunchFn launch)
{
  cuda::stream stream = make_current_device_stream();

  // allocation is not capturable, so everything the launches touch is set up before capture begins
  c2h::device_vector<int> vec(coords_extents{}.extent(0) * coords_extents{}.extent(1), 1);
  c2h::device_vector<int> count(1);
  c2h::device_vector<std::uint8_t> temp_storage(1);
  const capture_args args{
    thrust::raw_pointer_cast(vec.data()),
    thrust::raw_pointer_cast(count.data()),
    static_cast<int>(vec.size()),
    thrust::raw_pointer_cast(temp_storage.data()),
    temp_storage.size()};

  stream_capture_guard capture{stream.get()};

  SECTION("raw cudaStream_t")
  {
    launch(args, stream.get());
  }

  SECTION("cuda::stream_ref")
  {
    launch(args, cuda::stream_ref{stream});
  }

  SECTION("cuda::stream")
  {
    launch(args, stream);
  }

  SECTION("environment with stream")
  {
    launch(args, cuda::std::execution::env{cuda::stream_ref{stream}});
  }

  SECTION("non-copyable wrapper convertible to cudaStream_t")
  {
    non_copyable_stream_wrapper wrapper{stream.get()};
    launch(args, wrapper);
  }

  SECTION("type with a stream() member")
  {
    launch(args, with_stream_method{stream.get()});
  }

  SECTION("type with a get_stream() member")
  {
    launch(args, with_get_stream_method{stream.get()});
  }

  SECTION("environment with a get_stream prop")
  {
    // MSVC has trouble nesting two aggregate initializations with CTAD
    auto stream_prop = cuda::std::execution::prop{cuda::get_stream, cuda::stream_ref{stream}};
    launch(args, cuda::std::execution::env{cuda::std::move(stream_prop)});
  }

  SECTION("cuda::execution::gpu with stream")
  {
    launch(args, cuda::execution::gpu.with(cuda::get_stream, cuda::stream_ref{stream}));
  }

  CHECK(capture.finish_and_count_nodes() == 2); // one for the single-phase call, one for the two-phase call
}

CUB_TEST("DeviceFor::Bulk launches on the environment's stream", "[for][env]", CUB_SMALL)
{
  test_env_stream_routing(bulk_stream_routing{});
}

CUB_TEST("DeviceFor::ForEachN launches on the environment's stream", "[for][env]", CUB_SMALL)
{
  test_env_stream_routing(for_each_n_stream_routing{});
}

CUB_TEST("DeviceFor::ForEach launches on the environment's stream", "[for][env]", CUB_SMALL)
{
  test_env_stream_routing(for_each_stream_routing{});
}

CUB_TEST("DeviceFor::ForEachCopyN launches on the environment's stream", "[for][env]", CUB_SMALL)
{
  test_env_stream_routing(for_each_copy_n_stream_routing{});
}

CUB_TEST("DeviceFor::ForEachCopy launches on the environment's stream", "[for][env]", CUB_SMALL)
{
  test_env_stream_routing(for_each_copy_stream_routing{});
}

CUB_TEST("DeviceFor::ForEachInExtents launches on the environment's stream", "[for][env]", CUB_SMALL)
{
  test_env_stream_routing(for_each_in_extents_stream_routing{});
}

CUB_TEST("DeviceFor::ForEachInLayout launches on the environment's stream", "[for][env]", CUB_SMALL)
{
  test_env_stream_routing(for_each_in_layout_stream_routing{});
}

template <int ThreadsPerBlock>
struct for_each_tuning
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability) const -> cub::ForPolicy
  {
    return {ThreadsPerBlock, 2};
  }
};

struct block_size_extracting_op
{
  unsigned int* block_size;

  __device__ void operator()(int) const
  {
    if (threadIdx.x == 0)
    {
      atomicMax(block_size, blockDim.x);
    }
  }
};

using block_sizes =
  c2h::type_list<cuda::std::integral_constant<unsigned int, 64>, cuda::std::integral_constant<unsigned int, 128>>;

CUB_TEST("DeviceFor::Bulk can be tuned", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(4, op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceFor::ForEachN can be tuned", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachN(d_data.begin(), static_cast<int>(d_data.size()), op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceFor::ForEach can be tuned", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEach(d_data.begin(), d_data.end(), op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceFor::ForEachCopyN can be tuned", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopyN(d_data.begin(), static_cast<int>(d_data.size()), op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceFor::ForEachCopy can be tuned", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopy(d_data.begin(), d_data.end(), op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceFor::Bulk two-phase API propagates tuning", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(nullptr, temp_storage_bytes, 4, op, env));

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  REQUIRE(cudaSuccess == cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, 4, op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceFor::ForEachN two-phase API propagates tuning", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env       = cuda::execution::tune(for_each_tuning<target_block_size>{});
  const auto num_items = static_cast<int>(d_data.size());

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachN(nullptr, temp_storage_bytes, d_data.begin(), num_items, op, env));

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  REQUIRE(
    cudaSuccess == cub::DeviceFor::ForEachN(d_temp_storage, temp_storage_bytes, d_data.begin(), num_items, op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceFor::ForEach two-phase API propagates tuning", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == cub::DeviceFor::ForEach(nullptr, temp_storage_bytes, d_data.begin(), d_data.end(), op, env));

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  REQUIRE(
    cudaSuccess == cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, d_data.begin(), d_data.end(), op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceFor::ForEachCopyN two-phase API propagates tuning", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env       = cuda::execution::tune(for_each_tuning<target_block_size>{});
  const auto num_items = static_cast<int>(d_data.size());

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachCopyN(nullptr, temp_storage_bytes, d_data.begin(), num_items, op, env));

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  REQUIRE(cudaSuccess
          == cub::DeviceFor::ForEachCopyN(d_temp_storage, temp_storage_bytes, d_data.begin(), num_items, op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceFor::ForEachCopy two-phase API propagates tuning", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<int> d_data{1, 2, 3, 4};
  c2h::device_vector<unsigned int> d_block_size(1);
  const block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  size_t temp_storage_bytes = 0;
  REQUIRE(
    cudaSuccess == cub::DeviceFor::ForEachCopy(nullptr, temp_storage_bytes, d_data.begin(), d_data.end(), op, env));

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  REQUIRE(cudaSuccess
          == cub::DeviceFor::ForEachCopy(d_temp_storage, temp_storage_bytes, d_data.begin(), d_data.end(), op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

struct extents_block_size_extracting_op
{
  unsigned int* block_size;

  __device__ void operator()(int, int) const
  {
    if (threadIdx.x == 0)
    {
      atomicMax(block_size, blockDim.x);
    }
  }
};

CUB_TEST("DeviceFor::ForEachInExtents two-phase API propagates tuning", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  using extents_type = cuda::std::extents<int, 4>;
  const extents_block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInExtents(nullptr, temp_storage_bytes, extents_type{}, op, env));

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInExtents(d_temp_storage, temp_storage_bytes, extents_type{}, op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

CUB_TEST("DeviceFor::ForEachInLayout two-phase API propagates tuning", "[for][device]", CUB_SMALL, block_sizes)
{
  constexpr unsigned int target_block_size = c2h::get<0, TestType>::value;
  c2h::device_vector<unsigned int> d_block_size(1);
  using extents_type = cuda::std::extents<int, 4>;
  const auto mapping = cuda::std::layout_left::mapping<extents_type>{};
  const extents_block_size_extracting_op op{thrust::raw_pointer_cast(d_block_size.data())};
  const auto env = cuda::execution::tune(for_each_tuning<target_block_size>{});

  size_t temp_storage_bytes = 0;
  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInLayout(nullptr, temp_storage_bytes, mapping, op, env));

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  void* const d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());
  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInLayout(d_temp_storage, temp_storage_bytes, mapping, op, env));
  REQUIRE(d_block_size[0] == target_block_size);
}

#if _CCCL_COMPILER(GCC, >=, 8) // gcc 7 cannot preserve constexpr-ness from p1 to p2
CUB_TEST("Test ForPolicy properties", "[for][device]", CUB_SMALL)
{
  STATIC_REQUIRE(::cuda::std::semiregular<cub::ForPolicy>);
  STATIC_REQUIRE(::cuda::std::is_aggregate_v<cub::ForPolicy>);

  // aggregate init
  constexpr auto p1 = cub::ForPolicy{128, 4};

#  if _CCCL_STD_VER >= 2020
  // designated init
  constexpr auto p2 = cub::ForPolicy{.threads_per_block = 128, .items_per_thread = 4};
#  else // _CCCL_STD_VER >= 2020
  constexpr auto p2 = p1;
#  endif // _CCCL_STD_VER >= 2020

  // comparison
  STATIC_REQUIRE(p1 == p2);
  STATIC_REQUIRE_FALSE(p1 != p2);

  auto to_string = [](const auto& p) {
    std::ostringstream os;
    os << p;
    return os.str();
  };
  REQUIRE(to_string(p1) == "ForPolicy { .threads_per_block = 128, .items_per_thread = 4 }");
}
#endif // _CCCL_COMPILER(GCC, >=, 8)
