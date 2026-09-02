//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief The generic algorithms run over structures written independently of
 *        the container tier, with zero adapters:
 *
 *        - a fully foreign model: hand-rolled descriptors over raw
 *          cudaMallocAsync buffers, caller-created streams, a hand-rolled
 *          cuda::mr resource, and its own ADL default_envs — no container,
 *          no place_group, no places types anywhere in the model;
 *        - the adoption model: sharded_array::adopt over caller-owned memory
 *          and caller streams (foreign resources through the container type).
 *
 *        Both run the same generic transform / reduce / reduce_into and
 *        agree with references.
 */

#include <cuda/experimental/sharded.cuh>

#include <cstddef>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::data_place;
using cuda::experimental::places::exec_place;
using cuda::experimental::places::place_group;

namespace foreign
{

// A hand-rolled stream-ordered memory resource (cuda::mr::resource shape).
struct raw_async_mr
{
  void* allocate(::cuda::stream_ref s, ::std::size_t bytes, ::std::size_t = alignof(::std::max_align_t)) const
  {
    void* p = nullptr;
    cuda_safe_call(cudaMallocAsync(&p, bytes, s.get()));
    return p;
  }
  void deallocate(::cuda::stream_ref s, void* p, ::std::size_t, ::std::size_t = alignof(::std::max_align_t)) const
  {
    cuda_safe_call(cudaFreeAsync(p, s.get()));
  }
  void* allocate_sync(::std::size_t bytes, ::std::size_t = alignof(::std::max_align_t)) const
  {
    void* p = nullptr;
    cuda_safe_call(cudaMalloc(&p, bytes));
    return p;
  }
  void deallocate_sync(void* p, ::std::size_t, ::std::size_t = alignof(::std::max_align_t)) const
  {
    cuda_safe_call(cudaFree(p));
  }
  bool operator==(const raw_async_mr&) const
  {
    return true;
  }
  bool operator!=(const raw_async_mr&) const
  {
    return false;
  }
  friend constexpr void get_property(const raw_async_mr&, ::cuda::mr::device_accessible) noexcept {}
};

// A hand-rolled per-shard environment: member shapes of the two CPOs.
struct raw_env
{
  cudaStream_t s{};
  ::cuda::stream_ref get_stream() const noexcept
  {
    return ::cuda::stream_ref{s};
  }
  raw_async_mr get_memory_resource() const noexcept
  {
    return raw_async_mr{};
  }
};

// The fully foreign sharded structure: descriptors + its own environments.
struct raw_sharded
{
  ::std::vector<basic_shard_view<double, int>> shards_;
  ::std::vector<raw_env> envs_;

  ::std::size_t num_shards() const
  {
    return shards_.size();
  }
  const basic_shard_view<double, int>& shard(::std::size_t i) const
  {
    return shards_[i];
  }
};

::std::vector<raw_env> default_envs(const raw_sharded& v)
{
  return v.envs_;
}

} // namespace foreign

static_assert(sharded_view<foreign::raw_sharded>, "foreign model is a sharded_view");
static_assert(self_bound<foreign::raw_sharded>, "foreign model is self-bound via its own default_envs");
static_assert(sharded_alloc_env<foreign::raw_env>, "foreign env models sharded_alloc_env via member CPO shapes");

namespace
{

struct times3
{
  __host__ __device__ double operator()(double v) const
  {
    return v * 3.0;
  }
};

void test_fully_foreign_model()
{
  // Exercise every member of the hand-rolled resource at runtime (the
  // synchronous half and the comparisons are otherwise referenced only in
  // unevaluated concept checks; strict builds promote "never referenced").
  {
    foreign::raw_async_mr mr;
    void* p = mr.allocate_sync(64);
    EXPECT(p != nullptr);
    mr.deallocate_sync(p, 64);
    EXPECT(mr == foreign::raw_async_mr{});
    EXPECT(!(mr != foreign::raw_async_mr{}));
    get_property(mr, ::cuda::mr::device_accessible{}); // property tag, empty by design
  }

  // Two shards over raw buffers, two caller streams — no container anywhere.
  const ::std::size_t n0 = 300000, n1 = 200001, n = n0 + n1;
  cudaStream_t s0, s1;
  cuda_safe_call(cudaStreamCreate(&s0));
  cuda_safe_call(cudaStreamCreate(&s1));

  double *d0 = nullptr, *d1 = nullptr;
  cuda_safe_call(cudaMallocAsync(&d0, n0 * sizeof(double), s0));
  cuda_safe_call(cudaMallocAsync(&d1, n1 * sizeof(double), s1));

  foreign::raw_sharded v;
  v.shards_.push_back({d0, n0, 0, /*place=*/0});
  v.shards_.push_back({d1, n1, n0, /*place=*/1});
  v.envs_.push_back({s0});
  v.envs_.push_back({s1});
  EXPECT(validate(v));

  // Initialize to 1.0 through the generic tier itself (fill via transform
  // over memset-zeroed buffers)
  cuda_safe_call(cudaMemsetAsync(d0, 0, n0 * sizeof(double), s0));
  cuda_safe_call(cudaMemsetAsync(d1, 0, n1 * sizeof(double), s1));
  transform(v, [] __device__(double) {
    return 1.0;
  }); // synchronous form

  // Generic transform + reduce, self-bound
  transform(v, times3{});
  EXPECT(reduce(v, ::cuda::std::plus<double>{}, 0.0) == 3.0 * n);

  // reduce_into over the foreign model
  cudaStream_t cs;
  cuda_safe_call(cudaStreamCreate(&cs));
  const auto sp = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{cs}};
  const auto ce = ::cuda::std::execution::env{sp};
  double* h_out;
  cuda_safe_call(cudaMallocHost(&h_out, sizeof(double)));
  reduce_into(v, h_out, ::cuda::std::plus<double>{}, 0.0, ce);
  cuda_safe_call(cudaStreamSynchronize(cs));
  EXPECT(*h_out == 3.0 * n);

  cuda_safe_call(cudaFreeHost(h_out));
  cuda_safe_call(cudaStreamDestroy(cs));
  cuda_safe_call(cudaFreeAsync(d0, s0));
  cuda_safe_call(cudaFreeAsync(d1, s1));
  cuda_safe_call(cudaStreamSynchronize(s0));
  cuda_safe_call(cudaStreamSynchronize(s1));
  cuda_safe_call(cudaStreamDestroy(s0));
  cuda_safe_call(cudaStreamDestroy(s1));
}

void test_adopted_model()
{
  // Caller-owned memory + caller streams through sharded_array::adopt —
  // foreign resources, container type; the generic tier must treat it
  // exactly like the fully foreign model.
  const ::std::size_t n0 = 150000, n1 = 100001, n = n0 + n1;
  cudaStream_t s0, s1;
  cuda_safe_call(cudaStreamCreate(&s0));
  cuda_safe_call(cudaStreamCreate(&s1));
  double *d0 = nullptr, *d1 = nullptr;
  cuda_safe_call(cudaMalloc(&d0, n0 * sizeof(double)));
  cuda_safe_call(cudaMalloc(&d1, n1 * sizeof(double)));

  ::std::vector<shard<double>> shards(2);
  shards[0].data          = d0;
  shards[0].size          = n0;
  shards[0].capacity      = n0;
  shards[0].global_offset = 0;
  shards[0].place         = data_place::device(0);
  shards[0].exec          = exec_place::device(0);
  shards[0].stream        = s0;
  shards[1].data          = d1;
  shards[1].size          = n1;
  shards[1].capacity      = n1;
  shards[1].global_offset = n0;
  shards[1].place         = data_place::device(0);
  shards[1].exec          = exec_place::device(0);
  shards[1].stream        = s1;

  auto adopted = sharded_array<double>::adopt(::std::move(shards));
  EXPECT(validate(adopted));

  cuda_safe_call(cudaMemset(d0, 0, n0 * sizeof(double)));
  cuda_safe_call(cudaMemset(d1, 0, n1 * sizeof(double)));
  transform(adopted, [] __device__(double) {
    return 2.0;
  });
  transform(adopted, times3{});
  EXPECT(reduce(adopted, ::cuda::std::plus<double>{}, 0.0) == 6.0 * n);

  cuda_safe_call(cudaFree(d0));
  cuda_safe_call(cudaFree(d1));
  cuda_safe_call(cudaStreamDestroy(s0));
  cuda_safe_call(cudaStreamDestroy(s1));
}

} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  test_fully_foreign_model();
  test_adopted_model();

  return 0;
}
