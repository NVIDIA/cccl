//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Tests for `place_group`: construction (places / grid / factories),
 *        per-place stream pools (lazy creation, colors, isolation between
 *        groups), per-place memory resources, and pool BORROWING from an STF
 *        `async_resources_handle` (one pool owner when both layers coexist).
 *
 * Runs on any machine with at least one GPU; locality-domain factories
 * degrade to whole-device places where domains are not supported.
 */

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__stf/internal/async_resources_handle.cuh>

using namespace cuda::experimental::places;

namespace
{
__global__ void touch_kernel(int* ptr, int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n)
  {
    ptr[tid] = tid;
  }
}

void test_construction_and_factories()
{
  // From an explicit vector of places
  place_group g1(places_from_devices({0}));
  EXPECT(g1.size() == 1UL);
  EXPECT(g1.owns_resources());

  // From a grid (flattened) and from a scalar place
  auto grid = make_grid(::std::vector<exec_place>{exec_place::device(0), exec_place::device(0)});
  place_group g2(grid);
  EXPECT(g2.size() == grid.size());

  place_group g3(exec_place::device(0));
  EXPECT(g3.size() == 1UL);

  // by_devices covers every visible device
  const auto ndevs = all_device_ids().size();
  auto g4          = place_group::by_devices();
  EXPECT(g4.size() == ndevs);

  auto g5 = place_group::by_devices({0});
  EXPECT(g5.size() == 1UL);

  // by_locality_domains covers every domain of every device (>= one place
  // per device even without domain support)
  size_t total_domains = 0;
  for (int d : all_device_ids())
  {
    total_domains += locality_domain_count(d);
  }
  auto g6 = place_group::by_locality_domains();
  EXPECT(g6.size() == total_domains);
  EXPECT(g6.size() >= ndevs);
}

void test_streams()
{
  auto group = place_group::by_locality_domains();

  // A stream can be picked and used on every place, for every color
  EXPECT(group.num_stream_colors() >= 1UL);
  for (size_t i = 0; i < group.size(); i++)
  {
    for (size_t color = 0; color < group.num_stream_colors(); color++)
    {
      cudaStream_t s = group.get_stream(i, color);
      EXPECT(s != nullptr);
      // Stable: the same (place, color) always yields the same stream
      EXPECT(s == group.get_stream(i, color));

      exec_place_scope scope(group.place(i));
      cuda_safe_call(cudaStreamSynchronize(s));
    }
    // Different colors are different streams
    EXPECT(group.get_stream(i, 0) != group.get_stream(i, 1));
  }

  // Streams actually execute work on their place
  for (size_t i = 0; i < group.size(); i++)
  {
    exec_place_scope scope(group.place(i));
    const int n    = 1024;
    auto dplace    = group.place(i).affine_data_place();
    cudaStream_t s = group.get_stream(i);
    int* ptr       = static_cast<int*>(dplace.allocate(n * sizeof(int), s));
    touch_kernel<<<(n + 255) / 256, 256, 0, s>>>(ptr, n);
    cuda_safe_call(cudaGetLastError());
    cuda_safe_call(cudaStreamSynchronize(s));
    dplace.deallocate(ptr, n * sizeof(int), s);
    cuda_safe_call(cudaStreamSynchronize(s));
  }

  group.sync();

  // Two groups over the same places are distinct resource scopes: they own
  // distinct pools, hence distinct streams
  place_group a(places_from_devices({0}));
  place_group b(places_from_devices({0}));
  EXPECT(a.owns_resources());
  EXPECT(b.owns_resources());
  EXPECT(a.get_stream(0, 0) != b.get_stream(0, 0));
}

void test_memory_resources()
{
  auto group = place_group::by_devices({0});

  auto mr        = group.memory_resource(0);
  cudaStream_t s = group.get_stream(0);

  void* p = mr.allocate(::cuda::stream_ref{s}, 1024);
  EXPECT(p != nullptr);
  mr.deallocate(::cuda::stream_ref{s}, p, 1024);
  cuda_safe_call(cudaStreamSynchronize(s));

  void* q = mr.allocate_sync(2048);
  EXPECT(q != nullptr);
  mr.deallocate_sync(q, 2048);

  // Equality follows the place
  EXPECT(mr == group.memory_resource(0));
  EXPECT(mr != group.memory_resource(data_place::host()));

  // Host resource yields pinned memory usable from device code paths
  auto host_mr = group.memory_resource(data_place::host());
  void* h      = host_mr.allocate_sync(64);
  EXPECT(h != nullptr);
  host_mr.deallocate_sync(h, 64);
}

void test_borrowing_from_stf()
{
  using ::cuda::experimental::stf::async_resources_handle;

  async_resources_handle handle;
  auto places = places_from_devices({0});

  // Borrowing group: draws its pools from the handle's registry
  place_group borrowed(places, handle);
  EXPECT(!borrowed.owns_resources());
  EXPECT(&borrowed.resources() == &handle.get_place_resources());

  // One pool owner: the borrowed group's pool IS the handle's pool for the
  // same place (compare pool identity through stream_pool::operator==)
  auto& from_group  = borrowed.place(0).get_stream_pool(true, borrowed.resources());
  auto& from_handle = borrowed.place(0).get_stream_pool(true, handle.get_place_resources());
  EXPECT(from_group == from_handle);

  // The streams work
  cudaStream_t s = borrowed.get_stream(0);
  EXPECT(s != nullptr);
  exec_place_scope scope(borrowed.place(0));
  cuda_safe_call(cudaStreamSynchronize(s));

  // An owning group over the same places uses a DIFFERENT pool
  place_group owning(places);
  auto& from_owning = owning.place(0).get_stream_pool(true, owning.resources());
  EXPECT(!(from_owning == from_handle));

  // The low-level borrowing seam (raw registry reference) also works
  place_group raw_borrow(places, handle.get_place_resources());
  EXPECT(!raw_borrow.owns_resources());
  EXPECT(&raw_borrow.resources() == &handle.get_place_resources());
}

void test_move_semantics()
{
  place_group g(places_from_devices({0}));
  cudaStream_t s = g.get_stream(0);

  place_group moved(::std::move(g));
  EXPECT(moved.size() == 1UL);
  EXPECT(moved.owns_resources());
  // The cached stream survives the move
  EXPECT(moved.get_stream(0) == s);
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  test_construction_and_factories();
  test_streams();
  test_memory_resources();
  test_borrowing_from_stf();
  test_move_semantics();

  return 0;
}
