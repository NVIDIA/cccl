//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Inserts and lookups must stay correct when the slot storage is under-aligned for the packed
// atomic CAS, which forces the insert path onto the non-packed fallback.

#include <cuda/__memory/align_up.h>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/functional>
#include <cuda/memory_pool>
#include <cuda/std/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

namespace cudax = cuda::experimental;

template <class ValueType>
__global__ void fill_sentinel_kernel(ValueType* slots, int cap, ValueType sentinel)
{
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < cap)
  {
    slots[i] = sentinel;
  }
}

template <class RefType, class Key>
__global__ void insert_kernel(RefType ref, int num_keys)
{
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < num_keys)
  {
    [[maybe_unused]] const bool inserted =
      ref.insert(typename RefType::value_type{static_cast<Key>(i), static_cast<Key>(i)});
  }
}

template <class RefType, class Key>
__global__ void contains_kernel(RefType ref, int num_probes, int* out)
{
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < num_probes)
  {
    out[i] = ref.contains(static_cast<Key>(i)) ? 1 : 0;
  }
}

template <class Key, class Mapped>
void run_misaligned_external_storage()
{
  using probing_type        = cudax::cuco::linear_probing<1, cuda::hash<Key>>;
  constexpr int bucket_size = 1;
  using map_type            = cudax::cuco::fixed_capacity_map<
    Key,
    Mapped,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<Key>,
    probing_type,
    bucket_size>;
  using ref_type   = typename map_type::ref_type;
  using value_type = typename map_type::value_type;
  using span_type  = typename ref_type::storage_span_type;

  constexpr int num_keys = 200;
  const auto capacity =
    cudax::cuco::make_valid_capacity<probing_type, bucket_size>(static_cast<::cuda::std::size_t>(num_keys) * 2);

  const Key empty_k    = static_cast<Key>(-1);
  const Mapped empty_v = static_cast<Mapped>(-1);

  const ::cuda::std::size_t nbytes = (capacity + 2) * sizeof(value_type);
  void* raw                        = nullptr;
  REQUIRE(cudaMalloc(&raw, nbytes) == cudaSuccess);

  auto* const aligned_raw = ::cuda::align_up(static_cast<::cuda::std::byte*>(raw), sizeof(value_type));
  auto* const slots       = reinterpret_cast<value_type*>(aligned_raw + alignof(value_type));
  const auto slots_addr   = reinterpret_cast<::cuda::std::uintptr_t>(slots);
  REQUIRE(slots_addr % alignof(value_type) == 0);
  REQUIRE(slots_addr % sizeof(value_type) != 0);

  constexpr int block = 128;

  const int fill_grid = static_cast<int>((capacity + block - 1) / block);
  fill_sentinel_kernel<value_type>
    <<<fill_grid, block>>>(slots, static_cast<int>(capacity), value_type{empty_k, empty_v});
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const ref_type ref{
    cudax::cuco::empty_key<Key>{empty_k},
    cudax::cuco::empty_value<Mapped>{empty_v},
    ::cuda::std::equal_to<Key>{},
    probing_type{},
    span_type{slots, capacity}};

  insert_kernel<ref_type, Key><<<(num_keys + block - 1) / block, block>>>(ref, num_keys);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  constexpr int num_probes = 2 * num_keys;
  int* d_out               = nullptr;
  REQUIRE(cudaMalloc(&d_out, sizeof(int) * num_probes) == cudaSuccess);
  contains_kernel<ref_type, Key><<<(num_probes + block - 1) / block, block>>>(ref, num_probes, d_out);
  REQUIRE(cudaGetLastError() == cudaSuccess);

  int h_out[num_probes];
  REQUIRE(cudaMemcpy(h_out, d_out, sizeof(int) * num_probes, cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  for (int i = 0; i < num_probes; ++i)
  {
    REQUIRE(static_cast<bool>(h_out[i]) == (i < num_keys));
  }

  REQUIRE(cudaFree(d_out) == cudaSuccess);
  REQUIRE(cudaFree(raw) == cudaSuccess);
}

C2H_TEST("fixed_capacity_map insert and contains over misaligned external storage", "[container]")
{
  run_misaligned_external_storage<::cuda::std::int32_t, ::cuda::std::int32_t>();
  run_misaligned_external_storage<::cuda::std::uint16_t, ::cuda::std::uint16_t>();
}

struct publication_hash
{
  template <class Key>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::size_t operator()(Key key) const noexcept
  {
    return static_cast<::cuda::std::size_t>(key);
  }
};

template <bool DelayedPublication, class Ref>
__global__ void insert_and_find_publication_kernel(Ref ref, int* out)
{
  using key_type    = typename Ref::key_type;
  using mapped_type = typename Ref::mapped_type;
  using value_type  = typename Ref::value_type;
  const auto tile   = ::cooperative_groups::tiled_partition<Ref::cg_size>(::cooperative_groups::this_thread_block());
  auto* const slots = ref.storage_span().data();

  for (auto i = threadIdx.x; i < ref.capacity(); i += blockDim.x)
  {
    slots[i] = value_type{ref.empty_key_sentinel(), ref.empty_value_sentinel()};
  }
  __syncthreads();

  if constexpr (DelayedPublication)
  {
    // Model a successful key CAS whose dependent payload write has not happened yet.
    // Hashing key zero selects slot zero for both scalar and cooperative probing.
    if (threadIdx.x == 0)
    {
      slots[0].first = key_type{0};
    }
    __syncthreads();
    if (threadIdx.x == 32)
    {
      // A separate, resident warp publishes after a bounded delay. The writer never
      // waits for the reader to return, so the correct payload wait can finish.
      const auto start = clock64();
      while (clock64() - start < 1000000)
      {
      }
      ::cuda::atomic_ref<mapped_type, Ref::thread_scope>{slots[0].second}.store(
        mapped_type{7}, ::cuda::std::memory_order_relaxed);
    }
    if (threadIdx.x >= Ref::cg_size)
    {
      return;
    }
  }

  const int operation = static_cast<int>(threadIdx.x) / Ref::cg_size;
  const value_type value{key_type{0}, static_cast<mapped_type>(operation + 107)};
  const auto result = [&] {
    if constexpr (Ref::cg_size == 1)
    {
      return ref.insert_and_find(value);
    }
    else
    {
      return ref.insert_and_find(tile, value);
    }
  }();
  if (tile.thread_rank() == 0)
  {
    out[2 * operation]     = result.first == ref.end() ? -1 : static_cast<int>(result.first->second);
    out[2 * operation + 1] = result.second;
  }
}

template <class Key, int CgSize>
void run_insert_and_find_publication(bool misaligned, bool delayed_publication)
{
  using probing_type = cudax::cuco::linear_probing<CgSize, publication_hash>;
  using ref_type     = cudax::cuco::
    fixed_capacity_map_ref<Key, Key, ::cuda::thread_scope_device, ::cuda::std::equal_to<Key>, probing_type, 1>;
  using value_type         = typename ref_type::value_type;
  const auto capacity      = cudax::cuco::make_valid_capacity<probing_type, 1>(::cuda::std::size_t{16});
  constexpr int block      = 128;
  constexpr int operations = block / CgSize;
  CAPTURE(sizeof(Key), CgSize, misaligned, delayed_publication);

  ::cuda::stream stream{::cuda::device_ref{0}};
  const auto mr = ::cuda::device_default_memory_pool(stream.device());
  auto storage =
    ::cuda::make_buffer<::cuda::std::byte>(stream, mr, (capacity + 2) * sizeof(value_type), ::cuda::std::byte{});
  auto* const aligned_raw = ::cuda::align_up(storage.data(), sizeof(value_type));
  auto* const slots       = reinterpret_cast<value_type*>(aligned_raw + (misaligned ? alignof(value_type) : 0));
  REQUIRE(reinterpret_cast<::cuda::std::uintptr_t>(slots) % alignof(value_type) == 0);
  REQUIRE((reinterpret_cast<::cuda::std::uintptr_t>(slots) % sizeof(value_type) != 0) == misaligned);
  const ref_type ref{
    cudax::cuco::empty_key<Key>{static_cast<Key>(-1)},
    cudax::cuco::empty_value<Key>{static_cast<Key>(-1)},
    ::cuda::std::equal_to<Key>{},
    probing_type{},
    typename ref_type::storage_span_type{slots, capacity}};

  auto results = ::cuda::make_buffer<int>(stream, mr, 2 * operations, 0);
  if (delayed_publication)
  {
    insert_and_find_publication_kernel<true><<<1, block, 0, stream.get()>>>(ref, results.data());
  }
  else
  {
    insert_and_find_publication_kernel<false><<<1, block, 0, stream.get()>>>(ref, results.data());
  }
  REQUIRE(cudaGetLastError() == cudaSuccess);
  int out[2 * operations];
  const int result_count = delayed_publication ? 1 : operations;
  REQUIRE(cudaMemcpyAsync(out, results.data(), 2 * result_count * sizeof(int), cudaMemcpyDeviceToHost, stream.get())
          == cudaSuccess);
  stream.sync();

  if (delayed_publication)
  {
    REQUIRE(out[0] == 7);
    REQUIRE(out[1] == 0);
  }
  else
  {
    int winner         = -1;
    int inserted_count = 0;
    for (int i = 0; i < operations; ++i)
    {
      if (out[2 * i + 1])
      {
        winner = i;
        ++inserted_count;
      }
    }
    REQUIRE(inserted_count == 1);
    for (int i = 0; i < operations; ++i)
    {
      REQUIRE(out[2 * i] == winner + 107);
    }
  }
}

using publication_key_types = c2h::type_list<::cuda::std::int32_t, ::cuda::std::uint16_t>;
using publication_cg_sizes =
  c2h::type_list<::cuda::std::integral_constant<int, 1>, ::cuda::std::integral_constant<int, 2>>;

C2H_TEST("fixed_capacity_map insert_and_find waits for a pending payload",
         "[container]",
         publication_key_types,
         publication_cg_sizes)
{
  run_insert_and_find_publication<c2h::get<0, TestType>, c2h::get<1, TestType>::value>(true, true);
}

C2H_TEST("fixed_capacity_map insert_and_find concurrent duplicates over external storage",
         "[container]",
         publication_key_types,
         publication_cg_sizes)
{
  run_insert_and_find_publication<c2h::get<0, TestType>, c2h::get<1, TestType>::value>(false, false);
  run_insert_and_find_publication<c2h::get<0, TestType>, c2h::get<1, TestType>::value>(true, false);
}
