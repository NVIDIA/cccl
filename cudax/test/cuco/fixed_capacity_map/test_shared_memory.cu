//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Temporary nvcc workaround __host__ __device__ dtor conflict in cuda::buffer
#if defined(__CUDACC__)
#  pragma nv_diag_suppress 20011
#endif

#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map_ref.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

namespace cudax = cuda::experimental;

constexpr int empty_key   = -1;
constexpr int empty_value = -1;

// A static-capacity map with cg_size 1 so the test can use scalar device inserts.
using probing               = cudax::cuco::linear_probing<1, cuda::hash<int>>;
inline constexpr int bucket = 1;
inline constexpr ::cuda::std::size_t static_capacity =
  cudax::cuco::make_valid_capacity<probing, bucket>(::cuda::std::size_t{512});
using fixed_capacity_map_512_type = cudax::cuco::
  fixed_capacity_map<int, int, static_capacity, ::cuda::thread_scope_device, ::cuda::std::equal_to<int>, probing>;

template <class Pair>
struct iota_pair
{
  __host__ __device__ Pair operator()(typename Pair::first_type key) const noexcept
  {
    return Pair{key, static_cast<typename Pair::second_type>(key)};
  }
};

struct is_nonzero
{
  __device__ bool operator()(int v) const noexcept
  {
    return v != 0;
  }
};

// Demonstrates compile-time __shared__ sizing via ref_type::capacity_v.
template <class PairIt>
__global__ void insert_shmem_kernel(fixed_capacity_map_512_type::ref_type global_ref, PairIt pairs, int num_keys)
{
  using ref_t = fixed_capacity_map_512_type::ref_type;
  static_assert(ref_t::capacity_v != ::cuda::std::dynamic_extent,
                "capacity_v must be a compile-time constant for static extents");

  __shared__ ::cuda::__uninitialized_array<ref_t::value_type, ref_t::capacity_v> smem;

  const auto idx    = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  smem[threadIdx.x] = (idx < num_keys) ? pairs[idx] : ref_t::value_type{};
  __syncthreads();
  if (idx < num_keys)
  {
    global_ref.insert(smem[threadIdx.x]);
  }
}

C2H_TEST("fixed_capacity_map static extent — shared memory sizing via capacity_v", "[shmem][static]")
{
  constexpr int num_keys = 64;

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  fixed_capacity_map_512_type map{stream, mr, cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};

  const int block_size = 128;
  const int grid_size  = (num_keys + block_size - 1) / block_size;

  insert_shmem_kernel<<<grid_size, block_size, 0, stream.get()>>>(
    map.ref(),
    cuda::transform_iterator(cuda::counting_iterator<int>{0}, iota_pair<fixed_capacity_map_512_type::value_type>{}),
    num_keys);
  REQUIRE(cudaGetLastError() == cudaSuccess);

  // Verify the insertions actually landed in the global map
  auto found = ::cuda::make_buffer<int>(stream, mr, num_keys, 0);
  map.contains(stream, cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, found.begin());
  REQUIRE(::thrust::all_of(::thrust::cuda::par.on(stream.get()), found.data(), found.data() + num_keys, is_nonzero{}));
}

// A block-scoped ref over the same static capacity, for maps living entirely in shared memory.
using block_ref_512_type = cudax::cuco::
  fixed_capacity_map_ref<int, int, ::cuda::thread_scope_block, ::cuda::std::equal_to<int>, probing, bucket, static_capacity>;

// Builds a map over raw shared memory via device initialize, then inserts and finds per thread.
__global__ void shmem_map_lifecycle_kernel(int* thread_ok)
{
  using ref_t = block_ref_512_type;
  static_assert(ref_t::capacity_v != ::cuda::std::dynamic_extent,
                "capacity_v must be a compile-time constant for static extents");

  // Alignment to sizeof(value_type) enables packed CAS when value_type has a packable representation
  __shared__ ::cuda::__uninitialized_array<ref_t::value_type, ref_t::capacity_v, sizeof(ref_t::value_type)> smem;

  const auto block = ::cooperative_groups::this_thread_block();
  ref_t ref{cudax::cuco::empty_key{empty_key},
            cudax::cuco::empty_value{empty_value},
            ::cuda::std::equal_to<int>{},
            probing{},
            ref_t::storage_span_type{smem.data(), ref_t::capacity_v}};

  ref.initialize(block);

  const int rank = static_cast<int>(block.thread_rank());

  // After initialize and before any insert the map must be empty (read-only, so no data race)
  bool ok = !ref.contains(rank);
  block.sync();

  ref.insert(ref_t::value_type{rank, rank});
  block.sync();

  const auto it = ref.find(rank);
  ok            = ok && (it != ref.end()) && (it->first == rank) && (it->second == rank);

  // initialize can also clear a populated map; finish all earlier reads before clearing
  block.sync();
  ref.initialize(block);
  ok = ok && !ref.contains(rank);

  thread_ok[static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + rank] = ok ? 1 : 0;
}

C2H_TEST("fixed_capacity_map_ref device initialize — map fully in shared memory", "[shmem][initialize]")
{
  constexpr int num_blocks = 8;
  constexpr int block_size = 64;

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  auto thread_ok = ::cuda::make_buffer<int>(stream, mr, num_blocks * block_size, 0);
  shmem_map_lifecycle_kernel<<<num_blocks, block_size, 0, stream.get()>>>(thread_ok.data());
  REQUIRE(cudaGetLastError() == cudaSuccess);

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()), thread_ok.data(), thread_ok.data() + num_blocks * block_size, is_nonzero{}));
}

// Each block copies the global map into shared memory and probes the copy.
template <class GlobalRef>
__global__ void
make_copy_shmem_kernel(GlobalRef global_ref, int num_keys, int* keys_exist, int* pairs_correct, int* copy_mutable)
{
  using ref_t       = GlobalRef;
  using key_type    = typename ref_t::key_type;
  using mapped_type = typename ref_t::mapped_type;
  using value_type  = typename ref_t::value_type;

  // Alignment to sizeof(value_type) enables packed CAS when value_type has a packable representation
  __shared__ ::cuda::__uninitialized_array<value_type, ref_t::capacity_v, sizeof(value_type)> smem;

  const auto block = ::cooperative_groups::this_thread_block();
  // Exercise destruction and reinitialization of make_copy's shared barrier.
  (void) global_ref.template make_copy<::cuda::thread_scope_block>(
    block, typename ref_t::storage_span_type{smem.data(), ref_t::capacity_v});
  auto shared_ref = global_ref.template make_copy<::cuda::thread_scope_block>(
    block, typename ref_t::storage_span_type{smem.data(), ref_t::capacity_v});
  static_assert(::cuda::std::decay_t<decltype(shared_ref)>::thread_scope == ::cuda::thread_scope_block,
                "make_copy must rebind the thread scope of the returned ref");

  const int offset = static_cast<int>(blockIdx.x) * num_keys;
  for (int i = static_cast<int>(block.thread_rank()); i < num_keys; i += static_cast<int>(block.size()))
  {
    const auto it = shared_ref.find(static_cast<key_type>(i));
    if (it != shared_ref.end())
    {
      keys_exist[offset + i] = 1;
      pairs_correct[offset + i] =
        (it->first == static_cast<key_type>(i) && it->second == static_cast<mapped_type>(i)) ? 1 : 0;
    }
    else
    {
      keys_exist[offset + i]    = 0;
      pairs_correct[offset + i] = 1;
    }
  }

  block.sync();
  if (block.thread_rank() == 0)
  {
    const auto new_key     = static_cast<key_type>(num_keys);
    const auto new_value   = static_cast<mapped_type>(num_keys) + mapped_type{1};
    const bool inserted    = shared_ref.insert(value_type{new_key, new_value});
    const auto inserted_it = shared_ref.find(new_key);
    copy_mutable[blockIdx.x] =
      (inserted && inserted_it != shared_ref.end() && inserted_it->second == new_value
       && shared_ref.erased_key_sentinel() == global_ref.erased_key_sentinel())
        ? 1
        : 0;
  }
}

using make_copy_types = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;

C2H_TEST("fixed_capacity_map_ref make_copy — shared memory copy of a map",
         "[shmem][make_copy]",
         make_copy_types,
         make_copy_types)
{
  using key_type    = c2h::get<0, TestType>;
  using mapped_type = c2h::get<1, TestType>;

  using probing_t         = cudax::cuco::linear_probing<1, cuda::hash<key_type>>;
  constexpr auto capacity = cudax::cuco::make_valid_capacity<probing_t, bucket>(::cuda::std::size_t{512});
  using map_type          = cudax::cuco::fixed_capacity_map<
    key_type,
    mapped_type,
    capacity,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<key_type>,
    probing_t>;
  using value_type = typename map_type::value_type;

  constexpr int num_keys   = 300;
  constexpr int num_blocks = 8;
  constexpr int block_size = 128;

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  map_type map{stream,
               mr,
               cudax::cuco::empty_key{static_cast<key_type>(-1)},
               cudax::cuco::empty_value{static_cast<mapped_type>(-1)},
               cudax::cuco::erased_key{static_cast<key_type>(-2)}};

  auto keys_exist    = ::cuda::make_buffer<int>(stream, mr, num_blocks * num_keys, 0);
  auto pairs_correct = ::cuda::make_buffer<int>(stream, mr, num_blocks * num_keys, 0);
  auto copy_mutable  = ::cuda::make_buffer<int>(stream, mr, num_blocks, 0);

  SECTION("all keys are found after insertion and pairs are correct")
  {
    auto pairs = cuda::transform_iterator(cuda::counting_iterator<key_type>{0}, iota_pair<value_type>{});
    map.insert(stream, pairs, pairs + num_keys);

    make_copy_shmem_kernel<<<num_blocks, block_size, 0, stream.get()>>>(
      map.ref(), num_keys, keys_exist.data(), pairs_correct.data(), copy_mutable.data());
    REQUIRE(cudaGetLastError() == cudaSuccess);

    REQUIRE(::thrust::all_of(
      ::thrust::cuda::par.on(stream.get()), keys_exist.data(), keys_exist.data() + num_blocks * num_keys, is_nonzero{}));
    REQUIRE(::thrust::all_of(
      ::thrust::cuda::par.on(stream.get()),
      pairs_correct.data(),
      pairs_correct.data() + num_blocks * num_keys,
      is_nonzero{}));
    REQUIRE(::thrust::all_of(
      ::thrust::cuda::par.on(stream.get()), copy_mutable.data(), copy_mutable.data() + num_blocks, is_nonzero{}));
  }

  SECTION("no key is found before insertion")
  {
    make_copy_shmem_kernel<<<num_blocks, block_size, 0, stream.get()>>>(
      map.ref(), num_keys, keys_exist.data(), pairs_correct.data(), copy_mutable.data());
    REQUIRE(cudaGetLastError() == cudaSuccess);

    REQUIRE(!::thrust::any_of(
      ::thrust::cuda::par.on(stream.get()), keys_exist.data(), keys_exist.data() + num_blocks * num_keys, is_nonzero{}));
    REQUIRE(::thrust::all_of(
      ::thrust::cuda::par.on(stream.get()), copy_mutable.data(), copy_mutable.data() + num_blocks, is_nonzero{}));
  }
}

// Initializes a dynamic-extent map in global memory, fills it, and probes a same-scope copy.
template <class Ref>
__global__ void
dynamic_initialize_make_copy_kernel(Ref ref, typename Ref::storage_span_type copy_slots, int num_keys, int* thread_ok)
{
  const auto block = ::cooperative_groups::this_thread_block();
  ref.initialize(block);

  const int rank = static_cast<int>(block.thread_rank());

  // After initialize and before any insert the map must be empty (read-only, so no data race)
  bool ok = !ref.contains(rank);
  block.sync();

  if (rank < num_keys)
  {
    ref.insert(typename Ref::value_type{rank, rank + 1});
  }
  block.sync();

  auto copy_ref = ref.make_copy(block, copy_slots);
  static_assert(::cuda::std::is_same_v<decltype(copy_ref), Ref>,
                "make_copy without explicit scope must preserve the ref type");

  if (rank == 0)
  {
    copy_ref.insert(typename Ref::value_type{num_keys, num_keys + 1});
    ok = ok && copy_ref.erased_key_sentinel() == copy_ref.empty_key_sentinel();
  }
  block.sync();

  if (rank < num_keys)
  {
    const auto it = copy_ref.find(rank);
    ok            = ok && (it != copy_ref.end()) && (it->second == rank + 1);
  }
  else if (rank == num_keys)
  {
    const auto it = copy_ref.find(num_keys);
    ok            = ok && (it != copy_ref.end()) && (it->second == num_keys + 1);
  }
  thread_ok[rank] = ok ? 1 : 0;
}

C2H_TEST("fixed_capacity_map_ref initialize and make_copy — dynamic extent in global memory",
         "[global][initialize][make_copy]")
{
  constexpr int dynamic_bucket = 2;
  using dyn_ref_type           = cudax::cuco::
    fixed_capacity_map_ref<int, int, ::cuda::thread_scope_device, ::cuda::std::equal_to<int>, probing, dynamic_bucket>;
  using value_type = dyn_ref_type::value_type;

  constexpr int num_keys   = 100;
  constexpr int block_size = 128;
  const auto capacity      = cudax::cuco::make_valid_capacity<probing, dynamic_bucket>(::cuda::std::size_t{256});

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  auto slots      = ::cuda::make_buffer<value_type>(stream, mr, capacity, ::cuda::no_init);
  auto copy_slots = ::cuda::make_buffer<value_type>(stream, mr, capacity, ::cuda::no_init);
  auto thread_ok  = ::cuda::make_buffer<int>(stream, mr, block_size, 0);

  const dyn_ref_type ref{
    cudax::cuco::empty_key{empty_key},
    cudax::cuco::empty_value{empty_value},
    ::cuda::std::equal_to<int>{},
    probing{},
    dyn_ref_type::storage_span_type{slots.data(), capacity}};

  dynamic_initialize_make_copy_kernel<<<1, block_size, 0, stream.get()>>>(
    ref, dyn_ref_type::storage_span_type{copy_slots.data(), capacity}, num_keys, thread_ok.data());
  REQUIRE(cudaGetLastError() == cudaSuccess);

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()), thread_ok.data(), thread_ok.data() + block_size, is_nonzero{}));
}
