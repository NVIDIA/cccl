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

// Temporary nvcc workaround __host__ __device__ dtor conflict in cuda::buffer
#if defined(__CUDACC__)
#  pragma nv_diag_suppress 20011
#endif

#include <cuda/__memory/align_up.h>
#include <cuda/functional>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

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

  ref_type ref{cudax::cuco::empty_key<Key>{empty_k},
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
