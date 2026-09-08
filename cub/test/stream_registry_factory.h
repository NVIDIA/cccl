// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cub/util_device.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/std/__algorithm/find.h>
#include <cuda/std/execution>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>

//! @file
//! This file contains a CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY replacement,
//! `stream_registry_factory_t`, used by the environment-based launch wrappers in
//! catch2_test_launch_helper.h to verify that CUB device-scope algorithms use the stream and (optionally)
//! only the kernels provided through their environment.

struct get_allowed_kernels_t
{};

__host__ __device__ static cuda::std::execution::prop<get_allowed_kernels_t, cuda::std::span<void*>>
allowed_kernels(cuda::std::span<void*> allowed_kernels)
{
  return cuda::std::execution::prop{get_allowed_kernels_t{}, allowed_kernels};
}

struct stream_registry_factory_state_t
{
  cuda::std::optional<cudaStream_t> m_stream;
  cuda::std::span<void*> m_kernels;
};

static CUB_RUNTIME_FUNCTION stream_registry_factory_state_t* get_stream_registry_factory_state()
{
  stream_registry_factory_state_t* ptr{};
  NV_IF_ELSE_TARGET(NV_IS_HOST, (static stream_registry_factory_state_t state; ptr = &state;), (ptr = nullptr;));
  return ptr;
}

struct kernel_launcher_t : thrust::cuda_cub::detail::triple_chevron
{
  CUB_RUNTIME_FUNCTION kernel_launcher_t(
    dim3 grid, dim3 block, size_t shared_mem = 0, cudaStream_t stream = nullptr, bool dependent_launch = false)
      : thrust::cuda_cub::detail::triple_chevron(grid, block, shared_mem, stream, dependent_launch)
  {}

  template <class K, class... Args>
  CUB_RUNTIME_FUNCTION cudaError_t doit(K kernel, Args const&... args) const
  {
    NV_IF_TARGET(NV_IS_HOST, ({
                   auto& kernels = get_stream_registry_factory_state()->m_kernels;
                   if (!kernels.empty())
                   {
                     if (cuda::std::find(kernels.begin(), kernels.end(), reinterpret_cast<void*>(kernel))
                         == kernels.end())
                     {
                       FAIL("Kernel is not allowed: " << c2h::type_name<K>());
                     }
                   }
                 }));
    return thrust::cuda_cub::detail::triple_chevron::doit(kernel, args...);
  }
};

struct stream_registry_factory_t
{
  CUB_RUNTIME_FUNCTION kernel_launcher_t
  operator()(dim3 grid, dim3 block, size_t shared_mem, cudaStream_t stream, bool dependent_launch = false) const
  {
    NV_IF_TARGET(NV_IS_HOST, (if (get_stream_registry_factory_state()->m_stream) {
                   REQUIRE(stream == get_stream_registry_factory_state()->m_stream);
                 }));
    return kernel_launcher_t(grid, block, shared_mem, stream, dependent_launch);
  }

  CUB_RUNTIME_FUNCTION cudaError_t PtxVersion(int& version)
  {
    return cub::PtxVersion(version);
  }

  CUB_RUNTIME_FUNCTION cudaError_t PtxComputeCap(::cuda::compute_capability& cc) const
  {
    return cub::detail::ptx_compute_cap(cc);
  }

  CUB_RUNTIME_FUNCTION cudaError_t MultiProcessorCount(int& sm_count) const
  {
    int device_ordinal;
    const cudaError_t error = cudaGetDevice(&device_ordinal);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get SM count
    return cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal);
  }

  template <typename Kernel>
  CUB_RUNTIME_FUNCTION cudaError_t
  MaxSmOccupancy(int& sm_occupancy, Kernel kernel_ptr, int block_size, int dynamic_smem_bytes = 0)
  {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(&sm_occupancy, kernel_ptr, block_size, dynamic_smem_bytes);
  }

  _CCCL_HIDE_FROM_ABI CUB_RUNTIME_FUNCTION ::cudaError_t
  MemcpyAsync(void* dst, const void* src, size_t num_bytes, ::cudaMemcpyKind kind, ::cudaStream_t stream) const
  {
    NV_IF_TARGET(NV_IS_HOST, ({
                   if (get_stream_registry_factory_state()->m_stream)
                   {
                     REQUIRE(stream == get_stream_registry_factory_state()->m_stream);
                   }
                 }));
    return ::cudaMemcpyAsync(dst, src, num_bytes, kind, stream);
  }

  CUB_RUNTIME_FUNCTION cudaError_t MaxGridDimX(int& max_grid_dim_x) const
  {
    int device_ordinal;
    const cudaError_t error = cudaGetDevice(&device_ordinal);
    if (cudaSuccess != error)
    {
      return error;
    }

    // Get max grid dimension
    return cudaDeviceGetAttribute(&max_grid_dim_x, cudaDevAttrMaxGridDimX, device_ordinal);
  }

  CUB_RUNTIME_FUNCTION cudaError_t MemsetAsync(void* dst, unsigned char value, size_t num_bytes, cudaStream_t stream)
  {
    return cudaMemsetAsync(dst, value, num_bytes, stream);
  }

  CUB_RUNTIME_FUNCTION cudaError_t MaxSharedMemory(int& max_shared_memory) const
  {
    int device       = 0;
    const auto error = cudaGetDevice(&device);
    if (error != cudaSuccess)
    {
      return error;
    }

    return cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlock, device);
  }

  template <typename Kernel>
  CUB_RUNTIME_FUNCTION cudaError_t max_dynamic_smem_size_for(int& max_dynamic_smem_size, Kernel kernel_ptr)
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, //
                      ({ return cub::MaxPotentialDynamicSmemBytes(max_dynamic_smem_size, kernel_ptr); }),
                      ({
                        cudaFuncAttributes func_attrs{};
                        if (const auto error = cudaFuncGetAttributes(&func_attrs, kernel_ptr))
                        {
                          return error;
                        }
                        max_dynamic_smem_size = func_attrs.maxDynamicSharedSizeBytes;
                        return cudaSuccess;
                      }))
  }

  template <typename Kernel>
  CUB_RUNTIME_FUNCTION cudaError_t set_max_dynamic_smem_size_for(Kernel kernel_ptr, int smem_size)
  {
    return cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
};

struct stream_scope
{
  stream_scope(cudaStream_t stream)
  {
    get_stream_registry_factory_state()->m_stream = stream;
  }

  ~stream_scope()
  {
    get_stream_registry_factory_state()->m_stream = cuda::std::nullopt;
  }
};

struct kernel_scope
{
  kernel_scope(cuda::std::span<void*> allowed_kernels)
  {
    get_stream_registry_factory_state()->m_kernels = allowed_kernels;
  }

  ~kernel_scope()
  {
    get_stream_registry_factory_state()->m_kernels = {};
  }
};

// Checks that CUB's default kernel launcher, CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY, is
// `stream_registry_factory_t`.
template <class Dummy = int> // dummy template argument to make the static_assert dependent
void check_uses_stream_registry_factory()
{
  static_assert(
    sizeof(Dummy) > 0 && cuda::std::is_same_v<CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY, stream_registry_factory_t>,
    "Helper relies on the fact that CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY is "
    "`stream_registry_factory_t`");
}
