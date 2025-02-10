#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda.h>

#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{

struct CudaDriverLauncher
{
  dim3 grid;
  dim3 block;
  size_t shared_mem;
  CUstream stream;
  bool dependent_launch;

  template <typename... Args>
  _CCCL_HIDE_FROM_ABI cudaError_t doit(CUkernel kernel, Args const&... args) const
  {
    void* kernel_args[] = {const_cast<void*>(static_cast<void const*>(&args))...};

    CUfunction kernel_fn;
    auto status = static_cast<cudaError_t>(cuKernelGetFunction(&kernel_fn, kernel));
    if (status != cudaSuccess)
    {
      return status;
    }

#if _CCCL_HAS_PDL
    if (dependent_launch)
    {
      CUlaunchAttribute attribute[1];
      attribute[0].id                                           = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
      attribute[0].value.programmaticStreamSerializationAllowed = 1;

      CUlaunchConfig config{};
      config.gridDimX       = grid.x;
      config.gridDimY       = grid.y;
      config.gridDimZ       = grid.z;
      config.blockDimX      = block.x;
      config.blockDimY      = block.y;
      config.blockDimZ      = block.z;
      config.sharedMemBytes = shared_mem;
      config.hStream        = stream;
      config.attrs          = attribute;
      config.numAttrs       = 1;

      return static_cast<cudaError_t>(cuLaunchKernelEx(&config, kernel_fn, kernel_args, 0));
    }
    else
#endif
    {
      return static_cast<cudaError_t>(cuLaunchKernel(
        kernel_fn, grid.x, grid.y, grid.z, block.x, block.y, block.z, shared_mem, stream, kernel_args, 0));
    }
  }
};

struct CudaDriverLauncherFactory
{
  CudaDriverLauncher
  operator()(dim3 grid, dim3 block, _CUDA_VSTD::size_t shared_mem, CUstream stream, bool dependent_launch = false) const
  {
    return CudaDriverLauncher{grid, block, shared_mem, stream, dependent_launch};
  }

  cudaError_t PtxVersion(int& version) const
  {
    version = cc;
    return cudaSuccess;
  }

  _CCCL_HIDE_FROM_ABI cudaError_t MultiProcessorCount(int& sm_count) const
  {
    return static_cast<cudaError_t>(cuDeviceGetAttribute(&sm_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  }

  _CCCL_HIDE_FROM_ABI cudaError_t
  MaxSmOccupancy(int& sm_occupancy, CUkernel kernel, int block_size, int dynamic_smem_bytes = 0) const
  {
    // Older drivers have issues handling CUkernel in the occupancy queries, get the CUfunction instead.
    CUfunction kernel_fn;
    auto status = static_cast<cudaError_t>(cuKernelGetFunction(&kernel_fn, kernel));
    if (status != cudaSuccess)
    {
      return status;
    }

    return static_cast<cudaError_t>(
      cuOccupancyMaxActiveBlocksPerMultiprocessor(&sm_occupancy, kernel_fn, block_size, dynamic_smem_bytes));
  }

  CUdevice device;
  int cc;
};

} // namespace detail

CUB_NAMESPACE_END
