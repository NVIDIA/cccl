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
 * @brief Concrete implementations of data_place_interface
 *
 * This file contains implementations for standard data place types:
 * host, managed, device, invalid, and affine.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/places/data_place_interface.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>
#include <cuda/experimental/__stf/utility/scope_guard.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief Implementation for the invalid data place
 */
class data_place_invalid final : public data_place_interface
{
public:
  bool is_resolved() const override
  {
    return false;
  }

  int get_device_ordinal() const override
  {
    return data_place_interface::invalid;
  }

  ::std::string to_string() const override
  {
    return "invalid";
  }

  size_t hash() const override
  {
    return ::std::hash<int>()(data_place_interface::invalid);
  }

  int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    return 0;
  }

  void* allocate(::std::ptrdiff_t, cudaStream_t) const override
  {
    throw ::std::logic_error("Cannot allocate from invalid data_place");
  }

  void deallocate(void*, size_t, cudaStream_t) const override
  {
    throw ::std::logic_error("Cannot deallocate from invalid data_place");
  }

  bool allocation_is_stream_ordered() const override
  {
    return false;
  }
};

/**
 * @brief Implementation for the host (pinned memory) data place
 */
class data_place_host final : public data_place_interface
{
public:
  bool is_resolved() const override
  {
    return true;
  }

  int get_device_ordinal() const override
  {
    return data_place_interface::host;
  }

  ::std::string to_string() const override
  {
    return "host";
  }

  size_t hash() const override
  {
    return ::std::hash<int>()(data_place_interface::host);
  }

  int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    return 0;
  }

  void* allocate(::std::ptrdiff_t size, cudaStream_t) const override
  {
    void* result = nullptr;
    cuda_safe_call(cudaMallocHost(&result, size));
    return result;
  }

  void deallocate(void* ptr, size_t, cudaStream_t) const override
  {
    cuda_safe_call(cudaFreeHost(ptr));
  }

  bool allocation_is_stream_ordered() const override
  {
    return false;
  }

  CUresult mem_create(CUmemGenericAllocationHandle* handle, size_t size) const override
  {
#if _CCCL_CTK_AT_LEAST(12, 2)
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_HOST;
    prop.location.id         = 0;
    return cuMemCreate(handle, size, &prop, 0);
#else
    (void) handle;
    (void) size;
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
  }
};

/**
 * @brief Implementation for managed memory data place
 */
class data_place_managed final : public data_place_interface
{
public:
  bool is_resolved() const override
  {
    return true;
  }

  int get_device_ordinal() const override
  {
    return data_place_interface::managed;
  }

  ::std::string to_string() const override
  {
    return "managed";
  }

  size_t hash() const override
  {
    return ::std::hash<int>()(data_place_interface::managed);
  }

  int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    return 0;
  }

  void* allocate(::std::ptrdiff_t size, cudaStream_t) const override
  {
    void* result = nullptr;
    cuda_safe_call(cudaMallocManaged(&result, size));
    return result;
  }

  void deallocate(void* ptr, size_t, cudaStream_t) const override
  {
    cuda_safe_call(cudaFree(ptr));
  }

  bool allocation_is_stream_ordered() const override
  {
    return false;
  }
};

/**
 * @brief Implementation for a specific CUDA device data place
 */
class data_place_device final : public data_place_interface
{
public:
  bool is_resolved() const override
  {
    return true;
  }

  explicit data_place_device(int device_id)
      : device_id_(device_id)
  {
    _CCCL_ASSERT(device_id >= 0, "Device ID must be non-negative");
  }

  int get_device_ordinal() const override
  {
    return device_id_;
  }

  ::std::string to_string() const override
  {
    return "dev" + ::std::to_string(device_id_);
  }

  size_t hash() const override
  {
    return ::std::hash<int>()(device_id_);
  }

  int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    return (device_id_ > static_cast<const data_place_device&>(other).device_id_)
         - (device_id_ < static_cast<const data_place_device&>(other).device_id_);
  }

  void* allocate(::std::ptrdiff_t size, cudaStream_t stream) const override
  {
    void* result       = nullptr;
    const int prev_dev = cuda_try<cudaGetDevice>();

    if (prev_dev != device_id_)
    {
      cuda_safe_call(cudaSetDevice(device_id_));
    }

    SCOPE(exit)
    {
      if (prev_dev != device_id_)
      {
        cuda_safe_call(cudaSetDevice(prev_dev));
      }
    };

    cuda_safe_call(cudaMallocAsync(&result, size, stream));
    return result;
  }

  void deallocate(void* ptr, size_t, cudaStream_t stream) const override
  {
    const int prev_dev = cuda_try<cudaGetDevice>();

    if (prev_dev != device_id_)
    {
      cuda_safe_call(cudaSetDevice(device_id_));
    }

    SCOPE(exit)
    {
      if (prev_dev != device_id_)
      {
        cuda_safe_call(cudaSetDevice(prev_dev));
      }
    };

    cuda_safe_call(cudaFreeAsync(ptr, stream));
  }

  bool allocation_is_stream_ordered() const override
  {
    return true;
  }

  CUresult mem_create(CUmemGenericAllocationHandle* handle, size_t size) const override
  {
    CUmemAllocationProp prop = {};
    prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id         = device_id_;
    return cuMemCreate(handle, size, &prop, 0);
  }

private:
  int device_id_;
};

/**
 * @brief Implementation for the affine data place (uses exec_place's affine data place)
 */
class data_place_affine final : public data_place_interface
{
public:
  bool is_resolved() const override
  {
    return false;
  }

  int get_device_ordinal() const override
  {
    return data_place_interface::affine;
  }

  ::std::string to_string() const override
  {
    return "affine";
  }

  size_t hash() const override
  {
    return ::std::hash<int>()(data_place_interface::affine);
  }

  int cmp(const data_place_interface& other) const override
  {
    if (typeid(*this) != typeid(other))
    {
      return typeid(*this).before(typeid(other)) ? -1 : 1;
    }
    return 0;
  }

  void* allocate(::std::ptrdiff_t, cudaStream_t) const override
  {
    throw ::std::logic_error("Cannot allocate from affine data_place directly");
  }

  void deallocate(void*, size_t, cudaStream_t) const override
  {
    throw ::std::logic_error("Cannot deallocate from affine data_place directly");
  }

  bool allocation_is_stream_ordered() const override
  {
    return false;
  }
};
} // end namespace cuda::experimental::stf
