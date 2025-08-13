//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cstddef>
#include <cstdlib>

namespace cuda::experimental::cufile::test_utils
{

/**
 * @brief Check if CUDA is available on the system
 */
bool is_cuda_available();

/**
 * @brief Allocate GPU memory for testing
 */
void* allocate_gpu_memory(size_t size);

/**
 * @brief Allocate pinned host memory for testing
 */
void* allocate_host_memory(size_t size);

/**
 * @brief Free GPU memory
 */
void free_gpu_memory(void* ptr);

/**
 * @brief Free host memory
 */
void free_host_memory(void* ptr);

/**
 * @brief Allocate regular system memory for testing
 */
void* allocate_regular_memory(size_t size);

/**
 * @brief Free regular system memory
 */
void free_regular_memory(void* ptr);

/**
 * @brief RAII wrapper for GPU memory allocation
 */
class GPUMemoryRAII
{
public:
  explicit GPUMemoryRAII(size_t size);
  ~GPUMemoryRAII();

  GPUMemoryRAII(const GPUMemoryRAII&)            = delete;
  GPUMemoryRAII& operator=(const GPUMemoryRAII&) = delete;

  GPUMemoryRAII(GPUMemoryRAII&& other) noexcept
      : ptr_(other.ptr_)
      , size_(other.size_)
  {
    other.ptr_  = nullptr;
    other.size_ = 0;
  }

  GPUMemoryRAII& operator=(GPUMemoryRAII&& other) noexcept
  {
    if (this != &other)
    {
      free_gpu_memory(ptr_);
      ptr_        = other.ptr_;
      size_       = other.size_;
      other.ptr_  = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void* get() const;
  size_t size() const;

private:
  void* ptr_;
  size_t size_;
};

/**
 * @brief RAII wrapper for host memory allocation
 */
class HostMemoryRAII
{
public:
  explicit HostMemoryRAII(size_t size);
  ~HostMemoryRAII();

  HostMemoryRAII(const HostMemoryRAII&)            = delete;
  HostMemoryRAII& operator=(const HostMemoryRAII&) = delete;

  HostMemoryRAII(HostMemoryRAII&& other) noexcept
      : ptr_(other.ptr_)
      , size_(other.size_)
  {
    other.ptr_  = nullptr;
    other.size_ = 0;
  }

  HostMemoryRAII& operator=(HostMemoryRAII&& other) noexcept
  {
    if (this != &other)
    {
      free_host_memory(ptr_);
      ptr_        = other.ptr_;
      size_       = other.size_;
      other.ptr_  = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void* get() const;
  size_t size() const;

private:
  void* ptr_;
  size_t size_;
};

/**
 * @brief RAII wrapper for regular memory allocation
 */
class RegularMemoryRAII
{
public:
  explicit RegularMemoryRAII(size_t size);
  ~RegularMemoryRAII();

  RegularMemoryRAII(const RegularMemoryRAII&)            = delete;
  RegularMemoryRAII& operator=(const RegularMemoryRAII&) = delete;

  RegularMemoryRAII(RegularMemoryRAII&& other) noexcept
      : ptr_(other.ptr_)
      , size_(other.size_)
  {
    other.ptr_  = nullptr;
    other.size_ = 0;
  }

  RegularMemoryRAII& operator=(RegularMemoryRAII&& other) noexcept
  {
    if (this != &other)
    {
      free_regular_memory(ptr_);
      ptr_        = other.ptr_;
      size_       = other.size_;
      other.ptr_  = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void* get() const;
  size_t size() const;

private:
  void* ptr_;
  size_t size_;
};

} // namespace cuda::experimental::cufile::test_utils
