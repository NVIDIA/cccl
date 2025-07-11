//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_HOST_DEVICE_TYPES
#define TEST_SUPPORT_HOST_DEVICE_TYPES

#include <cuda/std/initializer_list>
#include <cuda/std/utility>

#if !_CCCL_COMPILER(NVRTC)
struct host_only_type
{
  int val_;

  host_only_type(const int val = 0) noexcept
      : val_(val)
  {}
  host_only_type(cuda::std::initializer_list<int>, const int val) noexcept
      : val_(val)
  {}

  host_only_type(const host_only_type& other) noexcept
      : val_(other.val_)
  {}
  host_only_type(host_only_type&& other) noexcept
      : val_(cuda::std::exchange(other.val_, -1))
  {}

  host_only_type& operator=(const host_only_type& other) noexcept
  {
    val_ = other.val_;
    return *this;
  }

  host_only_type& operator=(host_only_type&& other) noexcept

  {
    val_ = cuda::std::exchange(other.val_, -1);
    return *this;
  }

  ~host_only_type() noexcept {}

  [[nodiscard]] friend bool operator==(const host_only_type& lhs, const host_only_type& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  [[nodiscard]] friend bool operator!=(const host_only_type& lhs, const host_only_type& rhs) noexcept
  {
    return lhs.val_ != rhs.val_;
  }
  [[nodiscard]] friend bool operator<(const host_only_type& lhs, const host_only_type& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
  [[nodiscard]] friend bool operator<=(const host_only_type& lhs, const host_only_type& rhs) noexcept
  {
    return lhs.val_ <= rhs.val_;
  }
  [[nodiscard]] friend bool operator>(const host_only_type& lhs, const host_only_type& rhs) noexcept
  {
    return lhs.val_ > rhs.val_;
  }
  [[nodiscard]] friend bool operator>=(const host_only_type& lhs, const host_only_type& rhs) noexcept
  {
    return lhs.val_ >= rhs.val_;
  }

  void swap(host_only_type& other) noexcept
  {
    cuda::std::swap(val_, other.val_);
  }
};
#endif // !_CCCL_COMPILER(NVRTC)

#if _CCCL_CUDA_COMPILATION()
struct device_only_type
{
  int val_;

  __device__ device_only_type(const int val = 0) noexcept
      : val_(val)
  {}
  __device__ device_only_type(cuda::std::initializer_list<int>, const int val) noexcept
      : val_(val)
  {}

  __device__ device_only_type(const device_only_type& other) noexcept
      : val_(other.val_)
  {}
  __device__ device_only_type(device_only_type&& other) noexcept
      : val_(cuda::std::exchange(other.val_, -1))
  {}

  __device__ device_only_type& operator=(const device_only_type& other) noexcept
  {
    val_ = other.val_;
    return *this;
  }

  __device__ device_only_type& operator=(device_only_type&& other) noexcept

  {
    val_ = cuda::std::exchange(other.val_, -1);
    return *this;
  }

  __device__ ~device_only_type() noexcept {}

  [[nodiscard]] __device__ friend bool operator==(const device_only_type& lhs, const device_only_type& rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  [[nodiscard]] __device__ friend bool operator!=(const device_only_type& lhs, const device_only_type& rhs) noexcept
  {
    return lhs.val_ != rhs.val_;
  }
  [[nodiscard]] __device__ friend bool operator<(const device_only_type& lhs, const device_only_type& rhs) noexcept
  {
    return lhs.val_ < rhs.val_;
  }
  [[nodiscard]] __device__ friend bool operator<=(const device_only_type& lhs, const device_only_type& rhs) noexcept
  {
    return lhs.val_ <= rhs.val_;
  }
  [[nodiscard]] __device__ friend bool operator>(const device_only_type& lhs, const device_only_type& rhs) noexcept
  {
    return lhs.val_ > rhs.val_;
  }
  [[nodiscard]] __device__ friend bool operator>=(const device_only_type& lhs, const device_only_type& rhs) noexcept
  {
    return lhs.val_ >= rhs.val_;
  }

  __device__ void swap(device_only_type& other) noexcept
  {
    cuda::std::swap(val_, other.val_);
  }
};
#endif // _CCCL_CUDA_COMPILATION()

#endif // TEST_SUPPORT_HOST_DEVICE_TYPES
