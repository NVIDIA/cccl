// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/cstring>
#include <cuda/std/iterator>

namespace cuda_coop_detail
{
// Copying this cell never evaluates a possibly uninitialized numeric value.
// Conversion to T is used only after the caller's complete Store preflight.
template <typename T>
struct alignas(T) opaque_cell
{
  unsigned char bytes[sizeof(T)];

  __device__ opaque_cell() = default;
  // Warp-striped CUB Load constructs a cell from an initialized input value.
  __device__ opaque_cell(T value)
  {
    cuda::std::memcpy(bytes, &value, sizeof(T));
  }
  __device__ opaque_cell(const opaque_cell& other)
  {
    cuda::std::memcpy(bytes, other.bytes, sizeof(T));
  }
  __device__ opaque_cell& operator=(const opaque_cell& other)
  {
    cuda::std::memcpy(bytes, other.bytes, sizeof(T));
    return *this;
  }
  __device__ opaque_cell& operator=(T value)
  {
    cuda::std::memcpy(bytes, &value, sizeof(T));
    return *this;
  }
  __device__ operator T() const
  {
    T value;
    cuda::std::memcpy(&value, bytes, sizeof(T));
    return value;
  }
};

// CUB forms a per-lane iterator before testing its valid count. Keep that
// displacement as an integer; form a pointer only for an actually stored slot.
template <typename T>
struct guarded_output
{
  using value_type        = T;
  using difference_type   = int;
  using pointer           = T*;
  using reference         = T&;
  using iterator_category = cuda::std::random_access_iterator_tag;

  T* base;
  int offset;

  __device__ guarded_output operator+(int delta) const
  {
    return {base, offset + delta};
  }
  __device__ T& operator[](int index) const
  {
    return base[offset + index];
  }
};
} // namespace cuda_coop_detail
