//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Example: Thrust device_vector with an allocator that takes a data_place.
 *        Storage is allocated via data_place::allocate (device, composite/VMM,
 *        or other place types). The data_place_allocator class is defined in
 *        this example only.
 */

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cuda/experimental/__stf/places/blocked_partition.cuh>
#include <cuda/experimental/__stf/places/exec/green_context.cuh>
#include <cuda/experimental/stf.cuh>

#include <iostream>
#include <limits>

using namespace cuda::experimental::stf;

// Thrust device allocator that delegates to a data_place (example-only, not part of STF API).
template <typename T>
class data_place_allocator
{
public:
  using value_type      = T;
  using pointer         = thrust::device_ptr<T>;
  using const_pointer   = thrust::device_ptr<const T>;
  using reference       = thrust::device_reference<T>;
  using const_reference = thrust::device_reference<const T>;
  using size_type       = ::std::size_t;
  using difference_type = ::std::ptrdiff_t;

  template <typename U>
  struct rebind
  {
    using other = data_place_allocator<U>;
  };

  explicit data_place_allocator(const data_place& place)
      : place_(place)
  {}

  data_place_allocator() = default;

  template <typename U>
  data_place_allocator(const data_place_allocator<U>& other)
      : place_(other.place_)
  {}

  pointer allocate(size_type cnt, const_pointer = const_pointer(static_cast<T*>(nullptr)))
  {
    if (cnt == 0)
    {
      return pointer(nullptr);
    }
    if (cnt > max_size())
    {
      ::cuda::std::__throw_bad_alloc();
    }
    const size_type size_bytes = cnt * sizeof(T);
    void* raw                  = place_.allocate(static_cast<::std::ptrdiff_t>(size_bytes));
    return pointer(static_cast<T*>(raw));
  }

  void deallocate(pointer p, size_type cnt) noexcept
  {
    if (!p)
    {
      return;
    }
    const size_type size_bytes = cnt * sizeof(T);
    place_.deallocate(p.get(), size_bytes);
  }

  size_type max_size() const
  {
    return (::std::numeric_limits<size_type>::max)() / sizeof(T);
  }

  bool operator==(const data_place_allocator& rhs) const
  {
    return place_ == rhs.place_;
  }

  bool operator!=(const data_place_allocator& rhs) const
  {
    return !(*this == rhs);
  }

private:
  const data_place place_;

  template <typename U>
  friend class data_place_allocator;
};

// Run the Thrust example with the given data_place; returns true if the check passed.
bool run_with_place(const data_place& place, const char* label)
{
  const size_t n = 1024 * 1024;

  data_place_allocator<double> alloc(place);
  thrust::device_vector<double, data_place_allocator<double>> d_vec(n, 0.0, alloc);

  thrust::transform(
    thrust::device,
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator<size_t>(n),
    d_vec.begin(),
    [] _CCCL_DEVICE(size_t i) {
      return 2.0 * static_cast<double>(i);
    });

  thrust::host_vector<double> h_sample(4);
  thrust::copy(d_vec.begin(), d_vec.begin() + 4, h_sample.begin());

  bool ok = (h_sample[0] == 0.0 && h_sample[1] == 2.0 && h_sample[2] == 4.0 && h_sample[3] == 6.0);
  if (!ok)
  {
    std::cerr << "thrust_device_data_place_allocator: " << label << " (" << place.to_string() << "): FAILED\n";
  }
  return ok;
}

int main()
{
  bool all_ok = true;

  // Device 0
  all_ok &= run_with_place(data_place::device(0), "device(0)");

  // All devices (composite, VMM path when multiple devices)
  all_ok &= run_with_place(data_place::composite(blocked_partition(), exec_place::all_devices()),
                           "composite(blocked_partition, all_devices)");

#if _CCCL_CTK_AT_LEAST(12, 4)
  // Green context grid (composite, VMM path)
  {
    async_resources_handle handle;
    const int num_sms = 8;
    const int dev_id  = 0;
    auto gc_helper    = handle.get_gc_helper(dev_id, num_sms);
    if (gc_helper->get_count() >= 1)
    {
      auto where     = gc_helper->get_grid(true);
      data_place cdp = data_place::composite(blocked_partition(), where);
      all_ok &= run_with_place(cdp, "composite(blocked_partition, green_context_grid)");
    }
  }
}
#endif

return all_ok ? 0 : 1;
}
