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
 *
 * @brief Thrust device_vector with an allocator backed by a data_place.
 *
 * Wraps data_place::allocate/deallocate as a thrust::mr::memory_resource,
 * then uses thrust::mr::allocator to create a compatible allocator.
 * Storage is allocated via data_place (device, composite/VMM, or other
 * place types). The same Thrust code works unchanged for single-device,
 * multi-device (VMM), or green-context placement.
 */

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/memory_resource.h>
#include <thrust/transform.h>

#include <cuda/experimental/__places/partitions/blocked_partition.cuh>

#include <cstdio>

using namespace cuda::experimental::places;

// Minimal adapter: data_place is STF's abstraction; Thrust expects a
// memory_resource. This class bridges the two. The resource must outlive
// any vectors/allocators that use it.
class data_place_memory_resource final : public thrust::mr::memory_resource<thrust::device_ptr<void>>
{
public:
  explicit data_place_memory_resource(const data_place& place)
      : place_(place)
  {}

  pointer do_allocate(std::size_t bytes, std::size_t /*alignment*/) override
  {
    void* raw = place_.allocate(static_cast<std::ptrdiff_t>(bytes));
    return thrust::device_ptr<void>(raw);
  }

  void do_deallocate(pointer p, std::size_t bytes, std::size_t /*alignment*/) override
  {
    place_.deallocate(p.get(), bytes);
  }

  __host__ __device__ bool do_is_equal(const memory_resource& other) const noexcept override
  {
#if defined(__CUDA_ARCH__)
    (void) other;
    return false;
#else
    auto* o = dynamic_cast<const data_place_memory_resource*>(&other);
    return o && place_ == o->place_;
#endif
  }

private:
  data_place place_;
};

template <typename T>
using data_place_allocator = thrust::mr::allocator<T, data_place_memory_resource>;

bool run_with_place(const data_place& place, const char* label)
{
  const size_t n = 1024 * 1024;

  data_place_memory_resource memres(place);
  data_place_allocator<double> alloc(&memres);
  thrust::device_vector<double, data_place_allocator<double>> d_vec(n, 0.0, alloc);

  thrust::transform(
    thrust::device,
    thrust::counting_iterator<size_t>(0),
    thrust::counting_iterator<size_t>(n),
    d_vec.begin(),
    [] __device__(size_t i) {
      return 2.0 * static_cast<double>(i);
    });

  thrust::host_vector<double> h_sample(4);
  thrust::copy(d_vec.begin(), d_vec.begin() + 4, h_sample.begin());

  bool ok = (h_sample[0] == 0.0 && h_sample[1] == 2.0 && h_sample[2] == 4.0 && h_sample[3] == 6.0);
  printf(
    "thrust_device_data_place_allocator: %s (%s): %s\n", label, place.to_string().c_str(), ok ? "PASSED" : "FAILED");
  return ok;
}

int main()
{
  bool all_ok = true;

  all_ok &= run_with_place(data_place::device(0), "device(0)");

  all_ok &= run_with_place(data_place::composite(blocked_partition(), exec_place::all_devices()),
                           "composite(blocked_partition, all_devices)");

  return all_ok ? 0 : 1;
}
