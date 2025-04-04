#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include "unittest/testframework.h"
#include <c2h/catch2_test_helper.cuh>

// TODO expand this with other iterator types (forward, bidirectional, etc.)

using vector_list = c2h::type_list<
  // host
  thrust::host_vector<signed char>,
  thrust::host_vector<short>,
  thrust::host_vector<int>,
  thrust::host_vector<float>,
  thrust::host_vector<custom_numeric>,
  thrust::host_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::host_memory_resource>>,
  // device
  thrust::device_vector<signed char>,
  thrust::device_vector<short>,
  thrust::device_vector<int>,
  thrust::device_vector<float>,
  thrust::device_vector<custom_numeric>,
  thrust::device_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::device_memory_resource>>,
  // universal
  thrust::universal_vector<int>,
  thrust::universal_host_pinned_vector<int>>;
