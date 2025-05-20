#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include "unittest/testframework.h"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

// corresponds to DECLARE_VECTOR_UNITTEST
using vector_list = cuda::std::__type_list<
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

// corresponds to DECLARE_VARIABLE_UNITTEST
using variable_list =
  cuda::std::__type_list<signed char, unsigned char, short, unsigned short, int, unsigned int, float, double>;
