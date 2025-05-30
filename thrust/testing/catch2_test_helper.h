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

// workaround for error #3185-D: no '#pragma diagnostic push' was found to match this 'diagnostic pop'
#if _CCCL_COMPILER(NVHPC)
#  undef CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma("diag push")
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma("diag pop")
#endif
// workaround for error
// * MSVC14.39: #3185-D: no '#pragma diagnostic push' was found to match this 'diagnostic pop'
// * MSVC14.29: internal error: assertion failed: alloc_copy_of_pending_pragma: copied pragma has source sequence entry
//              (pragma.c, line 526 in alloc_copy_of_pending_pragma)
// see also upstream Catch2 issue: https://github.com/catchorg/Catch2/issues/2636
#if _CCCL_COMPILER(MSVC)
#  undef CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS
#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS
#endif

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
