/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// the purpose of this header is to #include the sort.h header
// of the sequential, host, and device systems. It should be #included in any
// code which uses adl to dispatch sort

#include <thrust/system/detail/sequential/sort.h>

// Some build systems need a hint to know which files we actually include
#if 0
#  include <thrust/system/cpp/detail/sort.h>
#  include <thrust/system/cuda/detail/sort.h>
#  include <thrust/system/omp/detail/sort.h>
#  include <thrust/system/tbb/detail/sort.h>
#endif

#include __THRUST_HOST_SYSTEM_ALGORITH_HEADER_INCLUDE(sort.h)
#include __THRUST_DEVICE_SYSTEM_ALGORITH_HEADER_INCLUDE(sort.h)
