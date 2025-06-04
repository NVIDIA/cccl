/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _CUDAX__CUCO_HASH_FUNCTIONS_CUH
#define _CUDAX__CUCO_HASH_FUNCTIONS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/functional.h>

#include <cuda/experimental/__cuco/detail/hash_functions/xxhash.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{

/**
 * @brief A 32-bit `XXH32` hash function to hash the given argument on host and device.
 *
 * @tparam Key The type of the values to hash
 */
template <typename Key>
using xxhash_32 = detail::XXHash_32<Key>;

} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_HASH_FUNCTIONS_CUH
