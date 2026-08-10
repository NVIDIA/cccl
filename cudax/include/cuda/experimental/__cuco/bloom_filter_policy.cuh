//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_BLOOM_FILTER_POLICY_CUH
#define _CUDAX___CUCO_BLOOM_FILTER_POLICY_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/detail/bloom_filter/bloom_filter_policy.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
//! @brief Sectorized Bloom filter policy with multiplicative-hashing fingerprint generation.
//!
//! Implements the Sectorized Bloom Filter (SBF) variant from "Optimizing Bloom Filters for Modern
//! GPU Architectures" (arXiv:2512.15595).
//!
//! Requires a 64-bit hash function: the result is split into upper 32 bits (block selection via
//! multiply-shift) and lower 32 bits (pattern generation).
//!
//! @tparam _Key Key type to hash
//! @tparam _Hash 64-bit hash functor type. Defaults to
//! `cuco::hash<_Key, hash_algorithm::xxhash_64>`
//! @tparam _Word Underlying word type of a filter block. Defaults to `cuda::std::uint32_t`
//! @tparam _WordsPerBlock Words per filter block. Defaults to the number of `_Word`s that fit in
//! one 32-byte sector
//! @tparam _PatternBits Fingerprint bits per key (the paper's k). Defaults to `_WordsPerBlock`
//! @tparam _AddHorizontalLayout Cooperative-group size for `add` (the paper's Theta). Defaults to
//! `_WordsPerBlock` for a fully horizontal add
//! @tparam _AddVerticalLayout Words per thread per `add` step (the paper's Phi). Defaults to `1`
//! for a fully horizontal add
//! @tparam _ContainsHorizontalLayout Cooperative-group size for `contains`. Defaults to `1` for a
//! fully vertical contains
//! @tparam _ContainsVerticalLayout Words per thread per `contains` step. Defaults to
//! `_WordsPerBlock` for a fully vertical contains
//! @tparam _ConditionalAdd Whether `add` reads each word before the atomic OR and skips the write
//! when the required bits are already set
//! @tparam _EarlyExitContains Whether `contains` short-circuits on the first missing fingerprint
//! slice
template <class _Key,
          class _Hash                                 = hash<_Key, hash_algorithm::xxhash_64>,
          class _Word                                 = ::cuda::std::uint32_t,
          int _WordsPerBlock                          = static_cast<int>(32 / sizeof(_Word)),
          int _PatternBits                            = _WordsPerBlock,
          int _AddHorizontalLayout                    = _WordsPerBlock,
          int _AddVerticalLayout                      = 1,
          int _ContainsHorizontalLayout               = 1,
          int _ContainsVerticalLayout                 = _WordsPerBlock,
          conditional_add_mode _ConditionalAdd        = conditional_add_mode::off,
          early_exit_contains_mode _EarlyExitContains = early_exit_contains_mode::off>
using bloom_filter_policy = __bloom_filter_ns::__bloom_filter_policy<
  _Hash,
  _Word,
  _WordsPerBlock,
  _PatternBits,
  _AddHorizontalLayout,
  _AddVerticalLayout,
  _ContainsHorizontalLayout,
  _ContainsVerticalLayout,
  _ConditionalAdd,
  _EarlyExitContains>;
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_BLOOM_FILTER_POLICY_CUH
