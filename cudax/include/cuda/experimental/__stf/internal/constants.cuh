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
 * @brief Different constant values and some related methods
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/utility/unittest.cuh>

namespace cuda::experimental::stf
{

/**
 * @brief Encodes an access mode (read, write, redux, ...)
 */
enum class access_mode : unsigned int
{
  none    = 0,
  read    = 1,
  write   = 2,
  rw      = 3, // READ + WRITE
  relaxed = 4, /* operator ? */
  reduce  = 8, // overwrite the content of the logical data (if any) with the result of the reduction (equivalent to
              // write)
  reduce_no_init = 16, // special case where the reduction will accumulate into the existing content (equivalent to rw)
};

// Trait to check if a type is a valid tag
template <typename T>
struct is_access_mode_tag : ::std::false_type
{};

// Helper for SFINAE or static_assert
template <typename T>
static constexpr bool is_access_mode_tag_v = is_access_mode_tag<T>::value;

/**
 * @brief Tag types when we want to do a static dispatch based on access modes
 */
struct access_mode_tag
{
  struct none
  {};
  struct read
  {};
  struct write
  {};
  struct rw
  {};
  struct relaxed
  {};
  struct reduce
  {};
  struct reduce_no_init
  {};

  // Function to get the runtime value for a tag type
  template <typename Tag>
  static constexpr access_mode to_access_mode(Tag)
  {
    static_assert(is_access_mode_tag_v<Tag>);

    if constexpr (::std::is_same_v<Tag, none>)
    {
      return access_mode::none;
    }
    else if constexpr (::std::is_same_v<Tag, read>)
    {
      return access_mode::read;
    }
    else if constexpr (::std::is_same_v<Tag, write>)
    {
      return access_mode::write;
    }
    else if constexpr (::std::is_same_v<Tag, rw>)
    {
      return access_mode::rw;
    }
    else if constexpr (::std::is_same_v<Tag, relaxed>)
    {
      return access_mode::relaxed;
    }
    else if constexpr (::std::is_same_v<Tag, reduce>)
    {
      return access_mode::reduce;
    }
    else
    {
      static_assert(::std::is_same_v<Tag, reduce_no_init>);
      return access_mode::reduce_no_init;
    }
  }
};

template <>
struct is_access_mode_tag<access_mode_tag::none> : ::std::true_type
{};

template <>
struct is_access_mode_tag<access_mode_tag::read> : ::std::true_type
{};

template <>
struct is_access_mode_tag<access_mode_tag::write> : ::std::true_type
{};

template <>
struct is_access_mode_tag<access_mode_tag::rw> : ::std::true_type
{};

template <>
struct is_access_mode_tag<access_mode_tag::relaxed> : ::std::true_type
{};

template <>
struct is_access_mode_tag<access_mode_tag::reduce> : ::std::true_type
{};

template <>
struct is_access_mode_tag<access_mode_tag::reduce_no_init> : ::std::true_type
{};

/**
 * @brief Combines two access mode into a compatible access mode for both
 *         accesses (eg. read+write = rw, read+read = read)
 */
inline access_mode operator|(access_mode lhs, access_mode rhs)
{
  assert(as_underlying(lhs) < 16);
  assert(as_underlying(rhs) < 16);
  EXPECT(lhs != access_mode::relaxed);
  EXPECT(rhs != access_mode::relaxed);
  return access_mode(as_underlying(lhs) | as_underlying(rhs));
}

inline access_mode& operator|=(access_mode& lhs, access_mode rhs)
{
  return lhs = lhs | rhs;
}

/**
 * @brief Convert an access mode into a C string
 */
inline const char* access_mode_string(access_mode mode)
{
  switch (mode)
  {
    case access_mode::none:
      return "none";
    case access_mode::read:
      return "read";
    case access_mode::rw:
      return "rw";
    case access_mode::write:
      return "write";
    case access_mode::relaxed:
      return "relaxed"; // op ?
    case access_mode::reduce_no_init:
      return "reduce (no init)"; // op ?
    case access_mode::reduce:
      return "reduce"; // op ?
    default:
      assert(false);
      abort();
  }
}

/**
 * @brief A tag type used in combination with the reduce access mode to
 * indicate that we should not overwrite a logical data, but instead accumulate
 * the result of the reduction with the existing content of the logical data
 * (thus making it a rw access, instead a of a write-only access)
 */
class no_init
{};

} // namespace cuda::experimental::stf
