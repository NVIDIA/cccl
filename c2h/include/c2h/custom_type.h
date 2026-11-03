// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/__cmath/round_up.h>
#include <cuda/std/limits>

#include <cstdint>
#include <memory>
#include <ostream>

namespace c2h
{
struct custom_type_state_t
{
  std::size_t key{};
  std::size_t val{};
};

template <template <typename> class... Policies>
class custom_type_t;

namespace detail
{
//! Number of policies \p CustomType was instantiated with.
template <class CustomType>
struct policy_count_t;

template <template <typename> class... Policies>
struct policy_count_t<custom_type_t<Policies...>>
{
  static constexpr std::size_t value = sizeof...(Policies);
};
} // namespace detail

template <template <typename> class... Policies>
class custom_type_t
    : public custom_type_state_t
    , public Policies<custom_type_t<Policies...>>...
{
public:
  friend __host__ std::ostream& operator<<(std::ostream& os, const custom_type_t& self)
  {
    return os << "{ " << self.key << ", " << self.val << " }";
  }
};

//! Inflates a custom type to exactly \p TotalSize bytes, so that algorithms have to fall back to
//! temporary storage backed by global memory (vsmem).
template <std::size_t TotalSize>
struct huge_data
{
  template <class CustomType>
  class type
  {
    // because custom types are 8-byte aligned we can only increase size in multiples of 8 bytes
    static_assert(TotalSize % sizeof(std::uint64_t) == 0, "huge_data<TotalSize>: TotalSize must be a multiple of 8");

    // Every other policy contributes exactly one byte (their MSVC workaround member), so counting
    // them is enough, and their order does not matter as they are all byte-aligned. The filler is 8-byte aligned, so
    // those bytes occupy a whole word, regardless of whether the filler comes before or after them.
    static constexpr std::size_t other_policy_bytes = detail::policy_count_t<CustomType>::value - 1;
    static constexpr std::size_t bytes_before_filler =
      sizeof(custom_type_state_t) + ::cuda::round_up(other_policy_bytes, sizeof(std::uint64_t));

    // A second huge_data policy, or any policy contributing more than its one-byte MSVC workaround
    // member, breaks the one-byte-per-policy assumption above and underflows the filler size below.
    static_assert(bytes_before_filler < TotalSize,
                  "huge_data expects to be the only size-contributing policy in the pack, and TotalSize to be "
                  "larger than custom_type_state_t plus one word for the remaining policies");

    // Whatever the other bases leave over becomes the filler. Use large words to minimize register allocation.
    std::uint64_t data[(TotalSize - bytes_before_filler) / sizeof(std::uint64_t)]{};
  };
};

template <class CustomType>
class less_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ bool operator<(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key < rhs.key;
  }
};

template <class CustomType>
class greater_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ bool operator>(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key > rhs.key;
  }
};

template <class CustomType>
class lexicographical_less_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ bool operator<(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key == rhs.key ? lhs.val < rhs.val : lhs.key < rhs.key;
  }
};

template <class CustomType>
class lexicographical_greater_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ bool operator>(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key == rhs.key ? lhs.val > rhs.val : lhs.key > rhs.key;
  }
};

template <class CustomType>
class equal_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ bool operator==(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key == rhs.key && lhs.val == rhs.val;
  }

  friend __host__ __device__ bool operator!=(const CustomType& lhs, const CustomType& rhs)
  {
    return !(lhs == rhs);
  }
};

template <class CustomType>
class subtractable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ CustomType operator-(const CustomType& lhs, const CustomType& rhs)
  {
    CustomType result{};

    result.key = lhs.key - rhs.key;
    result.val = lhs.val - rhs.val;

    return result;
  }
};

template <class CustomType>
class accumulateable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc{};

public:
  friend __host__ __device__ CustomType operator+(const CustomType& lhs, const CustomType& rhs)
  {
    CustomType result{};

    result.key = lhs.key + rhs.key;
    result.val = lhs.val + rhs.val;

    return result;
  }
};
} // namespace c2h

template <template <typename> class... Policies>
class cuda::std::numeric_limits<c2h::custom_type_t<Policies...>>
{
public:
  static constexpr bool is_specialized = true;

  // template <class SizeT = size_t> is a workaround for cudafe++ < 13.1 + gcc < 13 replacing `numeric_limits<size_t>`
  // with `numeric_limits<conditional<is_void_v<void>, __common_type2_imp<uint64_t, uint64_t>::type, void>::type>`

  template <class SizeT = std::size_t>
  static __host__ __device__ c2h::custom_type_t<Policies...> max()
  {
    c2h::custom_type_t<Policies...> val{};
    val.key = numeric_limits<SizeT>::max();
    val.val = numeric_limits<SizeT>::max();
    return val;
  }

  template <class SizeT = std::size_t>
  static __host__ __device__ c2h::custom_type_t<Policies...> min()
  {
    c2h::custom_type_t<Policies...> val{};
    val.key = numeric_limits<SizeT>::min();
    val.val = numeric_limits<SizeT>::min();
    return val;
  }

  template <class SizeT = std::size_t>
  static __host__ __device__ c2h::custom_type_t<Policies...> lowest()
  {
    c2h::custom_type_t<Policies...> val{};
    val.key = numeric_limits<SizeT>::lowest();
    val.val = numeric_limits<SizeT>::lowest();
    return val;
  }
};
