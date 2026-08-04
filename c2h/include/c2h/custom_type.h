// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/std/limits>

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

template <std::size_t TotalSize>
struct huge_data
{
  template <class CustomType>
  class type
  {
    static constexpr auto extra_member_bytes = (TotalSize - sizeof(custom_type_state_t));
    std::uint8_t data[extra_member_bytes];
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
    c2h::custom_type_t<Policies...> val;
    val.key = numeric_limits<SizeT>::max();
    val.val = numeric_limits<SizeT>::max();
    return val;
  }

  template <class SizeT = std::size_t>
  static __host__ __device__ c2h::custom_type_t<Policies...> min()
  {
    c2h::custom_type_t<Policies...> val;
    val.key = numeric_limits<SizeT>::min();
    val.val = numeric_limits<SizeT>::min();
    return val;
  }

  template <class SizeT = std::size_t>
  static __host__ __device__ c2h::custom_type_t<Policies...> lowest()
  {
    c2h::custom_type_t<Policies...> val;
    val.key = numeric_limits<SizeT>::lowest();
    val.val = numeric_limits<SizeT>::lowest();
    return val;
  }
};
