/******************************************************************************
* Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the NVIDIA CORPORATION nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************/

#pragma once

#include <limits>
#include <memory>
#include <ostream>

namespace c2h
{

struct custom_type_state_t
{
  std::size_t key{};
  std::size_t val{};
};

template <template<typename> class... Policies>
class custom_type_t : public custom_type_state_t
                    , public Policies<custom_type_t<Policies...>>...
{

public:
  friend __host__ std::ostream &operator<<(std::ostream &os,
                                           const custom_type_t &self)
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
  char workaround_msvc;

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
  char workaround_msvc;

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
  char workaround_msvc;

public:
  friend __host__ __device__ bool operator<(const CustomType &lhs, const CustomType &rhs)
  {
    return lhs.key == rhs.key ? lhs.val < rhs.val : lhs.key < rhs.key;
  }
};

template <class CustomType>
class lexicographical_greater_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc;

public:
  friend __host__ __device__ bool operator>(const CustomType &lhs, const CustomType &rhs)
  {
    return lhs.key == rhs.key ? lhs.val > rhs.val : lhs.key > rhs.key;
  }
};

template <class CustomType>
class equal_comparable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc;

public:
  friend __host__ __device__ bool operator==(const CustomType& lhs, const CustomType& rhs)
  {
    return lhs.key == rhs.key &&
           lhs.val == rhs.val;
  }
};

template <class CustomType>
class subtractable_t
{
  // The CUDA compiler follows the IA64 ABI for class layout, while the
  // Microsoft host compiler does not.
  char workaround_msvc;

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
  char workaround_msvc;

public:
  friend __host__ __device__ CustomType operator+(const CustomType& lhs, const CustomType& rhs)
  {
    CustomType result{};

    result.key = lhs.key + rhs.key;
    result.val = lhs.val + rhs.val;

    return result;
  }
};

} // c2h

namespace std {
  template<template<typename> class... Policies>
  class numeric_limits<c2h::custom_type_t<Policies...>>
  {
  public:
     static c2h::custom_type_t<Policies...> max()
     {
       c2h::custom_type_t<Policies...> val;
       val.key = std::numeric_limits<std::size_t>::max();
       val.val = std::numeric_limits<std::size_t>::max();
       return val;
     }

     static c2h::custom_type_t<Policies...> min()
     {
       c2h::custom_type_t<Policies...> val;
       val.key = std::numeric_limits<std::size_t>::min();
       val.val = std::numeric_limits<std::size_t>::min();
       return val;
     }

     static c2h::custom_type_t<Policies...> lowest()
     {
       c2h::custom_type_t<Policies...> val;
       val.key = std::numeric_limits<std::size_t>::lowest();
       val.val = std::numeric_limits<std::size_t>::lowest();
       return val;
     }
  };
}
