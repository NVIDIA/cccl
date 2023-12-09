/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS std::int32_tERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <iostream>

namespace c2h
{

namespace detail
{

template <std::int32_t ElementsPerObject_ = 128>
struct huge_data_type_t
{
  static constexpr std::int32_t ElementsPerObject = ElementsPerObject_;

  __device__ __host__ huge_data_type_t()
  {
    for (std::int32_t i = 0; i < ElementsPerObject; i++)
    {
      data[i] = 0;
    }
  }

  __device__ __host__ huge_data_type_t(const huge_data_type_t& rhs)
  {
    for (std::int32_t i = 0; i < ElementsPerObject; i++)
    {
      data[i] = rhs.data[i];
    }
  }

  explicit __device__ __host__ huge_data_type_t(std::int32_t val)
  {
    for (std::int32_t i = 0; i < ElementsPerObject; i++)
    {
      data[i] = val;
    }
  }

  __device__ __host__ huge_data_type_t& operator=(const huge_data_type_t& rhs)
  {
    if (this != &rhs)
    {
      for (std::int32_t i = 0; i < ElementsPerObject; i++)
      {
        data[i] = rhs.data[i];
      }
    }
    return *this;
  }

  std::int32_t data[ElementsPerObject];
};

template <std::int32_t ElementsPerObject>
inline __device__ __host__ bool
operator==(const huge_data_type_t<ElementsPerObject>& lhs, const huge_data_type_t<ElementsPerObject>& rhs)
{
  for (std::int32_t i = 0; i < ElementsPerObject; i++)
  {
    if (lhs.data[i] != rhs.data[i])
    {
      return false;
    }
  }

  return true;
}

template <std::int32_t ElementsPerObject>
inline __device__ __host__ bool
operator<(const huge_data_type_t<ElementsPerObject>& lhs, const huge_data_type_t<ElementsPerObject>& rhs)
{
  for (std::int32_t i = 0; i < ElementsPerObject; i++)
  {
    if (lhs.data[i] < rhs.data[i])
    {
      return true;
    }
  }

  return false;
}

template <typename DataType, std::int32_t ElementsPerObject>
__device__ __host__ bool operator!=(const huge_data_type_t<ElementsPerObject>& lhs, const DataType& rhs)
{
  for (std::int32_t i = 0; i < ElementsPerObject; i++)
  {
    if (lhs.data[i] != rhs)
    {
      return true;
    }
  }

  return false;
}

template <std::int32_t ElementsPerObject>
std::ostream& operator<<(std::ostream& os, const huge_data_type_t<ElementsPerObject>& val)
{
  os << '(';
  for (std::int32_t i = 0; i < ElementsPerObject; i++)
  {
    os << val.data[i];
    if (i < ElementsPerObject - 1)
    {
      os << ',';
    }
  }
  os << ')';
  return os;
}

} // namespace detail
} // namespace c2h
