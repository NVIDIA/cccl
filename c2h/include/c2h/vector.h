/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <thrust/detail/vector_base.h>

#include <memory>

#include <catch2/catch_tostring.hpp>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <c2h/checked_allocator.cuh>
#else
#  include <thrust/device_vector.h>
#  include <thrust/host_vector.h>
#endif

namespace c2h
{
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template <typename T>
using host_vector = THRUST_NS_QUALIFIER::detail::vector_base<T, c2h::checked_host_allocator<T>>;

template <typename T>
using device_vector = THRUST_NS_QUALIFIER::detail::vector_base<T, c2h::checked_cuda_allocator<T>>;
#else // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
using THRUST_NS_QUALIFIER::device_vector;
using THRUST_NS_QUALIFIER::host_vector;
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
} // namespace c2h

// We specialize how Catch2 prints ([signed|unsigned]) char vectors for better readability. Let's print them as numbers
// instead of characters. We need a full specialization here to avoid an ambiguity, since Catch2 already provides a
// partial specialization for range-like types.
#define _SPECIALIZE_VEC(V, T)                                              \
  template <>                                                              \
  struct ::Catch::StringMaker<c2h::V<T>>                                   \
  {                                                                        \
    /* Copied from `rangeToString` in catch_tostring.hpp */                \
    static auto convert(const c2h::V<T>& v) -> std::string                 \
    {                                                                      \
      auto first = v.begin();                                              \
      auto last  = v.end();                                                \
                                                                           \
      ReusableStringStream rss;                                            \
      rss << "{ ";                                                         \
      if (first != last)                                                   \
      {                                                                    \
        rss << Detail::stringify(static_cast<unsigned>(*first));           \
        for (++first; first != last; ++first)                              \
        {                                                                  \
          rss << ", " << Detail::stringify(static_cast<unsigned>(*first)); \
        }                                                                  \
      }                                                                    \
      rss << " }";                                                         \
      return rss.str();                                                    \
    }                                                                      \
  };
_SPECIALIZE_VEC(host_vector, char)
_SPECIALIZE_VEC(host_vector, signed char)
_SPECIALIZE_VEC(host_vector, unsigned char)
_SPECIALIZE_VEC(device_vector, char)
_SPECIALIZE_VEC(device_vector, signed char)
_SPECIALIZE_VEC(device_vector, unsigned char)
#undef _SPECIALIZE_VEC
