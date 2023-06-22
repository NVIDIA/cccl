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

#include <thrust/device_vector.h>

#include <c2h/custom_type.cuh>

namespace c2h
{

namespace detail
{

template <class T>
class value_wrapper_t
{
  T m_val{};

public:
  explicit value_wrapper_t(T val) : m_val(val) {}
  explicit value_wrapper_t(int val) : m_val(static_cast<T>(val)) {}
  T get() const { return m_val; }
};

}

class seed_t : public detail::value_wrapper_t<unsigned long long int> 
{
  using value_wrapper_t::value_wrapper_t;
};

class modulo_t : public detail::value_wrapper_t<std::size_t> 
{
  using value_wrapper_t::value_wrapper_t;
};

namespace detail
{
  
void gen(seed_t seed,
         char* data,
         c2h::custom_type_state_t min,
         c2h::custom_type_state_t max,
         std::size_t elements,
         std::size_t element_size);

}

template <template <typename> class... Ps>
void gen(
  seed_t seed,
  thrust::device_vector<c2h::custom_type_t<Ps...>> &data,
  c2h::custom_type_t<Ps...> min = std::numeric_limits<c2h::custom_type_t<Ps...>>::lowest(),
  c2h::custom_type_t<Ps...> max = std::numeric_limits<c2h::custom_type_t<Ps...>>::max())
{
  detail::gen(
      seed, 
      reinterpret_cast<char*>(thrust::raw_pointer_cast(data.data())),
      min,
      max,
      data.size(),
      sizeof(c2h::custom_type_t<Ps...>));
}

template <typename T>
void gen(seed_t seed,
         thrust::device_vector<T> &data,
         T min = std::numeric_limits<T>::min(),
         T max = std::numeric_limits<T>::max());

template <typename T>
void gen(modulo_t mod, thrust::device_vector<T> &data);

} // c2h

