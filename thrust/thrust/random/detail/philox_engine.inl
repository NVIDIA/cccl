/*
 *  Copyright 2025 NVIDIA Corporation
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

THRUST_NAMESPACE_BEGIN

namespace random
{
template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE philox_engine<UIntType, w, n, r, consts...>::philox_engine(result_type s)
{
  seed(s);
} 

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE void philox_engine<UIntType, w, n, r, consts...>::seed(result_type s)
{
    m_x = {};
    m_y = {};
    m_k = {};
    m_k[0] = s & max;
    m_j = n - 1
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
void
	philox_engine<UIntType, w, n, r, consts...>::set_counter(const array<result_type, n>& counter)
	{
	  for (size_t j = 0; j < n; ++j){
	    m_x[j] = counter[n - 1 - j] & max;
      }
	  m_j = n - 1;
	}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
void philox_engine<UIntType, w, n, r, consts...>::increment_counter() {
  // Increment the big integer m_x by 1, handling overflow.
  std::size_t i = 0;
  do {
    m_x[i] = (m_x[i] + 1) & max;
    ++i;
  } while (i < n && !m_x[i - 1]);
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
inline std::pair<typename philox_engine<UIntType, w, n, r, consts...>::result_type, typename philox_engine<UIntType, w, n, r, consts...>::result_type> philox_engine<UIntType, w, n, r, consts...>::mulhilo(result_type a, result_type b) {
    using upgraded_type = std::conditional_t<
        w <= 8, std::uint_fast16_t,
        std::conditional_t<
            w <= 16, std::uint_fast32_t,
            std::conditional_t<w <= 32, std::uint_fast64_t, uint128>>>;

    const upgraded_type ab =
        static_cast<upgraded_type>(a) * static_cast<upgraded_type>(b);
    return {static_cast<result_type>(ab >> w), static_cast<result_type>(ab) & this->max()};
}

template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
void philox_engine<UIntType, w, n, r, consts...>::philox() {
  // Only two variants are allowed, n=2 or n=4
  if constexpr(n == 2){
    result_type S0 = this->m_x[0];
    result_type S1 = this->m_x[1];
    result_type K0 = this->m_k[0];
    for (size_t j = 0; j < r; ++j) {
      auto [hi, lo] = this->mulhilo(S0, consts_arr[0]);
      S0 = hi ^ K0 ^ S1;
      S1 = lo;
      K0 = (K0 + consts_arr[1]) & in_mask;
    }
    this->m_y[0] = S0;
    this->m_y[1] = S1;
  }
  else if constexpr(n == 4){
    result_type S0 = this->m_x[0];
    result_type S1 = this->m_x[1];
    result_type S2 = this->m_x[2];
    result_type S3 = this->m_x[3];
    result_type K0 = this->m_k[0];
    result_type K1 = this->m_k[1];
    for (size_t j = 0; j < r; ++j) {
      result_type V0 = S2;
      result_type V1 = S1;
      result_type V2 = S0;
      result_type V3 = S3;

      auto [hi0, lo0] =
          this->mulhilo(V0, consts_arr[2]);
      auto [hi2, lo2] =
          this->mulhilo(V2, consts_arr[0]);

      S0 = hi0 ^ K0 ^ V1;
      S1 = lo0;

      S2 = hi2 ^ K1 ^ V3;

      S3 = lo2;

      K0 = (K0 + consts_arr[1]) & in_mask;
      K1 = (K1 + consts_arr[3]) & in_mask;
    }

    this->m_y[0] = S0;
    this->m_y[1] = S1;
    this->m_y[2] = S2;
    this->m_y[3] = S3;
  }

}


template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE typename philox_engine<UIntType, w, n, r, consts...>::result_type
philox_engine<UIntType, w, n, r, consts...>::operator()(void)
{
  m_j++;
  if (m_j == n){
    this->philox();
    this->increment_counter();
    m_j = 0;
  }
	return m_y[m_j];
}


} // namespace random

THRUST_NAMESPACE_END