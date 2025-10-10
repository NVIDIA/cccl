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

/*! \file philox_engine.h
 *  \brief A Philox counter-based pseudorandom number engine.
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

#include <thrust/random/detail/random_core_access.h>

#include <cuda/std/array>
#include <cuda/std/cstddef> // for size_t
#include <cuda/std/cstdint>

#include <array>
#include <iostream>
#include <limits>

THRUST_NAMESPACE_BEGIN

namespace random
{

/*! \addtogroup random_number_engine_templates Random Number Engine Class Templates
 *  \ingroup random
 *  \{
 */

/*! \class philox_engine
 *  \brief A \p philox_engine random number engine produces unsigned integer
 *         random numbers using a Philox counter-based random number generation algorithm
 *         as described in: Salmon, John K., et al. "Parallel random numbers: as easy as 1, 2, 3." Proceedings of 2011
 * international conference for high performance computing, networking, storage and analysis. 2011.
 *
 *
 *  \tparam UIntType The type of unsigned integer to produce.
 *  \tparam w The word size
 *  \tparam n The buffer size
 *  \tparam r The number of rounds
 *  \tparam consts The constants used in the generation algorithm.
 *
 *  \note Inexperienced users should not use this class template directly.  Instead, use
 *  \p philox4x32 or \p philox4x64 .
 *
 *  The following code snippet shows examples of use of a \p philox_engine instance:
 *
 *  \code
 *  #include <thrust/random/philox_engine.h>
 *  #include <iostream>
 *
 *  int main()
 *  {
 *    // create a philox4x64 object, which is an instance of philox_engine
 *    thrust::philox4x64 rng1;
 *    thrust::philox4x64 rng2;
 *    // Create two different streams of random numbers
 *    // The counter is set as a big integer with the least significant word last.
 *    // Each counter increment produces 4 new values
 *    rng1.set_counter({0, 0, 0, 0});
 *    rng2.set_counter({0, 0, 1, 0}); // rng2 is now 4*2^w values ahead of rng1
 *
 *    // Relation between discard and set_counter
 *    thrust::philox4x64 rng3;
 *    rng3.set_counter({0, 0, 0, 100});
 *    const int n = 4;
 *    rng1.discard(100*n); // rng1 is now at the same position as rng3
 *    std::cout << (rng1() == rng3()) << std::endl; // 1
 *
 *    return 0;
 *  }
 *
 *  \endcode
 *
 *  \see thrust::random::philox4x32
 *  \see thrust::random::philox4x64
 */
template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
class philox_engine
{
  static_assert(n == 2 || n == 4, "N argument must be either 2 or 4");
  static_assert(sizeof...(consts) == n, "consts array must be of length N");
  static_assert(0 < r, "rounds must be a natural number");
  static_assert((0 < w && w <= std::numeric_limits<UIntType>::digits),
                "Word size w must satisfy 0 < w <= numeric_limits<UIntType>::digits");

public:
  using result_type = UIntType;

  /*! The smallest value this engine may potentially produce.
   */
  static const result_type min = 0;
  /*! The largest value this engine may potentially produce.
   */
  static const result_type max = ((1ull << (w - 1)) | ((1ull << (w - 1)) - 1));

  /*! The default seed.
   */
  static constexpr result_type default_seed = 20111115u;

  /*! This constructor, which optionally accepts a seed, initializes a new
   *  \p philox_engine.
   *
   *  \param s The seed used to initialize this \p philox_engine's state.
   */
  _CCCL_HOST_DEVICE explicit philox_engine(result_type s = default_seed);

  /*! This method initializes this \p philox_engine's state, and optionally accepts
   *  a seed value.
   *
   *  \param s The seed used to initializes this \p philox_engine's state.
   */
  _CCCL_HOST_DEVICE void seed(result_type s = default_seed);

  /*! This method sets the internal counter. Each increment of the counter produces n new values. The array \p counter
   * can be thought of as a big integer. The n-1'th counter value is the least significant and the 0'th counter value is
   * the most significant. set_counter is related but distinct from discard:
   * - set_counter sets the engine's absolute position, while discard increments the engine.
   * - Each increment of the counter always produces n new values, while discard can increment by any number of values
   * equivalent to calling operator(). i.e. The sub-counter j is always set to n-1 after calling set_counter.
   * - set_counter exposes the full period of the engine as a big integer, while discard is limited by its word size
   * argument.
   *
   * set_counter is commonly used to initialize different streams of random numbers in parallel applications.
   * \code
   * Engine e1; // some philox_engine
   * Engine e2;
   * e1.set_counter({0, 0, 0, 100});
   * e2.set_counter({0, 0, 1, 100}); // e2 is now 4*2^w values ahead of e1
   * \endcode
   *
   *  \param counter The counter.
   */
  _CCCL_HOST_DEVICE void set_counter(const ::cuda::std::array<result_type, n>& counter);

  // generating functions

  /*! This member function produces a new random value and updates this \p philox_engine's state.
   *  \return A new random number.
   */
  _CCCL_HOST_DEVICE result_type operator()(void);

  /*! This member function advances this \p philox_engine's state a given number of times
   *  and discards the results. \p philox_engine is a counter-based engine, therefore can discard with O(1) complexity.
   *
   *  \param z The number of random values to discard.
   */
  _CCCL_HOST_DEVICE void discard(unsigned long long z);

  /*! \cond
   */

private:
  friend struct thrust::random::detail::random_core_access;

  _CCCL_HOST_DEVICE bool equal(const philox_engine& rhs) const;

  template <typename CharT, typename Traits>
  std::basic_ostream<CharT, Traits>& stream_out(std::basic_ostream<CharT, Traits>& os) const;

  template <typename CharT, typename Traits>
  std::basic_istream<CharT, Traits>& stream_in(std::basic_istream<CharT, Traits>& is);

  _CCCL_HOST_DEVICE void increment_counter();

  _CCCL_HOST_DEVICE void mulhilo(result_type a, result_type b, result_type& hi, result_type& lo) const;

  _CCCL_HOST_DEVICE void philox();

  // The counter X, a big integer stored as n w-bit words.
  // The least significant word is m_x[0].
  ::cuda::std::array<UIntType, n> m_x = {};
  // K is the "Key", storing the seed
  ::cuda::std::array<UIntType, n / 2> m_k = {};
  // The output buffer Y
  // Each time m_j reaches n, we generate n new values and store them in m_y.
  ::cuda::std::array<UIntType, n> m_y = {};
  // Each generation produces n random numbers, which are returned one at a time.
  // m_j cycles through [0, n-1].
  unsigned long long m_j = 0;

}; // end philox_engine

/*! This function checks two \p philox_engines for equality.
 *  \param lhs The first \p philox_engine to test.
 *  \param rhs The second \p philox_engine to test.
 *  \return \c true if \p lhs is equal to \p rhs; \c false, otherwise.
 */
template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE bool operator==(const philox_engine<UIntType, w, n, r, consts...>& lhs,
                                  const philox_engine<UIntType, w, n, r, consts...>& rhs);

/*! This function checks two \p philox_engines for inequality.
 *  \param lhs The first \p philox_engine to test.
 *  \param rhs The second \p philox_engine to test.
 *  \return \c true if \p lhs is not equal to \p rhs; \c false, otherwise.
 */
template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts>
_CCCL_HOST_DEVICE bool operator!=(const philox_engine<UIntType, w, n, r, consts...>& lhs,
                                  const philox_engine<UIntType, w, n, r, consts...>& rhs);

/*! This function streams a philox_engine to a \p std::basic_ostream.
 *  \param os The \p basic_ostream to stream out to.
 *  \param e The \p philox_engine to stream out.
 *  \return \p os
 */
template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts, typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, const philox_engine<UIntType, w, n, r, consts...>& e);

/*! This function streams a philox_engine in from a std::basic_istream.
 *  \param is The \p basic_istream to stream from.
 *  \param e The \p philox_engine to stream in.
 *  \return \p is
 */
template <typename UIntType, size_t w, size_t n, size_t r, UIntType... consts, typename CharT, typename Traits>
std::basic_istream<CharT, Traits>&
operator>>(std::basic_istream<CharT, Traits>& is, philox_engine<UIntType, w, n, r, consts...>& e);

} // namespace random

// import names into thrust::
using random::philox_engine;
THRUST_NAMESPACE_END

#include <thrust/random/detail/philox_engine.inl>
