/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file random.h
 *  \brief Pseudo-random number generators.
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
#include <cuda/std/cstdint>

// RNGs
#include <thrust/random/discard_block_engine.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/linear_feedback_shift_engine.h>
#include <thrust/random/subtract_with_carry_engine.h>
#include <thrust/random/xor_combine_engine.h>

// distributions
#include <thrust/random/normal_distribution.h>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/uniform_real_distribution.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup random Random Number Generation
 *  \{
 */

/*! \namespace thrust::random
 *  \brief \p thrust::random is the namespace which contains random number engine class templates,
 *  random number engine adaptor class templates, engines with predefined parameters,
 *  and random number distribution class templates. They are provided in a separate namespace
 *  for import convenience but are also aliased in the top-level \p thrust namespace for
 *  easy access.
 */
namespace random
{
/*! \addtogroup predefined_random Random Number Engines with Predefined Parameters
 *  \ingroup random
 *  \{
 */

/*! \typedef ranlux24
 *  \brief A random number engine with predefined parameters which implements the
 *         RANLUX level-3 random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p ranlux24
 *        shall produce the value \c 9901578 .
 */
using ranlux24 = discard_block_engine<ranlux24_base, 223, 23>;

/*! \typedef ranlux48
 *  \brief A random number engine with predefined parameters which implements the
 *         RANLUX level-4 random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p ranlux48
 *        shall produce the value \c 88229545517833 .
 */
using ranlux48 = discard_block_engine<ranlux48_base, 389, 11>;

/*! \typedef taus88
 *  \brief A random number engine with predefined parameters which implements
 *         L'Ecuyer's 1996 three-component Tausworthe random number generator.
 *
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p taus88
 *        shall produce the value \c 3535848941 .
 */
using taus88 =
  xor_combine_engine<linear_feedback_shift_engine<std::uint32_t, 32U, 31U, 13U, 12U>,
                     0,
                     xor_combine_engine<linear_feedback_shift_engine<std::uint32_t, 32U, 29U, 2U, 4U>,
                                        0,
                                        linear_feedback_shift_engine<std::uint32_t, 32U, 28U, 3U, 17U>,
                                        0>,
                     0>;

/*! \typedef default_random_engine
 *  \brief An implementation-defined "default" random number engine.
 *  \note \p default_random_engine is currently an alias for \p minstd_rand, and may change
 *        in a future version.
 */
using default_random_engine = minstd_rand;

/*! \} // end predefined_random
 */
} // namespace random

/*! \} // end random
 */

// import names into thrust::
using random::default_random_engine;
using random::ranlux24;
using random::ranlux48;
using random::taus88;

THRUST_NAMESPACE_END
