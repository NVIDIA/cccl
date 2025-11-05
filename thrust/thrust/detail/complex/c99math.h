/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Copyright 2013 Filipe RNC Maia
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

#include <thrust/detail/complex/math_private.h>

#include <cuda/std/cmath>

#include <math.h>

THRUST_NAMESPACE_BEGIN
namespace detail::complex
{
// Define basic arithmetic functions so we can use them without explicit scope
// keeping the code as close as possible to FreeBSDs for ease of maintenance.
// It also provides an easy way to support compilers with missing C99 functions.
// When possible, just use the names in the global scope.
// Some platforms define these as macros, others as free functions.
// Avoid using the std:: form of these as nvcc may treat std::foo() as __host__ functions.

using ::cuda::std::acos;
using ::cuda::std::asin;
using ::cuda::std::atan;
using ::cuda::std::atanh;
using ::cuda::std::copysign;
using ::cuda::std::cos;
using ::cuda::std::cosh;
using ::cuda::std::exp;
using ::cuda::std::hypot;
using ::cuda::std::isfinite;
using ::cuda::std::isinf;
using ::cuda::std::isnan;
using ::cuda::std::log;
using ::cuda::std::log1p;
using ::cuda::std::signbit;
using ::cuda::std::sin;
using ::cuda::std::sinh;
using ::cuda::std::sqrt;
using ::cuda::std::tan;
} // namespace detail::complex

THRUST_NAMESPACE_END
