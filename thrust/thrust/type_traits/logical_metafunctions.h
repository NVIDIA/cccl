// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/negation.h>

THRUST_NAMESPACE_BEGIN

using ::cuda::std::conjunction;
using ::cuda::std::conjunction_v;
using ::cuda::std::disjunction;
using ::cuda::std::disjunction_v;
using ::cuda::std::negation;
using ::cuda::std::negation_v;

THRUST_NAMESPACE_END
