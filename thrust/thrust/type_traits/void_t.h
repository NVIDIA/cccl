/*
 *  Copyright 2018-2021 NVIDIA Corporation
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

/*! \file
 *  \brief C++17's `void_t`.
 */

#pragma once

#include <thrust/detail/config.h>

_CCCL_IMPLICIT_SYSTEM_HEADER

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup type_traits Type Traits
 *  \{
 */

template <typename...>
struct THRUST_DEPRECATED_BECAUSE("Use ::cuda::std::void_t") voider
{
  using type = void;
};

template <typename... Ts>
using void_t THRUST_DEPRECATED_BECAUSE("Use ::cuda::std::void_t") = void;

/*! \} // type traits
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END
