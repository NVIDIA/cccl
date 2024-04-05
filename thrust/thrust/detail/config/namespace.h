/*
 *  Copyright 2021 NVIDIA Corporation
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

// Internal config header that is only included through thrust/detail/config/config.h

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config/device_system.h>
#include <thrust/version.h>

/**
 * \file namespace.h
 * \brief Utilities that allow `thrust::` to be placed inside an
 * application-specific namespace.
 */

/**
 * \def THRUST_CUB_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `thrust::` and `cub::` namespaces.
 * This macro should not be used with any other Thrust namespace macros.
 */
#ifdef THRUST_CUB_WRAPPED_NAMESPACE
#define THRUST_WRAPPED_NAMESPACE THRUST_CUB_WRAPPED_NAMESPACE
#endif

/**
 * \def THRUST_WRAPPED_NAMESPACE
 * If defined, this value will be used as the name of a namespace that wraps the
 * `thrust::` namespace.
 * If THRUST_CUB_WRAPPED_NAMESPACE is set, this will inherit that macro's value.
 * This macro should not be used with any other Thrust namespace macros.
 */
#ifdef THRUST_WRAPPED_NAMESPACE
#define THRUST_NS_PREFIX                                                       \
  namespace THRUST_WRAPPED_NAMESPACE                                           \
  {

#define THRUST_NS_POSTFIX }

#define THRUST_NS_QUALIFIER ::THRUST_WRAPPED_NAMESPACE::thrust
#endif

/**
 * \def THRUST_NS_PREFIX
 * This macro is inserted prior to all `namespace thrust { ... }` blocks. It is
 * derived from THRUST_WRAPPED_NAMESPACE, if set, and will be empty otherwise.
 * It may be defined by users, in which case THRUST_NS_PREFIX,
 * THRUST_NS_POSTFIX, and THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef THRUST_NS_PREFIX
#define THRUST_NS_PREFIX
#endif

/**
 * \def THRUST_NS_POSTFIX
 * This macro is inserted following the closing braces of all
 * `namespace thrust { ... }` block. It is defined appropriately when
 * THRUST_WRAPPED_NAMESPACE is set, and will be empty otherwise. It may be
 * defined by users, in which case THRUST_NS_PREFIX, THRUST_NS_POSTFIX, and
 * THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef THRUST_NS_POSTFIX
#define THRUST_NS_POSTFIX
#endif

/**
 * \def THRUST_NS_QUALIFIER
 * This macro is used to qualify members of thrust:: when accessing them from
 * outside of their namespace. By default, this is just `::thrust`, and will be
 * set appropriately when THRUST_WRAPPED_NAMESPACE is defined. This macro may be
 * defined by users, in which case THRUST_NS_PREFIX, THRUST_NS_POSTFIX, and
 * THRUST_NS_QUALIFIER must all be set consistently.
 */
#ifndef THRUST_NS_QUALIFIER
#define THRUST_NS_QUALIFIER ::thrust
#endif

// clang-format off
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  if !defined(THRUST_DETAIL_ABI_NS_NAME)
#    define THRUST_DETAIL_COUNT_N(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, \
                                  _14, _15, _16, _17, _18, _19, _20, N, ...)              \
                                  N
#    define THRUST_DETAIL_COUNT(...)                                                      \
      THRUST_DETAIL_IDENTITY(THRUST_DETAIL_COUNT_N(__VA_ARGS__, 20, 19, 18, 17, 16, 15, 14, 13, 12, \
                                                   11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1))
#    define THRUST_DETAIL_IDENTITY(N) N
#    define THRUST_DETAIL_APPLY(MACRO, ...) THRUST_DETAIL_IDENTITY(MACRO(__VA_ARGS__))
#    define THRUST_DETAIL_ABI_NS_NAME1(P1) \
        THRUST_##P1##_NS
#    define THRUST_DETAIL_ABI_NS_NAME2(P1, P2) \
        THRUST_##P1##_##P2##_NS
#    define THRUST_DETAIL_ABI_NS_NAME3(P1, P2, P3) \
        THRUST_##P1##_##P2##_##P3##_NS
#    define THRUST_DETAIL_ABI_NS_NAME4(P1, P2, P3, P4) \
        THRUST_##P1##_##P2##_##P3##_##P4##_NS
#    define THRUST_DETAIL_ABI_NS_NAME5(P1, P2, P3, P4, P5) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_NS
#    define THRUST_DETAIL_ABI_NS_NAME6(P1, P2, P3, P4, P5, P6) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_NS
#    define THRUST_DETAIL_ABI_NS_NAME7(P1, P2, P3, P4, P5, P6, P7) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_NS
#    define THRUST_DETAIL_ABI_NS_NAME8(P1, P2, P3, P4, P5, P6, P7, P8) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_NS
#    define THRUST_DETAIL_ABI_NS_NAME9(P1, P2, P3, P4, P5, P6, P7, P8, P9) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_NS
#    define THRUST_DETAIL_ABI_NS_NAME10(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_NS
#    define THRUST_DETAIL_ABI_NS_NAME11(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_NS
#    define THRUST_DETAIL_ABI_NS_NAME12(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_NS
#    define THRUST_DETAIL_ABI_NS_NAME13(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_NS
#    define THRUST_DETAIL_ABI_NS_NAME14(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_NS
#    define THRUST_DETAIL_ABI_NS_NAME15(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_NS
#    define THRUST_DETAIL_ABI_NS_NAME16(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_##P16##_NS
#    define THRUST_DETAIL_ABI_NS_NAME17(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_##P16##_##P17##_NS
#    define THRUST_DETAIL_ABI_NS_NAME18(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_##P16##_##P17##_##P18##_NS
#    define THRUST_DETAIL_ABI_NS_NAME19(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_##P16##_##P17##_##P18##_##P19##_NS
#    define THRUST_DETAIL_ABI_NS_NAME20(P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15, P16, P17, P18, P19, P20) \
        THRUST_##P1##_##P2##_##P3##_##P4##_##P5##_##P6##_##P7##_##P8##_##P9##_##P10##_##P11##_##P12##_##P13##_##P14##_##P15##_##P16##_##P17##_##P18##_##P19##_##P20##_NS
#    define THRUST_DETAIL_DISPATCH(N) THRUST_DETAIL_ABI_NS_NAME ## N
#    define THRUST_DETAIL_ABI_NS_NAME(...) THRUST_DETAIL_IDENTITY(THRUST_DETAIL_APPLY(THRUST_DETAIL_DISPATCH, THRUST_DETAIL_COUNT(__VA_ARGS__))(__VA_ARGS__))
#  endif // !defined(THRUST_DETAIL_ABI_NS_NAME)

#  if defined(THRUST_DISABLE_ABI_NAMESPACE) || defined(THRUST_WRAPPED_NAMESPACE)
#    if !defined(THRUST_WRAPPED_NAMESPACE)
#      if !defined(THRUST_IGNORE_ABI_NAMESPACE_ERROR)
#        error "Disabling ABI namespace is unsafe without wrapping namespace"
#      endif // !defined(THRUST_IGNORE_ABI_NAMESPACE_ERROR)
#    endif // !defined(THRUST_WRAPPED_NAMESPACE)
#    define THRUST_DETAIL_ABI_NS_BEGIN
#    define THRUST_DETAIL_ABI_NS_END
#  else // not defined(THRUST_DISABLE_ABI_NAMESPACE)
#    if defined(_NVHPC_CUDA)
#      define THRUST_DETAIL_ABI_NS_BEGIN inline namespace THRUST_DETAIL_ABI_NS_NAME(THRUST_VERSION, NV_TARGET_SM_INTEGER_LIST) {
#      define THRUST_DETAIL_ABI_NS_END }
#    else // not defined(_NVHPC_CUDA)
#      define THRUST_DETAIL_ABI_NS_BEGIN inline namespace THRUST_DETAIL_ABI_NS_NAME(THRUST_VERSION, __CUDA_ARCH_LIST__) {
#      define THRUST_DETAIL_ABI_NS_END }
#    endif // not defined(_NVHPC_CUDA)
#  endif // not defined(THRUST_DISABLE_ABI_NAMESPACE)
#else // THRUST_DEVICE_SYSTEM != THRUST_DEVICE_SYSTEM_CUDA
#  define THRUST_DETAIL_ABI_NS_BEGIN
#  define THRUST_DETAIL_ABI_NS_END
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
// clang-format on

/**
 * \def THRUST_NAMESPACE_BEGIN
 * This macro is used to open a `thrust::` namespace block, along with any
 * enclosing namespaces requested by THRUST_WRAPPED_NAMESPACE, etc.
 * This macro is defined by Thrust and may not be overridden.
 */
#define THRUST_NAMESPACE_BEGIN                                                 \
  THRUST_NS_PREFIX                                                             \
  namespace thrust                                                             \
  {                                                                            \
  THRUST_DETAIL_ABI_NS_BEGIN

/**
 * \def THRUST_NAMESPACE_END
 * This macro is used to close a `thrust::` namespace block, along with any
 * enclosing namespaces requested by THRUST_WRAPPED_NAMESPACE, etc.
 * This macro is defined by Thrust and may not be overridden.
 */
#define THRUST_NAMESPACE_END                                                   \
  THRUST_DETAIL_ABI_NS_END                                                     \
  } /* end namespace thrust */                                                 \
  THRUST_NS_POSTFIX

// The following is just here to add docs for the thrust namespace:

THRUST_NS_PREFIX

/*! \namespace thrust
 *  \brief \p thrust is the top-level namespace which contains all Thrust
 *         functions and types.
 */
namespace thrust
{
}

THRUST_NS_POSTFIX
