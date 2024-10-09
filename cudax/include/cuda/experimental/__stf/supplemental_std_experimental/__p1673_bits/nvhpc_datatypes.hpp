/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_NVHPC_DATATYPES_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_NVHPC_DATATYPES_HPP_

#include "exec_policy_wrapper_nvhpc.hpp"

namespace __nvhpc_std {

enum class __data_type {
    __Unknown,
    __Float16,
    __BFloat16,
    __Float32,
    __Float64,
    __Int8,
    __UInt8,
    __Int32,
    __ComplexFloat16,
    __ComplexBFloat16,
    __ComplexFloat32,
    __ComplexFloat64,
    __ComplexInt8,
    __ComplexUInt8,
    __ComplexInt32
};

template <typename>
constexpr __data_type __stdblas_output_data_type = __data_type::__Unknown;

#define __DEFINE_STDBLAS_TYPE_MAPPING(__type, __value)                                   \
    template <>                                                                          \
    constexpr __data_type __stdblas_output_data_type<__type> = __data_type::__##__value; \
    template <>                                                                          \
    constexpr __data_type __stdblas_output_data_type<std::complex<__type>> = __data_type::__Complex##__value

__DEFINE_STDBLAS_TYPE_MAPPING(float, Float32);
__DEFINE_STDBLAS_TYPE_MAPPING(double, Float64);
__DEFINE_STDBLAS_TYPE_MAPPING(signed char, Int8);
__DEFINE_STDBLAS_TYPE_MAPPING(unsigned char, UInt8);
#if __SIGNED_CHARS__
__DEFINE_STDBLAS_TYPE_MAPPING(char, Int8);
#else
__DEFINE_STDBLAS_TYPE_MAPPING(char, UInt8);
#endif
__DEFINE_STDBLAS_TYPE_MAPPING(std::byte, UInt8);
__DEFINE_STDBLAS_TYPE_MAPPING(int, Int32);
#if __SIZEOF_LONG__ == 4
__DEFINE_STDBLAS_TYPE_MAPPING(long, Int32);
#endif

#undef __DEFINE_STDBLAS_TYPE_MAPPING

template <typename _T>
constexpr __data_type __stdblas_data_type = __stdblas_output_data_type<std::remove_cv_t<_T>>;

// In general, both BLAS and cuBLAS support the case where all vectors/matrices
// are in the same type, which is one of float, double, std::complex<float>, or
// std::complex<double>. cuBLAS provides some APIs that support mixed-precision.

// TODO Add generic data type checks (for APIs that don't support mixed-precision)
constexpr bool __data_type_supported(__data_type __A_type) {
    return (__A_type == __data_type::__Float32 || __A_type == __data_type::__Float64 ||
            __A_type == __data_type::__ComplexFloat32 || __A_type == __data_type::__ComplexFloat64);
}

constexpr bool __data_types_supported(__data_type __A_type, __data_type __B_type) {
    return ((__A_type == __B_type) && __data_type_supported(__A_type));
}

constexpr bool __data_types_supported(__data_type __A_type, __data_type __B_type, __data_type __C_type) {
    return ((__A_type == __B_type) && (__A_type == __C_type) && __data_type_supported(__A_type));
}

constexpr bool __data_types_supported(
        __data_type __A_type, __data_type __B_type, __data_type __C_type, __data_type __D_type) {
    return ((__A_type == __B_type) && (__A_type == __C_type) && (__A_type == __D_type) &&
            __data_type_supported(__A_type));
}

template <class _ExecutionPolicy>
constexpr bool __gemm_data_types_supported(__data_type __A_type, __data_type __B_type, __data_type __C_type) {
    return (__data_types_supported(__A_type, __B_type, __C_type));
}

/**
 * cuBLAS cublasGemmEx supports mixed-precision where C is in one data type
 * while A and B are in a different data type (but both should be in the same
 * data type).
 * We currently support cases where
 * - C is in float, double, std::complex<float>, or std::complex<double>;
 * - When C is in float, A/B can be in either float or int8_t;
 * - When C is in std::complex<float>, A/B can be in either std::complex<float>
 *   or std::complex(int8_t)
 */
template <>
constexpr bool __gemm_data_types_supported<__nvhpc_exec<__cublas_exec_space<__nvhpc_sync>>>(
        __data_type __A_type, __data_type __B_type, __data_type __C_type) {
    switch (__C_type) {
    case __data_type::__Float32:
        return ((__A_type == __B_type) && (__A_type == __C_type || __A_type == __data_type::__Int8));
    case __data_type::__ComplexFloat32:
        return ((__A_type == __B_type) && (__B_type == __C_type || __B_type == __data_type::__ComplexInt8));
    case __data_type::__Float64:
    case __data_type::__ComplexFloat64: return (__A_type == __C_type && __B_type == __C_type);
    default: return false;
    }
}

}  // namespace __nvhpc_std

#endif
