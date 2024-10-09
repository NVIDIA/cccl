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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_CUBLAS_WRAPPER_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_CUBLAS_WRAPPER_HPP_

#include "cuComplex.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"

#include <complex>
#include <string>
#include <string_view>

namespace __cublas_std {

namespace __ex = std::experimental;

inline thread_local cublasHandle_t __handle = nullptr;
inline thread_local cudaStream_t __stream = nullptr;
inline thread_local float* __dummy = nullptr;

struct __handle_cleanup {
    bool __handle_created;

    ~__handle_cleanup() {
        if (__handle != nullptr) {
            cudaFreeAsync(__dummy, __stream);

            cublasDestroy(__handle);
            __handle = nullptr;
        }
    }
};

inline thread_local __handle_cleanup __cleaner;

inline cublasHandle_t __get_cublas_handle() {
    if (__handle == nullptr) {
        cublasStatus_t __stat = cublasCreate(&__handle);

        if (__stat != CUBLAS_STATUS_SUCCESS) {
            throw std::system_error(__ex::make_error_code(__stat), "cublasCreate failed");
        }

        __cleaner.__handle_created = true;

        cublasGetStream(__handle, &__stream);

        // Pre-allocate some space so that the pool allocator doesn't return memory to the OS
        cudaMallocAsync((void**) &__dummy, 8, __stream);
    }

    return __handle;
}

template <class _Exec>
inline void __synchronize(_Exec&&) {
}

inline void __synchronize(__nvhpc_std::__nvhpc_exec<__nvhpc_std::__cublas_exec_space<__nvhpc_std::__nvhpc_sync>>&&) {
    cudaError_t __error(cudaStreamSynchronize(__stream));

    if (__error != cudaSuccess) {
        throw std::system_error(__ex::make_error_code(__error), "CUDA call to cudaStreamSynchronize failed");
    }
}

inline void __check_cublas_status(
        cublasStatus_t __stat, std::string_view __function_name, std::string_view __cublas_api_name) {
    if (__stat != CUBLAS_STATUS_SUCCESS) {
        throw std::system_error(__ex::make_error_code(__stat),
                std::string("cuBLAS call to '") + std::string(__cublas_api_name) + std::string("' in function '") +
                        std::string(__function_name) + std::string("' failed"));
    }
}

template <typename>
constexpr cudaDataType_t __cuda_data_type = cudaDataType_t(-1);

#define __DEFINE_CUDA_TYPE_MAPPING(__type, __value)                       \
    template <>                                                           \
    constexpr cudaDataType_t __cuda_data_type<__type> = CUDA_R_##__value; \
    template <>                                                           \
    constexpr cudaDataType_t __cuda_data_type<std::complex<__type>> = CUDA_C_##__value

__DEFINE_CUDA_TYPE_MAPPING(float, 32F);
__DEFINE_CUDA_TYPE_MAPPING(signed char, 8I);
#if __SIGNED_CHARS__
__DEFINE_CUDA_TYPE_MAPPING(char, 8I);
#else
__DEFINE_CUDA_TYPE_MAPPING(char, 8U);
#endif

#undef __DEFINE_CUDA_TYPE_MAPPING

template <typename>
constexpr cublasComputeType_t __cublas_compute_type = cublasComputeType_t(-1);

template <>
constexpr cublasComputeType_t __cublas_compute_type<float> = CUBLAS_COMPUTE_32F;

// BLAS Level 1

// Copy

template <class _VectorValType>
cublasStatus_t __cublas_copy(cublasHandle_t __handle, int __n, _VectorValType const* __x, _VectorValType* __y);

template <>
cublasStatus_t __cublas_copy(cublasHandle_t __handle, int __n, float const* __x, float* __y) {
    return cublasScopy(__handle, __n, __x, 1, __y, 1);
}

template <>
cublasStatus_t __cublas_copy(cublasHandle_t __handle, int __n, double const* __x, double* __y) {
    return cublasDcopy(__handle, __n, __x, 1, __y, 1);
}

template <>
cublasStatus_t __cublas_copy(
        cublasHandle_t __handle, int __n, std::complex<float> const* __x, std::complex<float>* __y) {
    return cublasCcopy(__handle, __n, (cuComplex const*) __x, 1, (cuComplex*) __y, 1);
}

template <>
cublasStatus_t __cublas_copy(
        cublasHandle_t __handle, int __n, std::complex<double> const* __x, std::complex<double>* __y) {
    return cublasZcopy(__handle, __n, (cuDoubleComplex const*) __x, 1, (cuDoubleComplex*) __y, 1);
}

// Axpy

template <class _Scalar, class _VectorValType>
cublasStatus_t __cublas_axpy(
        cublasHandle_t __handle, int __n, _Scalar const* __alpha, _VectorValType const* __x, _VectorValType* __y);

template <>
cublasStatus_t __cublas_axpy(cublasHandle_t __handle, int __n, float const* __alpha, float const* __x, float* __y) {
    return cublasSaxpy(__handle, __n, __alpha, __x, 1, __y, 1);
}

template <>
cublasStatus_t __cublas_axpy(cublasHandle_t __handle, int __n, double const* __alpha, double const* __x, double* __y) {
    return cublasDaxpy(__handle, __n, __alpha, __x, 1, __y, 1);
}

template <>
cublasStatus_t __cublas_axpy(cublasHandle_t __handle, int __n, std::complex<float> const* __alpha,
        std::complex<float> const* __x, std::complex<float>* __y) {
    return cublasCaxpy(__handle, __n, (cuComplex const*) __alpha, (cuComplex const*) __x, 1, (cuComplex*) __y, 1);
}

template <>
cublasStatus_t __cublas_axpy(cublasHandle_t __handle, int __n, std::complex<double> const* __alpha,
        std::complex<double> const* __x, std::complex<double>* __y) {
    return cublasZaxpy(__handle, __n, (cuDoubleComplex const*) __alpha, (cuDoubleComplex const*) __x, 1,
            (cuDoubleComplex*) __y, 1);
}

// Dot

template <class _Scalar, class _VectorValType>
cublasStatus_t __cublas_dot(
        cublasHandle_t __handle, int __n, _VectorValType const* __x, _VectorValType const* __y, _Scalar* __result);

template <>
cublasStatus_t __cublas_dot(cublasHandle_t __handle, int __n, float const* __x, float const* __y, float* __result) {
    return cublasSdot(__handle, __n, __x, 1, __y, 1, __result);
}

template <>
cublasStatus_t __cublas_dot(cublasHandle_t __handle, int __n, double const* __x, double const* __y, double* __result) {
    return cublasDdot(__handle, __n, __x, 1, __y, 1, __result);
}

template <>
cublasStatus_t __cublas_dot(cublasHandle_t __handle, int __n, std::complex<float> const* __x,
        std::complex<float> const* __y, std::complex<float>* __result) {
    return cublasCdotu(__handle, __n, (cuComplex*) __x, 1, (cuComplex*) __y, 1, (cuComplex*) __result);
}

template <>
cublasStatus_t __cublas_dot(cublasHandle_t __handle, int __n, std::complex<double> const* __x,
        std::complex<double> const* __y, std::complex<double>* __result) {
    return cublasZdotu(
            __handle, __n, (cuDoubleComplex*) __x, 1, (cuDoubleComplex*) __y, 1, (cuDoubleComplex*) __result);
}

template <class _Scalar, class _VectorValType>
cublasStatus_t __cublas_dotc(
        cublasHandle_t __handle, int __n, _VectorValType const* __x, _VectorValType const* __y, _Scalar* __result);

template <>
cublasStatus_t __cublas_dotc(cublasHandle_t __handle, int __n, std::complex<float> const* __x,
        std::complex<float> const* __y, std::complex<float>* __result) {
    return cublasCdotc(__handle, __n, (cuComplex*) __x, 1, (cuComplex*) __y, 1, (cuComplex*) __result);
}

template <>
cublasStatus_t __cublas_dotc(cublasHandle_t __handle, int __n, std::complex<double> const* __x,
        std::complex<double> const* __y, std::complex<double>* __result) {
    return cublasZdotc(
            __handle, __n, (cuDoubleComplex*) __x, 1, (cuDoubleComplex*) __y, 1, (cuDoubleComplex*) __result);
}

// Scale

template <class _Scalar, class _VectorValType>
cublasStatus_t __cublas_scal(cublasHandle_t __handle, int __n, _Scalar __alpha, _VectorValType* __data);

template <>
cublasStatus_t __cublas_scal(cublasHandle_t __handle, int __n, float __alpha, float* __data) {
    return cublasSscal(__handle, __n, &__alpha, __data, 1);
}

template <>
cublasStatus_t __cublas_scal(cublasHandle_t __handle, int __n, double __alpha, double* __data) {
    return cublasDscal(__handle, __n, &__alpha, __data, 1);
}

template <>
cublasStatus_t __cublas_scal(
        cublasHandle_t __handle, int __n, std::complex<float> __alpha, std::complex<float>* __data) {
    return cublasCscal(__handle, __n, (cuComplex*) &__alpha, (cuComplex*) __data, 1);
}

template <>
cublasStatus_t __cublas_scal(
        cublasHandle_t __handle, int __n, std::complex<double> __alpha, std::complex<double>* __data) {
    return cublasZscal(__handle, __n, (cuDoubleComplex*) &__alpha, (cuDoubleComplex*) __data, 1);
}

// Calls cublas<T>scal to perform element-wise conjugate

// TODO remove these when all functions are updated to do out-of-place conj
template <typename _T>
cublasStatus_t __cublas_conj(cublasHandle_t __handle, int __n, _T* __data) {
    return CUBLAS_STATUS_SUCCESS;
}

template <typename _T>
cublasStatus_t __cublas_conj(cublasHandle_t __handle, int __n, std::complex<_T>* __data);

template <>
cublasStatus_t __cublas_conj(cublasHandle_t __handle, int __n, std::complex<float>* __data) {
    float const __alpha = -1.0f;
    cublasStatus_t __stat = cublasSscal(__handle, __n, &__alpha, reinterpret_cast<float*>(__data) + 1, 2);
    return __stat;
}

template <>
cublasStatus_t __cublas_conj(cublasHandle_t __handle, int __n, std::complex<double>* __data) {
    double const __alpha = -1.0;
    cublasStatus_t __stat = cublasDscal(__handle, __n, &__alpha, reinterpret_cast<double*>(__data) + 1, 2);
    return __stat;
}

template <typename _T>
cublasStatus_t __cublas_conj(cublasHandle_t __handle, int __n, _T const* __in, _T* __out) {
    cublasStatus_t __stat = __cublas_copy(__handle, __n, __in, __out);
    return __stat;
}

template <typename _T>
cublasStatus_t __cublas_conj(cublasHandle_t __handle, int __n, std::complex<_T> const* __in, std::complex<_T>* __out);

template <>
cublasStatus_t __cublas_conj(
        cublasHandle_t __handle, int __n, std::complex<float> const* __in, std::complex<float>* __out) {
    cublasStatus_t __stat = cublasCcopy(__handle, __n, (cuComplex const*) __in, 1, (cuComplex*) __out, 1);
    if (__stat != CUBLAS_STATUS_SUCCESS) {
        return __stat;
    }

    float const __alpha = -1.0f;
    __stat = cublasSscal(__handle, __n, &__alpha, reinterpret_cast<float*>(__out) + 1, 2);
    return __stat;
}

template <>
cublasStatus_t __cublas_conj(
        cublasHandle_t __handle, int __n, std::complex<double> const* __in, std::complex<double>* __out) {
    cublasStatus_t __stat = cublasZcopy(__handle, __n, (cuDoubleComplex const*) __in, 1, (cuDoubleComplex*) __out, 1);
    if (__stat != CUBLAS_STATUS_SUCCESS) {
        return __stat;
    }

    double const __alpha = -1.0f;
    __stat = cublasDscal(__handle, __n, &__alpha, reinterpret_cast<double*>(__out) + 1, 2);
    return __stat;
}

// nrm2

template <class _Scalar, class _VectorValType>
cublasStatus_t __cublas_nrm2(cublasHandle_t __handle, int __n, _VectorValType const* __x, _Scalar* __result);

template <>
cublasStatus_t __cublas_nrm2(cublasHandle_t __handle, int __n, float const* __x, float* __result) {
    return cublasSnrm2(__handle, __n, __x, 1, __result);
}

template <>
cublasStatus_t __cublas_nrm2(cublasHandle_t __handle, int __n, double const* __x, double* __result) {
    return cublasDnrm2(__handle, __n, __x, 1, __result);
}
template <>
cublasStatus_t __cublas_nrm2(cublasHandle_t __handle, int __n, std::complex<float> const* __x, float* __result) {
    return cublasScnrm2(__handle, __n, (cuComplex*) __x, 1, __result);
}
template <>
cublasStatus_t __cublas_nrm2(cublasHandle_t __handle, int __n, std::complex<double> const* __x, double* __result) {
    return cublasDznrm2(__handle, __n, (cuDoubleComplex*) __x, 1, __result);
}

// BLAS Level 2

// GEMV

template <class _Scalar, class _VectorValType, class _MatrixValType>
cublasStatus_t __cublas_gemv(cublasHandle_t __handle, cublasOperation_t __trans, int __m, int __n,
        _Scalar const* __alpha, _MatrixValType const* __A, int __lda, _VectorValType const* __x, _Scalar const* __beta,
        _VectorValType* __y);

template <>
cublasStatus_t __cublas_gemv(cublasHandle_t __handle, cublasOperation_t __trans, int __m, int __n, float const* __alpha,
        float const* __A, int __lda, float const* __x, float const* __beta, float* __y) {
    return cublasSgemv(__handle, __trans, __m, __n, __alpha, __A, __lda, __x, 1, __beta, __y, 1);
}

template <>
cublasStatus_t __cublas_gemv(cublasHandle_t __handle, cublasOperation_t __trans, int __m, int __n,
        double const* __alpha, double const* __A, int __lda, double const* __x, double const* __beta, double* __y) {
    return cublasDgemv(__handle, __trans, __m, __n, __alpha, __A, __lda, __x, 1, __beta, __y, 1);
}

template <>
cublasStatus_t __cublas_gemv(cublasHandle_t __handle, cublasOperation_t __trans, int __m, int __n,
        std::complex<float> const* __alpha, std::complex<float> const* __A, int __lda, std::complex<float> const* __x,
        std::complex<float> const* __beta, std::complex<float>* __y) {
    return cublasCgemv(__handle, __trans, __m, __n, (cuComplex const*) __alpha, (cuComplex const*) __A, __lda,
            (cuComplex const*) __x, 1, (cuComplex const*) __beta, (cuComplex*) __y, 1);
}

template <>
cublasStatus_t __cublas_gemv(cublasHandle_t __handle, cublasOperation_t __trans, int __m, int __n,
        std::complex<double> const* __alpha, std::complex<double> const* __A, int __lda,
        std::complex<double> const* __x, std::complex<double> const* __beta, std::complex<double>* __y) {
    return cublasZgemv(__handle, __trans, __m, __n, (cuDoubleComplex const*) __alpha, (cuDoubleComplex const*) __A,
            __lda, (cuDoubleComplex const*) __x, 1, (cuDoubleComplex const*) __beta, (cuDoubleComplex*) __y, 1);
}

// TRSV

template <class _VectorValType, class _MatrixValType>
cublasStatus_t __cublas_trsv(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans,
        cublasDiagType_t __diag, int __n, _MatrixValType const* __A, int __lda, _VectorValType* __x);

template <>
cublasStatus_t __cublas_trsv(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans,
        cublasDiagType_t __diag, int __n, float const* __A, int __lda, float* __x) {
    return cublasStrsv(__handle, __uplo, __trans, __diag, __n, __A, __lda, __x, 1);
}

template <>
cublasStatus_t __cublas_trsv(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans,
        cublasDiagType_t __diag, int __n, double const* __A, int __lda, double* __x) {
    return cublasDtrsv(__handle, __uplo, __trans, __diag, __n, __A, __lda, __x, 1);
}

template <>
cublasStatus_t __cublas_trsv(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans,
        cublasDiagType_t __diag, int __n, std::complex<float> const* __A, int __lda, std::complex<float>* __x) {
    return cublasCtrsv(__handle, __uplo, __trans, __diag, __n, (cuComplex*) __A, __lda, (cuComplex*) __x, 1);
}

template <>
cublasStatus_t __cublas_trsv(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans,
        cublasDiagType_t __diag, int __n, std::complex<double> const* __A, int __lda, std::complex<double>* __x) {
    return cublasZtrsv(
            __handle, __uplo, __trans, __diag, __n, (cuDoubleComplex*) __A, __lda, (cuDoubleComplex*) __x, 1);
}

// BLAS Level 3

// GEMM

template <class _Scalar, class _A_t, class _B_t, class _C_t>
cublasStatus_t __cublas_gemm(cublasHandle_t __handle, cublasOperation_t __trans_A, cublasOperation_t __trans_B, int __m,
        int __n, int __k, _Scalar const* __alpha, _A_t const* __A, int __lda, _B_t const* __B, int __ldb,
        _Scalar const* __beta, _C_t* __C, int __ldc) {
    return cublasGemmEx(__handle, __trans_A, __trans_B, __m, __n, __k, (void const*) __alpha, (void const*) __A,
            __cuda_data_type<_A_t>, __lda, (void const*) __B, __cuda_data_type<_B_t>, __ldb, (void const*) __beta,
            (void*) __C, __cuda_data_type<_C_t>, __ldc, __cublas_compute_type<_C_t>, CUBLAS_GEMM_DEFAULT);
}

template <>
cublasStatus_t __cublas_gemm(cublasHandle_t __handle, cublasOperation_t __trans_A, cublasOperation_t __trans_B, int __m,
        int __n, int __k, float const* __alpha, float const* __A, int __lda, float const* __B, int __ldb,
        float const* __beta, float* __C, int __ldc) {
    return cublasSgemm(
            __handle, __trans_A, __trans_B, __m, __n, __k, __alpha, __A, __lda, __B, __ldb, __beta, __C, __ldc);
}

template <>
cublasStatus_t __cublas_gemm(cublasHandle_t __handle, cublasOperation_t __trans_A, cublasOperation_t __trans_B, int __m,
        int __n, int __k, double const* __alpha, double const* __A, int __lda, double const* __B, int __ldb,
        double const* __beta, double* __C, int __ldc) {
    return cublasDgemm(
            __handle, __trans_A, __trans_B, __m, __n, __k, __alpha, __A, __lda, __B, __ldb, __beta, __C, __ldc);
}

template <>
cublasStatus_t __cublas_gemm(cublasHandle_t __handle, cublasOperation_t __trans_A, cublasOperation_t __trans_B, int __m,
        int __n, int __k, std::complex<float> const* __alpha, std::complex<float> const* __A, int __lda,
        std::complex<float> const* __B, int __ldb, std::complex<float> const* __beta, std::complex<float>* __C,
        int __ldc) {
    return cublasCgemm(__handle, __trans_A, __trans_B, __m, __n, __k, (cuComplex const*) __alpha,
            (cuComplex const*) __A, __lda, (cuComplex const*) __B, __ldb, (cuComplex const*) __beta, (cuComplex*) __C,
            __ldc);
}

template <>
cublasStatus_t __cublas_gemm(cublasHandle_t __handle, cublasOperation_t __trans_A, cublasOperation_t __trans_B, int __m,
        int __n, int __k, std::complex<double> const* __alpha, std::complex<double> const* __A, int __lda,
        std::complex<double> const* __B, int __ldb, std::complex<double> const* __beta, std::complex<double>* __C,
        int __ldc) {
    return cublasZgemm(__handle, __trans_A, __trans_B, __m, __n, __k, (cuDoubleComplex const*) __alpha,
            (cuDoubleComplex const*) __A, __lda, (cuDoubleComplex const*) __B, __ldb, (cuDoubleComplex const*) __beta,
            (cuDoubleComplex*) __C, __ldc);
}

// SYRK

template <class _Scalar, class _MatrixValType>
cublasStatus_t __cublas_syrk(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans, int __n,
        int __k, _Scalar const* __alpha, _MatrixValType const* __A, int __lda, _Scalar const* __beta,
        _MatrixValType* __C, int __ldc);

template <>
cublasStatus_t __cublas_syrk(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans, int __n,
        int __k, float const* __alpha, float const* __A, int __lda, float const* __beta, float* __C, int __ldc) {
    return cublasSsyrk(__handle, __uplo, __trans, __n, __k, __alpha, __A, __lda, __beta, __C, __ldc);
}

template <>
cublasStatus_t __cublas_syrk(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans, int __n,
        int __k, double const* __alpha, double const* __A, int __lda, double const* __beta, double* __C, int __ldc) {
    return cublasDsyrk(__handle, __uplo, __trans, __n, __k, __alpha, __A, __lda, __beta, __C, __ldc);
}

template <>
cublasStatus_t __cublas_syrk(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans, int __n,
        int __k, std::complex<float> const* __alpha, std::complex<float> const* __A, int __lda,
        std::complex<float> const* __beta, std::complex<float>* __C, int __ldc) {
    return cublasCsyrk(__handle, __uplo, __trans, __n, __k, (cuComplex const*) __alpha, (cuComplex const*) __A, __lda,
            (cuComplex const*) __beta, (cuComplex*) __C, __ldc);
}

template <>
cublasStatus_t __cublas_syrk(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans, int __n,
        int __k, std::complex<double> const* __alpha, std::complex<double> const* __A, int __lda,
        std::complex<double> const* __beta, std::complex<double>* __C, int __ldc) {
    return cublasZsyrk(__handle, __uplo, __trans, __n, __k, (cuDoubleComplex const*) __alpha,
            (cuDoubleComplex const*) __A, __lda, (cuDoubleComplex const*) __beta, (cuDoubleComplex*) __C, __ldc);
}

// SYR2K

template <class _Scalar, class _MatrixValType>
cublasStatus_t __cublas_syr2k(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans, int __n,
        int __k, _Scalar const* __alpha, _MatrixValType const* __A, int __lda, _MatrixValType const* __B, int __ldb,
        _Scalar const* __beta, _MatrixValType* __C, int __ldc);

template <>
cublasStatus_t __cublas_syr2k(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans, int __n,
        int __k, float const* __alpha, float const* __A, int __lda, float const* __B, int __ldb, float const* __beta,
        float* __C, int __ldc) {
    return cublasSsyr2k(__handle, __uplo, __trans, __n, __k, __alpha, __A, __lda, __B, __ldb, __beta, __C, __ldc);
}

template <>
cublasStatus_t __cublas_syr2k(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans, int __n,
        int __k, double const* __alpha, double const* __A, int __lda, double const* __B, int __ldb,
        double const* __beta, double* __C, int __ldc) {
    return cublasDsyr2k(__handle, __uplo, __trans, __n, __k, __alpha, __A, __lda, __B, __ldb, __beta, __C, __ldc);
}

template <>
cublasStatus_t __cublas_syr2k(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans, int __n,
        int __k, std::complex<float> const* __alpha, std::complex<float> const* __A, int __lda,
        std::complex<float> const* __B, int __ldb, std::complex<float> const* __beta, std::complex<float>* __C,
        int __ldc) {
    return cublasCsyr2k(__handle, __uplo, __trans, __n, __k, (cuComplex const*) __alpha, (cuComplex const*) __A, __lda,
            (cuComplex const*) __B, __ldb, (cuComplex const*) __beta, (cuComplex*) __C, __ldc);
}

template <>
cublasStatus_t __cublas_syr2k(cublasHandle_t __handle, cublasFillMode_t __uplo, cublasOperation_t __trans, int __n,
        int __k, std::complex<double> const* __alpha, std::complex<double> const* __A, int __lda,
        std::complex<double> const* __B, int __ldb, std::complex<double> const* __beta, std::complex<double>* __C,
        int __ldc) {
    return cublasZsyr2k(__handle, __uplo, __trans, __n, __k, (cuDoubleComplex const*) __alpha,
            (cuDoubleComplex const*) __A, __lda, (cuDoubleComplex const*) __B, __ldb, (cuDoubleComplex const*) __beta,
            (cuDoubleComplex*) __C, __ldc);
}

// TRSM

template <typename _Scalar, typename _MatrixValType>
cublasStatus_t __cublas_trsm(cublasHandle_t __handle, cublasSideMode_t __side, cublasFillMode_t __uplo,
        cublasOperation_t __trans, cublasDiagType_t __diag, int __m, int __n, _Scalar const* __alpha,
        _MatrixValType const* __A, int __lda, _MatrixValType* __B, int __ldb);

template <>
cublasStatus_t __cublas_trsm(cublasHandle_t __handle, cublasSideMode_t __side, cublasFillMode_t __uplo,
        cublasOperation_t __trans, cublasDiagType_t __diag, int __m, int __n, float const* __alpha, float const* __A,
        int __lda, float* __B, int __ldb) {
    return cublasStrsm(__handle, __side, __uplo, __trans, __diag, __m, __n, __alpha, __A, __lda, __B, __ldb);
}

template <>
cublasStatus_t __cublas_trsm(cublasHandle_t __handle, cublasSideMode_t __side, cublasFillMode_t __uplo,
        cublasOperation_t __trans, cublasDiagType_t __diag, int __m, int __n, double const* __alpha, double const* __A,
        int __lda, double* __B, int __ldb) {
    return cublasDtrsm(__handle, __side, __uplo, __trans, __diag, __m, __n, __alpha, __A, __lda, __B, __ldb);
}

template <>
cublasStatus_t __cublas_trsm(cublasHandle_t __handle, cublasSideMode_t __side, cublasFillMode_t __uplo,
        cublasOperation_t __trans, cublasDiagType_t __diag, int __m, int __n, std::complex<float> const* __alpha,
        std::complex<float> const* __A, int __lda, std::complex<float>* __B, int __ldb) {
    return cublasCtrsm(__handle, __side, __uplo, __trans, __diag, __m, __n, (cuComplex const*) __alpha,
            (cuComplex const*) __A, __lda, (cuComplex*) __B, __ldb);
}

template <>
cublasStatus_t __cublas_trsm(cublasHandle_t __handle, cublasSideMode_t __side, cublasFillMode_t __uplo,
        cublasOperation_t __trans, cublasDiagType_t __diag, int __m, int __n, std::complex<double> const* __alpha,
        std::complex<double> const* __A, int __lda, std::complex<double>* __B, int __ldb) {
    return cublasZtrsm(__handle, __side, __uplo, __trans, __diag, __m, __n, (cuDoubleComplex const*) __alpha,
            (cuDoubleComplex const*) __A, __lda, (cuDoubleComplex*) __B, __ldb);
}

// BLAS-like Extensions

// GEAM

template <typename _Scalar, typename _MatrixValType>
cublasStatus_t __cublas_geam(cublasHandle_t __handle, cublasOperation_t __trans_A, cublasOperation_t __trans_B, int __m,
        int __n, _Scalar const* __alpha, _MatrixValType const* __A, int __lda, _Scalar const* __beta,
        _MatrixValType const* __B, int __ldb, _MatrixValType* __C, int __ldc);

template <>
cublasStatus_t __cublas_geam(cublasHandle_t __handle, cublasOperation_t __trans_A, cublasOperation_t __trans_B, int __m,
        int __n, float const* __alpha, float const* __A, int __lda, float const* __beta, float const* __B, int __ldb,
        float* __C, int __ldc) {
    return cublasSgeam(__handle, __trans_A, __trans_B, __m, __n, __alpha, __A, __lda, __beta, __B, __ldb, __C, __ldc);
}

template <>
cublasStatus_t __cublas_geam(cublasHandle_t __handle, cublasOperation_t __trans_A, cublasOperation_t __trans_B, int __m,
        int __n, double const* __alpha, double const* __A, int __lda, double const* __beta, double const* __B,
        int __ldb, double* __C, int __ldc) {
    return cublasDgeam(__handle, __trans_A, __trans_B, __m, __n, __alpha, __A, __lda, __beta, __B, __ldb, __C, __ldc);
}

template <>
cublasStatus_t __cublas_geam(cublasHandle_t __handle, cublasOperation_t __trans_A, cublasOperation_t __trans_B, int __m,
        int __n, std::complex<float> const* __alpha, std::complex<float> const* __A, int __lda,
        std::complex<float> const* __beta, std::complex<float> const* __B, int __ldb, std::complex<float>* __C,
        int __ldc) {
    return cublasCgeam(__handle, __trans_A, __trans_B, __m, __n, (cuComplex const*) __alpha, (cuComplex const*) __A,
            __lda, (cuComplex const*) __beta, (cuComplex const*) __B, __ldb, (cuComplex*) __C, __ldc);
}

template <>
cublasStatus_t __cublas_geam(cublasHandle_t __handle, cublasOperation_t __trans_A, cublasOperation_t __trans_B, int __m,
        int __n, std::complex<double> const* __alpha, std::complex<double> const* __A, int __lda,
        std::complex<double> const* __beta, std::complex<double> const* __B, int __ldb, std::complex<double>* __C,
        int __ldc) {
    return cublasZgeam(__handle, __trans_A, __trans_B, __m, __n, (cuDoubleComplex const*) __alpha,
            (cuDoubleComplex const*) __A, __lda, (cuDoubleComplex const*) __beta, (cuDoubleComplex const*) __B, __ldb,
            (cuDoubleComplex*) __C, __ldc);
}

}  // namespace __cublas_std

#endif
