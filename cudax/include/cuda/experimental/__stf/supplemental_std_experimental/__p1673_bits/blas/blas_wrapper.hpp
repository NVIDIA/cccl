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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS_WRAPPER_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS_WRAPPER_HPP_

#include <complex.h>

namespace __blas_std {

// NOTE: I'm only exposing these extern declarations in a header file
// so that we can keep this a header-only library, for ease of testing
// and installation.  Exposing them in a real production
// implementation is really bad form.  This is because users may
// declare their own extern declarations of BLAS functions, and yours
// will collide with theirs at build time.

// NOTE: I'm assuming a particular BLAS ABI mangling here.  Typical
// BLAS C++ wrappers need to account for a variety of manglings that
// don't necessarily match the system's Fortran compiler (if it has
// one).  Lowercase with trailing underscore is a common pattern.
// Watch out for BLAS functions that return something, esp. a complex
// number.

extern "C" void scopy_(const int* __n, const float* __x, const int* __inc_x, float* __y, const int* __inc_y);

extern "C" void dcopy_(const int* __n, const double* __x, const int* __inc_x, double* __y, const int* __inc_y);

extern "C" void ccopy_(const int* __n, const void* __x, const int* __inc_x, void* __y, const int* __inc_y);

extern "C" void zcopy_(const int* __n, const void* __x, const int* __inc_x, void* __y, const int* __inc_y);

extern "C" void saxpy_(
        const int* __n, const float* __a, const float* __x, const int* __inc_x, float* __y, const int* __inc_y);

extern "C" void daxpy_(
        const int* __n, const double* __a, const double* __x, const int* __inc_x, double* __y, const int* __inc_y);

extern "C" void caxpy_(
        const int* __n, const void* __a, const void* __x, const int* __inc_x, void* __y, const int* __inc_y);

extern "C" void zaxpy_(
        const int* __n, const void* __a, const void* __x, const int* __inc_x, void* __y, const int* __inc_y);

extern "C" float sdot_(const int* __n, const float* __x, const int* __inc_x, const float* __y, const int* __inc_y);

extern "C" double ddot_(const int* __n, const double* __x, const int* __inc_x, const double* __y, const int* __inc_y);

#if defined(__powerpc__) || defined(__powerpc64__)
extern "C" std::complex<float> cdotu_(
        const int* __n, const void* __x, const int* __inc_x, const void* __y, const int* __inc_y);

extern "C" std::complex<double> zdotu_(
        const int* __n, const void* __x, const int* __inc_x, const void* __y, const int* __inc_y);

extern "C" std::complex<float> cdotc_(
        const int* __n, const void* __x, const int* __inc_x, const void* __y, const int* __inc_y);

extern "C" std::complex<double> zdotc_(
        const int* __n, const void* __x, const int* __inc_x, const void* __y, const int* __inc_y);
#else
extern "C" void cdotu_(
        void* __result, const int* __n, const void* __x, const int* __inc_x, const void* __y, const int* __inc_y);

extern "C" void zdotu_(
        void* __result, const int* __n, const void* __x, const int* __inc_x, const void* __y, const int* __inc_y);

extern "C" void cdotc_(
        void* __result, const int* __n, const void* __x, const int* __inc_x, const void* __y, const int* __inc_y);

extern "C" void zdotc_(
        void* __result, const int* __n, const void* __x, const int* __inc_x, const void* __y, const int* __inc_y);
#endif

extern "C" void sscal_(const int* __n, const float* __a, float* __x, const int* __inc_x);

extern "C" void dscal_(const int* __n, const double* __a, double* __x, const int* __inc_x);

extern "C" void cscal_(const int* __n, const void* __a, void* __x, const int* __inc_x);

extern "C" void zscal_(const int* __n, const void* __a, void* __x, const int* __inc_x);

extern "C" float snrm2_(const int* __n, const float* __x, const int* __inc_x);

extern "C" double dnrm2_(const int* __n, const double* __x, const int* __inc_x);

extern "C" float scnrm2_(const int* __n, const void* __x, const int* __inc_x);

extern "C" double dznrm2_(const int* __n, const void* __x, const int* __inc_x);

extern "C" void sgemv_(const char __trans[], const int* __m, const int* __n, const float* __alpha, const float* __a,
        const int* __lda, const float* __x, const int* __inc_x, const float* __beta, float* __y, const int* __inc_y);

extern "C" void dgemv_(const char __trans[], const int* __m, const int* __n, const double* __alpha, const double* __a,
        const int* __lda, const double* __x, const int* __inc_x, const double* __beta, double* __y, const int* __inc_y);

extern "C" void cgemv_(const char __trans[], const int* __m, const int* __n, const void* __alpha, const void* __a,
        const int* __lda, const void* __x, const int* __inc_x, const void* __beta, void* __y, const int* __inc_y);

extern "C" void zgemv_(const char __trans[], const int* __m, const int* __n, const void* __alpha, const void* __a,
        const int* __lda, const void* __x, const int* __inc_x, const void* __beta, void* __y, const int* __inc_y);

extern "C" void strsv_(const char __uplo[], const char __trans[], const char __diag[], const int* __n, const float* __a,
        const int* __lda, float* __x, const int* __inc_x);

extern "C" void dtrsv_(const char __uplo[], const char __trans[], const char __diag[], const int* __n,
        const double* __a, const int* __lda, double* __x, const int* __inc_x);

extern "C" void ctrsv_(const char __uplo[], const char __trans[], const char __diag[], const int* __n, const void* __a,
        const int* __lda, void* __x, const int* __inc_x);

extern "C" void ztrsv_(const char __uplo[], const char __trans[], const char __diag[], const int* __n, const void* __a,
        const int* __lda, void* __x, const int* __inc_x);

extern "C" void dgemm_(const char __trans_A[], const char __trans_B[], const int* __m, const int* __n, const int* __k,
        const double* __alpha, const double* __A, const int* __lda, const double* __B, const int* __ldb,
        const double* __beta, double* __C, const int* __ldc);

extern "C" void sgemm_(const char __trans_A[], const char __trans_B[], const int* __m, const int* __n, const int* __k,
        const float* __alpha, const float* __A, const int* __lda, const float* __B, const int* __ldb,
        const float* __beta, float* __C, const int* __ldc);

extern "C" void cgemm_(const char __trans_A[], const char __trans_B[], const int* __m, const int* __n, const int* __k,
        const void* __alpha, const void* __A, const int* __lda, const void* __B, const int* __ldb, const void* __beta,
        void* __C, const int* __ldc);

extern "C" void zgemm_(const char __trans_A[], const char __trans_B[], const int* __m, const int* __n, const int* __k,
        const void* __alpha, const void* __A, const int* __lda, const void* __B, const int* __ldb, const void* __beta,
        void* __C, const int* __ldc);

extern "C" void ssyrk_(const char __uplo[], const char __trans[], const int* __n, const int* __k, const float* __alpha,
        const float* __a, const int* __lda, const float* __beta, float* __c, const int* __ldc);

extern "C" void dsyrk_(const char __uplo[], const char __trans[], const int* __n, const int* __k, const double* __alpha,
        const double* __a, const int* __lda, const double* __beta, double* __c, const int* __ldc);

extern "C" void csyrk_(const char __uplo[], const char __trans[], const int* __n, const int* __k, const void* __alpha,
        const void* __a, const int* __lda, const void* __beta, void* __c, const int* __ldc);

extern "C" void zsyrk_(const char __uplo[], const char __trans[], const int* __n, const int* __k, const void* __alpha,
        const void* __a, const int* __lda, const void* __beta, void* __c, const int* __ldc);

extern "C" void ssyr2k_(const char __uplo[], const char __trans[], const int* __n, const int* __k, const float* __alpha,
        const float* __a, const int* __lda, const float* __b, const int* __ldb, const float* __beta, float* __c,
        const int* __ldc);

extern "C" void dsyr2k_(const char __uplo[], const char __trans[], const int* __n, const int* __k,
        const double* __alpha, const double* __a, const int* __lda, const double* __b, const int* __ldb,
        const double* __beta, double* __c, const int* __ldc);

extern "C" void csyr2k_(const char __uplo[], const char __trans[], const int* __n, const int* __k, const void* __alpha,
        const void* __a, const int* __lda, const void* __b, const int* __ldb, const void* __beta, void* __c,
        const int* __ldc);

extern "C" void zsyr2k_(const char __uplo[], const char __trans[], const int* __n, const int* __k, const void* __alpha,
        const void* __a, const int* __lda, const void* __b, const int* __ldb, const void* __beta, void* __c,
        const int* __ldc);

extern "C" void strsm_(const char __side[], const char __uplo[], const char __trans_a[], const char __diag[],
        const int* __m, const int* __n, const float* __alpha, const float* __a, const int* __lda, const float* __b,
        const int* __ldb);

extern "C" void dtrsm_(const char __side[], const char __uplo[], const char __trans_a[], const char __diag[],
        const int* __m, const int* __n, const double* __alpha, const double* __a, const int* __lda, const double* __b,
        const int* __ldb);

extern "C" void ctrsm_(const char __side[], const char __uplo[], const char __trans_a[], const char __diag[],
        const int* __m, const int* __n, const void* __alpha, const void* __a, const int* __lda, const void* __b,
        const int* __ldb);

extern "C" void ztrsm_(const char __side[], const char __uplo[], const char __trans_a[], const char __diag[],
        const int* __m, const int* __n, const void* __alpha, const void* __a, const int* __lda, const void* __b,
        const int* __ldb);

template <class _Scalar>
struct __blas_copy {};

template <>
struct __blas_copy<float> {
    static void __copy(const int __n, const float* __X, int __inc_x, float* __y, int __inc_y) {
        scopy_(&__n, __X, &__inc_x, __y, &__inc_y);
    }
};

template <>
struct __blas_copy<double> {
    static void __copy(const int __n, const double* __X, int __inc_x, double* __y, int __inc_y) {
        dcopy_(&__n, __X, &__inc_x, __y, &__inc_y);
    }
};

template <>
struct __blas_copy<std::complex<float>> {
    static void __copy(
            const int __n, const std::complex<float>* __X, int __inc_x, std::complex<float>* __y, int __inc_y) {
        ccopy_(&__n, __X, &__inc_x, __y, &__inc_y);
    }
};

template <>
struct __blas_copy<std::complex<double>> {
    static void __copy(
            const int __n, const std::complex<double>* __X, int __inc_x, std::complex<double>* __y, int __inc_y) {
        zcopy_(&__n, __X, &__inc_x, __y, &__inc_y);
    }
};

template <class _Scalar>
struct __blas_axpy {};

template <>
struct __blas_axpy<float> {
    static void __axpy(const int __n, const float __A, const float* __X, int __inc_x, float* __y, int __inc_y) {
        saxpy_(&__n, &__A, __X, &__inc_x, __y, &__inc_y);
    }
};

template <>
struct __blas_axpy<double> {
    static void __axpy(const int __n, const double __A, const double* __X, int __inc_x, double* __y, int __inc_y) {
        daxpy_(&__n, &__A, __X, &__inc_x, __y, &__inc_y);
    }
};

template <>
struct __blas_axpy<std::complex<float>> {
    static void __axpy(const int __n, const std::complex<float> __A, const std::complex<float>* __X, int __inc_x,
            std::complex<float>* __y, int __inc_y) {
        caxpy_(&__n, &__A, __X, &__inc_x, __y, &__inc_y);
    }
};

template <>
struct __blas_axpy<std::complex<double>> {
    static void __axpy(const int __n, const std::complex<double> __A, const std::complex<double>* __X, int __inc_x,
            std::complex<double>* __y, int __inc_y) {
        zaxpy_(&__n, &__A, __X, &__inc_x, __y, &__inc_y);
    }
};

template <class _Scalar>
struct __blas_dot_impl {};

template <>
struct __blas_dot_impl<float> {
    static float __dot(const int __n, const float* __X, int __inc_x, const float* __y, int __inc_y) {
        return sdot_(&__n, __X, &__inc_x, __y, &__inc_y);
    }
};

template <>
struct __blas_dot_impl<double> {
    static double __dot(int __n, const double* __X, int __inc_x, const double* __y, int __inc_y) {
        return ddot_(&__n, __X, &__inc_x, __y, &__inc_y);
    }
};

template <>
struct __blas_dot_impl<std::complex<float>> {
    static std::complex<float> __dot(
            int __n, const std::complex<float>* __X, int __inc_x, const std::complex<float>* __y, int __inc_y) {
#if defined(__powerpc__) || defined(__powerpc64__)
        return cdotu_(&__n, __X, &__inc_x, __y, &__inc_y);
#else
        std::complex<float> __res;
        cdotu_(&__res, &__n, __X, &__inc_x, __y, &__inc_y);
        return __res;
#endif
    }
};

template <>
struct __blas_dot_impl<std::complex<double>> {
    static std::complex<double> __dot(
            const int __n, const std::complex<double>* __X, int __inc_x, const std::complex<double>* __y, int __inc_y) {
#if defined(__powerpc__) || defined(__powerpc64__)
        return zdotu_(&__n, __X, &__inc_x, __y, &__inc_y);
#else
        std::complex<double> __res;
        zdotu_(&__res, &__n, __X, &__inc_x, __y, &__inc_y);
        return __res;
#endif
    }
};

template <class _Scalar>
using __blas_dot = __blas_dot_impl<std::remove_cv_t<_Scalar>>;

template <class _Scalar>
struct __blas_dotc_impl {};

template <>
struct __blas_dotc_impl<std::complex<float>> {
    static std::complex<float> __dotc(
            int __n, const std::complex<float>* __X, int __inc_x, const std::complex<float>* __y, int __inc_y) {
#if defined(__powerpc__) || defined(__powerpc64__)
        return cdotc_(&__n, __X, &__inc_x, __y, &__inc_y);
#else
        std::complex<float> __res;
        cdotc_(&__res, &__n, __X, &__inc_x, __y, &__inc_y);
        return __res;
#endif
    }
};

template <>
struct __blas_dotc_impl<std::complex<double>> {
    static std::complex<double> __dotc(
            const int __n, const std::complex<double>* __X, int __inc_x, const std::complex<double>* __y, int __inc_y) {
#if defined(__powerpc__) || defined(__powerpc64__)
        return zdotc_(&__n, __X, &__inc_x, __y, &__inc_y);
#else
        std::complex<double> __res;
        zdotc_(&__res, &__n, __X, &__inc_x, __y, &__inc_y);
        return __res;
#endif
    }
};

template <class _Scalar>
using __blas_dotc = __blas_dotc_impl<std::remove_cv_t<_Scalar>>;

template <class _Scalar>
struct __blas_scal {};

template <>
struct __blas_scal<float> {
    static void __scal(int __n, float __A, float* __X, int __inc_x) { sscal_(&__n, &__A, __X, &__inc_x); }
};

template <>
struct __blas_scal<double> {
    static void __scal(int __n, double __A, double* __X, int __inc_x) { dscal_(&__n, &__A, __X, &__inc_x); }
};

template <>
struct __blas_scal<std::complex<float>> {
    static void __scal(int __n, std::complex<float> __A, std::complex<float>* __X, int __inc_x) {
        cscal_(&__n, &__A, __X, &__inc_x);
    }
};

template <>
struct __blas_scal<std::complex<double>> {
    static void __scal(int __n, std::complex<double> __A, std::complex<double>* __X, int __inc_x) {
        zscal_(&__n, &__A, __X, &__inc_x);
    }
};

template <class _Scalar>
struct BlasNrm2Impl {};

template <>
struct BlasNrm2Impl<float> {
    static float __nrm2(int __n, const float* __X, int __inc_x) { return snrm2_(&__n, __X, &__inc_x); }
};

template <>
struct BlasNrm2Impl<double> {
    static double __nrm2(int __n, const double* __X, int __inc_x) { return dnrm2_(&__n, __X, &__inc_x); }
};

template <>
struct BlasNrm2Impl<std::complex<float>> {
    static float __nrm2(int __n, const std::complex<float>* __X, int __inc_x) { return scnrm2_(&__n, __X, &__inc_x); }
};

template <>
struct BlasNrm2Impl<std::complex<double>> {
    static double __nrm2(int __n, const std::complex<double>* __X, int __inc_x) { return dznrm2_(&__n, __X, &__inc_x); }
};

template <class _Scalar>
using BlasNrm2 = BlasNrm2Impl<std::remove_cv_t<_Scalar>>;

template <class _Scalar>
struct __blas_gemv {};

template <>
struct __blas_gemv<float> {
    static void __gemv(const char __trans_A[], const int __m, const int __n, const float __alpha, const float* __A,
            const int __lda, const float* __X, const int __inc_x, const float __beta, float* __y, const int __inc_y) {
        sgemv_(__trans_A, &__m, &__n, &__alpha, __A, &__lda, __X, &__inc_x, &__beta, __y, &__inc_y);
    }
};

template <>
struct __blas_gemv<double> {
    static void __gemv(const char __trans_A[], const int __m, const int __n, const double __alpha, const double* __A,
            const int __lda, const double* __X, const int __inc_x, const double __beta, double* __y,
            const int __inc_y) {
        dgemv_(__trans_A, &__m, &__n, &__alpha, __A, &__lda, __X, &__inc_x, &__beta, __y, &__inc_y);
    }
};

template <>
struct __blas_gemv<std::complex<float>> {
    static void __gemv(const char __trans_A[], const int __m, const int __n, const std::complex<float> __alpha,
            const std::complex<float>* __A, const int __lda, const std::complex<float>* __X, const int __inc_x,
            const std::complex<float> __beta, std::complex<float>* __y, const int __inc_y) {
        cgemv_(__trans_A, &__m, &__n, &__alpha, __A, &__lda, __X, &__inc_x, &__beta, __y, &__inc_y);
    }
};

template <>
struct __blas_gemv<std::complex<double>> {
    static void __gemv(const char __trans_A[], const int __m, const int __n, const std::complex<double> __alpha,
            const std::complex<double>* __A, const int __lda, const std::complex<double>* __X, const int __inc_x,
            const std::complex<double> __beta, std::complex<double>* __y, const int __inc_y) {
        zgemv_(__trans_A, &__m, &__n, &__alpha, __A, &__lda, __X, &__inc_x, &__beta, __y, &__inc_y);
    }
};

template <class _Scalar>
struct __blas_trsv {};

template <>
struct __blas_trsv<float> {
    static void __trsv(const char __uplo[], const char __trans[], const char __diag[], const int __n, const float* __A,
            const int __lda, float* __X, const int __inc_x) {
        strsv_(__uplo, __trans, __diag, &__n, __A, &__lda, __X, &__inc_x);
    }
};

template <>
struct __blas_trsv<double> {
    static void __trsv(const char __uplo[], const char __trans[], const char __diag[], const int __n, const double* __A,
            const int __lda, double* __X, const int __inc_x) {
        dtrsv_(__uplo, __trans, __diag, &__n, __A, &__lda, __X, &__inc_x);
    }
};

template <>
struct __blas_trsv<std::complex<float>> {
    static void __trsv(const char __uplo[], const char __trans[], const char __diag[], const int __n,
            const std::complex<float>* __A, const int __lda, std::complex<float>* __X, const int __inc_x) {
        ctrsv_(__uplo, __trans, __diag, &__n, __A, &__lda, __X, &__inc_x);
    }
};

template <>
struct __blas_trsv<std::complex<double>> {
    static void __trsv(const char __uplo[], const char __trans[], const char __diag[], const int __n,
            const std::complex<double>* __A, const int __lda, std::complex<double>* __X, const int __inc_x) {
        ztrsv_(__uplo, __trans, __diag, &__n, __A, &__lda, __X, &__inc_x);
    }
};

template <class _Scalar>
struct __blas_gemm {
    // static void
    // __gemm (const char __trans_A[], const char __trans_B[],
    //       const int __m, const int __n, const int __k,
    //       const _Scalar ALPHA,
    //       const _Scalar* __A, const int __lda,
    //       const _Scalar* __B, const int __ldb,
    //       const _Scalar BETA,
    //       _Scalar* __C, const int __ldc);
};

template <>
struct __blas_gemm<double> {
    static void __gemm(const char __trans_A[], const char __trans_B[], const int __m, const int __n, const int __k,
            const double ALPHA, const double* __A, const int __lda, const double* __B, const int __ldb,
            const double BETA, double* __C, const int __ldc) {
        dgemm_(__trans_A, __trans_B, &__m, &__n, &__k, &ALPHA, __A, &__lda, __B, &__ldb, &BETA, __C, &__ldc);
    }
};

template <>
struct __blas_gemm<float> {
    static void __gemm(const char __trans_A[], const char __trans_B[], const int __m, const int __n, const int __k,
            const float ALPHA, const float* __A, const int __lda, const float* __B, const int __ldb, const float BETA,
            float* __C, const int __ldc) {
        sgemm_(__trans_A, __trans_B, &__m, &__n, &__k, &ALPHA, __A, &__lda, __B, &__ldb, &BETA, __C, &__ldc);
    }
};

template <>
struct __blas_gemm<std::complex<double>> {
    static void __gemm(const char __trans_A[], const char __trans_B[], const int __m, const int __n, const int __k,
            const std::complex<double> ALPHA, const std::complex<double>* __A, const int __lda,
            const std::complex<double>* __B, const int __ldb, const std::complex<double> BETA,
            std::complex<double>* __C, const int __ldc) {
        zgemm_(__trans_A, __trans_B, &__m, &__n, &__k, &ALPHA, __A, &__lda, __B, &__ldb, &BETA, __C, &__ldc);
    }
};

template <>
struct __blas_gemm<std::complex<float>> {
    static void __gemm(const char __trans_A[], const char __trans_B[], const int __m, const int __n, const int __k,
            const std::complex<float> ALPHA, const std::complex<float>* __A, const int __lda,
            const std::complex<float>* __B, const int __ldb, const std::complex<float> BETA, std::complex<float>* __C,
            const int __ldc) {
        cgemm_(__trans_A, __trans_B, &__m, &__n, &__k, &ALPHA, __A, &__lda, __B, &__ldb, &BETA, __C, &__ldc);
    }
};

template <class _Scalar>
struct __blas_syrk {};

template <>
struct __blas_syrk<float> {
    static void __syrk(const char __uplo[], const char __trans[], const int __m, const int __k, const float __alpha,
            const float* __A, const int __lda, const float __beta, float* __C, const int __ldc) {
        ssyrk_(__uplo, __trans, &__m, &__k, &__alpha, __A, &__lda, &__beta, __C, &__ldc);
    }
};

template <>
struct __blas_syrk<double> {
    static void __syrk(const char __uplo[], const char __trans[], const int __m, const int __k, const double __alpha,
            const double* __A, const int __lda, const double __beta, double* __C, const int __ldc) {
        dsyrk_(__uplo, __trans, &__m, &__k, &__alpha, __A, &__lda, &__beta, __C, &__ldc);
    }
};

template <>
struct __blas_syrk<std::complex<float>> {
    static void __syrk(const char __uplo[], const char __trans[], const int __m, const int __k,
            const std::complex<float> __alpha, const std::complex<float>* __A, const int __lda,
            const std::complex<float> __beta, std::complex<float>* __C, const int __ldc) {
        csyrk_(__uplo, __trans, &__m, &__k, &__alpha, __A, &__lda, &__beta, __C, &__ldc);
    }
};

template <>
struct __blas_syrk<std::complex<double>> {
    static void __syrk(const char __uplo[], const char __trans[], const int __m, const int __k,
            const std::complex<double> __alpha, const std::complex<double>* __A, const int __lda,
            const std::complex<double> __beta, std::complex<double>* __C, const int __ldc) {
        zsyrk_(__uplo, __trans, &__m, &__k, &__alpha, __A, &__lda, &__beta, __C, &__ldc);
    }
};

template <class _Scalar>
struct __blas_syr2k {};

template <>
struct __blas_syr2k<float> {
    static void __syr2k(const char __uplo[], const char __trans[], const int __m, const int __k, const float __alpha,
            const float* __A, const int __lda, const float* __B, const int __ldb, const float __beta, float* __C,
            const int __ldc) {
        ssyr2k_(__uplo, __trans, &__m, &__k, &__alpha, __A, &__lda, __B, &__ldb, &__beta, __C, &__ldc);
    }
};

template <>
struct __blas_syr2k<double> {
    static void __syr2k(const char __uplo[], const char __trans[], const int __m, const int __k, const double __alpha,
            const double* __A, const int __lda, const double* __B, const int __ldb, const double __beta, double* __C,
            const int __ldc) {
        dsyr2k_(__uplo, __trans, &__m, &__k, &__alpha, __A, &__lda, __B, &__ldb, &__beta, __C, &__ldc);
    }
};

template <>
struct __blas_syr2k<std::complex<float>> {
    static void __syr2k(const char __uplo[], const char __trans[], const int __m, const int __k,
            const std::complex<float> __alpha, const std::complex<float>* __A, const int __lda,
            const std::complex<float>* __B, const int __ldb, const std::complex<float> __beta, std::complex<float>* __C,
            const int __ldc) {
        csyr2k_(__uplo, __trans, &__m, &__k, &__alpha, __A, &__lda, __B, &__ldb, &__beta, __C, &__ldc);
    }
};

template <>
struct __blas_syr2k<std::complex<double>> {
    static void __syr2k(const char __uplo[], const char __trans[], const int __m, const int __k,
            const std::complex<double> __alpha, const std::complex<double>* __A, const int __lda,
            const std::complex<double>* __B, const int __ldb, const std::complex<double> __beta,
            std::complex<double>* __C, const int __ldc) {
        zsyr2k_(__uplo, __trans, &__m, &__k, &__alpha, __A, &__lda, __B, &__ldb, &__beta, __C, &__ldc);
    }
};

template <class _Scalar>
struct __blas_trsm {};

template <>
struct __blas_trsm<float> {
    static void __trsm(const char __side[], const char __uplo[], const char __trans_A[], const char __diag[],
            const int __m, const int __n, const float __alpha, const float* __A, const int __lda, const float* __B,
            const int __ldb) {
        strsm_(__side, __uplo, __trans_A, __diag, &__m, &__n, &__alpha, __A, &__lda, __B, &__ldb);
    }
};

template <>
struct __blas_trsm<double> {
    static void __trsm(const char __side[], const char __uplo[], const char __trans_A[], const char __diag[],
            const int __m, const int __n, const double __alpha, const double* __A, const int __lda, const double* __B,
            const int __ldb) {
        dtrsm_(__side, __uplo, __trans_A, __diag, &__m, &__n, &__alpha, __A, &__lda, __B, &__ldb);
    }
};

template <>
struct __blas_trsm<std::complex<float>> {
    static void __trsm(const char __side[], const char __uplo[], const char __trans_A[], const char __diag[],
            const int __m, const int __n, const std::complex<float> __alpha, const std::complex<float>* __A,
            const int __lda, const std::complex<float>* __B, const int __ldb) {
        ctrsm_(__side, __uplo, __trans_A, __diag, &__m, &__n, &__alpha, __A, &__lda, __B, &__ldb);
    }
};

template <>
struct __blas_trsm<std::complex<double>> {
    static void __trsm(const char __side[], const char __uplo[], const char __trans_A[], const char __diag[],
            const int __m, const int __n, const std::complex<double> __alpha, const std::complex<double>* __A,
            const int __lda, const std::complex<double>* __B, const int __ldb) {
        ztrsm_(__side, __uplo, __trans_A, __diag, &__m, &__n, &__alpha, __A, &__lda, __B, &__ldb);
    }
};

template <class _Scalar>
struct __blas_conj {
    static void __conj(const int __n, _Scalar* __x, const int __inc = 1) {}

    static void __conj(const int __n, _Scalar const* __in, _Scalar* __out, const int __inc = 1) {
        __blas_copy<_Scalar>::__copy(__n, __in, __out, __inc);
    }
};

template <class _Scalar>
struct __blas_conj<std::complex<_Scalar>> {
    static void __conj(const int __n, std::complex<_Scalar>* __x, const int __inc = 1) {
        _Scalar const alpha { -1 };
        __blas_scal<_Scalar>::__scal(__n, alpha, reinterpret_cast<_Scalar*>(__x) + 1, 2 * __inc);
    }

    static void __conj(
            const int __n, std::complex<_Scalar> const* __in, std::complex<_Scalar>* __out, const int __inc = 1) {
        _Scalar const alpha { -1 };
        __blas_copy<std::complex<_Scalar>>::__copy(__n, __in, __inc, __out, __inc);
        __blas_scal<_Scalar>::__scal(__n, alpha, reinterpret_cast<_Scalar*>(__out) + 1, 2 * __inc);
    }
};

template <class _Scalar>
struct __blas_conj_2d {
    static void __conj(const int __m, const int __n, _Scalar* __X, const int __ldx) {
        for (int __i_col = 0; __i_col < __n; ++__i_col) {
            __blas_conj<_Scalar>::__conj(__m, __X + __i_col * __ldx);
        }
    }

    static void __conj(const int __m, const int __n, _Scalar const* __in, _Scalar* __out, const int __ldx) {
        for (int __i_col = 0; __i_col < __n; ++__i_col) {
            __blas_conj<_Scalar>::__conj(__m, __in + __i_col * __ldx, __out + __i_col * __ldx);
        }
    }
};

template <class _Scalar>
struct __blas_copy_2d {
    static void __copy(
            const char __trans, const int __m, const int __n, const _Scalar* __X, int __ldx, _Scalar* __y, int __ldy) {
        switch (__trans) {
        case 'N':
            for (int __i_col = 0; __i_col < __n; ++__i_col) {
                __blas_copy<_Scalar>::__copy(__m, __X + __i_col * __ldx, 1, __y + __i_col * __ldy, 1);
            }
            break;
        case 'T':
            for (int __i_col = 0; __i_col < __n; ++__i_col) {
                __blas_copy<_Scalar>::__copy(__m, __X + __i_col * __ldx, 1, __y + __i_col, __ldy);
            }
            break;
        case 'C':
            for (int __i_col = 0; __i_col < __n; ++__i_col) {
                __blas_copy<_Scalar>::__copy(__m, __X + __i_col * __ldx, 1, __y + __i_col, __ldy);
                __blas_conj<_Scalar>::__conj(__m, __y + __i_col, __ldy);
            }
            break;
        default:
            // Shouldn't be here
        }
    }
};

template <class _Scalar>
struct __blas_scal_2d {
    static void __scal(int __m, int __n, _Scalar __alpha, _Scalar* __X, int __ldx) {
        for (std::size_t __i_col = 0; __i_col < __n; ++__i_col) {
            __blas_scal<_Scalar>::__scal(__m, __alpha, __X + __i_col * __ldx, 1);
        }
    }
};

template <class _Scalar>
struct __blas_axpy_2d {
    static void __axpy(
            char __trans, int __m, int __n, _Scalar __A, const _Scalar* __X, int __ldx, _Scalar* __y, int __ldy) {
        switch (__trans) {
        case 'N':
            for (int __i_col = 0; __i_col < __n; ++__i_col) {
                __blas_axpy<_Scalar>::__axpy(__m, __A, __X + __i_col * __ldx, 1, __y + __i_col * __ldy, 1);
            }
            break;
        case 'T':
            for (int __i_col = 0; __i_col < __n; ++__i_col) {
                __blas_axpy<_Scalar>::__axpy(__m, __A, __X + __i_col * __ldx, 1, __y + __i_col, __ldy);
            }
            break;
        case 'C':
            // a*conj(x) + y = conj(conj(a)*x + conj(y))
            if constexpr (std::experimental::linalg::is_complex<_Scalar>::value) {
                __A = std::conj(__A);
            }

            for (int __i_col = 0; __i_col < __n; ++__i_col) {
                __blas_conj<_Scalar>::__conj(__m, __y + __i_col, __ldy);
                __blas_axpy<_Scalar>::__axpy(__m, __A, __X + __i_col * __ldx, 1, __y + __i_col, __ldy);
                __blas_conj<_Scalar>::__conj(__m, __y + __i_col, __ldy);
            }
            break;
        default:
            // Shouldn't be here
        }
    }
};

}  // namespace __blas_std

#endif
