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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_NVHPC_ERROR_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_NVHPC_ERROR_HPP_

#include <system_error>
#ifdef __LINALG_ENABLE_CUBLAS
#    include "cublas_v2.h"
#    include "driver_types.h"
#endif

namespace std { namespace experimental {

enum class __stdblas_errc { __SUCCESS, __UNSUPPORTED_INPLACE_OPS };

class __stdblas_category_t : public std::error_category {
public:
    virtual const char* name() const noexcept { return "stdblas"; }

    virtual std::string message(int __ev) const {
        switch (__ev) {
        case static_cast<int>(__stdblas_errc::__SUCCESS): return "Success";
        case static_cast<int>(__stdblas_errc::__UNSUPPORTED_INPLACE_OPS): return "Unsupported in-place transformation";
        default: return "Unknown error";
        }
    }
};

inline __stdblas_category_t __stdblas_category;

inline error_code make_error_code(__stdblas_errc __e) noexcept {
    return std::error_code(static_cast<int>(__e), __stdblas_category);
}

#ifdef __LINALG_ENABLE_CUBLAS
class __cublas_category_t : public std::error_category {
public:
    virtual const char* name() const noexcept { return "cublas"; }

    virtual std::string message(int __ev) const {
        return std::string(cublasGetStatusString(static_cast<cublasStatus_t>(__ev)));
    }
};

class __cuda_category_t : public std::error_category {
public:
    virtual const char* name() const noexcept { return "cuda"; }

    virtual std::string message(int __ev) const {
        return std::string(cudaGetErrorString(static_cast<cudaError_t>(__ev)));
    }
};

inline __cublas_category_t __cublas_category;

inline __cuda_category_t __cuda_category;

inline error_code make_error_code(cublasStatus_t __e) noexcept {
    return std::error_code(static_cast<int>(__e), __cublas_category);
}

inline error_code make_error_code(cudaError_t __e) noexcept {
    return std::error_code(static_cast<int>(__e), __cuda_category);
}
#endif
}}  // namespace std::experimental

#endif
