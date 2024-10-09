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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_EXEC_POLICY_WRAPPER_NVHPC_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_EXEC_POLICY_WRAPPER_NVHPC_HPP_

#include <execution>
#ifdef __LINALG_ENABLE_CUBLAS
#    include "nvhpc_par_no_sync.hpp"
#endif

namespace __nvhpc_std {

struct __nvhpc_no_sync {};

struct __nvhpc_sync {};

template <class _cublas_sync_type = __nvhpc_sync>
struct __cublas_exec_space {};

struct __blas_exec_space {};

#ifdef __LINALG_ENABLE_CUBLAS_DEFAULT
template <class _exec_space = __cublas_exec_space<>>
#else
template <class _exec_space = __blas_exec_space>
#endif
struct __nvhpc_exec {
};

template <class _ExecutionPolicy>
struct __base_exec_mapper {
    constexpr auto __map() { return _ExecutionPolicy {}; }
};

template <class _ExecutionPolicy>
struct __base_exec_mapper<_ExecutionPolicy&> {
    constexpr auto __map() { return _ExecutionPolicy {}; }
};

template <class _SyncType>
struct __base_exec_mapper<__nvhpc_exec<__cublas_exec_space<_SyncType>>> {
    constexpr auto __map() { return __nvhpc_exec<__cublas_exec_space<>> {}; }
};

template <class _SyncType>
struct __base_exec_mapper<__nvhpc_exec<__cublas_exec_space<_SyncType>>&> {
    constexpr auto __map() { return __nvhpc_exec<__cublas_exec_space<>> {}; }
};

template <class _exec_space>
auto execpolicy_mapper(__nvhpc_exec<_exec_space>) {
    return __nvhpc_exec<_exec_space>();
}
}  // namespace __nvhpc_std

namespace std::experimental {
template <>
struct is_execution_policy<__nvhpc_std::__nvhpc_exec<__nvhpc_std::__cublas_exec_space<__nvhpc_std::__nvhpc_sync>>>
        : std::true_type {};
template <>
struct is_execution_policy<__nvhpc_std::__nvhpc_exec<__nvhpc_std::__cublas_exec_space<__nvhpc_std::__nvhpc_no_sync>>>
        : std::true_type {};
template <>
struct is_execution_policy<__nvhpc_std::__nvhpc_exec<__nvhpc_std::__blas_exec_space>> : std::true_type {};
}  // namespace std::experimental

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {
auto execpolicy_mapper(std::execution::parallel_policy) {
    return __nvhpc_std::__nvhpc_exec<>();
}
auto execpolicy_mapper(std::execution::parallel_unsequenced_policy) {
    return __nvhpc_std::__nvhpc_exec<>();
}
#ifdef __LINALG_ENABLE_CUBLAS
auto execpolicy_mapper(no_sync_policy<std::execution::parallel_policy>) {
    return __nvhpc_std::__nvhpc_exec<__nvhpc_std::__cublas_exec_space<__nvhpc_std::__nvhpc_no_sync>>();
}
#endif
}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif
