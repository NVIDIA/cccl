/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_EXEC_POLICY_WRAPPER_CUBLAS_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_EXEC_POLICY_WRAPPER_CUBLAS_HPP_

#include <execution>

namespace __cublas_std
{

struct cublas_exec
{};

auto execpolicy_mapper(cublas_exec)
{
  return cublas_exec();
}
} // namespace __cublas_std

// Remap standard execution policies to cuBLAS
#ifdef __LINALG_ENABLE_CUBLAS_DEFAULT
namespace std
{
namespace experimental
{
inline namespace __p1673_version_0
{
namespace linalg
{
auto execpolicy_mapper(std::experimental::linalg::impl::default_exec_t)
{
  return __cublas_std::cublas_exec();
}
auto execpolicy_mapper(std::execution::parallel_policy)
{
  return __cublas_std::cublas_exec();
}
auto execpolicy_mapper(std::execution::parallel_unsequenced_policy)
{
  return __cublas_std::cublas_exec();
}
} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std
#endif

#endif