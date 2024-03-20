/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>
#include "cub/detail/meta.cuh"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/requirements.cuh>

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace guarantees
{

template <class... GuaranteesTs>
using guarantees_t = ::cuda::std::tuple<GuaranteesTs...>;

enum class determinism_t
{
  not_guaranteed,
  run_to_run
};

template <determinism_t Guarantee>
struct determinism_holder_t
{
  static constexpr determinism_t value = Guarantee;
};

template <determinism_t L, determinism_t R>
_CCCL_HOST_DEVICE constexpr bool operator<(determinism_holder_t<L>, determinism_holder_t<R>)
{
  return L < R;
}

using run_to_run_deterministic_t   = determinism_holder_t<determinism_t::run_to_run>;
using determinism_not_guaranteed_t = determinism_holder_t<determinism_t::not_guaranteed>;

template <class GuaranteesT, class RequirementsT>
struct statically_satisfy : ::cuda::std::false_type
{};

template <class... GuaranteesTs, class... RequirementsTs>
struct statically_satisfy<::cuda::std::tuple<GuaranteesTs...>, ::cuda::std::tuple<RequirementsTs...>>
    : ::cuda::std::integral_constant<bool,
                                     all_t<statically_less<RequirementsTs, GuaranteesTs>::value
                                           || statically_equal<RequirementsTs, GuaranteesTs>::value
                                           || !statically_ordered<GuaranteesTs, RequirementsTs>::value...>::value>
{};

} // namespace guarantees

} // namespace detail

inline namespace guarantees
{

// TODO CCCL version of `_LIBCUDACXX_CPO_ACCESSIBILITY`
_LIBCUDACXX_CPO_ACCESSIBILITY detail::guarantees::run_to_run_deterministic_t run_to_run_determinism{};
_LIBCUDACXX_CPO_ACCESSIBILITY detail::guarantees::determinism_not_guaranteed_t nondeterminism{};

} // namespace guarantees

CUB_NAMESPACE_END
