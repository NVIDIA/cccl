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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/meta.cuh>

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace requirements 
{

// The `operator<(T, T)` must exist
template <class T>
struct requirement : ordered<T, T>
{};

// True if all `Ts...` satisfy `requirement`
template <class... Ts>
struct list_of_requirements : all_t<requirement<Ts>::value...>
{};

template <class... Ts>
struct list_of_requirements_from_unique_categories_impl : ::cuda::std::true_type
{};

template <class T, class... Ts>
struct list_of_requirements_from_unique_categories_impl<T, Ts...>
    : ::cuda::std::integral_constant<
        bool,
        none_t<ordered<T, Ts>::value...>::value && list_of_requirements_from_unique_categories_impl<Ts...>::value>
{};

// True if `Ts...` is a list of requirements, and i-th type in `Ts...` is not ordered with j-th type for all i != j
template <class... Ts>
struct list_of_requirements_from_unique_categories
    : ::cuda::std::integral_constant<
        bool,
        list_of_requirements<Ts...>::value && list_of_requirements_from_unique_categories_impl<Ts...>::value>
{};

template <class CategoryRepresentativeT, class... Ts>
struct first_categorial_match_rev_id : ::cuda::std::integral_constant<::cuda::std::size_t, sizeof...(Ts)>
{};

template <class CategoryRepresentativeT>
struct first_categorial_match_rev_id<CategoryRepresentativeT> : ::cuda::std::integral_constant<::cuda::std::size_t, 0>
{};

template <class CategoryRepresentativeT, class H, class... Ts>
struct first_categorial_match_rev_id<CategoryRepresentativeT, H, Ts...>
    : ::cuda::std::integral_constant<::cuda::std::size_t,
                                     ordered<CategoryRepresentativeT, H>::value
                                       ? sizeof...(Ts) + 1
                                       : first_categorial_match_rev_id<CategoryRepresentativeT, Ts...>::value>
{};

// Index of the first type in `Ts...` that `CategoryRepresentativeT` is ordered with
// If no such type exists, returns `sizeof...(Ts)` (end)
template <class CategoryRepresentativeT, class... Ts>
struct first_categorial_match_id
    : ::cuda::std::integral_constant<::cuda::std::size_t,
                                     sizeof...(Ts) - first_categorial_match_rev_id<CategoryRepresentativeT, Ts...>::value>
{};

template <class CategoryRepresentativeT, ::cuda::std::size_t FirstMatchId, std::size_t E, class... Ts>
struct first_categorial_match_impl
{
  using type = typename ::cuda::std::tuple_element<FirstMatchId, ::cuda::std::tuple<Ts...>>::type;
};

template <class CategoryRepresentativeT, std::size_t E>
struct first_categorial_match_impl<CategoryRepresentativeT, E, E>
{};

template <class CategoryRepresentativeT, class... Ts>
using first_categorial_match =
  typename first_categorial_match_impl<CategoryRepresentativeT,
                                       first_categorial_match_id<CategoryRepresentativeT, Ts...>::value,
                                       sizeof...(Ts),
                                       Ts...>::type;

template <class GuaranteesT, class RequirementsT>
struct requirements_categorically_match_guarantees : ::cuda::std::true_type
{};

template <class RequirementT, class... RequirementsTs, class... GuaranteesTs>
struct requirements_categorically_match_guarantees<::cuda::std::tuple<GuaranteesTs...>,
                                                   ::cuda::std::tuple<RequirementT, RequirementsTs...>>
    : ::cuda::std::integral_constant<
        bool,
        !list_of_requirements_from_unique_categories<RequirementT, GuaranteesTs...>::value
          && requirements_categorically_match_guarantees<::cuda::std::tuple<GuaranteesTs...>,
                                                         ::cuda::std::tuple<RequirementsTs...>>::value>
{};

template <
  class CategoryRepresentativeT,
  class... Ts,
  typename ::cuda::std::enable_if<first_categorial_match_id<CategoryRepresentativeT, Ts...>::value != sizeof...(Ts),
                                  int>::type = 0>
first_categorial_match<CategoryRepresentativeT, Ts...> match(const ::cuda::std::tuple<Ts...>& tpl)
{
  return ::cuda::std::get<first_categorial_match_id<CategoryRepresentativeT, Ts...>::value>(tpl);
}

template <class CategoryRepresentativeT,
          class... GuaranteesTs,
          class... RequirementsTs,
          typename ::cuda::std::enable_if<
            first_categorial_match_id<CategoryRepresentativeT, RequirementsTs...>::value != sizeof...(RequirementsTs),
            int>::type = 0>
first_categorial_match<CategoryRepresentativeT, RequirementsTs...>
masked_value(const ::cuda::std::tuple<GuaranteesTs...>&, const ::cuda::std::tuple<RequirementsTs...>& requirements)
{
  return ::cuda::std::get<first_categorial_match_id<CategoryRepresentativeT, RequirementsTs...>::value>(requirements);
}

template <class CategoryRepresentativeT,
          class... GuaranteesTs,
          class... RequirementsTs,
          typename ::cuda::std::enable_if<
            first_categorial_match_id<CategoryRepresentativeT, RequirementsTs...>::value == sizeof...(RequirementsTs),
            int>::type = 0>
first_categorial_match<CategoryRepresentativeT, GuaranteesTs...>
masked_value(const ::cuda::std::tuple<GuaranteesTs...>& guarantees, const ::cuda::std::tuple<RequirementsTs...>&)
{
  return ::cuda::std::get<first_categorial_match_id<CategoryRepresentativeT, GuaranteesTs...>::value>(guarantees);
}

template <class... GuaranteesTs, class... RequirementsTs>
auto mask(const ::cuda::std::tuple<GuaranteesTs...>& guarantees,
          const ::cuda::std::tuple<RequirementsTs...>& requirements)
  -> decltype(::cuda::std::make_tuple(masked_value<GuaranteesTs>(guarantees, requirements)...))
{
  static_assert(list_of_requirements_from_unique_categories<GuaranteesTs...>::value,
                "Guarantees must be from unique categories");
  static_assert(list_of_requirements_from_unique_categories<RequirementsTs...>::value,
                "Requirements must be from unique categories");
  static_assert(requirements_categorically_match_guarantees<::cuda::std::tuple<GuaranteesTs...>,
                                                            ::cuda::std::tuple<RequirementsTs...>>::value,
                "Requested requirements are not supported by this algorithm");
  // static_assert(!contains_unknown_requirements<std::tuple<Ss...>, std::tuple<GuaranteesTs...>>::value,
  //               "Unknown requirements");
  return ::cuda::std::make_tuple(masked_value<GuaranteesTs>(guarantees, requirements)...);
}

} // namespace requirements

} // namespace detail

template <class... RequirementsTs>
::cuda::std::tuple<RequirementsTs...> require(RequirementsTs... requirements)
{
  static_assert(detail::requirements::list_of_requirements_from_unique_categories<RequirementsTs...>::value,
                "Requirements must be unique");
  return ::cuda::std::make_tuple(requirements...);
}

CUB_NAMESPACE_END
