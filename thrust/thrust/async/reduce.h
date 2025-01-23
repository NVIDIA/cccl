/*
 *  Copyright 2008-2021 NVIDIA Corporation
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
 *  \brief Algorithms for asynchronously reducing a range to a single value.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/cpp14_required.h>

#if _CCCL_STD_VER >= 2014

#  include <thrust/detail/select_system.h>
#  include <thrust/detail/static_assert.h>
#  include <thrust/future.h>
#  include <thrust/system/detail/adl/async/reduce.h>
#  include <thrust/type_traits/is_execution_policy.h>
#  include <thrust/type_traits/logical_metafunctions.h>

#  include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN

namespace async
{

/*! \cond
 */

namespace unimplemented
{

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <typename DerivedPolicy, typename ForwardIt, typename Sentinel, typename T, typename BinaryOp>
CCCL_DEPRECATED _CCCL_HOST future<DerivedPolicy, T>
async_reduce(thrust::execution_policy<DerivedPolicy>&, ForwardIt, Sentinel, T, BinaryOp)
{
  THRUST_STATIC_ASSERT_MSG((thrust::detail::depend_on_instantiation<ForwardIt, false>::value),
                           "this algorithm is not implemented for the specified system");
  return {};
}
_CCCL_SUPPRESS_DEPRECATED_POP

} // namespace unimplemented

namespace reduce_detail
{

using thrust::async::unimplemented::async_reduce;

struct reduce_fn final
{
  _CCCL_SUPPRESS_DEPRECATED_PUSH
  template <typename DerivedPolicy, typename ForwardIt, typename Sentinel, typename T, typename BinaryOp>
  _CCCL_HOST static auto
  call(thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
       ForwardIt&& first,
       Sentinel&& last,
       T&& init,
       BinaryOp&& op)
    // ADL dispatch.
    THRUST_RETURNS(async_reduce(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(init),
      THRUST_FWD(op))) _CCCL_SUPPRESS_DEPRECATED_POP

    _CCCL_SUPPRESS_DEPRECATED_PUSH template <typename DerivedPolicy, typename ForwardIt, typename Sentinel, typename T>
    _CCCL_HOST static auto call4(
      thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
      ForwardIt&& first,
      Sentinel&& last,
      T&& init,
      thrust::true_type)
    // ADL dispatch.
    THRUST_RETURNS(async_reduce(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(init),
      thrust::plus<::cuda::std::remove_cvref_t<T>>{})) _CCCL_SUPPRESS_DEPRECATED_POP

    _CCCL_SUPPRESS_DEPRECATED_PUSH template <typename DerivedPolicy, typename ForwardIt, typename Sentinel>
    _CCCL_HOST static auto call3(thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
                                 ForwardIt&& first,
                                 Sentinel&& last,
                                 thrust::true_type)
    // ADL dispatch.
    THRUST_RETURNS(async_reduce(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      typename iterator_traits<::cuda::std::remove_cvref_t<ForwardIt>>::value_type{},
      thrust::plus<
        ::cuda::std::remove_cvref_t<typename iterator_traits<::cuda::std::remove_cvref_t<ForwardIt>>::value_type>>{}))
      _CCCL_SUPPRESS_DEPRECATED_POP

    template <typename ForwardIt, typename Sentinel, typename T, typename BinaryOp>
    _CCCL_HOST static auto call4(ForwardIt&& first, Sentinel&& last, T&& init, BinaryOp&& op, thrust::false_type)
      THRUST_RETURNS(reduce_fn::call(
        thrust::detail::select_system(typename iterator_system<::cuda::std::remove_cvref_t<ForwardIt>>::type{}),
        THRUST_FWD(first),
        THRUST_FWD(last),
        THRUST_FWD(init),
        THRUST_FWD(op)))

        template <typename ForwardIt, typename Sentinel, typename T>
        _CCCL_HOST static auto call3(ForwardIt&& first, Sentinel&& last, T&& init, thrust::false_type)
          THRUST_RETURNS(reduce_fn::call(
            thrust::detail::select_system(typename iterator_system<::cuda::std::remove_cvref_t<ForwardIt>>::type{}),
            THRUST_FWD(first),
            THRUST_FWD(last),
            THRUST_FWD(init),
            thrust::plus<::cuda::std::remove_cvref_t<T>>{}))

    // MSVC WAR: MSVC gets angsty and eats all available RAM when we try to detect
    // if T1 is an execution_policy by using SFINAE. Switching to a static
    // dispatch pattern to prevent this.
    template <typename T1, typename T2, typename T3>
    _CCCL_HOST static auto call(T1&& t1, T2&& t2, T3&& t3) THRUST_RETURNS(reduce_fn::call3(
      THRUST_FWD(t1), THRUST_FWD(t2), THRUST_FWD(t3), thrust::is_execution_policy<::cuda::std::remove_cvref_t<T1>>{}))

      template <typename T1, typename T2, typename T3, typename T4>
      _CCCL_HOST static auto call(T1&& t1, T2&& t2, T3&& t3, T4&& t4) THRUST_RETURNS(reduce_fn::call4(
        THRUST_FWD(t1),
        THRUST_FWD(t2),
        THRUST_FWD(t3),
        THRUST_FWD(t4),
        thrust::is_execution_policy<::cuda::std::remove_cvref_t<T1>>{}))

        template <typename ForwardIt, typename Sentinel>
        _CCCL_HOST static auto call(ForwardIt&& first, Sentinel&& last) THRUST_RETURNS(reduce_fn::call(
          thrust::detail::select_system(typename iterator_system<::cuda::std::remove_cvref_t<ForwardIt>>::type{}),
          THRUST_FWD(first),
          THRUST_FWD(last),
          typename iterator_traits<::cuda::std::remove_cvref_t<ForwardIt>>::value_type{},
          thrust::plus<::cuda::std::remove_cvref_t<
            typename iterator_traits<::cuda::std::remove_cvref_t<ForwardIt>>::value_type>>{}))

          template <typename... Args>
          CCCL_DEPRECATED _CCCL_NODISCARD _CCCL_HOST auto operator()(Args&&... args) const
    THRUST_RETURNS(call(THRUST_FWD(args)...))
};

} // namespace reduce_detail

// note: cannot add a CCCL_DEPRECATED here because the global variable is emitted into cudafe1.stub.c and we cannot
// suppress the warning there
//! deprecated [Since 2.8.0]
_CCCL_GLOBAL_CONSTANT reduce_detail::reduce_fn reduce{};

///////////////////////////////////////////////////////////////////////////////

namespace unimplemented
{

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <typename DerivedPolicy, typename ForwardIt, typename Sentinel, typename OutputIt, typename T, typename BinaryOp>
CCCL_DEPRECATED _CCCL_HOST event<DerivedPolicy>
async_reduce_into(thrust::execution_policy<DerivedPolicy>&, ForwardIt, Sentinel, OutputIt, T, BinaryOp)
{
  THRUST_STATIC_ASSERT_MSG((thrust::detail::depend_on_instantiation<ForwardIt, false>::value),
                           "this algorithm is not implemented for the specified system");
  return {};
}
_CCCL_SUPPRESS_DEPRECATED_POP

} // namespace unimplemented

namespace reduce_into_detail
{

using thrust::async::unimplemented::async_reduce_into;

struct reduce_into_fn final
{
  _CCCL_SUPPRESS_DEPRECATED_PUSH
  template <typename DerivedPolicy, typename ForwardIt, typename Sentinel, typename OutputIt, typename T, typename BinaryOp>
  _CCCL_HOST static auto
  call(thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
       ForwardIt&& first,
       Sentinel&& last,
       OutputIt&& output,
       T&& init,
       BinaryOp&& op)
    // ADL dispatch.
    THRUST_RETURNS(async_reduce_into(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(output),
      THRUST_FWD(init),
      THRUST_FWD(op))) _CCCL_SUPPRESS_DEPRECATED_POP

    _CCCL_SUPPRESS_DEPRECATED_PUSH
    template <typename DerivedPolicy, typename ForwardIt, typename Sentinel, typename OutputIt, typename T>
    _CCCL_HOST static auto call5(
      thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
      ForwardIt&& first,
      Sentinel&& last,
      OutputIt&& output,
      T&& init,
      thrust::true_type)
    // ADL dispatch.
    THRUST_RETURNS(async_reduce_into(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(output),
      THRUST_FWD(init),
      thrust::plus<::cuda::std::remove_cvref_t<T>>{})) _CCCL_SUPPRESS_DEPRECATED_POP

    _CCCL_SUPPRESS_DEPRECATED_PUSH
    template <typename DerivedPolicy, typename ForwardIt, typename Sentinel, typename OutputIt>
    _CCCL_HOST static auto call4(
      thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
      ForwardIt&& first,
      Sentinel&& last,
      OutputIt&& output,
      thrust::true_type)
    // ADL dispatch.
    THRUST_RETURNS(async_reduce_into(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(output),
      typename iterator_traits<::cuda::std::remove_cvref_t<ForwardIt>>::value_type{},
      thrust::plus<
        ::cuda::std::remove_cvref_t<typename iterator_traits<::cuda::std::remove_cvref_t<ForwardIt>>::value_type>>{}))
      _CCCL_SUPPRESS_DEPRECATED_POP

    template <typename ForwardIt, typename Sentinel, typename OutputIt, typename T, typename BinaryOp>
    _CCCL_HOST static auto call5(
      ForwardIt&& first, Sentinel&& last, OutputIt&& output, T&& init, BinaryOp&& op, thrust::false_type)
      THRUST_RETURNS(reduce_into_fn::call(
        thrust::detail::select_system(typename iterator_system<::cuda::std::remove_cvref_t<ForwardIt>>::type{},
                                      typename iterator_system<::cuda::std::remove_cvref_t<OutputIt>>::type{}),
        THRUST_FWD(first),
        THRUST_FWD(last),
        THRUST_FWD(output),
        THRUST_FWD(init),
        THRUST_FWD(op)))

        template <typename ForwardIt, typename Sentinel, typename OutputIt, typename T>
        _CCCL_HOST static auto call4(ForwardIt&& first, Sentinel&& last, OutputIt&& output, T&& init, thrust::false_type)
          THRUST_RETURNS(reduce_into_fn::call(
            thrust::detail::select_system(typename iterator_system<::cuda::std::remove_cvref_t<ForwardIt>>::type{},
                                          typename iterator_system<::cuda::std::remove_cvref_t<OutputIt>>::type{}),
            THRUST_FWD(first),
            THRUST_FWD(last),
            THRUST_FWD(output),
            THRUST_FWD(init),
            thrust::plus<::cuda::std::remove_cvref_t<T>>{}))

            template <typename ForwardIt, typename Sentinel, typename OutputIt>
            _CCCL_HOST static auto call(ForwardIt&& first, Sentinel&& last, OutputIt&& output)
              THRUST_RETURNS(reduce_into_fn::call(
                thrust::detail::select_system(typename iterator_system<::cuda::std::remove_cvref_t<ForwardIt>>::type{},
                                              typename iterator_system<::cuda::std::remove_cvref_t<OutputIt>>::type{}),
                THRUST_FWD(first),
                THRUST_FWD(last),
                THRUST_FWD(output),
                typename iterator_traits<::cuda::std::remove_cvref_t<ForwardIt>>::value_type{},
                thrust::plus<::cuda::std::remove_cvref_t<
                  typename iterator_traits<::cuda::std::remove_cvref_t<ForwardIt>>::value_type>>{}))

    // MSVC WAR: MSVC gets angsty and eats all available RAM when we try to detect
    // if T1 is an execution_policy by using SFINAE. Switching to a static
    // dispatch pattern to prevent this.
    template <typename T1, typename T2, typename T3, typename T4>
    _CCCL_HOST static auto call(T1&& t1, T2&& t2, T3&& t3, T4&& t4) THRUST_RETURNS(reduce_into_fn::call4(
      THRUST_FWD(t1),
      THRUST_FWD(t2),
      THRUST_FWD(t3),
      THRUST_FWD(t4),
      thrust::is_execution_policy<::cuda::std::remove_cvref_t<T1>>{}))

      template <typename T1, typename T2, typename T3, typename T4, typename T5>
      _CCCL_HOST static auto call(T1&& t1, T2&& t2, T3&& t3, T4&& t4, T5&& t5) THRUST_RETURNS(reduce_into_fn::call5(
        THRUST_FWD(t1),
        THRUST_FWD(t2),
        THRUST_FWD(t3),
        THRUST_FWD(t4),
        THRUST_FWD(t5),
        thrust::is_execution_policy<::cuda::std::remove_cvref_t<T1>>{}))

        template <typename... Args>
        CCCL_DEPRECATED _CCCL_NODISCARD _CCCL_HOST auto operator()(Args&&... args) const
    THRUST_RETURNS(call(THRUST_FWD(args)...))
};

} // namespace reduce_into_detail

// note: cannot add a CCCL_DEPRECATED here because the global variable is emitted into cudafe1.stub.c and we cannot
// suppress the warning there
//! deprecated [Since 2.8.0]
_CCCL_GLOBAL_CONSTANT reduce_into_detail::reduce_into_fn reduce_into{};

/*! \endcond
 */

} // namespace async

THRUST_NAMESPACE_END

#endif
