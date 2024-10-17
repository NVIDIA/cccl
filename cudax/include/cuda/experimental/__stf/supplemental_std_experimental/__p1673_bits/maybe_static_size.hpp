/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software. //
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MAYBE_STATIC_SIZE_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MAYBE_STATIC_SIZE_HPP_

#include <experimental/mdspan>

namespace std
{
namespace experimental
{
inline namespace __p1673_version_0
{
namespace linalg
{
namespace detail
{

template <class T, T Value, T DynSentinel>
struct __maybe_static_value
{
  MDSPAN_INLINE_FUNCTION constexpr __maybe_static_value(T) noexcept {}
  MDSPAN_INLINE_FUNCTION _MDSPAN_CONSTEXPR_14 __maybe_static_value& operator=(T) noexcept {}

  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr __maybe_static_value() noexcept                            = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr __maybe_static_value(__maybe_static_value const&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED constexpr __maybe_static_value(__maybe_static_value&&) noexcept      = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED __maybe_static_value&
  operator=(__maybe_static_value const&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED _MDSPAN_CONSTEXPR_14_DEFAULTED __maybe_static_value&
  operator=(__maybe_static_value&&) noexcept = default;
  MDSPAN_INLINE_FUNCTION_DEFAULTED
  ~__maybe_static_value() = default;

  static constexpr auto value        = Value;
  static constexpr auto is_static    = true;
  static constexpr auto value_static = Value;
};

template <class T, T DynSentinel>
struct __maybe_static_value<T, DynSentinel, DynSentinel>
{
  T value                            = {};
  static constexpr auto is_static    = false;
  static constexpr auto value_static = DynSentinel;
};

template <::std::size_t StaticSize, ::std::size_t Sentinel = dynamic_extent>
using __maybe_static_extent = __maybe_static_value<::std::size_t, StaticSize, Sentinel>;

} // namespace detail
} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std

#endif // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_MAYBE_STATIC_SIZE_HPP_
