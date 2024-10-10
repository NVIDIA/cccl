/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_TAGS_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_TAGS_HPP_

#include <experimental/mdspan>

namespace std
{
namespace experimental
{
inline namespace __p1673_version_0
{
namespace linalg
{

// TODO @proposal-bug make sure these can't convert from `{}`

struct column_major_t
{};
_MDSPAN_INLINE_VARIABLE constexpr auto column_major = column_major_t{};
struct row_major_t
{};
_MDSPAN_INLINE_VARIABLE constexpr auto row_major = row_major_t{};

struct upper_triangle_t
{};
_MDSPAN_INLINE_VARIABLE constexpr auto upper_triangle = upper_triangle_t{};
struct lower_triangle_t
{};
_MDSPAN_INLINE_VARIABLE constexpr auto lower_triangle = lower_triangle_t{};

struct implicit_unit_diagonal_t
{};
_MDSPAN_INLINE_VARIABLE constexpr auto implicit_unit_diagonal = implicit_unit_diagonal_t{};
struct explicit_diagonal_t
{};
_MDSPAN_INLINE_VARIABLE constexpr auto explicit_diagonal = explicit_diagonal_t{};

struct left_side_t
{};
_MDSPAN_INLINE_VARIABLE constexpr auto left_side = left_side_t{};
struct right_side_t
{};
_MDSPAN_INLINE_VARIABLE constexpr auto right_side = right_side_t{};

} // namespace linalg
} // namespace __p1673_version_0
} // namespace experimental
} // namespace std

#endif // LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_LAYOUT_TAGS_HPP_
