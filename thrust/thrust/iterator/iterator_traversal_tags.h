// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

THRUST_NAMESPACE_BEGIN

//! \addtogroup iterators
//! \addtogroup iterator_traversal_tags Iterator Traversal Tags
//! \ingroup iterators
//! \{

// define Boost's traversal tags

//! Tag type for iterators allowing no traversal.
struct no_traversal_tag
{};

//! Tag type for iterators allowing incrementable traversal.
struct incrementable_traversal_tag : no_traversal_tag
{};

//! Tag type for iterators allowing single pass traversal.
struct single_pass_traversal_tag : incrementable_traversal_tag
{};

//! Tag type for iterators allowing forward traversal.
struct forward_traversal_tag : single_pass_traversal_tag
{};

//! Tag type for iterators allowing bidirectional traversal.
struct bidirectional_traversal_tag : forward_traversal_tag
{};

//! Tag type for iterators allowing random access traversal.
struct random_access_traversal_tag : bidirectional_traversal_tag
{};

//! \} // end iterator_traversal_tags

THRUST_NAMESPACE_END
