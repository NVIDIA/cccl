//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <typeindex>

// struct hash<type_index>
// {
//     size_t operator()(type_index index) const;
// };

// UNSUPPORTED: no-rtti

#include <typeindex>
#include <type_traits>

#include "test_macros.h"
#include "poisoned_hash_helper.h"

int main(int, char**)
{
  {
#if TEST_STD_VER <= 2014
    typedef std::hash<std::type_index> H;
    static_assert((std::is_same<typename H::argument_type, std::type_index>::value), "" );
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "" );
#endif
  }
  {
    test_hash_enabled_for_type<std::type_index>(std::type_index(typeid(int)));
  }

  return 0;
}
