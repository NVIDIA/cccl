//===---------------------------- any.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "any"

namespace std {
const char* bad_any_cast::what() const noexcept {
    return "bad any cast";
}
}


#include <experimental/__config>

//  Preserve std::experimental::any_bad_cast for ABI compatibility
//  Even though it no longer exists in a header file
_LIBCUDACXX_BEGIN_NAMESPACE_LFTS

class _LIBCUDACXX_EXCEPTION_ABI _LIBCUDACXX_AVAILABILITY_BAD_ANY_CAST bad_any_cast : public bad_cast
{
public:
    virtual const char* what() const noexcept;
};

const char* bad_any_cast::what() const noexcept {
    return "bad any cast";
}

_LIBCUDACXX_END_NAMESPACE_LFTS
