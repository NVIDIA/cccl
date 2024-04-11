//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// struct iterator_traits
// {
// };

#include <cuda/std/iterator>

#include "test_macros.h"

struct A
{};
struct NotAnIteratorEmpty
{};

struct NotAnIteratorNoDifference
{
  //     typedef int                       difference_type;
  typedef A value_type;
  typedef A* pointer;
  typedef A& reference;
  typedef cuda::std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoValue
{
  typedef int difference_type;
  //     typedef A                         value_type;
  typedef A* pointer;
  typedef A& reference;
  typedef cuda::std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoPointer
{
  typedef int difference_type;
  typedef A value_type;
  //     typedef A*                        pointer;
  typedef A& reference;
  typedef cuda::std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoReference
{
  typedef int difference_type;
  typedef A value_type;
  typedef A* pointer;
  //    typedef A&                        reference;
  typedef cuda::std::forward_iterator_tag iterator_category;
};

struct NotAnIteratorNoCategory
{
  typedef int difference_type;
  typedef A value_type;
  typedef A* pointer;
  typedef A& reference;
  //     typedef cuda::std::forward_iterator_tag iterator_category;
};

int main(int, char**)
{
  {
    typedef cuda::std::iterator_traits<NotAnIteratorEmpty> T;
    typedef T::difference_type DT; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::value_type VT; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::pointer PT; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::reference RT; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }

  {
    typedef cuda::std::iterator_traits<NotAnIteratorNoDifference> T;
    typedef T::difference_type DT; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::value_type VT; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::pointer PT; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::reference RT; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }

  {
    typedef cuda::std::iterator_traits<NotAnIteratorNoValue> T;
    typedef T::difference_type DT; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::value_type VT; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::pointer PT; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::reference RT; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }

  {
    typedef cuda::std::iterator_traits<NotAnIteratorNoPointer> T;
    typedef T::difference_type DT; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::value_type VT; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::pointer PT; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::reference RT; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }

  {
    typedef cuda::std::iterator_traits<NotAnIteratorNoReference> T;
    typedef T::difference_type DT; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::value_type VT; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::pointer PT; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::reference RT; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }

  {
    typedef cuda::std::iterator_traits<NotAnIteratorNoCategory> T;
    typedef T::difference_type DT; // expected-error-re {{no type named 'difference_type' in
                                   // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::value_type VT; // expected-error-re {{no type named 'value_type' in
                              // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::pointer PT; // expected-error-re {{no type named 'pointer' in
                           // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::reference RT; // expected-error-re {{no type named 'reference' in
                             // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
    typedef T::iterator_category CT; // expected-error-re {{no type named 'iterator_category' in
                                     // 'cuda::std::{{.+}}::iterator_traits<{{.+}}>}}
  }

  return 0;
}
