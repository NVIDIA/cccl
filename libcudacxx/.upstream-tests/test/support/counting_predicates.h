//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_COUNTING_PREDICATES_H
#define TEST_SUPPORT_COUNTING_PREDICATES_H

#ifndef __CUDACC_RTC__
#include <cstddef>
#endif

template <typename Predicate, typename Arg>
struct unary_counting_predicate {
public:
    typedef Arg argument_type;
    typedef bool result_type;

    __host__ __device__
    unary_counting_predicate(Predicate p) : p_(p), count_(0) {}
    __host__ __device__
    ~unary_counting_predicate() {}

    __host__ __device__
    bool operator () (const Arg &a) const { ++count_; return p_(a); }
    __host__ __device__
    size_t count() const { return count_; }
    __host__ __device__
    void reset() { count_ = 0; }

private:
    Predicate p_;
    mutable size_t count_;
};


template <typename Predicate, typename Arg1, typename Arg2=Arg1>
struct binary_counting_predicate {
public:
    typedef Arg1 first_argument_type;
    typedef Arg2 second_argument_type;
    typedef bool result_type;

    __host__ __device__
    binary_counting_predicate ( Predicate p ) : p_(p), count_(0) {}
    __host__ __device__
    ~binary_counting_predicate() {}

    __host__ __device__
    bool operator () (const Arg1 &a1, const Arg2 &a2) const { ++count_; return p_(a1, a2); }
    __host__ __device__
    size_t count() const { return count_; }
    __host__ __device__
    void reset() { count_ = 0; }

private:
    Predicate p_;
    mutable size_t count_;
};

#endif // TEST_SUPPORT_COUNTING_PREDICATES_H
