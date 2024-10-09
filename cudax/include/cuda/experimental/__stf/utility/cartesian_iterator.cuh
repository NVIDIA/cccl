//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cudastf/__stf/utility/cuda_attributes.h"
#include "cudastf/__stf/utility/unittest.h"

namespace cuda::experimental::stf {

#if 0
/**
 * @brief Implementation of a cartesian product of multiple iterators
 *
 * This is an experimental class for now, and has no user in the code yet.
 */
template <typename... Iterators>
class CartesianProduct {
public:
    CartesianProduct(Iterators... begins, Iterators... ends) : begins(begins...), ends(ends...), current(begins...) {}

    bool is_end() const { return end_reached; }

    CartesianProduct& operator++() {
        increment_helper(::std::index_sequence_for<Iterators...> {});
        end_reached = (current == ends);
        return *this;
    }

    auto operator*() const { return current; }

    bool operator==(const CartesianProduct& other) const { return current == other.current; }

    bool operator!=(const CartesianProduct& other) const { return !(*this == other); }

private:
    template <::std::size_t... Is>
    void increment_helper(::std::index_sequence<Is...>) {
        (((::std::get<Is>(current) != ::std::get<Is>(ends) &&
                  ++diagonal > ::std::distance(::std::get<Is>(begins), ::std::get<Is>(current)))
                         ? (++::std::get<Is>(current), 0)
                         : 0),
                ...);
    }

    ::std::tuple<Iterators...> begins;
    ::std::tuple<Iterators...> ends;
    ::std::tuple<Iterators...> current;
    bool end_reached = false;

    // Keep track of the total "diagonal" (sum of indices)
    size_t diagonal = 0;
};

#    ifdef UNITTESTED_FILE
// This currently fails...
UNITTEST("cartesian product") {
    ::std::vector<int> numbers { 1, 2, 3 };
    ::std::vector<char> letters { 'a', 'b', 'c' };
    ::std::vector<double> decimals { 0.1, 0.2 };

    auto product =
            CartesianProduct<::std::vector<int>::iterator, ::std::vector<char>::iterator, ::std::vector<double>::iterator>(
                    numbers.begin(), letters.begin(), decimals.begin(), numbers.end(), letters.end(), decimals.end());

    while (!product.is_end()) {
        auto tuple = *product;
        ::std::cout << "(" << *(::std::get<0>(tuple)) << ", " << *(::std::get<1>(tuple)) << ", " << *(::std::get<2>(tuple))
                  << ")" << ::std::endl;

        ++product;
    }
};

#    endif  // UNITTESTED_FILE

#endif

#if 0
class RangeIterator {
public:
    using value_type = size_t;
    CUDASTF_HOST_DEVICE explicit RangeIterator(int value) : currentValue(value) {}

    CUDASTF_HOST_DEVICE int operator*() const { return currentValue; }

    CUDASTF_HOST_DEVICE RangeIterator& operator++() {
        ++currentValue;
        return *this;
    }

    CUDASTF_HOST_DEVICE RangeIterator operator++(int) {
        RangeIterator temp = *this;
        ++(*this);
        return temp;
    }

    CUDASTF_HOST_DEVICE bool operator==(const RangeIterator& other) const { return currentValue == other.currentValue; }

    CUDASTF_HOST_DEVICE bool operator!=(const RangeIterator& other) const { return !(*this == other); }

private:
    int currentValue;
};
#endif

class RangeIterator {
public:
    using value_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;
    using iterator_category = ::std::random_access_iterator_tag;

    CUDASTF_HOST_DEVICE explicit RangeIterator(int value) : currentValue(value) {}

    CUDASTF_HOST_DEVICE value_type operator*() const { return currentValue; }

    CUDASTF_HOST_DEVICE RangeIterator& operator++() {
        ++currentValue;
        return *this;
    }

    CUDASTF_HOST_DEVICE RangeIterator operator++(int) {
        RangeIterator temp = *this;
        ++(*this);
        return temp;
    }

    CUDASTF_HOST_DEVICE RangeIterator& operator--() {
        --currentValue;
        return *this;
    }

    CUDASTF_HOST_DEVICE RangeIterator operator--(int) {
        RangeIterator temp = *this;
        --(*this);
        return temp;
    }

    CUDASTF_HOST_DEVICE RangeIterator& operator+=(difference_type n) {
        currentValue += n;
        return *this;
    }

    CUDASTF_HOST_DEVICE RangeIterator operator+(difference_type n) const {
        RangeIterator result = *this;
        result += n;
        return result;
    }

    CUDASTF_HOST_DEVICE RangeIterator& operator-=(difference_type n) {
        currentValue -= n;
        return *this;
    }

    CUDASTF_HOST_DEVICE RangeIterator operator-(difference_type n) const {
        RangeIterator result = *this;
        result -= n;
        return result;
    }

    CUDASTF_HOST_DEVICE difference_type operator-(const RangeIterator& other) const {
        return currentValue - other.currentValue;
    }

    CUDASTF_HOST_DEVICE value_type operator[](difference_type n) const { return *(*this + n); }

    CUDASTF_HOST_DEVICE bool operator==(const RangeIterator& other) const { return currentValue == other.currentValue; }

    CUDASTF_HOST_DEVICE bool operator!=(const RangeIterator& other) const { return !(*this == other); }

    CUDASTF_HOST_DEVICE bool operator<(const RangeIterator& other) const { return currentValue < other.currentValue; }

    CUDASTF_HOST_DEVICE bool operator>(const RangeIterator& other) const { return currentValue > other.currentValue; }

    CUDASTF_HOST_DEVICE bool operator<=(const RangeIterator& other) const { return currentValue <= other.currentValue; }

    CUDASTF_HOST_DEVICE bool operator>=(const RangeIterator& other) const { return currentValue >= other.currentValue; }

private:
    int currentValue;
};

class Range {
public:
    using value_type = size_t;
    using iterator = RangeIterator;
    using difference_type = iterator::difference_type;

    CUDASTF_HOST_DEVICE explicit Range(int n) : beginIterator(0), endIterator(n) {}

    CUDASTF_HOST_DEVICE iterator begin() const { return beginIterator; }

    CUDASTF_HOST_DEVICE iterator end() const { return endIterator; }

    CUDASTF_HOST_DEVICE value_type size() const { return endIterator - beginIterator; }

    CUDASTF_HOST_DEVICE value_type operator[](difference_type index) const { return *(beginIterator + index); }

private:
    iterator beginIterator;
    iterator endIterator;
};

#ifdef UNITTESTED_FILE
namespace reserved {
template <typename T>
__global__ void unit_test_range_func(T n) {
    Range range(n);
    int sum = 0;
    for (auto i: range) {
        // printf("CUDA %ld\n", i);
        sum++;
    }
    assert(sum == n);
}

}  // namespace reserved

UNITTEST("range") {
    int n = 10;
    Range range(n);

    int check[n];

    for (int num: range) {
        // fprintf(stderr, "->%d\n", num);
        check[num] = 1;
    }

    for (int i = 0; i < n; i++) {
        EXPECT(check[i] == 1);
    }

    reserved::unit_test_range_func<<<1, 1>>>(n);
    cudaDeviceSynchronize();
};
#endif  // UNITTESTED_FILE

class StridedRangeIterator {
public:
    using value_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = const value_type*;
    using reference = const value_type&;
    using iterator_category = ::std::random_access_iterator_tag;

    CUDASTF_HOST_DEVICE explicit StridedRangeIterator(value_type value, difference_type stride)
            : currentValue(value), stride(stride) {}

    CUDASTF_HOST_DEVICE value_type operator*() const { return currentValue; }

    CUDASTF_HOST_DEVICE StridedRangeIterator& operator++() {
        currentValue += stride;
        return *this;
    }

    CUDASTF_HOST_DEVICE StridedRangeIterator operator++(int) {
        StridedRangeIterator temp = *this;
        ++(*this);
        return temp;
    }

    CUDASTF_HOST_DEVICE StridedRangeIterator& operator--() {
        currentValue -= stride;
        return *this;
    }

    CUDASTF_HOST_DEVICE StridedRangeIterator operator--(int) {
        StridedRangeIterator temp = *this;
        --(*this);
        return temp;
    }

    CUDASTF_HOST_DEVICE StridedRangeIterator& operator+=(difference_type n) {
        currentValue += stride * n;
        return *this;
    }

    CUDASTF_HOST_DEVICE StridedRangeIterator& operator-=(difference_type n) {
        currentValue -= stride * n;
        return *this;
    }

    CUDASTF_HOST_DEVICE StridedRangeIterator operator+(difference_type n) const {
        return StridedRangeIterator(currentValue + stride * n, stride);
    }

    CUDASTF_HOST_DEVICE StridedRangeIterator operator-(difference_type n) const {
        return StridedRangeIterator(currentValue - stride * n, stride);
    }

    CUDASTF_HOST_DEVICE difference_type operator-(const StridedRangeIterator& other) const {
        return (currentValue - other.currentValue) / stride;
    }

    CUDASTF_HOST_DEVICE bool operator==(const StridedRangeIterator& other) const {
        return currentValue == other.currentValue;
    }

    CUDASTF_HOST_DEVICE bool operator!=(const StridedRangeIterator& other) const {
        return currentValue != other.currentValue;
    }

    CUDASTF_HOST_DEVICE bool operator<(const StridedRangeIterator& other) const {
        return currentValue < other.currentValue;
    }

    CUDASTF_HOST_DEVICE bool operator>(const StridedRangeIterator& other) const {
        return currentValue > other.currentValue;
    }

    CUDASTF_HOST_DEVICE bool operator<=(const StridedRangeIterator& other) const {
        return currentValue <= other.currentValue;
    }

    CUDASTF_HOST_DEVICE bool operator>=(const StridedRangeIterator& other) const {
        return currentValue >= other.currentValue;
    }

private:
    value_type currentValue;
    difference_type stride;
};

class StridedRange {
public:
    using value_type = size_t;
    using iterator = StridedRangeIterator;
    using difference_type = typename iterator::difference_type;

    CUDASTF_HOST_DEVICE explicit StridedRange(value_type start, value_type end, difference_type stride)
            : start_(start), end_(end), stride_(stride) {
        // Adjust end to the next multiple of stride if needed
        if (start > end) {
            end_ = start_;
        } else if ((end_ - start_) % stride_ != 0) {
            end_ = start_ + stride_ * ((end_ - start_) / stride_ + 1);
        }
    }

    CUDASTF_HOST_DEVICE iterator begin() const { return iterator(start_, stride_); }

    CUDASTF_HOST_DEVICE iterator end() const { return iterator(end_, stride_); }

private:
    value_type start_;
    value_type end_;
    difference_type stride_;
};

#ifdef UNITTESTED_FILE
UNITTEST("StridedRange") {
    StridedRange range(12, 1024, 17);
    size_t cnt = 0;
    size_t expected_cnt = (1024 - 12 + 17 - 1) / 17;

    for (auto it = range.begin(); it != range.end(); ++it) {
        //        ::std::cout << *it << " ";
        EXPECT(cnt < expected_cnt);
        cnt++;
    }

    //    ::std::cout << ::std::endl;

    EXPECT(cnt == expected_cnt);
};

UNITTEST("StridedRange loop") {
    size_t nthreads = 16;
    size_t n = 48;
    size_t cnt = 0;
    for (size_t tid = 0; tid < nthreads; tid++) {
        StridedRange range(tid, n, nthreads);
        //        ::std::cout << "Proc : " << tid << "=>";
        for (auto it = range.begin(); it != range.end(); ++it) {
            //            ::std::cout << *it << " ";
            EXPECT(cnt < n);
            cnt++;
        }
        //        ::std::cout << ::std::endl;
    }

    EXPECT(cnt == n);
};

namespace reserved {
template <typename T>
__global__ void unit_test_strided_range_func(T n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;
    StridedRange r(tid, n, nthreads);
    for (auto i: r) {
        //    printf("CUDA %ld\n", i);
    }
}
}  // namespace reserved

UNITTEST("StridedRange CUDA") {
    size_t n = 100;
    reserved::unit_test_strided_range_func<<<4, 2>>>(n);
    cudaDeviceSynchronize();
};
#endif  // UNITTESTED_FILE

}  // end namespace cuda::experimental::stf
