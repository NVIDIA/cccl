//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_H
#define ITERATORS_H

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_STDEXCEPT)
#include <cuda/std/stdexcept>
#endif
#include <cuda/std/cstddef>
#include <cuda/std/cassert>

#include "test_macros.h"

#if TEST_STD_VER >= 11
#define DELETE_FUNCTION = delete
#else
#define DELETE_FUNCTION
#endif

template <class It>
class output_iterator
{
    It it_;

    template <class U> friend class output_iterator;
public:
    typedef          cuda::std::output_iterator_tag                  iterator_category;
    typedef void                                               value_type;
    typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename cuda::std::iterator_traits<It>::reference       reference;

__host__ __device__
    It base() const {return it_;}

__host__ __device__
    output_iterator () {}
__host__ __device__
    explicit output_iterator(It it) : it_(it) {}
    template <class U>
__host__ __device__
        output_iterator(const output_iterator<U>& u) :it_(u.it_) {}

__host__ __device__
    reference operator*() const {return *it_;}

__host__ __device__
    output_iterator& operator++() {++it_; return *this;}
__host__ __device__
    output_iterator operator++(int)
        {output_iterator tmp(*this); ++(*this); return tmp;}

    template <class T>
__host__ __device__
    void operator,(T const &) DELETE_FUNCTION;
};

template <class It,
    class ItTraits = It>
class input_iterator
{
    typedef cuda::std::iterator_traits<ItTraits> Traits;
    It it_;

    template <class U, class T> friend class input_iterator;
public:
    typedef          cuda::std::input_iterator_tag                   iterator_category;
    typedef typename Traits::value_type                        value_type;
    typedef typename Traits::difference_type                   difference_type;
    typedef It                                                 pointer;
    typedef typename Traits::reference                         reference;

__host__ __device__
    TEST_CONSTEXPR_CXX14 It base() const {return it_;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 input_iterator() : it_() {}
__host__ __device__
    explicit TEST_CONSTEXPR_CXX14 input_iterator(It it) : it_(it) {}
    template <class U, class T>
__host__ __device__
        TEST_CONSTEXPR_CXX14 input_iterator(const input_iterator<U, T>& u) :it_(u.it_) {}

__host__ __device__
    TEST_CONSTEXPR_CXX14 reference operator*() const {return *it_;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 pointer operator->() const {return it_;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 input_iterator& operator++() {++it_; return *this;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 input_iterator operator++(int)
        {input_iterator tmp(*this); ++(*this); return tmp;}

__host__ __device__
    friend TEST_CONSTEXPR_CXX14 bool operator==(const input_iterator& x, const input_iterator& y)
        {return x.it_ == y.it_;}
__host__ __device__
    friend TEST_CONSTEXPR_CXX14 bool operator!=(const input_iterator& x, const input_iterator& y)
        {return !(x == y);}

    template <class T>
    void operator,(T const &) DELETE_FUNCTION;
};

template <class T, class TV, class U, class UV>
__host__ __device__
inline
bool
operator==(const input_iterator<T, TV>& x, const input_iterator<U, UV>& y)
{
    return x.base() == y.base();
}

template <class T, class TV, class U, class UV>
__host__ __device__
inline
bool
operator!=(const input_iterator<T, TV>& x, const input_iterator<U, UV>& y)
{
    return !(x == y);
}

template <class It>
class forward_iterator
{
    It it_;

    template <class U> friend class forward_iterator;
public:
    typedef          cuda::std::forward_iterator_tag                 iterator_category;
    typedef typename cuda::std::iterator_traits<It>::value_type      value_type;
    typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename cuda::std::iterator_traits<It>::reference       reference;

__host__ __device__
    TEST_CONSTEXPR_CXX14 It base() const {return it_;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 forward_iterator() : it_() {}
__host__ __device__
    explicit TEST_CONSTEXPR_CXX14 forward_iterator(It it) : it_(it) {}
    template <class U>
__host__ __device__
        TEST_CONSTEXPR_CXX14 forward_iterator(const forward_iterator<U>& u) :it_(u.it_) {}

__host__ __device__
    TEST_CONSTEXPR_CXX14 reference operator*() const {return *it_;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 pointer operator->() const {return it_;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 forward_iterator& operator++() {++it_; return *this;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 forward_iterator operator++(int)
        {forward_iterator tmp(*this); ++(*this); return tmp;}

__host__ __device__
    friend TEST_CONSTEXPR_CXX14 bool operator==(const forward_iterator& x, const forward_iterator& y)
        {return x.it_ == y.it_;}
__host__ __device__
    friend TEST_CONSTEXPR_CXX14 bool operator!=(const forward_iterator& x, const forward_iterator& y)
        {return !(x == y);}

    template <class T>
    void operator,(T const &) DELETE_FUNCTION;
};

template <class T, class U>
__host__ __device__
inline
bool TEST_CONSTEXPR_CXX14
operator==(const forward_iterator<T>& x, const forward_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
__host__ __device__
inline
bool TEST_CONSTEXPR_CXX14
operator!=(const forward_iterator<T>& x, const forward_iterator<U>& y)
{
    return !(x == y);
}

template <class It>
class bidirectional_iterator
{
    It it_;

    template <class U> friend class bidirectional_iterator;
public:
    typedef          cuda::std::bidirectional_iterator_tag           iterator_category;
    typedef typename cuda::std::iterator_traits<It>::value_type      value_type;
    typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename cuda::std::iterator_traits<It>::reference       reference;

__host__ __device__
    TEST_CONSTEXPR_CXX14 It base() const {return it_;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 bidirectional_iterator() : it_() {}
__host__ __device__
    explicit TEST_CONSTEXPR_CXX14 bidirectional_iterator(It it) : it_(it) {}
    template <class U>
__host__ __device__
        TEST_CONSTEXPR_CXX14 bidirectional_iterator(const bidirectional_iterator<U>& u) :it_(u.it_) {}

__host__ __device__
    TEST_CONSTEXPR_CXX14 reference operator*() const {return *it_;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 pointer operator->() const {return it_;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 bidirectional_iterator& operator++() {++it_; return *this;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 bidirectional_iterator operator++(int)
        {bidirectional_iterator tmp(*this); ++(*this); return tmp;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 bidirectional_iterator& operator--() {--it_; return *this;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 bidirectional_iterator operator--(int)
        {bidirectional_iterator tmp(*this); --(*this); return tmp;}

    template <class T>
__host__ __device__
    void operator,(T const &) DELETE_FUNCTION;
};

template <class T, class U>
__host__ __device__
inline
bool TEST_CONSTEXPR_CXX14
operator==(const bidirectional_iterator<T>& x, const bidirectional_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
__host__ __device__
inline
bool TEST_CONSTEXPR_CXX14
operator!=(const bidirectional_iterator<T>& x, const bidirectional_iterator<U>& y)
{
    return !(x == y);
}

template <class It>
class random_access_iterator
{
    It it_;

    template <class U> friend class random_access_iterator;
public:
    typedef          cuda::std::random_access_iterator_tag           iterator_category;
    typedef typename cuda::std::iterator_traits<It>::value_type      value_type;
    typedef typename cuda::std::iterator_traits<It>::difference_type difference_type;
    typedef It                                                 pointer;
    typedef typename cuda::std::iterator_traits<It>::reference       reference;

__host__ __device__
    TEST_CONSTEXPR_CXX14 It base() const {return it_;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 random_access_iterator() : it_() {}
__host__ __device__
    explicit TEST_CONSTEXPR_CXX14 random_access_iterator(It it) : it_(it) {}
    template <class U>
__host__ __device__
        TEST_CONSTEXPR_CXX14 random_access_iterator(const random_access_iterator<U>& u) :it_(u.it_) {}

__host__ __device__
    TEST_CONSTEXPR_CXX14 reference operator*() const {return *it_;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 pointer operator->() const {return it_;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 random_access_iterator& operator++() {++it_; return *this;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 random_access_iterator operator++(int)
        {random_access_iterator tmp(*this); ++(*this); return tmp;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 random_access_iterator& operator--() {--it_; return *this;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 random_access_iterator operator--(int)
        {random_access_iterator tmp(*this); --(*this); return tmp;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 random_access_iterator& operator+=(difference_type n) {it_ += n; return *this;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 random_access_iterator operator+(difference_type n) const
        {random_access_iterator tmp(*this); tmp += n; return tmp;}
__host__ __device__
    friend TEST_CONSTEXPR_CXX14 random_access_iterator operator+(difference_type n, random_access_iterator x)
        {x += n; return x;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 random_access_iterator& operator-=(difference_type n) {return *this += -n;}
__host__ __device__
    TEST_CONSTEXPR_CXX14 random_access_iterator operator-(difference_type n) const
        {random_access_iterator tmp(*this); tmp -= n; return tmp;}

__host__ __device__
    TEST_CONSTEXPR_CXX14 reference operator[](difference_type n) const {return it_[n];}

    template <class T>
    void operator,(T const &) DELETE_FUNCTION;
};

template <class T, class U>
__host__ __device__
inline
bool TEST_CONSTEXPR_CXX14
operator==(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return x.base() == y.base();
}

template <class T, class U>
__host__ __device__
inline
bool TEST_CONSTEXPR_CXX14
operator!=(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return !(x == y);
}

template <class T, class U>
__host__ __device__
inline
bool TEST_CONSTEXPR_CXX14
operator<(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return x.base() < y.base();
}

template <class T, class U>
__host__ __device__
inline
bool TEST_CONSTEXPR_CXX14
operator<=(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return !(y < x);
}

template <class T, class U>
__host__ __device__
inline
bool TEST_CONSTEXPR_CXX14
operator>(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return y < x;
}

template <class T, class U>
__host__ __device__
inline
bool TEST_CONSTEXPR_CXX14
operator>=(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return !(x < y);
}

template <class T, class U>
__host__ __device__
inline TEST_CONSTEXPR_CXX14
typename cuda::std::iterator_traits<T>::difference_type
operator-(const random_access_iterator<T>& x, const random_access_iterator<U>& y)
{
    return x.base() - y.base();
}

template <class Iter>
__host__ __device__
inline TEST_CONSTEXPR_CXX14 Iter base(output_iterator<Iter> i) { return i.base(); }

template <class Iter>
__host__ __device__
inline TEST_CONSTEXPR_CXX14 Iter base(input_iterator<Iter> i) { return i.base(); }

template <class Iter>
__host__ __device__
inline TEST_CONSTEXPR_CXX14 Iter base(forward_iterator<Iter> i) { return i.base(); }

template <class Iter>
__host__ __device__
inline TEST_CONSTEXPR_CXX14 Iter base(bidirectional_iterator<Iter> i) { return i.base(); }

template <class Iter>
__host__ __device__
inline TEST_CONSTEXPR_CXX14 Iter base(random_access_iterator<Iter> i) { return i.base(); }

template <class Iter>    // everything else
__host__ __device__
inline TEST_CONSTEXPR_CXX14 Iter base(Iter i) { return i; }

template <typename T>
struct ThrowingIterator {
    typedef cuda::std::bidirectional_iterator_tag iterator_category;
    typedef ptrdiff_t                       difference_type;
    typedef const T                         value_type;
    typedef const T *                       pointer;
    typedef const T &                       reference;

    enum ThrowingAction { TAIncrement, TADecrement, TADereference, TAAssignment, TAComparison };

//  Constructors
__host__ __device__
    ThrowingIterator ()
        : begin_(nullptr), end_(nullptr), current_(nullptr), action_(TADereference), index_(0) {}
__host__ __device__
    ThrowingIterator (const T *first, const T *last, size_t index = 0, ThrowingAction action = TADereference)
        : begin_(first), end_(last), current_(first), action_(action), index_(index) {}
__host__ __device__
    ThrowingIterator (const ThrowingIterator &rhs)
        : begin_(rhs.begin_), end_(rhs.end_), current_(rhs.current_), action_(rhs.action_), index_(rhs.index_) {}
__host__ __device__
    ThrowingIterator & operator= (const ThrowingIterator &rhs)
    {
    if (action_ == TAAssignment)
    {
        if (index_ == 0)
#ifndef TEST_HAS_NO_EXCEPTIONS
            throw cuda::std::runtime_error ("throw from iterator assignment");
#else
            assert(false);
#endif

        else
            --index_;
    }
    begin_   = rhs.begin_;
    end_     = rhs.end_;
    current_ = rhs.current_;
    action_  = rhs.action_;
    index_   = rhs.index_;
    return *this;
    }

//  iterator operations
__host__ __device__
    reference operator*() const
    {
    if (action_ == TADereference)
    {
        if (index_ == 0)
#ifndef TEST_HAS_NO_EXCEPTIONS
            throw cuda::std::runtime_error ("throw from iterator dereference");
#else
            assert(false);
#endif
        else
            --index_;
    }
    return *current_;
    }

__host__ __device__
    ThrowingIterator & operator++()
    {
    if (action_ == TAIncrement)
    {
        if (index_ == 0)
#ifndef TEST_HAS_NO_EXCEPTIONS
            throw cuda::std::runtime_error ("throw from iterator increment");
#else
            assert(false);
#endif
        else
            --index_;
    }
    ++current_;
    return *this;
    }

__host__ __device__
    ThrowingIterator operator++(int)
    {
        ThrowingIterator temp = *this;
        ++(*this);
        return temp;
    }

__host__ __device__
    ThrowingIterator & operator--()
    {
    if (action_ == TADecrement)
    {
        if (index_ == 0)
#ifndef TEST_HAS_NO_EXCEPTIONS
            throw cuda::std::runtime_error ("throw from iterator decrement");
#else
            assert(false);
#endif
        else
            --index_;
    }
    --current_;
    return *this;
    }

__host__ __device__
    ThrowingIterator operator--(int) {
        ThrowingIterator temp = *this;
        --(*this);
        return temp;
    }

__host__ __device__
    bool operator== (const ThrowingIterator &rhs) const
    {
    if (action_ == TAComparison)
    {
        if (index_ == 0)
#ifndef TEST_HAS_NO_EXCEPTIONS
            throw cuda::std::runtime_error ("throw from iterator comparison");
#else
            assert(false);
#endif
        else
            --index_;
    }
    bool atEndL =     current_ == end_;
    bool atEndR = rhs.current_ == rhs.end_;
    if (atEndL != atEndR) return false;  // one is at the end (or empty), the other is not.
    if (atEndL) return true;             // both are at the end (or empty)
    return current_ == rhs.current_;
    }

private:
    const T* begin_;
    const T* end_;
    const T* current_;
    ThrowingAction action_;
    mutable size_t index_;
};

template <typename T>
__host__ __device__
bool operator== (const ThrowingIterator<T>& a, const ThrowingIterator<T>& b)
{   return a.operator==(b); }

template <typename T>
__host__ __device__
bool operator!= (const ThrowingIterator<T>& a, const ThrowingIterator<T>& b)
{   return !a.operator==(b); }

template <typename T>
struct NonThrowingIterator {
    typedef cuda::std::bidirectional_iterator_tag iterator_category;
    typedef ptrdiff_t                       difference_type;
    typedef const T                         value_type;
    typedef const T *                       pointer;
    typedef const T &                       reference;

//  Constructors
__host__ __device__
    NonThrowingIterator ()
        : begin_(nullptr), end_(nullptr), current_(nullptr) {}
__host__ __device__
    NonThrowingIterator (const T *first, const T* last)
        : begin_(first), end_(last), current_(first) {}
__host__ __device__
    NonThrowingIterator (const NonThrowingIterator &rhs)
        : begin_(rhs.begin_), end_(rhs.end_), current_(rhs.current_) {}
__host__ __device__
    NonThrowingIterator & operator= (const NonThrowingIterator &rhs) TEST_NOEXCEPT
    {
    begin_   = rhs.begin_;
    end_     = rhs.end_;
    current_ = rhs.current_;
    return *this;
    }

//  iterator operations
__host__ __device__
    reference operator*() const TEST_NOEXCEPT
    {
    return *current_;
    }

__host__ __device__
    NonThrowingIterator & operator++() TEST_NOEXCEPT
    {
    ++current_;
    return *this;
    }

__host__ __device__
    NonThrowingIterator operator++(int) TEST_NOEXCEPT
    {
        NonThrowingIterator temp = *this;
        ++(*this);
        return temp;
    }

__host__ __device__
    NonThrowingIterator & operator--() TEST_NOEXCEPT
    {
    --current_;
    return *this;
    }

__host__ __device__
    NonThrowingIterator operator--(int) TEST_NOEXCEPT
    {
        NonThrowingIterator temp = *this;
        --(*this);
        return temp;
    }

__host__ __device__
    bool operator== (const NonThrowingIterator &rhs) const TEST_NOEXCEPT
    {
    bool atEndL =     current_ == end_;
    bool atEndR = rhs.current_ == rhs.end_;
    if (atEndL != atEndR) return false;  // one is at the end (or empty), the other is not.
    if (atEndL) return true;             // both are at the end (or empty)
    return current_ == rhs.current_;
    }

private:
    const T* begin_;
    const T* end_;
    const T* current_;
};

template <typename T>
__host__ __device__
bool operator== (const NonThrowingIterator<T>& a, const NonThrowingIterator<T>& b) TEST_NOEXCEPT
{   return a.operator==(b); }

template <typename T>
__host__ __device__
bool operator!= (const NonThrowingIterator<T>& a, const NonThrowingIterator<T>& b) TEST_NOEXCEPT
{   return !a.operator==(b); }

#undef DELETE_FUNCTION

#endif  // ITERATORS_H
