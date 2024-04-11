//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <cuda/std/span>

//  template<class Container>
//    constexpr explicit(Extent != dynamic_extent) span(Container&);
//  template<class Container>
//    constexpr explicit(Extent != dynamic_extent) span(Container const&);

// This test checks for libc++'s non-conforming temporary extension to cuda::std::span
// to support construction from containers that look like contiguous ranges.
//
// This extension is only supported when we don't ship <ranges>, and we can
// remove it once we get rid of _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES.

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

//  Look ma - I'm a container!
template <typename T>
struct IsAContainer
{
  __host__ __device__ constexpr IsAContainer()
      : v_{}
  {}
  __host__ __device__ constexpr size_t size() const
  {
    return 1;
  }
  __host__ __device__ constexpr T* data()
  {
    return &v_;
  }
  __host__ __device__ constexpr const T* data() const
  {
    return &v_;
  }

  __host__ __device__ constexpr const T* getV() const
  {
    return &v_;
  } // for checking
  T v_;
};

template <typename T>
struct NotAContainerNoData
{
  __host__ __device__ size_t size() const
  {
    return 0;
  }
};

template <typename T>
struct NotAContainerNoSize
{
  __host__ __device__ const T* data() const
  {
    return nullptr;
  }
};

template <typename T>
struct NotAContainerPrivate
{
private:
  __host__ __device__ size_t size() const
  {
    return 0;
  }
  __host__ __device__ const T* data() const
  {
    return nullptr;
  }
};

template <class T, size_t extent, class container>
__host__ __device__ cuda::std::span<T, extent> createImplicitSpan(container c)
{
  return {c}; // expected-error-re {{no matching constructor for initialization of 'cuda::std::span<{{.+}}>'}}
}

int main(int, char**)
{
  //  Making non-const spans from const sources (a temporary binds to `const &`)
  {
    cuda::std::span<int> s1{IsAContainer<int>()}; // expected-error {{no matching constructor for initialization of
                                                  // 'cuda::std::span<int>'}}
  }

  //  Missing size and/or data
  {
    cuda::std::span<const int> s1{NotAContainerNoData<int>()}; // expected-error {{no matching constructor for
                                                               // initialization of 'cuda::std::span<const int>'}}
    cuda::std::span<const int> s3{NotAContainerNoSize<int>()}; // expected-error {{no matching constructor for
                                                               // initialization of 'cuda::std::span<const int>'}}
    cuda::std::span<const int> s5{NotAContainerPrivate<int>()}; // expected-error {{no matching constructor for
                                                                // initialization of 'cuda::std::span<const int>'}}
  }

  //  Not the same type
  {
    IsAContainer<int> c;
    cuda::std::span<float> s1{c}; // expected-error {{no matching constructor for initialization of
                                  // 'cuda::std::span<float>'}}
  }

  //  CV wrong
  {
    IsAContainer<const int> c;
    IsAContainer<const volatile int> cv;
    IsAContainer<volatile int> v;

    cuda::std::span<int> s1{c}; // expected-error {{no matching constructor for initialization of
                                // 'cuda::std::span<int>'}}
    cuda::std::span<int> s2{v}; // expected-error {{no matching constructor for initialization of
                                // 'cuda::std::span<int>'}}
    cuda::std::span<int> s3{cv}; // expected-error {{no matching constructor for initialization of
                                 // 'cuda::std::span<int>'}}
    cuda::std::span<const int> s4{v}; // expected-error {{no matching constructor for initialization of
                                      // 'cuda::std::span<const int>'}}
    cuda::std::span<const int> s5{cv}; // expected-error {{no matching constructor for initialization of
                                       // 'cuda::std::span<const int>'}}
    cuda::std::span<volatile int> s6{c}; // expected-error {{no matching constructor for initialization of
                                         // 'cuda::std::span<volatile int>'}}
    cuda::std::span<volatile int> s7{cv}; // expected-error {{no matching constructor for initialization of
                                          // 'cuda::std::span<volatile int>'}}
  }

  // explicit constructor necessary
  {
    IsAContainer<int> c;
    const IsAContainer<int> cc;

    createImplicitSpan<int, 1>(c);
    createImplicitSpan<int, 1>(cc);
  }

  return 0;
}
