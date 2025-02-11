//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter>
//   requires LessThanComparable<Iter::value_type>
//   constexpr bool   // constexpr after C++17
//   is_heap(Iter first, Iter last);

#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/functional>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  typedef random_access_iterator<int*> RI;
  int i1[] = {0, 0};
  assert(cuda::std::is_heap(i1, i1, cuda::std::greater<int>()));
  assert(cuda::std::is_heap(i1, i1 + 1, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i1, i1 + 1, cuda::std::greater<int>()) == i1 + 1));
  assert(cuda::std::is_heap(RI(i1), RI(i1), cuda::std::greater<int>()));
  assert(cuda::std::is_heap(RI(i1), RI(i1 + 1), cuda::std::greater<int>())
         == (cuda::std::is_heap_until(RI(i1), RI(i1 + 1), cuda::std::greater<int>()) == RI(i1 + 1)));

  int i2[] = {0, 1};
  int i3[] = {1, 0};
  assert(cuda::std::is_heap(i1, i1 + 2, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i1, i1 + 2, cuda::std::greater<int>()) == i1 + 2));
  assert(cuda::std::is_heap(i2, i2 + 2, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i2, i2 + 2, cuda::std::greater<int>()) == i2 + 2));
  assert(cuda::std::is_heap(i3, i3 + 2, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i3, i3 + 2, cuda::std::greater<int>()) == i3 + 2));
  int i4[]  = {0, 0, 0};
  int i5[]  = {0, 0, 1};
  int i6[]  = {0, 1, 0};
  int i7[]  = {0, 1, 1};
  int i8[]  = {1, 0, 0};
  int i9[]  = {1, 0, 1};
  int i10[] = {1, 1, 0};
  assert(cuda::std::is_heap(i4, i4 + 3, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i4, i4 + 3, cuda::std::greater<int>()) == i4 + 3));
  assert(cuda::std::is_heap(i5, i5 + 3, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i5, i5 + 3, cuda::std::greater<int>()) == i5 + 3));
  assert(cuda::std::is_heap(i6, i6 + 3, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i6, i6 + 3, cuda::std::greater<int>()) == i6 + 3));
  assert(cuda::std::is_heap(i7, i7 + 3, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i7, i7 + 3, cuda::std::greater<int>()) == i7 + 3));
  assert(cuda::std::is_heap(i8, i8 + 3, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i8, i8 + 3, cuda::std::greater<int>()) == i8 + 3));
  assert(cuda::std::is_heap(i9, i9 + 3, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i9, i9 + 3, cuda::std::greater<int>()) == i9 + 3));
  assert(cuda::std::is_heap(i10, i10 + 3, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i10, i10 + 3, cuda::std::greater<int>()) == i10 + 3));
  int i11[] = {0, 0, 0, 0};
  int i12[] = {0, 0, 0, 1};
  int i13[] = {0, 0, 1, 0};
  int i14[] = {0, 0, 1, 1};
  int i15[] = {0, 1, 0, 0};
  int i16[] = {0, 1, 0, 1};
  int i17[] = {0, 1, 1, 0};
  int i18[] = {0, 1, 1, 1};
  int i19[] = {1, 0, 0, 0};
  int i20[] = {1, 0, 0, 1};
  int i21[] = {1, 0, 1, 0};
  int i22[] = {1, 0, 1, 1};
  int i23[] = {1, 1, 0, 0};
  int i24[] = {1, 1, 0, 1};
  int i25[] = {1, 1, 1, 0};
  assert(cuda::std::is_heap(i11, i11 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i11, i11 + 4, cuda::std::greater<int>()) == i11 + 4));
  assert(cuda::std::is_heap(i12, i12 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i12, i12 + 4, cuda::std::greater<int>()) == i12 + 4));
  assert(cuda::std::is_heap(i13, i13 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i13, i13 + 4, cuda::std::greater<int>()) == i13 + 4));
  assert(cuda::std::is_heap(i14, i14 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i14, i14 + 4, cuda::std::greater<int>()) == i14 + 4));
  assert(cuda::std::is_heap(i15, i15 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i15, i15 + 4, cuda::std::greater<int>()) == i15 + 4));
  assert(cuda::std::is_heap(i16, i16 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i16, i16 + 4, cuda::std::greater<int>()) == i16 + 4));
  assert(cuda::std::is_heap(i17, i17 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i17, i17 + 4, cuda::std::greater<int>()) == i17 + 4));
  assert(cuda::std::is_heap(i18, i18 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i18, i18 + 4, cuda::std::greater<int>()) == i18 + 4));
  assert(cuda::std::is_heap(i19, i19 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i19, i19 + 4, cuda::std::greater<int>()) == i19 + 4));
  assert(cuda::std::is_heap(i20, i20 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i20, i20 + 4, cuda::std::greater<int>()) == i20 + 4));
  assert(cuda::std::is_heap(i21, i21 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i21, i21 + 4, cuda::std::greater<int>()) == i21 + 4));
  assert(cuda::std::is_heap(i22, i22 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i22, i22 + 4, cuda::std::greater<int>()) == i22 + 4));
  assert(cuda::std::is_heap(i23, i23 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i23, i23 + 4, cuda::std::greater<int>()) == i23 + 4));
  assert(cuda::std::is_heap(i24, i24 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i24, i24 + 4, cuda::std::greater<int>()) == i24 + 4));
  assert(cuda::std::is_heap(i25, i25 + 4, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i25, i25 + 4, cuda::std::greater<int>()) == i25 + 4));
  int i26[] = {0, 0, 0, 0, 0};
  int i27[] = {0, 0, 0, 0, 1};
  int i28[] = {0, 0, 0, 1, 0};
  int i29[] = {0, 0, 0, 1, 1};
  int i30[] = {0, 0, 1, 0, 0};
  int i31[] = {0, 0, 1, 0, 1};
  int i32[] = {0, 0, 1, 1, 0};
  int i33[] = {0, 0, 1, 1, 1};
  int i34[] = {0, 1, 0, 0, 0};
  int i35[] = {0, 1, 0, 0, 1};
  int i36[] = {0, 1, 0, 1, 0};
  int i37[] = {0, 1, 0, 1, 1};
  int i38[] = {0, 1, 1, 0, 0};
  int i39[] = {0, 1, 1, 0, 1};
  int i40[] = {0, 1, 1, 1, 0};
  int i41[] = {0, 1, 1, 1, 1};
  int i42[] = {1, 0, 0, 0, 0};
  int i43[] = {1, 0, 0, 0, 1};
  int i44[] = {1, 0, 0, 1, 0};
  int i45[] = {1, 0, 0, 1, 1};
  int i46[] = {1, 0, 1, 0, 0};
  int i47[] = {1, 0, 1, 0, 1};
  int i48[] = {1, 0, 1, 1, 0};
  int i49[] = {1, 0, 1, 1, 1};
  int i50[] = {1, 1, 0, 0, 0};
  int i51[] = {1, 1, 0, 0, 1};
  int i52[] = {1, 1, 0, 1, 0};
  int i53[] = {1, 1, 0, 1, 1};
  int i54[] = {1, 1, 1, 0, 0};
  int i55[] = {1, 1, 1, 0, 1};
  int i56[] = {1, 1, 1, 1, 0};
  assert(cuda::std::is_heap(i26, i26 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i26, i26 + 5, cuda::std::greater<int>()) == i26 + 5));
  assert(cuda::std::is_heap(i27, i27 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i27, i27 + 5, cuda::std::greater<int>()) == i27 + 5));
  assert(cuda::std::is_heap(i28, i28 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i28, i28 + 5, cuda::std::greater<int>()) == i28 + 5));
  assert(cuda::std::is_heap(i29, i29 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i29, i29 + 5, cuda::std::greater<int>()) == i29 + 5));
  assert(cuda::std::is_heap(i30, i30 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i30, i30 + 5, cuda::std::greater<int>()) == i30 + 5));
  assert(cuda::std::is_heap(i31, i31 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i31, i31 + 5, cuda::std::greater<int>()) == i31 + 5));
  assert(cuda::std::is_heap(i32, i32 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i32, i32 + 5, cuda::std::greater<int>()) == i32 + 5));
  assert(cuda::std::is_heap(i33, i33 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i33, i33 + 5, cuda::std::greater<int>()) == i33 + 5));
  assert(cuda::std::is_heap(i34, i34 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i34, i34 + 5, cuda::std::greater<int>()) == i34 + 5));
  assert(cuda::std::is_heap(i35, i35 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i35, i35 + 5, cuda::std::greater<int>()) == i35 + 5));
  assert(cuda::std::is_heap(i36, i36 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i36, i36 + 5, cuda::std::greater<int>()) == i36 + 5));
  assert(cuda::std::is_heap(i37, i37 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i37, i37 + 5, cuda::std::greater<int>()) == i37 + 5));
  assert(cuda::std::is_heap(i38, i38 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i38, i38 + 5, cuda::std::greater<int>()) == i38 + 5));
  assert(cuda::std::is_heap(i39, i39 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i39, i39 + 5, cuda::std::greater<int>()) == i39 + 5));
  assert(cuda::std::is_heap(i40, i40 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i40, i40 + 5, cuda::std::greater<int>()) == i40 + 5));
  assert(cuda::std::is_heap(i41, i41 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i41, i41 + 5, cuda::std::greater<int>()) == i41 + 5));
  assert(cuda::std::is_heap(i42, i42 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i42, i42 + 5, cuda::std::greater<int>()) == i42 + 5));
  assert(cuda::std::is_heap(i43, i43 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i43, i43 + 5, cuda::std::greater<int>()) == i43 + 5));
  assert(cuda::std::is_heap(i44, i44 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i44, i44 + 5, cuda::std::greater<int>()) == i44 + 5));
  assert(cuda::std::is_heap(i45, i45 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i45, i45 + 5, cuda::std::greater<int>()) == i45 + 5));
  assert(cuda::std::is_heap(i46, i46 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i46, i46 + 5, cuda::std::greater<int>()) == i46 + 5));
  assert(cuda::std::is_heap(i47, i47 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i47, i47 + 5, cuda::std::greater<int>()) == i47 + 5));
  assert(cuda::std::is_heap(i48, i48 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i48, i48 + 5, cuda::std::greater<int>()) == i48 + 5));
  assert(cuda::std::is_heap(i49, i49 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i49, i49 + 5, cuda::std::greater<int>()) == i49 + 5));
  assert(cuda::std::is_heap(i50, i50 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i50, i50 + 5, cuda::std::greater<int>()) == i50 + 5));
  assert(cuda::std::is_heap(i51, i51 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i51, i51 + 5, cuda::std::greater<int>()) == i51 + 5));
  assert(cuda::std::is_heap(i52, i52 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i52, i52 + 5, cuda::std::greater<int>()) == i52 + 5));
  assert(cuda::std::is_heap(i53, i53 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i53, i53 + 5, cuda::std::greater<int>()) == i53 + 5));
  assert(cuda::std::is_heap(i54, i54 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i54, i54 + 5, cuda::std::greater<int>()) == i54 + 5));
  assert(cuda::std::is_heap(i55, i55 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i55, i55 + 5, cuda::std::greater<int>()) == i55 + 5));
  assert(cuda::std::is_heap(i56, i56 + 5, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i56, i56 + 5, cuda::std::greater<int>()) == i56 + 5));
  int i57[]  = {0, 0, 0, 0, 0, 0};
  int i58[]  = {0, 0, 0, 0, 0, 1};
  int i59[]  = {0, 0, 0, 0, 1, 0};
  int i60[]  = {0, 0, 0, 0, 1, 1};
  int i61[]  = {0, 0, 0, 1, 0, 0};
  int i62[]  = {0, 0, 0, 1, 0, 1};
  int i63[]  = {0, 0, 0, 1, 1, 0};
  int i64[]  = {0, 0, 0, 1, 1, 1};
  int i65[]  = {0, 0, 1, 0, 0, 0};
  int i66[]  = {0, 0, 1, 0, 0, 1};
  int i67[]  = {0, 0, 1, 0, 1, 0};
  int i68[]  = {0, 0, 1, 0, 1, 1};
  int i69[]  = {0, 0, 1, 1, 0, 0};
  int i70[]  = {0, 0, 1, 1, 0, 1};
  int i71[]  = {0, 0, 1, 1, 1, 0};
  int i72[]  = {0, 0, 1, 1, 1, 1};
  int i73[]  = {0, 1, 0, 0, 0, 0};
  int i74[]  = {0, 1, 0, 0, 0, 1};
  int i75[]  = {0, 1, 0, 0, 1, 0};
  int i76[]  = {0, 1, 0, 0, 1, 1};
  int i77[]  = {0, 1, 0, 1, 0, 0};
  int i78[]  = {0, 1, 0, 1, 0, 1};
  int i79[]  = {0, 1, 0, 1, 1, 0};
  int i80[]  = {0, 1, 0, 1, 1, 1};
  int i81[]  = {0, 1, 1, 0, 0, 0};
  int i82[]  = {0, 1, 1, 0, 0, 1};
  int i83[]  = {0, 1, 1, 0, 1, 0};
  int i84[]  = {0, 1, 1, 0, 1, 1};
  int i85[]  = {0, 1, 1, 1, 0, 0};
  int i86[]  = {0, 1, 1, 1, 0, 1};
  int i87[]  = {0, 1, 1, 1, 1, 0};
  int i88[]  = {0, 1, 1, 1, 1, 1};
  int i89[]  = {1, 0, 0, 0, 0, 0};
  int i90[]  = {1, 0, 0, 0, 0, 1};
  int i91[]  = {1, 0, 0, 0, 1, 0};
  int i92[]  = {1, 0, 0, 0, 1, 1};
  int i93[]  = {1, 0, 0, 1, 0, 0};
  int i94[]  = {1, 0, 0, 1, 0, 1};
  int i95[]  = {1, 0, 0, 1, 1, 0};
  int i96[]  = {1, 0, 0, 1, 1, 1};
  int i97[]  = {1, 0, 1, 0, 0, 0};
  int i98[]  = {1, 0, 1, 0, 0, 1};
  int i99[]  = {1, 0, 1, 0, 1, 0};
  int i100[] = {1, 0, 1, 0, 1, 1};
  int i101[] = {1, 0, 1, 1, 0, 0};
  int i102[] = {1, 0, 1, 1, 0, 1};
  int i103[] = {1, 0, 1, 1, 1, 0};
  int i104[] = {1, 0, 1, 1, 1, 1};
  int i105[] = {1, 1, 0, 0, 0, 0};
  int i106[] = {1, 1, 0, 0, 0, 1};
  int i107[] = {1, 1, 0, 0, 1, 0};
  int i108[] = {1, 1, 0, 0, 1, 1};
  int i109[] = {1, 1, 0, 1, 0, 0};
  int i110[] = {1, 1, 0, 1, 0, 1};
  int i111[] = {1, 1, 0, 1, 1, 0};
  int i112[] = {1, 1, 0, 1, 1, 1};
  int i113[] = {1, 1, 1, 0, 0, 0};
  int i114[] = {1, 1, 1, 0, 0, 1};
  int i115[] = {1, 1, 1, 0, 1, 0};
  int i116[] = {1, 1, 1, 0, 1, 1};
  int i117[] = {1, 1, 1, 1, 0, 0};
  int i118[] = {1, 1, 1, 1, 0, 1};
  int i119[] = {1, 1, 1, 1, 1, 0};
  assert(cuda::std::is_heap(i57, i57 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i57, i57 + 6, cuda::std::greater<int>()) == i57 + 6));
  assert(cuda::std::is_heap(i58, i58 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i58, i58 + 6, cuda::std::greater<int>()) == i58 + 6));
  assert(cuda::std::is_heap(i59, i59 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i59, i59 + 6, cuda::std::greater<int>()) == i59 + 6));
  assert(cuda::std::is_heap(i60, i60 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i60, i60 + 6, cuda::std::greater<int>()) == i60 + 6));
  assert(cuda::std::is_heap(i61, i61 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i61, i61 + 6, cuda::std::greater<int>()) == i61 + 6));
  assert(cuda::std::is_heap(i62, i62 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i62, i62 + 6, cuda::std::greater<int>()) == i62 + 6));
  assert(cuda::std::is_heap(i63, i63 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i63, i63 + 6, cuda::std::greater<int>()) == i63 + 6));
  assert(cuda::std::is_heap(i64, i64 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i64, i64 + 6, cuda::std::greater<int>()) == i64 + 6));
  assert(cuda::std::is_heap(i65, i65 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i65, i65 + 6, cuda::std::greater<int>()) == i65 + 6));
  assert(cuda::std::is_heap(i66, i66 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i66, i66 + 6, cuda::std::greater<int>()) == i66 + 6));
  assert(cuda::std::is_heap(i67, i67 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i67, i67 + 6, cuda::std::greater<int>()) == i67 + 6));
  assert(cuda::std::is_heap(i68, i68 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i68, i68 + 6, cuda::std::greater<int>()) == i68 + 6));
  assert(cuda::std::is_heap(i69, i69 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i69, i69 + 6, cuda::std::greater<int>()) == i69 + 6));
  assert(cuda::std::is_heap(i70, i70 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i70, i70 + 6, cuda::std::greater<int>()) == i70 + 6));
  assert(cuda::std::is_heap(i71, i71 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i71, i71 + 6, cuda::std::greater<int>()) == i71 + 6));
  assert(cuda::std::is_heap(i72, i72 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i72, i72 + 6, cuda::std::greater<int>()) == i72 + 6));
  assert(cuda::std::is_heap(i73, i73 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i73, i73 + 6, cuda::std::greater<int>()) == i73 + 6));
  assert(cuda::std::is_heap(i74, i74 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i74, i74 + 6, cuda::std::greater<int>()) == i74 + 6));
  assert(cuda::std::is_heap(i75, i75 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i75, i75 + 6, cuda::std::greater<int>()) == i75 + 6));
  assert(cuda::std::is_heap(i76, i76 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i76, i76 + 6, cuda::std::greater<int>()) == i76 + 6));
  assert(cuda::std::is_heap(i77, i77 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i77, i77 + 6, cuda::std::greater<int>()) == i77 + 6));
  assert(cuda::std::is_heap(i78, i78 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i78, i78 + 6, cuda::std::greater<int>()) == i78 + 6));
  assert(cuda::std::is_heap(i79, i79 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i79, i79 + 6, cuda::std::greater<int>()) == i79 + 6));
  assert(cuda::std::is_heap(i80, i80 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i80, i80 + 6, cuda::std::greater<int>()) == i80 + 6));
  assert(cuda::std::is_heap(i81, i81 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i81, i81 + 6, cuda::std::greater<int>()) == i81 + 6));
  assert(cuda::std::is_heap(i82, i82 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i82, i82 + 6, cuda::std::greater<int>()) == i82 + 6));
  assert(cuda::std::is_heap(i83, i83 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i83, i83 + 6, cuda::std::greater<int>()) == i83 + 6));
  assert(cuda::std::is_heap(i84, i84 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i84, i84 + 6, cuda::std::greater<int>()) == i84 + 6));
  assert(cuda::std::is_heap(i85, i85 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i85, i85 + 6, cuda::std::greater<int>()) == i85 + 6));
  assert(cuda::std::is_heap(i86, i86 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i86, i86 + 6, cuda::std::greater<int>()) == i86 + 6));
  assert(cuda::std::is_heap(i87, i87 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i87, i87 + 6, cuda::std::greater<int>()) == i87 + 6));
  assert(cuda::std::is_heap(i88, i88 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i88, i88 + 6, cuda::std::greater<int>()) == i88 + 6));
  assert(cuda::std::is_heap(i89, i89 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i89, i89 + 6, cuda::std::greater<int>()) == i89 + 6));
  assert(cuda::std::is_heap(i90, i90 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i90, i90 + 6, cuda::std::greater<int>()) == i90 + 6));
  assert(cuda::std::is_heap(i91, i91 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i91, i91 + 6, cuda::std::greater<int>()) == i91 + 6));
  assert(cuda::std::is_heap(i92, i92 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i92, i92 + 6, cuda::std::greater<int>()) == i92 + 6));
  assert(cuda::std::is_heap(i93, i93 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i93, i93 + 6, cuda::std::greater<int>()) == i93 + 6));
  assert(cuda::std::is_heap(i94, i94 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i94, i94 + 6, cuda::std::greater<int>()) == i94 + 6));
  assert(cuda::std::is_heap(i95, i95 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i95, i95 + 6, cuda::std::greater<int>()) == i95 + 6));
  assert(cuda::std::is_heap(i96, i96 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i96, i96 + 6, cuda::std::greater<int>()) == i96 + 6));
  assert(cuda::std::is_heap(i97, i97 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i97, i97 + 6, cuda::std::greater<int>()) == i97 + 6));
  assert(cuda::std::is_heap(i98, i98 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i98, i98 + 6, cuda::std::greater<int>()) == i98 + 6));
  assert(cuda::std::is_heap(i99, i99 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i99, i99 + 6, cuda::std::greater<int>()) == i99 + 6));
  assert(cuda::std::is_heap(i100, i100 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i100, i100 + 6, cuda::std::greater<int>()) == i100 + 6));
  assert(cuda::std::is_heap(i101, i101 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i101, i101 + 6, cuda::std::greater<int>()) == i101 + 6));
  assert(cuda::std::is_heap(i102, i102 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i102, i102 + 6, cuda::std::greater<int>()) == i102 + 6));
  assert(cuda::std::is_heap(i103, i103 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i103, i103 + 6, cuda::std::greater<int>()) == i103 + 6));
  assert(cuda::std::is_heap(i104, i104 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i104, i104 + 6, cuda::std::greater<int>()) == i104 + 6));
  assert(cuda::std::is_heap(i105, i105 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i105, i105 + 6, cuda::std::greater<int>()) == i105 + 6));
  assert(cuda::std::is_heap(i106, i106 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i106, i106 + 6, cuda::std::greater<int>()) == i106 + 6));
  assert(cuda::std::is_heap(i107, i107 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i107, i107 + 6, cuda::std::greater<int>()) == i107 + 6));
  assert(cuda::std::is_heap(i108, i108 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i108, i108 + 6, cuda::std::greater<int>()) == i108 + 6));
  assert(cuda::std::is_heap(i109, i109 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i109, i109 + 6, cuda::std::greater<int>()) == i109 + 6));
  assert(cuda::std::is_heap(i110, i110 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i110, i110 + 6, cuda::std::greater<int>()) == i110 + 6));
  assert(cuda::std::is_heap(i111, i111 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i111, i111 + 6, cuda::std::greater<int>()) == i111 + 6));
  assert(cuda::std::is_heap(i112, i112 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i112, i112 + 6, cuda::std::greater<int>()) == i112 + 6));
  assert(cuda::std::is_heap(i113, i113 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i113, i113 + 6, cuda::std::greater<int>()) == i113 + 6));
  assert(cuda::std::is_heap(i114, i114 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i114, i114 + 6, cuda::std::greater<int>()) == i114 + 6));
  assert(cuda::std::is_heap(i115, i115 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i115, i115 + 6, cuda::std::greater<int>()) == i115 + 6));
  assert(cuda::std::is_heap(i116, i116 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i116, i116 + 6, cuda::std::greater<int>()) == i116 + 6));
  assert(cuda::std::is_heap(i117, i117 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i117, i117 + 6, cuda::std::greater<int>()) == i117 + 6));
  assert(cuda::std::is_heap(i118, i118 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i118, i118 + 6, cuda::std::greater<int>()) == i118 + 6));
  assert(cuda::std::is_heap(i119, i119 + 6, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i119, i119 + 6, cuda::std::greater<int>()) == i119 + 6));
  int i120[] = {0, 0, 0, 0, 0, 0, 0};
  int i121[] = {0, 0, 0, 0, 0, 0, 1};
  int i122[] = {0, 0, 0, 0, 0, 1, 0};
  int i123[] = {0, 0, 0, 0, 0, 1, 1};
  int i124[] = {0, 0, 0, 0, 1, 0, 0};
  int i125[] = {0, 0, 0, 0, 1, 0, 1};
  int i126[] = {0, 0, 0, 0, 1, 1, 0};
  int i127[] = {0, 0, 0, 0, 1, 1, 1};
  int i128[] = {0, 0, 0, 1, 0, 0, 0};
  int i129[] = {0, 0, 0, 1, 0, 0, 1};
  int i130[] = {0, 0, 0, 1, 0, 1, 0};
  int i131[] = {0, 0, 0, 1, 0, 1, 1};
  int i132[] = {0, 0, 0, 1, 1, 0, 0};
  int i133[] = {0, 0, 0, 1, 1, 0, 1};
  int i134[] = {0, 0, 0, 1, 1, 1, 0};
  int i135[] = {0, 0, 0, 1, 1, 1, 1};
  int i136[] = {0, 0, 1, 0, 0, 0, 0};
  int i137[] = {0, 0, 1, 0, 0, 0, 1};
  int i138[] = {0, 0, 1, 0, 0, 1, 0};
  int i139[] = {0, 0, 1, 0, 0, 1, 1};
  int i140[] = {0, 0, 1, 0, 1, 0, 0};
  int i141[] = {0, 0, 1, 0, 1, 0, 1};
  int i142[] = {0, 0, 1, 0, 1, 1, 0};
  int i143[] = {0, 0, 1, 0, 1, 1, 1};
  int i144[] = {0, 0, 1, 1, 0, 0, 0};
  int i145[] = {0, 0, 1, 1, 0, 0, 1};
  int i146[] = {0, 0, 1, 1, 0, 1, 0};
  int i147[] = {0, 0, 1, 1, 0, 1, 1};
  int i148[] = {0, 0, 1, 1, 1, 0, 0};
  int i149[] = {0, 0, 1, 1, 1, 0, 1};
  int i150[] = {0, 0, 1, 1, 1, 1, 0};
  int i151[] = {0, 0, 1, 1, 1, 1, 1};
  int i152[] = {0, 1, 0, 0, 0, 0, 0};
  int i153[] = {0, 1, 0, 0, 0, 0, 1};
  int i154[] = {0, 1, 0, 0, 0, 1, 0};
  int i155[] = {0, 1, 0, 0, 0, 1, 1};
  int i156[] = {0, 1, 0, 0, 1, 0, 0};
  int i157[] = {0, 1, 0, 0, 1, 0, 1};
  int i158[] = {0, 1, 0, 0, 1, 1, 0};
  int i159[] = {0, 1, 0, 0, 1, 1, 1};
  int i160[] = {0, 1, 0, 1, 0, 0, 0};
  int i161[] = {0, 1, 0, 1, 0, 0, 1};
  int i162[] = {0, 1, 0, 1, 0, 1, 0};
  int i163[] = {0, 1, 0, 1, 0, 1, 1};
  int i164[] = {0, 1, 0, 1, 1, 0, 0};
  int i165[] = {0, 1, 0, 1, 1, 0, 1};
  int i166[] = {0, 1, 0, 1, 1, 1, 0};
  int i167[] = {0, 1, 0, 1, 1, 1, 1};
  int i168[] = {0, 1, 1, 0, 0, 0, 0};
  int i169[] = {0, 1, 1, 0, 0, 0, 1};
  int i170[] = {0, 1, 1, 0, 0, 1, 0};
  int i171[] = {0, 1, 1, 0, 0, 1, 1};
  int i172[] = {0, 1, 1, 0, 1, 0, 0};
  int i173[] = {0, 1, 1, 0, 1, 0, 1};
  int i174[] = {0, 1, 1, 0, 1, 1, 0};
  int i175[] = {0, 1, 1, 0, 1, 1, 1};
  int i176[] = {0, 1, 1, 1, 0, 0, 0};
  int i177[] = {0, 1, 1, 1, 0, 0, 1};
  int i178[] = {0, 1, 1, 1, 0, 1, 0};
  int i179[] = {0, 1, 1, 1, 0, 1, 1};
  int i180[] = {0, 1, 1, 1, 1, 0, 0};
  int i181[] = {0, 1, 1, 1, 1, 0, 1};
  int i182[] = {0, 1, 1, 1, 1, 1, 0};
  int i183[] = {0, 1, 1, 1, 1, 1, 1};
  int i184[] = {1, 0, 0, 0, 0, 0, 0};
  int i185[] = {1, 0, 0, 0, 0, 0, 1};
  int i186[] = {1, 0, 0, 0, 0, 1, 0};
  int i187[] = {1, 0, 0, 0, 0, 1, 1};
  int i188[] = {1, 0, 0, 0, 1, 0, 0};
  int i189[] = {1, 0, 0, 0, 1, 0, 1};
  int i190[] = {1, 0, 0, 0, 1, 1, 0};
  int i191[] = {1, 0, 0, 0, 1, 1, 1};
  int i192[] = {1, 0, 0, 1, 0, 0, 0};
  int i193[] = {1, 0, 0, 1, 0, 0, 1};
  int i194[] = {1, 0, 0, 1, 0, 1, 0};
  int i195[] = {1, 0, 0, 1, 0, 1, 1};
  int i196[] = {1, 0, 0, 1, 1, 0, 0};
  int i197[] = {1, 0, 0, 1, 1, 0, 1};
  int i198[] = {1, 0, 0, 1, 1, 1, 0};
  int i199[] = {1, 0, 0, 1, 1, 1, 1};
  int i200[] = {1, 0, 1, 0, 0, 0, 0};
  int i201[] = {1, 0, 1, 0, 0, 0, 1};
  int i202[] = {1, 0, 1, 0, 0, 1, 0};
  int i203[] = {1, 0, 1, 0, 0, 1, 1};
  int i204[] = {1, 0, 1, 0, 1, 0, 0};
  int i205[] = {1, 0, 1, 0, 1, 0, 1};
  int i206[] = {1, 0, 1, 0, 1, 1, 0};
  int i207[] = {1, 0, 1, 0, 1, 1, 1};
  int i208[] = {1, 0, 1, 1, 0, 0, 0};
  int i209[] = {1, 0, 1, 1, 0, 0, 1};
  int i210[] = {1, 0, 1, 1, 0, 1, 0};
  int i211[] = {1, 0, 1, 1, 0, 1, 1};
  int i212[] = {1, 0, 1, 1, 1, 0, 0};
  int i213[] = {1, 0, 1, 1, 1, 0, 1};
  int i214[] = {1, 0, 1, 1, 1, 1, 0};
  int i215[] = {1, 0, 1, 1, 1, 1, 1};
  int i216[] = {1, 1, 0, 0, 0, 0, 0};
  int i217[] = {1, 1, 0, 0, 0, 0, 1};
  int i218[] = {1, 1, 0, 0, 0, 1, 0};
  int i219[] = {1, 1, 0, 0, 0, 1, 1};
  int i220[] = {1, 1, 0, 0, 1, 0, 0};
  int i221[] = {1, 1, 0, 0, 1, 0, 1};
  int i222[] = {1, 1, 0, 0, 1, 1, 0};
  int i223[] = {1, 1, 0, 0, 1, 1, 1};
  int i224[] = {1, 1, 0, 1, 0, 0, 0};
  int i225[] = {1, 1, 0, 1, 0, 0, 1};
  int i226[] = {1, 1, 0, 1, 0, 1, 0};
  int i227[] = {1, 1, 0, 1, 0, 1, 1};
  int i228[] = {1, 1, 0, 1, 1, 0, 0};
  int i229[] = {1, 1, 0, 1, 1, 0, 1};
  int i230[] = {1, 1, 0, 1, 1, 1, 0};
  int i231[] = {1, 1, 0, 1, 1, 1, 1};
  int i232[] = {1, 1, 1, 0, 0, 0, 0};
  int i233[] = {1, 1, 1, 0, 0, 0, 1};
  int i234[] = {1, 1, 1, 0, 0, 1, 0};
  int i235[] = {1, 1, 1, 0, 0, 1, 1};
  int i236[] = {1, 1, 1, 0, 1, 0, 0};
  int i237[] = {1, 1, 1, 0, 1, 0, 1};
  int i238[] = {1, 1, 1, 0, 1, 1, 0};
  int i239[] = {1, 1, 1, 0, 1, 1, 1};
  int i240[] = {1, 1, 1, 1, 0, 0, 0};
  int i241[] = {1, 1, 1, 1, 0, 0, 1};
  int i242[] = {1, 1, 1, 1, 0, 1, 0};
  int i243[] = {1, 1, 1, 1, 0, 1, 1};
  int i244[] = {1, 1, 1, 1, 1, 0, 0};
  int i245[] = {1, 1, 1, 1, 1, 0, 1};
  int i246[] = {1, 1, 1, 1, 1, 1, 0};
  assert(cuda::std::is_heap(i120, i120 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i120, i120 + 7, cuda::std::greater<int>()) == i120 + 7));
  assert(cuda::std::is_heap(i121, i121 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i121, i121 + 7, cuda::std::greater<int>()) == i121 + 7));
  assert(cuda::std::is_heap(i122, i122 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i122, i122 + 7, cuda::std::greater<int>()) == i122 + 7));
  assert(cuda::std::is_heap(i123, i123 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i123, i123 + 7, cuda::std::greater<int>()) == i123 + 7));
  assert(cuda::std::is_heap(i124, i124 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i124, i124 + 7, cuda::std::greater<int>()) == i124 + 7));
  assert(cuda::std::is_heap(i125, i125 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i125, i125 + 7, cuda::std::greater<int>()) == i125 + 7));
  assert(cuda::std::is_heap(i126, i126 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i126, i126 + 7, cuda::std::greater<int>()) == i126 + 7));
  assert(cuda::std::is_heap(i127, i127 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i127, i127 + 7, cuda::std::greater<int>()) == i127 + 7));
  assert(cuda::std::is_heap(i128, i128 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i128, i128 + 7, cuda::std::greater<int>()) == i128 + 7));
  assert(cuda::std::is_heap(i129, i129 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i129, i129 + 7, cuda::std::greater<int>()) == i129 + 7));
  assert(cuda::std::is_heap(i130, i130 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i130, i130 + 7, cuda::std::greater<int>()) == i130 + 7));
  assert(cuda::std::is_heap(i131, i131 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i131, i131 + 7, cuda::std::greater<int>()) == i131 + 7));
  assert(cuda::std::is_heap(i132, i132 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i132, i132 + 7, cuda::std::greater<int>()) == i132 + 7));
  assert(cuda::std::is_heap(i133, i133 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i133, i133 + 7, cuda::std::greater<int>()) == i133 + 7));
  assert(cuda::std::is_heap(i134, i134 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i134, i134 + 7, cuda::std::greater<int>()) == i134 + 7));
  assert(cuda::std::is_heap(i135, i135 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i135, i135 + 7, cuda::std::greater<int>()) == i135 + 7));
  assert(cuda::std::is_heap(i136, i136 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i136, i136 + 7, cuda::std::greater<int>()) == i136 + 7));
  assert(cuda::std::is_heap(i137, i137 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i137, i137 + 7, cuda::std::greater<int>()) == i137 + 7));
  assert(cuda::std::is_heap(i138, i138 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i138, i138 + 7, cuda::std::greater<int>()) == i138 + 7));
  assert(cuda::std::is_heap(i139, i139 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i139, i139 + 7, cuda::std::greater<int>()) == i139 + 7));
  assert(cuda::std::is_heap(i140, i140 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i140, i140 + 7, cuda::std::greater<int>()) == i140 + 7));
  assert(cuda::std::is_heap(i141, i141 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i141, i141 + 7, cuda::std::greater<int>()) == i141 + 7));
  assert(cuda::std::is_heap(i142, i142 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i142, i142 + 7, cuda::std::greater<int>()) == i142 + 7));
  assert(cuda::std::is_heap(i143, i143 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i143, i143 + 7, cuda::std::greater<int>()) == i143 + 7));
  assert(cuda::std::is_heap(i144, i144 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i144, i144 + 7, cuda::std::greater<int>()) == i144 + 7));
  assert(cuda::std::is_heap(i145, i145 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i145, i145 + 7, cuda::std::greater<int>()) == i145 + 7));
  assert(cuda::std::is_heap(i146, i146 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i146, i146 + 7, cuda::std::greater<int>()) == i146 + 7));
  assert(cuda::std::is_heap(i147, i147 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i147, i147 + 7, cuda::std::greater<int>()) == i147 + 7));
  assert(cuda::std::is_heap(i148, i148 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i148, i148 + 7, cuda::std::greater<int>()) == i148 + 7));
  assert(cuda::std::is_heap(i149, i149 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i149, i149 + 7, cuda::std::greater<int>()) == i149 + 7));
  assert(cuda::std::is_heap(i150, i150 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i150, i150 + 7, cuda::std::greater<int>()) == i150 + 7));
  assert(cuda::std::is_heap(i151, i151 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i151, i151 + 7, cuda::std::greater<int>()) == i151 + 7));
  assert(cuda::std::is_heap(i152, i152 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i152, i152 + 7, cuda::std::greater<int>()) == i152 + 7));
  assert(cuda::std::is_heap(i153, i153 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i153, i153 + 7, cuda::std::greater<int>()) == i153 + 7));
  assert(cuda::std::is_heap(i154, i154 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i154, i154 + 7, cuda::std::greater<int>()) == i154 + 7));
  assert(cuda::std::is_heap(i155, i155 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i155, i155 + 7, cuda::std::greater<int>()) == i155 + 7));
  assert(cuda::std::is_heap(i156, i156 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i156, i156 + 7, cuda::std::greater<int>()) == i156 + 7));
  assert(cuda::std::is_heap(i157, i157 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i157, i157 + 7, cuda::std::greater<int>()) == i157 + 7));
  assert(cuda::std::is_heap(i158, i158 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i158, i158 + 7, cuda::std::greater<int>()) == i158 + 7));
  assert(cuda::std::is_heap(i159, i159 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i159, i159 + 7, cuda::std::greater<int>()) == i159 + 7));
  assert(cuda::std::is_heap(i160, i160 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i160, i160 + 7, cuda::std::greater<int>()) == i160 + 7));
  assert(cuda::std::is_heap(i161, i161 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i161, i161 + 7, cuda::std::greater<int>()) == i161 + 7));
  assert(cuda::std::is_heap(i162, i162 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i162, i162 + 7, cuda::std::greater<int>()) == i162 + 7));
  assert(cuda::std::is_heap(i163, i163 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i163, i163 + 7, cuda::std::greater<int>()) == i163 + 7));
  assert(cuda::std::is_heap(i164, i164 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i164, i164 + 7, cuda::std::greater<int>()) == i164 + 7));
  assert(cuda::std::is_heap(i165, i165 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i165, i165 + 7, cuda::std::greater<int>()) == i165 + 7));
  assert(cuda::std::is_heap(i166, i166 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i166, i166 + 7, cuda::std::greater<int>()) == i166 + 7));
  assert(cuda::std::is_heap(i167, i167 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i167, i167 + 7, cuda::std::greater<int>()) == i167 + 7));
  assert(cuda::std::is_heap(i168, i168 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i168, i168 + 7, cuda::std::greater<int>()) == i168 + 7));
  assert(cuda::std::is_heap(i169, i169 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i169, i169 + 7, cuda::std::greater<int>()) == i169 + 7));
  assert(cuda::std::is_heap(i170, i170 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i170, i170 + 7, cuda::std::greater<int>()) == i170 + 7));
  assert(cuda::std::is_heap(i171, i171 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i171, i171 + 7, cuda::std::greater<int>()) == i171 + 7));
  assert(cuda::std::is_heap(i172, i172 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i172, i172 + 7, cuda::std::greater<int>()) == i172 + 7));
  assert(cuda::std::is_heap(i173, i173 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i173, i173 + 7, cuda::std::greater<int>()) == i173 + 7));
  assert(cuda::std::is_heap(i174, i174 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i174, i174 + 7, cuda::std::greater<int>()) == i174 + 7));
  assert(cuda::std::is_heap(i175, i175 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i175, i175 + 7, cuda::std::greater<int>()) == i175 + 7));
  assert(cuda::std::is_heap(i176, i176 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i176, i176 + 7, cuda::std::greater<int>()) == i176 + 7));
  assert(cuda::std::is_heap(i177, i177 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i177, i177 + 7, cuda::std::greater<int>()) == i177 + 7));
  assert(cuda::std::is_heap(i178, i178 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i178, i178 + 7, cuda::std::greater<int>()) == i178 + 7));
  assert(cuda::std::is_heap(i179, i179 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i179, i179 + 7, cuda::std::greater<int>()) == i179 + 7));
  assert(cuda::std::is_heap(i180, i180 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i180, i180 + 7, cuda::std::greater<int>()) == i180 + 7));
  assert(cuda::std::is_heap(i181, i181 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i181, i181 + 7, cuda::std::greater<int>()) == i181 + 7));
  assert(cuda::std::is_heap(i182, i182 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i182, i182 + 7, cuda::std::greater<int>()) == i182 + 7));
  assert(cuda::std::is_heap(i183, i183 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i183, i183 + 7, cuda::std::greater<int>()) == i183 + 7));
  assert(cuda::std::is_heap(i184, i184 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i184, i184 + 7, cuda::std::greater<int>()) == i184 + 7));
  assert(cuda::std::is_heap(i185, i185 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i185, i185 + 7, cuda::std::greater<int>()) == i185 + 7));
  assert(cuda::std::is_heap(i186, i186 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i186, i186 + 7, cuda::std::greater<int>()) == i186 + 7));
  assert(cuda::std::is_heap(i187, i187 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i187, i187 + 7, cuda::std::greater<int>()) == i187 + 7));
  assert(cuda::std::is_heap(i188, i188 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i188, i188 + 7, cuda::std::greater<int>()) == i188 + 7));
  assert(cuda::std::is_heap(i189, i189 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i189, i189 + 7, cuda::std::greater<int>()) == i189 + 7));
  assert(cuda::std::is_heap(i190, i190 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i190, i190 + 7, cuda::std::greater<int>()) == i190 + 7));
  assert(cuda::std::is_heap(i191, i191 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i191, i191 + 7, cuda::std::greater<int>()) == i191 + 7));
  assert(cuda::std::is_heap(i192, i192 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i192, i192 + 7, cuda::std::greater<int>()) == i192 + 7));
  assert(cuda::std::is_heap(i193, i193 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i193, i193 + 7, cuda::std::greater<int>()) == i193 + 7));
  assert(cuda::std::is_heap(i194, i194 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i194, i194 + 7, cuda::std::greater<int>()) == i194 + 7));
  assert(cuda::std::is_heap(i195, i195 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i195, i195 + 7, cuda::std::greater<int>()) == i195 + 7));
  assert(cuda::std::is_heap(i196, i196 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i196, i196 + 7, cuda::std::greater<int>()) == i196 + 7));
  assert(cuda::std::is_heap(i197, i197 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i197, i197 + 7, cuda::std::greater<int>()) == i197 + 7));
  assert(cuda::std::is_heap(i198, i198 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i198, i198 + 7, cuda::std::greater<int>()) == i198 + 7));
  assert(cuda::std::is_heap(i199, i199 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i199, i199 + 7, cuda::std::greater<int>()) == i199 + 7));
  assert(cuda::std::is_heap(i200, i200 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i200, i200 + 7, cuda::std::greater<int>()) == i200 + 7));
  assert(cuda::std::is_heap(i201, i201 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i201, i201 + 7, cuda::std::greater<int>()) == i201 + 7));
  assert(cuda::std::is_heap(i202, i202 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i202, i202 + 7, cuda::std::greater<int>()) == i202 + 7));
  assert(cuda::std::is_heap(i203, i203 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i203, i203 + 7, cuda::std::greater<int>()) == i203 + 7));
  assert(cuda::std::is_heap(i204, i204 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i204, i204 + 7, cuda::std::greater<int>()) == i204 + 7));
  assert(cuda::std::is_heap(i205, i205 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i205, i205 + 7, cuda::std::greater<int>()) == i205 + 7));
  assert(cuda::std::is_heap(i206, i206 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i206, i206 + 7, cuda::std::greater<int>()) == i206 + 7));
  assert(cuda::std::is_heap(i207, i207 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i207, i207 + 7, cuda::std::greater<int>()) == i207 + 7));
  assert(cuda::std::is_heap(i208, i208 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i208, i208 + 7, cuda::std::greater<int>()) == i208 + 7));
  assert(cuda::std::is_heap(i209, i209 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i209, i209 + 7, cuda::std::greater<int>()) == i209 + 7));
  assert(cuda::std::is_heap(i210, i210 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i210, i210 + 7, cuda::std::greater<int>()) == i210 + 7));
  assert(cuda::std::is_heap(i211, i211 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i211, i211 + 7, cuda::std::greater<int>()) == i211 + 7));
  assert(cuda::std::is_heap(i212, i212 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i212, i212 + 7, cuda::std::greater<int>()) == i212 + 7));
  assert(cuda::std::is_heap(i213, i213 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i213, i213 + 7, cuda::std::greater<int>()) == i213 + 7));
  assert(cuda::std::is_heap(i214, i214 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i214, i214 + 7, cuda::std::greater<int>()) == i214 + 7));
  assert(cuda::std::is_heap(i215, i215 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i215, i215 + 7, cuda::std::greater<int>()) == i215 + 7));
  assert(cuda::std::is_heap(i216, i216 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i216, i216 + 7, cuda::std::greater<int>()) == i216 + 7));
  assert(cuda::std::is_heap(i217, i217 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i217, i217 + 7, cuda::std::greater<int>()) == i217 + 7));
  assert(cuda::std::is_heap(i218, i218 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i218, i218 + 7, cuda::std::greater<int>()) == i218 + 7));
  assert(cuda::std::is_heap(i219, i219 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i219, i219 + 7, cuda::std::greater<int>()) == i219 + 7));
  assert(cuda::std::is_heap(i220, i220 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i220, i220 + 7, cuda::std::greater<int>()) == i220 + 7));
  assert(cuda::std::is_heap(i221, i221 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i221, i221 + 7, cuda::std::greater<int>()) == i221 + 7));
  assert(cuda::std::is_heap(i222, i222 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i222, i222 + 7, cuda::std::greater<int>()) == i222 + 7));
  assert(cuda::std::is_heap(i223, i223 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i223, i223 + 7, cuda::std::greater<int>()) == i223 + 7));
  assert(cuda::std::is_heap(i224, i224 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i224, i224 + 7, cuda::std::greater<int>()) == i224 + 7));
  assert(cuda::std::is_heap(i225, i225 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i225, i225 + 7, cuda::std::greater<int>()) == i225 + 7));
  assert(cuda::std::is_heap(i226, i226 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i226, i226 + 7, cuda::std::greater<int>()) == i226 + 7));
  assert(cuda::std::is_heap(i227, i227 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i227, i227 + 7, cuda::std::greater<int>()) == i227 + 7));
  assert(cuda::std::is_heap(i228, i228 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i228, i228 + 7, cuda::std::greater<int>()) == i228 + 7));
  assert(cuda::std::is_heap(i229, i229 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i229, i229 + 7, cuda::std::greater<int>()) == i229 + 7));
  assert(cuda::std::is_heap(i230, i230 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i230, i230 + 7, cuda::std::greater<int>()) == i230 + 7));
  assert(cuda::std::is_heap(i231, i231 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i231, i231 + 7, cuda::std::greater<int>()) == i231 + 7));
  assert(cuda::std::is_heap(i232, i232 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i232, i232 + 7, cuda::std::greater<int>()) == i232 + 7));
  assert(cuda::std::is_heap(i233, i233 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i233, i233 + 7, cuda::std::greater<int>()) == i233 + 7));
  assert(cuda::std::is_heap(i234, i234 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i234, i234 + 7, cuda::std::greater<int>()) == i234 + 7));
  assert(cuda::std::is_heap(i235, i235 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i235, i235 + 7, cuda::std::greater<int>()) == i235 + 7));
  assert(cuda::std::is_heap(i236, i236 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i236, i236 + 7, cuda::std::greater<int>()) == i236 + 7));
  assert(cuda::std::is_heap(i237, i237 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i237, i237 + 7, cuda::std::greater<int>()) == i237 + 7));
  assert(cuda::std::is_heap(i238, i238 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i238, i238 + 7, cuda::std::greater<int>()) == i238 + 7));
  assert(cuda::std::is_heap(i239, i239 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i239, i239 + 7, cuda::std::greater<int>()) == i239 + 7));
  assert(cuda::std::is_heap(i240, i240 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i240, i240 + 7, cuda::std::greater<int>()) == i240 + 7));
  assert(cuda::std::is_heap(i241, i241 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i241, i241 + 7, cuda::std::greater<int>()) == i241 + 7));
  assert(cuda::std::is_heap(i242, i242 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i242, i242 + 7, cuda::std::greater<int>()) == i242 + 7));
  assert(cuda::std::is_heap(i243, i243 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i243, i243 + 7, cuda::std::greater<int>()) == i243 + 7));
  assert(cuda::std::is_heap(i244, i244 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i244, i244 + 7, cuda::std::greater<int>()) == i244 + 7));
  assert(cuda::std::is_heap(i245, i245 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i245, i245 + 7, cuda::std::greater<int>()) == i245 + 7));
  assert(cuda::std::is_heap(i246, i246 + 7, cuda::std::greater<int>())
         == (cuda::std::is_heap_until(i246, i246 + 7, cuda::std::greater<int>()) == i246 + 7));

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif

  return 0;
}
