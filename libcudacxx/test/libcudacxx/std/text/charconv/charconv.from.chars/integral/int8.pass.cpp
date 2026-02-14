//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/array>
#include <cuda/std/charconv>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>
#include <cuda/utility>

#include "test_macros.h"

constexpr int first_base = 2;
constexpr int last_base  = 36;

struct TestItem
{
  cuda::std::int8_t val;
  const char* str_signed;
  const char* str_unsigned;
};

template <int Base>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items();

// Source code for the generation of the test items
// #include <iostream>
// #include <charconv>
// #include <cstdint>
// #include <cstddef>
// #include <type_traits>
//
// constexpr std::int8_t list[] = {
//   0,
//   1,
//   2,
//   3,
//   4,
//   7,
//   8,
//   14,
//   16,
//   23,
//   32,
//   36,
//   37,
//   52,
//   60,
//   99,
//   127,
//   -1,
//   -5,
//   -9,
//   -17,
//   -28,
//   -43,
//   -66,
//   -113,
//   -128,
// };
//
// int main()
// {
//   for (int base = 2; base <= 36; ++base)
//   {
//     std::cout <<
// "\n"
// "template <>\n"
// "__host__ __device__ constexpr cuda::std::array<TestItem, " << std::size(list) << "> get_test_items<" << base <<
// ">()\n" <<
// "{\n"
// "  return {{\n";
//     for (auto v : list)
//     {
//       constexpr std::size_t buff_size = 64;
//       char signed_buff[buff_size]{};
//       char unsigned_buff[buff_size]{};
//       std::to_chars(signed_buff, signed_buff + buff_size, v, base);
//       std::to_chars(unsigned_buff, unsigned_buff + buff_size, static_cast<std::make_unsigned_t<decltype(v)>>(v),
//       base);
//
//       std::cout <<
// "    TestItem{" << int(v) << ", \"" << signed_buff << "\", \"" << unsigned_buff << "\"},\n";
//     }
//     std::cout <<
// "  }};\n" <<
// "}\n";
//   }
// }

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<2>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "10", "10"},
    TestItem{3, "11", "11"},
    TestItem{4, "100", "100"},
    TestItem{7, "111", "111"},
    TestItem{8, "1000", "1000"},
    TestItem{14, "1110", "1110"},
    TestItem{16, "10000", "10000"},
    TestItem{23, "10111", "10111"},
    TestItem{32, "100000", "100000"},
    TestItem{36, "100100", "100100"},
    TestItem{37, "100101", "100101"},
    TestItem{52, "110100", "110100"},
    TestItem{60, "111100", "111100"},
    TestItem{99, "1100011", "1100011"},
    TestItem{127, "1111111", "1111111"},
    TestItem{-1, "-1", "11111111"},
    TestItem{-5, "-101", "11111011"},
    TestItem{-9, "-1001", "11110111"},
    TestItem{-17, "-10001", "11101111"},
    TestItem{-28, "-11100", "11100100"},
    TestItem{-43, "-101011", "11010101"},
    TestItem{-66, "-1000010", "10111110"},
    TestItem{-113, "-1110001", "10001111"},
    TestItem{-128, "-10000000", "10000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<3>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{3, "10", "10"},
    TestItem{4, "11", "11"},
    TestItem{7, "21", "21"},
    TestItem{8, "22", "22"},
    TestItem{14, "112", "112"},
    TestItem{16, "121", "121"},
    TestItem{23, "212", "212"},
    TestItem{32, "1012", "1012"},
    TestItem{36, "1100", "1100"},
    TestItem{37, "1101", "1101"},
    TestItem{52, "1221", "1221"},
    TestItem{60, "2020", "2020"},
    TestItem{99, "10200", "10200"},
    TestItem{127, "11201", "11201"},
    TestItem{-1, "-1", "100110"},
    TestItem{-5, "-12", "100022"},
    TestItem{-9, "-100", "100011"},
    TestItem{-17, "-122", "22212"},
    TestItem{-28, "-1001", "22110"},
    TestItem{-43, "-1121", "21220"},
    TestItem{-66, "-2110", "21001"},
    TestItem{-113, "-11012", "12022"},
    TestItem{-128, "-11202", "11202"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<4>()
{
  return {{
    TestItem{0, "0", "0"},           TestItem{1, "1", "1"},           TestItem{2, "2", "2"},
    TestItem{3, "3", "3"},           TestItem{4, "10", "10"},         TestItem{7, "13", "13"},
    TestItem{8, "20", "20"},         TestItem{14, "32", "32"},        TestItem{16, "100", "100"},
    TestItem{23, "113", "113"},      TestItem{32, "200", "200"},      TestItem{36, "210", "210"},
    TestItem{37, "211", "211"},      TestItem{52, "310", "310"},      TestItem{60, "330", "330"},
    TestItem{99, "1203", "1203"},    TestItem{127, "1333", "1333"},   TestItem{-1, "-1", "3333"},
    TestItem{-5, "-11", "3323"},     TestItem{-9, "-21", "3313"},     TestItem{-17, "-101", "3233"},
    TestItem{-28, "-130", "3210"},   TestItem{-43, "-223", "3111"},   TestItem{-66, "-1002", "2332"},
    TestItem{-113, "-1301", "2033"}, TestItem{-128, "-2000", "2000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<5>()
{
  return {{
    TestItem{0, "0", "0"},          TestItem{1, "1", "1"},           TestItem{2, "2", "2"},
    TestItem{3, "3", "3"},          TestItem{4, "4", "4"},           TestItem{7, "12", "12"},
    TestItem{8, "13", "13"},        TestItem{14, "24", "24"},        TestItem{16, "31", "31"},
    TestItem{23, "43", "43"},       TestItem{32, "112", "112"},      TestItem{36, "121", "121"},
    TestItem{37, "122", "122"},     TestItem{52, "202", "202"},      TestItem{60, "220", "220"},
    TestItem{99, "344", "344"},     TestItem{127, "1002", "1002"},   TestItem{-1, "-1", "2010"},
    TestItem{-5, "-10", "2001"},    TestItem{-9, "-14", "1442"},     TestItem{-17, "-32", "1424"},
    TestItem{-28, "-103", "1403"},  TestItem{-43, "-133", "1323"},   TestItem{-66, "-231", "1230"},
    TestItem{-113, "-423", "1033"}, TestItem{-128, "-1003", "1003"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<6>()
{
  return {{
    TestItem{0, "0", "0"},         TestItem{1, "1", "1"},         TestItem{2, "2", "2"},
    TestItem{3, "3", "3"},         TestItem{4, "4", "4"},         TestItem{7, "11", "11"},
    TestItem{8, "12", "12"},       TestItem{14, "22", "22"},      TestItem{16, "24", "24"},
    TestItem{23, "35", "35"},      TestItem{32, "52", "52"},      TestItem{36, "100", "100"},
    TestItem{37, "101", "101"},    TestItem{52, "124", "124"},    TestItem{60, "140", "140"},
    TestItem{99, "243", "243"},    TestItem{127, "331", "331"},   TestItem{-1, "-1", "1103"},
    TestItem{-5, "-5", "1055"},    TestItem{-9, "-13", "1051"},   TestItem{-17, "-25", "1035"},
    TestItem{-28, "-44", "1020"},  TestItem{-43, "-111", "553"},  TestItem{-66, "-150", "514"},
    TestItem{-113, "-305", "355"}, TestItem{-128, "-332", "332"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<7>()
{
  return {{
    TestItem{0, "0", "0"},         TestItem{1, "1", "1"},         TestItem{2, "2", "2"},
    TestItem{3, "3", "3"},         TestItem{4, "4", "4"},         TestItem{7, "10", "10"},
    TestItem{8, "11", "11"},       TestItem{14, "20", "20"},      TestItem{16, "22", "22"},
    TestItem{23, "32", "32"},      TestItem{32, "44", "44"},      TestItem{36, "51", "51"},
    TestItem{37, "52", "52"},      TestItem{52, "103", "103"},    TestItem{60, "114", "114"},
    TestItem{99, "201", "201"},    TestItem{127, "241", "241"},   TestItem{-1, "-1", "513"},
    TestItem{-5, "-5", "506"},     TestItem{-9, "-12", "502"},    TestItem{-17, "-23", "461"},
    TestItem{-28, "-40", "444"},   TestItem{-43, "-61", "423"},   TestItem{-66, "-123", "361"},
    TestItem{-113, "-221", "263"}, TestItem{-128, "-242", "242"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<8>()
{
  return {{
    TestItem{0, "0", "0"},         TestItem{1, "1", "1"},         TestItem{2, "2", "2"},
    TestItem{3, "3", "3"},         TestItem{4, "4", "4"},         TestItem{7, "7", "7"},
    TestItem{8, "10", "10"},       TestItem{14, "16", "16"},      TestItem{16, "20", "20"},
    TestItem{23, "27", "27"},      TestItem{32, "40", "40"},      TestItem{36, "44", "44"},
    TestItem{37, "45", "45"},      TestItem{52, "64", "64"},      TestItem{60, "74", "74"},
    TestItem{99, "143", "143"},    TestItem{127, "177", "177"},   TestItem{-1, "-1", "377"},
    TestItem{-5, "-5", "373"},     TestItem{-9, "-11", "367"},    TestItem{-17, "-21", "357"},
    TestItem{-28, "-34", "344"},   TestItem{-43, "-53", "325"},   TestItem{-66, "-102", "276"},
    TestItem{-113, "-161", "217"}, TestItem{-128, "-200", "200"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<9>()
{
  return {{
    TestItem{0, "0", "0"},         TestItem{1, "1", "1"},         TestItem{2, "2", "2"},
    TestItem{3, "3", "3"},         TestItem{4, "4", "4"},         TestItem{7, "7", "7"},
    TestItem{8, "8", "8"},         TestItem{14, "15", "15"},      TestItem{16, "17", "17"},
    TestItem{23, "25", "25"},      TestItem{32, "35", "35"},      TestItem{36, "40", "40"},
    TestItem{37, "41", "41"},      TestItem{52, "57", "57"},      TestItem{60, "66", "66"},
    TestItem{99, "120", "120"},    TestItem{127, "151", "151"},   TestItem{-1, "-1", "313"},
    TestItem{-5, "-5", "308"},     TestItem{-9, "-10", "304"},    TestItem{-17, "-18", "285"},
    TestItem{-28, "-31", "273"},   TestItem{-43, "-47", "256"},   TestItem{-66, "-73", "231"},
    TestItem{-113, "-135", "168"}, TestItem{-128, "-152", "152"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<10>()
{
  return {{
    TestItem{0, "0", "0"},         TestItem{1, "1", "1"},         TestItem{2, "2", "2"},
    TestItem{3, "3", "3"},         TestItem{4, "4", "4"},         TestItem{7, "7", "7"},
    TestItem{8, "8", "8"},         TestItem{14, "14", "14"},      TestItem{16, "16", "16"},
    TestItem{23, "23", "23"},      TestItem{32, "32", "32"},      TestItem{36, "36", "36"},
    TestItem{37, "37", "37"},      TestItem{52, "52", "52"},      TestItem{60, "60", "60"},
    TestItem{99, "99", "99"},      TestItem{127, "127", "127"},   TestItem{-1, "-1", "255"},
    TestItem{-5, "-5", "251"},     TestItem{-9, "-9", "247"},     TestItem{-17, "-17", "239"},
    TestItem{-28, "-28", "228"},   TestItem{-43, "-43", "213"},   TestItem{-66, "-66", "190"},
    TestItem{-113, "-113", "143"}, TestItem{-128, "-128", "128"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<11>()
{
  return {{
    TestItem{0, "0", "0"},        TestItem{1, "1", "1"},         TestItem{2, "2", "2"},
    TestItem{3, "3", "3"},        TestItem{4, "4", "4"},         TestItem{7, "7", "7"},
    TestItem{8, "8", "8"},        TestItem{14, "13", "13"},      TestItem{16, "15", "15"},
    TestItem{23, "21", "21"},     TestItem{32, "2a", "2a"},      TestItem{36, "33", "33"},
    TestItem{37, "34", "34"},     TestItem{52, "48", "48"},      TestItem{60, "55", "55"},
    TestItem{99, "90", "90"},     TestItem{127, "106", "106"},   TestItem{-1, "-1", "212"},
    TestItem{-5, "-5", "209"},    TestItem{-9, "-9", "205"},     TestItem{-17, "-16", "1a8"},
    TestItem{-28, "-26", "198"},  TestItem{-43, "-3a", "184"},   TestItem{-66, "-60", "163"},
    TestItem{-113, "-a3", "120"}, TestItem{-128, "-107", "107"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<12>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},       TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},       TestItem{14, "12", "12"},
    TestItem{16, "14", "14"},    TestItem{23, "1b", "1b"},    TestItem{32, "28", "28"},    TestItem{36, "30", "30"},
    TestItem{37, "31", "31"},    TestItem{52, "44", "44"},    TestItem{60, "50", "50"},    TestItem{99, "83", "83"},
    TestItem{127, "a7", "a7"},   TestItem{-1, "-1", "193"},   TestItem{-5, "-5", "18b"},   TestItem{-9, "-9", "187"},
    TestItem{-17, "-15", "17b"}, TestItem{-28, "-24", "170"}, TestItem{-43, "-37", "159"}, TestItem{-66, "-56", "13a"},
    TestItem{-113, "-95", "bb"}, TestItem{-128, "-a8", "a8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<13>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},       TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},       TestItem{14, "11", "11"},
    TestItem{16, "13", "13"},    TestItem{23, "1a", "1a"},    TestItem{32, "26", "26"},    TestItem{36, "2a", "2a"},
    TestItem{37, "2b", "2b"},    TestItem{52, "40", "40"},    TestItem{60, "48", "48"},    TestItem{99, "78", "78"},
    TestItem{127, "9a", "9a"},   TestItem{-1, "-1", "168"},   TestItem{-5, "-5", "164"},   TestItem{-9, "-9", "160"},
    TestItem{-17, "-14", "155"}, TestItem{-28, "-22", "147"}, TestItem{-43, "-34", "135"}, TestItem{-66, "-51", "118"},
    TestItem{-113, "-89", "b0"}, TestItem{-128, "-9b", "9b"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<14>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},       TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},       TestItem{14, "10", "10"},
    TestItem{16, "12", "12"},    TestItem{23, "19", "19"},    TestItem{32, "24", "24"},    TestItem{36, "28", "28"},
    TestItem{37, "29", "29"},    TestItem{52, "3a", "3a"},    TestItem{60, "44", "44"},    TestItem{99, "71", "71"},
    TestItem{127, "91", "91"},   TestItem{-1, "-1", "143"},   TestItem{-5, "-5", "13d"},   TestItem{-9, "-9", "139"},
    TestItem{-17, "-13", "131"}, TestItem{-28, "-20", "124"}, TestItem{-43, "-31", "113"}, TestItem{-66, "-4a", "d8"},
    TestItem{-113, "-81", "a3"}, TestItem{-128, "-92", "92"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<15>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "11", "11"},    TestItem{23, "18", "18"},    TestItem{32, "22", "22"},   TestItem{36, "26", "26"},
    TestItem{37, "27", "27"},    TestItem{52, "37", "37"},    TestItem{60, "40", "40"},   TestItem{99, "69", "69"},
    TestItem{127, "87", "87"},   TestItem{-1, "-1", "120"},   TestItem{-5, "-5", "11b"},  TestItem{-9, "-9", "117"},
    TestItem{-17, "-12", "10e"}, TestItem{-28, "-1d", "103"}, TestItem{-43, "-2d", "e3"}, TestItem{-66, "-46", "ca"},
    TestItem{-113, "-78", "98"}, TestItem{-128, "-88", "88"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<16>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "10", "10"},    TestItem{23, "17", "17"},    TestItem{32, "20", "20"},   TestItem{36, "24", "24"},
    TestItem{37, "25", "25"},    TestItem{52, "34", "34"},    TestItem{60, "3c", "3c"},   TestItem{99, "63", "63"},
    TestItem{127, "7f", "7f"},   TestItem{-1, "-1", "ff"},    TestItem{-5, "-5", "fb"},   TestItem{-9, "-9", "f7"},
    TestItem{-17, "-11", "ef"},  TestItem{-28, "-1c", "e4"},  TestItem{-43, "-2b", "d5"}, TestItem{-66, "-42", "be"},
    TestItem{-113, "-71", "8f"}, TestItem{-128, "-80", "80"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<17>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "16", "16"},    TestItem{32, "1f", "1f"},   TestItem{36, "22", "22"},
    TestItem{37, "23", "23"},    TestItem{52, "31", "31"},    TestItem{60, "39", "39"},   TestItem{99, "5e", "5e"},
    TestItem{127, "78", "78"},   TestItem{-1, "-1", "f0"},    TestItem{-5, "-5", "ed"},   TestItem{-9, "-9", "e9"},
    TestItem{-17, "-10", "e1"},  TestItem{-28, "-1b", "d7"},  TestItem{-43, "-29", "c9"}, TestItem{-66, "-3f", "b3"},
    TestItem{-113, "-6b", "87"}, TestItem{-128, "-79", "79"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<18>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "15", "15"},    TestItem{32, "1e", "1e"},   TestItem{36, "20", "20"},
    TestItem{37, "21", "21"},    TestItem{52, "2g", "2g"},    TestItem{60, "36", "36"},   TestItem{99, "59", "59"},
    TestItem{127, "71", "71"},   TestItem{-1, "-1", "e3"},    TestItem{-5, "-5", "dh"},   TestItem{-9, "-9", "dd"},
    TestItem{-17, "-h", "d5"},   TestItem{-28, "-1a", "cc"},  TestItem{-43, "-27", "bf"}, TestItem{-66, "-3c", "aa"},
    TestItem{-113, "-65", "7h"}, TestItem{-128, "-72", "72"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<19>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "14", "14"},    TestItem{32, "1d", "1d"},   TestItem{36, "1h", "1h"},
    TestItem{37, "1i", "1i"},    TestItem{52, "2e", "2e"},    TestItem{60, "33", "33"},   TestItem{99, "54", "54"},
    TestItem{127, "6d", "6d"},   TestItem{-1, "-1", "d8"},    TestItem{-5, "-5", "d4"},   TestItem{-9, "-9", "d0"},
    TestItem{-17, "-h", "cb"},   TestItem{-28, "-19", "c0"},  TestItem{-43, "-25", "b4"}, TestItem{-66, "-39", "a0"},
    TestItem{-113, "-5i", "7a"}, TestItem{-128, "-6e", "6e"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<20>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "13", "13"},    TestItem{32, "1c", "1c"},   TestItem{36, "1g", "1g"},
    TestItem{37, "1h", "1h"},    TestItem{52, "2c", "2c"},    TestItem{60, "30", "30"},   TestItem{99, "4j", "4j"},
    TestItem{127, "67", "67"},   TestItem{-1, "-1", "cf"},    TestItem{-5, "-5", "cb"},   TestItem{-9, "-9", "c7"},
    TestItem{-17, "-h", "bj"},   TestItem{-28, "-18", "b8"},  TestItem{-43, "-23", "ad"}, TestItem{-66, "-36", "9a"},
    TestItem{-113, "-5d", "73"}, TestItem{-128, "-68", "68"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<21>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "12", "12"},    TestItem{32, "1b", "1b"},   TestItem{36, "1f", "1f"},
    TestItem{37, "1g", "1g"},    TestItem{52, "2a", "2a"},    TestItem{60, "2i", "2i"},   TestItem{99, "4f", "4f"},
    TestItem{127, "61", "61"},   TestItem{-1, "-1", "c3"},    TestItem{-5, "-5", "bk"},   TestItem{-9, "-9", "bg"},
    TestItem{-17, "-h", "b8"},   TestItem{-28, "-17", "ai"},  TestItem{-43, "-21", "a3"}, TestItem{-66, "-33", "91"},
    TestItem{-113, "-58", "6h"}, TestItem{-128, "-62", "62"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<22>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "11", "11"},    TestItem{32, "1a", "1a"},   TestItem{36, "1e", "1e"},
    TestItem{37, "1f", "1f"},    TestItem{52, "28", "28"},    TestItem{60, "2g", "2g"},   TestItem{99, "4b", "4b"},
    TestItem{127, "5h", "5h"},   TestItem{-1, "-1", "bd"},    TestItem{-5, "-5", "b9"},   TestItem{-9, "-9", "b5"},
    TestItem{-17, "-h", "aj"},   TestItem{-28, "-16", "a8"},  TestItem{-43, "-1l", "9f"}, TestItem{-66, "-30", "8e"},
    TestItem{-113, "-53", "6b"}, TestItem{-128, "-5i", "5i"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<23>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "10", "10"},    TestItem{32, "19", "19"},   TestItem{36, "1d", "1d"},
    TestItem{37, "1e", "1e"},    TestItem{52, "26", "26"},    TestItem{60, "2e", "2e"},   TestItem{99, "47", "47"},
    TestItem{127, "5c", "5c"},   TestItem{-1, "-1", "b2"},    TestItem{-5, "-5", "al"},   TestItem{-9, "-9", "ah"},
    TestItem{-17, "-h", "a9"},   TestItem{-28, "-15", "9l"},  TestItem{-43, "-1k", "96"}, TestItem{-66, "-2k", "86"},
    TestItem{-113, "-4l", "65"}, TestItem{-128, "-5d", "5d"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<24>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "18", "18"},   TestItem{36, "1c", "1c"},
    TestItem{37, "1d", "1d"},    TestItem{52, "24", "24"},    TestItem{60, "2c", "2c"},   TestItem{99, "43", "43"},
    TestItem{127, "57", "57"},   TestItem{-1, "-1", "af"},    TestItem{-5, "-5", "ab"},   TestItem{-9, "-9", "a7"},
    TestItem{-17, "-h", "9n"},   TestItem{-28, "-14", "9c"},  TestItem{-43, "-1j", "8l"}, TestItem{-66, "-2i", "7m"},
    TestItem{-113, "-4h", "5n"}, TestItem{-128, "-58", "58"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<25>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "17", "17"},   TestItem{36, "1b", "1b"},
    TestItem{37, "1c", "1c"},    TestItem{52, "22", "22"},    TestItem{60, "2a", "2a"},   TestItem{99, "3o", "3o"},
    TestItem{127, "52", "52"},   TestItem{-1, "-1", "a5"},    TestItem{-5, "-5", "a1"},   TestItem{-9, "-9", "9m"},
    TestItem{-17, "-h", "9e"},   TestItem{-28, "-13", "93"},  TestItem{-43, "-1i", "8d"}, TestItem{-66, "-2g", "7f"},
    TestItem{-113, "-4d", "5i"}, TestItem{-128, "-53", "53"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<26>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "16", "16"},   TestItem{36, "1a", "1a"},
    TestItem{37, "1b", "1b"},    TestItem{52, "20", "20"},    TestItem{60, "28", "28"},   TestItem{99, "3l", "3l"},
    TestItem{127, "4n", "4n"},   TestItem{-1, "-1", "9l"},    TestItem{-5, "-5", "9h"},   TestItem{-9, "-9", "9d"},
    TestItem{-17, "-h", "95"},   TestItem{-28, "-12", "8k"},  TestItem{-43, "-1h", "85"}, TestItem{-66, "-2e", "78"},
    TestItem{-113, "-49", "5d"}, TestItem{-128, "-4o", "4o"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<27>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "15", "15"},   TestItem{36, "19", "19"},
    TestItem{37, "1a", "1a"},    TestItem{52, "1p", "1p"},    TestItem{60, "26", "26"},   TestItem{99, "3i", "3i"},
    TestItem{127, "4j", "4j"},   TestItem{-1, "-1", "9c"},    TestItem{-5, "-5", "98"},   TestItem{-9, "-9", "94"},
    TestItem{-17, "-h", "8n"},   TestItem{-28, "-11", "8c"},  TestItem{-43, "-1g", "7o"}, TestItem{-66, "-2c", "71"},
    TestItem{-113, "-45", "58"}, TestItem{-128, "-4k", "4k"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<28>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "14", "14"},   TestItem{36, "18", "18"},
    TestItem{37, "19", "19"},    TestItem{52, "1o", "1o"},    TestItem{60, "24", "24"},   TestItem{99, "3f", "3f"},
    TestItem{127, "4f", "4f"},   TestItem{-1, "-1", "93"},    TestItem{-5, "-5", "8r"},   TestItem{-9, "-9", "8n"},
    TestItem{-17, "-h", "8f"},   TestItem{-28, "-10", "84"},  TestItem{-43, "-1f", "7h"}, TestItem{-66, "-2a", "6m"},
    TestItem{-113, "-41", "53"}, TestItem{-128, "-4g", "4g"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<29>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "13", "13"},   TestItem{36, "17", "17"},
    TestItem{37, "18", "18"},    TestItem{52, "1n", "1n"},    TestItem{60, "22", "22"},   TestItem{99, "3c", "3c"},
    TestItem{127, "4b", "4b"},   TestItem{-1, "-1", "8n"},    TestItem{-5, "-5", "8j"},   TestItem{-9, "-9", "8f"},
    TestItem{-17, "-h", "87"},   TestItem{-28, "-s", "7p"},   TestItem{-43, "-1e", "7a"}, TestItem{-66, "-28", "6g"},
    TestItem{-113, "-3q", "4r"}, TestItem{-128, "-4c", "4c"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<30>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "12", "12"},   TestItem{36, "16", "16"},
    TestItem{37, "17", "17"},    TestItem{52, "1m", "1m"},    TestItem{60, "20", "20"},   TestItem{99, "39", "39"},
    TestItem{127, "47", "47"},   TestItem{-1, "-1", "8f"},    TestItem{-5, "-5", "8b"},   TestItem{-9, "-9", "87"},
    TestItem{-17, "-h", "7t"},   TestItem{-28, "-s", "7i"},   TestItem{-43, "-1d", "73"}, TestItem{-66, "-26", "6a"},
    TestItem{-113, "-3n", "4n"}, TestItem{-128, "-48", "48"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<31>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "11", "11"},   TestItem{36, "15", "15"},
    TestItem{37, "16", "16"},    TestItem{52, "1l", "1l"},    TestItem{60, "1t", "1t"},   TestItem{99, "36", "36"},
    TestItem{127, "43", "43"},   TestItem{-1, "-1", "87"},    TestItem{-5, "-5", "83"},   TestItem{-9, "-9", "7u"},
    TestItem{-17, "-h", "7m"},   TestItem{-28, "-s", "7b"},   TestItem{-43, "-1c", "6r"}, TestItem{-66, "-24", "64"},
    TestItem{-113, "-3k", "4j"}, TestItem{-128, "-44", "44"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<32>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "10", "10"},   TestItem{36, "14", "14"},
    TestItem{37, "15", "15"},    TestItem{52, "1k", "1k"},    TestItem{60, "1s", "1s"},   TestItem{99, "33", "33"},
    TestItem{127, "3v", "3v"},   TestItem{-1, "-1", "7v"},    TestItem{-5, "-5", "7r"},   TestItem{-9, "-9", "7n"},
    TestItem{-17, "-h", "7f"},   TestItem{-28, "-s", "74"},   TestItem{-43, "-1b", "6l"}, TestItem{-66, "-22", "5u"},
    TestItem{-113, "-3h", "4f"}, TestItem{-128, "-40", "40"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<33>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "w", "w"},     TestItem{36, "13", "13"},
    TestItem{37, "14", "14"},    TestItem{52, "1j", "1j"},    TestItem{60, "1r", "1r"},   TestItem{99, "30", "30"},
    TestItem{127, "3s", "3s"},   TestItem{-1, "-1", "7o"},    TestItem{-5, "-5", "7k"},   TestItem{-9, "-9", "7g"},
    TestItem{-17, "-h", "78"},   TestItem{-28, "-s", "6u"},   TestItem{-43, "-1a", "6f"}, TestItem{-66, "-20", "5p"},
    TestItem{-113, "-3e", "4b"}, TestItem{-128, "-3t", "3t"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<34>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "w", "w"},     TestItem{36, "12", "12"},
    TestItem{37, "13", "13"},    TestItem{52, "1i", "1i"},    TestItem{60, "1q", "1q"},   TestItem{99, "2v", "2v"},
    TestItem{127, "3p", "3p"},   TestItem{-1, "-1", "7h"},    TestItem{-5, "-5", "7d"},   TestItem{-9, "-9", "79"},
    TestItem{-17, "-h", "71"},   TestItem{-28, "-s", "6o"},   TestItem{-43, "-19", "69"}, TestItem{-66, "-1w", "5k"},
    TestItem{-113, "-3b", "47"}, TestItem{-128, "-3q", "3q"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<35>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "w", "w"},     TestItem{36, "11", "11"},
    TestItem{37, "12", "12"},    TestItem{52, "1h", "1h"},    TestItem{60, "1p", "1p"},   TestItem{99, "2t", "2t"},
    TestItem{127, "3m", "3m"},   TestItem{-1, "-1", "7a"},    TestItem{-5, "-5", "76"},   TestItem{-9, "-9", "72"},
    TestItem{-17, "-h", "6t"},   TestItem{-28, "-s", "6i"},   TestItem{-43, "-18", "63"}, TestItem{-66, "-1v", "5f"},
    TestItem{-113, "-38", "43"}, TestItem{-128, "-3n", "3n"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 26> get_test_items<36>()
{
  return {{
    TestItem{0, "0", "0"},       TestItem{1, "1", "1"},       TestItem{2, "2", "2"},      TestItem{3, "3", "3"},
    TestItem{4, "4", "4"},       TestItem{7, "7", "7"},       TestItem{8, "8", "8"},      TestItem{14, "e", "e"},
    TestItem{16, "g", "g"},      TestItem{23, "n", "n"},      TestItem{32, "w", "w"},     TestItem{36, "10", "10"},
    TestItem{37, "11", "11"},    TestItem{52, "1g", "1g"},    TestItem{60, "1o", "1o"},   TestItem{99, "2r", "2r"},
    TestItem{127, "3j", "3j"},   TestItem{-1, "-1", "73"},    TestItem{-5, "-5", "6z"},   TestItem{-9, "-9", "6v"},
    TestItem{-17, "-h", "6n"},   TestItem{-28, "-s", "6c"},   TestItem{-43, "-17", "5x"}, TestItem{-66, "-1u", "5a"},
    TestItem{-113, "-35", "3z"}, TestItem{-128, "-3k", "3k"},
  }};
}

template <class T>
__host__ __device__ constexpr void test_from_chars(
  const char* data,
  cuda::std::ptrdiff_t size,
  int base,
  cuda::std::ptrdiff_t exp_ptr_offset,
  cuda::std::errc exp_errc,
  T exp_val = {})
{
  constexpr cuda::std::size_t buff_size = 150;
  constexpr auto init_val               = static_cast<T>(-23);

  char buff[buff_size]{};
  cuda::std::strncpy(buff, data, size);

  {
    T value(init_val);
    const auto result = cuda::std::from_chars(buff, buff + size, value, base);

    // Check if the result matches the expected result
    assert(result.ptr == buff + exp_ptr_offset);
    assert(result.ec == exp_errc);

    // Check if the value matches the expected value
    assert((exp_errc == cuda::std::errc{}) ? (value == exp_val) : (value == init_val));
  }

  // If the base is greater than 10, we need to handle case sensitivity for letters
  if (base > 10)
  {
    for (cuda::std::ptrdiff_t i = 0; i < size; ++i)
    {
      if (buff[i] >= 'a' && buff[i] <= 'z')
      {
        buff[i] -= 'a' - 'A'; // Convert lowercase to uppercase
      }
    }

    T value(init_val);
    const auto result = cuda::std::from_chars(buff, buff + size, value, base);

    // Check if the result matches the expected result
    assert(result.ptr == buff + exp_ptr_offset);
    assert(result.ec == exp_errc);

    // Check if the value matches the expected value
    assert((exp_errc == cuda::std::errc{}) ? (value == exp_val) : (value == init_val));
  }
}

template <class T>
__host__ __device__ constexpr void test_from_chars(const TestItem& item, int base, bool overflow = false)
{
  static_assert(
    cuda::std::is_same_v<
      cuda::std::from_chars_result,
      decltype(cuda::std::from_chars(
        cuda::std::declval<const char*>(), cuda::std::declval<const char*>(), cuda::std::declval<T&>(), int{}))>);
  static_assert(noexcept(cuda::std::from_chars(
    cuda::std::declval<const char*>(), cuda::std::declval<const char*>(), cuda::std::declval<T&>(), int{})));

  const auto ref_val = static_cast<T>(item.val);
  const char* str    = (cuda::std::is_signed_v<T>) ? item.str_signed : item.str_unsigned;
  const auto len     = cuda::std::strlen(str);

  constexpr cuda::std::size_t buff_size = 150;
  char buff[buff_size]{};

  const auto success_errc = (overflow) ? cuda::std::errc::result_out_of_range : cuda::std::errc{};

  // 1. Test the original string
  cuda::std::strncpy(buff, str, len);
  test_from_chars<T>(buff, len, base, len, success_errc, ref_val);

  // 2. Test a string that has invalid characters at the end
  cuda::std::strncpy(buff, str, len);
  buff[len] = '9' + 1;
  test_from_chars<T>(buff, len + 1, base, len, success_errc, ref_val);

  // 3. Test a string that has invalid characters at the beginning
  buff[0] = '9' + 1;
  cuda::std::strncpy(buff + 1, str, len);
  test_from_chars<T>(buff, len + 1, base, 0, cuda::std::errc::invalid_argument);

  // 3. Test a string that has white characters at the beginning
  buff[0] = ' ';
  cuda::std::strncpy(buff + 1, str, len);
  test_from_chars<T>(buff, len + 1, base, 0, cuda::std::errc::invalid_argument);

  // 5. Test a string that has '+' at the beginning
  buff[0] = '+';
  cuda::std::strncpy(buff + 1, str, len);
  test_from_chars<T>(buff, len + 1, base, 0, cuda::std::errc::invalid_argument);

  // 6. Test a string that has '-' at the beginning (unsigned only)
  if constexpr (cuda::std::is_unsigned_v<T>)
  {
    buff[0] = '-';
    cuda::std::strncpy(buff + 1, str, len);
    test_from_chars<T>(buff, len + 1, base, 0, cuda::std::errc::invalid_argument);
  }
}

template <class T>
__host__ __device__ constexpr void test_overflow()
{
  constexpr int base = 10;

  TestItem item{};

  // 1. Test `max + 1` overflow
  {
    item.str_signed   = "128";
    item.str_unsigned = "256";
    test_from_chars<T>(item, base, true);
  }

  // 2. Test positive huge value overflow
  {
    item.str_signed   = "12390981233333333333333333333";
    item.str_unsigned = "12390981233333333333333333333";
    test_from_chars<T>(item, base, true);
  }

  // 3. Test `min - 1` overflow
  if constexpr (cuda::std::is_signed_v<T>)
  {
    item.str_signed   = "-129";
    item.str_unsigned = "";
    test_from_chars<T>(item, base, true);
  }

  // 4. Test negative huge value overflow
  if constexpr (cuda::std::is_signed_v<T>)
  {
    item.str_signed   = "-12390981233333333333333333333";
    item.str_unsigned = "";
    test_from_chars<T>(item, base, true);
  }
}

template <int Base>
__host__ __device__ constexpr bool test_base()
{
  constexpr auto items = get_test_items<Base>();

  for (const auto& item : items)
  {
    test_from_chars<char>(item, Base);
    test_from_chars<cuda::std::int8_t>(item, Base);
    test_from_chars<cuda::std::uint8_t>(item, Base);
  }

  // Test overflow cases (base 10 only)
  if constexpr (Base == 10)
  {
    test_overflow<char>();
    test_overflow<cuda::std::int8_t>();
    test_overflow<cuda::std::uint8_t>();
  }

  return true;
}

struct TestBaseInvoker
{
  template <int Base>
  __host__ __device__ constexpr void operator()(cuda::std::integral_constant<int, Base>) const
  {
    test_base<Base>();
    static_assert(test_base<Base>());
  }
};

__host__ __device__ constexpr void test()
{
  cuda::static_for<int, first_base, last_base + 1>(TestBaseInvoker{});
}

int main(int, char**)
{
  test();
  return 0;
}
