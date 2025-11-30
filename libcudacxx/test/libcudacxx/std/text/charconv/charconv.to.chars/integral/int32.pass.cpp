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
  cuda::std::int32_t val;
  const char* str_signed;
  const char* str_unsigned;
};

template <int Base>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items();

// Source code for the generation of the test items
// #include <iostream>
// #include <charconv>
// #include <cstdint>
// #include <cstddef>
// #include <type_traits>
// #include <limits>
//
// constexpr std::int32_t list[] = {
//   0,
//   1,
//   9,
//   194,
//   814,
//   6'322,
//   12'345,
//   33'312,
//   901'369,
//   1'579'542,
//   123'345'345,
//   2'147'483'647,
//   -1,
//   -437,
//   -1'459,
//   -8'103,
//   -90'000,
//   -790'301,
//   -8'999'099,
//   -542'185'444,
//   -2'147'483'647 - 1,
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
//       constexpr std::size_t buff_size = 150;
//       char signed_buff[buff_size]{};
//       char unsigned_buff[buff_size]{};
//       std::to_chars(signed_buff, signed_buff + buff_size, v, base);
//       std::to_chars(unsigned_buff, unsigned_buff + buff_size, static_cast<std::make_unsigned_t<decltype(v)>>(v),
//       base);
//
//       std::cout <<
// "    TestItem{";
//       if (v == std::numeric_limits<decltype(v)>::min())
//       {
//         std::cout << (-std::numeric_limits<decltype(v)>::max()) << " - 1";
//       }
//       else
//       {
//         std::cout << v;
//       }
//       std::cout << ", \"" << signed_buff << "\", \"" << unsigned_buff << "\"},\n";
//     }
//     std::cout <<
// "  }};\n" <<
// "}\n";
//   }
// }

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<2>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "1001", "1001"},
    TestItem{194, "11000010", "11000010"},
    TestItem{814, "1100101110", "1100101110"},
    TestItem{6322, "1100010110010", "1100010110010"},
    TestItem{12345, "11000000111001", "11000000111001"},
    TestItem{33312, "1000001000100000", "1000001000100000"},
    TestItem{901369, "11011100000011111001", "11011100000011111001"},
    TestItem{1579542, "110000001101000010110", "110000001101000010110"},
    TestItem{123345345, "111010110100001100111000001", "111010110100001100111000001"},
    TestItem{2147483647, "1111111111111111111111111111111", "1111111111111111111111111111111"},
    TestItem{-1, "-1", "11111111111111111111111111111111"},
    TestItem{-437, "-110110101", "11111111111111111111111001001011"},
    TestItem{-1459, "-10110110011", "11111111111111111111101001001101"},
    TestItem{-8103, "-1111110100111", "11111111111111111110000001011001"},
    TestItem{-90000, "-10101111110010000", "11111111111111101010000001110000"},
    TestItem{-790301, "-11000000111100011101", "11111111111100111111000011100011"},
    TestItem{-8999099, "-100010010101000010111011", "11111111011101101010111101000101"},
    TestItem{-542185444, "-100000010100010001011111100100", "11011111101011101110100000011100"},
    TestItem{-2147483647 - 1, "-10000000000000000000000000000000", "10000000000000000000000000000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<3>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "100", "100"},
    TestItem{194, "21012", "21012"},
    TestItem{814, "1010011", "1010011"},
    TestItem{6322, "22200011", "22200011"},
    TestItem{12345, "121221020", "121221020"},
    TestItem{33312, "1200200210", "1200200210"},
    TestItem{901369, "1200210110001", "1200210110001"},
    TestItem{1579542, "2222020201120", "2222020201120"},
    TestItem{123345345, "22121002121000010", "22121002121000010"},
    TestItem{2147483647, "12112122212110202101", "12112122212110202101"},
    TestItem{-1, "-1", "102002022201221111210"},
    TestItem{-437, "-121012", "102002022201220220122"},
    TestItem{-1459, "-2000001", "102002022201212111210"},
    TestItem{-8103, "-102010010", "102002022201112101201"},
    TestItem{-90000, "-11120110100", "102002022120101001111"},
    TestItem{-790301, "-1111011002102", "102002021020210102102"},
    TestItem{-8999099, "-121221012102222", "102001200210202001212"},
    TestItem{-542185444, "-1101210012212121101", "100200112112001220110"},
    TestItem{-2147483647 - 1, "-12112122212110202102", "12112122212110202102"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<4>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "21", "21"},
    TestItem{194, "3002", "3002"},
    TestItem{814, "30232", "30232"},
    TestItem{6322, "1202302", "1202302"},
    TestItem{12345, "3000321", "3000321"},
    TestItem{33312, "20020200", "20020200"},
    TestItem{901369, "3130003321", "3130003321"},
    TestItem{1579542, "12001220112", "12001220112"},
    TestItem{123345345, "13112201213001", "13112201213001"},
    TestItem{2147483647, "1333333333333333", "1333333333333333"},
    TestItem{-1, "-1", "3333333333333333"},
    TestItem{-437, "-12311", "3333333333321023"},
    TestItem{-1459, "-112303", "3333333333221031"},
    TestItem{-8103, "-1332213", "3333333332001121"},
    TestItem{-90000, "-111332100", "3333333222001300"},
    TestItem{-790301, "-3000330131", "3333330333003203"},
    TestItem{-8999099, "-202111002323", "3333131222331011"},
    TestItem{-542185444, "-200110101133210", "3133223232200130"},
    TestItem{-2147483647 - 1, "-2000000000000000", "2000000000000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<5>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "14", "14"},
    TestItem{194, "1234", "1234"},
    TestItem{814, "11224", "11224"},
    TestItem{6322, "200242", "200242"},
    TestItem{12345, "343340", "343340"},
    TestItem{33312, "2031222", "2031222"},
    TestItem{901369, "212320434", "212320434"},
    TestItem{1579542, "401021132", "401021132"},
    TestItem{123345345, "223034022340", "223034022340"},
    TestItem{2147483647, "13344223434042", "13344223434042"},
    TestItem{-1, "-1", "32244002423140"},
    TestItem{-437, "-3222", "32244002414414"},
    TestItem{-1459, "-21314", "32244002401322"},
    TestItem{-8103, "-224403", "32244002143233"},
    TestItem{-90000, "-10340000", "32243442033141"},
    TestItem{-790301, "-200242201", "32243302130440"},
    TestItem{-8999099, "-4300432344", "32234201440242"},
    TestItem{-542185444, "-2102244413234", "30141203004402"},
    TestItem{-2147483647 - 1, "-13344223434043", "13344223434043"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<6>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "13", "13"},
    TestItem{194, "522", "522"},
    TestItem{814, "3434", "3434"},
    TestItem{6322, "45134", "45134"},
    TestItem{12345, "133053", "133053"},
    TestItem{33312, "414120", "414120"},
    TestItem{901369, "31153001", "31153001"},
    TestItem{1579542, "53504410", "53504410"},
    TestItem{123345345, "20123415133", "20123415133"},
    TestItem{2147483647, "553032005531", "553032005531"},
    TestItem{-1, "-1", "1550104015503"},
    TestItem{-437, "-2005", "1550104013455"},
    TestItem{-1459, "-10431", "1550104005033"},
    TestItem{-8103, "-101303", "1550103514201"},
    TestItem{-90000, "-1532400", "1550102043104"},
    TestItem{-790301, "-24534445", "1550035041015"},
    TestItem{-8999099, "-520514255", "1545143101205"},
    TestItem{-542185444, "-125444525444", "1420215050020"},
    TestItem{-2147483647 - 1, "-553032005532", "553032005532"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<7>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "12", "12"},
    TestItem{194, "365", "365"},
    TestItem{814, "2242", "2242"},
    TestItem{6322, "24301", "24301"},
    TestItem{12345, "50664", "50664"},
    TestItem{33312, "166056", "166056"},
    TestItem{901369, "10442620", "10442620"},
    TestItem{1579542, "16266036", "16266036"},
    TestItem{123345345, "3025263264", "3025263264"},
    TestItem{2147483647, "104134211161", "104134211161"},
    TestItem{-1, "-1", "211301422353"},
    TestItem{-437, "-1163", "211301421161"},
    TestItem{-1459, "-4153", "211301415201"},
    TestItem{-8103, "-32424", "211301356630"},
    TestItem{-90000, "-523251", "211300566103"},
    TestItem{-790301, "-6501041", "211261621313"},
    TestItem{-8999099, "-136330304", "211132062050"},
    TestItem{-542185444, "-16302333403", "161666055651"},
    TestItem{-2147483647 - 1, "-104134211162", "104134211162"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<8>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "11", "11"},
    TestItem{194, "302", "302"},
    TestItem{814, "1456", "1456"},
    TestItem{6322, "14262", "14262"},
    TestItem{12345, "30071", "30071"},
    TestItem{33312, "101040", "101040"},
    TestItem{901369, "3340371", "3340371"},
    TestItem{1579542, "6015026", "6015026"},
    TestItem{123345345, "726414701", "726414701"},
    TestItem{2147483647, "17777777777", "17777777777"},
    TestItem{-1, "-1", "37777777777"},
    TestItem{-437, "-665", "37777777113"},
    TestItem{-1459, "-2663", "37777775115"},
    TestItem{-8103, "-17647", "37777760131"},
    TestItem{-90000, "-257620", "37777520160"},
    TestItem{-790301, "-3007435", "37774770343"},
    TestItem{-8999099, "-42250273", "37735527505"},
    TestItem{-542185444, "-4024213744", "33753564034"},
    TestItem{-2147483647 - 1, "-20000000000", "20000000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<9>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "10", "10"},
    TestItem{194, "235", "235"},
    TestItem{814, "1104", "1104"},
    TestItem{6322, "8604", "8604"},
    TestItem{12345, "17836", "17836"},
    TestItem{33312, "50623", "50623"},
    TestItem{901369, "1623401", "1623401"},
    TestItem{1579542, "2866646", "2866646"},
    TestItem{123345345, "277077003", "277077003"},
    TestItem{2147483647, "5478773671", "5478773671"},
    TestItem{-1, "-1", "12068657453"},
    TestItem{-437, "-535", "12068656818"},
    TestItem{-1459, "-2001", "12068655453"},
    TestItem{-8103, "-12103", "12068645351"},
    TestItem{-90000, "-146410", "12068511044"},
    TestItem{-790301, "-1434072", "12067223372"},
    TestItem{-8999099, "-17835388", "12050722055"},
    TestItem{-542185444, "-1353185541", "10615461813"},
    TestItem{-2147483647 - 1, "-5478773672", "5478773672"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<10>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "194", "194"},
    TestItem{814, "814", "814"},
    TestItem{6322, "6322", "6322"},
    TestItem{12345, "12345", "12345"},
    TestItem{33312, "33312", "33312"},
    TestItem{901369, "901369", "901369"},
    TestItem{1579542, "1579542", "1579542"},
    TestItem{123345345, "123345345", "123345345"},
    TestItem{2147483647, "2147483647", "2147483647"},
    TestItem{-1, "-1", "4294967295"},
    TestItem{-437, "-437", "4294966859"},
    TestItem{-1459, "-1459", "4294965837"},
    TestItem{-8103, "-8103", "4294959193"},
    TestItem{-90000, "-90000", "4294877296"},
    TestItem{-790301, "-790301", "4294176995"},
    TestItem{-8999099, "-8999099", "4285968197"},
    TestItem{-542185444, "-542185444", "3752781852"},
    TestItem{-2147483647 - 1, "-2147483648", "2147483648"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<11>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "167", "167"},
    TestItem{814, "680", "680"},
    TestItem{6322, "4828", "4828"},
    TestItem{12345, "9303", "9303"},
    TestItem{33312, "23034", "23034"},
    TestItem{901369, "566237", "566237"},
    TestItem{1579542, "989808", "989808"},
    TestItem{123345345, "63697202", "63697202"},
    TestItem{2147483647, "a02220281", "a02220281"},
    TestItem{-1, "-1", "1904440553"},
    TestItem{-437, "-368", "1904440197"},
    TestItem{-1459, "-1107", "190443a448"},
    TestItem{-8103, "-60a7", "1904435458"},
    TestItem{-90000, "-61689", "1904389976"},
    TestItem{-790301, "-49a846", "1903a50809"},
    TestItem{-8999099, "-509717a", "18aa354385"},
    TestItem{-542185444, "-25905aa49", "1656390606"},
    TestItem{-2147483647 - 1, "-a02220282", "a02220282"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<12>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "142", "142"},
    TestItem{814, "57a", "57a"},
    TestItem{6322, "37aa", "37aa"},
    TestItem{12345, "7189", "7189"},
    TestItem{33312, "17340", "17340"},
    TestItem{901369, "375761", "375761"},
    TestItem{1579542, "642106", "642106"},
    TestItem{123345345, "353844a9", "353844a9"},
    TestItem{2147483647, "4bb2308a7", "4bb2308a7"},
    TestItem{-1, "-1", "9ba461593"},
    TestItem{-437, "-305", "9ba46128b"},
    TestItem{-1459, "-a17", "9ba460779"},
    TestItem{-8103, "-4833", "9ba458961"},
    TestItem{-90000, "-44100", "9ba419494"},
    TestItem{-790301, "-321425", "9ba14016b"},
    TestItem{-8999099, "-301b98b", "9b7441805"},
    TestItem{-542185444, "-1316b0884", "888970910"},
    TestItem{-2147483647 - 1, "-4bb2308a8", "4bb2308a8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<13>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "11c", "11c"},
    TestItem{814, "4a8", "4a8"},
    TestItem{6322, "2b54", "2b54"},
    TestItem{12345, "5808", "5808"},
    TestItem{33312, "12216", "12216"},
    TestItem{901369, "257371", "257371"},
    TestItem{1579542, "433c53", "433c53"},
    TestItem{123345345, "1c728816", "1c728816"},
    TestItem{2147483647, "282ba4aaa", "282ba4aaa"},
    TestItem{-1, "-1", "535a79888"},
    TestItem{-437, "-278", "535a79611"},
    TestItem{-1459, "-883", "535a79006"},
    TestItem{-8103, "-38c4", "535a75c95"},
    TestItem{-90000, "-31c71", "535a47918"},
    TestItem{-790301, "-218945", "535860c44"},
    TestItem{-8999099, "-1b31115", "533c48774"},
    TestItem{-542185444, "-884355b8", "47a6442a1"},
    TestItem{-2147483647 - 1, "-282ba4aab", "282ba4aab"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<14>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "dc", "dc"},
    TestItem{814, "422", "422"},
    TestItem{6322, "2438", "2438"},
    TestItem{12345, "46db", "46db"},
    TestItem{33312, "c1d6", "c1d6"},
    TestItem{901369, "1966b7", "1966b7"},
    TestItem{1579542, "2d18c6", "2d18c6"},
    TestItem{123345345, "1254acdb", "1254acdb"},
    TestItem{2147483647, "1652ca931", "1652ca931"},
    TestItem{-1, "-1", "2ca5b7463"},
    TestItem{-437, "-233", "2ca5b7231"},
    TestItem{-1459, "-763", "2ca5b6b01"},
    TestItem{-8103, "-2d4b", "2ca5b4517"},
    TestItem{-90000, "-24b28", "2ca59273a"},
    TestItem{-790301, "-168021", "2ca44d443"},
    TestItem{-8999099, "-12a37ab", "2c9313a97"},
    TestItem{-542185444, "-5201763a", "27859dc28"},
    TestItem{-2147483647 - 1, "-1652ca932", "1652ca932"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<15>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "ce", "ce"},
    TestItem{814, "394", "394"},
    TestItem{6322, "1d17", "1d17"},
    TestItem{12345, "39d0", "39d0"},
    TestItem{33312, "9d0c", "9d0c"},
    TestItem{901369, "12c114", "12c114"},
    TestItem{1579542, "21302c", "21302c"},
    TestItem{123345345, "ac66b80", "ac66b80"},
    TestItem{2147483647, "c87e66b7", "c87e66b7"},
    TestItem{-1, "-1", "1a20dcd80"},
    TestItem{-437, "-1e2", "1a20dcb8e"},
    TestItem{-1459, "-674", "1a20dc70c"},
    TestItem{-8103, "-2603", "1a20da77d"},
    TestItem{-90000, "-1ba00", "1a20c1381"},
    TestItem{-790301, "-10926b", "1a1ed3b15"},
    TestItem{-8999099, "-bcb5ee", "1a1411782"},
    TestItem{-542185444, "-328ec814", "16e6e056c"},
    TestItem{-2147483647 - 1, "-c87e66b8", "c87e66b8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<16>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "c2", "c2"},
    TestItem{814, "32e", "32e"},
    TestItem{6322, "18b2", "18b2"},
    TestItem{12345, "3039", "3039"},
    TestItem{33312, "8220", "8220"},
    TestItem{901369, "dc0f9", "dc0f9"},
    TestItem{1579542, "181a16", "181a16"},
    TestItem{123345345, "75a19c1", "75a19c1"},
    TestItem{2147483647, "7fffffff", "7fffffff"},
    TestItem{-1, "-1", "ffffffff"},
    TestItem{-437, "-1b5", "fffffe4b"},
    TestItem{-1459, "-5b3", "fffffa4d"},
    TestItem{-8103, "-1fa7", "ffffe059"},
    TestItem{-90000, "-15f90", "fffea070"},
    TestItem{-790301, "-c0f1d", "fff3f0e3"},
    TestItem{-8999099, "-8950bb", "ff76af45"},
    TestItem{-542185444, "-205117e4", "dfaee81c"},
    TestItem{-2147483647 - 1, "-80000000", "80000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<17>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "b7", "b7"},
    TestItem{814, "2df", "2df"},
    TestItem{6322, "14ef", "14ef"},
    TestItem{12345, "28c3", "28c3"},
    TestItem{33312, "6d49", "6d49"},
    TestItem{901369, "ad7fc", "ad7fc"},
    TestItem{1579542, "11f894", "11f894"},
    TestItem{123345345, "51edf89", "51edf89"},
    TestItem{2147483647, "53g7f548", "53g7f548"},
    TestItem{-1, "-1", "a7ffda90"},
    TestItem{-437, "-18c", "a7ffd906"},
    TestItem{-1459, "-50e", "a7ffd584"},
    TestItem{-8103, "-1b0b", "a7ffbg87"},
    TestItem{-90000, "-11572", "a7fec51g"},
    TestItem{-790301, "-97ea5", "a7f65cfd"},
    TestItem{-8999099, "-65cbcd", "a79a0fd5"},
    TestItem{-542185444, "-157ea537", "9281355b"},
    TestItem{-2147483647 - 1, "-53g7f549", "53g7f549"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<18>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "ae", "ae"},
    TestItem{814, "294", "294"},
    TestItem{6322, "1194", "1194"},
    TestItem{12345, "221f", "221f"},
    TestItem{33312, "5cec", "5cec"},
    TestItem{901369, "8aa01", "8aa01"},
    TestItem{1579542, "f0f26", "f0f26"},
    TestItem{123345345, "3b4hd93", "3b4hd93"},
    TestItem{2147483647, "3928g3h1", "3928g3h1"},
    TestItem{-1, "-1", "704he7g3"},
    TestItem{-437, "-165", "704he69h"},
    TestItem{-1459, "-491", "704he373"},
    TestItem{-8103, "-1703", "704hd0g1"},
    TestItem{-90000, "-f7e0", "704gh024"},
    TestItem{-790301, "-7993b", "704a4gcb"},
    TestItem{-8999099, "-4dd0hh", "700416g5"},
    TestItem{-542185444, "-fggf5fa", "6260h20c"},
    TestItem{-2147483647 - 1, "-3928g3h2", "3928g3h2"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<19>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "a4", "a4"},
    TestItem{814, "24g", "24g"},
    TestItem{6322, "h9e", "h9e"},
    TestItem{12345, "1f3e", "1f3e"},
    TestItem{33312, "4g55", "4g55"},
    TestItem{901369, "6h7g9", "6h7g9"},
    TestItem{1579542, "c258f", "c258f"},
    TestItem{123345345, "2bf8ig5", "2bf8ig5"},
    TestItem{2147483647, "27c57h32", "27c57h32"},
    TestItem{-1, "-1", "4f5aff65"},
    TestItem{-437, "-140", "4f5afe26"},
    TestItem{-1459, "-40f", "4f5afb5a"},
    TestItem{-8103, "-1389", "4f5aebgg"},
    TestItem{-90000, "-d25g", "4f5a2d09"},
    TestItem{-790301, "-6143f", "4f54eb2a"},
    TestItem{-8999099, "-3c104f", "4f1hef1a"},
    TestItem{-542185444, "-b9i75e0", "43eb89b6"},
    TestItem{-2147483647 - 1, "-27c57h33", "27c57h33"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<20>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "9e", "9e"},
    TestItem{814, "20e", "20e"},
    TestItem{6322, "fg2", "fg2"},
    TestItem{12345, "1ah5", "1ah5"},
    TestItem{33312, "435c", "435c"},
    TestItem{901369, "5cd89", "5cd89"},
    TestItem{1579542, "9h8h2", "9h8h2"},
    TestItem{123345345, "1iai375", "1iai375"},
    TestItem{2147483647, "1db1f927", "1db1f927"},
    TestItem{-1, "-1", "3723ai4f"},
    TestItem{-437, "-11h", "3723ah2j"},
    TestItem{-1459, "-3cj", "3723aebh"},
    TestItem{-8103, "-1053", "37239hjd"},
    TestItem{-90000, "-b500", "3722jd4g"},
    TestItem{-790301, "-4iff1", "371ic29f"},
    TestItem{-8999099, "-2g4hej", "36j7609h"},
    TestItem{-542185444, "-898d3c4", "2icehecc"},
    TestItem{-2147483647 - 1, "-1db1f928", "1db1f928"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<21>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "95", "95"},
    TestItem{814, "1hg", "1hg"},
    TestItem{6322, "e71", "e71"},
    TestItem{12345, "16ki", "16ki"},
    TestItem{33312, "3cb6", "3cb6"},
    TestItem{901369, "4d6j7", "4d6j7"},
    TestItem{1579542, "82bf6", "82bf6"},
    TestItem{123345345, "1944gdi", "1944gdi"},
    TestItem{2147483647, "140h2d91", "140h2d91"},
    TestItem{-1, "-1", "281d55i3"},
    TestItem{-437, "-kh", "281d54i8"},
    TestItem{-1459, "-36a", "281d52bf"},
    TestItem{-8103, "-i7i", "281d48a7"},
    TestItem{-90000, "-9f1f", "281cgbga"},
    TestItem{-790301, "-41718", "28193jgh"},
    TestItem{-8999099, "-245f2b", "27k8kbfe"},
    TestItem{-542185444, "-66fi09a", "21fi858f"},
    TestItem{-2147483647 - 1, "-140h2d92", "140h2d92"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<22>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "8i", "8i"},
    TestItem{814, "1f0", "1f0"},
    TestItem{6322, "d18", "d18"},
    TestItem{12345, "13b3", "13b3"},
    TestItem{33312, "32i4", "32i4"},
    TestItem{901369, "3ie77", "3ie77"},
    TestItem{1579542, "6g7b8", "6g7b8"},
    TestItem{123345345, "11kbjgd", "11kbjgd"},
    TestItem{2147483647, "ikf5bf1", "ikf5bf1"},
    TestItem{-1, "-1", "1fj8b183"},
    TestItem{-437, "-jj", "1fj8b0a7"},
    TestItem{-1459, "-307", "1fj8ak7j"},
    TestItem{-8103, "-gg7", "1fj8a6dj"},
    TestItem{-90000, "-89kk", "1fj82d96"},
    TestItem{-790301, "-384ih", "1fj52ib9"},
    TestItem{-8999099, "-1g933l", "1fhe1k45"},
    TestItem{-542185444, "-4h4alik", "1b2401b6"},
    TestItem{-2147483647 - 1, "-ikf5bf2", "ikf5bf2"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<23>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "8a", "8a"},
    TestItem{814, "1c9", "1c9"},
    TestItem{6322, "blk", "blk"},
    TestItem{12345, "107h", "107h"},
    TestItem{33312, "2gm8", "2gm8"},
    TestItem{901369, "351km", "351km"},
    TestItem{1579542, "5eikh", "5eikh"},
    TestItem{123345345, "j3hg02", "j3hg02"},
    TestItem{2147483647, "ebelf95", "ebelf95"},
    TestItem{-1, "-1", "1606k7ib"},
    TestItem{-437, "-j0", "1606k6mc"},
    TestItem{-1459, "-2ha", "1606k512"},
    TestItem{-8103, "-f77", "1606jfb5"},
    TestItem{-90000, "-7931", "1606clfb"},
    TestItem{-790301, "-2illl", "160418je"},
    TestItem{-8999099, "-193ec4", "15lkgg68"},
    TestItem{-542185444, "-3f5am54", "128198d8"},
    TestItem{-2147483647 - 1, "-ebelf96", "ebelf96"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<24>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "82", "82"},
    TestItem{814, "19m", "19m"},
    TestItem{6322, "ana", "ana"},
    TestItem{12345, "la9", "la9"},
    TestItem{33312, "29k0", "29k0"},
    TestItem{901369, "2h4l1", "2h4l1"},
    TestItem{1579542, "4i666", "4i666"},
    TestItem{123345345, "fbid59", "fbid59"},
    TestItem{2147483647, "b5gge57", "b5gge57"},
    TestItem{-1, "-1", "mb994af"},
    TestItem{-437, "-i5", "mb993gb"},
    TestItem{-1459, "-2cj", "mb991ll"},
    TestItem{-8103, "-e1f", "mb98e91"},
    TestItem{-90000, "-6c60", "mb92g4g"},
    TestItem{-790301, "-29415", "mb7009b"},
    TestItem{-8999099, "-132nab", "ma66505"},
    TestItem{-542185444, "-2k24e44", "jf74e6c"},
    TestItem{-2147483647 - 1, "-b5gge58", "b5gge58"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<25>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "7j", "7j"},
    TestItem{814, "17e", "17e"},
    TestItem{6322, "a2m", "a2m"},
    TestItem{12345, "jik", "jik"},
    TestItem{33312, "237c", "237c"},
    TestItem{901369, "27h4j", "27h4j"},
    TestItem{1579542, "4126h", "4126h"},
    TestItem{123345345, "cfj2dk", "cfj2dk"},
    TestItem{2147483647, "8jmdnkm", "8jmdnkm"},
    TestItem{-1, "-1", "hek2mgk"},
    TestItem{-437, "-hc", "hek2lo9"},
    TestItem{-1459, "-289", "hek2k8c"},
    TestItem{-8103, "-co3", "hek29hi"},
    TestItem{-90000, "-5j00", "hejm3gl"},
    TestItem{-790301, "-20ec1", "hei284k"},
    TestItem{-8999099, "-n0ndo", "hdm1o2m"},
    TestItem{-542185444, "-25colhj", "f9730o2"},
    TestItem{-2147483647 - 1, "-8jmdnkn", "8jmdnkn"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<26>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "7c", "7c"},
    TestItem{814, "158", "158"},
    TestItem{6322, "994", "994"},
    TestItem{12345, "i6l", "i6l"},
    TestItem{33312, "1n76", "1n76"},
    TestItem{901369, "1p7a1", "1p7a1"},
    TestItem{1579542, "3bmfg", "3bmfg"},
    TestItem{123345345, "a9nldj", "a9nldj"},
    TestItem{2147483647, "6oj8ion", "6oj8ion"},
    TestItem{-1, "-1", "dnchbnl"},
    TestItem{-437, "-gl", "dnchb71"},
    TestItem{-1459, "-243", "dnch9jj"},
    TestItem{-8103, "-bph", "dncgpo5"},
    TestItem{-90000, "-533e", "dncc8k8"},
    TestItem{-790301, "-1ip25", "dnaoclh"},
    TestItem{-8999099, "-ji075", "dmipbgh"},
    TestItem{-542185444, "-1jgc1c8", "c3m5abe"},
    TestItem{-2147483647 - 1, "-6oj8ioo", "6oj8ioo"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<27>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "75", "75"},
    TestItem{814, "134", "134"},
    TestItem{6322, "8i4", "8i4"},
    TestItem{12345, "gp6", "gp6"},
    TestItem{33312, "1iil", "1iil"},
    TestItem{901369, "1ilc1", "1ilc1"},
    TestItem{1579542, "2q6jf", "2q6jf"},
    TestItem{123345345, "8g2g03", "8g2g03"},
    TestItem{2147483647, "5ehncka", "5ehncka"},
    TestItem{-1, "-1", "b28jpdl"},
    TestItem{-437, "-g5", "b28jooh"},
    TestItem{-1459, "-201", "b28jndl"},
    TestItem{-8103, "-b33", "b28jeaj"},
    TestItem{-90000, "-4fc9", "b28fa1d"},
    TestItem{-790301, "-1d42b", "b276lbb"},
    TestItem{-8999099, "-gp5bq", "b1ilk1n"},
    TestItem{-542185444, "-1al5nga", "9iee1oc"},
    TestItem{-2147483647 - 1, "-5ehnckb", "5ehnckb"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<28>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "6q", "6q"},
    TestItem{814, "112", "112"},
    TestItem{6322, "81m", "81m"},
    TestItem{12345, "fkp", "fkp"},
    TestItem{33312, "1edk", "1edk"},
    TestItem{901369, "1d1jl", "1d1jl"},
    TestItem{1579542, "2fqk6", "2fqk6"},
    TestItem{123345345, "74io6p", "74io6p"},
    TestItem{2147483647, "4clm98f", "4clm98f"},
    TestItem{-1, "-1", "8pfgih3"},
    TestItem{-437, "-fh", "8pfgi1f"},
    TestItem{-1459, "-1o3", "8pfggl1"},
    TestItem{-8103, "-a9b", "8pfg87l"},
    TestItem{-90000, "-42m8", "8pfcfmo"},
    TestItem{-790301, "-18011", "8pe8ig3"},
    TestItem{-8999099, "-ehqcb", "8p0qk4l"},
    TestItem{-542185444, "-13e2j1o", "7m1drf8"},
    TestItem{-2147483647 - 1, "-4clm98g", "4clm98g"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<29>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "6k", "6k"},
    TestItem{814, "s2", "s2"},
    TestItem{6322, "7f0", "7f0"},
    TestItem{12345, "ejk", "ejk"},
    TestItem{33312, "1ahk", "1ahk"},
    TestItem{901369, "17rmk", "17rmk"},
    TestItem{1579542, "26m4s", "26m4s"},
    TestItem{123345345, "60bc2m", "60bc2m"},
    TestItem{2147483647, "3hk7987", "3hk7987"},
    TestItem{-1, "-1", "76beigf"},
    TestItem{-437, "-f2", "76bei1e"},
    TestItem{-1459, "-1l9", "76bego7"},
    TestItem{-8103, "-9ic", "76be8r4"},
    TestItem{-90000, "-3k0d", "76barg3"},
    TestItem{-790301, "-13bkm", "76ab6on"},
    TestItem{-8999099, "-cksdm", "75rmj2n"},
    TestItem{-542185444, "-qcglan", "68rqq5m"},
    TestItem{-2147483647 - 1, "-3hk7988", "3hk7988"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<30>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "6e", "6e"},
    TestItem{814, "r4", "r4"},
    TestItem{6322, "70m", "70m"},
    TestItem{12345, "dlf", "dlf"},
    TestItem{33312, "170c", "170c"},
    TestItem{901369, "13bfj", "13bfj"},
    TestItem{1579542, "1sf1c", "1sf1c"},
    TestItem{123345345, "528abf", "528abf"},
    TestItem{2147483647, "2sb6cs7", "2sb6cs7"},
    TestItem{-1, "-1", "5qmcpqf"},
    TestItem{-437, "-eh", "5qmcpbt"},
    TestItem{-1459, "-1ij", "5qmco7r"},
    TestItem{-8103, "-903", "5qmcgqd"},
    TestItem{-90000, "-3a00", "5qm9fqg"},
    TestItem{-790301, "-t83b", "5qldhn5"},
    TestItem{-8999099, "-b38tt", "5qb9gqh"},
    TestItem{-542185444, "-m9as84", "54d1ric"},
    TestItem{-2147483647 - 1, "-2sb6cs8", "2sb6cs8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<31>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "68", "68"},
    TestItem{814, "q8", "q8"},
    TestItem{6322, "6ht", "6ht"},
    TestItem{12345, "cq7", "cq7"},
    TestItem{33312, "13ki", "13ki"},
    TestItem{901369, "u7td", "u7td"},
    TestItem{1579542, "1m0ju", "1m0ju"},
    TestItem{123345345, "49hb13", "49hb13"},
    TestItem{2147483647, "2d09uc1", "2d09uc1"},
    TestItem{-1, "-1", "4q0jto3"},
    TestItem{-437, "-e3", "4q0jta1"},
    TestItem{-1459, "-1g2", "4q0js82"},
    TestItem{-8103, "-8dc", "4q0jlan"},
    TestItem{-90000, "-30k7", "4q0gt3s"},
    TestItem{-790301, "-qgbi", "4puodch"},
    TestItem{-8999099, "-9n29g", "4plrrej"},
    TestItem{-542185444, "-it2jp1", "472h9u3"},
    TestItem{-2147483647 - 1, "-2d09uc2", "2d09uc2"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<32>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "62", "62"},
    TestItem{814, "pe", "pe"},
    TestItem{6322, "65i", "65i"},
    TestItem{12345, "c1p", "c1p"},
    TestItem{33312, "10h0", "10h0"},
    TestItem{901369, "rg7p", "rg7p"},
    TestItem{1579542, "1g6gm", "1g6gm"},
    TestItem{123345345, "3lk6e1", "3lk6e1"},
    TestItem{2147483647, "1vvvvvv", "1vvvvvv"},
    TestItem{-1, "-1", "3vvvvvv"},
    TestItem{-437, "-dl", "3vvvvib"},
    TestItem{-1459, "-1dj", "3vvvuid"},
    TestItem{-8103, "-7t7", "3vvvo2p"},
    TestItem{-90000, "-2nsg", "3vvt83g"},
    TestItem{-790301, "-o3ot", "3vv7s73"},
    TestItem{-8999099, "-8ik5r", "3vndbq5"},
    TestItem{-542185444, "-g525v4", "3fqtq0s"},
    TestItem{-2147483647 - 1, "-2000000", "2000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<33>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "5t", "5t"},
    TestItem{814, "om", "om"},
    TestItem{6322, "5qj", "5qj"},
    TestItem{12345, "bb3", "bb3"},
    TestItem{33312, "ujf", "ujf"},
    TestItem{901369, "p2n7", "p2n7"},
    TestItem{1579542, "1aveu", "1aveu"},
    TestItem{123345345, "3508po", "3508po"},
    TestItem{2147483647, "1lsqtl1", "1lsqtl1"},
    TestItem{-1, "-1", "3aokq93"},
    TestItem{-437, "-d8", "3aokpst"},
    TestItem{-1459, "-1b7", "3aokouu"},
    TestItem{-8103, "-7ei", "3aokirj"},
    TestItem{-90000, "-2gl9", "3aoi9ks"},
    TestItem{-790301, "-lwnh", "3anvqik"},
    TestItem{-8999099, "-7jdkw", "3ah1cl5"},
    TestItem{-542185444, "-ds63jv", "2ttemm6"},
    TestItem{-2147483647 - 1, "-1lsqtl2", "1lsqtl2"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<34>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "5o", "5o"},
    TestItem{814, "nw", "nw"},
    TestItem{6322, "5fw", "5fw"},
    TestItem{12345, "an3", "an3"},
    TestItem{33312, "srq", "srq"},
    TestItem{901369, "mvot", "mvot"},
    TestItem{1579542, "166d4", "166d4"},
    TestItem{123345345, "2oa849", "2oa849"},
    TestItem{2147483647, "1d8xqrp", "1d8xqrp"},
    TestItem{-1, "-1", "2qhxjlh"},
    TestItem{-437, "-ct", "2qhxj8n"},
    TestItem{-1459, "-18v", "2qhxicl"},
    TestItem{-8103, "-70b", "2qhxcl7"},
    TestItem{-90000, "-29t2", "2qhv9qg"},
    TestItem{-790301, "-k3m5", "2qhdfxd"},
    TestItem{-8999099, "-6ownd", "2qb8kw5"},
    TestItem{-542185444, "-bvomio", "2ek8v2s"},
    TestItem{-2147483647 - 1, "-1d8xqrq", "1d8xqrq"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<35>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "5j", "5j"},
    TestItem{814, "n9", "n9"},
    TestItem{6322, "55m", "55m"},
    TestItem{12345, "a2p", "a2p"},
    TestItem{33312, "r6r", "r6r"},
    TestItem{901369, "l0se", "l0se"},
    TestItem{1579542, "11ter", "11ter"},
    TestItem{123345345, "2c6u2p", "2c6u2p"},
    TestItem{2147483647, "15v22um", "15v22um"},
    TestItem{-1, "-1", "2br45qa"},
    TestItem{-437, "-ch", "2br45dt"},
    TestItem{-1459, "-16o", "2br44jm"},
    TestItem{-8103, "-6li", "2br3y4s"},
    TestItem{-90000, "-23gf", "2br229v"},
    TestItem{-790301, "-if51", "2bqkpla"},
    TestItem{-8999099, "-5yv74", "2bl49j7"},
    TestItem{-542185444, "-abapco", "21fsfdm"},
    TestItem{-2147483647 - 1, "-15v22un", "15v22un"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 21> get_test_items<36>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{9, "9", "9"},
    TestItem{194, "5e", "5e"},
    TestItem{814, "mm", "mm"},
    TestItem{6322, "4vm", "4vm"},
    TestItem{12345, "9ix", "9ix"},
    TestItem{33312, "ppc", "ppc"},
    TestItem{901369, "jbi1", "jbi1"},
    TestItem{1579542, "xus6", "xus6"},
    TestItem{123345345, "21fpvl", "21fpvl"},
    TestItem{2147483647, "zik0zj", "zik0zj"},
    TestItem{-1, "-1", "1z141z3"},
    TestItem{-437, "-c5", "1z141mz"},
    TestItem{-1459, "-14j", "1z140ul"},
    TestItem{-8103, "-693", "1z13vq1"},
    TestItem{-90000, "-1xg0", "1z124j4"},
    TestItem{-790301, "-gxst", "1z0n46b"},
    TestItem{-8999099, "-5cvqz", "1yvr685"},
    TestItem{-542185444, "-8yswys", "1q2b50c"},
    TestItem{-2147483647 - 1, "-zik0zk", "zik0zk"},
  }};
}

template <class T, int Base>
__host__ __device__ constexpr void test_to_chars(const TestItem& item)
{
  constexpr cuda::std::size_t buff_size = 150;

  static_assert(cuda::std::is_same_v<
                cuda::std::to_chars_result,
                decltype(cuda::std::to_chars(cuda::std::declval<char*>(), cuda::std::declval<char*>(), T{}, int{}))>);
  static_assert(noexcept(cuda::std::to_chars(cuda::std::declval<char*>(), cuda::std::declval<char*>(), T{}, int{})));

  char buff[buff_size + 1]{};
  char* buff_start = buff + 1;

  const auto value    = static_cast<T>(item.val);
  const char* ref_str = (cuda::std::is_signed_v<T>) ? item.str_signed : item.str_unsigned;
  const auto ref_len  = cuda::std::strlen(ref_str);

  // Check valid buffer size
  {
    const auto result = cuda::std::to_chars(buff_start, buff_start + buff_size, value, Base);
    assert(result.ec == cuda::std::errc{});
    assert(result.ptr == buff_start + ref_len);

    // Compare with reference string
    assert(cuda::std::strncmp(buff_start, ref_str, buff_size) == 0);

    // Check that the operation did not underflow the buffer
    assert(buff[0] == '\0');
  }

  // Check too small buffer
  {
    const auto result = cuda::std::to_chars(buff_start, buff_start + ref_len - 1, value, Base);
    assert(result.ec == cuda::std::errc::value_too_large);
    assert(result.ptr == buff_start + ref_len - 1);
  }

  // Check zero buffer
  {
    const auto result = cuda::std::to_chars(buff_start, buff_start, value, Base);
    assert(result.ec == cuda::std::errc::value_too_large);
    assert(result.ptr == buff_start);
  }
}

template <int Base>
__host__ __device__ constexpr bool test_base()
{
  constexpr auto items = get_test_items<Base>();

  for (const auto& item : items)
  {
    test_to_chars<cuda::std::int32_t, Base>(item);
    test_to_chars<cuda::std::uint32_t, Base>(item);
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
