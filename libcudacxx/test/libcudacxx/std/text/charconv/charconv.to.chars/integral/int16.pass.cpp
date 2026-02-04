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
  cuda::std::int16_t val;
  const char* str_signed;
  const char* str_unsigned;
};

template <int Base>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items();

// Source code for the generation of the test items
// #include <iostream>
// #include <charconv>
// #include <cstdint>
// #include <cstddef>
// #include <type_traits>
//
// constexpr std::int16_t list[] = {
//   0,
//   1,
//   2,
//   7,
//   14,
//   32,
//   227,
//   893,
//   911,
//   1'019,
//   3'333,
//   9'704,
//   23'232,
//   32'766,
//   32'767,
//   -1,
//   -17,
//   -294,
//   -777,
//   -1'123,
//   -3'123,
//   -12'345,
//   -32'767,
//   -32'768,
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
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<2>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "10", "10"},
    TestItem{7, "111", "111"},
    TestItem{14, "1110", "1110"},
    TestItem{32, "100000", "100000"},
    TestItem{227, "11100011", "11100011"},
    TestItem{893, "1101111101", "1101111101"},
    TestItem{911, "1110001111", "1110001111"},
    TestItem{1019, "1111111011", "1111111011"},
    TestItem{3333, "110100000101", "110100000101"},
    TestItem{9704, "10010111101000", "10010111101000"},
    TestItem{23232, "101101011000000", "101101011000000"},
    TestItem{32766, "111111111111110", "111111111111110"},
    TestItem{32767, "111111111111111", "111111111111111"},
    TestItem{-1, "-1", "1111111111111111"},
    TestItem{-17, "-10001", "1111111111101111"},
    TestItem{-294, "-100100110", "1111111011011010"},
    TestItem{-777, "-1100001001", "1111110011110111"},
    TestItem{-1123, "-10001100011", "1111101110011101"},
    TestItem{-3123, "-110000110011", "1111001111001101"},
    TestItem{-12345, "-11000000111001", "1100111111000111"},
    TestItem{-32767, "-111111111111111", "1000000000000001"},
    TestItem{-32768, "-1000000000000000", "1000000000000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<3>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "21", "21"},
    TestItem{14, "112", "112"},
    TestItem{32, "1012", "1012"},
    TestItem{227, "22102", "22102"},
    TestItem{893, "1020002", "1020002"},
    TestItem{911, "1020202", "1020202"},
    TestItem{1019, "1101202", "1101202"},
    TestItem{3333, "11120110", "11120110"},
    TestItem{9704, "111022102", "111022102"},
    TestItem{23232, "1011212110", "1011212110"},
    TestItem{32766, "1122221120", "1122221120"},
    TestItem{32767, "1122221121", "1122221121"},
    TestItem{-1, "-1", "10022220020"},
    TestItem{-17, "-122", "10022212122"},
    TestItem{-294, "-101220", "10022111101"},
    TestItem{-777, "-1001210", "10021211111"},
    TestItem{-1123, "-1112121", "10021100200"},
    TestItem{-3123, "-11021200", "10011121121"},
    TestItem{-12345, "-121221020", "2200222001"},
    TestItem{-32767, "-1122221121", "1122221200"},
    TestItem{-32768, "-1122221122", "1122221122"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<4>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "13", "13"},
    TestItem{14, "32", "32"},
    TestItem{32, "200", "200"},
    TestItem{227, "3203", "3203"},
    TestItem{893, "31331", "31331"},
    TestItem{911, "32033", "32033"},
    TestItem{1019, "33323", "33323"},
    TestItem{3333, "310011", "310011"},
    TestItem{9704, "2113220", "2113220"},
    TestItem{23232, "11223000", "11223000"},
    TestItem{32766, "13333332", "13333332"},
    TestItem{32767, "13333333", "13333333"},
    TestItem{-1, "-1", "33333333"},
    TestItem{-17, "-101", "33333233"},
    TestItem{-294, "-10212", "33323122"},
    TestItem{-777, "-30021", "33303313"},
    TestItem{-1123, "-101203", "33232131"},
    TestItem{-3123, "-300303", "33033031"},
    TestItem{-12345, "-3000321", "30333013"},
    TestItem{-32767, "-13333333", "20000001"},
    TestItem{-32768, "-20000000", "20000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<5>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "12", "12"},
    TestItem{14, "24", "24"},
    TestItem{32, "112", "112"},
    TestItem{227, "1402", "1402"},
    TestItem{893, "12033", "12033"},
    TestItem{911, "12121", "12121"},
    TestItem{1019, "13034", "13034"},
    TestItem{3333, "101313", "101313"},
    TestItem{9704, "302304", "302304"},
    TestItem{23232, "1220412", "1220412"},
    TestItem{32766, "2022031", "2022031"},
    TestItem{32767, "2022032", "2022032"},
    TestItem{-1, "-1", "4044120"},
    TestItem{-17, "-32", "4044034"},
    TestItem{-294, "-2134", "4041432"},
    TestItem{-777, "-11102", "4033014"},
    TestItem{-1123, "-13443", "4030123"},
    TestItem{-3123, "-44443", "3444123"},
    TestItem{-12345, "-343340", "3200231"},
    TestItem{-32767, "-2022032", "2022034"},
    TestItem{-32768, "-2022033", "2022033"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<6>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "11", "11"},
    TestItem{14, "22", "22"},
    TestItem{32, "52", "52"},
    TestItem{227, "1015", "1015"},
    TestItem{893, "4045", "4045"},
    TestItem{911, "4115", "4115"},
    TestItem{1019, "4415", "4415"},
    TestItem{3333, "23233", "23233"},
    TestItem{9704, "112532", "112532"},
    TestItem{23232, "255320", "255320"},
    TestItem{32766, "411410", "411410"},
    TestItem{32767, "411411", "411411"},
    TestItem{-1, "-1", "1223223"},
    TestItem{-17, "-25", "1223155"},
    TestItem{-294, "-1210", "1222014"},
    TestItem{-777, "-3333", "1215451"},
    TestItem{-1123, "-5111", "1214113"},
    TestItem{-3123, "-22243", "1200541"},
    TestItem{-12345, "-133053", "1050131"},
    TestItem{-32767, "-411411", "411413"},
    TestItem{-32768, "-411412", "411412"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<7>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "10", "10"},
    TestItem{14, "20", "20"},
    TestItem{32, "44", "44"},
    TestItem{227, "443", "443"},
    TestItem{893, "2414", "2414"},
    TestItem{911, "2441", "2441"},
    TestItem{1019, "2654", "2654"},
    TestItem{3333, "12501", "12501"},
    TestItem{9704, "40202", "40202"},
    TestItem{23232, "124506", "124506"},
    TestItem{32766, "164346", "164346"},
    TestItem{32767, "164350", "164350"},
    TestItem{-1, "-1", "362031"},
    TestItem{-17, "-23", "362006"},
    TestItem{-294, "-600", "361132"},
    TestItem{-777, "-2160", "356542"},
    TestItem{-1123, "-3163", "355536"},
    TestItem{-3123, "-12051", "346651"},
    TestItem{-12345, "-50664", "311035"},
    TestItem{-32767, "-164350", "164352"},
    TestItem{-32768, "-164351", "164351"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<8>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "16", "16"},
    TestItem{32, "40", "40"},
    TestItem{227, "343", "343"},
    TestItem{893, "1575", "1575"},
    TestItem{911, "1617", "1617"},
    TestItem{1019, "1773", "1773"},
    TestItem{3333, "6405", "6405"},
    TestItem{9704, "22750", "22750"},
    TestItem{23232, "55300", "55300"},
    TestItem{32766, "77776", "77776"},
    TestItem{32767, "77777", "77777"},
    TestItem{-1, "-1", "177777"},
    TestItem{-17, "-21", "177757"},
    TestItem{-294, "-446", "177332"},
    TestItem{-777, "-1411", "176367"},
    TestItem{-1123, "-2143", "175635"},
    TestItem{-3123, "-6063", "171715"},
    TestItem{-12345, "-30071", "147707"},
    TestItem{-32767, "-77777", "100001"},
    TestItem{-32768, "-100000", "100000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<9>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "15", "15"},
    TestItem{32, "35", "35"},
    TestItem{227, "272", "272"},
    TestItem{893, "1202", "1202"},
    TestItem{911, "1222", "1222"},
    TestItem{1019, "1352", "1352"},
    TestItem{3333, "4513", "4513"},
    TestItem{9704, "14272", "14272"},
    TestItem{23232, "34773", "34773"},
    TestItem{32766, "48846", "48846"},
    TestItem{32767, "48847", "48847"},
    TestItem{-1, "-1", "108806"},
    TestItem{-17, "-18", "108778"},
    TestItem{-294, "-356", "108441"},
    TestItem{-777, "-1053", "107744"},
    TestItem{-1123, "-1477", "107320"},
    TestItem{-3123, "-4250", "104547"},
    TestItem{-12345, "-17836", "80861"},
    TestItem{-32767, "-48847", "48850"},
    TestItem{-32768, "-48848", "48848"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<10>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "14", "14"},
    TestItem{32, "32", "32"},
    TestItem{227, "227", "227"},
    TestItem{893, "893", "893"},
    TestItem{911, "911", "911"},
    TestItem{1019, "1019", "1019"},
    TestItem{3333, "3333", "3333"},
    TestItem{9704, "9704", "9704"},
    TestItem{23232, "23232", "23232"},
    TestItem{32766, "32766", "32766"},
    TestItem{32767, "32767", "32767"},
    TestItem{-1, "-1", "65535"},
    TestItem{-17, "-17", "65519"},
    TestItem{-294, "-294", "65242"},
    TestItem{-777, "-777", "64759"},
    TestItem{-1123, "-1123", "64413"},
    TestItem{-3123, "-3123", "62413"},
    TestItem{-12345, "-12345", "53191"},
    TestItem{-32767, "-32767", "32769"},
    TestItem{-32768, "-32768", "32768"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<11>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "13", "13"},
    TestItem{32, "2a", "2a"},
    TestItem{227, "197", "197"},
    TestItem{893, "742", "742"},
    TestItem{911, "759", "759"},
    TestItem{1019, "847", "847"},
    TestItem{3333, "2560", "2560"},
    TestItem{9704, "7322", "7322"},
    TestItem{23232, "16500", "16500"},
    TestItem{32766, "22688", "22688"},
    TestItem{32767, "22689", "22689"},
    TestItem{-1, "-1", "45268"},
    TestItem{-17, "-16", "45253"},
    TestItem{-294, "-248", "45021"},
    TestItem{-777, "-647", "44722"},
    TestItem{-1123, "-931", "44438"},
    TestItem{-3123, "-238a", "4298a"},
    TestItem{-12345, "-9303", "36a66"},
    TestItem{-32767, "-22689", "22690"},
    TestItem{-32768, "-2268a", "2268a"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<12>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "12", "12"},
    TestItem{32, "28", "28"},
    TestItem{227, "16b", "16b"},
    TestItem{893, "625", "625"},
    TestItem{911, "63b", "63b"},
    TestItem{1019, "70b", "70b"},
    TestItem{3333, "1b19", "1b19"},
    TestItem{9704, "5748", "5748"},
    TestItem{23232, "11540", "11540"},
    TestItem{32766, "16b66", "16b66"},
    TestItem{32767, "16b67", "16b67"},
    TestItem{-1, "-1", "31b13"},
    TestItem{-17, "-15", "31abb"},
    TestItem{-294, "-206", "3190a"},
    TestItem{-777, "-549", "31587"},
    TestItem{-1123, "-797", "31339"},
    TestItem{-3123, "-1983", "30151"},
    TestItem{-12345, "-7189", "26947"},
    TestItem{-32767, "-16b67", "16b69"},
    TestItem{-32768, "-16b68", "16b68"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<13>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "11", "11"},
    TestItem{32, "26", "26"},
    TestItem{227, "146", "146"},
    TestItem{893, "539", "539"},
    TestItem{911, "551", "551"},
    TestItem{1019, "605", "605"},
    TestItem{3333, "1695", "1695"},
    TestItem{9704, "4556", "4556"},
    TestItem{23232, "a761", "a761"},
    TestItem{32766, "11bb6", "11bb6"},
    TestItem{32767, "11bb7", "11bb7"},
    TestItem{-1, "-1", "23aa2"},
    TestItem{-17, "-14", "23a8c"},
    TestItem{-294, "-198", "23908"},
    TestItem{-777, "-47a", "23626"},
    TestItem{-1123, "-685", "2341b"},
    TestItem{-3123, "-1563", "22540"},
    TestItem{-12345, "-5808", "1b298"},
    TestItem{-32767, "-11bb7", "11bb9"},
    TestItem{-32768, "-11bb8", "11bb8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<14>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "10", "10"},
    TestItem{32, "24", "24"},
    TestItem{227, "123", "123"},
    TestItem{893, "47b", "47b"},
    TestItem{911, "491", "491"},
    TestItem{1019, "52b", "52b"},
    TestItem{3333, "1301", "1301"},
    TestItem{9704, "3772", "3772"},
    TestItem{23232, "8676", "8676"},
    TestItem{32766, "bd26", "bd26"},
    TestItem{32767, "bd27", "bd27"},
    TestItem{-1, "-1", "19c51"},
    TestItem{-17, "-13", "19c3d"},
    TestItem{-294, "-170", "19ac2"},
    TestItem{-777, "-3d7", "19859"},
    TestItem{-1123, "-5a3", "1968d"},
    TestItem{-3123, "-11d1", "18a61"},
    TestItem{-12345, "-46db", "15555"},
    TestItem{-32767, "-bd27", "bd29"},
    TestItem{-32768, "-bd28", "bd28"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<15>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "22", "22"},
    TestItem{227, "102", "102"},
    TestItem{893, "3e8", "3e8"},
    TestItem{911, "40b", "40b"},
    TestItem{1019, "47e", "47e"},
    TestItem{3333, "ec3", "ec3"},
    TestItem{9704, "2d1e", "2d1e"},
    TestItem{23232, "6d3c", "6d3c"},
    TestItem{32766, "9a96", "9a96"},
    TestItem{32767, "9a97", "9a97"},
    TestItem{-1, "-1", "14640"},
    TestItem{-17, "-12", "1462e"},
    TestItem{-294, "-149", "144e7"},
    TestItem{-777, "-36c", "142c4"},
    TestItem{-1123, "-4ed", "14143"},
    TestItem{-3123, "-dd3", "1375d"},
    TestItem{-12345, "-39d0", "10b61"},
    TestItem{-32767, "-9a97", "9a99"},
    TestItem{-32768, "-9a98", "9a98"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<16>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "20", "20"},
    TestItem{227, "e3", "e3"},
    TestItem{893, "37d", "37d"},
    TestItem{911, "38f", "38f"},
    TestItem{1019, "3fb", "3fb"},
    TestItem{3333, "d05", "d05"},
    TestItem{9704, "25e8", "25e8"},
    TestItem{23232, "5ac0", "5ac0"},
    TestItem{32766, "7ffe", "7ffe"},
    TestItem{32767, "7fff", "7fff"},
    TestItem{-1, "-1", "ffff"},
    TestItem{-17, "-11", "ffef"},
    TestItem{-294, "-126", "feda"},
    TestItem{-777, "-309", "fcf7"},
    TestItem{-1123, "-463", "fb9d"},
    TestItem{-3123, "-c33", "f3cd"},
    TestItem{-12345, "-3039", "cfc7"},
    TestItem{-32767, "-7fff", "8001"},
    TestItem{-32768, "-8000", "8000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<17>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "1f", "1f"},
    TestItem{227, "d6", "d6"},
    TestItem{893, "319", "319"},
    TestItem{911, "32a", "32a"},
    TestItem{1019, "38g", "38g"},
    TestItem{3333, "b91", "b91"},
    TestItem{9704, "1g9e", "1g9e"},
    TestItem{23232, "4c6a", "4c6a"},
    TestItem{32766, "6b67", "6b67"},
    TestItem{32767, "6b68", "6b68"},
    TestItem{-1, "-1", "d5d0"},
    TestItem{-17, "-10", "d5c1"},
    TestItem{-294, "-105", "d4cd"},
    TestItem{-777, "-2bc", "d316"},
    TestItem{-1123, "-3f1", "d1f0"},
    TestItem{-3123, "-adc", "cbg6"},
    TestItem{-12345, "-28c3", "ae0f"},
    TestItem{-32767, "-6b68", "6b6a"},
    TestItem{-32768, "-6b69", "6b69"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<18>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "1e", "1e"},
    TestItem{227, "cb", "cb"},
    TestItem{893, "2db", "2db"},
    TestItem{911, "2eb", "2eb"},
    TestItem{1019, "32b", "32b"},
    TestItem{3333, "a53", "a53"},
    TestItem{9704, "1bh2", "1bh2"},
    TestItem{23232, "3hcc", "3hcc"},
    TestItem{32766, "5b26", "5b26"},
    TestItem{32767, "5b27", "5b27"},
    TestItem{-1, "-1", "b44f"},
    TestItem{-17, "-h", "b43h"},
    TestItem{-294, "-g6", "b36a"},
    TestItem{-777, "-273", "b1fd"},
    TestItem{-1123, "-387", "b0e9"},
    TestItem{-3123, "-9b9", "acb7"},
    TestItem{-12345, "-221f", "9231"},
    TestItem{-32767, "-5b27", "5b29"},
    TestItem{-32768, "-5b28", "5b28"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<19>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "1d", "1d"},
    TestItem{227, "bi", "bi"},
    TestItem{893, "290", "290"},
    TestItem{911, "29i", "29i"},
    TestItem{1019, "2fc", "2fc"},
    TestItem{3333, "948", "948"},
    TestItem{9704, "17ge", "17ge"},
    TestItem{23232, "376e", "376e"},
    TestItem{32766, "4eea", "4eea"},
    TestItem{32767, "4eeb", "4eeb"},
    TestItem{-1, "-1", "9aa4"},
    TestItem{-17, "-h", "9a97"},
    TestItem{-294, "-f9", "99df"},
    TestItem{-777, "-22h", "9877"},
    TestItem{-1123, "-322", "9783"},
    TestItem{-3123, "-8c7", "91gh"},
    TestItem{-12345, "-1f3e", "7e6a"},
    TestItem{-32767, "-4eeb", "4eed"},
    TestItem{-32768, "-4eec", "4eec"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<20>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "1c", "1c"},
    TestItem{227, "b7", "b7"},
    TestItem{893, "24d", "24d"},
    TestItem{911, "25b", "25b"},
    TestItem{1019, "2aj", "2aj"},
    TestItem{3333, "86d", "86d"},
    TestItem{9704, "1454", "1454"},
    TestItem{23232, "2i1c", "2i1c"},
    TestItem{32766, "41i6", "41i6"},
    TestItem{32767, "41i7", "41i7"},
    TestItem{-1, "-1", "83gf"},
    TestItem{-17, "-h", "83fj"},
    TestItem{-294, "-ee", "8322"},
    TestItem{-777, "-1ih", "81hj"},
    TestItem{-1123, "-2g3", "810d"},
    TestItem{-3123, "-7g3", "7g0d"},
    TestItem{-12345, "-1ah5", "6cjb"},
    TestItem{-32767, "-41i7", "41i9"},
    TestItem{-32768, "-41i8", "41i8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<21>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "1b", "1b"},
    TestItem{227, "ah", "ah"},
    TestItem{893, "20b", "20b"},
    TestItem{911, "218", "218"},
    TestItem{1019, "26b", "26b"},
    TestItem{3333, "7bf", "7bf"},
    TestItem{9704, "1102", "1102"},
    TestItem{23232, "2ae6", "2ae6"},
    TestItem{32766, "3b66", "3b66"},
    TestItem{32767, "3b67", "3b67"},
    TestItem{-1, "-1", "71cf"},
    TestItem{-17, "-h", "71bk"},
    TestItem{-294, "-e0", "70jg"},
    TestItem{-777, "-1g0", "6khg"},
    TestItem{-1123, "-2ba", "6k16"},
    TestItem{-3123, "-71f", "6fb1"},
    TestItem{-12345, "-16ki", "5fcj"},
    TestItem{-32767, "-3b67", "3b69"},
    TestItem{-32768, "-3b68", "3b68"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<22>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "1a", "1a"},
    TestItem{227, "a7", "a7"},
    TestItem{893, "1id", "1id"},
    TestItem{911, "1j9", "1j9"},
    TestItem{1019, "227", "227"},
    TestItem{3333, "6jb", "6jb"},
    TestItem{9704, "k12", "k12"},
    TestItem{23232, "2400", "2400"},
    TestItem{32766, "31f8", "31f8"},
    TestItem{32767, "31f9", "31f9"},
    TestItem{-1, "-1", "638j"},
    TestItem{-17, "-h", "6383"},
    TestItem{-294, "-d8", "62hc"},
    TestItem{-777, "-1d7", "61hd"},
    TestItem{-1123, "-271", "611j"},
    TestItem{-3123, "-69l", "5ikl"},
    TestItem{-12345, "-13b3", "4ljh"},
    TestItem{-32767, "-31f9", "31fb"},
    TestItem{-32768, "-31fa", "31fa"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<23>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "19", "19"},
    TestItem{227, "9k", "9k"},
    TestItem{893, "1fj", "1fj"},
    TestItem{911, "1ge", "1ge"},
    TestItem{1019, "1l7", "1l7"},
    TestItem{3333, "66l", "66l"},
    TestItem{9704, "i7l", "i7l"},
    TestItem{23232, "1kl2", "1kl2"},
    TestItem{32766, "2fle", "2fle"},
    TestItem{32767, "2flf", "2flf"},
    TestItem{-1, "-1", "58k8"},
    TestItem{-17, "-h", "58jf"},
    TestItem{-294, "-ci", "587e"},
    TestItem{-777, "-1ai", "579e"},
    TestItem{-1123, "-22j", "56hd"},
    TestItem{-3123, "-5ki", "52me"},
    TestItem{-12345, "-107h", "48cf"},
    TestItem{-32767, "-2flf", "2flh"},
    TestItem{-32768, "-2flg", "2flg"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<24>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "18", "18"},
    TestItem{227, "9b", "9b"},
    TestItem{893, "1d5", "1d5"},
    TestItem{911, "1dn", "1dn"},
    TestItem{1019, "1ib", "1ib"},
    TestItem{3333, "5il", "5il"},
    TestItem{9704, "gk8", "gk8"},
    TestItem{23232, "1g80", "1g80"},
    TestItem{32766, "28l6", "28l6"},
    TestItem{32767, "28l7", "28l7"},
    TestItem{-1, "-1", "4hif"},
    TestItem{-17, "-h", "4hhn"},
    TestItem{-294, "-c6", "4h6a"},
    TestItem{-777, "-189", "4ga7"},
    TestItem{-1123, "-1mj", "4fjl"},
    TestItem{-3123, "-5a3", "4c8d"},
    TestItem{-12345, "-la9", "3k87"},
    TestItem{-32767, "-28l7", "28l9"},
    TestItem{-32768, "-28l8", "28l8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<25>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "17", "17"},
    TestItem{227, "92", "92"},
    TestItem{893, "1ai", "1ai"},
    TestItem{911, "1bb", "1bb"},
    TestItem{1019, "1fj", "1fj"},
    TestItem{3333, "588", "588"},
    TestItem{9704, "fd4", "fd4"},
    TestItem{23232, "1c47", "1c47"},
    TestItem{32766, "22ag", "22ag"},
    TestItem{32767, "22ah", "22ah"},
    TestItem{-1, "-1", "44la"},
    TestItem{-17, "-h", "44kj"},
    TestItem{-294, "-bj", "449h"},
    TestItem{-777, "-162", "43f9"},
    TestItem{-1123, "-1jn", "431d"},
    TestItem{-3123, "-4on", "3old"},
    TestItem{-12345, "-jik", "3a2g"},
    TestItem{-32767, "-22ah", "22aj"},
    TestItem{-32768, "-22ai", "22ai"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<26>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "16", "16"},
    TestItem{227, "8j", "8j"},
    TestItem{893, "189", "189"},
    TestItem{911, "191", "191"},
    TestItem{1019, "1d5", "1d5"},
    TestItem{3333, "4o5", "4o5"},
    TestItem{9704, "e96", "e96"},
    TestItem{23232, "189e", "189e"},
    TestItem{32766, "1mc6", "1mc6"},
    TestItem{32767, "1mc7", "1mc7"},
    TestItem{-1, "-1", "3iof"},
    TestItem{-17, "-h", "3inp"},
    TestItem{-294, "-b8", "3id8"},
    TestItem{-777, "-13n", "3hkj"},
    TestItem{-1123, "-1h5", "3h7b"},
    TestItem{-3123, "-4g3", "3e8d"},
    TestItem{-12345, "-i6l", "30hl"},
    TestItem{-32767, "-1mc7", "1mc9"},
    TestItem{-32768, "-1mc8", "1mc8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<27>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "15", "15"},
    TestItem{227, "8b", "8b"},
    TestItem{893, "162", "162"},
    TestItem{911, "16k", "16k"},
    TestItem{1019, "1ak", "1ak"},
    TestItem{3333, "4fc", "4fc"},
    TestItem{9704, "d8b", "d8b"},
    TestItem{23232, "14nc", "14nc"},
    TestItem{32766, "1hpf", "1hpf"},
    TestItem{32767, "1hpg", "1hpg"},
    TestItem{-1, "-1", "38o6"},
    TestItem{-17, "-h", "38nh"},
    TestItem{-294, "-ao", "38da"},
    TestItem{-777, "-11l", "37md"},
    TestItem{-1123, "-1eg", "379i"},
    TestItem{-3123, "-47i", "34gg"},
    TestItem{-12345, "-gp6", "2iq1"},
    TestItem{-32767, "-1hpg", "1hpi"},
    TestItem{-32768, "-1hph", "1hph"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<28>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "14", "14"},
    TestItem{227, "83", "83"},
    TestItem{893, "13p", "13p"},
    TestItem{911, "14f", "14f"},
    TestItem{1019, "18b", "18b"},
    TestItem{3333, "471", "471"},
    TestItem{9704, "cag", "cag"},
    TestItem{23232, "11hk", "11hk"},
    TestItem{32766, "1dm6", "1dm6"},
    TestItem{32767, "1dm7", "1dm7"},
    TestItem{-1, "-1", "2rgf"},
    TestItem{-17, "-h", "2rfr"},
    TestItem{-294, "-ae", "2r62"},
    TestItem{-777, "-rl", "2qgn"},
    TestItem{-1123, "-1c3", "2q4d"},
    TestItem{-3123, "-3rf", "2nh1"},
    TestItem{-12345, "-fkp", "2bnj"},
    TestItem{-32767, "-1dm7", "1dm9"},
    TestItem{-32768, "-1dm8", "1dm8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<29>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "13", "13"},
    TestItem{227, "7o", "7o"},
    TestItem{893, "11n", "11n"},
    TestItem{911, "12c", "12c"},
    TestItem{1019, "164", "164"},
    TestItem{3333, "3rr", "3rr"},
    TestItem{9704, "bfi", "bfi"},
    TestItem{23232, "ri3", "ri3"},
    TestItem{32766, "19rp", "19rp"},
    TestItem{32767, "19rq", "19rq"},
    TestItem{-1, "-1", "2jqo"},
    TestItem{-17, "-h", "2jq8"},
    TestItem{-294, "-a4", "2jgl"},
    TestItem{-777, "-qn", "2j02"},
    TestItem{-1123, "-19l", "2ih4"},
    TestItem{-3123, "-3kk", "2g65"},
    TestItem{-12345, "-ejk", "2575"},
    TestItem{-32767, "-19rq", "19rs"},
    TestItem{-32768, "-19rr", "19rr"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<30>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "12", "12"},
    TestItem{227, "7h", "7h"},
    TestItem{893, "tn", "tn"},
    TestItem{911, "10b", "10b"},
    TestItem{1019, "13t", "13t"},
    TestItem{3333, "3l3", "3l3"},
    TestItem{9704, "ane", "ane"},
    TestItem{23232, "poc", "poc"},
    TestItem{32766, "16c6", "16c6"},
    TestItem{32767, "16c7", "16c7"},
    TestItem{-1, "-1", "2cof"},
    TestItem{-17, "-h", "2cnt"},
    TestItem{-294, "-9o", "2cem"},
    TestItem{-777, "-pr", "2bsj"},
    TestItem{-1123, "-17d", "2bh3"},
    TestItem{-3123, "-3e3", "29ad"},
    TestItem{-12345, "-dlf", "1t31"},
    TestItem{-32767, "-16c7", "16c9"},
    TestItem{-32768, "-16c8", "16c8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<31>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "11", "11"},
    TestItem{227, "7a", "7a"},
    TestItem{893, "sp", "sp"},
    TestItem{911, "tc", "tc"},
    TestItem{1019, "11r", "11r"},
    TestItem{3333, "3eg", "3eg"},
    TestItem{9704, "a31", "a31"},
    TestItem{23232, "o5d", "o5d"},
    TestItem{32766, "132u", "132u"},
    TestItem{32767, "1330", "1330"},
    TestItem{-1, "-1", "2661"},
    TestItem{-17, "-h", "265g"},
    TestItem{-294, "-9f", "25ri"},
    TestItem{-777, "-p2", "25c0"},
    TestItem{-1123, "-157", "250q"},
    TestItem{-3123, "-37n", "22ta"},
    TestItem{-12345, "-cq7", "1oaq"},
    TestItem{-32767, "-1330", "1332"},
    TestItem{-32768, "-1331", "1331"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<32>()
{
  return {{
    TestItem{0, "0", "0"},
    TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},
    TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},
    TestItem{32, "10", "10"},
    TestItem{227, "73", "73"},
    TestItem{893, "rt", "rt"},
    TestItem{911, "sf", "sf"},
    TestItem{1019, "vr", "vr"},
    TestItem{3333, "385", "385"},
    TestItem{9704, "9f8", "9f8"},
    TestItem{23232, "mm0", "mm0"},
    TestItem{32766, "vvu", "vvu"},
    TestItem{32767, "vvv", "vvv"},
    TestItem{-1, "-1", "1vvv"},
    TestItem{-17, "-h", "1vvf"},
    TestItem{-294, "-96", "1vmq"},
    TestItem{-777, "-o9", "1v7n"},
    TestItem{-1123, "-133", "1ust"},
    TestItem{-3123, "-31j", "1sud"},
    TestItem{-12345, "-c1p", "1ju7"},
    TestItem{-32767, "-vvv", "1001"},
    TestItem{-32768, "-1000", "1000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<33>()
{
  return {{
    TestItem{0, "0", "0"},           TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},           TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},          TestItem{32, "w", "w"},
    TestItem{227, "6t", "6t"},       TestItem{893, "r2", "r2"},
    TestItem{911, "rk", "rk"},       TestItem{1019, "ut", "ut"},
    TestItem{3333, "320", "320"},    TestItem{9704, "8u2", "8u2"},
    TestItem{23232, "lb0", "lb0"},   TestItem{32766, "u2u", "u2u"},
    TestItem{32767, "u2v", "u2v"},   TestItem{-1, "-1", "1r5u"},
    TestItem{-17, "-h", "1r5e"},     TestItem{-294, "-8u", "1qu1"},
    TestItem{-777, "-ni", "1qfd"},   TestItem{-1123, "-111", "1q4u"},
    TestItem{-3123, "-2sl", "1oaa"}, TestItem{-12345, "-bb3", "1frs"},
    TestItem{-32767, "-u2v", "u30"}, TestItem{-32768, "-u2w", "u2w"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<34>()
{
  return {{
    TestItem{0, "0", "0"},           TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},           TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},          TestItem{32, "w", "w"},
    TestItem{227, "6n", "6n"},       TestItem{893, "q9", "q9"},
    TestItem{911, "qr", "qr"},       TestItem{1019, "tx", "tx"},
    TestItem{3333, "2u1", "2u1"},    TestItem{9704, "8de", "8de"},
    TestItem{23232, "k3a", "k3a"},   TestItem{32766, "sbo", "sbo"},
    TestItem{32767, "sbp", "sbp"},   TestItem{-1, "-1", "1mnh"},
    TestItem{-17, "-h", "1mn1"},     TestItem{-294, "-8m", "1meu"},
    TestItem{-777, "-mt", "1m0n"},   TestItem{-1123, "-x1", "1loh"},
    TestItem{-3123, "-2nt", "1jxn"}, TestItem{-12345, "-an3", "1c0f"},
    TestItem{-32767, "-sbp", "sbr"}, TestItem{-32768, "-sbq", "sbq"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<35>()
{
  return {{
    TestItem{0, "0", "0"},           TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},           TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},          TestItem{32, "w", "w"},
    TestItem{227, "6h", "6h"},       TestItem{893, "pi", "pi"},
    TestItem{911, "q1", "q1"},       TestItem{1019, "t4", "t4"},
    TestItem{3333, "2p8", "2p8"},    TestItem{9704, "7w9", "7w9"},
    TestItem{23232, "ixr", "ixr"},   TestItem{32766, "qq6", "qq6"},
    TestItem{32767, "qq7", "qq7"},   TestItem{-1, "-1", "1ihf"},
    TestItem{-17, "-h", "1igy"},     TestItem{-294, "-8e", "1i92"},
    TestItem{-777, "-m7", "1hu9"},   TestItem{-1123, "-w3", "1hkd"},
    TestItem{-3123, "-2j8", "1fx8"}, TestItem{-12345, "-a2p", "18eq"},
    TestItem{-32767, "-qq7", "qq9"}, TestItem{-32768, "-qq8", "qq8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 24> get_test_items<36>()
{
  return {{
    TestItem{0, "0", "0"},           TestItem{1, "1", "1"},
    TestItem{2, "2", "2"},           TestItem{7, "7", "7"},
    TestItem{14, "e", "e"},          TestItem{32, "w", "w"},
    TestItem{227, "6b", "6b"},       TestItem{893, "ot", "ot"},
    TestItem{911, "pb", "pb"},       TestItem{1019, "sb", "sb"},
    TestItem{3333, "2kl", "2kl"},    TestItem{9704, "7hk", "7hk"},
    TestItem{23232, "hxc", "hxc"},   TestItem{32766, "pa6", "pa6"},
    TestItem{32767, "pa7", "pa7"},   TestItem{-1, "-1", "1ekf"},
    TestItem{-17, "-h", "1ejz"},     TestItem{-294, "-86", "1eca"},
    TestItem{-777, "-ll", "1dyv"},   TestItem{-1123, "-v7", "1dp9"},
    TestItem{-3123, "-2er", "1c5p"}, TestItem{-12345, "-9ix", "151j"},
    TestItem{-32767, "-pa7", "pa9"}, TestItem{-32768, "-pa8", "pa8"},
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
    test_to_chars<cuda::std::int16_t, Base>(item);
    test_to_chars<cuda::std::uint16_t, Base>(item);
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
