//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo: Once we can pass additional flags to the compiler, remove the `test_constexpr` and execute the whole `test`
// instead

#include <cuda/std/__charconv_>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

constexpr int first_base = 2;
constexpr int last_base  = 36;

struct TestItem
{
  int64_t val; // todo: we currently test values in the range of int64_t, expand it to int128_t
  const char* str_signed;
  const char* str_unsigned = str_signed;
};

template <int Base>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items();

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<2>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "10"},
    TestItem{3ll, "11"},
    TestItem{4ll, "100"},
    TestItem{7ll, "111"},
    TestItem{8ll, "1000"},
    TestItem{14ll, "1110"},
    TestItem{16ll, "10000"},
    TestItem{23ll, "10111"},
    TestItem{32ll, "100000"},
    TestItem{36ll, "100100"},
    TestItem{127ll, "1111111"},
    TestItem{200ll, "11001000"},
    TestItem{255ll, "11111111"},
    TestItem{8158ll, "1111111011110"},
    TestItem{50243ll, "1100010001000011"},
    TestItem{32767ll, "111111111111111"},
    TestItem{87231ll, "10101010010111111"},
    TestItem{65535ll, "1111111111111111"},
    TestItem{17875098ll, "1000100001100000010011010"},
    TestItem{1787987597ll, "1101010100100101000011010001101"},
    TestItem{2147483647ll, "1111111111111111111111111111111"},
    TestItem{4294967295ll, "11111111111111111111111111111111"},
    TestItem{657687465411ll, "1001100100100001001110100100110111000011"},
    TestItem{89098098408713ll, "10100010000100011000100100000010111110100001001"},
    TestItem{4987654351689798ll, "10001101110000011111011000101111001010001100001000110"},
    TestItem{9223372036854775807ll, "111111111111111111111111111111111111111111111111111111111111111"},
    TestItem{-1ll, "-1", "1111111111111111111111111111111111111111111111111111111111111111"},
    TestItem{-13ll, "-1101", "1111111111111111111111111111111111111111111111111111111111110011"},
    TestItem{-128ll, "-10000000", "1111111111111111111111111111111111111111111111111111111110000000"},
    TestItem{-879ll, "-1101101111", "1111111111111111111111111111111111111111111111111111110010010001"},
    TestItem{-12345ll, "-11000000111001", "1111111111111111111111111111111111111111111111111100111111000111"},
    TestItem{-32768ll, "-1000000000000000", "1111111111111111111111111111111111111111111111111000000000000000"},
    TestItem{-5165781ll, "-10011101101001011010101", "1111111111111111111111111111111111111111101100010010110100101011"},
    TestItem{
      -97897347ll, "-101110101011100101110000011", "1111111111111111111111111111111111111010001010100011010001111101"},
    TestItem{-2147483648ll,
             "-10000000000000000000000000000000",
             "1111111111111111111111111111111110000000000000000000000000000000"},
    TestItem{-165789751156ll,
             "-10011010011001110101101101011101110100",
             "1111111111111111111111111101100101100110001010010010100010001100"},
    TestItem{-8798798743521135ll,
             "-11111010000100111010111111001100011101100011101101111",
             "1111111111100000101111011000101000000110011100010011100010010001"},
    TestItem{-9223372036854775807ll - 1,
             "-1000000000000000000000000000000000000000000000000000000000000000",
             "1000000000000000000000000000000000000000000000000000000000000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<3>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "10"},
    TestItem{4ll, "11"},
    TestItem{7ll, "21"},
    TestItem{8ll, "22"},
    TestItem{14ll, "112"},
    TestItem{16ll, "121"},
    TestItem{23ll, "212"},
    TestItem{32ll, "1012"},
    TestItem{36ll, "1100"},
    TestItem{127ll, "11201"},
    TestItem{200ll, "21102"},
    TestItem{255ll, "100110"},
    TestItem{8158ll, "102012011"},
    TestItem{50243ll, "2112220212"},
    TestItem{32767ll, "1122221121"},
    TestItem{87231ll, "11102122210"},
    TestItem{65535ll, "10022220020"},
    TestItem{17875098ll, "1020122011000200"},
    TestItem{1787987597ll, "11121121102011212212"},
    TestItem{2147483647ll, "12112122212110202101"},
    TestItem{4294967295ll, "102002022201221111210"},
    TestItem{657687465411ll, "2022212121100222221011020"},
    TestItem{89098098408713ll, "102200110200202200010100101002"},
    TestItem{4987654351689798ll, "220020001211002200102102220021000"},
    TestItem{9223372036854775807ll, "2021110011022210012102010021220101220221"},
    TestItem{-1ll, "-1", "11112220022122120101211020120210210211220"},
    TestItem{-13ll, "-111", "11112220022122120101211020120210210211110"},
    TestItem{-128ll, "-11202", "11112220022122120101211020120210210200012"},
    TestItem{-879ll, "-1012120", "11112220022122120101211020120210202122101"},
    TestItem{-12345ll, "-121221020", "11112220022122120101211020120210011220201"},
    TestItem{-32768ll, "-1122221122", "11112220022122120101211020120202010220022"},
    TestItem{-5165781ll, "-100201110010020", "11112220022122120101211020020002100201201"},
    TestItem{-97897347ll, "-20211012200220200", "11112220022122120101210222202121002221021"},
    TestItem{-2147483648ll, "-12112122212110202102", "11112220022122120101121200220221100002112"},
    TestItem{-165789751156ll, "-120211221011122222110221", "11112220022122112210222022102010211101000"},
    TestItem{-8798798743521135ll, "-1120201211221221111212210201011020", "11112211201220201102212201201000002200201"},
    TestItem{-9223372036854775807ll - 1,
             "-2021110011022210012102010021220101220222",
             "2021110011022210012102010021220101220222"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<4>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "10"},
    TestItem{7ll, "13"},
    TestItem{8ll, "20"},
    TestItem{14ll, "32"},
    TestItem{16ll, "100"},
    TestItem{23ll, "113"},
    TestItem{32ll, "200"},
    TestItem{36ll, "210"},
    TestItem{127ll, "1333"},
    TestItem{200ll, "3020"},
    TestItem{255ll, "3333"},
    TestItem{8158ll, "1333132"},
    TestItem{50243ll, "30101003"},
    TestItem{32767ll, "13333333"},
    TestItem{87231ll, "111102333"},
    TestItem{65535ll, "33333333"},
    TestItem{17875098ll, "1010030002122"},
    TestItem{1787987597ll, "1222210220122031"},
    TestItem{2147483647ll, "1333333333333333"},
    TestItem{4294967295ll, "3333333333333333"},
    TestItem{657687465411ll, "21210201032210313003"},
    TestItem{89098098408713ll, "110100203010200113310021"},
    TestItem{4987654351689798ll, "101232003323011321101201012"},
    TestItem{9223372036854775807ll, "13333333333333333333333333333333"},
    TestItem{-1ll, "-1", "33333333333333333333333333333333"},
    TestItem{-13ll, "-31", "33333333333333333333333333333303"},
    TestItem{-128ll, "-2000", "33333333333333333333333333332000"},
    TestItem{-879ll, "-31233", "33333333333333333333333333302101"},
    TestItem{-12345ll, "-3000321", "33333333333333333333333330333013"},
    TestItem{-32768ll, "-20000000", "33333333333333333333333320000000"},
    TestItem{-5165781ll, "-103231023111", "33333333333333333333230102310223"},
    TestItem{-97897347ll, "-11311130232003", "33333333333333333322022203101331"},
    TestItem{-2147483648ll, "-2000000000000000", "33333333333333332000000000000000"},
    TestItem{-165789751156ll, "-2122121311231131310", "33333333333331211212022102202030"},
    TestItem{-8798798743521135ll, "-133100213113321203230131233", "33333200233120220012130103202101"},
    TestItem{-9223372036854775807ll - 1, "-20000000000000000000000000000000", "20000000000000000000000000000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<5>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "12"},
    TestItem{8ll, "13"},
    TestItem{14ll, "24"},
    TestItem{16ll, "31"},
    TestItem{23ll, "43"},
    TestItem{32ll, "112"},
    TestItem{36ll, "121"},
    TestItem{127ll, "1002"},
    TestItem{200ll, "1300"},
    TestItem{255ll, "2010"},
    TestItem{8158ll, "230113"},
    TestItem{50243ll, "3101433"},
    TestItem{32767ll, "2022032"},
    TestItem{87231ll, "10242411"},
    TestItem{65535ll, "4044120"},
    TestItem{17875098ll, "14034000343"},
    TestItem{1787987597ll, "12130211100342"},
    TestItem{2147483647ll, "13344223434042"},
    TestItem{4294967295ll, "32244002423140"},
    TestItem{657687465411ll, "41233420442343121"},
    TestItem{89098098408713ll, "43134240401143034323"},
    TestItem{4987654351689798ll, "20212220212103013033143"},
    TestItem{9223372036854775807ll, "1104332401304422434310311212"},
    TestItem{-1ll, "-1", "2214220303114400424121122430"},
    TestItem{-13ll, "-23", "2214220303114400424121122403"},
    TestItem{-128ll, "-1003", "2214220303114400424121121423"},
    TestItem{-879ll, "-12004", "2214220303114400424121110422"},
    TestItem{-12345ll, "-343340", "2214220303114400424120224041"},
    TestItem{-32768ll, "-2022033", "2214220303114400424114100343"},
    TestItem{-5165781ll, "-2310301111", "2214220303114400421310321320"},
    TestItem{-97897347ll, "-200030203342", "2214220303114400224040414034"},
    TestItem{-2147483648ll, "-13344223434043", "2214220303114332024342133333"},
    TestItem{-165789751156ll, "-10204014134014111", "2214220303104141404432103320"},
    TestItem{-8798798743521135ll, "-33211234004311320134020", "2214132041330341112300433411"},
    TestItem{-9223372036854775807ll - 1, "-1104332401304422434310311213", "1104332401304422434310311213"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<6>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "11"},
    TestItem{8ll, "12"},
    TestItem{14ll, "22"},
    TestItem{16ll, "24"},
    TestItem{23ll, "35"},
    TestItem{32ll, "52"},
    TestItem{36ll, "100"},
    TestItem{127ll, "331"},
    TestItem{200ll, "532"},
    TestItem{255ll, "1103"},
    TestItem{8158ll, "101434"},
    TestItem{50243ll, "1024335"},
    TestItem{32767ll, "411411"},
    TestItem{87231ll, "1511503"},
    TestItem{65535ll, "1223223"},
    TestItem{17875098ll, "1435043030"},
    TestItem{1787987597ll, "453230440205"},
    TestItem{2147483647ll, "553032005531"},
    TestItem{4294967295ll, "1550104015503"},
    TestItem{657687465411ll, "1222045404520523"},
    TestItem{89098098408713ll, "513255033520304345"},
    TestItem{4987654351689798ll, "121035504435532434130"},
    TestItem{9223372036854775807ll, "1540241003031030222122211"},
    TestItem{-1ll, "-1", "3520522010102100444244423"},
    TestItem{-13ll, "-21", "3520522010102100444244403"},
    TestItem{-128ll, "-332", "3520522010102100444244052"},
    TestItem{-879ll, "-4023", "3520522010102100444240401"},
    TestItem{-12345ll, "-133053", "3520522010102100444111331"},
    TestItem{-32768ll, "-411412", "3520522010102100443433012"},
    TestItem{-5165781ll, "-302415353", "3520522010102100141425031"},
    TestItem{-97897347ll, "-13414140243", "3520522010102043030104141"},
    TestItem{-2147483648ll, "-553032005532", "3520522010101103412234452"},
    TestItem{-165789751156ll, "-204055053424124", "3520522005454001350420300"},
    TestItem{-8798798743521135ll, "-222345252201335025223", "3520255220405455105215201"},
    TestItem{-9223372036854775807ll - 1, "-1540241003031030222122212", "1540241003031030222122212"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<7>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "10"},
    TestItem{8ll, "11"},
    TestItem{14ll, "20"},
    TestItem{16ll, "22"},
    TestItem{23ll, "32"},
    TestItem{32ll, "44"},
    TestItem{36ll, "51"},
    TestItem{127ll, "241"},
    TestItem{200ll, "404"},
    TestItem{255ll, "513"},
    TestItem{8158ll, "32533"},
    TestItem{50243ll, "266324"},
    TestItem{32767ll, "164350"},
    TestItem{87231ll, "512214"},
    TestItem{65535ll, "362031"},
    TestItem{17875098ll, "304635663"},
    TestItem{1787987597ll, "62210433554"},
    TestItem{2147483647ll, "104134211161"},
    TestItem{4294967295ll, "211301422353"},
    TestItem{657687465411ll, "65342052134244"},
    TestItem{89098098408713ll, "24524060636446322"},
    TestItem{4987654351689798ll, "3030400064303245065"},
    TestItem{9223372036854775807ll, "22341010611245052052300"},
    TestItem{-1ll, "-1", "45012021522523134134601"},
    TestItem{-13ll, "-16", "45012021522523134134553"},
    TestItem{-128ll, "-242", "45012021522523134134330"},
    TestItem{-879ll, "-2364", "45012021522523134132205"},
    TestItem{-12345ll, "-50664", "45012021522523134053605"},
    TestItem{-32768ll, "-164351", "45012021522523133640221"},
    TestItem{-5165781ll, "-61623405", "45012021522523042211164"},
    TestItem{-97897347ll, "-2266054002", "45012021522520535050600"},
    TestItem{-2147483648ll, "-104134211162", "45012021522415666623410"},
    TestItem{-165789751156ll, "-14656265430121", "45012021504533535404451"},
    TestItem{-8798798743521135ll, "-5255221142205524046", "45003433301350625310523"},
    TestItem{-9223372036854775807ll - 1, "-22341010611245052052301", "22341010611245052052301"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<8>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "10"},
    TestItem{14ll, "16"},
    TestItem{16ll, "20"},
    TestItem{23ll, "27"},
    TestItem{32ll, "40"},
    TestItem{36ll, "44"},
    TestItem{127ll, "177"},
    TestItem{200ll, "310"},
    TestItem{255ll, "377"},
    TestItem{8158ll, "17736"},
    TestItem{50243ll, "142103"},
    TestItem{32767ll, "77777"},
    TestItem{87231ll, "252277"},
    TestItem{65535ll, "177777"},
    TestItem{17875098ll, "104140232"},
    TestItem{1787987597ll, "15244503215"},
    TestItem{2147483647ll, "17777777777"},
    TestItem{4294967295ll, "37777777777"},
    TestItem{657687465411ll, "11444116446703"},
    TestItem{89098098408713ll, "2420430440276411"},
    TestItem{4987654351689798ll, "215603730571214106"},
    TestItem{9223372036854775807ll, "777777777777777777777"},
    TestItem{-1ll, "-1", "1777777777777777777777"},
    TestItem{-13ll, "-15", "1777777777777777777763"},
    TestItem{-128ll, "-200", "1777777777777777777600"},
    TestItem{-879ll, "-1557", "1777777777777777776221"},
    TestItem{-12345ll, "-30071", "1777777777777777747707"},
    TestItem{-32768ll, "-100000", "1777777777777777700000"},
    TestItem{-5165781ll, "-23551325", "1777777777777754226453"},
    TestItem{-97897347ll, "-565345603", "1777777777777212432175"},
    TestItem{-2147483648ll, "-20000000000", "1777777777760000000000"},
    TestItem{-165789751156ll, "-2323165553564", "1777777775454612224214"},
    TestItem{-8798798743521135ll, "-372047277143543557", "1777405730500634234221"},
    TestItem{-9223372036854775807ll - 1, "-1000000000000000000000", "1000000000000000000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<9>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "15"},
    TestItem{16ll, "17"},
    TestItem{23ll, "25"},
    TestItem{32ll, "35"},
    TestItem{36ll, "40"},
    TestItem{127ll, "151"},
    TestItem{200ll, "242"},
    TestItem{255ll, "313"},
    TestItem{8158ll, "12164"},
    TestItem{50243ll, "75825"},
    TestItem{32767ll, "48847"},
    TestItem{87231ll, "142583"},
    TestItem{65535ll, "108806"},
    TestItem{17875098ll, "36564020"},
    TestItem{1787987597ll, "4547364785"},
    TestItem{2147483647ll, "5478773671"},
    TestItem{4294967295ll, "12068657453"},
    TestItem{657687465411ll, "2285540887136"},
    TestItem{89098098408713ll, "380420680110332"},
    TestItem{4987654351689798ll, "26201732612386230"},
    TestItem{9223372036854775807ll, "67404283172107811827"},
    TestItem{-1ll, "-1", "145808576354216723756"},
    TestItem{-13ll, "-14", "145808576354216723743"},
    TestItem{-128ll, "-152", "145808576354216723605"},
    TestItem{-879ll, "-1176", "145808576354216722571"},
    TestItem{-12345ll, "-17836", "145808576354216704821"},
    TestItem{-32768ll, "-48848", "145808576354216663808"},
    TestItem{-5165781ll, "-10643106", "145808576354206070651"},
    TestItem{-97897347ll, "-224180820", "145808576353882532837"},
    TestItem{-2147483648ll, "-5478773672", "145808576347626840075"},
    TestItem{-165789751156ll, "-524834588427", "145808575728272124330"},
    TestItem{-8798798743521135ll, "-46654857455721136", "145751821385651002621"},
    TestItem{-9223372036854775807ll - 1, "-67404283172107811828", "67404283172107811828"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<10>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "14"},
    TestItem{16ll, "16"},
    TestItem{23ll, "23"},
    TestItem{32ll, "32"},
    TestItem{36ll, "36"},
    TestItem{127ll, "127"},
    TestItem{200ll, "200"},
    TestItem{255ll, "255"},
    TestItem{8158ll, "8158"},
    TestItem{50243ll, "50243"},
    TestItem{32767ll, "32767"},
    TestItem{87231ll, "87231"},
    TestItem{65535ll, "65535"},
    TestItem{17875098ll, "17875098"},
    TestItem{1787987597ll, "1787987597"},
    TestItem{2147483647ll, "2147483647"},
    TestItem{4294967295ll, "4294967295"},
    TestItem{657687465411ll, "657687465411"},
    TestItem{89098098408713ll, "89098098408713"},
    TestItem{4987654351689798ll, "4987654351689798"},
    TestItem{9223372036854775807ll, "9223372036854775807"},
    TestItem{-1ll, "-1", "18446744073709551615"},
    TestItem{-13ll, "-13", "18446744073709551603"},
    TestItem{-128ll, "-128", "18446744073709551488"},
    TestItem{-879ll, "-879", "18446744073709550737"},
    TestItem{-12345ll, "-12345", "18446744073709539271"},
    TestItem{-32768ll, "-32768", "18446744073709518848"},
    TestItem{-5165781ll, "-5165781", "18446744073704385835"},
    TestItem{-97897347ll, "-97897347", "18446744073611654269"},
    TestItem{-2147483648ll, "-2147483648", "18446744071562067968"},
    TestItem{-165789751156ll, "-165789751156", "18446743907919800460"},
    TestItem{-8798798743521135ll, "-8798798743521135", "18437945274966030481"},
    TestItem{-9223372036854775807ll - 1, "-9223372036854775808", "9223372036854775808"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<11>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "13"},
    TestItem{16ll, "15"},
    TestItem{23ll, "21"},
    TestItem{32ll, "2a"},
    TestItem{36ll, "33"},
    TestItem{127ll, "106"},
    TestItem{200ll, "172"},
    TestItem{255ll, "212"},
    TestItem{8158ll, "6147"},
    TestItem{50243ll, "34826"},
    TestItem{32767ll, "22689"},
    TestItem{87231ll, "5a5a1"},
    TestItem{65535ll, "45268"},
    TestItem{17875098ll, "a0a990a"},
    TestItem{1787987597ll, "8382aa600"},
    TestItem{2147483647ll, "a02220281"},
    TestItem{4294967295ll, "1904440553"},
    TestItem{657687465411ll, "233a18479149"},
    TestItem{89098098408713ll, "26431322850719"},
    TestItem{4987654351689798ll, "1215247371714788"},
    TestItem{9223372036854775807ll, "1728002635214590697"},
    TestItem{-1ll, "-1", "335500516a429071284"},
    TestItem{-13ll, "-12", "335500516a429071273"},
    TestItem{-128ll, "-107", "335500516a429071179"},
    TestItem{-879ll, "-72a", "335500516a429070656"},
    TestItem{-12345ll, "-9303", "335500516a429062a82"},
    TestItem{-32768ll, "-2268a", "335500516a4290496a6"},
    TestItem{-5165781ll, "-2a09145", "335500516a426163140"},
    TestItem{-97897347ll, "-502957a9", "335500516a388886587"},
    TestItem{-2147483648ll, "-a02220282", "3355005169526951003"},
    TestItem{-165789751156ll, "-64346aa0977", "3355005106092080409"},
    TestItem{-8798798743521135ll, "-21196286565646a8", "3352996641882607688"},
    TestItem{-9223372036854775807ll - 1, "-1728002635214590698", "1728002635214590698"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<12>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "12"},
    TestItem{16ll, "14"},
    TestItem{23ll, "1b"},
    TestItem{32ll, "28"},
    TestItem{36ll, "30"},
    TestItem{127ll, "a7"},
    TestItem{200ll, "148"},
    TestItem{255ll, "193"},
    TestItem{8158ll, "487a"},
    TestItem{50243ll, "250ab"},
    TestItem{32767ll, "16b67"},
    TestItem{87231ll, "42593"},
    TestItem{65535ll, "31b13"},
    TestItem{17875098ll, "5ba0476"},
    TestItem{1787987597ll, "41a963065"},
    TestItem{2147483647ll, "4bb2308a7"},
    TestItem{4294967295ll, "9ba461593"},
    TestItem{657687465411ll, "a756a250143"},
    TestItem{89098098408713ll, "9bab984a720b5"},
    TestItem{4987654351689798ll, "3a74949b2918946"},
    TestItem{9223372036854775807ll, "41a792678515120367"},
    TestItem{-1ll, "-1", "839365134a2a240713"},
    TestItem{-13ll, "-11", "839365134a2a240703"},
    TestItem{-128ll, "-a8", "839365134a2a240628"},
    TestItem{-879ll, "-613", "839365134a2a240101"},
    TestItem{-12345ll, "-7189", "839365134a2a235547"},
    TestItem{-32768ll, "-16b68", "839365134a2a225768"},
    TestItem{-5165781ll, "-1891559", "839365134a2856b177"},
    TestItem{-97897347ll, "-28951683", "839365134a014ab051"},
    TestItem{-2147483648ll, "-4bb2308a8", "83936513452b00ba28"},
    TestItem{-165789751156ll, "-2816a79b044", "83936510887b661690"},
    TestItem{-8798798743521135ll, "-6a2a1618850b213", "83888231a861931501"},
    TestItem{-9223372036854775807ll - 1, "-41a792678515120368", "41a792678515120368"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<13>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "11"},
    TestItem{16ll, "13"},
    TestItem{23ll, "1a"},
    TestItem{32ll, "26"},
    TestItem{36ll, "2a"},
    TestItem{127ll, "9a"},
    TestItem{200ll, "125"},
    TestItem{255ll, "168"},
    TestItem{8158ll, "3937"},
    TestItem{50243ll, "19b3b"},
    TestItem{32767ll, "11bb7"},
    TestItem{87231ll, "30921"},
    TestItem{65535ll, "23aa2"},
    TestItem{17875098ll, "391b1a7"},
    TestItem{1787987597ll, "226575536"},
    TestItem{2147483647ll, "282ba4aaa"},
    TestItem{4294967295ll, "535a79888"},
    TestItem{657687465411ll, "4a034274122"},
    TestItem{89098098408713ll, "3a93bb75916a4"},
    TestItem{4987654351689798ll, "136106a306aa2b9"},
    TestItem{9223372036854775807ll, "10b269549075433c37"},
    TestItem{-1ll, "-1", "219505a9511a867b72"},
    TestItem{-13ll, "-10", "219505a9511a867b63"},
    TestItem{-128ll, "-9b", "219505a9511a867aa5"},
    TestItem{-879ll, "-528", "219505a9511a867648"},
    TestItem{-12345ll, "-5808", "219505a9511a862368"},
    TestItem{-32768ll, "-11bb8", "219505a9511a855c88"},
    TestItem{-5165781ll, "-10bb39a", "219505a951197797a6"},
    TestItem{-97897347ll, "-17388732", "219505a951034ac441"},
    TestItem{-2147483648ll, "-282ba4aab", "219505a94b67993095"},
    TestItem{-165789751156ll, "-12831900493", "219505a825b8c676b0"},
    TestItem{-8798798743521135ll, "-23087b21342b37b", "2192a5216c074397c5"},
    TestItem{-9223372036854775807ll - 1, "-10b269549075433c38", "10b269549075433c38"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<14>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "10"},
    TestItem{16ll, "12"},
    TestItem{23ll, "19"},
    TestItem{32ll, "24"},
    TestItem{36ll, "28"},
    TestItem{127ll, "91"},
    TestItem{200ll, "104"},
    TestItem{255ll, "143"},
    TestItem{8158ll, "2d8a"},
    TestItem{50243ll, "1444b"},
    TestItem{32767ll, "bd27"},
    TestItem{87231ll, "23b0b"},
    TestItem{65535ll, "19c51"},
    TestItem{17875098ll, "253436a"},
    TestItem{1787987597ll, "12d66ad9b"},
    TestItem{2147483647ll, "1652ca931"},
    TestItem{4294967295ll, "2ca5b7463"},
    TestItem{657687465411ll, "23b91964ccb"},
    TestItem{89098098408713ll, "1800529d3b449"},
    TestItem{4987654351689798ll, "63d91a89b041dc"},
    TestItem{9223372036854775807ll, "4340724c6c71dc7a7"},
    TestItem{-1ll, "-1", "8681049adb03db171"},
    TestItem{-13ll, "-d", "8681049adb03db163"},
    TestItem{-128ll, "-92", "8681049adb03db0c0"},
    TestItem{-879ll, "-46b", "8681049adb03dab05"},
    TestItem{-12345ll, "-46db", "8681049adb03d6875"},
    TestItem{-32768ll, "-bd28", "8681049adb03cd248"},
    TestItem{-5165781ll, "-986805", "8681049adad85476b"},
    TestItem{-97897347ll, "-d004c39", "8681049ada13d6337"},
    TestItem{-2147483648ll, "-1652ca932", "8681049ac49110640"},
    TestItem{-165789751156ll, "-804a833748", "86810492d639a7828"},
    TestItem{-8798798743521135ll, "-b12ac1ba9b155d", "8673d1ccbd3829a13"},
    TestItem{-9223372036854775807ll - 1, "-4340724c6c71dc7a8", "4340724c6c71dc7a8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<15>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "11"},
    TestItem{23ll, "18"},
    TestItem{32ll, "22"},
    TestItem{36ll, "26"},
    TestItem{127ll, "87"},
    TestItem{200ll, "d5"},
    TestItem{255ll, "120"},
    TestItem{8158ll, "263d"},
    TestItem{50243ll, "ed48"},
    TestItem{32767ll, "9a97"},
    TestItem{87231ll, "1aca6"},
    TestItem{65535ll, "14640"},
    TestItem{17875098ll, "18814d3"},
    TestItem{1787987597ll, "a6e84182"},
    TestItem{2147483647ll, "c87e66b7"},
    TestItem{4294967295ll, "1a20dcd80"},
    TestItem{657687465411ll, "121945751c6"},
    TestItem{89098098408713ll, "a479a542a828"},
    TestItem{4987654351689798ll, "2869550810bbd3"},
    TestItem{9223372036854775807ll, "160e2ad3246366807"},
    TestItem{-1ll, "-1", "2c1d56b648c6cd110"},
    TestItem{-13ll, "-d", "2c1d56b648c6cd103"},
    TestItem{-128ll, "-88", "2c1d56b648c6cd078"},
    TestItem{-879ll, "-3d9", "2c1d56b648c6ccc27"},
    TestItem{-12345ll, "-39d0", "2c1d56b648c6c9631"},
    TestItem{-32768ll, "-9a98", "2c1d56b648c6c3568"},
    TestItem{-5165781ll, "-6c0906", "2c1d56b648c00c70a"},
    TestItem{-97897347ll, "-88db94c", "2c1d56b6483ce16b4"},
    TestItem{-2147483648ll, "-c87e66b8", "2c1d56b63b3dd6948"},
    TestItem{-165789751156ll, "-44a4de4371", "2c1d56b1ed77d8c90"},
    TestItem{-8798798743521135ll, "-47c36e329abc40", "2c18c97e559c213c1"},
    TestItem{-9223372036854775807ll - 1, "-160e2ad3246366808", "160e2ad3246366808"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<16>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "10"},
    TestItem{23ll, "17"},
    TestItem{32ll, "20"},
    TestItem{36ll, "24"},
    TestItem{127ll, "7f"},
    TestItem{200ll, "c8"},
    TestItem{255ll, "ff"},
    TestItem{8158ll, "1fde"},
    TestItem{50243ll, "c443"},
    TestItem{32767ll, "7fff"},
    TestItem{87231ll, "154bf"},
    TestItem{65535ll, "ffff"},
    TestItem{17875098ll, "110c09a"},
    TestItem{1787987597ll, "6a92868d"},
    TestItem{2147483647ll, "7fffffff"},
    TestItem{4294967295ll, "ffffffff"},
    TestItem{657687465411ll, "99213a4dc3"},
    TestItem{89098098408713ll, "5108c4817d09"},
    TestItem{4987654351689798ll, "11b83ec5e51846"},
    TestItem{9223372036854775807ll, "7fffffffffffffff"},
    TestItem{-1ll, "-1", "ffffffffffffffff"},
    TestItem{-13ll, "-d", "fffffffffffffff3"},
    TestItem{-128ll, "-80", "ffffffffffffff80"},
    TestItem{-879ll, "-36f", "fffffffffffffc91"},
    TestItem{-12345ll, "-3039", "ffffffffffffcfc7"},
    TestItem{-32768ll, "-8000", "ffffffffffff8000"},
    TestItem{-5165781ll, "-4ed2d5", "ffffffffffb12d2b"},
    TestItem{-97897347ll, "-5d5cb83", "fffffffffa2a347d"},
    TestItem{-2147483648ll, "-80000000", "ffffffff80000000"},
    TestItem{-165789751156ll, "-2699d6d774", "ffffffd96629288c"},
    TestItem{-8798798743521135ll, "-1f4275f98ec76f", "ffe0bd8a06713891"},
    TestItem{-9223372036854775807ll - 1, "-8000000000000000", "8000000000000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<17>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "16"},
    TestItem{32ll, "1f"},
    TestItem{36ll, "22"},
    TestItem{127ll, "78"},
    TestItem{200ll, "bd"},
    TestItem{255ll, "f0"},
    TestItem{8158ll, "1b3f"},
    TestItem{50243ll, "a3e8"},
    TestItem{32767ll, "6b68"},
    TestItem{87231ll, "10ce4"},
    TestItem{65535ll, "d5d0"},
    TestItem{17875098ll, "ca0596"},
    TestItem{1787987597ll, "4614af50"},
    TestItem{2147483647ll, "53g7f548"},
    TestItem{4294967295ll, "a7ffda90"},
    TestItem{657687465411ll, "594d7e2g76"},
    TestItem{89098098408713ll, "2a3591510d44"},
    TestItem{4987654351689798ll, "8990c43d7g311"},
    TestItem{9223372036854775807ll, "33d3d8307b214008"},
    TestItem{-1ll, "-1", "67979g60f5428010"},
    TestItem{-13ll, "-d", "67979g60f5428005"},
    TestItem{-128ll, "-79", "67979g60f5427ga9"},
    TestItem{-879ll, "-30c", "67979g60f5427e06"},
    TestItem{-12345ll, "-28c3", "67979g60f542585f"},
    TestItem{-32768ll, "-6b69", "67979g60f54215b9"},
    TestItem{-5165781ll, "-3ae7b8", "67979g60f508a96a"},
    TestItem{-97897347ll, "-40g2328", "67979g60f1335dfa"},
    TestItem{-2147483648ll, "-53g7f549", "67979g60a14b9bd9"},
    TestItem{-165789751156ll, "-16d091d405", "67979g4b24c0bd0d"},
    TestItem{-8798798743521135ll, "-f1c8795a892be", "678983ea5gaafe64"},
    TestItem{-9223372036854775807ll - 1, "-33d3d8307b214009", "33d3d8307b214009"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<18>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "15"},
    TestItem{32ll, "1e"},
    TestItem{36ll, "20"},
    TestItem{127ll, "71"},
    TestItem{200ll, "b2"},
    TestItem{255ll, "e3"},
    TestItem{8158ll, "1734"},
    TestItem{50243ll, "8b15"},
    TestItem{32767ll, "5b27"},
    TestItem{87231ll, "eh43"},
    TestItem{65535ll, "b44f"},
    TestItem{17875098ll, "985010"},
    TestItem{1787987597ll, "2ga46445"},
    TestItem{2147483647ll, "3928g3h1"},
    TestItem{4294967295ll, "704he7g3"},
    TestItem{657687465411ll, "35c4e632af"},
    TestItem{89098098408713ll, "16h32e8da31b"},
    TestItem{4987654351689798ll, "45ag9ehdb42f0"},
    TestItem{9223372036854775807ll, "16agh595df825fa7"},
    TestItem{-1ll, "-1", "2d3fgb0b9cg4bd2f"},
    TestItem{-13ll, "-d", "2d3fgb0b9cg4bd23"},
    TestItem{-128ll, "-72", "2d3fgb0b9cg4bcde"},
    TestItem{-879ll, "-2cf", "2d3fgb0b9cg4ba81"},
    TestItem{-12345ll, "-221f", "2d3fgb0b9cg49b11"},
    TestItem{-32768ll, "-5b28", "2d3fgb0b9cg46208"},
    TestItem{-5165781ll, "-2d3ddf", "2d3fgb0b9cd97h71"},
    TestItem{-97897347ll, "-2fea459", "2d3fgb0b9a0818f7"},
    TestItem{-2147483648ll, "-3928g3h2", "2d3fgb0b63ddd93e"},
    TestItem{-165789751156ll, "-f0e78hgeg", "2d3fgahe8g8dbe60"},
    TestItem{-8798798743521135ll, "-7ag5fghea4baf", "2d385ccdad1c71a1"},
    TestItem{-9223372036854775807ll - 1, "-16agh595df825fa8", "16agh595df825fa8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<19>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "14"},
    TestItem{32ll, "1d"},
    TestItem{36ll, "1h"},
    TestItem{127ll, "6d"},
    TestItem{200ll, "aa"},
    TestItem{255ll, "d8"},
    TestItem{8158ll, "13b7"},
    TestItem{50243ll, "7637"},
    TestItem{32767ll, "4eeb"},
    TestItem{87231ll, "cdc2"},
    TestItem{65535ll, "9aa4"},
    TestItem{17875098ll, "74319c"},
    TestItem{1787987597ll, "2001gb47"},
    TestItem{2147483647ll, "27c57h32"},
    TestItem{4294967295ll, "4f5aff65"},
    TestItem{657687465411ll, "20ded6i0c9"},
    TestItem{89098098408713ll, "ea22b8cah4b"},
    TestItem{4987654351689798ll, "24f9b74c4f2i5"},
    TestItem{9223372036854775807ll, "ba643dci0ffeehh"},
    TestItem{-1ll, "-1", "141c8786h1ccaagg"},
    TestItem{-13ll, "-d", "141c8786h1ccaag4"},
    TestItem{-128ll, "-6e", "141c8786h1ccaaa3"},
    TestItem{-879ll, "-285", "141c8786h1cca88c"},
    TestItem{-12345ll, "-1f3e", "141c8786h1cc8ed3"},
    TestItem{-32768ll, "-4eec", "141c8786h1cc5f25"},
    TestItem{-5165781ll, "-21c2c4", "141c8786h1aah84d"},
    TestItem{-97897347ll, "-21a3fei", "141c8786gib26e1i"},
    TestItem{-2147483648ll, "-27c57h33", "141c8786ed072cde"},
    TestItem{-165789751156ll, "-9e9009d4c", "141c877g2bcc0gc5"},
    TestItem{-8798798743521135ll, "-3ia245if4bh3d", "14188g62b1g7hcd4"},
    TestItem{-9223372036854775807ll - 1, "-ba643dci0ffeehi", "ba643dci0ffeehi"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<20>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "13"},
    TestItem{32ll, "1c"},
    TestItem{36ll, "1g"},
    TestItem{127ll, "67"},
    TestItem{200ll, "a0"},
    TestItem{255ll, "cf"},
    TestItem{8158ll, "107i"},
    TestItem{50243ll, "65c3"},
    TestItem{32767ll, "41i7"},
    TestItem{87231ll, "ai1b"},
    TestItem{65535ll, "83gf"},
    TestItem{17875098ll, "5be7ei"},
    TestItem{1787987597ll, "17iei8jh"},
    TestItem{2147483647ll, "1db1f927"},
    TestItem{4294967295ll, "3723ai4f"},
    TestItem{657687465411ll, "15dg76d3ab"},
    TestItem{89098098408713ll, "8e07hff11fd"},
    TestItem{4987654351689798ll, "1471a4j4i149i"},
    TestItem{9223372036854775807ll, "5cbfjia3fh26ja7"},
    TestItem{-1ll, "-1", "b53bjh07be4dj0f"},
    TestItem{-13ll, "-d", "b53bjh07be4dj03"},
    TestItem{-128ll, "-68", "b53bjh07be4die8"},
    TestItem{-879ll, "-23j", "b53bjh07be4dggh"},
    TestItem{-12345ll, "-1ah5", "b53bjh07be4c83b"},
    TestItem{-32768ll, "-41i8", "b53bjh07be49h28"},
    TestItem{-5165781ll, "-1c5e91", "b53bjh07bcc84bf"},
    TestItem{-97897347ll, "-1abh377", "b53bjh07a3cgfd9"},
    TestItem{-2147483648ll, "-1db1f928", "b53bjh05i32i9i8"},
    TestItem{-165789751156ll, "-69a95ihhg", "b53bjgdi14if130"},
    TestItem{-8798798743521135ll, "-22j531a7702gf", "b5190bh616hdg41"},
    TestItem{-9223372036854775807ll - 1, "-5cbfjia3fh26ja8", "5cbfjia3fh26ja8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<21>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "12"},
    TestItem{32ll, "1b"},
    TestItem{36ll, "1f"},
    TestItem{127ll, "61"},
    TestItem{200ll, "9b"},
    TestItem{255ll, "c3"},
    TestItem{8158ll, "iaa"},
    TestItem{50243ll, "58jb"},
    TestItem{32767ll, "3b67"},
    TestItem{87231ll, "98gi"},
    TestItem{65535ll, "71cf"},
    TestItem{17875098ll, "47j323"},
    TestItem{1787987597ll, "khgd7db"},
    TestItem{2147483647ll, "140h2d91"},
    TestItem{4294967295ll, "281d55i3"},
    TestItem{657687465411ll, "h8380j1hi"},
    TestItem{89098098408713ll, "573e0c1i832"},
    TestItem{4987654351689798ll, "e509g876icbc"},
    TestItem{9223372036854775807ll, "2heiciiie82dh97"},
    TestItem{-1ll, "-1", "5e8g4ggg7g56dif"},
    TestItem{-13ll, "-d", "5e8g4ggg7g56di3"},
    TestItem{-128ll, "-62", "5e8g4ggg7g56dce"},
    TestItem{-879ll, "-1ki", "5e8g4ggg7g56bij"},
    TestItem{-12345ll, "-16ki", "5e8g4ggg7g556ij"},
    TestItem{-32768ll, "-3b68", "5e8g4ggg7g532c8"},
    TestItem{-5165781ll, "-15bggc", "5e8g4ggg7ekfi24"},
    TestItem{-97897347ll, "-12k7j99", "5e8g4ggg6d5jf97"},
    TestItem{-2147483648ll, "-140h2d92", "5e8g4ggf3f9409e"},
    TestItem{-165789751156ll, "-4810jfbh1", "5e8g4gc86f6c21f"},
    TestItem{-8798798743521135ll, "-142aeg3hh6d66", "5e7c26203j900ca"},
    TestItem{-9223372036854775807ll - 1, "-2heiciiie82dh98", "2heiciiie82dh98"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<22>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "11"},
    TestItem{32ll, "1a"},
    TestItem{36ll, "1e"},
    TestItem{127ll, "5h"},
    TestItem{200ll, "92"},
    TestItem{255ll, "bd"},
    TestItem{8158ll, "gii"},
    TestItem{50243ll, "4fhh"},
    TestItem{32767ll, "31f9"},
    TestItem{87231ll, "8451"},
    TestItem{65535ll, "638j"},
    TestItem{17875098ll, "3a6g0a"},
    TestItem{1787987597ll, "fgkdf5b"},
    TestItem{2147483647ll, "ikf5bf1"},
    TestItem{4294967295ll, "1fj8b183"},
    TestItem{657687465411ll, "bleg6ejd9"},
    TestItem{89098098408713ll, "37hdih28769"},
    TestItem{4987654351689798ll, "8bh7ga79fff8"},
    TestItem{9223372036854775807ll, "1adaibb21dckfa7"},
    TestItem{-1ll, "-1", "2l4lf104353j8kf"},
    TestItem{-13ll, "-d", "2l4lf104353j8k3"},
    TestItem{-128ll, "-5i", "2l4lf104353j8ek"},
    TestItem{-879ll, "-1hl", "2l4lf104353j72h"},
    TestItem{-12345ll, "-13b3", "2l4lf104353i59d"},
    TestItem{-32768ll, "-31fa", "2l4lf104353g756"},
    TestItem{-5165781ll, "-101325", "2l4lf104343i5ib"},
    TestItem{-97897347ll, "-iljl59", "2l4lf104283l9f7"},
    TestItem{-2147483648ll, "-ikf5bf2", "2l4lf10366adj5e"},
    TestItem{-165789751156ll, "-30a5akg3i", "2l4lf0j3elekegk"},
    TestItem{-8798798743521135ll, "-f1640a92a4aj", "2l46dgi3ei1949j"},
    TestItem{-9223372036854775807ll - 1, "-1adaibb21dckfa8", "1adaibb21dckfa8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<23>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "10"},
    TestItem{32ll, "19"},
    TestItem{36ll, "1d"},
    TestItem{127ll, "5c"},
    TestItem{200ll, "8g"},
    TestItem{255ll, "b2"},
    TestItem{8158ll, "f9g"},
    TestItem{50243ll, "42mb"},
    TestItem{32767ll, "2flf"},
    TestItem{87231ll, "73kf"},
    TestItem{65535ll, "58k8"},
    TestItem{17875098ll, "2hk384"},
    TestItem{1787987597ll, "c1i6jh4"},
    TestItem{2147483647ll, "ebelf95"},
    TestItem{4294967295ll, "1606k7ib"},
    TestItem{657687465411ll, "893h9911i"},
    TestItem{89098098408713ll, "23ah4568ij5"},
    TestItem{4987654351689798ll, "5593832cgbh5"},
    TestItem{9223372036854775807ll, "i6k448cf4192c2"},
    TestItem{-1ll, "-1", "1ddh88h2782i515"},
    TestItem{-13ll, "-d", "1ddh88h2782i50g"},
    TestItem{-128ll, "-5d", "1ddh88h2782i4ig"},
    TestItem{-879ll, "-1f5", "1ddh88h2782i391"},
    TestItem{-12345ll, "-107h", "1ddh88h2782h4gc"},
    TestItem{-32768ll, "-2flg", "1ddh88h2782fc2d"},
    TestItem{-5165781ll, "-iad44", "1ddh88h27777ek2"},
    TestItem{-97897347ll, "-f4j339", "1ddh88h26fkm1kk"},
    TestItem{-2147483648ll, "-ebelf96", "1ddh88h1fjajcf0"},
    TestItem{-165789751156ll, "-22fl8fd26", "1ddh88eme9h2em0"},
    TestItem{-8798798743521135ll, "-959237id6l1d", "1dd82melmccb6mg"},
    TestItem{-9223372036854775807ll - 1, "-i6k448cf4192c3", "i6k448cf4192c3"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<24>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "18"},
    TestItem{36ll, "1c"},
    TestItem{127ll, "57"},
    TestItem{200ll, "88"},
    TestItem{255ll, "af"},
    TestItem{8158ll, "e3m"},
    TestItem{50243ll, "3f5b"},
    TestItem{32767ll, "28l7"},
    TestItem{87231ll, "67af"},
    TestItem{65535ll, "4hif"},
    TestItem{17875098ll, "25l13i"},
    TestItem{1787987597ll, "98d3935"},
    TestItem{2147483647ll, "b5gge57"},
    TestItem{4294967295ll, "mb994af"},
    TestItem{657687465411ll, "5n9cjjc83"},
    TestItem{89098098408713ll, "19ha6jjmi5h"},
    TestItem{4987654351689798ll, "36fn6m8cke86"},
    TestItem{9223372036854775807ll, "acd772jnc9l0l7"},
    TestItem{-1ll, "-1", "l12ee5fn0ji1if"},
    TestItem{-13ll, "-d", "l12ee5fn0ji1i3"},
    TestItem{-128ll, "-58", "l12ee5fn0ji1d8"},
    TestItem{-879ll, "-1cf", "l12ee5fn0ji061"},
    TestItem{-12345ll, "-la9", "l12ee5fn0jh487"},
    TestItem{-32768ll, "-28l8", "l12ee5fn0jfgl8"},
    TestItem{-5165781ll, "-fdg8l", "l12ee5fn04499j"},
    TestItem{-97897347ll, "-c71gg3", "l12ee5fmccg92d"},
    TestItem{-2147483648ll, "-b5gge58", "l12ee5fbj31bd8"},
    TestItem{-165789751156ll, "-1c3cnkl24", "l12ee43jbjl4gc"},
    TestItem{-8798798743521135ll, "-5iie89l3j9cf", "l0kjjf7d3fmg61"},
    TestItem{-9223372036854775807ll - 1, "-acd772jnc9l0l8", "acd772jnc9l0l8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<25>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "17"},
    TestItem{36ll, "1b"},
    TestItem{127ll, "52"},
    TestItem{200ll, "80"},
    TestItem{255ll, "a5"},
    TestItem{8158ll, "d18"},
    TestItem{50243ll, "359i"},
    TestItem{32767ll, "22ah"},
    TestItem{87231ll, "5ee6"},
    TestItem{65535ll, "44la"},
    TestItem{17875098ll, "1kj03n"},
    TestItem{1787987597ll, "782653m"},
    TestItem{2147483647ll, "8jmdnkm"},
    TestItem{4294967295ll, "hek2mgk"},
    TestItem{657687465411ll, "47im4mjgb"},
    TestItem{89098098408713ll, "n8mkk6n3nd"},
    TestItem{4987654351689798ll, "227c275f83gn"},
    TestItem{9223372036854775807ll, "64ie1focnn5g77"},
    TestItem{-1ll, "-1", "c9c336o0mlb7ef"},
    TestItem{-13ll, "-d", "c9c336o0mlb7e3"},
    TestItem{-128ll, "-53", "c9c336o0mlb79d"},
    TestItem{-879ll, "-1a4", "c9c336o0mlb64c"},
    TestItem{-12345ll, "-jik", "c9c336o0mlackl"},
    TestItem{-32768ll, "-22ai", "c9c336o0ml953n"},
    TestItem{-5165781ll, "-d5f66", "c9c336o0m85h8a"},
    TestItem{-97897347ll, "-a0faim", "c9c336o0ckklkj"},
    TestItem{-2147483648ll, "-8jmdnkn", "c9c336nh2nm8ii"},
    TestItem{-165789751156ll, "-1241lj1l6", "c9c335llkoh5ia"},
    TestItem{-8798798743521135ll, "-3h6dk4g8a8ka", "c98ali3l6d0nj6"},
    TestItem{-9223372036854775807ll - 1, "-64ie1focnn5g78", "64ie1focnn5g78"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<26>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "16"},
    TestItem{36ll, "1a"},
    TestItem{127ll, "4n"},
    TestItem{200ll, "7i"},
    TestItem{255ll, "9l"},
    TestItem{8158ll, "c1k"},
    TestItem{50243ll, "2m8b"},
    TestItem{32767ll, "1mc7"},
    TestItem{87231ll, "4p11"},
    TestItem{65535ll, "3iof"},
    TestItem{17875098ll, "1d30bk"},
    TestItem{1787987597ll, "5kcgo1j"},
    TestItem{2147483647ll, "6oj8ion"},
    TestItem{4294967295ll, "dnchbnl"},
    TestItem{657687465411ll, "33n0cgjkf"},
    TestItem{89098098408713ll, "gah3p9d4oh"},
    TestItem{4987654351689798ll, "198g3f10hdim"},
    TestItem{9223372036854775807ll, "3igoecjbmca687"},
    TestItem{-1ll, "-1", "7b7n2pcniokcgf"},
    TestItem{-13ll, "-d", "7b7n2pcniokcg3"},
    TestItem{-128ll, "-4o", "7b7n2pcniokcbi"},
    TestItem{-879ll, "-17l", "7b7n2pcniokb8l"},
    TestItem{-12345ll, "-i6l", "7b7n2pcniojk9l"},
    TestItem{-32768ll, "-1mc8", "7b7n2pcnioig48"},
    TestItem{-5165781ll, "-b7nhn", "7b7n2pcnidceoj"},
    TestItem{-97897347ll, "-865oef", "7b7n2pcnaiee21"},
    TestItem{-2147483648ll, "-6oj8ioo", "7b7n2pcgk5bjhi"},
    TestItem{-165789751156ll, "-kghjd14g", "7b7n2oi7157bc0"},
    TestItem{-8798798743521135ll, "-2a8e9ma7ckab", "7b5ckb318h7i65"},
    TestItem{-9223372036854775807ll - 1, "-3igoecjbmca688", "3igoecjbmca688"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<27>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "15"},
    TestItem{36ll, "19"},
    TestItem{127ll, "4j"},
    TestItem{200ll, "7b"},
    TestItem{255ll, "9c"},
    TestItem{8158ll, "b54"},
    TestItem{50243ll, "2eon"},
    TestItem{32767ll, "1hpg"},
    TestItem{87231ll, "4bhl"},
    TestItem{65535ll, "38o6"},
    TestItem{17875098ll, "16h40i"},
    TestItem{1787987597ll, "4ggb4nn"},
    TestItem{2147483647ll, "5ehncka"},
    TestItem{4294967295ll, "b28jpdl"},
    TestItem{657687465411ll, "28ng9qp46"},
    TestItem{89098098408713ll, "biciki39a2"},
    TestItem{4987654351689798ll, "o61m2ibbo70"},
    TestItem{9223372036854775807ll, "27c48l5b37oaop"},
    TestItem{-1ll, "-1", "4eo8hfam6fllmo"},
    TestItem{-13ll, "-d", "4eo8hfam6fllmc"},
    TestItem{-128ll, "-4k", "4eo8hfam6flli5"},
    TestItem{-879ll, "-15f", "4eo8hfam6flkha"},
    TestItem{-12345ll, "-gp6", "4eo8hfam6fl4oj"},
    TestItem{-32768ll, "-1hph", "4eo8hfam6fk3o8"},
    TestItem{-5165781ll, "-9jc36", "4eo8hfam6629jj"},
    TestItem{-97897347ll, "-6m5ioi", "4eo8hfalqkg2p7"},
    TestItem{-2147483648ll, "-5ehnckb", "4eo8hfagiop92e"},
    TestItem{-165789751156ll, "-fmp4hqcp", "4eo8helq8b3ma0"},
    TestItem{-8798798743521135ll, "-1fjmppdnlj46", "4emjojbnjj02ij"},
    TestItem{-9223372036854775807ll - 1, "-27c48l5b37oaoq", "27c48l5b37oaoq"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<28>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "14"},
    TestItem{36ll, "18"},
    TestItem{127ll, "4f"},
    TestItem{200ll, "74"},
    TestItem{255ll, "93"},
    TestItem{8158ll, "aba"},
    TestItem{50243ll, "282b"},
    TestItem{32767ll, "1dm7"},
    TestItem{87231ll, "3r7b"},
    TestItem{65535ll, "2rgf"},
    TestItem{17875098ll, "1127oa"},
    TestItem{1787987597ll, "3jopobp"},
    TestItem{2147483647ll, "4clm98f"},
    TestItem{4294967295ll, "8pfgih3"},
    TestItem{657687465411ll, "1kkmh0h6b"},
    TestItem{89098098408713ll, "8bn8p9o4g9"},
    TestItem{4987654351689798ll, "gndle47p0dq"},
    TestItem{9223372036854775807ll, "1bk39f3ah3dmq7"},
    TestItem{-1ll, "-1", "2nc6j26l66rhof"},
    TestItem{-13ll, "-d", "2nc6j26l66rho3"},
    TestItem{-128ll, "-4g", "2nc6j26l66rhk0"},
    TestItem{-879ll, "-13b", "2nc6j26l66rgl5"},
    TestItem{-12345ll, "-fkp", "2nc6j26l66r23j"},
    TestItem{-32768ll, "-1dm8", "2nc6j26l66q428"},
    TestItem{-5165781ll, "-8b905", "2nc6j26l5qg8ob"},
    TestItem{-97897347ll, "-5j7h1n", "2nc6j26l0fk0ml"},
    TestItem{-2147483648ll, "-4clm98g", "2nc6j26gld58g0"},
    TestItem{-165789751156ll, "-c813jj98", "2nc6j1md537qf8"},
    TestItem{-8798798743521135ll, "-11jlcchmfpnr", "2nb4r8m8gcbk0h"},
    TestItem{-9223372036854775807ll - 1, "-1bk39f3ah3dmq8", "1bk39f3ah3dmq8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<29>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "13"},
    TestItem{36ll, "17"},
    TestItem{127ll, "4b"},
    TestItem{200ll, "6q"},
    TestItem{255ll, "8n"},
    TestItem{8158ll, "9k9"},
    TestItem{50243ll, "21lf"},
    TestItem{32767ll, "19rq"},
    TestItem{87231ll, "3gks"},
    TestItem{65535ll, "2jqo"},
    TestItem{17875098ll, "p7qgk"},
    TestItem{1787987597ll, "304s6jl"},
    TestItem{2147483647ll, "3hk7987"},
    TestItem{4294967295ll, "76beigf"},
    TestItem{657687465411ll, "193jpcaqi"},
    TestItem{89098098408713ll, "643457jb33"},
    TestItem{4987654351689798ll, "bonbd6e6crj"},
    TestItem{9223372036854775807ll, "q1se8f0m04isb"},
    TestItem{-1ll, "-1", "1n3rsh11f098rn"},
    TestItem{-13ll, "-d", "1n3rsh11f098rb"},
    TestItem{-128ll, "-4c", "1n3rsh11f098nc"},
    TestItem{-879ll, "-119", "1n3rsh11f097qf"},
    TestItem{-12345ll, "-ejk", "1n3rsh11f08n84"},
    TestItem{-32768ll, "-19rr", "1n3rsh11f07rsq"},
    TestItem{-5165781ll, "-78ncb", "1n3rsh11em0efd"},
    TestItem{-97897347ll, "-4mbsph", "1n3rsh11a6q927"},
    TestItem{-2147483648ll, "-3hk7988", "1n3rsh0qq91sjg"},
    TestItem{-165789751156ll, "-9hkqadrq", "1n3rsgkcn2rnsr"},
    TestItem{-8798798743521135ll, "-kqeqrdc8q2l", "1n3722331h0bp3"},
    TestItem{-9223372036854775807ll - 1, "-q1se8f0m04isc", "q1se8f0m04isc"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<30>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "12"},
    TestItem{36ll, "16"},
    TestItem{127ll, "47"},
    TestItem{200ll, "6k"},
    TestItem{255ll, "8f"},
    TestItem{8158ll, "91s"},
    TestItem{50243ll, "1pon"},
    TestItem{32767ll, "16c7"},
    TestItem{87231ll, "36rl"},
    TestItem{65535ll, "2cof"},
    TestItem{17875098ll, "m216i"},
    TestItem{1787987597ll, "2dhbmqh"},
    TestItem{2147483647ll, "2sb6cs7"},
    TestItem{4294967295ll, "5qmcpqf"},
    TestItem{657687465411ll, "10259p0dl"},
    TestItem{89098098408713ll, "4fnticaknn"},
    TestItem{4987654351689798ll, "8dbt5ff1e6i"},
    TestItem{9223372036854775807ll, "hajppbc1fc207"},
    TestItem{-1ll, "-1", "14l9lkmo30o40f"},
    TestItem{-13ll, "-d", "14l9lkmo30o403"},
    TestItem{-128ll, "-48", "14l9lkmo30o3q8"},
    TestItem{-879ll, "-t9", "14l9lkmo30o317"},
    TestItem{-12345ll, "-dlf", "14l9lkmo30nk91"},
    TestItem{-32768ll, "-16c8", "14l9lkmo30mri8"},
    TestItem{-5165781ll, "-6b9ml", "14l9lkmo2oco7p"},
    TestItem{-97897347ll, "-40poor", "14l9lkmnsts95j"},
    TestItem{-2147483648ll, "-2sb6cs8", "14l9lkml4jhl28"},
    TestItem{-165789751156ll, "-7hcil4ig", "14l9lkf6kc2tc0"},
    TestItem{-8798798743521135ll, "-er0mmn856of", "14kook019mir61"},
    TestItem{-9223372036854775807ll - 1, "-hajppbc1fc208", "hajppbc1fc208"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<31>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "11"},
    TestItem{36ll, "15"},
    TestItem{127ll, "43"},
    TestItem{200ll, "6e"},
    TestItem{255ll, "87"},
    TestItem{8158ll, "8f5"},
    TestItem{50243ll, "1l8n"},
    TestItem{32767ll, "1330"},
    TestItem{87231ll, "2sns"},
    TestItem{65535ll, "2661"},
    TestItem{17875098ll, "jb0g2"},
    TestItem{1787987597ll, "20e1m08"},
    TestItem{2147483647ll, "2d09uc1"},
    TestItem{4294967295ll, "4q0jto3"},
    TestItem{657687465411ll, "ns1k4jpl"},
    TestItem{89098098408713ll, "3bedp0lfkl"},
    TestItem{4987654351689798ll, "62jt2sednp7"},
    TestItem{9223372036854775807ll, "bm03i95hia437"},
    TestItem{-1ll, "-1", "nd075ib45k86f"},
    TestItem{-13ll, "-d", "nd075ib45k863"},
    TestItem{-128ll, "-44", "nd075ib45k82c"},
    TestItem{-879ll, "-sb", "nd075ib45k795"},
    TestItem{-12345ll, "-cq7", "nd075ib45jqb9"},
    TestItem{-32768ll, "-1331", "nd075ib45j53f"},
    TestItem{-5165781ll, "-5icd3", "nd075ib401qod"},
    TestItem{-97897347ll, "-3d048t", "nd075ib0nk3si"},
    TestItem{-2147483648ll, "-2d09uc2", "nd075i8m5a8pe"},
    TestItem{-165789751156ll, "-60ot667s", "nd075caa7e1tj"},
    TestItem{-8798798743521135ll, "-amodj9ieogq", "nckfc4mpi5ekl"},
    TestItem{-9223372036854775807ll - 1, "-bm03i95hia438", "bm03i95hia438"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<32>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "10"},
    TestItem{36ll, "14"},
    TestItem{127ll, "3v"},
    TestItem{200ll, "68"},
    TestItem{255ll, "7v"},
    TestItem{8158ll, "7uu"},
    TestItem{50243ll, "1h23"},
    TestItem{32767ll, "vvv"},
    TestItem{87231ll, "2l5v"},
    TestItem{65535ll, "1vvv"},
    TestItem{17875098ll, "h1g4q"},
    TestItem{1787987597ll, "1l951kd"},
    TestItem{2147483647ll, "1vvvvvv"},
    TestItem{4294967295ll, "3vvvvvv"},
    TestItem{657687465411ll, "j4gjkje3"},
    TestItem{89098098408713ll, "2h13282v89"},
    TestItem{4987654351689798ll, "4do7r2ua626"},
    TestItem{9223372036854775807ll, "7vvvvvvvvvvvv"},
    TestItem{-1ll, "-1", "fvvvvvvvvvvvv"},
    TestItem{-13ll, "-d", "fvvvvvvvvvvvj"},
    TestItem{-128ll, "-40", "fvvvvvvvvvvs0"},
    TestItem{-879ll, "-rf", "fvvvvvvvvvv4h"},
    TestItem{-12345ll, "-c1p", "fvvvvvvvvvju7"},
    TestItem{-32768ll, "-1000", "fvvvvvvvvv000"},
    TestItem{-5165781ll, "-4tkml", "fvvvvvvvr2b9b"},
    TestItem{-97897347ll, "-2tbis3", "fvvvvvvt2kd3t"},
    TestItem{-2147483648ll, "-2000000", "fvvvvvu000000"},
    TestItem{-165789751156ll, "-4qctdlrk", "fvvvvr5j2ia4c"},
    TestItem{-8798798743521135ll, "-7q2ensothrf", "fvo5th8372e4h"},
    TestItem{-9223372036854775807ll - 1, "-8000000000000", "8000000000000"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<33>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "w"},
    TestItem{36ll, "13"},
    TestItem{127ll, "3s"},
    TestItem{200ll, "62"},
    TestItem{255ll, "7o"},
    TestItem{8158ll, "7g7"},
    TestItem{50243ll, "1d4h"},
    TestItem{32767ll, "u2v"},
    TestItem{87231ll, "2e3c"},
    TestItem{65535ll, "1r5u"},
    TestItem{17875098ll, "f2d7l"},
    TestItem{1787987597ll, "1cmmctb"},
    TestItem{2147483647ll, "1lsqtl1"},
    TestItem{4294967295ll, "3aokq93"},
    TestItem{657687465411ll, "fe8eg7g9"},
    TestItem{89098098408713ll, "1ubjpukd0k"},
    TestItem{4987654351689798ll, "38fcdu5s9du"},
    TestItem{9223372036854775807ll, "5hg4ck9jd4u37"},
    TestItem{-1ll, "-1", "b1w8p7j5q9r6f"},
    TestItem{-13ll, "-d", "b1w8p7j5q9r63"},
    TestItem{-128ll, "-3t", "b1w8p7j5q9r2k"},
    TestItem{-879ll, "-ql", "b1w8p7j5q9qcs"},
    TestItem{-12345ll, "-bb3", "b1w8p7j5q9fsd"},
    TestItem{-32768ll, "-u2w", "b1w8p7j5q8u3h"},
    TestItem{-5165781ll, "-4bojr", "b1w8p7j5lv2jm"},
    TestItem{-97897347ll, "-2gi4i9", "b1w8p7j39oml7"},
    TestItem{-2147483648ll, "-1lsqtl2", "b1w8p7hgufuie"},
    TestItem{-165789751156ll, "-3tcaa767", "b1w8p3mqfwk09"},
    TestItem{-8798798743521135ll, "-5oj768v1cwu", "b1qh60cts8e6j"},
    TestItem{-9223372036854775807ll - 1, "-5hg4ck9jd4u38", "5hg4ck9jd4u38"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<34>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "w"},
    TestItem{36ll, "12"},
    TestItem{127ll, "3p"},
    TestItem{200ll, "5u"},
    TestItem{255ll, "7h"},
    TestItem{8158ll, "71w"},
    TestItem{50243ll, "19fp"},
    TestItem{32767ll, "sbp"},
    TestItem{87231ll, "27fl"},
    TestItem{65535ll, "1mnh"},
    TestItem{17875098ll, "dcqu6"},
    TestItem{1787987597ll, "15bx82h"},
    TestItem{2147483647ll, "1d8xqrp"},
    TestItem{4294967295ll, "2qhxjlh"},
    TestItem{657687465411ll, "chp78tkn"},
    TestItem{89098098408713ll, "1fubx38kal"},
    TestItem{4987654351689798ll, "2e4wnws450i"},
    TestItem{9223372036854775807ll, "3tdtk1v8j6tpp"},
    TestItem{-1ll, "-1", "7orp63sh4dphh"},
    TestItem{-13ll, "-d", "7orp63sh4dph5"},
    TestItem{-128ll, "-3q", "7orp63sh4dpdq"},
    TestItem{-879ll, "-pt", "7orp63sh4dopn"},
    TestItem{-12345ll, "-an3", "7orp63sh4desf"},
    TestItem{-32768ll, "-sbq", "7orp63sh4cv5q"},
    TestItem{-5165781ll, "-3temp", "7orp63sh0iasr"},
    TestItem{-97897347ll, "-258q9p", "7orp63sex4x7r"},
    TestItem{-2147483648ll, "-1d8xqrq", "7orp63r3tdwnq"},
    TestItem{-165789751156ll, "-35auvdpm", "7orp60n67gbpu"},
    TestItem{-8798798743521135ll, "-48v3mdkohmv", "7ong9063hn7sl"},
    TestItem{-9223372036854775807ll - 1, "-3tdtk1v8j6tpq", "3tdtk1v8j6tpq"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<35>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "w"},
    TestItem{36ll, "11"},
    TestItem{127ll, "3m"},
    TestItem{200ll, "5p"},
    TestItem{255ll, "7a"},
    TestItem{8158ll, "6n3"},
    TestItem{50243ll, "160i"},
    TestItem{32767ll, "qq7"},
    TestItem{87231ll, "217b"},
    TestItem{65535ll, "1ihf"},
    TestItem{17875098ll, "bvvw3"},
    TestItem{1787987597ll, "y1hbow"},
    TestItem{2147483647ll, "15v22um"},
    TestItem{4294967295ll, "2br45qa"},
    TestItem{657687465411ll, "a7r5obub"},
    TestItem{89098098408713ll, "14jsjs3s4n"},
    TestItem{4987654351689798ll, "1s9v46xlmyx"},
    TestItem{9223372036854775807ll, "2pijmikexrxp7"},
    TestItem{-1ll, "-1", "5g24a25twkwff"},
    TestItem{-13ll, "-d", "5g24a25twkwf3"},
    TestItem{-128ll, "-3n", "5g24a25twkwbs"},
    TestItem{-879ll, "-p4", "5g24a25twkvpc"},
    TestItem{-12345ll, "-a2p", "5g24a25twkmcq"},
    TestItem{-32768ll, "-qq8", "5g24a25twk5o8"},
    TestItem{-5165781ll, "-3fgxq", "5g24a25tt5fgp"},
    TestItem{-97897347ll, "-1u8b72", "5g24a25s2cl8e"},
    TestItem{-2147483648ll, "-15v22un", "5g24a24o1itjs"},
    TestItem{-165789751156ll, "-2k6kgcd1", "5g249yknc4k2f"},
    TestItem{-8798798743521135ll, "-36mb7iv78yk", "5fxwmpxb1dnfv"},
    TestItem{-9223372036854775807ll - 1, "-2pijmikexrxp8", "2pijmikexrxp8"},
  }};
}

template <>
__host__ __device__ constexpr cuda::std::array<TestItem, 40> get_test_items<36>()
{
  return {{
    TestItem{0ll, "0"},
    TestItem{1ll, "1"},
    TestItem{2ll, "2"},
    TestItem{3ll, "3"},
    TestItem{4ll, "4"},
    TestItem{7ll, "7"},
    TestItem{8ll, "8"},
    TestItem{14ll, "e"},
    TestItem{16ll, "g"},
    TestItem{23ll, "n"},
    TestItem{32ll, "w"},
    TestItem{36ll, "10"},
    TestItem{127ll, "3j"},
    TestItem{200ll, "5k"},
    TestItem{255ll, "73"},
    TestItem{8158ll, "6am"},
    TestItem{50243ll, "12rn"},
    TestItem{32767ll, "pa7"},
    TestItem{87231ll, "1vb3"},
    TestItem{65535ll, "1ekf"},
    TestItem{17875098ll, "an4ii"},
    TestItem{1787987597ll, "tkis25"},
    TestItem{2147483647ll, "zik0zj"},
    TestItem{4294967295ll, "1z141z3"},
    TestItem{657687465411ll, "8e4y4w5f"},
    TestItem{89098098408713ll, "vkz3ncirt"},
    TestItem{4987654351689798ll, "1d3z4rzkrpi"},
    TestItem{9223372036854775807ll, "1y2p0ij32e8e7"},
    TestItem{-1ll, "-1", "3w5e11264sgsf"},
    TestItem{-13ll, "-d", "3w5e11264sgs3"},
    TestItem{-128ll, "-3k", "3w5e11264sgow"},
    TestItem{-879ll, "-of", "3w5e11264sg41"},
    TestItem{-12345ll, "-9ix", "3w5e11264s79j"},
    TestItem{-32768ll, "-pa8", "3w5e11264rri8"},
    TestItem{-5165781ll, "-32pxx", "3w5e11261pquj"},
    TestItem{-97897347ll, "-1maa2r", "3w5e1124ii6pp"},
    TestItem{-2147483648ll, "-zik0zk", "3w5e1116m8fsw"},
    TestItem{-165789751156ll, "-245uxqpg", "3w5e0yy09uq30"},
    TestItem{-8798798743521135ll, "-2emwwc9n2wf", "3w2ze45tv5dw1"},
    TestItem{-9223372036854775807ll - 1, "-1y2p0ij32e8e8", "1y2p0ij32e8e8"},
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

  // std::in_range accepts integer types only
  using RangeT =
    cuda::std::conditional_t<cuda::std::is_same_v<T, char>,
                             cuda::std::conditional_t<cuda::std::is_signed_v<T>, signed char, unsigned char>,
                             T>;

  if (cuda::std::in_range<RangeT>(item.val))
  {
    const auto ref_len = cuda::std::strlen(item.str_signed);
    char buff[buff_size + 1]{};
    char* buff_start = buff + 1;

    const auto value    = static_cast<T>(item.val);
    const char* ref_str = (cuda::std::is_signed_v<T>) ? item.str_signed : item.str_unsigned;

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
}

template <int Base>
__host__ __device__ constexpr void test_base()
{
  constexpr auto items = get_test_items<Base>();

  for (const auto& item : items)
  {
    test_to_chars<char, Base>(item);

    test_to_chars<signed char, Base>(item);
    test_to_chars<signed short, Base>(item);
    test_to_chars<signed int, Base>(item);
    test_to_chars<signed long, Base>(item);
    test_to_chars<signed long long, Base>(item);
#if _CCCL_HAS_INT128()
    test_to_chars<__int128_t, Base>(item);
#endif // _CCCL_HAS_INT128()

    test_to_chars<unsigned char, Base>(item);
    test_to_chars<unsigned short, Base>(item);
    test_to_chars<unsigned int, Base>(item);
    test_to_chars<unsigned long, Base>(item);
    test_to_chars<unsigned long long, Base>(item);
#if _CCCL_HAS_INT128()
    test_to_chars<__uint128_t, Base>(item);
#endif // _CCCL_HAS_INT128()
  }
}

template <int Base>
__host__ __device__ constexpr void test_constexpr_base()
{
  // Test only value 127 (index 12) which is representable by all types
  constexpr auto item = get_test_items<Base>()[12];

  test_to_chars<char, Base>(item);

  test_to_chars<signed char, Base>(item);
  test_to_chars<signed short, Base>(item);
  test_to_chars<signed int, Base>(item);
  test_to_chars<signed long, Base>(item);
  test_to_chars<signed long long, Base>(item);
#if _CCCL_HAS_INT128()
  test_to_chars<__int128_t, Base>(item);
#endif // _CCCL_HAS_INT128()

  test_to_chars<unsigned char, Base>(item);
  test_to_chars<unsigned short, Base>(item);
  test_to_chars<unsigned int, Base>(item);
  test_to_chars<unsigned long, Base>(item);
  test_to_chars<unsigned long long, Base>(item);
#if _CCCL_HAS_INT128()
  test_to_chars<__uint128_t, Base>(item);
#endif // _CCCL_HAS_INT128()
}

template <int... Base>
__host__ __device__ constexpr void test_helper(cuda::std::integer_sequence<int, Base...>)
{
  (test_base<Base + first_base>(), ...);
}

template <int... Base>
__host__ __device__ constexpr void test_constexpr_helper(cuda::std::integer_sequence<int, Base...>)
{
  (test_constexpr_base<Base + first_base>(), ...);
}

__host__ __device__ constexpr bool test()
{
  test_helper(cuda::std::make_integer_sequence<int, last_base - first_base + 1>{});

  return true;
}

__host__ __device__ constexpr bool test_constexpr()
{
  test_constexpr_helper(cuda::std::make_integer_sequence<int, last_base - first_base + 1>{});

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test_constexpr());
  return 0;
}
