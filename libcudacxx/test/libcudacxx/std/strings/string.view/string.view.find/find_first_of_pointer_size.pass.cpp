//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr size_type find_first_of(const charT* s, size_type pos = 0) const;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void
test_find_first_of(const SV& sv, const typename SV::value_type* str, typename SV::size_type x)
{
  assert(sv.find_first_of(str) == x);
  if (x != SV::npos)
  {
    assert(x < sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void test_find_first_of(
  const SV& sv, const typename SV::value_type* str, typename SV::size_type pos, typename SV::size_type x)
{
  assert(sv.find_first_of(str, pos) == x);
  if (x != SV::npos)
  {
    assert(pos <= x && x < sv.size());
  }
}

#define TEST_FIND_FIRST_OF(SV_T, SV_STR, STR, ...) \
  test_find_first_of(                              \
    SV_T(TEST_STRLIT(typename SV_T::value_type, SV_STR)), TEST_STRLIT(typename SV_T::value_type, STR), __VA_ARGS__)

template <class SV>
__host__ __device__ constexpr void test_find_first_of()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find_first_of(cuda::std::declval<const CharT*>()))>);
  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find_first_of(cuda::std::declval<const CharT*>(), SizeT{}))>);

  static_assert(noexcept(SV{}.find_first_of(cuda::std::declval<const CharT*>())));
  static_assert(noexcept(SV{}.find_first_of(cuda::std::declval<const CharT*>(), SizeT{})));

  TEST_FIND_FIRST_OF(SV, "", "", SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "laenf", SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "pqlnkmbdjo", SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "qkamfogpnljdcshbreti", SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "", 0, SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "laenf", 0, SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "pqlnkmbdjo", 0, SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "qkamfogpnljdcshbreti", 0, SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "", 1, SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "bjaht", 1, SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "hjlcmgpket", 1, SV::npos);
  TEST_FIND_FIRST_OF(SV, "", "htaobedqikfplcgjsmrn", 1, SV::npos);

  TEST_FIND_FIRST_OF(SV, "nhmko", "", SV::npos);
  TEST_FIND_FIRST_OF(SV, "lahfb", "irkhs", 2);
  TEST_FIND_FIRST_OF(SV, "gmfhd", "kantesmpgj", 0);
  TEST_FIND_FIRST_OF(SV, "odaft", "oknlrstdpiqmjbaghcfe", 0);
  TEST_FIND_FIRST_OF(SV, "fodgq", "", 0, SV::npos);
  TEST_FIND_FIRST_OF(SV, "qanej", "dfkap", 0, 1);
  TEST_FIND_FIRST_OF(SV, "clbao", "ihqrfebgad", 0, 2);
  TEST_FIND_FIRST_OF(SV, "mekdn", "ngtjfcalbseiqrphmkdo", 0, 0);
  TEST_FIND_FIRST_OF(SV, "srdfq", "", 1, SV::npos);
  TEST_FIND_FIRST_OF(SV, "oemth", "ikcrq", 1, SV::npos);
  TEST_FIND_FIRST_OF(SV, "cdaih", "dmajblfhsg", 1, 1);
  TEST_FIND_FIRST_OF(SV, "qohtk", "oqftjhdmkgsblacenirp", 1, 1);
  TEST_FIND_FIRST_OF(SV, "cshmd", "", 2, SV::npos);
  TEST_FIND_FIRST_OF(SV, "lhcdo", "oebqi", 2, 4);
  TEST_FIND_FIRST_OF(SV, "qnsoh", "kojhpmbsfe", 2, 2);
  TEST_FIND_FIRST_OF(SV, "pkrof", "acbsjqogpltdkhinfrem", 2, 2);
  TEST_FIND_FIRST_OF(SV, "fmtsp", "", 4, SV::npos);
  TEST_FIND_FIRST_OF(SV, "khbpm", "aobjd", 4, SV::npos);
  TEST_FIND_FIRST_OF(SV, "pbsji", "pcbahntsje", 4, SV::npos);
  TEST_FIND_FIRST_OF(SV, "mprdj", "fhepcrntkoagbmldqijs", 4, 4);
  TEST_FIND_FIRST_OF(SV, "eqmpa", "", 5, SV::npos);
  TEST_FIND_FIRST_OF(SV, "omigs", "kocgb", 5, SV::npos);
  TEST_FIND_FIRST_OF(SV, "onmje", "fbslrjiqkm", 5, SV::npos);
  TEST_FIND_FIRST_OF(SV, "oqmrj", "jeidpcmalhfnqbgtrsko", 5, SV::npos);
  TEST_FIND_FIRST_OF(SV, "schfa", "", 6, SV::npos);
  TEST_FIND_FIRST_OF(SV, "igdsc", "qngpd", 6, SV::npos);
  TEST_FIND_FIRST_OF(SV, "brqgo", "rodhqklgmb", 6, SV::npos);
  TEST_FIND_FIRST_OF(SV, "tnrph", "thdjgafrlbkoiqcspmne", 6, SV::npos);

  TEST_FIND_FIRST_OF(SV, "eolhfgpjqk", "", SV::npos);
  TEST_FIND_FIRST_OF(SV, "nbatdlmekr", "bnrpe", 0);
  TEST_FIND_FIRST_OF(SV, "jdmciepkaq", "jtdaefblso", 0);
  TEST_FIND_FIRST_OF(SV, "hkbgspoflt", "oselktgbcapndfjihrmq", 0);
  TEST_FIND_FIRST_OF(SV, "hcjitbfapl", "", 0, SV::npos);
  TEST_FIND_FIRST_OF(SV, "daiprenocl", "ashjd", 0, 0);
  TEST_FIND_FIRST_OF(SV, "litpcfdghe", "mgojkldsqh", 0, 0);
  TEST_FIND_FIRST_OF(SV, "aidjksrolc", "imqnaghkfrdtlopbjesc", 0, 0);
  TEST_FIND_FIRST_OF(SV, "qpghtfbaji", "", 1, SV::npos);
  TEST_FIND_FIRST_OF(SV, "gfshlcmdjr", "nadkh", 1, 3);
  TEST_FIND_FIRST_OF(SV, "nkodajteqp", "ofdrqmkebl", 1, 1);
  TEST_FIND_FIRST_OF(SV, "gbmetiprqd", "bdfjqgatlksriohemnpc", 1, 1);
  TEST_FIND_FIRST_OF(SV, "crnklpmegd", "", 5, SV::npos);
  TEST_FIND_FIRST_OF(SV, "jsbtafedoc", "prqgn", 5, SV::npos);
  TEST_FIND_FIRST_OF(SV, "qnmodrtkeb", "pejafmnokr", 5, 5);
  TEST_FIND_FIRST_OF(SV, "cpebqsfmnj", "odnqkgijrhabfmcestlp", 5, 5);
  TEST_FIND_FIRST_OF(SV, "lmofqdhpki", "", 9, SV::npos);
  TEST_FIND_FIRST_OF(SV, "hnefkqimca", "rtjpa", 9, 9);
  TEST_FIND_FIRST_OF(SV, "drtasbgmfp", "ktsrmnqagd", 9, SV::npos);
  TEST_FIND_FIRST_OF(SV, "lsaijeqhtr", "rtdhgcisbnmoaqkfpjle", 9, 9);
  TEST_FIND_FIRST_OF(SV, "elgofjmbrq", "", 10, SV::npos);
  TEST_FIND_FIRST_OF(SV, "mjqdgalkpc", "dplqa", 10, SV::npos);
  TEST_FIND_FIRST_OF(SV, "kthqnfcerm", "dkacjoptns", 10, SV::npos);
  TEST_FIND_FIRST_OF(SV, "dfsjhanorc", "hqfimtrgnbekpdcsjalo", 10, SV::npos);
  TEST_FIND_FIRST_OF(SV, "eqsgalomhb", "", 11, SV::npos);
  TEST_FIND_FIRST_OF(SV, "akiteljmoh", "lofbc", 11, SV::npos);
  TEST_FIND_FIRST_OF(SV, "hlbdfreqjo", "astoegbfpn", 11, SV::npos);
  TEST_FIND_FIRST_OF(SV, "taqobhlerg", "pdgreqomsncafklhtibj", 11, SV::npos);

  TEST_FIND_FIRST_OF(SV, "gprdcokbnjhlsfmtieqa", "", SV::npos);
  TEST_FIND_FIRST_OF(SV, "qjghlnftcaismkropdeb", "bjaht", 1);
  TEST_FIND_FIRST_OF(SV, "pnalfrdtkqcmojiesbhg", "hjlcmgpket", 0);
  TEST_FIND_FIRST_OF(SV, "pniotcfrhqsmgdkjbael", "htaobedqikfplcgjsmrn", 0);
  TEST_FIND_FIRST_OF(SV, "snafbdlghrjkpqtoceim", "", 0, SV::npos);
  TEST_FIND_FIRST_OF(SV, "aemtbrgcklhndjisfpoq", "lbtqd", 0, 3);
  TEST_FIND_FIRST_OF(SV, "pnracgfkjdiholtbqsem", "tboimldpjh", 0, 0);
  TEST_FIND_FIRST_OF(SV, "dicfltehbsgrmojnpkaq", "slcerthdaiqjfnobgkpm", 0, 0);
  TEST_FIND_FIRST_OF(SV, "jlnkraeodhcspfgbqitm", "", 1, SV::npos);
  TEST_FIND_FIRST_OF(SV, "lhosrngtmfjikbqpcade", "aqibs", 1, 3);
  TEST_FIND_FIRST_OF(SV, "rbtaqjhgkneisldpmfoc", "gtfblmqinc", 1, 1);
  TEST_FIND_FIRST_OF(SV, "gpifsqlrdkbonjtmheca", "mkqpbtdalgniorhfescj", 1, 1);
  TEST_FIND_FIRST_OF(SV, "hdpkobnsalmcfijregtq", "", 10, SV::npos);
  TEST_FIND_FIRST_OF(SV, "jtlshdgqaiprkbcoenfm", "pblas", 10, 10);
  TEST_FIND_FIRST_OF(SV, "fkdrbqltsgmcoiphneaj", "arosdhcfme", 10, 10);
  TEST_FIND_FIRST_OF(SV, "crsplifgtqedjohnabmk", "blkhjeogicatqfnpdmsr", 10, 10);
  TEST_FIND_FIRST_OF(SV, "niptglfbosehkamrdqcj", "", 19, SV::npos);
  TEST_FIND_FIRST_OF(SV, "copqdhstbingamjfkler", "djkqc", 19, SV::npos);
  TEST_FIND_FIRST_OF(SV, "mrtaefilpdsgocnhqbjk", "lgokshjtpb", 19, 19);
  TEST_FIND_FIRST_OF(SV, "kojatdhlcmigpbfrqnes", "bqjhtkfepimcnsgrlado", 19, 19);
  TEST_FIND_FIRST_OF(SV, "eaintpchlqsbdgrkjofm", "", 20, SV::npos);
  TEST_FIND_FIRST_OF(SV, "gjnhidfsepkrtaqbmclo", "nocfa", 20, SV::npos);
  TEST_FIND_FIRST_OF(SV, "spocfaktqdbiejlhngmr", "bgtajmiedc", 20, SV::npos);
  TEST_FIND_FIRST_OF(SV, "rphmlekgfscndtaobiqj", "lsckfnqgdahejiopbtmr", 20, SV::npos);
  TEST_FIND_FIRST_OF(SV, "liatsqdoegkmfcnbhrpj", "", 21, SV::npos);
  TEST_FIND_FIRST_OF(SV, "binjagtfldkrspcomqeh", "gfsrt", 21, SV::npos);
  TEST_FIND_FIRST_OF(SV, "latkmisecnorjbfhqpdg", "pfsocbhjtm", 21, SV::npos);
  TEST_FIND_FIRST_OF(SV, "lecfratdjkhnsmqpoigb", "tpflmdnoicjgkberhqsa", 21, SV::npos);
}

__host__ __device__ constexpr bool test()
{
  test_find_first_of<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_find_first_of<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_find_first_of<cuda::std::u16string_view>();
  test_find_first_of<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_find_first_of<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
