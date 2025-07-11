//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// size_type find_last_not_of(const basic_string& str, size_type pos = npos) const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_find_last_not_of(const SV& sv, const SV& str, typename SV::size_type x)
{
  assert(sv.find_last_not_of(str) == x);
  if (x != SV::npos)
  {
    assert(x < sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void
test_find_last_not_of(const SV& sv, const SV& str, typename SV::size_type pos, typename SV::size_type x)
{
  assert(sv.find_last_not_of(str, pos) == x);
  if (x != SV::npos)
  {
    assert(x <= pos && x < sv.size());
  }
}

#define TEST_FIND_LAST_NOT_OF(SV_T, SV1_STR, SV2_STR, ...)                     \
  test_find_last_not_of(SV_T(TEST_STRLIT(typename SV_T::value_type, SV1_STR)), \
                        SV_T(TEST_STRLIT(typename SV_T::value_type, SV2_STR)), \
                        __VA_ARGS__)

template <class SV>
__host__ __device__ constexpr void test_find_last_not_of()
{
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find_last_not_of(SV{}))>);
  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find_last_not_of(SV{}, SizeT{}))>);

  static_assert(noexcept(SV{}.find_last_not_of(SV{})));
  static_assert(noexcept(SV{}.find_last_not_of(SV{}, SizeT{})));

  TEST_FIND_LAST_NOT_OF(SV, "", "", SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "laenf", SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "pqlnkmbdjo", SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "qkamfogpnljdcshbreti", SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "", 0, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "laenf", 0, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "pqlnkmbdjo", 0, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "qkamfogpnljdcshbreti", 0, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "", 1, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "bjaht", 1, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "hjlcmgpket", 1, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "", "htaobedqikfplcgjsmrn", 1, SV::npos);

  TEST_FIND_LAST_NOT_OF(SV, "nhmko", "", 4);
  TEST_FIND_LAST_NOT_OF(SV, "lahfb", "irkhs", 4);
  TEST_FIND_LAST_NOT_OF(SV, "gmfhd", "kantesmpgj", 4);
  TEST_FIND_LAST_NOT_OF(SV, "odaft", "oknlrstdpiqmjbaghcfe", SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "fodgq", "", 0, 0);
  TEST_FIND_LAST_NOT_OF(SV, "qanej", "dfkap", 0, 0);
  TEST_FIND_LAST_NOT_OF(SV, "clbao", "ihqrfebgad", 0, 0);
  TEST_FIND_LAST_NOT_OF(SV, "mekdn", "ngtjfcalbseiqrphmkdo", 0, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "srdfq", "", 1, 1);
  TEST_FIND_LAST_NOT_OF(SV, "oemth", "ikcrq", 1, 1);
  TEST_FIND_LAST_NOT_OF(SV, "cdaih", "dmajblfhsg", 1, 0);
  TEST_FIND_LAST_NOT_OF(SV, "qohtk", "oqftjhdmkgsblacenirp", 1, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "cshmd", "", 2, 2);
  TEST_FIND_LAST_NOT_OF(SV, "lhcdo", "oebqi", 2, 2);
  TEST_FIND_LAST_NOT_OF(SV, "qnsoh", "kojhpmbsfe", 2, 1);
  TEST_FIND_LAST_NOT_OF(SV, "pkrof", "acbsjqogpltdkhinfrem", 2, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "fmtsp", "", 4, 4);
  TEST_FIND_LAST_NOT_OF(SV, "khbpm", "aobjd", 4, 4);
  TEST_FIND_LAST_NOT_OF(SV, "pbsji", "pcbahntsje", 4, 4);
  TEST_FIND_LAST_NOT_OF(SV, "mprdj", "fhepcrntkoagbmldqijs", 4, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "eqmpa", "", 5, 4);
  TEST_FIND_LAST_NOT_OF(SV, "omigs", "kocgb", 5, 4);
  TEST_FIND_LAST_NOT_OF(SV, "onmje", "fbslrjiqkm", 5, 4);
  TEST_FIND_LAST_NOT_OF(SV, "oqmrj", "jeidpcmalhfnqbgtrsko", 5, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "schfa", "", 6, 4);
  TEST_FIND_LAST_NOT_OF(SV, "igdsc", "qngpd", 6, 4);
  TEST_FIND_LAST_NOT_OF(SV, "brqgo", "rodhqklgmb", 6, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "tnrph", "thdjgafrlbkoiqcspmne", 6, SV::npos);

  TEST_FIND_LAST_NOT_OF(SV, "eolhfgpjqk", "", 9);
  TEST_FIND_LAST_NOT_OF(SV, "nbatdlmekr", "bnrpe", 8);
  TEST_FIND_LAST_NOT_OF(SV, "jdmciepkaq", "jtdaefblso", 9);
  TEST_FIND_LAST_NOT_OF(SV, "hkbgspoflt", "oselktgbcapndfjihrmq", SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "hcjitbfapl", "", 0, 0);
  TEST_FIND_LAST_NOT_OF(SV, "daiprenocl", "ashjd", 0, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "litpcfdghe", "mgojkldsqh", 0, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "aidjksrolc", "imqnaghkfrdtlopbjesc", 0, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "qpghtfbaji", "", 1, 1);
  TEST_FIND_LAST_NOT_OF(SV, "gfshlcmdjr", "nadkh", 1, 1);
  TEST_FIND_LAST_NOT_OF(SV, "nkodajteqp", "ofdrqmkebl", 1, 0);
  TEST_FIND_LAST_NOT_OF(SV, "gbmetiprqd", "bdfjqgatlksriohemnpc", 1, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "crnklpmegd", "", 5, 5);
  TEST_FIND_LAST_NOT_OF(SV, "jsbtafedoc", "prqgn", 5, 5);
  TEST_FIND_LAST_NOT_OF(SV, "qnmodrtkeb", "pejafmnokr", 5, 4);
  TEST_FIND_LAST_NOT_OF(SV, "cpebqsfmnj", "odnqkgijrhabfmcestlp", 5, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "lmofqdhpki", "", 9, 9);
  TEST_FIND_LAST_NOT_OF(SV, "hnefkqimca", "rtjpa", 9, 8);
  TEST_FIND_LAST_NOT_OF(SV, "drtasbgmfp", "ktsrmnqagd", 9, 9);
  TEST_FIND_LAST_NOT_OF(SV, "lsaijeqhtr", "rtdhgcisbnmoaqkfpjle", 9, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "elgofjmbrq", "", 10, 9);
  TEST_FIND_LAST_NOT_OF(SV, "mjqdgalkpc", "dplqa", 10, 9);
  TEST_FIND_LAST_NOT_OF(SV, "kthqnfcerm", "dkacjoptns", 10, 9);
  TEST_FIND_LAST_NOT_OF(SV, "dfsjhanorc", "hqfimtrgnbekpdcsjalo", 10, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "eqsgalomhb", "", 11, 9);
  TEST_FIND_LAST_NOT_OF(SV, "akiteljmoh", "lofbc", 11, 9);
  TEST_FIND_LAST_NOT_OF(SV, "hlbdfreqjo", "astoegbfpn", 11, 8);
  TEST_FIND_LAST_NOT_OF(SV, "taqobhlerg", "pdgreqomsncafklhtibj", 11, SV::npos);

  TEST_FIND_LAST_NOT_OF(SV, "gprdcokbnjhlsfmtieqa", "", 19);
  TEST_FIND_LAST_NOT_OF(SV, "qjghlnftcaismkropdeb", "bjaht", 18);
  TEST_FIND_LAST_NOT_OF(SV, "pnalfrdtkqcmojiesbhg", "hjlcmgpket", 17);
  TEST_FIND_LAST_NOT_OF(SV, "pniotcfrhqsmgdkjbael", "htaobedqikfplcgjsmrn", SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "snafbdlghrjkpqtoceim", "", 0, 0);
  TEST_FIND_LAST_NOT_OF(SV, "aemtbrgcklhndjisfpoq", "lbtqd", 0, 0);
  TEST_FIND_LAST_NOT_OF(SV, "pnracgfkjdiholtbqsem", "tboimldpjh", 0, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "dicfltehbsgrmojnpkaq", "slcerthdaiqjfnobgkpm", 0, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "jlnkraeodhcspfgbqitm", "", 1, 1);
  TEST_FIND_LAST_NOT_OF(SV, "lhosrngtmfjikbqpcade", "aqibs", 1, 1);
  TEST_FIND_LAST_NOT_OF(SV, "rbtaqjhgkneisldpmfoc", "gtfblmqinc", 1, 0);
  TEST_FIND_LAST_NOT_OF(SV, "gpifsqlrdkbonjtmheca", "mkqpbtdalgniorhfescj", 1, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "hdpkobnsalmcfijregtq", "", 10, 10);
  TEST_FIND_LAST_NOT_OF(SV, "jtlshdgqaiprkbcoenfm", "pblas", 10, 9);
  TEST_FIND_LAST_NOT_OF(SV, "fkdrbqltsgmcoiphneaj", "arosdhcfme", 10, 9);
  TEST_FIND_LAST_NOT_OF(SV, "crsplifgtqedjohnabmk", "blkhjeogicatqfnpdmsr", 10, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "niptglfbosehkamrdqcj", "", 19, 19);
  TEST_FIND_LAST_NOT_OF(SV, "copqdhstbingamjfkler", "djkqc", 19, 19);
  TEST_FIND_LAST_NOT_OF(SV, "mrtaefilpdsgocnhqbjk", "lgokshjtpb", 19, 16);
  TEST_FIND_LAST_NOT_OF(SV, "kojatdhlcmigpbfrqnes", "bqjhtkfepimcnsgrlado", 19, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "eaintpchlqsbdgrkjofm", "", 20, 19);
  TEST_FIND_LAST_NOT_OF(SV, "gjnhidfsepkrtaqbmclo", "nocfa", 20, 18);
  TEST_FIND_LAST_NOT_OF(SV, "spocfaktqdbiejlhngmr", "bgtajmiedc", 20, 19);
  TEST_FIND_LAST_NOT_OF(SV, "rphmlekgfscndtaobiqj", "lsckfnqgdahejiopbtmr", 20, SV::npos);
  TEST_FIND_LAST_NOT_OF(SV, "liatsqdoegkmfcnbhrpj", "", 21, 19);
  TEST_FIND_LAST_NOT_OF(SV, "binjagtfldkrspcomqeh", "gfsrt", 21, 19);
  TEST_FIND_LAST_NOT_OF(SV, "latkmisecnorjbfhqpdg", "pfsocbhjtm", 21, 19);
  TEST_FIND_LAST_NOT_OF(SV, "lecfratdjkhnsmqpoigb", "tpflmdnoicjgkberhqsa", 21, SV::npos);
}

__host__ __device__ constexpr bool test()
{
  test_find_last_not_of<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_find_last_not_of<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_find_last_not_of<cuda::std::u16string_view>();
  test_find_last_not_of<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_find_last_not_of<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
