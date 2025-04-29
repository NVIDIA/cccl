//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr size_type find_last_of(const charT* s, size_type pos = npos) const;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void
test_find_last_of(const SV& sv, const typename SV::value_type* str, typename SV::size_type x)
{
  assert(sv.find_last_of(str) == x);
  if (x != SV::npos)
  {
    assert(x < sv.size());
  }
}

template <class SV>
__host__ __device__ constexpr void test_find_last_of(
  const SV& sv, const typename SV::value_type* str, typename SV::size_type pos, typename SV::size_type x)
{
  assert(sv.find_last_of(str, pos) == x);
  if (x != SV::npos)
  {
    assert(x <= pos && x < sv.size());
  }
}

#define TEST_FIND_LAST_OF(SV_T, SV_STR, STR, ...) \
  test_find_last_of(                              \
    SV_T(TEST_STRLIT(typename SV_T::value_type, SV_STR)), TEST_STRLIT(typename SV_T::value_type, STR), __VA_ARGS__)

template <class SV>
__host__ __device__ constexpr void test_find_last_of()
{
  using CharT = typename SV::value_type;
  using SizeT = typename SV::size_type;

  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find_last_of(cuda::std::declval<const CharT*>()))>);
  static_assert(cuda::std::is_same_v<SizeT, decltype(SV{}.find_last_of(cuda::std::declval<const CharT*>(), SizeT{}))>);

  static_assert(noexcept(SV{}.find_last_of(cuda::std::declval<const CharT*>())));
  static_assert(noexcept(SV{}.find_last_of(cuda::std::declval<const CharT*>(), SizeT{})));

  TEST_FIND_LAST_OF(SV, "", "", SV::npos);
  TEST_FIND_LAST_OF(SV, "", "laenf", SV::npos);
  TEST_FIND_LAST_OF(SV, "", "pqlnkmbdjo", SV::npos);
  TEST_FIND_LAST_OF(SV, "", "qkamfogpnljdcshbreti", SV::npos);
  TEST_FIND_LAST_OF(SV, "", "", 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "laenf", 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "pqlnkmbdjo", 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "qkamfogpnljdcshbreti", 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "", 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "bjaht", 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "hjlcmgpket", 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "htaobedqikfplcgjsmrn", 1, SV::npos);

  TEST_FIND_LAST_OF(SV, "nhmko", "", SV::npos);
  TEST_FIND_LAST_OF(SV, "lahfb", "irkhs", 2);
  TEST_FIND_LAST_OF(SV, "gmfhd", "kantesmpgj", 1);
  TEST_FIND_LAST_OF(SV, "odaft", "oknlrstdpiqmjbaghcfe", 4);
  TEST_FIND_LAST_OF(SV, "fodgq", "", 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "qanej", "dfkap", 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "clbao", "ihqrfebgad", 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "mekdn", "ngtjfcalbseiqrphmkdo", 0, 0);
  TEST_FIND_LAST_OF(SV, "srdfq", "", 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "oemth", "ikcrq", 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "cdaih", "dmajblfhsg", 1, 1);
  TEST_FIND_LAST_OF(SV, "qohtk", "oqftjhdmkgsblacenirp", 1, 1);
  TEST_FIND_LAST_OF(SV, "cshmd", "", 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "lhcdo", "oebqi", 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "qnsoh", "kojhpmbsfe", 2, 2);
  TEST_FIND_LAST_OF(SV, "pkrof", "acbsjqogpltdkhinfrem", 2, 2);
  TEST_FIND_LAST_OF(SV, "fmtsp", "", 4, SV::npos);
  TEST_FIND_LAST_OF(SV, "khbpm", "aobjd", 4, 2);
  TEST_FIND_LAST_OF(SV, "pbsji", "pcbahntsje", 4, 3);
  TEST_FIND_LAST_OF(SV, "mprdj", "fhepcrntkoagbmldqijs", 4, 4);
  TEST_FIND_LAST_OF(SV, "eqmpa", "", 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "omigs", "kocgb", 5, 3);
  TEST_FIND_LAST_OF(SV, "onmje", "fbslrjiqkm", 5, 3);
  TEST_FIND_LAST_OF(SV, "oqmrj", "jeidpcmalhfnqbgtrsko", 5, 4);
  TEST_FIND_LAST_OF(SV, "schfa", "", 6, SV::npos);
  TEST_FIND_LAST_OF(SV, "igdsc", "qngpd", 6, 2);
  TEST_FIND_LAST_OF(SV, "brqgo", "rodhqklgmb", 6, 4);
  TEST_FIND_LAST_OF(SV, "tnrph", "thdjgafrlbkoiqcspmne", 6, 4);

  TEST_FIND_LAST_OF(SV, "eolhfgpjqk", "", SV::npos);
  TEST_FIND_LAST_OF(SV, "nbatdlmekr", "bnrpe", 9);
  TEST_FIND_LAST_OF(SV, "jdmciepkaq", "jtdaefblso", 8);
  TEST_FIND_LAST_OF(SV, "hkbgspoflt", "oselktgbcapndfjihrmq", 9);
  TEST_FIND_LAST_OF(SV, "hcjitbfapl", "", 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "daiprenocl", "ashjd", 0, 0);
  TEST_FIND_LAST_OF(SV, "litpcfdghe", "mgojkldsqh", 0, 0);
  TEST_FIND_LAST_OF(SV, "aidjksrolc", "imqnaghkfrdtlopbjesc", 0, 0);
  TEST_FIND_LAST_OF(SV, "qpghtfbaji", "", 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "gfshlcmdjr", "nadkh", 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "nkodajteqp", "ofdrqmkebl", 1, 1);
  TEST_FIND_LAST_OF(SV, "gbmetiprqd", "bdfjqgatlksriohemnpc", 1, 1);
  TEST_FIND_LAST_OF(SV, "crnklpmegd", "", 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "jsbtafedoc", "prqgn", 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "qnmodrtkeb", "pejafmnokr", 5, 5);
  TEST_FIND_LAST_OF(SV, "cpebqsfmnj", "odnqkgijrhabfmcestlp", 5, 5);
  TEST_FIND_LAST_OF(SV, "lmofqdhpki", "", 9, SV::npos);
  TEST_FIND_LAST_OF(SV, "hnefkqimca", "rtjpa", 9, 9);
  TEST_FIND_LAST_OF(SV, "drtasbgmfp", "ktsrmnqagd", 9, 7);
  TEST_FIND_LAST_OF(SV, "lsaijeqhtr", "rtdhgcisbnmoaqkfpjle", 9, 9);
  TEST_FIND_LAST_OF(SV, "elgofjmbrq", "", 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "mjqdgalkpc", "dplqa", 10, 8);
  TEST_FIND_LAST_OF(SV, "kthqnfcerm", "dkacjoptns", 10, 6);
  TEST_FIND_LAST_OF(SV, "dfsjhanorc", "hqfimtrgnbekpdcsjalo", 10, 9);
  TEST_FIND_LAST_OF(SV, "eqsgalomhb", "", 11, SV::npos);
  TEST_FIND_LAST_OF(SV, "akiteljmoh", "lofbc", 11, 8);
  TEST_FIND_LAST_OF(SV, "hlbdfreqjo", "astoegbfpn", 11, 9);
  TEST_FIND_LAST_OF(SV, "taqobhlerg", "pdgreqomsncafklhtibj", 11, 9);

  TEST_FIND_LAST_OF(SV, "gprdcokbnjhlsfmtieqa", "", SV::npos);
  TEST_FIND_LAST_OF(SV, "qjghlnftcaismkropdeb", "bjaht", 19);
  TEST_FIND_LAST_OF(SV, "pnalfrdtkqcmojiesbhg", "hjlcmgpket", 19);
  TEST_FIND_LAST_OF(SV, "pniotcfrhqsmgdkjbael", "htaobedqikfplcgjsmrn", 19);
  TEST_FIND_LAST_OF(SV, "snafbdlghrjkpqtoceim", "", 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "aemtbrgcklhndjisfpoq", "lbtqd", 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "pnracgfkjdiholtbqsem", "tboimldpjh", 0, 0);
  TEST_FIND_LAST_OF(SV, "dicfltehbsgrmojnpkaq", "slcerthdaiqjfnobgkpm", 0, 0);
  TEST_FIND_LAST_OF(SV, "jlnkraeodhcspfgbqitm", "", 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "lhosrngtmfjikbqpcade", "aqibs", 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "rbtaqjhgkneisldpmfoc", "gtfblmqinc", 1, 1);
  TEST_FIND_LAST_OF(SV, "gpifsqlrdkbonjtmheca", "mkqpbtdalgniorhfescj", 1, 1);
  TEST_FIND_LAST_OF(SV, "hdpkobnsalmcfijregtq", "", 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "jtlshdgqaiprkbcoenfm", "pblas", 10, 10);
  TEST_FIND_LAST_OF(SV, "fkdrbqltsgmcoiphneaj", "arosdhcfme", 10, 10);
  TEST_FIND_LAST_OF(SV, "crsplifgtqedjohnabmk", "blkhjeogicatqfnpdmsr", 10, 10);
  TEST_FIND_LAST_OF(SV, "niptglfbosehkamrdqcj", "", 19, SV::npos);
  TEST_FIND_LAST_OF(SV, "copqdhstbingamjfkler", "djkqc", 19, 16);
  TEST_FIND_LAST_OF(SV, "mrtaefilpdsgocnhqbjk", "lgokshjtpb", 19, 19);
  TEST_FIND_LAST_OF(SV, "kojatdhlcmigpbfrqnes", "bqjhtkfepimcnsgrlado", 19, 19);
  TEST_FIND_LAST_OF(SV, "eaintpchlqsbdgrkjofm", "", 20, SV::npos);
  TEST_FIND_LAST_OF(SV, "gjnhidfsepkrtaqbmclo", "nocfa", 20, 19);
  TEST_FIND_LAST_OF(SV, "spocfaktqdbiejlhngmr", "bgtajmiedc", 20, 18);
  TEST_FIND_LAST_OF(SV, "rphmlekgfscndtaobiqj", "lsckfnqgdahejiopbtmr", 20, 19);
  TEST_FIND_LAST_OF(SV, "liatsqdoegkmfcnbhrpj", "", 21, SV::npos);
  TEST_FIND_LAST_OF(SV, "binjagtfldkrspcomqeh", "gfsrt", 21, 12);
  TEST_FIND_LAST_OF(SV, "latkmisecnorjbfhqpdg", "pfsocbhjtm", 21, 17);
  TEST_FIND_LAST_OF(SV, "lecfratdjkhnsmqpoigb", "tpflmdnoicjgkberhqsa", 21, 19);
}

__host__ __device__ constexpr bool test()
{
  test_find_last_of<cuda::std::string_view>();
#if _CCCL_HAS_CHAR8_T()
  test_find_last_of<cuda::std::u8string_view>();
#endif // _CCCL_HAS_CHAR8_T()
  test_find_last_of<cuda::std::u16string_view>();
  test_find_last_of<cuda::std::u32string_view>();
#if _CCCL_HAS_WCHAR_T()
  test_find_last_of<cuda::std::wstring_view>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
