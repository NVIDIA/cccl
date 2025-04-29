//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr size_type find_last_of(const charT* s, size_type pos, size_type n) const;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "literal.h"

template <class SV>
__host__ __device__ constexpr void test_find_last_of(
  const SV& sv,
  const typename SV::value_type* str,
  typename SV::size_type pos,
  typename SV::size_type n,
  typename SV::size_type x)
{
  assert(sv.find_last_of(str, pos, n) == x);
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

  static_assert(
    cuda::std::is_same_v<SizeT, decltype(SV{}.find_last_of(cuda::std::declval<const CharT*>(), SizeT{}, SizeT{}))>);
  static_assert(noexcept(SV{}.find_last_of(cuda::std::declval<const CharT*>(), SizeT{}, SizeT{})));

  TEST_FIND_LAST_OF(SV, "", "", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "irkhs", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "kante", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "oknlr", 0, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "pcdro", 0, 4, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "bnrpe", 0, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "jtdaefblso", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "oselktgbca", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "eqgaplhckj", 0, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "bjahtcmnlp", 0, 9, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "hjlcmgpket", 0, 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "htaobedqikfplcgjsmrn", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "hpqiarojkcdlsgnmfetb", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "dfkaprhjloqetcsimnbg", 0, 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "ihqrfebgadntlpmjksoc", 0, 19, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "ngtjfcalbseiqrphmkdo", 0, 20, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "lbtqd", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "tboim", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "slcer", 1, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "cbjfs", 1, 4, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "aqibs", 1, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "gtfblmqinc", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "mkqpbtdalg", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "kphatlimcd", 1, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "pblasqogic", 1, 9, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "arosdhcfme", 1, 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "blkhjeogicatqfnpdmsr", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "bmhineprjcoadgstflqk", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "djkqcmetslnghpbarfoi", 1, 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "lgokshjtpbemarcdqnfi", 1, 19, SV::npos);
  TEST_FIND_LAST_OF(SV, "", "bqjhtkfepimcnsgrlado", 1, 20, SV::npos);

  TEST_FIND_LAST_OF(SV, "eaint", "", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "binja", "gfsrt", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "latkm", "pfsoc", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "lecfr", "tpflm", 0, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "eqkst", "sgkec", 0, 4, 0);
  TEST_FIND_LAST_OF(SV, "cdafr", "romds", 0, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "prbhe", "qhjistlgmr", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "lbisk", "pedfirsglo", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "hrlpd", "aqcoslgrmk", 0, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "ehmja", "dabckmepqj", 0, 9, 0);
  TEST_FIND_LAST_OF(SV, "mhqgd", "pqscrjthli", 0, 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "tgklq", "kfphdcsjqmobliagtren", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "bocjs", "rokpefncljibsdhqtagm", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "grbsd", "afionmkphlebtcjqsgrd", 0, 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "ofjqr", "aenmqplidhkofrjbctsg", 0, 19, 0);
  TEST_FIND_LAST_OF(SV, "btlfi", "osjmbtcadhiklegrpqnf", 0, 20, 0);
  TEST_FIND_LAST_OF(SV, "clrgb", "", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "tjmek", "osmia", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "bgstp", "ckonl", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "hstrk", "ilcaj", 1, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "kmspj", "lasiq", 1, 4, SV::npos);
  TEST_FIND_LAST_OF(SV, "tjboh", "kfqmr", 1, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "ilbcj", "klnitfaobg", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "jkngf", "gjhmdlqikp", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "gfcql", "skbgtahqej", 1, 5, 0);
  TEST_FIND_LAST_OF(SV, "dqtlg", "bjsdgtlpkf", 1, 9, 0);
  TEST_FIND_LAST_OF(SV, "bthpg", "bjgfmnlkio", 1, 10, 0);
  TEST_FIND_LAST_OF(SV, "dgsnq", "lbhepotfsjdqigcnamkr", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "rmfhp", "tebangckmpsrqdlfojhi", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "jfdam", "joflqbdkhtegimscpanr", 1, 10, 1);
  TEST_FIND_LAST_OF(SV, "edapb", "adpmcohetfbsrjinlqkg", 1, 19, 1);
  TEST_FIND_LAST_OF(SV, "brfsm", "iacldqjpfnogbsrhmetk", 1, 20, 1);
  TEST_FIND_LAST_OF(SV, "ndrhl", "", 2, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "mrecp", "otkgb", 2, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "qlasf", "cqsjl", 2, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "smaqd", "dpifl", 2, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "hjeni", "oapht", 2, 4, 0);
  TEST_FIND_LAST_OF(SV, "ocmfj", "cifts", 2, 5, 1);
  TEST_FIND_LAST_OF(SV, "hmftq", "nmsckbgalo", 2, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "fklad", "tpksqhamle", 2, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "dirnm", "tpdrchmkji", 2, 5, 2);
  TEST_FIND_LAST_OF(SV, "hrgdc", "ijagfkblst", 2, 9, 2);
  TEST_FIND_LAST_OF(SV, "ifakg", "kpocsignjb", 2, 10, 0);
  TEST_FIND_LAST_OF(SV, "ebrgd", "pecqtkjsnbdrialgmohf", 2, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "rcjml", "aiortphfcmkjebgsndql", 2, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "peqmt", "sdbkeamglhipojqftrcn", 2, 10, 1);
  TEST_FIND_LAST_OF(SV, "frehn", "ljqncehgmfktroapidbs", 2, 19, 2);
  TEST_FIND_LAST_OF(SV, "tqolf", "rtcfodilamkbenjghqps", 2, 20, 2);
  TEST_FIND_LAST_OF(SV, "cjgao", "", 4, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "kjplq", "mabns", 4, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "herni", "bdnrp", 4, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "tadrb", "scidp", 4, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "pkfeo", "agbjl", 4, 4, SV::npos);
  TEST_FIND_LAST_OF(SV, "hoser", "jfmpr", 4, 5, 4);
  TEST_FIND_LAST_OF(SV, "kgrsp", "rbpefghsmj", 4, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "pgejb", "apsfntdoqc", 4, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "thlnq", "ndkjeisgcl", 4, 5, 3);
  TEST_FIND_LAST_OF(SV, "nbmit", "rnfpqatdeo", 4, 9, 4);
  TEST_FIND_LAST_OF(SV, "jgmib", "bntjlqrfik", 4, 10, 4);
  TEST_FIND_LAST_OF(SV, "ncrfj", "kcrtmpolnaqejghsfdbi", 4, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "ncsik", "lobheanpkmqidsrtcfgj", 4, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "sgbfh", "athdkljcnreqbgpmisof", 4, 10, 4);
  TEST_FIND_LAST_OF(SV, "dktbn", "qkdmjialrscpbhefgont", 4, 19, 4);
  TEST_FIND_LAST_OF(SV, "fthqm", "dmasojntqleribkgfchp", 4, 20, 4);
  TEST_FIND_LAST_OF(SV, "klopi", "", 5, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "dajhn", "psthd", 5, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "jbgno", "rpmjd", 5, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "hkjae", "dfsmk", 5, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "gbhqo", "skqne", 5, 4, 3);
  TEST_FIND_LAST_OF(SV, "ktdor", "kipnf", 5, 5, 0);
  TEST_FIND_LAST_OF(SV, "ldprn", "hmrnqdgifl", 5, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "egmjk", "fsmjcdairn", 5, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "armql", "pcdgltbrfj", 5, 5, 4);
  TEST_FIND_LAST_OF(SV, "cdhjo", "aekfctpirg", 5, 9, 0);
  TEST_FIND_LAST_OF(SV, "jcons", "ledihrsgpf", 5, 10, 4);
  TEST_FIND_LAST_OF(SV, "cbrkp", "mqcklahsbtirgopefndj", 5, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "fhgna", "kmlthaoqgecrnpdbjfis", 5, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "ejfcd", "sfhbamcdptojlkrenqgi", 5, 10, 4);
  TEST_FIND_LAST_OF(SV, "kqjhe", "pbniofmcedrkhlstgaqj", 5, 19, 4);
  TEST_FIND_LAST_OF(SV, "pbdjl", "mongjratcskbhqiepfdl", 5, 20, 4);
  TEST_FIND_LAST_OF(SV, "gajqn", "", 6, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "stedk", "hrnat", 6, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "tjkaf", "gsqdt", 6, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "dthpe", "bspkd", 6, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "klhde", "ohcmb", 6, 4, 2);
  TEST_FIND_LAST_OF(SV, "bhlki", "heatr", 6, 5, 1);
  TEST_FIND_LAST_OF(SV, "lqmoh", "pmblckedfn", 6, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "mtqin", "aceqmsrbik", 6, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "dpqbr", "lmbtdehjrn", 6, 5, 3);
  TEST_FIND_LAST_OF(SV, "kdhmo", "teqmcrlgib", 6, 9, 3);
  TEST_FIND_LAST_OF(SV, "jblqp", "njolbmspac", 6, 10, 4);
  TEST_FIND_LAST_OF(SV, "qmjgl", "pofnhidklamecrbqjgst", 6, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "rothp", "jbhckmtgrqnosafedpli", 6, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "ghknq", "dobntpmqklicsahgjerf", 6, 10, 4);
  TEST_FIND_LAST_OF(SV, "eopfi", "tpdshainjkbfoemlrgcq", 6, 19, 4);
  TEST_FIND_LAST_OF(SV, "dsnmg", "oldpfgeakrnitscbjmqh", 6, 20, 4);

  TEST_FIND_LAST_OF(SV, "jnkrfhotgl", "", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "dltjfngbko", "rqegt", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "bmjlpkiqde", "dashm", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "skrflobnqm", "jqirk", 0, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "jkpldtshrm", "rckeg", 0, 4, SV::npos);
  TEST_FIND_LAST_OF(SV, "ghasdbnjqo", "jscie", 0, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "igrkhpbqjt", "efsphndliq", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "ikthdgcamf", "gdicosleja", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "pcofgeniam", "qcpjibosfl", 0, 5, 0);
  TEST_FIND_LAST_OF(SV, "rlfjgesqhc", "lrhmefnjcq", 0, 9, 0);
  TEST_FIND_LAST_OF(SV, "itphbqsker", "dtablcrseo", 0, 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "skjafcirqm", "apckjsftedbhgomrnilq", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "tcqomarsfd", "pcbrgflehjtiadnsokqm", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "rocfeldqpk", "nsiadegjklhobrmtqcpf", 0, 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "cfpegndlkt", "cpmajdqnolikhgsbretf", 0, 19, 0);
  TEST_FIND_LAST_OF(SV, "fqbtnkeasj", "jcflkntmgiqrphdosaeb", 0, 20, 0);
  TEST_FIND_LAST_OF(SV, "shbcqnmoar", "", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "bdoshlmfin", "ontrs", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "khfrebnsgq", "pfkna", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "getcrsaoji", "ekosa", 1, 2, 1);
  TEST_FIND_LAST_OF(SV, "fjiknedcpq", "anqhk", 1, 4, SV::npos);
  TEST_FIND_LAST_OF(SV, "tkejgnafrm", "jekca", 1, 5, 1);
  TEST_FIND_LAST_OF(SV, "jnakolqrde", "ikemsjgacf", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "lcjptsmgbe", "arolgsjkhm", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "itfsmcjorl", "oftkbldhre", 1, 5, 1);
  TEST_FIND_LAST_OF(SV, "omchkfrjea", "gbkqdoeftl", 1, 9, 0);
  TEST_FIND_LAST_OF(SV, "cigfqkated", "sqcflrgtim", 1, 10, 1);
  TEST_FIND_LAST_OF(SV, "tscenjikml", "fmhbkislrjdpanogqcet", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "qcpaemsinf", "rnioadktqlgpbcjsmhef", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "gltkojeipd", "oakgtnldpsefihqmjcbr", 1, 10, 1);
  TEST_FIND_LAST_OF(SV, "qistfrgnmp", "gbnaelosidmcjqktfhpr", 1, 19, 1);
  TEST_FIND_LAST_OF(SV, "bdnpfcqaem", "akbripjhlosndcmqgfet", 1, 20, 1);
  TEST_FIND_LAST_OF(SV, "ectnhskflp", "", 5, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "fgtianblpq", "pijag", 5, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "mfeqklirnh", "jrckd", 5, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "astedncjhk", "qcloh", 5, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "fhlqgcajbr", "thlmp", 5, 4, 2);
  TEST_FIND_LAST_OF(SV, "epfhocmdng", "qidmo", 5, 5, 4);
  TEST_FIND_LAST_OF(SV, "apcnsibger", "lnegpsjqrd", 5, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "aqkocrbign", "rjqdablmfs", 5, 1, 5);
  TEST_FIND_LAST_OF(SV, "ijsmdtqgce", "enkgpbsjaq", 5, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "clobgsrken", "kdsgoaijfh", 5, 9, 5);
  TEST_FIND_LAST_OF(SV, "jbhcfposld", "trfqgmckbe", 5, 10, 4);
  TEST_FIND_LAST_OF(SV, "oqnpblhide", "igetsracjfkdnpoblhqm", 5, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "lroeasctif", "nqctfaogirshlekbdjpm", 5, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "bpjlgmiedh", "csehfgomljdqinbartkp", 5, 10, 5);
  TEST_FIND_LAST_OF(SV, "pamkeoidrj", "qahoegcmplkfsjbdnitr", 5, 19, 5);
  TEST_FIND_LAST_OF(SV, "espogqbthk", "dpteiajrqmsognhlfbkc", 5, 20, 5);
  TEST_FIND_LAST_OF(SV, "shoiedtcjb", "", 9, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "ebcinjgads", "tqbnh", 9, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "dqmregkcfl", "akmle", 9, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "ngcrieqajf", "iqfkm", 9, 2, 6);
  TEST_FIND_LAST_OF(SV, "qosmilgnjb", "tqjsr", 9, 4, 8);
  TEST_FIND_LAST_OF(SV, "ikabsjtdfl", "jplqg", 9, 5, 9);
  TEST_FIND_LAST_OF(SV, "ersmicafdh", "oilnrbcgtj", 9, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "fdnplotmgh", "morkglpesn", 9, 1, 7);
  TEST_FIND_LAST_OF(SV, "fdbicojerm", "dmicerngat", 9, 5, 9);
  TEST_FIND_LAST_OF(SV, "mbtafndjcq", "radgeskbtc", 9, 9, 6);
  TEST_FIND_LAST_OF(SV, "mlenkpfdtc", "ljikprsmqo", 9, 10, 5);
  TEST_FIND_LAST_OF(SV, "ahlcifdqgs", "trqihkcgsjamfdbolnpe", 9, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "bgjemaltks", "lqmthbsrekajgnofcipd", 9, 1, 6);
  TEST_FIND_LAST_OF(SV, "pdhslbqrfc", "jtalmedribkgqsopcnfh", 9, 10, 7);
  TEST_FIND_LAST_OF(SV, "dirhtsnjkc", "spqfoiclmtagejbndkrh", 9, 19, 9);
  TEST_FIND_LAST_OF(SV, "dlroktbcja", "nmotklspigjrdhcfaebq", 9, 20, 9);
  TEST_FIND_LAST_OF(SV, "ncjpmaekbs", "", 10, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "hlbosgmrak", "hpmsd", 10, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "pqfhsgilen", "qnpor", 10, 1, 1);
  TEST_FIND_LAST_OF(SV, "gqtjsbdckh", "otdma", 10, 2, 2);
  TEST_FIND_LAST_OF(SV, "cfkqpjlegi", "efhjg", 10, 4, 7);
  TEST_FIND_LAST_OF(SV, "beanrfodgj", "odpte", 10, 5, 7);
  TEST_FIND_LAST_OF(SV, "adtkqpbjfi", "bctdgfmolr", 10, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "iomkfthagj", "oaklidrbqg", 10, 1, 1);
  TEST_FIND_LAST_OF(SV, "sdpcilonqj", "dnjfsagktr", 10, 5, 9);
  TEST_FIND_LAST_OF(SV, "gtfbdkqeml", "nejaktmiqg", 10, 9, 8);
  TEST_FIND_LAST_OF(SV, "bmeqgcdorj", "pjqonlebsf", 10, 10, 9);
  TEST_FIND_LAST_OF(SV, "etqlcanmob", "dshmnbtolcjepgaikfqr", 10, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "roqmkbdtia", "iogfhpabtjkqlrnemcds", 10, 1, 8);
  TEST_FIND_LAST_OF(SV, "kadsithljf", "ngridfabjsecpqltkmoh", 10, 10, 9);
  TEST_FIND_LAST_OF(SV, "sgtkpbfdmh", "athmknplcgofrqejsdib", 10, 19, 9);
  TEST_FIND_LAST_OF(SV, "qgmetnabkl", "ldobhmqcafnjtkeisgrp", 10, 20, 9);
  TEST_FIND_LAST_OF(SV, "cqjohampgd", "", 11, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "hobitmpsan", "aocjb", 11, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "tjehkpsalm", "jbrnk", 11, 1, 1);
  TEST_FIND_LAST_OF(SV, "ngfbojitcl", "tqedg", 11, 2, 7);
  TEST_FIND_LAST_OF(SV, "rcfkdbhgjo", "nqskp", 11, 4, 3);
  TEST_FIND_LAST_OF(SV, "qghptonrea", "eaqkl", 11, 5, 9);
  TEST_FIND_LAST_OF(SV, "hnprfgqjdl", "reaoicljqm", 11, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "hlmgabenti", "lsftgajqpm", 11, 1, 1);
  TEST_FIND_LAST_OF(SV, "ofcjanmrbs", "rlpfogmits", 11, 5, 7);
  TEST_FIND_LAST_OF(SV, "jqedtkornm", "shkncmiaqj", 11, 9, 9);
  TEST_FIND_LAST_OF(SV, "rfedlasjmg", "fpnatrhqgs", 11, 10, 9);
  TEST_FIND_LAST_OF(SV, "talpqjsgkm", "sjclemqhnpdbgikarfot", 11, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "lrkcbtqpie", "otcmedjikgsfnqbrhpla", 11, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "cipogdskjf", "bonsaefdqiprkhlgtjcm", 11, 10, 9);
  TEST_FIND_LAST_OF(SV, "nqedcojahi", "egpscmahijlfnkrodqtb", 11, 19, 9);
  TEST_FIND_LAST_OF(SV, "hefnrkmctj", "kmqbfepjthgilscrndoa", 11, 20, 9);

  TEST_FIND_LAST_OF(SV, "atqirnmekfjolhpdsgcb", "", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "echfkmlpribjnqsaogtd", "prboq", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "qnhiftdgcleajbpkrosm", "fjcqh", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "chamfknorbedjitgslpq", "fmosa", 0, 2, SV::npos);
  TEST_FIND_LAST_OF(SV, "njhqpibfmtlkaecdrgso", "qdbok", 0, 4, SV::npos);
  TEST_FIND_LAST_OF(SV, "ebnghfsqkprmdcljoiat", "amslg", 0, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "letjomsgihfrpqbkancd", "smpltjneqb", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "nblgoipcrqeaktshjdmf", "flitskrnge", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "cehkbngtjoiflqapsmrd", "pgqihmlbef", 0, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "mignapfoklbhcqjetdrs", "cfpdqjtgsb", 0, 9, SV::npos);
  TEST_FIND_LAST_OF(SV, "ceatbhlsqjgpnokfrmdi", "htpsiaflom", 0, 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "ocihkjgrdelpfnmastqb", "kpjfiaceghsrdtlbnomq", 0, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "noelgschdtbrjfmiqkap", "qhtbomidljgafneksprc", 0, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "dkclqfombepritjnghas", "nhtjobkcefldimpsaqgr", 0, 10, SV::npos);
  TEST_FIND_LAST_OF(SV, "miklnresdgbhqcojftap", "prabcjfqnoeskilmtgdh", 0, 19, 0);
  TEST_FIND_LAST_OF(SV, "htbcigojaqmdkfrnlsep", "dtrgmchilkasqoebfpjn", 0, 20, 0);
  TEST_FIND_LAST_OF(SV, "febhmqtjanokscdirpgl", "", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "loakbsqjpcrdhftniegm", "sqome", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "reagphsqflbitdcjmkno", "smfte", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "jitlfrqemsdhkopncabg", "ciboh", 1, 2, 1);
  TEST_FIND_LAST_OF(SV, "mhtaepscdnrjqgbkifol", "haois", 1, 4, 1);
  TEST_FIND_LAST_OF(SV, "tocesrfmnglpbjihqadk", "abfki", 1, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "lpfmctjrhdagneskbqoi", "frdkocntmq", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "lsmqaepkdhncirbtjfgo", "oasbpedlnr", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "epoiqmtldrabnkjhcfsg", "kltqmhgand", 1, 5, SV::npos);
  TEST_FIND_LAST_OF(SV, "emgasrilpknqojhtbdcf", "gdtfjchpmr", 1, 9, 1);
  TEST_FIND_LAST_OF(SV, "hnfiagdpcklrjetqbsom", "ponmcqblet", 1, 10, 1);
  TEST_FIND_LAST_OF(SV, "nsdfebgajhmtricpoklq", "sgphqdnofeiklatbcmjr", 1, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "atjgfsdlpobmeiqhncrk", "ljqprsmigtfoneadckbh", 1, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "sitodfgnrejlahcbmqkp", "ligeojhafnkmrcsqtbdp", 1, 10, 1);
  TEST_FIND_LAST_OF(SV, "fraghmbiceknltjpqosd", "lsimqfnjarbopedkhcgt", 1, 19, 1);
  TEST_FIND_LAST_OF(SV, "pmafenlhqtdbkirjsogc", "abedmfjlghniorcqptks", 1, 20, 1);
  TEST_FIND_LAST_OF(SV, "pihgmoeqtnakrjslcbfd", "", 10, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "gjdkeprctqblnhiafsom", "hqtoa", 10, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "mkpnblfdsahrcqijteog", "cahif", 10, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "gckarqnelodfjhmbptis", "kehis", 10, 2, 7);
  TEST_FIND_LAST_OF(SV, "gqpskidtbclomahnrjfe", "kdlmh", 10, 4, 10);
  TEST_FIND_LAST_OF(SV, "pkldjsqrfgitbhmaecno", "paeql", 10, 5, 6);
  TEST_FIND_LAST_OF(SV, "aftsijrbeklnmcdqhgop", "aghoqiefnb", 10, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "mtlgdrhafjkbiepqnsoc", "jrbqaikpdo", 10, 1, 9);
  TEST_FIND_LAST_OF(SV, "pqgirnaefthokdmbsclj", "smjonaeqcl", 10, 5, 5);
  TEST_FIND_LAST_OF(SV, "kpdbgjmtherlsfcqoina", "eqbdrkcfah", 10, 9, 10);
  TEST_FIND_LAST_OF(SV, "jrlbothiknqmdgcfasep", "kapmsienhf", 10, 10, 9);
  TEST_FIND_LAST_OF(SV, "mjogldqferckabinptsh", "jpqotrlenfcsbhkaimdg", 10, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "apoklnefbhmgqcdrisjt", "jlbmhnfgtcqprikeados", 10, 1, SV::npos);
  TEST_FIND_LAST_OF(SV, "ifeopcnrjbhkdgatmqls", "stgbhfmdaljnpqoicker", 10, 10, 10);
  TEST_FIND_LAST_OF(SV, "ckqhaiesmjdnrgolbtpf", "oihcetflbjagdsrkmqpn", 10, 19, 10);
  TEST_FIND_LAST_OF(SV, "bnlgapfimcoterskqdjh", "adtclebmnpjsrqfkigoh", 10, 20, 10);
  TEST_FIND_LAST_OF(SV, "kgdlrobpmjcthqsafeni", "", 19, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "dfkechomjapgnslbtqir", "beafg", 19, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "rloadknfbqtgmhcsipje", "iclat", 19, 1, 16);
  TEST_FIND_LAST_OF(SV, "mgjhkolrnadqbpetcifs", "rkhnf", 19, 2, 7);
  TEST_FIND_LAST_OF(SV, "cmlfakiojdrgtbsphqen", "clshq", 19, 4, 16);
  TEST_FIND_LAST_OF(SV, "kghbfipeomsntdalrqjc", "dtcoj", 19, 5, 19);
  TEST_FIND_LAST_OF(SV, "eldiqckrnmtasbghjfpo", "rqosnjmfth", 19, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "abqjcfedgotihlnspkrm", "siatdfqglh", 19, 1, 15);
  TEST_FIND_LAST_OF(SV, "qfbadrtjsimkolcenhpg", "mrlshtpgjq", 19, 5, 17);
  TEST_FIND_LAST_OF(SV, "abseghclkjqifmtodrnp", "adlcskgqjt", 19, 9, 16);
  TEST_FIND_LAST_OF(SV, "ibmsnlrjefhtdokacqpg", "drshcjknaf", 19, 10, 16);
  TEST_FIND_LAST_OF(SV, "mrkfciqjebaponsthldg", "etsaqroinghpkjdlfcbm", 19, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "mjkticdeoqshpalrfbgn", "sgepdnkqliambtrocfhj", 19, 1, 10);
  TEST_FIND_LAST_OF(SV, "rqnoclbdejgiphtfsakm", "nlmcjaqgbsortfdihkpe", 19, 10, 19);
  TEST_FIND_LAST_OF(SV, "plkqbhmtfaeodjcrsing", "racfnpmosldibqkghjet", 19, 19, 19);
  TEST_FIND_LAST_OF(SV, "oegalhmstjrfickpbndq", "fjhdsctkqeiolagrnmbp", 19, 20, 19);
  TEST_FIND_LAST_OF(SV, "rdtgjcaohpblniekmsfq", "", 20, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "ofkqbnjetrmsaidphglc", "ejanp", 20, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "grkpahljcftesdmonqib", "odife", 20, 1, 15);
  TEST_FIND_LAST_OF(SV, "jimlgbhfqkteospardcn", "okaqd", 20, 2, 12);
  TEST_FIND_LAST_OF(SV, "gftenihpmslrjkqadcob", "lcdbi", 20, 4, 19);
  TEST_FIND_LAST_OF(SV, "bmhldogtckrfsanijepq", "fsqbj", 20, 5, 19);
  TEST_FIND_LAST_OF(SV, "nfqkrpjdesabgtlcmoih", "bigdomnplq", 20, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "focalnrpiqmdkstehbjg", "apiblotgcd", 20, 1, 3);
  TEST_FIND_LAST_OF(SV, "rhqdspkmebiflcotnjga", "acfhdenops", 20, 5, 19);
  TEST_FIND_LAST_OF(SV, "rahdtmsckfboqlpniegj", "jopdeamcrk", 20, 9, 19);
  TEST_FIND_LAST_OF(SV, "fbkeiopclstmdqranjhg", "trqncbkgmh", 20, 10, 19);
  TEST_FIND_LAST_OF(SV, "lifhpdgmbconstjeqark", "tomglrkencbsfjqpihda", 20, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "pboqganrhedjmltsicfk", "gbkhdnpoietfcmrslajq", 20, 1, 4);
  TEST_FIND_LAST_OF(SV, "klchabsimetjnqgorfpd", "rtfnmbsglkjaichoqedp", 20, 10, 17);
  TEST_FIND_LAST_OF(SV, "sirfgmjqhctndbklaepo", "ohkmdpfqbsacrtjnlgei", 20, 19, 19);
  TEST_FIND_LAST_OF(SV, "rlbdsiceaonqjtfpghkm", "dlbrteoisgphmkncajfq", 20, 20, 19);
  TEST_FIND_LAST_OF(SV, "ecgdanriptblhjfqskom", "", 21, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "fdmiarlpgcskbhoteqjn", "sjrlo", 21, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "rlbstjqopignecmfadkh", "qjpor", 21, 1, 6);
  TEST_FIND_LAST_OF(SV, "grjpqmbshektdolcafni", "odhfn", 21, 2, 13);
  TEST_FIND_LAST_OF(SV, "sakfcohtqnibprjmlged", "qtfin", 21, 4, 10);
  TEST_FIND_LAST_OF(SV, "mjtdglasihqpocebrfkn", "hpqfo", 21, 5, 17);
  TEST_FIND_LAST_OF(SV, "okaplfrntghqbmeicsdj", "fabmertkos", 21, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "sahngemrtcjidqbklfpo", "brqtgkmaej", 21, 1, 14);
  TEST_FIND_LAST_OF(SV, "dlmsipcnekhbgoaftqjr", "nfrdeihsgl", 21, 5, 19);
  TEST_FIND_LAST_OF(SV, "ahegrmqnoiklpfsdbcjt", "hlfrosekpi", 21, 9, 14);
  TEST_FIND_LAST_OF(SV, "hdsjbnmlegtkqripacof", "atgbkrjdsm", 21, 10, 16);
  TEST_FIND_LAST_OF(SV, "pcnedrfjihqbalkgtoms", "blnrptjgqmaifsdkhoec", 21, 0, SV::npos);
  TEST_FIND_LAST_OF(SV, "qjidealmtpskrbfhocng", "ctpmdahebfqjgknloris", 21, 1, 17);
  TEST_FIND_LAST_OF(SV, "qeindtagmokpfhsclrbj", "apnkeqthrmlbfodiscgj", 21, 10, 17);
  TEST_FIND_LAST_OF(SV, "kpfegbjhsrnodltqciam", "jdgictpframeoqlsbknh", 21, 19, 19);
  TEST_FIND_LAST_OF(SV, "hnbrcplsjfgiktoedmaq", "qprlsfojamgndekthibc", 21, 20, 19);
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
