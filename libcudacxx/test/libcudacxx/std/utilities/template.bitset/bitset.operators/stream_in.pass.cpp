//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization

// test:

// template <class charT, class traits, size_t N>
// basic_istream<charT, traits>&
// operator>>(basic_istream<charT, traits>& is, bitset<N>& x);

#include <cuda/std/version>

#ifndef _LIBCUDACXX_HAS_SSTREAM
int main(int, char**)
{
  return 0;
}
#else

#  include <cuda/std/bitset>
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::istringstream in("01011010");
    cuda::std::bitset<8> b;
    in >> b;
    assert(b.to_ulong() == 0x5A);
  }
  {
    // Make sure that input-streaming an empty bitset does not cause the
    // failbit to be set (LWG 3199).
    cuda::std::istringstream in("01011010");
    cuda::std::bitset<0> b;
    in >> b;
    assert(b.to_string() == "");
    assert(!in.bad());
    assert(!in.fail());
    assert(!in.eof());
    assert(in.good());
  }
#  ifndef TEST_HAS_NO_EXCEPTIONS
  {
    cuda::std::stringbuf sb;
    cuda::std::istream is(&sb);
    is.exceptions(cuda::std::ios::failbit);

    bool threw = false;
    try
    {
      cuda::std::bitset<8> b;
      is >> b;
    }
    catch (cuda::std::ios::failure const&)
    {
      threw = true;
    }

    assert(!is.bad());
    assert(is.fail());
    assert(is.eof());
    assert(threw);
  }
  {
    cuda::std::stringbuf sb;
    cuda::std::istream is(&sb);
    is.exceptions(cuda::std::ios::eofbit);

    bool threw = false;
    try
    {
      cuda::std::bitset<8> b;
      is >> b;
    }
    catch (cuda::std::ios::failure const&)
    {
      threw = true;
    }

    assert(!is.bad());
    assert(is.fail());
    assert(is.eof());
    assert(threw);
  }
#  endif // TEST_HAS_NO_EXCEPTIONS

  return 0;
}

#endif
