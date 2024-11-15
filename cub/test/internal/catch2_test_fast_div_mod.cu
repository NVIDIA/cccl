/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#include <cub/config.cuh>

#if __cccl_lib_mdspan

#  include "c2h/catch2_test_helper.h"
#  include "c2h/utility.h"
#  include "catch2_test_launch_helper.h"

// %PARAM% TEST_LAUNCH lid 0:1:2

DECLARE_LAUNCH_WRAPPER(cub::DeviceFor::ForEachInExtents, device_for_each_in_extents);

/***********************************************************************************************************************
 * TEST CASES
 **********************************************************************************************************************/

using index_types =
  c2h::type_list<int8_t,
                 uint8_t,
                 int16_t,
                 uint16_t,
                 int32_t,
                 uint32_t
#  if CUB_IS_INT128_ENABLED
                 ,
                 int64_t,
                 uint64_t
#  endif
                 >;

C2H_TEST("FastDivMod", "[FastDivMod][Random]", index_types)
{
  using cub::detail::fast_div_mod;
  using index_type         = c2h::get<0, TestType>;
  constexpr auto max_value = cuda::std::numeric_limits<index_type>::max();
  auto divisor             = GENERATE(take(100, random(index_type{1}, max_value)));
  auto dividend            = GENERATE(take(100, random(index_type{1}, max_value)));
  fast_div_mod<index_type> div_mod(divisor);
  static_cast<void>(div_mod(dividend));
}

C2H_TEST("FastDivMod", "[FastDivMod][EdgeCases]", index_types)
{
  using cub::detail::fast_div_mod;
  using index_type         = c2h::get<0, TestType>;
  constexpr auto max_value = cuda::std::numeric_limits<index_type>::max();
  // divisor/dividend == max
  fast_div_mod<index_type> div_mod_max(max_value);
  static_cast<void>(div_mod_max(max_value));
  // divisor == 0, /dividend == max
  fast_div_mod<index_type> div_mod_min(max_value);
  static_cast<void>(div_mod_min(10));
}

#endif // __cccl_lib_mdspan
