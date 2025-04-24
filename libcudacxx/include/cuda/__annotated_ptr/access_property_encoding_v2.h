/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA SOFTWARE LICENSE
 *
 * This license is a legal agreement between you and NVIDIA Corporation ("NVIDIA") and governs your use of the
 * NVIDIA/CUDA C++ Library software and materials provided hereunder (“SOFTWARE”).
 *
 * This license can be accepted only by an adult of legal age of majority in the country in which the SOFTWARE is used.
 * If you are under the legal age of majority, you must ask your parent or legal guardian to consent to this license. By
 * taking delivery of the SOFTWARE, you affirm that you have reached the legal age of majority, you accept the terms of
 * this license, and you take legal and financial responsibility for the actions of your permitted users.
 *
 * You agree to use the SOFTWARE only for purposes that are permitted by (a) this license, and (b) any applicable law,
 * regulation or generally accepted practices or guidelines in the relevant jurisdictions.
 *
 * 1. LICENSE. Subject to the terms of this license, NVIDIA grants you a non-exclusive limited license to: (a) install
 * and use the SOFTWARE, and (b) distribute the SOFTWARE subject to the distribution requirements described in this
 * license. NVIDIA reserves all rights, title and interest in and to the SOFTWARE not expressly granted to you under
 * this license.
 *
 * 2. DISTRIBUTION REQUIREMENTS. These are the distribution requirements for you to exercise the distribution grant:
 * a.      The terms under which you distribute the SOFTWARE must be consistent with the terms of this license,
 * including (without limitation) terms relating to the license grant and license restrictions and protection of
 * NVIDIA’s intellectual property rights. b.      You agree to notify NVIDIA in writing of any known or suspected
 * distribution or use of the SOFTWARE not in compliance with the requirements of this license, and to enforce the terms
 * of your agreements with respect to distributed SOFTWARE.
 *
 * 3. LIMITATIONS. Your license to use the SOFTWARE is restricted as follows:
 * a.      The SOFTWARE is licensed for you to develop applications only for use in systems with NVIDIA GPUs.
 * b.      You may not reverse engineer, decompile or disassemble, or remove copyright or other proprietary notices from
 * any portion of the SOFTWARE or copies of the SOFTWARE. c.      You may not modify or create derivative works of any
 * portion of the SOFTWARE. d.      You may not bypass, disable, or circumvent any technical measure, encryption,
 * security, digital rights management or authentication mechanism in the SOFTWARE. e.      You may not use the SOFTWARE
 * in any manner that would cause it to become subject to an open source software license. As examples, licenses that
 * require as a condition of use, modification, and/or distribution that the SOFTWARE be (i) disclosed or distributed in
 * source code form; (ii) licensed for the purpose of making derivative works; or (iii) redistributable at no charge. f.
 * Unless you have an agreement with NVIDIA for this purpose, you may not use the SOFTWARE with any system or
 * application where the use or failure of the system or application can reasonably be expected to threaten or result in
 * personal injury, death, or catastrophic loss. Examples include use in avionics, navigation, military, medical, life
 * support or other life critical applications. NVIDIA does not design, test or manufacture the SOFTWARE for these
 * critical uses and NVIDIA shall not be liable to you or any third party, in whole or in part, for any claims or
 * damages arising from such uses. g.      You agree to defend, indemnify and hold harmless NVIDIA and its affiliates,
 * and their respective employees, contractors, agents, officers and directors, from and against any and all claims,
 * damages, obligations, losses, liabilities, costs or debt, fines, restitutions and expenses (including but not limited
 * to attorney’s fees and costs incident to establishing the right of indemnification) arising out of or related to use
 * of the SOFTWARE outside of the scope of this Agreement, or not in compliance with its terms.
 *
 * 4. PRE-RELEASE. SOFTWARE versions identified as alpha, beta, preview, early access or otherwise as pre-release may
 * not be fully functional, may contain errors or design flaws, and may have reduced or different security, privacy,
 * availability, and reliability standards relative to commercial versions of NVIDIA software and materials. You may use
 * a pre-release SOFTWARE version at your own risk, understanding that these versions are not intended for use in
 * production or business-critical systems.
 *
 * 5. OWNERSHIP. The SOFTWARE and the related intellectual property rights therein are and will remain the sole and
 * exclusive property of NVIDIA or its licensors. The SOFTWARE is copyrighted and protected by the laws of the United
 * States and other countries, and international treaty provisions. NVIDIA may make changes to the SOFTWARE, at any time
 * without notice, but is not obligated to support or update the SOFTWARE.
 *
 * 6. COMPONENTS UNDER OTHER LICENSES. The SOFTWARE may include NVIDIA or third-party components with separate legal
 * notices or terms as may be described in proprietary notices accompanying the SOFTWARE. If and to the extent there is
 * a conflict between the terms in this license and the license terms associated with a component, the license terms
 * associated with the components control only to the extent necessary to resolve the conflict.
 *
 * 7. FEEDBACK. You may, but don’t have to, provide to NVIDIA any Feedback. “Feedback” means any suggestions, bug fixes,
 * enhancements, modifications, feature requests or other feedback regarding the SOFTWARE. For any Feedback that you
 * voluntarily provide, you hereby grant NVIDIA and its affiliates a perpetual, non-exclusive, worldwide, irrevocable
 * license to use, reproduce, modify, license, sublicense (through multiple tiers of sublicensees), and distribute
 * (through multiple tiers of distributors) the Feedback without the payment of any royalties or fees to you. NVIDIA
 * will use Feedback at its choice.
 *
 * 8. NO WARRANTIES. THE SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY EXPRESS OR IMPLIED WARRANTY OF ANY KIND INCLUDING, BUT
 * NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR FITNESS FOR A PARTICULAR PURPOSE. NVIDIA DOES NOT
 * WARRANT THAT THE SOFTWARE WILL MEET YOUR REQUIREMENTS OR THAT THE OPERATION THEREOF WILL BE UNINTERRUPTED OR
 * ERROR-FREE, OR THAT ALL ERRORS WILL BE CORRECTED.
 *
 * 9. LIMITATIONS OF LIABILITY. TO THE MAXIMUM EXTENT PERMITTED BY LAW, NVIDIA AND ITS AFFILIATES SHALL NOT BE LIABLE
 * FOR ANY SPECIAL, INCIDENTAL, PUNITIVE OR CONSEQUENTIAL DAMAGES, OR ANY LOST PROFITS, PROJECT DELAYS, LOSS OF USE,
 * LOSS OF DATA OR LOSS OF GOODWILL, OR THE COSTS OF PROCURING SUBSTITUTE PRODUCTS, ARISING OUT OF OR IN CONNECTION WITH
 * THIS LICENSE OR THE USE OR PERFORMANCE OF THE SOFTWARE, WHETHER SUCH LIABILITY ARISES FROM ANY CLAIM BASED UPON
 * BREACH OF CONTRACT, BREACH OF WARRANTY, TORT (INCLUDING NEGLIGENCE), PRODUCT LIABILITY OR ANY OTHER CAUSE OF ACTION
 * OR THEORY OF LIABILITY, EVEN IF NVIDIA HAS PREVIOUSLY BEEN ADVISED OF, OR COULD REASONABLY HAVE FORESEEN, THE
 * POSSIBILITY OF SUCH DAMAGES. IN NO EVENT WILL NVIDIA’S AND ITS AFFILIATES TOTAL CUMULATIVE LIABILITY UNDER OR ARISING
 * OUT OF THIS LICENSE EXCEED US$10.00. THE NATURE OF THE LIABILITY OR THE NUMBER OF CLAIMS OR SUITS SHALL NOT ENLARGE
 * OR EXTEND THIS LIMIT.
 *
 * 10. TERMINATION. Your rights under this license will terminate automatically without notice from NVIDIA if you fail
 * to comply with any term and condition of this license or if you commence or participate in any legal proceeding
 * against NVIDIA with respect to the SOFTWARE. NVIDIA may terminate this license with advance written notice to you if
 * NVIDIA decides to no longer provide the SOFTWARE in a country or, in NVIDIA’s sole discretion, the continued use of
 * it is no longer commercially viable. Upon any termination of this license, you agree to promptly discontinue use of
 * the SOFTWARE and destroy all copies in your possession or control. Your prior distributions in accordance with this
 * license are not affected by the termination of this license. All provisions of this license will survive termination,
 * except for the license granted to you.
 *
 * 11. APPLICABLE LAW. This license will be governed in all respects by the laws of the United States and of the State
 * of Delaware as those laws are applied to contracts entered into and performed entirely within Delaware by Delaware
 * residents, without regard to the conflicts of laws principles. The United Nations Convention on Contracts for the
 * International Sale of Goods is specifically disclaimed. You agree to all terms of this Agreement in the English
 * language. The state or federal courts residing in Santa Clara County, California shall have exclusive jurisdiction
 * over any dispute or claim arising out of this license. Notwithstanding this, you agree that NVIDIA shall still be
 * allowed to apply for injunctive remedies or an equivalent type of urgent legal relief in any jurisdiction.
 *
 * 12. NO ASSIGNMENT. This license and your rights and obligations thereunder may not be assigned by you by any means or
 * operation of law without NVIDIA’s permission. Any attempted assignment not approved by NVIDIA in writing shall be
 * void and of no effect.
 *
 * 13. EXPORT. The SOFTWARE is subject to United States export laws and regulations. You agree that you will not ship,
 * transfer or export the SOFTWARE into any country, or use the SOFTWARE in any manner, prohibited by the United States
 * Bureau of Industry and Security or economic sanctions regulations administered by the U.S. Department of Treasury’s
 * Office of Foreign Assets Control (OFAC), or any applicable export laws, restrictions or regulations. These laws
 * include restrictions on destinations, end users and end use. By accepting this license, you confirm that you are not
 * a resident or citizen of any country currently embargoed by the U.S. and that you are not otherwise prohibited from
 * receiving the SOFTWARE.
 *
 * 14. GOVERNMENT USE. The SOFTWARE has been developed entirely at private expense and is “commercial items” consisting
 * of “commercial computer software” and “commercial computer software documentation” provided with RESTRICTED RIGHTS.
 * Use, duplication or disclosure by the U.S. Government or a U.S. Government subcontractor is subject to the
 * restrictions in this license pursuant to DFARS 227.7202-3(a) or as set forth in subparagraphs (b)(1) and (2) of the
 * Commercial Computer Software - Restricted Rights clause at FAR 52.227-19, as applicable. Contractor/manufacturer is
 * NVIDIA, 2788 San Tomas Expressway, Santa Clara, CA 95051.
 *
 * 15. ENTIRE AGREEMENT. This license is the final, complete and exclusive agreement between the parties relating to the
 * subject matter of this license and supersedes all prior or contemporaneous understandings and agreements relating to
 * this subject matter, whether oral or written. If any court of competent jurisdiction determines that any provision of
 * this license is illegal, invalid or unenforceable, the remaining provisions will remain in full force and effect.
 * This license may only be modified in a writing signed by an authorized representative of each party.
 *
 * (v. August 20, 2021)
 */
#ifndef _CUDA___ANNOTATED_PTR_ACCESS_PROPERTY_ENCODING
#define _CUDA___ANNOTATED_PTR_ACCESS_PROPERTY_ENCODING

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// #include <cuda_runtime_api.h>

#include <cuda/__annotated_ptr/createpolicy.h>
#include <cuda/__cmath/ilog.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

enum class __l2_descriptor_mode_t : uint32_t
{
  _Desc_Implicit    = 0,
  _Desc_Interleaved = 2,
  _Desc_Block_Type  = 3
};

struct __block_desc_t // 64 bits
{
  uint64_t                   : 37;
  uint32_t __block_count     : 7;
  uint32_t __block_start     : 7;
  uint32_t                   : 1;
  uint32_t __block_size_enum : 4; // 56 bits

  uint32_t __l2_cop_off             : 1;
  uint32_t __l2_cop_on              : 2;
  uint32_t __l2_descriptor_mode     : 2;
  uint32_t __l1_inv_dont_allocate   : 1;
  uint32_t __l2_sector_promote_256B : 1;
  uint32_t                          : 1;
};
static_assert(sizeof(__block_desc_t) == 8, "__block_desc_t should be 8 bytes");

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_BIT_CAST uint64_t __block_encoding(
  void* __ptr, uint32_t __primary_bytes, uint32_t __total_bytes, __l2_evict_t __primary, __l2_evict_t __secondary)
{
  if (!_CUDA_VSTD::is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_DEVICE, (_CCCL_ASSERT(__isGlobal(__ptr), "ptr must be global");))
  }
  _CCCL_ASSERT(__primary_bytes > 0, "primary_size must be greater than 0");
  _CCCL_ASSERT(__primary_bytes <= __total_bytes, "primary_size must be less than or equal to total_size");
  _CCCL_ASSERT(__secondary == __l2_evict_t::_L2_Evict_First || __secondary == __l2_evict_t::_L2_Evict_Unchanged,
               "secondary policy must be evict_first or evict_unchanged");
  auto __raw_ptr         = _CUDA_VSTD::bit_cast<uintptr_t>(__ptr);
  auto __log2_total_size = ::cuda::ceil_ilog2(__total_bytes);
  // replace with _CUDA_VSTD::add_sat when available PR #3449
  auto __block_size_enum = static_cast<uint32_t>(_CUDA_VSTD::max(__log2_total_size - 19, 0)); // min block size = 4K
  auto __log2_block_size = 12u + __block_size_enum;
  auto __block_size      = 1u << __log2_block_size;
  auto __block_start     = static_cast<uint32_t>(__raw_ptr >> __log2_block_size); // ptr / block_size
  // vvvv block_end = ceil_div(ptr + primary_size, block_size)
  auto __block_end = static_cast<uint32_t>((__raw_ptr + __primary_bytes + __block_size - 1) >> __log2_block_size);
  _CCCL_ASSERT(__block_end >= __block_start, "block_end < block_start");
  auto __block_count        = _CUDA_VSTD::clamp(__block_end - __block_start, 1u, 127u);
  auto __l2_cop_off         = _CUDA_VSTD::to_underlying(__secondary);
  auto __l2_cop_on          = _CUDA_VSTD::to_underlying(__primary);
  auto __l2_descriptor_mode = _CUDA_VSTD::to_underlying(__l2_descriptor_mode_t::_Desc_Block_Type);
  __block_desc_t __block_desc{
    __block_count, __block_start, __block_size_enum, __l2_cop_off, __l2_cop_on, __l2_descriptor_mode, 0, 0};
  return __block_desc.__block_count << 23;
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_CONSTEXPR_BIT_CAST uint64_t __l2_interleave(float __fraction)
{
  if (!_CUDA_VSTD::is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_DEVICE, (_CCCL_ASSERT(__isGlobal(__ptr), "ptr must be global");))
  }
  _CCCL_ASSERT(__primary_bytes > 0, "primary_size must be greater than 0");
  _CCCL_ASSERT(__primary_bytes <= __total_bytes, "primary_size must be less than or equal to total_size");
  _CCCL_ASSERT(__secondary == __l2_evict_t::_L2_Evict_First || __secondary == __l2_evict_t::_L2_Evict_Unchanged,
               "secondary policy must be evict_first or evict_unchanged");
  auto __raw_ptr         = _CUDA_VSTD::bit_cast<uintptr_t>(__ptr);
  auto __log2_total_size = ::cuda::ceil_ilog2(__total_bytes);
  // replace with _CUDA_VSTD::add_sat when available PR #3449
  auto __block_size_enum = static_cast<uint32_t>(_CUDA_VSTD::max(__log2_total_size - 19, 0)); // min block size = 4K
  auto __log2_block_size = 12u + __block_size_enum;
  auto __block_size      = 1u << __log2_block_size;
  auto __block_start     = static_cast<uint32_t>(__raw_ptr >> __log2_block_size); // ptr / block_size
  // vvvv block_end = ceil_div(ptr + primary_size, block_size)
  auto __block_end = static_cast<uint32_t>((__raw_ptr + __primary_bytes + __block_size - 1) >> __log2_block_size);
  _CCCL_ASSERT(__block_end >= __block_start, "block_end < block_start");
  auto __block_count        = _CUDA_VSTD::clamp(__block_end - __block_start, 1u, 127u);
  auto __l2_cop_off         = _CUDA_VSTD::to_underlying(__secondary);
  auto __l2_cop_on          = _CUDA_VSTD::to_underlying(__primary);
  auto __l2_descriptor_mode = _CUDA_VSTD::to_underlying(__l2_descriptor_mode_t::_Desc_Block_Type);
  __block_desc_t __block_desc{
    __block_count, __block_start, __block_size_enum, __l2_cop_off, __l2_cop_on, __l2_descriptor_mode, 0, 0};
  return __block_desc.__block_count << 23;
}

// enum class __l2_cop_off_t
//{
//   _L2_Evict_Normal = 0,
//   _L2_Evict_First  = 1,
// };

enum class __l2_evict_t
{
  _L2_Evict_Normal        = 0,
  _L2_Evict_First         = 1,
  _L2_Evict_Last          = 2,
  _L2_Evict_Normal_Demote = 3,
};

enum class __l2_descriptor_mode_t
{
  _Desc_Implicit    = 0,
  _Desc_Interleaved = 2,
  _Desc_Block_Type  = 3,
};

enum class __l2_eviction_max_way_t
{
  _CUDA_Ampere_Max_L2_Ways = 16,
};

enum class __block_size_t
{
  _BlockSize_4K   = 0,
  _BlockSize_8K   = 1,
  _BlockSize_16K  = 2,
  _BlockSize_32K  = 3,
  _BlockSize_64K  = 4,
  _BlockSize_128K = 5,
  _BlockSize_256K = 6,
  _BlockSize_512K = 7,
  _BlockSize_1M   = 8,
  _BlockSize_2M   = 9,
  _BlockSize_4M   = 10,
  _BlockSize_8M   = 11,
  _BlockSize_16M  = 12,
  _BlockSize_32M  = 13,
};

enum class __l2_cop_off_t
{
  _L2_Evict_Normal = 0,
  _L2_Evict_First  = 1,
};

enum class __l2_cop_on_t
{
  _L2_Evict_Normal        = 0,
  _L2_Evict_First         = 1,
  _L2_Evict_Last          = 2,
  _L2_Evict_Normal_Demote = 3,
};

struct __block_desc_t // 56 bits
{
  uint64_t                   : 37;
  uint32_t __block_count     : 7;
  uint32_t __block_start     : 7;
  uint32_t                   : 1;
  uint32_t __block_size_enum : 4;

  uint32_t __l2_cop_off             : 1;
  uint32_t __l2_cop_on              : 2;
  uint32_t __l2_descriptor_mode     : 2;
  uint32_t __l1_inv_dont_allocate   : 1;
  uint32_t __l2_sector_promote_256B : 1;
  uint32_t                          : 1;
};
static_assert(sizeof(__block_desc_t) == 8, "__block_desc_t should be 8 bytes");

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint64_t __block_encoding(
  void* __ptr, uint32_t __primary_size, uint32_t __total_bytes, __l2_evict_t __primary, __l2_evict_t __secondary)
{
  auto __raw_ptr         = _CUDA_VSTD::bit_cast<uintptr_t>(__ptr);
  auto __log2_total_size = ::cuda::ceil_ilog2(__total_bytes);
  auto __block_size_enum = _CUDA_VSTD::max(__log2_total_size - 19u, 0u);
  auto __log2_block_size = 12u + __block_size_enum;
  auto __block_size      = 1u << __log2_block_size;
  auto __block_start     = static_cast<uint32_t>(__raw_ptr >> __log2_block_size);
  // vvvv __block_end = ceil_div(__raw_ptr + __primary_size, __block_size)
  auto __block_end   = static_cast<uint32_t>((__raw_ptr + __primary_size + __block_size - 1) >> __log2_block_size);
  auto __block_count = __block_end - __block_start;
  auto __l2_cop_off  = static_cast<uint32_t>(__secondary);
  auto __l2_cop_on   = static_cast<uint32_t>(__primary);
  auto __l2_descriptor_mode = static_cast<uint32_t>(__l2_descriptor_mode_t::_Desc_Block_Type);
  __block_desc_t __block_desc{
    __block_count, __block_start, __block_size_enum, __l2_cop_off, __l2_cop_on, __l2_descriptor_mode, 0, 0};
  return cuda::std::bit_cast<uint64_t>(__block_desc);
}

// uint32_t __l2_cop_off             : 1;
// uint32_t __l2_cop_on              : 2;
// uint32_t __l2_descriptor_mode     : 2;
// uint32_t __l1_inv_dont_allocate   : 1;
// uint32_t __l2_sector_promote_256B : 1;
// uint32_t __ap_reserved3           : 1;

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint64_t __get_descriptor_cexpr() const noexcept
{
  // clang-format off
    return static_cast<uint64_t>(__reserved1) << 0
         | static_cast<uint64_t>(__block_count) << 37
         | static_cast<uint64_t>(__block_start) << 44
         | static_cast<uint64_t>(__reserved2) << 51
         | static_cast<uint64_t>(__log2_block_size) << 52
         | static_cast<uint64_t>(__l2_cop_off) << 56
         | static_cast<uint64_t>(__l2_cop_on) << 57
         | static_cast<uint64_t>(__l2_descriptor_mode) << 59
         | static_cast<uint64_t>(__l1_inv_dont_allocate) << 61
         | static_cast<uint64_t>(__l2_sector_promote_256B) << 62
         | static_cast<uint64_t>(__ap_reserved3) << 63;
  // clang-format on
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI uint64_t __get_descriptor_non_cexpr() const noexcept
{
  return *reinterpret_cast<const uint64_t*>(this);
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint64_t __get_descriptor() const noexcept
{
  return _CUDA_VSTD::__cccl_default_is_constant_evaluated() ? __get_descriptor_cexpr() : __get_descriptor_non_cexpr();
}
}
;
static_assert(sizeof(__block_desc_t) == 8, "__block_desc_t should be 8 bytes");
static_assert(sizeof(__block_desc_t) == sizeof(uint64_t));
static_assert(
  __block_desc_t{
    uint64_t{1},
    uint64_t{1},
    uint64_t{1},
    uint64_t{1},
    __block_size_t::_BlockSize_8K,
    __off::_L2_Evict_First,
    __on::_L2_Evict_First,
    __l2_descriptor_mode_t::_Desc_Interleaved,
    uint64_t{1},
    uint64_t{1},
    uint64_t{1}}
    .__get_descriptor()
  == 0xF318102000000001);

/* Factory like struct to build a __block_desc_t due to constexpr C++11 */
struct __block_descriptor_builder
{ // variable declaration order matters == usage order
  uint32_t __offset;
  __block_size_t __log2_block_size;
  uint32_t __block_start, __end_hit;
  uint32_t __block_count;
  __off::__l2_cop_off_t __l2_cop_off;
  __on::__l2_cop_on_t __l2_cop_on;
  __l2_descriptor_mode_t __l2_descriptor_mode;
  bool __l1_inv_dont_allocate;
  bool __l2_sector_promote_256B;

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr uint32_t __calc_offset(size_t __total_bytes) noexcept
  {
    return _CUDA_VSTD::max(uint32_t{12}, ::cuda::ceil_ilog2(__total_bytes) - 7u);
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr uint32_t
  __calc_block_start(uintptr_t __ptr, size_t __total_bytes) noexcept
  {
    return static_cast<uint32_t>(__ptr >> __calc_offset(static_cast<uint32_t>(__total_bytes)));
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI static constexpr uint32_t
  __calc_end_hit(uintptr_t __ptr, size_t __hit_bytes, size_t __total_bytes) noexcept
  {
    return static_cast<uint32_t>(
      (__ptr + __hit_bytes + (uintptr_t{1} << (__calc_offset(static_cast<uint32_t>(__total_bytes)))) - 1)
      >> __calc_offset(static_cast<uint32_t>(__total_bytes)));
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __block_descriptor_builder(
    uintptr_t __ptr,
    size_t __hit_bytes,
    size_t __total_bytes,
    __on::__l2_cop_on_t __hit_prop,
    __off::__l2_cop_off_t __miss_prop) noexcept
      : __offset{__calc_offset(__total_bytes)}
      , __log2_block_size{static_cast<__block_size_t>(__calc_offset(__total_bytes) - uint32_t{12})}
      , __block_start{__calc_block_start(__ptr, __total_bytes)}
      , __end_hit{__calc_end_hit(__ptr, __hit_bytes, __total_bytes)}
      , __block_count{__calc_end_hit(__ptr, __hit_bytes, __total_bytes) - __calc_block_start(__ptr, __total_bytes)}
      , __l2_cop_off{__miss_prop}
      , __l2_cop_on{__hit_prop}
      , __l2_descriptor_mode{_Desc_Block_Type}
      , __l1_inv_dont_allocate{false}
      , __l2_sector_promote_256B{false}
  {}

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr __block_desc_t __get_block() const noexcept
  {
    return __block_desc_t{
      0,
      _CUDA_VSTD::min(uint32_t{0x7F}, __block_count),
      __block_start & uint32_t{0x7F},
      0,
      __log2_block_size,
      __l2_cop_off,
      __l2_cop_on,
      _Desc_Block_Type,
      uint64_t{false},
      uint64_t{false},
      uint64_t{0}};
  }
};
static_assert(sizeof(uintptr_t) == 8, "uintptr_t needs at least 5 bytes for this code to work");

struct __interleave_descriptor_t
{
  uint64_t __reserved1                        : 52;
  uint64_t __fraction                         : 4;
  __off::__l2_cop_off_t __l2_cop_off          : 1;
  __on::__l2_cop_on_t __l2_cop_on             : 2;
  __l2_descriptor_mode_t __l2_descriptor_mode : 2;
  uint64_t __l1_inv_dont_allocate             : 1;
  uint64_t __l2_sector_promote_256B           : 1;
  uint64_t __reserved2                        : 1;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __interleave_descriptor_t(
    __on::__l2_cop_on_t __hit_prop, uint32_t __hit_ratio, __off::__l2_cop_off_t __miss_prop) noexcept
      : __reserved1{0}
      , __fraction{__hit_ratio}
      , __l2_cop_off{__miss_prop}
      , __l2_cop_on{__hit_prop}
      , __l2_descriptor_mode{_Desc_Interleaved}
      , __l1_inv_dont_allocate{0}
      , __l2_sector_promote_256B{0}
      , __reserved2{0}
  {}

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint64_t __get_descriptor_cexpr() const noexcept
  {
    // clang-format off
    return static_cast<uint64_t>(__reserved1)
         | static_cast<uint64_t>(__fraction) << 52
         | static_cast<uint64_t>(__l2_cop_off) << 56
         | static_cast<uint64_t>(__l2_cop_on) << 57
         | static_cast<uint64_t>(__l2_descriptor_mode) << 59
         | static_cast<uint64_t>(__l1_inv_dont_allocate) << 61
         | static_cast<uint64_t>(__l2_sector_promote_256B) << 62
         | static_cast<uint64_t>(__reserved2) << 63;
    // clang-format on
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI uint64_t __get_descriptor_non_cexpr() const noexcept
  {
    return *reinterpret_cast<const uint64_t*>(this);
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint64_t __get_descriptor() const noexcept
  {
    return _CUDA_VSTD::__cccl_default_is_constant_evaluated() ? __get_descriptor_cexpr() : __get_descriptor_non_cexpr();
  }
};
static_assert(sizeof(__interleave_descriptor_t) == 8, "__interleave_descriptor_t should be 8 bytes");
static_assert(sizeof(__interleave_descriptor_t) == sizeof(uint64_t));

inline constexpr auto __interleave_normal = uint64_t{0x10F0000000000000};

inline constexpr auto __interleave_streaming = uint64_t{0x12F0000000000000};

inline constexpr auto __interleave_persisting = uint64_t{0x14F0000000000000};

inline constexpr auto __interleave_normal_demote = uint64_t{0x16F0000000000000};

} // namespace __sm_80

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint64_t __interleave(
  cudaAccessProperty __hit_prop, float __hit_ratio, cudaAccessProperty __miss_prop = cudaAccessPropertyNormal) noexcept
{
  _CCCL_ASSERT(__hit_ratio > 0.0f && __hit_ratio <= 1.0f, "__hit_ratio must be between 0.0f and 1.0f");
  auto __l2_cop = __hit_prop == cudaAccessPropertyNormal
                  ? __sm_80::__on::__l2_cop_on_t::_L2_Evict_Normal_Demote
                  : static_cast<__sm_80::__on::__l2_cop_on_t>(__hit_prop);
  auto __hit_ratio1 =
    static_cast<uint32_t>(__hit_ratio * uint32_t{__sm_80::__l2_eviction_max_way_t::_CUDA_Ampere_Max_L2_Ways});
  auto __fraction =
    _CUDA_VSTD::min(__hit_ratio1, uint32_t{__sm_80::__l2_eviction_max_way_t::_CUDA_Ampere_Max_L2_Ways} - 1);
  return __sm_80::__interleave_descriptor_t{
    __l2_cop, __fraction, static_cast<__sm_80::__off::__l2_cop_off_t>(__miss_prop)}
    .__get_descriptor();
}

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI uint64_t _LIBCUDACXX_CONSTEXPR_BIT_CAST __block(
  void* __ptr,
  size_t __hit_bytes,
  size_t __total_bytes,
  cudaAccessProperty __hit_prop,
  cudaAccessProperty __miss_prop = cudaAccessPropertyNormal) noexcept
{
  _CCCL_ASSERT(__ptr != nullptr, "ptr must not be null");
  NV_IF_TARGET(NV_IS_DEVICE, (_CCCL_ASSERT(__isGlobal(__ptr), "ptr must be global");))
  _CCCL_ASSERT(__hit_bytes > 0, "hit_bytes must be greater than 0");
  _CCCL_ASSERT(__hit_bytes <= __total_bytes, "hit_bytes must be less than or equal to total_bytes");
  _CCCL_ASSERT(__total_bytes <= size_t{0xFFFFFFFF}, "total_bytes must be less than or equal to 4GB");
  auto __l2_cop = (__hit_prop == cudaAccessPropertyNormal)
                  ? __sm_80::__on::_L2_Evict_Normal_Demote
                  : static_cast<__sm_80::__on::__l2_cop_on_t>(__hit_prop);
  return __sm_80::__block_descriptor_builder{
    _CUDA_VSTD::bit_cast<uintptr_t>(__ptr),
    __hit_bytes,
    __total_bytes,
    __l2_cop,
    static_cast<__sm_80::__off::__l2_cop_off_t>(__miss_prop)}
    .__get_block()
    .__get_descriptor();
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___ANNOTATED_PTR_ACCESS_PROPERTY_ENCODING
