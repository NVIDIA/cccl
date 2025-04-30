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

#include <cuda/__annotated_ptr/createpolicy.h>
#include <cuda/__cmath/ilog.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__utility/to_underlying.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

enum class __l2_descriptor_mode_t : uint32_t
{
  _Desc_Implicit    = 0,
  _Desc_Interleaved = 2,
  _Desc_Block_Type  = 3
};

/***********************************************************************************************************************
 * Range Block Descriptor
 **********************************************************************************************************************/

// MemoryDescriptor:blockDesc_t  reference
//
// struct __block_desc_t // 64 bits
// {
//   uint64_t __reserved1       : 37;
//   uint32_t __block_count     : 7;
//   uint32_t __block_start     : 7;
//   uint32_t __reserved2       : 1;
//   uint32_t __block_size_enum : 4; // 56 bits
//
//   uint32_t __l2_cop_off             : 1;
//   uint32_t __l2_cop_on              : 2;
//   uint32_t __l2_descriptor_mode     : 2;
//   uint32_t __l1_inv_dont_allocate   : 1;
//   uint32_t __l2_sector_promote_256B : 1;
//   uint32_t __reserved3              : 1;
// };

#if !_CCCL_CUDA_COMPILER(NVRTC)

[[nodiscard]] _CCCL_HIDE_FROM_ABI uint64_t __block_encoding_host(
  __l2_evict_t __primary, __l2_evict_t __secondary, const void* __ptr, uint32_t __primary_bytes, uint32_t __total_bytes)
{
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
  // NOTE: there is a bug in PTX createpolicy when __block_size_enum == 13. The *incorrect* behavior matches the
  //       following code:
  // auto __block_count        = (__block_size_enum == 13)
  //                            ? ((__block_end - __block_start <= 127u) ? (__block_end - __block_start) : 1)
  //                            : _CUDA_VSTD::clamp(__block_end - __block_start, 1u, 127u);
  auto __block_count        = _CUDA_VSTD::clamp(__block_end - __block_start, 1u, 127u);
  auto __l2_cop_off         = _CUDA_VSTD::to_underlying(__secondary);
  auto __l2_cop_on          = _CUDA_VSTD::to_underlying(__primary);
  auto __l2_descriptor_mode = _CUDA_VSTD::to_underlying(__l2_descriptor_mode_t::_Desc_Block_Type);
  return static_cast<uint64_t>(__block_count) << 37 //
       | static_cast<uint64_t>(__block_start) << 44 //
       | static_cast<uint64_t>(__block_size_enum) << 52 //
       | static_cast<uint64_t>(__l2_cop_off) << 56 //
       | static_cast<uint64_t>(__l2_cop_on) << 57 //
       | static_cast<uint64_t>(__l2_descriptor_mode) << 59;
}

#endif // !_CCCL_CUDA_COMPILER(NVRTC)

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI uint64_t __block_encoding(
  __l2_evict_t __primary, __l2_evict_t __secondary, const void* __ptr, size_t __primary_bytes, size_t __total_bytes)
{
  _CCCL_ASSERT(__primary_bytes <= size_t{0xFFFFFFFF}, "primary size must be less than 4GB");
  _CCCL_ASSERT(__total_bytes <= size_t{0xFFFFFFFF}, "total size must be less than 4GB");
  auto __primary_bytes1 = static_cast<uint32_t>(__primary_bytes);
  auto __total_bytes1   = static_cast<uint32_t>(__total_bytes);
  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
    (return ::cuda::__block_encoding_host(__primary, __secondary, __ptr, __primary_bytes1, __total_bytes1);),
    (return ::cuda::__createpolicy_range(__primary, __secondary, __ptr, __primary_bytes1, __total_bytes1);))
}

/***********************************************************************************************************************
 * Interleaved Descriptor
 **********************************************************************************************************************/

// MemoryDescriptor:interleaveDesc_t reference
//
// struct __interleaved_desc_t // 64 bits
// {
//   uint64_t            : 52;
//   uint32_t __fraction : 4; // 56 bits
//
//   uint32_t __l2_cop_off             : 1;
//   uint32_t __l2_cop_on              : 2;
//   uint32_t __l2_descriptor_mode     : 2;
//   uint32_t __l1_inv_dont_allocate   : 1;
//   uint32_t __l2_sector_promote_256B : 1;
//   uint32_t                          : 1;
// };

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr uint64_t
__l2_interleave(__l2_evict_t __primary, __l2_evict_t __secondary, float __fraction)
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_80, (return ::cuda::__createpolicy_fraction(__primary, __secondary, __fraction);), (return 0;))
  }
  _CCCL_ASSERT(__fraction > 0.0f && __fraction <= 1.0f, "fraction must be between 0.0f and 1.0f");
  _CCCL_ASSERT(__secondary == __l2_evict_t::_L2_Evict_First || __secondary == __l2_evict_t::_L2_Evict_Unchanged,
               "secondary policy must be evict_first or evict_unchanged");
  constexpr auto __epsilon  = _CUDA_VSTD::numeric_limits<float>::epsilon();
  auto __num                = static_cast<uint32_t>((__fraction - __epsilon) * 16.0f); // fraction = num / 16
  auto __l2_cop_off         = _CUDA_VSTD::to_underlying(__secondary);
  auto __l2_cop_on          = _CUDA_VSTD::to_underlying(__primary);
  auto __l2_descriptor_mode = _CUDA_VSTD::to_underlying(__l2_descriptor_mode_t::_Desc_Interleaved);
  return static_cast<uint64_t>(__num) << 52 //
       | static_cast<uint64_t>(__l2_cop_off) << 56 //
       | static_cast<uint64_t>(__l2_cop_on) << 57 //
       | static_cast<uint64_t>(__l2_descriptor_mode) << 59;
}

inline constexpr auto __l2_interleave_normal = uint64_t{0x10F0000000000000};

inline constexpr auto __l2_interleave_streaming = uint64_t{0x12F0000000000000};

inline constexpr auto __l2_interleave_persisting = uint64_t{0x14F0000000000000};

inline constexpr auto __l2_interleave_normal_demote = uint64_t{0x16F0000000000000};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___ANNOTATED_PTR_ACCESS_PROPERTY_ENCODING
