// This file is intended to be force included by the compiler (using -include=tbb_preinclude.h). It's purpose is to fix
// unrecognized builtins by cudafe++ that are used in <immintrin.h>. We simply forward declare them in global namespace.

#if defined(_IMMINTRIN_H_INCLUDED)
#  error "This file must be included before <immintrin.h>"
#endif // _IMMINTRIN_H_INCLUDED

#if defined(__NVCC__) && defined(__CUDACC__)
// Forward declare builtins used by gcc 12. Clang and nvc++ define __GNUC__, too, so we need to explicitly leave them
// out.
#  if defined(__GNUC__) && !defined(__clang__) && !defined(__NVCOMPILER)
#    if __CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ == 0 && __GNUC__ == 12
void __builtin_ia32_ldtilecfg(const void*);
void __builtin_ia32_sttilecfg(void*);
#    endif // __CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ == 0 && __GNUC__ == 12
#  endif // __GNUC__ && !__clang__ && !__NVCOMPILER

// cudafe++ has problems with many builtins used in <avx512fp16intrin.h> and <avx512vlfp16intrin.h> when compiling with
// nvc++ as the host compiler. Since those headers are not used by thrust nor tbb, we can prevent their inclusion by
// defining their include guard macros.
#  if defined(__NVCOMPILER)
#    define __AVX512FP16INTRIN_H
#    define __AVX512VLFP16INTRIN_H
#  endif // __NVCOMPILER
#endif // __NVCC__ && __CUDACC__
