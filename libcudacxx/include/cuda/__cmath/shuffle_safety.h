
#ifndef _CUDA_FUNCTIONAL_SHUFFLE_SAFETY_H
#define _CUDA_FUNCTIONAL_SHUFFLE_SAFETY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/type_traits>
#include <cuda/std/bit>
#include <cuda/std/memory>

#include <cuda_fp16.h>

#include <cuda_bf16.h>


#define _CCCL_HAS_CUDA_COMPILER 1 //fix for now -- to be deleted later

#if _CCCL_HAS_CUDA_COMPILER
_LIBCUDACXX_BEGIN_NAMESPACE_CUDA
    template <typename T>
    constexpr bool is_supported_type_v = false;
    template <> constexpr bool is_supported_type_v<int> = true;
    template <> constexpr bool is_supported_type_v<unsigned int> = true;
    template <> constexpr bool is_supported_type_v<long> = true;
    template <> constexpr bool is_supported_type_v<unsigned long> = true;
    template <> constexpr bool is_supported_type_v<long long> = true;
    template <> constexpr bool is_supported_type_v<unsigned long long> = true;
    template <> constexpr bool is_supported_type_v<float> = true;
    template <> constexpr bool is_supported_type_v<double> = true;
    template <> constexpr bool is_supported_type_v<__half> = true;
    template <> constexpr bool is_supported_type_v<__half2> = true;
    template <> constexpr bool is_supported_type_v<__nv_bfloat16> = true;
    template <> constexpr bool is_supported_type_v<__nv_bfloat162> = true;

    template <typename T>
    T shfl(T var, int srcLane, unsigned mask = 0xFFFFFFFF, int width = warpSize)
    {
        _CCCL_ASSERT(is_supported_type_v<T>, "T must be a supported type for warp shuffle operations"); // T must be a supported type for warp shuffle operations
        _CCCL_ASSERT((width > 0 && (width & (width - 1)) == 0), "Width must be a power of two"); // width must be a power of two
        if constexpr(sizeof(T)==4)
        {
            if constexpr(cuda::std::is_same_v<T, __half2>)//check for __half2
            {
                __half part_arr[2];
                cuda::std::memcpy(part_arr, &var, sizeof(var));

                float h2f_one = __half2float(part_arr[0]);
                float h2f_two = __half2float(part_arr[1]);

                float result_one = __shfl_sync(mask, h2f_one, srcLane, width);
                float result_two = __shfl_sync(mask, h2f_two, srcLane, width);

                __half f2h_one = __float2half(result_one);
                __half f2h_two = __float2half(result_two);

                __half2 result = __halves2half2(f2h_one, f2h_two);
            }
            else if(cuda::std::is_same_v<T, __nv_bfloat162>)//check for __nv_bfloat162
            {
                __nv_bfloat16 part_arr[2];
                cuda::std::memcpy(part_arr, &var, sizeof(var));

                float b2f_one = __nv_bfloat162float(part_arr[0]);
                float b2f_two = __nv_bfloat162float(part_arr[1]);

                float result_one = __shfl_sync(mask, b2f_one, srcLane, width);
                float result_two = __shfl_sync(mask, b2f_two, srcLane, width);

                __nv_bfloat16 f2b_one = __float2nv_bfloat162(result_one);
                __nv_bfloat16 f2b_two = __float2nv_bfloat162(result_two);

                __nv_bfloat162 result = __nv_bfloat162(f2b_one, f2b_two);
            }
            else if constexpr(cuda::std::is_same_v<T, int> || cuda::std::is_same_v<T, unsigned int> || cuda::std::is_same_v<T, long> 
                || cuda::std::is_same_v<T, unsigned long>)
            {
                int var_int = cuda::std::bit_cast<int>(var);
                int result = __shfl_sync(mask, var_int, srcLane, width);
                T result_t = cuda::std::bit_cast<T>(result);
                return result_t;
            }
        }
    }
_LIBCUDACXX_END_NAMESPACE_CUDA
#endif // _CCCL_HAS_CUDA_COMPILER

#endif // _CUDA_FUNCTIONAL_SHUFFLE_SAFETY_H