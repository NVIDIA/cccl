/**
 * @brief Example demonstrating exponential function implementation for both native double and fp64emu types
 * 
 * This example implements and compares the exponential function (exp) using:
 * 1. Standard library implementation (std::exp)
 * 2. Template-based implementation for native double-precision
 * 3. Template-based implementation for fp64emu emulated floating-point
 * 
 * The implementation features:
 * - Polynomial approximation with range reduction
 * - Special case handling (NaN, overflow, underflow)
 * - Subnormal number support
 * - High precision constants for ln(2) and 1/ln(2)
 * - Bit manipulation for exponent reconstruction
 * 
 * The example tests various input values:
 * - Zero and small values
 * - Normal range values (positive and negative)
 * - Edge cases near overflow/underflow limits
 * - Special values (NaN, Inf)
 * 
 * Results are compared with relative error calculations and validation against
 * a small epsilon threshold to determine accuracy.
 * 
 * The implementation is CUDA-compatible and can run on both host and device.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdint>

#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)


constexpr double epsilon = 1e-4;

// similar to C++20 std::bit_cast
#if __cplusplus >= 202002L
    #include <bit>
    template<typename T, typename R>
    __FPEMU_DEVICE_DECL__ T bit_cast(const R value) {
        return std::bit_cast<T>(value);
    }
#else
    // Custom implementation for C++17 and earlier
    template<typename T, typename R>
    __FPEMU_DEVICE_DECL__ T bit_cast(const R value) {
        T dst;
        // memcpy implementation
        std::memcpy(static_cast<void*>(&dst), static_cast<const void*>(&value), sizeof(T));
        return dst;
    }
#endif

#if __CUDACC__
    #define __EXP_TARGET__ __device__
    #define __RUN_TARGET__ __global__
    #define RUN_TEST run_test<<<1,1>>>
    #define DEVICE_SYNC() cudaDeviceSynchronize()
#else
    #define __EXP_TARGET__
    #define __RUN_TARGET__
    #define RUN_TEST run_test
    #define DEVICE_SYNC()
#endif

/**
 * @brief Template function implementing exponential function for both double and fp64emu types
 * 
 * This function implements the exponential function using a polynomial approximation
 * with range reduction. It works for both native double-precision and fp64emu types.
 * 
 * @tparam T The floating-point type (double or fp64emu)
 * @param x The input value
 * @return T The result of exp(x)
 */
 #if defined __EXP_KERNELS__
 // Version wrriten using fp64emu kernels
static inline __EXP_TARGET__ fpbits64_t exp_efp_kernel(fpbits64_t x)
 {
    fpbits64_t ln2_hi  = __nv_fp64emu_from_double(0x1.62e42fefa39efp-1);
    fpbits64_t ln2_lo  = __nv_fp64emu_from_double(0x1.abc9e3b39803fp-34);
    fpbits64_t inv_ln2 = __nv_fp64emu_from_double(0x1.71547652b82fep+0);
    fpbits64_t ovf     = __nv_fp64emu_from_double(709.782712893384);
    fpbits64_t udf     = __nv_fp64emu_from_double(-745.1332191019411);
    fpbits64_t scale   = __nv_fp64emu_from_double(0x1.0p-52);
    fpbits64_t pinf;  pinf  = 0x7ff0000000000000;
    fpbits64_t pzero; pzero = 0x0000000000000000;

    if (__nv_fp64emu_cmp_ne(x, x))   return x;     // NaN
    if (__nv_fp64emu_cmp_gt(x, ovf)) return pinf;  // Overflow
    if (__nv_fp64emu_cmp_lt(x, udf)) return pzero; // Underflow

    uint64_t xsign = x & 0x8000000000000000;

    fpbits64_t r = __nv_fp64emu_from_double(0.5);
    r |= xsign;

    r = __nv_fp64emu_dadd_rn( __nv_fp64emu_dmul_rn(x, inv_ln2), r);
    int64_t k = __nv_fp64emu_to_int_rz(r);
    fpbits64_t kf = __nv_fp64emu_from_int(k);

    r = __nv_fp64emu_dsub_rn(x, __nv_fp64emu_dmul_rn(kf, ln2_hi));
    r = __nv_fp64emu_dsub_rn(r, __nv_fp64emu_dmul_rn(kf, ln2_lo));

    fpbits64_t poly = __nv_fp64emu_from_double(0x1.ae64567f544e4p-13);
    poly = __nv_fp64emu_fma_rn(poly, r, __nv_fp64emu_from_double(0x1.71de3a556c734p-11));
    poly = __nv_fp64emu_fma_rn(poly, r, __nv_fp64emu_from_double(0x1.a01a01a01a01ap-9));
    poly = __nv_fp64emu_fma_rn(poly, r, __nv_fp64emu_from_double(0x1.6c16c16c16c17p-7));
    poly = __nv_fp64emu_fma_rn(poly, r, __nv_fp64emu_from_double(0x1.999999999999ap-5));
    poly = __nv_fp64emu_fma_rn(poly, r, __nv_fp64emu_from_double(0x1.5555555555555p-3));
    poly = __nv_fp64emu_fma_rn(poly, r, __nv_fp64emu_from_double(0x1p-1));
    poly = __nv_fp64emu_fma_rn(poly, r, __nv_fp64emu_from_double(0x1p+0));
    poly = __nv_fp64emu_fma_rn(poly, r, __nv_fp64emu_from_double(0x1p+0));
    
    int exponent = k + 1023;
    if (exponent <= 0) {
        if (exponent < -52) return pzero;
        uint64_t uexp = (uint64_t)(exponent + 52) << 52;
        fpbits64_t dexp = bit_cast<fpbits64_t>(uexp);
        dexp = __nv_fp64emu_dmul_rn(poly, dexp);
        dexp = __nv_fp64emu_dmul_rn(dexp, scale);
        return dexp;
    }

    if (exponent >= 2047) return pinf;

    uint64_t uexp = (uint64_t)exponent << 52;
    fpbits64_t dexp = bit_cast<fpbits64_t>(uexp);
    return __nv_fp64emu_dmul_rn(poly, dexp);
 }
 #endif

// Version wrriten using C/C++ double precision
template<typename T>
__EXP_TARGET__ T exp_impl(T x)
{
    #define LN2_HI   0x1.62e42fefa39efp-1 // High part of ln(2)
    #define LN2_LO   0x1.abc9e3b39803fp-34 // Low part for extra precision
    #define INV_LN2  0x1.71547652b82fep+0  // 1 / ln(2)    

    // Handle special cases
    if (x != x) return x;  // NaN
    if (x > 709.782712893384) return T(1.0) / 0.0;  // Overflow
    if (x < -745.1332191019411) return 0.0;  // Underflow

    // Range reduction: x = k * ln2 + r,  |r| <= ln2/2
    int k = (int)(x * INV_LN2 + (x >= 0 ? 0.5 : -0.5));

#if defined __CUDACC__
    T r   = __fma_rn(-k, LN2_HI, x);
    r     = __fma_rn(-k, LN2_LO, r);
#else
    T r   = fma(-k, LN2_HI, x);
    r     = fma(-k, LN2_LO, r);
#endif

    // Polynomial approximation of exp(r), r in [-ln2/2, ln2/2]
    T poly = 0x1p+0+r*(
               0x1p+0+r*(
                 0x1p-1+r*(
                   0x1.5555555555555p-3+r*(
                     0x1.999999999999ap-5+r*(
                       0x1.6c16c16c16c17p-7+r*(
                         0x1.a01a01a01a01ap-9))))));
            
    // Reconstruct exp(x) = 2^k * exp(r)
    int exponent = k + 1023;  // Bias = 1023 for double
    if (exponent <= 0) {  // Subnormal
        if (exponent < -52) return 0.0;
        uint64_t uexp = (uint64_t)(exponent + 52) << 52;
        T dexp = bit_cast<T>(uexp);
        return poly * dexp * 0x1.0p-52;
    }

    if (exponent >= 2047) return T(1.0) / 0.0;

    uint64_t uexp = (uint64_t)exponent << 52;
    T dexp = bit_cast<T>(uexp);
    return poly * dexp;
}

__RUN_TARGET__ void run_test (double* inputs, 
                              double* results_std, 
                              double* results_native, 
                              double* results_fp64emu, int n) 
{
    for (int i = 0; i < n; i++) 
    {
        results_std[i]     = exp (inputs[i]);
        results_native[i]  = exp_impl<double> (inputs[i]);
        // Use fp64emu kernels if enabled
        #if defined __KERNELS__
            results_fp64emu[i]   = __nv_fp64emu_to_double(
                 exp_efp_kernel(__nv_fp64emu_from_double(inputs[i])));
        #else
            results_fp64emu[i]   = exp_impl<fp64emu> (inputs[i]);
        #endif
    }
}

int main() 
{
    const int num_tests = 10;
    
    // Test values
    double test_values[num_tests] = 
    {
        0.0,        // exp(0) = 1
        0.00001,    // very small positive value
        1.0,        // exp(1) = e
        -1.0,       // exp(-1) = 1/e
        0.5,        // exp(0.5)
        -0.5,       // exp(-0.5)
        10.0,       // exp(10) - larger value
        -10.0,      // exp(-10) - smaller value
        700.0,      // near overflow limit
        -700.0,     // near underflow limit
    };
    
    double* inputs;
    double* results_std;      // Standard exp results
    double* results_native;   // Our template exp results for double
    double* results_fp64emu;    // Our template exp results for fp64emu
    
#if defined __FPEMU_DEVICE__
    // Allocate device memory
    cudaMallocManaged(&inputs, num_tests * sizeof(double));
    cudaMallocManaged(&results_std, num_tests * sizeof(double));
    cudaMallocManaged(&results_native, num_tests * sizeof(double));
    cudaMallocManaged(&results_fp64emu, num_tests * sizeof(double));
    // Copy test values to device memory
    for (int i = 0; i < num_tests; i++) 
    {
        inputs[i] = test_values[i];
    }
#else
    // Use stack memory for host-only execution
    inputs         = test_values;
    results_std    = new double[num_tests];
    results_native = new double[num_tests];
    results_fp64emu  = new double[num_tests];
#endif
    
    // Run the test
    RUN_TEST(inputs, results_std, results_native, results_fp64emu, num_tests);
    DEVICE_SYNC();

    // Print results
    printf("  ** This example demonstrates how to use fp64emu library to implement exp function\n\n");
    for (int i = 0; i < num_tests; i++) 
    {
        double err_native = (results_std[i] != 0.0)?fabs(results_native[i] - results_std[i]) / results_std[i]:0.0;
        std::string err_native_str = err_native < epsilon ? " OK " : "BAD ";

        double err_fp64emu  = (results_std[i] != 0.0)?fabs(results_fp64emu[i]  - results_std[i]) / results_std[i]:0.0;
        std::string err_fp64emu_str = err_fp64emu < epsilon ? " OK " : "BAD ";

        printf("exp(%6.2g) = %9.3g (std), = %9.3g (native, err=%-7.2g [%s]), = %9.3g (fp64emu, err=%-7.2g [%s])\n", 
               inputs[i], results_std[i], 
               results_native[i], err_native, err_native_str.c_str(), 
               results_fp64emu[i],  err_fp64emu,  err_fp64emu_str.c_str());  
    }
    // Basic verification: count failures
    int errors = 0;
    for (int i = 0; i < num_tests; i++) {
        double err_fp64emu = (results_std[i] != 0.0) ? fabs(results_fp64emu[i] - results_std[i]) / results_std[i] : 0.0;
        if (err_fp64emu >= epsilon) errors++;
    }
    printf("fpemu_exp: %s (%d errors)\n\n", errors ? "FAIL" : "PASS", errors);
    
#if defined __FPEMU_DEVICE__
    // Free device memory
    cudaFree(inputs);
    cudaFree(results_std);
    cudaFree(results_native);
    cudaFree(results_fp64emu);
#else
    // Free host memory
    delete[] results_std;
    delete[] results_native;
    delete[] results_fp64emu;
#endif
    
    return errors ? 1 : 0;
} 