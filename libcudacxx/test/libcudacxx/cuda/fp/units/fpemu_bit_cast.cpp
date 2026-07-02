#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cinttypes>
#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if __FPEMU_UNPACKED__ == 1

/*
 * @brief Example demonstrating the usage of bit_cast for unpacked floating-point types
 * 
 * This example shows how to use the bit_cast function to get the IEEE-754 
 * double precision bit representation of an unpacked floating-point value.
 * 
 * The workflow:
 * 1. Create unpacked floating-point values from doubles
 * 2. Perform operations on unpacked values
 * 3. Use bit_cast to get the IEEE-754 bit representation
 * 4. Compare with expected bit patterns
 * 
 * This is useful for:
 * - Debugging floating-point operations
 * - Verifying bit-exact results
 * - Serialization and storage of floating-point values
 * - Low-level floating-point analysis
 */

//  Define the macros for the host and device runs
#if __CUDACC__
    #define TARGET_DEVICE __global__
    #define HOST_DEVICE __host__ __device__
#else
    #define TARGET_DEVICE
    #define HOST_DEVICE
#endif

HOST_DEVICE void print_bits(fpbits64_t bits, const char* label)
{
    printf("%s: 0x%016" PRIx64 "\n", label, bits);
}

TARGET_DEVICE void run_bit_cast_example()
{
    printf("=== bit_cast Example for Unpacked Types ===\n\n");

    // Example 1: Simple value - bit_cast to uint64_t
    {
        printf("Example 1: Simple value 1.5 - bit_cast to uint64_t\n");
        double val = 1.5;
        
        // Create unpacked value
        fp64emu_unpacked_t<fp64emu_accuracy::def> x_unpacked(val);
        
        // Get IEEE-754 bit representation using C++20-style bit_cast
        uint64_t x_bits = bit_cast<uint64_t>(x_unpacked);
        
        print_bits(x_bits, "  Bits of 1.5");
        
        // Also can bit_cast directly to double
        double val_back = bit_cast<double>(x_unpacked);
        printf("  Original: %.17g\n", val);
        printf("  bit_cast<double>: %.17g\n", val_back);
        printf("  Match: %s\n\n", (val == val_back) ? "YES" : "NO");
    }

    // Example 2: Result of arithmetic operation
    {
        printf("Example 2: Arithmetic operation (2.0 * 3.0 + 1.0)\n");
        
        fp64emu_unpacked_t<fp64emu_accuracy::def> x(2.0);
        fp64emu_unpacked_t<fp64emu_accuracy::def> y(3.0);
        fp64emu_unpacked_t<fp64emu_accuracy::def> z(1.0);
        
        // Perform operation
        fp64emu_unpacked_t<fp64emu_accuracy::def> result = x * y + z;
        
        // Get bit representation using C++20-style bit_cast
        uint64_t result_bits = bit_cast<uint64_t>(result);
        
        print_bits(result_bits, "  Bits of result");
        
        double result_double = bit_cast<double>(result);
        printf("  Result value: %.17g\n", result_double);
        printf("  Expected: 7.0\n\n");
    }

    // Example 3: Special values
    {
        printf("Example 3: Special values\n");
        
        // Zero
        fp64emu_unpacked_t<fp64emu_accuracy::def> zero(0.0);
        uint64_t zero_bits = bit_cast<uint64_t>(zero);
        print_bits(zero_bits, "  Bits of +0.0");
        
        // Negative zero
        fp64emu_unpacked_t<fp64emu_accuracy::def> neg_zero(-0.0);
        uint64_t neg_zero_bits = bit_cast<uint64_t>(neg_zero);
        print_bits(neg_zero_bits, "  Bits of -0.0");
        
        // One
        fp64emu_unpacked_t<fp64emu_accuracy::def> one(1.0);
        uint64_t one_bits = bit_cast<uint64_t>(one);
        print_bits(one_bits, "  Bits of 1.0");
        
        // Negative one
        fp64emu_unpacked_t<fp64emu_accuracy::def> neg_one(-1.0);
        uint64_t neg_one_bits = bit_cast<uint64_t>(neg_one);
        print_bits(neg_one_bits, "  Bits of -1.0");
        
        printf("\n");
    }

    // Example 4: Different accuracy levels
    {
        printf("Example 4: Same value with different accuracy levels\n");
        double val = 3.14159265358979323846;
        
        // Default accuracy (== high: correct rounding, full range)
        fp64emu_unpacked_t<fp64emu_accuracy::def> def_val(val);
        uint64_t def_bits = bit_cast<uint64_t>(def_val);
        print_bits(def_bits, "  def ");
        
        // High accuracy (correct rounding, full range)
        fp64emu_unpacked_t<fp64emu_accuracy::high> accurate_val(val);
        uint64_t accurate_bits = bit_cast<uint64_t>(accurate_val);
        print_bits(accurate_bits, "  high");
        
        // Low accuracy (normal range)
        fp64emu_unpacked_t<fp64emu_accuracy::low> fast_val(val);
        uint64_t fast_bits = bit_cast<uint64_t>(fast_val);
        print_bits(fast_bits, "  low ");
        
        // They should be the same for this simple case
        printf("  All accuracy levels match: %s\n\n", (def_bits == accurate_bits && def_bits == fast_bits) ? "YES" : "NO");
    }

    // Example 5: Bit manipulation and reconstruction
    {
        printf("Example 5: Bit manipulation\n");
        double val = 42.0;
        
        fp64emu_unpacked_t<fp64emu_accuracy::def> x(val);
        uint64_t bits = bit_cast<uint64_t>(x);
        
        print_bits(bits, "  Original bits");
        
        // Extract sign bit (bit 63)
        uint64_t sign = (bits >> 63) & 1;
        // Extract exponent (bits 62-52)
        uint64_t exponent = (bits >> 52) & 0x7FF;
        // Extract mantissa (bits 51-0)
        uint64_t mantissa = bits & 0xFFFFFFFFFFFFFull;
        
        printf("  Sign: %llu\n", (unsigned long long)sign);
        printf("  Exponent: 0x%03llx (%lld)\n", (unsigned long long)exponent, (long long)exponent);
        printf("  Mantissa: 0x%013llx\n\n", (unsigned long long)mantissa);
    }

    printf("=== bit_cast Example Complete ===\n");
}

// Verification helper: returns number of errors
int verify_bit_cast()
{
    int errors = 0;

    // Check round-trip: double -> unpacked -> bit_cast<double> should preserve value
    double test_vals[] = {1.5, -2.0, 0.0, 42.0, 3.14159265358979323846};
    for (int i = 0; i < 5; i++) {
        fp64emu_unpacked_t<fp64emu_accuracy::def> x(test_vals[i]);
        double back = bit_cast<double>(x);
        if (back != test_vals[i]) {
            printf("FAIL: bit_cast round-trip for %.17g (got %.17g)\n", test_vals[i], back);
            errors++;
        }
    }

    // Check arithmetic result: 2*3+1 = 7
    fp64emu_unpacked_t<fp64emu_accuracy::def> a(2.0), b(3.0), c(1.0);
    double r = bit_cast<double>(a * b + c);
    if (fabs(r - 7.0) > 1e-10) {
        printf("FAIL: 2*3+1 = %.17g (expected 7.0)\n", r);
        errors++;
    }

    // Check accuracy levels give same result for simple conversion
    double pi = 3.14159265358979323846;
    uint64_t b1 = bit_cast<uint64_t>(fp64emu_unpacked_t<fp64emu_accuracy::def>(pi));
    uint64_t b2 = bit_cast<uint64_t>(fp64emu_unpacked_t<fp64emu_accuracy::high>(pi));
    uint64_t b3 = bit_cast<uint64_t>(fp64emu_unpacked_t<fp64emu_accuracy::low>(pi));
    if (b1 != b2 || b1 != b3) {
        printf("FAIL: accuracy levels give different bits for pi\n");
        errors++;
    }

    return errors;
}

int main()
{
#if __CUDACC__
    run_bit_cast_example<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        return EXIT_FAILURE;
    }
#else
    run_bit_cast_example();
#endif

    int errors = verify_bit_cast();
    printf("fpemu_bit_cast: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
}

#else // __FPEMU_UNPACKED__ != 1

int main()
{
    printf("This example requires __FPEMU_UNPACKED__ to be enabled.\n");
    printf("Please compile with -D__FPEMU_UNPACKED__=1\n");
    return 0;
}

#endif // __FPEMU_UNPACKED__ == 1

