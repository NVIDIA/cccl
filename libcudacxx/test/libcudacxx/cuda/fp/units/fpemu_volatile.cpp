/*
    fpemu_volatile.cpp - Unit Test for Volatile Constructors/Copies and Trivial Copyability
    ======================================================================================================
    Author:  Andrei Kolesov
    Date:    2026

    This test verifies that fp64emu_t and fp64emu_unpacked_t types:
    1. Are trivially copyable (required for cooperative_groups, __shfl intrinsics, etc.)
    2. Correctly support volatile construction (copy from volatile object)
    3. Correctly support volatile assignment (assign to/from volatile object)
    4. Preserve values through volatile round-trips on both host and device
*/

#include <cmath>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <cxxabi.h>
#include <cstdlib>

#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Demangle GCC/Clang typeid names into human-readable form
static std::string demangle(const char* mangled) {
    int status = 0;
    char* demangled = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    std::string result = (status == 0 && demangled) ? demangled : mangled;
    std::free(demangled);
    return result;
}

// ============================================================
// Compile-time checks: trivially copyable
// ============================================================
static_assert(std::is_trivially_copyable<fp64emu>::value,
              "fp64emu must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64emu_low>::value,
              "fp64emu_low must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64emu_high>::value,
              "fp64emu_high must be trivially copyable");

#if __FPEMU_UNPACKED__ == 1
static_assert(std::is_trivially_copyable<fp64emu_unpacked>::value,
              "fp64emu_unpacked must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64emu_unpacked_low>::value,
              "fp64emu_unpacked_low must be trivially copyable");
static_assert(std::is_trivially_copyable<fp64emu_unpacked_high>::value,
              "fp64emu_unpacked_high must be trivially copyable");
#endif

// ============================================================
// Host-side volatile tests
// ============================================================

template <typename emu_type>
bool test_volatile_host() {
    const double test_val     = 3.141592653589793;
    const double test_val2    = 2.718281828459045;
    const double tolerance    = 1e-12;
    bool         all_passed   = true;

    std::cout << "\n  --- Host volatile tests for " << demangle(typeid(emu_type).name())
              << " (size=" << sizeof(emu_type) << ") ---" << std::endl;

    // Test 1: Construct from volatile
    {
        volatile emu_type vol;
        // Assign through volatile assignment operator
        const emu_type tmp(test_val);
        vol = tmp;
        // Construct from volatile
        emu_type non_vol(vol);
        double   result = static_cast<double>(non_vol);
        double   error  = std::abs(result - test_val);
        bool     passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    Construct from volatile:   "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    // Test 2: Assign to volatile
    {
        emu_type          src(test_val);
        volatile emu_type vol;
        vol = src;
        // Read back via construct-from-volatile
        emu_type readback(vol);
        double   result = static_cast<double>(readback);
        double   error  = std::abs(result - test_val);
        bool     passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    Assign to volatile:        "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    // Test 3: Assign from volatile
    {
        volatile emu_type vol;
        const emu_type    tmp(test_val2);
        vol = tmp;
        emu_type dst;
        dst = vol;
        double result = static_cast<double>(dst);
        double error  = std::abs(result - test_val2);
        bool   passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    Assign from volatile:      "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    // Test 4: Volatile round-trip preserves bits
    {
        emu_type          src(test_val);
        volatile emu_type vol;
        vol = src;
        emu_type dst(vol);
        bool     bits_match = (src.bits == dst.bits);
        bool     passed     = bits_match;
        all_passed &= passed;
        std::cout << "    Volatile round-trip bits:   "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (bits " << (bits_match ? "OK" : "MISMATCH") << ")" << std::endl;
    }

    return all_passed;
}

#if __FPEMU_UNPACKED__ == 1
template <typename emu_type>
bool test_volatile_host_unpacked() {
    const double test_val     = 3.141592653589793;
    const double test_val2    = 2.718281828459045;
    const double tolerance    = 1e-12;
    bool         all_passed   = true;

    std::cout << "\n  --- Host volatile tests for " << demangle(typeid(emu_type).name())
              << " (size=" << sizeof(emu_type) << ") ---" << std::endl;

    // Test 1: Construct from volatile
    {
        volatile emu_type vol;
        const emu_type tmp(test_val);
        vol = tmp;
        emu_type non_vol(vol);
        double   result = static_cast<double>(non_vol);
        double   error  = std::abs(result - test_val);
        bool     passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    Construct from volatile:   "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    // Test 2: Assign to volatile
    {
        emu_type          src(test_val);
        volatile emu_type vol;
        vol = src;
        emu_type readback(vol);
        double   result = static_cast<double>(readback);
        double   error  = std::abs(result - test_val);
        bool     passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    Assign to volatile:        "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    // Test 3: Assign from volatile
    {
        volatile emu_type vol;
        const emu_type    tmp(test_val2);
        vol = tmp;
        emu_type dst;
        dst = vol;
        double result = static_cast<double>(dst);
        double error  = std::abs(result - test_val2);
        bool   passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    Assign from volatile:      "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    // Test 4: Volatile round-trip preserves unpacked fields
    {
        emu_type          src(test_val);
        volatile emu_type vol;
        vol = src;
        emu_type dst(vol);
        bool sign_match     = (src.bits.sign     == dst.bits.sign);
        bool exponent_match = (src.bits.exponent  == dst.bits.exponent);
        bool mantissa_match = (src.bits.mantissa  == dst.bits.mantissa);
        bool passed         = sign_match && exponent_match && mantissa_match;
        all_passed &= passed;
        std::cout << "    Volatile round-trip bits:   "
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (sign " << (sign_match ? "OK" : "MISMATCH")
                  << ", exp " << (exponent_match ? "OK" : "MISMATCH")
                  << ", man " << (mantissa_match ? "OK" : "MISMATCH") << ")" << std::endl;
    }

    return all_passed;
}
#endif // __FPEMU_UNPACKED__

// ============================================================
// Device-side volatile tests (CUDA)
// ============================================================
#if defined(__CUDACC__)

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = call;                                                          \
        if (err != cudaSuccess) {                                                        \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "         \
                      << cudaGetErrorString(err) << std::endl;                           \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

// Kernel: construct from volatile, assign to volatile, assign from volatile
template <typename emu_type>
__global__ void test_volatile_kernel(const emu_type input, emu_type *results) {
    // results[0] = construct from volatile
    // results[1] = assign to volatile then read back
    // results[2] = assign from volatile

    // Use raw aligned storage for __shared__ to avoid dynamic initialization warning
    __shared__ alignas(alignof(emu_type)) unsigned char shared_buf[sizeof(emu_type)];
    volatile emu_type& shared_vol = *reinterpret_cast<volatile emu_type*>(shared_buf);

    if (threadIdx.x == 0) {
        // Write to volatile via assignment
        shared_vol = input;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // Test 1: Construct from volatile
        emu_type from_vol(shared_vol);
        results[0] = from_vol;

        // Test 2: Assign to volatile then read back
        alignas(alignof(emu_type)) unsigned char local_buf[sizeof(emu_type)];
        volatile emu_type& local_vol = *reinterpret_cast<volatile emu_type*>(local_buf);
        local_vol = input;
        emu_type readback(local_vol);
        results[1] = readback;

        // Test 3: Assign from volatile
        emu_type assigned;
        assigned   = shared_vol;
        results[2] = assigned;
    }
} // test_volatile_kernel

template <typename emu_type>
bool test_volatile_device() {
    const double test_val  = 3.141592653589793;
    const double tolerance = 1e-12;
    bool         all_passed = true;

    std::cout << "\n  --- Device volatile tests for " << demangle(typeid(emu_type).name())
              << " (size=" << sizeof(emu_type) << ") ---" << std::endl;

    constexpr int num_results = 3;
    emu_type     *d_results;
    emu_type      h_results[num_results];
    emu_type      h_input(test_val);

    CUDA_CHECK(cudaMalloc(&d_results, num_results * sizeof(emu_type)));
    CUDA_CHECK(cudaMemset(d_results, 0, num_results * sizeof(emu_type)));

    test_volatile_kernel<emu_type><<<1, 32>>>(h_input, d_results);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_results, d_results, num_results * sizeof(emu_type),
                          cudaMemcpyDeviceToHost));

    const char *test_names[] = {
        "Construct from volatile (device): ",
        "Assign to volatile (device):      ",
        "Assign from volatile (device):    ",
    };

    for (int i = 0; i < num_results; ++i) {
        double result = static_cast<double>(h_results[i]);
        double error  = std::abs(result - test_val);
        bool   passed = error < tolerance;
        all_passed &= passed;
        std::cout << "    " << test_names[i]
                  << (passed ? "+ PASSED" : "- FAILED")
                  << "  (error=" << std::scientific << std::setprecision(4) << error << ")" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_results));
    return all_passed;
}

#endif // __CUDACC__

// ============================================================
// Main
// ============================================================
int main() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Volatile and Trivial Copyability Tests for fp64emu_t" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

#if defined(__CUDACC__)
    // Get device properties
    int            device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "\nDevice Information:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
#endif

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "COMPILE-TIME: trivially copyable static_assert checks passed" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    bool all_passed = true;

    // Host tests - packed types
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "HOST VOLATILE TESTS (packed)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    all_passed &= test_volatile_host<fp64emu>();
    all_passed &= test_volatile_host<fp64emu_low>();
    all_passed &= test_volatile_host<fp64emu_high>();

#if __FPEMU_UNPACKED__ == 1
    // Host tests - unpacked types
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "HOST VOLATILE TESTS (unpacked)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    all_passed &= test_volatile_host_unpacked<fp64emu_unpacked>();
    all_passed &= test_volatile_host_unpacked<fp64emu_unpacked_low>();
    all_passed &= test_volatile_host_unpacked<fp64emu_unpacked_high>();
#endif

#if defined(__CUDACC__)
    // Device tests - packed types
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "DEVICE VOLATILE TESTS (packed)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    all_passed &= test_volatile_device<fp64emu>();
    all_passed &= test_volatile_device<fp64emu_low>();
    all_passed &= test_volatile_device<fp64emu_high>();

#if __FPEMU_UNPACKED__ == 1
    // Device tests - unpacked types
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "DEVICE VOLATILE TESTS (unpacked)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    all_passed &= test_volatile_device<fp64emu_unpacked>();
    all_passed &= test_volatile_device<fp64emu_unpacked_low>();
    all_passed &= test_volatile_device<fp64emu_unpacked_high>();
#endif
#endif // __CUDACC__

    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (all_passed) { std::cout << "+ ALL TESTS PASSED"  << std::endl; }
    else            { std::cout << "- SOME TESTS FAILED" << std::endl; }
    std::cout << std::string(60, '=') << std::endl << std::endl;

    return all_passed ? 0 : 1;
} // main
