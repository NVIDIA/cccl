# cuFile C++ Utils Unit Tests

This directory contains comprehensive unit tests for the cuFile C++ utility functions using Google Test (gtest).

## Overview

The tests cover all utility functions in `cuda/io/utils.hpp`:

- **Memory Type Detection**: `is_gpu_memory()`, `is_cufile_compatible()`
- **Device Management**: `get_device_id()`
- **Alignment Functions**: `is_aligned()`, `align_up()`, `get_optimal_alignment()`
- **Integration Tests**: Combined functionality testing

## Test Structure

### Files

- `CMakeLists.txt` - CMake configuration for building tests
- `test_utils.h` - Header for test utility functions
- `test_utils.cpp` - Implementation of test utilities
- `test_utils_functions.cpp` - Main test file with all unit tests
- `README.md` - This file

### Test Categories

1. **Alignment Tests** (`UtilsAlignmentTest`)
   - Tests for `is_aligned()` function
   - Tests for `align_up()` function
   - Tests for `get_optimal_alignment()` function

2. **CUDA Memory Tests** (`UtilsTest`)
   - Tests for `is_gpu_memory()` function
   - Tests for `get_device_id()` function
   - Tests for `is_cufile_compatible()` function

3. **Corner Cases** (`UtilsCornerCasesTest`)
   - Null pointer handling
   - Edge cases for alignment functions

4. **Integration Tests**
   - Combined functionality testing
   - Typical usage workflows

## Building and Running

### Prerequisites

1. **CMake** (version 3.18 or higher)
2. **Google Test** (gtest)
3. **CUDA Toolkit** (for CUDA-dependent tests)
4. **C++17 compatible compiler**

### Building

From the project root directory:

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake .. -DBUILD_CUFILE_CPP_TESTS=ON

# Build the tests
make -j$(nproc)

# Or build just the test targets
make test_utils_functions
```

### Running Tests

```bash
# Run all tests
ctest --verbose

# Or run the test executable directly
./tests/test_utils_functions

# Run with gtest filters
./tests/test_utils_functions --gtest_filter="*Alignment*"

# Run with detailed output
./tests/test_utils_functions --gtest_output=xml:test_results.xml
```

### Using the Custom Test Target

```bash
# Run tests using the custom target
make run_tests
```

## Test Behavior

### CUDA Availability

The tests automatically detect whether CUDA is available:

- **CUDA Available**: Runs full test suite including GPU memory tests
- **CUDA Not Available**: Skips CUDA-dependent tests, runs alignment tests only

### Test Output

When CUDA is available:
```
CUDA is available - running full test suite
[==========] Running 12 tests from 4 test fixtures.
[----------] 6 tests from UtilsAlignmentTest
[ RUN      ] UtilsAlignmentTest.IsAligned
[       OK ] UtilsAlignmentTest.IsAligned (0 ms)
...
```

When CUDA is not available:
```
CUDA is not available - running limited test suite
[==========] Running 6 tests from 2 test fixtures.
[----------] 6 tests from UtilsAlignmentTest
[ RUN      ] UtilsAlignmentTest.IsAligned
[       OK ] UtilsAlignmentTest.IsAligned (0 ms)
...
```

## Memory Management

The tests use RAII classes for safe memory management:

- `GPUMemoryRAII` - Automatic GPU memory allocation/deallocation
- `HostMemoryRAII` - Automatic pinned host memory allocation/deallocation
- `RegularMemoryRAII` - Automatic regular memory allocation/deallocation

This ensures no memory leaks even if tests fail or throw exceptions.

## Adding New Tests

To add tests for new utility functions:

1. **Add test utility functions** to `test_utils.h/.cpp` if needed
2. **Create test cases** in `test_utils_functions.cpp`
3. **Update CMakeLists.txt** if additional dependencies are required
4. **Update this README** to document the new tests

### Example Test Case

```cpp
TEST(UtilsNewFeatureTest, NewFunction) {
    // Test setup
    test_utils::GPUMemoryRAII gpu_mem(1024);
    
    // Test execution
    bool result = new_function(gpu_mem.get());
    
    // Assertions
    EXPECT_TRUE(result);
}
```

## Test Environment Variables

- `CUDA_VISIBLE_DEVICES` - Set to control which GPU devices are visible
- `GTEST_COLOR` - Control colored output (`yes`/`no`/`auto`)
- `GTEST_FILTER` - Filter which tests to run

## Troubleshooting

### Common Issues

1. **CUDA not found**: Ensure CUDA toolkit is installed and `nvcc` is in PATH
2. **Google Test not found**: Install gtest development package
3. **Compilation errors**: Check C++17 compiler support

### Debug Builds

For debugging test failures:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_CUFILE_CPP_TESTS=ON
make test_utils_functions
gdb ./tests/test_utils_functions
```

### Memory Error Detection

Run with memory error detection tools:

```bash
# Valgrind
valgrind --leak-check=full ./tests/test_utils_functions

# Address Sanitizer
cmake .. -DCMAKE_CXX_FLAGS="-fsanitize=address" -DBUILD_CUFILE_CPP_TESTS=ON
make test_utils_functions
./tests/test_utils_functions
```

## Contributing

When contributing new tests:

1. Follow the existing test structure and naming conventions
2. Use the provided RAII classes for memory management
3. Include both positive and negative test cases
4. Add appropriate comments explaining complex test scenarios
5. Update this README if adding new test categories 