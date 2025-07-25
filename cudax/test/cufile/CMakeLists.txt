# ================================================================================================
# cuFile Tests - Modern CMake Configuration
# ================================================================================================

# Only build cufile tests if the cufile library is available
if(NOT TARGET cufile_cpp)
  # Try to find or build the cufile library
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../include/cuda/experimental/__cufile/CMakeLists.txt")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../include/cuda/experimental/__cufile cufile_lib)
  else()
    message(STATUS "cuFile library not available, skipping cuFile tests")
    return()
  endif()
endif()

# ================================================================================================
# Dependencies
# ================================================================================================

# Find required packages
find_package(GTest QUIET)
if(NOT GTest_FOUND)
  message(STATUS "Google Test not found, skipping cuFile tests")
  return()
endif()

find_package(CUDAToolkit REQUIRED)

# Include Google Test discovery module
include(GoogleTest)

# ================================================================================================
# Test Utilities Library
# ================================================================================================

add_library(test_utils STATIC
    test_utils.cpp
)

target_include_directories(test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/cuda/experimental/__cufile
)

target_link_libraries(test_utils PUBLIC
    cufile_cpp
    CUDA::cudart
    GTest::gtest
    GTest::gtest_main
)

target_compile_features(test_utils PUBLIC cxx_std_17)

# Modern CMake: Use generator expressions for build-type-specific flags
target_compile_options(test_utils PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:RelWithDebInfo>:-g -O2>
)

# ================================================================================================
# Test Executables
# ================================================================================================

add_executable(test_utils_functions
    test_utils_functions.cpp
)

target_link_libraries(test_utils_functions PRIVATE
    test_utils
    GTest::gtest_main
)

target_compile_features(test_utils_functions PRIVATE cxx_std_17)

# Apply same modern compiler options pattern
target_compile_options(test_utils_functions PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:RelWithDebInfo>:-g -O2>
)

add_executable(test_driver_functions
    test_driver_functions.cpp
)

target_link_libraries(test_driver_functions PRIVATE
    test_utils
    GTest::gtest_main
)

target_compile_features(test_driver_functions PRIVATE cxx_std_17)

# Apply same modern compiler options pattern
target_compile_options(test_driver_functions PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:RelWithDebInfo>:-g -O2>
)

add_executable(test_file_handle_functions
    test_file_handle_functions.cpp
)

target_link_libraries(test_file_handle_functions PRIVATE
    test_utils
    GTest::gtest_main
)

target_compile_features(test_file_handle_functions PRIVATE cxx_std_17)

# Apply same modern compiler options pattern
target_compile_options(test_file_handle_functions PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:RelWithDebInfo>:-g -O2>
)

# ================================================================================================
# Register Tests with Google Test Discovery
# ================================================================================================

gtest_discover_tests(test_utils_functions
    PROPERTIES
        LABELS "unit"
        TIMEOUT 30
    TEST_PREFIX "Utils_Functions_"
    DISCOVERY_TIMEOUT 60
)

gtest_discover_tests(test_driver_functions
    PROPERTIES
        LABELS "unit"
        TIMEOUT 30
    TEST_PREFIX "Driver_Functions_"
    DISCOVERY_TIMEOUT 60
)

gtest_discover_tests(test_file_handle_functions
    PROPERTIES
        LABELS "unit"
        TIMEOUT 60
    TEST_PREFIX "File_Handle_Functions_"
    DISCOVERY_TIMEOUT 60
)

# ================================================================================================
# Convenience Targets
# ================================================================================================

add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --output-on-failure
    DEPENDS test_utils_functions test_driver_functions test_file_handle_functions
    COMMENT "Running all cuFile tests"
)