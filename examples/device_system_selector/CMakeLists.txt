cmake_minimum_required(VERSION 3.26)

set(THRUST_BACKEND "CPP" CACHE STRING "THRUST_DEVICE_SYSTEM to use (CPP, OMP, CUDA)")
string(TOUPPER "${THRUST_BACKEND}" THRUST_BACKEND)

# Conditionally enable CUDA
if(THRUST_BACKEND STREQUAL "CUDA")
    project(thrust_selector LANGUAGES CUDA CXX)
else()
    project(thrust_selector LANGUAGES CXX)
endif()

# Decide source file
if(THRUST_BACKEND STREQUAL "CUDA")
    set(SRC "main.cu")
else()
    set(SRC "main.cpp")
endif()

add_executable(thrust_selector ${SRC})

# Include Thrust (from parent directory)
target_include_directories(thrust_selector PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../thrust)

# Set THRUST_DEVICE_SYSTEM
if(THRUST_BACKEND STREQUAL "CPP")
    target_compile_definitions(thrust_selector PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
elseif(THRUST_BACKEND STREQUAL "OMP")
    find_package(OpenMP REQUIRED)
    target_compile_definitions(thrust_selector PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
    target_link_libraries(thrust_selector PRIVATE OpenMP::OpenMP_CXX)
elseif(THRUST_BACKEND STREQUAL "CUDA")
    target_compile_definitions(thrust_selector PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
else()
    message(FATAL_ERROR "Invalid THRUST_BACKEND selected")
endif()

target_compile_features(thrust_selector PRIVATE cxx_std_17)


add_subdirectory(device_system_selector)
