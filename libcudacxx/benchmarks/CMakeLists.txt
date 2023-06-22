cmake_minimum_required(VERSION 3.18)

project(libcudacxx-benchmarks LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

set(GPU_ARCHS "70;72;75;80" CACHE STRING "List of GPU architectures to compile for.")

message(STATUS "Enabled CUDA architectures:${GPU_ARCHS}")

find_package(Threads REQUIRED)
find_package(OpenMP)

function(ConfigureHostBench BENCH_NAME BENCH_SRC)
   add_executable("${BENCH_NAME}" "${BENCH_SRC}")
   target_link_libraries("${BENCH_NAME}" PRIVATE Threads::Threads)

   # TODO: Link against libcudaxx interface target instead
   target_include_directories("${BENCH_NAME}" PRIVATE
                                              "${CMAKE_CURRENT_SOURCE_DIR}/../include")
   if(OpenMP_CXX_FOUND)
      target_link_libraries("${BENCH_NAME}" PRIVATE OpenMP::OpenMP_CXX)
   endif()
endfunction(ConfigureHostBench)

function(ConfigureDeviceBench BENCH_NAME BENCH_SRC)
   add_executable("${BENCH_NAME}" "${BENCH_SRC}")
   set_property(TARGET "${BENCH_NAME}" PROPERTY CUDA_ARCHITECTURES "${GPU_ARCHS}")
   # TODO: Link against libcudaxx interface target instead
   target_include_directories("${BENCH_NAME}" PRIVATE
                                              "${CMAKE_CURRENT_SOURCE_DIR}/../include")
   target_compile_options("${BENCH_NAME}" PRIVATE --expt-extended-lambda --expt-relaxed-constexpr)
endfunction(ConfigureDeviceBench)

ConfigureHostBench(concurrency_host concurrency.cpp)                                        

ConfigureDeviceBench(concurrency_device concurrency.cu)

