cmake_minimum_required(VERSION 3.18)

# ================================================================================================
# Project Configuration
# ================================================================================================

project(cufile 
    VERSION 1.0.0 
    LANGUAGES CXX CUDA
    DESCRIPTION "cuFile library for GPU-accelerated file I/O"
)

# Set CUDA architectures if not already set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90;120")
endif()

# Use GNU install directories
include(GNUInstallDirs)

# Testing option
option(BUILD_TESTING "Build tests" ON)
include(CTest)

# ================================================================================================
# Dependencies
# ================================================================================================

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)

# Find system dependencies
pkg_check_modules(LIBUDEV REQUIRED IMPORTED_TARGET libudev)
pkg_check_modules(NUMA IMPORTED_TARGET numa)

# Find other system libraries
find_library(URCU_LIB urcu REQUIRED)
find_library(AIO_LIB aio REQUIRED)
find_library(URING_LIB uring)

# Optional RDMA support
find_library(IBVERBS_LIB ibverbs)
find_library(RDMACM_LIB rdmacm)

# Build jsoncpp if not found
pkg_check_modules(JSONCPP jsoncpp)
if(NOT JSONCPP_FOUND)
    message(STATUS "Building jsoncpp from source")
    set(BUILD_STATIC_LIBS ON)
    set(BUILD_SHARED_LIBS OFF)
    set(JSONCPP_WITH_CMAKE_PACKAGE OFF)
    add_subdirectory(../third-party/json jsoncpp_build)
    set(JSONCPP_TARGET jsoncpp_lib_static)
    set_target_properties(jsoncpp_lib_static PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
    )
else()
    # Create imported target for system jsoncpp
    add_library(jsoncpp_system INTERFACE)
    target_link_libraries(jsoncpp_system INTERFACE ${JSONCPP_LIBRARIES})
    target_include_directories(jsoncpp_system INTERFACE ${JSONCPP_INCLUDE_DIRS})
    set(JSONCPP_TARGET jsoncpp_system)
endif()

# ConcurrencyKit - Modern CMake approach
pkg_check_modules(LIBCK libck)
if(LIBCK_FOUND)
    message(STATUS "Using system ConcurrencyKit")
    # Create imported target for system libck
    add_library(ConcurrencyKit::ck INTERFACE IMPORTED)
    target_link_libraries(ConcurrencyKit::ck INTERFACE ${LIBCK_LIBRARIES})
    target_include_directories(ConcurrencyKit::ck INTERFACE ${LIBCK_INCLUDE_DIRS})
    set(CK_TARGET ConcurrencyKit::ck)
else()
    message(STATUS "Using bundled ConcurrencyKit")
    
    # Determine architecture
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(CK_ARCH "aarch64")
    else()
        set(CK_ARCH "x86_64")
    endif()
    
    # Check if pre-built libraries exist
    set(CK_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/ckit/${CK_ARCH})
    set(CK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/ckit/include)
    
    if(EXISTS ${CK_LIB_DIR}/libck.so AND EXISTS ${CK_LIB_DIR}/libck.a)
        message(STATUS "Using pre-built ConcurrencyKit libraries for ${CK_ARCH}")
        
        # Create imported targets for pre-built libraries
        add_library(ConcurrencyKit::ck_shared SHARED IMPORTED)
        set_target_properties(ConcurrencyKit::ck_shared PROPERTIES
            IMPORTED_LOCATION ${CK_LIB_DIR}/libck.so
            INTERFACE_INCLUDE_DIRECTORIES ${CK_INCLUDE_DIR}
        )
        
        add_library(ConcurrencyKit::ck_static STATIC IMPORTED)
        set_target_properties(ConcurrencyKit::ck_static PROPERTIES
            IMPORTED_LOCATION ${CK_LIB_DIR}/libck.a
            INTERFACE_INCLUDE_DIRECTORIES ${CK_INCLUDE_DIR}
        )
        
        # Use static library since shared library doesn't export symbols properly
        set(CK_TARGET ConcurrencyKit::ck_static)
        
    else()
        message(STATUS "Building ConcurrencyKit from source")
        
        # Build from source using ExternalProject
        ExternalProject_Add(concurrencykit_build
            SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/ckit/srctree
            CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
            BUILD_COMMAND make
            INSTALL_COMMAND make install
            BUILD_BYPRODUCTS 
                <INSTALL_DIR>/lib/libck.so
                <INSTALL_DIR>/lib/libck.a
        )
        
        # Create imported target from built library
        add_library(ConcurrencyKit::ck_built SHARED IMPORTED)
        ExternalProject_Get_Property(concurrencykit_build INSTALL_DIR)
        set_target_properties(ConcurrencyKit::ck_built PROPERTIES
            IMPORTED_LOCATION ${INSTALL_DIR}/lib/libck.so
            INTERFACE_INCLUDE_DIRECTORIES ${INSTALL_DIR}/include
        )
        add_dependencies(ConcurrencyKit::ck_built concurrencykit_build)
        
        set(CK_TARGET ConcurrencyKit::ck_built)
    endif()
endif()

# Build NVFS using external project
include(ExternalProject)
ExternalProject_Add(nvfs_build
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../nvfs
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make build
    BUILD_IN_SOURCE ON
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ../nvfs/nvfs.o ../nvfs/nvfs_async.o ../nvfs/nvfs_batch.o 
                    ../nvfs/nvfs_bounce_buffer.o ../nvfs/nvfs_stats.o
)

# Create NVFS imported library
add_library(nvfs_combined STATIC IMPORTED)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libnvfs.a
    COMMAND ${CMAKE_AR} rcs ${CMAKE_CURRENT_BINARY_DIR}/libnvfs.a
        ${CMAKE_CURRENT_SOURCE_DIR}/../nvfs/*.o
    DEPENDS nvfs_build
    COMMENT "Creating NVFS static library"
)
add_custom_target(nvfs_target DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libnvfs.a)
set_target_properties(nvfs_combined PROPERTIES 
    IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/libnvfs.a
)

# ================================================================================================
# cuFile Library
# ================================================================================================

# Get GDS version
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../nvfsdriver/GDS_VERSION")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../nvfsdriver/GDS_VERSION" GDS_VERSION)
    string(STRIP "${GDS_VERSION}" GDS_VERSION)
else()
    set(GDS_VERSION "unknown")
endif()

# Define source files
set(CUFILE_SOURCES
    cufile.cc cufile_iouring.cc cufile_core.cc cufile_posix_async.cc
    cufile_async.cc cufile_numa_helpers.cc cufile_work_item.cc
    cufile_work_queue.cc cufile_wait_queue.cc cufile_config_params.cc
    cufile_config_reader.cc cufile_platform_topology_nvfs.cc
    cufile_platform_topology_udev.cc cufile_platform_topology_sysfs.cc
    cufile_driver.cc cufile_fdhash.cc cufile_fsinfo.cc
    cufile_generic_cacheops.cc cufile_nvinfo_hash.cc cufile_object.cc
    cufile_platform.cc cufile_dynamic_routing.cc cufile_p2p.cc
    cufile_posix.cc cufile_posix_pool.cc cufile_posix_pool_stats.cc
    cufile_rdma.cc cufile_rdma_loadbalancer.cc cufile_rdma_stats.cc
    cufile_thread_pool_stats.cc cufile_stats.cc cufile_stats_plugin.cc
    cufile_udev_helpers.cc cufile_udev_utils.cc cufile_ext4.cc
    cufile_lustre.cc cufile_wekafs.cc cufile_nfs.cc cufile_gpfs.cc
    cufile_xfs.cc cufile_beegfs.cc cufile_mount_helpers.cc
    cufile_urcucds_helpers.cc cufile_urcubp_helpers.cc cufile_cuda_helpers.cc
    cufile_aio.cc cufile_batch.cc cufile_batch_pool.cc cufile_logger_ext.cc
    cufile_scatefs.cc cufile_config_store.cc
    cufile_gpu.cu
)

# Create main library
add_library(cufile SHARED ${CUFILE_SOURCES})
add_dependencies(cufile nvfs_target)

# Set modern C++ and CUDA standards per target
target_compile_features(cufile PRIVATE cxx_std_17)
set_target_properties(cufile PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Set library properties
set_target_properties(cufile PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH FALSE
)

# Configure compile definitions
target_compile_definitions(cufile PRIVATE
    CUFILE_RELEASE_VERSION="${GDS_VERSION}"
    CONFIG_CUFILE_LUSTRE CONFIG_CUFILE_BEEGFS CONFIG_CUFILE_WEKAFS
    CONFIG_CUFILE_NFS CONFIG_CUFILE_STATS CONFIG_CUFILE_ASSERT
    CONFIG_CUFILE_GPFS CONFIG_CUFILE_THREADPOOL CONFIG_CUFILE_SCATEFS
    NUMA_AWARE_CUFILE_THREADPOOL SDT_USE_VARIADIC CUFILE_LIB_BUILD
    NVFS_ENABLE_KERN_RDMA_SUPPORT _GNU_SOURCE _FILE_OFFSET_BITS=64
    $<$<CONFIG:Release>:CUFILE_RELEASE_BUILD>
    $<$<NOT:$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>>:USE_STAP_PROBE>
    $<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},aarch64>:AARCH64_PLATFORM>
    $<$<BOOL:${URING_LIB}>:CONFIG_USE_IO_URING>
)

# Modern CMake: Use generator expressions for build-type-specific flags
target_compile_options(cufile PRIVATE
    -Wall -fpermissive -fPIC
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O2 -s -fvisibility=hidden -fvisibility-inlines-hidden>
    $<$<CONFIG:RelWithDebInfo>:-g -O2>
    $<$<CONFIG:MinSizeRel>:-Os -s -fvisibility=hidden -fvisibility-inlines-hidden>
)

# Configure include directories
target_include_directories(cufile PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../nvfs
    ${CMAKE_CURRENT_SOURCE_DIR}/../mlx
    ${CMAKE_CURRENT_SOURCE_DIR}/../third-party/json/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# check CK_TARGET exists
if(NOT CK_TARGET)
    message(FATAL_ERROR "CK_TARGET is not set")
endif()

# Link libraries
target_link_libraries(cufile 
    PRIVATE
        # Core dependencies
        nvfs_combined
        ${JSONCPP_TARGET}
        
        # CUDA libraries
        CUDA::cudart
        CUDA::cuda_driver
        
        # System libraries
        ${URCU_LIB}
        ${AIO_LIB}
        PkgConfig::LIBUDEV
        Threads::Threads
        dl rt
        
        # Optional libraries
        $<$<BOOL:${NUMA_FOUND}>:PkgConfig::NUMA>
        $<$<BOOL:${URING_LIB}>:${URING_LIB}>
        $<$<AND:$<BOOL:${IBVERBS_LIB}>,$<BOOL:${RDMACM_LIB}>>:${IBVERBS_LIB};${RDMACM_LIB}>
    PUBLIC
        # Dependencies that need to be available to consumers
        ${CK_TARGET}
)

# ================================================================================================
# C++ Interface Target
# ================================================================================================

add_library(cufile_cpp INTERFACE)

target_include_directories(cufile_cpp INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(cufile_cpp INTERFACE 
    cufile
    CUDA::cudart
)

# ================================================================================================
# Testing
# ================================================================================================

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# ================================================================================================
# Installation
# ================================================================================================

# Install library
install(TARGETS cufile cufile_cpp
    EXPORT cufile-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install CMake config
install(EXPORT cufile-targets
    FILE cufile-targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cufile
)

# Generate and install config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    cufile-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/cufile-config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cufile-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cufile
) 