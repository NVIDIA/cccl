# InstallSamples.cmake
# Configuration for installing CUDA samples to organized directory structure
#
# This module sets up installation paths organized by:
#   - Target Architecture (x86_64, aarch64, etc.)
#   - Target OS (linux, windows, darwin)
#   - Build Type (release, debug)
#
# Default installation path: build/bin/${TARGET_ARCH}/${TARGET_OS}/${BUILD_TYPE}
#
# Installation structure:
#   - Executables: installed to flat root directory only (easy access)
#   - Data files (.ll, .ptx, .fatbin, etc.): installed to subdirectories (preserves relative paths)
#   - run_tests.py handles path resolution automatically for both nested and flat structures
#
# Users can override by setting CMAKE_INSTALL_PREFIX or CUDA_SAMPLES_INSTALL_DIR

# Configure paths only once (but always define the function below)
if (NOT CUDA_SAMPLES_INSTALL_CONFIGURED)
  set(
    CUDA_SAMPLES_INSTALL_CONFIGURED
    TRUE
    CACHE INTERNAL
    "InstallSamples configuration guard"
  )

  # Detect target architecture - use lowercase of CMAKE_SYSTEM_PROCESSOR
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" TARGET_ARCH_TEMP)
  set(TARGET_ARCH "${TARGET_ARCH_TEMP}" CACHE INTERNAL "Target architecture")

  # Detect target OS
  if (WIN32)
    set(TARGET_OS_TEMP "windows")
  elseif (APPLE)
    set(TARGET_OS_TEMP "darwin")
  elseif (UNIX)
    if (CMAKE_SYSTEM_NAME MATCHES QNX)
      set(TARGET_OS_TEMP "qnx")
    else()
      set(TARGET_OS_TEMP "linux")
    endif()
  else()
    set(TARGET_OS_TEMP "unknown")
  endif()
  set(TARGET_OS "${TARGET_OS_TEMP}" CACHE INTERNAL "Target OS")

  # Detect if using multi-config generator (Visual Studio, Xcode, Ninja Multi-Config)
  get_property(MULTI_CONFIG_TEMP GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  set(
    MULTI_CONFIG
    "${MULTI_CONFIG_TEMP}"
    CACHE INTERNAL
    "Multi-config generator flag"
  )

  # Get build type
  if (MULTI_CONFIG)
    # Multi-config generators: use $<CONFIG> which is evaluated at build/install time
    set(
      BUILD_TYPE_EXPR
      "$<LOWER_CASE:$<CONFIG>>"
      CACHE INTERNAL
      "Build type expression"
    )
    set(BUILD_TYPE_MSG "multi-config (specified at build time)")
  else()
    # Single-config generators: use CMAKE_BUILD_TYPE (default to release if not specified)
    if (NOT CMAKE_BUILD_TYPE)
      set(CMAKE_BUILD_TYPE "Release")
    endif()
    string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER_TEMP)
    set(
      BUILD_TYPE_EXPR
      "${BUILD_TYPE_LOWER_TEMP}"
      CACHE INTERNAL
      "Build type expression"
    )
    set(BUILD_TYPE_MSG "${BUILD_TYPE_LOWER_TEMP}")
  endif()

  # Set default install prefix to build/bin if not explicitly set by user
  # Use the root binary directory (where CMakeCache.txt is located)
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    # Find the directory containing CMakeCache.txt (the root build directory)
    set(ROOT_BINARY_DIR "${CMAKE_BINARY_DIR}")
    set(PREV_DIR "")
    set(MAX_ITERATIONS 50)
    set(ITERATION 0)
    while(
      NOT EXISTS "${ROOT_BINARY_DIR}/CMakeCache.txt"
      AND NOT "${ROOT_BINARY_DIR}" STREQUAL "/"
      AND NOT "${ROOT_BINARY_DIR}" STREQUAL ""
      AND NOT "${ROOT_BINARY_DIR}" STREQUAL "${PREV_DIR}"
      AND ITERATION LESS MAX_ITERATIONS
    )
      set(PREV_DIR "${ROOT_BINARY_DIR}")
      get_filename_component(ROOT_BINARY_DIR "${ROOT_BINARY_DIR}" DIRECTORY)
      math(EXPR ITERATION "${ITERATION} + 1")
    endwhile()
    # If CMakeCache.txt wasn't found, fall back to CMAKE_BINARY_DIR
    if (NOT EXISTS "${ROOT_BINARY_DIR}/CMakeCache.txt")
      set(ROOT_BINARY_DIR "${CMAKE_BINARY_DIR}")
    endif()
    # If user didn't specify CMAKE_INSTALL_PREFIX, default to build/bin
    set(
      CMAKE_INSTALL_PREFIX
      "${ROOT_BINARY_DIR}/bin"
      CACHE PATH
      "Installation directory"
      FORCE
    )
  endif()

  # Create the installation path: bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
  # For multi-config generators, this will be evaluated at install time
  if (NOT DEFINED CUDA_SAMPLES_INSTALL_DIR)
    set(
      CUDA_SAMPLES_INSTALL_DIR
      "${CMAKE_INSTALL_PREFIX}/${TARGET_ARCH}/${TARGET_OS}/${BUILD_TYPE_EXPR}"
      CACHE PATH
      "Install directory for samples"
    )
  endif()

  # Check if we should use the default dynamic installation logic
  if (
    "${CUDA_SAMPLES_INSTALL_DIR}"
      STREQUAL
      "${CMAKE_INSTALL_PREFIX}/${TARGET_ARCH}/${TARGET_OS}/${BUILD_TYPE_EXPR}"
  )
    set(CUDA_SAMPLES_INSTALL_USE_DEFAULT TRUE)
  else()
    set(CUDA_SAMPLES_INSTALL_USE_DEFAULT FALSE)
  endif()

  # Print installation configuration
  message(STATUS "CUDA Samples installation configured:")
  message(STATUS "  Architecture: ${TARGET_ARCH}")
  message(STATUS "  OS: ${TARGET_OS}")
  message(STATUS "  Build Type: ${BUILD_TYPE_MSG}")
  message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
  if (NOT MULTI_CONFIG)
    message(STATUS "  Install Directory: ${CUDA_SAMPLES_INSTALL_DIR}")
  else()
    message(
      STATUS
      "  Install Directory: ${CMAKE_INSTALL_PREFIX}/${TARGET_ARCH}/${TARGET_OS}/<config>"
    )
  endif()
endif()

# Function to setup installation for regular samples
# This should be called after all targets are defined
function(setup_samples_install)
  # Create an install script that copies executables, data files, and shared libraries
  # This script only installs - build must be done separately
  install(
    CODE
      "
        if(\"${CUDA_SAMPLES_INSTALL_USE_DEFAULT}\" STREQUAL \"TRUE\")
            # Determine the actual install directory based on the configuration
            # CMAKE_INSTALL_CONFIG_NAME is available at install time for multi-config generators
            if(DEFINED CMAKE_INSTALL_CONFIG_NAME)
                string(TOLOWER \"\${CMAKE_INSTALL_CONFIG_NAME}\" INSTALL_BUILD_TYPE)
            else()
                set(INSTALL_BUILD_TYPE \"${BUILD_TYPE_EXPR}\")
            endif()
            set(INSTALL_DIR \"${CMAKE_INSTALL_PREFIX}/${TARGET_ARCH}/${TARGET_OS}/\${INSTALL_BUILD_TYPE}\")
        else()
            set(INSTALL_DIR \"${CUDA_SAMPLES_INSTALL_DIR}\")
        endif()

        # Search in the current project's binary directory for built executables
        file(GLOB_RECURSE BINARY_FILES
             LIST_DIRECTORIES false
             \"${CMAKE_CURRENT_BINARY_DIR}/*\")

        # Copy data files from sample's own data directory
        file(GLOB_RECURSE SAMPLE_DATA_FILES
             LIST_DIRECTORIES false
             \"${CMAKE_CURRENT_SOURCE_DIR}/data/*\")

        # Copy shared data files from Common/data directory.
        # With CCCL's flat samples layout (samples/<name>/), Common is one
        # level up. Keep the deeper paths as fallbacks in case a sample is
        # dropped into a nested category-style layout.
        set(COMMON_DATA_FILES \"\")
        if(EXISTS \"${CMAKE_CURRENT_SOURCE_DIR}/../Common/data\")
            file(GLOB_RECURSE COMMON_DATA_FILES
                 LIST_DIRECTORIES false
                 \"${CMAKE_CURRENT_SOURCE_DIR}/../Common/data/*\")
        elseif(EXISTS \"${CMAKE_CURRENT_SOURCE_DIR}/../../../Common/data\")
            file(GLOB_RECURSE COMMON_DATA_FILES
                 LIST_DIRECTORIES false
                 \"${CMAKE_CURRENT_SOURCE_DIR}/../../../Common/data/*\")
        elseif(EXISTS \"${CMAKE_CURRENT_SOURCE_DIR}/../../../../Common/data\")
            file(GLOB_RECURSE COMMON_DATA_FILES
                 LIST_DIRECTORIES false
                 \"${CMAKE_CURRENT_SOURCE_DIR}/../../../../Common/data/*\")
        endif()

        # Copy shared library files from bin/win64 directory (Windows only)
        # These are pre-built DLLs like freeglut.dll, glew64.dll, etc.
        set(SHARED_LIB_FILES \"\")
        if(CMAKE_HOST_WIN32)
            # Determine build configuration at install time
            # CMAKE_INSTALL_CONFIG_NAME is set by CMake at install time for multi-config generators
            if(DEFINED CMAKE_INSTALL_CONFIG_NAME)
                string(TOLOWER \"\${CMAKE_INSTALL_CONFIG_NAME}\" INSTALL_CONFIG_LOWER)
            else()
                # Fallback for single-config generators
                set(INSTALL_CONFIG_LOWER \"${BUILD_TYPE_EXPR}\")
            endif()

            # Try multiple possible paths for bin/win64 directory.
            # CCCL's flat samples layout puts bin/ one level up; keep the
            # deeper paths as fallbacks for other layouts.
            set(BIN_WIN64_PATHS
                \"${CMAKE_CURRENT_SOURCE_DIR}/../bin/win64/\${INSTALL_CONFIG_LOWER}\"
                \"${CMAKE_CURRENT_SOURCE_DIR}/../../../bin/win64/\${INSTALL_CONFIG_LOWER}\"
                \"${CMAKE_CURRENT_SOURCE_DIR}/../../../../bin/win64/\${INSTALL_CONFIG_LOWER}\"
                \"${CMAKE_SOURCE_DIR}/bin/win64/\${INSTALL_CONFIG_LOWER}\"
            )
            foreach(BIN_PATH IN LISTS BIN_WIN64_PATHS)
                if(EXISTS \"\${BIN_PATH}\")
                    file(GLOB SHARED_LIB_FILES
                         LIST_DIRECTORIES false
                         \"\${BIN_PATH}/*.dll\")
                    if(SHARED_LIB_FILES)
                        break()
                    endif()
                endif()
            endforeach()
        endif()

        # Combine all lists
        set(SAMPLE_FILES \${BINARY_FILES} \${SAMPLE_DATA_FILES} \${COMMON_DATA_FILES} \${SHARED_LIB_FILES})

        # Remove duplicates to avoid copying the same file multiple times
        # This preserves the order, so files from earlier sources take precedence
        list(REMOVE_DUPLICATES SAMPLE_FILES)

        set(INSTALLED_COUNT 0)

        # Filter to include executable files and specific file types
        foreach(SAMPLE_FILE IN LISTS SAMPLE_FILES)
            # Skip non-files
            if(NOT IS_DIRECTORY \"\${SAMPLE_FILE}\")
                get_filename_component(SAMPLE_EXT \"\${SAMPLE_FILE}\" EXT)
                get_filename_component(SAMPLE_NAME \"\${SAMPLE_FILE}\" NAME)

                set(SHOULD_INSTALL FALSE)

                # Skip build artifacts, source files, and CMake files
                # Note: .lib (Windows import libs) and .a (static libs) are excluded - link-time only
                # .so (Linux shared libs) and .dll (Windows DLLs) are included - runtime dependencies
                # Source files (.cu, .cpp, .c, .h, etc.) are excluded - not needed at runtime
                if(NOT SAMPLE_EXT MATCHES \"\\\\.(o|a|cmake|obj|lib|exp|ilk|pdb|cu|cpp|cxx|cc|c|h|hpp|hxx|cuh|inl)$\" AND
                   NOT SAMPLE_NAME MATCHES \"^(Makefile|cmake_install\\\\.cmake)$\" AND
                   NOT \"\${SAMPLE_FILE}\" MATCHES \"/CMakeFiles/\" AND
                   NOT \"\${SAMPLE_FILE}\" MATCHES \"\\\\\\\\CMakeFiles\\\\\\\\\")

                    # Check if file has required extension (fatbin, ptx, bc, raw, ppm) or is executable
                    if(SAMPLE_EXT MATCHES \"\\\\.(fatbin|ptx|bc|raw|ppm)$\")
                        set(SHOULD_INSTALL TRUE)
                    # Check for shared libraries: .dll (Windows) or .so (Linux)
                    elseif(SAMPLE_EXT MATCHES \"\\\\.(dll|so)$\")
                        set(SHOULD_INSTALL TRUE)
                    # On Windows, check for .exe extension
                    elseif(CMAKE_HOST_WIN32 AND SAMPLE_EXT MATCHES \"\\\\.(exe)$\")
                        set(SHOULD_INSTALL TRUE)
                    else()
                        # On Unix-like systems, check if file has executable permissions
                        if(NOT CMAKE_HOST_WIN32)
                            if(IS_SYMLINK \"\${SAMPLE_FILE}\" OR
                               (EXISTS \"\${SAMPLE_FILE}\" AND NOT IS_DIRECTORY \"\${SAMPLE_FILE}\"))
                                # Use test -x to check if file has executable permissions
                                execute_process(
                                    COMMAND test -x \"\${SAMPLE_FILE}\"
                                    RESULT_VARIABLE IS_EXEC
                                    OUTPUT_QUIET ERROR_QUIET
                                )
                                if(IS_EXEC EQUAL 0)
                                    set(SHOULD_INSTALL TRUE)
                                endif()
                            endif()
                        endif()
                    endif()
                endif()

                if(SHOULD_INSTALL)
                    get_filename_component(FILE_NAME \"\${SAMPLE_FILE}\" NAME)
                    set(DEST_FILE \"\${INSTALL_DIR}/\${FILE_NAME}\")

                    # Determine file type based on extension
                    get_filename_component(FILE_EXT \"\${SAMPLE_FILE}\" EXT)
                    set(IS_EXECUTABLE FALSE)
                    set(IS_SHARED_LIB FALSE)
                    set(IS_DATA_FILE FALSE)

                    # Check for known data file extensions first
                    if(FILE_EXT MATCHES \"\\\\.(fatbin|ptx|bc|raw|ppm)$\")
                        set(IS_DATA_FILE TRUE)
                    # Check if it's a shared library
                    elseif(FILE_EXT MATCHES \"\\\\.(dll|so)$\")
                        set(IS_SHARED_LIB TRUE)
                    # Check if it's an executable
                    else()
                        # On Windows, check for .exe extension (not .dll - those are libraries)
                        if(CMAKE_HOST_WIN32)
                            if(FILE_EXT MATCHES \"\\\\.(exe)$\")
                                set(IS_EXECUTABLE TRUE)
                            endif()
                        else()
                            # On Unix-like systems, check for no extension (typical for executables)
                            # .so files are shared libraries, not executables
                            if(FILE_EXT STREQUAL \"\")
                                set(IS_EXECUTABLE TRUE)
                            endif()
                        endif()
                    endif()

                    get_filename_component(DEST_DIR \"\${DEST_FILE}\" DIRECTORY)

                    # Check if this is a Common data file that already exists
                    # Skip copying to avoid redundant operations when multiple samples use the same files
                    set(SKIP_COPY FALSE)
                    if(\"\${SAMPLE_FILE}\" MATCHES \"/Common/data/\" AND EXISTS \"\${DEST_FILE}\")
                        set(SKIP_COPY TRUE)
                    endif()

                    if(NOT SKIP_COPY)
                    if(IS_DATA_FILE)
                        # Data file (.raw, .ppm, .ptx, .fatbin, .bc) - copy without execute permissions
                        message(STATUS \"Installing data file: \${DEST_FILE}\")
                        if(CMAKE_HOST_WIN32)
                            file(COPY \"\${SAMPLE_FILE}\" DESTINATION \"\${DEST_DIR}\")
                        else()
                            file(COPY \"\${SAMPLE_FILE}\"
                                 DESTINATION \"\${DEST_DIR}\"
                                 FILE_PERMISSIONS OWNER_READ OWNER_WRITE
                                                  GROUP_READ
                                                  WORLD_READ)
                        endif()
                        math(EXPR INSTALLED_COUNT \"\${INSTALLED_COUNT} + 1\")
                    elseif(IS_EXECUTABLE)
                        # File is executable - copy with execute permissions (Unix) or as-is (Windows)
                        message(STATUS \"Installing executable: \${DEST_FILE}\")
                        if(CMAKE_HOST_WIN32)
                            file(COPY \"\${SAMPLE_FILE}\" DESTINATION \"\${DEST_DIR}\")
                        else()
                            file(COPY \"\${SAMPLE_FILE}\"
                                 DESTINATION \"\${DEST_DIR}\"
                                 FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                                                  GROUP_READ GROUP_EXECUTE
                                                  WORLD_READ WORLD_EXECUTE)
                        endif()
                        math(EXPR INSTALLED_COUNT \"\${INSTALLED_COUNT} + 1\")
                    elseif(IS_SHARED_LIB)
                        # Shared library - copy with appropriate permissions
                        message(STATUS \"Installing shared library: \${DEST_FILE}\")
                        if(CMAKE_HOST_WIN32)
                            file(COPY \"\${SAMPLE_FILE}\" DESTINATION \"\${DEST_DIR}\")
                        else()
                            file(COPY \"\${SAMPLE_FILE}\"
                                 DESTINATION \"\${DEST_DIR}\"
                                 FILE_PERMISSIONS OWNER_READ OWNER_WRITE
                                                  GROUP_READ
                                                  WORLD_READ)
                        endif()
                        math(EXPR INSTALLED_COUNT \"\${INSTALLED_COUNT} + 1\")
                    else()
                        # Unknown file type - copy as regular file without execute permissions
                        message(STATUS \"Installing file: \${DEST_FILE}\")
                        if(CMAKE_HOST_WIN32)
                            file(COPY \"\${SAMPLE_FILE}\" DESTINATION \"\${DEST_DIR}\")
                        else()
                            file(COPY \"\${SAMPLE_FILE}\"
                                 DESTINATION \"\${DEST_DIR}\"
                                 FILE_PERMISSIONS OWNER_READ OWNER_WRITE
                                                  GROUP_READ
                                                  WORLD_READ)
                        endif()
                        math(EXPR INSTALLED_COUNT \"\${INSTALLED_COUNT} + 1\")
                    endif()
                    endif() # NOT SKIP_COPY
                endif()
            endif()
        endforeach()

        message(STATUS \"Installation complete: \${INSTALLED_COUNT} files installed to \${INSTALL_DIR}\")
    "
  )
endfunction()
