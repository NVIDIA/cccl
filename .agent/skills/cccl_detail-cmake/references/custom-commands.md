# Custom command signatures

## cccl_add_executable

```cmake
cccl_add_executable(
  <target_name>
  SOURCES <file> [<file>…]
  [ADD_CTEST]             # register a no-arg CTest with the same name
  [NO_METATARGETS]        # skip dot-path metatarget registration
  [NO_CLANG_TIDY]         # skip clang-tidy target hookup
  [DIALECT <std>]         # override CXX/CUDA standard (e.g. 17, 20)
  [METATARGET_PATH <path>]# override the dot-path used for metatarget registration
)
```

Calls `cccl_configure_target`, then optionally `add_test`, `cccl_ensure_metatargets`, `cccl_tidy_add_target`.

## cccl_configure_target

```cmake
cccl_configure_target(<target> [DIALECT <std>])
```

Sets: `CXX_EXTENSIONS OFF`, `CUDA_EXTENSIONS OFF`, standard properties, `cxx_std_N`/`cuda_std_N` compile features, and output dirs (`ARCHIVE_OUTPUT_DIRECTORY`, etc.) to `${CCCL_BINARY_DIR}/lib` / `bin`. INTERFACE libraries receive `INTERFACE` compile features instead of `PUBLIC`.

## cccl_add_compile_test

```cmake
cccl_add_compile_test(
  <result_var>        # receives the generated test name
  <name_prefix>       # e.g. cccl.example
  <subdir>            # relative path to the sub-project directory
  <test_id>           # unique suffix (allows same subdir, multiple ids)
  [CTEST_COMMAND <cmd>]
  <cmake_configure_args>…
)
```

Registers a CTest that runs `ctest --build-and-test <src> <build> --build-generator … --build-options <args> --test-command ctest --output-on-failure`. Build directory is `${CMAKE_CURRENT_BINARY_DIR}/<subdir>/<test_id>`.

## cccl_add_xfail_compile_target_test

```cmake
cccl_add_xfail_compile_target_test(
  <target_name>
  [TEST_NAME <name>]
  [ERROR_REGEX <regex>]
  [SOURCE_FILE <path>]
  [ERROR_REGEX_LABEL <label>]          # scan SOURCE_FILE for // LABEL {{"regex"}}
  [ERROR_NUMBER <n>]                   # match // LABEL-N {{"regex"}}
  [ERROR_NUMBER_TARGET_NAME_REGEX <r>] # extract N from target name
)
```

Marks the target `EXCLUDE_FROM_ALL`, adds a cleanup fixture that deletes the output file, registers a CTest that builds the target. Test passes if: (a) `PASS_REGULAR_EXPRESSION` matches the output, or (b) no regex provided and build fails (`WILL_FAIL true`). Source file changes re-trigger CMake via `CMAKE_CONFIGURE_DEPENDS`.

## cccl_generate_header_tests

```cmake
cccl_generate_header_tests(
  <target_name>
  <project_include_path>  # relative to CCCL_SOURCE_DIR, e.g. libcudacxx/include
  [DIALECT <std>]         # passed to cccl_configure_target
  [LANGUAGE CXX|CUDA]     # default: CUDA
  [HEADER_TEMPLATE <file>]
  [GLOBS <pattern>…]
  [EXCLUDES <pattern>…]
  [HEADERS <file>…]
  [PER_HEADER_DEFINES DEFINE <def> <regex>… [DEFINE …]]
  [NO_METATARGETS]
)
```

Configures `cmake/header_test.cu.in` (or custom template) for each matched header, replacing `@header@`. Builds an OBJECT library. Also creates `<target_name>.link_check` that links the objects twice — this causes a duplicate-symbol error if any header function lacks `inline`.

## cccl_generate_install_rules

```cmake
cccl_generate_install_rules(
  <PROJECT_NAME>        # case-sensitive, used for option name
  <DEFAULT_ENABLE>      # ON/OFF default for the install option
  [NO_HEADERS]
  [HEADERS_SUBDIRS <dir>…]
  [HEADERS_INCLUDE <glob>…]
  [HEADERS_EXCLUDE <glob>…]
  [PACKAGE]             # install the CMake package from lib/cmake/<name_lower>/
)
```

Creates cache option `<PROJECT_NAME>_ENABLE_INSTALL_RULES`. Headers installed to `CMAKE_INSTALL_INCLUDEDIR`. Package to `CMAKE_INSTALL_LIBDIR/cmake/`. If a `<name>-header-search.cmake.in` exists in the package dir, it is configured with `_CCCL_RELATIVE_LIBDIR` and installed.

## cccl_parse_variant_params / cccl_get_variant_data

Source files embed `%PARAM%` comment lines:

```cpp
// %PARAM% DEFINITION_NAME label value1:value2:value3
```

`cccl_parse_variant_params(src num_var labels_var defs_var)` extracts these into cartesian-product variant lists. `cccl_get_variant_data(labels defs idx label_out defs_out)` retrieves one variant by index (also appends `VAR_IDX=N`).

## cccl_ensure_metatargets

```cmake
cccl_ensure_metatargets(<target_name> [METATARGET_PATH <path>])
```

Splits `METATARGET_PATH` on `.`, creating custom targets for each prefix segment if they don't exist, chaining `add_dependencies` up the tree. The real target is made a dependency of the leaf segment.
