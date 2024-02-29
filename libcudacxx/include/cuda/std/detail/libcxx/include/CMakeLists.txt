set(files
  __assert
  __algorithm/adjacent_find.h
  __algorithm/all_of.h
  __algorithm/any_of.h
  __algorithm/binary_search.h
  __algorithm/clamp.h
  __algorithm/comp_ref_type.h
  __algorithm/comp.h
  __algorithm/copy_backward.h
  __algorithm/copy_if.h
  __algorithm/copy_n.h
  __algorithm/copy.h
  __algorithm/count.h
  __algorithm/count_if.h
  __algorithm/equal_range.h
  __algorithm/equal.h
  __algorithm/fill_n.h
  __algorithm/fill.h
  __algorithm/find_end.h
  __algorithm/find_first_of.h
  __algorithm/find_if_not.h
  __algorithm/find_if.h
  __algorithm/find.h
  __algorithm/for_each_n.h
  __algorithm/for_each.h
  __algorithm/generate_n.h
  __algorithm/generate.h
  __algorithm/half_positive.h
  __algorithm/includes.h
  __algorithm/is_heap_until.h
  __algorithm/is_heap.h
  __algorithm/is_partitioned.h
  __algorithm/is_permutation.h
  __algorithm/is_sorted_until.h
  __algorithm/is_sorted.h
  __algorithm/iter_swap.h
  __algorithm/iterator_operations.h
  __algorithm/lexicographical_compare.h
  __algorithm/lower_bound.h
  __algorithm/make_heap.h
  __algorithm/max_element.h
  __algorithm/max.h
  __algorithm/merge.h
  __algorithm/min_element.h
  __algorithm/min.h
  __algorithm/minmax_element.h
  __algorithm/minmax.h
  __algorithm/mismatch.h
  __algorithm/move_backward.h
  __algorithm/move.h
  __algorithm/next_permutation.h
  __algorithm/none_of.h
  __algorithm/partial_sort_copy.h
  __algorithm/partial_sort.h
  __algorithm/partition_copy.h
  __algorithm/partition_point.h
  __algorithm/partition.h
  __algorithm/pop_heap.h
  __algorithm/prev_permutation.h
  __algorithm/push_heap.h
  __algorithm/ranges_iterator_concept.h
  __algorithm/remove_copy_if.h
  __algorithm/remove_copy.h
  __algorithm/remove_if.h
  __algorithm/remove.h
  __algorithm/replace_copy_if.h
  __algorithm/replace_copy.h
  __algorithm/replace_if.h
  __algorithm/replace.h
  __algorithm/reverse_copy.h
  __algorithm/reverse.h
  __algorithm/rotate_copy.h
  __algorithm/rotate.h
  __algorithm/search_n.h
  __algorithm/search.h
  __algorithm/set_difference.h
  __algorithm/set_intersection.h
  __algorithm/set_symmetric_difference.h
  __algorithm/set_union.h
  __algorithm/shift_left.h
  __algorithm/shift_right.h
  __algorithm/sift_down.h
  __algorithm/sort_heap.h
  __algorithm/swap_ranges.h
  __algorithm/transform.h
  __algorithm/unique_copy.h
  __algorithm/unique.h
  __algorithm/upper_bound.h
  __availability
  __bit_reference
  __bsd_locale_defaults.h
  __bsd_locale_fallbacks.h
  __errc
  __cccl_config
  __cccl/compiler.h
  __cccl/diagnostic.h
  __cccl/dialect.h
  __cccl/execution_space.h
  __cccl/system_header.h
  __cccl/version.h
  __cccl/visibility.h
  __concepts/__concept_macros.h
  __concepts/_One_of.h
  __concepts/arithmetic.h
  __concepts/assignable.h
  __concepts/boolean_testable.h
  __concepts/class_or_enum.h
  __concepts/common_reference_with.h
  __concepts/common_with.h
  __concepts/constructible.h
  __concepts/convertible_to.h
  __concepts/copyable.h
  __concepts/derived_from.h
  __concepts/destructible.h
  __concepts/different_from.h
  __concepts/equality_comparable.h
  __concepts/invocable.h
  __concepts/movable.h
  __concepts/predicate.h
  __concepts/regular.h
  __concepts/relation.h
  __concepts/same_as.h
  __concepts/semiregular.h
  __concepts/swappable.h
  __concepts/totally_ordered.h
  __cuda/atomic_prelude.h
  __cuda/atomic.h
  __cuda/barrier.h
  __cuda/chrono.h
  __cuda/climits_prelude.h
  __cuda/cstddef_prelude.h
  __cuda/cstdint_prelude.h
  __cuda/latch.h
  __cuda/semaphore.h
  __debug
  __expected/bad_expected_access.h
  __expected/expected.h
  __expected/unexpected.h
  __expected/unexpect.h
  __expected/unexpected.h
  __functional/binary_function.h
  __functional/binary_negate.h
  __functional/bind_back.h
  __functional/bind_front.h
  __functional/bind.h
  __functional/binder1st.h
  __functional/binder2nd.h
  __functional/compose.h
  __functional/default_searcher.h
  __functional/function.h
  __functional/hash.h
  __functional/identity.h
  __functional/invoke.h
  __functional/is_transparent.h
  __functional/mem_fn.h
  __functional/mem_fun_ref.h
  __functional/not_fn.h
  __functional/operations.h
  __functional/perfect_forward.h
  __functional/pointer_to_binary_function.h
  __functional/pointer_to_unary_function.h
  __functional/ranges_operations.h
  __functional/reference_wrapper.h
  __functional/unary_function.h
  __functional/unary_negate.h
  __functional/unwrap_ref.h
  __functional/weak_result_type.h
  __functional_base
  __fwd/array.h
  __fwd/get.h
  __fwd/hash.h
  __fwd/pair.h
  __fwd/subrange.h
  __fwd/tuple.h
  __hash_table
  __iterator/access.h
  __iterator/advance.h
  __iterator/back_insert_iterator.h
  __iterator/bounded_iterator.h
  __iterator/concepts.h
  __iterator/data.h
  __iterator/default_sentinel.h
  __iterator/distance.h
  __iterator/empty.h
  __iterator/erase_if_container.h
  __iterator/front_insert_iterator.h
  __iterator/insert_iterator.h
  __iterator/incrementable_traits.h
  __iterator/indirectly_comparable.h
  __iterator/istream_iterator.h
  __iterator/istreambuf_iterator.h
  __iterator/iter_move.h
  __iterator/iter_swap.h
  __iterator/iterator.h
  __iterator/iterator_traits.h
  __iterator/mergeable.h
  __iterator/move_iterator.h
  __iterator/move_sentinel.h
  __iterator/next.h
  __iterator/ostream_iterator.h
  __iterator/ostreambuf_iterator.h
  __iterator/permutable.h
  __iterator/prev.h
  __iterator/projected.h
  __iterator/readable_traits.h
  __iterator/reverse_access.h
  __iterator/reverse_iterator.h
  __iterator/size.h
  __iterator/sortable.h
  __iterator/unreachable_sentinel.h
  __iterator/wrap_iter.h
  __libcpp_version
  __locale
  __mdspan/compressed_pair.hpp
  __mdspan/config.hpp
  __mdspan/default_accessor.hpp
  __mdspan/dynamic_extent.hpp
  __mdspan/extents.hpp
  __mdspan/full_extent_t.hpp
  __mdspan/layout_left.hpp
  __mdspan/layout_right.hpp
  __mdspan/layout_stride.hpp
  __mdspan/macros.hpp
  __mdspan/maybe_static_value.hpp
  __mdspan/mdspan.hpp
  __mdspan/no_unique_address.hpp
  __mdspan/standard_layout_static_array.hpp
  __mdspan/static_array.hpp
  __mdspan/submdspan.hpp
  __mdspan/type_list.hpp
  __memory/addressof.h
  __memory/construct_at.h
  __memory/pointer_traits.h
  __memory/voidify.h
  __mutex_base
  __node_handle
  __nullptr
  __ranges/access.h
  __ranges/concepts.h
  __ranges/dangling.h
  __ranges/data.h
  __ranges/empty.h
  __ranges/enable_borrowed_range.h
  __ranges/enable_view.h
  __ranges/rbegin.h
  __ranges/rend.h
  __ranges/size.h
  __ranges/subrange.h
  __ranges/view_interface.h
  __ranges/views.h
  __split_buffer
  __sso_allocator
  __std_stream
  __string
  __threading_support
  __tree
  __tuple_dir/apply_cv.h
  __tuple_dir/make_tuple_types.h
  __tuple_dir/sfinae_helpers.h
  __tuple_dir/structured_bindings.h
  __tuple_dir/tuple_element.h
  __tuple_dir/tuple_indices.h
  __tuple_dir/tuple_like_ext.h
  __tuple_dir/tuple_like.h
  __tuple_dir/tuple_size.h
  __tuple_dir/tuple_types.h
  __type_traits/add_const.h
  __type_traits/add_cv.h
  __type_traits/add_lvalue_reference.h
  __type_traits/add_pointer.h
  __type_traits/add_rvalue_reference.h
  __type_traits/add_volatile.h
  __type_traits/aligned_storage.h
  __type_traits/aligned_union.h
  __type_traits/alignment_of.h
  __type_traits/apply_cv.h
  __type_traits/can_extract_key.h
  __type_traits/common_type.h
  __type_traits/conditional.h
  __type_traits/conjunction.h
  __type_traits/copy_cv.h
  __type_traits/copy_cvref.h
  __type_traits/decay.h
  __type_traits/dependent_type.h
  __type_traits/disjunction.h
  __type_traits/enable_if.h
  __type_traits/extent.h
  __type_traits/has_unique_object_representation.h
  __type_traits/has_virtual_destructor.h
  __type_traits/integral_constant.h
  __type_traits/is_abstract.h
  __type_traits/is_aggregate.h
  __type_traits/is_allocator.h
  __type_traits/is_arithmetic.h
  __type_traits/is_array.h
  __type_traits/is_assignable.h
  __type_traits/is_base_of.h
  __type_traits/is_bounded_array.h
  __type_traits/is_callable.h
  __type_traits/is_char_like_type.h
  __type_traits/is_class.h
  __type_traits/is_compound.h
  __type_traits/is_constant_evaluated.h
  __type_traits/is_const.h
  __type_traits/is_constructible.h
  __type_traits/is_convertible.h
  __type_traits/is_copy_assignable.h
  __type_traits/is_copy_constructible.h
  __type_traits/is_core_convertible.h
  __type_traits/is_default_constructible.h
  __type_traits/is_destructible.h
  __type_traits/is_empty.h
  __type_traits/is_enum.h
  __type_traits/is_final.h
  __type_traits/is_floating_point.h
  __type_traits/is_function.h
  __type_traits/is_fundamental.h
  __type_traits/is_implicitly_default_constructible.h
  __type_traits/is_integral.h
  __type_traits/is_literal_type.h
  __type_traits/is_member_function_pointer.h
  __type_traits/is_member_object_pointer.h
  __type_traits/is_member_pointer.h
  __type_traits/is_move_assignable.h
  __type_traits/is_move_constructible.h
  __type_traits/is_nothrow_assignable.h
  __type_traits/is_nothrow_constructible.h
  __type_traits/is_nothrow_convertible.h
  __type_traits/is_nothrow_copy_assignable.h
  __type_traits/is_nothrow_copy_constructible.h
  __type_traits/is_nothrow_default_constructible.h
  __type_traits/is_nothrow_destructible.h
  __type_traits/is_nothrow_move_assignable.h
  __type_traits/is_nothrow_move_constructible.h
  __type_traits/is_null_pointer.h
  __type_traits/is_object.h
  __type_traits/is_pod.h
  __type_traits/is_pointer.h
  __type_traits/is_polymorphic.h
  __type_traits/is_primary_template.h
  __type_traits/is_referenceable.h
  __type_traits/is_reference.h
  __type_traits/is_reference_wrapper.h
  __type_traits/is_same.h
  __type_traits/is_scalar.h
  __type_traits/is_scoped_enum.h
  __type_traits/is_signed.h
  __type_traits/is_signed_integer.h
  __type_traits/is_standard_layout.h
  __type_traits/is_swappable.h
  __type_traits/is_trivial.h
  __type_traits/is_trivially_assignable.h
  __type_traits/is_trivially_constructible.h
  __type_traits/is_trivially_copyable.h
  __type_traits/is_trivially_copy_assignable.h
  __type_traits/is_trivially_copy_constructible.h
  __type_traits/is_trivially_default_constructible.h
  __type_traits/is_trivially_destructible.h
  __type_traits/is_trivially_move_assignable.h
  __type_traits/is_trivially_move_constructible.h
  __type_traits/is_unbounded_array.h
  __type_traits/is_union.h
  __type_traits/is_unsigned.h
  __type_traits/is_unsigned_integer.h
  __type_traits/is_valid_expansion.h
  __type_traits/is_void.h
  __type_traits/is_volatile.h
  __type_traits/lazy.h
  __type_traits/make_32_64_or_128_bit.h
  __type_traits/make_const_lvalue_ref.h
  __type_traits/make_signed.h
  __type_traits/make_unsigned.h
  __type_traits/maybe_const.h
  __type_traits/nat.h
  __type_traits/negation.h
  __type_traits/promote.h
  __type_traits/rank.h
  __type_traits/remove_all_extents.h
  __type_traits/remove_const.h
  __type_traits/remove_const_ref.h
  __type_traits/remove_cv.h
  __type_traits/remove_cvref.h
  __type_traits/remove_extent.h
  __type_traits/remove_pointer.h
  __type_traits/remove_reference.h
  __type_traits/remove_volatile.h
  __type_traits/result_of.h
  __type_traits/type_identity.h
  __type_traits/type_list.h
  __type_traits/underlying_type.h
  __type_traits/void_t.h
  __type_traits/is_constant_evaluated.h
  __undef_macros
  __utility/as_const.h
  __utility/auto_cast.h
  __utility/cmp.h
  __utility/convert_to_integral.h
  __utility/declval.h
  __utility/exception_guard.h
  __utility/exchange.h
  __utility/forward_like.h
  __utility/forward.h
  __utility/in_place.h
  __utility/integer_sequence.h
  __utility/move.h
  __utility/pair.h
  __utility/piecewise_construct.h
  __utility/priority_tag.h
  __utility/rel_ops.h
  __utility/swap.h
  __utility/to_underlying.h
  __utility/unreachable.h
  __variant/monostate.h
  __verbose_abort
  algorithm
  any
  array
  atomic
  barrier
  bit
  bitset
  cassert
  ccomplex
  cctype
  cerrno
  cfenv
  cfloat
  charconv
  chrono
  cinttypes
  ciso646
  climits
  clocale
  cmath
  codecvt
  compare
  complex
  complex.h
  concepts
  condition_variable
  csetjmp
  csignal
  cstdarg
  cstdbool
  cstddef
  cstdint
  cstdio
  cstdlib
  cstring
  ctgmath
  ctime
  ctype.h
  cwchar
  cwctype
  deque
  errno.h
  exception
  execution
  expected
  experimental/__config
  experimental/__memory
  experimental/algorithm
  experimental/coroutine
  experimental/deque
  experimental/filesystem
  experimental/forward_list
  experimental/functional
  experimental/iterator
  experimental/list
  experimental/map
  experimental/memory_resource
  experimental/propagate_const
  experimental/regex
  experimental/set
  experimental/simd
  experimental/string
  experimental/type_traits
  experimental/unordered_map
  experimental/unordered_set
  experimental/utility
  experimental/vector
  ext/__hash
  ext/hash_map
  ext/hash_set
  fenv.h
  filesystem
  float.h
  forward_list
  fstream
  functional
  future
  initializer_list
  inttypes.h
  iomanip
  ios
  iosfwd
  iostream
  istream
  iterator
  latch
  limits
  limits.h
  list
  locale
  locale.h
  map
  math.h
  mdspan
  memory
  module.modulemap
  mutex
  new
  numeric
  optional
  ostream
  queue
  random
  ranges
  ratio
  regex
  scoped_allocator
  semaphore
  set
  setjmp.h
  shared_mutex
  span
  sstream
  stack
  stdbool.h
  stddef.h
  stdexcept
  stdint.h
  stdio.h
  stdlib.h
  streambuf
  string
  string.h
  string_view
  strstream
  system_error
  tgmath.h
  thread
  tuple
  type_traits
  typeindex
  typeinfo
  unordered_map
  unordered_set
  utility
  valarray
  variant
  vector
  version
  wchar.h
  wctype.h
  )

if(LIBCXX_INSTALL_SUPPORT_HEADERS)
  set(files
    ${files}
    support/android/locale_bionic.h
    support/fuchsia/xlocale.h
    support/ibm/limits.h
    support/ibm/locale_mgmt_aix.h
    support/ibm/support.h
    support/ibm/xlocale.h
    support/musl/xlocale.h
    support/newlib/xlocale.h
    support/solaris/floatingpoint.h
    support/solaris/wchar.h
    support/solaris/xlocale.h
    support/win32/limits_msvc_win32.h
    support/win32/locale_win32.h
    support/xlocale/__nop_locale_mgmt.h
    support/xlocale/__posix_l_fallback.h
    support/xlocale/__strtonum_fallback.h
    )
endif()

if (LIBCXX_NEEDS_SITE_CONFIG)
  # Generate a custom __config header. The new header is created
  # by prepending __config_site to the current __config header.
  add_custom_command(OUTPUT ${LIBCXX_BINARY_DIR}/__generated_config
    COMMAND ${PYTHON_EXECUTABLE} ${LIBCXX_SOURCE_DIR}/utils/cat_files.py
      ${LIBCXX_BINARY_DIR}/__config_site
      ${LIBCXX_SOURCE_DIR}/include/__config
      -o ${LIBCXX_BINARY_DIR}/__generated_config
    DEPENDS ${LIBCXX_SOURCE_DIR}/include/__config
            ${LIBCXX_BINARY_DIR}/__config_site
  )
  # Add a target that executes the generation commands.
  add_custom_target(cxx-generated-config ALL
    DEPENDS ${LIBCXX_BINARY_DIR}/__generated_config)
  set(generated_config_deps cxx-generated-config)
else()
  set(files
    ${files}
    __config
    )
endif()

# In some build configuraitons (like boostrapping clang), we need to be able to
# install the libcxx headers before CMake configuraiton for libcxx runs. Making
# the name of this target configurable allows LLVM/runtimes/CMakeLists.txt to
# add this subdirectory to the LLVM build to put libcxx's headers in place
# before libcxx's build configuration is run.
if (NOT CXX_HEADER_TARGET)
  set(CXX_HEADER_TARGET cxx-headers)
endif()
if(NOT LIBCXX_USING_INSTALLED_LLVM AND LIBCXX_HEADER_DIR)
  set(output_dir ${LIBCXX_HEADER_DIR}/include/c++/v1)

  set(out_files)
  foreach(f ${files})
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${f})
    set(dst ${output_dir}/${f})
    add_custom_command(OUTPUT ${dst}
      DEPENDS ${src}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
      COMMENT "Copying CXX header ${f}")
    list(APPEND out_files ${dst})
  endforeach()

  if (LIBCXX_NEEDS_SITE_CONFIG)
    # Copy the generated header as __config into build directory.
    set(src ${LIBCXX_BINARY_DIR}/__generated_config)
    set(dst ${output_dir}/__config)
    add_custom_command(OUTPUT ${dst}
        DEPENDS ${src} ${generated_config_deps}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
        COMMENT "Copying CXX __config")
    list(APPEND out_files ${dst})
  endif()

  add_custom_target(${CXX_HEADER_TARGET} ALL DEPENDS ${out_files} ${LIBCXX_CXX_ABI_HEADER_TARGET})
else()
  add_custom_target(${CXX_HEADER_TARGET})
endif()
set_target_properties(${CXX_HEADER_TARGET} PROPERTIES FOLDER "Misc")

if (LIBCXX_INSTALL_HEADERS)
  foreach(file ${files})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file}
      DESTINATION ${LIBCXX_INSTALL_HEADER_PREFIX}include/c++/v1/${dir}
      COMPONENT ${CXX_HEADER_TARGET}
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
    )
  endforeach()

  if (LIBCXX_NEEDS_SITE_CONFIG)
    # Install the generated header as __config.
    install(FILES ${LIBCXX_BINARY_DIR}/__generated_config
      DESTINATION ${LIBCXX_INSTALL_HEADER_PREFIX}include/c++/v1
      PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
      RENAME __config
      COMPONENT ${CXX_HEADER_TARGET})
  endif()

  if (NOT CMAKE_CONFIGURATION_TYPES)
    add_custom_target(install-${CXX_HEADER_TARGET}
                      DEPENDS ${CXX_HEADER_TARGET} ${generated_config_deps}
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=${CXX_HEADER_TARGET}
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    # Stripping is a no-op for headers
    add_custom_target(install-${CXX_HEADER_TARGET}-stripped DEPENDS install-${CXX_HEADER_TARGET})

    add_custom_target(install-libcxx-headers DEPENDS install-${CXX_HEADER_TARGET})
    add_custom_target(install-libcxx-headers-stripped DEPENDS install-${CXX_HEADER_TARGET}-stripped)
  endif()
endif()
