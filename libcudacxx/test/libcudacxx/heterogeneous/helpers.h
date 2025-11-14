//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HETEROGENEOUS_HELPERS_H
#define HETEROGENEOUS_HELPERS_H

#include <cuda/std/type_traits>

#include <cstdio>
#include <new>
#include <thread>
#include <vector>

#include "meta.h"

#define DEFINE_ASYNC_TRAIT(...)                                             \
  template <typename T, typename = cuda::std::true_type>                    \
  struct async##__VA_ARGS__##_trait_impl                                    \
  {                                                                         \
    using type = cuda::std::false_type;                                     \
  };                                                                        \
                                                                            \
  template <typename T>                                                     \
  struct async##__VA_ARGS__##_trait_impl<T, typename T::async##__VA_ARGS__> \
  {                                                                         \
    using type = cuda::std::true_type;                                      \
  };                                                                        \
                                                                            \
  template <typename T>                                                     \
  using async##__VA_ARGS__##_trait = typename async##__VA_ARGS__##_trait_impl<T>::type;

DEFINE_ASYNC_TRAIT()
DEFINE_ASYNC_TRAIT(_initialize)
DEFINE_ASYNC_TRAIT(_validate)

#undef DEFINE_ASYNC_TRAIT

template <typename T, typename = int>
struct has_threadcount : std::false_type
{};
template <typename T>
struct has_threadcount<T, decltype((void) T::threadcount, (int) 0)> : std::true_type
{};

template <typename T, bool = has_threadcount<T>::value>
struct threadcount_trait_impl
{
  static constexpr size_t value = 1;
};

template <typename T>
struct threadcount_trait_impl<T, true>
{
  static constexpr size_t value = T::threadcount;
};

template <typename T>
using threadcount_trait = threadcount_trait_impl<T>;

#define HETEROGENEOUS_SAFE_CALL(...)                                                  \
  do                                                                                  \
  {                                                                                   \
    cudaError_t err = __VA_ARGS__;                                                    \
    if (err != cudaSuccess)                                                           \
    {                                                                                 \
      printf("CUDA ERROR: %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
      fflush(stdout);                                                                 \
      abort();                                                                        \
    }                                                                                 \
  } while (false)

__host__ inline std::vector<std::thread>& host_threads()
{
  static std::vector<std::thread> threads;
  return threads;
}

__host__ inline void sync_host_threads()
{
  for (auto&& thread : host_threads())
  {
    thread.join();
  }
  host_threads().clear();
}

__host__ inline std::vector<std::thread>& device_threads()
{
  static std::vector<std::thread> threads;
  return threads;
}

__host__ inline void sync_device_threads()
{
  for (auto&& thread : device_threads())
  {
    thread.join();
  }
  device_threads().clear();
}

__host__ void sync_all()
{
  sync_host_threads();
  sync_device_threads();
}

struct async_tester_fence
{
  template <typename T>
  __host__ __device__ static void initialize(T&)
  {}

  template <typename T>
  __host__ __device__ static void validate(T&)
  {}

  template <typename T>
  __host__ __device__ static void perform(T&)
  {}
};

template <typename... Testers>
using tester_list = type_list<Testers...>;

template <typename Tester, typename T>
__host__ __device__ void initialize(T& object)
{
  Tester::initialize(object);
}

template <typename Tester, typename T>
__host__ __device__ auto validate_impl(T& object) -> decltype(Tester::validate(object), void())
{
  Tester::validate(object);
}

template <typename, typename... Ts>
__host__ __device__ void validate_impl(Ts&&...)
{}

template <typename Tester, typename T>
__host__ __device__ void validate(T& object)
{
  validate_impl<Tester>(object);
}

template <typename T, typename... Args>
__global__ void construct_kernel(void* address, Args... args)
{
  new (address) T(args...);
}

template <typename T>
__global__ void destroy_kernel(T* object)
{
  object->~T();
}

template <typename Tester, typename T>
__global__ void initialization_kernel(T& object)
{
  initialize<Tester>(object);
}

template <typename Tester, typename T>
__global__ void validation_kernel(T& object)
{
  validate<Tester>(object);
}

template <typename T, typename... Args>
T* device_construct(void* address, Args... args)
{
  construct_kernel<T><<<1, 1>>>(address, args...);
  HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
  HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
  return reinterpret_cast<T*>(address);
}

template <typename T>
void device_destroy(T* object)
{
  destroy_kernel<<<1, 1>>>(object);
  HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
  HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
}
template <typename Fn>
void device_launch_async(Fn& launcher)
{
  auto streamManager = [launcher]() {
    cudaStream_t stream;
    HETEROGENEOUS_SAFE_CALL(cudaStreamCreate(&stream));
    launcher(stream);
    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());

    HETEROGENEOUS_SAFE_CALL(cudaStreamSynchronize(stream));
    HETEROGENEOUS_SAFE_CALL(cudaStreamDestroy(stream));
  };

  device_threads().push_back(std::thread(streamManager));
}

template <typename Tester, typename T>
void device_initialize(T& object)
{
#ifdef DEBUG_TESTERS
  printf("    %s\n", __PRETTY_FUNCTION__);
  fflush(stdout);
#endif

  auto kernel_launcher = [&object](cudaStream_t stream) {
    constexpr auto tc = threadcount_trait<Tester>::value;
#ifdef DEBUG_TESTERS
    printf("      %i device init threads launched\r\n", (int) tc);
    fflush(stdout);
#endif
    initialization_kernel<Tester><<<1, tc, 0, stream>>>(object);
  };

  device_launch_async(kernel_launcher);

  if (!async_initialize_trait<Tester>::value)
  {
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
    sync_all();
  }
}

template <typename Tester, typename T>
void device_validate(T& object)
{
#ifdef DEBUG_TESTERS
  printf("    %s\n", __PRETTY_FUNCTION__);
  fflush(stdout);
#endif

  auto kernel_launcher = [&object](cudaStream_t stream) {
    constexpr auto tc = threadcount_trait<Tester>::value;
#ifdef DEBUG_TESTERS
    printf("     %i device validate threads launched\r\n", (int) tc);
    fflush(stdout);
#endif
    validation_kernel<Tester><<<1, tc, 0, stream>>>(object);
  };

  device_launch_async(kernel_launcher);

  if (!async_validate_trait<Tester>::value)
  {
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
    sync_all();
  }
}

template <typename Tester, typename T>
void host_initialize(T& object)
{
#ifdef DEBUG_TESTERS
  printf("    %s\n", __PRETTY_FUNCTION__);
  fflush(stdout);
#endif

  constexpr auto tc = threadcount_trait<Tester>::value;
#ifdef DEBUG_TESTERS
  printf("      %i host init threads launched\r\n", (int) tc);
  fflush(stdout);
#endif

  for (size_t i = 0; i < tc; i++)
  {
    host_threads().emplace_back([&] {
      initialize<Tester>(object);
    });
  }

  if (!async_initialize_trait<Tester>::value)
  {
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
    sync_all();
  }
}

template <typename Tester, typename T>
void host_validate(T& object)
{
#ifdef DEBUG_TESTERS
  printf("    %s\n", __PRETTY_FUNCTION__);
  fflush(stdout);
#endif

  constexpr auto tc = threadcount_trait<Tester>::value;
#ifdef DEBUG_TESTERS
  printf("      %i host validate threads launched\r\n", (int) tc);
  fflush(stdout);
#endif

  for (size_t i = 0; i < tc; i++)
  {
    host_threads().emplace_back([&] {
      validate<Tester>(object);
    });
  }

  if (!async_initialize_trait<Tester>::value)
  {
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
    sync_all();
  }
}

template <typename T>
using performer = void (*)(T&);

template <typename T>
struct initializer_validator
{
  performer<T> initializer;
  performer<T> validator;
};

template <typename T>
struct host_launcher
{
  template <typename Tester>
  static initializer_validator<T> get_exec()
  {
    return initializer_validator<T>{host_initialize<Tester>, host_validate<Tester>};
  }
};

template <typename T>
struct device_launcher
{
  template <typename Tester>
  static initializer_validator<T> get_exec()
  {
    return initializer_validator<T>{device_initialize<Tester>, device_validate<Tester>};
  }
};

template <typename T, typename... Testers, typename... Launchers>
void do_heterogeneous_test(T* test_input, type_list<Testers...>, type_list<Launchers...>)
{
  initializer_validator<T> performers[] = {{Launchers::template get_exec<Testers>()}...};

  for (auto&& performer : performers)
  {
    performer.initializer(*test_input);
    performer.validator(*test_input);
  }

  HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
  HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());

  sync_all();
}

template <size_t Idx>
using enable_if_permutations_remain = typename std::enable_if<Idx != 0, int>::type;
template <size_t Idx>
using enable_if_no_permutations_remain = typename std::enable_if<Idx == 0, int>::type;

template <size_t Idx, typename Fn, typename Launchers, enable_if_permutations_remain<Idx> = 0>
void permute_tests(const Fn& fn, Launchers launchers)
{
#ifdef DEBUG_TESTERS
  printf("  Testing permutation %zd (%s)\r\n", Idx, __PRETTY_FUNCTION__);
  fflush(stdout);
#endif
  fn(launchers);
  permute_tests<Idx - 1>(fn, rotl<Launchers>{});
}

template <size_t Idx, typename Fn, typename Launchers, enable_if_no_permutations_remain<Idx> = 0>
void permute_tests(const Fn&, Launchers)
{}

template <typename Fn, typename... Launchers>
void permute_tests(const Fn& fn, type_list<Launchers...> launchers)
{
  permute_tests<sizeof...(Launchers)>(fn, launchers);
}

template <typename Testers, typename InputCreator, typename InputDestructor>
struct test_wrapper
{
  InputCreator creator;
  InputDestructor destructor;

  template <typename Launchers>
  void operator()(Launchers) const
  {
    auto input = creator();
    do_heterogeneous_test(input, Testers{}, Launchers{});
    destructor(input);
  }
};

template <typename T, typename... Testers, typename... Args>
void validate_device_dynamic(tester_list<Testers...> testers, Args... args)
{
  auto test_input_creator = [args...]() -> T* {
    void* pointer = nullptr;
    HETEROGENEOUS_SAFE_CALL(cudaMallocHost(&pointer, sizeof(T)));
    return device_construct<T>(pointer, args...);
  };

  auto test_input_destructor = [](T* test_input) {
    device_destroy(test_input);
    HETEROGENEOUS_SAFE_CALL(cudaFreeHost(test_input));
  };

  test_wrapper<tester_list<Testers...>, decltype(test_input_creator), decltype(test_input_destructor)> test_harness{
    test_input_creator, test_input_destructor};

  // ex: type_list<device_launcher, host_launcher, host_launcher>
  using initial_launcher_list = append_n<sizeof...(Testers) - 1, type_list<device_launcher<T>>, host_launcher<T>>;
#ifdef DEBUG_TESTERS
  printf("Launching %zd permutations\r\n", sizeof...(Testers));
  fflush(stdout);
#endif
  permute_tests(test_harness, initial_launcher_list{});
}

template <typename T>
struct manual_object
{
  template <typename... Args>
  void construct(Args... args)
  {
    new (static_cast<void*>(&data.object)) T(args...);
  }
  template <typename... Args>
  void device_construct(Args... args)
  {
    ::device_construct<T>(&data.object, args...);
  }

  void destroy()
  {
    data.object.~T();
  }
  void device_destroy()
  {
    ::device_destroy(&data.object);
  }

  T& get()
  {
    return data.object;
  }

  union data
  {
    __host__ __device__ constexpr data() noexcept
        : dummy(){};
    char dummy = {};
    T object;
  } data;

  constexpr manual_object() noexcept {}
};

template <typename T>
__managed__ manual_object<T> managed_variable{};

template <typename Creator, typename Destroyer, typename Validator, std::size_t N>
void validate_in_managed_memory_helper(const Creator& creator, const Destroyer& destroyer, Validator (&performers)[N])
{
  auto object = creator();

  for (auto&& performer : performers)
  {
    performer.initializer(*object);
    performer.validator(*object);
  }

  HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
  HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());

  sync_all();

  destroyer(object);
}

template <typename T, typename... Testers, typename... Args>
void validate_managed(tester_list<Testers...>, Args... args)
{
  initializer_validator<T> host_init_device_check[] = {{host_initialize<Testers>, device_validate<Testers>}...};

  initializer_validator<T> device_init_host_check[] = {{device_initialize<Testers>, host_validate<Testers>}...};

  auto host_constructor = [args...]() -> T* {
    void* pointer;
    HETEROGENEOUS_SAFE_CALL(cudaMallocManaged(&pointer, sizeof(T)));
    return new (pointer) T(args...);
  };

  auto device_constructor = [args...]() -> T* {
    void* pointer;
    HETEROGENEOUS_SAFE_CALL(cudaMallocManaged(&pointer, sizeof(T)));
    return device_construct<T>(pointer, args...);
  };

  auto host_destructor = [](T* object) {
    object->~T();
    HETEROGENEOUS_SAFE_CALL(cudaFree(object));
  };

  auto device_destructor = [](T* object) {
    device_destroy(object);
    HETEROGENEOUS_SAFE_CALL(cudaFree(object));
  };

  validate_in_managed_memory_helper(host_constructor, host_destructor, host_init_device_check);
  validate_in_managed_memory_helper(host_constructor, host_destructor, device_init_host_check);
  validate_in_managed_memory_helper(host_constructor, device_destructor, host_init_device_check);
  validate_in_managed_memory_helper(host_constructor, device_destructor, device_init_host_check);
  validate_in_managed_memory_helper(device_constructor, host_destructor, host_init_device_check);
  validate_in_managed_memory_helper(device_constructor, host_destructor, device_init_host_check);
  validate_in_managed_memory_helper(device_constructor, device_destructor, host_init_device_check);
  validate_in_managed_memory_helper(device_constructor, device_destructor, device_init_host_check);

#if !defined(__clang__)
  // The managed variable template part of this test is disabled under clang, pending nvbug 2790305 being fixed.

  auto host_variable_constructor = [args...]() -> T* {
    managed_variable<T>.construct(args...);
    return &managed_variable<T>.get();
  };

  auto device_variable_constructor = [args...]() -> T* {
    managed_variable<T>.device_construct(args...);
    return &managed_variable<T>.get();
  };

  auto host_variable_destructor = [](T*) {
    managed_variable<T>.destroy();
  };

  auto device_variable_destructor = [](T*) {
    managed_variable<T>.device_destroy();
  };

  validate_in_managed_memory_helper(host_variable_constructor, host_variable_destructor, host_init_device_check);
  validate_in_managed_memory_helper(host_variable_constructor, host_variable_destructor, device_init_host_check);
  validate_in_managed_memory_helper(host_variable_constructor, device_variable_destructor, host_init_device_check);
  validate_in_managed_memory_helper(host_variable_constructor, device_variable_destructor, device_init_host_check);
  validate_in_managed_memory_helper(device_variable_constructor, host_variable_destructor, host_init_device_check);
  validate_in_managed_memory_helper(device_variable_constructor, host_variable_destructor, device_init_host_check);
  validate_in_managed_memory_helper(device_variable_constructor, device_variable_destructor, host_init_device_check);
  validate_in_managed_memory_helper(device_variable_constructor, device_variable_destructor, device_init_host_check);
#endif
}

bool check_managed_memory_support(bool is_async)
{
  int current_device, property_value;
  HETEROGENEOUS_SAFE_CALL(cudaGetDevice(&current_device));
  HETEROGENEOUS_SAFE_CALL(cudaDeviceGetAttribute(
    &property_value, is_async ? cudaDevAttrConcurrentManagedAccess : cudaDevAttrManagedMemory, current_device));
  return property_value == 1;
}

struct dummy_tester
{
  template <typename... Ts>
  __host__ __device__ static void initialize(Ts&&...)
  {}

  template <typename... Ts>
  __host__ __device__ static void validate(Ts&&...)
  {}
};

template <bool PerformerCombinations, typename List>
struct validate_list
{
  using type = List;
};

template <bool PerformerCombinations>
struct validate_list<PerformerCombinations, tester_list<>>
{
  using type = tester_list<dummy_tester>;
};

template <bool... Values>
struct any_of;

template <bool... Tail>
struct any_of<true, Tail...> : cuda::std::true_type
{};

template <bool... Tail>
struct any_of<false, Tail...> : any_of<Tail...>
{};

template <>
struct any_of<> : cuda::std::false_type
{};

template <typename TesterList>
struct is_tester_list_async;

template <typename... Testers>
struct is_tester_list_async<tester_list<Testers...>>
    : any_of<async_initialize_trait<Testers>::value..., async_validate_trait<Testers>::value...>
{};

template <typename T, typename TesterList, typename... Args>
void validate_pinned(Args... args)
{
  using list_t = typename validate_list<false, TesterList>::type;
  list_t list0;
#ifdef DEBUG_TESTERS
  printf("%s\n", "Launching permuted H/D tests");
  fflush(stdout);
#endif
  validate_device_dynamic<T>(list0, args...);

  if (check_managed_memory_support(is_tester_list_async<list_t>::value))
  {
#ifdef DEBUG_TESTERS
    printf("%s\n", "Launching mixed H/D tests");
    fflush(stdout);
#endif
    typename validate_list<true, TesterList>::type list1;
    validate_managed<T>(list1, args...);
  }
}

enum class performer_side
{
  initialize,
  validate
};

template <typename Performer, performer_side>
struct performer_adapter;

template <typename Performer>
struct performer_adapter<Performer, performer_side::initialize>
{
  using async_initialize = async_trait<Performer>;
  using async_validate   = async_trait<Performer>;

  static constexpr auto threadcount = threadcount_trait<Performer>::value;

  template <typename T>
  __host__ __device__ static void initialize(T& t)
  {
    Performer::perform(t);
  }
};

template <typename Performer>
struct performer_adapter<Performer, performer_side::validate>
{
  using async_initialize = async_trait<Performer>;
  using async_validate   = async_trait<Performer>;

  static constexpr auto threadcount = threadcount_trait<Performer>::value;

  template <typename T>
  __host__ __device__ static void initialize(T&)
  {}

  template <typename T>
  __host__ __device__ static void validate(T& t)
  {
    Performer::perform(t);
  }
};

template <typename... Ts>
struct performer_list
{};

template <typename Front, typename... Performers>
struct generate_variants_t;

template <typename... Fronts>
struct generate_variants_t<tester_list<Fronts...>>
{
  using type = tester_list<Fronts..., async_tester_fence>;
};

template <typename... Fronts, typename First, typename... Performers>
struct generate_variants_t<tester_list<Fronts...>, First, Performers...>
{
  using type =
    append<typename generate_variants_t<tester_list<Fronts..., performer_adapter<First, performer_side::initialize>>,
                                        Performers...>::type,
           typename generate_variants_t<tester_list<Fronts..., performer_adapter<First, performer_side::validate>>,
                                        Performers...>::type>;
};

template <typename Front, typename... Performers>
using generate_variants = typename generate_variants_t<Front, Performers...>::type;

template <typename First, typename... Rest>
struct validate_list<true, performer_list<First, Rest...>>
{
  using type = generate_variants<
    // Lock the first performer to initialize only.
    // Otherwise, here's what would get generated (for 2 performers):
    //
    // ii, iv, vi, vv (i - initialize, v - validate)
    //
    // but because the testing process itself is symmetrical, that means that
    // testing for `ii` also covers `vv`, and testing for `iv` also covers `vi`.
    // Therefore, only the first half of the generated sequence is actually
    // meaningfully useful.
    tester_list<performer_adapter<First, performer_side::initialize>>,
    Rest...>;
};

template <typename... All>
struct validate_list<false, performer_list<All...>>
{
  using type = tester_list<performer_adapter<All, performer_side::initialize>...>;
};

#endif
