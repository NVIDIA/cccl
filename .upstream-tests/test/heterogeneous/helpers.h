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

#include <new>
#include <thread>
#include <vector>
#include <stdlib.h>

#define DEFINE_ASYNC_TRAIT(...) \
    template<typename T, typename = cuda::std::true_type> \
    struct async ## __VA_ARGS__ ## _trait_impl \
    { \
        using type = cuda::std::false_type; \
    }; \
    \
    template<typename T> \
    struct async ## __VA_ARGS__ ## _trait_impl<T, typename T::async ## __VA_ARGS__> \
    { \
        using type = cuda::std::true_type; \
    }; \
    \
    template<typename T> \
    using async ## __VA_ARGS__ ## _trait = typename async ## __VA_ARGS__ ## _trait_impl<T>::type;

DEFINE_ASYNC_TRAIT()
DEFINE_ASYNC_TRAIT(_initialize)
DEFINE_ASYNC_TRAIT(_validate)

#undef DEFINE_ASYNC_TRAIT

#define HETEROGENEOUS_SAFE_CALL(...) \
    do { \
        cudaError_t err = __VA_ARGS__; \
        if (err != cudaSuccess) \
        { \
            printf("CUDA ERROR: %s: %s\n", \
                cudaGetErrorName(err), cudaGetErrorString(err)); \
            abort(); \
        } \
    } while (false)

__host__
inline std::vector<std::thread> & host_threads()
{
    static std::vector<std::thread> threads;
    return threads;
}

__host__
inline void sync_host_threads()
{
    for (auto && thread : host_threads())
    {
        thread.join();
    }
    host_threads().clear();
}

__host__
inline std::vector<cudaStream_t> & device_streams()
{
    static std::vector<cudaStream_t> streams;
    return streams;
}

__host__
inline void sync_device_streams()
{
    for (auto && stream : device_streams())
    {
        HETEROGENEOUS_SAFE_CALL(cudaStreamSynchronize(stream));
        HETEROGENEOUS_SAFE_CALL(cudaStreamDestroy(stream));
    }

    device_streams().clear();
}

__host__
void sync_all()
{
    sync_host_threads();
    sync_device_streams();
}

struct async_tester_fence
{
    template<typename T>
    __host__ __device__
    static void initialize(T &)
    {
    }

    template<typename T>
    __host__ __device__
    static void validate(T &)
    {
    }

    template<typename T>
    __host__ __device__
    static void perform(T &)
    {
    }
};

template<typename ...Testers>
struct tester_list
{
};

template<typename TesterList, typename ...Testers>
struct extend_tester_list_t;

template<typename ...Original, typename ...Additional>
struct extend_tester_list_t<tester_list<Original...>, Additional...>
{
    using type = tester_list<Original..., Additional...>;
};

template<typename TesterList, typename ...Testers>
using extend_tester_list = typename extend_tester_list_t<TesterList, Testers...>::type;

template<typename Tester, typename T>
__host__ __device__
void initialize(T & object)
{
    Tester::initialize(object);
}

template<typename Tester, typename T>
__host__ __device__
auto validate_impl(T & object)
    -> decltype(Tester::validate(object), void())
{
    Tester::validate(object);
}

template<typename, typename ...Ts>
__host__ __device__
void validate_impl(Ts && ...)
{
}

template<typename Tester, typename T>
__host__ __device__
void validate(T & object)
{
    validate_impl<Tester>(object);
}

template<typename T, typename ...Args>
__global__
void construct_kernel(void * address, Args ...args)
{
    new (address) T(args...);
}

template<typename T>
__global__
void destroy_kernel(T * object)
{
    object->~T();
}

template<typename Tester, typename T>
__global__
void initialization_kernel(T & object)
{
    initialize<Tester>(object);
}

template<typename Tester, typename T>
__global__
void validation_kernel(T & object)
{
    validate<Tester>(object);
}

template<typename T, typename ...Args>
T * device_construct(void * address, Args... args)
{
    construct_kernel<T><<<1, 1>>>(address, args...);
    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
    return reinterpret_cast<T *>(address);
}

template<typename T>
void device_destroy(T * object)
{
    destroy_kernel<<<1, 1>>>(object);
    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
}

template<typename Tester, typename T>
void device_initialize(T & object)
{
#ifdef DEBUG_TESTERS
    printf("%s\n", __PRETTY_FUNCTION__);
    fflush(stdout);
#endif

    cudaStream_t s;
    HETEROGENEOUS_SAFE_CALL(cudaStreamCreate(&s));
    initialization_kernel<Tester><<<1, 1, 0, s>>>(object);
    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
    device_streams().push_back(s);

    if (!async_initialize_trait<Tester>::value)
    {
        HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
        sync_all();
    }
}

template<typename Tester, typename T>
void device_validate(T & object)
{
#ifdef DEBUG_TESTERS
    printf("%s\n", __PRETTY_FUNCTION__);
    fflush(stdout);
#endif

    cudaStream_t s;
    HETEROGENEOUS_SAFE_CALL(cudaStreamCreate(&s));
    validation_kernel<Tester><<<1, 1, 0, s>>>(object);
    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
    device_streams().push_back(s);

    if (!async_validate_trait<Tester>::value)
    {
        HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
        sync_all();
    }
}

template<typename Tester, typename T>
void host_initialize(T & object)
{
#ifdef DEBUG_TESTERS
    printf("%s\n", __PRETTY_FUNCTION__);
    fflush(stdout);
#endif

    if (async_initialize_trait<Tester>::value)
    {
        host_threads().emplace_back([&]{ initialize<Tester>(object); });
    }

    else
    {
        initialize<Tester>(object);
        sync_all();
    }
}

template<typename Tester, typename T>
void host_validate(T & object)
{
#ifdef DEBUG_TESTERS
    printf("%s\n", __PRETTY_FUNCTION__);
    fflush(stdout);
#endif

    if (async_validate_trait<Tester>::value)
    {
        host_threads().emplace_back([&]{ validate<Tester>(object); });
    }

    else
    {
        validate<Tester>(object);
        sync_all();
    }
}

template<typename T, typename ...Args>
using creator = T & (*)(Args...);

template<typename T>
using performer = void (*)(T &);

template<typename T>
struct initializer_validator
{
    performer<T> initializer;
    performer<T> validator;
};

template<typename T, typename ...Testers, typename ...Args>
void validate_device_dynamic(tester_list<Testers...>, Args ...args)
{
    void * pointer;
    HETEROGENEOUS_SAFE_CALL(cudaMalloc(&pointer, sizeof(T)));

    T & object = *device_construct<T>(pointer, args...);

    initializer_validator<T> performers[] = {
        {
            device_initialize<Testers>,
            device_validate<Testers>
        }...
    };

    for (auto && performer : performers)
    {
        performer.initializer(object);
        performer.validator(object);
    }

    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());

    sync_all();

    device_destroy(&object);
    HETEROGENEOUS_SAFE_CALL(cudaFree(pointer));
}

#if __cplusplus >= 201402L
template<typename T>
struct manual_object
{
    template<typename ...Args>
    void construct(Args ...args) { new (static_cast<void *>(&data.object)) T(args...); }
    template<typename ...Args>
    void device_construct(Args ...args) { ::device_construct<T>(&data.object, args...); }

    void destroy() { data.object.~T(); }
    void device_destroy() { ::device_destroy(&data.object); }

    T & get() { return data.object; }

    union data
    {
        __host__ __device__
        data() {}

        __host__ __device__
        ~data() {}

        T object;
    } data;
};

template<typename T>
__managed__ manual_object<T> managed_variable;
#endif

template<typename T, std::size_t N, typename ...Args>
void validate_in_managed_memory_helper(
    creator<T, Args...> creator_,
    performer<T> destroyer,
    initializer_validator<T> (&performers)[N],
    Args ...args
)
{
    T & object = creator_(args...);

    for (auto && performer : performers)
    {
        performer.initializer(object);
        performer.validator(object);
    }

    HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
    HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());

    sync_all();

    destroyer(object);
}

template<typename T, typename ...Testers, typename ...Args>
void validate_managed(tester_list<Testers...>, Args ...args)
{
    initializer_validator<T> host_init_device_check[] = {
        {
            host_initialize<Testers>,
            device_validate<Testers>
        }...
    };

    initializer_validator<T> device_init_host_check[] = {
        {
            device_initialize<Testers>,
            host_validate<Testers>
        }...
    };

    creator<T, Args...> host_constructor = [](Args ...args) -> T & {
        void * pointer;
        HETEROGENEOUS_SAFE_CALL(cudaMallocManaged(&pointer, sizeof(T)));
        return *new (pointer) T(args...);
    };

    creator<T, Args...> device_constructor = [](Args ...args) -> T & {
        void * pointer;
        HETEROGENEOUS_SAFE_CALL(cudaMallocManaged(&pointer, sizeof(T)));
        return *device_construct<T>(pointer, args...);
    };

    performer<T> host_destructor = [](T & object){
        object.~T();
        HETEROGENEOUS_SAFE_CALL(cudaFree(&object));
    };

    performer<T> device_destructor = [](T & object){
        device_destroy(&object);
        HETEROGENEOUS_SAFE_CALL(cudaFree(&object));
    };

    validate_in_managed_memory_helper<T>(host_constructor, host_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(host_constructor, host_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(host_constructor, device_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(host_constructor, device_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(device_constructor, host_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(device_constructor, host_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(device_constructor, device_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(device_constructor, device_destructor, device_init_host_check, args...);

#if __cplusplus >= 201402L && !defined(__clang__)
    // The managed variable template part of this test is disabled under clang, pending nvbug 2790305 being fixed.

    creator<T, Args...> host_variable_constructor = [](Args ...args) -> T & {
        managed_variable<T>.construct(args...);
        return managed_variable<T>.get();
    };

    creator<T, Args...> device_variable_constructor = [](Args ...args) -> T & {
        managed_variable<T>.device_construct(args...);
        return managed_variable<T>.get();
    };

    performer<T> host_variable_destructor = [](T &){
        managed_variable<T>.destroy();
    };

    performer<T> device_variable_destructor = [](T &){
        managed_variable<T>.device_destroy();
    };

    validate_in_managed_memory_helper<T>(host_variable_constructor, host_variable_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(host_variable_constructor, host_variable_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(host_variable_constructor, device_variable_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(host_variable_constructor, device_variable_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(device_variable_constructor, host_variable_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(device_variable_constructor, host_variable_destructor, device_init_host_check, args...);
    validate_in_managed_memory_helper<T>(device_variable_constructor, device_variable_destructor, host_init_device_check, args...);
    validate_in_managed_memory_helper<T>(device_variable_constructor, device_variable_destructor, device_init_host_check, args...);
#endif
}

bool check_managed_memory_support(bool is_async)
{
    int current_device, property_value;
    HETEROGENEOUS_SAFE_CALL(cudaGetDevice(&current_device));
    HETEROGENEOUS_SAFE_CALL(cudaDeviceGetAttribute(&property_value,
        is_async ? cudaDevAttrConcurrentManagedAccess : cudaDevAttrManagedMemory,
        current_device));
    return property_value == 1;
}

struct dummy_tester
{
    template<typename ...Ts>
    __host__ __device__
    static void initialize(Ts &&...) {}

    template<typename ...Ts>
    __host__ __device__
    static void validate(Ts &&...) {}
};

template<bool PerformerCombinations, typename List>
struct validate_list
{
    using type = List;
};

template<bool PerformerCombinations>
struct validate_list<PerformerCombinations, tester_list<>>
{
    using type = tester_list<dummy_tester>;
};

template<bool ...Values>
struct any_of;

template<bool ...Tail>
struct any_of<true, Tail...> : cuda::std::true_type {};

template<bool ...Tail>
struct any_of<false, Tail...> : any_of<Tail...> {};

template<>
struct any_of<> : cuda::std::false_type {};

template<typename TesterList>
struct is_tester_list_async;

template<typename ...Testers>
struct is_tester_list_async<tester_list<Testers...>>
    : any_of<async_initialize_trait<Testers>::value..., async_validate_trait<Testers>::value...>
{
};

template<typename T, typename TesterList, typename ...Args>
void validate_not_movable(Args ...args)
{
    using list_t = typename validate_list<false, TesterList>::type;
    list_t list0;
    validate_device_dynamic<T>(list0, args...);

    if (check_managed_memory_support(is_tester_list_async<list_t>::value))
    {
        typename validate_list<true, TesterList>::type list1;
        validate_managed<T>(list1, args...);
    }
}

enum class performer_side
{
    initialize,
    validate
};

template<typename Performer, performer_side>
struct performer_adapter;

template<typename Performer>
struct performer_adapter<Performer, performer_side::initialize>
{
    using async_initialize = async_trait<Performer>;
    using async_validate = async_trait<Performer>;

    template<typename T>
    __host__ __device__
    static void initialize(T & t)
    {
        Performer::perform(t);
    }
};

template<typename Performer>
struct performer_adapter<Performer, performer_side::validate>
{
    using async_initialize = async_trait<Performer>;
    using async_validate = async_trait<Performer>;

    template<typename T>
    __host__ __device__
    static void initialize(T &)
    {
    }

    template<typename T>
    __host__ __device__
    static void validate(T & t)
    {
        Performer::perform(t);
    }
};

template<typename ...Ts>
struct performer_list
{
};

template<typename ...Lists>
struct cat_tester_lists_t;

template<typename ...Only>
struct cat_tester_lists_t<
    tester_list<Only...>
>
{
    using type = tester_list<Only...>;
};

template<typename ...First, typename ...Second, typename ...Tail>
struct cat_tester_lists_t<
    tester_list<First...>,
    tester_list<Second...>,
    Tail...
>
{
    using type = typename cat_tester_lists_t<
        tester_list<First..., Second...>,
        Tail...
    >::type;
};

template<typename ...Lists>
using cat_tester_lists = typename cat_tester_lists_t<Lists...>::type;

template<typename Front, typename ...Performers>
struct generate_variants_t;

template<typename ...Fronts>
struct generate_variants_t<tester_list<Fronts...>>
{
    using type = tester_list<Fronts..., async_tester_fence>;
};

template<typename ...Fronts, typename First, typename ...Performers>
struct generate_variants_t<tester_list<Fronts...>, First, Performers...>
{
    using type = cat_tester_lists<
        typename generate_variants_t<
            tester_list<
                Fronts...,
                performer_adapter<First, performer_side::initialize>
            >,
            Performers...
        >::type,
        typename generate_variants_t<
            tester_list<
                Fronts...,
                performer_adapter<First, performer_side::validate>
            >,
            Performers...
        >::type
    >;
};

template<typename Front, typename ...Performers>
using generate_variants = typename generate_variants_t<Front, Performers...>::type;

template<typename First, typename ...Rest>
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
        Rest...
    >;
};

template<typename ...All>
struct validate_list<false, performer_list<All...>>
{
    using type = tester_list<performer_adapter<All, performer_side::initialize>...>;
};

#endif
