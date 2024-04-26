#ifndef __LIBCUDACXX_ATOMIC_SCOPES_H
#define __LIBCUDACXX_ATOMIC_SCOPES_H

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// REMEMBER CHANGES TO THESE ARE ABI BREAKING
// TODO: Space values out for potential new scopes
#ifndef __ATOMIC_BLOCK
#define __ATOMIC_SYSTEM 0 // 0 indicates default
#define __ATOMIC_DEVICE 1
#define __ATOMIC_BLOCK 2
#define __ATOMIC_THREAD 10
#endif //__ATOMIC_BLOCK

enum thread_scope {
    thread_scope_system = __ATOMIC_SYSTEM,
    thread_scope_device = __ATOMIC_DEVICE,
    thread_scope_block = __ATOMIC_BLOCK,
    thread_scope_thread = __ATOMIC_THREAD
};

#define _LIBCUDACXX_ATOMIC_SCOPE_TYPE ::cuda::thread_scope
#define _LIBCUDACXX_ATOMIC_SCOPE_DEFAULT ::cuda::thread_scope::system

struct __thread_scope_thread_tag { };
struct __thread_scope_block_tag { };
struct __thread_scope_device_tag { };
struct __thread_scope_system_tag { };

template<int _Scope>  struct __scope_enum_to_tag { };
/* This would be the implementation once an actual thread-scope backend exists.
template<> struct __scope_enum_to_tag<(int)thread_scope_thread> {
    using type = __thread_scope_thread_tag; };
Until then: */
template<>
struct __scope_enum_to_tag<(int)thread_scope_thread> {
    using __tag = __thread_scope_block_tag;
};
template<>
struct __scope_enum_to_tag<(int)thread_scope_block> {
    using __tag = __thread_scope_block_tag;
};
template<>
struct __scope_enum_to_tag<(int)thread_scope_device> {
    using __tag = __thread_scope_device_tag;
};
template<>
struct __scope_enum_to_tag<(int)thread_scope_system> {
    using __tag = __thread_scope_system_tag;
};

template <int _Scope>
using __scope_to_tag = typename __scope_enum_to_tag<_Scope>::__tag;

_LIBCUDACXX_END_NAMESPACE_STD

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

using _CUDA_VSTD::thread_scope;
using _CUDA_VSTD::thread_scope_block;
using _CUDA_VSTD::thread_scope_device;
using _CUDA_VSTD::thread_scope_system;
using _CUDA_VSTD::thread_scope_thread;

using _CUDA_VSTD::__thread_scope_block_tag;
using _CUDA_VSTD::__thread_scope_device_tag;
using _CUDA_VSTD::__thread_scope_system_tag;

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // __LIBCUDACXX_ATOMIC_SCOPES_H
