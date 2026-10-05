// Minimal pthread mutex stub for host JIT.
// Host JIT's include path has no system <pthread.h>. This declares the glibc
// entry points so a freestanding TU can call them; the dynamic loader resolves
// the symbols. Host JIT targets x86_64, where glibc and musl pthread_mutex_t
// are 40 bytes and 8-byte aligned.
#ifndef _HOSTJIT_PTHREAD_H
#define _HOSTJIT_PTHREAD_H

#if defined(__cplusplus)
#  define _HOSTJIT_ALIGNAS(n) alignas(n)
#else
#  define _HOSTJIT_ALIGNAS(n) _Alignas(n)
#endif

typedef struct
{
  _HOSTJIT_ALIGNAS(8) unsigned char __size[40];
} pthread_mutex_t;

#ifdef __cplusplus
extern "C"
{
#endif

  int pthread_mutex_lock(pthread_mutex_t*);
  int pthread_mutex_trylock(pthread_mutex_t*);
  int pthread_mutex_unlock(pthread_mutex_t*);

#ifdef __cplusplus
}
#endif

#endif // _HOSTJIT_PTHREAD_H
