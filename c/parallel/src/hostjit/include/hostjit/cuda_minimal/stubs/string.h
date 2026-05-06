#ifndef _HOSTJIT_STRING_H
#define _HOSTJIT_STRING_H

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
inline void* memcpy(void* __s1, const void* __s2, size_t __n)
{
  return __builtin_memcpy(__s1, __s2, __n);
}
inline void* memmove(void* __s1, const void* __s2, size_t __n)
{
  return __builtin_memmove(__s1, __s2, __n);
}
inline int memcmp(const void* __s1, const void* __s2, size_t __n)
{
  return __builtin_memcmp(__s1, __s2, __n);
}
inline char* strchr(char* __s, int __c)
{
  return __builtin_strchr(__s, __c);
}
inline char* strpbrk(char* __s1, const char* __s2)
{
  return __builtin_strpbrk(__s1, __s2);
}
inline char* strrchr(char* __s, int __c)
{
  return __builtin_strrchr(__s, __c);
}
inline void* memchr(void* __s, int __c, size_t __n)
{
  return __builtin_memchr(__s, __c, __n);
}
inline char* strstr(char* __s1, const char* __s2)
{
  return __builtin_strstr(__s1, __s2);
}
inline char* strcpy(char* __s1, const char* __s2)
{
  return __builtin_strcpy(__s1, __s2);
}
inline char* strncpy(char* __s1, const char* __s2, size_t __n)
{
  return __builtin_strncpy(__s1, __s2, __n);
}
inline int strcmp(const char* __s1, const char* __s2)
{
  return __builtin_strcmp(__s1, __s2);
}
inline int strncmp(const char* __s1, const char* __s2, size_t __n)
{
  return __builtin_strncmp(__s1, __s2, __n);
}
inline size_t strlen(const char* __s)
{
  return __builtin_strlen(__s);
}
}
#else // ^^^ __cplusplus ^^^ / vvv !__cplusplus vvv
void* memcpy(void*, const void*, size_t);
void* memset(void*, int, size_t);
int memcmp(const void*, const void*, size_t);
void* memmove(void*, const void*, size_t);
size_t strlen(const char*);
#endif // !__cplusplus

#endif //_HOSTJIT_STRING_H
