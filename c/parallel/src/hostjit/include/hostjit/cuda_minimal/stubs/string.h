#ifndef _HOSTJIT_STRING_H
#define _HOSTJIT_STRING_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void* memcpy(void*, const void*, size_t);
void* memset(void*, int, size_t);
int memcmp(const void*, const void*, size_t);
void* memmove(void*, const void*, size_t);
size_t strlen(const char*);
#ifdef __cplusplus
}
#endif
#endif
