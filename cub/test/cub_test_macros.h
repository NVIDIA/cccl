#pragma once

#include <c2h/catch2_test_helper.h>

// Every CUB Catch2 test must specify CUB_SMALL or CUB_LARGE.
// Small tests may share a GPU; large tests run alone. Use CUB_LARGE when unsure.

#define CUB_TEST_MEMORY_TAG_CUB_SMALL "[small-mem]"
#define CUB_TEST_MEMORY_TAG_CUB_LARGE "[large-mem]"
#define CUB_TEST_MEMORY_TAG(MEMORY)   C2H_TEST_CONCAT(CUB_TEST_MEMORY_TAG_, MEMORY)

#define CUB_TEST(NAME, TAGS, MEMORY, ...) C2H_TEST(NAME, TAGS CUB_TEST_MEMORY_TAG(MEMORY), __VA_ARGS__)

#define CUB_TEST_CASE(NAME, TAGS, MEMORY) TEST_CASE(NAME, TAGS CUB_TEST_MEMORY_TAG(MEMORY))

#define CUB_TEST_LIST(NAME, TAGS, MEMORY, ...) C2H_TEST_LIST(NAME, TAGS CUB_TEST_MEMORY_TAG(MEMORY), __VA_ARGS__)
