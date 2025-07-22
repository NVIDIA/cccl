#pragma once

// this is the only header included by unittests
// it pulls in all the others used for unittesting

#include "catch2_test_helper.h"
#include <unittest/random.h>
#include <unittest/special_types.h>
#include <unittest/testframework.h>
#include <unittest/util.h>

#define ASSERT_EQUAL(X, Y)           REQUIRE(X == Y)
#define ASSERT_EQUAL_QUIET(X, Y)     REQUIRE((X == Y))
#define ASSERT_NOT_EQUAL(X, Y)       REQUIRE(X != Y)
#define ASSERT_NOT_EQUAL_QUIET(X, Y) REQUIRE((X != Y))
#define ASSERT_LEQUAL(X, Y)          REQUIRE(X <= Y)
#define ASSERT_GEQUAL(X, Y)          REQUIRE(X >= Y)
#define ASSERT_LESS(X, Y)            REQUIRE(X < Y)
#define ASSERT_GREATER(X, Y)         REQUIRE(X > Y)
#define ASSERT_ALMOST_EQUAL(X, Y)    REQUIRE_APPROX_EQ(X, Y)

#define ASSERT_THROWS(EXPR, EXCEPTION_TYPE) CHECK_THROWS_AS(EXPR, EXCEPTION_TYPE)
#define KNOWN_FAILURE                       FAIL()
