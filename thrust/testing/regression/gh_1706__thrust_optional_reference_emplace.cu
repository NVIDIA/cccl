#include <thrust/optional.h>

#include <unittest/unittest.h>

int main()
{
  {
    int a = 10;

    thrust::optional<int&> maybe(a);

    int b = 20;
    maybe.emplace(b);

    ASSERT_EQUAL(maybe.value(), 20);
    // Emplacing with b shouldn't change a
    ASSERT_EQUAL(a, 10);

    int c = 30;
    maybe.emplace(c);

    ASSERT_EQUAL(maybe.value(), 30);
    ASSERT_EQUAL(b, 20);
  }

  {
    thrust::optional<int&> maybe;

    int b = 21;
    maybe.emplace(b);

    ASSERT_EQUAL(maybe.value(), 21);

    int c = 31;
    maybe.emplace(c);

    ASSERT_EQUAL(maybe.value(), 31);
    ASSERT_EQUAL(b, 21);
  }
}
