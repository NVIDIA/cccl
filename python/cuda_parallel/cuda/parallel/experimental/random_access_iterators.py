# These types are not the usual Python iterators at all, but are closely
# aligned with C++ Random Access Iterators:
#   - https://en.cppreference.com/w/cpp/iterator/random_access_iterator

# C++ `operator+`  -> `advance`
# C++ `operator[]` -> `dereference`


class ConstantRAI:

    def __init__(self, *, value):
        self.value = value

    def advance(self, distance):
        del distance
        return self

    def dereference(self, distance):
        del distance
        return self.value


class CountingRAI:

    def __init__(self, origin):
        self.origin = origin

    def advance(self, distance):
        return CountingRAI(origin=self.origin + distance)

    def dereference(self, distance):
        return self.origin + distance


class TransformRAI:

    def __init__(self, *, unary_op, origin):
        self.unary_op = unary_op
        self.origin = origin

    def advance(self, distance):
        return TransformRAI(unary_op=self.unary_op, origin=self.origin + distance)

    def dereference(self, distance):
        return self.unary_op(self.origin + distance)
