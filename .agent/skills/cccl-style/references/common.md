# Extra Common Style Guidance

- Static member functions of a class template inherit the class's namespace.
- Variables that are unsigned, or that can become unsigned after template instantiation, must not check for negative values directly. Use `cuda::std::is_unsigned_v<T> ? false : (var < 0)` instead.
