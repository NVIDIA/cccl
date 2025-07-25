#pragma once

#include <cstddef>
#include <type_traits>
#include <iterator>

#if __cplusplus >= 202002L && __has_include(<span>)
#include <span>
#define CUDA_IO_HAS_STD_SPAN 1
#else
#define CUDA_IO_HAS_STD_SPAN 0
#endif

namespace cuda::experimental::cufile::detail {

#if CUDA_IO_HAS_STD_SPAN
    // Use standard std::span when available
    template<typename T, std::size_t Extent = std::dynamic_extent>
    using span = std::span<T, Extent>;

    constexpr std::size_t dynamic_extent = std::dynamic_extent;
#else
    // Simple span implementation for pre-C++20
    constexpr std::size_t dynamic_extent = std::size_t(-1);

    template<typename T, std::size_t Extent = dynamic_extent>
    class span {
    private:
        T* data_;
        std::size_t size_;

    public:
        using element_type = T;
        using value_type = std::remove_cv_t<T>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using iterator = T*;
        using const_iterator = const T*;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        static constexpr size_type extent = Extent;

        // Constructors
        constexpr span() noexcept : data_(nullptr), size_(0) {}

        constexpr span(pointer ptr, size_type count) noexcept
            : data_(ptr), size_(count) {}

        constexpr span(pointer first, pointer last) noexcept
            : data_(first), size_(last - first) {}

        template<std::size_t N>
        constexpr span(element_type (&arr)[N]) noexcept
            : data_(arr), size_(N) {}

        template<typename Container,
                 typename = std::enable_if_t<
                     !std::is_array_v<Container> &&
                     !std::is_same_v<Container, span> &&
                     std::is_convertible_v<typename Container::pointer, pointer>
                 >>
        constexpr span(Container& cont) noexcept
            : data_(cont.data()), size_(cont.size()) {}

        template<typename Container,
                 typename = std::enable_if_t<
                     !std::is_array_v<Container> &&
                     !std::is_same_v<Container, span> &&
                     std::is_convertible_v<typename Container::pointer, pointer>
                 >>
        constexpr span(const Container& cont) noexcept
            : data_(cont.data()), size_(cont.size()) {}

        // Copy constructor
        constexpr span(const span& other) noexcept = default;

        // Assignment
        constexpr span& operator=(const span& other) noexcept = default;

        // Iterators
        constexpr iterator begin() const noexcept { return data_; }
        constexpr iterator end() const noexcept { return data_ + size_; }
        constexpr const_iterator cbegin() const noexcept { return data_; }
        constexpr const_iterator cend() const noexcept { return data_ + size_; }

        constexpr reverse_iterator rbegin() const noexcept { return reverse_iterator(end()); }
        constexpr reverse_iterator rend() const noexcept { return reverse_iterator(begin()); }
        constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }
        constexpr const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

        // Element access
        constexpr reference operator[](size_type idx) const noexcept { return data_[idx]; }
        constexpr reference at(size_type idx) const {
            if (idx >= size_) throw std::out_of_range("span::at");
            return data_[idx];
        }
        constexpr reference front() const noexcept { return data_[0]; }
        constexpr reference back() const noexcept { return data_[size_ - 1]; }

        // Observers
        constexpr pointer data() const noexcept { return data_; }
        constexpr size_type size() const noexcept { return size_; }
        constexpr size_type size_bytes() const noexcept { return size_ * sizeof(element_type); }
        constexpr bool empty() const noexcept { return size_ == 0; }

        // Subviews
        constexpr span<element_type> first(size_type count) const noexcept {
            return span<element_type>(data_, count);
        }

        constexpr span<element_type> last(size_type count) const noexcept {
            return span<element_type>(data_ + size_ - count, count);
        }

        constexpr span<element_type> subspan(size_type offset, size_type count = dynamic_extent) const noexcept {
            return span<element_type>(data_ + offset, count == dynamic_extent ? size_ - offset : count);
        }
    };

    // Deduction guides for pre-C++20
    template<typename T>
    span(T*, std::size_t) -> span<T>;

    template<typename T>
    span(T*, T*) -> span<T>;

    template<typename T, std::size_t N>
    span(T(&)[N]) -> span<T, N>;

    template<typename Container>
    span(Container&) -> span<typename Container::value_type>;

    template<typename Container>
    span(const Container&) -> span<const typename Container::value_type>;
#endif

} // namespace cuda::experimental::cufile::detail

// Convenience aliases in cuda::experimental namespace
namespace cuda::experimental::cufile {
    template<typename T, std::size_t Extent = detail::dynamic_extent>
    using span = detail::span<T, Extent>;

    constexpr std::size_t dynamic_extent = detail::dynamic_extent;
}