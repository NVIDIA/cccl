#pragma once

namespace cuda::experimental::cufile::detail {
    /**
 * @brief RAII wrapper for automatic resource cleanup
 *
 * Generic RAII wrapper that manages a resource with a custom deleter function.
 * Provides move semantics and automatic cleanup on destruction.
 */
template<typename T, typename Deleter>
class raii_resource {
private:
    T resource_;
    Deleter deleter_;
    bool owns_resource_ = false;

public:

    /**
     * @brief Default constructor
     *
     * This constructor is required for the default constructor of the derived class.
     */
    raii_resource() = default;

    /**
     * @brief Construct with resource and deleter
     * @param resource The resource to manage
     * @param deleter Function/lambda to call for cleanup
     */
    explicit raii_resource(T resource, Deleter deleter)
        : resource_(resource), deleter_(std::move(deleter)), owns_resource_(true) {}

    /**
     * @brief Destructor automatically calls deleter if resource is owned
     */
    ~raii_resource() {
        if (owns_resource_) {
            deleter_(resource_);
        }
    }

    // Non-copyable
    raii_resource(const raii_resource&) = delete;
    raii_resource& operator=(const raii_resource&) = delete;

    /**
     * @brief Move constructor transfers ownership
     */
    raii_resource(raii_resource&& other) noexcept
        : resource_(other.resource_), deleter_(std::move(other.deleter_)), owns_resource_(other.owns_resource_) {
        other.owns_resource_ = false;
    }

    /**
     * @brief Move assignment transfers ownership
     */
    raii_resource& operator=(raii_resource&& other) noexcept {
        if (this != &other) {
            if (owns_resource_) {
                deleter_(resource_);
            }
            resource_ = other.resource_;
            deleter_ = std::move(other.deleter_);
            owns_resource_ = other.owns_resource_;
            other.owns_resource_ = false;
        }
        return *this;
    }

    /**
     * @brief Get the managed resource
     */
    T get() const noexcept { return resource_; }

    /**
     * @brief Release ownership of the resource without calling deleter
     */
    T release() noexcept {
        owns_resource_ = false;
        return resource_;
    }

    /**
     * @brief Check if this wrapper owns the resource
     */
    bool has_value() const noexcept { return owns_resource_; }


    /**
     * @brief Emplace a new resource (destroys current if owned)
     * @param resource The resource to manage
     * @param deleter Function/lambda to call for cleanup
     */
    void emplace(T resource, Deleter deleter) {
        if (owns_resource_) {
            deleter_(resource_);
        }
        resource_ = resource;
        deleter_ = std::move(deleter);
        owns_resource_ = true;
    }

    /**
     * @brief Reset to empty state (destroys current if owned)
     */
    void reset() noexcept {
        if (owns_resource_) {
            deleter_(resource_);
        }
        owns_resource_ = false;
    }
};

/**
 * @brief Helper function to create a raii_resource with type deduction
 */
template<typename T, typename Deleter>
raii_resource<T, Deleter> make_raii_resource(T resource, Deleter deleter) {
    return raii_resource<T, Deleter>(resource, std::move(deleter));
}
} // namespace cuda::experimental::cufile::detail