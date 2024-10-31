//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Stackable context and logical data to nest contexts
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "cuda/experimental/__stf/allocators/adapters.cuh"
#include "cuda/experimental/stf.cuh"

namespace cuda::experimental::stf {

template <typename T>
class stackable_logical_data;

class stackable_ctx {
public:
    class impl {
    public:
        impl() { push(stream_ctx(), nullptr); }

        ~impl() = default;

        void push(context ctx, cudaStream_t stream) {
            s.push_back(mv(ctx));
            s_stream.push_back(stream);
        }

        void pop() {
            s.back().finalize();

            s.pop_back();

            s_stream.pop_back();

            assert(alloc_adapters.size() > 0);
            alloc_adapters.back().clear();
            alloc_adapters.pop_back();
        }

        size_t depth() const { return s.size() - 1; }

        auto& get() { return s.back(); }

        const auto& get() const { return s.back(); }

        auto& at(size_t level) {
            assert(level < s.size());
            return s[level];
        }

        const auto& at(size_t level) const {
            assert(level < s.size());

            return s[level];
        }

        cudaStream_t stream_at(size_t level) const { return s_stream[level]; }

        void push_graph() {
            cudaStream_t stream = get().pick_stream();

            // These resources are not destroyed when we pop, so we create it only if needed
            if (async_handles.size() < s_stream.size()) {
                async_handles.emplace_back();
            }

            auto gctx = graph_ctx(stream, async_handles.back());

            auto wrapper = stream_adapter(gctx, stream);
            // FIXME : issue with the deinit phase
            //   gctx.update_uncached_allocator(wrapper.allocator());

            alloc_adapters.push_back(wrapper);

            push(mv(gctx), stream);
        }

    private:
        ::std::vector<context> s;
        ::std::vector<cudaStream_t> s_stream;
        ::std::vector<async_resources_handle> async_handles;
        ::std::vector<stream_adapter> alloc_adapters;
    };

    stackable_ctx() : pimpl(::std::make_shared<impl>()) {}

    const auto& get() const { return pimpl->get(); }
    auto& get() { return pimpl->get(); }

    const auto& at(size_t level) const { return pimpl->at(level); }
    auto& at(size_t level) { return pimpl->at(level); }

    cudaStream_t stream_at(size_t level) const { return pimpl->stream_at(level); }

    const auto& operator()() const { return get(); }

    auto& operator()() { return get(); }

    void push_graph() { pimpl->push_graph(); }

    void pop() { pimpl->pop(); }

    size_t depth() const { return pimpl->depth(); }

    template <typename... Pack>
    auto logical_data(Pack&&... pack) {
        return stackable_logical_data(*this, depth(), get().logical_data(::std::forward<Pack>(pack)...));
    }

    template <typename... Pack>
    auto task(Pack&&... pack) {
        return get().task(::std::forward<Pack>(pack)...);
    }

    template <typename... Pack>
    auto parallel_for(Pack&&... pack) {
        return get().parallel_for(::std::forward<Pack>(pack)...);
    }

    template <typename... Pack>
    auto host_launch(Pack&&... pack) {
        return get().host_launch(::std::forward<Pack>(pack)...);
    }

    void finalize() {
        // There must be only one level left
        assert(depth() == 0);

        get().finalize();
    }

public:
    ::std::shared_ptr<impl> pimpl;
};

template <typename T>
class stackable_logical_data {
    class impl {
    public:
        impl() = default;
        impl(stackable_ctx sctx, size_t depth, logical_data<T> ld) : base_depth(depth), sctx(mv(sctx)) {
            s.push_back(ld);
        }

        const auto& get() const {
            check_level_mismatch();
            return s.back();
        }
        auto& get() {
            check_level_mismatch();
            return s.back();
        }

        void push(access_mode m, data_place where = data_place::invalid) {
            // We have not pushed yet, so the current depth is the one before pushing
            context& from_ctx = sctx.at(depth());
            context& to_ctx = sctx.at(depth() + 1);

            // Ensure this will match the depth of the context after pushing
            assert(sctx.depth() == depth() + 1);

            if (where == data_place::invalid) {
                // use the default place
                where = from_ctx.default_exec_place().affine_data_place();
            }

            assert(where != data_place::invalid);

            // Freeze the logical data at the top
            logical_data<T>& from_data = s.back();
            frozen_logical_data<T> f = from_ctx.freeze(from_data, m, mv(where));

            // Save the frozen data in a separate vector
            frozen_s.push_back(f);

            // FAKE IMPORT : use the stream needed to support the (graph) ctx
            cudaStream_t stream = sctx.stream_at(depth());

            T inst = f.get(where, stream);
            auto ld = to_ctx.logical_data(inst, where);

            if (!symbol.empty()) {
                ld.set_symbol(symbol + "." + ::std::to_string(depth() + 1));
            }

            s.push_back(mv(ld));
        }

        void pop() {
            // We are going to unfreeze the data, which is currently being used
            // in a (graph) ctx that uses this stream to launch the graph
            cudaStream_t stream = sctx.stream_at(depth());

            frozen_logical_data<T>& f = frozen_s.back();
            f.unfreeze(stream);

            // Remove frozen logical data
            frozen_s.pop_back();
            // Remove aliased logical data
            s.pop_back();
        }

        size_t depth() const { return s.size() - 1 + base_depth; }

        void set_symbol(::std::string symbol_) {
            symbol = mv(symbol_);
            s.back().set_symbol(symbol + "." + ::std::to_string(depth()));
        }

    private:
        void check_level_mismatch() const {
            if (depth() != sctx.depth()) {
                fprintf(stderr, "Warning: mismatch between ctx level %ld and data level %ld\n", sctx.depth(), depth());
            }
        }

        mutable ::std::vector<logical_data<T>> s;

        // When stacking data, we freeze data from the lower levels, these are
        // their frozen counterparts. This vector has one item less than the
        // vector of logical data.
        mutable ::std::vector<frozen_logical_data<T>> frozen_s;

        // If the logical data was created at a level that is not directly the root of the context, we remember this
        // offset
        size_t base_depth = 0;
        stackable_ctx sctx;  // in which stackable context was this created ?

        ::std::string symbol;
    };

public:
    stackable_logical_data() = default;

    template <typename... Args>
    stackable_logical_data(stackable_ctx sctx, size_t depth, logical_data<T> ld)
            : pimpl(::std::make_shared<impl>(mv(sctx), depth, mv(ld))) {}

    const auto& get() const { return pimpl->get(); }
    auto& get() { return pimpl->get(); }

    const auto& operator()() const { return get(); }
    auto& operator()() { return get(); }

    size_t depth() const { return pimpl->depth(); }

    void push(access_mode m, data_place where = data_place::invalid) { pimpl->push(m, mv(where)); }
    void pop() { pimpl->pop(); }

    // Helpers
    template <typename... Pack>
    auto read(Pack&&... pack) const {
        return get().read(::std::forward<Pack>(pack)...);
    }

    template <typename... Pack>
    auto write(Pack&&... pack) {
        return get().write(::std::forward<Pack>(pack)...);
    }

    template <typename... Pack>
    auto rw(Pack&&... pack) {
        return get().rw(::std::forward<Pack>(pack)...);
    }

    auto shape() const { return get().shape(); }

    auto& set_symbol(::std::string symbol) {
        pimpl->set_symbol(mv(symbol));
        return *this;
    }

private:
    ::std::shared_ptr<impl> pimpl;
};

}  // end namespace cuda::experimental::stf
