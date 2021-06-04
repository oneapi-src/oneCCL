#pragma once

#include "sched/entry/entry.hpp"

class deps_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "DEPS";
    }

    deps_entry() = delete;
    deps_entry(ccl_sched* sched) : sched_entry(sched) {}

    void start() override {
        std::vector<ccl::event>& deps = sched->get_deps();
        for (size_t idx = 0; idx < deps.size(); idx++) {
#ifdef CCL_ENABLE_SYCL
            /* TODO: detect pure sycl::event and ccl::event for device op */
            deps[idx].get_native().wait();
#else /* CCL_ENABLE_SYCL */
            deps[idx].wait();
#endif /* CCL_ENABLE_SYCL */
        }
        status = ccl_sched_entry_status_complete;
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str, "deps.size ", sched->get_deps().size(), "\n");
    }
};
