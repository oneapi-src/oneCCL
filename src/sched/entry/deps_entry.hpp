#pragma once

#ifdef CCL_ENABLE_SYCL
#include "common/utils/sycl_utils.hpp"
#endif // CCL_ENABLE_SYCL

#ifdef CCL_ENABLE_ITT
#include "sched/sched_timer.hpp"
#endif // CCL_ENABLE_ITT

#include "sched/entry/entry.hpp"

class deps_entry : public sched_entry {
public:
    static constexpr const char* class_name() noexcept {
        return "DEPS";
    }

    deps_entry() = delete;
    deps_entry(ccl_sched* sched) : sched_entry(sched, false /*is_barrier*/, true /*is_urgent*/) {}

    void start() override {
        status = ccl_sched_entry_status_started;
    }

    void update() override {
        bool all_completed = true;
        std::vector<ccl::event>& deps = sched->get_deps();

        // Note: ccl event caches the true result of test() method, so we can just iterate over whole
        // array of deps each update() call without any overhead.
        for (size_t idx = 0; idx < deps.size(); idx++) {
            bool completed = deps[idx].test();

            all_completed = all_completed && completed;
        }

        if (all_completed) {
            status = ccl_sched_entry_status_complete;
        }
    }

    const char* name() const override {
        return class_name();
    }

protected:
    void dump_detail(std::stringstream& str) const override {
        ccl_logger::format(str, "deps.size ", sched->get_deps().size(), "\n");
    }
};
