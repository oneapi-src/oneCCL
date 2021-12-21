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
        status = ccl_sched_entry_status_started;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
        if (ccl::global_data::env().enable_kernel_profile && sched->coll_param.stream) {
            sched->master_sched->get_kernel_timer().set_deps_start_time(
                ccl::ze::calculate_global_time(sched->coll_param.stream->get_ze_device()));
        }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
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
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
            if (ccl::global_data::env().enable_kernel_profile && sched->coll_param.stream) {
                sched->master_sched->get_kernel_timer().set_deps_end_time(
                    ccl::ze::calculate_global_time(sched->coll_param.stream->get_ze_device()));
            }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
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
