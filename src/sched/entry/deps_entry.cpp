#include "sched/entry/deps_entry.hpp"
#include "sched/entry/subsched_entry.hpp"
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
#include "sched/entry/ze/ze_event_wait_entry.hpp"
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

using namespace ccl;

void deps_entry::start() {
    status = ccl_sched_entry_status_started;
#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    if (!sched->is_deps_barrier()) {
        if (!out_event) {
            out_event = sched->get_memory().event_manager->create();
        }
        else {
            ZE_CALL(zeEventHostReset, (out_event));
        }
        CCL_THROW_IF_NOT(out_event);
    }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
}

void deps_entry::update() {
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
        if (out_event) {
            ZE_CALL(zeEventHostSignal, (out_event));
        }
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE
    }
}

void deps_entry::dump_detail(std::stringstream& str) const {
    ccl_logger::format(str, "deps.size ", sched->get_deps().size(), "\n");
}
