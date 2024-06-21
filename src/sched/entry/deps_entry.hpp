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
    deps_entry(ccl_sched* sched)
            : sched_entry(sched, false /*is_barrier*/, false /*is_coll*/, true /*is_deps*/) {}

    void start() override;

    void update() override;

    const char* name() const override {
        return class_name();
    }

#if defined(CCL_ENABLE_SYCL) && defined(CCL_ENABLE_ZE)
    ze_event_handle_t out_event{ nullptr };
#endif // CCL_ENABLE_SYCL && CCL_ENABLE_ZE

protected:
    void dump_detail(std::stringstream& str) const override;
};
