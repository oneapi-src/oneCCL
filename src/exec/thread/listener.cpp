/*
 Copyright 2016-2020 Intel Corporation
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/
#include "exec/exec.hpp"
#include "exec/thread/listener.hpp"

void* ccl_update_comm_world_info(void* args);

ccl_listener::ccl_listener(ccl_global_data* gl_data)
    : ccl_base_thread(0, ccl_update_comm_world_info),
      gl_data(gl_data)
{}

void* ccl_update_comm_world_info(void* args)
{
    ccl_listener* listener = static_cast<ccl_listener*>(args);
    ccl_global_data* gl_data = listener->gl_data;
    int res = 0;
    listener->started = true;

    while (true)
    {
        /*
         * atl_wait_notification return values:
         * 0 - got notification, should to do some updates
         * 1 - finished by timeout, should to check if thread need to stop, in another case should to recall this func
         * TODO: change nums to enum
         * */
        res = atl_wait_notification(gl_data->executor->get_atl_ctx());

        if (res == 1)
        {
            if (listener->should_stop.load(std::memory_order_acquire))
                break;
            else
                continue;
        }
        gl_data->executor->is_locked = true;
        ccl_executor::worker_guard guard = gl_data->executor->get_worker_lock();

        ccl_reset_resize_dependent_objects(*gl_data);

        atl_update(gl_data->executor->get_atl_ctx());

        ccl_init_resize_dependent_objects(*gl_data);

        gl_data->executor->update_workers();
        gl_data->executor->is_locked = false;
    }

    listener->started = false;

    return nullptr;
}
