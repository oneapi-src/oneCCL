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

ccl_listener::ccl_listener(ccl_global_data *gl_data)
    : ccl_base_thread(0, ccl_update_comm_world_info),
      gl_data(gl_data)
{}

void* ccl_update_comm_world_info(void* args)
{
    ccl_listener* listener = static_cast<ccl_listener*>(args);
    ccl_global_data* gl_data = listener->gl_data;

    while (true)
    {
        atl_wait_notification(gl_data->executor->get_atl_ctx());

        gl_data->executor->is_locked = true;
        ccl_executor::worker_guard guard = gl_data->executor->get_worker_lock();

        ccl_reset_for_size_update(gl_data);

        atl_update(gl_data->executor->get_atl_ctx());

        ccl_init_global_objects(*gl_data);

        gl_data->executor->update_workers();
        gl_data->executor->is_locked = false;
    }

    return nullptr;
}
