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
#include <cassert>
#include <cstring>
#include <functional>

#include "oneapi/ccl/native_device_api/l0/base_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/device.hpp"
#include "oneapi/ccl/native_device_api/l0/subdevice.hpp"
#include "oneapi/ccl/native_device_api/l0/driver.hpp"
#include "oneapi/ccl/native_device_api/l0/primitives_impl.hpp"
#include "oneapi/ccl/native_device_api/l0/utils.hpp"

namespace native {

uint32_t get_device_properties_from_handle(ccl_device::handle_t handle) {
    ze_device_properties_t device_properties;
    ze_result_t ret = zeDeviceGetProperties(handle, &device_properties);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("zeDeviceGetProperties failed, error: ") +
                                 native::to_string(ret));
    }

    return device_properties.deviceId;
}

details::cross_device_rating property_p2p_rating_calculator(const native::ccl_device& lhs,
                                                            const native::ccl_device& rhs,
                                                            size_t weight) {
    ze_device_p2p_properties_t p2p = lhs.get_p2p_properties(rhs);
    if (p2p.flags & ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS)
        return weight;
    else
        return 0;
}

CCL_API
std::shared_ptr<ccl_device> ccl_device::create(
    handle_t handle,
    owner_ptr_t&& driver,
    const ccl::device_indices_t& indexes /* = ccl::device_indices_t()*/) {
    // TODO - dirty code
    owner_ptr_t shared_driver(std::move(driver));
    std::shared_ptr<ccl_device> device =
        std::make_shared<ccl_device>(handle, shared_driver.lock()->get_ptr(), shared_driver.lock()->get_driver_contexts());

    auto collected_subdevices_list = ccl_subdevice::get_handles(*device);

    try {
        for (const auto& val : collected_subdevices_list) {
            if (indexes.empty()) {
                device->sub_devices.emplace(
                    val.first,
                    ccl_subdevice::create(
                        val.second, device->get_ptr(), shared_driver.lock()->get_ptr()));
            }
            else {
                //collect device_index only for device specific index
                for (const auto& affitinity : indexes) {
                    // TODO add driver index checking
                    if (std::get<ccl::device_index_enum::subdevice_index_id>(affitinity) ==
                        val.first) {
                        device->sub_devices.emplace(
                            val.first,
                            ccl_subdevice::create(
                                val.second, device->get_ptr(), shared_driver.lock()->get_ptr()));
                    }
                }
            }
        }
    }
    catch (const std::exception& ex) {
        std::stringstream ss;
        ss << "Cannot create subdevices: ";
        for (const auto& index : indexes) {
            ss << index << ", ";
        }
        ss << "\nError: " << ex.what();
        throw;
    }

    return device;
}

CCL_API
ccl_device::indexed_handles ccl_device::get_handles(
    const ccl_device_driver& driver,
    const ccl::device_indices_t& requested_device_indexes /* = indices()*/) {
    uint32_t devices_count = 0;
    ze_result_t err = zeDeviceGet(driver.handle, &devices_count, nullptr);
    if (err != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - zeDeviceGet failed, error: " + native::to_string(err));
    }

    std::vector<ccl_device::handle_t> handles;
    handles.resize(devices_count);

    err = zeDeviceGet(driver.handle, &devices_count, handles.data());
    if (err != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(
            std::string(__FUNCTION__) +
            " - zeDeviceGet failed for device request, error: " + native::to_string(err));
    }

    //filter indices by driver id
    auto parent_id = driver.get_driver_id();
    ccl::device_indices_t filtered_ids;
    if (!requested_device_indexes.empty()) {
        for (const auto& index : requested_device_indexes) {
            if (std::get<ccl::device_index_enum::driver_index_id>(index) == parent_id) {
                filtered_ids.insert(index);
            }
        }
        if (filtered_ids.empty()) {
            throw std::runtime_error(std::string(__FUNCTION__) + " - Failed, nothing to get");
        }
    }

    //collect device by indices
    indexed_handles ret;
    try {
        ret = detail::collect_indexed_data<ccl::device_index_enum::device_index_id>(
            filtered_ids,
            handles,
            std::bind(get_device_properties_from_handle, std::placeholders::_1));

        // set parent_index forcibly
        /*
        std::transform(ret.begin(), ret.end(), ret.begin(),
                       [parent_id](typename indexed_handles::value_type &val)
                       {
                           std::get<ccl::device_index_enum::driver_index_id>(val.first) = parent_id;
                       });
        */
    }
    catch (const std::exception& ex) {
        throw std::runtime_error(std::string(__FUNCTION__) + " - Cannot add device: " + ex.what());
    }
    return ret;
}

void ccl_device::initialize_device_data() {
    ze_result_t ret = zeDeviceGetProperties(handle, &device_properties);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("zeDeviceGetProperties failed, error: ") +
                                 native::to_string(ret));
    }

    uint32_t memory_prop_count = 0;
    ret = zeDeviceGetMemoryProperties(handle, &memory_prop_count, nullptr);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(
            std::string("zeDeviceGetMemoryProperties failed for nullptr, error: ") +
            native::to_string(ret));
    }
    memory_properties.resize(memory_prop_count);

    ret = zeDeviceGetMemoryProperties(handle, &memory_prop_count, memory_properties.data());
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(
            std::string("zeDeviceGetMemoryProperties failed for memory_properties, error: ") +
            native::to_string(ret));
    }

    ret = zeDeviceGetMemoryAccessProperties(handle, &memory_access_properties);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(
            std::string(
                "zeDeviceGetMemoryAccessProperties failed for memory_access_properties, error: ") +
            native::to_string(ret));
    }

    ret = zeDeviceGetComputeProperties(handle, &compute_properties);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("zeDeviceGetComputeProperties failed, error: ") +
                                 native::to_string(ret));
    }
}

ccl_device::ccl_device(handle_t h, owner_ptr_t&& parent, std::weak_ptr<ccl_context_holder>&& ctx, std::false_type)
        : base(h, std::move(parent), std::move(ctx)) {}

ccl_device::ccl_device(handle_t h, owner_ptr_t&& parent, std::weak_ptr<ccl_context_holder>&& ctx) : base(h, std::move(parent), std::move(ctx)) {
    initialize_device_data();
}

CCL_API ccl_device::~ccl_device() {
    cmd_queus.clear();
    sub_devices.clear();
}

CCL_API ccl_device::sub_devices_container_type& ccl_device::get_subdevices() {
    return const_cast<sub_devices_container_type&>(
        static_cast<const ccl_device*>(this)->get_subdevices());
}

CCL_API const ccl_device::sub_devices_container_type& ccl_device::get_subdevices() const {
    return sub_devices;
}

CCL_API ccl_device::subdevice_ptr ccl_device::get_subdevice(const ccl::device_index_type& path) {
    return std::const_pointer_cast<ccl_subdevice>(
        static_cast<const ccl_device*>(this)->get_subdevice(path));
}

CCL_API ccl_device::const_subdevice_ptr ccl_device::get_subdevice(
    const ccl::device_index_type& path) const {
    const auto driver = get_owner().lock();
    if (!driver) {
        assert(false && "because ccl_device has no owner");
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 " - cannot get get_device_path() because ccl_device has no owner");
    }

    ccl::index_type driver_idx = std::get<ccl::device_index_enum::driver_index_id>(path);
    if (driver_idx != driver->get_driver_id()) {
        assert(false && "incorrect owner driver");
        throw std::runtime_error(
            std::string(__PRETTY_FUNCTION__) + " - incorrect driver, expected: " +
            std::to_string(driver->get_driver_id()) + ", requested: " + ccl::to_string(path));
    }

    ccl::index_type device_index = std::get<ccl::device_index_enum::device_index_id>(path);
    if (device_index != get_device_id()) {
        assert(false && "incorrect device index");
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 " - incorrect device, expected: " + std::to_string(device_index) +
                                 ", requested: " + ccl::to_string(path));
    }

    ccl::index_type subdevice_index = std::get<ccl::device_index_enum::subdevice_index_id>(path);
    if (ccl::unused_index_value == subdevice_index) {
        assert(false && "incorrect subdevice index");
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 " - incorrect subdevice requested: " + ccl::to_string(path));
    }

    auto it = sub_devices.find(subdevice_index);
    if (it == sub_devices.end()) {
        assert(false && "subdevice is not found");
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 " - incorrect subdevice index requested: " + ccl::to_string(path) +
                                 ". Total subdevices count: " + std::to_string(sub_devices.size()));
    }

    return it->second;
}

CCL_API const ze_device_properties_t& ccl_device::get_device_properties() const {
    return device_properties;
}

CCL_API bool ccl_device::is_subdevice() const noexcept {
    return false;
}

CCL_API ccl::index_type ccl_device::get_device_id() const {
    assert(!(device_properties.flags & ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE) &&
           "Must NOT be subdevice");
    return get_device_properties().deviceId;
}

CCL_API ccl::device_index_type ccl_device::get_device_path() const {
    const auto driver = get_owner().lock();
    if (!driver) {
        throw std::runtime_error("cannot get get_device_path() because ccl_device has no owner");
    }

    ccl::device_index_type device_path = std::make_tuple(
        driver->get_driver_id(), get_device_id(), std::numeric_limits<uint32_t>::max());
    return device_path;
}

CCL_API ze_device_p2p_properties_t
ccl_device::get_p2p_properties(const ccl_device& remote_device) const {
    ze_device_p2p_properties_t pP2PProperties;
    ze_result_t ret = zeDeviceGetP2PProperties(handle, remote_device.handle, &pP2PProperties);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("Cannot execute zeDeviceGetP2PProperties, error: ") +
                                 native::to_string(ret));
    }
    return pP2PProperties;
}

CCL_API const ze_command_queue_desc_t& ccl_device::get_default_queue_desc() {
    static ze_command_queue_desc_t common{
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .pNext = NULL,
        .ordinal = 0,
        .index = 0,
        .flags = 0,
        .mode = ZE_COMMAND_QUEUE_MODE_DEFAULT,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL,
    };
    return common;
}

CCL_API const ze_command_list_desc_t& ccl_device::get_default_list_desc() {
    static ze_command_list_desc_t common{
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .pNext = NULL,
        .commandQueueGroupOrdinal = 0,
        .flags = 0,
    };
    return common;
}

CCL_API const ze_device_mem_alloc_desc_t& ccl_device::get_default_mem_alloc_desc() {
    static ze_device_mem_alloc_desc_t common{
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC,
        .pNext = NULL,
        .flags = 0,
        .ordinal = 0,
    };
    return common;
}

CCL_API const ze_host_mem_alloc_desc_t& ccl_device::get_default_host_alloc_desc() {
    static const ze_host_mem_alloc_desc_t common{
        .stype      = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        .pNext      = NULL,
        .flags      = 0,
    };
    return common;
}

CCL_API ccl_device::device_queue ccl_device::create_cmd_queue(std::shared_ptr<ccl_context> ctx,
    const ze_command_queue_desc_t& properties /* = get_default_queue_desc()*/) {

    if(!ctx) {
        ctx = get_default_device_context();
    }

    ze_command_queue_handle_t hCommandQueue;
    ze_result_t ret = zeCommandQueueCreate(ctx->get(), handle, &properties, &hCommandQueue);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot allocate queue, error: ") +
                                 native::to_string(ret));
    }
    return device_queue(hCommandQueue, get_ptr(), ctx);
}

CCL_API ze_fence_handle_t ccl_device::create_or_get_fence(const device_queue& queue, 
                                                          std::shared_ptr<ccl_context> ctx) {
    //TODO not optimal
    std::unique_lock<std::mutex> lock(queue_mutex);
    auto fence_it = queue_fences.find(queue.handle);
    if (fence_it == queue_fences.end()) {
        ze_fence_handle_t h;
        ze_fence_desc_t desc{
            .stype = ZE_STRUCTURE_TYPE_FENCE_DESC,
            .pNext = NULL,
            .flags = 0,
        };

        ze_result_t ret = zeFenceCreate(queue.handle, &desc, &h);
        if (ret != ZE_RESULT_SUCCESS) {
            throw std::runtime_error(std::string("cannot allocate fence, error: ") +
                                     native::to_string(ret));
        }
        device_queue_fence f(h, get_ptr(), ctx);
        fence_it = queue_fences.emplace(queue.handle, std::move(f)).first;
    }
    return fence_it->second.handle;
}

CCL_API void* ccl_device::device_alloc_memory(size_t bytes_count,
                                              size_t alignment,
                                              const ze_device_mem_alloc_desc_t& mem_descr,
                                              const ze_host_mem_alloc_desc_t& host_descr,
                                              std::shared_ptr<ccl_context> ctx) {
    void* out_ptr = nullptr;
    if(!ctx) {
        ctx = get_default_device_context();
    }

    ze_result_t
        ret = //zeDriverAllocSharedMem(get_owner()->handle, handle, flags, ordinal, ZE_HOST_MEM_ALLOC_FLAG_DEFAULT, bytes_count, alignment, &out_ptr);
        //zeDriverAllocHostMem(get_owner()->handle, ZE_HOST_MEM_ALLOC_FLAG_DEFAULT, bytes_count, alignment, &out_ptr);
        zeMemAllocShared(
            ctx->get(), &mem_descr, &host_descr, bytes_count, alignment, handle, &out_ptr);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot allocate memory, error: ") +
                                 std::to_string(ret));
    }

    return out_ptr;
}

CCL_API void* ccl_device::device_alloc_shared_memory(size_t bytes_count,
                                                     size_t alignment,
                                                     const ze_host_mem_alloc_desc_t& host_desc,
                                                     const ze_device_mem_alloc_desc_t& mem_descr,
                                                     std::shared_ptr<ccl_context> ctx) {
    void* out_ptr = nullptr;
    if(!ctx) {
        ctx = get_default_device_context();
    }

    ze_result_t ret = zeMemAllocShared(
        ctx->get(), &mem_descr, &host_desc, bytes_count, alignment, handle, &out_ptr);

    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot allocate shared memory, error: ") +
                                 std::to_string(ret));
    }

    return out_ptr;
}

CCL_API ccl_device::handle_t ccl_device::get_assoc_device_handle(const void* ptr,
                                                                 const ccl_device_driver* driver,
                                                                 std::shared_ptr<ccl_context> ctx) {
    assert(driver && "Driver must exist!");
    ze_memory_allocation_properties_t mem_prop;
    ze_device_handle_t alloc_device_handle{};
    // TODO: empty
    ze_context_handle_t ctx_tmp = nullptr;

    ze_result_t result =
        zeMemGetAllocProperties(ctx_tmp, ptr, &mem_prop, &alloc_device_handle);
    if (result != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("Cannot zeMemGetAllocProperties: ") +
                                 native::to_string(result));
    }
    return alloc_device_handle;
}

CCL_API void ccl_device::device_free_memory(void* mem_handle, std::shared_ptr<ccl_context> ctx) {
    if (!mem_handle) {
        return;
    }
    if(!ctx) {
        ctx = get_default_device_context();
    }

    if (zeMemFree(ctx->get(), mem_handle) != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot release memory"));
    }
}

CCL_API ccl_device::device_ipc_memory_handle ccl_device::create_ipc_memory_handle(
    void* device_mem_ptr, std::shared_ptr<ccl_context> ctx) {
    ze_ipc_mem_handle_t ipc_handle;

    if(!ctx) {
        ctx = get_default_device_context();
    }

    ze_result_t ret = zeMemGetIpcHandle(
        ctx->get(), device_mem_ptr, &ipc_handle);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot get ipc mem handle, error: ") +
                                 native::to_string(ret));
    }
    return device_ipc_memory_handle(ipc_handle, get_ptr(), ctx);
}

CCL_API std::shared_ptr<ccl_device::device_ipc_memory_handle>
ccl_device::create_shared_ipc_memory_handle(void* device_mem_ptr, std::shared_ptr<ccl_context> ctx) {
    ze_ipc_mem_handle_t ipc_handle;
    //TODO thread-safety
    auto it = ipc_storage.find(device_mem_ptr);
    if (it != ipc_storage.end()) {
        return it->second;
    }

    if(!ctx) {
        ctx = get_default_device_context();
    }

    ze_result_t ret = zeMemGetIpcHandle(
        ctx->get(), device_mem_ptr, &ipc_handle);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot get ipc mem handle, error: ") +
                                 native::to_string(ret));
    }

    return ipc_storage
        .insert({ device_mem_ptr,
                  std::shared_ptr<device_ipc_memory_handle>(
                      new device_ipc_memory_handle(ipc_handle, get_ptr(), ctx)) })
        .first->second;
}

void CCL_API ccl_device::on_delete(ze_ipc_mem_handle_t& ipc_mem_handle, ze_context_handle_t& ctx) {
    /*
    //No need to destroy ipc handle on parent process?

    ze_result_t ret = xeIpcCloseMemHandle(ipc_mem_handle);
    if(ret != ZE_RESULT_SUCCESS)
    {
        throw std::runtime_error(std::string("cannot close ipc handle, error: ") + std::to_string(ret));
    }
    */

    //todo thread safety
    for (auto ipc_it = ipc_storage.begin(); ipc_it != ipc_storage.end(); ++ipc_it) {
        if (!strncmp(ipc_it->second->handle.data, ipc_mem_handle.data, ZE_MAX_IPC_HANDLE_SIZE)) {
            ipc_storage.erase(ipc_it);
        }
    }
}

CCL_API ccl_device::device_ipc_memory ccl_device::get_ipc_memory(
    std::shared_ptr<device_ipc_memory_handle>&& ipc_handle, std::shared_ptr<ccl_context> ctx) {
    assert(ipc_handle->get_owner().lock().get() == this && "IPC handle doesn't belong to device: ");
    //, this,
    // ", expected device: ", ipc_handle.get_owner());

    ze_ipc_memory_flag_t flag = ZE_IPC_MEMORY_FLAG_TBD;
    ip_memory_elem_t ipc_memory{};

    if(!ctx) {
        ctx = get_default_device_context();
    }

    ze_result_t ret = zeMemOpenIpcHandle(
        ctx->get(), handle, ipc_handle->handle, flag, &(ipc_memory.pointer));
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot get open ipc mem handle from: ") +
                                 native::to_string(ipc_handle->handle) +
                                 ", error: " + native::to_string(ret));
    }

    assert(ipc_memory.pointer && "opened ipc memory handle is nullptr");

    //no need to clear ipc handle in remote process (?)
    //ipc_handle.handle = nullptr;
    ipc_handle->owner.reset();

    return device_ipc_memory(ipc_memory, get_ptr(), ctx);
}

CCL_API std::shared_ptr<ccl_device::device_ipc_memory> ccl_device::restore_shared_ipc_memory(
    std::shared_ptr<device_ipc_memory_handle>&& ipc_handle, std::shared_ptr<ccl_context> ctx) {
    assert(ipc_handle->get_owner().lock().get() == this && "IPC handle doesn't belong to device: ");
    ze_ipc_memory_flag_t flag = ZE_IPC_MEMORY_FLAG_TBD;
    ip_memory_elem_t ipc_memory{};

    if(!ctx) {
        ctx = get_default_device_context();
    }

    ze_result_t ret = zeMemOpenIpcHandle(
        ctx->get(), handle, ipc_handle->handle, flag, &(ipc_memory.pointer));
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot get open ipc mem handle from: ") +
                                 native::to_string(ipc_handle->handle) +
                                 ", error: " + native::to_string(ret));
    }

    assert(ipc_memory.pointer && "opened ipc memory handle is nullptr");

    //no need to clear ipc handle in remote process (?)
    //ipc_handle.handle = nullptr;
    ipc_handle->owner.reset();

    return std::shared_ptr<device_ipc_memory>(new device_ipc_memory(ipc_memory, get_ptr(), ctx));
}

void CCL_API ccl_device::on_delete(ip_memory_elem_t& ipc_mem, ze_context_handle_t& ctx) {
    // if(!ctx) {
    //     ctx = std::shared_ptr<ccl_context>(get_owner().lock()->context->map_context.begin()->second.front().lock());
    // }

    // TODO: empty
    ze_context_handle_t ctx_tmp = nullptr;
    ze_result_t ret = zeMemCloseIpcHandle(ctx_tmp, ipc_mem.pointer);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot close ipc mem handle, error: ") +
                                 native::to_string(ret));
    }
}

CCL_API ccl_device::device_queue& ccl_device::get_cmd_queue(
    const ze_command_queue_desc_t& properties, std::shared_ptr<ccl_context> ctx) {
    std::unique_lock<std::mutex> lock(queue_mutex);
    auto it = cmd_queus.find(properties);
    if (it == cmd_queus.end()) {
        it = cmd_queus.emplace(properties, create_cmd_queue(ctx, properties)).first;
    }
    return it->second;
}

CCL_API
ccl_device::context_storage_type ccl_device::get_device_contexts() {
    return get_ctx().lock();
}

CCL_API
std::shared_ptr<ccl_context> ccl_device::get_default_device_context() {
    auto ctx_holder = get_device_contexts();
    if (ctx_holder->map_context.empty())
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 " - no default driver in context map");
    auto &default_driver_ptr = *ctx_holder->map_context.begin();
    if (default_driver_ptr.second.empty())
        throw std::runtime_error(std::string(__PRETTY_FUNCTION__) +
                                 " - no default context for default driver");
    auto ctx = *default_driver_ptr.second.begin();

    return ctx;
}

ccl_device::device_cmd_list CCL_API
ccl_device::create_cmd_list(std::shared_ptr<ccl_context> ctx,
                            const ze_command_list_desc_t& properties) {
    // Create a command queue
    if(!ctx) {
        ctx = get_default_device_context();
    }

    ze_command_list_handle_t hCommandList;
    ze_result_t ret = zeCommandListCreate(ctx->get(), handle, &properties, &hCommandList);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot allocate command list, error: ") +
                                 native::to_string(ret));
    }
    return device_cmd_list(hCommandList, get_ptr(), ctx);
}

CCL_API ccl_device::device_cmd_list& ccl_device::get_cmd_list(std::shared_ptr<ccl_context> ctx,
    const ze_command_list_desc_t& properties /* = get_default_list_desc()*/) {
    auto it = cmd_lists.find(properties);
    if (it == cmd_lists.end()) {
        it = cmd_lists.emplace(properties, create_cmd_list( ctx, properties)).first;
    }
    return it->second;
}

CCL_API ccl_device::device_module_ptr ccl_device::create_module(const ze_module_desc_t& descr,
                                                                size_t hash,
                                                                std::shared_ptr<ccl_context> ctx) {
    auto it = modules.find(hash);
    if (it != modules.end()) {
        return it->second;
    }

    std::string build_log_string;
    ze_module_handle_t module = nullptr;
    ze_module_build_log_handle_t build_log = nullptr;

    if(!ctx) {
        ctx = get_default_device_context();
    }

    ze_result_t result = zeModuleCreate(ctx->get(), handle, &descr, &module, &build_log);
    if (result != ZE_RESULT_SUCCESS) {
        build_log_string = get_build_log_string(build_log);
        throw std::runtime_error("zeModuleCreate failed: " + native::to_string(result) +
                                 ", Log: " + build_log_string);
    }

    result = zeModuleBuildLogDestroy(build_log);
    if (result) {
        throw std::runtime_error("zeModuleBuildLogDestroy failed: " + native::to_string(result));
    }

    ccl_device::device_module_ptr module_ptr(new device_module(module, get_ptr(), ctx));
    if (!modules.insert({ hash, module_ptr }).second) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - failed: hash conflict for: " + std::to_string(hash));
    }
    return module_ptr;
}

void CCL_API ccl_device::on_delete(ze_command_queue_handle_t& handle, ze_context_handle_t& ctx) {
    //todo not thread safe
    ze_result_t ret = zeCommandQueueDestroy(handle);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot destroy queue, error: ") +
                                 native::to_string(ret));
    }

    //TODO remove from map
}
void CCL_API ccl_device::on_delete(ze_command_list_handle_t& handle, ze_context_handle_t& ctx) {
    //todo not thread safe
    ze_result_t ret = zeCommandListDestroy(handle);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot destroy cmd list, error: ") +
                                 native::to_string(ret));
    }
    //TODO remove from map: cmd_lists;
}

void CCL_API ccl_device::on_delete(ze_device_handle_t& sub_device_handle, ze_context_handle_t& ctx) {
    auto& subdevices = get_subdevices();
    auto it = std::find_if(
        subdevices.begin(),
        subdevices.end(),
        [sub_device_handle](const typename sub_devices_container_type::value_type& subdevice) {
            return subdevice.second->get() == sub_device_handle;
        });

    if (it == subdevices.end()) {
        throw std::runtime_error(
            std::string("cannot destroy subdevice handle: orphant subddevice: ") +
            std::to_string(reinterpret_cast<size_t>(sub_device_handle)));
    }
    subdevices.erase(it);
}

void CCL_API ccl_device::on_delete(ze_module_handle_t& module_handle, ze_context_handle_t& ctx) {
    ze_result_t ret = zeModuleDestroy(module_handle);
    if (ret != ZE_RESULT_SUCCESS) {
        throw std::runtime_error(std::string("cannot destroy module handle, error: ") +
                                 native::to_string(ret));
    }
}

size_t ccl_device::serialize(std::vector<uint8_t>& out,
                             size_t from_pos,
                             size_t expected_size) const {
    // check parent existence
    const auto driver = get_owner().lock();
    if (!driver) {
        throw std::runtime_error("cannot serialize ccl_device without owner");
    }

    constexpr size_t expected_device_bytes = sizeof(device_properties.deviceId);
    size_t serialized_bytes = driver->serialize(
        out, from_pos, expected_device_bytes + expected_size); //resize vector inside

    // serialize from position
    uint8_t* data_start = out.data() + from_pos + serialized_bytes;
    *(reinterpret_cast<decltype(device_properties.deviceId)*>(data_start)) = get_device_id();
    serialized_bytes += expected_device_bytes;

    return serialized_bytes;
}

std::weak_ptr<ccl_device> ccl_device::deserialize(const uint8_t** data,
                                                  size_t& size,
                                                  ccl_device_platform& platform) {
    //restore driver
    auto driver = ccl_device_driver::deserialize(data, size, platform).lock();
    if (!driver) {
        throw std::runtime_error("cannot deserialize ccl_device, because owner is nullptr");
    }

    constexpr size_t expected_bytes = sizeof(device_properties.deviceId);
    if (size < expected_bytes) {
        throw std::runtime_error("cannot deserialize ccl_device, not enough data");
    }

    //restore device index
    decltype(device_properties.deviceId) recovered_index =
        *(reinterpret_cast<const decltype(device_properties.deviceId)*>(*data));
    size -= expected_bytes;
    *data += expected_bytes;

    //find device with requested handle
    ccl::device_index_type path{ driver->get_driver_id(),
                                 recovered_index,
                                 ccl::unused_index_value };
    auto device_ptr = driver->get_device(path);
    if (!device_ptr) {
        throw std::runtime_error(std::string(__FUNCTION__) +
                                 " - invalid device index: " + ccl::to_string(path));
    }
    return device_ptr;
}

std::string CCL_API ccl_device::to_string(const std::string& prefix) const {
    std::stringstream ss;
    ss << prefix << "Device: " << handle << std::endl;
    ss << prefix << "[\n";

    std::string param_prefix = prefix + "\t";
    std::string inner_param_prefix = std::string("\n\t") + param_prefix;
    ss << param_prefix << "DeviceProperties:\n"
       << param_prefix << "{" << param_prefix
       << native::to_string(device_properties, inner_param_prefix) << "\n"
       << param_prefix << "}\n"
       << param_prefix << "MemoryProp:\n"
       << param_prefix << "{";

    for (auto it = memory_properties.begin(); it != memory_properties.end(); ++it) {
        ss << param_prefix << native::to_string(*it, inner_param_prefix) << ",\n";
    }
    ss << param_prefix << "}\n";
    ss << param_prefix << "MemoryAccessProp:\n"
       << param_prefix << "{" << param_prefix
       << native::to_string(memory_access_properties, inner_param_prefix) << "\n"
       << param_prefix << "}\n"
       << param_prefix << "ComputeProp:\n"
       << param_prefix << "{" << param_prefix
       << native::to_string(compute_properties, inner_param_prefix) << "\n"
       << param_prefix << "}" << std::endl;

    ss << param_prefix << "Subdevices count: " << sub_devices.size() << std::endl;
    for (const auto& tile : sub_devices) {
        ss << tile.second->to_string(param_prefix);
    }
    ss << prefix << "],\n";
    return ss.str();
}

std::ostream& operator<<(std::ostream& out, const ccl_device& node) {
    out << "==========Device:==========\n" << node.to_string();
    if (!node.sub_devices.empty()) {
        out << "\n[\n";
        for (const auto& subdevice_ptr : node.sub_devices) {
            out << *subdevice_ptr.second << ",\n";
        }
        out << "\n]";
    }
    return out;
}
} // namespace native
#include "native_device_api/api_explicit_instantiation.hpp"
