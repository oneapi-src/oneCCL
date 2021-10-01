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
#include <algorithm>
#include <fstream>

#include "common/global/global.hpp"
#include "common/log/log.hpp"
#include "sched/entry/gpu/ze_primitives.hpp"

namespace ccl {

namespace ze {

void load_module(std::string dir,
                 std::string file_name,
                 ze_device_handle_t device,
                 ze_context_handle_t context,
                 ze_module_handle_t* module) {
    LOG_DEBUG("module loading started: directory: ", dir, ", file: ", file_name);

    if (!dir.empty()) {
        if (*dir.rbegin() != '/') {
            dir += '/';
        }
    }

    std::string file_path = dir + file_name;
    std::ifstream file(file_path, std::ios_base::in | std::ios_base::binary);
    if (!file.good() || dir.empty() || file_name.empty()) {
        CCL_THROW("failed to load module: file: ", file_path);
    }

    file.seekg(0, file.end);
    size_t filesize = file.tellg();
    file.seekg(0, file.beg);

    std::vector<uint8_t> module_data(filesize);
    file.read(reinterpret_cast<char*>(module_data.data()), filesize);
    file.close();

    ze_module_desc_t desc = {};
    ze_module_format_t format = ZE_MODULE_FORMAT_IL_SPIRV;
    desc.format = format;
    desc.pInputModule = reinterpret_cast<const uint8_t*>(module_data.data());
    desc.inputSize = module_data.size();
    ZE_CALL(zeModuleCreate, (context, device, &desc, module, nullptr));
    LOG_DEBUG("module loading completed: directory: ", dir, ", file: ", file_name);
}

void create_kernel(ze_module_handle_t module, std::string kernel_name, ze_kernel_handle_t* kernel) {
    ze_kernel_desc_t desc = default_kernel_desc;
    // convert to lowercase
    std::transform(kernel_name.begin(), kernel_name.end(), kernel_name.begin(), ::tolower);
    desc.pKernelName = kernel_name.c_str();
    ze_result_t res = zeKernelCreate(module, &desc, kernel);
    if (res != ZE_RESULT_SUCCESS) {
        CCL_THROW("error at zeKernelCreate: kernel name: ", kernel_name, " ret: ", to_string(res));
    }
}

void get_suggested_group_size(ze_kernel_handle_t kernel,
                              size_t count,
                              ze_group_size_t* group_size) {
    CCL_ASSERT(count > 0, "count == 0");
    ZE_CALL(zeKernelSuggestGroupSize,
            (kernel,
             count,
             1,
             1,
             &group_size->groupSizeX,
             &group_size->groupSizeY,
             &group_size->groupSizeZ));
    CCL_THROW_IF_NOT(group_size->groupSizeX >= 1,
                     "wrong group size calculation: group size: ",
                     to_string(*group_size),
                     ", count: ",
                     count);
}

void get_suggested_group_count(const ze_group_size_t& group_size,
                               size_t count,
                               ze_group_count_t* group_count) {
    group_count->groupCountX = count / group_size.groupSizeX;
    group_count->groupCountY = 1;
    group_count->groupCountZ = 1;

    auto rem = count % group_size.groupSizeX;
    CCL_THROW_IF_NOT(group_count->groupCountX >= 1 && rem == 0,
                     "wrong group count calculation: group size: ",
                     to_string(group_size),
                     ", group count: ",
                     to_string(*group_count),
                     ", count: ",
                     std::to_string(count));
}

void set_kernel_args(ze_kernel_handle_t kernel, const ze_kernel_args_t& kernel_args) {
    uint32_t idx = 0;
    for (const auto& arg : kernel_args) {
        auto res = zeKernelSetArgumentValue(kernel, idx, arg.first, arg.second);
        if (res != ZE_RESULT_SUCCESS) {
            CCL_THROW("zeKernelSetArgumentValue failed with error ",
                      to_string(res),
                      " on idx ",
                      idx,
                      " with value ",
                      *((void**)arg.second));
        }
        ++idx;
    }
}

void get_num_queue_groups(ze_device_handle_t device, uint32_t* num) {
    *num = 0;
    ZE_CALL(zeDeviceGetCommandQueueGroupProperties, (device, num, nullptr));
    CCL_THROW_IF_NOT(*num != 0, "no queue groups found");
}

void get_queues_properties(ze_device_handle_t device,
                           uint32_t num_queue_groups,
                           ze_queue_properties_t* props) {
    props->resize(num_queue_groups);
    ZE_CALL(zeDeviceGetCommandQueueGroupProperties, (device, &num_queue_groups, props->data()));
}

void get_comp_queue_ordinal(ze_device_handle_t device,
                            const ze_queue_properties_t& props,
                            uint32_t* ordinal) {
    uint32_t comp_ordinal = std::numeric_limits<uint32_t>::max();

    for (uint32_t idx = 0; idx < props.size(); ++idx) {
        if (props[idx].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            comp_ordinal = idx;
            break;
        }
    }

    LOG_DEBUG("find queue: { ordinal: ",
              comp_ordinal,
              ", queue properties params: ",
              to_string(props[comp_ordinal]),
              " }");

    if (comp_ordinal != std::numeric_limits<uint32_t>::max()) {
        *ordinal = comp_ordinal;
    }
    else {
        LOG_WARN("could not find queue ordinal, ordinal 0 will be used");
        *ordinal = 0;
    }
}

void get_copy_queue_ordinal(ze_device_handle_t device,
                            const ze_queue_properties_t& props,
                            uint32_t* ordinal) {
    uint32_t copy_ordinal = std::numeric_limits<uint32_t>::max();

    for (uint32_t idx = 0; idx < props.size(); ++idx) {
        /* only compute property */
        if ((props[idx].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) &&
            global_data::env().ze_copy_engine == ccl_ze_copy_engine_none) {
            copy_ordinal = idx;
            break;
        }

        /* only copy property */
        if ((props[idx].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) &&
            ((props[idx].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) == 0)) {
            /* main */
            if (props[idx].numQueues == 1 &&
                global_data::env().ze_copy_engine == ccl_ze_copy_engine_main) {
                copy_ordinal = idx;
                break;
            }
            /* link */
            if (props[idx].numQueues > 1 &&
                global_data::env().ze_copy_engine == ccl_ze_copy_engine_link) {
                copy_ordinal = idx;
                break;
            }
        }
    }

    LOG_DEBUG("find copy queue: { ordinal: ",
              copy_ordinal,
              ", queue properties params: ",
              to_string(props[copy_ordinal]),
              " }");

    if (copy_ordinal != std::numeric_limits<uint32_t>::max()) {
        *ordinal = copy_ordinal;
    }
    else {
        LOG_WARN("could not find queue ordinal for copy engine mode: ",
                 global_data::env().ze_copy_engine,
                 ", ordinal 0 will be used");
        *ordinal = 0;
    }
}

void get_queue_index(const ze_queue_properties_t& props,
                     uint32_t ordinal,
                     int idx,
                     uint32_t* index) {
    CCL_ASSERT(props.size() > ordinal, "props.size() <= ordinal");
    *index = idx % props[ordinal].numQueues;
    LOG_DEBUG("set queue index: ", *index);
}

std::string to_string(const ze_result_t result) {
    switch (result) {
        case ZE_RESULT_SUCCESS: return "ZE_RESULT_SUCCESS";
        case ZE_RESULT_NOT_READY: return "ZE_RESULT_NOT_READY";
        case ZE_RESULT_ERROR_DEVICE_LOST: return "ZE_RESULT_ERROR_DEVICE_LOST";
        case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
        case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY: return "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
        case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE: return "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
        case ZE_RESULT_ERROR_MODULE_LINK_FAILURE: return "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
        case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
            return "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
        case ZE_RESULT_ERROR_NOT_AVAILABLE: return "ZE_RESULT_ERROR_NOT_AVAILABLE";
        case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
            return "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
        case ZE_RESULT_ERROR_UNINITIALIZED: return "ZE_RESULT_ERROR_UNINITIALIZED";
        case ZE_RESULT_ERROR_UNSUPPORTED_VERSION: return "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
        case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE: return "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
        case ZE_RESULT_ERROR_INVALID_ARGUMENT: return "ZE_RESULT_ERROR_INVALID_ARGUMENT";
        case ZE_RESULT_ERROR_INVALID_NULL_HANDLE: return "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
        case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE: return "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
        case ZE_RESULT_ERROR_INVALID_NULL_POINTER: return "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
        case ZE_RESULT_ERROR_INVALID_SIZE: return "ZE_RESULT_ERROR_INVALID_SIZE";
        case ZE_RESULT_ERROR_UNSUPPORTED_SIZE: return "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
        case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT: return "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
        case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
            return "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
        case ZE_RESULT_ERROR_INVALID_ENUMERATION: return "ZE_RESULT_ERROR_INVALID_ENUMERATION";
        case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
            return "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
        case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
            return "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
        case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY: return "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
        case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME: return "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
        case ZE_RESULT_ERROR_INVALID_KERNEL_NAME: return "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
        case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME: return "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
        case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
            return "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
        case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
            return "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
            return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
            return "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
        case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
            return "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
        case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
            return "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
        case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
            return "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
        case ZE_RESULT_ERROR_OVERLAPPING_REGIONS: return "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
        case ZE_RESULT_ERROR_UNKNOWN: return "ZE_RESULT_ERROR_UNKNOWN";
        case ZE_RESULT_FORCE_UINT32: return "ZE_RESULT_FORCE_UINT32";
        default: return "unknown ze_result_t value: " + std::to_string(static_cast<int>(result));
    }
}

std::string to_string(const ze_group_size_t& group_size) {
    std::stringstream ss;
    ss << "{ x: " << group_size.groupSizeX << ", y: " << group_size.groupSizeY
       << ", z: " << group_size.groupSizeZ << " }";
    return ss.str();
}

std::string to_string(const ze_group_count_t& group_count) {
    std::stringstream ss;
    ss << "{ x: " << group_count.groupCountX << ", y: " << group_count.groupCountY
       << ", z: " << group_count.groupCountZ << " }";
    return ss.str();
}

std::string to_string(const ze_kernel_args_t& kernel_args) {
    std::stringstream ss;
    ss << "{\n";
    size_t idx = 0;
    for (const auto& arg : kernel_args) {
        // TODO: can we distinguish argument types in order to properly print them instead of printing
        // as a void* ptr?
        ss << "  idx: " << idx << ", { " << arg.first << ", " << *(void**)arg.second << " }\n";
        ++idx;
    }
    ss << "}";
    return ss.str();
}

std::string to_string(const ze_command_queue_group_property_flag_t& flag) {
    switch (flag) {
        case ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE:
            return "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE";
        case ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY:
            return "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY";
        case ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS:
            return "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COOPERATIVE_KERNELS";
        case ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS:
            return "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_METRICS";
        case ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32:
            return "ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_FORCE_UINT32";
        default:
            return "unknown ze_command_queue_group_property_flag_t value: " +
                   std::to_string(static_cast<int>(flag));
    }
}

std::string to_string(const ze_command_queue_group_properties_t& queue_property) {
    std::stringstream ss;
    ss << "stype: " << queue_property.stype << ", pNext: " << (void*)queue_property.pNext
       << ", flags: "
       << flags_to_string<ze_command_queue_group_property_flag_t>(queue_property.flags)
       << ", maxMemoryFillPatternSize: " << queue_property.maxMemoryFillPatternSize
       << ", numQueues: " << queue_property.numQueues;
    return ss.str();
}

std::string join_strings(const std::vector<std::string>& tokens, const std::string& delimeter) {
    std::stringstream ss;
    for (size_t i = 0; i < tokens.size(); ++i) {
        ss << tokens[i];
        if (i < tokens.size() - 1) {
            ss << delimeter;
        }
    }
    return ss.str();
}

template <typename T>
std::string flags_to_string(uint32_t flags) {
    const size_t bits = 8;
    std::vector<std::string> output;
    for (size_t i = 0; i < sizeof(flags) * bits; ++i) {
        const size_t mask = 1UL << i;
        const auto flag = flags & mask;
        if (flag != 0) {
            output.emplace_back(to_string(static_cast<T>(flag)));
        }
    }
    return join_strings(output, " | ");
}

} // namespace ze
} // namespace ccl
