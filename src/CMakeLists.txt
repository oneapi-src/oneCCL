#
# Copyright 2016-2020 Intel Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#builds ccl

set (EXTENSIONS_SRC)

if (CCL_ENABLE_SYCL)
    list (APPEND EXTENSIONS_SRC
                    ccl_cpp_gpu_api.cpp

                    native_device_api/l0/utils.cpp
                    native_device_api/sycl/export.cpp
                    native_device_api/interop_utils.cpp
                    #common/comm/l0/comm_context.cpp
                    #common/comm/l0/comm_context_storage.cpp
                    common/comm/single_device_communicator/single_device_communicator.cpp
        )
endif(CCL_ENABLE_SYCL)

if (MULTI_GPU_SUPPORT)
list (APPEND EXTENSIONS_SRC
                    ccl_cpp_gpu_api.cpp
                    ccl_gpu_modules.cpp
                    ccl_cpp_utils.cpp

                    native_device_api/l0/base.cpp
                    native_device_api/l0/device.cpp
                    native_device_api/l0/context.cpp
                    native_device_api/l0/subdevice.cpp
                    native_device_api/l0/driver.cpp
                    native_device_api/l0/export.cpp
                    native_device_api/l0/platform.cpp
                    native_device_api/l0/utils.cpp
                    native_device_api/l0/primitives.cpp
                    native_device_api/interop_utils.cpp
                    sched/gpu_sched.cpp
                    sched/gpu_concurrent_sched.cpp
                    common/event/impls/gpu_event.cpp

                    common/comm/l0/comm_context.cpp
                    common/comm/l0/comm_context_storage.cpp

                    common/comm/l0/devices/ccl_gpu_comm.cpp
                    common/comm/l0/devices/ccl_virtual_gpu_comm.cpp
                    common/comm/l0/devices/ccl_ipc_gpu_comm.cpp

                    common/comm/single_device_communicator/single_device_communicator.cpp
                    common/comm/l0/communicator/device_group/device_ring_communicator.cpp
                    common/comm/l0/communicator/device_group/device_a2a_communicator.cpp
                    common/comm/l0/communicator/thread_group/thread_ring_communicator.cpp
                    common/comm/l0/communicator/thread_group/thread_a2a_communicator.cpp
                    common/comm/l0/communicator/process_group/process_ring_communicator.cpp
                    common/comm/l0/communicator/process_group/process_a2a_communicator.cpp

                    common/comm/l0/context/process_group_ctx.cpp
                    common/comm/l0/context/thread_group_ctx.cpp
                    common/comm/l0/context/device_group_ctx.cpp
                    common/comm/l0/context/device_storage.cpp

                    common/comm/l0/topology/topology_serializer.cpp
                    common/comm/l0/topology/ring/device_group_ring_creator.cpp
                    common/comm/l0/topology/ring/thread_group_ring_creator.cpp
                    common/comm/l0/topology/ring/process_group_ring_creator.cpp
                    common/comm/l0/topology/topology_construction_utils.cpp

                    common/comm/l0/gpu_comm_attr.cpp
                    common/comm/l0/modules/base_entry_module.cpp
                    common/comm/l0/modules/modules_source_data.cpp)
endif(MULTI_GPU_SUPPORT)

set(CCL_SRC
    ccl.cpp
    ccl_cpp_api.cpp
    ccl_cpp_communicator.cpp
    ccl_cpp_environment.cpp
    ccl_api_functions.cpp
    ccl_app_api_coll_attr.cpp
    ccl_app_api_comm_split_attr.cpp
    ccl_app_api_datatype_attr.cpp
    ccl_app_api_event.cpp
    ccl_cpp_kvs.cpp
    ccl_cpp_device.cpp
    ccl_cpp_stream.cpp
    ccl_cpp_context.cpp
    ccl_cpp_utils.cpp
    ccl_empty_attr.cpp
    ccl_empty_coll_attr.cpp
    ccl_empty_stream.cpp
    native_device_api/sycl_l0/export.cpp
    native_device_api/empty/export.cpp
    atl/atl.cpp
    atl/atl_wrapper.cpp
    atl/mpi/atl_mpi.cpp
    atl/ofi/atl_ofi.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable_simple.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/helper.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs_keeper.c
    atl/util/pm/pmi_resizable_rt/pmi_resizable/pmi_listener.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/rank_list.c
    atl/util/pm/pmi_resizable_rt/pmi_resizable/request_wrappers_k8s.c
    atl/util/pm/pmi_resizable_rt/pmi_resizable/resizable_pmi.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/shift_list.c
        atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/users_kvs.cpp
    atl/util/pm/pmi_rt/pmi_simple.cpp
    atl/util/pm/pmi_rt/pmi/simple_pmi.c
    atl/util/pm/pmi_rt/pmi/simple_pmiutil.c
    coll/coll_common_attributes.cpp
    coll/ccl_allgather_op_attr.cpp
    coll/ccl_allreduce_op_attr.cpp
    coll/ccl_alltoall_op_attr.cpp
    coll/ccl_alltoallv_op_attr.cpp
    coll/ccl_bcast_op_attr.cpp
    coll/ccl_reduce_op_attr.cpp
    coll/ccl_reduce_scatter_op_attr.cpp
    coll/ccl_sparse_allreduce_op_attr.cpp
    coll/ccl_barrier_op_attr.cpp
    coll/algorithms/allgatherv.cpp
    coll/algorithms/allreduce/allreduce.cpp
    coll/algorithms/allreduce/allreduce_2d.cpp
    coll/algorithms/allreduce/allreduce_rma.cpp
    coll/algorithms/alltoall.cpp
    coll/algorithms/alltoallv.cpp
    coll/algorithms/barrier.cpp
    coll/algorithms/bcast.cpp
    coll/algorithms/double_tree_ops.cpp
    coll/algorithms/reduce.cpp
    coll/algorithms/reduce_scatter.cpp
    coll/coll.cpp
    coll/selection/selector_allgatherv.cpp
    coll/selection/selector_allreduce.cpp
    coll/selection/selector_alltoall.cpp
    coll/selection/selector_alltoallv.cpp
    coll/selection/selector_barrier.cpp
    coll/selection/selector_bcast.cpp
    coll/selection/selector_reduce.cpp
    coll/selection/selector_reduce_scatter.cpp
    coll/selection/selector_sparse_allreduce.cpp
    comp/bf16/bf16.cpp
    comp/bf16/bf16_intrisics.cpp
    comp/comp.cpp
    sched/sched.cpp
    sched/extra_sched.cpp
    sched/master_sched.cpp
    sched/sched_base.cpp
    sched/cache/cache.cpp
    sched/cache/key.cpp
    sched/queue/strict_queue.cpp
    sched/queue/queue.cpp
    sched/entry/coll/coll_entry.cpp
    sched/entry/coll/coll_entry_helper.cpp
    sched/entry/entry.cpp
    sched/entry/factory/chunked_entry_factory.cpp
    exec/exec.cpp
    exec/thread/base_thread.cpp
    exec/thread/listener.cpp
    exec/thread/service_worker.cpp
    exec/thread/worker.cpp
    fusion/fusion.cpp
    parallelizer/parallelizer.cpp
    unordered_coll/unordered_coll.cpp
    common/comm/atl_tag.cpp
    common/comm/comm.cpp
    common/comm/comm_interface.cpp
    common/comm/host_communicator/host_communicator.cpp

    common/context/context.cpp
    common/datatype/datatype.cpp
    common/device/device.cpp
    common/event/ccl_event.cpp
    common/event/event_internal/event_internal.cpp
    common/stream/stream.cpp

    common/env/env.cpp
    common/global/global.cpp
    common/log/log.cpp
    common/event/impls/host_event.cpp
    common/event/impls/native_event.cpp
    common/request/request.cpp
    common/utils/spinlock.cpp
    common/utils/yield.cpp
    ${EXTENSIONS_SRC})

list(APPEND CCL_INC_DIRS
                 ${PROJECT_SOURCE_DIR}/include
                 ${PROJECT_SOURCE_DIR}/mpi/include
                 ${LIBFABRIC_INCLUDE_DIR}
                 ${PROJECT_SOURCE_DIR}/src
                 ${PROJECT_SOURCE_DIR}/src/atl)

message(STATUS "CCL_INC_DIRS: ${CCL_INC_DIRS}")
message(STATUS "oneCCL lib LIBFABRIC_LIB_DIR: ${LIBFABRIC_LIB_DIR}")
message(STATUS "oneCCL lib LIBFABRIC_INCLUDE_DIR: ${LIBFABRIC_INCLUDE_DIR}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

#special library that holds objects only
add_library(ccl-objects OBJECT ${CCL_SRC})
set_target_properties(ccl-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_compile_definitions(ccl-objects PRIVATE PT_LOCK_SPIN=1)
target_compile_definitions(ccl-objects PRIVATE ATL_TRANSPORT_DL_DIR=\"${CCL_BUILD_DIR}\")
target_include_directories(ccl-objects PRIVATE ${CCL_INC_DIRS})

if(COMPUTE_RUNTIME_TARGET_NAME)
    target_include_directories(ccl-objects PRIVATE $<TARGET_PROPERTY:${COMPUTE_RUNTIME_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# add library search directory
link_directories(${PROJECT_SOURCE_DIR}/mpi/lib)
link_directories(${LIBFABRIC_LIB_DIR})
#shared library
add_library(ccl SHARED $<TARGET_OBJECTS:ccl-objects>)
target_include_directories(ccl PUBLIC ${CCL_INC_DIRS})
# link with release_mt libmpi.so for oneAPI Base toolkit
set_target_properties(ccl PROPERTIES LINK_FLAGS "-Wl,-rpath,../../../../mpi/latest/lib/release_mt/")
target_link_libraries(ccl PUBLIC dl pthread ${EXTERNAL_LIBS} ${COMPUTE_RUNTIME_TARGET_NAME} fabric mpi)
if (NOT LIB_SO_VERSION AND NOT LIB_MAJOR_VERSION)
        set_target_properties(ccl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
else()
        set_target_properties(ccl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR} VERSION ${LIB_SO_VERSION} SOVERSION ${LIB_MAJOR_VERSION})
endif()
install(TARGETS ccl LIBRARY DESTINATION ${CCL_INSTALL_LIB})
install(FILES
    "../cmake/FindOpenCL.cmake"
    "../cmake/FindL0.cmake"
    "../cmake/FindSYCL.cmake"
    "../cmake/FindIntelSYCL.cmake"
    "../cmake/FindComputeCpp.cmake"
    DESTINATION ${CCL_INSTALL_LIB})


#static library
add_library(ccl-static STATIC $<TARGET_OBJECTS:ccl-objects>)
set_target_properties(ccl-static PROPERTIES OUTPUT_NAME ccl)
set_target_properties(ccl-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS ccl-static ARCHIVE DESTINATION ${CCL_INSTALL_LIB})

if(MULTI_GPU_SUPPORT)
    message("Turn on L0 multi-gpu unit tests")
    add_library(ccl-ut-static STATIC $<TARGET_OBJECTS:ccl-objects>)
    target_link_libraries(ccl-ut-static PUBLIC dl pthread ${EXTERNAL_LIBS} ${COMPUTE_RUNTIME_TARGET_NAME} fabric mpi)
    set_target_properties(ccl-ut-static PROPERTIES COMPILE_FLAGS "-fvisibility=default -DUNIT_TESTS")
    set_target_properties(ccl-ut-static PROPERTIES OUTPUT_NAME cclut)
    set_target_properties(ccl-ut-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
    set_target_properties(ccl-ut-static PROPERTIES INTERFACE_INCLUDE_DIRECTORIES  $<TARGET_PROPERTY:${COMPUTE_RUNTIME_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>)
endif(MULTI_GPU_SUPPORT)

#headers installation
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CCL_INSTALL_INCLUDE} FILES_MATCHING REGEX ".*\\.(h|hpp)$")

#mpi & ofi rt
file(GLOB mpi_bins "${PROJECT_SOURCE_DIR}/mpi/bin/*")
install(PROGRAMS ${mpi_bins} DESTINATION ${CCL_INSTALL_BIN})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/ofi/lib/
        DESTINATION ${CCL_INSTALL_LIB})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/mpi/include/
        DESTINATION ${CCL_INSTALL_INCLUDE})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/mpi/lib/
        DESTINATION ${CCL_INSTALL_LIB})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/mpi/etc/
        DESTINATION ${CCL_INSTALL_ETC})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/mpi/licensing/
        DESTINATION ${CCL_INSTALL_LICENSE}/mpi/)
