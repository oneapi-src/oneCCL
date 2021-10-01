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
# builds CCL

set (EXTENSIONS_SRC)

if (CCL_ENABLE_SYCL)
    list (APPEND EXTENSIONS_SRC
                    native_device_api/l0/utils.cpp
                    native_device_api/sycl/export.cpp
                    native_device_api/interop_utils.cpp
        )
endif(CCL_ENABLE_SYCL)

if (MULTI_GPU_SUPPORT)
list (APPEND EXTENSIONS_SRC
                    ccl_cpp_utils.cpp

                    native_device_api/l0/base.cpp
                    native_device_api/l0/device.cpp
                    native_device_api/l0/context.cpp
                    native_device_api/l0/event_pool.cpp
                    native_device_api/l0/subdevice.cpp
                    native_device_api/l0/driver.cpp
                    native_device_api/l0/export.cpp
                    native_device_api/l0/platform.cpp
                    native_device_api/l0/utils.cpp
                    native_device_api/l0/primitives.cpp
                    native_device_api/interop_utils.cpp

                    common/comm/l0/comm_context.cpp
                    common/comm/l0/comm_context_storage.cpp
                    common/comm/l0/context_comm_addr.cpp

                    common/comm/l0/devices/ccl_gpu_base_comm.cpp
                    common/comm/l0/devices/ccl_gpu_comm.cpp
                    common/comm/l0/devices/ccl_virtual_gpu_comm.cpp
                    common/comm/l0/devices/ccl_ipc_gpu_comm.cpp

                    common/comm/l0/devices/communication_structs/connection.cpp
                    common/comm/l0/devices/communication_structs/ipc_connection.cpp
                    common/comm/l0/devices/communication_structs/ipc_server.cpp
                    common/comm/l0/devices/communication_structs/ipc_client.cpp

                    common/comm/l0/context/process_group_ctx.cpp
                    common/comm/l0/context/thread_group_ctx.cpp
                    common/comm/l0/context/device_group_ctx.cpp
                    common/comm/l0/context/device_storage.cpp

                    common/comm/l0/topology/topology_serializer.cpp
                    common/comm/l0/topology/ring/device_group_ring_creator.cpp
                    common/comm/l0/topology/ring/thread_group_ring_creator.cpp
                    common/comm/l0/topology/ring/process_group_ring_creator.cpp
                    common/comm/l0/topology/topology_construction_utils.cpp

                    common/comm/l0/context/scale/ipc/ipc_ctx_session.cpp
                    common/comm/l0/context/scale/ipc/ipc_ctx_utils.cpp
                    common/comm/l0/context/scale/ipc/ipc_session_key.cpp

                    common/comm/l0/context/scale/base/base_session.cpp
                    common/comm/l0/context/scale/scale_out/scale_out_session.cpp

                    common/comm/l0/gpu_comm_attr.cpp
                    common/comm/l0/modules/base_entry_module.cpp
                    common/comm/l0/modules/modules_source_data.cpp
                    common/comm/l0/modules/kernel_utils.cpp

                    sched/gpu_sched.cpp
                    sched/gpu_concurrent_sched.cpp
                    sched/entry/gpu/ze_cache.cpp
                    sched/entry/gpu/ze_call.cpp
                    sched/entry/gpu/ze_primitives.cpp)
endif(MULTI_GPU_SUPPORT)

if (CCL_ENABLE_SYCL AND MULTI_GPU_SUPPORT)
    list (APPEND EXTENSIONS_SRC
                    sched/entry/gpu/ze_base_entry.cpp
                    sched/entry/gpu/ze_allreduce_entry.cpp
                    sched/entry/gpu/ze_copy_entry.cpp
                    sched/entry/gpu/ze_handle_exchange_entry.cpp
                    sched/entry/gpu/ze_event_signal_entry.cpp
                    sched/entry/gpu/ze_event_wait_entry.cpp
                    sched/entry/gpu/ze_reduce_entry.cpp
                    sched/entry/reduce_local_entry.cpp
                    sched/ze_handle_manager.cpp
        )
endif(CCL_ENABLE_SYCL AND MULTI_GPU_SUPPORT)

set(CCL_SRC
    ccl_cpp_communicator.cpp
    ccl_cpp_environment.cpp
    ccl_api_functions.cpp
    ccl_app_api_coll_attr.cpp
    ccl_app_api_comm_attr.cpp
    ccl_app_api_comm_split_attr.cpp
    ccl_app_api_datatype_attr.cpp
    ccl_app_api_kvs_attr.cpp
    ccl_app_api_event.cpp
    ccl_app_api_init_attr.cpp
    ccl_cpp_kvs.cpp
    ccl_cpp_device.cpp
    ccl_cpp_stream.cpp
    ccl_cpp_context.cpp
    ccl_cpp_utils.cpp
    ccl_empty_attr.cpp
    ccl_empty_coll_attr.cpp
    ccl_empty_comm_attr.cpp
    ccl_empty_init_attr.cpp
    ccl_empty_comm_split_attr.cpp
    ccl_empty_kvs_attr.cpp
    ccl_empty_stream.cpp
    native_device_api/sycl_l0/export.cpp
    native_device_api/empty/export.cpp
    atl/atl_wrapper.cpp
    atl/mpi/atl_mpi.cpp
    atl/ofi/atl_ofi.cpp
    atl/ofi/atl_ofi_helper.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable_simple.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable_simple_internal.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/helper.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs_keeper.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/pmi_listener.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/rank_list.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/request_wrappers_k8s.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/resizable_pmi.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/shift_list.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs.cpp
    atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs/internal_kvs_server.cpp
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
    coll/coll_param.cpp
    coll/algorithms/allgatherv.cpp
    coll/algorithms/allreduce/allreduce.cpp
    coll/algorithms/allreduce/allreduce_2d.cpp
    coll/algorithms/allreduce/allreduce_rma.cpp
    coll/algorithms/algorithm_utils.cpp
    coll/algorithms/alltoall.cpp
    coll/algorithms/alltoallv.cpp
    coll/algorithms/barrier.cpp
    coll/algorithms/bcast.cpp
    coll/algorithms/double_tree_ops.cpp
    coll/algorithms/reduce.cpp
    coll/algorithms/reduce_scatter.cpp
    coll/coll.cpp
    coll/coll_check.cpp
    coll/selection/selection.cpp
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
    comp/fp16/fp16.cpp
    comp/fp16/fp16_intrisics.cpp
    hwloc/hwloc_wrapper.cpp
    sched/sched.cpp
    sched/buffer_cache.cpp
    sched/extra_sched.cpp
    sched/master_sched.cpp
    sched/sched_base.cpp
    sched/sched_timer.cpp
    sched/cache/cache.cpp
    sched/cache/key.cpp
    sched/queue/flow_control.cpp
    sched/queue/strict_queue.cpp
    sched/queue/queue.cpp
    sched/entry/coll/coll_entry.cpp
    sched/entry/coll/coll_entry_helper.cpp
    sched/entry/copy/copy_entry.cpp
    sched/entry/copy/copy_helper.cpp
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
    common/comm/compiler_comm_interface_dispatcher.cpp
    common/comm/host_communicator/host_communicator.cpp

    common/context/context.cpp
    common/datatype/datatype.cpp
    common/device/device.cpp
    common/event/ccl_event.cpp
    common/stream/stream.cpp

    common/env/env.cpp
    common/global/global.cpp
    common/log/log.cpp
    common/event/impls/host_event.cpp
    common/event/impls/native_event.cpp
    common/framework/framework.cpp
    common/request/request.cpp
    common/utils/spinlock.cpp
    common/utils/version.cpp
    common/utils/yield.cpp

    ${EXTENSIONS_SRC})

set(SRC_C_FLAGS)
set(SRC_CXX_FLAGS)
set(SRC_SHARED_LINKER_FLAGS)
set(SRC_INCLUDE_DIRS)
set(SRC_LINK_DIRS)
set(SRC_LINK_LIBS)

# common settings of security options
if (USE_SECURITY_FLAGS)
    set(SRC_C_FLAGS "${SRC_C_FLAGS} -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector")
    set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector")
    set(SRC_SHARED_LINKER_FLAGS "${SRC_SHARED_LINKER_FLAGS} -fPIE -fPIC -z noexecstack -z relro -z now")
    if (${CMAKE_C_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            set(SRC_C_FLAGS "${SRC_C_FLAGS} -fstack-protector-strong")
            set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -fstack-protector-strong")
        endif()
    endif()
endif()

set(SRC_SHARED_LINKER_FLAGS "${SRC_SHARED_LINKER_FLAGS} -Wl,--version-script=${PROJECT_SOURCE_DIR}/ccl.map")

if (${CMAKE_C_COMPILER_ID} STREQUAL "Intel" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    if (USE_CODECOV_FLAGS)
        set(SRC_C_FLAGS "${SRC_C_FLAGS} -prof-gen=srcpos -prof-src-root-cwd")
        set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -prof-gen=srcpos -prof-src-root-cwd")
    endif()
endif()

list(APPEND SRC_INCLUDE_DIRS
     ${PROJECT_SOURCE_DIR}/include
     ${PROJECT_SOURCE_DIR}/src
     ${PROJECT_SOURCE_DIR}/src/atl
     ${LIBFABRIC_INCLUDE_DIR}
     ${HWLOC_INCLUDE_DIR})

list(APPEND SRC_LINK_DIRS ${LIBFABRIC_LIB_DIR})

list(APPEND SRC_LINK_LIBS
     dl
     pthread
     ${EXTERNAL_LIBS}
     ${COMPUTE_BACKEND_TARGET_NAME}
     fabric
     ${HWLOC_LIB_DIR}/libhwloc.a)

if (ENABLE_MPI)
    set(SRC_C_FLAGS "${SRC_C_FLAGS} -DCCL_ENABLE_MPI")
    set(SRC_CXX_FLAGS "${SRC_CXX_FLAGS} -DCCL_ENABLE_MPI")
    list(APPEND SRC_INCLUDE_DIRS ${MPI_INCLUDE_DIR})
    list(APPEND SRC_LINK_DIRS ${MPI_LIB_DIR})
    list(APPEND SRC_LINK_LIBS mpi)
endif()

link_directories(${SRC_LINK_DIRS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SRC_C_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SRC_CXX_FLAGS} -pthread")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SRC_SHARED_LINKER_FLAGS}")

# special library that holds objects only
add_library(ccl-objects OBJECT ${CCL_SRC})
set_target_properties(ccl-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_include_directories(ccl-objects PRIVATE ${SRC_INCLUDE_DIRS})

if (COMPUTE_BACKEND_TARGET_NAME)
    target_include_directories(ccl-objects PRIVATE $<TARGET_PROPERTY:${COMPUTE_BACKEND_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>)
endif()

# shared library
add_library(ccl SHARED $<TARGET_OBJECTS:ccl-objects>)
target_include_directories(ccl PUBLIC ${SRC_INCLUDE_DIRS})

# link with release_mt libmpi.so for oneAPI Base toolkit
# libccl.so -> cpu_icc/cpu_gpu_dpcpp -> lib -> latest -> ccl -> mpi -> ...
set(ONEAPI_IMPI_RPATH "'$ORIGIN'/../../../../mpi/latest/lib/release_mt/")
set_target_properties(ccl PROPERTIES LINK_FLAGS "-Wl,-rpath,${ONEAPI_IMPI_RPATH}")

target_link_libraries(ccl PUBLIC ${SRC_LINK_LIBS})

if (NOT LIB_SO_VERSION AND NOT LIB_MAJOR_VERSION)
    set_target_properties(ccl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
else()
    set_target_properties(ccl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR} VERSION ${LIB_SO_VERSION} SOVERSION ${LIB_MAJOR_VERSION})
endif()

message(STATUS "SRC C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "SRC CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "SRC SHARED_LINKER_FLAGS: ${CMAKE_SHARED_LINKER_FLAGS}")
message(STATUS "SRC INCLUDE_DIRS: ${SRC_INCLUDE_DIRS}")
message(STATUS "SRC LINK_DIRS: ${SRC_LINK_DIRS}")
message(STATUS "SRC LINK_LIBS: ${SRC_LINK_LIBS}")

install(TARGETS ccl LIBRARY DESTINATION ${CCL_INSTALL_LIB})
install(FILES
    "${PROJECT_SOURCE_DIR}/cmake/FindComputeCpp.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/FindIntelSYCL.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/FindIntelSYCL_level_zero.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/Findlevel_zero.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/FindNUMA.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/FindOpenCL.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/FindSYCL.cmake"
    DESTINATION ${CCL_INSTALL_LIB})

# static library
add_library(ccl-static STATIC $<TARGET_OBJECTS:ccl-objects>)
set_target_properties(ccl-static PROPERTIES OUTPUT_NAME ccl)
set_target_properties(ccl-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS ccl-static ARCHIVE DESTINATION ${CCL_INSTALL_LIB} OPTIONAL)

# API headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CCL_INSTALL_INCLUDE} FILES_MATCHING REGEX ".*\\.(h|hpp)$")

if ("${LIBFABRIC_DIR}" STREQUAL "")
    # internal libfabric is used, install it into package
    install(DIRECTORY ${DEPS_DIR}/ofi/lib/
            DESTINATION ${CCL_INSTALL_LIB})
endif()

if (ENABLE_MPI)
    file(GLOB mpi_bins "${DEPS_DIR}/mpi/bin/*")
    install(PROGRAMS ${mpi_bins} DESTINATION ${CCL_INSTALL_BIN})
    install(DIRECTORY ${DEPS_DIR}/mpi/include/
            DESTINATION ${CCL_INSTALL_INCLUDE})
    install(DIRECTORY ${DEPS_DIR}/mpi/lib/
           DESTINATION ${CCL_INSTALL_LIB})
    install(DIRECTORY ${DEPS_DIR}/mpi/etc/
            DESTINATION ${CCL_INSTALL_ETC})
    install(DIRECTORY ${DEPS_DIR}/mpi/licensing/
            DESTINATION ${CCL_INSTALL_LICENSE}/mpi/)
endif()
