#builds ccl

add_subdirectory(atl)

set(CCL_SRC
    ccl.cpp
    ccl_cpp_api.cpp
    atl/atl.cpp
    coll/algorithms/allgatherv.cpp
    coll/algorithms/allreduce/allreduce.cpp
    coll/algorithms/allreduce/allreduce_2d.cpp
    coll/algorithms/allreduce/allreduce_rma.cpp
    coll/algorithms/alltoall.cpp
    coll/algorithms/barrier.cpp
    coll/algorithms/bcast.cpp
    coll/algorithms/double_tree_ops.cpp
    coll/algorithms/reduce.cpp
    coll/algorithms/reduce_scatter.cpp
    coll/coll.cpp
    coll/selection/selector_allgatherv.cpp
    coll/selection/selector_allreduce.cpp
    coll/selection/selector_alltoall.cpp
    coll/selection/selector_barrier.cpp
    coll/selection/selector_bcast.cpp
    coll/selection/selector_reduce.cpp
    coll/selection/selector_reduce_scatter.cpp
    coll/selection/selector_sparse_allreduce.cpp
    comp/comp.cpp
    sched/sched.cpp
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
    common/datatype/datatype.cpp
    common/env/env.cpp
    common/global/global.cpp
    common/log/log.cpp
    common/stream/stream.cpp
    common/utils/spinlock.cpp
    common/utils/yield.cpp)

if(${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CCL_INC_DIRS "${CCL_INC_DIRS} ${SYCL_BUNDLE_ROOT}/include")
endif()
set(CCL_INC_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

#special library that holds objects only
add_library(ccl-objects OBJECT ${CCL_SRC})
set_target_properties(ccl-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_compile_definitions(ccl-objects PRIVATE PT_LOCK_SPIN=1)
target_compile_definitions(ccl-objects PRIVATE ATL_TRANSPORT_DL_DIR=\"${CCL_BUILD_DIR}\")
target_include_directories(ccl-objects PRIVATE ${CCL_INC_DIRS})

#shared library
add_library(ccl SHARED $<TARGET_OBJECTS:ccl-objects>)
target_include_directories(ccl PUBLIC ${CCL_INC_DIRS})
target_link_libraries(ccl PUBLIC dl pthread)
set_target_properties(ccl PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS ccl LIBRARY DESTINATION ${CCL_INSTALL_LIB})
if(${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
install(FILES
    "../cmake/FindOpenCL.cmake"
    "../cmake/FindSYCL.cmake"
    "../cmake/FindIntelSYCL.cmake"
    "../cmake/FindComputeCpp.cmake"
    DESTINATION ${CCL_INSTALL_LIB})
endif()

#static library
add_library(ccl-static STATIC $<TARGET_OBJECTS:ccl-objects>)
set_target_properties(ccl-static PROPERTIES OUTPUT_NAME ccl)
set_target_properties(ccl-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS ccl-static ARCHIVE DESTINATION ${CCL_INSTALL_LIB})

#headers installation
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
        DESTINATION ${CCL_INSTALL_INCLUDE} FILES_MATCHING REGEX ".*\\.(h|hpp)$")

#mpi & ofi rt
file(GLOB mpi_bins "${PROJECT_SOURCE_DIR}/mpi/bin/*")
install(PROGRAMS ${mpi_bins} DESTINATION ${CCL_INSTALL_BIN})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/ofi/lib/
        DESTINATION ${CCL_INSTALL_LIB})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/mpi/lib/
        DESTINATION ${CCL_INSTALL_LIB})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/mpi/etc/
        DESTINATION ${CCL_INSTALL_ETC})

install(DIRECTORY ${PROJECT_SOURCE_DIR}/mpi/licensing/
        DESTINATION ${CCL_INSTALL_LICENSE}/mpi/)
