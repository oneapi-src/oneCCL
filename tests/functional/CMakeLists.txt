cmake_minimum_required (VERSION 2.8)
file(GLOB sources "*.c" "*.cpp")

enable_testing()
ADD_SUBDIRECTORY (googletest-release-1.8.1/googletest/)
include_directories(${PROJECT_SOURCE_DIR}/tests/googletest-release-1.8.1/googletest/include ${PROJECT_SOURCE_DIR}/tests/googletest-release-1.8.1/googletest/src)
include_directories($ENV{CCL_ROOT}/include)
link_directories($ENV{CCL_ROOT}/lib)

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)
    add_executable(${executable} ${src})
    add_dependencies(${executable} gtest)
    add_dependencies(${executable} ccl)
    target_link_libraries(${executable}  PUBLIC gtest_main)
    target_link_libraries(${executable} PUBLIC gtest )
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PRIVATE ccl)
    install(TARGETS ${executable} RUNTIME DESTINATION ${CCL_INSTALL_TESTS})
    # FIXME: enable allreduce_custom for non-direct case only
    if(NOT ${executable} STREQUAL "allreduce_custom_test")
        add_test (NAME ${executable} CONFIGURATIONS Default COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/${executable} --gtest_output=xml:${CCL_INSTALL_TESTS}/${executable}_default_report.junit.xml)
    endif()
endforeach()

foreach(algo direct; naive; flat; multi_bcast)
add_test (NAME allgatherv_test_mpi_${algo} CONFIGURATIONS mpi_allgatherv_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allgatherv_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allgatherv_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()

foreach(algo direct; rabenseifner; starlike; ring; ring_rma; double_tree; recursive_doubling)
add_test (NAME allreduce_test_mpi_${algo} CONFIGURATIONS mpi_allreduce_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allreduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allreduce_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()

add_test (NAME allreduce_custom CONFIGURATIONS Custom COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/allreduce_custom_test --gtest_output=xml:${CCL_INSTALL_TESTS}/allreduce_custom_report.junit.xml)

foreach(algo direct; ring; double_tree; naive)
add_test (NAME bcast_test_mpi_${algo} CONFIGURATIONS mpi_bcast_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/bcast_test --gtest_output=xml:${CCL_INSTALL_TESTS}/bcast_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()

foreach(algo direct; rabenseifner; tree; double_tree)
add_test (NAME reduce_test_mpi_${algo} CONFIGURATIONS mpi_reduce_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/reduce_test --gtest_output=xml:${CCL_INSTALL_TESTS}/reduce_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()

foreach(algo direct; scatter; scatter_message)
add_test (NAME alltoall_test_mpi_${algo} CONFIGURATIONS mpi_alltoall_${algo} COMMAND mpiexec.hydra -l -n 2 -ppn 1 ${CCL_INSTALL_TESTS}/alltoall_test --gtest_output=xml:${CCL_INSTALL_TESTS}/alltoall_test_mpi_runtime_adjust_${algo}_report.junit.xml)
endforeach()
