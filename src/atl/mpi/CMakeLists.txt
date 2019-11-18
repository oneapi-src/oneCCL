
set(MPI_SRC
    atl_mpi.c)

set(COMMON_MPI_INC_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${PROJECT_SOURCE_DIR}/mpi_rt/include/)

# special library that holds objects only
add_library(ccl_atl_mpi-objects OBJECT ${MPI_SRC})
set_target_properties(ccl_atl_mpi-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_include_directories(ccl_atl_mpi-objects PRIVATE ${COMMON_MPI_INC_DIRS})

# add library search directory
link_directories(${PROJECT_SOURCE_DIR}/mpi_rt/lib)

# shared
add_library(ccl_atl_mpi SHARED $<TARGET_OBJECTS:ccl_atl_mpi-objects>)
target_include_directories(ccl_atl_mpi PRIVATE ${COMMON_MPI_INC_DIRS})

# target_link_libraries(ccl_atl_mpi PRIVATE fabric m)
target_link_libraries(ccl_atl_mpi PRIVATE mpi)
set_target_properties(ccl_atl_mpi PROPERTIES VERSION 1 SOVERSION 1.0)
set_target_properties(ccl_atl_mpi PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS ccl_atl_mpi LIBRARY DESTINATION ${CCL_INSTALL_LIB})

# static
add_library(ccl_atl_mpi-static STATIC $<TARGET_OBJECTS:ccl_atl_mpi-objects>)
set_target_properties(ccl_atl_mpi-static PROPERTIES OUTPUT_NAME ccl_atl_mpi)
set_target_properties(ccl_atl_mpi-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS ccl_atl_mpi-static ARCHIVE DESTINATION ${CCL_INSTALL_LIB})
