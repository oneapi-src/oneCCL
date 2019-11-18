#builds resizable_pmi

set(KVS_SRC
    helper.c
    rank_list.c
    shift_list.c
    kvs_keeper.c
    resizable_pmi.c
    listener.c
    request_wrappers_k8s.c
    kvs/kvs.c)

set(COMMON_KVS_INC_DIRS
    ${PROJECT_SOURCE_DIR}/src/atl/util/pm/pmi_resizable_rt/pmi_resizable
    ${PROJECT_SOURCE_DIR}/src/atl/util/pm/pmi_resizable_rt/pmi_resizable/kvs
    ${PROJECT_SOURCE_DIR}/ofi_rt/include)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pthread")

#special library that holds objects only
add_library(resizable_pmi-objects OBJECT ${KVS_SRC})
set_target_properties(resizable_pmi-objects PROPERTIES POSITION_INDEPENDENT_CODE 1)
target_include_directories(resizable_pmi-objects PUBLIC ${COMMON_KVS_INC_DIRS})
target_compile_definitions(resizable_pmi-objects PRIVATE HAVE_UNISTD_H HAVE_STDLIB_H HAVE_STRING_H HAVE_STRINGS_H)

#shared lib
add_library(resizable_pmi SHARED $<TARGET_OBJECTS:resizable_pmi-objects>)
target_include_directories(resizable_pmi PUBLIC ${COMMON_KVS_INC_DIRS})
set_target_properties(resizable_pmi PROPERTIES VERSION 1 SOVERSION 1.0)
set_target_properties(resizable_pmi PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})

install(TARGETS resizable_pmi LIBRARY DESTINATION ${CCL_INSTALL_LIB})

#static lib
add_library(resizable_pmi-static STATIC $<TARGET_OBJECTS:resizable_pmi-objects>)
set_target_properties(resizable_pmi-static PROPERTIES OUTPUT_NAME resizable_pmi)
set_target_properties(resizable_pmi-static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CCL_BUILD_DIR})
install(TARGETS resizable_pmi-static ARCHIVE DESTINATION ${CCL_INSTALL_LIB})
