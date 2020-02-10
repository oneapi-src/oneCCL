#common CMakeList.txt to build CCL, ATL, tests

cmake_minimum_required (VERSION 2.8)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

set(PROJECT_NAME "oneCCL")
set(PROJECT_FULL_NAME "oneAPI Collective Communications Library")

project(${PROJECT_NAME})

include(cmake/helpers.cmake)

check_compiler_version()

#set default build types.
#Available build types are: Debug, Release, RelWithDebInfo and MinSizeRel
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
if (NOT CMAKE_DISABLE_SYCL)
    set(CMAKE_DISABLE_SYCL 0)
endif()

#show build info
message(STATUS "Installation directory: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C compiler : ${CMAKE_C_COMPILER}")
message(STATUS "CXX compiler : ${CMAKE_CXX_COMPILER}")

#predefined options
option(USE_SECURITY_FLAGS "Use security flags" TRUE)
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set(USE_SECURITY_FLAGS FALSE)
endif()

option(USE_CODECOV_FLAGS "Calculate code coverage" FALSE)
option(WITH_ASAN "Use address sanitizer, can only be used in Debug build" FALSE)

#installation path variables
include(GNUInstallDirs)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/_install" CACHE PATH "Default install path" FORCE)
endif()

set(CMAKE_INSTALL_LIBDIR "lib")
set(CCL_INSTALL_LIB "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(CCL_INSTALL_INCLUDE "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
set(CCL_INSTALL_DOC "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DOCDIR}")
set(CCL_INSTALL_BIN "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
set(CCL_INSTALL_ENV "${CMAKE_INSTALL_PREFIX}/env")
set(CCL_INSTALL_ETC "${CMAKE_INSTALL_PREFIX}/etc")
set(CCL_INSTALL_LICENSE "${CMAKE_INSTALL_PREFIX}/licensing")
set(CCL_INSTALL_MODULE "${CMAKE_INSTALL_PREFIX}/modulefiles")
set(CCL_INSTALL_EXAMPLES "${CMAKE_INSTALL_PREFIX}/examples")
set(CCL_INSTALL_TESTS "${CMAKE_INSTALL_PREFIX}/tests")

set(CMAKE_SKIP_INSTALL_RPATH TRUE)
set(CMAKE_SKIP_RPATH TRUE)

if (${CMAKE_VERSION} VERSION_LESS 3.1)
#cmake version below 3.1 does not support CMAKE_C[XX}_STANDARD flags
#set manually
    set(CXX_COMPILER_FLAGS "-std=gnu++11")
    set(C_COMPILER_FLAGS "-std=gnu99")
endif()

#common settings of security options
if(USE_SECURITY_FLAGS)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat -Wformat-security -D_FORTIFY_SOURCE=2 -fstack-protector")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fPIE -fPIC -z noexecstack -z relro -z now")
    if(${CMAKE_C_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-protector-strong")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-strong")
        endif()
    endif()
endif()

if(${CMAKE_C_COMPILER_ID} STREQUAL "Intel" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
	if (USE_CODECOV_FLAGS)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -prof-gen=srcpos -prof-src-root-cwd")
		set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -prof-gen=srcpos -prof-src-root-cwd")
	endif()
endif()

#TODO: add -Wextra to c/cxx flags

#common release/debug compilation settings
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_COMPILER_FLAGS} -Wall -Werror -D_GNU_SOURCE -fvisibility=internal")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${C_COMPILER_FLAGS} -O0 -g -DENABLE_DEBUG")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${C_COMPILER_FLAGS} -O3")
set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${C_COMPILER_FLAGS} -O2 -g")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_COMPILER_FLAGS} -Wall -Werror -D_GNU_SOURCE -fvisibility=internal")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_COMPILER_FLAGS} -O0 -g -DENABLE_DEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_COMPILER_FLAGS} -O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${CXX_COMPILER_FLAGS} -O2 -g")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if( ${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    if (${CMAKE_DISABLE_SYCL} EQUAL 0)
        set(CMAKE_CLANG_FLAGS "-fsycl")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lsycl")
    endif()
    set(CMAKE_CLANG_FLAGS "${CMAKE_CLANG_FLAGS} -Wno-pessimizing-move -Wno-dangling-field -Wno-return-type -Wno-defaulted-function-deleted -Wno-mismatched-tags -Wno-unused-private-field -Wno-unneeded-internal-declaration -Wno-delete-non-abstract-non-virtual-dtor -Wno-reorder -D__STRICT_ANSI__ ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CLANG_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CLANG_FLAGS} -std=c++11")
endif()

if(${CMAKE_C_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
        #c++17 introduces algined new operator, use it
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -faligned-new")
    endif()
endif()

if(WITH_ASAN AND ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message(STATUS "Compiling with address sanitizer")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_C_FLAGS_DEBUG  "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lasan")
endif()

set(CCL_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/src)

enable_testing()

set(EXAMPLES_INC_DIRS ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/examples/include)

#include other CMakeLists
add_subdirectory(src)
add_subdirectory(examples/cpu)

set (CCL_ENABLCE_SYCL_CHECK_CONTRACT "#if !defined(__clang__) || __clang_major__ < 9 || !defined(CL_SYCL_LANGUAGE_VERSION)\n#error This version of CCL configured only for DPC++ compiler\n#endif")
if(${CMAKE_DISABLE_SYCL} EQUAL 0 AND ${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    option (CCL_ENABLE_SYCL "Enable CCL SYCL runtime" ON)
    message(STATUS "Enable CCL SYCL runtime")
    add_subdirectory(examples/sycl)
endif()

add_subdirectory(examples/common)
add_subdirectory(tests/functional)

#generate & install vars.sh
configure_file(cmake/vars.sh.in ${CMAKE_CURRENT_BINARY_DIR}/vars.sh @ONLY)
configure_file(cmake/setvars.sh.in ${CMAKE_CURRENT_BINARY_DIR}/setvars.sh @ONLY)
configure_file(cmake/ccl ${CMAKE_CURRENT_BINARY_DIR}/ccl @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/vars.sh DESTINATION ${CCL_INSTALL_ENV})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/setvars.sh DESTINATION ${CCL_INSTALL_ENV})
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/ccl DESTINATION ${CCL_INSTALL_MODULE})
install(PROGRAMS ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${CCL_INSTALL_LICENSE})
install(PROGRAMS ${PROJECT_SOURCE_DIR}/third-party-programs.txt DESTINATION ${CCL_INSTALL_LICENSE}
)
set(CCL_MAJOR_VERSION     "0")
set(CCL_MINOR_VERSION     "4")
set(CCL_UPDATE_VERSION    "0")
set(CCL_PRODUCT_STATUS    "beta")
string(TIMESTAMP CCL_PRODUCT_BUILD_DATE "%Y-%m-%dT %H:%M:%SZ")
get_vcs_properties("git")
set(CCL_PRODUCT_FULL "${CCL_PRODUCT_STATUS}-${CCL_MAJOR_VERSION}.${CCL_MINOR_VERSION}.${CCL_UPDATE_VERSION} ${CCL_PRODUCT_BUILD_DATE} ${VCS_INFO}")
configure_file(${PROJECT_SOURCE_DIR}/include/ccl_config.h.in "${CMAKE_CURRENT_BINARY_DIR}/include/ccl_config.h")
file(COPY "${CMAKE_CURRENT_BINARY_DIR}/include/ccl_config.h" DESTINATION ${PROJECT_SOURCE_DIR}/include/)

