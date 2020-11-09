#pragma once

/* All symbols shall be internal unless marked as CCL_API */
#ifdef __linux__
#   if __GNUC__ >= 4
#       define CCL_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
#   else
#       define CCL_HELPER_DLL_EXPORT
#   endif
#else
#error "unexpected OS"
#endif

#define CCL_API CCL_HELPER_DLL_EXPORT

#define ONECCL_SPEC_VERSION "1.0"

#define CCL_MAJOR_VERSION                   2021
#define CCL_MINOR_VERSION                   1
#define CCL_UPDATE_VERSION                  0
#define CCL_PRODUCT_STATUS             "Gold"
#define CCL_PRODUCT_BUILD_DATE         "2020-11-09T 20:29:17Z"
#define CCL_PRODUCT_FULL               "Gold-2021.1.0 2020-11-09T 20:29:17Z (/)"

/* Auto-generated configuration settings for SYCL support */
/* #undef CCL_ENABLE_SYCL */

#ifdef CCL_ENABLE_SYCL

#endif

/* Auto-generated configuration settings for multi GPU support*/
/* #undef MULTI_GPU_SUPPORT */
