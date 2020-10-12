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

#define CCL_MAJOR_VERSION                   0
#define CCL_MINOR_VERSION                   7
#define CCL_UPDATE_VERSION                  0
#define CCL_PRODUCT_STATUS             "beta"
#define CCL_PRODUCT_BUILD_DATE         "2020-06-19T 21:29:18Z"
#define CCL_PRODUCT_FULL               "beta-0.7.0 2020-06-19T 21:29:18Z (mlp/a46ba7b)"


/* Configuration settings for multi GPU extension support*/
/* #undef MULTI_GPU_SUPPORT */

/* Auto-generated configuration settings for SYCL support */
/* #undef CCL_ENABLE_SYCL */

#ifdef CCL_ENABLE_SYCL

#endif

#define CCL_ENABLE_SYCL_V              
#define CCL_ENABLE_SYCL_TRUE                1
#define CCL_ENABLE_SYCL_FALSE               0
/* #undef CCL_GPU_DEVICES_AFFINITY_ENABLE */
/* #undef CCL_GPU_DEVICES_AFFINITY_MASK_SIZE */
