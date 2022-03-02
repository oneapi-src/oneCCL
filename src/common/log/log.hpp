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
#pragma once

#include <assert.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>

#include "oneapi/ccl/exception.hpp"
#include "oneapi/ccl/types.hpp"
#include "common/utils/spinlock.hpp"
#include "common/utils/utils.hpp"

std::ostream& operator<<(std::ostream& os, const ccl::datatype& dt);

#define __FILENAME__ \
    ({ \
        const char* ptr = strrchr(__FILE__, '/'); \
        if (ptr) { \
            ++ptr; \
        } \
        else { \
            ptr = __FILE__; \
        } \
        ptr; \
    })

constexpr size_t LOGGER_BUFFER_SIZE = 256 * 1024;

constexpr const char* get_str_end(const char* str) {
    return *str ? get_str_end(str + 1) : str;
}

constexpr bool is_slash(const char* str) {
    return *str == '/' ? true : (*str ? is_slash(str + 1) : false);
}

constexpr const char* trim_slash(const char* str) {
    return *str == '/' ? (str + 1) : trim_slash(str - 1);
}

constexpr const char* basedir_static(const char* str) {
    return is_slash(str) ? trim_slash(get_str_end(str)) : str;
}

enum class ccl_log_level { error = 0, warn, info, debug, trace };

/**
 * Wrapper over streambuf class to provide presistent buffer
 */
class ccl_streambuf : public std::streambuf {
public:
    explicit ccl_streambuf(size_t s) : size(s), buffer(new char[size]) {
        reset();
    }

    ccl_streambuf(const ccl_streambuf& other) = delete;
    ccl_streambuf(ccl_streambuf&& other) = delete;

    ccl_streambuf& operator=(const ccl_streambuf& other) = delete;
    ccl_streambuf& operator=(ccl_streambuf&& other) = delete;

    friend std::ostream& operator<<(std::ostream& os, ccl_streambuf& buf);

private:
    size_t size;
    std::unique_ptr<char[]> buffer;

    void reset() {
        //reset pointer to start/cur/end positions in streambuf
        setp(buffer.get(), buffer.get() + size);
    }

    void set_eol() {
        auto pos = pptr();
        if (pos) {
            *pos = '\0';
        }
    }
};

/**
 * Logging interface
 * All methods support c++ stream format listed in <iomanip> or https://en.cppreference.com/w/cpp/io/manip
 *
 * Set witdth:
 * printf approach:
 * printf("Formatted digit: %10d, usual format: %d\n", 123, 234);
 * new approach:
 * log.info("Formatted digit:", std::setw(10), "123", ", usual format: ", 234);
 *
 * Floating point precision:
 * printf approach:
 * printf("Formatted digit: %.4f\n", 213,345355);
 * new approach:
 * To format output of floating point types one can use std::fixed with std::setprecision or std::scientific
 * log.info("Formatted digit: ", std::setprecision(4), 213,345355);
 *
 * To format base of digital values one can use std::dec, std::hex, std::oct as a parameter of logger interface
 * To set justification one can use std::left, std::right, std::internal as a parameter of logger interface
 */
class ccl_logger {
    using ccl_logger_lock_t = ccl_spinlock;

public:
    ccl_logger()
            : streambuf(LOGGER_BUFFER_SIZE),
              out_stream(&streambuf),
              initial_flags(out_stream.flags()) {}

    ccl_logger(const ccl_logger& other) = delete;
    ccl_logger(ccl_logger&& other) = delete;

    ccl_logger& operator=(const ccl_logger& other) = delete;
    ccl_logger& operator=(ccl_logger&& other) = delete;

    static void set_log_level(ccl_log_level lvl) {
        level = lvl;
    }

    static ccl_log_level get_log_level() noexcept {
        return level;
    }

    template <typename T, typename... Tpackage>
    void error(T&& first, Tpackage&&... others) {
        std::lock_guard<ccl_logger_lock_t> lock{ guard };

        write_stream_wrapper(
            out_stream, std::cerr, std::forward<T>(first), std::forward<Tpackage>(others)...);

        std::cerr << streambuf;
        std::flush(std::cerr);

        out_stream.flags(initial_flags);
    }

    template <typename T, typename... Tpackage>
    void warn(T&& first, Tpackage&&... others) {
        std::lock_guard<ccl_logger_lock_t> lock{ guard };

        write_stream_wrapper(
            out_stream, std::cout, std::forward<T>(first), std::forward<Tpackage>(others)...);
    }

    template <typename T, typename... Tpackage>
    void info(T&& first, Tpackage&&... others) {
        std::lock_guard<ccl_logger_lock_t> lock{ guard };

        write_stream_wrapper(
            out_stream, std::cout, std::forward<T>(first), std::forward<Tpackage>(others)...);
    }

    template <typename T, typename... Tpackage>
    void debug(T&& first, Tpackage&&... others) {
        std::lock_guard<ccl_logger_lock_t> lock{ guard };

        write_stream_wrapper(
            out_stream, std::cout, std::forward<T>(first), std::forward<Tpackage>(others)...);
    }

    template <typename T, typename... Tpackage>
    void trace(T&& first, Tpackage&&... others) {
        std::lock_guard<ccl_logger_lock_t> lock{ guard };

        write_stream_wrapper(
            out_stream, std::cout, std::forward<T>(first), std::forward<Tpackage>(others)...);
    }

    /**
     * General purpose method, used to fill @b stream with content
     * @tparam stream any type that supports << operator as a left hand operand
     * @tparam T any type that has << operator
     * @tparam Tpackage any package of types that support << operator
     */
    template <typename stream, typename T, typename... Tpackage>
    static void format(stream& ss, T&& first, Tpackage&&... others) {
        write_stream(ss, std::forward<T>(first), std::forward<Tpackage>(others)...);
    }

    static void write_backtrace(std::ostream& str);

    static std::map<ccl_log_level, std::string> level_names;

    static void set_abort_on_throw(int val) {
        abort_on_throw = val;
    }

    static bool is_abort_on_throw_enabled() {
        return abort_on_throw;
    }

private:
    static ccl_log_level level;
    static bool abort_on_throw;

    ccl_streambuf streambuf;
    std::ostream out_stream;
    std::ios::fmtflags initial_flags;

    ccl_logger_lock_t guard{};

    template <typename stream, typename T>
    static void write_stream(stream& ss, T&& tail) {
        ss << tail;
    }

    template <typename stream, typename T, typename... Tpackage>
    static void write_stream(stream& ss, T&& first, Tpackage&&... others) {
        ss << first;
        write_stream(ss, std::forward<Tpackage>(others)...);
    }

    /**
     * Internal wrapper over write_stream methods. Formats message header, writes arguments to stream and redirects
     * result to the passed output stream
     */
    template <typename stream, typename output, typename T, typename... Tpackage>
    void write_stream_wrapper(stream& ss, output& out, T&& first, Tpackage&&... others) {
        write_prefix(ss);
        write_stream(ss, std::forward<T>(first), std::forward<Tpackage>(others)...);
        out << streambuf << std::endl;

        ss.flags(initial_flags);
    }

    static void write_prefix(std::ostream& str);
};

extern ccl_logger logger;

#define LOG_ERROR(...) \
    { \
        if (logger.get_log_level() >= ccl_log_level::error) { \
            logger.error("|CCL_ERROR| ", \
                         basedir_static(__FILE__), \
                         ":", \
                         __LINE__, \
                         " ", \
                         __FUNCTION__, \
                         ": ", \
                         ##__VA_ARGS__); \
        } \
    }

#define LOG_WARN(...) \
    { \
        if (logger.get_log_level() >= ccl_log_level::warn) { \
            logger.warn("|CCL_WARN| ", ##__VA_ARGS__); \
        } \
    }

#define LOG_INFO(...) \
    { \
        if (logger.get_log_level() >= ccl_log_level::info) { \
            logger.info("|CCL_INFO| ", ##__VA_ARGS__); \
        } \
    }

#define LOG_DEBUG(...) \
    { \
        if (logger.get_log_level() >= ccl_log_level::debug) { \
            logger.debug("|CCL_DEBUG| ", \
                         basedir_static(__FILE__), \
                         ":", \
                         __LINE__, \
                         " ", \
                         __FUNCTION__, \
                         ": ", \
                         ##__VA_ARGS__); \
        } \
    }

#define LOG_TRACE(...) \
    { \
        if (logger.get_log_level() >= ccl_log_level::trace) { \
            logger.trace("|CCL_TRACE| ", \
                         basedir_static(__FILE__), \
                         ":", \
                         __LINE__, \
                         " ", \
                         __FUNCTION__, \
                         ": ", \
                         ##__VA_ARGS__); \
        } \
    }

/**
 * Macro to handle critical unrecoverable error. Can be used in destructors
 */
#define CCL_FATAL(...) \
    do { \
        LOG_ERROR(__VA_ARGS__) \
        std::terminate(); \
    } while (0)

/**
 * Helper macro to throw ccl::exception exception. Must never be used in destructors
 */
#define CCL_THROW(...) \
    do { \
        std::stringstream throw_msg_ss; \
        ccl_logger::format(throw_msg_ss, \
                           basedir_static(__FILE__), \
                           ":", \
                           __LINE__, \
                           " ", \
                           __FUNCTION__, \
                           ": EXCEPTION: ", \
                           ##__VA_ARGS__); \
        if (ccl_logger::is_abort_on_throw_enabled()) { \
            LOG_ERROR(throw_msg_ss.str()); \
            abort(); \
        } \
        else { \
            throw ccl::exception(throw_msg_ss.str()); \
        } \
    } while (0)

/**
 * Helper macro to throw ccl::exception exception. Must never be used in destructors
 */
#define CCL_THROW_WITH_ERROR(...) \
    do { \
        std::stringstream throw_msg_ss; \
        ccl_logger::format(throw_msg_ss, \
                           basedir_static(__FILE__), \
                           ":", \
                           __LINE__, \
                           " ", \
                           __FUNCTION__, \
                           ": EXCEPTION: ", \
                           ##__VA_ARGS__); \
        LOG_ERROR("Error - ", ##__VA_ARGS__); \
        if (ccl_logger::is_abort_on_throw_enabled()) { \
            abort(); \
        } \
        else { \
            throw ccl::exception(throw_msg_ss.str()); \
        } \
    } while (0)
/**
 * Helper macro to throw ccl::exception exception if provided condition is not true.
 * Must never be used in destructors
 */
#define CCL_THROW_IF_NOT(cond, ...) \
    do { \
        if (!(cond)) { \
            LOG_ERROR("condition ", #cond, " failed\n", ##__VA_ARGS__); \
            CCL_THROW(__VA_ARGS__); \
        } \
    } while (0)

#define CCL_UNUSED(expr) \
    do { \
        (void)sizeof(expr); \
    } while (0)

#ifdef ENABLE_DEBUG

/**
 * Raises failed assertion if provided condition is not true. Works in debug build only
 */
#define CCL_ASSERT(cond, ...) \
    do { \
        if (!(cond)) { \
            LOG_ERROR("ASSERT failed, cond:  ", #cond, " ", ##__VA_ARGS__); \
            assert(0); \
        } \
    } while (0)

#else

/**
 * Raises failed assertion if provided condition is not true. Works in debug build only
 */
#define CCL_ASSERT(cond, ...) CCL_UNUSED(cond)

#endif
