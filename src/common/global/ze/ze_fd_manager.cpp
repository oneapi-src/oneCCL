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
#include "common/global/global.hpp"
#include "common/global/ze/ze_fd_manager.hpp"
#include "common/log/log.hpp"
#include "common/utils/exchange_utils.hpp"
#include "common/utils/utils.hpp"
#include "common/utils/yield.hpp"

#include <dirent.h>
#ifdef CCL_ENABLE_DRM
#include "i915_drm.h"
#endif // CCL_ENABLE_DRM

// pidfd system calls
#ifndef __NR_pidfd_open
#define __NR_pidfd_open 434
#endif // __NR_pidfd_open
#ifndef __NR_pidfd_getfd
#define __NR_pidfd_getfd 438
#endif // __NR_pidfd_getfd

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/un.h>

namespace ccl {
namespace ze {

fd_manager::fd_manager() {
    device_fds = init_device_fds();
    exchange_device_fds();
    LOG_DEBUG("init completed");
}

fd_manager::~fd_manager() {
    all_socks.clear();
    all_pids.clear();
    for (auto fd : device_fds) {
        close(fd);
    }
    device_fds.clear();
}

bool fd_manager::is_pidfd_supported() {
    int pid = getpid();
    char filename[] = "/tmp/oneccl_pidfd_check_getXXXXXXXXXX";
    std::vector<int> fds;
    bool result = true;

    auto check_fd = [&](int fd) {
        if (fd == ccl::utils::invalid_fd) {
            result = false;
        }
        fds.push_back(fd);
    };

    int file_fd = mkstemp(filename);
    check_fd(file_fd);

    int pidfd = syscall(__NR_pidfd_open, pid, 0);
    check_fd(pidfd);

    int dupfd = syscall(__NR_pidfd_getfd, pidfd, file_fd, 0);
    check_fd(dupfd);

    for (auto &fd : fds) {
        close(fd);
    }
    unlink(filename);
    return result;
}

void fd_manager::barrier(void *mem) {
    static int call_count = 1;

    int local_count = ccl::global_data::get().get_local_proc_count();
    std::atomic<int> *barrier_counter = static_cast<std::atomic<int> *>(mem);
    CCL_THROW_IF_NOT(barrier_counter == mem,
                     "barrier_counter: ",
                     barrier_counter,
                     " and mem:",
                     mem,
                     " must be the same");

    ++(*barrier_counter);
    LOG_DEBUG("barrier_counter: ", *barrier_counter);

    while ((*barrier_counter) < (call_count * local_count)) {
        ccl_yield(ccl::global_data::env().yield_type);
    }
    call_count++;
}

std::string fd_manager::get_shm_filename() {
    std::string filename = "/dev/shm/ccl-shm-file";
    uid_t uid = getuid();
    std::stringstream ss;
    ss << filename << "-" << std::to_string(uid);
    return ss.str();
}

void *fd_manager::create_shared_memory() {
    int local_count = ccl::global_data::get().get_local_proc_count();
    auto length = size_per_proc * local_count + counter_offset;
    int prot = PROT_READ | PROT_WRITE;
    int flags = MAP_SHARED;

    auto shm_filename = get_shm_filename();
    int fd = open(shm_filename.c_str(), O_CREAT | O_RDWR, 0666);
    CCL_THROW_IF_NOT(fd > 0, "open failed: fd: ", fd, ", errno: ", strerror(errno));
    int ret = ftruncate(fd, length);
    CCL_THROW_IF_NOT(ret != ccl::utils::invalid_err_code,
                     "ioctl failed: ret: ",
                     ret,
                     ", errno: ",
                     strerror(errno));

    void *mem = mmap(nullptr, length, prot, flags, fd, 0);
    CCL_THROW_IF_NOT(mem != MAP_FAILED, "mmap failed: mem: ", mem, ", errno: ", strerror(errno));

    LOG_DEBUG("shm_filename: ", shm_filename, ", mem: ", mem, ", fd: ", fd);
    barrier(mem);

    close(fd);
    unlink(shm_filename.c_str());
    return mem;
}

std::vector<int> fd_manager::get_device_fds() {
    return device_fds;
}

std::vector<int> fd_manager::init_device_fds() {
    const char *device_dir = "/dev/dri/by-path/";
    const char *suffix = "-render";
    char device_name[NAME_MAX];
    struct dirent *ent = nullptr;
    std::vector<int> fds;

    DIR *dir = opendir(device_dir);
    CCL_THROW_IF_NOT(dir, "opendir failed: could not open device directory");

    LOG_DEBUG("search for all devices in the device directory");
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.' || strstr(ent->d_name, suffix) == nullptr) {
            continue;
        }

        memset(device_name, 0, sizeof(device_name));
        int ret = snprintf(device_name, NAME_MAX, "%s%s", device_dir, ent->d_name);
        CCL_THROW_IF_NOT(ret > 0 || ret <= NAME_MAX, "could not create device name");

        int fd = open(device_name, O_RDWR);
        CCL_THROW_IF_NOT(fd > 0, "open failed: fd: ", fd, ", errno: ", strerror(errno));
        fds.push_back(fd);
        CCL_THROW_IF_NOT(
            fds.back() != ccl::utils::invalid_fd, "unexpected device fd: ", fds.back());
        LOG_DEBUG("device_name: ", device_name, " device_fd: ", fds.back());
    }
    CCL_THROW_IF_NOT(!fds.empty(), "fds is empty");
    LOG_DEBUG("completed, fds size: ", fds.size());
    return fds;
}

int fd_manager::pidfd_open(const int pid) {
    int fd = syscall(__NR_pidfd_open, pid, 0);
    CCL_THROW_IF_NOT(fd != ccl::utils::invalid_pid,
                     "pidfd_open failed: fd: ",
                     fd,
                     ", pid: ",
                     pid,
                     ", errno: ",
                     strerror(errno));
    LOG_DEBUG("pidfd_open: pid: ", pid, ", fd: ", fd);
    return fd;
}

int fd_manager::fd_to_mem_handle(int dev_fd, int fd) {
#ifdef CCL_ENABLE_DRM
    struct drm_prime_handle req = { 0, 0, 0 };
    req.fd = fd;

    int ret = ioctl(dev_fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &req);
    CCL_THROW_IF_NOT(ret != ccl::utils::invalid_err_code,
                     "ioctl failed: ret: ",
                     ret,
                     ", errno: ",
                     strerror(errno),
                     ", dev_fd: ",
                     dev_fd,
                     ", fd: ",
                     fd);
    LOG_DEBUG("dev_fd: ", dev_fd, ", req.fd: ", req.fd, ", handle: ", req.handle);
    return req.handle;
#else // CCL_ENABLE_DRM
    return ccl::utils::invalid_mem_handle;
#endif // CCL_ENABLE_DRM
}

int fd_manager::convert_fd_drmfd(int convert_from_fd, int handle) {
#ifdef CCL_ENABLE_DRM
    struct drm_prime_handle req = { 0, 0, 0 };
    req.flags = DRM_CLOEXEC | DRM_RDWR;
    req.handle = handle;

    int ret = ioctl(convert_from_fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &req);
    CCL_THROW_IF_NOT(ret != ccl::utils::invalid_err_code,
                     "ioctl failed: ret: ",
                     ret,
                     ", errno: ",
                     strerror(errno),
                     ", dev_fd: ",
                     convert_from_fd,
                     ", handle: ",
                     handle);
    LOG_DEBUG("drm: dev_fd: ", convert_from_fd, ", req.handle: ", handle, ", fd: ", req.fd);
    return req.fd;
#else // CCL_ENABLE_DRM
    return ccl::utils::invalid_fd;
#endif // CCL_ENABLE_DRM
}

int fd_manager::convert_fd_pidfd(int convert_from_fd, int handle) {
    int fd = syscall(__NR_pidfd_getfd, convert_from_fd, handle, 0);
    CCL_THROW_IF_NOT(fd != ccl::utils::invalid_fd,
                     "pidfd_getfd failed: "
                     "convert_from_fd: ",
                     convert_from_fd,
                     ", fd: ",
                     fd,
                     ", handle: ",
                     handle,
                     ", errno: ",
                     strerror(errno));
    LOG_DEBUG(
        "pidfd_getfd: convert_from_fd: ", convert_from_fd, ", handle: ", handle, ", fd: ", fd);
    return fd;
}

int fd_manager::mem_handle_to_fd(int convert_from_fd, int handle) {
    int fd = ccl::utils::invalid_fd;
    if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::drmfd) {
        fd = convert_fd_drmfd(convert_from_fd, handle);
    }
    else if (ccl::global_data::env().ze_ipc_exchange == ccl::ze::ipc_exchange_mode::pidfd) {
        fd = convert_fd_pidfd(convert_from_fd, handle);
    }
    else {
        CCL_THROW("unexpected ipc_exchange_mode");
    }
    return fd;
}

std::vector<int> fd_manager::setup_device_fds(int local_count, int proc_idx) {
    std::vector<int> fds;
    if (proc_idx == 0) {
        fds = device_fds;
        // send the fds to all other local processes
        for (int p_idx = 1; p_idx < local_count; p_idx++) {
            for (auto &fd : fds) {
                ccl::utils::sendmsg_call(all_socks[p_idx], fd, nullptr, 0, proc_idx);
            }
        }
    }
    else {
        // receive the fds from local process 0
        for (auto fd : device_fds) {
            close(fd);
        }
        fds.resize(device_fds.size());
        for (auto &fd : fds) {
            ccl::utils::recvmsg_call(all_socks[0], &fd, nullptr, 0, proc_idx);
        }
    }
    return fds;
}

void fd_manager::exchange_device_fds() {
    int sock_err;
    std::string sock_name;
    struct sockaddr_un sockaddr;
    memset(&sockaddr, 0, sizeof(sockaddr));
    unsigned int sockaddr_len = sizeof(sockaddr);
    int enable = 1;

    int local_count = ccl::global_data::get().get_local_proc_count();
    int local_idx = ccl::global_data::get().get_local_proc_idx();

    auto length = size_per_proc * local_count + counter_offset;

    all_pids.resize(local_count, ccl::utils::invalid_pid);
    all_socks.resize(local_count, ccl::utils::invalid_fd);

    pid_t pid = getpid();

    // send own pid to all processes via shm
    void *mem = create_shared_memory();
    void *shmem = (char *)mem + counter_offset;

    ((pid_t *)shmem)[local_idx] = pid;

    barrier(mem);

    for (int i = 0; i < local_count; i++) {
        all_pids[i] = ((pid_t *)shmem)[i];
    }
    CCL_THROW_IF_NOT(!all_pids.empty(), "all_pids shouldn't be empty");
    LOG_DEBUG("pid exchange is done: ", all_pids.size());

    // create a named socket between local_idx
    // 0 and all other local processes
    if (local_idx == 0) {
        barrier(mem);
        for (int i = 1; i < local_count; ++i) {
            std::string remote_sock_name;
            struct sockaddr_un remote_sockaddr;

            remote_sock_name = "/tmp/ccl-ipc-fd-sock-" + std::to_string(all_pids[i]) + ":" +
                               std::to_string(i) + "-" + std::to_string(local_idx);
            sock_name = "/tmp/ccl-ipc-fd-sock-" + std::to_string(pid) + ":" +
                        std::to_string(local_idx) + "-" + std::to_string(i);

            // create a socket for local proc j
            all_socks[i] = socket(AF_UNIX, SOCK_STREAM, 0);
            CCL_THROW_IF_NOT(all_socks[i] != ccl::utils::invalid_fd,
                             "socket failed: sock_err: ",
                             all_socks[i],
                             ", errno: ",
                             strerror(errno));

            setsockopt(all_socks[i], SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
            sockaddr.sun_family = AF_UNIX;
            strcpy(sockaddr.sun_path, sock_name.c_str());

            sock_err = bind(all_socks[i], (struct sockaddr *)&sockaddr, sockaddr_len);
            CCL_THROW_IF_NOT(sock_err != ccl::utils::invalid_err_code,
                             "bind failed: sock_err: ",
                             sock_err,
                             ", errno: ",
                             strerror(errno));

            // connect to remote socket for local proc j
            remote_sockaddr.sun_family = AF_UNIX;
            strcpy(remote_sockaddr.sun_path, remote_sock_name.c_str());

            sock_err = connect(all_socks[i], (struct sockaddr *)&remote_sockaddr, sockaddr_len);
            if (sock_err < 0) {
                if (errno == ECONNREFUSED || errno == ENOENT) {
                    return;
                }
                CCL_THROW("connect failed: error: ",
                          sock_err,
                          ", errno: ",
                          strerror(errno),
                          ", sock_name: ",
                          sock_name);
            }
        }
    }
    else {
        int sock;
        // create the local socket name
        sock_name = "/tmp/ccl-ipc-fd-sock-" + std::to_string(pid) + ":" +
                    std::to_string(local_idx) + "-" + std::to_string(0);
        // create a socket for local proc i
        sock = socket(AF_UNIX, SOCK_STREAM, 0);
        CCL_THROW_IF_NOT(sock != ccl::utils::invalid_fd,
                         "socket failed: sock: ",
                         sock,
                         ", errno: ",
                         strerror(errno));

        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
        sockaddr.sun_family = AF_UNIX;
        strcpy(sockaddr.sun_path, sock_name.c_str());

        sock_err = bind(sock, (struct sockaddr *)&sockaddr, sockaddr_len);
        CCL_THROW_IF_NOT(sock_err != ccl::utils::invalid_err_code,
                         "bind failed: sock_err: ",
                         sock_err,
                         ", errno: ",
                         strerror(errno));

        // listen to the socket to accept a connection to the other process
        sock_err = listen(sock, local_count);
        CCL_THROW_IF_NOT(sock_err != ccl::utils::invalid_err_code,
                         "listen failed: sock_err: ",
                         sock_err,
                         ", errno: ",
                         strerror(errno));

        // notify the other process that the socket is created and being listened to
        barrier(mem);

        all_socks[0] = accept(sock, (struct sockaddr *)&sockaddr, &sockaddr_len);
        CCL_THROW_IF_NOT(all_socks[0] != ccl::utils::invalid_err_code,
                         "accept failed: sock: ",
                         all_socks[0],
                         ", errno: ",
                         strerror(errno));

        setsockopt(all_socks[0], SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
        close(sock);
    }

    LOG_DEBUG("connection is set up");
    device_fds = setup_device_fds(local_count, local_idx);

    // close sockets
    if (local_idx == 0) {
        close_sockets(local_count, local_idx);
        barrier(mem);
    }
    else {
        barrier(mem);
        close_sockets(local_count, local_idx);
    }

    int ret = munmap(mem, length);
    CCL_THROW_IF_NOT(ret == 0, "munmap failed: ret: ", ret, ", errno: ", strerror(errno));
}

void fd_manager::close_sockets(int local_count, int proc_idx) {
    int sock_err;
    std::string sock_name;
    for (int i = 0; i < local_count; ++i) {
        if (all_socks[i] != ccl::utils::invalid_fd) {
            sock_err = close(all_socks[i]);
            CCL_THROW_IF_NOT(sock_err != ccl::utils::invalid_err_code,
                             "close failed: ret",
                             sock_err,
                             ", errno: ",
                             strerror(errno));
        }

        if (all_pids[proc_idx] != ccl::utils::invalid_pid && proc_idx != i) {
            sock_name = "/tmp/ccl-ipc-fd-sock-" + std::to_string(all_pids[proc_idx]) + ":" +
                        std::to_string(proc_idx) + "-" + std::to_string(i);
            sock_err = unlink(sock_name.c_str());
        }
    }
}

} // namespace ze
} // namespace ccl
