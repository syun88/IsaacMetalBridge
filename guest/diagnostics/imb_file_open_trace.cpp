#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

namespace {

bool shouldTrace(const char* path) {
    return path != nullptr
        && (std::strstr(path, "shadercache") != nullptr
            || std::strstr(path, ".desc") != nullptr
            || std::strstr(path, "8ffb883e060d5f4c492fcc4e2b66a396") != nullptr);
}

void traceResult(const char* operation, const char* path, long result, int savedErrno) {
    if (!shouldTrace(path)) {
        return;
    }
    char line[2048];
    const int length = std::snprintf(
        line,
        sizeof(line),
        "imb-file-open-trace: %s path='%s' result=%ld errno=%d (%s)\n",
        operation,
        path,
        result,
        savedErrno,
        std::strerror(savedErrno));
    if (length > 0) {
        const size_t bytes = static_cast<size_t>(length) < sizeof(line)
            ? static_cast<size_t>(length)
            : sizeof(line) - 1;
        const ssize_t written = ::write(STDERR_FILENO, line, bytes);
        (void)written;
    }
}

template <typename Function>
Function resolveNext(const char* name) {
    return reinterpret_cast<Function>(::dlsym(RTLD_NEXT, name));
}

} // namespace

extern "C" int open(const char* path, int flags, ...) {
    using Function = int (*)(const char*, int, ...);
    static const Function realOpen = resolveNext<Function>("open");
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list arguments;
        va_start(arguments, flags);
        mode = static_cast<mode_t>(va_arg(arguments, int));
        va_end(arguments);
    }
    errno = 0;
    const int result = (flags & O_CREAT) != 0
        ? realOpen(path, flags, mode)
        : realOpen(path, flags);
    const int savedErrno = errno;
    traceResult("open", path, result, savedErrno);
    errno = savedErrno;
    return result;
}

extern "C" int open64(const char* path, int flags, ...) {
    using Function = int (*)(const char*, int, ...);
    static const Function realOpen64 = resolveNext<Function>("open64");
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list arguments;
        va_start(arguments, flags);
        mode = static_cast<mode_t>(va_arg(arguments, int));
        va_end(arguments);
    }
    errno = 0;
    const int result = (flags & O_CREAT) != 0
        ? realOpen64(path, flags, mode)
        : realOpen64(path, flags);
    const int savedErrno = errno;
    traceResult("open64", path, result, savedErrno);
    errno = savedErrno;
    return result;
}

extern "C" int openat(int directory, const char* path, int flags, ...) {
    using Function = int (*)(int, const char*, int, ...);
    static const Function realOpenAt = resolveNext<Function>("openat");
    mode_t mode = 0;
    if ((flags & O_CREAT) != 0) {
        va_list arguments;
        va_start(arguments, flags);
        mode = static_cast<mode_t>(va_arg(arguments, int));
        va_end(arguments);
    }
    errno = 0;
    const int result = (flags & O_CREAT) != 0
        ? realOpenAt(directory, path, flags, mode)
        : realOpenAt(directory, path, flags);
    const int savedErrno = errno;
    traceResult("openat", path, result, savedErrno);
    errno = savedErrno;
    return result;
}

extern "C" FILE* fopen(const char* path, const char* mode) {
    using Function = FILE* (*)(const char*, const char*);
    static const Function realFopen = resolveNext<Function>("fopen");
    errno = 0;
    FILE* const result = realFopen(path, mode);
    const int savedErrno = errno;
    traceResult("fopen", path, reinterpret_cast<long>(result), savedErrno);
    errno = savedErrno;
    return result;
}

extern "C" FILE* fopen64(const char* path, const char* mode) {
    using Function = FILE* (*)(const char*, const char*);
    static const Function realFopen64 = resolveNext<Function>("fopen64");
    errno = 0;
    FILE* const result = realFopen64(path, mode);
    const int savedErrno = errno;
    traceResult("fopen64", path, reinterpret_cast<long>(result), savedErrno);
    errno = savedErrno;
    return result;
}
