#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <time.h>
#include <unistd.h>

namespace {

using ClockGettimeFunction = int (*)(clockid_t, struct timespec*);

struct CallSite {
    std::atomic<std::uintptr_t> address{0};
    std::atomic<std::uint32_t> delayedCalls{0};
    std::atomic<std::uint8_t> isCarboniteCalibration{0};
};

constexpr std::size_t kCallSiteCount = 256;
constexpr std::uint32_t kMaximumDelayedCalls = 512;
constexpr std::uint64_t kMinimumCallNanoseconds = 2'000;
std::array<CallSite, kCallSiteCount> gCallSites{};
std::atomic<ClockGettimeFunction> gRealClockGettime{nullptr};
std::atomic<bool> gReportedDelay{false};

ClockGettimeFunction resolveClockGettime() {
    auto function = gRealClockGettime.load(std::memory_order_acquire);
    if (function != nullptr) return function;
    function = reinterpret_cast<ClockGettimeFunction>(
        dlsym(RTLD_NEXT, "clock_gettime")
    );
    if (function != nullptr) {
        gRealClockGettime.store(function, std::memory_order_release);
    }
    return function;
}

bool pathUsesCarboniteTscCalibration(const char* path) {
    if (path == nullptr) return false;
    return std::strstr(path, "libomni.anim.behavior.core.plugin.so") != nullptr
        || std::strstr(path, "libcarb.tasking.plugin.so") != nullptr
        || std::strstr(path, "libcarb.profiler-cpu.plugin.so") != nullptr;
}

CallSite* callSiteFor(std::uintptr_t address) {
    std::size_t slot = static_cast<std::size_t>(
        (address >> 4U) ^ (address >> 13U)
    ) % kCallSiteCount;
    for (std::size_t probe = 0; probe < kCallSiteCount; ++probe) {
        auto& candidate = gCallSites[(slot + probe) % kCallSiteCount];
        std::uintptr_t expected = 0;
        if (candidate.address.compare_exchange_strong(
                expected,
                address,
                std::memory_order_acq_rel,
                std::memory_order_acquire
            )
            || expected == address) {
            return &candidate;
        }
    }
    return nullptr;
}

bool shouldDelayCall(void* returnAddress) {
    const auto address = reinterpret_cast<std::uintptr_t>(returnAddress);
    if (address == 0) return false;
    auto* callSite = callSiteFor(address);
    if (callSite == nullptr) return false;

    auto decision = callSite->isCarboniteCalibration.load(std::memory_order_acquire);
    if (decision == 0) {
        Dl_info info{};
        const bool matches = dladdr(returnAddress, &info) != 0
            && pathUsesCarboniteTscCalibration(info.dli_fname);
        const std::uint8_t resolved = matches ? 2 : 1;
        std::uint8_t expected = 0;
        callSite->isCarboniteCalibration.compare_exchange_strong(
            expected,
            resolved,
            std::memory_order_acq_rel,
            std::memory_order_acquire
        );
        decision = callSite->isCarboniteCalibration.load(std::memory_order_acquire);
    }
    if (decision != 2) return false;

    const auto delayed = callSite->delayedCalls.fetch_add(
        1,
        std::memory_order_relaxed
    );
    return delayed < kMaximumDelayedCalls;
}

std::uint64_t timespecNanoseconds(const struct timespec& value) {
    return static_cast<std::uint64_t>(value.tv_sec) * UINT64_C(1'000'000'000)
        + static_cast<std::uint64_t>(value.tv_nsec);
}

void reportDelayOnce() {
    const char* trace = std::getenv("IMB_CLOCK_SHIM_TRACE");
    if (trace == nullptr || std::strcmp(trace, "0") == 0
        || gReportedDelay.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    constexpr char message[] =
        "imb-clock-shim: extending Carbonite TSC calibration samples for the Apple VM counter\n";
    const auto written = ::write(STDERR_FILENO, message, sizeof(message) - 1);
    (void)written;
}

} // namespace

extern "C" __attribute__((visibility("default"))) int clock_gettime(
    clockid_t clockID,
    struct timespec* result
) {
    const auto realClockGettime = resolveClockGettime();
    if (realClockGettime == nullptr) return -1;

    const int status = realClockGettime(clockID, result);
    if (status != 0 || clockID != CLOCK_MONOTONIC
        || !shouldDelayCall(__builtin_return_address(0))) {
        return status;
    }

    reportDelayOnce();
    struct timespec now{};
    const std::uint64_t start = timespecNanoseconds(*result);
    while (realClockGettime(CLOCK_MONOTONIC, &now) == 0
        && timespecNanoseconds(now) - start < kMinimumCallNanoseconds) {
    }
    return status;
}
