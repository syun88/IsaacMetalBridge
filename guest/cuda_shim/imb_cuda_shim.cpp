#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <pthread.h>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <utility>

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUdeviceptr = std::uint64_t;
using CUstream = void*;
using CUevent = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUlibrary = void*;
using CUkernel = void*;
using CUarray = void*;
using CUmipmappedArray = void*;
using CUexternalMemory = void*;
using CUexternalSemaphore = void*;
using CUtexObject = std::uint64_t;
using CUarray_format = int;
using CUmemGenericAllocationHandle = std::uint64_t;
using CUmemAllocationHandleType = int;
using CUmemAllocationGranularity_flags = int;

struct CUmemAllocationProp;
struct CUmemAccessDesc;

struct CUDA_ARRAY3D_DESCRIPTOR {
    std::size_t Width;
    std::size_t Height;
    std::size_t Depth;
    CUarray_format Format;
    unsigned int NumChannels;
    unsigned int Flags;
};

struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
    int type;
    union {
        int fd;
        struct {
            void* handle;
            const void* name;
        } win32;
        const void* nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};

struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};

struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC {
    unsigned long long offset;
    CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
    unsigned int numLevels;
    unsigned int reserved[16];
};

struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
    int type;
    union {
        int fd;
        struct {
            void* handle;
            const void* name;
        } win32;
        const void* nvSciSyncObject;
    } handle;
    unsigned int flags;
    unsigned int reserved[16];
};

struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
    union {
        struct {
            unsigned long long value;
        } fence;
        struct {
            void* fence;
        } nvSciSync;
    } params;
    unsigned int flags;
    unsigned int reserved[16];
};

struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
    union {
        struct {
            unsigned long long value;
        } fence;
        struct {
            void* fence;
        } nvSciSync;
    } params;
    unsigned int flags;
    unsigned int reserved[16];
};

constexpr CUresult CUDA_SUCCESS = 0;
constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
constexpr CUresult CUDA_ERROR_INVALID_DEVICE = 101;
constexpr CUresult CUDA_ERROR_INVALID_HANDLE = 400;
constexpr CUresult CUDA_ERROR_NOT_FOUND = 500;
constexpr CUresult CUDA_ERROR_NOT_READY = 600;
constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = 801;
constexpr int kDriverVersion = 12080;
constexpr std::size_t kFallbackTotalMemory = 16ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint8_t kDeviceUUID[16] = {
    'I', 'M', 'B', '-', 'A', 'P', 'P', 'L', 'E', '-', 'M', '4', 0, 0, 0, 1,
};
constexpr std::uint8_t kCudartExportTableUUID[16] = {
    0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
    0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9,
};
constexpr std::uint8_t kRuntimeCallbackHooksUUID[16] = {
    0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
    0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66,
};
constexpr std::uint8_t kToolsTlsUUID[16] = {
    0x42, 0xd8, 0x5a, 0x81, 0x23, 0xf6, 0xcb, 0x47,
    0x82, 0x98, 0xf6, 0xe7, 0x8a, 0x3a, 0xec, 0xdc,
};
constexpr std::uint8_t kContextLocalStorageUUID[16] = {
    0xc6, 0x93, 0x33, 0x6e, 0x11, 0x21, 0xdf, 0x11,
    0xa8, 0xc3, 0x68, 0xf3, 0x55, 0xd8, 0x95, 0x93,
};
constexpr std::uint8_t kContextChecksUUID[16] = {
    0x26, 0x3e, 0x88, 0x60, 0x7c, 0xd2, 0x61, 0x43,
    0x92, 0xf6, 0xbb, 0xd5, 0x00, 0x6d, 0xfa, 0x7e,
};
constexpr std::uint8_t kIntegrityCheckUUID[16] = {
    0xd4, 0x08, 0x20, 0x55, 0xbd, 0xe6, 0x70, 0x4b,
    0x8d, 0x34, 0xba, 0x12, 0x3c, 0x66, 0xe1, 0xf2,
};

std::uint8_t gPrimaryContextToken = 0;
std::uint8_t gMemoryPoolToken = 0;
std::uint32_t gRuntimeCallbackBuffer1[1024] = {};
std::uint32_t gRuntimeCallbackBuffer2[14] = {};
thread_local CUcontext gCurrentContext = nullptr;

std::size_t scaledSystemMemory(unsigned long value, unsigned int unit) {
    const auto scaled = static_cast<unsigned long long>(value) * static_cast<unsigned long long>(unit);
    return static_cast<std::size_t>(std::min<unsigned long long>(scaled, SIZE_MAX));
}

std::size_t systemTotalMemory() {
    struct sysinfo information {};
    if (::sysinfo(&information) == 0 && information.totalram > 0) {
        return scaledSystemMemory(information.totalram, information.mem_unit);
    }
    return kFallbackTotalMemory;
}

std::size_t systemFreeMemory() {
    struct sysinfo information {};
    if (::sysinfo(&information) == 0) {
        const auto free = scaledSystemMemory(information.freeram, information.mem_unit);
        const auto buffers = scaledSystemMemory(information.bufferram, information.mem_unit);
        return std::min(systemTotalMemory(), free + buffers);
    }
    return systemTotalMemory() * 3 / 4;
}

using ContextStorageDestructor = void (*)(CUcontext, void*, void*);
struct ContextStorageEntry {
    CUcontext context = nullptr;
    void* key = nullptr;
    void* value = nullptr;
    ContextStorageDestructor destructor = nullptr;
    bool used = false;
};
ContextStorageEntry gContextStorage[64] = {};

struct StreamState {
    unsigned int flags = 0;
    int priority = 0;
};

struct EventState {
    unsigned int flags = 0;
    bool recorded = false;
};

struct ModuleState {
    const void* image = nullptr;
};
struct FunctionState {
    ModuleState* module = nullptr;
};
struct LibraryState {
    ModuleState* module = nullptr;
};
struct KernelState {
    LibraryState* library = nullptr;
    FunctionState* function = nullptr;
    std::string name;
};
struct ExternalMemoryState;
struct MipmappedArrayState;
struct ArrayState {
    MipmappedArrayState* mipmap = nullptr;
    unsigned int level = 0;
};
struct ExternalMemoryState {
    int fd = -1;
    std::size_t size = 0;
    void* mapping = MAP_FAILED;
};
struct ExternalSemaphoreState {
    int fd = -1;
};
struct MipmappedArrayState {
    ExternalMemoryState* memory = nullptr;
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC descriptor{};
    std::vector<ArrayState*> levels;
};

struct GenericAllocationState {
    int fd = -1;
    std::size_t size = 0;
};

std::mutex gCudaStateMutex;
std::unordered_map<CUdeviceptr, std::size_t> gDeviceAllocations;
std::unordered_map<CUdeviceptr, std::size_t> gVirtualReservations;
std::unordered_map<CUmemGenericAllocationHandle, GenericAllocationState>
    gGenericAllocations;
std::unordered_map<void*, std::size_t> gHostAllocations;
std::unordered_set<StreamState*> gStreams;
std::unordered_set<EventState*> gEvents;
std::unordered_set<ModuleState*> gModules;
std::unordered_set<FunctionState*> gFunctions;
std::unordered_set<LibraryState*> gLibraries;
std::unordered_set<KernelState*> gKernels;
std::unordered_set<ArrayState*> gArrays;
std::unordered_set<ExternalMemoryState*> gExternalMemories;
std::unordered_set<ExternalSemaphoreState*> gExternalSemaphores;
std::unordered_set<MipmappedArrayState*> gMipmappedArrays;
std::unordered_set<CUdeviceptr> gExternalDevicePointers;
std::atomic<std::uint64_t> gNextTextureObject{1};
std::atomic<std::uint64_t> gNextGenericAllocationHandle{1};

bool resolveDeviceRange(CUdeviceptr pointer, std::size_t size, void** hostPointer) {
    if (hostPointer == nullptr) return false;
    for (const auto& [base, allocationSize] : gDeviceAllocations) {
        if (pointer >= base && pointer - base <= allocationSize
            && size <= allocationSize - static_cast<std::size_t>(pointer - base)) {
            *hostPointer = reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer));
            return true;
        }
    }
    for (const auto& [basePointer, allocationSize] : gHostAllocations) {
        const auto base = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(basePointer));
        if (pointer >= base && pointer - base <= allocationSize
            && size <= allocationSize - static_cast<std::size_t>(pointer - base)) {
            *hostPointer = reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer));
            return true;
        }
    }
    return false;
}

template <typename Value>
CUresult fillDeviceMemory(CUdeviceptr destination, Value value, std::size_t count) {
    if (count > SIZE_MAX / sizeof(Value)) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(gCudaStateMutex);
    void* host = nullptr;
    if (!resolveDeviceRange(destination, count * sizeof(Value), &host)) return CUDA_ERROR_INVALID_VALUE;
    std::fill_n(static_cast<Value*>(host), count, value);
    return CUDA_SUCCESS;
}

// libcudart resolves the complete Driver API table up front, then calls only a
// small subset during device discovery.  Give every unresolved symbol a stable
// trampoline so traces identify the function that was actually called.  This
// is diagnostic compatibility plumbing; real operations still get typed
// implementations below as soon as they are observed.
constexpr std::size_t kUnsupportedSlotCount = 1024;
constexpr std::size_t kUnsupportedNameSize = 96;
char gUnsupportedNames[kUnsupportedSlotCount][kUnsupportedNameSize] = {};

bool traceEnabled() {
    const char* value = std::getenv("IMB_CUDA_TRACE");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

void trace(const char* name) {
    if (traceEnabled()) std::fprintf(stderr, "imb-cuda-shim: %s\n", name);
}

CUresult loadModuleToken(CUmodule* module, const void* image) {
    if (module == nullptr || image == nullptr) return CUDA_ERROR_INVALID_VALUE;
    auto* state = new (std::nothrow) ModuleState{image};
    if (state == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    {
        std::lock_guard lock(gCudaStateMutex);
        gModules.insert(state);
    }
    *module = state;
    return CUDA_SUCCESS;
}

CUresult loadLibraryToken(CUlibrary* library, const void* image) {
    if (library == nullptr || image == nullptr) return CUDA_ERROR_INVALID_VALUE;

    CUmodule module = nullptr;
    const CUresult moduleResult = loadModuleToken(&module, image);
    if (moduleResult != CUDA_SUCCESS) return moduleResult;

    auto* state = new (std::nothrow) LibraryState{static_cast<ModuleState*>(module)};
    if (state == nullptr) {
        std::lock_guard lock(gCudaStateMutex);
        gModules.erase(static_cast<ModuleState*>(module));
        delete static_cast<ModuleState*>(module);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    {
        std::lock_guard lock(gCudaStateMutex);
        gLibraries.insert(state);
    }
    *library = state;
    return CUDA_SUCCESS;
}

bool validDevice(CUdevice device) {
    return device == 0;
}

template <std::size_t Slot>
CUresult unsupportedSlot(
    std::uintptr_t argument0,
    std::uintptr_t argument1,
    std::uintptr_t argument2,
    std::uintptr_t argument3,
    std::uintptr_t,
    std::uintptr_t,
    std::uintptr_t,
    std::uintptr_t
) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-cuda-shim: generic symbol=%s args=%#llx,%#llx,%#llx,%#llx\n",
            gUnsupportedNames[Slot],
            static_cast<unsigned long long>(argument0),
            static_cast<unsigned long long>(argument1),
            static_cast<unsigned long long>(argument2),
            static_cast<unsigned long long>(argument3)
        );
    }
    return CUDA_SUCCESS;
}

using UnsupportedFunction = CUresult (*)(
    std::uintptr_t,
    std::uintptr_t,
    std::uintptr_t,
    std::uintptr_t,
    std::uintptr_t,
    std::uintptr_t,
    std::uintptr_t,
    std::uintptr_t
);

template <std::size_t... Slots>
constexpr std::array<UnsupportedFunction, sizeof...(Slots)> makeUnsupportedFunctions(
    std::index_sequence<Slots...>
) {
    return {&unsupportedSlot<Slots>...};
}

constexpr auto gUnsupportedFunctions = makeUnsupportedFunctions(
    std::make_index_sequence<kUnsupportedSlotCount>{}
);

std::size_t unsupportedSlotFor(const char* symbol) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char* cursor = reinterpret_cast<const unsigned char*>(symbol); *cursor != 0; ++cursor) {
        hash ^= *cursor;
        hash *= 1099511628211ULL;
    }

    for (std::size_t probe = 0; probe < kUnsupportedSlotCount; ++probe) {
        const std::size_t slot = (static_cast<std::size_t>(hash) + probe) % kUnsupportedSlotCount;
        if (gUnsupportedNames[slot][0] == '\0') {
            std::snprintf(gUnsupportedNames[slot], kUnsupportedNameSize, "%s", symbol);
            return slot;
        }
        if (std::strcmp(gUnsupportedNames[slot], symbol) == 0) return slot;
    }

    return 0;
}

CUresult cudartGetModuleFromCubin(CUmodule* module, const void* fatbinWrapper) {
    trace("cudart export getModuleFromCubin");
    return loadModuleToken(module, fatbinWrapper);
}

CUresult cudartGetPrimaryContext(CUcontext* context, CUdevice device) {
    trace("cudart export getPrimaryContext");
    if (context == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    *context = &gPrimaryContextToken;
    return CUDA_SUCCESS;
}

CUresult cudartGetModuleFromCubinExt1(
    CUmodule* module,
    const void* fatbinWrapper,
    void* optionKeys,
    void* optionValues,
    unsigned int optionCount
) {
    trace("cudart export getModuleFromCubinExt1");
    if (optionKeys != nullptr || optionValues != nullptr || optionCount != 0) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return loadModuleToken(module, fatbinWrapper);
}

void cudartInterfaceNoop(std::uintptr_t) {
    trace("cudart export interfaceNoop");
}

CUresult cudartGetModuleFromCubinExt2(
    const void* fatbinHeader,
    CUmodule* module,
    void* optionKeys,
    void* optionValues,
    unsigned int optionCount
) {
    trace("cudart export getModuleFromCubinExt2");
    if (optionKeys != nullptr || optionValues != nullptr || optionCount != 0) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return loadModuleToken(module, fatbinHeader);
}

CUresult cudartLoadCompilers() {
    trace("cudart export loadCompilers");
    return CUDA_SUCCESS;
}

const void* const gCudartExportTable[13] = {
    reinterpret_cast<const void*>(sizeof(gCudartExportTable)),
    reinterpret_cast<const void*>(cudartGetModuleFromCubin),
    reinterpret_cast<const void*>(cudartGetPrimaryContext),
    nullptr,
    nullptr,
    nullptr,
    reinterpret_cast<const void*>(cudartGetModuleFromCubinExt1),
    reinterpret_cast<const void*>(cudartInterfaceNoop),
    reinterpret_cast<const void*>(cudartGetModuleFromCubinExt2),
    nullptr,
    nullptr,
    nullptr,
    reinterpret_cast<const void*>(cudartLoadCompilers),
};

void getRuntimeCallbackBuffer1(void** buffer, std::size_t* size) {
    trace("runtime callback export getBuffer1");
    if (buffer != nullptr) *buffer = gRuntimeCallbackBuffer1;
    if (size != nullptr) *size = std::size(gRuntimeCallbackBuffer1);
}

void getRuntimeCallbackBuffer2(void** buffer, std::size_t* size) {
    trace("runtime callback export getBuffer2");
    if (buffer != nullptr) *buffer = gRuntimeCallbackBuffer2;
    if (size != nullptr) *size = std::size(gRuntimeCallbackBuffer2);
}

const void* const gRuntimeCallbackHooksTable[7] = {
    reinterpret_cast<const void*>(sizeof(gRuntimeCallbackHooksTable)),
    nullptr,
    reinterpret_cast<const void*>(getRuntimeCallbackBuffer1),
    nullptr,
    nullptr,
    nullptr,
    reinterpret_cast<const void*>(getRuntimeCallbackBuffer2),
};

const void* const gToolsTlsTable[4] = {
    reinterpret_cast<const void*>(sizeof(gToolsTlsTable)),
    nullptr,
    nullptr,
    nullptr,
};

CUcontext resolveContext(CUcontext context) {
    if (context != nullptr) return context;
    if (gCurrentContext != nullptr) return gCurrentContext;
    return &gPrimaryContextToken;
}

CUresult contextStoragePut(
    CUcontext context,
    void* key,
    void* value,
    ContextStorageDestructor destructor
) {
    trace("context local storage put");
    context = resolveContext(context);
    for (auto& entry : gContextStorage) {
        if (entry.used && entry.context == context && entry.key == key) {
            entry.value = value;
            entry.destructor = destructor;
            return CUDA_SUCCESS;
        }
    }
    for (auto& entry : gContextStorage) {
        if (!entry.used) {
            entry = {context, key, value, destructor, true};
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult contextStorageDelete(CUcontext context, void* key) {
    trace("context local storage delete");
    context = resolveContext(context);
    for (auto& entry : gContextStorage) {
        if (entry.used && entry.context == context && entry.key == key) {
            entry = {};
            return CUDA_SUCCESS;
        }
    }
    return CUDA_SUCCESS;
}

CUresult contextStorageGet(void** value, CUcontext context, void* key) {
    trace("context local storage get");
    if (value == nullptr) return CUDA_ERROR_INVALID_VALUE;
    context = resolveContext(context);
    for (const auto& entry : gContextStorage) {
        if (entry.used && entry.context == context && entry.key == key) {
            *value = entry.value;
            return CUDA_SUCCESS;
        }
    }
    *value = nullptr;
    return CUDA_ERROR_INVALID_HANDLE;
}

const void* const gContextLocalStorageTable[4] = {
    reinterpret_cast<const void*>(contextStoragePut),
    reinterpret_cast<const void*>(contextStorageDelete),
    reinterpret_cast<const void*>(contextStorageGet),
    nullptr,
};

CUresult contextCheck(CUcontext, std::uint32_t* flags, const void**) {
    trace("context check");
    if (flags == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *flags = 0;
    return CUDA_SUCCESS;
}

std::uint32_t contextCheckFlags() {
    trace("context check flags");
    return 0;
}

const void* const gContextChecksTable[4] = {
    reinterpret_cast<const void*>(sizeof(gContextChecksTable)),
    nullptr,
    reinterpret_cast<const void*>(contextCheck),
    reinterpret_cast<const void*>(contextCheckFlags),
};

constexpr std::uint8_t kIntegrityMixingTable[256] = {
    0x29, 0x2e, 0x43, 0xc9, 0xa2, 0xd8, 0x7c, 0x01, 0x3d, 0x36, 0x54, 0xa1, 0xec, 0xf0, 0x06, 0x13,
    0x62, 0xa7, 0x05, 0xf3, 0xc0, 0xc7, 0x73, 0x8c, 0x98, 0x93, 0x2b, 0xd9, 0xbc, 0x4c, 0x82, 0xca,
    0x1e, 0x9b, 0x57, 0x3c, 0xfd, 0xd4, 0xe0, 0x16, 0x67, 0x42, 0x6f, 0x18, 0x8a, 0x17, 0xe5, 0x12,
    0xbe, 0x4e, 0xc4, 0xd6, 0xda, 0x9e, 0xde, 0x49, 0xa0, 0xfb, 0xf5, 0x8e, 0xbb, 0x2f, 0xee, 0x7a,
    0xa9, 0x68, 0x79, 0x91, 0x15, 0xb2, 0x07, 0x3f, 0x94, 0xc2, 0x10, 0x89, 0x0b, 0x22, 0x5f, 0x21,
    0x80, 0x7f, 0x5d, 0x9a, 0x5a, 0x90, 0x32, 0x27, 0x35, 0x3e, 0xcc, 0xe7, 0xbf, 0xf7, 0x97, 0x03,
    0xff, 0x19, 0x30, 0xb3, 0x48, 0xa5, 0xb5, 0xd1, 0xd7, 0x5e, 0x92, 0x2a, 0xac, 0x56, 0xaa, 0xc6,
    0x4f, 0xb8, 0x38, 0xd2, 0x96, 0xa4, 0x7d, 0xb6, 0x76, 0xfc, 0x6b, 0xe2, 0x9c, 0x74, 0x04, 0xf1,
    0x45, 0x9d, 0x70, 0x59, 0x64, 0x71, 0x87, 0x20, 0x86, 0x5b, 0xcf, 0x65, 0xe6, 0x2d, 0xa8, 0x02,
    0x1b, 0x60, 0x25, 0xad, 0xae, 0xb0, 0xb9, 0xf6, 0x1c, 0x46, 0x61, 0x69, 0x34, 0x40, 0x7e, 0x0f,
    0x55, 0x47, 0xa3, 0x23, 0xdd, 0x51, 0xaf, 0x3a, 0xc3, 0x5c, 0xf9, 0xce, 0xba, 0xc5, 0xea, 0x26,
    0x2c, 0x53, 0x0d, 0x6e, 0x85, 0x28, 0x84, 0x09, 0xd3, 0xdf, 0xcd, 0xf4, 0x41, 0x81, 0x4d, 0x52,
    0x6a, 0xdc, 0x37, 0xc8, 0x6c, 0xc1, 0xab, 0xfa, 0x24, 0xe1, 0x7b, 0x08, 0x0c, 0xbd, 0xb1, 0x4a,
    0x78, 0x88, 0x95, 0x8b, 0xe3, 0x63, 0xe8, 0x6d, 0xe9, 0xcb, 0xd5, 0xfe, 0x3b, 0x00, 0x1d, 0x39,
    0xf2, 0xef, 0xb7, 0x0e, 0x66, 0x58, 0xd0, 0xe4, 0xa6, 0x77, 0x72, 0xf8, 0xeb, 0x75, 0x4b, 0x0a,
    0x31, 0x44, 0x50, 0xb4, 0x8f, 0xed, 0x1f, 0x1a, 0xdb, 0x99, 0x8d, 0x33, 0x9f, 0x11, 0x83, 0x14,
};

void integritySinglePass(std::array<std::uint8_t, 66>& accumulator, std::uint8_t byte) {
    const std::uint8_t position = accumulator[0x40];
    accumulator[static_cast<std::size_t>(position) + 0x10] = byte;
    const std::uint8_t nextPosition = static_cast<std::uint8_t>((position + 1) & 0x0f);
    accumulator[static_cast<std::size_t>(position) + 0x20] = accumulator[position] ^ byte;
    const std::uint8_t mixed = kIntegrityMixingTable[byte ^ accumulator[0x41]];
    const std::uint8_t previous = accumulator[static_cast<std::size_t>(position) + 0x30];
    accumulator[static_cast<std::size_t>(position) + 0x30] = mixed ^ previous;
    accumulator[0x41] = mixed ^ previous;
    accumulator[0x40] = nextPosition;
    if (nextPosition != 0) return;

    std::uint8_t state = 0x29;
    std::uint8_t round = 0;
    while (true) {
        state ^= accumulator[0];
        accumulator[0] = state;
        for (std::size_t index = 1; index < 0x30; ++index) {
            state = accumulator[index] ^ kIntegrityMixingTable[state];
            accumulator[index] = state;
        }
        state = static_cast<std::uint8_t>(state + round);
        ++round;
        if (round == 0x12) break;
        state = kIntegrityMixingTable[state];
    }
}

void integrityHashBytes(
    std::array<std::uint8_t, 66>& accumulator,
    const void* bytes,
    std::size_t size,
    std::uint8_t xorMask
) {
    const auto* input = static_cast<const std::uint8_t*>(bytes);
    for (std::size_t index = 0; index < size; ++index) {
        integritySinglePass(accumulator, input[index] ^ xorMask);
    }
}

std::array<std::uint64_t, 2> integrityFinalize(std::array<std::uint8_t, 66>& accumulator) {
    const std::uint8_t padding = static_cast<std::uint8_t>(16 - accumulator[0x40]);
    for (std::uint8_t index = 0; index < padding; ++index) integritySinglePass(accumulator, padding);
    for (std::size_t index = 0x30; index < 0x40; ++index) integritySinglePass(accumulator, accumulator[index]);

    std::array<std::uint64_t, 2> result = {};
    std::memcpy(result.data(), accumulator.data(), 16);
    return result;
}

struct IntegrityPass3Input {
    std::uint32_t driverVersion;
    std::uint32_t requestedVersion;
    std::uint32_t process;
    std::uint32_t thread;
    const void* cudartTable;
    const void* integrityTable;
    const void* integrityFunction;
    std::uint64_t unixSeconds;
};

struct IntegrityDeviceInfo {
    std::uint8_t uuid[16];
    std::int32_t pciDomain;
    std::int32_t pciBus;
    std::int32_t pciDevice;
};

CUresult integrityCheck(std::uint32_t version, std::uint64_t unixSeconds, std::uint64_t* result);
extern const void* const gIntegrityCheckTable[3];

CUresult integrityCheck(std::uint32_t version, std::uint64_t unixSeconds, std::uint64_t* result) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-cuda-shim: integrity check version=%u unixSeconds=%llu\n",
            version,
            static_cast<unsigned long long>(unixSeconds)
        );
    }
    if (result == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (version % 10 == 0) {
        result[0] = 0x3341181c03cb675cULL;
        result[1] = 0x8ed383aa1f4cd1e8ULL;
    } else if (version % 10 == 1) {
        result[0] = 0x1841181c03cb675cULL;
        result[1] = 0x8ed383aa1f4cd1e8ULL;
    } else {
        constexpr std::uint8_t pass1[16] = {
            0x14, 0x6a, 0xdd, 0xae, 0x53, 0xa9, 0xa7, 0x52,
            0xaa, 0x08, 0x41, 0x36, 0x0b, 0xf5, 0x5a, 0x9f,
        };
        std::array<std::uint8_t, 66> accumulator = {};
        integrityHashBytes(accumulator, pass1, sizeof(pass1), 0x36);

        const IntegrityPass3Input processInfo = {
            static_cast<std::uint32_t>(kDriverVersion),
            version,
            static_cast<std::uint32_t>(::getpid()),
            static_cast<std::uint32_t>(::pthread_self()),
            gCudartExportTable,
            gIntegrityCheckTable,
            reinterpret_cast<const void*>(integrityCheck),
            unixSeconds,
        };
        integrityHashBytes(accumulator, &processInfo, sizeof(processInfo), 0);

        IntegrityDeviceInfo deviceInfo = {};
        std::memcpy(deviceInfo.uuid, kDeviceUUID, sizeof(kDeviceUUID));
        integrityHashBytes(accumulator, &deviceInfo, sizeof(deviceInfo), 0);

        const auto firstPass = integrityFinalize(accumulator);
        std::fill(accumulator.begin(), accumulator.begin() + 16, 0);
        std::fill(accumulator.begin() + 48, accumulator.end(), 0);
        integrityHashBytes(accumulator, pass1, sizeof(pass1), 0x5c);
        integrityHashBytes(accumulator, firstPass.data(), sizeof(firstPass), 0);
        const auto finalPass = integrityFinalize(accumulator);
        result[0] = finalPass[0];
        result[1] = finalPass[1];
    }
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-cuda-shim: integrity result=%016llx%016llx\n",
            static_cast<unsigned long long>(result[0]),
            static_cast<unsigned long long>(result[1])
        );
    }
    return CUDA_SUCCESS;
}

const void* const gIntegrityCheckTable[3] = {
    reinterpret_cast<const void*>(sizeof(gIntegrityCheckTable)),
    reinterpret_cast<const void*>(integrityCheck),
    nullptr,
};

}  // namespace

extern "C" {

#define IMB_CUDA_EXPORT __attribute__((visibility("default")))

IMB_CUDA_EXPORT CUresult cuInit(unsigned int flags) {
    trace("cuInit");
    return flags == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

IMB_CUDA_EXPORT CUresult cuDriverGetVersion(int* version) {
    trace("cuDriverGetVersion");
    if (version == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *version = kDriverVersion;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetCount(int* count) {
    trace("cuDeviceGetCount -> 1");
    if (count == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *count = 1;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGet(CUdevice* device, int ordinal) {
    trace("cuDeviceGet");
    if (device == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (ordinal != 0) return CUDA_ERROR_INVALID_DEVICE;
    *device = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetName(char* name, int length, CUdevice device) {
    trace("cuDeviceGetName");
    if (name == nullptr || length <= 0) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    std::snprintf(name, static_cast<std::size_t>(length), "%s", "IsaacMetalBridge CUDA-compat (Apple M4)");
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetUuid(void* uuid, CUdevice device) {
    trace("cuDeviceGetUuid");
    if (uuid == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    std::memcpy(uuid, kDeviceUUID, sizeof(kDeviceUUID));
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetUuid_v2(void* uuid, CUdevice device) {
    return cuDeviceGetUuid(uuid, device);
}

IMB_CUDA_EXPORT CUresult cuDeviceTotalMem(std::size_t* bytes, CUdevice device) {
    trace("cuDeviceTotalMem");
    if (bytes == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    *bytes = systemTotalMemory();
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceTotalMem_v2(std::size_t* bytes, CUdevice device) {
    return cuDeviceTotalMem(bytes, device);
}

IMB_CUDA_EXPORT CUresult cuDeviceGetAttribute(int* value, int attribute, CUdevice device) {
    if (traceEnabled()) std::fprintf(stderr, "imb-cuda-shim: cuDeviceGetAttribute attribute=%d\n", attribute);
    if (value == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    switch (attribute) {
    case 1: *value = 1024; break;       // max threads per block
    case 2: *value = 1024; break;       // max block dimension X
    case 3: *value = 1024; break;       // max block dimension Y
    case 4: *value = 64; break;         // max block dimension Z
    case 5: *value = 2147483647; break; // max grid dimension X
    case 6: *value = 65535; break;
    case 7: *value = 65535; break;
    case 8: *value = 49152; break;      // shared memory per block
    case 10: *value = 32; break;        // warp size
    case 13: *value = 1000000; break;   // clock rate, kHz
    case 16: *value = 10; break;        // multiprocessor count
    case 17: *value = 0; break;         // kernel timeout
    case 18: *value = 1; break;         // integrated
    case 20: *value = 0; break;         // default compute mode
    case 33: *value = 0; break;         // PCI bus
    case 34: *value = 0; break;         // PCI device
    case 36: *value = 1000000; break;   // memory clock
    case 37: *value = 128; break;       // memory bus width
    case 38: *value = 4 * 1024 * 1024; break;
    case 41: *value = 1; break;         // unified addressing
    case 50: *value = 0; break;         // PCI domain
    case 75: *value = 8; break;         // compute capability major
    case 76: *value = 9; break;         // compute capability minor
    case 83: *value = 1; break;         // managed memory
    case 86: *value = 1; break;         // host native atomics
    case 89: *value = 1; break;         // concurrent managed access
    default: *value = 1; break;
    }
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetPCIBusId(char* identifier, int length, CUdevice device) {
    trace("cuDeviceGetPCIBusId");
    if (identifier == nullptr || length <= 0) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    std::snprintf(identifier, static_cast<std::size_t>(length), "%s", "0000:00:00.0");
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetByPCIBusId(CUdevice* device, const char* identifier) {
    trace("cuDeviceGetByPCIBusId");
    if (device == nullptr || identifier == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *device = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetLuid(char* luid, unsigned int* nodeMask, CUdevice device) {
    trace("cuDeviceGetLuid");
    if (luid == nullptr || nodeMask == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    std::memset(luid, 0, 8);
    *nodeMask = 1;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetP2PAttribute(
    int* value,
    int,
    CUdevice sourceDevice,
    CUdevice destinationDevice
) {
    trace("cuDeviceGetP2PAttribute");
    if (value == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(sourceDevice) || !validDevice(destinationDevice)) return CUDA_ERROR_INVALID_DEVICE;
    *value = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetTexture1DLinearMaxWidth(
    std::size_t* maxWidth,
    int,
    unsigned int,
    CUdevice device
) {
    trace("cuDeviceGetTexture1DLinearMaxWidth");
    if (maxWidth == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    *maxWidth = 1ULL << 27;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetDefaultMemPool(void** pool, CUdevice device) {
    trace("cuDeviceGetDefaultMemPool");
    if (pool == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    *pool = &gMemoryPoolToken;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceGetMemPool(void** pool, CUdevice device) {
    trace("cuDeviceGetMemPool");
    return cuDeviceGetDefaultMemPool(pool, device);
}

IMB_CUDA_EXPORT CUresult cuDeviceSetMemPool(CUdevice device, void*) {
    trace("cuDeviceSetMemPool");
    return validDevice(device) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
}

IMB_CUDA_EXPORT CUresult cuFlushGPUDirectRDMAWrites(int, int) {
    trace("cuFlushGPUDirectRDMAWrites");
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDevicePrimaryCtxRetain(CUcontext* context, CUdevice device) {
    trace("cuDevicePrimaryCtxRetain");
    if (context == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    *context = &gPrimaryContextToken;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDevicePrimaryCtxRelease(CUdevice device) {
    trace("cuDevicePrimaryCtxRelease");
    return validDevice(device) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
}

IMB_CUDA_EXPORT CUresult cuDevicePrimaryCtxRelease_v2(CUdevice device) {
    return cuDevicePrimaryCtxRelease(device);
}

IMB_CUDA_EXPORT CUresult cuDevicePrimaryCtxReset(CUdevice device) {
    trace("cuDevicePrimaryCtxReset");
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    gCurrentContext = nullptr;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDevicePrimaryCtxReset_v2(CUdevice device) {
    return cuDevicePrimaryCtxReset(device);
}

IMB_CUDA_EXPORT CUresult cuDevicePrimaryCtxGetState(CUdevice device, unsigned int* flags, int* active) {
    trace("cuDevicePrimaryCtxGetState");
    if (flags == nullptr || active == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    *flags = 0;
    *active = gCurrentContext != nullptr ? 1 : 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDevicePrimaryCtxSetFlags(CUdevice device, unsigned int) {
    trace("cuDevicePrimaryCtxSetFlags");
    return validDevice(device) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_DEVICE;
}

IMB_CUDA_EXPORT CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice device, unsigned int flags) {
    return cuDevicePrimaryCtxSetFlags(device, flags);
}

IMB_CUDA_EXPORT CUresult cuCtxSetCurrent(CUcontext context) {
    trace("cuCtxSetCurrent");
    gCurrentContext = context;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxCreate(CUcontext* context, unsigned int, CUdevice device) {
    trace("cuCtxCreate");
    if (context == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device)) return CUDA_ERROR_INVALID_DEVICE;
    *context = &gPrimaryContextToken;
    gCurrentContext = *context;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxCreate_v2(CUcontext* context, unsigned int flags, CUdevice device) {
    return cuCtxCreate(context, flags, device);
}

IMB_CUDA_EXPORT CUresult cuCtxDetach(CUcontext context) {
    trace("cuCtxDetach");
    if (gCurrentContext == context) gCurrentContext = nullptr;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxGetCurrent(CUcontext* context) {
    trace("cuCtxGetCurrent");
    if (context == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *context = gCurrentContext;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxGetDevice(CUdevice* device) {
    trace("cuCtxGetDevice");
    if (device == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *device = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxGetApiVersion(CUcontext, unsigned int* version) {
    trace("cuCtxGetApiVersion");
    if (version == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *version = 3020;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxSynchronize() {
    trace("cuCtxSynchronize");
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxGetFlags(unsigned int* flags) {
    trace("cuCtxGetFlags");
    if (flags == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *flags = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxGetLimit(std::size_t* value, int) {
    trace("cuCtxGetLimit");
    if (value == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *value = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxSetLimit(int, std::size_t) {
    trace("cuCtxSetLimit");
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxGetCacheConfig(int* config) {
    trace("cuCtxGetCacheConfig");
    if (config == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *config = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxSetCacheConfig(int) {
    trace("cuCtxSetCacheConfig");
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxGetSharedMemConfig(int* config) {
    trace("cuCtxGetSharedMemConfig");
    if (config == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *config = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxSetSharedMemConfig(int) {
    trace("cuCtxSetSharedMemConfig");
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    trace("cuCtxGetStreamPriorityRange");
    if (leastPriority == nullptr || greatestPriority == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *leastPriority = 0;
    *greatestPriority = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxPushCurrent(CUcontext context) {
    trace("cuCtxPushCurrent");
    gCurrentContext = context;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxPushCurrent_v2(CUcontext context) {
    return cuCtxPushCurrent(context);
}

IMB_CUDA_EXPORT CUresult cuCtxPopCurrent(CUcontext* context) {
    trace("cuCtxPopCurrent");
    if (context == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *context = gCurrentContext;
    gCurrentContext = nullptr;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxPopCurrent_v2(CUcontext* context) {
    return cuCtxPopCurrent(context);
}

IMB_CUDA_EXPORT CUresult cuCtxResetPersistingL2Cache() {
    trace("cuCtxResetPersistingL2Cache");
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDeviceCanAccessPeer(int* canAccess, CUdevice device, CUdevice peerDevice) {
    trace("cuDeviceCanAccessPeer");
    if (canAccess == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (!validDevice(device) || !validDevice(peerDevice)) return CUDA_ERROR_INVALID_DEVICE;
    *canAccess = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemGetInfo(std::size_t* freeBytes, std::size_t* totalBytes) {
    trace("cuMemGetInfo");
    if (freeBytes == nullptr || totalBytes == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *totalBytes = systemTotalMemory();
    *freeBytes = systemFreeMemory();
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemGetInfo_v2(std::size_t* freeBytes, std::size_t* totalBytes) {
    return cuMemGetInfo(freeBytes, totalBytes);
}

IMB_CUDA_EXPORT CUresult cuCtxDestroy(CUcontext context) {
    trace("cuCtxDestroy");
    if (context == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (gCurrentContext == context) gCurrentContext = nullptr;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuCtxDestroy_v2(CUcontext context) {
    return cuCtxDestroy(context);
}

IMB_CUDA_EXPORT CUresult cuMemAlloc(CUdeviceptr* pointer, std::size_t size) {
    trace("cuMemAlloc");
    if (pointer == nullptr || size == 0) return CUDA_ERROR_INVALID_VALUE;
    void* allocation = std::malloc(size);
    if (allocation == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto devicePointer = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(allocation));
    {
        std::lock_guard lock(gCudaStateMutex);
        gDeviceAllocations[devicePointer] = size;
    }
    *pointer = devicePointer;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemAlloc_v2(CUdeviceptr* pointer, std::size_t size) {
    return cuMemAlloc(pointer, size);
}

IMB_CUDA_EXPORT CUresult cuMemFree(CUdeviceptr pointer) {
    trace("cuMemFree");
    if (pointer == 0) return CUDA_SUCCESS;
    bool external = false;
    {
        std::lock_guard lock(gCudaStateMutex);
        if (gDeviceAllocations.erase(pointer) == 0) return CUDA_ERROR_INVALID_VALUE;
        external = gExternalDevicePointers.erase(pointer) != 0;
    }
    if (!external) std::free(reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer)));
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemFree_v2(CUdeviceptr pointer) {
    return cuMemFree(pointer);
}

IMB_CUDA_EXPORT CUresult cuMemAddressReserve(
    CUdeviceptr* pointer,
    std::size_t size,
    std::size_t alignment,
    CUdeviceptr requestedAddress,
    unsigned long long flags
) {
    trace("cuMemAddressReserve");
    if (pointer == nullptr || size == 0 || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    constexpr std::size_t defaultAlignment = 64 * 1024;
    if (alignment == 0) alignment = defaultAlignment;
    if ((alignment & (alignment - 1)) != 0 || alignment < 4096) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    void* reservation = MAP_FAILED;
    if (requestedAddress != 0) {
#ifdef MAP_FIXED_NOREPLACE
        reservation = ::mmap(
            reinterpret_cast<void*>(static_cast<std::uintptr_t>(requestedAddress)),
            size,
            PROT_NONE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE,
            -1,
            0
        );
#endif
    } else {
        if (size > SIZE_MAX - alignment) return CUDA_ERROR_OUT_OF_MEMORY;
        void* raw = ::mmap(
            nullptr,
            size + alignment,
            PROT_NONE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0
        );
        if (raw != MAP_FAILED) {
            const auto rawAddress = reinterpret_cast<std::uintptr_t>(raw);
            const auto alignedAddress =
                (rawAddress + alignment - 1) & ~(static_cast<std::uintptr_t>(alignment) - 1);
            const std::size_t prefix = alignedAddress - rawAddress;
            const std::size_t suffix = alignment - prefix;
            if (prefix != 0) (void)::munmap(raw, prefix);
            if (suffix != 0) {
                (void)::munmap(
                    reinterpret_cast<void*>(alignedAddress + size),
                    suffix
                );
            }
            reservation = reinterpret_cast<void*>(alignedAddress);
        }
    }
    if (reservation == MAP_FAILED) return CUDA_ERROR_OUT_OF_MEMORY;

    const auto devicePointer =
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(reservation));
    {
        std::lock_guard lock(gCudaStateMutex);
        gVirtualReservations[devicePointer] = size;
    }
    *pointer = devicePointer;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemAddressFree(CUdeviceptr pointer, std::size_t size) {
    trace("cuMemAddressFree");
    if (pointer == 0 || size == 0) return CUDA_ERROR_INVALID_VALUE;
    {
        std::lock_guard lock(gCudaStateMutex);
        const auto reservation = gVirtualReservations.find(pointer);
        if (reservation == gVirtualReservations.end() || reservation->second != size) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        gVirtualReservations.erase(reservation);
        gDeviceAllocations.erase(pointer);
    }
    return ::munmap(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer)),
        size
    ) == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

IMB_CUDA_EXPORT CUresult cuMemCreate(
    CUmemGenericAllocationHandle* handle,
    std::size_t size,
    const CUmemAllocationProp* properties,
    unsigned long long flags
) {
    trace("cuMemCreate");
    if (handle == nullptr || properties == nullptr || size == 0 || flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const int fd = static_cast<int>(
        ::syscall(SYS_memfd_create, "imb-cuda-vmm", 0x0001U)
    );
    if (fd < 0) return CUDA_ERROR_OUT_OF_MEMORY;
    if (::ftruncate(fd, static_cast<off_t>(size)) != 0) {
        ::close(fd);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    const auto allocationHandle =
        gNextGenericAllocationHandle.fetch_add(1, std::memory_order_relaxed);
    try {
        std::lock_guard lock(gCudaStateMutex);
        gGenericAllocations.emplace(
            allocationHandle,
            GenericAllocationState{fd, size}
        );
    } catch (const std::bad_alloc&) {
        ::close(fd);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *handle = allocationHandle;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    trace("cuMemRelease");
    GenericAllocationState allocation{};
    {
        std::lock_guard lock(gCudaStateMutex);
        const auto entry = gGenericAllocations.find(handle);
        if (entry == gGenericAllocations.end()) return CUDA_ERROR_INVALID_HANDLE;
        allocation = entry->second;
        gGenericAllocations.erase(entry);
    }
    if (allocation.fd >= 0) ::close(allocation.fd);
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemMap(
    CUdeviceptr pointer,
    std::size_t size,
    std::size_t offset,
    CUmemGenericAllocationHandle handle,
    unsigned long long flags
) {
    trace("cuMemMap");
    if (pointer == 0 || size == 0 || flags != 0) return CUDA_ERROR_INVALID_VALUE;
    GenericAllocationState allocation{};
    {
        std::lock_guard lock(gCudaStateMutex);
        const auto entry = gGenericAllocations.find(handle);
        if (entry == gGenericAllocations.end()) return CUDA_ERROR_INVALID_HANDLE;
        allocation = entry->second;
        bool insideReservation = false;
        for (const auto& [base, reservationSize] : gVirtualReservations) {
            if (pointer >= base && pointer - base <= reservationSize
                && size <= reservationSize - (pointer - base)) {
                insideReservation = true;
                break;
            }
        }
        if (!insideReservation || offset > allocation.size
            || size > allocation.size - offset) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    void* mapped = ::mmap(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer)),
        size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_FIXED,
        allocation.fd,
        static_cast<off_t>(offset)
    );
    if (mapped == MAP_FAILED
        || reinterpret_cast<std::uintptr_t>(mapped) != static_cast<std::uintptr_t>(pointer)) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    {
        std::lock_guard lock(gCudaStateMutex);
        gDeviceAllocations[pointer] = size;
    }
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemUnmap(CUdeviceptr pointer, std::size_t size) {
    trace("cuMemUnmap");
    if (pointer == 0 || size == 0) return CUDA_ERROR_INVALID_VALUE;
    {
        std::lock_guard lock(gCudaStateMutex);
        const auto allocation = gDeviceAllocations.find(pointer);
        if (allocation == gDeviceAllocations.end() || allocation->second != size) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        gDeviceAllocations.erase(allocation);
    }
    void* reserved = ::mmap(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(pointer)),
        size,
        PROT_NONE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
        -1,
        0
    );
    return reserved == MAP_FAILED ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemSetAccess(
    CUdeviceptr pointer,
    std::size_t size,
    const CUmemAccessDesc* descriptors,
    std::size_t descriptorCount
) {
    trace("cuMemSetAccess");
    if (pointer == 0 || size == 0 || descriptors == nullptr || descriptorCount == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard lock(gCudaStateMutex);
    void* hostPointer = nullptr;
    return resolveDeviceRange(pointer, size, &hostPointer)
        ? CUDA_SUCCESS
        : CUDA_ERROR_INVALID_VALUE;
}

IMB_CUDA_EXPORT CUresult cuMemGetAllocationGranularity(
    std::size_t* granularity,
    const CUmemAllocationProp* properties,
    CUmemAllocationGranularity_flags option
) {
    trace("cuMemGetAllocationGranularity");
    if (granularity == nullptr || properties == nullptr || (option != 0 && option != 1)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *granularity = 64 * 1024;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemExportToShareableHandle(
    void* shareableHandle,
    CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType handleType,
    unsigned long long flags
) {
    trace("cuMemExportToShareableHandle");
    if (shareableHandle == nullptr || handleType != 1 || flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard lock(gCudaStateMutex);
    const auto entry = gGenericAllocations.find(handle);
    if (entry == gGenericAllocations.end()) return CUDA_ERROR_INVALID_HANDLE;
    const int exported = ::dup(entry->second.fd);
    if (exported < 0) return CUDA_ERROR_OUT_OF_MEMORY;
    *static_cast<int*>(shareableHandle) = exported;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemHostAlloc(void** pointer, std::size_t size, unsigned int) {
    trace("cuMemHostAlloc");
    if (pointer == nullptr || size == 0) return CUDA_ERROR_INVALID_VALUE;
    void* allocation = std::malloc(size);
    if (allocation == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    {
        std::lock_guard lock(gCudaStateMutex);
        gHostAllocations[allocation] = size;
    }
    *pointer = allocation;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemFreeHost(void* pointer) {
    trace("cuMemFreeHost");
    if (pointer == nullptr) return CUDA_SUCCESS;
    {
        std::lock_guard lock(gCudaStateMutex);
        if (gHostAllocations.erase(pointer) == 0) return CUDA_ERROR_INVALID_VALUE;
    }
    std::free(pointer);
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemHostGetDevicePointer(
    CUdeviceptr* devicePointer,
    void* hostPointer,
    unsigned int
) {
    trace("cuMemHostGetDevicePointer");
    if (devicePointer == nullptr || hostPointer == nullptr) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(gCudaStateMutex);
    if (!gHostAllocations.contains(hostPointer)) return CUDA_ERROR_INVALID_VALUE;
    *devicePointer = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(hostPointer));
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemHostGetDevicePointer_v2(
    CUdeviceptr* devicePointer,
    void* hostPointer,
    unsigned int flags
) {
    return cuMemHostGetDevicePointer(devicePointer, hostPointer, flags);
}

IMB_CUDA_EXPORT CUresult cuMemcpyHtoD(CUdeviceptr destination, const void* source, std::size_t size) {
    trace("cuMemcpyHtoD");
    if (source == nullptr && size != 0) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(gCudaStateMutex);
    void* destinationHost = nullptr;
    if (!resolveDeviceRange(destination, size, &destinationHost)) return CUDA_ERROR_INVALID_VALUE;
    std::memcpy(destinationHost, source, size);
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemcpyHtoD_v2(CUdeviceptr destination, const void* source, std::size_t size) {
    return cuMemcpyHtoD(destination, source, size);
}

IMB_CUDA_EXPORT CUresult cuMemcpyHtoDAsync_v2(
    CUdeviceptr destination,
    const void* source,
    std::size_t size,
    CUstream
) {
    return cuMemcpyHtoD(destination, source, size);
}

IMB_CUDA_EXPORT CUresult cuMemcpyDtoH(void* destination, CUdeviceptr source, std::size_t size) {
    trace("cuMemcpyDtoH");
    if (destination == nullptr && size != 0) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(gCudaStateMutex);
    void* sourceHost = nullptr;
    if (!resolveDeviceRange(source, size, &sourceHost)) return CUDA_ERROR_INVALID_VALUE;
    std::memcpy(destination, sourceHost, size);
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemcpyDtoH_v2(void* destination, CUdeviceptr source, std::size_t size) {
    return cuMemcpyDtoH(destination, source, size);
}

IMB_CUDA_EXPORT CUresult cuMemcpyDtoHAsync_v2(
    void* destination,
    CUdeviceptr source,
    std::size_t size,
    CUstream
) {
    return cuMemcpyDtoH(destination, source, size);
}

IMB_CUDA_EXPORT CUresult cuMemcpyDtoD(CUdeviceptr destination, CUdeviceptr source, std::size_t size) {
    trace("cuMemcpyDtoD");
    std::lock_guard lock(gCudaStateMutex);
    void* destinationHost = nullptr;
    void* sourceHost = nullptr;
    if (!resolveDeviceRange(destination, size, &destinationHost)
        || !resolveDeviceRange(source, size, &sourceHost)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::memmove(destinationHost, sourceHost, size);
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemcpyDtoD_v2(CUdeviceptr destination, CUdeviceptr source, std::size_t size) {
    return cuMemcpyDtoD(destination, source, size);
}

IMB_CUDA_EXPORT CUresult cuMemcpyDtoDAsync_v2(
    CUdeviceptr destination,
    CUdeviceptr source,
    std::size_t size,
    CUstream
) {
    return cuMemcpyDtoD(destination, source, size);
}

IMB_CUDA_EXPORT CUresult cuMemcpyPeerAsync(
    CUdeviceptr destination,
    CUcontext,
    CUdeviceptr source,
    CUcontext,
    std::size_t size,
    CUstream
) {
    return cuMemcpyDtoD(destination, source, size);
}

IMB_CUDA_EXPORT CUresult cuMemcpy3DAsync_v2(const void*, CUstream) {
    trace("cuMemcpy3DAsync_v2");
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMemsetD8_v2(CUdeviceptr destination, unsigned char value, std::size_t count) {
    trace("cuMemsetD8_v2");
    return fillDeviceMemory(destination, value, count);
}

IMB_CUDA_EXPORT CUresult cuMemsetD16_v2(CUdeviceptr destination, unsigned short value, std::size_t count) {
    trace("cuMemsetD16_v2");
    return fillDeviceMemory(destination, value, count);
}

IMB_CUDA_EXPORT CUresult cuMemsetD32_v2(CUdeviceptr destination, unsigned int value, std::size_t count) {
    trace("cuMemsetD32_v2");
    return fillDeviceMemory(destination, value, count);
}

IMB_CUDA_EXPORT CUresult cuMemsetD8Async(
    CUdeviceptr destination,
    unsigned char value,
    std::size_t count,
    CUstream
) {
    return cuMemsetD8_v2(destination, value, count);
}

IMB_CUDA_EXPORT CUresult cuMemsetD32Async(
    CUdeviceptr destination,
    unsigned int value,
    std::size_t count,
    CUstream
) {
    return cuMemsetD32_v2(destination, value, count);
}

IMB_CUDA_EXPORT CUresult cuStreamCreate(CUstream* stream, unsigned int flags) {
    trace("cuStreamCreate");
    if (stream == nullptr) return CUDA_ERROR_INVALID_VALUE;
    auto* state = new (std::nothrow) StreamState{flags, 0};
    if (state == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    {
        std::lock_guard lock(gCudaStateMutex);
        gStreams.insert(state);
    }
    *stream = state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuStreamCreateWithPriority(
    CUstream* stream,
    unsigned int flags,
    int priority
) {
    const CUresult result = cuStreamCreate(stream, flags);
    if (result == CUDA_SUCCESS) static_cast<StreamState*>(*stream)->priority = priority;
    return result;
}

IMB_CUDA_EXPORT CUresult cuStreamDestroy(CUstream stream) {
    trace("cuStreamDestroy");
    if (stream == nullptr) return CUDA_SUCCESS;
    auto* state = static_cast<StreamState*>(stream);
    {
        std::lock_guard lock(gCudaStateMutex);
        if (gStreams.erase(state) == 0) return CUDA_ERROR_INVALID_HANDLE;
    }
    delete state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuStreamDestroy_v2(CUstream stream) {
    return cuStreamDestroy(stream);
}

IMB_CUDA_EXPORT CUresult cuStreamQuery(CUstream stream) {
    trace("cuStreamQuery");
    if (stream == nullptr) return CUDA_SUCCESS;
    std::lock_guard lock(gCudaStateMutex);
    return gStreams.contains(static_cast<StreamState*>(stream)) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

IMB_CUDA_EXPORT CUresult cuStreamSynchronize(CUstream stream) {
    trace("cuStreamSynchronize");
    return cuStreamQuery(stream);
}

IMB_CUDA_EXPORT CUresult cuEventCreate(CUevent* event, unsigned int flags) {
    trace("cuEventCreate");
    if (event == nullptr) return CUDA_ERROR_INVALID_VALUE;
    auto* state = new (std::nothrow) EventState{flags, false};
    if (state == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    {
        std::lock_guard lock(gCudaStateMutex);
        gEvents.insert(state);
    }
    *event = state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuEventRecord(CUevent event, CUstream) {
    trace("cuEventRecord");
    std::lock_guard lock(gCudaStateMutex);
    auto* state = static_cast<EventState*>(event);
    if (!gEvents.contains(state)) return CUDA_ERROR_INVALID_HANDLE;
    state->recorded = true;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuEventQuery(CUevent event) {
    trace("cuEventQuery");
    std::lock_guard lock(gCudaStateMutex);
    auto* state = static_cast<EventState*>(event);
    if (!gEvents.contains(state)) return CUDA_ERROR_INVALID_HANDLE;
    return state->recorded ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}

IMB_CUDA_EXPORT CUresult cuEventSynchronize(CUevent event) {
    trace("cuEventSynchronize");
    std::lock_guard lock(gCudaStateMutex);
    auto* state = static_cast<EventState*>(event);
    if (!gEvents.contains(state)) return CUDA_ERROR_INVALID_HANDLE;
    state->recorded = true;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuEventDestroy(CUevent event) {
    trace("cuEventDestroy");
    if (event == nullptr) return CUDA_SUCCESS;
    auto* state = static_cast<EventState*>(event);
    {
        std::lock_guard lock(gCudaStateMutex);
        if (gEvents.erase(state) == 0) return CUDA_ERROR_INVALID_HANDLE;
    }
    delete state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuEventDestroy_v2(CUevent event) {
    return cuEventDestroy(event);
}

IMB_CUDA_EXPORT CUresult cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int) {
    trace("cuStreamWaitEvent");
    if (stream != nullptr) {
        std::lock_guard lock(gCudaStateMutex);
        if (!gStreams.contains(static_cast<StreamState*>(stream))) return CUDA_ERROR_INVALID_HANDLE;
    }
    return cuEventSynchronize(event);
}

IMB_CUDA_EXPORT CUresult cuModuleLoadDataEx(
    CUmodule* module,
    const void* image,
    unsigned int,
    int*,
    void**
) {
    trace("cuModuleLoadDataEx");
    return loadModuleToken(module, image);
}

IMB_CUDA_EXPORT CUresult cuModuleLoadData(CUmodule* module, const void* image) {
    trace("cuModuleLoadData");
    return loadModuleToken(module, image);
}

IMB_CUDA_EXPORT CUresult cuModuleLoadFatBinary(CUmodule* module, const void* fatbin) {
    trace("cuModuleLoadFatBinary");
    return loadModuleToken(module, fatbin);
}

IMB_CUDA_EXPORT CUresult cuLibraryLoadData(
    CUlibrary* library,
    const void* code,
    int* jitOptions,
    void** jitOptionValues,
    unsigned int jitOptionCount,
    int* libraryOptions,
    void** libraryOptionValues,
    unsigned int libraryOptionCount
) {
    trace("cuLibraryLoadData");
    if ((jitOptionCount != 0 && (jitOptions == nullptr || jitOptionValues == nullptr))
        || (libraryOptionCount != 0
            && (libraryOptions == nullptr || libraryOptionValues == nullptr))) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return loadLibraryToken(library, code);
}

IMB_CUDA_EXPORT CUresult cuLibraryLoadFromFile(
    CUlibrary* library,
    const char* fileName,
    int* jitOptions,
    void** jitOptionValues,
    unsigned int jitOptionCount,
    int* libraryOptions,
    void** libraryOptionValues,
    unsigned int libraryOptionCount
) {
    trace("cuLibraryLoadFromFile");
    return cuLibraryLoadData(
        library,
        fileName,
        jitOptions,
        jitOptionValues,
        jitOptionCount,
        libraryOptions,
        libraryOptionValues,
        libraryOptionCount
    );
}

IMB_CUDA_EXPORT CUresult cuLibraryGetModule(CUmodule* module, CUlibrary library) {
    trace("cuLibraryGetModule");
    if (module == nullptr || library == nullptr) return CUDA_ERROR_INVALID_VALUE;
    auto* state = static_cast<LibraryState*>(library);
    std::lock_guard lock(gCudaStateMutex);
    if (!gLibraries.contains(state) || !gModules.contains(state->module)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *module = state->module;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuLibraryGetKernel(
    CUkernel* kernel,
    CUlibrary library,
    const char* name
) {
    trace("cuLibraryGetKernel");
    if (kernel == nullptr || library == nullptr || name == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto* libraryState = static_cast<LibraryState*>(library);
    std::lock_guard lock(gCudaStateMutex);
    if (!gLibraries.contains(libraryState) || !gModules.contains(libraryState->module)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }

    auto* functionState = new (std::nothrow) FunctionState{libraryState->module};
    if (functionState == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    auto* kernelState = new (std::nothrow) KernelState{libraryState, functionState, name};
    if (kernelState == nullptr) {
        delete functionState;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    gFunctions.insert(functionState);
    gKernels.insert(kernelState);
    *kernel = kernelState;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuKernelGetFunction(CUfunction* function, CUkernel kernel) {
    trace("cuKernelGetFunction");
    if (function == nullptr || kernel == nullptr) return CUDA_ERROR_INVALID_VALUE;
    auto* state = static_cast<KernelState*>(kernel);
    std::lock_guard lock(gCudaStateMutex);
    if (!gKernels.contains(state) || !gFunctions.contains(state->function)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *function = state->function;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuKernelGetName(const char** name, CUkernel kernel) {
    trace("cuKernelGetName");
    if (name == nullptr || kernel == nullptr) return CUDA_ERROR_INVALID_VALUE;
    auto* state = static_cast<KernelState*>(kernel);
    std::lock_guard lock(gCudaStateMutex);
    if (!gKernels.contains(state)) return CUDA_ERROR_INVALID_HANDLE;
    *name = state->name.c_str();
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuKernelGetAttribute(
    int* value,
    int,
    CUkernel kernel,
    CUdevice device
) {
    trace("cuKernelGetAttribute");
    if (value == nullptr || kernel == nullptr || !validDevice(device)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard lock(gCudaStateMutex);
    if (!gKernels.contains(static_cast<KernelState*>(kernel))) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *value = 0;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuKernelSetAttribute(int, int, CUkernel kernel, CUdevice device) {
    trace("cuKernelSetAttribute");
    if (kernel == nullptr || !validDevice(device)) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(gCudaStateMutex);
    return gKernels.contains(static_cast<KernelState*>(kernel))
        ? CUDA_SUCCESS
        : CUDA_ERROR_INVALID_HANDLE;
}

IMB_CUDA_EXPORT CUresult cuKernelSetCacheConfig(CUkernel kernel, int, CUdevice device) {
    trace("cuKernelSetCacheConfig");
    if (kernel == nullptr || !validDevice(device)) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(gCudaStateMutex);
    return gKernels.contains(static_cast<KernelState*>(kernel))
        ? CUDA_SUCCESS
        : CUDA_ERROR_INVALID_HANDLE;
}

IMB_CUDA_EXPORT CUresult cuLibraryUnload(CUlibrary library) {
    trace("cuLibraryUnload");
    if (library == nullptr) return CUDA_SUCCESS;
    auto* state = static_cast<LibraryState*>(library);
    {
        std::lock_guard lock(gCudaStateMutex);
        if (!gLibraries.contains(state)) return CUDA_ERROR_INVALID_HANDLE;
        for (auto iterator = gKernels.begin(); iterator != gKernels.end();) {
            KernelState* kernel = *iterator;
            if (kernel->library != state) {
                ++iterator;
                continue;
            }
            gFunctions.erase(kernel->function);
            delete kernel->function;
            iterator = gKernels.erase(iterator);
            delete kernel;
        }
        gModules.erase(state->module);
        gLibraries.erase(state);
    }
    delete state->module;
    delete state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuModuleGetFunction(CUfunction* function, CUmodule module, const char* name) {
    trace("cuModuleGetFunction");
    if (function == nullptr || module == nullptr || name == nullptr) return CUDA_ERROR_INVALID_VALUE;
    auto* moduleState = static_cast<ModuleState*>(module);
    std::lock_guard lock(gCudaStateMutex);
    if (!gModules.contains(moduleState)) return CUDA_ERROR_INVALID_HANDLE;
    auto* state = new (std::nothrow) FunctionState{moduleState};
    if (state == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    gFunctions.insert(state);
    *function = state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuModuleUnload(CUmodule module) {
    trace("cuModuleUnload");
    if (module == nullptr) return CUDA_SUCCESS;
    auto* state = static_cast<ModuleState*>(module);
    {
        std::lock_guard lock(gCudaStateMutex);
        if (gModules.erase(state) == 0) return CUDA_ERROR_INVALID_HANDLE;
    }
    delete state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuLaunchKernel(
    CUfunction function,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    unsigned int,
    CUstream,
    void**,
    void**
) {
    trace("cuLaunchKernel");
    std::lock_guard lock(gCudaStateMutex);
    return gFunctions.contains(static_cast<FunctionState*>(function))
        ? CUDA_SUCCESS
        : CUDA_ERROR_INVALID_HANDLE;
}

IMB_CUDA_EXPORT CUresult cuImportExternalMemory(
    CUexternalMemory* externalMemory,
    const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* descriptor
) {
    trace("cuImportExternalMemory");
    if (externalMemory == nullptr || descriptor == nullptr || descriptor->type != 1
        || descriptor->handle.fd < 0 || descriptor->size == 0
        || (descriptor->flags & ~1U) != 0 || descriptor->size > SIZE_MAX) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    struct stat status {};
    if (::fstat(descriptor->handle.fd, &status) != 0 || status.st_size < 0
        || static_cast<unsigned long long>(status.st_size) < descriptor->size) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    void* mapping = ::mmap(
        nullptr,
        static_cast<std::size_t>(descriptor->size),
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        descriptor->handle.fd,
        0
    );
    if (mapping == MAP_FAILED) return CUDA_ERROR_OUT_OF_MEMORY;
    auto* state = new (std::nothrow) ExternalMemoryState{
        descriptor->handle.fd,
        static_cast<std::size_t>(descriptor->size),
        mapping,
    };
    if (state == nullptr) {
        ::munmap(mapping, static_cast<std::size_t>(descriptor->size));
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    {
        std::lock_guard lock(gCudaStateMutex);
        gExternalMemories.insert(state);
    }
    *externalMemory = state;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-cuda-shim: imported OPAQUE_FD=%d bytes=%zu external=%p\n",
            state->fd,
            state->size,
            static_cast<void*>(state)
        );
    }
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuExternalMemoryGetMappedBuffer(
    CUdeviceptr* devicePointer,
    CUexternalMemory externalMemory,
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* descriptor
) {
    trace("cuExternalMemoryGetMappedBuffer");
    if (devicePointer == nullptr || descriptor == nullptr || descriptor->size == 0
        || descriptor->flags != 0 || descriptor->offset > SIZE_MAX
        || descriptor->size > SIZE_MAX) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard lock(gCudaStateMutex);
    auto* state = static_cast<ExternalMemoryState*>(externalMemory);
    if (!gExternalMemories.contains(state)
        || descriptor->offset > state->size
        || descriptor->size > state->size - static_cast<std::size_t>(descriptor->offset)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    const auto pointer = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(state->mapping))
        + descriptor->offset;
    gDeviceAllocations[pointer] = static_cast<std::size_t>(descriptor->size);
    gExternalDevicePointers.insert(pointer);
    *devicePointer = pointer;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray* mipmap,
    CUexternalMemory externalMemory,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* descriptor
) {
    trace("cuExternalMemoryGetMappedMipmappedArray");
    if (mipmap == nullptr || descriptor == nullptr || descriptor->numLevels == 0
        || descriptor->arrayDesc.Width == 0 || descriptor->arrayDesc.Height == 0
        || descriptor->arrayDesc.NumChannels == 0 || descriptor->arrayDesc.NumChannels > 4
        || descriptor->offset > SIZE_MAX) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard lock(gCudaStateMutex);
    auto* memory = static_cast<ExternalMemoryState*>(externalMemory);
    if (!gExternalMemories.contains(memory)
        || static_cast<std::size_t>(descriptor->offset) >= memory->size) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    auto* state = new (std::nothrow) MipmappedArrayState{};
    if (state == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    state->memory = memory;
    state->descriptor = *descriptor;
    try {
        state->levels.resize(descriptor->numLevels, nullptr);
        for (unsigned int level = 0; level < descriptor->numLevels; ++level) {
            auto* array = new ArrayState{state, level};
            state->levels[level] = array;
            gArrays.insert(array);
        }
    } catch (const std::bad_alloc&) {
        for (auto* array : state->levels) {
            if (array != nullptr) {
                gArrays.erase(array);
                delete array;
            }
        }
        delete state;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    gMipmappedArrays.insert(state);
    *mipmap = state;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-cuda-shim: mapped mipmap external=%p mipmap=%p extent=%zux%zux%zu channels=%u format=%d levels=%u\n",
            static_cast<void*>(memory),
            static_cast<void*>(state),
            descriptor->arrayDesc.Width,
            descriptor->arrayDesc.Height,
            descriptor->arrayDesc.Depth,
            descriptor->arrayDesc.NumChannels,
            descriptor->arrayDesc.Format,
            descriptor->numLevels
        );
    }
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMipmappedArrayGetLevel(
    CUarray* array,
    CUmipmappedArray mipmap,
    unsigned int level
) {
    trace("cuMipmappedArrayGetLevel");
    if (array == nullptr) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard lock(gCudaStateMutex);
    auto* state = static_cast<MipmappedArrayState*>(mipmap);
    if (!gMipmappedArrays.contains(state) || level >= state->levels.size()
        || state->levels[level] == nullptr) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *array = state->levels[level];
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuMipmappedArrayDestroy(CUmipmappedArray mipmap) {
    trace("cuMipmappedArrayDestroy");
    if (mipmap == nullptr) return CUDA_SUCCESS;
    std::lock_guard lock(gCudaStateMutex);
    auto* state = static_cast<MipmappedArrayState*>(mipmap);
    if (gMipmappedArrays.erase(state) == 0) return CUDA_ERROR_INVALID_HANDLE;
    for (auto* array : state->levels) {
        if (array != nullptr) {
            gArrays.erase(array);
            delete array;
        }
    }
    delete state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDestroyExternalMemory(CUexternalMemory externalMemory) {
    trace("cuDestroyExternalMemory");
    if (externalMemory == nullptr) return CUDA_SUCCESS;
    std::lock_guard lock(gCudaStateMutex);
    auto* state = static_cast<ExternalMemoryState*>(externalMemory);
    if (!gExternalMemories.contains(state)) return CUDA_ERROR_INVALID_HANDLE;
    for (const auto* mipmap : gMipmappedArrays) {
        if (mipmap->memory == state) return CUDA_ERROR_INVALID_HANDLE;
    }
    gExternalMemories.erase(state);
    if (state->mapping != MAP_FAILED) ::munmap(state->mapping, state->size);
    if (state->fd >= 0) ::close(state->fd);
    delete state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuImportExternalSemaphore(
    CUexternalSemaphore* externalSemaphore,
    const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC* descriptor
) {
    trace("cuImportExternalSemaphore");
    if (externalSemaphore == nullptr || descriptor == nullptr
        || descriptor->type != 1 || descriptor->handle.fd < 0
        || descriptor->flags != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    struct stat status {};
    if (::fstat(descriptor->handle.fd, &status) != 0
        || status.st_size < static_cast<off_t>(sizeof(std::uint64_t))) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto* state = new (std::nothrow) ExternalSemaphoreState{descriptor->handle.fd};
    if (state == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    {
        std::lock_guard lock(gCudaStateMutex);
        gExternalSemaphores.insert(state);
    }
    *externalSemaphore = state;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-cuda-shim: imported semaphore OPAQUE_FD=%d external=%p\n",
            state->fd,
            static_cast<void*>(state)
        );
    }
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuSignalExternalSemaphoresAsync(
    const CUexternalSemaphore* externalSemaphores,
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS* parameters,
    unsigned int count,
    CUstream
) {
    trace("cuSignalExternalSemaphoresAsync");
    if (count != 0 && (externalSemaphores == nullptr || parameters == nullptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard lock(gCudaStateMutex);
    for (unsigned int index = 0; index < count; ++index) {
        auto* state = static_cast<ExternalSemaphoreState*>(externalSemaphores[index]);
        if (!gExternalSemaphores.contains(state) || parameters[index].flags != 0) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        const std::uint64_t value = parameters[index].params.fence.value == 0
            ? 1
            : parameters[index].params.fence.value;
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
        std::size_t written = 0;
        while (written < sizeof(value)) {
            const ssize_t result = ::pwrite(
                state->fd,
                bytes + written,
                sizeof(value) - written,
                static_cast<off_t>(written)
            );
            if (result < 0 && errno == EINTR) continue;
            if (result <= 0) return CUDA_ERROR_INVALID_HANDLE;
            written += static_cast<std::size_t>(result);
        }
    }
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuWaitExternalSemaphoresAsync(
    const CUexternalSemaphore* externalSemaphores,
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS* parameters,
    unsigned int count,
    CUstream
) {
    trace("cuWaitExternalSemaphoresAsync");
    if (count != 0 && (externalSemaphores == nullptr || parameters == nullptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::lock_guard lock(gCudaStateMutex);
    for (unsigned int index = 0; index < count; ++index) {
        auto* state = static_cast<ExternalSemaphoreState*>(externalSemaphores[index]);
        if (!gExternalSemaphores.contains(state) || parameters[index].flags != 0) {
            return CUDA_ERROR_INVALID_HANDLE;
        }
        std::uint64_t current = 0;
        auto* bytes = reinterpret_cast<std::uint8_t*>(&current);
        std::size_t read = 0;
        while (read < sizeof(current)) {
            const ssize_t result = ::pread(
                state->fd,
                bytes + read,
                sizeof(current) - read,
                static_cast<off_t>(read)
            );
            if (result < 0 && errno == EINTR) continue;
            if (result <= 0) return CUDA_ERROR_INVALID_HANDLE;
            read += static_cast<std::size_t>(result);
        }
        // The compatibility driver executes CUDA work synchronously. A Vulkan
        // value below the requested timeline point is therefore retained as a
        // pending stream dependency rather than reported as a CPU-side error.
        (void)current;
    }
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuDestroyExternalSemaphore(
    CUexternalSemaphore externalSemaphore
) {
    trace("cuDestroyExternalSemaphore");
    if (externalSemaphore == nullptr) return CUDA_SUCCESS;
    auto* state = static_cast<ExternalSemaphoreState*>(externalSemaphore);
    {
        std::lock_guard lock(gCudaStateMutex);
        if (gExternalSemaphores.erase(state) == 0) return CUDA_ERROR_INVALID_HANDLE;
    }
    if (state->fd >= 0) ::close(state->fd);
    delete state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuArray3DCreate_v2(CUarray* array, const void* descriptor) {
    trace("cuArray3DCreate_v2");
    if (array == nullptr || descriptor == nullptr) return CUDA_ERROR_INVALID_VALUE;
    auto* state = new (std::nothrow) ArrayState{};
    if (state == nullptr) return CUDA_ERROR_OUT_OF_MEMORY;
    {
        std::lock_guard lock(gCudaStateMutex);
        gArrays.insert(state);
    }
    *array = state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuArrayDestroy(CUarray array) {
    trace("cuArrayDestroy");
    if (array == nullptr) return CUDA_SUCCESS;
    auto* state = static_cast<ArrayState*>(array);
    {
        std::lock_guard lock(gCudaStateMutex);
        if (gArrays.erase(state) == 0) return CUDA_ERROR_INVALID_HANDLE;
        if (state->mipmap != nullptr && gMipmappedArrays.contains(state->mipmap)
            && state->level < state->mipmap->levels.size()
            && state->mipmap->levels[state->level] == state) {
            state->mipmap->levels[state->level] = nullptr;
        }
    }
    delete state;
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuTexObjectCreate(
    CUtexObject* texture,
    const void* resourceDescriptor,
    const void* textureDescriptor,
    const void*
) {
    trace("cuTexObjectCreate");
    if (texture == nullptr || resourceDescriptor == nullptr || textureDescriptor == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *texture = gNextTextureObject.fetch_add(1);
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuTexObjectDestroy(CUtexObject texture) {
    trace("cuTexObjectDestroy");
    return texture == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuModuleGetLoadingMode(int* mode) {
    trace("cuModuleGetLoadingMode -> eager");
    if (mode == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *mode = 1;  // CU_MODULE_EAGER_LOADING
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuGetErrorName(CUresult error, const char** name) {
    if (name == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *name = error == CUDA_SUCCESS ? "CUDA_SUCCESS" : "CUDA_ERROR_NOT_SUPPORTED";
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuGetErrorString(CUresult error, const char** text) {
    if (text == nullptr) return CUDA_ERROR_INVALID_VALUE;
    *text = error == CUDA_SUCCESS ? "success" : "operation is not implemented by IsaacMetalBridge";
    return CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuGetProcAddress(
    const char* symbol,
    void** function,
    int cudaVersion,
    std::uint64_t flags
);

IMB_CUDA_EXPORT CUresult cuGetProcAddress_v2(
    const char* symbol,
    void** function,
    int cudaVersion,
    std::uint64_t flags,
    int* symbolStatus
);

IMB_CUDA_EXPORT CUresult cuGetExportTable(const void** table, const void* tableId);

void* lookup(const char* symbol) {
    if (symbol == nullptr) return nullptr;
#define IMB_CUDA_SYMBOL(name) if (std::strcmp(symbol, #name) == 0) return reinterpret_cast<void*>(name)
    IMB_CUDA_SYMBOL(cuInit);
    IMB_CUDA_SYMBOL(cuDriverGetVersion);
    IMB_CUDA_SYMBOL(cuDeviceGetCount);
    IMB_CUDA_SYMBOL(cuDeviceGet);
    IMB_CUDA_SYMBOL(cuDeviceGetName);
    IMB_CUDA_SYMBOL(cuDeviceGetUuid);
    IMB_CUDA_SYMBOL(cuDeviceGetUuid_v2);
    IMB_CUDA_SYMBOL(cuDeviceTotalMem);
    IMB_CUDA_SYMBOL(cuDeviceTotalMem_v2);
    IMB_CUDA_SYMBOL(cuDeviceGetAttribute);
    IMB_CUDA_SYMBOL(cuDeviceGetPCIBusId);
    IMB_CUDA_SYMBOL(cuDeviceGetByPCIBusId);
    IMB_CUDA_SYMBOL(cuDeviceGetLuid);
    IMB_CUDA_SYMBOL(cuDeviceGetP2PAttribute);
    IMB_CUDA_SYMBOL(cuDeviceGetTexture1DLinearMaxWidth);
    IMB_CUDA_SYMBOL(cuDeviceGetDefaultMemPool);
    IMB_CUDA_SYMBOL(cuDeviceGetMemPool);
    IMB_CUDA_SYMBOL(cuDeviceSetMemPool);
    IMB_CUDA_SYMBOL(cuFlushGPUDirectRDMAWrites);
    IMB_CUDA_SYMBOL(cuDevicePrimaryCtxRetain);
    IMB_CUDA_SYMBOL(cuDevicePrimaryCtxRelease);
    IMB_CUDA_SYMBOL(cuDevicePrimaryCtxRelease_v2);
    IMB_CUDA_SYMBOL(cuDevicePrimaryCtxReset);
    IMB_CUDA_SYMBOL(cuDevicePrimaryCtxReset_v2);
    IMB_CUDA_SYMBOL(cuDevicePrimaryCtxGetState);
    IMB_CUDA_SYMBOL(cuDevicePrimaryCtxSetFlags);
    IMB_CUDA_SYMBOL(cuDevicePrimaryCtxSetFlags_v2);
    IMB_CUDA_SYMBOL(cuCtxSetCurrent);
    IMB_CUDA_SYMBOL(cuCtxCreate);
    IMB_CUDA_SYMBOL(cuCtxCreate_v2);
    IMB_CUDA_SYMBOL(cuCtxDestroy);
    IMB_CUDA_SYMBOL(cuCtxDestroy_v2);
    IMB_CUDA_SYMBOL(cuCtxDetach);
    IMB_CUDA_SYMBOL(cuCtxGetCurrent);
    IMB_CUDA_SYMBOL(cuCtxGetDevice);
    IMB_CUDA_SYMBOL(cuCtxGetApiVersion);
    IMB_CUDA_SYMBOL(cuCtxSynchronize);
    IMB_CUDA_SYMBOL(cuCtxGetFlags);
    IMB_CUDA_SYMBOL(cuCtxGetLimit);
    IMB_CUDA_SYMBOL(cuCtxSetLimit);
    IMB_CUDA_SYMBOL(cuCtxGetCacheConfig);
    IMB_CUDA_SYMBOL(cuCtxSetCacheConfig);
    IMB_CUDA_SYMBOL(cuCtxGetSharedMemConfig);
    IMB_CUDA_SYMBOL(cuCtxSetSharedMemConfig);
    IMB_CUDA_SYMBOL(cuCtxGetStreamPriorityRange);
    IMB_CUDA_SYMBOL(cuCtxPushCurrent);
    IMB_CUDA_SYMBOL(cuCtxPushCurrent_v2);
    IMB_CUDA_SYMBOL(cuCtxPopCurrent);
    IMB_CUDA_SYMBOL(cuCtxPopCurrent_v2);
    IMB_CUDA_SYMBOL(cuCtxResetPersistingL2Cache);
    IMB_CUDA_SYMBOL(cuDeviceCanAccessPeer);
    IMB_CUDA_SYMBOL(cuMemGetInfo);
    IMB_CUDA_SYMBOL(cuMemGetInfo_v2);
    IMB_CUDA_SYMBOL(cuMemAlloc);
    IMB_CUDA_SYMBOL(cuMemAlloc_v2);
    IMB_CUDA_SYMBOL(cuMemFree);
    IMB_CUDA_SYMBOL(cuMemFree_v2);
    IMB_CUDA_SYMBOL(cuMemAddressReserve);
    IMB_CUDA_SYMBOL(cuMemAddressFree);
    IMB_CUDA_SYMBOL(cuMemCreate);
    IMB_CUDA_SYMBOL(cuMemRelease);
    IMB_CUDA_SYMBOL(cuMemMap);
    IMB_CUDA_SYMBOL(cuMemUnmap);
    IMB_CUDA_SYMBOL(cuMemSetAccess);
    IMB_CUDA_SYMBOL(cuMemGetAllocationGranularity);
    IMB_CUDA_SYMBOL(cuMemExportToShareableHandle);
    IMB_CUDA_SYMBOL(cuMemHostAlloc);
    IMB_CUDA_SYMBOL(cuMemFreeHost);
    IMB_CUDA_SYMBOL(cuMemHostGetDevicePointer);
    IMB_CUDA_SYMBOL(cuMemHostGetDevicePointer_v2);
    IMB_CUDA_SYMBOL(cuMemcpyHtoD);
    IMB_CUDA_SYMBOL(cuMemcpyHtoD_v2);
    IMB_CUDA_SYMBOL(cuMemcpyHtoDAsync_v2);
    IMB_CUDA_SYMBOL(cuMemcpyDtoH);
    IMB_CUDA_SYMBOL(cuMemcpyDtoH_v2);
    IMB_CUDA_SYMBOL(cuMemcpyDtoHAsync_v2);
    IMB_CUDA_SYMBOL(cuMemcpyDtoD);
    IMB_CUDA_SYMBOL(cuMemcpyDtoD_v2);
    IMB_CUDA_SYMBOL(cuMemcpyDtoDAsync_v2);
    IMB_CUDA_SYMBOL(cuMemcpyPeerAsync);
    IMB_CUDA_SYMBOL(cuMemcpy3DAsync_v2);
    IMB_CUDA_SYMBOL(cuMemsetD8_v2);
    IMB_CUDA_SYMBOL(cuMemsetD16_v2);
    IMB_CUDA_SYMBOL(cuMemsetD32_v2);
    IMB_CUDA_SYMBOL(cuMemsetD8Async);
    IMB_CUDA_SYMBOL(cuMemsetD32Async);
    IMB_CUDA_SYMBOL(cuStreamCreate);
    IMB_CUDA_SYMBOL(cuStreamCreateWithPriority);
    IMB_CUDA_SYMBOL(cuStreamDestroy);
    IMB_CUDA_SYMBOL(cuStreamDestroy_v2);
    IMB_CUDA_SYMBOL(cuStreamQuery);
    IMB_CUDA_SYMBOL(cuStreamSynchronize);
    IMB_CUDA_SYMBOL(cuEventCreate);
    IMB_CUDA_SYMBOL(cuEventRecord);
    IMB_CUDA_SYMBOL(cuEventQuery);
    IMB_CUDA_SYMBOL(cuEventSynchronize);
    IMB_CUDA_SYMBOL(cuEventDestroy);
    IMB_CUDA_SYMBOL(cuEventDestroy_v2);
    IMB_CUDA_SYMBOL(cuStreamWaitEvent);
    IMB_CUDA_SYMBOL(cuModuleLoadData);
    IMB_CUDA_SYMBOL(cuModuleLoadDataEx);
    IMB_CUDA_SYMBOL(cuModuleLoadFatBinary);
    IMB_CUDA_SYMBOL(cuModuleGetFunction);
    IMB_CUDA_SYMBOL(cuModuleUnload);
    IMB_CUDA_SYMBOL(cuLibraryLoadData);
    IMB_CUDA_SYMBOL(cuLibraryLoadFromFile);
    IMB_CUDA_SYMBOL(cuLibraryGetModule);
    IMB_CUDA_SYMBOL(cuLibraryGetKernel);
    IMB_CUDA_SYMBOL(cuLibraryUnload);
    IMB_CUDA_SYMBOL(cuKernelGetFunction);
    IMB_CUDA_SYMBOL(cuKernelGetName);
    IMB_CUDA_SYMBOL(cuKernelGetAttribute);
    IMB_CUDA_SYMBOL(cuKernelSetAttribute);
    IMB_CUDA_SYMBOL(cuKernelSetCacheConfig);
    IMB_CUDA_SYMBOL(cuLaunchKernel);
    IMB_CUDA_SYMBOL(cuImportExternalMemory);
    IMB_CUDA_SYMBOL(cuExternalMemoryGetMappedBuffer);
    IMB_CUDA_SYMBOL(cuExternalMemoryGetMappedMipmappedArray);
    IMB_CUDA_SYMBOL(cuMipmappedArrayGetLevel);
    IMB_CUDA_SYMBOL(cuMipmappedArrayDestroy);
    IMB_CUDA_SYMBOL(cuDestroyExternalMemory);
    IMB_CUDA_SYMBOL(cuImportExternalSemaphore);
    IMB_CUDA_SYMBOL(cuSignalExternalSemaphoresAsync);
    IMB_CUDA_SYMBOL(cuWaitExternalSemaphoresAsync);
    IMB_CUDA_SYMBOL(cuDestroyExternalSemaphore);
    IMB_CUDA_SYMBOL(cuArray3DCreate_v2);
    IMB_CUDA_SYMBOL(cuArrayDestroy);
    IMB_CUDA_SYMBOL(cuTexObjectCreate);
    IMB_CUDA_SYMBOL(cuTexObjectDestroy);
    IMB_CUDA_SYMBOL(cuModuleGetLoadingMode);
    IMB_CUDA_SYMBOL(cuGetErrorName);
    IMB_CUDA_SYMBOL(cuGetErrorString);
    IMB_CUDA_SYMBOL(cuGetProcAddress);
    IMB_CUDA_SYMBOL(cuGetProcAddress_v2);
    IMB_CUDA_SYMBOL(cuGetExportTable);
#undef IMB_CUDA_SYMBOL
    return reinterpret_cast<void*>(gUnsupportedFunctions[unsupportedSlotFor(symbol)]);
}

IMB_CUDA_EXPORT CUresult cuGetProcAddress(
    const char* symbol,
    void** function,
    int cudaVersion,
    std::uint64_t flags
) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-cuda-shim: cuGetProcAddress symbol=%s version=%d flags=%llu\n",
            symbol == nullptr ? "(null)" : symbol,
            cudaVersion,
            static_cast<unsigned long long>(flags)
        );
    }
    if (symbol == nullptr || function == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (*symbol == '\0') {
        *function = nullptr;
        return CUDA_ERROR_NOT_FOUND;
    }
    *function = lookup(symbol);
    return *function == nullptr ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS;
}

IMB_CUDA_EXPORT CUresult cuGetProcAddress_v2(
    const char* symbol,
    void** function,
    int cudaVersion,
    std::uint64_t flags,
    int* symbolStatus
) {
    const CUresult result = cuGetProcAddress(symbol, function, cudaVersion, flags);
    if (symbolStatus != nullptr) *symbolStatus = result == CUDA_SUCCESS ? 0 : 2;
    return result;
}

IMB_CUDA_EXPORT CUresult cuGetExportTable(const void** table, const void* tableId) {
    if (traceEnabled()) {
        const auto* bytes = static_cast<const unsigned char*>(tableId);
        std::fprintf(stderr, "imb-cuda-shim: cuGetExportTable id=");
        if (bytes == nullptr) {
            std::fprintf(stderr, "(null)");
        } else {
            for (std::size_t index = 0; index < 16; ++index) std::fprintf(stderr, "%02x", bytes[index]);
        }
        std::fprintf(stderr, "\n");
    }
    if (table == nullptr || tableId == nullptr) return CUDA_ERROR_INVALID_VALUE;
    if (std::memcmp(tableId, kCudartExportTableUUID, sizeof(kCudartExportTableUUID)) == 0) {
        *table = gCudartExportTable;
        trace("cuGetExportTable -> cudart interface");
        return CUDA_SUCCESS;
    }
    if (std::memcmp(tableId, kRuntimeCallbackHooksUUID, sizeof(kRuntimeCallbackHooksUUID)) == 0) {
        *table = gRuntimeCallbackHooksTable;
        trace("cuGetExportTable -> runtime callback hooks");
        return CUDA_SUCCESS;
    }
    if (std::memcmp(tableId, kToolsTlsUUID, sizeof(kToolsTlsUUID)) == 0) {
        *table = gToolsTlsTable;
        trace("cuGetExportTable -> tools TLS");
        return CUDA_SUCCESS;
    }
    if (std::memcmp(tableId, kContextLocalStorageUUID, sizeof(kContextLocalStorageUUID)) == 0) {
        *table = gContextLocalStorageTable;
        trace("cuGetExportTable -> context local storage");
        return CUDA_SUCCESS;
    }
    if (std::memcmp(tableId, kContextChecksUUID, sizeof(kContextChecksUUID)) == 0) {
        *table = gContextChecksTable;
        trace("cuGetExportTable -> context checks");
        return CUDA_SUCCESS;
    }
    if (std::memcmp(tableId, kIntegrityCheckUUID, sizeof(kIntegrityCheckUUID)) == 0) {
        *table = gIntegrityCheckTable;
        trace("cuGetExportTable -> integrity check");
        return CUDA_SUCCESS;
    }
    *table = nullptr;
    trace("cuGetExportTable -> not supported");
    return CUDA_ERROR_NOT_SUPPORTED;
}

}  // extern "C"
