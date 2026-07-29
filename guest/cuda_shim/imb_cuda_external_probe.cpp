#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <fcntl.h>
#include <unistd.h>

using CUresult = int;
using CUarray = void*;
using CUmipmappedArray = void*;
using CUexternalMemory = void*;
using CUarray_format = int;
using CUdeviceptr = std::uint64_t;
using CUmemGenericAllocationHandle = std::uint64_t;

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

struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC {
    unsigned long long offset;
    CUDA_ARRAY3D_DESCRIPTOR arrayDesc;
    unsigned int numLevels;
    unsigned int reserved[16];
};

struct CUmemAllocationProp {
    std::uint8_t bytes[64];
};

struct CUmemAccessDesc {
    std::uint8_t bytes[16];
};

extern "C" CUresult cuImportExternalMemory(
    CUexternalMemory*,
    const CUDA_EXTERNAL_MEMORY_HANDLE_DESC*
);
extern "C" CUresult cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray*,
    CUexternalMemory,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC*
);
extern "C" CUresult cuMipmappedArrayGetLevel(CUarray*, CUmipmappedArray, unsigned int);
extern "C" CUresult cuMipmappedArrayDestroy(CUmipmappedArray);
extern "C" CUresult cuDestroyExternalMemory(CUexternalMemory);
extern "C" CUresult cuMemAddressReserve(
    CUdeviceptr*,
    std::size_t,
    std::size_t,
    CUdeviceptr,
    unsigned long long
);
extern "C" CUresult cuMemAddressFree(CUdeviceptr, std::size_t);
extern "C" CUresult cuMemCreate(
    CUmemGenericAllocationHandle*,
    std::size_t,
    const CUmemAllocationProp*,
    unsigned long long
);
extern "C" CUresult cuMemRelease(CUmemGenericAllocationHandle);
extern "C" CUresult cuMemMap(
    CUdeviceptr,
    std::size_t,
    std::size_t,
    CUmemGenericAllocationHandle,
    unsigned long long
);
extern "C" CUresult cuMemUnmap(CUdeviceptr, std::size_t);
extern "C" CUresult cuMemSetAccess(
    CUdeviceptr,
    std::size_t,
    const CUmemAccessDesc*,
    std::size_t
);
extern "C" CUresult cuMemGetAllocationGranularity(
    std::size_t*,
    const CUmemAllocationProp*,
    int
);
extern "C" CUresult cuMemExportToShareableHandle(
    void*,
    CUmemGenericAllocationHandle,
    int,
    unsigned long long
);

void require(CUresult result, const char* operation) {
    if (result != 0) {
        std::fprintf(stderr, "imb-cuda-external-probe: %s failed with %d\n", operation, result);
        std::exit(1);
    }
}

int main() {
    constexpr std::size_t byteCount = 1280 * 720 * 4;
    char path[] = "/tmp/imb-cuda-external.XXXXXX";
    const int fd = ::mkstemp(path);
    if (fd < 0) return 1;
    ::unlink(path);
    if (::ftruncate(fd, static_cast<off_t>(byteCount)) != 0) return 1;

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memoryDescriptor{};
    memoryDescriptor.type = 1;
    memoryDescriptor.handle.fd = fd;
    memoryDescriptor.size = byteCount;
    CUexternalMemory memory = nullptr;
    require(cuImportExternalMemory(&memory, &memoryDescriptor), "cuImportExternalMemory");
    if (memory == nullptr) return 1;

    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mipmapDescriptor{};
    mipmapDescriptor.arrayDesc.Width = 1280;
    mipmapDescriptor.arrayDesc.Height = 720;
    mipmapDescriptor.arrayDesc.Depth = 0;
    mipmapDescriptor.arrayDesc.Format = 1;
    mipmapDescriptor.arrayDesc.NumChannels = 4;
    mipmapDescriptor.arrayDesc.Flags = 0x20;
    mipmapDescriptor.numLevels = 1;
    CUmipmappedArray mipmap = nullptr;
    require(
        cuExternalMemoryGetMappedMipmappedArray(&mipmap, memory, &mipmapDescriptor),
        "cuExternalMemoryGetMappedMipmappedArray"
    );
    if (mipmap == nullptr) return 1;
    CUarray level = nullptr;
    require(cuMipmappedArrayGetLevel(&level, mipmap, 0), "cuMipmappedArrayGetLevel");
    if (level == nullptr) return 1;
    require(cuMipmappedArrayDestroy(mipmap), "cuMipmappedArrayDestroy");
    require(cuDestroyExternalMemory(memory), "cuDestroyExternalMemory");
    if (::fcntl(fd, F_GETFD) != -1) {
        std::fprintf(stderr, "imb-cuda-external-probe: imported FD ownership was not consumed\n");
        return 1;
    }

    CUmemAllocationProp allocationProperties{};
    std::size_t granularity = 0;
    require(
        cuMemGetAllocationGranularity(
            &granularity,
            &allocationProperties,
            0
        ),
        "cuMemGetAllocationGranularity"
    );
    if (granularity != 64 * 1024) return 1;
    CUdeviceptr virtualAddress = 0;
    require(
        cuMemAddressReserve(
            &virtualAddress,
            granularity,
            granularity,
            0,
            0
        ),
        "cuMemAddressReserve"
    );
    CUmemGenericAllocationHandle allocation = 0;
    require(
        cuMemCreate(
            &allocation,
            granularity,
            &allocationProperties,
            0
        ),
        "cuMemCreate"
    );
    require(
        cuMemMap(
            virtualAddress,
            granularity,
            0,
            allocation,
            0
        ),
        "cuMemMap"
    );
    CUmemAccessDesc access{};
    require(
        cuMemSetAccess(
            virtualAddress,
            granularity,
            &access,
            1
        ),
        "cuMemSetAccess"
    );
    auto* virtualBytes = reinterpret_cast<std::uint8_t*>(
        static_cast<std::uintptr_t>(virtualAddress)
    );
    virtualBytes[0] = 0x5a;
    virtualBytes[granularity - 1] = 0xa5;
    int shareableFD = -1;
    require(
        cuMemExportToShareableHandle(
            &shareableFD,
            allocation,
            1,
            0
        ),
        "cuMemExportToShareableHandle"
    );
    if (shareableFD < 0) return 1;
    std::uint8_t exportedByte = 0;
    if (::pread(shareableFD, &exportedByte, 1, 0) != 1 || exportedByte != 0x5a) {
        return 1;
    }
    ::close(shareableFD);
    require(cuMemRelease(allocation), "cuMemRelease");
    require(cuMemUnmap(virtualAddress, granularity), "cuMemUnmap");
    require(
        cuMemAddressFree(virtualAddress, granularity),
        "cuMemAddressFree"
    );

    std::puts(
        "CUDA_EXTERNAL_MEMORY image=1280x720xRGBA8 handle=OPAQUE_FD mipmap=passed "
        "virtual_memory=passed"
    );
    return 0;
}
