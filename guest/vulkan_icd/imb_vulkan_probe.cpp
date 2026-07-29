#include <vulkan/vulkan.h>

#include "texel_add_spv.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <unistd.h>

namespace {

void require(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " + std::to_string(result));
    }
}

std::vector<std::uint32_t> readSpirV(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error(std::string("could not open ") + path);
    const auto size = stream.tellg();
    if (size <= 0 || size % 4 != 0) throw std::runtime_error("SPIR-V file has invalid size");
    std::vector<std::uint32_t> words(static_cast<std::size_t>(size) / sizeof(std::uint32_t));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(words.data()), size);
    if (!stream) throw std::runtime_error("could not read SPIR-V file");
    return words;
}

void writePPM(
    const char* path,
    const std::uint8_t* rgba,
    std::uint32_t width,
    std::uint32_t height,
    VkDeviceSize rowPitch
) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error(std::string("could not create ") + path);
    output << "P6\n" << width << ' ' << height << "\n255\n";
    for (std::uint32_t y = 0; y < height; ++y) {
        const auto* row = rgba + static_cast<std::size_t>(y * rowPitch);
        for (std::uint32_t x = 0; x < width; ++x) {
            output.put(static_cast<char>(row[x * 4]));
            output.put(static_cast<char>(row[x * 4 + 1]));
            output.put(static_cast<char>(row[x * 4 + 2]));
        }
    }
    if (!output) throw std::runtime_error(std::string("could not finish ") + path);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::runtime_error("usage: imb-vulkan-probe [triangle.ppm]");
        const char* imageOutput = argc == 2 ? argv[1] : nullptr;
        VkApplicationInfo applicationInfo{};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "imb-vulkan-probe";
        applicationInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
        applicationInfo.pEngineName = "none";
        applicationInfo.engineVersion = 0;
        applicationInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &applicationInfo;

        VkInstance instance = VK_NULL_HANDLE;
        require(vkCreateInstance(&instanceInfo, nullptr, &instance), "vkCreateInstance");

        std::uint32_t physicalDeviceCount = 0;
        require(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, nullptr), "vkEnumeratePhysicalDevices(count)");
        if (physicalDeviceCount != 1) throw std::runtime_error("expected exactly one IMB physical device");
        std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
        require(vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data()), "vkEnumeratePhysicalDevices");

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevices[0], &properties);
        if (std::strncmp(properties.deviceName, "IsaacMetalBridge (", 18) != 0) {
            throw std::runtime_error("unexpected physical device name");
        }
        if (properties.apiVersion < VK_API_VERSION_1_1) {
            throw std::runtime_error("prototype ICD must advertise Vulkan 1.1 or newer");
        }
        if (properties.sparseProperties.residencyStandard2DBlockShape != VK_TRUE) {
            throw std::runtime_error("IMB did not advertise its 64 KiB Vulkan sparse block aggregation");
        }
        if (properties.limits.sparseAddressSpaceSize < (UINT64_C(1) << 30)) {
            throw std::runtime_error("IMB did not advertise a usable sparse virtual address space");
        }

        std::uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[0], &queueFamilyCount, nullptr);
        if (queueFamilyCount != 1) throw std::runtime_error("expected one IMB queue family");
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevices[0], &queueFamilyCount, queueFamilies.data());
        if ((queueFamilies[0].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
            throw std::runtime_error("IMB queue did not advertise compute");
        }

        std::uint32_t deviceExtensionCount = 0;
        require(
            vkEnumerateDeviceExtensionProperties(
                physicalDevices[0],
                nullptr,
                &deviceExtensionCount,
                nullptr
            ),
            "vkEnumerateDeviceExtensionProperties(count)"
        );
        std::vector<VkExtensionProperties> deviceExtensions(deviceExtensionCount);
        require(
            vkEnumerateDeviceExtensionProperties(
                physicalDevices[0],
                nullptr,
                &deviceExtensionCount,
                deviceExtensions.data()
            ),
            "vkEnumerateDeviceExtensionProperties"
        );
        const auto hasDeviceExtension = [&deviceExtensions](const char* name) {
            for (const auto& extension : deviceExtensions) {
                if (std::strcmp(extension.extensionName, name) == 0) return true;
            }
            return false;
        };
        const char* requiredDeviceExtensions[] = {
            VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
            VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
            VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        };
        for (const char* extension : requiredDeviceExtensions) {
            if (!hasDeviceExtension(extension)) {
                throw std::runtime_error(std::string("IMB device did not advertise ") + extension);
            }
        }

        VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{};
        accelerationStructureFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{};
        rayTracingPipelineFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
        accelerationStructureFeatures.pNext = &rayTracingPipelineFeatures;
        VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{};
        bufferDeviceAddressFeatures.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
        bufferDeviceAddressFeatures.pNext = &accelerationStructureFeatures;
        VkPhysicalDeviceFeatures2 physicalDeviceFeatures{};
        physicalDeviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        physicalDeviceFeatures.pNext = &bufferDeviceAddressFeatures;
        vkGetPhysicalDeviceFeatures2(physicalDevices[0], &physicalDeviceFeatures);
        if (bufferDeviceAddressFeatures.bufferDeviceAddress != VK_TRUE
            || accelerationStructureFeatures.accelerationStructure != VK_TRUE
            || rayTracingPipelineFeatures.rayTracingPipeline != VK_TRUE) {
            throw std::runtime_error("IMB device did not advertise Metal ray-tracing features");
        }
        if (physicalDeviceFeatures.features.sparseBinding != VK_TRUE
            || physicalDeviceFeatures.features.sparseResidencyBuffer != VK_TRUE
            || physicalDeviceFeatures.features.sparseResidencyImage2D != VK_TRUE
            || physicalDeviceFeatures.features.sparseResidencyImage3D != VK_FALSE
            || physicalDeviceFeatures.features.sparseResidencyAliased != VK_FALSE) {
            throw std::runtime_error(
                "IMB sparse profile does not match Metal 2D tile residency"
            );
        }
        std::uint32_t sparseImagePropertyCount = 1;
        vkGetPhysicalDeviceSparseImageFormatProperties(
            physicalDevices[0],
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TYPE_2D,
            VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_TILING_OPTIMAL,
            &sparseImagePropertyCount,
            nullptr
        );
        if (sparseImagePropertyCount != 1) {
            throw std::runtime_error(
                "IMB Metal sparse profile did not return RGBA8 properties"
            );
        }

        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties{};
        rayTracingProperties.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &rayTracingProperties;
        vkGetPhysicalDeviceProperties2(physicalDevices[0], &properties2);
        if (rayTracingProperties.maxRayRecursionDepth < 3) {
            throw std::runtime_error("IMB ray-tracing recursion limit is below Isaac RTX's required depth 3");
        }

        VkPhysicalDeviceExternalImageFormatInfo externalImageInfo{};
        externalImageInfo.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO;
        externalImageInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkPhysicalDeviceImageFormatInfo2 externalImageQuery{};
        externalImageQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2;
        externalImageQuery.pNext = &externalImageInfo;
        externalImageQuery.format = VK_FORMAT_R8G8B8A8_UNORM;
        externalImageQuery.type = VK_IMAGE_TYPE_2D;
        externalImageQuery.tiling = VK_IMAGE_TILING_OPTIMAL;
        externalImageQuery.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        VkExternalImageFormatProperties externalImageProperties{};
        externalImageProperties.sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES;
        VkImageFormatProperties2 externalImageResult{};
        externalImageResult.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2;
        externalImageResult.pNext = &externalImageProperties;
        require(
            vkGetPhysicalDeviceImageFormatProperties2(
                physicalDevices[0],
                &externalImageQuery,
                &externalImageResult
            ),
            "vkGetPhysicalDeviceImageFormatProperties2(OPAQUE_FD)"
        );
        constexpr VkExternalMemoryFeatureFlags requiredExternalFeatures =
            VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
        if ((externalImageProperties.externalMemoryProperties.externalMemoryFeatures
                & requiredExternalFeatures) != requiredExternalFeatures
            || (externalImageProperties.externalMemoryProperties.compatibleHandleTypes
                & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) == 0) {
            throw std::runtime_error("IMB external RGBA8 image is not OPAQUE_FD compatible");
        }

        float priorities[2] = {1.0f, 1.0f};
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = 0;
        queueInfo.queueCount = 2;
        queueInfo.pQueuePriorities = priorities;

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.pNext = &bufferDeviceAddressFeatures;
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.enabledExtensionCount =
            static_cast<std::uint32_t>(sizeof(requiredDeviceExtensions) / sizeof(requiredDeviceExtensions[0]));
        deviceInfo.ppEnabledExtensionNames = requiredDeviceExtensions;

        VkDevice device = VK_NULL_HANDLE;
        require(vkCreateDevice(physicalDevices[0], &deviceInfo, nullptr, &device), "vkCreateDevice");
        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, 0, 0, &queue);
        if (queue == VK_NULL_HANDLE) throw std::runtime_error("vkGetDeviceQueue returned null");
        VkQueue copyQueue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, 0, 1, &copyQueue);
        if (copyQueue == VK_NULL_HANDLE || copyQueue == queue) {
            throw std::runtime_error("IMB did not create distinct logical render/copy queues");
        }
        VkDeviceQueueInfo2 queueInfo2{};
        queueInfo2.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2;
        queueInfo2.queueFamilyIndex = 0;
        queueInfo2.queueIndex = 1;
        VkQueue queue2 = VK_NULL_HANDLE;
        vkGetDeviceQueue2(device, &queueInfo2, &queue2);
        if (queue2 == VK_NULL_HANDLE || queue2 != copyQueue) {
            throw std::runtime_error("vkGetDeviceQueue2 did not return the created Metal-backed queue");
        }

        std::cout << "VULKAN_ICD discovered=\"" << properties.deviceName
                  << "\" api=" << VK_API_VERSION_MAJOR(properties.apiVersion)
                  << "." << VK_API_VERSION_MINOR(properties.apiVersion)
                  << " queue=compute transport=vsock\n";

        constexpr VkDeviceSize bufferSize = 4 * sizeof(std::uint32_t);
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bufferSize;
        bufferInfo.usage =
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buffer = VK_NULL_HANDLE;
        require(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer), "vkCreateBuffer");

        VkMemoryRequirements memoryRequirements{};
        vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);
        if (memoryRequirements.memoryTypeBits == 0 || memoryRequirements.size < bufferSize) {
            throw std::runtime_error("unexpected IMB buffer memory requirements");
        }
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevices[0], &memoryProperties);
        std::uint32_t memoryTypeIndex = UINT32_MAX;
        constexpr VkMemoryPropertyFlags requiredMemoryFlags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (std::uint32_t index = 0; index < memoryProperties.memoryTypeCount; ++index) {
            const bool supported = (memoryRequirements.memoryTypeBits & (1U << index)) != 0;
            const bool mappable = (memoryProperties.memoryTypes[index].propertyFlags & requiredMemoryFlags)
                == requiredMemoryFlags;
            if (supported && mappable) {
                memoryTypeIndex = index;
                break;
            }
        }
        if (memoryTypeIndex == UINT32_MAX) {
            throw std::runtime_error("IMB buffer has no host-visible coherent memory type");
        }

        VkSparseImageFormatProperties sparseFormatProperties{};
        sparseImagePropertyCount = 1;
        vkGetPhysicalDeviceSparseImageFormatProperties(
            physicalDevices[0],
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_TYPE_2D,
            VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_TILING_OPTIMAL,
            &sparseImagePropertyCount,
            &sparseFormatProperties
        );
        if (sparseImagePropertyCount != 1
            || sparseFormatProperties.imageGranularity.width == 0
            || sparseFormatProperties.imageGranularity.height == 0
            || sparseFormatProperties.imageGranularity.depth != 1
            || (sparseFormatProperties.flags
                & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT) != 0) {
            throw std::runtime_error("IMB returned invalid standard Vulkan sparse block granularity");
        }
        VkImageCreateInfo sparseImageInfo{};
        sparseImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        sparseImageInfo.flags =
            VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
        sparseImageInfo.imageType = VK_IMAGE_TYPE_2D;
        sparseImageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        sparseImageInfo.extent = {
            sparseFormatProperties.imageGranularity.width * 2,
            sparseFormatProperties.imageGranularity.height * 2,
            1,
        };
        sparseImageInfo.mipLevels = 1;
        sparseImageInfo.arrayLayers = 1;
        sparseImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        sparseImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        sparseImageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        sparseImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sparseImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage sparseImage = VK_NULL_HANDLE;
        require(
            vkCreateImage(device, &sparseImageInfo, nullptr, &sparseImage),
            "vkCreateImage(sparse Metal)"
        );
        std::uint32_t sparseRequirementCount = 0;
        vkGetImageSparseMemoryRequirements(
            device,
            sparseImage,
            &sparseRequirementCount,
            nullptr
        );
        if (sparseRequirementCount != 1) {
            throw std::runtime_error("IMB sparse image did not report exactly one Metal-backed color requirement");
        }
        std::vector<VkSparseImageMemoryRequirements> sparseRequirements(sparseRequirementCount);
        vkGetImageSparseMemoryRequirements(
            device,
            sparseImage,
            &sparseRequirementCount,
            sparseRequirements.data()
        );
        if (sparseRequirementCount != 1
            || sparseRequirements[0].formatProperties.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT
            || (sparseRequirements[0].formatProperties.flags
                & VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT) != 0) {
            throw std::runtime_error("IMB sparse image requirements are malformed");
        }
        VkMemoryRequirements sparseMemoryRequirements{};
        vkGetImageMemoryRequirements(device, sparseImage, &sparseMemoryRequirements);
        if (sparseMemoryRequirements.alignment == 0
            || (sparseMemoryRequirements.memoryTypeBits & (1U << memoryTypeIndex)) == 0) {
            throw std::runtime_error("IMB sparse image returned invalid memory requirements");
        }
        VkMemoryAllocateInfo sparseMemoryInfo{};
        sparseMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        sparseMemoryInfo.allocationSize = sparseMemoryRequirements.alignment;
        sparseMemoryInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory sparseMemory = VK_NULL_HANDLE;
        require(
            vkAllocateMemory(device, &sparseMemoryInfo, nullptr, &sparseMemory),
            "vkAllocateMemory(sparse tile)"
        );
        VkSparseImageMemoryBind sparseTileBind{};
        sparseTileBind.subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        sparseTileBind.subresource.mipLevel = 0;
        sparseTileBind.subresource.arrayLayer = 0;
        sparseTileBind.extent = sparseFormatProperties.imageGranularity;
        sparseTileBind.memory = sparseMemory;
        VkSparseImageMemoryBindInfo sparseBindInfo{};
        sparseBindInfo.image = sparseImage;
        sparseBindInfo.bindCount = 1;
        sparseBindInfo.pBinds = &sparseTileBind;
        VkBindSparseInfo sparseQueueBind{};
        sparseQueueBind.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
        sparseQueueBind.imageBindCount = 1;
        sparseQueueBind.pImageBinds = &sparseBindInfo;
        require(vkQueueBindSparse(queue, 1, &sparseQueueBind, VK_NULL_HANDLE), "vkQueueBindSparse(map)");
        sparseTileBind.memory = VK_NULL_HANDLE;
        require(vkQueueBindSparse(queue, 1, &sparseQueueBind, VK_NULL_HANDLE), "vkQueueBindSparse(unmap)");
        vkDestroyImage(device, sparseImage, nullptr);
        vkFreeMemory(device, sparseMemory, nullptr);
        std::cout
            << "VULKAN_SPARSE_IMAGE format=RGBA8 tile="
            << sparseFormatProperties.imageGranularity.width << "x"
            << sparseFormatProperties.imageGranularity.height
            << " map=passed unmap=passed backend=Metal\n";

        const auto getMemoryFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR")
        );
        const auto getMemoryFdProperties = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
            vkGetDeviceProcAddr(device, "vkGetMemoryFdPropertiesKHR")
        );
        if (getMemoryFd == nullptr || getMemoryFdProperties == nullptr) {
            throw std::runtime_error("Vulkan loader did not expose OPAQUE_FD memory entry points");
        }
        VkExportMemoryAllocateInfo exportMemoryInfo{};
        exportMemoryInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportMemoryInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        VkMemoryAllocateInfo externalAllocationInfo{};
        externalAllocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        externalAllocationInfo.pNext = &exportMemoryInfo;
        externalAllocationInfo.allocationSize = 4096;
        externalAllocationInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory exportedMemory = VK_NULL_HANDLE;
        require(
            vkAllocateMemory(device, &externalAllocationInfo, nullptr, &exportedMemory),
            "vkAllocateMemory(export OPAQUE_FD)"
        );
        void* externalMapped = nullptr;
        require(vkMapMemory(device, exportedMemory, 0, 4096, 0, &externalMapped),
                "vkMapMemory(export OPAQUE_FD)");
        std::memset(externalMapped, 0x5a, 4096);
        vkUnmapMemory(device, exportedMemory);
        VkMemoryGetFdInfoKHR getFdInfo{};
        getFdInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        getFdInfo.memory = exportedMemory;
        getFdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        int externalFd = -1;
        require(getMemoryFd(device, &getFdInfo, &externalFd), "vkGetMemoryFdKHR");
        VkMemoryFdPropertiesKHR fdProperties{};
        fdProperties.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
        require(
            getMemoryFdProperties(
                device,
                VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
                externalFd,
                &fdProperties
            ),
            "vkGetMemoryFdPropertiesKHR"
        );
        if ((fdProperties.memoryTypeBits & (1U << memoryTypeIndex)) == 0) {
            ::close(externalFd);
            throw std::runtime_error("exported OPAQUE_FD did not support its source memory type");
        }
        VkImportMemoryFdInfoKHR importMemoryInfo{};
        importMemoryInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
        importMemoryInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        importMemoryInfo.fd = externalFd;
        VkMemoryAllocateInfo importAllocationInfo{};
        importAllocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        importAllocationInfo.pNext = &importMemoryInfo;
        importAllocationInfo.allocationSize = 4096;
        importAllocationInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory importedMemory = VK_NULL_HANDLE;
        require(
            vkAllocateMemory(device, &importAllocationInfo, nullptr, &importedMemory),
            "vkAllocateMemory(import OPAQUE_FD)"
        );
        externalMapped = nullptr;
        require(vkMapMemory(device, importedMemory, 0, 4096, 0, &externalMapped),
                "vkMapMemory(import OPAQUE_FD)");
        const auto* externalBytes = static_cast<const std::uint8_t*>(externalMapped);
        for (std::size_t index = 0; index < 4096; ++index) {
            if (externalBytes[index] != 0x5a) {
                throw std::runtime_error("OPAQUE_FD import did not preserve allocation bytes");
            }
        }
        vkUnmapMemory(device, importedMemory);
        std::cout << "VULKAN_EXTERNAL_MEMORY image=RGBA8 handle=OPAQUE_FD roundtrip=passed\n";
        VkMemoryAllocateInfo memoryInfo{};
        memoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        memoryInfo.allocationSize = memoryRequirements.size;
        memoryInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        require(vkAllocateMemory(device, &memoryInfo, nullptr, &memory), "vkAllocateMemory");
        require(vkBindBufferMemory(device, buffer, memory, 0), "vkBindBufferMemory");

        void* mapped = nullptr;
        require(vkMapMemory(device, memory, 0, bufferSize, 0, &mapped), "vkMapMemory(input)");
        const std::uint32_t input[] = {1, 2, 3, 4};
        std::memcpy(mapped, input, sizeof(input));
        vkUnmapMemory(device, memory);

        VkDescriptorSetLayoutBinding layoutBinding{};
        layoutBinding.binding = 0;
        layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBinding.descriptorCount = 1;
        layoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{};
        descriptorLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorLayoutInfo.bindingCount = 1;
        descriptorLayoutInfo.pBindings = &layoutBinding;
        VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
        require(
            vkCreateDescriptorSetLayout(device, &descriptorLayoutInfo, nullptr, &descriptorLayout),
            "vkCreateDescriptorSetLayout"
        );

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0;
        pushConstantRange.size = sizeof(std::uint32_t);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        require(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout), "vkCreatePipelineLayout");

        const auto spirV = readSpirV("/usr/local/share/imb/add_u32.spv");
        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = spirV.size() * sizeof(std::uint32_t);
        shaderInfo.pCode = spirV.data();
        VkShaderModule shader = VK_NULL_HANDLE;
        require(vkCreateShaderModule(device, &shaderInfo, nullptr, &shader), "vkCreateShaderModule");

        const auto createRayTracingPipelines = reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(
            vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR")
        );
        if (createRayTracingPipelines == nullptr) {
            throw std::runtime_error("Vulkan loader did not expose vkCreateRayTracingPipelinesKHR");
        }
        VkPipelineShaderStageCreateInfo rayGenerationStage{};
        rayGenerationStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        rayGenerationStage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
        rayGenerationStage.module = shader;
        rayGenerationStage.pName = "main";
        VkRayTracingShaderGroupCreateInfoKHR rayGenerationGroup{};
        rayGenerationGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
        rayGenerationGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        rayGenerationGroup.generalShader = 0;
        rayGenerationGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
        rayGenerationGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
        rayGenerationGroup.intersectionShader = VK_SHADER_UNUSED_KHR;
        VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{};
        rayPipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        rayPipelineInfo.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR;
        rayPipelineInfo.stageCount = 1;
        rayPipelineInfo.pStages = &rayGenerationStage;
        rayPipelineInfo.groupCount = 1;
        rayPipelineInfo.pGroups = &rayGenerationGroup;
        rayPipelineInfo.maxPipelineRayRecursionDepth = 3;
        rayPipelineInfo.layout = pipelineLayout;
        VkPipeline rayPipeline = VK_NULL_HANDLE;
        require(
            createRayTracingPipelines(
                device,
                VK_NULL_HANDLE,
                VK_NULL_HANDLE,
                1,
                &rayPipelineInfo,
                nullptr,
                &rayPipeline
            ),
            "vkCreateRayTracingPipelinesKHR(depth=3)"
        );
        VkPipelineLibraryCreateInfoKHR rayLibraryInfo{};
        rayLibraryInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR;
        rayLibraryInfo.libraryCount = 1;
        rayLibraryInfo.pLibraries = &rayPipeline;
        VkRayTracingPipelineCreateInfoKHR linkedRayPipelineInfo{};
        linkedRayPipelineInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
        linkedRayPipelineInfo.pLibraryInfo = &rayLibraryInfo;
        linkedRayPipelineInfo.maxPipelineRayRecursionDepth = 3;
        linkedRayPipelineInfo.layout = pipelineLayout;
        VkPipeline linkedRayPipeline = VK_NULL_HANDLE;
        require(
            createRayTracingPipelines(
                device,
                VK_NULL_HANDLE,
                VK_NULL_HANDLE,
                1,
                &linkedRayPipelineInfo,
                nullptr,
                &linkedRayPipeline
            ),
            "vkCreateRayTracingPipelinesKHR(link)"
        );
        const auto getRayTracingShaderGroupHandles =
            reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(
                vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR")
            );
        if (getRayTracingShaderGroupHandles == nullptr) {
            throw std::runtime_error("Vulkan loader did not expose vkGetRayTracingShaderGroupHandlesKHR");
        }
        std::uint8_t rayShaderHandle[32]{};
        require(
            getRayTracingShaderGroupHandles(
                device,
                linkedRayPipeline,
                0,
                1,
                sizeof(rayShaderHandle),
                rayShaderHandle
            ),
            "vkGetRayTracingShaderGroupHandlesKHR(linked group)"
        );
        std::cout << "VULKAN_RT_PIPELINE stages=1 groups=1 recursion=3 linked_groups=1 accepted=1\n";

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shader;
        stageInfo.pName = "main";
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        require(
            vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline),
            "vkCreateComputePipelines"
        );

        VkDescriptorPoolSize poolSizes[3]{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 1;
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        poolSizes[1].descriptorCount = 1;
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSizes[2].descriptorCount = 1;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        poolInfo.maxSets = 2;
        poolInfo.poolSizeCount = 3;
        poolInfo.pPoolSizes = poolSizes;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        require(vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool), "vkCreateDescriptorPool");

        VkDescriptorSetAllocateInfo descriptorAllocateInfo{};
        descriptorAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorAllocateInfo.descriptorPool = descriptorPool;
        descriptorAllocateInfo.descriptorSetCount = 1;
        descriptorAllocateInfo.pSetLayouts = &descriptorLayout;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        require(vkAllocateDescriptorSets(device, &descriptorAllocateInfo, &descriptorSet), "vkAllocateDescriptorSets");

        VkDescriptorBufferInfo descriptorBufferInfo{};
        descriptorBufferInfo.buffer = buffer;
        descriptorBufferInfo.offset = 0;
        descriptorBufferInfo.range = bufferSize;
        VkWriteDescriptorSet descriptorWrite{};
        descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrite.dstSet = descriptorSet;
        descriptorWrite.dstBinding = 0;
        descriptorWrite.descriptorCount = 1;
        descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrite.pBufferInfo = &descriptorBufferInfo;
        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);

        VkCommandPoolCreateInfo commandPoolInfo{};
        commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolInfo.queueFamilyIndex = 0;
        VkCommandPool commandPool = VK_NULL_HANDLE;
        require(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool), "vkCreateCommandPool");

        VkCommandBufferAllocateInfo commandAllocateInfo{};
        commandAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandAllocateInfo.commandPool = commandPool;
        commandAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandAllocateInfo.commandBufferCount = 1;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        require(vkAllocateCommandBuffers(device, &commandAllocateInfo, &commandBuffer), "vkAllocateCommandBuffers");

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        require(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout,
            0,
            1,
            &descriptorSet,
            0,
            nullptr
        );
        const std::uint32_t addend = 5;
        vkCmdPushConstants(
            commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(addend),
            &addend
        );
        vkCmdDispatch(commandBuffer, 4, 1, 1);
        require(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence = VK_NULL_HANDLE;
        require(vkCreateFence(device, &fenceInfo, nullptr, &fence), "vkCreateFence");

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        require(vkQueueSubmit(queue, 1, &submitInfo, fence), "vkQueueSubmit");
        require(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        require(vkGetFenceStatus(device, fence), "vkGetFenceStatus");

        mapped = nullptr;
        require(vkMapMemory(device, memory, 0, bufferSize, 0, &mapped), "vkMapMemory(output)");
        const std::uint32_t expected[] = {6, 7, 8, 9};
        if (std::memcmp(mapped, expected, sizeof(expected)) != 0) {
            throw std::runtime_error("Vulkan compute readback did not match [6,7,8,9]");
        }
        vkUnmapMemory(device, memory);

        std::cout << "VULKAN_COMPUTE input=[1,2,3,4] addend=5 output=[6,7,8,9]"
                  << " backend=Metal fence=signaled\n";

        mapped = nullptr;
        require(vkMapMemory(device, memory, 0, bufferSize, 0, &mapped),
                "vkMapMemory(texel input)");
        const std::uint32_t texelInput[] = {1, 2, 3, 4};
        std::memcpy(mapped, texelInput, sizeof(texelInput));
        vkUnmapMemory(device, memory);

        VkBufferViewCreateInfo texelViewInfo{};
        texelViewInfo.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
        texelViewInfo.buffer = buffer;
        texelViewInfo.format = VK_FORMAT_R32_UINT;
        texelViewInfo.offset = 0;
        texelViewInfo.range = bufferSize;
        VkBufferView texelView = VK_NULL_HANDLE;
        require(
            vkCreateBufferView(device, &texelViewInfo, nullptr, &texelView),
            "vkCreateBufferView(Metal texel)"
        );
        if (texelView == VK_NULL_HANDLE) {
            throw std::runtime_error("IMB returned a null Metal texel buffer view");
        }

        std::vector<std::uint32_t> texelShaderCode(
            (imb_texel_add_spv_len + sizeof(std::uint32_t) - 1)
                / sizeof(std::uint32_t)
        );
        std::memcpy(
            texelShaderCode.data(),
            imb_texel_add_spv,
            imb_texel_add_spv_len
        );
        VkShaderModuleCreateInfo texelShaderInfo{};
        texelShaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        texelShaderInfo.codeSize = imb_texel_add_spv_len;
        texelShaderInfo.pCode = texelShaderCode.data();
        VkShaderModule texelShader = VK_NULL_HANDLE;
        require(
            vkCreateShaderModule(device, &texelShaderInfo, nullptr, &texelShader),
            "vkCreateShaderModule(texel)"
        );

        VkDescriptorSetLayoutBinding texelLayoutBinding{};
        texelLayoutBinding.binding = 0;
        texelLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        texelLayoutBinding.descriptorCount = 1;
        texelLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo texelLayoutInfo{};
        texelLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        texelLayoutInfo.bindingCount = 1;
        texelLayoutInfo.pBindings = &texelLayoutBinding;
        VkDescriptorSetLayout texelDescriptorLayout = VK_NULL_HANDLE;
        require(
            vkCreateDescriptorSetLayout(
                device,
                &texelLayoutInfo,
                nullptr,
                &texelDescriptorLayout
            ),
            "vkCreateDescriptorSetLayout(texel)"
        );

        VkPushConstantRange texelPushRange{};
        texelPushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        texelPushRange.size = sizeof(std::uint32_t);
        VkPipelineLayoutCreateInfo texelPipelineLayoutInfo{};
        texelPipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        texelPipelineLayoutInfo.setLayoutCount = 1;
        texelPipelineLayoutInfo.pSetLayouts = &texelDescriptorLayout;
        texelPipelineLayoutInfo.pushConstantRangeCount = 1;
        texelPipelineLayoutInfo.pPushConstantRanges = &texelPushRange;
        VkPipelineLayout texelPipelineLayout = VK_NULL_HANDLE;
        require(
            vkCreatePipelineLayout(
                device,
                &texelPipelineLayoutInfo,
                nullptr,
                &texelPipelineLayout
            ),
            "vkCreatePipelineLayout(texel)"
        );

        VkPipelineShaderStageCreateInfo texelStageInfo{};
        texelStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        texelStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        texelStageInfo.module = texelShader;
        texelStageInfo.pName = "main";
        VkComputePipelineCreateInfo texelPipelineInfo{};
        texelPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        texelPipelineInfo.stage = texelStageInfo;
        texelPipelineInfo.layout = texelPipelineLayout;
        VkPipeline texelPipeline = VK_NULL_HANDLE;
        require(
            vkCreateComputePipelines(
                device,
                VK_NULL_HANDLE,
                1,
                &texelPipelineInfo,
                nullptr,
                &texelPipeline
            ),
            "vkCreateComputePipelines(texel)"
        );

        VkDescriptorPoolSize texelPoolSize{};
        texelPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        texelPoolSize.descriptorCount = 1;
        VkDescriptorPoolCreateInfo texelPoolInfo{};
        texelPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        texelPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        texelPoolInfo.maxSets = 1;
        texelPoolInfo.poolSizeCount = 1;
        texelPoolInfo.pPoolSizes = &texelPoolSize;
        VkDescriptorPool texelDescriptorPool = VK_NULL_HANDLE;
        require(
            vkCreateDescriptorPool(
                device,
                &texelPoolInfo,
                nullptr,
                &texelDescriptorPool
            ),
            "vkCreateDescriptorPool(texel)"
        );
        VkDescriptorSetAllocateInfo texelSetAllocateInfo{};
        texelSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        texelSetAllocateInfo.descriptorPool = texelDescriptorPool;
        texelSetAllocateInfo.descriptorSetCount = 1;
        texelSetAllocateInfo.pSetLayouts = &texelDescriptorLayout;
        VkDescriptorSet texelDescriptorSet = VK_NULL_HANDLE;
        require(
            vkAllocateDescriptorSets(
                device,
                &texelSetAllocateInfo,
                &texelDescriptorSet
            ),
            "vkAllocateDescriptorSets(texel)"
        );
        VkWriteDescriptorSet texelDescriptorWrite{};
        texelDescriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        texelDescriptorWrite.dstSet = texelDescriptorSet;
        texelDescriptorWrite.dstBinding = 0;
        texelDescriptorWrite.descriptorCount = 1;
        texelDescriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
        texelDescriptorWrite.pTexelBufferView = &texelView;
        vkUpdateDescriptorSets(device, 1, &texelDescriptorWrite, 0, nullptr);

        VkCommandBuffer texelCommand = VK_NULL_HANDLE;
        require(
            vkAllocateCommandBuffers(
                device,
                &commandAllocateInfo,
                &texelCommand
            ),
            "vkAllocateCommandBuffers(texel)"
        );
        require(
            vkBeginCommandBuffer(texelCommand, &beginInfo),
            "vkBeginCommandBuffer(texel)"
        );
        vkCmdBindPipeline(
            texelCommand,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            texelPipeline
        );
        vkCmdBindDescriptorSets(
            texelCommand,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            texelPipelineLayout,
            0,
            1,
            &texelDescriptorSet,
            0,
            nullptr
        );
        const std::uint32_t texelAddend = 7;
        vkCmdPushConstants(
            texelCommand,
            texelPipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            sizeof(texelAddend),
            &texelAddend
        );
        vkCmdDispatch(texelCommand, 4, 1, 1);
        require(vkEndCommandBuffer(texelCommand), "vkEndCommandBuffer(texel)");
        VkFence texelFence = VK_NULL_HANDLE;
        require(
            vkCreateFence(device, &fenceInfo, nullptr, &texelFence),
            "vkCreateFence(texel)"
        );
        VkSubmitInfo texelSubmit{};
        texelSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        texelSubmit.commandBufferCount = 1;
        texelSubmit.pCommandBuffers = &texelCommand;
        require(
            vkQueueSubmit(queue, 1, &texelSubmit, texelFence),
            "vkQueueSubmit(texel)"
        );
        require(
            vkWaitForFences(device, 1, &texelFence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences(texel)"
        );
        mapped = nullptr;
        require(vkMapMemory(device, memory, 0, bufferSize, 0, &mapped),
                "vkMapMemory(texel output)");
        const std::uint32_t texelExpected[] = {8, 9, 10, 11};
        if (std::memcmp(mapped, texelExpected, sizeof(texelExpected)) != 0) {
            throw std::runtime_error(
                "Vulkan Metal texel-buffer readback did not match [8,9,10,11]"
            );
        }
        vkUnmapMemory(device, memory);
        std::cout << "VULKAN_TEXEL_BUFFER format=R32_UINT output=[8,9,10,11]"
                  << " backend=Metal fence=signaled\n";

        vkDestroyFence(device, texelFence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &texelCommand);
        require(
            vkFreeDescriptorSets(
                device,
                texelDescriptorPool,
                1,
                &texelDescriptorSet
            ),
            "vkFreeDescriptorSets(texel)"
        );
        vkDestroyDescriptorPool(device, texelDescriptorPool, nullptr);
        vkDestroyPipeline(device, texelPipeline, nullptr);
        vkDestroyPipelineLayout(device, texelPipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, texelDescriptorLayout, nullptr);
        vkDestroyShaderModule(device, texelShader, nullptr);
        vkDestroyBufferView(device, texelView, nullptr);

        VkBufferCreateInfo transferBufferInfo{};
        transferBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        transferBufferInfo.size = 16;
        transferBufferInfo.usage =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        transferBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer transferBuffer = VK_NULL_HANDLE;
        require(
            vkCreateBuffer(device, &transferBufferInfo, nullptr, &transferBuffer),
            "vkCreateBuffer(ordered transfer)"
        );
        VkMemoryRequirements transferBufferRequirements{};
        vkGetBufferMemoryRequirements(device, transferBuffer, &transferBufferRequirements);
        VkMemoryAllocateInfo transferBufferMemoryInfo{};
        transferBufferMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        transferBufferMemoryInfo.allocationSize = transferBufferRequirements.size;
        transferBufferMemoryInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory transferBufferMemory = VK_NULL_HANDLE;
        require(
            vkAllocateMemory(
                device,
                &transferBufferMemoryInfo,
                nullptr,
                &transferBufferMemory
            ),
            "vkAllocateMemory(ordered transfer)"
        );
        require(
            vkBindBufferMemory(device, transferBuffer, transferBufferMemory, 0),
            "vkBindBufferMemory(ordered transfer)"
        );

        constexpr std::uint32_t transferImageWidth = 4;
        constexpr std::uint32_t transferImageHeight = 4;
        VkImageCreateInfo transferImageInfo{};
        transferImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        transferImageInfo.imageType = VK_IMAGE_TYPE_2D;
        transferImageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        transferImageInfo.extent = {transferImageWidth, transferImageHeight, 1};
        transferImageInfo.mipLevels = 1;
        transferImageInfo.arrayLayers = 1;
        transferImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        transferImageInfo.tiling = VK_IMAGE_TILING_LINEAR;
        transferImageInfo.usage =
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        transferImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        transferImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage transferSourceImage = VK_NULL_HANDLE;
        VkImage transferDestinationImage = VK_NULL_HANDLE;
        require(
            vkCreateImage(
                device,
                &transferImageInfo,
                nullptr,
                &transferSourceImage
            ),
            "vkCreateImage(ordered transfer source)"
        );
        require(
            vkCreateImage(
                device,
                &transferImageInfo,
                nullptr,
                &transferDestinationImage
            ),
            "vkCreateImage(ordered transfer destination)"
        );
        VkMemoryRequirements transferImageRequirements{};
        vkGetImageMemoryRequirements(
            device,
            transferSourceImage,
            &transferImageRequirements
        );
        VkMemoryAllocateInfo transferImageMemoryInfo{};
        transferImageMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        transferImageMemoryInfo.allocationSize = transferImageRequirements.size;
        transferImageMemoryInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory transferSourceMemory = VK_NULL_HANDLE;
        VkDeviceMemory transferDestinationMemory = VK_NULL_HANDLE;
        require(
            vkAllocateMemory(
                device,
                &transferImageMemoryInfo,
                nullptr,
                &transferSourceMemory
            ),
            "vkAllocateMemory(ordered transfer source)"
        );
        require(
            vkAllocateMemory(
                device,
                &transferImageMemoryInfo,
                nullptr,
                &transferDestinationMemory
            ),
            "vkAllocateMemory(ordered transfer destination)"
        );
        require(
            vkBindImageMemory(device, transferSourceImage, transferSourceMemory, 0),
            "vkBindImageMemory(ordered transfer source)"
        );
        require(
            vkBindImageMemory(
                device,
                transferDestinationImage,
                transferDestinationMemory,
                0
            ),
            "vkBindImageMemory(ordered transfer destination)"
        );

        VkCommandBuffer transferCommand = VK_NULL_HANDLE;
        require(
            vkAllocateCommandBuffers(
                device,
                &commandAllocateInfo,
                &transferCommand
            ),
            "vkAllocateCommandBuffers(ordered transfer)"
        );
        require(
            vkBeginCommandBuffer(transferCommand, &beginInfo),
            "vkBeginCommandBuffer(ordered transfer)"
        );
        vkCmdFillBuffer(
            transferCommand,
            transferBuffer,
            0,
            VK_WHOLE_SIZE,
            UINT32_C(0x11223344)
        );
        const std::uint32_t updatedWord = UINT32_C(0xaabbccdd);
        vkCmdUpdateBuffer(
            transferCommand,
            transferBuffer,
            sizeof(std::uint32_t),
            sizeof(updatedWord),
            &updatedWord
        );
        vkCmdFillBuffer(
            transferCommand,
            transferBuffer,
            sizeof(std::uint32_t) * 2,
            sizeof(std::uint32_t) * 2,
            UINT32_C(0x55667788)
        );
        VkImageSubresourceRange transferRange{};
        transferRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        transferRange.levelCount = 1;
        transferRange.layerCount = 1;
        VkClearColorValue transferClear{};
        transferClear.float32[0] = 0.25F;
        transferClear.float32[1] = 0.5F;
        transferClear.float32[2] = 0.75F;
        transferClear.float32[3] = 1.0F;
        vkCmdClearColorImage(
            transferCommand,
            transferSourceImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            &transferClear,
            1,
            &transferRange
        );
        VkImageCopy transferImageCopy{};
        transferImageCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        transferImageCopy.srcSubresource.layerCount = 1;
        transferImageCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        transferImageCopy.dstSubresource.layerCount = 1;
        transferImageCopy.extent = {
            transferImageWidth,
            transferImageHeight,
            1,
        };
        vkCmdCopyImage(
            transferCommand,
            transferSourceImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            transferDestinationImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &transferImageCopy
        );
        transferClear.float32[0] = 1.0F;
        transferClear.float32[1] = 0.0F;
        transferClear.float32[2] = 0.0F;
        vkCmdClearColorImage(
            transferCommand,
            transferSourceImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            &transferClear,
            1,
            &transferRange
        );
        require(
            vkEndCommandBuffer(transferCommand),
            "vkEndCommandBuffer(ordered transfer)"
        );
        VkFence transferFence = VK_NULL_HANDLE;
        require(
            vkCreateFence(device, &fenceInfo, nullptr, &transferFence),
            "vkCreateFence(ordered transfer)"
        );
        VkSubmitInfo transferSubmit{};
        transferSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        transferSubmit.commandBufferCount = 1;
        transferSubmit.pCommandBuffers = &transferCommand;
        require(
            vkQueueSubmit(queue, 1, &transferSubmit, transferFence),
            "vkQueueSubmit(ordered transfer)"
        );
        require(
            vkWaitForFences(device, 1, &transferFence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences(ordered transfer)"
        );

        void* transferMapped = nullptr;
        require(
            vkMapMemory(
                device,
                transferBufferMemory,
                0,
                transferBufferInfo.size,
                0,
                &transferMapped
            ),
            "vkMapMemory(ordered transfer buffer)"
        );
        const std::uint32_t expectedTransferWords[] = {
            UINT32_C(0x11223344),
            UINT32_C(0xaabbccdd),
            UINT32_C(0x55667788),
            UINT32_C(0x55667788),
        };
        if (std::memcmp(
                transferMapped,
                expectedTransferWords,
                sizeof(expectedTransferWords)
            ) != 0) {
            throw std::runtime_error("ordered Vulkan buffer fill/update mismatch");
        }
        vkUnmapMemory(device, transferBufferMemory);

        require(
            vkMapMemory(
                device,
                transferDestinationMemory,
                0,
                transferImageRequirements.size,
                0,
                &transferMapped
            ),
            "vkMapMemory(ordered transfer image)"
        );
        const std::uint8_t expectedTransferTexel[] = {64, 128, 191, 255};
        for (std::size_t pixel = 0;
             pixel < transferImageWidth * transferImageHeight;
             ++pixel) {
            if (std::memcmp(
                    static_cast<const std::uint8_t*>(transferMapped) + pixel * 4,
                    expectedTransferTexel,
                    sizeof(expectedTransferTexel)
                ) != 0) {
                throw std::runtime_error("ordered Vulkan image clear/copy mismatch");
            }
        }
        vkUnmapMemory(device, transferDestinationMemory);
        std::cout
            << "VULKAN_TRANSFER fill=passed update_order=passed"
            << " clear=passed image_copy=passed\n";

        vkDestroyFence(device, transferFence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &transferCommand);
        vkDestroyImage(device, transferDestinationImage, nullptr);
        vkDestroyImage(device, transferSourceImage, nullptr);
        vkFreeMemory(device, transferDestinationMemory, nullptr);
        vkFreeMemory(device, transferSourceMemory, nullptr);
        vkDestroyBuffer(device, transferBuffer, nullptr);
        vkFreeMemory(device, transferBufferMemory, nullptr);

        constexpr std::uint32_t mipSourceWidth = 8;
        constexpr std::uint32_t mipSourceHeight = 8;
        constexpr std::uint32_t mipLevelCount = 4;
        constexpr VkDeviceSize mipOffsets[mipLevelCount] = {0, 512, 640, 672};
        constexpr VkDeviceSize mipTransferBytes = 680;
        VkBufferCreateInfo mipBufferInfo{};
        mipBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        mipBufferInfo.size = mipTransferBytes;
        mipBufferInfo.usage =
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        mipBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer mipBuffer = VK_NULL_HANDLE;
        require(vkCreateBuffer(device, &mipBufferInfo, nullptr, &mipBuffer), "vkCreateBuffer(mip transfer)");
        VkMemoryRequirements mipBufferRequirements{};
        vkGetBufferMemoryRequirements(device, mipBuffer, &mipBufferRequirements);
        VkMemoryAllocateInfo mipBufferMemoryInfo{};
        mipBufferMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mipBufferMemoryInfo.allocationSize = mipBufferRequirements.size;
        mipBufferMemoryInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory mipBufferMemory = VK_NULL_HANDLE;
        require(
            vkAllocateMemory(device, &mipBufferMemoryInfo, nullptr, &mipBufferMemory),
            "vkAllocateMemory(mip transfer)"
        );
        require(
            vkBindBufferMemory(device, mipBuffer, mipBufferMemory, 0),
            "vkBindBufferMemory(mip transfer)"
        );
        void* mipMapped = nullptr;
        require(
            vkMapMemory(device, mipBufferMemory, 0, mipTransferBytes, 0, &mipMapped),
            "vkMapMemory(mip upload)"
        );
        std::memset(mipMapped, 0, static_cast<std::size_t>(mipTransferBytes));
        auto* mipWords = static_cast<std::uint16_t*>(mipMapped);
        for (std::size_t index = 0; index < 512 / sizeof(std::uint16_t); ++index) {
            mipWords[index] = UINT16_C(0x1234);
        }
        vkUnmapMemory(device, mipBufferMemory);

        VkImageCreateInfo mipImageInfo{};
        mipImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        mipImageInfo.imageType = VK_IMAGE_TYPE_2D;
        mipImageInfo.format = VK_FORMAT_R16G16B16A16_UNORM;
        mipImageInfo.extent = {mipSourceWidth, mipSourceHeight, 1};
        mipImageInfo.mipLevels = mipLevelCount;
        mipImageInfo.arrayLayers = 1;
        mipImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        mipImageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        mipImageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        mipImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        mipImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage mipImage = VK_NULL_HANDLE;
        require(vkCreateImage(device, &mipImageInfo, nullptr, &mipImage), "vkCreateImage(mip transfer)");
        VkMemoryRequirements mipImageRequirements{};
        vkGetImageMemoryRequirements(device, mipImage, &mipImageRequirements);
        VkMemoryAllocateInfo mipImageMemoryInfo{};
        mipImageMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mipImageMemoryInfo.allocationSize = mipImageRequirements.size;
        mipImageMemoryInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory mipImageMemory = VK_NULL_HANDLE;
        require(
            vkAllocateMemory(device, &mipImageMemoryInfo, nullptr, &mipImageMemory),
            "vkAllocateMemory(mip image)"
        );
        require(vkBindImageMemory(device, mipImage, mipImageMemory, 0), "vkBindImageMemory(mip image)");

        VkCommandBuffer mipCommand = VK_NULL_HANDLE;
        require(vkAllocateCommandBuffers(device, &commandAllocateInfo, &mipCommand), "vkAllocateCommandBuffers(mip)");
        require(vkBeginCommandBuffer(mipCommand, &beginInfo), "vkBeginCommandBuffer(mip)");
        VkBufferImageCopy mipUpload{};
        mipUpload.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        mipUpload.imageSubresource.layerCount = 1;
        mipUpload.imageExtent = {mipSourceWidth, mipSourceHeight, 1};
        vkCmdCopyBufferToImage(
            mipCommand,
            mipBuffer,
            mipImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &mipUpload
        );
        VkBufferImageCopy mipReadbacks[mipLevelCount]{};
        for (std::uint32_t level = 0; level < mipLevelCount; ++level) {
            mipReadbacks[level].bufferOffset = mipOffsets[level];
            mipReadbacks[level].imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            mipReadbacks[level].imageSubresource.mipLevel = level;
            mipReadbacks[level].imageSubresource.layerCount = 1;
            mipReadbacks[level].imageExtent = {
                std::max(1U, mipSourceWidth >> level),
                std::max(1U, mipSourceHeight >> level),
                1,
            };
        }
        vkCmdCopyImageToBuffer(
            mipCommand,
            mipImage,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            mipBuffer,
            mipLevelCount,
            mipReadbacks
        );
        require(vkEndCommandBuffer(mipCommand), "vkEndCommandBuffer(mip)");
        VkFence mipFence = VK_NULL_HANDLE;
        require(vkCreateFence(device, &fenceInfo, nullptr, &mipFence), "vkCreateFence(mip)");
        VkSubmitInfo mipSubmit{};
        mipSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        mipSubmit.commandBufferCount = 1;
        mipSubmit.pCommandBuffers = &mipCommand;
        require(vkQueueSubmit(queue, 1, &mipSubmit, mipFence), "vkQueueSubmit(mip)");
        require(
            vkWaitForFences(device, 1, &mipFence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences(mip)"
        );
        mipMapped = nullptr;
        require(
            vkMapMemory(device, mipBufferMemory, 0, mipTransferBytes, 0, &mipMapped),
            "vkMapMemory(mip readback)"
        );
        const auto* mipReadbackWords = static_cast<const std::uint16_t*>(mipMapped);
        for (std::size_t index = 0; index < mipTransferBytes / sizeof(std::uint16_t); ++index) {
            if (mipReadbackWords[index] != UINT16_C(0x1234)) {
                throw std::runtime_error("IMB RGBA16 image-to-buffer mip readback mismatch");
            }
        }
        vkUnmapMemory(device, mipBufferMemory);
        vkDestroyFence(device, mipFence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &mipCommand);
        vkDestroyImage(device, mipImage, nullptr);
        vkFreeMemory(device, mipImageMemory, nullptr);
        vkDestroyBuffer(device, mipBuffer, nullptr);
        vkFreeMemory(device, mipBufferMemory, nullptr);
        std::cout
            << "VULKAN_IMAGE_READBACK format=RGBA16 mips=4 bytes=680"
            << " buffer_to_image=passed image_to_buffer=passed\n";

        constexpr std::uint32_t imageWidth = 64;
        constexpr std::uint32_t imageHeight = 64;
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageInfo.extent = {imageWidth, imageHeight, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage colorImage = VK_NULL_HANDLE;
        require(vkCreateImage(device, &imageInfo, nullptr, &colorImage), "vkCreateImage(raster)");

        VkMemoryRequirements imageMemoryRequirements{};
        vkGetImageMemoryRequirements(device, colorImage, &imageMemoryRequirements);
        if (imageMemoryRequirements.size < imageWidth * imageHeight * 4
            || (imageMemoryRequirements.memoryTypeBits & (1U << memoryTypeIndex)) == 0) {
            throw std::runtime_error("unexpected IMB image memory requirements");
        }
        VkMemoryAllocateInfo imageMemoryInfo{};
        imageMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        imageMemoryInfo.allocationSize = imageMemoryRequirements.size;
        imageMemoryInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory imageMemory = VK_NULL_HANDLE;
        require(vkAllocateMemory(device, &imageMemoryInfo, nullptr, &imageMemory), "vkAllocateMemory(raster)");
        require(vkBindImageMemory(device, colorImage, imageMemory, 0), "vkBindImageMemory(raster)");

        VkImageViewCreateInfo imageViewInfo{};
        imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        imageViewInfo.image = colorImage;
        imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        imageViewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageViewInfo.subresourceRange.levelCount = 1;
        imageViewInfo.subresourceRange.layerCount = 1;
        VkImageView colorView = VK_NULL_HANDLE;
        require(vkCreateImageView(device, &imageViewInfo, nullptr, &colorView), "vkCreateImageView(raster)");

        VkAttachmentDescription attachment{};
        attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkAttachmentReference colorReference{};
        colorReference.attachment = 0;
        colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorReference;
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &attachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        VkRenderPass renderPass = VK_NULL_HANDLE;
        require(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass), "vkCreateRenderPass(raster)");

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = &colorView;
        framebufferInfo.width = imageWidth;
        framebufferInfo.height = imageHeight;
        framebufferInfo.layers = 1;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        require(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &framebuffer), "vkCreateFramebuffer(raster)");

        VkPipelineLayoutCreateInfo rasterLayoutInfo{};
        rasterLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        VkPipelineLayout rasterLayout = VK_NULL_HANDLE;
        require(vkCreatePipelineLayout(device, &rasterLayoutInfo, nullptr, &rasterLayout), "vkCreatePipelineLayout(raster)");

        const auto vertexSpirV = readSpirV("/usr/local/share/imb/triangle.vert.spv");
        const auto fragmentSpirV = readSpirV("/usr/local/share/imb/triangle.frag.spv");
        VkShaderModuleCreateInfo vertexShaderInfo{};
        vertexShaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        vertexShaderInfo.codeSize = vertexSpirV.size() * sizeof(std::uint32_t);
        vertexShaderInfo.pCode = vertexSpirV.data();
        VkShaderModule vertexShader = VK_NULL_HANDLE;
        require(vkCreateShaderModule(device, &vertexShaderInfo, nullptr, &vertexShader), "vkCreateShaderModule(vertex)");
        VkShaderModuleCreateInfo fragmentShaderInfo{};
        fragmentShaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        fragmentShaderInfo.codeSize = fragmentSpirV.size() * sizeof(std::uint32_t);
        fragmentShaderInfo.pCode = fragmentSpirV.data();
        VkShaderModule fragmentShader = VK_NULL_HANDLE;
        require(vkCreateShaderModule(device, &fragmentShaderInfo, nullptr, &fragmentShader), "vkCreateShaderModule(fragment)");

        VkPipelineShaderStageCreateInfo graphicsStages[2]{};
        graphicsStages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        graphicsStages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        graphicsStages[0].module = vertexShader;
        graphicsStages[0].pName = "main";
        graphicsStages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        graphicsStages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        graphicsStages[1].module = fragmentShader;
        graphicsStages[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkViewport viewport{};
        viewport.width = static_cast<float>(imageWidth);
        viewport.height = static_cast<float>(imageHeight);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        VkRect2D scissor{{0, 0}, {imageWidth, imageHeight}};
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo rasterization{};
        rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blendState{};
        blendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        blendState.attachmentCount = 1;
        blendState.pAttachments = &blendAttachment;
        VkGraphicsPipelineCreateInfo graphicsInfo{};
        graphicsInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        graphicsInfo.stageCount = 2;
        graphicsInfo.pStages = graphicsStages;
        graphicsInfo.pVertexInputState = &vertexInput;
        graphicsInfo.pInputAssemblyState = &inputAssembly;
        graphicsInfo.pViewportState = &viewportState;
        graphicsInfo.pRasterizationState = &rasterization;
        graphicsInfo.pMultisampleState = &multisample;
        graphicsInfo.pColorBlendState = &blendState;
        graphicsInfo.layout = rasterLayout;
        graphicsInfo.renderPass = renderPass;
        graphicsInfo.subpass = 0;
        VkPipeline graphicsPipeline = VK_NULL_HANDLE;
        require(
            vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsInfo, nullptr, &graphicsPipeline),
            "vkCreateGraphicsPipelines(raster)"
        );

        VkCommandBuffer rasterCommand = VK_NULL_HANDLE;
        require(vkAllocateCommandBuffers(device, &commandAllocateInfo, &rasterCommand), "vkAllocateCommandBuffers(raster)");
        require(vkBeginCommandBuffer(rasterCommand, &beginInfo), "vkBeginCommandBuffer(raster)");
        VkClearValue clearValue{};
        clearValue.color.float32[0] = 16.0f / 255.0f;
        clearValue.color.float32[1] = 24.0f / 255.0f;
        clearValue.color.float32[2] = 32.0f / 255.0f;
        clearValue.color.float32[3] = 1.0f;
        VkRenderPassBeginInfo renderBegin{};
        renderBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderBegin.renderPass = renderPass;
        renderBegin.framebuffer = framebuffer;
        renderBegin.renderArea.extent = {imageWidth, imageHeight};
        renderBegin.clearValueCount = 1;
        renderBegin.pClearValues = &clearValue;
        vkCmdBeginRenderPass(rasterCommand, &renderBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(rasterCommand, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vkCmdDraw(rasterCommand, 3, 1, 0, 0);
        vkCmdEndRenderPass(rasterCommand);
        require(vkEndCommandBuffer(rasterCommand), "vkEndCommandBuffer(raster)");

        VkFence rasterFence = VK_NULL_HANDLE;
        require(vkCreateFence(device, &fenceInfo, nullptr, &rasterFence), "vkCreateFence(raster)");
        VkSubmitInfo rasterSubmit{};
        rasterSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        rasterSubmit.commandBufferCount = 1;
        rasterSubmit.pCommandBuffers = &rasterCommand;
        require(vkQueueSubmit(queue, 1, &rasterSubmit, rasterFence), "vkQueueSubmit(raster)");
        require(vkWaitForFences(device, 1, &rasterFence, VK_TRUE, UINT64_MAX), "vkWaitForFences(raster)");

        VkImageSubresource subresource{};
        subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        VkSubresourceLayout imageLayout{};
        vkGetImageSubresourceLayout(device, colorImage, &subresource, &imageLayout);
        if (imageLayout.rowPitch < imageWidth * 4 || imageLayout.size < imageWidth * imageHeight * 4) {
            throw std::runtime_error("invalid IMB linear image layout");
        }
        mapped = nullptr;
        require(
            vkMapMemory(device, imageMemory, 0, imageMemoryRequirements.size, 0, &mapped),
            "vkMapMemory(raster output)"
        );
        const auto* rgba = static_cast<const std::uint8_t*>(mapped) + imageLayout.offset;
        const std::uint8_t clearBytes[4]{0x10, 0x18, 0x20, 0xff};
        if (std::memcmp(rgba, clearBytes, sizeof(clearBytes)) != 0) {
            throw std::runtime_error("Vulkan raster clear-color corner mismatch");
        }
        std::size_t changedPixels = 0;
        for (std::uint32_t y = 0; y < imageHeight; ++y) {
            const auto* row = rgba + static_cast<std::size_t>(y * imageLayout.rowPitch);
            for (std::uint32_t x = 0; x < imageWidth; ++x) {
                const auto* pixel = row + x * 4;
                if (pixel[3] != 0xff) throw std::runtime_error("Vulkan raster produced a transparent pixel");
                if (std::memcmp(pixel, clearBytes, sizeof(clearBytes)) != 0) ++changedPixels;
            }
        }
        if (changedPixels < 512 || changedPixels >= imageWidth * imageHeight) {
            throw std::runtime_error("Vulkan raster changed-pixel coverage is invalid");
        }
        if (imageOutput != nullptr) {
            writePPM(imageOutput, rgba, imageWidth, imageHeight, imageLayout.rowPitch);
        }
        vkUnmapMemory(device, imageMemory);
        std::cout << "VULKAN_RASTER triangle=" << imageWidth << 'x' << imageHeight
                  << " format=RGBA8 changed_pixels=" << changedPixels
                  << " backend=Metal fence=signaled";
        if (imageOutput != nullptr) std::cout << " image=\"" << imageOutput << '"';
        std::cout << '\n';

        const auto getBufferDeviceAddress = reinterpret_cast<PFN_vkGetBufferDeviceAddressKHR>(
            vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR")
        );
        const auto getAccelerationStructureBuildSizes =
            reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
                vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR")
            );
        const auto createAccelerationStructure = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
            vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR")
        );
        const auto destroyAccelerationStructure = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
            vkGetDeviceProcAddr(device, "vkDestroyAccelerationStructureKHR")
        );
        const auto getAccelerationStructureDeviceAddress =
            reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
                vkGetDeviceProcAddr(device, "vkGetAccelerationStructureDeviceAddressKHR")
            );
        const auto cmdBuildAccelerationStructures =
            reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
                vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR")
            );
        if (getBufferDeviceAddress == nullptr || getAccelerationStructureBuildSizes == nullptr
            || createAccelerationStructure == nullptr || destroyAccelerationStructure == nullptr
            || getAccelerationStructureDeviceAddress == nullptr
            || cmdBuildAccelerationStructures == nullptr) {
            throw std::runtime_error("Vulkan loader did not expose KHR acceleration-structure entry points");
        }

        struct AddressBuffer {
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceMemory memory = VK_NULL_HANDLE;
            VkDeviceSize allocationSize = 0;
            VkDeviceAddress address = 0;
        };
        const auto createAddressBuffer = [
            device,
            memoryTypeIndex,
            getBufferDeviceAddress
        ](VkDeviceSize size, VkBufferUsageFlags usage, const char* label) {
            AddressBuffer result{};
            VkBufferCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            createInfo.size = size;
            createInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            require(vkCreateBuffer(device, &createInfo, nullptr, &result.buffer), label);

            VkMemoryRequirements requirements{};
            vkGetBufferMemoryRequirements(device, result.buffer, &requirements);
            if ((requirements.memoryTypeBits & (1U << memoryTypeIndex)) == 0) {
                throw std::runtime_error(std::string(label) + " has no host-visible memory type");
            }
            VkMemoryAllocateFlagsInfo allocationFlags{};
            allocationFlags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
            allocationFlags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            VkMemoryAllocateInfo allocationInfo{};
            allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocationInfo.pNext = &allocationFlags;
            allocationInfo.allocationSize = requirements.size;
            allocationInfo.memoryTypeIndex = memoryTypeIndex;
            require(vkAllocateMemory(device, &allocationInfo, nullptr, &result.memory), label);
            result.allocationSize = requirements.size;
            require(vkBindBufferMemory(device, result.buffer, result.memory, 0), label);

            VkBufferDeviceAddressInfo addressInfo{};
            addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
            addressInfo.buffer = result.buffer;
            result.address = getBufferDeviceAddress(device, &addressInfo);
            if (result.address == 0) {
                throw std::runtime_error(std::string(label) + " returned a zero device address");
            }
            return result;
        };

        constexpr float triangleVertices[] = {
            -1.0f, -1.0f, 0.0f,
             1.0f, -1.0f, 0.0f,
             0.0f,  1.0f, 0.0f,
        };
        AddressBuffer blasVertices = createAddressBuffer(
            sizeof(triangleVertices),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            "create BLAS vertex buffer"
        );
        mapped = nullptr;
        require(
            vkMapMemory(device, blasVertices.memory, 0, sizeof(triangleVertices), 0, &mapped),
            "vkMapMemory(BLAS vertices)"
        );
        std::memcpy(mapped, triangleVertices, sizeof(triangleVertices));
        vkUnmapMemory(device, blasVertices.memory);

        VkAccelerationStructureGeometryTrianglesDataKHR triangleData{};
        triangleData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangleData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangleData.vertexData.deviceAddress = blasVertices.address;
        triangleData.vertexStride = 3 * sizeof(float);
        triangleData.maxVertex = 2;
        triangleData.indexType = VK_INDEX_TYPE_NONE_KHR;
        VkAccelerationStructureGeometryKHR blasGeometry{};
        blasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        blasGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        blasGeometry.geometry.triangles = triangleData;
        blasGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        VkAccelerationStructureBuildGeometryInfoKHR blasBuildInfo{};
        blasBuildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        blasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        blasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        blasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        blasBuildInfo.geometryCount = 1;
        blasBuildInfo.pGeometries = &blasGeometry;
        constexpr std::uint32_t blasPrimitiveCount = 1;
        VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
        blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        getAccelerationStructureBuildSizes(
            device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &blasBuildInfo,
            &blasPrimitiveCount,
            &blasSizes
        );
        if (blasSizes.accelerationStructureSize == 0 || blasSizes.buildScratchSize == 0) {
            throw std::runtime_error("IMB returned invalid BLAS build sizes");
        }

        AddressBuffer blasStorage = createAddressBuffer(
            blasSizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            "create BLAS storage buffer"
        );
        VkAccelerationStructureCreateInfoKHR blasCreateInfo{};
        blasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        blasCreateInfo.buffer = blasStorage.buffer;
        blasCreateInfo.size = blasSizes.accelerationStructureSize;
        blasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
        require(
            createAccelerationStructure(device, &blasCreateInfo, nullptr, &blas),
            "vkCreateAccelerationStructureKHR(BLAS)"
        );

        AddressBuffer blasScratch = createAddressBuffer(
            blasSizes.buildScratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            "create BLAS scratch buffer"
        );
        blasBuildInfo.dstAccelerationStructure = blas;
        blasBuildInfo.scratchData.deviceAddress = blasScratch.address;
        VkAccelerationStructureBuildRangeInfoKHR blasRange{};
        blasRange.primitiveCount = blasPrimitiveCount;
        const VkAccelerationStructureBuildRangeInfoKHR* blasRanges[] = {&blasRange};

        VkCommandBuffer blasCommand = VK_NULL_HANDLE;
        require(vkAllocateCommandBuffers(device, &commandAllocateInfo, &blasCommand),
                "vkAllocateCommandBuffers(BLAS)");
        require(vkBeginCommandBuffer(blasCommand, &beginInfo), "vkBeginCommandBuffer(BLAS)");
        cmdBuildAccelerationStructures(blasCommand, 1, &blasBuildInfo, blasRanges);
        require(vkEndCommandBuffer(blasCommand), "vkEndCommandBuffer(BLAS)");
        VkFence blasFence = VK_NULL_HANDLE;
        require(vkCreateFence(device, &fenceInfo, nullptr, &blasFence), "vkCreateFence(BLAS)");
        VkSubmitInfo blasSubmit{};
        blasSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        blasSubmit.commandBufferCount = 1;
        blasSubmit.pCommandBuffers = &blasCommand;
        require(vkQueueSubmit(queue, 1, &blasSubmit, blasFence), "vkQueueSubmit(BLAS)");
        require(
            vkWaitForFences(device, 1, &blasFence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences(BLAS)"
        );
        require(vkGetFenceStatus(device, blasFence), "vkGetFenceStatus(BLAS)");
        std::cout << "VULKAN_BLAS triangles=1 geometry=opaque backend=Metal fence=signaled\n";

        VkAccelerationStructureDeviceAddressInfoKHR blasAddressInfo{};
        blasAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        blasAddressInfo.accelerationStructure = blas;
        const VkDeviceAddress blasAddress = getAccelerationStructureDeviceAddress(device, &blasAddressInfo);
        if (blasAddress == 0) throw std::runtime_error("IMB returned a zero BLAS device address");

        VkAccelerationStructureInstanceKHR tlasInstance{};
        tlasInstance.transform.matrix[0][0] = 1.0f;
        tlasInstance.transform.matrix[1][1] = 1.0f;
        tlasInstance.transform.matrix[2][2] = 1.0f;
        tlasInstance.instanceCustomIndex = 7;
        tlasInstance.mask = 0xff;
        tlasInstance.instanceShaderBindingTableRecordOffset = 3;
        tlasInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        tlasInstance.accelerationStructureReference = blasAddress;
        AddressBuffer tlasInstances = createAddressBuffer(
            sizeof(tlasInstance),
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
            "create TLAS instance buffer"
        );
        mapped = nullptr;
        require(
            vkMapMemory(device, tlasInstances.memory, 0, sizeof(tlasInstance), 0, &mapped),
            "vkMapMemory(TLAS instance)"
        );
        std::memcpy(mapped, &tlasInstance, sizeof(tlasInstance));
        vkUnmapMemory(device, tlasInstances.memory);

        VkAccelerationStructureGeometryInstancesDataKHR instancesData{};
        instancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instancesData.arrayOfPointers = VK_FALSE;
        instancesData.data.deviceAddress = tlasInstances.address;
        VkAccelerationStructureGeometryKHR tlasGeometry{};
        tlasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        tlasGeometry.geometry.instances = instancesData;
        VkAccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{};
        tlasBuildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        tlasBuildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        tlasBuildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        tlasBuildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        tlasBuildInfo.geometryCount = 1;
        tlasBuildInfo.pGeometries = &tlasGeometry;
        constexpr std::uint32_t tlasPrimitiveCount = 1;
        VkAccelerationStructureBuildSizesInfoKHR tlasSizes{};
        tlasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        getAccelerationStructureBuildSizes(
            device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &tlasBuildInfo,
            &tlasPrimitiveCount,
            &tlasSizes
        );
        if (tlasSizes.accelerationStructureSize == 0 || tlasSizes.buildScratchSize == 0) {
            throw std::runtime_error("IMB returned invalid TLAS build sizes");
        }
        AddressBuffer tlasStorage = createAddressBuffer(
            tlasSizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            "create TLAS storage buffer"
        );
        VkAccelerationStructureCreateInfoKHR tlasCreateInfo{};
        tlasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        tlasCreateInfo.buffer = tlasStorage.buffer;
        tlasCreateInfo.size = tlasSizes.accelerationStructureSize;
        tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        VkAccelerationStructureKHR tlas = VK_NULL_HANDLE;
        require(
            createAccelerationStructure(device, &tlasCreateInfo, nullptr, &tlas),
            "vkCreateAccelerationStructureKHR(TLAS)"
        );
        AddressBuffer tlasScratch = createAddressBuffer(
            tlasSizes.buildScratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            "create TLAS scratch buffer"
        );
        tlasBuildInfo.dstAccelerationStructure = tlas;
        tlasBuildInfo.scratchData.deviceAddress = tlasScratch.address;
        VkAccelerationStructureBuildRangeInfoKHR tlasRange{};
        tlasRange.primitiveCount = tlasPrimitiveCount;
        const VkAccelerationStructureBuildRangeInfoKHR* tlasRanges[] = {&tlasRange};
        VkCommandBuffer tlasCommand = VK_NULL_HANDLE;
        require(vkAllocateCommandBuffers(device, &commandAllocateInfo, &tlasCommand),
                "vkAllocateCommandBuffers(TLAS)");
        require(vkBeginCommandBuffer(tlasCommand, &beginInfo), "vkBeginCommandBuffer(TLAS)");
        cmdBuildAccelerationStructures(tlasCommand, 1, &tlasBuildInfo, tlasRanges);
        require(vkEndCommandBuffer(tlasCommand), "vkEndCommandBuffer(TLAS)");
        VkFence tlasFence = VK_NULL_HANDLE;
        require(vkCreateFence(device, &fenceInfo, nullptr, &tlasFence), "vkCreateFence(TLAS)");
        VkSubmitInfo tlasSubmit{};
        tlasSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        tlasSubmit.commandBufferCount = 1;
        tlasSubmit.pCommandBuffers = &tlasCommand;
        require(vkQueueSubmit(queue, 1, &tlasSubmit, tlasFence), "vkQueueSubmit(TLAS)");
        require(
            vkWaitForFences(device, 1, &tlasFence, VK_TRUE, UINT64_MAX),
            "vkWaitForFences(TLAS)"
        );
        require(vkGetFenceStatus(device, tlasFence), "vkGetFenceStatus(TLAS)");
        std::cout << "VULKAN_TLAS instances=1 child_blas=1 backend=Metal fence=signaled\n";

        VkImageCreateInfo rayImageInfo{};
        rayImageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        rayImageInfo.imageType = VK_IMAGE_TYPE_2D;
        rayImageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        rayImageInfo.extent = {imageWidth, imageHeight, 1};
        rayImageInfo.mipLevels = 1;
        rayImageInfo.arrayLayers = 1;
        rayImageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        rayImageInfo.tiling = VK_IMAGE_TILING_LINEAR;
        rayImageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        rayImageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        rayImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImage rayImage = VK_NULL_HANDLE;
        require(vkCreateImage(device, &rayImageInfo, nullptr, &rayImage), "vkCreateImage(ray trace)");
        VkMemoryRequirements rayImageRequirements{};
        vkGetImageMemoryRequirements(device, rayImage, &rayImageRequirements);
        VkMemoryAllocateInfo rayImageMemoryInfo{};
        rayImageMemoryInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        rayImageMemoryInfo.allocationSize = rayImageRequirements.size;
        rayImageMemoryInfo.memoryTypeIndex = memoryTypeIndex;
        VkDeviceMemory rayImageMemory = VK_NULL_HANDLE;
        require(vkAllocateMemory(device, &rayImageMemoryInfo, nullptr, &rayImageMemory),
                "vkAllocateMemory(ray trace)");
        require(vkBindImageMemory(device, rayImage, rayImageMemory, 0), "vkBindImageMemory(ray trace)");
        VkImageViewCreateInfo rayImageViewInfo{};
        rayImageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        rayImageViewInfo.image = rayImage;
        rayImageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        rayImageViewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        rayImageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        rayImageViewInfo.subresourceRange.levelCount = 1;
        rayImageViewInfo.subresourceRange.layerCount = 1;
        VkImageView rayImageView = VK_NULL_HANDLE;
        require(vkCreateImageView(device, &rayImageViewInfo, nullptr, &rayImageView),
                "vkCreateImageView(ray trace)");

        VkDescriptorSet rayDescriptorSet = VK_NULL_HANDLE;
        require(vkAllocateDescriptorSets(device, &descriptorAllocateInfo, &rayDescriptorSet),
                "vkAllocateDescriptorSets(ray trace)");
        VkWriteDescriptorSetAccelerationStructureKHR rayAccelerationWrite{};
        rayAccelerationWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        rayAccelerationWrite.accelerationStructureCount = 1;
        rayAccelerationWrite.pAccelerationStructures = &tlas;
        VkDescriptorImageInfo rayDescriptorImage{};
        rayDescriptorImage.imageView = rayImageView;
        rayDescriptorImage.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet rayDescriptorWrites[2]{};
        rayDescriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        rayDescriptorWrites[0].pNext = &rayAccelerationWrite;
        rayDescriptorWrites[0].dstSet = rayDescriptorSet;
        rayDescriptorWrites[0].dstBinding = 1;
        rayDescriptorWrites[0].descriptorCount = 1;
        rayDescriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        rayDescriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        rayDescriptorWrites[1].dstSet = rayDescriptorSet;
        rayDescriptorWrites[1].dstBinding = 2;
        rayDescriptorWrites[1].descriptorCount = 1;
        rayDescriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        rayDescriptorWrites[1].pImageInfo = &rayDescriptorImage;
        vkUpdateDescriptorSets(device, 2, rayDescriptorWrites, 0, nullptr);

        AddressBuffer shaderBindingTable = createAddressBuffer(
            sizeof(rayShaderHandle),
            VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
            "create ray shader binding table"
        );
        mapped = nullptr;
        require(vkMapMemory(device, shaderBindingTable.memory, 0, sizeof(rayShaderHandle), 0, &mapped),
                "vkMapMemory(ray shader binding table)");
        std::memcpy(mapped, rayShaderHandle, sizeof(rayShaderHandle));
        vkUnmapMemory(device, shaderBindingTable.memory);

        const auto cmdTraceRays = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(
            vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR")
        );
        if (cmdTraceRays == nullptr) throw std::runtime_error("Vulkan loader did not expose vkCmdTraceRaysKHR");
        VkCommandBuffer rayCommand = VK_NULL_HANDLE;
        require(vkAllocateCommandBuffers(device, &commandAllocateInfo, &rayCommand),
                "vkAllocateCommandBuffers(ray trace)");
        require(vkBeginCommandBuffer(rayCommand, &beginInfo), "vkBeginCommandBuffer(ray trace)");
        vkCmdBindPipeline(rayCommand, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, linkedRayPipeline);
        vkCmdBindDescriptorSets(
            rayCommand,
            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
            pipelineLayout,
            0,
            1,
            &rayDescriptorSet,
            0,
            nullptr
        );
        VkStridedDeviceAddressRegionKHR raygenRegion{};
        raygenRegion.deviceAddress = shaderBindingTable.address;
        raygenRegion.stride = sizeof(rayShaderHandle);
        raygenRegion.size = sizeof(rayShaderHandle);
        VkStridedDeviceAddressRegionKHR emptyRegion{};
        cmdTraceRays(
            rayCommand,
            &raygenRegion,
            &emptyRegion,
            &emptyRegion,
            &emptyRegion,
            imageWidth,
            imageHeight,
            1
        );
        require(vkEndCommandBuffer(rayCommand), "vkEndCommandBuffer(ray trace)");
        VkFence rayFence = VK_NULL_HANDLE;
        require(vkCreateFence(device, &fenceInfo, nullptr, &rayFence), "vkCreateFence(ray trace)");
        VkSubmitInfo raySubmit{};
        raySubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        raySubmit.commandBufferCount = 1;
        raySubmit.pCommandBuffers = &rayCommand;
        require(vkQueueSubmit(queue, 1, &raySubmit, rayFence), "vkQueueSubmit(ray trace)");
        require(vkWaitForFences(device, 1, &rayFence, VK_TRUE, UINT64_MAX),
                "vkWaitForFences(ray trace)");
        VkImageSubresource raySubresource{};
        raySubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        VkSubresourceLayout rayImageLayout{};
        vkGetImageSubresourceLayout(device, rayImage, &raySubresource, &rayImageLayout);
        mapped = nullptr;
        require(vkMapMemory(device, rayImageMemory, 0, rayImageRequirements.size, 0, &mapped),
                "vkMapMemory(ray trace output)");
        const auto* rayPixels = static_cast<const std::uint8_t*>(mapped) + rayImageLayout.offset;
        const auto* rayCorner = rayPixels;
        const auto* rayCenter = rayPixels
            + static_cast<std::size_t>(imageHeight / 2) * rayImageLayout.rowPitch
            + static_cast<std::size_t>(imageWidth / 2) * 4;
        const std::uint8_t missColor[4]{0, 0, 0, 255};
        const std::uint8_t hitColor[4]{0, 255, 0, 255};
        if (std::memcmp(rayCorner, missColor, sizeof(missColor)) != 0
            || std::memcmp(rayCenter, hitColor, sizeof(hitColor)) != 0) {
            throw std::runtime_error("Metal ray dispatch did not produce distinct miss and triangle-hit pixels");
        }
        vkUnmapMemory(device, rayImageMemory);
        std::cout << "VULKAN_RAY_DISPATCH rays=" << imageWidth << 'x' << imageHeight
                  << " center=hit corner=miss backend=Metal fence=signaled\n";

        vkDestroyFence(device, rayFence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &rayCommand);
        vkDestroyBuffer(device, shaderBindingTable.buffer, nullptr);
        vkFreeMemory(device, shaderBindingTable.memory, nullptr);
        require(vkFreeDescriptorSets(device, descriptorPool, 1, &rayDescriptorSet),
                "vkFreeDescriptorSets(ray trace)");
        vkDestroyImageView(device, rayImageView, nullptr);
        vkDestroyImage(device, rayImage, nullptr);
        vkFreeMemory(device, rayImageMemory, nullptr);

        vkDestroyFence(device, tlasFence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &tlasCommand);
        destroyAccelerationStructure(device, tlas, nullptr);
        vkDestroyBuffer(device, tlasScratch.buffer, nullptr);
        vkFreeMemory(device, tlasScratch.memory, nullptr);
        vkDestroyBuffer(device, tlasStorage.buffer, nullptr);
        vkFreeMemory(device, tlasStorage.memory, nullptr);
        vkDestroyBuffer(device, tlasInstances.buffer, nullptr);
        vkFreeMemory(device, tlasInstances.memory, nullptr);

        vkDestroyFence(device, blasFence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &blasCommand);
        destroyAccelerationStructure(device, blas, nullptr);
        vkDestroyBuffer(device, blasScratch.buffer, nullptr);
        vkFreeMemory(device, blasScratch.memory, nullptr);
        vkDestroyBuffer(device, blasStorage.buffer, nullptr);
        vkFreeMemory(device, blasStorage.memory, nullptr);
        vkDestroyBuffer(device, blasVertices.buffer, nullptr);
        vkFreeMemory(device, blasVertices.memory, nullptr);

        vkDestroyFence(device, rasterFence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &rasterCommand);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyShaderModule(device, fragmentShader, nullptr);
        vkDestroyShaderModule(device, vertexShader, nullptr);
        vkDestroyPipelineLayout(device, rasterLayout, nullptr);
        vkDestroyFramebuffer(device, framebuffer, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyImageView(device, colorView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, imageMemory, nullptr);

        require(vkQueueWaitIdle(queue), "vkQueueWaitIdle");
        require(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
        vkDestroyFence(device, fence, nullptr);
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
        vkDestroyCommandPool(device, commandPool, nullptr);
        require(vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet), "vkFreeDescriptorSets");
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyPipeline(device, linkedRayPipeline, nullptr);
        vkDestroyPipeline(device, rayPipeline, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyShaderModule(device, shader, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
        vkDestroyBuffer(device, buffer, nullptr);
        vkFreeMemory(device, memory, nullptr);
        vkFreeMemory(device, importedMemory, nullptr);
        vkFreeMemory(device, exportedMemory, nullptr);

        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "imb-vulkan-probe: " << error.what() << '\n';
        return 1;
    }
}
