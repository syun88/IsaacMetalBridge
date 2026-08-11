#include "imb_protocol.h"
#include "imb_wire.hpp"
#include "add_u32_spv.h"
#include "triangle_vert_spv.h"
#include "triangle_frag_spv.h"

#include <vulkan/vk_icd.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <linux/vm_sockets.h>
#include <unistd.h>

namespace {

constexpr std::uint32_t kMaxRayRecursionDepth = 3;
constexpr std::uint32_t kAdvertisedQueueCount = 16;
constexpr std::size_t kShaderGroupHandleSize = 32;
// Vulkan's standard 2D sparse image block shapes represent 64 KiB. Apple M4
// exposes 16 KiB Metal sparse texture tiles, so one guest-visible Vulkan block
// is backed by a 2x2 group of Metal tiles.
constexpr VkDeviceSize kMetalSparseImageBlockBytes = 16384;
constexpr VkDeviceSize kSparseImageBlockBytes = 65536;
constexpr std::uint32_t kCameraStateMagic = UINT32_C(0x31434d49);
constexpr std::uint16_t kCameraStateLegacyVersion = 2;
constexpr std::uint16_t kSceneStatePreviousVersion = 3;
constexpr std::uint16_t kSceneStateLightingVersion = 4;
constexpr std::uint16_t kSceneStateGeometryVersion = 5;
constexpr std::uint16_t kSceneStateNormalsVersion = 6;
constexpr std::uint16_t kSceneStateTextureVersion = 7;
constexpr std::uint16_t kSceneStateMaterialVersion = 8;
constexpr std::uint16_t kSceneStateEmissionVersion = 9;
constexpr std::uint16_t kSceneStateParameterTextureVersion = 10;
constexpr std::uint16_t kSceneStateNormalTextureVersion = 11;
constexpr std::uint16_t kSceneStateSplitSequenceVersion = 12;
constexpr std::uint16_t kSceneStateDeduplicatedTextureVersion = 13;
constexpr std::uint16_t kSceneStateOpacityVersion = 14;
constexpr std::uint16_t kSceneStateLightArrayVersion = 15;
constexpr std::uint16_t kSceneStateAreaLightVersion = 16;
constexpr std::uint16_t kSceneStateCylinderLightVersion = 17;
constexpr std::uint16_t kSceneStateShapingVersion = 18;
constexpr std::uint16_t kSceneStateLightTextureVersion = 19;
constexpr std::uint16_t kSceneStateVersion = kSceneStateLightTextureVersion;
constexpr std::size_t kCameraStateSize = 96;
constexpr std::size_t kSceneStatePreviousHeaderSize = 100;
constexpr std::size_t kSceneStatePreviousFixedHeaderSize = 148;
constexpr std::size_t kSceneStateSplitSequenceHeaderSize = 156;
constexpr std::size_t kSceneStateHeaderSize = 160;
constexpr std::size_t kSceneLegacyLightRecordSize = 48;
constexpr std::size_t kSceneAreaLightRecordSize = 80;
constexpr std::size_t kSceneShapingLightRecordSize = 120;
constexpr std::size_t kSceneLightRecordSize = 152;
constexpr std::size_t kSceneMeshRecordSize = 88;
constexpr std::size_t kSceneMeshGeometryRecordSize = 96;
constexpr std::size_t kSceneMeshNormalsRecordSize = 100;
constexpr std::size_t kSceneMeshTextureRecordSize = 116;
constexpr std::size_t kSceneMeshMaterialRecordSize = 124;
constexpr std::size_t kSceneMeshEmissionRecordSize = 140;
constexpr std::size_t kSceneMeshParameterTextureRecordSize = 192;
constexpr std::size_t kSceneMeshNormalTextureRecordSize = 204;
constexpr std::size_t kSceneMeshDeduplicatedTextureRecordSize = 224;
constexpr std::size_t kSceneMeshOpacityRecordSize = 232;
constexpr std::size_t kSceneTextureRecordSize = 20;
constexpr std::uint32_t kMaxSceneMeshCount = 4096;
constexpr std::uint32_t kMaxSceneTextureCount = 126;
constexpr std::uint32_t kMaxSceneLegacyAdditionalLightCount = 13;
constexpr std::uint32_t kMaxSceneLightCount = 16;
constexpr std::uint32_t kMaxSceneLightTextureCount = 32;
constexpr std::size_t kMaxSceneLightTextureBytes = 16U * 1024U * 1024U;
constexpr std::uint32_t kMaxSceneMeshVertexCount = 16U * 1024U * 1024U;
constexpr std::uint32_t kMaxSceneMeshIndexCount = 48U * 1024U * 1024U;
constexpr std::size_t kMaxSceneStateBytes = 512U * 1024U * 1024U;

constexpr bool sceneMaterialFlagsValid(
    std::uint32_t flags,
    bool standardOpacity = false
) {
    // Versions through v13 reserve bit 7. v14 uses it for standard USD
    // opacity while bits 8-31 continue to carry packed RGB.
    return standardOpacity || (flags & 128u) == 0;
}

static_assert(sceneMaterialFlagsValid(UINT32_C(0x2525251b)));
static_assert(sceneMaterialFlagsValid(UINT32_C(0xffffff7f)));
static_assert(!sceneMaterialFlagsValid(UINT32_C(0x00000080)));
static_assert(sceneMaterialFlagsValid(UINT32_C(0x00000080), true));

struct BridgeAdditionalLight {
    std::uint32_t kind = 0;
    std::uint32_t schema = 0;
    std::array<float, 16> values{};
    std::uint64_t pathHash = 0;
    std::array<float, 3> shapingAxis{};
    float shapingConeAngleDegrees = 0.0f;
    float shapingConeSoftness = 0.0f;
    float shapingFocus = 0.0f;
    std::array<float, 3> shapingFocusTint{};
    std::uint32_t shapingFlags = IMB_TRACE_LIGHT_SHAPING_NONE;
    std::uint64_t emissionTextureIndex = UINT64_MAX;
    std::uint64_t iesTextureIndex = UINT64_MAX;
    float iesAngleScale = 0.0f;
    float iesMultiplier = 0.0f;
    std::uint32_t textureFlags = IMB_TRACE_LIGHT_TEXTURE_NONE;
    std::uint64_t emissionTextureResourceID = 0;
    std::uint64_t iesTextureResourceID = 0;
    std::uint32_t emissionTextureWidth = 0;
    std::uint32_t emissionTextureHeight = 0;
    std::shared_ptr<const std::vector<std::uint8_t>> emissionTextureRGBA8;
    std::uint32_t iesTextureWidth = 0;
    std::uint32_t iesTextureHeight = 0;
    std::shared_ptr<const std::vector<std::uint8_t>> iesTextureRGBA8;
};

struct BridgeRayCamera {
    std::uint64_t sequence = 0;
    std::array<float, 3> position{};
    std::array<float, 3> forward{};
    std::array<float, 3> up{};
    float verticalFOVRadians = 0.0f;
    float nearDistance = 0.0f;
    float farDistance = 0.0f;
    bool hasSphereLight = false;
    std::array<float, 3> sphereLightPosition{};
    std::array<float, 3> sphereLightColor{};
    float sphereLightIntensity = 0.0f;
    float sphereLightRadius = 0.0f;
    bool hasDistantLight = false;
    std::array<float, 3> distantLightDirection{};
    std::array<float, 3> distantLightColor{};
    float distantLightIntensity = 0.0f;
    float distantLightAngleDegrees = 0.0f;
    bool hasDomeLight = false;
    std::array<float, 3> domeLightColor{};
    float domeLightIntensity = 0.0f;
    bool completeLightList = false;
    std::vector<BridgeAdditionalLight> additionalLights;
};

struct BridgeSceneTextureMap {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channel = 0;
    std::shared_ptr<const std::vector<std::uint8_t>> rgba8;
};

struct BridgeSceneMesh {
    std::uint64_t pathHash = 0;
    std::uint32_t triangleCount = 0;
    std::uint32_t materialFlags = 0;
    std::array<float, 3> boundsMinimum{};
    std::array<float, 3> boundsMaximum{};
    std::array<float, 12> worldTransform{};
    std::vector<float> vertices;
    std::vector<std::uint32_t> indices;
    std::vector<float> cornerNormals;
    std::vector<float> cornerUVs;
    std::uint32_t textureWidth = 0;
    std::uint32_t textureHeight = 0;
    std::shared_ptr<const std::vector<std::uint8_t>> textureRGBA8;
    float roughness = 0.5f;
    float metallic = 0.0f;
    std::array<float, 3> emissionColor{};
    float emissionIntensity = 0.0f;
    float opacity = 1.0f;
    float opacityThreshold = 0.0f;
    bool opacityFromBaseAlpha = false;
    BridgeSceneTextureMap roughnessTexture;
    BridgeSceneTextureMap metallicTexture;
    BridgeSceneTextureMap emissionTexture;
    BridgeSceneTextureMap normalTexture;
};

struct BridgeSceneState {
    BridgeRayCamera camera{};
    std::vector<BridgeSceneMesh> meshes;
    bool hasMeshManifest = false;
    std::uint64_t meshSequence = 0;
};

struct BridgeSceneMetalMesh {
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t contentHash = 0;
    std::uint64_t pathHash = 0;
    std::uint64_t vertexBufferResourceID = 0;
    std::uint64_t indexBufferResourceID = 0;
    std::uint64_t textureResourceID = 0;
    std::uint64_t roughnessTextureResourceID = 0;
    std::uint64_t metallicTextureResourceID = 0;
    std::uint64_t emissionTextureResourceID = 0;
    std::uint64_t normalTextureResourceID = 0;
    std::uint64_t materialDescriptorResourceID = 0;
    std::uint64_t accelerationStructureResourceID = 0;
};

struct BridgeSceneMetalTexture {
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t contentHash = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::shared_ptr<const std::vector<std::uint8_t>> rgba8;
    std::uint64_t resourceID = 0;
    std::size_t referenceCount = 0;
};

std::uint16_t readCameraU16(const std::array<std::uint8_t, kCameraStateSize>& bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(bytes[offset])
        | static_cast<std::uint16_t>(bytes[offset + 1]) << 8;
}

std::uint32_t readCameraU32(const std::array<std::uint8_t, kCameraStateSize>& bytes, std::size_t offset) {
    std::uint32_t result = 0;
    for (std::size_t index = 0; index < 4; ++index) {
        result |= static_cast<std::uint32_t>(bytes[offset + index]) << (index * 8);
    }
    return result;
}

std::uint64_t readCameraU64(const std::array<std::uint8_t, kCameraStateSize>& bytes, std::size_t offset) {
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < 8; ++index) {
        result |= static_cast<std::uint64_t>(bytes[offset + index]) << (index * 8);
    }
    return result;
}

float readCameraFloat(const std::array<std::uint8_t, kCameraStateSize>& bytes, std::size_t offset) {
    return std::bit_cast<float>(readCameraU32(bytes, offset));
}

float vectorLengthSquared(const std::array<float, 3>& value) {
    return value[0] * value[0] + value[1] * value[1] + value[2] * value[2];
}

std::optional<BridgeSceneState> readLiveSceneState(bool includeMeshes = true) {
    const char* path = std::getenv("IMB_CAMERA_STATE_FILE");
    if (path == nullptr || path[0] == '\0') return std::nullopt;

    std::ifstream input(path, std::ios::binary);
    if (!input) return std::nullopt;
    std::vector<std::uint8_t> allBytes;
    if (includeMeshes) {
        allBytes.assign(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()
        );
    } else {
        // Camera/light state lives entirely in the fixed header. Current Full
        // payloads can be tens of MiB, so do not copy and decode all Mesh
        // geometry and textures for every ray dispatch merely to update the
        // camera uniforms.
        allBytes.resize(kSceneStateHeaderSize);
        input.read(
            reinterpret_cast<char*>(allBytes.data()),
            static_cast<std::streamsize>(allBytes.size())
        );
        allBytes.resize(static_cast<std::size_t>(input.gcount()));
        if (allBytes.size() >= kSceneStateHeaderSize) {
            const std::uint16_t headerVersion =
                static_cast<std::uint16_t>(allBytes[4])
                | static_cast<std::uint16_t>(allBytes[5]) << 8;
            if (headerVersion >= kSceneStateLightArrayVersion) {
                std::uint32_t additionalLightCount = 0;
                for (std::size_t index = 0; index < 4; ++index) {
                    additionalLightCount |=
                        static_cast<std::uint32_t>(allBytes[156 + index])
                        << (index * 8);
                }
                const std::uint32_t maximumLightCount =
                    headerVersion >= kSceneStateAreaLightVersion
                    ? kMaxSceneLightCount
                    : kMaxSceneLegacyAdditionalLightCount;
                if (additionalLightCount > maximumLightCount) {
                    return std::nullopt;
                }
                const std::size_t lightRecordSize =
                    headerVersion >= kSceneStateLightTextureVersion
                    ? kSceneLightRecordSize
                    : (headerVersion >= kSceneStateShapingVersion
                    ? kSceneShapingLightRecordSize
                    : (headerVersion >= kSceneStateAreaLightVersion
                        ? kSceneAreaLightRecordSize
                        : kSceneLegacyLightRecordSize));
                const std::size_t additionalBytes =
                    static_cast<std::size_t>(additionalLightCount)
                    * lightRecordSize;
                const std::size_t originalSize = allBytes.size();
                allBytes.resize(originalSize + additionalBytes);
                input.read(
                    reinterpret_cast<char*>(allBytes.data() + originalSize),
                    static_cast<std::streamsize>(additionalBytes)
                );
                allBytes.resize(
                    originalSize + static_cast<std::size_t>(input.gcount())
                );
                if (allBytes.size() != originalSize + additionalBytes) {
                    return std::nullopt;
                }
                if (headerVersion >= kSceneStateLightTextureVersion) {
                    std::array<std::uint8_t, sizeof(std::uint32_t)> countBytes{};
                    input.read(
                        reinterpret_cast<char*>(countBytes.data()),
                        static_cast<std::streamsize>(countBytes.size())
                    );
                    if (static_cast<std::size_t>(input.gcount())
                        != countBytes.size()) {
                        return std::nullopt;
                    }
                    std::uint32_t textureCount = 0;
                    for (std::size_t index = 0; index < countBytes.size(); ++index) {
                        textureCount |= static_cast<std::uint32_t>(countBytes[index])
                            << (index * 8);
                    }
                    if (textureCount > kMaxSceneLightTextureCount) {
                        return std::nullopt;
                    }
                    allBytes.insert(
                        allBytes.end(), countBytes.begin(), countBytes.end()
                    );
                    std::size_t texturePayloadBytes = 0;
                    for (std::uint32_t textureIndex = 0;
                         textureIndex < textureCount; ++textureIndex) {
                        std::array<std::uint8_t, kSceneTextureRecordSize> record{};
                        input.read(
                            reinterpret_cast<char*>(record.data()),
                            static_cast<std::streamsize>(record.size())
                        );
                        if (static_cast<std::size_t>(input.gcount())
                            != record.size()) {
                            return std::nullopt;
                        }
                        const auto recordU32 = [&record](std::size_t offset) {
                            std::uint32_t result = 0;
                            for (std::size_t index = 0; index < 4; ++index) {
                                result |= static_cast<std::uint32_t>(
                                    record[offset + index]
                                ) << (index * 8);
                            }
                            return result;
                        };
                        const std::uint32_t width = recordU32(8);
                        const std::uint32_t height = recordU32(12);
                        const std::uint32_t byteCount = recordU32(16);
                        const std::uint64_t expectedBytes =
                            static_cast<std::uint64_t>(width) * height * 4;
                        if (width == 0 || height == 0
                            || width > 512 || height > 512
                            || expectedBytes != byteCount
                            || byteCount > kMaxSceneLightTextureBytes
                                - texturePayloadBytes) {
                            return std::nullopt;
                        }
                        allBytes.insert(
                            allBytes.end(), record.begin(), record.end()
                        );
                        const std::size_t payloadOffset = allBytes.size();
                        allBytes.resize(payloadOffset + byteCount);
                        input.read(
                            reinterpret_cast<char*>(
                                allBytes.data() + payloadOffset
                            ),
                            static_cast<std::streamsize>(byteCount)
                        );
                        if (static_cast<std::size_t>(input.gcount()) != byteCount) {
                            return std::nullopt;
                        }
                        texturePayloadBytes += byteCount;
                    }
                }
            }
        }
    }
    if (allBytes.size() < kCameraStateSize) return std::nullopt;
    std::array<std::uint8_t, kCameraStateSize> bytes{};
    std::copy_n(allBytes.begin(), bytes.size(), bytes.begin());
    const std::uint16_t version = readCameraU16(bytes, 4);
    if (readCameraU32(bytes, 0) != kCameraStateMagic
        || !(version == kCameraStateLegacyVersion
            || version == kSceneStatePreviousVersion
            || (version >= kSceneStateLightingVersion
                && version <= kSceneStateVersion))
        || (readCameraU16(bytes, 6) & 1u) == 0) {
        return std::nullopt;
    }
    if ((version == kCameraStateLegacyVersion && allBytes.size() != kCameraStateSize)
        || (version == kSceneStatePreviousVersion
            && allBytes.size() < kSceneStatePreviousHeaderSize)
        || (version >= kSceneStateLightingVersion
            && version < kSceneStateSplitSequenceVersion
            && allBytes.size() < kSceneStatePreviousFixedHeaderSize)
        || (version >= kSceneStateSplitSequenceVersion
            && version < kSceneStateLightArrayVersion
            && allBytes.size() < kSceneStateSplitSequenceHeaderSize)
        || (version >= kSceneStateLightArrayVersion
            && version <= kSceneStateVersion
            && allBytes.size() < kSceneStateHeaderSize)) {
        return std::nullopt;
    }

    BridgeSceneState scene{};
    auto& camera = scene.camera;
    camera.sequence = readCameraU64(bytes, 8);
    for (std::size_t index = 0; index < 3; ++index) {
        camera.position[index] = readCameraFloat(bytes, 16 + index * 4);
        camera.forward[index] = readCameraFloat(bytes, 28 + index * 4);
        camera.up[index] = readCameraFloat(bytes, 40 + index * 4);
    }
    camera.verticalFOVRadians = readCameraFloat(bytes, 52);
    camera.nearDistance = readCameraFloat(bytes, 56);
    camera.farDistance = readCameraFloat(bytes, 60);
    const std::uint16_t stateFlags = readCameraU16(bytes, 6);
    camera.hasSphereLight = (stateFlags & 2u) != 0;
    for (std::size_t index = 0; index < 3; ++index) {
        camera.sphereLightPosition[index] = readCameraFloat(bytes, 64 + index * 4);
        camera.sphereLightColor[index] = readCameraFloat(bytes, 76 + index * 4);
    }
    camera.sphereLightIntensity = readCameraFloat(bytes, 88);
    camera.sphereLightRadius = readCameraFloat(bytes, 92);
    const auto finiteVector = [](const std::array<float, 3>& value) {
        return std::all_of(value.begin(), value.end(), [](float component) {
            return std::isfinite(component);
        });
    };
    if (camera.sequence == 0
        || !finiteVector(camera.position)
        || !finiteVector(camera.forward)
        || !finiteVector(camera.up)
        || !std::isfinite(camera.verticalFOVRadians)
        || !std::isfinite(camera.nearDistance)
        || !std::isfinite(camera.farDistance)
        || vectorLengthSquared(camera.forward) <= 0.000001f
        || vectorLengthSquared(camera.up) <= 0.000001f
        || camera.verticalFOVRadians <= 0.01f
        || camera.verticalFOVRadians >= 3.13f
        || camera.nearDistance <= 0.0f
        || camera.farDistance <= camera.nearDistance) {
        return std::nullopt;
    }
    if (camera.hasSphereLight
        && (!finiteVector(camera.sphereLightPosition)
            || !finiteVector(camera.sphereLightColor)
            || !std::isfinite(camera.sphereLightIntensity)
            || !std::isfinite(camera.sphereLightRadius)
            || std::any_of(
                camera.sphereLightColor.begin(),
                camera.sphereLightColor.end(),
                [](float component) { return component < 0.0f; }
            )
            || camera.sphereLightIntensity < 0.0f
            || camera.sphereLightRadius <= 0.0f)) {
        return std::nullopt;
    }

    const auto readAllU32 = [&allBytes](std::size_t offset) {
        std::uint32_t result = 0;
        for (std::size_t index = 0; index < 4; ++index) {
            result |= static_cast<std::uint32_t>(allBytes[offset + index])
                << (index * 8);
        }
        return result;
    };
    const auto readAllU64 = [&allBytes](std::size_t offset) {
        std::uint64_t result = 0;
        for (std::size_t index = 0; index < 8; ++index) {
            result |= static_cast<std::uint64_t>(allBytes[offset + index])
                << (index * 8);
        }
        return result;
    };
    const auto readAllFloat = [&readAllU32](std::size_t offset) {
        return std::bit_cast<float>(readAllU32(offset));
    };
    if (version >= kSceneStateLightingVersion
        && version <= kSceneStateVersion) {
        camera.hasDistantLight = (stateFlags & 8u) != 0;
        camera.hasDomeLight = (stateFlags & 16u) != 0;
        for (std::size_t index = 0; index < 3; ++index) {
            camera.distantLightDirection[index] = readAllFloat(96 + index * 4);
            camera.distantLightColor[index] = readAllFloat(108 + index * 4);
            camera.domeLightColor[index] = readAllFloat(128 + index * 4);
        }
        camera.distantLightIntensity = readAllFloat(120);
        camera.distantLightAngleDegrees = readAllFloat(124);
        camera.domeLightIntensity = readAllFloat(140);
        if (camera.hasDistantLight
            && (!finiteVector(camera.distantLightDirection)
                || !finiteVector(camera.distantLightColor)
                || vectorLengthSquared(camera.distantLightDirection) <= 0.000001f
                || !std::isfinite(camera.distantLightIntensity)
                || !std::isfinite(camera.distantLightAngleDegrees)
                || std::any_of(
                    camera.distantLightColor.begin(),
                    camera.distantLightColor.end(),
                    [](float component) { return component < 0.0f; }
                )
                || camera.distantLightIntensity < 0.0f
                || camera.distantLightAngleDegrees < 0.0f
                || camera.distantLightAngleDegrees > 180.0f)) {
            return std::nullopt;
        }
        if (camera.hasDomeLight
            && (!finiteVector(camera.domeLightColor)
                || !std::isfinite(camera.domeLightIntensity)
                || std::any_of(
                    camera.domeLightColor.begin(),
                    camera.domeLightColor.end(),
                    [](float component) { return component < 0.0f; }
                )
                || camera.domeLightIntensity < 0.0f)) {
            return std::nullopt;
        }
    }

    std::uint32_t additionalLightCount = 0;
    std::size_t additionalLightBytes = 0;
    std::size_t lightTextureBytes = 0;
    if (version >= kSceneStateLightArrayVersion) {
        camera.completeLightList = version >= kSceneStateAreaLightVersion;
        additionalLightCount = readAllU32(156);
        const std::uint32_t maximumLightCount = camera.completeLightList
            ? kMaxSceneLightCount
            : kMaxSceneLegacyAdditionalLightCount;
        if (additionalLightCount > maximumLightCount) {
            return std::nullopt;
        }
        const std::size_t lightRecordSize =
            version >= kSceneStateLightTextureVersion
            ? kSceneLightRecordSize
            : (version >= kSceneStateShapingVersion
            ? kSceneShapingLightRecordSize
            : (camera.completeLightList
                ? kSceneAreaLightRecordSize
                : kSceneLegacyLightRecordSize));
        additionalLightBytes =
            static_cast<std::size_t>(additionalLightCount)
            * lightRecordSize;
        if (kSceneStateHeaderSize > allBytes.size()
            || additionalLightBytes
                > allBytes.size() - kSceneStateHeaderSize) {
            return std::nullopt;
        }
        try {
            camera.additionalLights.reserve(additionalLightCount);
        } catch (const std::bad_alloc&) {
            return std::nullopt;
        }
        for (std::uint32_t lightIndex = 0;
             lightIndex < additionalLightCount; ++lightIndex) {
            const std::size_t lightOffset = kSceneStateHeaderSize
                + static_cast<std::size_t>(lightIndex)
                    * lightRecordSize;
            BridgeAdditionalLight light{};
            light.kind = readAllU32(lightOffset);
            light.schema = readAllU32(lightOffset + 4);
            const std::size_t valueCount = camera.completeLightList ? 16 : 8;
            for (std::size_t valueIndex = 0; valueIndex < valueCount; ++valueIndex) {
                light.values[valueIndex] = readAllFloat(
                    lightOffset + 8 + valueIndex * 4
                );
            }
            light.pathHash = readAllU64(
                lightOffset + (camera.completeLightList ? 72 : 40)
            );
            if (version >= kSceneStateShapingVersion) {
                for (std::size_t index = 0; index < 3; ++index) {
                    light.shapingAxis[index] = readAllFloat(
                        lightOffset + 80 + index * 4
                    );
                }
                light.shapingConeAngleDegrees = readAllFloat(lightOffset + 92);
                light.shapingConeSoftness = readAllFloat(lightOffset + 96);
                light.shapingFocus = readAllFloat(lightOffset + 100);
                for (std::size_t index = 0; index < 3; ++index) {
                    light.shapingFocusTint[index] = readAllFloat(
                        lightOffset + 104 + index * 4
                    );
                }
                light.shapingFlags = readAllU32(lightOffset + 116);
            }
            if (version >= kSceneStateLightTextureVersion) {
                light.emissionTextureIndex = readAllU64(lightOffset + 120);
                light.iesTextureIndex = readAllU64(lightOffset + 128);
                light.iesAngleScale = readAllFloat(lightOffset + 136);
                light.iesMultiplier = readAllFloat(lightOffset + 140);
                light.textureFlags = readAllU32(lightOffset + 144);
                if (readAllU32(lightOffset + 148) != 0) {
                    return std::nullopt;
                }
            }
            if (light.pathHash == 0
                || !std::all_of(
                    light.values.begin(),
                    light.values.end(),
                    [](float value) { return std::isfinite(value); }
                )
                || !finiteVector(light.shapingAxis)
                || !std::isfinite(light.shapingConeAngleDegrees)
                || !std::isfinite(light.shapingConeSoftness)
                || !std::isfinite(light.shapingFocus)
                || !finiteVector(light.shapingFocusTint)
                || !std::isfinite(light.iesAngleScale)
                || !std::isfinite(light.iesMultiplier)
                || (light.shapingFlags
                    & ~static_cast<std::uint32_t>(
                        IMB_TRACE_LIGHT_SHAPING_APPLIED
                    )) != 0
                || (light.textureFlags
                    & ~(static_cast<std::uint32_t>(
                            IMB_TRACE_LIGHT_TEXTURE_RECT_EMISSION
                        )
                        | static_cast<std::uint32_t>(
                            IMB_TRACE_LIGHT_TEXTURE_IES_PROFILE
                        )
                        | static_cast<std::uint32_t>(
                            IMB_TRACE_LIGHT_TEXTURE_IES_NORMALIZED
                        ))) != 0) {
                return std::nullopt;
            }
            const bool hasShaping =
                (light.shapingFlags & IMB_TRACE_LIGHT_SHAPING_APPLIED) != 0;
            const bool hasEmissionTexture =
                (light.textureFlags
                    & IMB_TRACE_LIGHT_TEXTURE_RECT_EMISSION) != 0;
            const bool hasIES =
                (light.textureFlags & IMB_TRACE_LIGHT_TEXTURE_IES_PROFILE) != 0;
            const bool normalizedIES =
                (light.textureFlags
                    & IMB_TRACE_LIGHT_TEXTURE_IES_NORMALIZED) != 0;
            if ((hasEmissionTexture
                    ? light.emissionTextureIndex == UINT64_MAX
                    : light.emissionTextureIndex != UINT64_MAX)
                || (hasIES
                    ? light.iesTextureIndex == UINT64_MAX
                    : light.iesTextureIndex != UINT64_MAX)
                || (!hasIES && (normalizedIES
                    || light.iesAngleScale != 0.0f
                    || light.iesMultiplier != 0.0f))
                || (hasIES && (!hasShaping
                    || light.kind != IMB_TRACE_LIGHT_KIND_POSITIONAL
                    || light.iesMultiplier < 0.0f
                    || light.iesMultiplier > 1000000.0f))
                || (hasEmissionTexture
                    && (light.kind != IMB_TRACE_LIGHT_KIND_POSITIONAL
                        || light.schema != IMB_TRACE_LIGHT_SCHEMA_RECT))) {
                return std::nullopt;
            }
            if (hasShaping) {
                if (light.kind != IMB_TRACE_LIGHT_KIND_POSITIONAL
                    || vectorLengthSquared(light.shapingAxis) <= 0.000001f
                    || light.shapingConeAngleDegrees < 0.0f
                    || light.shapingConeAngleDegrees > 180.0f
                    || light.shapingConeSoftness < 0.0f
                    || light.shapingConeSoftness > 1.0f
                    || light.shapingFocus < 0.0f
                    || std::any_of(
                        light.shapingFocusTint.begin(),
                        light.shapingFocusTint.end(),
                        [](float value) { return value < 0.0f; }
                    )) {
                    return std::nullopt;
                }
            } else if (version >= kSceneStateShapingVersion
                && (std::any_of(
                        light.shapingAxis.begin(),
                        light.shapingAxis.end(),
                        [](float value) { return value != 0.0f; }
                    )
                    || light.shapingConeAngleDegrees != 0.0f
                    || light.shapingConeSoftness != 0.0f
                    || light.shapingFocus != 0.0f
                    || std::any_of(
                        light.shapingFocusTint.begin(),
                        light.shapingFocusTint.end(),
                        [](float value) { return value != 0.0f; }
                    ))) {
                return std::nullopt;
            }
            const auto nonnegativeColor = [&light](std::size_t offset) {
                return light.values[offset] >= 0.0f
                    && light.values[offset + 1] >= 0.0f
                    && light.values[offset + 2] >= 0.0f;
            };
            if (light.kind == IMB_TRACE_LIGHT_KIND_POSITIONAL) {
                if (light.schema < IMB_TRACE_LIGHT_SCHEMA_SPHERE
                    || (light.schema > IMB_TRACE_LIGHT_SCHEMA_DISK
                        && light.schema != IMB_TRACE_LIGHT_SCHEMA_CYLINDER)
                    || (light.schema == IMB_TRACE_LIGHT_SCHEMA_CYLINDER
                        && version < kSceneStateCylinderLightVersion)
                    || light.values[3] < 0.0f
                    || !nonnegativeColor(4)
                    || (light.schema == IMB_TRACE_LIGHT_SCHEMA_CYLINDER
                        ? light.values[7] < 0.0f
                        : light.values[7] <= 0.0f)) {
                    return std::nullopt;
                }
                if (camera.completeLightList) {
                    const std::array<float, 3> axisU{
                        light.values[8], light.values[9], light.values[10]
                    };
                    const std::array<float, 3> axisV{
                        light.values[12], light.values[13], light.values[14]
                    };
                    if (light.schema == IMB_TRACE_LIGHT_SCHEMA_SPHERE) {
                        const bool hasIESBasis = hasIES
                            && vectorLengthSquared(axisU) > 0.000001f
                            && vectorLengthSquared(axisV) > 0.000001f
                            && light.values[11] == 0.0f
                            && light.values[15] == 0.0f;
                        const bool hasNoBasis = std::all_of(
                            light.values.begin() + 8,
                            light.values.end(),
                            [](float value) { return value == 0.0f; }
                        );
                        if (!(hasIESBasis || (!hasIES && hasNoBasis))) {
                            return std::nullopt;
                        }
                    } else {
                        const bool cylinder =
                            light.schema == IMB_TRACE_LIGHT_SCHEMA_CYLINDER;
                        if (vectorLengthSquared(axisU) <= 0.000001f
                            || vectorLengthSquared(axisV) <= 0.000001f
                            || light.values[11] <= 0.0f
                            || (cylinder
                                ? !((light.values[7] > 0.0f
                                        && light.values[15] > 0.0f)
                                    || (light.values[7] == 0.0f
                                        && light.values[15] == 0.0f))
                                : light.values[15] <= 0.0f)) {
                            return std::nullopt;
                        }
                    }
                }
            } else if (light.kind == IMB_TRACE_LIGHT_KIND_DISTANT) {
                const std::array<float, 3> direction{
                    light.values[0], light.values[1], light.values[2]
                };
                if (light.schema != IMB_TRACE_LIGHT_SCHEMA_DISTANT
                    || vectorLengthSquared(direction) <= 0.000001f
                    || light.values[3] < 0.0f
                    || !nonnegativeColor(4)
                    || light.values[7] < 0.0f
                    || light.values[7] >= 360.0f
                    || (camera.completeLightList
                        && std::any_of(
                            light.values.begin() + 8,
                            light.values.end(),
                            [](float value) { return value != 0.0f; }
                        ))) {
                    return std::nullopt;
                }
            } else if (light.kind == IMB_TRACE_LIGHT_KIND_DOME) {
                if (light.schema != IMB_TRACE_LIGHT_SCHEMA_DOME
                    || !nonnegativeColor(0)
                    || light.values[3] < 0.0f
                    || std::any_of(
                        light.values.begin() + 4,
                        light.values.end(),
                        [](float value) { return value != 0.0f; }
                    )) {
                    return std::nullopt;
                }
            } else {
                return std::nullopt;
            }
            camera.additionalLights.push_back(light);
        }
    }

    if (version >= kSceneStateLightTextureVersion) {
        std::size_t textureOffset = kSceneStateHeaderSize
            + additionalLightBytes;
        if (textureOffset > allBytes.size()
            || sizeof(std::uint32_t) > allBytes.size() - textureOffset) {
            return std::nullopt;
        }
        const std::size_t tableStart = textureOffset;
        const std::uint32_t textureCount = readAllU32(textureOffset);
        textureOffset += sizeof(std::uint32_t);
        if (textureCount > kMaxSceneLightTextureCount) {
            return std::nullopt;
        }
        std::vector<BridgeSceneTextureMap> lightTextures;
        std::size_t texturePayloadBytes = 0;
        try {
            lightTextures.reserve(textureCount);
            for (std::uint32_t textureIndex = 0;
                 textureIndex < textureCount; ++textureIndex) {
                if (textureOffset > allBytes.size()
                    || kSceneTextureRecordSize
                        > allBytes.size() - textureOffset) {
                    return std::nullopt;
                }
                const std::uint32_t width = readAllU32(textureOffset + 8);
                const std::uint32_t height = readAllU32(textureOffset + 12);
                const std::uint32_t byteCount = readAllU32(textureOffset + 16);
                const std::uint64_t expectedBytes =
                    static_cast<std::uint64_t>(width) * height * 4;
                textureOffset += kSceneTextureRecordSize;
                if (width == 0 || height == 0
                    || width > 512 || height > 512
                    || expectedBytes != byteCount
                    || byteCount > kMaxSceneLightTextureBytes
                        - texturePayloadBytes
                    || textureOffset > allBytes.size()
                    || byteCount > allBytes.size() - textureOffset) {
                    return std::nullopt;
                }
                BridgeSceneTextureMap texture{};
                texture.width = width;
                texture.height = height;
                texture.rgba8 = std::make_shared<
                    const std::vector<std::uint8_t>
                >(
                    allBytes.begin() + textureOffset,
                    allBytes.begin() + textureOffset + byteCount
                );
                lightTextures.push_back(std::move(texture));
                textureOffset += byteCount;
                texturePayloadBytes += byteCount;
            }
        } catch (const std::bad_alloc&) {
            return std::nullopt;
        }
        lightTextureBytes = textureOffset - tableStart;
        for (auto& light : camera.additionalLights) {
            if ((light.textureFlags
                    & IMB_TRACE_LIGHT_TEXTURE_RECT_EMISSION) != 0) {
                if (light.emissionTextureIndex >= lightTextures.size()) {
                    return std::nullopt;
                }
                const auto& texture = lightTextures[
                    static_cast<std::size_t>(light.emissionTextureIndex)
                ];
                light.emissionTextureWidth = texture.width;
                light.emissionTextureHeight = texture.height;
                light.emissionTextureRGBA8 = texture.rgba8;
            }
            if ((light.textureFlags
                    & IMB_TRACE_LIGHT_TEXTURE_IES_PROFILE) != 0) {
                if (light.iesTextureIndex >= lightTextures.size()) {
                    return std::nullopt;
                }
                const auto& texture = lightTextures[
                    static_cast<std::size_t>(light.iesTextureIndex)
                ];
                light.iesTextureWidth = texture.width;
                light.iesTextureHeight = texture.height;
                light.iesTextureRGBA8 = texture.rgba8;
            }
        }
    }

    if (version == kSceneStatePreviousVersion
        || (version >= kSceneStateLightingVersion
            && version <= kSceneStateVersion)) {
        const std::size_t sceneHeaderSize =
            version >= kSceneStateLightArrayVersion
            ? kSceneStateHeaderSize
            : (version >= kSceneStateSplitSequenceVersion
                ? kSceneStateSplitSequenceHeaderSize
            : (version >= kSceneStateLightingVersion
                ? kSceneStatePreviousFixedHeaderSize
                : kSceneStatePreviousHeaderSize));
        const std::size_t scenePayloadOffset =
            sceneHeaderSize + additionalLightBytes + lightTextureBytes;
        const std::size_t meshCountOffset =
            version >= kSceneStateSplitSequenceVersion
            ? 152
            : (version >= kSceneStateLightingVersion
                ? 144
                : kCameraStateSize);
        scene.meshSequence = version >= kSceneStateSplitSequenceVersion
            ? readAllU64(144)
            : camera.sequence;
        const std::uint32_t meshCount = readAllU32(meshCountOffset);
        const bool hasMeshGeometry = version >= kSceneStateGeometryVersion;
        const bool hasCornerNormals = version >= kSceneStateNormalsVersion;
        const bool hasFileTextures = version >= kSceneStateTextureVersion;
        const bool hasMaterialParameters = version >= kSceneStateMaterialVersion;
        const bool hasEmission = version >= kSceneStateEmissionVersion;
        const bool hasParameterTextures =
            version >= kSceneStateParameterTextureVersion;
        const bool hasNormalTextures = version >= kSceneStateNormalTextureVersion;
        const bool hasDeduplicatedTextures =
            version >= kSceneStateDeduplicatedTextureVersion;
        const bool hasStandardOpacity =
            version >= kSceneStateOpacityVersion;
        if (scene.meshSequence == 0 || meshCount > kMaxSceneMeshCount
            || allBytes.size() > kMaxSceneStateBytes) {
            return std::nullopt;
        }
        if (includeMeshes && !hasMeshGeometry
            && (meshCount > (std::numeric_limits<std::size_t>::max()
                    - scenePayloadOffset) / kSceneMeshRecordSize
                || allBytes.size() != scenePayloadOffset
                    + static_cast<std::size_t>(meshCount)
                        * kSceneMeshRecordSize)) {
            return std::nullopt;
        }
        scene.hasMeshManifest = (readCameraU16(bytes, 6) & 4u) != 0;
        if (!scene.hasMeshManifest && meshCount != 0) return std::nullopt;
        if (hasMeshGeometry && (readCameraU16(bytes, 6) & 32u) == 0) {
            return std::nullopt;
        }
        if (hasCornerNormals && (readCameraU16(bytes, 6) & 64u) == 0) {
            return std::nullopt;
        }
        if (hasFileTextures && (readCameraU16(bytes, 6) & 128u) == 0) {
            return std::nullopt;
        }
        if (hasMaterialParameters && (readCameraU16(bytes, 6) & 256u) == 0) {
            return std::nullopt;
        }
        if (hasEmission && (readCameraU16(bytes, 6) & 512u) == 0) {
            return std::nullopt;
        }
        if (hasParameterTextures && (readCameraU16(bytes, 6) & 1024u) == 0) {
            return std::nullopt;
        }
        if (hasNormalTextures && (readCameraU16(bytes, 6) & 2048u) == 0) {
            return std::nullopt;
        }
        if (!includeMeshes) return scene;
        try {
            scene.meshes.reserve(meshCount);
        } catch (const std::bad_alloc&) {
            return std::nullopt;
        }
        std::size_t recordOffset = scenePayloadOffset;
        std::vector<BridgeSceneTextureMap> sceneTextures;
        if (hasDeduplicatedTextures) {
            if (recordOffset > allBytes.size()
                || sizeof(std::uint32_t) > allBytes.size() - recordOffset) {
                return std::nullopt;
            }
            const std::uint32_t textureCount = readAllU32(recordOffset);
            recordOffset += sizeof(std::uint32_t);
            if (textureCount > kMaxSceneTextureCount) return std::nullopt;
            try {
                sceneTextures.reserve(textureCount);
                for (std::uint32_t textureIndex = 0;
                     textureIndex < textureCount; ++textureIndex) {
                    if (recordOffset > allBytes.size()
                        || kSceneTextureRecordSize
                            > allBytes.size() - recordOffset) {
                        return std::nullopt;
                    }
                    // The stable source-path hash occupies the first 8 bytes;
                    // dimensions and payload length follow it.
                    const std::uint32_t width = readAllU32(recordOffset + 8);
                    const std::uint32_t height = readAllU32(recordOffset + 12);
                    const std::uint32_t byteCount = readAllU32(recordOffset + 16);
                    const std::uint64_t expectedBytes =
                        static_cast<std::uint64_t>(width) * height * 4;
                    recordOffset += kSceneTextureRecordSize;
                    if (width == 0 || height == 0
                        || width > 4096 || height > 4096
                        || expectedBytes != byteCount
                        || recordOffset > allBytes.size()
                        || byteCount > allBytes.size() - recordOffset) {
                        return std::nullopt;
                    }
                    BridgeSceneTextureMap texture{};
                    texture.width = width;
                    texture.height = height;
                    texture.rgba8 = std::make_shared<
                        const std::vector<std::uint8_t>
                    >(
                        allBytes.begin() + recordOffset,
                        allBytes.begin() + recordOffset + byteCount
                    );
                    sceneTextures.push_back(std::move(texture));
                    recordOffset += byteCount;
                }
            } catch (const std::bad_alloc&) {
                return std::nullopt;
            }
        }
        for (std::uint32_t meshIndex = 0; meshIndex < meshCount; ++meshIndex) {
            const std::size_t recordSize = hasStandardOpacity
                ? kSceneMeshOpacityRecordSize
                : (hasDeduplicatedTextures
                ? kSceneMeshDeduplicatedTextureRecordSize
                : (hasNormalTextures
                ? kSceneMeshNormalTextureRecordSize
                : (hasParameterTextures
                ? kSceneMeshParameterTextureRecordSize
                : (hasEmission
                ? kSceneMeshEmissionRecordSize
                : (hasMaterialParameters
                ? kSceneMeshMaterialRecordSize
                : (hasFileTextures
                ? kSceneMeshTextureRecordSize
                : (hasCornerNormals
                    ? kSceneMeshNormalsRecordSize
                : (hasMeshGeometry
                    ? kSceneMeshGeometryRecordSize
                    : kSceneMeshRecordSize))))))));
            if (recordOffset > allBytes.size()
                || recordSize > allBytes.size() - recordOffset) {
                return std::nullopt;
            }
            BridgeSceneMesh mesh{};
            mesh.pathHash = readAllU64(recordOffset);
            mesh.triangleCount = readAllU32(recordOffset + 8);
            mesh.materialFlags = readAllU32(recordOffset + 12);
            // Bits 0-6 are legacy material capabilities, v14 adds standard
            // opacity in bit 7, and bits 8-31 carry packed RGB base color.
            if (!sceneMaterialFlagsValid(
                    mesh.materialFlags,
                    hasStandardOpacity
                )) {
                return std::nullopt;
            }
            if (!hasMaterialParameters && (mesh.materialFlags & 16u) != 0) {
                return std::nullopt;
            }
            if (!hasEmission && (mesh.materialFlags & 32u) != 0) {
                return std::nullopt;
            }
            for (std::size_t component = 0; component < 3; ++component) {
                mesh.boundsMinimum[component] =
                    readAllFloat(recordOffset + 16 + component * 4);
                mesh.boundsMaximum[component] =
                    readAllFloat(recordOffset + 28 + component * 4);
            }
            for (std::size_t component = 0; component < 12; ++component) {
                mesh.worldTransform[component] =
                    readAllFloat(recordOffset + 40 + component * 4);
            }
            if (hasMaterialParameters) {
                mesh.roughness = readAllFloat(recordOffset + 88);
                mesh.metallic = readAllFloat(recordOffset + 92);
                const bool meshHasMaterialParameters =
                    (mesh.materialFlags & 16u) != 0;
                if ((meshHasMaterialParameters
                        && (!std::isfinite(mesh.roughness)
                            || !std::isfinite(mesh.metallic)
                            || mesh.roughness < 0.0f || mesh.roughness > 1.0f
                            || mesh.metallic < 0.0f || mesh.metallic > 1.0f))
                    || (!meshHasMaterialParameters
                        && (mesh.roughness != 0.5f || mesh.metallic != 0.0f))) {
                    return std::nullopt;
                }
            }
            if (hasEmission) {
                for (std::size_t component = 0; component < 3; ++component) {
                    mesh.emissionColor[component] =
                        readAllFloat(recordOffset + 96 + component * 4);
                }
                mesh.emissionIntensity = readAllFloat(recordOffset + 108);
                const bool meshHasEmission = (mesh.materialFlags & 32u) != 0;
                const bool emissionValid = finiteVector(mesh.emissionColor)
                    && std::all_of(
                        mesh.emissionColor.begin(), mesh.emissionColor.end(),
                        [](float value) { return value >= 0.0f && value <= 1.0f; }
                    )
                    && std::isfinite(mesh.emissionIntensity)
                    && mesh.emissionIntensity >= 0.0f
                    && mesh.emissionIntensity <= 1000000.0f;
                if ((meshHasEmission && !emissionValid)
                    || (!meshHasEmission
                        && (mesh.emissionColor != std::array<float, 3>{}
                            || mesh.emissionIntensity != 0.0f))) {
                    return std::nullopt;
                }
            }
            if (hasStandardOpacity) {
                mesh.opacity = readAllFloat(recordOffset + 112);
                mesh.opacityThreshold = readAllFloat(recordOffset + 116);
                const bool meshHasStandardOpacity =
                    (mesh.materialFlags & 128u) != 0;
                const bool opacityValid = std::isfinite(mesh.opacity)
                    && std::isfinite(mesh.opacityThreshold)
                    && mesh.opacity >= 0.0f && mesh.opacity <= 1.0f
                    && mesh.opacityThreshold >= 0.0f
                    && mesh.opacityThreshold <= 1.0f;
                if ((meshHasStandardOpacity && !opacityValid)
                    || (!meshHasStandardOpacity
                        && (mesh.opacity != 1.0f
                            || mesh.opacityThreshold != 0.0f))) {
                    return std::nullopt;
                }
            }
            bool boundsOrdered = true;
            for (std::size_t component = 0; component < 3; ++component) {
                if (mesh.boundsMinimum[component] > mesh.boundsMaximum[component]) {
                    boundsOrdered = false;
                    break;
                }
            }
            if (mesh.pathHash == 0 || mesh.triangleCount == 0
                || !finiteVector(mesh.boundsMinimum)
                || !finiteVector(mesh.boundsMaximum)
                || !boundsOrdered
                || std::any_of(
                    mesh.worldTransform.begin(),
                    mesh.worldTransform.end(),
                    [](float component) { return !std::isfinite(component); }
                )) {
                return std::nullopt;
            }
            recordOffset += recordSize;
            if (hasMeshGeometry) {
                const std::size_t recordStart = recordOffset - recordSize;
                const std::size_t geometryFieldsOffset =
                    hasStandardOpacity ? 120 : 112;
                const std::uint32_t vertexCount = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset)
                    : readAllU32(recordOffset - (hasFileTextures
                        ? 28 : (hasCornerNormals ? 12 : 8)));
                const std::uint32_t indexCount = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 4)
                    : readAllU32(recordOffset - (hasFileTextures
                        ? 24 : (hasCornerNormals ? 8 : 4)));
                const std::uint32_t normalCount = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 8)
                    : (hasCornerNormals
                        ? readAllU32(recordOffset - (hasFileTextures ? 20 : 4))
                        : 0);
                const std::uint32_t uvCount = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 12)
                    : (hasFileTextures ? readAllU32(recordOffset - 16) : 0);
                const std::uint32_t textureWidth = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 16)
                    : (hasFileTextures ? readAllU32(recordOffset - 12) : 0);
                const std::uint32_t textureHeight = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 20)
                    : (hasFileTextures ? readAllU32(recordOffset - 8) : 0);
                const std::uint32_t textureByteCount = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 24)
                    : (hasFileTextures ? readAllU32(recordOffset - 4) : 0);
                const std::uint32_t textureFlags = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 28) : 0;
                const std::uint32_t roughnessTextureWidth = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 32) : 0;
                const std::uint32_t roughnessTextureHeight = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 36) : 0;
                const std::uint32_t roughnessTextureByteCount = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 40) : 0;
                const std::uint32_t roughnessTextureChannel = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 44) : 0;
                const std::uint32_t metallicTextureWidth = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 48) : 0;
                const std::uint32_t metallicTextureHeight = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 52) : 0;
                const std::uint32_t metallicTextureByteCount = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 56) : 0;
                const std::uint32_t metallicTextureChannel = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 60) : 0;
                const std::uint32_t emissionTextureWidth = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 64) : 0;
                const std::uint32_t emissionTextureHeight = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 68) : 0;
                const std::uint32_t emissionTextureByteCount = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 72) : 0;
                const std::uint32_t emissionTextureChannel = hasParameterTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 76) : 0;
                const std::uint32_t normalTextureWidth = hasNormalTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 80) : 0;
                const std::uint32_t normalTextureHeight = hasNormalTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 84) : 0;
                const std::uint32_t normalTextureByteCount = hasNormalTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 88) : 0;
                const std::uint32_t noSceneTexture =
                    std::numeric_limits<std::uint32_t>::max();
                const std::uint32_t baseTextureIndex = hasDeduplicatedTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 92) : noSceneTexture;
                const std::uint32_t roughnessTextureIndex = hasDeduplicatedTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 96) : noSceneTexture;
                const std::uint32_t metallicTextureIndex = hasDeduplicatedTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 100) : noSceneTexture;
                const std::uint32_t emissionTextureIndex = hasDeduplicatedTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 104) : noSceneTexture;
                const std::uint32_t normalTextureIndex = hasDeduplicatedTextures
                    ? readAllU32(recordStart + geometryFieldsOffset + 108) : noSceneTexture;
                const bool hasTextureUVs = uvCount != 0;
                const bool hasBaseTexture = hasDeduplicatedTextures
                    ? baseTextureIndex != noSceneTexture
                    : textureByteCount != 0;
                const bool hasRoughnessTexture = (textureFlags & 1u) != 0;
                const bool hasMetallicTexture = (textureFlags & 2u) != 0;
                const bool hasEmissionTexture = (textureFlags & 4u) != 0;
                const bool hasNormalTexture = (textureFlags & 8u) != 0;
                const bool hasOpacityTexture = (textureFlags & 16u) != 0;
                if (vertexCount == 0 || vertexCount > kMaxSceneMeshVertexCount
                    || indexCount == 0 || indexCount > kMaxSceneMeshIndexCount
                    || indexCount % 3 != 0
                    || indexCount / 3 != mesh.triangleCount
                    || (normalCount != 0 && normalCount != indexCount)
                    || (hasTextureUVs && uvCount != indexCount)
                    || (!hasTextureUVs && (hasBaseTexture
                        || hasRoughnessTexture || hasMetallicTexture
                        || hasEmissionTexture || hasNormalTexture
                        || hasOpacityTexture))
                    || (textureFlags & ~(hasStandardOpacity ? 31u : 15u)) != 0
                    || (hasOpacityTexture
                        && ((mesh.materialFlags & 128u) == 0
                            || !hasBaseTexture))) {
                    return std::nullopt;
                }
                const std::uint64_t expectedTextureBytes =
                    static_cast<std::uint64_t>(textureWidth)
                    * textureHeight * 4;
                const auto deduplicatedTextureValid = [
                    &sceneTextures, noSceneTexture
                ](
                    bool present,
                    std::uint32_t textureIndex,
                    std::uint32_t width,
                    std::uint32_t height,
                    std::uint32_t byteCount
                ) {
                    if (!present) {
                        return textureIndex == noSceneTexture
                            && width == 0 && height == 0 && byteCount == 0;
                    }
                    return textureIndex < sceneTextures.size()
                        && byteCount == 0
                        && width == sceneTextures[textureIndex].width
                        && height == sceneTextures[textureIndex].height;
                };
                const bool baseTextureValid = hasDeduplicatedTextures
                    ? deduplicatedTextureValid(
                        hasBaseTexture, baseTextureIndex,
                        textureWidth, textureHeight, textureByteCount
                    )
                    : (hasBaseTexture
                        ? textureWidth > 0 && textureHeight > 0
                            && textureWidth <= 4096 && textureHeight <= 4096
                            && expectedTextureBytes == textureByteCount
                        : textureWidth == 0 && textureHeight == 0
                            && textureByteCount == 0);
                if (!baseTextureValid
                    || (hasBaseTexture && (mesh.materialFlags & 8u) == 0)
                    || (!hasBaseTexture && (mesh.materialFlags & 8u) != 0)
                    || ((mesh.materialFlags & 64u) != 0
                        && !hasBaseTexture)) {
                    return std::nullopt;
                }
                const auto parameterTextureValid = [
                    &deduplicatedTextureValid,
                    hasDeduplicatedTextures,
                    noSceneTexture
                ](
                    bool present,
                    std::uint32_t textureIndex,
                    std::uint32_t width,
                    std::uint32_t height,
                    std::uint32_t byteCount,
                    std::uint32_t channel,
                    bool colorOutput
                ) {
                    if (hasDeduplicatedTextures) {
                        return deduplicatedTextureValid(
                            present, textureIndex, width, height, byteCount
                        ) && (present
                            ? (colorOutput ? channel == 4u : channel <= 3u)
                            : channel == 0u);
                    }
                    if (!present) {
                        return textureIndex == noSceneTexture
                            && width == 0 && height == 0 && byteCount == 0
                            && channel == 0;
                    }
                    const std::uint64_t expectedBytes =
                        static_cast<std::uint64_t>(width) * height * 4;
                    return width > 0 && height > 0
                        && width <= 4096 && height <= 4096
                        && expectedBytes == byteCount
                        && (colorOutput ? channel == 4u : channel <= 3u);
                };
                if (!parameterTextureValid(
                        hasRoughnessTexture,
                        roughnessTextureIndex,
                        roughnessTextureWidth,
                        roughnessTextureHeight,
                        roughnessTextureByteCount,
                        roughnessTextureChannel,
                        false
                    )
                    || !parameterTextureValid(
                        hasMetallicTexture,
                        metallicTextureIndex,
                        metallicTextureWidth,
                        metallicTextureHeight,
                        metallicTextureByteCount,
                        metallicTextureChannel,
                        false
                    )
                    || !parameterTextureValid(
                        hasEmissionTexture,
                        emissionTextureIndex,
                        emissionTextureWidth,
                        emissionTextureHeight,
                        emissionTextureByteCount,
                        emissionTextureChannel,
                        true
                    )
                    || !parameterTextureValid(
                        hasNormalTexture,
                        normalTextureIndex,
                        normalTextureWidth,
                        normalTextureHeight,
                        normalTextureByteCount,
                        hasNormalTexture ? 4u : 0u,
                        true
                    )) {
                    return std::nullopt;
                }
                mesh.opacityFromBaseAlpha = hasOpacityTexture;
                const std::size_t vertexValueCount =
                    static_cast<std::size_t>(vertexCount) * 3;
                const std::size_t vertexBytes = vertexValueCount * sizeof(float);
                const std::size_t indexBytes =
                    static_cast<std::size_t>(indexCount) * sizeof(std::uint32_t);
                const std::size_t normalValueCount =
                    static_cast<std::size_t>(normalCount) * 3;
                const std::size_t normalBytes =
                    normalValueCount * sizeof(float);
                const std::size_t uvValueCount =
                    static_cast<std::size_t>(uvCount) * 2;
                const std::size_t uvBytes = uvValueCount * sizeof(float);
                const std::size_t textureBytes = textureByteCount;
                const std::size_t roughnessTextureBytes =
                    roughnessTextureByteCount;
                const std::size_t metallicTextureBytes =
                    metallicTextureByteCount;
                const std::size_t emissionTextureBytes =
                    emissionTextureByteCount;
                const std::size_t normalTextureBytes = normalTextureByteCount;
                const std::uint64_t payloadByteCount =
                    static_cast<std::uint64_t>(vertexBytes) + indexBytes
                    + normalBytes + uvBytes + textureBytes
                    + roughnessTextureBytes + metallicTextureBytes
                    + emissionTextureBytes + normalTextureBytes;
                if (recordOffset > allBytes.size()
                    || payloadByteCount > allBytes.size() - recordOffset) {
                    return std::nullopt;
                }
                try {
                    mesh.vertices.reserve(vertexValueCount);
                    for (std::size_t valueIndex = 0;
                         valueIndex < vertexValueCount;
                         ++valueIndex) {
                        const float value = readAllFloat(
                            recordOffset + valueIndex * sizeof(float)
                        );
                        if (!std::isfinite(value)) return std::nullopt;
                        mesh.vertices.push_back(value);
                    }
                    const std::size_t indexOffset = recordOffset + vertexBytes;
                    mesh.indices.reserve(indexCount);
                    for (std::uint32_t indexIndex = 0;
                         indexIndex < indexCount;
                         ++indexIndex) {
                        const std::uint32_t vertexIndex = readAllU32(
                            indexOffset
                                + static_cast<std::size_t>(indexIndex)
                                    * sizeof(std::uint32_t)
                        );
                        if (vertexIndex >= vertexCount) return std::nullopt;
                        mesh.indices.push_back(vertexIndex);
                    }
                    const std::size_t normalOffset =
                        indexOffset + indexBytes;
                    mesh.cornerNormals.reserve(normalValueCount);
                    for (std::size_t valueIndex = 0;
                         valueIndex < normalValueCount;
                         ++valueIndex) {
                        const float value = readAllFloat(
                            normalOffset + valueIndex * sizeof(float)
                        );
                        if (!std::isfinite(value)) return std::nullopt;
                        mesh.cornerNormals.push_back(value);
                    }
                    for (std::uint32_t normalIndex = 0;
                         normalIndex < normalCount;
                         ++normalIndex) {
                        const std::size_t componentOffset =
                            static_cast<std::size_t>(normalIndex) * 3;
                        const float x = mesh.cornerNormals[componentOffset];
                        const float y = mesh.cornerNormals[componentOffset + 1];
                        const float z = mesh.cornerNormals[componentOffset + 2];
                        const float lengthSquared = x * x + y * y + z * z;
                        if (!std::isfinite(lengthSquared)
                            || lengthSquared <= 0.000000000001F) {
                            return std::nullopt;
                        }
                    }
                    const std::size_t uvOffset = normalOffset + normalBytes;
                    mesh.cornerUVs.reserve(uvValueCount);
                    for (std::size_t valueIndex = 0;
                         valueIndex < uvValueCount;
                         ++valueIndex) {
                        const float value = readAllFloat(
                            uvOffset + valueIndex * sizeof(float)
                        );
                        if (!std::isfinite(value)) return std::nullopt;
                        mesh.cornerUVs.push_back(value);
                    }
                    mesh.textureWidth = textureWidth;
                    mesh.textureHeight = textureHeight;
                    const std::size_t textureOffset = uvOffset + uvBytes;
                    mesh.textureRGBA8 = hasDeduplicatedTextures
                        && hasBaseTexture
                        ? sceneTextures[baseTextureIndex].rgba8
                        : std::make_shared<const std::vector<std::uint8_t>>(
                            allBytes.begin() + textureOffset,
                            allBytes.begin() + textureOffset + textureBytes
                        );
                    const std::size_t roughnessTextureOffset =
                        textureOffset + textureBytes;
                    mesh.roughnessTexture.width = roughnessTextureWidth;
                    mesh.roughnessTexture.height = roughnessTextureHeight;
                    mesh.roughnessTexture.channel = roughnessTextureChannel;
                    mesh.roughnessTexture.rgba8 = hasDeduplicatedTextures
                        && hasRoughnessTexture
                        ? sceneTextures[roughnessTextureIndex].rgba8
                        : std::make_shared<const std::vector<std::uint8_t>>(
                            allBytes.begin() + roughnessTextureOffset,
                            allBytes.begin() + roughnessTextureOffset
                                + roughnessTextureBytes
                        );
                    const std::size_t metallicTextureOffset =
                        roughnessTextureOffset + roughnessTextureBytes;
                    mesh.metallicTexture.width = metallicTextureWidth;
                    mesh.metallicTexture.height = metallicTextureHeight;
                    mesh.metallicTexture.channel = metallicTextureChannel;
                    mesh.metallicTexture.rgba8 = hasDeduplicatedTextures
                        && hasMetallicTexture
                        ? sceneTextures[metallicTextureIndex].rgba8
                        : std::make_shared<const std::vector<std::uint8_t>>(
                            allBytes.begin() + metallicTextureOffset,
                            allBytes.begin() + metallicTextureOffset
                                + metallicTextureBytes
                        );
                    const std::size_t emissionTextureOffset =
                        metallicTextureOffset + metallicTextureBytes;
                    mesh.emissionTexture.width = emissionTextureWidth;
                    mesh.emissionTexture.height = emissionTextureHeight;
                    mesh.emissionTexture.channel = emissionTextureChannel;
                    mesh.emissionTexture.rgba8 = hasDeduplicatedTextures
                        && hasEmissionTexture
                        ? sceneTextures[emissionTextureIndex].rgba8
                        : std::make_shared<const std::vector<std::uint8_t>>(
                            allBytes.begin() + emissionTextureOffset,
                            allBytes.begin() + emissionTextureOffset
                                + emissionTextureBytes
                        );
                    const std::size_t normalTextureOffset =
                        emissionTextureOffset + emissionTextureBytes;
                    mesh.normalTexture.width = normalTextureWidth;
                    mesh.normalTexture.height = normalTextureHeight;
                    mesh.normalTexture.channel = 4;
                    mesh.normalTexture.rgba8 = hasDeduplicatedTextures
                        && hasNormalTexture
                        ? sceneTextures[normalTextureIndex].rgba8
                        : std::make_shared<const std::vector<std::uint8_t>>(
                            allBytes.begin() + normalTextureOffset,
                            allBytes.begin() + normalTextureOffset
                                + normalTextureBytes
                        );
                } catch (const std::bad_alloc&) {
                    return std::nullopt;
                }
                recordOffset += vertexBytes + indexBytes + normalBytes
                    + uvBytes + textureBytes + roughnessTextureBytes
                    + metallicTextureBytes + emissionTextureBytes
                    + normalTextureBytes;
            }
            try {
                scene.meshes.push_back(std::move(mesh));
            } catch (const std::bad_alloc&) {
                return std::nullopt;
            }
        }
        if (recordOffset != allBytes.size()) return std::nullopt;
    }
    return scene;
}

std::optional<BridgeRayCamera> readLiveCameraState() {
    const auto scene = readLiveSceneState(false);
    if (!scene.has_value()) return std::nullopt;
    return scene->camera;
}

bool publishMetalCameraSensorFrame(
    const char* path,
    std::uint32_t sourceWidth,
    std::uint32_t sourceHeight,
    const std::uint8_t* rgba8
) {
    if (path == nullptr || path[0] == '\0' || sourceWidth == 0 || sourceHeight == 0
        || rgba8 == nullptr) {
        return false;
    }
    auto requestedDimension = [](const char* name, std::uint32_t fallback) {
        const char* value = std::getenv(name);
        if (value == nullptr || value[0] == '\0') return fallback;
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        return end != value && *end == '\0' && parsed >= 16 && parsed <= 8192
            ? static_cast<std::uint32_t>(parsed)
            : fallback;
    };
    const std::uint32_t width = requestedDimension(
        "IMB_CAMERA_SENSOR_WIDTH",
        sourceWidth
    );
    const std::uint32_t height = requestedDimension(
        "IMB_CAMERA_SENSOR_HEIGHT",
        sourceHeight
    );
    const std::uint64_t pixelCount =
        static_cast<std::uint64_t>(width) * height;
    if (pixelCount > std::numeric_limits<std::size_t>::max() / 3) {
        return false;
    }
    std::vector<std::uint8_t> rgb;
    try {
        rgb.resize(static_cast<std::size_t>(pixelCount) * 3);
    } catch (const std::bad_alloc&) {
        return false;
    }
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint32_t sourceY = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(y) * sourceHeight / height
        );
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::uint32_t sourceX = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(x) * sourceWidth / width
            );
            const std::size_t sourceOffset = (
                static_cast<std::size_t>(sourceY) * sourceWidth + sourceX
            ) * 4;
            const std::size_t outputOffset = (
                static_cast<std::size_t>(y) * width + x
            ) * 3;
            rgb[outputOffset] = rgba8[sourceOffset];
            rgb[outputOffset + 1] = rgba8[sourceOffset + 1];
            rgb[outputOffset + 2] = rgba8[sourceOffset + 2];
        }
    }
    const std::string temporaryPath = std::string(path)
        + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
    {
        std::ofstream output(temporaryPath, std::ios::binary | std::ios::trunc);
        if (!output) return false;
        output << "P6\n" << width << " " << height << "\n255\n";
        output.write(
            reinterpret_cast<const char*>(rgb.data()),
            static_cast<std::streamsize>(rgb.size())
        );
        if (!output) {
            output.close();
            std::remove(temporaryPath.c_str());
            return false;
        }
    }
    if (std::rename(temporaryPath.c_str(), path) != 0) {
        std::remove(temporaryPath.c_str());
        return false;
    }
    return true;
}

struct FormatBlockInfo {
    std::uint32_t width = 1;
    std::uint32_t height = 1;
    std::uint32_t depth = 1;
    std::uint32_t bytes = 4;
};

FormatBlockInfo formatBlockInfo(VkFormat format) {
    const auto value = static_cast<std::uint32_t>(format);
    if (value >= VK_FORMAT_R8_UNORM && value <= VK_FORMAT_R8_SRGB) {
        return {1, 1, 1, 1};
    }
    if (value >= VK_FORMAT_R8G8_UNORM && value <= VK_FORMAT_R8G8_SRGB) {
        return {1, 1, 1, 2};
    }
    if (value >= VK_FORMAT_R8G8B8_UNORM && value <= VK_FORMAT_B8G8R8_SRGB) {
        return {1, 1, 1, 3};
    }
    if (value >= VK_FORMAT_R8G8B8A8_UNORM && value <= VK_FORMAT_A2B10G10R10_SINT_PACK32) {
        return {1, 1, 1, 4};
    }
    if (value >= VK_FORMAT_R16_UNORM && value <= VK_FORMAT_R16_SFLOAT) {
        return {1, 1, 1, 2};
    }
    if (value >= VK_FORMAT_R16G16_UNORM && value <= VK_FORMAT_R16G16_SFLOAT) {
        return {1, 1, 1, 4};
    }
    if (value >= VK_FORMAT_R16G16B16_UNORM && value <= VK_FORMAT_R16G16B16_SFLOAT) {
        return {1, 1, 1, 6};
    }
    if (value >= VK_FORMAT_R16G16B16A16_UNORM
        && value <= VK_FORMAT_R16G16B16A16_SFLOAT) {
        return {1, 1, 1, 8};
    }
    if (value >= VK_FORMAT_R32_UINT && value <= VK_FORMAT_R32_SFLOAT) {
        return {1, 1, 1, 4};
    }
    if (value >= VK_FORMAT_R32G32_UINT && value <= VK_FORMAT_R32G32_SFLOAT) {
        return {1, 1, 1, 8};
    }
    if (value >= VK_FORMAT_R32G32B32_UINT && value <= VK_FORMAT_R32G32B32_SFLOAT) {
        return {1, 1, 1, 12};
    }
    if (value >= VK_FORMAT_R32G32B32A32_UINT
        && value <= VK_FORMAT_R32G32B32A32_SFLOAT) {
        return {1, 1, 1, 16};
    }
    if (value >= VK_FORMAT_BC1_RGB_UNORM_BLOCK
        && value <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK) {
        return {4, 4, 1, 8};
    }
    if (value >= VK_FORMAT_BC2_UNORM_BLOCK
        && value <= VK_FORMAT_BC3_SRGB_BLOCK) {
        return {4, 4, 1, 16};
    }
    if (value >= VK_FORMAT_BC4_UNORM_BLOCK
        && value <= VK_FORMAT_BC4_SNORM_BLOCK) {
        return {4, 4, 1, 8};
    }
    if (value >= VK_FORMAT_BC5_UNORM_BLOCK
        && value <= VK_FORMAT_BC7_SRGB_BLOCK) {
        return {4, 4, 1, 16};
    }
    return {};
}

bool texelBufferFormatSupported(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM:
    case VK_FORMAT_R8_SNORM:
    case VK_FORMAT_R8_UINT:
    case VK_FORMAT_R8_SINT:
    case VK_FORMAT_R8G8_UNORM:
    case VK_FORMAT_R8G8_SNORM:
    case VK_FORMAT_R8G8_UINT:
    case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SNORM:
    case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R16_UNORM:
    case VK_FORMAT_R16_SNORM:
    case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16G16_UNORM:
    case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_UINT:
    case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_SFLOAT:
    case VK_FORMAT_R16G16B16A16_UNORM:
    case VK_FORMAT_R16G16B16A16_SNORM:
    case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return true;
    default:
        return false;
    }
}

VkImageAspectFlags formatAspectMask(VkFormat format) {
    if (format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_X8_D24_UNORM_PACK32
        || format == VK_FORMAT_D32_SFLOAT) {
        return VK_IMAGE_ASPECT_DEPTH_BIT;
    }
    if (format == VK_FORMAT_S8_UINT) return VK_IMAGE_ASPECT_STENCIL_BIT;
    if (format == VK_FORMAT_D16_UNORM_S8_UINT
        || format == VK_FORMAT_D24_UNORM_S8_UINT
        || format == VK_FORMAT_D32_SFLOAT_S8_UINT) {
        return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    }
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

std::uint64_t clampedMultiply(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > UINT64_MAX / left) return UINT64_MAX;
    return left * right;
}

std::uint64_t clampedAdd(std::uint64_t left, std::uint64_t right) {
    if (right > UINT64_MAX - left) return UINT64_MAX;
    return left + right;
}

std::uint64_t alignClamped(std::uint64_t value, std::uint64_t alignment) {
    if (alignment == 0 || value > UINT64_MAX - (alignment - 1)) return UINT64_MAX;
    return (value + alignment - 1) & ~(alignment - 1);
}

std::uint64_t mipByteSize(VkFormat format, VkExtent3D extent) {
    const auto block = formatBlockInfo(format);
    const std::uint64_t blocksX = (extent.width + block.width - 1) / block.width;
    const std::uint64_t blocksY = (extent.height + block.height - 1) / block.height;
    const std::uint64_t blocksZ = (extent.depth + block.depth - 1) / block.depth;
    return clampedMultiply(
        clampedMultiply(clampedMultiply(blocksX, blocksY), blocksZ),
        block.bytes
    );
}

VkExtent3D metalSparseImageGranularity(VkFormat format, VkImageType type) {
    const auto block = formatBlockInfo(format);
    std::uint32_t blockCountX = 64;
    std::uint32_t blockCountY = 64;
    if (block.bytes <= 1) {
        blockCountX = 128;
        blockCountY = 128;
    } else if (block.bytes == 2) {
        blockCountX = 128;
        blockCountY = 64;
    } else if (block.bytes == 8) {
        blockCountX = 64;
        blockCountY = 32;
    } else if (block.bytes >= 16) {
        blockCountX = 32;
        blockCountY = 32;
    }
    return {
        blockCountX * block.width,
        blockCountY * block.height,
        type == VK_IMAGE_TYPE_3D ? 4U : 1U,
    };
}

VkExtent3D sparseImageGranularity(VkFormat format, VkImageType type) {
    const auto metal = metalSparseImageGranularity(format, type);
    return {
        metal.width * 2,
        metal.height * 2,
        metal.depth,
    };
}

VkSparseImageMemoryRequirements sparseImageRequirements(
    VkFormat format,
    VkImageType type,
    VkExtent3D extent,
    std::uint32_t mipLevels,
    std::uint32_t arrayLayers
) {
    VkSparseImageMemoryRequirements requirements{};
    requirements.formatProperties.aspectMask = formatAspectMask(format);
    requirements.formatProperties.imageGranularity = sparseImageGranularity(format, type);
    requirements.formatProperties.flags = 0;

    std::uint64_t tiledBytesPerLayer = 0;
    std::uint64_t tailBytesPerLayer = 0;
    requirements.imageMipTailFirstLod = mipLevels;
    for (std::uint32_t level = 0; level < mipLevels; ++level) {
        const VkExtent3D levelExtent{
            std::max(1U, extent.width >> level),
            std::max(1U, extent.height >> level),
            std::max(1U, extent.depth >> level),
        };
        if (requirements.imageMipTailFirstLod == mipLevels
            && (levelExtent.width < requirements.formatProperties.imageGranularity.width
                || levelExtent.height < requirements.formatProperties.imageGranularity.height
                || levelExtent.depth < requirements.formatProperties.imageGranularity.depth)) {
            requirements.imageMipTailFirstLod = level;
        }
        if (level < requirements.imageMipTailFirstLod) {
            const auto granularity = requirements.formatProperties.imageGranularity;
            const std::uint64_t blocksX = (levelExtent.width + granularity.width - 1)
                / granularity.width;
            const std::uint64_t blocksY = (levelExtent.height + granularity.height - 1)
                / granularity.height;
            const std::uint64_t blocksZ = (levelExtent.depth + granularity.depth - 1)
                / granularity.depth;
            tiledBytesPerLayer = clampedAdd(
                tiledBytesPerLayer,
                clampedMultiply(
                    clampedMultiply(clampedMultiply(blocksX, blocksY), blocksZ),
                    kSparseImageBlockBytes
                )
            );
        } else {
            const auto metalGranularity = metalSparseImageGranularity(format, type);
            const std::uint64_t metalTilesX =
                (levelExtent.width + metalGranularity.width - 1) / metalGranularity.width;
            const std::uint64_t metalTilesY =
                (levelExtent.height + metalGranularity.height - 1) / metalGranularity.height;
            const std::uint64_t metalTilesZ =
                (levelExtent.depth + metalGranularity.depth - 1) / metalGranularity.depth;
            tailBytesPerLayer = clampedAdd(
                tailBytesPerLayer,
                clampedMultiply(
                    clampedMultiply(
                        clampedMultiply(metalTilesX, metalTilesY),
                        metalTilesZ
                    ),
                    kMetalSparseImageBlockBytes
                )
            );
        }
    }

    requirements.imageMipTailOffset = alignClamped(
        clampedMultiply(tiledBytesPerLayer, arrayLayers),
        kSparseImageBlockBytes
    );
    if (requirements.imageMipTailFirstLod < mipLevels) {
        requirements.imageMipTailSize = alignClamped(
            std::max<std::uint64_t>(tailBytesPerLayer, 1),
            kSparseImageBlockBytes
        );
        requirements.imageMipTailStride = requirements.imageMipTailSize;
    }
    return requirements;
}

VkDeviceSize imageAllocationSize(
    VkFormat format,
    VkImageType type,
    VkExtent3D extent,
    std::uint32_t mipLevels,
    std::uint32_t arrayLayers,
    VkSampleCountFlagBits samples,
    VkImageCreateFlags flags
) {
    if ((flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) != 0) {
        const auto sparse = sparseImageRequirements(format, type, extent, mipLevels, arrayLayers);
        // Metal owns any implementation-private texture metadata. Expose only
        // the Vulkan color aspect and its per-layer mip tails to the guest.
        if (sparse.imageMipTailSize == 0) return sparse.imageMipTailOffset;
        return clampedAdd(
            sparse.imageMipTailOffset,
            clampedAdd(
                clampedMultiply(
                    sparse.imageMipTailStride,
                    arrayLayers > 0 ? arrayLayers - 1 : 0
                ),
                sparse.imageMipTailSize
            )
        );
    }
    std::uint64_t bytes = 0;
    for (std::uint32_t level = 0; level < mipLevels; ++level) {
        const VkExtent3D levelExtent{
            std::max(1U, extent.width >> level),
            std::max(1U, extent.height >> level),
            std::max(1U, extent.depth >> level),
        };
        bytes = clampedAdd(bytes, mipByteSize(format, levelExtent));
    }
    bytes = clampedMultiply(bytes, arrayLayers);
    bytes = clampedMultiply(bytes, static_cast<std::uint32_t>(samples));
    return alignClamped(std::max<std::uint64_t>(bytes, 1), 256);
}

std::uint64_t imageLayerByteSize(
    VkFormat format,
    VkExtent3D extent,
    std::uint32_t mipLevels
) {
    std::uint64_t bytes = 0;
    for (std::uint32_t level = 0; level < mipLevels; ++level) {
        const VkExtent3D levelExtent{
            std::max(1U, extent.width >> level),
            std::max(1U, extent.height >> level),
            std::max(1U, extent.depth >> level),
        };
        bytes = clampedAdd(bytes, mipByteSize(format, levelExtent));
    }
    return bytes;
}

std::uint64_t imageSubresourceByteOffset(
    VkFormat format,
    VkExtent3D extent,
    std::uint32_t mipLevels,
    std::uint32_t mipLevel,
    std::uint32_t arrayLayer
) {
    std::uint64_t offset = clampedMultiply(
        imageLayerByteSize(format, extent, mipLevels),
        arrayLayer
    );
    for (std::uint32_t level = 0; level < mipLevel; ++level) {
        const VkExtent3D levelExtent{
            std::max(1U, extent.width >> level),
            std::max(1U, extent.height >> level),
            std::max(1U, extent.depth >> level),
        };
        offset = clampedAdd(offset, mipByteSize(format, levelExtent));
    }
    return offset;
}

constexpr std::uint64_t kRequiredCapabilities = IMB_CAP_METAL_AVAILABLE
    | IMB_CAP_METAL_BUFFER
    | IMB_CAP_METAL_COMPUTE
    | IMB_CAP_METAL_IMAGE
    | IMB_CAP_METAL_RASTER
    | IMB_CAP_METAL_UI_RASTER
    | IMB_CAP_RESOURCE_IO
    | IMB_CAP_REAL_FENCE;

constexpr std::uint8_t kDeviceUUID[VK_UUID_SIZE] = {
    'I', 'M', 'B', '-', 'A', 'P', 'P', 'L', 'E', '-', 'M', '4', 0, 0, 0, 1,
};

std::uint64_t physicalMemoryBytes() {
    struct sysinfo information {};
    if (::sysinfo(&information) != 0 || information.totalram == 0) return 0;
    return static_cast<std::uint64_t>(information.totalram)
        * static_cast<std::uint64_t>(information.mem_unit);
}

struct Capabilities {
    std::uint64_t bits = 0;
    std::uint64_t maxBufferLength = 0;
    std::string deviceName;
};

struct BridgeSparseImageProperties {
    std::uint32_t tileWidth = 0;
    std::uint32_t tileHeight = 0;
    std::uint32_t tileDepth = 0;
    std::uint64_t tileSizeBytes = 0;
};

struct BridgeUIDraw {
    std::uint64_t textureID = 0;
    std::uint32_t indexCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    std::uint32_t scissorX = 0;
    std::uint32_t scissorY = 0;
    std::uint32_t scissorWidth = 0;
    std::uint32_t scissorHeight = 0;
};

struct BridgeComputeBinding {
    std::uint32_t descriptorSet = 0;
    std::uint32_t binding = 0;
    std::uint32_t arrayElement = 0;
    std::uint32_t kind = 0;
    std::uint32_t format = 0;
    std::uint64_t resourceID = 0;
    std::uint64_t offset = 0;
    std::uint64_t length = 0;
};

struct BridgePrimitiveAccelerationStructureGeometry {
    std::uint32_t kind = 0;
    std::uint32_t flags = 0;
    std::uint64_t dataResourceID = 0;
    std::uint64_t dataOffset = 0;
    std::uint32_t primitiveCount = 0;
    std::uint32_t stride = 0;
    std::uint64_t indexResourceID = 0;
    std::uint64_t indexOffset = 0;
    std::uint32_t indexType = 0;
    std::uint32_t vertexFormat = 0;
    std::uint64_t transformResourceID = 0;
    std::uint64_t transformOffset = 0;
};

struct BridgeInstanceAccelerationStructureInstance {
    std::array<std::uint32_t, 12> transformationBits{};
    std::uint32_t options = 0;
    std::uint32_t mask = 0;
    std::uint32_t intersectionFunctionTableOffset = 0;
    std::uint32_t userID = 0;
    std::uint64_t accelerationStructureResourceID = 0;
};

class BridgeConnection {
public:
    explicit BridgeConnection(std::uint32_t port) : fd_(acceptHost(port)) {
        hello();
        capabilities_ = queryCapabilities();
        if ((capabilities_.bits & kRequiredCapabilities) != kRequiredCapabilities) {
            throw std::runtime_error("host does not advertise the verified IMB Metal data path");
        }
    }

    ~BridgeConnection() {
        if (fd_ >= 0) ::close(fd_);
    }

    const Capabilities& capabilities() const { return capabilities_; }

    void shutdown() {
        if (fd_ < 0) return;
        try {
            const auto reply = exchange(IMB_MSG_SHUTDOWN);
            expectType(reply, IMB_MSG_SHUTDOWN);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "imb-vulkan-icd: shutdown warning: %s\n", error.what());
        }
        ::close(fd_);
        fd_ = -1;
    }

private:
    struct Frame {
        imb_message_header header{};
        imb::Bytes payload;
    };

    static int acceptHost(std::uint32_t port) {
        const int listener = ::socket(AF_VSOCK, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (listener < 0) throw std::runtime_error("vsock socket failed: " + std::string(std::strerror(errno)));

        sockaddr_vm address{};
        address.svm_family = AF_VSOCK;
        address.svm_port = port;
        address.svm_cid = VMADDR_CID_ANY;
        if (::bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
            const std::string message = "vsock bind failed: " + std::string(std::strerror(errno));
            ::close(listener);
            throw std::runtime_error(message);
        }
        if (::listen(listener, 1) != 0) {
            const std::string message = "vsock listen failed: " + std::string(std::strerror(errno));
            ::close(listener);
            throw std::runtime_error(message);
        }

        std::fprintf(stderr, "imb-vulkan-icd: listening on vsock port %u\n", port);
        int connection = -1;
        do {
            connection = ::accept(listener, nullptr, nullptr);
        } while (connection < 0 && errno == EINTR);
        const int savedErrno = errno;
        ::close(listener);
        if (connection < 0) throw std::runtime_error("vsock accept failed: " + std::string(std::strerror(savedErrno)));
        std::fprintf(stderr, "imb-vulkan-icd: host vsock connected\n");
        return connection;
    }

    void writeAll(const std::uint8_t* bytes, std::size_t size) {
        std::size_t offset = 0;
        while (offset < size) {
            const ssize_t written = ::send(fd_, bytes + offset, size - offset, MSG_NOSIGNAL);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) throw std::runtime_error("vsock write failed: " + std::string(std::strerror(errno)));
            offset += static_cast<std::size_t>(written);
        }
    }

    void readAll(std::uint8_t* bytes, std::size_t size) {
        std::size_t offset = 0;
        while (offset < size) {
            const ssize_t count = ::read(fd_, bytes + offset, size - offset);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) throw std::runtime_error("vsock read failed: " + std::string(std::strerror(errno)));
            if (count == 0) throw std::runtime_error("host closed the IMB stream");
            offset += static_cast<std::size_t>(count);
        }
    }

    Frame exchange(std::uint16_t type, imb::Bytes payload = {}, std::uint64_t resourceID = 0) {
        const std::uint64_t requestID = nextRequestID_++;
        const imb_message_header request{
            IMB_PROTOCOL_MAGIC,
            IMB_PROTOCOL_VERSION_MAJOR,
            IMB_PROTOCOL_VERSION_MINOR,
            type,
            IMB_FLAG_NONE,
            static_cast<std::uint32_t>(payload.size()),
            requestID,
            resourceID,
        };
        const auto headerBytes = imb::encodeHeader(request);
        writeAll(headerBytes.data(), headerBytes.size());
        if (!payload.empty()) writeAll(payload.data(), payload.size());

        imb::Bytes responseHeaderBytes(IMB_PROTOCOL_HEADER_SIZE);
        readAll(responseHeaderBytes.data(), responseHeaderBytes.size());
        Frame response;
        response.header = imb::decodeHeader(responseHeaderBytes);
        if (response.header.magic != IMB_PROTOCOL_MAGIC
            || response.header.version_major != IMB_PROTOCOL_VERSION_MAJOR
            || response.header.version_minor != IMB_PROTOCOL_VERSION_MINOR
            || response.header.flags != IMB_FLAG_RESPONSE
            || response.header.request_id != requestID
            || response.header.payload_length > IMB_PROTOCOL_MAX_PAYLOAD) {
            throw std::runtime_error("invalid IMB response header");
        }
        response.payload.resize(response.header.payload_length);
        if (!response.payload.empty()) readAll(response.payload.data(), response.payload.size());
        if (response.header.message_type == IMB_MSG_ERROR) throwProtocolError(response.payload);
        return response;
    }

    static void expectType(const Frame& frame, std::uint16_t expected) {
        if (frame.header.message_type != expected) {
            throw std::runtime_error("unexpected IMB reply type " + std::to_string(frame.header.message_type));
        }
    }

    [[noreturn]] static void throwProtocolError(const imb::Bytes& payload) {
        if (payload.size() < sizeof(imb_error_payload)) throw std::runtime_error("malformed IMB error");
        const auto code = imb::readLittleEndian<std::uint32_t>(payload, 0);
        const auto length = imb::readLittleEndian<std::uint32_t>(payload, 4);
        if (length > payload.size() - sizeof(imb_error_payload)) throw std::runtime_error("malformed IMB error text");
        const std::string message(
            payload.begin() + sizeof(imb_error_payload),
            payload.begin() + sizeof(imb_error_payload) + length
        );
        throw std::runtime_error("IMB error " + std::to_string(code) + ": " + message);
    }

    void hello() {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, std::uint16_t{IMB_PROTOCOL_VERSION_MAJOR});
        imb::appendLittleEndian(payload, std::uint16_t{IMB_PROTOCOL_VERSION_MINOR});
        imb::appendLittleEndian(payload, std::uint16_t{IMB_PROTOCOL_VERSION_MAJOR});
        imb::appendLittleEndian(payload, std::uint16_t{IMB_PROTOCOL_VERSION_MINOR});
        const auto reply = exchange(IMB_MSG_HELLO, std::move(payload));
        expectType(reply, IMB_MSG_HELLO_REPLY);
        if (reply.payload.size() != sizeof(imb_hello_reply_payload)) {
            throw std::runtime_error("invalid IMB HELLO_REPLY size");
        }
    }

    Capabilities queryCapabilities() {
        const auto reply = exchange(IMB_MSG_QUERY_CAPABILITIES);
        expectType(reply, IMB_MSG_CAPABILITIES_REPLY);
        if (reply.payload.size() < sizeof(imb_capabilities_payload)) {
            throw std::runtime_error("invalid IMB capability payload");
        }
        Capabilities result;
        result.bits = imb::readLittleEndian<std::uint64_t>(reply.payload, 0);
        result.maxBufferLength = imb::readLittleEndian<std::uint64_t>(reply.payload, 8);
        const auto nameLength = imb::readLittleEndian<std::uint32_t>(reply.payload, 16);
        if (nameLength > reply.payload.size() - sizeof(imb_capabilities_payload)) {
            throw std::runtime_error("invalid IMB Metal device name length");
        }
        result.deviceName.assign(
            reply.payload.begin() + sizeof(imb_capabilities_payload),
            reply.payload.begin() + sizeof(imb_capabilities_payload) + nameLength
        );
        return result;
    }

public:
    std::uint64_t createBuffer(std::uint64_t size) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, size);
        imb::appendLittleEndian(payload, std::uint32_t{IMB_RESOURCE_BUFFER});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto reply = exchange(IMB_MSG_CREATE_RESOURCE, std::move(payload));
        expectType(reply, IMB_MSG_CREATE_RESOURCE);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned resource ID zero");
        return reply.header.resource_id;
    }

    std::uint64_t createImage(
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t format,
        std::uint32_t depth = 1,
        bool texture3D = false
    ) {
        if (width == 0 || height == 0 || depth == 0
            || (texture3D && depth > 0xffffU)) {
            throw std::runtime_error("invalid ordinary image extent");
        }
        std::uint64_t size = width;
        if (height > UINT64_MAX / size) throw std::runtime_error("ordinary image size overflow");
        size *= height;
        if (depth > UINT64_MAX / size) throw std::runtime_error("ordinary image size overflow");
        size *= depth;
        if (size > UINT64_MAX / 4) throw std::runtime_error("ordinary image size overflow");
        size *= 4;
        const std::uint32_t options = texture3D
            ? IMB_IMAGE_OPTION_3D
                | (depth << IMB_IMAGE_OPTION_DEPTH_SHIFT)
            : IMB_IMAGE_OPTION_NONE;
        imb::Bytes payload;
        imb::appendLittleEndian(payload, size);
        imb::appendLittleEndian(payload, std::uint32_t{IMB_RESOURCE_IMAGE});
        imb::appendLittleEndian(payload, options);
        imb::appendLittleEndian(payload, width);
        imb::appendLittleEndian(payload, height);
        imb::appendLittleEndian(payload, format);
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto reply = exchange(IMB_MSG_CREATE_RESOURCE, std::move(payload));
        expectType(reply, IMB_MSG_CREATE_RESOURCE);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned image resource ID zero");
        return reply.header.resource_id;
    }

    BridgeSparseImageProperties querySparseImageProperties(
        std::uint32_t format,
        std::uint32_t textureType = IMB_TEXTURE_TYPE_2D,
        std::uint32_t sampleCount = 1
    ) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, format);
        imb::appendLittleEndian(payload, textureType);
        imb::appendLittleEndian(payload, sampleCount);
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto reply = exchange(IMB_MSG_QUERY_SPARSE_IMAGE_PROPERTIES, std::move(payload));
        expectType(reply, IMB_MSG_QUERY_SPARSE_IMAGE_PROPERTIES);
        if (reply.payload.size() != sizeof(imb_sparse_image_properties_payload)) {
            throw std::runtime_error("invalid sparse image properties reply");
        }
        BridgeSparseImageProperties properties;
        properties.tileWidth = imb::readLittleEndian<std::uint32_t>(reply.payload, 0);
        properties.tileHeight = imb::readLittleEndian<std::uint32_t>(reply.payload, 4);
        properties.tileDepth = imb::readLittleEndian<std::uint32_t>(reply.payload, 8);
        properties.tileSizeBytes = imb::readLittleEndian<std::uint64_t>(reply.payload, 16);
        if (properties.tileWidth == 0 || properties.tileHeight == 0
            || properties.tileDepth == 0 || properties.tileSizeBytes == 0) {
            throw std::runtime_error("host returned empty sparse image properties");
        }
        return properties;
    }

    std::uint64_t createSparseImage(
        std::uint64_t virtualSize,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t format,
        std::uint32_t mipLevels,
        std::uint32_t arrayLayers,
        std::uint32_t sampleCount = 1,
        std::uint32_t textureType = IMB_TEXTURE_TYPE_2D
    ) {
        imb::Bytes payload;
        payload.reserve(sizeof(imb_create_sparse_image_resource_payload));
        imb::appendLittleEndian(payload, virtualSize);
        imb::appendLittleEndian(payload, std::uint32_t{IMB_RESOURCE_IMAGE});
        imb::appendLittleEndian(payload, std::uint32_t{IMB_IMAGE_OPTION_SPARSE});
        imb::appendLittleEndian(payload, width);
        imb::appendLittleEndian(payload, height);
        imb::appendLittleEndian(payload, format);
        imb::appendLittleEndian(payload, mipLevels);
        imb::appendLittleEndian(payload, arrayLayers);
        imb::appendLittleEndian(payload, sampleCount);
        imb::appendLittleEndian(payload, textureType);
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto reply = exchange(IMB_MSG_CREATE_RESOURCE, std::move(payload));
        expectType(reply, IMB_MSG_CREATE_RESOURCE);
        if (reply.header.resource_id == 0) {
            throw std::runtime_error("host returned sparse image resource ID zero");
        }
        return reply.header.resource_id;
    }

    void updateSparseImageMapping(
        std::uint64_t resourceID,
        bool map,
        std::uint32_t mipLevel,
        std::uint32_t slice,
        std::uint32_t tileX,
        std::uint32_t tileY,
        std::uint32_t tileZ,
        std::uint32_t tileWidth,
        std::uint32_t tileHeight,
        std::uint32_t tileDepth
    ) {
        imb::Bytes payload;
        payload.reserve(sizeof(imb_update_sparse_image_mapping_payload));
        imb::appendLittleEndian(
            payload,
            std::uint32_t{map ? IMB_SPARSE_MAPPING_MAP : IMB_SPARSE_MAPPING_UNMAP}
        );
        imb::appendLittleEndian(payload, mipLevel);
        imb::appendLittleEndian(payload, slice);
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, tileX);
        imb::appendLittleEndian(payload, tileY);
        imb::appendLittleEndian(payload, tileZ);
        imb::appendLittleEndian(payload, tileWidth);
        imb::appendLittleEndian(payload, tileHeight);
        imb::appendLittleEndian(payload, tileDepth);
        const auto reply = exchange(
            IMB_MSG_UPDATE_SPARSE_IMAGE_MAPPING,
            std::move(payload),
            resourceID
        );
        expectType(reply, IMB_MSG_UPDATE_SPARSE_IMAGE_MAPPING);
    }

    std::uint64_t createComputePipeline(
        const std::vector<std::uint32_t>& code,
        const char* entryPoint,
        std::uint32_t* creationFlags
    ) {
        if (code.empty() || entryPoint == nullptr || *entryPoint == '\0') {
            throw std::runtime_error("invalid SPIR-V compute pipeline request");
        }
        const std::size_t spirvLength = code.size() * sizeof(std::uint32_t);
        const std::size_t entryPointLength = std::strlen(entryPoint);
        const std::size_t payloadLength = sizeof(imb_create_compute_pipeline_payload)
            + spirvLength + entryPointLength;
        if (spirvLength > UINT32_MAX || entryPointLength > 1024
            || payloadLength > IMB_PROTOCOL_MAX_PAYLOAD) {
            throw std::runtime_error("SPIR-V compute pipeline request exceeds IMB limits");
        }
        imb::Bytes payload;
        payload.reserve(payloadLength);
        imb::appendLittleEndian(payload, static_cast<std::uint32_t>(spirvLength));
        imb::appendLittleEndian(payload, static_cast<std::uint32_t>(entryPointLength));
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto* codeBytes = reinterpret_cast<const std::uint8_t*>(code.data());
        payload.insert(payload.end(), codeBytes, codeBytes + spirvLength);
        payload.insert(payload.end(), entryPoint, entryPoint + entryPointLength);
        const auto reply = exchange(IMB_MSG_CREATE_COMPUTE_PIPELINE, std::move(payload));
        expectType(reply, IMB_MSG_CREATE_COMPUTE_PIPELINE);
        if (reply.header.resource_id == 0) {
            throw std::runtime_error("host returned compute pipeline resource ID zero");
        }
        if (creationFlags != nullptr) {
            *creationFlags = reply.payload.size() >= sizeof(std::uint32_t)
                ? imb::readLittleEndian<std::uint32_t>(reply.payload, 0)
                : static_cast<std::uint32_t>(
                    IMB_COMPUTE_PIPELINE_FLAG_SOFTWARE_FP64_EXECUTION_REQUIRED
                );
        }
        return reply.header.resource_id;
    }

    std::uint64_t createAccelerationStructure(std::uint32_t type, std::uint64_t requestedSize) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, requestedSize);
        imb::appendLittleEndian(payload, type);
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto reply = exchange(IMB_MSG_CREATE_ACCELERATION_STRUCTURE, std::move(payload));
        expectType(reply, IMB_MSG_CREATE_ACCELERATION_STRUCTURE);
        if (reply.header.resource_id == 0) {
            throw std::runtime_error("host returned acceleration structure resource ID zero");
        }
        return reply.header.resource_id;
    }

    void buildPrimitiveAccelerationStructure(
        std::uint64_t resourceID,
        std::uint32_t buildFlags,
        const std::vector<BridgePrimitiveAccelerationStructureGeometry>& geometries
    ) {
        if (resourceID == 0 || geometries.empty() || geometries.size() > 65535) {
            throw std::runtime_error("invalid primitive acceleration structure build request");
        }
        const std::size_t payloadSize = sizeof(imb_build_primitive_acceleration_structure_payload)
            + geometries.size() * sizeof(imb_primitive_acceleration_structure_geometry_payload);
        if (payloadSize > IMB_PROTOCOL_MAX_PAYLOAD) {
            throw std::runtime_error("primitive acceleration structure build exceeds IMB payload limit");
        }
        imb::Bytes payload;
        payload.reserve(payloadSize);
        imb::appendLittleEndian(payload, static_cast<std::uint32_t>(geometries.size()));
        imb::appendLittleEndian(payload, buildFlags);
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        for (const auto& geometry : geometries) {
            imb::appendLittleEndian(payload, geometry.kind);
            imb::appendLittleEndian(payload, geometry.flags);
            imb::appendLittleEndian(payload, geometry.dataResourceID);
            imb::appendLittleEndian(payload, geometry.dataOffset);
            imb::appendLittleEndian(payload, geometry.primitiveCount);
            imb::appendLittleEndian(payload, geometry.stride);
            imb::appendLittleEndian(payload, geometry.indexResourceID);
            imb::appendLittleEndian(payload, geometry.indexOffset);
            imb::appendLittleEndian(payload, geometry.indexType);
            imb::appendLittleEndian(payload, geometry.vertexFormat);
            imb::appendLittleEndian(payload, geometry.transformResourceID);
            imb::appendLittleEndian(payload, geometry.transformOffset);
        }
        const auto reply = exchange(
            IMB_MSG_BUILD_PRIMITIVE_ACCELERATION_STRUCTURE,
            std::move(payload),
            resourceID
        );
        expectType(reply, IMB_MSG_BUILD_PRIMITIVE_ACCELERATION_STRUCTURE);
        if (reply.header.resource_id != resourceID) {
            throw std::runtime_error("host returned the wrong acceleration structure resource ID");
        }
    }

    void buildInstanceAccelerationStructure(
        std::uint64_t resourceID,
        std::uint32_t buildFlags,
        const std::vector<BridgeInstanceAccelerationStructureInstance>& instances
    ) {
        if (resourceID == 0 || instances.empty() || instances.size() > 1048576) {
            throw std::runtime_error("invalid instance acceleration structure build request");
        }
        const std::size_t payloadSize = sizeof(imb_build_instance_acceleration_structure_payload)
            + instances.size() * sizeof(imb_acceleration_structure_instance_payload);
        if (payloadSize > IMB_PROTOCOL_MAX_PAYLOAD) {
            throw std::runtime_error("instance acceleration structure build exceeds IMB payload limit");
        }
        imb::Bytes payload;
        payload.reserve(payloadSize);
        imb::appendLittleEndian(payload, static_cast<std::uint32_t>(instances.size()));
        imb::appendLittleEndian(payload, buildFlags);
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        for (const auto& instance : instances) {
            for (const auto component : instance.transformationBits) {
                imb::appendLittleEndian(payload, component);
            }
            imb::appendLittleEndian(payload, instance.options);
            imb::appendLittleEndian(payload, instance.mask);
            imb::appendLittleEndian(payload, instance.intersectionFunctionTableOffset);
            imb::appendLittleEndian(payload, instance.userID);
            imb::appendLittleEndian(payload, instance.accelerationStructureResourceID);
            imb::appendLittleEndian(payload, std::uint64_t{0});
        }
        const auto reply = exchange(
            IMB_MSG_BUILD_INSTANCE_ACCELERATION_STRUCTURE,
            std::move(payload),
            resourceID
        );
        expectType(reply, IMB_MSG_BUILD_INSTANCE_ACCELERATION_STRUCTURE);
        if (reply.header.resource_id != resourceID) {
            throw std::runtime_error("host returned the wrong instance acceleration structure resource ID");
        }
    }

    void destroyBuffer(std::uint64_t resourceID) {
        const auto reply = exchange(IMB_MSG_DESTROY_RESOURCE, {}, resourceID);
        expectType(reply, IMB_MSG_DESTROY_RESOURCE);
    }

    void writeBuffer(
        std::uint64_t resourceID,
        const std::uint8_t* bytes,
        std::uint64_t size,
        std::uint64_t destinationOffset = 0
    ) {
        constexpr std::uint64_t maxChunk = IMB_PROTOCOL_MAX_PAYLOAD - sizeof(imb_write_resource_payload);
        std::uint64_t offset = 0;
        while (offset < size) {
            const std::uint32_t chunk = static_cast<std::uint32_t>(std::min(maxChunk, size - offset));
            imb::Bytes payload;
            payload.reserve(sizeof(imb_write_resource_payload) + chunk);
            imb::appendLittleEndian(payload, destinationOffset + offset);
            imb::appendLittleEndian(payload, chunk);
            imb::appendLittleEndian(payload, std::uint32_t{0});
            payload.insert(payload.end(), bytes + offset, bytes + offset + chunk);
            const auto reply = exchange(IMB_MSG_WRITE_RESOURCE, std::move(payload), resourceID);
            expectType(reply, IMB_MSG_WRITE_RESOURCE);
            offset += chunk;
        }
    }

    void writeImage(std::uint64_t resourceID, const std::uint8_t* bytes, std::uint64_t size) {
        writeBuffer(resourceID, bytes, size);
    }

    void readBuffer(
        std::uint64_t resourceID,
        std::uint8_t* bytes,
        std::uint64_t size,
        std::uint64_t remoteOffset = 0
    ) {
        std::uint64_t offset = 0;
        while (offset < size) {
            const std::uint64_t chunk = std::min<std::uint64_t>(IMB_PROTOCOL_MAX_PAYLOAD, size - offset);
            imb::Bytes payload;
            imb::appendLittleEndian(payload, remoteOffset + offset);
            imb::appendLittleEndian(payload, chunk);
            const auto reply = exchange(IMB_MSG_READ_RESOURCE, std::move(payload), resourceID);
            expectType(reply, IMB_MSG_READ_RESOURCE);
            if (reply.payload.size() != chunk) throw std::runtime_error("host returned wrong readback size");
            std::memcpy(bytes + offset, reply.payload.data(), reply.payload.size());
            offset += chunk;
        }
    }

    void readImage(std::uint64_t resourceID, std::uint8_t* bytes, std::uint64_t size) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, std::uint64_t{0});
        imb::appendLittleEndian(payload, size);
        const auto reply = exchange(IMB_MSG_READ_RESOURCE, std::move(payload), resourceID);
        expectType(reply, IMB_MSG_READ_RESOURCE);
        if (reply.payload.size() != size) throw std::runtime_error("host returned wrong image readback size");
        std::memcpy(bytes, reply.payload.data(), reply.payload.size());
    }

    std::uint64_t submitNoop() {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, std::uint16_t{IMB_COMMAND_NOOP});
        imb::appendLittleEndian(payload, std::uint16_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto reply = exchange(IMB_MSG_SUBMIT_COMMAND, std::move(payload), 0);
        expectType(reply, IMB_MSG_SUBMIT_COMMAND);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned no-op fence ID zero");
        return reply.header.resource_id;
    }

    std::uint64_t submitAdd(std::uint64_t resourceID, std::uint32_t elementCount, std::uint32_t addend) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, std::uint16_t{IMB_COMMAND_ADD_U32});
        imb::appendLittleEndian(payload, std::uint16_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, elementCount);
        imb::appendLittleEndian(payload, addend);
        const auto reply = exchange(IMB_MSG_SUBMIT_COMMAND, std::move(payload), resourceID);
        expectType(reply, IMB_MSG_SUBMIT_COMMAND);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned fence ID zero");
        return reply.header.resource_id;
    }

    std::uint64_t submitCompute(
        std::uint64_t pipelineID,
        std::uint32_t groupCountX,
        std::uint32_t groupCountY,
        std::uint32_t groupCountZ,
        const std::vector<BridgeComputeBinding>& bindings,
        const imb::Bytes& pushConstants
    ) {
        if (pipelineID == 0 || groupCountX == 0 || groupCountY == 0 || groupCountZ == 0
            || bindings.size() > UINT32_MAX || pushConstants.size() > 4096) {
            throw std::runtime_error("invalid Metal compute dispatch request");
        }
        const std::size_t payloadSize = sizeof(imb_dispatch_compute_command_payload)
            + bindings.size() * sizeof(imb_compute_binding_payload)
            + pushConstants.size();
        if (payloadSize > IMB_PROTOCOL_MAX_PAYLOAD) {
            throw std::runtime_error("Metal compute dispatch exceeds IMB payload limit");
        }
        imb::Bytes payload;
        payload.reserve(payloadSize);
        imb::appendLittleEndian(payload, std::uint16_t{IMB_COMMAND_DISPATCH_COMPUTE});
        imb::appendLittleEndian(payload, std::uint16_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, groupCountX);
        imb::appendLittleEndian(payload, groupCountY);
        imb::appendLittleEndian(payload, groupCountZ);
        imb::appendLittleEndian(payload, static_cast<std::uint32_t>(bindings.size()));
        imb::appendLittleEndian(
            payload,
            static_cast<std::uint32_t>(pushConstants.size())
        );
        imb::appendLittleEndian(payload, std::uint32_t{0});
        for (const auto& binding : bindings) {
            imb::appendLittleEndian(payload, binding.descriptorSet);
            imb::appendLittleEndian(payload, binding.binding);
            imb::appendLittleEndian(payload, binding.arrayElement);
            imb::appendLittleEndian(payload, binding.kind);
            imb::appendLittleEndian(payload, binding.format);
            imb::appendLittleEndian(payload, std::uint32_t{0});
            imb::appendLittleEndian(payload, binding.resourceID);
            imb::appendLittleEndian(payload, binding.offset);
            imb::appendLittleEndian(payload, binding.length);
        }
        payload.insert(payload.end(), pushConstants.begin(), pushConstants.end());
        const auto reply = exchange(
            IMB_MSG_SUBMIT_COMMAND,
            std::move(payload),
            pipelineID
        );
        expectType(reply, IMB_MSG_SUBMIT_COMMAND);
        if (reply.header.resource_id == 0) {
            throw std::runtime_error("host returned compute fence ID zero");
        }
        return reply.header.resource_id;
    }

    std::uint64_t submitTriangle(std::uint64_t resourceID, std::uint32_t clearRGBA8) {
        imb::Bytes payload;
        imb::appendLittleEndian(payload, std::uint16_t{IMB_COMMAND_DRAW_TRIANGLE});
        imb::appendLittleEndian(payload, std::uint16_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, clearRGBA8);
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto reply = exchange(IMB_MSG_SUBMIT_COMMAND, std::move(payload), resourceID);
        expectType(reply, IMB_MSG_SUBMIT_COMMAND);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned raster fence ID zero");
        return reply.header.resource_id;
    }

    std::uint64_t submitIndexedUI(
        std::uint64_t imageID,
        std::uint64_t vertexBufferID,
        std::uint64_t indexBufferID,
        std::uint64_t vertexBufferOffset,
        std::uint64_t indexBufferOffset,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t clearRGBA8,
        const std::vector<BridgeUIDraw>& draws
    ) {
        if (draws.empty() || draws.size() > UINT32_MAX) {
            throw std::runtime_error("invalid Kit UI draw count");
        }
        const std::size_t payloadSize = sizeof(imb_draw_indexed_ui_command_payload)
            + draws.size() * sizeof(imb_ui_draw_payload);
        if (payloadSize > IMB_PROTOCOL_MAX_PAYLOAD) {
            throw std::runtime_error("Kit UI command exceeds IMB payload limit");
        }
        imb::Bytes payload;
        payload.reserve(payloadSize);
        imb::appendLittleEndian(payload, std::uint16_t{IMB_COMMAND_DRAW_INDEXED_UI});
        imb::appendLittleEndian(payload, std::uint16_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, vertexBufferID);
        imb::appendLittleEndian(payload, indexBufferID);
        imb::appendLittleEndian(payload, vertexBufferOffset);
        imb::appendLittleEndian(payload, indexBufferOffset);
        imb::appendLittleEndian(payload, width);
        imb::appendLittleEndian(payload, height);
        imb::appendLittleEndian(payload, clearRGBA8);
        imb::appendLittleEndian(payload, static_cast<std::uint32_t>(draws.size()));
        for (const auto& draw : draws) {
            imb::appendLittleEndian(payload, draw.textureID);
            imb::appendLittleEndian(payload, draw.indexCount);
            imb::appendLittleEndian(payload, draw.firstIndex);
            imb::appendLittleEndian(payload, static_cast<std::uint32_t>(draw.vertexOffset));
            imb::appendLittleEndian(payload, draw.scissorX);
            imb::appendLittleEndian(payload, draw.scissorY);
            imb::appendLittleEndian(payload, draw.scissorWidth);
            imb::appendLittleEndian(payload, draw.scissorHeight);
            imb::appendLittleEndian(payload, std::uint32_t{0});
        }
        const auto reply = exchange(IMB_MSG_SUBMIT_COMMAND, std::move(payload), imageID);
        expectType(reply, IMB_MSG_SUBMIT_COMMAND);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned Kit UI fence ID zero");
        return reply.header.resource_id;
    }

    std::uint64_t submitRayTrace(
        std::uint64_t imageID,
        std::uint64_t accelerationStructureID,
        std::uint32_t width,
        std::uint32_t height,
        std::uint32_t missRGBA8,
        std::uint32_t hitRGBA8,
        const BridgeRayCamera* camera,
        bool emptyStageGrid = false
    ) {
        if (imageID == 0 || (!emptyStageGrid && accelerationStructureID == 0)
            || (emptyStageGrid && accelerationStructureID != 0)
            || width == 0 || height == 0
            || (camera != nullptr
                && camera->additionalLights.size()
                    > IMB_TRACE_RAYS_MAX_ADDITIONAL_LIGHTS)) {
            throw std::runtime_error("invalid Metal ray dispatch request");
        }
        imb::Bytes payload;
        const std::size_t additionalLightCount =
            camera == nullptr || emptyStageGrid
            ? 0 : camera->additionalLights.size();
        payload.reserve(
            sizeof(imb_trace_rays_command_payload)
            + (additionalLightCount == 0
                ? 0
                : sizeof(imb_trace_rays_light_list_header)
                    + additionalLightCount * sizeof(imb_trace_light_payload))
        );
        imb::appendLittleEndian(payload, std::uint16_t{IMB_COMMAND_TRACE_RAYS});
        imb::appendLittleEndian(payload, std::uint16_t{0});
        imb::appendLittleEndian(payload, std::uint32_t{0});
        imb::appendLittleEndian(payload, accelerationStructureID);
        imb::appendLittleEndian(payload, width);
        imb::appendLittleEndian(payload, height);
        imb::appendLittleEndian(payload, missRGBA8);
        imb::appendLittleEndian(payload, hitRGBA8);
        std::uint32_t rayOptions = IMB_TRACE_RAYS_OPTION_NONE;
        if (camera != nullptr) {
            rayOptions |= static_cast<std::uint32_t>(IMB_TRACE_RAYS_OPTION_LIVE_CAMERA);
            if (!emptyStageGrid && !camera->completeLightList
                && camera->hasSphereLight) {
                rayOptions |= static_cast<std::uint32_t>(
                    IMB_TRACE_RAYS_OPTION_LIVE_SPHERE_LIGHT
                );
            }
            if (!emptyStageGrid && !camera->completeLightList
                && camera->hasDistantLight) {
                rayOptions |= static_cast<std::uint32_t>(
                    IMB_TRACE_RAYS_OPTION_LIVE_DISTANT_LIGHT
                );
            }
            if (!emptyStageGrid && !camera->completeLightList
                && camera->hasDomeLight) {
                rayOptions |= static_cast<std::uint32_t>(
                    IMB_TRACE_RAYS_OPTION_LIVE_DOME_LIGHT
                );
            }
            if (!emptyStageGrid && !camera->additionalLights.empty()) {
                rayOptions |= static_cast<std::uint32_t>(
                    IMB_TRACE_RAYS_OPTION_ADDITIONAL_LIGHTS
                );
            }
        }
        if (emptyStageGrid) {
            rayOptions |= static_cast<std::uint32_t>(
                IMB_TRACE_RAYS_OPTION_EMPTY_STAGE_GRID
            );
        }
        imb::appendLittleEndian(payload, rayOptions);
        imb::appendLittleEndian(payload, std::uint32_t{0});
        const auto appendFloat = [&payload](float value) {
            imb::appendLittleEndian(payload, std::bit_cast<std::uint32_t>(value));
        };
        const BridgeRayCamera emptyCamera{};
        const auto& encodedCamera = camera == nullptr ? emptyCamera : *camera;
        for (float value : encodedCamera.position) appendFloat(value);
        for (float value : encodedCamera.forward) appendFloat(value);
        for (float value : encodedCamera.up) appendFloat(value);
        appendFloat(encodedCamera.verticalFOVRadians);
        appendFloat(encodedCamera.nearDistance);
        appendFloat(encodedCamera.farDistance);
        for (float value : encodedCamera.sphereLightPosition) appendFloat(value);
        for (float value : encodedCamera.sphereLightColor) appendFloat(value);
        appendFloat(encodedCamera.sphereLightIntensity);
        appendFloat(encodedCamera.sphereLightRadius);
        for (float value : encodedCamera.distantLightDirection) appendFloat(value);
        for (float value : encodedCamera.distantLightColor) appendFloat(value);
        appendFloat(encodedCamera.distantLightIntensity);
        appendFloat(encodedCamera.distantLightAngleDegrees);
        for (float value : encodedCamera.domeLightColor) appendFloat(value);
        appendFloat(encodedCamera.domeLightIntensity);
        if (additionalLightCount != 0) {
            imb::appendLittleEndian(
                payload, static_cast<std::uint32_t>(additionalLightCount)
            );
            imb::appendLittleEndian(payload, std::uint32_t{0});
            for (const auto& light : encodedCamera.additionalLights) {
                imb::appendLittleEndian(payload, light.kind);
                imb::appendLittleEndian(payload, light.schema);
                for (float value : light.values) appendFloat(value);
                imb::appendLittleEndian(payload, light.pathHash);
                for (float value : light.shapingAxis) appendFloat(value);
                appendFloat(light.shapingConeAngleDegrees);
                appendFloat(light.shapingConeSoftness);
                appendFloat(light.shapingFocus);
                for (float value : light.shapingFocusTint) appendFloat(value);
                imb::appendLittleEndian(payload, light.shapingFlags);
                imb::appendLittleEndian(
                    payload, light.emissionTextureResourceID
                );
                imb::appendLittleEndian(payload, light.iesTextureResourceID);
                appendFloat(light.iesAngleScale);
                appendFloat(light.iesMultiplier);
                imb::appendLittleEndian(payload, light.textureFlags);
                imb::appendLittleEndian(payload, std::uint32_t{0});
            }
        }
        const auto reply = exchange(IMB_MSG_SUBMIT_COMMAND, std::move(payload), imageID);
        expectType(reply, IMB_MSG_SUBMIT_COMMAND);
        if (reply.header.resource_id == 0) throw std::runtime_error("host returned ray dispatch fence ID zero");
        return reply.header.resource_id;
    }

    void waitFence(std::uint64_t fenceID) {
        const auto reply = exchange(IMB_MSG_WAIT_FENCE, {}, fenceID);
        expectType(reply, IMB_MSG_WAIT_FENCE);
        if (reply.payload.size() != sizeof(imb_wait_fence_reply_payload)
            || imb::readLittleEndian<std::uint32_t>(reply.payload, 0) != 1) {
            throw std::runtime_error("host Metal fence did not signal");
        }
    }

private:
    int fd_ = -1;
    std::uint64_t nextRequestID_ = 1;
    Capabilities capabilities_;
};

struct DeviceMemoryState {
    VkDevice device = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    std::uint64_t resourceID = 0;
    imb::Bytes bytes;
    bool mapped = false;
    VkDeviceSize mappedOffset = 0;
    VkDeviceSize mappedSize = 0;
    std::uint32_t mapReferenceCount = 0;
    VkDeviceSize dirtyOffset = VK_WHOLE_SIZE;
    VkDeviceSize dirtyEnd = 0;
    VkExternalMemoryHandleTypeFlags exportHandleTypes = 0;
    // Retained duplicate of the anonymous OPAQUE_FD backing. CUDA receives a
    // separate duplicate, so both APIs observe writes to the same file pages.
    int externalFD = -1;
};

struct SparseBufferBinding {
    VkDeviceSize resourceOffset = 0;
    VkDeviceSize size = 0;
    DeviceMemoryState* memory = nullptr;
    VkDeviceSize memoryOffset = 0;
    VkSparseMemoryBindFlags flags = 0;
};

struct SparseImageBinding {
    VkImageSubresource subresource{};
    VkOffset3D offset{};
    VkExtent3D extent{};
    DeviceMemoryState* memory = nullptr;
    VkDeviceSize memoryOffset = 0;
    VkSparseMemoryBindFlags flags = 0;
};

struct SparseImageOpaqueBinding {
    VkDeviceSize resourceOffset = 0;
    VkDeviceSize size = 0;
    DeviceMemoryState* memory = nullptr;
    VkDeviceSize memoryOffset = 0;
    VkSparseMemoryBindFlags flags = 0;
};

struct BufferState {
    VkDevice device = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VkBufferCreateFlags flags = 0;
    DeviceMemoryState* memory = nullptr;
    VkDeviceSize memoryOffset = 0;
    VkDeviceAddress deviceAddress = 0;
    std::vector<SparseBufferBinding> sparseBindings;
};

struct BufferViewState {
    VkDevice device = VK_NULL_HANDLE;
    BufferState* buffer = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkDeviceSize offset = 0;
    VkDeviceSize range = 0;
};

struct ImageState {
    VkDevice device = VK_NULL_HANDLE;
    VkExtent3D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageUsageFlags usage = 0;
    VkImageCreateFlags flags = 0;
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    std::uint32_t mipLevels = 1;
    std::uint32_t arrayLayers = 1;
    VkDeviceSize size = 0;
    std::uint64_t resourceID = 0;
    DeviceMemoryState* memory = nullptr;
    VkDeviceSize memoryOffset = 0;
    imb::Bytes sparseBytes;
    VkExtent3D sparseGranularity{};
    VkDeviceSize sparseTileBytes = 0;
    VkExtent3D metalSparseGranularity{};
    VkDeviceSize metalSparseTileBytes = 0;
    std::vector<bool> initializedSubresources;
    std::vector<SparseImageBinding> sparseBindings;
    std::vector<SparseImageOpaqueBinding> sparseOpaqueBindings;
    std::uint64_t storageDescriptorSequence = 0;
    bool dirty = false;
};

struct ImageViewState {
    VkDevice device = VK_NULL_HANDLE;
    ImageState* image = nullptr;
};

struct SamplerState {
    VkDevice device = VK_NULL_HANDLE;
    VkFilter magFilter = VK_FILTER_NEAREST;
    VkFilter minFilter = VK_FILTER_NEAREST;
    VkSamplerMipmapMode mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    VkSamplerAddressMode addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    float mipLodBias = 0.0f;
    VkBool32 anisotropyEnable = VK_FALSE;
    float maxAnisotropy = 1.0f;
    VkBool32 compareEnable = VK_FALSE;
    VkCompareOp compareOp = VK_COMPARE_OP_NEVER;
    float minLod = 0.0f;
    float maxLod = 0.0f;
    VkBorderColor borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VkBool32 unnormalizedCoordinates = VK_FALSE;
};

std::uint32_t bridgeSamplerOptions(const SamplerState& sampler) {
    return (static_cast<std::uint32_t>(sampler.magFilter)
                << IMB_COMPUTE_SAMPLER_MAG_FILTER_SHIFT)
        | (static_cast<std::uint32_t>(sampler.minFilter)
                << IMB_COMPUTE_SAMPLER_MIN_FILTER_SHIFT)
        | (static_cast<std::uint32_t>(sampler.mipmapMode)
                << IMB_COMPUTE_SAMPLER_MIPMAP_MODE_SHIFT)
        | (static_cast<std::uint32_t>(sampler.addressModeU)
                << IMB_COMPUTE_SAMPLER_ADDRESS_U_SHIFT)
        | (static_cast<std::uint32_t>(sampler.addressModeV)
                << IMB_COMPUTE_SAMPLER_ADDRESS_V_SHIFT)
        | (static_cast<std::uint32_t>(sampler.addressModeW)
                << IMB_COMPUTE_SAMPLER_ADDRESS_W_SHIFT)
        | (static_cast<std::uint32_t>(sampler.anisotropyEnable != VK_FALSE)
                << IMB_COMPUTE_SAMPLER_ANISOTROPY_ENABLE_SHIFT)
        | (static_cast<std::uint32_t>(sampler.compareEnable != VK_FALSE)
                << IMB_COMPUTE_SAMPLER_COMPARE_ENABLE_SHIFT)
        | (static_cast<std::uint32_t>(sampler.compareOp)
                << IMB_COMPUTE_SAMPLER_COMPARE_OP_SHIFT)
        | (static_cast<std::uint32_t>(sampler.unnormalizedCoordinates != VK_FALSE)
                << IMB_COMPUTE_SAMPLER_UNNORMALIZED_SHIFT)
        | (static_cast<std::uint32_t>(sampler.borderColor)
                << IMB_COMPUTE_SAMPLER_BORDER_COLOR_SHIFT);
}

std::uint64_t bridgeSamplerFloatPair(float low, float high) {
    return static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(low))
        | (static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(high)) << 32);
}

struct RenderPassState {
    VkDevice device = VK_NULL_HANDLE;
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
};

struct FramebufferState {
    VkDevice device = VK_NULL_HANDLE;
    RenderPassState* renderPass = nullptr;
    ImageState* colorImage = nullptr;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
};

struct PipelineCacheState {
    VkDevice device = VK_NULL_HANDLE;
};

struct ShaderModuleState {
    VkDevice device = VK_NULL_HANDLE;
    bool isAddUInt32 = false;
    bool isTriangleVertex = false;
    bool isTriangleFragment = false;
    bool usesFloat64Matrix = false;
    std::uint64_t hash = 0;
    std::vector<std::uint32_t> code;
};

struct DescriptorSetLayoutState {
    VkDevice device = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
};

struct PipelineLayoutState {
    VkDevice device = VK_NULL_HANDLE;
    std::vector<DescriptorSetLayoutState*> setLayouts;
    std::vector<VkPushConstantRange> pushConstantRanges;
};

struct PipelineState {
    VkDevice device = VK_NULL_HANDLE;
    PipelineLayoutState* layout = nullptr;
    bool isAddUInt32 = false;
    bool usesFloat64Matrix = false;
    bool requiresSoftwareFloat64Execution = false;
    std::uint64_t computeHash = 0;
    std::uint64_t bridgeComputePipelineID = 0;
    bool graphics = false;
    bool isFixedTriangle = false;
    bool isKitUI = false;
    bool rayTracingNV = false;
    bool rayTracingKHR = false;
    bool isMetalRayProbe = false;
    std::vector<ShaderModuleState*> rayTracingShaders;
    std::vector<VkShaderStageFlagBits> rayTracingStages;
    std::vector<VkRayTracingShaderGroupCreateInfoNV> rayTracingGroups;
    std::uint32_t maxRecursionDepth = 0;
    std::uint64_t rayTracingHash = 0;
    std::vector<PipelineState*> rayTracingLibraries;
};

struct AccelerationStructureNVState {
    VkDevice device = VK_NULL_HANDLE;
    VkDeviceSize compactedSize = 0;
    VkAccelerationStructureTypeNV type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_NV;
    VkBuildAccelerationStructureFlagsNV flags = 0;
    std::uint32_t instanceCount = 0;
    std::vector<VkGeometryNV> geometries;
    VkDeviceSize objectSize = 0;
    VkDeviceSize buildScratchSize = 0;
    VkDeviceSize updateScratchSize = 0;
    DeviceMemoryState* memory = nullptr;
    VkDeviceSize memoryOffset = 0;
    std::uint64_t opaqueHandle = 0;
    bool built = false;
};

struct AccelerationStructureKHRState {
    VkDevice device = VK_NULL_HANDLE;
    BufferState* buffer = nullptr;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    VkAccelerationStructureTypeKHR type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VkDeviceAddress deviceAddress = 0;
    std::uint64_t bridgeResourceID = 0;
    std::uint64_t triangleCount = 0;
    std::array<float, 3> triangleBoundsMinimum{};
    std::array<float, 3> triangleBoundsMaximum{};
    bool hasTriangleBounds = false;
    bool built = false;
    bool builtFromSceneState = false;
    std::uint64_t sceneStateSequence = 0;
    std::uint64_t sceneStateMeshSequence = 0;
};

struct DeferredOperationKHRState {
    VkDevice device = VK_NULL_HANDLE;
    VkResult result = VK_SUCCESS;
};

struct DescriptorPoolState {
    VkDevice device = VK_NULL_HANDLE;
    std::uint32_t maxSets = 0;
    std::uint32_t allocatedSets = 0;
};

struct DescriptorSetState {
    struct BufferBinding {
        BufferState* buffer = nullptr;
        VkDeviceSize offset = 0;
        VkDeviceSize range = 0;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
        VkFormat format = VK_FORMAT_UNDEFINED;
    };

    struct ImageBinding {
        ImageState* image = nullptr;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
    };

    VkDevice device = VK_NULL_HANDLE;
    DescriptorPoolState* pool = nullptr;
    DescriptorSetLayoutState* layout = nullptr;
    BufferState* buffer = nullptr;
    VkDeviceSize offset = 0;
    VkDeviceSize range = 0;
    std::unordered_map<std::uint32_t, ImageState*> sampledImages;
    std::unordered_map<std::uint64_t, ImageState*> storageImages;
    std::unordered_map<std::uint64_t, AccelerationStructureKHRState*> accelerationStructuresKHR;
    std::unordered_map<std::uint64_t, BufferBinding> computeBuffers;
    std::unordered_map<std::uint64_t, ImageBinding> computeImages;
    std::unordered_map<std::uint64_t, SamplerState*> computeSamplers;
};

struct CommandPoolState {
    VkDevice device = VK_NULL_HANDLE;
};

struct RecordedUIDraw {
    ImageState* texture = nullptr;
    std::uint32_t indexCount = 0;
    std::uint32_t firstIndex = 0;
    std::int32_t vertexOffset = 0;
    VkRect2D scissor{};
};

struct RecordedAccelerationStructureBuildNV {
    AccelerationStructureNVState* destination = nullptr;
    AccelerationStructureNVState* source = nullptr;
    BufferState* instanceData = nullptr;
    VkDeviceSize instanceOffset = 0;
    BufferState* scratch = nullptr;
    VkDeviceSize scratchOffset = 0;
    bool update = false;
};

struct RecordedAccelerationStructureCopyNV {
    AccelerationStructureNVState* destination = nullptr;
    AccelerationStructureNVState* source = nullptr;
    VkCopyAccelerationStructureModeNV mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_NV;
};

struct RecordedAccelerationStructureBuildKHR {
    std::uint64_t sequence = 0;
    AccelerationStructureKHRState* destination = nullptr;
    AccelerationStructureKHRState* source = nullptr;
    VkAccelerationStructureTypeKHR type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    VkBuildAccelerationStructureFlagsKHR flags = 0;
    VkBuildAccelerationStructureModeKHR mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    VkDeviceAddress scratchAddress = 0;
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
};

struct RecordedAccelerationStructureCopyKHR {
    std::uint64_t sequence = 0;
    AccelerationStructureKHRState* destination = nullptr;
    AccelerationStructureKHRState* source = nullptr;
    VkCopyAccelerationStructureModeKHR mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_CLONE_KHR;
};

struct RecordedRayTraceNV {
    PipelineState* pipeline = nullptr;
    BufferState* raygenShaderBindingTable = nullptr;
    BufferState* missShaderBindingTable = nullptr;
    BufferState* hitShaderBindingTable = nullptr;
    BufferState* callableShaderBindingTable = nullptr;
    VkDeviceSize raygenOffset = 0;
    VkDeviceSize missOffset = 0;
    VkDeviceSize missStride = 0;
    VkDeviceSize hitOffset = 0;
    VkDeviceSize hitStride = 0;
    VkDeviceSize callableOffset = 0;
    VkDeviceSize callableStride = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
};

struct RecordedRayTraceKHR {
    PipelineState* pipeline = nullptr;
    VkStridedDeviceAddressRegionKHR raygen{};
    VkStridedDeviceAddressRegionKHR miss{};
    VkStridedDeviceAddressRegionKHR hit{};
    VkStridedDeviceAddressRegionKHR callable{};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::array<DescriptorSetState*, 32> descriptorSets{};
};

struct RecordedComputeDispatch {
    std::uint64_t sequence = 0;
    PipelineState* pipeline = nullptr;
    bool bridgeEligible = false;
    std::array<DescriptorSetState*, 32> descriptorSets{};
    std::uint32_t groupCountX = 0;
    std::uint32_t groupCountY = 0;
    std::uint32_t groupCountZ = 0;
    imb::Bytes pushConstants;
};

struct RecordedBufferCopy {
    std::uint64_t sequence = 0;
    BufferState* source = nullptr;
    BufferState* destination = nullptr;
    std::vector<VkBufferCopy> regions;
};

struct RecordedBufferUpdate {
    std::uint64_t sequence = 0;
    BufferState* destination = nullptr;
    VkDeviceSize offset = 0;
    imb::Bytes data;
};

struct RecordedBufferFill {
    std::uint64_t sequence = 0;
    BufferState* destination = nullptr;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    std::uint32_t data = 0;
};

struct RecordedImageClear {
    std::uint64_t sequence = 0;
    ImageState* image = nullptr;
    VkClearColorValue color{};
    std::vector<VkImageSubresourceRange> ranges;
};

struct RecordedImageCopy {
    std::uint64_t sequence = 0;
    ImageState* source = nullptr;
    ImageState* destination = nullptr;
    std::vector<VkImageCopy> regions;
};

struct RecordedBufferToImageCopy {
    std::uint64_t sequence = 0;
    BufferState* source = nullptr;
    ImageState* destination = nullptr;
    std::vector<VkBufferImageCopy> regions;
};

struct RecordedImageToBufferCopy {
    std::uint64_t sequence = 0;
    ImageState* source = nullptr;
    BufferState* destination = nullptr;
    std::vector<VkBufferImageCopy> regions;
};

struct CommandBufferState {
    VK_LOADER_DATA loaderData{};
    VkDevice device = VK_NULL_HANDLE;
    CommandPoolState* pool = nullptr;
    bool recording = false;
    bool executable = false;
    PipelineState* pipeline = nullptr;
    DescriptorSetState* descriptorSet = nullptr;
    std::uint32_t addend = 0;
    bool hasPushConstant = false;
    imb::Bytes computePushConstants;
    std::uint32_t groupCountX = 0;
    FramebufferState* framebuffer = nullptr;
    bool insideRenderPass = false;
    std::uint32_t clearRGBA8 = 0;
    bool drewTriangle = false;
    std::array<DescriptorSetState*, 2> graphicsDescriptorSets{};
    std::array<DescriptorSetState*, 32> computeDescriptorSets{};
    std::array<DescriptorSetState*, 32> rayTracingDescriptorSets{};
    BufferState* vertexBuffer = nullptr;
    VkDeviceSize vertexBufferOffset = 0;
    BufferState* indexBuffer = nullptr;
    VkDeviceSize indexBufferOffset = 0;
    VkIndexType indexType = VK_INDEX_TYPE_UINT16;
    std::uint32_t textureIndex = 0;
    VkRect2D scissor{};
    std::vector<RecordedUIDraw> uiDraws;
    std::vector<RecordedAccelerationStructureBuildNV> accelerationStructureBuildsNV;
    std::vector<RecordedAccelerationStructureCopyNV> accelerationStructureCopiesNV;
    std::vector<RecordedRayTraceNV> rayTracesNV;
    std::vector<RecordedAccelerationStructureBuildKHR> accelerationStructureBuildsKHR;
    std::vector<RecordedAccelerationStructureCopyKHR> accelerationStructureCopiesKHR;
    std::vector<RecordedRayTraceKHR> rayTracesKHR;
    std::vector<RecordedComputeDispatch> computeDispatches;
    std::uint64_t nextCommandSequence = 1;
    std::vector<RecordedBufferCopy> bufferCopies;
    std::vector<RecordedBufferUpdate> bufferUpdates;
    std::vector<RecordedBufferFill> bufferFills;
    std::vector<RecordedImageClear> imageClears;
    std::vector<RecordedImageCopy> imageCopies;
    std::vector<RecordedBufferToImageCopy> bufferToImageCopies;
    std::vector<RecordedImageToBufferCopy> imageToBufferCopies;
};

struct FenceState {
    VkDevice device = VK_NULL_HANDLE;
    std::uint64_t bridgeFenceID = 0;
    DeviceMemoryState* resultMemory = nullptr;
    ImageState* resultImage = nullptr;
    bool submitted = false;
    bool signaled = false;
};

struct SemaphoreState {
    VkDevice device = VK_NULL_HANDLE;
    bool signaled = false;
    bool timeline = false;
    std::uint64_t value = 0;
    VkExternalSemaphoreHandleTypeFlags exportHandleTypes = 0;
    // OPAQUE_FD synchronization payload shared with the CUDA compatibility
    // library. The first uint64_t stores 0/1 for binary semaphores and the
    // monotonically increasing counter for timeline semaphores.
    int externalFD = -1;
};

struct QueryPoolState {
    VkDevice device = VK_NULL_HANDLE;
    VkQueryType type = VK_QUERY_TYPE_OCCLUSION;
    std::uint32_t count = 0;
    VkQueryPipelineStatisticFlags pipelineStatistics = 0;
    std::vector<std::uint64_t> values;
    std::vector<bool> available;
};

struct DriverState {
    std::recursive_mutex mutex;
    std::unique_ptr<BridgeConnection> bridge;
    std::unordered_set<VkInstance> instances;
    std::unordered_map<VkInstance, VkPhysicalDevice> physicalDevices;
    // Carbonite asks one Vulkan family for separate render, compute, copy, and
    // sparse-binding queues. Metal exposes one command queue to the bridge, so
    // these Vulkan handles are logical lanes serialized through the same host
    // connection and recursive driver lock.
    std::unordered_map<VkDevice, std::vector<VkQueue>> devices;
    std::unordered_set<DeviceMemoryState*> memories;
    std::unordered_set<BufferState*> buffers;
    std::unordered_set<BufferViewState*> bufferViews;
    std::unordered_set<ImageState*> images;
    std::unordered_set<ImageViewState*> imageViews;
    std::unordered_set<SamplerState*> samplers;
    std::unordered_set<RenderPassState*> renderPasses;
    std::unordered_set<FramebufferState*> framebuffers;
    std::unordered_set<ShaderModuleState*> shaderModules;
    std::unordered_set<DescriptorSetLayoutState*> descriptorSetLayouts;
    std::unordered_set<PipelineLayoutState*> pipelineLayouts;
    std::unordered_set<PipelineState*> pipelines;
    std::unordered_set<AccelerationStructureNVState*> accelerationStructuresNV;
    std::unordered_set<AccelerationStructureKHRState*> accelerationStructuresKHR;
    std::unordered_set<DeferredOperationKHRState*> deferredOperationsKHR;
    std::unordered_set<PipelineCacheState*> pipelineCaches;
    std::unordered_set<DescriptorPoolState*> descriptorPools;
    std::unordered_set<DescriptorSetState*> descriptorSets;
    std::unordered_set<CommandPoolState*> commandPools;
    std::unordered_set<CommandBufferState*> commandBuffers;
    std::unordered_set<FenceState*> fences;
    std::unordered_set<SemaphoreState*> semaphores;
    std::unordered_set<QueryPoolState*> queryPools;
    std::uint64_t nextAccelerationStructureHandle = UINT64_C(0x494d420000000001);
    VkDeviceAddress nextBufferDeviceAddress = UINT64_C(0x0000010000000000);
    std::uint64_t nextStorageDescriptorSequence = 1;
    // Exact local-space Mesh points and triangulated indices published by the
    // real USD stage use dedicated Metal resources.  They remain alive for
    // the lifetime of the device because the fallback TLAS references them.
    std::vector<BridgeSceneMetalMesh> sceneMetalMeshes;
    // Scene-state v13 transports every unique image once. Mirror that
    // deduplication in Metal so repeated USD Mesh/material occurrences share
    // one image resource and one shader texture slot.
    std::vector<BridgeSceneMetalTexture> sceneMetalTextures;
    // The RTX render graph writes the primary ray result into one pair of
    // RGBA images, then Kit presents a separate triple-buffered RGBA image in
    // its UI draw.  Keep the most recent real Metal result so that queue
    // submission can synchronize it into whichever presentation image Kit
    // selected for this frame.
    VkDevice latestMetalSceneDevice = VK_NULL_HANDLE;
    std::uint32_t latestMetalSceneWidth = 0;
    std::uint32_t latestMetalSceneHeight = 0;
    imb::Bytes latestMetalSceneRGBA8;
    bool emptyStageGridLoggedWithoutCamera = false;
    bool emptyStageGridLoggedWithCamera = false;
    VkDevice cameraSensorFrameDevice = VK_NULL_HANDLE;
    bool cameraSensorFrameAttempted = false;
    bool cameraSensorFramePublished = false;
};

DriverState gState;

std::uint64_t sceneTextureContentHash(const BridgeSceneTextureMap& texture) {
    if (!texture.rgba8) return 0;
    std::uint64_t hash = UINT64_C(14695981039346656037);
    const auto append = [&hash](const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= UINT64_C(1099511628211);
        }
    };
    append(&texture.width, sizeof(texture.width));
    append(&texture.height, sizeof(texture.height));
    append(texture.rgba8->data(), texture.rgba8->size());
    return hash;
}

std::uint64_t acquireSceneTextureResourceLocked(
    VkDevice device,
    const BridgeSceneTextureMap& texture
) {
    if (gState.bridge == nullptr || !texture.rgba8) return 0;
    const std::uint64_t contentHash = sceneTextureContentHash(texture);
    for (auto& cached : gState.sceneMetalTextures) {
        if (cached.device == device
            && cached.contentHash == contentHash
            && cached.width == texture.width
            && cached.height == texture.height
            && cached.rgba8 && *cached.rgba8 == *texture.rgba8) {
            ++cached.referenceCount;
            return cached.resourceID;
        }
    }
    const std::uint64_t resourceID = gState.bridge->createImage(
        texture.width, texture.height, IMB_IMAGE_FORMAT_RGBA8_UNORM
    );
    try {
        gState.bridge->writeImage(
            resourceID, texture.rgba8->data(), texture.rgba8->size()
        );
        gState.sceneMetalTextures.push_back(BridgeSceneMetalTexture{
            device, contentHash, texture.width, texture.height,
            texture.rgba8, resourceID, 1,
        });
    } catch (...) {
        try {
            gState.bridge->destroyBuffer(resourceID);
        } catch (const std::exception&) {
        }
        throw;
    }
    return resourceID;
}

void releaseSceneTextureResourceLocked(std::uint64_t resourceID) {
    if (resourceID == 0) return;
    const auto found = std::find_if(
        gState.sceneMetalTextures.begin(),
        gState.sceneMetalTextures.end(),
        [resourceID](const auto& cached) {
            return cached.resourceID == resourceID;
        }
    );
    if (found == gState.sceneMetalTextures.end()) return;
    if (found->referenceCount > 1) {
        --found->referenceCount;
        return;
    }
    if (gState.bridge != nullptr) {
        try {
            gState.bridge->destroyBuffer(found->resourceID);
        } catch (const std::exception&) {
        }
    }
    gState.sceneMetalTextures.erase(found);
}

struct BridgeSceneTextureReferenceGuard {
    std::vector<std::uint64_t> resourceIDs;

    ~BridgeSceneTextureReferenceGuard() {
        for (auto iterator = resourceIDs.rbegin();
             iterator != resourceIDs.rend(); ++iterator) {
            releaseSceneTextureResourceLocked(*iterator);
        }
    }
};

template <typename Handle>
Handle makeDispatchableHandle() {
    auto* data = new VK_LOADER_DATA{};
    set_loader_magic_value(data);
    return reinterpret_cast<Handle>(data);
}

template <typename Handle>
void destroyDispatchableHandle(Handle handle) {
    delete reinterpret_cast<VK_LOADER_DATA*>(handle);
}

template <typename Handle, typename State>
Handle makeObjectHandle(State* state) {
    return reinterpret_cast<Handle>(state);
}

template <typename State, typename Handle>
State* objectState(Handle handle) {
    return reinterpret_cast<State*>(handle);
}

bool validPhysicalDevice(VkPhysicalDevice physicalDevice) {
    return std::any_of(
        gState.physicalDevices.begin(),
        gState.physicalDevices.end(),
        [physicalDevice](const auto& entry) { return entry.second == physicalDevice; }
    );
}

std::uint32_t configuredVsockPort() {
    const char* value = std::getenv("IMB_VSOCK_PORT");
    if (value == nullptr || *value == '\0') {
        throw std::runtime_error("IMB_VSOCK_PORT is required");
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0 || parsed > UINT32_MAX) {
        throw std::runtime_error("IMB_VSOCK_PORT is invalid");
    }
    return static_cast<std::uint32_t>(parsed);
}

bool traceEnabled() {
    const char* value = std::getenv("IMB_VULKAN_TRACE");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

bool semaphoreTraceEnabled() {
    const char* value = std::getenv("IMB_VULKAN_SEMAPHORE_TRACE");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

constexpr std::size_t kDiagnosticSlotCount = 512;
constexpr std::size_t kDiagnosticNameSize = 96;
char gDiagnosticDeviceFunctionNames[kDiagnosticSlotCount][kDiagnosticNameSize] = {};

template <std::size_t Slot>
VKAPI_ATTR VkResult VKAPI_CALL diagnosticDeviceFunction(
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
            "imb-vulkan-icd: diagnostic call %s args=%#llx,%#llx,%#llx,%#llx\n",
            gDiagnosticDeviceFunctionNames[Slot],
            static_cast<unsigned long long>(argument0),
            static_cast<unsigned long long>(argument1),
            static_cast<unsigned long long>(argument2),
            static_cast<unsigned long long>(argument3)
        );
    }
    return VK_SUCCESS;
}

using DiagnosticDeviceFunction = VkResult (VKAPI_PTR *)(
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
constexpr std::array<DiagnosticDeviceFunction, sizeof...(Slots)> makeDiagnosticDeviceFunctions(
    std::index_sequence<Slots...>
) {
    return {&diagnosticDeviceFunction<Slots>...};
}

constexpr auto gDiagnosticDeviceFunctions = makeDiagnosticDeviceFunctions(
    std::make_index_sequence<kDiagnosticSlotCount>{}
);

bool endsWith(const char* value, const char* suffix) {
    const std::size_t valueLength = std::strlen(value);
    const std::size_t suffixLength = std::strlen(suffix);
    return valueLength >= suffixLength
        && std::memcmp(value + valueLength - suffixLength, suffix, suffixLength) == 0;
}

bool isCoreDeviceFunctionName(const char* name) {
    constexpr const char* vendorSuffixes[] = {
        "KHR", "EXT", "NV", "NVX", "AMD", "AMDX", "INTEL", "GOOGLE",
        "QCOM", "HUAWEI", "VALVE", "FUCHSIA", "ANDROID", "GGP", "MVK",
        "NN", "ARM", "SEC", "MESA", "MSFT",
    };
    for (const char* suffix : vendorSuffixes) {
        if (endsWith(name, suffix)) return false;
    }
    return std::strncmp(name, "vk", 2) == 0;
}

std::size_t diagnosticSlotFor(const char* name) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char* cursor = reinterpret_cast<const unsigned char*>(name); *cursor != 0; ++cursor) {
        hash ^= *cursor;
        hash *= 1099511628211ULL;
    }
    for (std::size_t probe = 0; probe < kDiagnosticSlotCount; ++probe) {
        const std::size_t slot = (static_cast<std::size_t>(hash) + probe) % kDiagnosticSlotCount;
        if (gDiagnosticDeviceFunctionNames[slot][0] == '\0') {
            std::snprintf(gDiagnosticDeviceFunctionNames[slot], kDiagnosticNameSize, "%s", name);
            return slot;
        }
        if (std::strcmp(gDiagnosticDeviceFunctionNames[slot], name) == 0) return slot;
    }
    return 0;
}

bool nvidiaCompatibilityIdentityEnabled() {
    const char* value = std::getenv("IMB_VULKAN_NVIDIA_COMPAT");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

bool spirvComputeBridgeEnabled() {
    const char* value = std::getenv("IMB_VULKAN_SPIRV_COMPUTE");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

bool genericComputeBridgeEnabled() {
    const char* value = std::getenv("IMB_VULKAN_GENERIC_COMPUTE");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

bool float64MatrixExecutionEnabled() {
    const char* value = std::getenv("IMB_VULKAN_FP64_MATRIX_EXECUTION");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

bool validatedSoftwareFloat64MatrixShader(std::uint64_t hash) {
    // Each exact hash below has been dispatched from its captured Isaac SPIR-V
    // on Apple M4 and had a deterministic Metal output checked. Keep the
    // allowlist exact; other software-FP64 matrix shaders remain compile-only
    // until each one has equivalent output validation.
    switch (hash) {
        case 0xf5dd5704d7491f17ULL:
            // World-transform output matrix with a decoded binary64 position.
            return true;
        case 0x64d94f5901ee0e4fULL:
            // Camera/depth reconstruction with two independent samplers and
            // an FP64-transformed clipping-plane pixel.
            return true;
        case 0x75cdce75f76c7184ULL:
            // Splat-record generation variant whose binary64 camera origin
            // changes four checked hierarchy counters through the exact-zero
            // clipping decision and real storage-buffer array writes.
            return true;
        case 0x0a553b2a8825d0ffULL:
            // Temporal-consistency variant with current-ray and four-neighbor
            // binary64 clipping loops. Both loops update only local maximum
            // ray lengths; the following reprojections use the original
            // sampled depths. A two-origin 3x3 dispatch checks identical final
            // consistency pixels after both loops complete.
            return true;
        case 0xd1b78c3914cb1874ULL:
            // Depth-consistency variant whose two clipping loops consume the
            // software-binary64 camera origin. Both clipped lengths are
            // output-dead by construction; a signed-neighbor 3x3 dispatch
            // checks unchanged depth plus completion after the second loop.
            return true;
        case 0x75321ea922defdfcULL:
            // Panoramic depth reconstruction whose software-FP64 transform
            // writes a checked world-space position pixel.
            return true;
        case 0x361fde9aceebb12bULL:
            // Panoramic depth reconstruction variant whose software-FP64
            // world position is checked through its storage-buffer output.
            return true;
        case 0x2eb5c9558c6d3e5aULL:
            // Motion-vector variant whose two software-FP64 world origins
            // produce a checked image-space position difference.
            return true;
        case 0x610a7b9aed9f7b98ULL:
            // Projection variant whose software-FP64 relative position length
            // is checked through its dedicated depth image.
            return true;
        case 0xb258118a36125408ULL:
            // Final-composite variant whose software-FP64 camera origin hits
            // a checked clipping-plane early-output pixel.
            return true;
        case 0xc93eebdfc4dd964bULL:
            // Volumetric integration variant whose binary64 camera origin
            // moves a real 3D noise lookup from a dark to a bright voxel and
            // writes a checked 3D storage-image pixel.
            return true;
        case 0xbdd2d21d53978c2eULL:
            // Volume-composite variant whose binary64 camera origin changes
            // the checked output from its scaled path to an exact clipping
            // early return while a real 3D texture remains bound.
            return true;
        case 0xd38fa4ca78558061ULL:
            // Render-output inspection variant whose mode-3 position path
            // writes software-binary64 origin changes as checked RGBA8 color.
            return true;
        case 0xf72cb1589be7be96ULL:
            // The final captured module uses two UInt64 compare-exchanges in
            // its hash-table insertion path. Metal serializes its independent
            // GlobalInvocationID-only workload on one thread; the exact
            // captured shader inserts one checked key and increments its real
            // insertion counter from zero to one on Apple M4.
            return true;
        default:
            return false;
    }
}

bool computeTraceEnabled() {
    const char* value = std::getenv("IMB_VULKAN_COMPUTE_TRACE");
    return traceEnabled() || (value != nullptr && std::strcmp(value, "0") != 0);
}

bool nullDescriptorBridgeEnabled() {
    const char* value = std::getenv("IMB_VULKAN_NULL_DESCRIPTOR");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

bool rayTracingTraceEnabled() {
    const char* value = std::getenv("IMB_VULKAN_RT_TRACE");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

bool scenePresentationEnabled() {
    // The current scene renderer is a diagnostic Metal hit/miss visualization,
    // not Kit's normal material renderer. Do not inject it into an ordinary
    // empty-stage viewport unless the launcher explicitly selected a
    // validation/environment stage.
    const char* value = std::getenv("IMB_VULKAN_SCENE_PRESENTATION");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool sceneGridPresentationEnabled() {
    const char* value = std::getenv("IMB_VULKAN_SCENE_GRID");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool emptyStageGridPresentationEnabled() {
    const char* value = std::getenv("IMB_VULKAN_EMPTY_STAGE_GRID");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool fullWorkspaceOnlyEnabled() {
    // Kit can keep an undocked Viewport window alive after Isaac Sim's Full
    // layout is restored. Both windows use the same UI graphics pipeline, but
    // the native bridge currently has one presentation surface. In Full mode
    // the launcher therefore asks us to present only the root that contains
    // the right and bottom dock regions of the normal Isaac workspace.
    const char* value = std::getenv("IMB_VULKAN_FULL_WORKSPACE_ONLY");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool fullWorkspaceLayoutReady() {
    const char* path = std::getenv("IMB_VULKAN_FULL_WORKSPACE_READY_FILE");
    return path == nullptr || *path == '\0' || ::access(path, F_OK) == 0;
}

bool uiPresentationTraceEnabled() {
    const char* value = std::getenv("IMB_VULKAN_UI_TRACE");
    return traceEnabled() || (value != nullptr && std::strcmp(value, "0") != 0);
}

bool uiSnapshotLayoutChangesEnabled() {
    const char* value = std::getenv("IMB_VULKAN_UI_SNAPSHOT_CHANGES");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

bool sparseImagesEnabled() {
    // The current protocol maps Vulkan image pages through a real Metal sparse heap.
    // The launcher enables this after both the dedicated tile probe and the
    // real Kit texture-streaming path completed through mip 0 without a crash.
    const char* value = std::getenv("IMB_VULKAN_SPARSE_IMAGES");
    return value != nullptr && std::strcmp(value, "1") == 0
        && gState.bridge != nullptr
        && (gState.bridge->capabilities().bits & IMB_CAP_METAL_SPARSE_IMAGE) != 0;
}

std::uint32_t bridgeImageFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8G8B8A8_UNORM:
        return IMB_IMAGE_FORMAT_RGBA8_UNORM;
    case VK_FORMAT_B8G8R8A8_UNORM:
        return IMB_IMAGE_FORMAT_BGRA8_UNORM;
    case VK_FORMAT_BC3_UNORM_BLOCK:
        return IMB_IMAGE_FORMAT_BC3_UNORM;
    case VK_FORMAT_BC3_SRGB_BLOCK:
        return IMB_IMAGE_FORMAT_BC3_SRGB;
    case VK_FORMAT_BC5_UNORM_BLOCK:
        return IMB_IMAGE_FORMAT_BC5_UNORM;
    case VK_FORMAT_R16_UNORM:
        return IMB_IMAGE_FORMAT_R16_UNORM;
    case VK_FORMAT_R16G16B16A16_UNORM:
        return IMB_IMAGE_FORMAT_RGBA16_UNORM;
    case VK_FORMAT_R8G8B8A8_UINT:
        return IMB_IMAGE_FORMAT_RGBA8_UINT;
    case VK_FORMAT_R8G8B8A8_SNORM:
        return IMB_IMAGE_FORMAT_RGBA8_SNORM;
    case VK_FORMAT_R8G8B8A8_SRGB:
        return IMB_IMAGE_FORMAT_RGBA8_SRGB;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
        return IMB_IMAGE_FORMAT_RGBA32_SFLOAT;
    default:
        return 0;
    }
}

bool queryMetalSparseImageProperties(
    VkFormat format,
    BridgeSparseImageProperties& properties
) {
    const auto bridgeFormat = bridgeImageFormat(format);
    if (!sparseImagesEnabled() || bridgeFormat == 0 || gState.bridge == nullptr) return false;
    try {
        properties = gState.bridge->querySparseImageProperties(bridgeFormat);
        return true;
    } catch (const std::exception& error) {
        if (rayTracingTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: Metal sparse image query failed format=%d: %s\n",
                format,
                error.what()
            );
        }
        return false;
    }
}

bool externalMemoryTraceEnabled() {
    const char* value = std::getenv("IMB_VULKAN_EXTERNAL_TRACE");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

std::uint32_t nvidiaCompatibilityFeatureMask() {
    if (!nvidiaCompatibilityIdentityEnabled()) return 0;
    const char* value = std::getenv("IMB_VULKAN_PROFILE_MASK");
    if (value == nullptr || *value == '\0') return 0;
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 0);
    if (end == value || *end != '\0') return 0;
    return static_cast<std::uint32_t>(parsed);
}

bool rayTracingAvailable() {
    return gState.bridge != nullptr
        && (gState.bridge->capabilities().bits & (IMB_CAP_RAY_TRACING | IMB_CAP_METAL_ACCELERATION_STRUCTURE))
            == (IMB_CAP_RAY_TRACING | IMB_CAP_METAL_ACCELERATION_STRUCTURE);
}

constexpr VkExtensionProperties kInstanceExtensions[] = {
    {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_SURFACE_SPEC_VERSION},
    {"VK_KHR_xlib_surface", 6},
    {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_SPEC_VERSION},
    {VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_SPEC_VERSION},
    {VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_SPEC_VERSION},
    {VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME, VK_KHR_EXTERNAL_FENCE_CAPABILITIES_SPEC_VERSION},
    {VK_KHR_DEVICE_GROUP_CREATION_EXTENSION_NAME, VK_KHR_DEVICE_GROUP_CREATION_SPEC_VERSION},
    {VK_EXT_DEBUG_UTILS_EXTENSION_NAME, VK_EXT_DEBUG_UTILS_SPEC_VERSION},
    {VK_EXT_DEBUG_REPORT_EXTENSION_NAME, VK_EXT_DEBUG_REPORT_SPEC_VERSION},
    {VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME, VK_KHR_GET_SURFACE_CAPABILITIES_2_SPEC_VERSION},
};

struct DeviceExtensionEntry {
    VkExtensionProperties properties;
    bool requiresRayTracing;
    bool requiresNvidiaCompatibility;
};

constexpr DeviceExtensionEntry kDeviceExtensions[] = {
    {{VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_SPEC_VERSION}, false, false},
    {{VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_SPEC_VERSION}, false, false},
    {{VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_SPEC_VERSION}, false, false},
    {{VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_SPEC_VERSION}, false, false},
    {{VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_GET_MEMORY_REQUIREMENTS_2_SPEC_VERSION}, false, false},
    {{VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_SPEC_VERSION}, false, false},
    {{VK_KHR_BIND_MEMORY_2_EXTENSION_NAME, VK_KHR_BIND_MEMORY_2_SPEC_VERSION}, false, false},
    {{VK_KHR_MAINTENANCE_3_EXTENSION_NAME, VK_KHR_MAINTENANCE_3_SPEC_VERSION}, false, false},
    {{VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME, VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_SPEC_VERSION}, false, false},
    {{VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, VK_EXT_DESCRIPTOR_INDEXING_SPEC_VERSION}, false, false},
    {{VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, VK_EXT_MEMORY_BUDGET_SPEC_VERSION}, false, false},
    {{VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_SPEC_VERSION}, false, false},

    // Isaac Sim 6.0.1's aarch64 shader cache was generated for the DGX Spark
    // Vulkan 1.3 profile.  Report the feature-bearing extensions that Kit
    // actually queries so ShaderDb selects those bundled SPIR-V variants.
    {{VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME, VK_EXT_SCALAR_BLOCK_LAYOUT_SPEC_VERSION}, false, true},
    {{VK_KHR_MAINTENANCE_4_EXTENSION_NAME, VK_KHR_MAINTENANCE_4_SPEC_VERSION}, false, true},
    {{VK_NV_OPTICAL_FLOW_EXTENSION_NAME, VK_NV_OPTICAL_FLOW_SPEC_VERSION}, false, true},
    {{VK_KHR_SHADER_DRAW_PARAMETERS_EXTENSION_NAME, VK_KHR_SHADER_DRAW_PARAMETERS_SPEC_VERSION}, false, true},
    {{VK_KHR_SHADER_CLOCK_EXTENSION_NAME, VK_KHR_SHADER_CLOCK_SPEC_VERSION}, false, true},
    {{VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, VK_KHR_SYNCHRONIZATION_2_SPEC_VERSION}, false, true},
    {{VK_KHR_8BIT_STORAGE_EXTENSION_NAME, VK_KHR_8BIT_STORAGE_SPEC_VERSION}, false, true},
    {{VK_KHR_16BIT_STORAGE_EXTENSION_NAME, VK_KHR_16BIT_STORAGE_SPEC_VERSION}, false, true},
    {{VK_NV_PRESENT_BARRIER_EXTENSION_NAME, VK_NV_PRESENT_BARRIER_SPEC_VERSION}, false, true},
    {{VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_SPEC_VERSION}, false, true},
    {{VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, VK_EXT_ROBUSTNESS_2_SPEC_VERSION}, false, true},
    {{VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME, VK_EXT_HOST_QUERY_RESET_SPEC_VERSION}, false, true},
    {{VK_NV_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME, VK_NV_DEVICE_GENERATED_COMMANDS_SPEC_VERSION}, false, true},
    {{VK_NV_REPRESENTATIVE_FRAGMENT_TEST_EXTENSION_NAME, VK_NV_REPRESENTATIVE_FRAGMENT_TEST_SPEC_VERSION}, false, true},
    {{VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, VK_EXT_SHADER_ATOMIC_FLOAT_SPEC_VERSION}, false, true},
    {{VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME, VK_KHR_SHADER_ATOMIC_INT64_SPEC_VERSION}, false, true},
    {{VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME, VK_KHR_SHADER_FLOAT16_INT8_SPEC_VERSION}, false, true},

    {{VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME, VK_KHR_SHADER_FLOAT_CONTROLS_SPEC_VERSION}, true, false},
    {{VK_KHR_SPIRV_1_4_EXTENSION_NAME, VK_KHR_SPIRV_1_4_SPEC_VERSION}, true, false},
    {{VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME, VK_KHR_BUFFER_DEVICE_ADDRESS_SPEC_VERSION}, true, false},
    {{VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_SPEC_VERSION}, true, false},
    {{VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, VK_KHR_PIPELINE_LIBRARY_SPEC_VERSION}, true, false},
    {{VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_SPEC_VERSION}, true, false},
    {{VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, VK_KHR_RAY_TRACING_PIPELINE_SPEC_VERSION}, true, false},
    {{VK_NV_RAY_TRACING_EXTENSION_NAME, VK_NV_RAY_TRACING_SPEC_VERSION}, true, false},
    {{VK_KHR_RAY_QUERY_EXTENSION_NAME, VK_KHR_RAY_QUERY_SPEC_VERSION}, true, true},
    {{VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME, VK_NV_RAY_TRACING_MOTION_BLUR_SPEC_VERSION}, true, true},
    {{VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, VK_NV_RAY_TRACING_INVOCATION_REORDER_SPEC_VERSION}, true, true},
};

bool deviceExtensionAvailable(const DeviceExtensionEntry& entry) {
    if (entry.requiresRayTracing && !rayTracingAvailable()) return false;
    if (entry.requiresNvidiaCompatibility && !nvidiaCompatibilityIdentityEnabled()) return false;
    return true;
}

bool supportedDeviceExtension(const char* name) {
    if (name == nullptr) return false;
    const auto found = std::find_if(
        std::begin(kDeviceExtensions),
        std::end(kDeviceExtensions),
        [name](const auto& entry) {
            return std::strcmp(name, entry.properties.extensionName) == 0;
        }
    );
    return found != std::end(kDeviceExtensions) && deviceExtensionAvailable(*found);
}

bool supportedInstanceExtension(const char* name) {
    return name != nullptr && std::any_of(
        std::begin(kInstanceExtensions),
        std::end(kInstanceExtensions),
        [name](const auto& extension) { return std::strcmp(name, extension.extensionName) == 0; }
    );
}

template <typename Function>
PFN_vkVoidFunction toVoidFunction(Function function) {
    return reinterpret_cast<PFN_vkVoidFunction>(function);
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkEnumerateInstanceExtensionProperties(
    const char* layerName,
    std::uint32_t* propertyCount,
    VkExtensionProperties* properties
) {
    if (propertyCount == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    if (layerName != nullptr) return VK_ERROR_LAYER_NOT_PRESENT;
    constexpr std::uint32_t available = static_cast<std::uint32_t>(std::size(kInstanceExtensions));
    if (properties == nullptr) {
        *propertyCount = available;
        return VK_SUCCESS;
    }
    const std::uint32_t written = std::min(*propertyCount, available);
    std::copy_n(kInstanceExtensions, written, properties);
    *propertyCount = written;
    return written < available ? VK_INCOMPLETE : VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkEnumerateInstanceLayerProperties(
    std::uint32_t* propertyCount,
    VkLayerProperties* properties
) {
    if (propertyCount == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    *propertyCount = 0;
    (void)properties;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkEnumerateInstanceVersion(std::uint32_t* apiVersion) {
    if (apiVersion == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    // Isaac Sim 6's bundled aarch64 shader cache is generated against the
    // modern Vulkan feature model exposed by its supported NVIDIA driver.
    // Advertising only 1.1 changes ShaderDb's platform key even when the same
    // KHR features are available through extensions.
    *apiVersion = VK_API_VERSION_1_3;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateInstance(
    const VkInstanceCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkInstance* instance
) {
    if (createInfo == nullptr || instance == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    if (traceEnabled()) {
        const std::uint32_t apiVersion = createInfo->pApplicationInfo == nullptr
            ? VK_API_VERSION_1_0
            : createInfo->pApplicationInfo->apiVersion;
        std::fprintf(
            stderr,
            "imb-vulkan-icd: vkCreateInstance api=%u.%u extensions=%u\n",
            VK_VERSION_MAJOR(apiVersion),
            VK_VERSION_MINOR(apiVersion),
            createInfo->enabledExtensionCount
        );
    }
    for (std::uint32_t index = 0; index < createInfo->enabledExtensionCount; ++index) {
        const char* extension = createInfo->ppEnabledExtensionNames[index];
        if (traceEnabled()) std::fprintf(stderr, "imb-vulkan-icd: instance extension %s\n", extension);
        if (!supportedInstanceExtension(extension)) return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    std::lock_guard lock(gState.mutex);
    if (!gState.instances.empty()) return VK_ERROR_TOO_MANY_OBJECTS;
    try {
        gState.bridge = std::make_unique<BridgeConnection>(configuredVsockPort());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: initialization failed: %s\n", error.what());
        gState.bridge.reset();
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    *instance = makeDispatchableHandle<VkInstance>();
    const auto physicalDevice = makeDispatchableHandle<VkPhysicalDevice>();
    gState.instances.insert(*instance);
    gState.physicalDevices[*instance] = physicalDevice;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyInstance(VkInstance instance, const VkAllocationCallbacks*) {
    std::lock_guard lock(gState.mutex);
    const auto physical = gState.physicalDevices.find(instance);
    if (physical != gState.physicalDevices.end()) {
        destroyDispatchableHandle(physical->second);
        gState.physicalDevices.erase(physical);
    }
    if (gState.instances.erase(instance) != 0) destroyDispatchableHandle(instance);
    if (gState.instances.empty() && gState.bridge) {
        gState.bridge->shutdown();
        gState.bridge.reset();
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkEnumeratePhysicalDevices(
    VkInstance instance,
    std::uint32_t* count,
    VkPhysicalDevice* devices
) {
    if (count == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    const auto found = gState.physicalDevices.find(instance);
    if (found == gState.physicalDevices.end()) return VK_ERROR_INITIALIZATION_FAILED;
    if (devices == nullptr) {
        *count = 1;
        return VK_SUCCESS;
    }
    if (*count == 0) return VK_INCOMPLETE;
    devices[0] = found->second;
    *count = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceFeatures(
    VkPhysicalDevice physicalDevice,
    VkPhysicalDeviceFeatures* features
) {
    std::lock_guard lock(gState.mutex);
    if (features == nullptr || !validPhysicalDevice(physicalDevice)) return;
    std::memset(features, 0, sizeof(*features));
    features->robustBufferAccess = VK_TRUE;
    features->fullDrawIndexUint32 = VK_TRUE;
    features->imageCubeArray = VK_TRUE;
    features->independentBlend = VK_TRUE;
    features->geometryShader = VK_TRUE;
    features->tessellationShader = VK_TRUE;
    features->sampleRateShading = VK_TRUE;
    features->dualSrcBlend = VK_TRUE;
    features->multiDrawIndirect = VK_TRUE;
    features->drawIndirectFirstInstance = VK_TRUE;
    features->depthClamp = VK_TRUE;
    features->depthBiasClamp = VK_TRUE;
    features->fillModeNonSolid = VK_TRUE;
    features->samplerAnisotropy = VK_TRUE;
    features->textureCompressionBC = VK_TRUE;
    features->shaderStorageImageExtendedFormats = VK_TRUE;
    features->shaderStorageImageWriteWithoutFormat = VK_TRUE;
    features->shaderInt16 = VK_TRUE;
    features->shaderInt64 = VK_TRUE;
    // RTX uses a large sparse buffer as a virtual-address arena for shader
    // binding tables.  The bridge backs only the pages bound through
    // vkQueueBindSparse rather than allocating the full multi-gigabyte arena.
    features->sparseBinding = VK_TRUE;
    features->sparseResidencyBuffer = VK_TRUE;
    features->sparseResidencyImage2D = sparseImagesEnabled() ? VK_TRUE : VK_FALSE;
    features->sparseResidencyImage3D = VK_FALSE;
    features->sparseResidency2Samples = VK_FALSE;
    features->sparseResidency4Samples = VK_FALSE;
    features->sparseResidency8Samples = VK_FALSE;
    // Metal sparse heaps do not expose Vulkan memory-offset alias semantics.
    features->sparseResidencyAliased = VK_FALSE;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: physical features shaderInt64=%u geometry=%u tessellation=%u\n",
            features->shaderInt64,
            features->geometryShader,
            features->tessellationShader
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceProperties(
    VkPhysicalDevice physicalDevice,
    VkPhysicalDeviceProperties* properties
) {
    std::lock_guard lock(gState.mutex);
    if (properties == nullptr || !validPhysicalDevice(physicalDevice) || !gState.bridge) return;
    std::memset(properties, 0, sizeof(*properties));
    properties->apiVersion = VK_API_VERSION_1_3;
    properties->driverVersion = nvidiaCompatibilityIdentityEnabled()
        // Isaac Sim 6.0's official aarch64 target is DGX Spark with the
        // 580.142 driver family.  ShaderDb selects pre-generated Vulkan
        // permutations from this compatibility identity, so keep both the
        // packed Vulkan value and VkPhysicalDeviceDriverProperties in sync.
        ? ((580U << 22) | (142U << 14))
        : VK_MAKE_VERSION(0, 1, 0);
    properties->vendorID = nvidiaCompatibilityIdentityEnabled() ? 0x10de : 0;
    properties->deviceID = nvidiaCompatibilityIdentityEnabled() ? 0xffff : 0;
    properties->deviceType = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    const std::string name = nvidiaCompatibilityIdentityEnabled()
        ? "IsaacMetalBridge NVIDIA-compat (" + gState.bridge->capabilities().deviceName + ")"
        : "IsaacMetalBridge (" + gState.bridge->capabilities().deviceName + ")";
    std::snprintf(properties->deviceName, sizeof(properties->deviceName), "%s", name.c_str());
    std::memcpy(properties->pipelineCacheUUID, kDeviceUUID, sizeof(kDeviceUUID));
    properties->limits.maxImageDimension1D = 16384;
    properties->limits.maxImageDimension2D = 16384;
    properties->limits.maxImageDimension3D = 2048;
    properties->limits.maxImageDimensionCube = 16384;
    properties->limits.maxImageArrayLayers = 2048;
    properties->limits.maxTexelBufferElements = 1 << 27;
    // Keep guest buffer-view offsets compatible with the strictest alignment
    // used by the M4 Metal buffer-backed texture formats exposed below.
    properties->limits.minTexelBufferOffsetAlignment = 256;
    properties->limits.maxUniformBufferRange = 1 << 27;
    properties->limits.maxStorageBufferRange = 1U << 30;
    properties->limits.maxPushConstantsSize = 256;
    properties->limits.maxMemoryAllocationCount = 4096;
    properties->limits.maxSamplerAllocationCount = 4096;
    properties->limits.maxBoundDescriptorSets = 32;
    properties->limits.maxPerStageDescriptorSamplers = 1024;
    properties->limits.maxPerStageDescriptorUniformBuffers = 1024;
    properties->limits.maxPerStageDescriptorStorageBuffers = 1024;
    properties->limits.maxPerStageDescriptorSampledImages = 1024;
    properties->limits.maxPerStageDescriptorStorageImages = 1024;
    properties->limits.maxPerStageDescriptorInputAttachments = 1024;
    properties->limits.maxPerStageResources = 4096;
    properties->limits.maxDescriptorSetSamplers = 4096;
    properties->limits.maxDescriptorSetUniformBuffers = 4096;
    properties->limits.maxDescriptorSetUniformBuffersDynamic = 1024;
    properties->limits.maxDescriptorSetStorageBuffers = 4096;
    properties->limits.maxDescriptorSetStorageBuffersDynamic = 1024;
    properties->limits.maxDescriptorSetSampledImages = 4096;
    properties->limits.maxDescriptorSetStorageImages = 4096;
    properties->limits.maxDescriptorSetInputAttachments = 4096;
    properties->limits.maxComputeWorkGroupCount[0] = 65535;
    properties->limits.maxComputeWorkGroupCount[1] = 65535;
    properties->limits.maxComputeWorkGroupCount[2] = 65535;
    properties->limits.maxComputeWorkGroupInvocations = 1024;
    properties->limits.maxComputeWorkGroupSize[0] = 1024;
    properties->limits.maxComputeWorkGroupSize[1] = 1024;
    properties->limits.maxComputeWorkGroupSize[2] = 64;
    // Sparse resources reserve virtual addresses and consume host/Metal
    // memory only for the pages bound through vkQueueBindSparse.  Keep this
    // comfortably above Isaac Sim's texture arenas without claiming that the
    // 32 GiB unified-memory machine has the same amount of resident memory.
    // A zero value is inconsistent with sparseBinding=VK_TRUE and causes
    // clients to treat every sparse resource as exceeding the device limit.
    properties->limits.sparseAddressSpaceSize = UINT64_C(1) << 40;
    properties->limits.nonCoherentAtomSize = 1;
    properties->limits.bufferImageGranularity = 1;
    properties->sparseProperties.residencyStandard2DBlockShape = VK_TRUE;
    properties->sparseProperties.residencyStandard2DMultisampleBlockShape = VK_FALSE;
    properties->sparseProperties.residencyStandard3DBlockShape = VK_FALSE;
    properties->sparseProperties.residencyAlignedMipSize = VK_FALSE;
    properties->sparseProperties.residencyNonResidentStrict = VK_FALSE;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: physical properties name=%s api=%#x vendor=%#x device=%#x type=%d driver=%#x\n",
            properties->deviceName,
            properties->apiVersion,
            properties->vendorID,
            properties->deviceID,
            properties->deviceType,
            properties->driverVersion
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceMemoryProperties(
    VkPhysicalDevice physicalDevice,
    VkPhysicalDeviceMemoryProperties* properties
) {
    std::lock_guard lock(gState.mutex);
    if (properties == nullptr || !validPhysicalDevice(physicalDevice) || !gState.bridge) return;
    std::memset(properties, 0, sizeof(*properties));
    properties->memoryHeapCount = 1;
    const auto guestMemory = physicalMemoryBytes();
    properties->memoryHeaps[0].size = guestMemory > 0
        ? guestMemory
        : gState.bridge->capabilities().maxBufferLength;
    properties->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
    properties->memoryTypeCount = 4;
    properties->memoryTypes[0].heapIndex = 0;
    properties->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    properties->memoryTypes[1].heapIndex = 0;
    properties->memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    properties->memoryTypes[2].heapIndex = 0;
    properties->memoryTypes[2].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    properties->memoryTypes[3].heapIndex = 0;
    properties->memoryTypes[3].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceQueueFamilyProperties(
    VkPhysicalDevice physicalDevice,
    std::uint32_t* count,
    VkQueueFamilyProperties* properties
) {
    std::lock_guard lock(gState.mutex);
    if (count == nullptr || !validPhysicalDevice(physicalDevice)) return;
    if (properties == nullptr) {
        *count = 1;
        return;
    }
    if (*count == 0) return;
    std::memset(&properties[0], 0, sizeof(properties[0]));
    // One Metal command queue backs every advertised Vulkan queue capability.
    // Graphics commands are introduced incrementally, but Isaac/Kit requires a
    // graphics-capable family before it will create the logical device.
    properties[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT
        | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT;
    properties[0].queueCount = kAdvertisedQueueCount;
    properties[0].minImageTransferGranularity = {1, 1, 1};
    *count = 1;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceFormatProperties(
    VkPhysicalDevice,
    VkFormat format,
    VkFormatProperties* properties
) {
    if (properties == nullptr) return;
    std::memset(properties, 0, sizeof(*properties));
    if (format == VK_FORMAT_UNDEFINED) return;
    constexpr VkFormatFeatureFlags imageFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT
        | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT
        | VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT
        | VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT
        | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    properties->linearTilingFeatures = imageFeatures;
    properties->optimalTilingFeatures = imageFeatures;
    properties->bufferFeatures = VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT;
    if (texelBufferFormatSupported(format)) {
        properties->bufferFeatures |= VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT
            | VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetPhysicalDeviceImageFormatProperties(
    VkPhysicalDevice,
    VkFormat format,
    VkImageType imageType,
    VkImageTiling,
    VkImageUsageFlags,
    VkImageCreateFlags,
    VkImageFormatProperties* properties
) {
    if (properties == nullptr || format == VK_FORMAT_UNDEFINED) return VK_ERROR_FORMAT_NOT_SUPPORTED;
    std::memset(properties, 0, sizeof(*properties));
    properties->maxExtent = imageType == VK_IMAGE_TYPE_3D
        ? VkExtent3D{2048, 2048, 2048}
        : VkExtent3D{16384, 16384, 1};
    properties->maxMipLevels = 15;
    properties->maxArrayLayers = 2048;
    properties->sampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_2_BIT
        | VK_SAMPLE_COUNT_4_BIT | VK_SAMPLE_COUNT_8_BIT;
    properties->maxResourceSize = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceSparseImageFormatProperties(
    VkPhysicalDevice physicalDevice,
    VkFormat format,
    VkImageType imageType,
    VkSampleCountFlagBits samples,
    VkImageUsageFlags,
    VkImageTiling tiling,
    std::uint32_t* count,
    VkSparseImageFormatProperties* properties
) {
    if (count == nullptr) return;
    std::lock_guard lock(gState.mutex);
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: sparse image format query format=%d type=%d samples=%#x tiling=%d output=%d requested=%u\n",
            format,
            imageType,
            samples,
            tiling,
            properties != nullptr ? 1 : 0,
            *count
        );
    }
    BridgeSparseImageProperties metalProperties;
    const bool supported = sparseImagesEnabled()
        && validPhysicalDevice(physicalDevice)
        && imageType == VK_IMAGE_TYPE_2D
        && tiling == VK_IMAGE_TILING_OPTIMAL
        && samples == VK_SAMPLE_COUNT_1_BIT
        && queryMetalSparseImageProperties(format, metalProperties);
    if (!supported) {
        *count = 0;
        return;
    }
    if (properties == nullptr) {
        *count = 1;
        return;
    }
    if (*count == 0) return;
    properties[0].aspectMask = formatAspectMask(format);
    properties[0].imageGranularity = sparseImageGranularity(format, imageType);
    properties[0].flags = 0;
    *count = 1;
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: sparse image format result aspect=%#x granularity=%ux%ux%u count=1\n",
            properties[0].aspectMask,
            properties[0].imageGranularity.width,
            properties[0].imageGranularity.height,
            properties[0].imageGranularity.depth
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroySurfaceKHR(
    VkInstance,
    VkSurfaceKHR,
    const VkAllocationCallbacks*
) {
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetPhysicalDeviceSurfaceSupportKHR(
    VkPhysicalDevice physicalDevice,
    std::uint32_t queueFamilyIndex,
    VkSurfaceKHR,
    VkBool32* supported
) {
    if (supported == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    if (!validPhysicalDevice(physicalDevice) || queueFamilyIndex != 0) return VK_ERROR_INITIALIZATION_FAILED;
    *supported = VK_FALSE;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
    VkPhysicalDevice,
    VkSurfaceKHR,
    VkSurfaceCapabilitiesKHR*
) {
    return VK_ERROR_SURFACE_LOST_KHR;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetPhysicalDeviceSurfaceFormatsKHR(
    VkPhysicalDevice physicalDevice,
    VkSurfaceKHR,
    std::uint32_t* count,
    VkSurfaceFormatKHR*
) {
    if (count == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    if (!validPhysicalDevice(physicalDevice)) return VK_ERROR_INITIALIZATION_FAILED;
    *count = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetPhysicalDeviceSurfacePresentModesKHR(
    VkPhysicalDevice physicalDevice,
    VkSurfaceKHR,
    std::uint32_t* count,
    VkPresentModeKHR*
) {
    if (count == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    if (!validPhysicalDevice(physicalDevice)) return VK_ERROR_INITIALIZATION_FAILED;
    *count = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateXlibSurfaceKHR(
    VkInstance,
    const void*,
    const VkAllocationCallbacks*,
    VkSurfaceKHR* surface
) {
    if (surface != nullptr) *surface = VK_NULL_HANDLE;
    return VK_ERROR_SURFACE_LOST_KHR;
}

VKAPI_ATTR VkBool32 VKAPI_CALL imb_vkGetPhysicalDeviceXlibPresentationSupportKHR(
    VkPhysicalDevice,
    std::uint32_t,
    void*,
    unsigned long
) {
    return VK_FALSE;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceFeatures2(
    VkPhysicalDevice physicalDevice,
    VkPhysicalDeviceFeatures2* features
) {
    if (features == nullptr) return;
    imb_vkGetPhysicalDeviceFeatures(physicalDevice, &features->features);
    const std::uint32_t profileMask = nvidiaCompatibilityFeatureMask();
    const bool layoutFeatures = (profileMask & (1U << 0)) != 0;
    const bool numericFeatures = (profileMask & (1U << 1)) != 0;
    const bool synchronizationFeatures = (profileMask & (1U << 2)) != 0;
    const bool shaderFeatures = (profileMask & (1U << 3)) != 0;
    const bool nvidiaUtilityFeatures = (profileMask & (1U << 4)) != 0;
    const bool advancedRayTracingFeatures = (profileMask & (1U << 5)) != 0;
    const bool rayTracing = rayTracingAvailable();
    auto* next = reinterpret_cast<VkBaseOutStructure*>(features->pNext);
    while (next != nullptr) {
        if (traceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: vkGetPhysicalDeviceFeatures2 pNext sType=%d\n",
                static_cast<int>(next->sType)
            );
        }
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES) {
            auto* scalar = reinterpret_cast<VkPhysicalDeviceScalarBlockLayoutFeatures*>(next);
            scalar->scalarBlockLayout = layoutFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_FEATURES) {
            auto* maintenance = reinterpret_cast<VkPhysicalDeviceMaintenance4Features*>(next);
            maintenance->maintenance4 = layoutFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV) {
            auto* opticalFlow = reinterpret_cast<VkPhysicalDeviceOpticalFlowFeaturesNV*>(next);
            opticalFlow->opticalFlow = nvidiaUtilityFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES) {
            auto* drawParameters = reinterpret_cast<VkPhysicalDeviceShaderDrawParametersFeatures*>(next);
            drawParameters->shaderDrawParameters = layoutFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV) {
            auto* reorder = reinterpret_cast<VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV*>(next);
            reorder->rayTracingInvocationReorder = advancedRayTracingFeatures && rayTracing ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR) {
            auto* clock = reinterpret_cast<VkPhysicalDeviceShaderClockFeaturesKHR*>(next);
            clock->shaderSubgroupClock = shaderFeatures ? VK_TRUE : VK_FALSE;
            clock->shaderDeviceClock = shaderFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES) {
            auto* synchronization = reinterpret_cast<VkPhysicalDeviceSynchronization2Features*>(next);
            synchronization->synchronization2 = synchronizationFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES) {
            auto* storage = reinterpret_cast<VkPhysicalDevice8BitStorageFeatures*>(next);
            storage->storageBuffer8BitAccess = numericFeatures ? VK_TRUE : VK_FALSE;
            storage->uniformAndStorageBuffer8BitAccess = numericFeatures ? VK_TRUE : VK_FALSE;
            storage->storagePushConstant8 = numericFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES) {
            auto* storage = reinterpret_cast<VkPhysicalDevice16BitStorageFeatures*>(next);
            storage->storageBuffer16BitAccess = numericFeatures ? VK_TRUE : VK_FALSE;
            storage->uniformAndStorageBuffer16BitAccess = numericFeatures ? VK_TRUE : VK_FALSE;
            storage->storagePushConstant16 = numericFeatures ? VK_TRUE : VK_FALSE;
            storage->storageInputOutput16 = numericFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_BARRIER_FEATURES_NV) {
            auto* presentBarrier = reinterpret_cast<VkPhysicalDevicePresentBarrierFeaturesNV*>(next);
            presentBarrier->presentBarrier = nvidiaUtilityFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR) {
            auto* barycentric = reinterpret_cast<VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR*>(next);
            barycentric->fragmentShaderBarycentric = shaderFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV) {
            auto* motionBlur = reinterpret_cast<VkPhysicalDeviceRayTracingMotionBlurFeaturesNV*>(next);
            motionBlur->rayTracingMotionBlur = advancedRayTracingFeatures && rayTracing ? VK_TRUE : VK_FALSE;
            motionBlur->rayTracingMotionBlurPipelineTraceRaysIndirect = VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT) {
            auto* robustness = reinterpret_cast<VkPhysicalDeviceRobustness2FeaturesEXT*>(next);
            robustness->robustBufferAccess2 = synchronizationFeatures ? VK_TRUE : VK_FALSE;
            robustness->robustImageAccess2 = synchronizationFeatures ? VK_TRUE : VK_FALSE;
            // Isaac RTX can intentionally pass a null TLAS instance descriptor
            // while the scene is empty.  Keep that narrow behavior independent
            // from robustness2 and synchronization2, whose command paths are not
            // yet generally implemented by the bridge.
            robustness->nullDescriptor = nullDescriptorBridgeEnabled() && rayTracing
                ? VK_TRUE
                : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR) {
            auto* query = reinterpret_cast<VkPhysicalDeviceRayQueryFeaturesKHR*>(next);
            query->rayQuery = advancedRayTracingFeatures && rayTracing ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES) {
            auto* hostQuery = reinterpret_cast<VkPhysicalDeviceHostQueryResetFeatures*>(next);
            hostQuery->hostQueryReset = synchronizationFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_NV) {
            auto* generatedCommands = reinterpret_cast<VkPhysicalDeviceDeviceGeneratedCommandsFeaturesNV*>(next);
            generatedCommands->deviceGeneratedCommands = nvidiaUtilityFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_REPRESENTATIVE_FRAGMENT_TEST_FEATURES_NV) {
            auto* representative = reinterpret_cast<VkPhysicalDeviceRepresentativeFragmentTestFeaturesNV*>(next);
            representative->representativeFragmentTest = shaderFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT) {
            auto* atomicFloat = reinterpret_cast<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT*>(next);
            atomicFloat->shaderBufferFloat32Atomics = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->shaderBufferFloat32AtomicAdd = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->shaderBufferFloat64Atomics = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->shaderBufferFloat64AtomicAdd = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->shaderSharedFloat32Atomics = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->shaderSharedFloat32AtomicAdd = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->shaderSharedFloat64Atomics = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->shaderSharedFloat64AtomicAdd = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->shaderImageFloat32Atomics = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->shaderImageFloat32AtomicAdd = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->sparseImageFloat32Atomics = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicFloat->sparseImageFloat32AtomicAdd = numericFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES) {
            auto* atomicInt64 = reinterpret_cast<VkPhysicalDeviceShaderAtomicInt64Features*>(next);
            atomicInt64->shaderBufferInt64Atomics = numericFeatures ? VK_TRUE : VK_FALSE;
            atomicInt64->shaderSharedInt64Atomics = numericFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES) {
            auto* float16Int8 = reinterpret_cast<VkPhysicalDeviceShaderFloat16Int8Features*>(next);
            float16Int8->shaderFloat16 = numericFeatures ? VK_TRUE : VK_FALSE;
            float16Int8->shaderInt8 = numericFeatures ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES) {
            auto* indexing = reinterpret_cast<VkPhysicalDeviceDescriptorIndexingFeatures*>(next);
            indexing->shaderInputAttachmentArrayDynamicIndexing = VK_TRUE;
            indexing->shaderUniformTexelBufferArrayDynamicIndexing = VK_TRUE;
            indexing->shaderStorageTexelBufferArrayDynamicIndexing = VK_TRUE;
            indexing->shaderUniformBufferArrayNonUniformIndexing = VK_TRUE;
            indexing->shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
            indexing->shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
            indexing->shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
            indexing->shaderInputAttachmentArrayNonUniformIndexing = VK_TRUE;
            indexing->shaderUniformTexelBufferArrayNonUniformIndexing = VK_TRUE;
            indexing->shaderStorageTexelBufferArrayNonUniformIndexing = VK_TRUE;
            indexing->descriptorBindingUniformBufferUpdateAfterBind = VK_TRUE;
            indexing->descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
            indexing->descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
            indexing->descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
            indexing->descriptorBindingUniformTexelBufferUpdateAfterBind = VK_TRUE;
            indexing->descriptorBindingStorageTexelBufferUpdateAfterBind = VK_TRUE;
            indexing->descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
            indexing->descriptorBindingPartiallyBound = VK_TRUE;
            indexing->descriptorBindingVariableDescriptorCount = VK_TRUE;
            indexing->runtimeDescriptorArray = VK_TRUE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES) {
            auto* timeline = reinterpret_cast<VkPhysicalDeviceTimelineSemaphoreFeatures*>(next);
            timeline->timelineSemaphore = VK_TRUE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES) {
            auto* address = reinterpret_cast<VkPhysicalDeviceBufferDeviceAddressFeatures*>(next);
            address->bufferDeviceAddress = rayTracing ? VK_TRUE : VK_FALSE;
            address->bufferDeviceAddressCaptureReplay = VK_FALSE;
            address->bufferDeviceAddressMultiDevice = VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR) {
            auto* acceleration = reinterpret_cast<VkPhysicalDeviceAccelerationStructureFeaturesKHR*>(next);
            acceleration->accelerationStructure = rayTracing ? VK_TRUE : VK_FALSE;
            acceleration->accelerationStructureCaptureReplay = VK_FALSE;
            acceleration->accelerationStructureIndirectBuild = VK_FALSE;
            acceleration->accelerationStructureHostCommands = VK_FALSE;
            acceleration->descriptorBindingAccelerationStructureUpdateAfterBind = rayTracing ? VK_TRUE : VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR) {
            auto* rayTracing = reinterpret_cast<VkPhysicalDeviceRayTracingPipelineFeaturesKHR*>(next);
            rayTracing->rayTracingPipeline = rayTracingAvailable() ? VK_TRUE : VK_FALSE;
            rayTracing->rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE;
            rayTracing->rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE;
            rayTracing->rayTracingPipelineTraceRaysIndirect = VK_FALSE;
            rayTracing->rayTraversalPrimitiveCulling = rayTracingAvailable() ? VK_TRUE : VK_FALSE;
        }
        next = next->pNext;
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceProperties2(
    VkPhysicalDevice physicalDevice,
    VkPhysicalDeviceProperties2* properties
) {
    if (properties == nullptr) return;
    imb_vkGetPhysicalDeviceProperties(physicalDevice, &properties->properties);
    auto* next = reinterpret_cast<VkBaseOutStructure*>(properties->pNext);
    while (next != nullptr) {
        if (traceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: vkGetPhysicalDeviceProperties2 pNext sType=%d\n",
                static_cast<int>(next->sType)
            );
        }
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES) {
            auto* identity = reinterpret_cast<VkPhysicalDeviceIDProperties*>(next);
            std::memcpy(identity->deviceUUID, kDeviceUUID, sizeof(kDeviceUUID));
            std::memcpy(identity->driverUUID, kDeviceUUID, sizeof(kDeviceUUID));
            std::memset(identity->deviceLUID, 0, sizeof(identity->deviceLUID));
            identity->deviceNodeMask = 0;
            identity->deviceLUIDValid = VK_FALSE;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES) {
            auto* driver = reinterpret_cast<VkPhysicalDeviceDriverProperties*>(next);
            const bool nvidiaCompatibility = nvidiaCompatibilityIdentityEnabled();
            driver->driverID = nvidiaCompatibility ? VK_DRIVER_ID_NVIDIA_PROPRIETARY : VK_DRIVER_ID_MOLTENVK;
            std::snprintf(
                driver->driverName,
                sizeof(driver->driverName),
                "%s",
                nvidiaCompatibility ? "NVIDIA" : "IsaacMetalBridge Metal ICD"
            );
            // gpu.foundation parses NVIDIA driverInfo as a dotted numeric
            // version.  Keep the compatibility identity in that exact ABI
            // shape; explanatory prose here causes its std::stoi parser to
            // abort before logical-device creation.
            std::snprintf(
                driver->driverInfo,
                sizeof(driver->driverInfo),
                "%s",
                nvidiaCompatibility ? "580.142.00" : "Vulkan to Apple Metal bridge"
            );
            driver->conformanceVersion = {1, 1, 0, 0};
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES) {
            auto* maintenance = reinterpret_cast<VkPhysicalDeviceMaintenance3Properties*>(next);
            maintenance->maxPerSetDescriptors = 16384;
            maintenance->maxMemoryAllocationSize = 8ULL * 1024ULL * 1024ULL * 1024ULL;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES) {
            auto* indexing = reinterpret_cast<VkPhysicalDeviceDescriptorIndexingProperties*>(next);
            indexing->maxUpdateAfterBindDescriptorsInAllPools = 16384;
            indexing->shaderUniformBufferArrayNonUniformIndexingNative = VK_TRUE;
            indexing->shaderSampledImageArrayNonUniformIndexingNative = VK_TRUE;
            indexing->shaderStorageBufferArrayNonUniformIndexingNative = VK_TRUE;
            indexing->shaderStorageImageArrayNonUniformIndexingNative = VK_TRUE;
            indexing->shaderInputAttachmentArrayNonUniformIndexingNative = VK_TRUE;
            indexing->robustBufferAccessUpdateAfterBind = VK_TRUE;
            indexing->quadDivergentImplicitLod = VK_TRUE;
            indexing->maxPerStageDescriptorUpdateAfterBindSamplers = 4096;
            indexing->maxPerStageDescriptorUpdateAfterBindUniformBuffers = 4096;
            indexing->maxPerStageDescriptorUpdateAfterBindStorageBuffers = 4096;
            indexing->maxPerStageDescriptorUpdateAfterBindSampledImages = 4096;
            indexing->maxPerStageDescriptorUpdateAfterBindStorageImages = 4096;
            indexing->maxPerStageDescriptorUpdateAfterBindInputAttachments = 4096;
            indexing->maxPerStageUpdateAfterBindResources = 16384;
            indexing->maxDescriptorSetUpdateAfterBindSamplers = 4096;
            indexing->maxDescriptorSetUpdateAfterBindUniformBuffers = 4096;
            indexing->maxDescriptorSetUpdateAfterBindUniformBuffersDynamic = 1024;
            indexing->maxDescriptorSetUpdateAfterBindStorageBuffers = 4096;
            indexing->maxDescriptorSetUpdateAfterBindStorageBuffersDynamic = 1024;
            indexing->maxDescriptorSetUpdateAfterBindSampledImages = 4096;
            indexing->maxDescriptorSetUpdateAfterBindStorageImages = 4096;
            indexing->maxDescriptorSetUpdateAfterBindInputAttachments = 4096;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES) {
            auto* timeline = reinterpret_cast<VkPhysicalDeviceTimelineSemaphoreProperties*>(next);
            timeline->maxTimelineSemaphoreValueDifference = std::numeric_limits<std::uint64_t>::max();
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PROPERTIES_NV) {
            auto* rayTracing = reinterpret_cast<VkPhysicalDeviceRayTracingPropertiesNV*>(next);
            rayTracing->shaderGroupHandleSize = 32;
            rayTracing->maxRecursionDepth = kMaxRayRecursionDepth;
            rayTracing->maxShaderGroupStride = 4096;
            rayTracing->shaderGroupBaseAlignment = 64;
            rayTracing->maxGeometryCount = 1ULL << 24;
            rayTracing->maxInstanceCount = 1ULL << 24;
            rayTracing->maxTriangleCount = 1ULL << 29;
            rayTracing->maxDescriptorSetAccelerationStructures = 4096;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR) {
            auto* acceleration = reinterpret_cast<VkPhysicalDeviceAccelerationStructurePropertiesKHR*>(next);
            acceleration->maxGeometryCount = (1ULL << 24) - 1;
            acceleration->maxInstanceCount = (1ULL << 24) - 1;
            acceleration->maxPrimitiveCount = (1ULL << 29) - 1;
            acceleration->maxPerStageDescriptorAccelerationStructures = 4096;
            acceleration->maxPerStageDescriptorUpdateAfterBindAccelerationStructures = 4096;
            acceleration->maxDescriptorSetAccelerationStructures = 4096;
            acceleration->maxDescriptorSetUpdateAfterBindAccelerationStructures = 4096;
            acceleration->minAccelerationStructureScratchOffsetAlignment = 256;
        } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR) {
            auto* rayTracing = reinterpret_cast<VkPhysicalDeviceRayTracingPipelinePropertiesKHR*>(next);
            rayTracing->shaderGroupHandleSize = 32;
            rayTracing->maxRayRecursionDepth = kMaxRayRecursionDepth;
            rayTracing->maxShaderGroupStride = 4096;
            rayTracing->shaderGroupBaseAlignment = 64;
            rayTracing->shaderGroupHandleCaptureReplaySize = 32;
            rayTracing->maxRayDispatchInvocationCount = 1U << 30;
            rayTracing->shaderGroupHandleAlignment = 32;
            rayTracing->maxRayHitAttributeSize = 32;
        }
        next = next->pNext;
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceFormatProperties2(
    VkPhysicalDevice physicalDevice,
    VkFormat format,
    VkFormatProperties2* properties
) {
    if (properties != nullptr) {
        imb_vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &properties->formatProperties);
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetPhysicalDeviceImageFormatProperties2(
    VkPhysicalDevice physicalDevice,
    const VkPhysicalDeviceImageFormatInfo2* imageFormatInfo,
    VkImageFormatProperties2* imageFormatProperties
) {
    if (imageFormatInfo == nullptr || imageFormatProperties == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    const auto* inputNext = reinterpret_cast<const VkBaseInStructure*>(imageFormatInfo->pNext);
    const VkPhysicalDeviceExternalImageFormatInfo* externalInfo = nullptr;
    while (inputNext != nullptr) {
        if (inputNext->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO) {
            externalInfo = reinterpret_cast<const VkPhysicalDeviceExternalImageFormatInfo*>(inputNext);
            break;
        }
        inputNext = inputNext->pNext;
    }
    if (externalInfo != nullptr
        && externalInfo->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
        if (externalMemoryTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: external image handle type %#x unsupported\n",
                externalInfo->handleType
            );
        }
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }

    const VkResult result = imb_vkGetPhysicalDeviceImageFormatProperties(
        physicalDevice,
        imageFormatInfo->format,
        imageFormatInfo->type,
        imageFormatInfo->tiling,
        imageFormatInfo->usage,
        imageFormatInfo->flags,
        &imageFormatProperties->imageFormatProperties
    );
    if (result != VK_SUCCESS) return result;

    auto* outputNext = reinterpret_cast<VkBaseOutStructure*>(imageFormatProperties->pNext);
    while (outputNext != nullptr) {
        if (outputNext->sType == VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES) {
            auto* externalProperties = reinterpret_cast<VkExternalImageFormatProperties*>(outputNext);
            std::memset(
                &externalProperties->externalMemoryProperties,
                0,
                sizeof(externalProperties->externalMemoryProperties)
            );
            if (externalInfo != nullptr) {
                externalProperties->externalMemoryProperties.externalMemoryFeatures =
                    VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT
                    | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
                externalProperties->externalMemoryProperties.exportFromImportedHandleTypes =
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
                externalProperties->externalMemoryProperties.compatibleHandleTypes =
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
            }
        }
        outputNext = outputNext->pNext;
    }
    if (externalMemoryTraceEnabled() && externalInfo != nullptr) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: external image OPAQUE_FD compatible format=%d type=%d usage=%#x flags=%#x\n",
            imageFormatInfo->format,
            imageFormatInfo->type,
            imageFormatInfo->usage,
            imageFormatInfo->flags
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceQueueFamilyProperties2(
    VkPhysicalDevice physicalDevice,
    std::uint32_t* count,
    VkQueueFamilyProperties2* properties
) {
    if (count == nullptr) return;
    if (properties == nullptr) {
        imb_vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, count, nullptr);
        return;
    }
    std::uint32_t coreCount = *count;
    VkQueueFamilyProperties coreProperties{};
    imb_vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &coreCount, &coreProperties);
    if (coreCount != 0 && *count != 0) properties[0].queueFamilyProperties = coreProperties;
    *count = coreCount;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceMemoryProperties2(
    VkPhysicalDevice physicalDevice,
    VkPhysicalDeviceMemoryProperties2* properties
) {
    if (properties == nullptr) return;
    std::lock_guard lock(gState.mutex);
    imb_vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties->memoryProperties);
    auto* next = reinterpret_cast<VkBaseOutStructure*>(properties->pNext);
    while (next != nullptr) {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT) {
            auto* budget = reinterpret_cast<VkPhysicalDeviceMemoryBudgetPropertiesEXT*>(next);
            for (std::uint32_t index = 0; index < VK_MAX_MEMORY_HEAPS; ++index) {
                budget->heapBudget[index] = 0;
                budget->heapUsage[index] = 0;
            }
            const VkDeviceSize heapSize = properties->memoryProperties.memoryHeaps[0].size;
            VkDeviceSize allocated = 0;
            for (const auto* memory : gState.memories) {
                if (memory == nullptr || allocated >= heapSize) continue;
                const VkDeviceSize remaining = heapSize - allocated;
                allocated += std::min(memory->size, remaining);
            }
            budget->heapBudget[0] = heapSize;
            budget->heapUsage[0] = allocated;
        }
        next = next->pNext;
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceSparseImageFormatProperties2(
    VkPhysicalDevice physicalDevice,
    const VkPhysicalDeviceSparseImageFormatInfo2* info,
    std::uint32_t* count,
    VkSparseImageFormatProperties2* properties
) {
    if (count == nullptr || info == nullptr) return;
    if (properties == nullptr) {
        imb_vkGetPhysicalDeviceSparseImageFormatProperties(
            physicalDevice,
            info->format,
            info->type,
            info->samples,
            info->usage,
            info->tiling,
            count,
            nullptr
        );
        return;
    }
    if (*count == 0) return;
    VkSparseImageFormatProperties core{};
    std::uint32_t coreCount = 1;
    imb_vkGetPhysicalDeviceSparseImageFormatProperties(
        physicalDevice,
        info->format,
        info->type,
        info->samples,
        info->usage,
        info->tiling,
        &coreCount,
        &core
    );
    if (coreCount != 0) properties[0].properties = core;
    *count = coreCount;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceExternalBufferProperties(
    VkPhysicalDevice physicalDevice,
    const VkPhysicalDeviceExternalBufferInfo* info,
    VkExternalBufferProperties* properties
) {
    if (properties == nullptr) return;
    std::memset(&properties->externalMemoryProperties, 0, sizeof(properties->externalMemoryProperties));
    std::lock_guard lock(gState.mutex);
    if (!validPhysicalDevice(physicalDevice) || info == nullptr
        || info->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
        return;
    }
    properties->externalMemoryProperties.externalMemoryFeatures =
        VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT;
    properties->externalMemoryProperties.exportFromImportedHandleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    properties->externalMemoryProperties.compatibleHandleTypes =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    if (externalMemoryTraceEnabled()) {
        std::fprintf(stderr, "imb-vulkan-icd: external buffer OPAQUE_FD compatible\n");
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceExternalFenceProperties(
    VkPhysicalDevice,
    const VkPhysicalDeviceExternalFenceInfo*,
    VkExternalFenceProperties* properties
) {
    if (properties != nullptr) {
        properties->exportFromImportedHandleTypes = 0;
        properties->compatibleHandleTypes = 0;
        properties->externalFenceFeatures = 0;
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetPhysicalDeviceExternalSemaphoreProperties(
    VkPhysicalDevice physicalDevice,
    const VkPhysicalDeviceExternalSemaphoreInfo* info,
    VkExternalSemaphoreProperties* properties
) {
    if (properties == nullptr) return;
    properties->exportFromImportedHandleTypes = 0;
    properties->compatibleHandleTypes = 0;
    properties->externalSemaphoreFeatures = 0;
    std::lock_guard lock(gState.mutex);
    if (!validPhysicalDevice(physicalDevice) || info == nullptr
        || info->handleType != VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
        return;
    }
    properties->exportFromImportedHandleTypes =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    properties->compatibleHandleTypes =
        VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    properties->externalSemaphoreFeatures =
        VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT
        | VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT;
    if (externalMemoryTraceEnabled() || rayTracingTraceEnabled()) {
        std::fprintf(stderr, "imb-vulkan-icd: external semaphore OPAQUE_FD compatible\n");
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkEnumeratePhysicalDeviceGroups(
    VkInstance instance,
    std::uint32_t* count,
    VkPhysicalDeviceGroupProperties* groups
) {
    if (count == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    const auto found = gState.physicalDevices.find(instance);
    if (found == gState.physicalDevices.end()) return VK_ERROR_INITIALIZATION_FAILED;
    if (groups == nullptr) {
        *count = 1;
        return VK_SUCCESS;
    }
    if (*count == 0) return VK_INCOMPLETE;
    groups[0].physicalDeviceCount = 1;
    groups[0].physicalDevices[0] = found->second;
    groups[0].subsetAllocation = VK_FALSE;
    *count = 1;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateDebugUtilsMessengerEXT(
    VkInstance,
    const VkDebugUtilsMessengerCreateInfoEXT*,
    const VkAllocationCallbacks*,
    VkDebugUtilsMessengerEXT* messenger
) {
    if (messenger == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    auto* token = new (std::nothrow) std::uint8_t{0};
    if (token == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *messenger = reinterpret_cast<VkDebugUtilsMessengerEXT>(token);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyDebugUtilsMessengerEXT(
    VkInstance,
    VkDebugUtilsMessengerEXT messenger,
    const VkAllocationCallbacks*
) {
    delete reinterpret_cast<std::uint8_t*>(messenger);
}

VKAPI_ATTR void VKAPI_CALL imb_vkSubmitDebugUtilsMessageEXT(
    VkInstance,
    VkDebugUtilsMessageSeverityFlagBitsEXT,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT*
) {
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateDebugReportCallbackEXT(
    VkInstance,
    const VkDebugReportCallbackCreateInfoEXT*,
    const VkAllocationCallbacks*,
    VkDebugReportCallbackEXT* callback
) {
    if (callback == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    auto* token = new (std::nothrow) std::uint8_t{0};
    if (token == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *callback = reinterpret_cast<VkDebugReportCallbackEXT>(token);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyDebugReportCallbackEXT(
    VkInstance,
    VkDebugReportCallbackEXT callback,
    const VkAllocationCallbacks*
) {
    delete reinterpret_cast<std::uint8_t*>(callback);
}

VKAPI_ATTR void VKAPI_CALL imb_vkDebugReportMessageEXT(
    VkInstance,
    VkDebugReportFlagsEXT,
    VkDebugReportObjectTypeEXT,
    std::uint64_t,
    std::size_t,
    std::int32_t,
    const char*,
    const char*
) {
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetPhysicalDeviceSurfaceCapabilities2KHR(
    VkPhysicalDevice,
    const VkPhysicalDeviceSurfaceInfo2KHR*,
    VkSurfaceCapabilities2KHR*
) {
    return VK_ERROR_SURFACE_LOST_KHR;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetPhysicalDeviceSurfaceFormats2KHR(
    VkPhysicalDevice physicalDevice,
    const VkPhysicalDeviceSurfaceInfo2KHR*,
    std::uint32_t* count,
    VkSurfaceFormat2KHR*
) {
    if (count == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    if (!validPhysicalDevice(physicalDevice)) return VK_ERROR_INITIALIZATION_FAILED;
    *count = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkEnumerateDeviceExtensionProperties(
    VkPhysicalDevice physicalDevice,
    const char* layerName,
    std::uint32_t* count,
    VkExtensionProperties* properties
) {
    if (count == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    if (layerName != nullptr) return VK_ERROR_LAYER_NOT_PRESENT;
    std::lock_guard lock(gState.mutex);
    if (!validPhysicalDevice(physicalDevice)) return VK_ERROR_INITIALIZATION_FAILED;
    const bool exposeRayTracing = rayTracingAvailable();
    const std::uint32_t available = static_cast<std::uint32_t>(std::count_if(
        std::begin(kDeviceExtensions),
        std::end(kDeviceExtensions),
        deviceExtensionAvailable
    ));
    if (properties == nullptr) {
        *count = available;
        if (traceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: enumerate device extensions count=%u rayTracing=%d\n",
                available,
                exposeRayTracing ? 1 : 0
            );
        }
        return VK_SUCCESS;
    }
    const std::uint32_t capacity = *count;
    std::uint32_t written = 0;
    for (const auto& entry : kDeviceExtensions) {
        if (!deviceExtensionAvailable(entry)) continue;
        if (written >= capacity) break;
        properties[written++] = entry.properties;
    }
    *count = written;
    if (traceEnabled()) {
        std::fprintf(stderr, "imb-vulkan-icd: enumerate device extensions wrote=%u", written);
        for (std::uint32_t index = 0; index < written; ++index) {
            std::fprintf(stderr, " %s", properties[index].extensionName);
        }
        std::fputc('\n', stderr);
    }
    return written < available ? VK_INCOMPLETE : VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkEnumerateDeviceLayerProperties(
    VkPhysicalDevice,
    std::uint32_t* count,
    VkLayerProperties* properties
) {
    if (count == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    *count = 0;
    (void)properties;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateDevice(
    VkPhysicalDevice physicalDevice,
    const VkDeviceCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkDevice* device
) {
    if (createInfo == nullptr || device == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: vkCreateDevice extensions=%u queueFamilies=%u\n",
            createInfo->enabledExtensionCount,
            createInfo->queueCreateInfoCount
        );
        for (std::uint32_t index = 0; index < createInfo->enabledExtensionCount; ++index) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: device extension %s\n",
                createInfo->ppEnabledExtensionNames[index]
            );
        }
        for (std::uint32_t index = 0; index < createInfo->queueCreateInfoCount; ++index) {
            const auto& queueInfo = createInfo->pQueueCreateInfos[index];
            std::fprintf(
                stderr,
                "imb-vulkan-icd: device queue family=%u count=%u\n",
                queueInfo.queueFamilyIndex,
                queueInfo.queueCount
            );
        }
        auto* next = reinterpret_cast<const VkBaseInStructure*>(createInfo->pNext);
        while (next != nullptr) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: vkCreateDevice pNext sType=%d\n",
                static_cast<int>(next->sType)
            );
            next = next->pNext;
        }
    }
    std::lock_guard lock(gState.mutex);
    if (!validPhysicalDevice(physicalDevice)) return VK_ERROR_INITIALIZATION_FAILED;
    if (createInfo->enabledExtensionCount != 0 && createInfo->ppEnabledExtensionNames == nullptr) {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
    for (std::uint32_t index = 0; index < createInfo->enabledExtensionCount; ++index) {
        if (!supportedDeviceExtension(createInfo->ppEnabledExtensionNames[index])) {
            return VK_ERROR_EXTENSION_NOT_PRESENT;
        }
    }
    if (createInfo->queueCreateInfoCount != 1
        || createInfo->pQueueCreateInfos == nullptr
        || createInfo->pQueueCreateInfos[0].queueFamilyIndex != 0
        || createInfo->pQueueCreateInfos[0].queueCount == 0
        || createInfo->pQueueCreateInfos[0].queueCount > kAdvertisedQueueCount) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    VkDevice newDevice = VK_NULL_HANDLE;
    std::vector<VkQueue> queues;
    try {
        newDevice = makeDispatchableHandle<VkDevice>();
        queues.reserve(createInfo->pQueueCreateInfos[0].queueCount);
        for (std::uint32_t index = 0;
             index < createInfo->pQueueCreateInfos[0].queueCount;
             ++index) {
            queues.push_back(makeDispatchableHandle<VkQueue>());
        }
        gState.devices.emplace(newDevice, std::move(queues));
    } catch (const std::bad_alloc&) {
        for (VkQueue queue : queues) destroyDispatchableHandle(queue);
        if (newDevice != VK_NULL_HANDLE) destroyDispatchableHandle(newDevice);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *device = newDevice;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyDevice(VkDevice device, const VkAllocationCallbacks*) {
    std::lock_guard lock(gState.mutex);
    const auto found = gState.devices.find(device);
    if (found == gState.devices.end()) return;
    for (auto iterator = gState.sceneMetalMeshes.begin();
         iterator != gState.sceneMetalMeshes.end();) {
        if (iterator->device != device) {
            ++iterator;
            continue;
        }
        if (gState.bridge != nullptr) {
            const std::array<std::uint64_t, 4> resourceIDs{
                iterator->accelerationStructureResourceID,
                iterator->materialDescriptorResourceID,
                iterator->indexBufferResourceID,
                iterator->vertexBufferResourceID,
            };
            for (const auto resourceID : resourceIDs) {
                if (resourceID == 0) continue;
                try {
                    gState.bridge->destroyBuffer(resourceID);
                } catch (const std::exception& error) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: USD Mesh Metal resource destroy warning: %s\n",
                        error.what()
                    );
                }
            }
            releaseSceneTextureResourceLocked(iterator->normalTextureResourceID);
            releaseSceneTextureResourceLocked(iterator->emissionTextureResourceID);
            releaseSceneTextureResourceLocked(iterator->metallicTextureResourceID);
            releaseSceneTextureResourceLocked(iterator->roughnessTextureResourceID);
            releaseSceneTextureResourceLocked(iterator->textureResourceID);
        }
        iterator = gState.sceneMetalMeshes.erase(iterator);
    }
    for (auto iterator = gState.accelerationStructuresNV.begin();
         iterator != gState.accelerationStructuresNV.end();) {
        auto* state = *iterator;
        if (state->device == device) {
            iterator = gState.accelerationStructuresNV.erase(iterator);
            delete state;
        } else {
            ++iterator;
        }
    }
    for (auto iterator = gState.accelerationStructuresKHR.begin();
         iterator != gState.accelerationStructuresKHR.end();) {
        auto* state = *iterator;
        if (state->device == device) {
            iterator = gState.accelerationStructuresKHR.erase(iterator);
            delete state;
        } else {
            ++iterator;
        }
    }
    for (auto iterator = gState.deferredOperationsKHR.begin();
         iterator != gState.deferredOperationsKHR.end();) {
        auto* state = *iterator;
        if (state->device == device) {
            iterator = gState.deferredOperationsKHR.erase(iterator);
            delete state;
        } else {
            ++iterator;
        }
    }
    for (VkQueue queue : found->second) destroyDispatchableHandle(queue);
    gState.devices.erase(found);
    if (gState.latestMetalSceneDevice == device) {
        gState.latestMetalSceneDevice = VK_NULL_HANDLE;
        gState.latestMetalSceneWidth = 0;
        gState.latestMetalSceneHeight = 0;
        gState.latestMetalSceneRGBA8.clear();
    }
    if (gState.cameraSensorFrameDevice == device) {
        gState.cameraSensorFrameDevice = VK_NULL_HANDLE;
        gState.cameraSensorFrameAttempted = false;
        gState.cameraSensorFramePublished = false;
    }
    destroyDispatchableHandle(device);
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetDeviceQueue(
    VkDevice device,
    std::uint32_t queueFamilyIndex,
    std::uint32_t queueIndex,
    VkQueue* queue
) {
    if (queue == nullptr) return;
    *queue = VK_NULL_HANDLE;
    if (queueFamilyIndex != 0) return;
    std::lock_guard lock(gState.mutex);
    const auto found = gState.devices.find(device);
    if (found != gState.devices.end() && queueIndex < found->second.size()) {
        *queue = found->second[queueIndex];
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetDeviceQueue2(
    VkDevice device,
    const VkDeviceQueueInfo2* queueInfo,
    VkQueue* queue
) {
    if (queue == nullptr) return;
    *queue = VK_NULL_HANDLE;
    if (queueInfo == nullptr
        || queueInfo->sType != VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2
        || queueInfo->flags != 0) {
        return;
    }
    imb_vkGetDeviceQueue(
        device,
        queueInfo->queueFamilyIndex,
        queueInfo->queueIndex,
        queue
    );
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: get device queue2 family=%u index=%u flags=%#x queue=%p\n",
            queueInfo->queueFamilyIndex,
            queueInfo->queueIndex,
            queueInfo->flags,
            static_cast<void*>(*queue)
        );
    }
}

bool validDevice(VkDevice device) {
    return gState.devices.contains(device);
}

VkDevice deviceForQueue(VkQueue queue) {
    const auto found = std::find_if(
        gState.devices.begin(),
        gState.devices.end(),
        [queue](const auto& entry) {
            return std::find(entry.second.begin(), entry.second.end(), queue)
                != entry.second.end();
        }
    );
    return found == gState.devices.end() ? VK_NULL_HANDLE : found->first;
}

VkResult ensureMemoryBackingLocked(DeviceMemoryState* memory, bool needsMetalResource) {
    if (memory == nullptr || !gState.memories.contains(memory)) return VK_ERROR_MEMORY_MAP_FAILED;
    try {
        if (memory->bytes.empty()) memory->bytes.resize(static_cast<std::size_t>(memory->size));
        if (needsMetalResource && memory->resourceID == 0) {
            if (!gState.bridge) return VK_ERROR_DEVICE_LOST;
            memory->resourceID = gState.bridge->createBuffer(memory->size);
        }
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: backing allocation failed: %s\n", error.what());
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return VK_SUCCESS;
}

VkResult syncMemoryToExternalFDLocked(
    DeviceMemoryState* memory,
    VkDeviceSize offset,
    VkDeviceSize size
) {
    if (memory == nullptr || memory->externalFD < 0 || size == 0) return VK_SUCCESS;
    const VkResult backingResult = ensureMemoryBackingLocked(memory, false);
    if (backingResult != VK_SUCCESS) return backingResult;
    if (offset > memory->size || size > memory->size - offset) return VK_ERROR_MEMORY_MAP_FAILED;
    VkDeviceSize written = 0;
    while (written < size) {
        const ssize_t count = ::pwrite(
            memory->externalFD,
            memory->bytes.data() + static_cast<std::size_t>(offset + written),
            static_cast<std::size_t>(size - written),
            static_cast<off_t>(offset + written)
        );
        if (count <= 0) return VK_ERROR_OUT_OF_HOST_MEMORY;
        written += static_cast<VkDeviceSize>(count);
    }
    return VK_SUCCESS;
}

VkResult refreshMemoryFromExternalFDLocked(
    DeviceMemoryState* memory,
    VkDeviceSize offset,
    VkDeviceSize size
) {
    if (memory == nullptr || memory->externalFD < 0 || size == 0) return VK_SUCCESS;
    const VkResult backingResult = ensureMemoryBackingLocked(memory, false);
    if (backingResult != VK_SUCCESS) return backingResult;
    if (offset > memory->size || size > memory->size - offset) return VK_ERROR_MEMORY_MAP_FAILED;
    VkDeviceSize read = 0;
    while (read < size) {
        const ssize_t count = ::pread(
            memory->externalFD,
            memory->bytes.data() + static_cast<std::size_t>(offset + read),
            static_cast<std::size_t>(size - read),
            static_cast<off_t>(offset + read)
        );
        if (count <= 0) return VK_ERROR_MEMORY_MAP_FAILED;
        read += static_cast<VkDeviceSize>(count);
    }
    return VK_SUCCESS;
}

VkResult ensureImageBackingLocked(ImageState* image) {
    const bool ordinary3D = image != nullptr
        && image->type == VK_IMAGE_TYPE_3D;
    if (image == nullptr || !gState.images.contains(image) || !gState.bridge
        || (image->format != VK_FORMAT_R8G8B8A8_UNORM
            && image->format != VK_FORMAT_B8G8R8A8_UNORM)
        || image->extent.width == 0 || image->extent.height == 0
        || image->extent.depth == 0
        // The established 2D UI path deliberately bridges only level zero of
        // mipmapped Kit textures. Preserve that compatibility. Ordinary 3D
        // resources, whose whole volume is transported as one tight payload,
        // require the exact one-mip/one-layer/single-sample shape.
        || (ordinary3D
            && (image->mipLevels != 1 || image->arrayLayers != 1
                || image->samples != VK_SAMPLE_COUNT_1_BIT))
        || (!ordinary3D && image->extent.depth != 1)) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    const std::uint64_t byteCount = static_cast<std::uint64_t>(image->extent.width)
        * image->extent.height * image->extent.depth * 4;
    if (byteCount > IMB_PROTOCOL_MAX_PAYLOAD) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    try {
        if (image->resourceID == 0) {
            const std::uint32_t format = image->format == VK_FORMAT_R8G8B8A8_UNORM
                ? IMB_IMAGE_FORMAT_RGBA8_UNORM
                : IMB_IMAGE_FORMAT_BGRA8_UNORM;
            image->resourceID = gState.bridge->createImage(
                image->extent.width,
                image->extent.height,
                format,
                image->extent.depth,
                ordinary3D
            );
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: image backing allocation failed: %s\n", error.what());
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return VK_SUCCESS;
}

VkResult ensureSparseImageBackingLocked(ImageState* image) {
    if (image == nullptr || !gState.images.contains(image) || !gState.bridge
        || (image->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) == 0
        || image->type != VK_IMAGE_TYPE_2D
        || image->samples != VK_SAMPLE_COUNT_1_BIT
        || image->sparseTileBytes == 0
        || image->extent.depth != 1
        || image->extent.width == 0 || image->extent.height == 0) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    const auto format = bridgeImageFormat(image->format);
    if (format == 0) return VK_ERROR_FORMAT_NOT_SUPPORTED;
    try {
        if (image->resourceID == 0) {
            image->resourceID = gState.bridge->createSparseImage(
                image->size,
                image->extent.width,
                image->extent.height,
                format,
                image->mipLevels,
                image->arrayLayers
            );
        }
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: sparse image backing allocation failed: %s\n",
            error.what()
        );
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return VK_SUCCESS;
}

std::size_t imageSubresourceIndex(
    const ImageState* image,
    std::uint32_t mipLevel,
    std::uint32_t arrayLayer
) {
    return static_cast<std::size_t>(arrayLayer) * image->mipLevels + mipLevel;
}

VkResult ensureLinearImageMipInitializedLocked(
    ImageState* image,
    std::uint32_t mipLevel,
    std::uint32_t arrayLayer
) {
    if (image == nullptr || !gState.images.contains(image) || image->memory == nullptr
        || mipLevel >= image->mipLevels || arrayLayer >= image->arrayLayers
        || image->samples != VK_SAMPLE_COUNT_1_BIT) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    const std::size_t subresource = imageSubresourceIndex(image, mipLevel, arrayLayer);
    if (subresource >= image->initializedSubresources.size()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (image->initializedSubresources[subresource]) return VK_SUCCESS;
    const VkResult backingResult = ensureMemoryBackingLocked(image->memory, false);
    if (backingResult != VK_SUCCESS) return backingResult;

    const auto block = formatBlockInfo(image->format);
    if (block.width != 1 || block.height != 1 || block.depth != 1 || block.bytes == 0) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    if (mipLevel == 0) {
        // Device memory is zero-initialized by the bridge. Treat an untouched
        // base level as a valid transparent/black source rather than reading
        // outside its backing range.
        image->initializedSubresources[subresource] = true;
        return VK_SUCCESS;
    }

    const VkResult sourceResult =
        ensureLinearImageMipInitializedLocked(image, mipLevel - 1, arrayLayer);
    if (sourceResult != VK_SUCCESS) return sourceResult;
    const VkExtent3D sourceExtent{
        std::max(1U, image->extent.width >> (mipLevel - 1)),
        std::max(1U, image->extent.height >> (mipLevel - 1)),
        std::max(1U, image->extent.depth >> (mipLevel - 1)),
    };
    const VkExtent3D destinationExtent{
        std::max(1U, image->extent.width >> mipLevel),
        std::max(1U, image->extent.height >> mipLevel),
        std::max(1U, image->extent.depth >> mipLevel),
    };
    const std::uint64_t sourceOffset = clampedAdd(
        image->memoryOffset,
        imageSubresourceByteOffset(
            image->format,
            image->extent,
            image->mipLevels,
            mipLevel - 1,
            arrayLayer
        )
    );
    const std::uint64_t destinationOffset = clampedAdd(
        image->memoryOffset,
        imageSubresourceByteOffset(
            image->format,
            image->extent,
            image->mipLevels,
            mipLevel,
            arrayLayer
        )
    );
    const std::uint64_t sourceBytes = mipByteSize(image->format, sourceExtent);
    const std::uint64_t destinationBytes = mipByteSize(image->format, destinationExtent);
    if (sourceOffset > image->memory->size
        || sourceBytes > image->memory->size - sourceOffset
        || destinationOffset > image->memory->size
        || destinationBytes > image->memory->size - destinationOffset) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    const auto* source = image->memory->bytes.data() + sourceOffset;
    auto* destination = image->memory->bytes.data() + destinationOffset;
    for (std::uint32_t z = 0; z < destinationExtent.depth; ++z) {
        const std::uint32_t sourceZ = std::min(z * 2, sourceExtent.depth - 1);
        for (std::uint32_t y = 0; y < destinationExtent.height; ++y) {
            const std::uint32_t sourceY = std::min(y * 2, sourceExtent.height - 1);
            for (std::uint32_t x = 0; x < destinationExtent.width; ++x) {
                const std::uint32_t sourceX = std::min(x * 2, sourceExtent.width - 1);
                std::memcpy(
                    destination
                        + ((static_cast<std::uint64_t>(z) * destinationExtent.height + y)
                            * destinationExtent.width + x) * block.bytes,
                    source
                        + ((static_cast<std::uint64_t>(sourceZ) * sourceExtent.height + sourceY)
                            * sourceExtent.width + sourceX) * block.bytes,
                    block.bytes
                );
            }
        }
    }
    image->initializedSubresources[subresource] = true;
    return VK_SUCCESS;
}

VkResult completeFenceLocked(FenceState* fence) {
    if (fence == nullptr || !gState.fences.contains(fence)) return VK_ERROR_DEVICE_LOST;
    if (fence->signaled) return VK_SUCCESS;
    if (!fence->submitted || fence->bridgeFenceID == 0
        || (fence->resultMemory == nullptr && fence->resultImage == nullptr) || !gState.bridge) {
        return VK_NOT_READY;
    }
    try {
        gState.bridge->waitFence(fence->bridgeFenceID);
        if (fence->resultImage != nullptr) {
            auto* image = fence->resultImage;
            if (image->memory == nullptr) return VK_ERROR_MEMORY_MAP_FAILED;
            const VkResult memoryResult = ensureMemoryBackingLocked(image->memory, false);
            if (memoryResult != VK_SUCCESS) return memoryResult;
            const std::uint64_t byteCount = static_cast<std::uint64_t>(image->extent.width)
                * image->extent.height * image->extent.depth * 4;
            if (image->memoryOffset > image->memory->size
                || byteCount > image->memory->size - image->memoryOffset) {
                return VK_ERROR_MEMORY_MAP_FAILED;
            }
            gState.bridge->readImage(
                image->resourceID,
                image->memory->bytes.data() + image->memoryOffset,
                byteCount
            );
        } else {
            gState.bridge->readBuffer(
                fence->resultMemory->resourceID,
                fence->resultMemory->bytes.data(),
                fence->resultMemory->size
            );
        }
        fence->signaled = true;
        return VK_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: fence wait failed: %s\n", error.what());
        return VK_ERROR_DEVICE_LOST;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateBuffer(
    VkDevice device,
    const VkBufferCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkBuffer* buffer
) {
    if (createInfo == nullptr || buffer == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)
        || (createInfo->sharingMode != VK_SHARING_MODE_EXCLUSIVE
            && createInfo->sharingMode != VK_SHARING_MODE_CONCURRENT)) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    auto* state = new (std::nothrow) BufferState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    state->size = std::max<VkDeviceSize>(1, createInfo->size);
    state->usage = createInfo->usage;
    state->flags = createInfo->flags;
    state->deviceAddress = gState.nextBufferDeviceAddress;
    const VkDeviceAddress addressSpan = (state->size + 4095) & ~VkDeviceAddress{4095};
    if (addressSpan <= std::numeric_limits<VkDeviceAddress>::max() - gState.nextBufferDeviceAddress) {
        gState.nextBufferDeviceAddress += std::max<VkDeviceAddress>(addressSpan, 4096);
    }
    if (traceEnabled() || (rayTracingTraceEnabled() && state->size >= (VkDeviceSize{1} << 30))) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: buffer handle=%p address=%#llx size=%llu usage=%#x flags=%#x sharing=%d\n",
            static_cast<void*>(state),
            static_cast<unsigned long long>(state->deviceAddress),
            static_cast<unsigned long long>(state->size),
            state->usage,
            createInfo->flags,
            createInfo->sharingMode
        );
    }
    gState.buffers.insert(state);
    *buffer = makeObjectHandle<VkBuffer>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyBuffer(
    VkDevice,
    VkBuffer buffer,
    const VkAllocationCallbacks*
) {
    if (buffer == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<BufferState>(buffer);
    if (gState.buffers.erase(state) != 0) delete state;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetBufferMemoryRequirements(
    VkDevice,
    VkBuffer buffer,
    VkMemoryRequirements* requirements
) {
    if (buffer == VK_NULL_HANDLE || requirements == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<BufferState>(buffer);
    if (!gState.buffers.contains(state)) return;
    const bool sparse = (state->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) != 0;
    const VkDeviceSize alignment = sparse ? VkDeviceSize{65536} : VkDeviceSize{4};
    requirements->size = (state->size + alignment - 1) & ~(alignment - 1);
    requirements->alignment = alignment;
    requirements->memoryTypeBits = 0x0f;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetBufferMemoryRequirements2(
    VkDevice device,
    const VkBufferMemoryRequirementsInfo2* info,
    VkMemoryRequirements2* requirements
) {
    if (info == nullptr || requirements == nullptr) return;
    imb_vkGetBufferMemoryRequirements(device, info->buffer, &requirements->memoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetDeviceBufferMemoryRequirements(
    VkDevice device,
    const VkDeviceBufferMemoryRequirements* info,
    VkMemoryRequirements2* requirements
) {
    if (info == nullptr || info->pCreateInfo == nullptr || requirements == nullptr) return;
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return;
    const auto* createInfo = info->pCreateInfo;
    const bool sparse = (createInfo->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) != 0;
    const VkDeviceSize alignment = sparse ? VkDeviceSize{65536} : VkDeviceSize{4};
    const VkDeviceSize size = std::max<VkDeviceSize>(1, createInfo->size);
    requirements->memoryRequirements.size = (size + alignment - 1) & ~(alignment - 1);
    requirements->memoryRequirements.alignment = alignment;
    requirements->memoryRequirements.memoryTypeBits = 0x0f;
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: maintenance4 buffer requirements size=%llu flags=%#x result=%llu alignment=%llu\n",
            static_cast<unsigned long long>(createInfo->size),
            createInfo->flags,
            static_cast<unsigned long long>(requirements->memoryRequirements.size),
            static_cast<unsigned long long>(alignment)
        );
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateImage(
    VkDevice device,
    const VkImageCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkImage* image
) {
    if (createInfo == nullptr || image == nullptr || createInfo->extent.width == 0
        || createInfo->extent.height == 0 || createInfo->extent.depth == 0
        || createInfo->arrayLayers == 0 || createInfo->mipLevels == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    const bool sparseResidency =
        (createInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) != 0;
    const VkDeviceSize byteSize = imageAllocationSize(
        createInfo->format,
        createInfo->imageType,
        createInfo->extent,
        createInfo->mipLevels,
        createInfo->arrayLayers,
        createInfo->samples,
        createInfo->flags
    );
    if (byteSize == UINT64_MAX) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    BridgeSparseImageProperties metalSparseProperties;
    if (sparseResidency) {
        if ((createInfo->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) == 0
            || createInfo->imageType != VK_IMAGE_TYPE_2D
            || createInfo->samples != VK_SAMPLE_COUNT_1_BIT
            || !queryMetalSparseImageProperties(createInfo->format, metalSparseProperties)) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: sparse image create rejected format=%d type=%d samples=%#x flags=%#x extent=%ux%ux%u mips=%u layers=%u\n",
                createInfo->format,
                createInfo->imageType,
                createInfo->samples,
                createInfo->flags,
                createInfo->extent.width,
                createInfo->extent.height,
                createInfo->extent.depth,
                createInfo->mipLevels,
                createInfo->arrayLayers
            );
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        const VkExtent3D expected = sparseImageGranularity(createInfo->format, createInfo->imageType);
        if (metalSparseProperties.tileWidth > UINT32_MAX / 2
            || metalSparseProperties.tileHeight > UINT32_MAX / 2
            || metalSparseProperties.tileWidth * 2 != expected.width
            || metalSparseProperties.tileHeight * 2 != expected.height
            || metalSparseProperties.tileDepth != expected.depth
            || metalSparseProperties.tileSizeBytes != kMetalSparseImageBlockBytes
            || metalSparseProperties.tileSizeBytes > UINT64_MAX / 4
            || metalSparseProperties.tileSizeBytes * 4 != kSparseImageBlockBytes) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: Metal sparse tile mismatch format=%d Vulkan=%ux%ux%u/%llu Metal=%ux%ux%u/%llu\n",
                createInfo->format,
                expected.width,
                expected.height,
                expected.depth,
                static_cast<unsigned long long>(kSparseImageBlockBytes),
                metalSparseProperties.tileWidth,
                metalSparseProperties.tileHeight,
                metalSparseProperties.tileDepth,
                static_cast<unsigned long long>(metalSparseProperties.tileSizeBytes)
            );
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
    }
    auto* state = new (std::nothrow) ImageState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    state->extent = createInfo->extent;
    state->format = createInfo->format;
    state->usage = createInfo->usage;
    state->flags = createInfo->flags;
    state->type = createInfo->imageType;
    state->tiling = createInfo->tiling;
    state->samples = createInfo->samples;
    state->mipLevels = createInfo->mipLevels;
    state->arrayLayers = createInfo->arrayLayers;
    state->size = byteSize;
    try {
        state->initializedSubresources.resize(
            static_cast<std::size_t>(createInfo->mipLevels)
                * createInfo->arrayLayers,
            false
        );
    } catch (const std::bad_alloc&) {
        delete state;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (sparseResidency) {
        state->sparseGranularity =
            sparseImageGranularity(createInfo->format, createInfo->imageType);
        state->sparseTileBytes = kSparseImageBlockBytes;
        state->metalSparseGranularity = {
            metalSparseProperties.tileWidth,
            metalSparseProperties.tileHeight,
            metalSparseProperties.tileDepth,
        };
        state->metalSparseTileBytes = metalSparseProperties.tileSizeBytes;
    }
    if (traceEnabled()
        || (rayTracingTraceEnabled()
            && (createInfo->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) != 0)) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: image handle=%p format=%d extent=%ux%ux%u usage=%#x tiling=%d size=%llu\n",
            static_cast<void*>(state),
            createInfo->format,
            createInfo->extent.width,
            createInfo->extent.height,
            createInfo->extent.depth,
            createInfo->usage,
            createInfo->tiling,
            static_cast<unsigned long long>(byteSize)
        );
    }
    gState.images.insert(state);
    *image = makeObjectHandle<VkImage>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyImage(
    VkDevice,
    VkImage image,
    const VkAllocationCallbacks*
) {
    if (image == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<ImageState>(image);
    if (gState.images.erase(state) == 0) return;
    if (gState.bridge && state->resourceID != 0) {
        try {
            gState.bridge->destroyBuffer(state->resourceID);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "imb-vulkan-icd: image destroy warning: %s\n", error.what());
        }
    }
    delete state;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetImageMemoryRequirements(
    VkDevice,
    VkImage image,
    VkMemoryRequirements* requirements
) {
    if (image == VK_NULL_HANDLE || requirements == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<ImageState>(image);
    if (!gState.images.contains(state)) return;
    requirements->size = state->size;
    requirements->alignment = (state->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) != 0
        ? kSparseImageBlockBytes
        : VkDeviceSize{256};
    requirements->memoryTypeBits = 0x0f;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetImageMemoryRequirements2(
    VkDevice device,
    const VkImageMemoryRequirementsInfo2* info,
    VkMemoryRequirements2* requirements
) {
    if (info == nullptr || requirements == nullptr) return;
    imb_vkGetImageMemoryRequirements(device, info->image, &requirements->memoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetDeviceImageMemoryRequirements(
    VkDevice device,
    const VkDeviceImageMemoryRequirements* info,
    VkMemoryRequirements2* requirements
) {
    if (info == nullptr || info->pCreateInfo == nullptr || requirements == nullptr) return;
    const auto* createInfo = info->pCreateInfo;
    const VkDeviceSize size = imageAllocationSize(
        createInfo->format,
        createInfo->imageType,
        createInfo->extent,
        createInfo->mipLevels,
        createInfo->arrayLayers,
        createInfo->samples,
        createInfo->flags
    );
    if (size == UINT64_MAX) {
        requirements->memoryRequirements = {};
        return;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return;
    requirements->memoryRequirements.size = size;
    requirements->memoryRequirements.alignment =
        (createInfo->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) != 0
        ? kSparseImageBlockBytes
        : VkDeviceSize{256};
    requirements->memoryRequirements.memoryTypeBits = 0x0f;
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: maintenance4 image requirements format=%d extent=%ux%ux%u mips=%u layers=%u flags=%#x size=%llu\n",
            createInfo->format,
            createInfo->extent.width,
            createInfo->extent.height,
            createInfo->extent.depth,
            createInfo->mipLevels,
            createInfo->arrayLayers,
            createInfo->flags,
            static_cast<unsigned long long>(size)
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetImageSparseMemoryRequirements(
    VkDevice device,
    VkImage image,
    std::uint32_t* count,
    VkSparseImageMemoryRequirements* requirements
) {
    if (count == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<ImageState>(image);
    const bool supported = sparseImagesEnabled()
        && validDevice(device) && gState.images.contains(state)
        && state->device == device
        && (state->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) != 0
        && state->sparseTileBytes != 0;
    if (!supported) {
        if (rayTracingTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: sparse image requirements unsupported image=%p known=%d flags=%#x\n",
                static_cast<void*>(state),
                gState.images.contains(state) ? 1 : 0,
                gState.images.contains(state) ? state->flags : 0
            );
        }
        *count = 0;
        return;
    }
    if (requirements == nullptr) {
        *count = 1;
        return;
    }
    if (*count == 0) return;
    const std::uint32_t capacity = *count;
    requirements[0] = sparseImageRequirements(
        state->format,
        state->type,
        state->extent,
        state->mipLevels,
        state->arrayLayers
    );
    *count = std::min(capacity, 1U);
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: sparse image requirements image=%p extent=%ux%ux%u mips=%u layers=%u size=%llu tailFirst=%u tailSize=%llu\n",
            static_cast<void*>(state),
            state->extent.width,
            state->extent.height,
            state->extent.depth,
            state->mipLevels,
            state->arrayLayers,
            static_cast<unsigned long long>(state->size),
            requirements[0].imageMipTailFirstLod,
            static_cast<unsigned long long>(requirements[0].imageMipTailSize)
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetImageSparseMemoryRequirements2(
    VkDevice device,
    const VkImageSparseMemoryRequirementsInfo2* info,
    std::uint32_t* count,
    VkSparseImageMemoryRequirements2* requirements
) {
    if (info == nullptr || count == nullptr) return;
    if (requirements == nullptr) {
        imb_vkGetImageSparseMemoryRequirements(device, info->image, count, nullptr);
        return;
    }
    if (*count == 0) return;
    const std::uint32_t capacity = *count;
    std::array<VkSparseImageMemoryRequirements, 1> core{};
    std::uint32_t coreCount = std::min(capacity, 1U);
    imb_vkGetImageSparseMemoryRequirements(device, info->image, &coreCount, core.data());
    for (std::uint32_t index = 0; index < coreCount; ++index) {
        requirements[index].memoryRequirements = core[index];
    }
    *count = coreCount;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetDeviceImageSparseMemoryRequirements(
    VkDevice device,
    const VkDeviceImageMemoryRequirements* info,
    std::uint32_t* count,
    VkSparseImageMemoryRequirements2* requirements
) {
    if (count == nullptr) return;
    std::lock_guard lock(gState.mutex);
    const auto* createInfo = info == nullptr ? nullptr : info->pCreateInfo;
    BridgeSparseImageProperties metalProperties;
    const bool supported = sparseImagesEnabled()
        && validDevice(device) && createInfo != nullptr
        && (createInfo->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) != 0
        && createInfo->imageType == VK_IMAGE_TYPE_2D
        && createInfo->samples == VK_SAMPLE_COUNT_1_BIT
        && queryMetalSparseImageProperties(createInfo->format, metalProperties);
    if (!supported) {
        *count = 0;
        return;
    }
    if (requirements == nullptr) {
        *count = 1;
        if (rayTracingTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: maintenance4 sparse image requirements query format=%d extent=%ux%ux%u mips=%u layers=%u count=1\n",
                createInfo->format,
                createInfo->extent.width,
                createInfo->extent.height,
                createInfo->extent.depth,
                createInfo->mipLevels,
                createInfo->arrayLayers
            );
        }
        return;
    }
    if (*count == 0) return;
    const std::uint32_t capacity = *count;
    auto& core = requirements[0].memoryRequirements;
    core = sparseImageRequirements(
        createInfo->format,
        createInfo->imageType,
        createInfo->extent,
        createInfo->mipLevels,
        createInfo->arrayLayers
    );
    *count = std::min(capacity, 1U);
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: maintenance4 sparse image requirements result format=%d aspect=%#x granularity=%ux%ux%u tailFirst=%u tailSize=%llu\n",
            createInfo->format,
            core.formatProperties.aspectMask,
            core.formatProperties.imageGranularity.width,
            core.formatProperties.imageGranularity.height,
            core.formatProperties.imageGranularity.depth,
            core.imageMipTailFirstLod,
            static_cast<unsigned long long>(core.imageMipTailSize)
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetImageSubresourceLayout(
    VkDevice,
    VkImage image,
    const VkImageSubresource* subresource,
    VkSubresourceLayout* layout
) {
    if (image == VK_NULL_HANDLE || subresource == nullptr || layout == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<ImageState>(image);
    if (!gState.images.contains(state) || subresource->mipLevel != 0 || subresource->arrayLayer != 0) return;
    std::memset(layout, 0, sizeof(*layout));
    layout->offset = 0;
    layout->rowPitch = static_cast<VkDeviceSize>(state->extent.width) * 4;
    layout->arrayPitch = layout->rowPitch * state->extent.height;
    layout->depthPitch = layout->arrayPitch;
    layout->size = layout->arrayPitch;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkBindImageMemory(
    VkDevice device,
    VkImage image,
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset
) {
    if (image == VK_NULL_HANDLE || memory == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    auto* imageState = objectState<ImageState>(image);
    auto* memoryState = objectState<DeviceMemoryState>(memory);
    if (!gState.images.contains(imageState) || !gState.memories.contains(memoryState)
        || imageState->device != device || memoryState->device != device
        || memoryOffset > memoryState->size || imageState->size > memoryState->size - memoryOffset) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    imageState->memory = memoryState;
    imageState->memoryOffset = memoryOffset;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: bind image=%p memory=%p offset=%llu\n",
            static_cast<void*>(imageState),
            static_cast<void*>(memoryState),
            static_cast<unsigned long long>(memoryOffset)
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkBindImageMemory2(
    VkDevice device,
    std::uint32_t bindInfoCount,
    const VkBindImageMemoryInfo* bindInfos
) {
    if (bindInfoCount != 0 && bindInfos == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    for (std::uint32_t index = 0; index < bindInfoCount; ++index) {
        const VkResult result = imb_vkBindImageMemory(
            device,
            bindInfos[index].image,
            bindInfos[index].memory,
            bindInfos[index].memoryOffset
        );
        if (result != VK_SUCCESS) return result;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkAllocateMemory(
    VkDevice device,
    const VkMemoryAllocateInfo* allocateInfo,
    const VkAllocationCallbacks*,
    VkDeviceMemory* memory
) {
    if (allocateInfo == nullptr || memory == nullptr || allocateInfo->allocationSize == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device) || allocateInfo->memoryTypeIndex >= 4 || !gState.bridge) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (allocateInfo->allocationSize > gState.bridge->capabilities().maxBufferLength) {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    VkExternalMemoryHandleTypeFlags exportHandleTypes = 0;
    const VkImportMemoryFdInfoKHR* importInfo = nullptr;
    auto* next = reinterpret_cast<const VkBaseInStructure*>(allocateInfo->pNext);
    while (next != nullptr) {
        if (next->sType == VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO) {
            exportHandleTypes = reinterpret_cast<const VkExportMemoryAllocateInfo*>(next)->handleTypes;
        } else if (next->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR) {
            importInfo = reinterpret_cast<const VkImportMemoryFdInfoKHR*>(next);
        }
        next = next->pNext;
    }
    if ((exportHandleTypes & ~VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) != 0
        || (importInfo != nullptr
            && (importInfo->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
                || importInfo->fd < 0))) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    auto* state = new (std::nothrow) DeviceMemoryState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    state->size = allocateInfo->allocationSize;
    state->exportHandleTypes = exportHandleTypes;
    if (importInfo != nullptr) {
        struct stat importedStat {};
        if (::fstat(importInfo->fd, &importedStat) != 0
            || importedStat.st_size < 0
            || static_cast<std::uint64_t>(importedStat.st_size) < state->size) {
            delete state;
            return VK_ERROR_INVALID_EXTERNAL_HANDLE;
        }
        try {
            state->bytes.resize(static_cast<std::size_t>(state->size));
        } catch (const std::bad_alloc&) {
            delete state;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        std::size_t readOffset = 0;
        while (readOffset < state->bytes.size()) {
            const ssize_t count = ::pread(
                importInfo->fd,
                state->bytes.data() + readOffset,
                state->bytes.size() - readOffset,
                static_cast<off_t>(readOffset)
            );
            if (count <= 0) {
                ::close(importInfo->fd);
                delete state;
                return VK_ERROR_INVALID_EXTERNAL_HANDLE;
            }
            readOffset += static_cast<std::size_t>(count);
        }
        state->externalFD = importInfo->fd;
        state->exportHandleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        state->dirtyOffset = 0;
        state->dirtyEnd = state->size;
    }
    if (traceEnabled() || externalMemoryTraceEnabled() || rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: memory handle=%p size=%llu type=%u exportTypes=%#x imported=%d\n",
            static_cast<void*>(state),
            static_cast<unsigned long long>(state->size),
            allocateInfo->memoryTypeIndex,
            state->exportHandleTypes,
            importInfo != nullptr ? 1 : 0
        );
    }
    gState.memories.insert(state);
    *memory = makeObjectHandle<VkDeviceMemory>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetMemoryFdKHR(
    VkDevice device,
    const VkMemoryGetFdInfoKHR* info,
    int* fd
) {
    if (info == nullptr || fd == nullptr
        || info->sType != VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR
        || info->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<DeviceMemoryState>(info->memory);
    if (!validDevice(device) || !gState.memories.contains(state) || state->device != device
        || (state->exportHandleTypes & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) == 0) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    const VkResult backingResult = ensureMemoryBackingLocked(state, false);
    if (backingResult != VK_SUCCESS) return backingResult;
    if (state->externalFD < 0) {
        char path[] = "/tmp/imb-vulkan-memory.XXXXXX";
        const int backingFD = ::mkstemp(path);
        if (backingFD < 0) return VK_ERROR_TOO_MANY_OBJECTS;
        ::unlink(path);
        if (::ftruncate(backingFD, static_cast<off_t>(state->size)) != 0) {
            ::close(backingFD);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        state->externalFD = backingFD;
        const VkResult syncResult = syncMemoryToExternalFDLocked(state, 0, state->size);
        if (syncResult != VK_SUCCESS) {
            ::close(state->externalFD);
            state->externalFD = -1;
            return syncResult;
        }
        state->dirtyOffset = VK_WHOLE_SIZE;
        state->dirtyEnd = 0;
    } else if (state->dirtyOffset != VK_WHOLE_SIZE && state->dirtyEnd > state->dirtyOffset) {
        const VkResult syncResult = syncMemoryToExternalFDLocked(
            state,
            state->dirtyOffset,
            state->dirtyEnd - state->dirtyOffset
        );
        if (syncResult != VK_SUCCESS) return syncResult;
        state->dirtyOffset = VK_WHOLE_SIZE;
        state->dirtyEnd = 0;
    }
    const int output = ::dup(state->externalFD);
    if (output < 0) return VK_ERROR_TOO_MANY_OBJECTS;
    *fd = output;
    if (externalMemoryTraceEnabled() || rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: exported shared memory=%p OPAQUE_FD=%d backingFD=%d bytes=%llu\n",
            static_cast<void*>(state),
            output,
            state->externalFD,
            static_cast<unsigned long long>(state->size)
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetMemoryFdPropertiesKHR(
    VkDevice device,
    VkExternalMemoryHandleTypeFlagBits handleType,
    int fd,
    VkMemoryFdPropertiesKHR* properties
) {
    if (properties == nullptr
        || properties->sType != VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR
        || handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
        || fd < 0) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    std::lock_guard lock(gState.mutex);
    struct stat importedStat {};
    if (!validDevice(device) || ::fstat(fd, &importedStat) != 0 || importedStat.st_size <= 0) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    properties->memoryTypeBits = 0x0f;
    if (externalMemoryTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: queried OPAQUE_FD=%d bytes=%lld memoryTypes=%#x\n",
            fd,
            static_cast<long long>(importedStat.st_size),
            properties->memoryTypeBits
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkFreeMemory(
    VkDevice,
    VkDeviceMemory memory,
    const VkAllocationCallbacks*
) {
    if (memory == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<DeviceMemoryState>(memory);
    if (gState.memories.erase(state) == 0) return;
    for (auto* buffer : gState.buffers) {
        std::erase_if(
            buffer->sparseBindings,
            [state](const SparseBufferBinding& binding) { return binding.memory == state; }
        );
    }
    if (gState.bridge && state->resourceID != 0) {
        try {
            gState.bridge->destroyBuffer(state->resourceID);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "imb-vulkan-icd: memory free warning: %s\n", error.what());
        }
    }
    if (state->externalFD >= 0) ::close(state->externalFD);
    delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkBindBufferMemory(
    VkDevice device,
    VkBuffer buffer,
    VkDeviceMemory memory,
    VkDeviceSize memoryOffset
) {
    if (buffer == VK_NULL_HANDLE || memory == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    auto* bufferState = objectState<BufferState>(buffer);
    auto* memoryState = objectState<DeviceMemoryState>(memory);
    if (!gState.buffers.contains(bufferState)
        || !gState.memories.contains(memoryState)
        || bufferState->device != device
        || memoryState->device != device
        || memoryOffset > memoryState->size
        || bufferState->size > memoryState->size - memoryOffset) {
        if (rayTracingTraceEnabled() && gState.buffers.contains(bufferState)
            && bufferState->size >= (VkDeviceSize{1} << 30)) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: RT large-buffer bind rejected buffer=%p size=%llu memory=%p memorySize=%llu offset=%llu knownMemory=%d\n",
                static_cast<void*>(bufferState),
                static_cast<unsigned long long>(bufferState->size),
                static_cast<void*>(memoryState),
                static_cast<unsigned long long>(gState.memories.contains(memoryState) ? memoryState->size : 0),
                static_cast<unsigned long long>(memoryOffset),
                gState.memories.contains(memoryState) ? 1 : 0
            );
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    bufferState->memory = memoryState;
    bufferState->memoryOffset = memoryOffset;
    if (traceEnabled() || (rayTracingTraceEnabled() && bufferState->size >= (VkDeviceSize{1} << 30))) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: bind buffer=%p memory=%p offset=%llu\n",
            static_cast<void*>(bufferState),
            static_cast<void*>(memoryState),
            static_cast<unsigned long long>(memoryOffset)
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkBindBufferMemory2(
    VkDevice device,
    std::uint32_t bindInfoCount,
    const VkBindBufferMemoryInfo* bindInfos
) {
    if (bindInfoCount != 0 && bindInfos == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    for (std::uint32_t index = 0; index < bindInfoCount; ++index) {
        const VkResult result = imb_vkBindBufferMemory(
            device,
            bindInfos[index].buffer,
            bindInfos[index].memory,
            bindInfos[index].memoryOffset
        );
        if (result != VK_SUCCESS) return result;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkMapMemory(
    VkDevice,
    VkDeviceMemory memory,
    VkDeviceSize offset,
    VkDeviceSize size,
    VkMemoryMapFlags,
    void** data
) {
    if (memory == VK_NULL_HANDLE || data == nullptr) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: vkMapMemory rejected null argument memory=%p data=%p\n",
            reinterpret_cast<void*>(memory),
            static_cast<void*>(data)
        );
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<DeviceMemoryState>(memory);
    if (!gState.memories.contains(state) || offset > state->size) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: vkMapMemory rejected memory=%p tracked=%u offset=%llu allocationSize=%llu requestedSize=%llu\n",
            static_cast<void*>(state),
            gState.memories.contains(state) ? 1u : 0u,
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(
                gState.memories.contains(state) ? state->size : 0
            ),
            static_cast<unsigned long long>(size)
        );
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    const VkDeviceSize mapSize = size == VK_WHOLE_SIZE ? state->size - offset : size;
    if (mapSize > state->size - offset) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: vkMapMemory rejected out-of-range memory=%p offset=%llu mapSize=%llu allocationSize=%llu\n",
            static_cast<void*>(state),
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(mapSize),
            static_cast<unsigned long long>(state->size)
        );
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if (state->mapped) {
        // Isaac Kit can ask two subsystems to map the same persistent upload
        // allocation before either subsystem releases it. Vulkan normally
        // requires one map per allocation, but the backing byte vector is
        // stable for the allocation lifetime. Accept only an identical range
        // and keep it mapped until the matching number of unmaps arrives;
        // overlapping or different-range remaps remain rejected.
        if (state->mappedOffset != offset || state->mappedSize != mapSize
            || state->mapReferenceCount == UINT32_MAX) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: vkMapMemory rejected incompatible remap memory=%p offset=%llu size=%llu activeOffset=%llu activeSize=%llu references=%u\n",
                static_cast<void*>(state),
                static_cast<unsigned long long>(offset),
                static_cast<unsigned long long>(mapSize),
                static_cast<unsigned long long>(state->mappedOffset),
                static_cast<unsigned long long>(state->mappedSize),
                state->mapReferenceCount
            );
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        ++state->mapReferenceCount;
        *data = state->bytes.data() + static_cast<std::size_t>(offset);
        std::fprintf(
            stderr,
            "imb-vulkan-icd: accepted identical persistent-memory remap memory=%p offset=%llu size=%llu references=%u\n",
            static_cast<void*>(state),
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(mapSize),
            state->mapReferenceCount
        );
        return VK_SUCCESS;
    }
    const VkResult backingResult = ensureMemoryBackingLocked(state, false);
    if (backingResult != VK_SUCCESS) return backingResult;
    state->mapped = true;
    state->mappedOffset = offset;
    state->mappedSize = mapSize;
    state->mapReferenceCount = 1;
    state->dirtyOffset = std::min(state->dirtyOffset, offset);
    state->dirtyEnd = std::max(state->dirtyEnd, offset + mapSize);
    *data = state->bytes.data() + static_cast<std::size_t>(offset);
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: map memory=%p offset=%llu size=%llu ptr=%p\n",
            static_cast<void*>(state),
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(mapSize),
            *data
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkUnmapMemory(VkDevice, VkDeviceMemory memory) {
    if (memory == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<DeviceMemoryState>(memory);
    if (gState.memories.contains(state)) {
        if (!state->mapped || state->mapReferenceCount == 0) return;
        if (state->externalFD >= 0 && state->dirtyOffset != VK_WHOLE_SIZE
            && state->dirtyEnd > state->dirtyOffset) {
            (void)syncMemoryToExternalFDLocked(
                state,
                state->dirtyOffset,
                state->dirtyEnd - state->dirtyOffset
            );
        }
        --state->mapReferenceCount;
        if (state->mapReferenceCount != 0) return;
        if (state->externalFD >= 0) {
            state->dirtyOffset = VK_WHOLE_SIZE;
            state->dirtyEnd = 0;
        }
        state->mapped = false;
        state->mappedOffset = 0;
        state->mappedSize = 0;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkFlushMappedMemoryRanges(
    VkDevice,
    std::uint32_t memoryRangeCount,
    const VkMappedMemoryRange* memoryRanges
) {
    if (memoryRangeCount != 0 && memoryRanges == nullptr) return VK_ERROR_MEMORY_MAP_FAILED;
    std::lock_guard lock(gState.mutex);
    for (std::uint32_t index = 0; index < memoryRangeCount; ++index) {
        auto* state = objectState<DeviceMemoryState>(memoryRanges[index].memory);
        if (!gState.memories.contains(state)) return VK_ERROR_MEMORY_MAP_FAILED;
        const VkDeviceSize offset = memoryRanges[index].offset;
        if (offset > state->size) return VK_ERROR_MEMORY_MAP_FAILED;
        const VkDeviceSize size = memoryRanges[index].size == VK_WHOLE_SIZE
            ? state->size - offset
            : memoryRanges[index].size;
        if (size > state->size - offset) {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        const VkResult syncResult = syncMemoryToExternalFDLocked(state, offset, size);
        if (syncResult != VK_SUCCESS) return syncResult;
        if (state->externalFD >= 0) {
            state->dirtyOffset = VK_WHOLE_SIZE;
            state->dirtyEnd = 0;
        }
        if (traceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: flush memory=%p offset=%llu size=%llu\n",
                static_cast<void*>(state),
                static_cast<unsigned long long>(memoryRanges[index].offset),
                static_cast<unsigned long long>(memoryRanges[index].size)
            );
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkInvalidateMappedMemoryRanges(
    VkDevice,
    std::uint32_t memoryRangeCount,
    const VkMappedMemoryRange* memoryRanges
) {
    if (memoryRangeCount != 0 && memoryRanges == nullptr) return VK_ERROR_MEMORY_MAP_FAILED;
    std::lock_guard lock(gState.mutex);
    for (std::uint32_t index = 0; index < memoryRangeCount; ++index) {
        auto* state = objectState<DeviceMemoryState>(memoryRanges[index].memory);
        if (!gState.memories.contains(state)) return VK_ERROR_MEMORY_MAP_FAILED;
        const VkDeviceSize offset = memoryRanges[index].offset;
        if (offset > state->size) return VK_ERROR_MEMORY_MAP_FAILED;
        const VkDeviceSize size = memoryRanges[index].size == VK_WHOLE_SIZE
            ? state->size - offset
            : memoryRanges[index].size;
        if (size > state->size - offset) return VK_ERROR_MEMORY_MAP_FAILED;
        const VkResult result = refreshMemoryFromExternalFDLocked(state, offset, size);
        if (result != VK_SUCCESS) return result;
    }
    return VK_SUCCESS;
}

std::uint64_t shaderHash(const std::uint32_t* words, std::size_t byteCount) {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(words);
    for (std::size_t index = 0; index < byteCount; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool spirvUsesFloat64Matrix(const std::uint32_t* words, std::size_t byteCount) {
    constexpr std::uint16_t kOpTypeFloat = 22;
    constexpr std::uint16_t kOpTypeVector = 23;
    constexpr std::uint16_t kOpTypeMatrix = 24;
    const std::size_t wordCount = byteCount / sizeof(std::uint32_t);
    if (words == nullptr || wordCount < 5) return false;

    auto collectResultTypes = [words, wordCount](
                                  std::uint16_t wantedOpcode,
                                  const std::vector<std::uint32_t>* referencedTypes
                              ) {
        std::vector<std::uint32_t> results;
        for (std::size_t offset = 5; offset < wordCount;) {
            const std::uint16_t instructionWords =
                static_cast<std::uint16_t>(words[offset] >> 16);
            const std::uint16_t opcode =
                static_cast<std::uint16_t>(words[offset] & UINT32_C(0xffff));
            if (instructionWords == 0 || instructionWords > wordCount - offset) break;
            if (opcode == wantedOpcode && instructionWords >= 3) {
                const bool matchesReference = referencedTypes == nullptr
                    || std::find(
                        referencedTypes->begin(),
                        referencedTypes->end(),
                        words[offset + 2]
                    ) != referencedTypes->end();
                // OpTypeFloat's third word is its bit width. Vector/Matrix use
                // the third word as their component/column type.
                if (matchesReference) results.push_back(words[offset + 1]);
            }
            offset += instructionWords;
        }
        return results;
    };

    std::vector<std::uint32_t> float64Types;
    for (std::size_t offset = 5; offset < wordCount;) {
        const std::uint16_t instructionWords =
            static_cast<std::uint16_t>(words[offset] >> 16);
        const std::uint16_t opcode =
            static_cast<std::uint16_t>(words[offset] & UINT32_C(0xffff));
        if (instructionWords == 0 || instructionWords > wordCount - offset) break;
        if (opcode == kOpTypeFloat && instructionWords >= 3
            && words[offset + 2] == 64) {
            float64Types.push_back(words[offset + 1]);
        }
        offset += instructionWords;
    }
    if (float64Types.empty()) return false;
    const auto float64VectorTypes = collectResultTypes(kOpTypeVector, &float64Types);
    if (float64VectorTypes.empty()) return false;
    return !collectResultTypes(kOpTypeMatrix, &float64VectorTypes).empty();
}

void dumpShaderIfRequested(const VkShaderModuleCreateInfo* createInfo, std::uint64_t hash) {
    const char* directory = std::getenv("IMB_VULKAN_SHADER_DUMP_DIR");
    if (directory == nullptr || *directory == '\0') return;
    if (::mkdir(directory, 0755) != 0 && errno != EEXIST) {
        std::fprintf(stderr, "imb-vulkan-icd: shader dump mkdir failed: %s\n", std::strerror(errno));
        return;
    }
    char filename[64]{};
    std::snprintf(filename, sizeof(filename), "/%016llx.spv", static_cast<unsigned long long>(hash));
    std::ofstream output(std::string(directory) + filename, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::fprintf(stderr, "imb-vulkan-icd: shader dump open failed for %s\n", filename);
        return;
    }
    output.write(reinterpret_cast<const char*>(createInfo->pCode), static_cast<std::streamsize>(createInfo->codeSize));
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateShaderModule(
    VkDevice device,
    const VkShaderModuleCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkShaderModule* shaderModule
) {
    if (createInfo == nullptr || shaderModule == nullptr || createInfo->pCode == nullptr
        || createInfo->codeSize < sizeof(std::uint32_t) || (createInfo->codeSize % 4) != 0
        || createInfo->pCode[0] != 0x07230203) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    const bool isAddUInt32 = createInfo->codeSize == imb_add_u32_spv_len
        && std::memcmp(createInfo->pCode, imb_add_u32_spv, imb_add_u32_spv_len) == 0;
    const bool isTriangleVertex = createInfo->codeSize == imb_triangle_vert_spv_len
        && std::memcmp(createInfo->pCode, imb_triangle_vert_spv, imb_triangle_vert_spv_len) == 0;
    const bool isTriangleFragment = createInfo->codeSize == imb_triangle_frag_spv_len
        && std::memcmp(createInfo->pCode, imb_triangle_frag_spv, imb_triangle_frag_spv_len) == 0;
    const std::uint64_t hash = shaderHash(createInfo->pCode, createInfo->codeSize);
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: shader module bytes=%zu hash=%016llx known=%s\n",
            createInfo->codeSize,
            static_cast<unsigned long long>(hash),
            isAddUInt32 ? "add-u32" : (isTriangleVertex ? "triangle-vertex" : (isTriangleFragment ? "triangle-fragment" : "no"))
        );
    }
    dumpShaderIfRequested(createInfo, hash);
    auto* state = new (std::nothrow) ShaderModuleState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    state->isAddUInt32 = isAddUInt32;
    state->isTriangleVertex = isTriangleVertex;
    state->isTriangleFragment = isTriangleFragment;
    try {
        state->usesFloat64Matrix = spirvUsesFloat64Matrix(
            createInfo->pCode,
            createInfo->codeSize
        );
    } catch (const std::bad_alloc&) {
        delete state;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    state->hash = hash;
    try {
        state->code.assign(createInfo->pCode, createInfo->pCode + createInfo->codeSize / 4);
    } catch (const std::bad_alloc&) {
        delete state;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    gState.shaderModules.insert(state);
    *shaderModule = makeObjectHandle<VkShaderModule>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyShaderModule(
    VkDevice,
    VkShaderModule shaderModule,
    const VkAllocationCallbacks*
) {
    if (shaderModule == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<ShaderModuleState>(shaderModule);
    if (gState.shaderModules.erase(state) != 0) delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateBufferView(
    VkDevice device,
    const VkBufferViewCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkBufferView* bufferView
) {
    if (createInfo == nullptr || bufferView == nullptr
        || createInfo->buffer == VK_NULL_HANDLE
        || createInfo->format == VK_FORMAT_UNDEFINED) {
        static bool reportedInvalidArguments = false;
        if (!reportedInvalidArguments) {
            reportedInvalidArguments = true;
            std::fprintf(
                stderr,
                "imb-vulkan-icd: invalid buffer-view arguments info=%p out=%p buffer=%p format=%d\n",
                static_cast<const void*>(createInfo),
                static_cast<void*>(bufferView),
                reinterpret_cast<void*>(
                    createInfo == nullptr ? VK_NULL_HANDLE : createInfo->buffer
                ),
                createInfo == nullptr ? VK_FORMAT_UNDEFINED : createInfo->format
            );
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *bufferView = VK_NULL_HANDLE;
    std::lock_guard lock(gState.mutex);
    auto* buffer = objectState<BufferState>(createInfo->buffer);
    if (!validDevice(device) || !gState.buffers.contains(buffer)
        || buffer->device != device || createInfo->offset >= buffer->size) {
        static bool reportedInvalidBuffer = false;
        if (!reportedInvalidBuffer) {
            reportedInvalidBuffer = true;
            std::fprintf(
                stderr,
                "imb-vulkan-icd: invalid buffer-view source device=%p buffer=%p state=%p validDevice=%d known=%d owner=%p offset=%llu size=%llu format=%d\n",
                reinterpret_cast<void*>(device),
                reinterpret_cast<void*>(createInfo->buffer),
                static_cast<void*>(buffer),
                validDevice(device) ? 1 : 0,
                gState.buffers.contains(buffer) ? 1 : 0,
                static_cast<void*>(
                    gState.buffers.contains(buffer) ? buffer->device : VK_NULL_HANDLE
                ),
                static_cast<unsigned long long>(createInfo->offset),
                static_cast<unsigned long long>(
                    gState.buffers.contains(buffer) ? buffer->size : 0
                ),
                createInfo->format
            );
        }
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    const auto block = formatBlockInfo(createInfo->format);
    if (!texelBufferFormatSupported(createInfo->format)) {
        static std::unordered_set<std::uint32_t> reportedFormats;
        if (reportedFormats.insert(static_cast<std::uint32_t>(createInfo->format)).second) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: unsupported Metal texel-buffer Vulkan format=%d\n",
                createInfo->format
            );
        }
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    if (block.width != 1 || block.height != 1 || block.depth != 1
        || block.bytes == 0 || (createInfo->offset % block.bytes) != 0) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    const VkDeviceSize remaining = buffer->size - createInfo->offset;
    const VkDeviceSize requestedRange = createInfo->range == VK_WHOLE_SIZE
        ? remaining
        : createInfo->range;
    if (requestedRange == 0 || requestedRange > remaining) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    // VK_WHOLE_SIZE is defined as the largest complete texel range. Real
    // Isaac RTX also submits the equivalent raw remaining-byte count (for
    // example 1161 bytes for R32_UINT). Accept that narrow non-conformant
    // spelling by discarding only the incomplete final texel.
    const VkDeviceSize range = requestedRange - (requestedRange % block.bytes);
    if (range != requestedRange) {
        static bool reportedRoundedTail = false;
        if (!reportedRoundedTail) {
            reportedRoundedTail = true;
            std::fprintf(
                stderr,
                "imb-vulkan-icd: rounded incomplete texel-buffer tail requested=%llu usable=%llu texel=%u\n",
                static_cast<unsigned long long>(requestedRange),
                static_cast<unsigned long long>(range),
                block.bytes
            );
        }
    }
    if (range == 0 || (range % block.bytes) != 0
        || range / block.bytes > (UINT64_C(1) << 27)) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: invalid Metal texel-buffer range format=%d offset=%llu range=%llu buffer=%llu texel=%u\n",
            createInfo->format,
            static_cast<unsigned long long>(createInfo->offset),
            static_cast<unsigned long long>(range),
            static_cast<unsigned long long>(buffer->size),
            block.bytes
        );
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    auto* state = new (std::nothrow) BufferViewState{
        device,
        buffer,
        createInfo->format,
        createInfo->offset,
        range,
    };
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    gState.bufferViews.insert(state);
    *bufferView = makeObjectHandle<VkBufferView>(state);
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: buffer view=%p buffer=%p format=%d offset=%llu range=%llu\n",
            static_cast<void*>(state),
            static_cast<void*>(buffer),
            createInfo->format,
            static_cast<unsigned long long>(createInfo->offset),
            static_cast<unsigned long long>(range)
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyBufferView(
    VkDevice,
    VkBufferView bufferView,
    const VkAllocationCallbacks*
) {
    if (bufferView == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<BufferViewState>(bufferView);
    if (gState.bufferViews.erase(state) != 0) delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateImageView(
    VkDevice device,
    const VkImageViewCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkImageView* imageView
) {
    if (createInfo == nullptr || imageView == nullptr || createInfo->image == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    auto* image = objectState<ImageState>(createInfo->image);
    if (!validDevice(device) || !gState.images.contains(image) || image->device != device) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    auto* state = new (std::nothrow) ImageViewState{device, image};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    gState.imageViews.insert(state);
    *imageView = makeObjectHandle<VkImageView>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyImageView(
    VkDevice,
    VkImageView imageView,
    const VkAllocationCallbacks*
) {
    if (imageView == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<ImageViewState>(imageView);
    if (gState.imageViews.erase(state) != 0) delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateSampler(
    VkDevice device,
    const VkSamplerCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkSampler* sampler
) {
    if (createInfo == nullptr || sampler == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    auto* state = new (std::nothrow) SamplerState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    state->magFilter = createInfo->magFilter;
    state->minFilter = createInfo->minFilter;
    state->mipmapMode = createInfo->mipmapMode;
    state->addressModeU = createInfo->addressModeU;
    state->addressModeV = createInfo->addressModeV;
    state->addressModeW = createInfo->addressModeW;
    state->mipLodBias = createInfo->mipLodBias;
    state->anisotropyEnable = createInfo->anisotropyEnable;
    state->maxAnisotropy = createInfo->maxAnisotropy;
    state->compareEnable = createInfo->compareEnable;
    state->compareOp = createInfo->compareOp;
    state->minLod = createInfo->minLod;
    state->maxLod = createInfo->maxLod;
    state->borderColor = createInfo->borderColor;
    state->unnormalizedCoordinates = createInfo->unnormalizedCoordinates;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: sampler handle=%p mag=%d min=%d mip=%d address=%d/%d/%d anisotropy=%u max=%g\n",
            static_cast<void*>(state),
            createInfo->magFilter,
            createInfo->minFilter,
            createInfo->mipmapMode,
            createInfo->addressModeU,
            createInfo->addressModeV,
            createInfo->addressModeW,
            createInfo->anisotropyEnable,
            static_cast<double>(createInfo->maxAnisotropy)
        );
    }
    gState.samplers.insert(state);
    *sampler = makeObjectHandle<VkSampler>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroySampler(
    VkDevice,
    VkSampler sampler,
    const VkAllocationCallbacks*
) {
    if (sampler == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<SamplerState>(sampler);
    if (gState.samplers.erase(state) != 0) delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateRenderPass(
    VkDevice device,
    const VkRenderPassCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkRenderPass* renderPass
) {
    if (createInfo == nullptr || renderPass == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    if (createInfo->attachmentCount == 1 && createInfo->pAttachments != nullptr
        && createInfo->subpassCount == 1 && createInfo->pSubpasses != nullptr
        && createInfo->pSubpasses[0].colorAttachmentCount == 1
        && createInfo->pSubpasses[0].pColorAttachments != nullptr
        && createInfo->pSubpasses[0].pColorAttachments[0].attachment == 0) {
        colorFormat = createInfo->pAttachments[0].format;
    }
    auto* state = new (std::nothrow) RenderPassState{device, colorFormat};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: render pass handle=%p attachments=%u subpasses=%u colorFormat=%d\n",
            static_cast<void*>(state),
            createInfo->attachmentCount,
            createInfo->subpassCount,
            colorFormat
        );
    }
    gState.renderPasses.insert(state);
    *renderPass = makeObjectHandle<VkRenderPass>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyRenderPass(
    VkDevice,
    VkRenderPass renderPass,
    const VkAllocationCallbacks*
) {
    if (renderPass == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<RenderPassState>(renderPass);
    if (gState.renderPasses.erase(state) != 0) delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateFramebuffer(
    VkDevice device,
    const VkFramebufferCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkFramebuffer* framebuffer
) {
    if (createInfo == nullptr || framebuffer == nullptr || createInfo->renderPass == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    auto* renderPass = objectState<RenderPassState>(createInfo->renderPass);
    if (!validDevice(device) || !gState.renderPasses.contains(renderPass)
        || renderPass->device != device) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    ImageState* colorImage = nullptr;
    if (createInfo->attachmentCount == 1 && createInfo->pAttachments != nullptr) {
        auto* view = objectState<ImageViewState>(createInfo->pAttachments[0]);
        if (gState.imageViews.contains(view) && view->device == device) colorImage = view->image;
    }
    auto* state = new (std::nothrow) FramebufferState{
        device,
        renderPass,
        colorImage,
        createInfo->width,
        createInfo->height,
    };
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: framebuffer handle=%p renderPass=%p image=%p extent=%ux%u attachments=%u\n",
            static_cast<void*>(state),
            static_cast<void*>(renderPass),
            static_cast<void*>(colorImage),
            createInfo->width,
            createInfo->height,
            createInfo->attachmentCount
        );
    }
    gState.framebuffers.insert(state);
    *framebuffer = makeObjectHandle<VkFramebuffer>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyFramebuffer(
    VkDevice,
    VkFramebuffer framebuffer,
    const VkAllocationCallbacks*
) {
    if (framebuffer == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<FramebufferState>(framebuffer);
    if (gState.framebuffers.erase(state) != 0) delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreatePipelineCache(
    VkDevice device,
    const VkPipelineCacheCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkPipelineCache* pipelineCache
) {
    if (createInfo == nullptr || pipelineCache == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    auto* state = new (std::nothrow) PipelineCacheState{device};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    gState.pipelineCaches.insert(state);
    *pipelineCache = makeObjectHandle<VkPipelineCache>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyPipelineCache(
    VkDevice,
    VkPipelineCache pipelineCache,
    const VkAllocationCallbacks*
) {
    if (pipelineCache == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<PipelineCacheState>(pipelineCache);
    if (gState.pipelineCaches.erase(state) != 0) delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetPipelineCacheData(
    VkDevice,
    VkPipelineCache pipelineCache,
    std::size_t* dataSize,
    void*
) {
    if (dataSize == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<PipelineCacheState>(pipelineCache);
    if (!gState.pipelineCaches.contains(state)) return VK_ERROR_INITIALIZATION_FAILED;
    *dataSize = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkMergePipelineCaches(
    VkDevice,
    VkPipelineCache destinationCache,
    std::uint32_t sourceCacheCount,
    const VkPipelineCache* sourceCaches
) {
    if (sourceCacheCount != 0 && sourceCaches == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    auto* destination = objectState<PipelineCacheState>(destinationCache);
    if (!gState.pipelineCaches.contains(destination)) return VK_ERROR_INITIALIZATION_FAILED;
    for (std::uint32_t index = 0; index < sourceCacheCount; ++index) {
        if (!gState.pipelineCaches.contains(objectState<PipelineCacheState>(sourceCaches[index]))) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateDescriptorSetLayout(
    VkDevice device,
    const VkDescriptorSetLayoutCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkDescriptorSetLayout* setLayout
) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: vkCreateDescriptorSetLayout device=%p info=%p out=%p bindings=%u ptr=%p flags=%#x\n",
            reinterpret_cast<void*>(device),
            static_cast<const void*>(createInfo),
            static_cast<void*>(setLayout),
            createInfo == nullptr ? 0 : createInfo->bindingCount,
            createInfo == nullptr ? nullptr : static_cast<const void*>(createInfo->pBindings),
            createInfo == nullptr ? 0 : createInfo->flags
        );
    }
    if (createInfo == nullptr || setLayout == nullptr
        || (createInfo->bindingCount != 0 && createInfo->pBindings == nullptr)) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    std::unordered_set<std::uint32_t> bindingNumbers;
    for (std::uint32_t index = 0; index < createInfo->bindingCount; ++index) {
        const auto& binding = createInfo->pBindings[index];
        if (traceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: descriptor binding=%u type=%d count=%u stages=%#x immutable=%s\n",
                binding.binding,
                static_cast<int>(binding.descriptorType),
                binding.descriptorCount,
                binding.stageFlags,
                binding.pImmutableSamplers == nullptr ? "no" : "yes"
            );
        }
        // Kit compatibility mode uses zero-count bindings with a max-enum stage
        // sentinel for inactive descriptor slots. Preserve those placeholders so
        // its pipeline layout stays structurally identical to the NVIDIA path.
        if (!bindingNumbers.insert(binding.binding).second) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }

    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) {
        if (traceEnabled()) std::fprintf(stderr, "imb-vulkan-icd: descriptor layout device is unknown\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    auto* state = new (std::nothrow) DescriptorSetLayoutState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    try {
        if (createInfo->bindingCount != 0) {
            state->bindings.assign(createInfo->pBindings, createInfo->pBindings + createInfo->bindingCount);
        }
        for (auto& binding : state->bindings) binding.pImmutableSamplers = nullptr;
    } catch (const std::bad_alloc&) {
        delete state;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    gState.descriptorSetLayouts.insert(state);
    *setLayout = makeObjectHandle<VkDescriptorSetLayout>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyDescriptorSetLayout(
    VkDevice,
    VkDescriptorSetLayout setLayout,
    const VkAllocationCallbacks*
) {
    if (setLayout == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<DescriptorSetLayoutState>(setLayout);
    if (gState.descriptorSetLayouts.erase(state) != 0) delete state;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetDescriptorSetLayoutSupport(
    VkDevice device,
    const VkDescriptorSetLayoutCreateInfo* createInfo,
    VkDescriptorSetLayoutSupport* support
) {
    if (support == nullptr) return;
    std::lock_guard lock(gState.mutex);
    support->supported = createInfo != nullptr && validDevice(device) ? VK_TRUE : VK_FALSE;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreatePipelineLayout(
    VkDevice device,
    const VkPipelineLayoutCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkPipelineLayout* pipelineLayout
) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: vkCreatePipelineLayout device=%p info=%p out=%p sets=%u setPtr=%p pushes=%u pushPtr=%p flags=%#x\n",
            reinterpret_cast<void*>(device),
            static_cast<const void*>(createInfo),
            static_cast<void*>(pipelineLayout),
            createInfo == nullptr ? 0 : createInfo->setLayoutCount,
            createInfo == nullptr ? nullptr : static_cast<const void*>(createInfo->pSetLayouts),
            createInfo == nullptr ? 0 : createInfo->pushConstantRangeCount,
            createInfo == nullptr ? nullptr : static_cast<const void*>(createInfo->pPushConstantRanges),
            createInfo == nullptr ? 0 : createInfo->flags
        );
    }
    if (createInfo == nullptr || pipelineLayout == nullptr
        || (createInfo->setLayoutCount != 0 && createInfo->pSetLayouts == nullptr)
        || (createInfo->pushConstantRangeCount != 0 && createInfo->pPushConstantRanges == nullptr)) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;

    std::vector<DescriptorSetLayoutState*> setLayoutStates;
    try {
        setLayoutStates.reserve(createInfo->setLayoutCount);
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    for (std::uint32_t index = 0; index < createInfo->setLayoutCount; ++index) {
        auto* setLayoutState = objectState<DescriptorSetLayoutState>(createInfo->pSetLayouts[index]);
        if (!gState.descriptorSetLayouts.contains(setLayoutState) || setLayoutState->device != device) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        setLayoutStates.push_back(setLayoutState);
    }
    for (std::uint32_t index = 0; index < createInfo->pushConstantRangeCount; ++index) {
        const auto& range = createInfo->pPushConstantRanges[index];
        if (range.stageFlags == 0 || range.size == 0 || (range.offset % 4) != 0 || (range.size % 4) != 0) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }

    auto* state = new (std::nothrow) PipelineLayoutState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    state->setLayouts = std::move(setLayoutStates);
    try {
        if (createInfo->pushConstantRangeCount != 0) {
            state->pushConstantRanges.assign(
                createInfo->pPushConstantRanges,
                createInfo->pPushConstantRanges + createInfo->pushConstantRangeCount
            );
        }
    } catch (const std::bad_alloc&) {
        delete state;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: pipeline layout sets=%u pushRanges=%u\n",
            createInfo->setLayoutCount,
            createInfo->pushConstantRangeCount
        );
        for (std::uint32_t index = 0; index < createInfo->pushConstantRangeCount; ++index) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: push range stages=%#x offset=%u size=%u\n",
                createInfo->pPushConstantRanges[index].stageFlags,
                createInfo->pPushConstantRanges[index].offset,
                createInfo->pPushConstantRanges[index].size
            );
        }
    }
    gState.pipelineLayouts.insert(state);
    *pipelineLayout = makeObjectHandle<VkPipelineLayout>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyPipelineLayout(
    VkDevice,
    VkPipelineLayout pipelineLayout,
    const VkAllocationCallbacks*
) {
    if (pipelineLayout == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<PipelineLayoutState>(pipelineLayout);
    if (gState.pipelineLayouts.erase(state) != 0) delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateComputePipelines(
    VkDevice device,
    VkPipelineCache,
    std::uint32_t createInfoCount,
    const VkComputePipelineCreateInfo* createInfos,
    const VkAllocationCallbacks*,
    VkPipeline* pipelines
) {
    if (createInfoCount == 0 || createInfos == nullptr || pipelines == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    for (std::uint32_t index = 0; index < createInfoCount; ++index) pipelines[index] = VK_NULL_HANDLE;
    for (std::uint32_t index = 0; index < createInfoCount; ++index) {
        const auto& info = createInfos[index];
        auto* shader = objectState<ShaderModuleState>(info.stage.module);
        auto* layout = objectState<PipelineLayoutState>(info.layout);
        if (info.stage.stage != VK_SHADER_STAGE_COMPUTE_BIT || info.stage.pName == nullptr
            || !gState.shaderModules.contains(shader) || !gState.pipelineLayouts.contains(layout)
            || shader->device != device || layout->device != device) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        const bool isAddUInt32 = shader->isAddUInt32 && std::strcmp(info.stage.pName, "main") == 0
            && info.stage.pSpecializationInfo == nullptr;
        auto* state = new (std::nothrow) PipelineState{};
        if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
        state->device = device;
        state->layout = layout;
        state->isAddUInt32 = isAddUInt32;
        state->usesFloat64Matrix = shader->usesFloat64Matrix;
        state->computeHash = shader->hash;
        if (spirvComputeBridgeEnabled()
            && (gState.bridge->capabilities().bits & IMB_CAP_METAL_SPIRV_COMPUTE) != 0
            && info.stage.pSpecializationInfo == nullptr) {
            try {
                std::uint32_t creationFlags = IMB_COMPUTE_PIPELINE_FLAG_NONE;
                state->bridgeComputePipelineID = gState.bridge->createComputePipeline(
                    shader->code,
                    info.stage.pName,
                    &creationFlags
                );
                state->requiresSoftwareFloat64Execution =
                    (creationFlags
                        & IMB_COMPUTE_PIPELINE_FLAG_SOFTWARE_FP64_EXECUTION_REQUIRED) != 0;
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: Metal SPIR-V compute pipeline created hash=%016llx host=%llu execution=%s\n",
                    static_cast<unsigned long long>(shader->hash),
                    static_cast<unsigned long long>(state->bridgeComputePipelineID),
                    state->usesFloat64Matrix
                            && state->requiresSoftwareFloat64Execution
                            && !validatedSoftwareFloat64MatrixShader(state->computeHash)
                            && !float64MatrixExecutionEnabled()
                        ? "compile-only-fp64-matrix"
                        : "enabled"
                );
            } catch (const std::exception& error) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: Metal SPIR-V compute pipeline skipped hash=%016llx: %s\n",
                    static_cast<unsigned long long>(shader->hash),
                    error.what()
                );
            }
        }
        gState.pipelines.insert(state);
        pipelines[index] = makeObjectHandle<VkPipeline>(state);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateGraphicsPipelines(
    VkDevice device,
    VkPipelineCache,
    std::uint32_t createInfoCount,
    const VkGraphicsPipelineCreateInfo* createInfos,
    const VkAllocationCallbacks*,
    VkPipeline* pipelines
) {
    if (createInfoCount == 0 || createInfos == nullptr || pipelines == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    for (std::uint32_t index = 0; index < createInfoCount; ++index) pipelines[index] = VK_NULL_HANDLE;
    for (std::uint32_t index = 0; index < createInfoCount; ++index) {
        const auto& info = createInfos[index];
        auto* layout = objectState<PipelineLayoutState>(info.layout);
        if (!gState.pipelineLayouts.contains(layout) || layout->device != device) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        bool hasVertex = false;
        bool hasFragment = false;
        std::uint64_t vertexHash = 0;
        std::uint64_t fragmentHash = 0;
        for (std::uint32_t stageIndex = 0; stageIndex < info.stageCount; ++stageIndex) {
            const auto& stage = info.pStages[stageIndex];
            auto* shader = objectState<ShaderModuleState>(stage.module);
            if (!gState.shaderModules.contains(shader) || shader->device != device
                || stage.pName == nullptr || std::strcmp(stage.pName, "main") != 0) {
                continue;
            }
            if (stage.stage == VK_SHADER_STAGE_VERTEX_BIT) {
                vertexHash = shader->hash;
                if (shader->isTriangleVertex) hasVertex = true;
            }
            if (stage.stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
                fragmentHash = shader->hash;
                if (shader->isTriangleFragment) hasFragment = true;
            }
        }
        auto* renderPass = objectState<RenderPassState>(info.renderPass);
        const bool isFixedTriangle = hasVertex && hasFragment
            && gState.renderPasses.contains(renderPass)
            && renderPass->colorFormat == VK_FORMAT_R8G8B8A8_UNORM
            && info.pInputAssemblyState != nullptr
            && info.pInputAssemblyState->topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        constexpr std::uint64_t kKitUIVertexHash = 0x7982ff2b309ae080ULL;
        constexpr std::uint64_t kKitUIFragmentHash = 0xf9b23989da5b7895ULL;
        const bool isKitUI = vertexHash == kKitUIVertexHash
            && fragmentHash == kKitUIFragmentHash
            && gState.renderPasses.contains(renderPass)
            && renderPass->colorFormat == VK_FORMAT_B8G8R8A8_UNORM
            && info.pInputAssemblyState != nullptr
            && info.pInputAssemblyState->topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        if (traceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: graphics pipeline index=%u stages=%u renderPass=%p subpass=%u topology=%d fixedTriangle=%d kitUI=%d\n",
                index,
                info.stageCount,
                static_cast<void*>(renderPass),
                info.subpass,
                info.pInputAssemblyState == nullptr ? -1 : static_cast<int>(info.pInputAssemblyState->topology),
                isFixedTriangle ? 1 : 0,
                isKitUI ? 1 : 0
            );
            for (std::uint32_t stageIndex = 0; stageIndex < info.stageCount; ++stageIndex) {
                auto* shader = objectState<ShaderModuleState>(info.pStages[stageIndex].module);
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: graphics stage=%#x hash=%016llx entry=%s\n",
                    info.pStages[stageIndex].stage,
                    gState.shaderModules.contains(shader) ? static_cast<unsigned long long>(shader->hash) : 0ULL,
                    info.pStages[stageIndex].pName == nullptr ? "(null)" : info.pStages[stageIndex].pName
                );
            }
            if (info.pVertexInputState != nullptr) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: vertex input bindings=%u attributes=%u\n",
                    info.pVertexInputState->vertexBindingDescriptionCount,
                    info.pVertexInputState->vertexAttributeDescriptionCount
                );
                for (std::uint32_t bindingIndex = 0;
                     bindingIndex < info.pVertexInputState->vertexBindingDescriptionCount;
                     ++bindingIndex) {
                    const auto& binding = info.pVertexInputState->pVertexBindingDescriptions[bindingIndex];
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: vertex binding=%u stride=%u rate=%d\n",
                        binding.binding,
                        binding.stride,
                        binding.inputRate
                    );
                }
                for (std::uint32_t attributeIndex = 0;
                     attributeIndex < info.pVertexInputState->vertexAttributeDescriptionCount;
                     ++attributeIndex) {
                    const auto& attribute = info.pVertexInputState->pVertexAttributeDescriptions[attributeIndex];
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: vertex attribute location=%u binding=%u format=%d offset=%u\n",
                        attribute.location,
                        attribute.binding,
                        attribute.format,
                        attribute.offset
                    );
                }
            }
            if (info.pRasterizationState != nullptr) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: raster polygon=%d cull=%#x front=%d line=%g discard=%u\n",
                    info.pRasterizationState->polygonMode,
                    info.pRasterizationState->cullMode,
                    info.pRasterizationState->frontFace,
                    static_cast<double>(info.pRasterizationState->lineWidth),
                    info.pRasterizationState->rasterizerDiscardEnable
                );
            }
            if (info.pColorBlendState != nullptr) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: color blend attachments=%u logic=%u op=%d\n",
                    info.pColorBlendState->attachmentCount,
                    info.pColorBlendState->logicOpEnable,
                    info.pColorBlendState->logicOp
                );
                for (std::uint32_t attachmentIndex = 0;
                     attachmentIndex < info.pColorBlendState->attachmentCount;
                     ++attachmentIndex) {
                    const auto& attachment = info.pColorBlendState->pAttachments[attachmentIndex];
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: blend attachment=%u enabled=%u color=%d/%d/%d alpha=%d/%d/%d mask=%#x\n",
                        attachmentIndex,
                        attachment.blendEnable,
                        attachment.srcColorBlendFactor,
                        attachment.dstColorBlendFactor,
                        attachment.colorBlendOp,
                        attachment.srcAlphaBlendFactor,
                        attachment.dstAlphaBlendFactor,
                        attachment.alphaBlendOp,
                        attachment.colorWriteMask
                    );
                }
            }
            if (info.pDynamicState != nullptr) {
                std::fprintf(stderr, "imb-vulkan-icd: dynamic states=%u", info.pDynamicState->dynamicStateCount);
                for (std::uint32_t dynamicIndex = 0;
                     dynamicIndex < info.pDynamicState->dynamicStateCount;
                     ++dynamicIndex) {
                    std::fprintf(stderr, " %d", info.pDynamicState->pDynamicStates[dynamicIndex]);
                }
                std::fputc('\n', stderr);
            }
        }
        auto* state = new (std::nothrow) PipelineState{};
        if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
        state->device = device;
        state->layout = layout;
        state->graphics = true;
        state->isFixedTriangle = isFixedTriangle;
        state->isKitUI = isKitUI;
        gState.pipelines.insert(state);
        pipelines[index] = makeObjectHandle<VkPipeline>(state);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyPipeline(
    VkDevice,
    VkPipeline pipeline,
    const VkAllocationCallbacks*
) {
    if (pipeline == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<PipelineState>(pipeline);
    if (gState.pipelines.erase(state) != 0) {
        if (state->bridgeComputePipelineID != 0 && gState.bridge != nullptr) {
            try {
                gState.bridge->destroyBuffer(state->bridgeComputePipelineID);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "imb-vulkan-icd: compute pipeline destroy warning: %s\n", error.what());
            }
        }
        delete state;
    }
}

VkDeviceSize alignedAccelerationStructureSize(VkDeviceSize value) {
    constexpr VkDeviceSize alignment = 256;
    if (value > std::numeric_limits<VkDeviceSize>::max() - (alignment - 1)) {
        return std::numeric_limits<VkDeviceSize>::max() & ~(alignment - 1);
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

VkDeviceSize saturatingAccelerationStructureAdd(VkDeviceSize left, VkDeviceSize right) {
    if (right > std::numeric_limits<VkDeviceSize>::max() - left) {
        return std::numeric_limits<VkDeviceSize>::max();
    }
    return left + right;
}

VkDeviceSize saturatingAccelerationStructureMultiply(VkDeviceSize left, VkDeviceSize right) {
    if (left != 0 && right > std::numeric_limits<VkDeviceSize>::max() / left) {
        return std::numeric_limits<VkDeviceSize>::max();
    }
    return left * right;
}

VkDeviceSize accelerationStructureGeometryBytes(const VkAccelerationStructureInfoNV& info) {
    VkDeviceSize bytes = 4096;
    bytes = saturatingAccelerationStructureAdd(
        bytes,
        saturatingAccelerationStructureMultiply(info.instanceCount, sizeof(VkAccelerationStructureInstanceNV))
    );
    for (std::uint32_t index = 0; index < info.geometryCount; ++index) {
        const auto& geometry = info.pGeometries[index];
        if (geometry.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_NV) {
            const auto& triangles = geometry.geometry.triangles;
            const VkDeviceSize vertexStride = std::max<VkDeviceSize>(triangles.vertexStride, 12);
            bytes = saturatingAccelerationStructureAdd(
                bytes,
                saturatingAccelerationStructureMultiply(triangles.vertexCount, vertexStride)
            );
            const VkDeviceSize indexSize = triangles.indexType == VK_INDEX_TYPE_UINT16 ? 2 : 4;
            bytes = saturatingAccelerationStructureAdd(
                bytes,
                saturatingAccelerationStructureMultiply(triangles.indexCount, indexSize)
            );
        } else if (geometry.geometryType == VK_GEOMETRY_TYPE_AABBS_NV) {
            const auto& aabbs = geometry.geometry.aabbs;
            bytes = saturatingAccelerationStructureAdd(
                bytes,
                saturatingAccelerationStructureMultiply(aabbs.numAABBs, std::max<std::uint32_t>(aabbs.stride, 24))
            );
        }
    }
    return alignedAccelerationStructureSize(bytes);
}

bool validAccelerationStructureGeometryLocked(VkDevice device, const VkGeometryNV& geometry) {
    if (geometry.sType != VK_STRUCTURE_TYPE_GEOMETRY_NV) return false;
    if (geometry.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_NV) {
        const auto& triangles = geometry.geometry.triangles;
        if (triangles.sType != VK_STRUCTURE_TYPE_GEOMETRY_TRIANGLES_NV) return false;
        const auto validBuffer = [device](VkBuffer handle) {
            if (handle == VK_NULL_HANDLE) return true;
            auto* buffer = objectState<BufferState>(handle);
            return gState.buffers.contains(buffer) && buffer->device == device;
        };
        return validBuffer(triangles.vertexData)
            && validBuffer(triangles.indexData)
            && validBuffer(triangles.transformData);
    }
    if (geometry.geometryType == VK_GEOMETRY_TYPE_AABBS_NV) {
        const auto& aabbs = geometry.geometry.aabbs;
        if (aabbs.sType != VK_STRUCTURE_TYPE_GEOMETRY_AABB_NV) return false;
        if (aabbs.aabbData == VK_NULL_HANDLE) return aabbs.numAABBs == 0;
        auto* buffer = objectState<BufferState>(aabbs.aabbData);
        return gState.buffers.contains(buffer) && buffer->device == device;
    }
    return false;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateAccelerationStructureNV(
    VkDevice device,
    const VkAccelerationStructureCreateInfoNV* createInfo,
    const VkAllocationCallbacks*,
    VkAccelerationStructureNV* accelerationStructure
) {
    if (createInfo == nullptr || accelerationStructure == nullptr
        || createInfo->sType != VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_NV
        || createInfo->info.sType != VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_INFO_NV
        || (createInfo->info.geometryCount != 0 && createInfo->info.pGeometries == nullptr)) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_DEVICE_LOST;
    for (std::uint32_t index = 0; index < createInfo->info.geometryCount; ++index) {
        if (!validAccelerationStructureGeometryLocked(device, createInfo->info.pGeometries[index])) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }

    auto* state = new (std::nothrow) AccelerationStructureNVState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    state->compactedSize = createInfo->compactedSize;
    state->type = createInfo->info.type;
    state->flags = createInfo->info.flags;
    state->instanceCount = createInfo->info.instanceCount;
    try {
        if (createInfo->info.geometryCount != 0) {
            state->geometries.assign(
                createInfo->info.pGeometries,
                createInfo->info.pGeometries + createInfo->info.geometryCount
            );
            for (auto& geometry : state->geometries) {
                geometry.pNext = nullptr;
                if (geometry.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_NV) {
                    geometry.geometry.triangles.pNext = nullptr;
                } else if (geometry.geometryType == VK_GEOMETRY_TYPE_AABBS_NV) {
                    geometry.geometry.aabbs.pNext = nullptr;
                }
            }
        }
    } catch (const std::bad_alloc&) {
        delete state;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    const VkDeviceSize estimatedSize = accelerationStructureGeometryBytes(createInfo->info);
    state->objectSize = alignedAccelerationStructureSize(std::max(createInfo->compactedSize, estimatedSize));
    state->buildScratchSize = alignedAccelerationStructureSize(
        saturatingAccelerationStructureAdd(4096, state->objectSize / 2)
    );
    state->updateScratchSize = alignedAccelerationStructureSize(
        saturatingAccelerationStructureAdd(4096, state->objectSize / 4)
    );
    state->opaqueHandle = gState.nextAccelerationStructureHandle++;
    gState.accelerationStructuresNV.insert(state);
    *accelerationStructure = makeObjectHandle<VkAccelerationStructureNV>(state);
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: create AS NV handle=%p opaque=%#llx type=%d geometries=%u instances=%u object=%llu scratch=%llu/%llu\n",
            static_cast<void*>(state),
            static_cast<unsigned long long>(state->opaqueHandle),
            state->type,
            createInfo->info.geometryCount,
            state->instanceCount,
            static_cast<unsigned long long>(state->objectSize),
            static_cast<unsigned long long>(state->buildScratchSize),
            static_cast<unsigned long long>(state->updateScratchSize)
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyAccelerationStructureNV(
    VkDevice,
    VkAccelerationStructureNV accelerationStructure,
    const VkAllocationCallbacks*
) {
    if (accelerationStructure == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<AccelerationStructureNVState>(accelerationStructure);
    if (gState.accelerationStructuresNV.erase(state) != 0) delete state;
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetAccelerationStructureMemoryRequirementsNV(
    VkDevice device,
    const VkAccelerationStructureMemoryRequirementsInfoNV* info,
    VkMemoryRequirements2KHR* memoryRequirements
) {
    if (info == nullptr || memoryRequirements == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<AccelerationStructureNVState>(info->accelerationStructure);
    if (!validDevice(device) || !gState.accelerationStructuresNV.contains(state)
        || state->device != device) {
        return;
    }
    VkDeviceSize size = state->objectSize;
    if (info->type == VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_BUILD_SCRATCH_NV) {
        size = state->buildScratchSize;
    } else if (info->type == VK_ACCELERATION_STRUCTURE_MEMORY_REQUIREMENTS_TYPE_UPDATE_SCRATCH_NV) {
        size = state->updateScratchSize;
    }
    memoryRequirements->memoryRequirements.size = size;
    memoryRequirements->memoryRequirements.alignment = 256;
    memoryRequirements->memoryRequirements.memoryTypeBits = 0x0f;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: AS NV requirements handle=%p type=%d size=%llu\n",
            static_cast<void*>(state),
            info->type,
            static_cast<unsigned long long>(size)
        );
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkBindAccelerationStructureMemoryNV(
    VkDevice device,
    std::uint32_t bindInfoCount,
    const VkBindAccelerationStructureMemoryInfoNV* bindInfos
) {
    if (bindInfoCount != 0 && bindInfos == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_DEVICE_LOST;
    for (std::uint32_t index = 0; index < bindInfoCount; ++index) {
        auto* acceleration = objectState<AccelerationStructureNVState>(bindInfos[index].accelerationStructure);
        auto* memory = objectState<DeviceMemoryState>(bindInfos[index].memory);
        if (!gState.accelerationStructuresNV.contains(acceleration)
            || !gState.memories.contains(memory)
            || acceleration->device != device || memory->device != device
            || bindInfos[index].deviceIndexCount != 0
            || (bindInfos[index].memoryOffset % 256) != 0
            || bindInfos[index].memoryOffset > memory->size
            || acceleration->objectSize > memory->size - bindInfos[index].memoryOffset) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    for (std::uint32_t index = 0; index < bindInfoCount; ++index) {
        auto* acceleration = objectState<AccelerationStructureNVState>(bindInfos[index].accelerationStructure);
        acceleration->memory = objectState<DeviceMemoryState>(bindInfos[index].memory);
        acceleration->memoryOffset = bindInfos[index].memoryOffset;
        if (traceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: bind AS NV handle=%p memory=%p offset=%llu\n",
                static_cast<void*>(acceleration),
                static_cast<void*>(acceleration->memory),
                static_cast<unsigned long long>(acceleration->memoryOffset)
            );
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL imb_vkGetBufferDeviceAddressKHR(
    VkDevice device,
    const VkBufferDeviceAddressInfo* info
) {
    if (info == nullptr || info->buffer == VK_NULL_HANDLE) return 0;
    std::lock_guard lock(gState.mutex);
    auto* buffer = objectState<BufferState>(info->buffer);
    if (!validDevice(device) || !gState.buffers.contains(buffer) || buffer->device != device) return 0;
    return buffer->deviceAddress;
}

VKAPI_ATTR std::uint64_t VKAPI_CALL imb_vkGetBufferOpaqueCaptureAddressKHR(
    VkDevice,
    const VkBufferDeviceAddressInfo*
) {
    return 0;
}

VKAPI_ATTR std::uint64_t VKAPI_CALL imb_vkGetDeviceMemoryOpaqueCaptureAddressKHR(
    VkDevice,
    const VkDeviceMemoryOpaqueCaptureAddressInfo*
) {
    return 0;
}

BufferState* bufferForDeviceAddressLocked(VkDevice device, VkDeviceAddress address) {
    if (address == 0) return nullptr;
    for (auto* buffer : gState.buffers) {
        if (buffer->device != device || address < buffer->deviceAddress) continue;
        const VkDeviceAddress offset = address - buffer->deviceAddress;
        if (offset < buffer->size) return buffer;
    }
    return nullptr;
}

AccelerationStructureKHRState* accelerationStructureForDeviceAddressLocked(
    VkDevice device,
    VkDeviceAddress address
) {
    if (address == 0) return nullptr;
    for (auto* accelerationStructure : gState.accelerationStructuresKHR) {
        if (accelerationStructure->device == device
            && accelerationStructure->deviceAddress == address) {
            return accelerationStructure;
        }
    }
    return nullptr;
}

VkResult copyDeviceAddressBytesLocked(
    VkDevice device,
    VkDeviceAddress address,
    std::size_t size,
    imb::Bytes& output
) {
    auto* buffer = bufferForDeviceAddressLocked(device, address);
    if (buffer == nullptr || size == 0) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    const VkDeviceSize bufferOffset = address - buffer->deviceAddress;
    if (bufferOffset > buffer->size || size > buffer->size - bufferOffset) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if (buffer->memory == nullptr) {
        try {
            output.resize(size);
        } catch (const std::bad_alloc&) {
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        VkDeviceSize copied = 0;
        while (copied < size) {
            const VkDeviceSize resourceOffset = bufferOffset + copied;
            const auto found = std::find_if(
                buffer->sparseBindings.begin(),
                buffer->sparseBindings.end(),
                [resourceOffset](const SparseBufferBinding& binding) {
                    return resourceOffset >= binding.resourceOffset
                        && resourceOffset - binding.resourceOffset < binding.size;
                }
            );
            if (found == buffer->sparseBindings.end() || found->memory == nullptr) {
                output.clear();
                return VK_ERROR_MEMORY_MAP_FAILED;
            }
            const VkDeviceSize withinBinding = resourceOffset - found->resourceOffset;
            const VkDeviceSize chunk = std::min<VkDeviceSize>(
                size - copied,
                found->size - withinBinding
            );
            const VkResult backingResult = ensureMemoryBackingLocked(found->memory, false);
            if (backingResult != VK_SUCCESS) {
                output.clear();
                return backingResult;
            }
            const VkDeviceSize sourceOffset = found->memoryOffset + withinBinding;
            if (sourceOffset > found->memory->size
                || chunk > found->memory->size - sourceOffset) {
                output.clear();
                return VK_ERROR_MEMORY_MAP_FAILED;
            }
            const VkResult refreshResult = refreshMemoryFromExternalFDLocked(
                found->memory,
                sourceOffset,
                chunk
            );
            if (refreshResult != VK_SUCCESS) {
                output.clear();
                return refreshResult;
            }
            std::memcpy(
                output.data() + static_cast<std::size_t>(copied),
                found->memory->bytes.data() + static_cast<std::size_t>(sourceOffset),
                static_cast<std::size_t>(chunk)
            );
            copied += chunk;
        }
        return VK_SUCCESS;
    }
    auto* memory = buffer->memory;
    const VkResult backingResult = ensureMemoryBackingLocked(memory, false);
    if (backingResult != VK_SUCCESS) return backingResult;
    if (buffer->memoryOffset > memory->size
        || bufferOffset > memory->size - buffer->memoryOffset) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    const VkDeviceSize sourceOffset = buffer->memoryOffset + bufferOffset;
    if (sourceOffset > memory->size || size > memory->size - sourceOffset) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    const VkResult refreshResult = refreshMemoryFromExternalFDLocked(memory, sourceOffset, size);
    if (refreshResult != VK_SUCCESS) return refreshResult;
    try {
        output.assign(
            memory->bytes.begin() + static_cast<std::size_t>(sourceOffset),
            memory->bytes.begin() + static_cast<std::size_t>(sourceOffset + size)
        );
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return VK_SUCCESS;
}

VkResult writeBufferBytesLocked(
    BufferState* buffer,
    VkDeviceSize bufferOffset,
    const std::uint8_t* source,
    std::size_t size
) {
    if (buffer == nullptr || source == nullptr || size == 0
        || bufferOffset > buffer->size || size > buffer->size - bufferOffset) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if (buffer->memory != nullptr) {
        auto* memory = buffer->memory;
        const VkResult backingResult = ensureMemoryBackingLocked(memory, false);
        if (backingResult != VK_SUCCESS) return backingResult;
        if (buffer->memoryOffset > memory->size
            || bufferOffset > memory->size - buffer->memoryOffset) {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        const VkDeviceSize destinationOffset = buffer->memoryOffset + bufferOffset;
        if (size > memory->size - destinationOffset) return VK_ERROR_MEMORY_MAP_FAILED;
        std::memcpy(
            memory->bytes.data() + static_cast<std::size_t>(destinationOffset),
            source,
            size
        );
        memory->dirtyOffset = std::min(memory->dirtyOffset, destinationOffset);
        memory->dirtyEnd = std::max(memory->dirtyEnd, destinationOffset + size);
        const VkResult syncResult = syncMemoryToExternalFDLocked(
            memory,
            destinationOffset,
            size
        );
        if (syncResult == VK_SUCCESS && memory->externalFD >= 0) {
            memory->dirtyOffset = VK_WHOLE_SIZE;
            memory->dirtyEnd = 0;
        }
        return syncResult;
    }

    VkDeviceSize written = 0;
    while (written < size) {
        const VkDeviceSize resourceOffset = bufferOffset + written;
        const auto found = std::find_if(
            buffer->sparseBindings.begin(),
            buffer->sparseBindings.end(),
            [resourceOffset](const SparseBufferBinding& binding) {
                return resourceOffset >= binding.resourceOffset
                    && resourceOffset - binding.resourceOffset < binding.size;
            }
        );
        if (found == buffer->sparseBindings.end() || found->memory == nullptr) {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        const VkDeviceSize withinBinding = resourceOffset - found->resourceOffset;
        const VkDeviceSize chunk = std::min<VkDeviceSize>(size - written, found->size - withinBinding);
        const VkResult backingResult = ensureMemoryBackingLocked(found->memory, false);
        if (backingResult != VK_SUCCESS) return backingResult;
        const VkDeviceSize destinationOffset = found->memoryOffset + withinBinding;
        if (destinationOffset > found->memory->size
            || chunk > found->memory->size - destinationOffset) {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        std::memcpy(
            found->memory->bytes.data() + static_cast<std::size_t>(destinationOffset),
            source + static_cast<std::size_t>(written),
            static_cast<std::size_t>(chunk)
        );
        found->memory->dirtyOffset = std::min(found->memory->dirtyOffset, destinationOffset);
        found->memory->dirtyEnd = std::max(found->memory->dirtyEnd, destinationOffset + chunk);
        const VkResult syncResult = syncMemoryToExternalFDLocked(
            found->memory,
            destinationOffset,
            chunk
        );
        if (syncResult != VK_SUCCESS) return syncResult;
        if (found->memory->externalFD >= 0) {
            found->memory->dirtyOffset = VK_WHOLE_SIZE;
            found->memory->dirtyEnd = 0;
        }
        written += chunk;
    }
    return VK_SUCCESS;
}

VkResult copyBufferBytesLocked(
    BufferState* source,
    BufferState* destination,
    const VkBufferCopy& region
) {
    if (source == nullptr || destination == nullptr || region.size == 0
        || region.srcOffset > source->size || region.size > source->size - region.srcOffset
        || region.dstOffset > destination->size || region.size > destination->size - region.dstOffset) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    imb::Bytes bytes;
    const VkResult readResult = copyDeviceAddressBytesLocked(
        source->device,
        source->deviceAddress + region.srcOffset,
        static_cast<std::size_t>(region.size),
        bytes
    );
    if (readResult != VK_SUCCESS) return readResult;
    return writeBufferBytesLocked(
        destination,
        region.dstOffset,
        bytes.data(),
        bytes.size()
    );
}

std::uint16_t floatToHalf(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & UINT32_C(0x8000);
    const std::uint32_t exponent = (bits >> 23) & UINT32_C(0xff);
    std::uint32_t mantissa = bits & UINT32_C(0x7fffff);
    if (exponent == UINT32_C(0xff)) {
        return static_cast<std::uint16_t>(
            sign | UINT32_C(0x7c00) | (mantissa == 0 ? 0 : UINT32_C(0x0200))
        );
    }
    const std::int32_t halfExponent = static_cast<std::int32_t>(exponent) - 127 + 15;
    if (halfExponent >= 31) {
        return static_cast<std::uint16_t>(sign | UINT32_C(0x7c00));
    }
    if (halfExponent <= 0) {
        if (halfExponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= UINT32_C(0x800000);
        const std::uint32_t shift = static_cast<std::uint32_t>(14 - halfExponent);
        const std::uint32_t rounded = (mantissa + (UINT32_C(1) << (shift - 1))) >> shift;
        return static_cast<std::uint16_t>(sign | rounded);
    }
    const std::uint32_t roundedMantissa = mantissa + UINT32_C(0x1000);
    if ((roundedMantissa & UINT32_C(0x800000)) != 0) {
        const std::uint32_t roundedExponent = static_cast<std::uint32_t>(halfExponent + 1);
        return static_cast<std::uint16_t>(
            sign | (roundedExponent >= 31 ? UINT32_C(0x7c00) : roundedExponent << 10)
        );
    }
    return static_cast<std::uint16_t>(
        sign | (static_cast<std::uint32_t>(halfExponent) << 10)
            | (roundedMantissa >> 13)
    );
}

template <typename Value>
void storeClearComponent(imb::Bytes& output, std::size_t offset, Value value) {
    if (offset <= output.size() && sizeof(value) <= output.size() - offset) {
        std::memcpy(output.data() + offset, &value, sizeof(value));
    }
}

bool clearColorTexel(VkFormat format, const VkClearColorValue& color, imb::Bytes& output) {
    const auto block = formatBlockInfo(format);
    if (block.width != 1 || block.height != 1 || block.depth != 1 || block.bytes == 0) {
        return false;
    }
    try {
        output.assign(block.bytes, 0);
    } catch (const std::bad_alloc&) {
        return false;
    }

    if (format == VK_FORMAT_R8_UNORM) {
        output[0] = static_cast<std::uint8_t>(
            std::clamp(color.float32[0], 0.0F, 1.0F) * 255.0F + 0.5F
        );
        return true;
    }
    if (format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB
        || format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB) {
        std::array<std::uint8_t, 4> components{};
        for (std::size_t index = 0; index < components.size(); ++index) {
            components[index] = static_cast<std::uint8_t>(
                std::clamp(color.float32[index], 0.0F, 1.0F) * 255.0F + 0.5F
            );
        }
        if (format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB) {
            std::swap(components[0], components[2]);
        }
        std::copy(components.begin(), components.end(), output.begin());
        return true;
    }

    const std::uint32_t formatValue = static_cast<std::uint32_t>(format);
    if (formatValue >= static_cast<std::uint32_t>(VK_FORMAT_R16_UNORM)
        && formatValue <= static_cast<std::uint32_t>(VK_FORMAT_R16G16B16A16_SFLOAT)) {
        const std::uint32_t family =
            (formatValue - static_cast<std::uint32_t>(VK_FORMAT_R16_UNORM)) / 7;
        const std::uint32_t kind =
            (formatValue - static_cast<std::uint32_t>(VK_FORMAT_R16_UNORM)) % 7;
        const std::uint32_t componentCount = family + 1;
        for (std::uint32_t component = 0; component < componentCount; ++component) {
            std::uint16_t value = 0;
            if (kind == 0) {
                value = static_cast<std::uint16_t>(
                    std::clamp(color.float32[component], 0.0F, 1.0F) * 65535.0F + 0.5F
                );
            } else if (kind == 4) {
                value = static_cast<std::uint16_t>(color.uint32[component]);
            } else if (kind == 5) {
                value = static_cast<std::uint16_t>(
                    static_cast<std::int16_t>(color.int32[component])
                );
            } else if (kind == 6) {
                value = floatToHalf(color.float32[component]);
            } else {
                return color.uint32[0] == 0 && color.uint32[1] == 0
                    && color.uint32[2] == 0 && color.uint32[3] == 0;
            }
            storeClearComponent(output, component * sizeof(value), value);
        }
        return true;
    }
    if (formatValue >= static_cast<std::uint32_t>(VK_FORMAT_R32_UINT)
        && formatValue <= static_cast<std::uint32_t>(VK_FORMAT_R32G32B32A32_SFLOAT)) {
        const std::uint32_t family =
            (formatValue - static_cast<std::uint32_t>(VK_FORMAT_R32_UINT)) / 3;
        const std::uint32_t kind =
            (formatValue - static_cast<std::uint32_t>(VK_FORMAT_R32_UINT)) % 3;
        const std::uint32_t componentCount = family + 1;
        for (std::uint32_t component = 0; component < componentCount; ++component) {
            std::uint32_t value = 0;
            if (kind == 0) {
                value = color.uint32[component];
            } else if (kind == 1) {
                std::memcpy(&value, &color.int32[component], sizeof(value));
            } else {
                std::memcpy(&value, &color.float32[component], sizeof(value));
            }
            storeClearComponent(output, component * sizeof(value), value);
        }
        return true;
    }
    return color.uint32[0] == 0 && color.uint32[1] == 0
        && color.uint32[2] == 0 && color.uint32[3] == 0;
}

VkResult imageHostBytesLocked(
    ImageState* image,
    std::uint8_t** bytes,
    VkDeviceSize* byteCount
) {
    if (image == nullptr || bytes == nullptr || byteCount == nullptr
        || !gState.images.contains(image)) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if (image->memory != nullptr) {
        const VkResult backingResult = ensureMemoryBackingLocked(image->memory, false);
        if (backingResult != VK_SUCCESS) return backingResult;
        if (image->memoryOffset > image->memory->size
            || image->size > image->memory->size - image->memoryOffset) {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        *bytes = image->memory->bytes.data() + static_cast<std::size_t>(image->memoryOffset);
        *byteCount = image->size;
        return VK_SUCCESS;
    }
    if ((image->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) == 0) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    try {
        if (image->sparseBytes.empty()) {
            image->sparseBytes.resize(static_cast<std::size_t>(image->size));
        }
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    *bytes = image->sparseBytes.data();
    *byteCount = image->sparseBytes.size();
    return VK_SUCCESS;
}

void markImageSubresourceInitialized(
    ImageState* image,
    std::uint32_t mipLevel,
    std::uint32_t arrayLayer
) {
    const std::size_t index = imageSubresourceIndex(image, mipLevel, arrayLayer);
    if (index < image->initializedSubresources.size()) {
        image->initializedSubresources[index] = true;
    }
    image->dirty = true;
}

VkResult executeBufferFillLocked(const RecordedBufferFill& fill) {
    if (fill.destination == nullptr || fill.size == 0) return VK_ERROR_MEMORY_MAP_FAILED;
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: execute buffer fill destination=%p address=%#llx offset=%llu size=%llu data=%#x\n",
            static_cast<void*>(fill.destination),
            static_cast<unsigned long long>(fill.destination->deviceAddress),
            static_cast<unsigned long long>(fill.offset),
            static_cast<unsigned long long>(fill.size),
            fill.data
        );
        std::fprintf(
            stderr,
            "imb-vulkan-icd: execute buffer fill backing memory=%p memoryOffset=%llu memorySize=%llu\n",
            static_cast<void*>(fill.destination->memory),
            static_cast<unsigned long long>(fill.destination->memoryOffset),
            static_cast<unsigned long long>(
                fill.destination->memory == nullptr ? 0 : fill.destination->memory->size
            )
        );
    }
    std::array<std::uint8_t, 4096> pattern{};
    for (std::size_t offset = 0; offset < pattern.size(); offset += sizeof(fill.data)) {
        std::memcpy(pattern.data() + offset, &fill.data, sizeof(fill.data));
    }
    VkDeviceSize written = 0;
    while (written < fill.size) {
        const VkDeviceSize chunk = std::min<VkDeviceSize>(
            pattern.size(),
            fill.size - written
        );
        const VkResult result = writeBufferBytesLocked(
            fill.destination,
            fill.offset + written,
            pattern.data(),
            static_cast<std::size_t>(chunk)
        );
        if (result != VK_SUCCESS) return result;
        written += chunk;
    }
    return VK_SUCCESS;
}

VkResult executeImageClearLocked(const RecordedImageClear& clear) {
    std::uint8_t* destination = nullptr;
    VkDeviceSize destinationSize = 0;
    const VkResult backingResult =
        imageHostBytesLocked(clear.image, &destination, &destinationSize);
    if (backingResult != VK_SUCCESS) return backingResult;
    imb::Bytes texel;
    if (!clearColorTexel(clear.image->format, clear.color, texel) || texel.empty()) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    const bool zero = std::all_of(
        texel.begin(),
        texel.end(),
        [](std::uint8_t value) { return value == 0; }
    );
    for (const auto& range : clear.ranges) {
        const std::uint32_t levelCount = range.levelCount == VK_REMAINING_MIP_LEVELS
            ? clear.image->mipLevels - range.baseMipLevel
            : range.levelCount;
        const std::uint32_t layerCount = range.layerCount == VK_REMAINING_ARRAY_LAYERS
            ? clear.image->arrayLayers - range.baseArrayLayer
            : range.layerCount;
        for (std::uint32_t layer = 0; layer < layerCount; ++layer) {
            const std::uint32_t arrayLayer = range.baseArrayLayer + layer;
            for (std::uint32_t level = 0; level < levelCount; ++level) {
                const std::uint32_t mipLevel = range.baseMipLevel + level;
                const VkExtent3D extent{
                    std::max(1U, clear.image->extent.width >> mipLevel),
                    std::max(1U, clear.image->extent.height >> mipLevel),
                    std::max(1U, clear.image->extent.depth >> mipLevel),
                };
                const VkDeviceSize offset = imageSubresourceByteOffset(
                    clear.image->format,
                    clear.image->extent,
                    clear.image->mipLevels,
                    mipLevel,
                    arrayLayer
                );
                const VkDeviceSize size = mipByteSize(clear.image->format, extent);
                if (offset > destinationSize || size > destinationSize - offset) {
                    return VK_ERROR_MEMORY_MAP_FAILED;
                }
                auto* output = destination + static_cast<std::size_t>(offset);
                if (zero) {
                    std::memset(output, 0, static_cast<std::size_t>(size));
                } else {
                    for (VkDeviceSize byte = 0; byte < size; byte += texel.size()) {
                        std::memcpy(
                            output + static_cast<std::size_t>(byte),
                            texel.data(),
                            texel.size()
                        );
                    }
                }
                markImageSubresourceInitialized(clear.image, mipLevel, arrayLayer);
            }
        }
    }
    return VK_SUCCESS;
}

VkResult executeImageCopyLocked(const RecordedImageCopy& copy) {
    std::uint8_t* source = nullptr;
    std::uint8_t* destination = nullptr;
    VkDeviceSize sourceSize = 0;
    VkDeviceSize destinationSize = 0;
    VkResult result = imageHostBytesLocked(copy.source, &source, &sourceSize);
    if (result != VK_SUCCESS) return result;
    result = imageHostBytesLocked(copy.destination, &destination, &destinationSize);
    if (result != VK_SUCCESS) return result;
    const auto sourceBlock = formatBlockInfo(copy.source->format);
    const auto destinationBlock = formatBlockInfo(copy.destination->format);
    if (sourceBlock.width != destinationBlock.width
        || sourceBlock.height != destinationBlock.height
        || sourceBlock.depth != destinationBlock.depth
        || sourceBlock.bytes != destinationBlock.bytes
        || sourceBlock.bytes == 0) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    for (const auto& region : copy.regions) {
        const VkExtent3D sourceExtent{
            std::max(1U, copy.source->extent.width >> region.srcSubresource.mipLevel),
            std::max(1U, copy.source->extent.height >> region.srcSubresource.mipLevel),
            std::max(1U, copy.source->extent.depth >> region.srcSubresource.mipLevel),
        };
        const VkExtent3D destinationExtent{
            std::max(1U, copy.destination->extent.width >> region.dstSubresource.mipLevel),
            std::max(1U, copy.destination->extent.height >> region.dstSubresource.mipLevel),
            std::max(1U, copy.destination->extent.depth >> region.dstSubresource.mipLevel),
        };
        const std::uint32_t blockWidth = sourceBlock.width;
        const std::uint32_t blockHeight = sourceBlock.height;
        const std::uint32_t blockDepth = sourceBlock.depth;
        const std::uint32_t copyBlocksX =
            (region.extent.width + blockWidth - 1) / blockWidth;
        const std::uint32_t copyBlocksY =
            (region.extent.height + blockHeight - 1) / blockHeight;
        const std::uint32_t copyBlocksZ =
            (region.extent.depth + blockDepth - 1) / blockDepth;
        const std::uint32_t sourceBlocksX =
            (sourceExtent.width + blockWidth - 1) / blockWidth;
        const std::uint32_t sourceBlocksY =
            (sourceExtent.height + blockHeight - 1) / blockHeight;
        const std::uint32_t destinationBlocksX =
            (destinationExtent.width + blockWidth - 1) / blockWidth;
        const std::uint32_t destinationBlocksY =
            (destinationExtent.height + blockHeight - 1) / blockHeight;
        const VkDeviceSize rowBytes =
            static_cast<VkDeviceSize>(copyBlocksX) * sourceBlock.bytes;
        for (std::uint32_t layer = 0;
             layer < region.srcSubresource.layerCount;
             ++layer) {
            const std::uint32_t sourceLayer = region.srcSubresource.baseArrayLayer + layer;
            const std::uint32_t destinationLayer =
                region.dstSubresource.baseArrayLayer + layer;
            const VkDeviceSize sourceBase = imageSubresourceByteOffset(
                copy.source->format,
                copy.source->extent,
                copy.source->mipLevels,
                region.srcSubresource.mipLevel,
                sourceLayer
            );
            const VkDeviceSize destinationBase = imageSubresourceByteOffset(
                copy.destination->format,
                copy.destination->extent,
                copy.destination->mipLevels,
                region.dstSubresource.mipLevel,
                destinationLayer
            );
            for (std::uint32_t z = 0; z < copyBlocksZ; ++z) {
                for (std::uint32_t y = 0; y < copyBlocksY; ++y) {
                    const VkDeviceSize sourceBlockOffset =
                        ((static_cast<VkDeviceSize>(region.srcOffset.z / blockDepth + z)
                            * sourceBlocksY
                            + static_cast<std::uint32_t>(region.srcOffset.y) / blockHeight + y)
                            * sourceBlocksX
                            + static_cast<std::uint32_t>(region.srcOffset.x) / blockWidth)
                        * sourceBlock.bytes;
                    const VkDeviceSize destinationBlockOffset =
                        ((static_cast<VkDeviceSize>(region.dstOffset.z / blockDepth + z)
                            * destinationBlocksY
                            + static_cast<std::uint32_t>(region.dstOffset.y) / blockHeight + y)
                            * destinationBlocksX
                            + static_cast<std::uint32_t>(region.dstOffset.x) / blockWidth)
                        * destinationBlock.bytes;
                    const VkDeviceSize sourceOffset = sourceBase + sourceBlockOffset;
                    const VkDeviceSize destinationOffset =
                        destinationBase + destinationBlockOffset;
                    if (sourceOffset > sourceSize || rowBytes > sourceSize - sourceOffset
                        || destinationOffset > destinationSize
                        || rowBytes > destinationSize - destinationOffset) {
                        return VK_ERROR_MEMORY_MAP_FAILED;
                    }
                    std::memmove(
                        destination + static_cast<std::size_t>(destinationOffset),
                        source + static_cast<std::size_t>(sourceOffset),
                        static_cast<std::size_t>(rowBytes)
                    );
                }
            }
            markImageSubresourceInitialized(
                copy.destination,
                region.dstSubresource.mipLevel,
                destinationLayer
            );
        }
    }
    return VK_SUCCESS;
}

VkResult executeBufferToImageCopyLocked(const RecordedBufferToImageCopy& copy) {
    if (copy.source == nullptr || copy.destination == nullptr
        || copy.source->memory == nullptr) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    const VkResult sourceResult = ensureMemoryBackingLocked(copy.source->memory, false);
    if (sourceResult != VK_SUCCESS) return sourceResult;
    std::uint8_t* destinationBytes = nullptr;
    VkDeviceSize destinationSize = 0;
    const VkResult destinationResult = imageHostBytesLocked(
        copy.destination,
        &destinationBytes,
        &destinationSize
    );
    if (destinationResult != VK_SUCCESS) return destinationResult;

    const FormatBlockInfo block = formatBlockInfo(copy.destination->format);
    if (block.width == 0 || block.height == 0 || block.depth == 0
        || block.bytes == 0) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    const auto ceilDivide = [](std::uint64_t value, std::uint64_t divisor) {
        return (value + divisor - 1) / divisor;
    };
    for (const auto& region : copy.regions) {
        if (region.imageSubresource.aspectMask
                != formatAspectMask(copy.destination->format)
            || region.imageSubresource.mipLevel >= copy.destination->mipLevels
            || region.imageSubresource.layerCount == 0
            || region.imageSubresource.baseArrayLayer >= copy.destination->arrayLayers
            || region.imageSubresource.layerCount
                > copy.destination->arrayLayers
                    - region.imageSubresource.baseArrayLayer
            || region.imageOffset.x < 0 || region.imageOffset.y < 0
            || region.imageOffset.z < 0 || region.imageExtent.depth == 0
            || region.imageExtent.width == 0 || region.imageExtent.height == 0) {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        const VkExtent3D mipExtent{
            std::max(1U, copy.destination->extent.width
                >> region.imageSubresource.mipLevel),
            std::max(1U, copy.destination->extent.height
                >> region.imageSubresource.mipLevel),
            std::max(1U, copy.destination->extent.depth
                >> region.imageSubresource.mipLevel),
        };
        const std::uint32_t destinationX =
            static_cast<std::uint32_t>(region.imageOffset.x);
        const std::uint32_t destinationY =
            static_cast<std::uint32_t>(region.imageOffset.y);
        const std::uint32_t destinationZ =
            static_cast<std::uint32_t>(region.imageOffset.z);
        const std::uint32_t copyWidth = region.imageExtent.width;
        const std::uint32_t copyHeight = region.imageExtent.height;
        const std::uint32_t copyDepth = region.imageExtent.depth;
        if (destinationX > mipExtent.width
            || copyWidth > mipExtent.width - destinationX
            || destinationY > mipExtent.height
            || copyHeight > mipExtent.height - destinationY
            || destinationZ > mipExtent.depth
            || copyDepth > mipExtent.depth - destinationZ
            || destinationX % block.width != 0
            || destinationY % block.height != 0
            || destinationZ % block.depth != 0
            || (copyWidth % block.width != 0
                && destinationX + copyWidth != mipExtent.width)
            || (copyHeight % block.height != 0
                && destinationY + copyHeight != mipExtent.height)
            || (copyDepth % block.depth != 0
                && destinationZ + copyDepth != mipExtent.depth)
            || (region.bufferRowLength != 0
                && region.bufferRowLength % block.width != 0)
            || (region.bufferImageHeight != 0
                && region.bufferImageHeight % block.height != 0)) {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }

        const std::uint64_t copyBlocksX = ceilDivide(copyWidth, block.width);
        const std::uint64_t copyBlocksY = ceilDivide(copyHeight, block.height);
        const std::uint64_t copyBlocksZ = ceilDivide(copyDepth, block.depth);
        const std::uint64_t sourceRowBlocks = region.bufferRowLength == 0
            ? copyBlocksX
            : ceilDivide(region.bufferRowLength, block.width);
        const std::uint64_t sourceImageBlockRows = region.bufferImageHeight == 0
            ? copyBlocksY
            : ceilDivide(region.bufferImageHeight, block.height);
        if (sourceRowBlocks < copyBlocksX
            || sourceImageBlockRows < copyBlocksY) {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        const std::uint64_t destinationBlocksX =
            ceilDivide(mipExtent.width, block.width);
        const std::uint64_t destinationBlocksY =
            ceilDivide(mipExtent.height, block.height);
        const std::uint64_t destinationXBlock = destinationX / block.width;
        const std::uint64_t destinationYBlock = destinationY / block.height;
        const std::uint64_t destinationZBlock = destinationZ / block.depth;
        const VkDeviceSize sourceImageStride = clampedMultiply(
            clampedMultiply(sourceRowBlocks, sourceImageBlockRows),
            block.bytes
        );
        const VkDeviceSize sourceLayerStride = clampedMultiply(
            sourceImageStride,
            copyBlocksZ
        );
        const VkDeviceSize rowBytes = clampedMultiply(copyBlocksX, block.bytes);
        for (std::uint32_t layer = 0;
             layer < region.imageSubresource.layerCount;
             ++layer) {
            const std::uint32_t arrayLayer =
                region.imageSubresource.baseArrayLayer + layer;
            const VkDeviceSize destinationBase = clampedAdd(
                imageSubresourceByteOffset(
                    copy.destination->format,
                    copy.destination->extent,
                    copy.destination->mipLevels,
                    region.imageSubresource.mipLevel,
                    arrayLayer
                ),
                clampedMultiply(
                    (destinationZBlock * destinationBlocksY + destinationYBlock)
                        * destinationBlocksX + destinationXBlock,
                    block.bytes
                )
            );
            for (std::uint64_t z = 0; z < copyBlocksZ; ++z) {
                for (std::uint64_t row = 0; row < copyBlocksY; ++row) {
                    const VkDeviceSize sourceRelativeOffset = clampedAdd(
                        region.bufferOffset,
                        clampedAdd(
                            clampedMultiply(layer, sourceLayerStride),
                            clampedAdd(
                                clampedMultiply(z, sourceImageStride),
                                clampedMultiply(
                                    clampedMultiply(row, sourceRowBlocks),
                                    block.bytes
                                )
                            )
                        )
                    );
                    const VkDeviceSize sourceOffset = clampedAdd(
                        copy.source->memoryOffset,
                        sourceRelativeOffset
                    );
                    const VkDeviceSize destinationOffset = clampedAdd(
                        destinationBase,
                        clampedMultiply(
                            (z * destinationBlocksY + row) * destinationBlocksX,
                            block.bytes
                        )
                    );
                    if (sourceRelativeOffset > copy.source->size
                        || rowBytes > copy.source->size - sourceRelativeOffset
                        || sourceOffset > copy.source->memory->size
                        || rowBytes > copy.source->memory->size - sourceOffset
                        || destinationOffset > destinationSize
                        || rowBytes > destinationSize - destinationOffset) {
                        return VK_ERROR_MEMORY_MAP_FAILED;
                    }
                    std::memcpy(
                        destinationBytes + static_cast<std::size_t>(destinationOffset),
                        copy.source->memory->bytes.data()
                            + static_cast<std::size_t>(sourceOffset),
                        static_cast<std::size_t>(rowBytes)
                    );
                }
            }
            markImageSubresourceInitialized(
                copy.destination,
                region.imageSubresource.mipLevel,
                arrayLayer
            );
        }
    }
    return VK_SUCCESS;
}

VkResult executeImageToBufferCopyLocked(const RecordedImageToBufferCopy& copy) {
    if (copy.source == nullptr || copy.destination == nullptr
        || copy.destination->memory == nullptr) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    const VkResult destinationResult = ensureMemoryBackingLocked(
        copy.destination->memory,
        false
    );
    if (destinationResult != VK_SUCCESS) return destinationResult;
    std::uint8_t* sourceBytes = nullptr;
    VkDeviceSize sourceSize = 0;
    const VkResult sourceResult = imageHostBytesLocked(
        copy.source,
        &sourceBytes,
        &sourceSize
    );
    if (sourceResult != VK_SUCCESS) return sourceResult;
    const FormatBlockInfo block = formatBlockInfo(copy.source->format);
    if (block.width == 0 || block.height == 0 || block.depth == 0
        || block.bytes == 0) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    const auto ceilDivide = [](std::uint64_t value, std::uint64_t divisor) {
        return (value + divisor - 1) / divisor;
    };
    VkDeviceSize dirtyStart = VK_WHOLE_SIZE;
    VkDeviceSize dirtyEnd = 0;
    for (const auto& region : copy.regions) {
        if (region.imageSubresource.aspectMask != formatAspectMask(copy.source->format)
            || region.imageSubresource.mipLevel >= copy.source->mipLevels
            || region.imageSubresource.layerCount == 0
            || region.imageSubresource.baseArrayLayer >= copy.source->arrayLayers
            || region.imageSubresource.layerCount
                > copy.source->arrayLayers - region.imageSubresource.baseArrayLayer
            || region.imageOffset.x < 0 || region.imageOffset.y < 0
            || region.imageOffset.z < 0 || region.imageExtent.depth == 0
            || region.imageExtent.width == 0 || region.imageExtent.height == 0) {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        const VkExtent3D mipExtent{
            std::max(1U, copy.source->extent.width
                >> region.imageSubresource.mipLevel),
            std::max(1U, copy.source->extent.height
                >> region.imageSubresource.mipLevel),
            std::max(1U, copy.source->extent.depth
                >> region.imageSubresource.mipLevel),
        };
        const std::uint32_t sourceX = static_cast<std::uint32_t>(region.imageOffset.x);
        const std::uint32_t sourceY = static_cast<std::uint32_t>(region.imageOffset.y);
        const std::uint32_t sourceZ = static_cast<std::uint32_t>(region.imageOffset.z);
        const std::uint32_t copyWidth = region.imageExtent.width;
        const std::uint32_t copyHeight = region.imageExtent.height;
        const std::uint32_t copyDepth = region.imageExtent.depth;
        if (sourceX > mipExtent.width || copyWidth > mipExtent.width - sourceX
            || sourceY > mipExtent.height || copyHeight > mipExtent.height - sourceY
            || sourceZ > mipExtent.depth || copyDepth > mipExtent.depth - sourceZ
            || sourceX % block.width != 0 || sourceY % block.height != 0
            || sourceZ % block.depth != 0
            || (copyWidth % block.width != 0 && sourceX + copyWidth != mipExtent.width)
            || (copyHeight % block.height != 0 && sourceY + copyHeight != mipExtent.height)
            || (copyDepth % block.depth != 0 && sourceZ + copyDepth != mipExtent.depth)
            || (region.bufferRowLength != 0
                && region.bufferRowLength % block.width != 0)
            || (region.bufferImageHeight != 0
                && region.bufferImageHeight % block.height != 0)) {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        const std::uint64_t copyBlocksX = ceilDivide(copyWidth, block.width);
        const std::uint64_t copyBlocksY = ceilDivide(copyHeight, block.height);
        const std::uint64_t copyBlocksZ = ceilDivide(copyDepth, block.depth);
        const std::uint64_t destinationRowBlocks = region.bufferRowLength == 0
            ? copyBlocksX
            : ceilDivide(region.bufferRowLength, block.width);
        const std::uint64_t destinationImageBlockRows =
            region.bufferImageHeight == 0
                ? copyBlocksY
                : ceilDivide(region.bufferImageHeight, block.height);
        if (destinationRowBlocks < copyBlocksX
            || destinationImageBlockRows < copyBlocksY) {
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
        }
        const std::uint64_t sourceBlocksX = ceilDivide(mipExtent.width, block.width);
        const std::uint64_t sourceBlocksY = ceilDivide(mipExtent.height, block.height);
        const std::uint64_t sourceXBlock = sourceX / block.width;
        const std::uint64_t sourceYBlock = sourceY / block.height;
        const std::uint64_t sourceZBlock = sourceZ / block.depth;
        const VkDeviceSize destinationImageStride = clampedMultiply(
            clampedMultiply(destinationRowBlocks, destinationImageBlockRows),
            block.bytes
        );
        const VkDeviceSize destinationLayerStride = clampedMultiply(
            destinationImageStride,
            copyBlocksZ
        );
        const VkDeviceSize destinationBase = clampedAdd(
            copy.destination->memoryOffset,
            region.bufferOffset
        );
        const VkDeviceSize rowBytes = clampedMultiply(copyBlocksX, block.bytes);
        for (std::uint32_t layer = 0;
             layer < region.imageSubresource.layerCount;
             ++layer) {
            const std::uint32_t arrayLayer =
                region.imageSubresource.baseArrayLayer + layer;
            if (copy.source->memory != nullptr) {
                const VkResult initializedResult = ensureLinearImageMipInitializedLocked(
                    copy.source,
                    region.imageSubresource.mipLevel,
                    arrayLayer
                );
                if (initializedResult != VK_SUCCESS) return initializedResult;
            }
            const VkDeviceSize sourceBase = clampedAdd(
                imageSubresourceByteOffset(
                    copy.source->format,
                    copy.source->extent,
                    copy.source->mipLevels,
                    region.imageSubresource.mipLevel,
                    arrayLayer
                ),
                clampedMultiply(
                    (sourceZBlock * sourceBlocksY + sourceYBlock)
                        * sourceBlocksX + sourceXBlock,
                    block.bytes
                )
            );
            for (std::uint64_t z = 0; z < copyBlocksZ; ++z) {
                for (std::uint64_t row = 0; row < copyBlocksY; ++row) {
                    const VkDeviceSize sourceOffset = clampedAdd(
                        sourceBase,
                        clampedMultiply(
                            (z * sourceBlocksY + row) * sourceBlocksX,
                            block.bytes
                        )
                    );
                    const VkDeviceSize destinationOffset = clampedAdd(
                        destinationBase,
                        clampedAdd(
                            clampedMultiply(layer, destinationLayerStride),
                            clampedAdd(
                                clampedMultiply(z, destinationImageStride),
                                clampedMultiply(
                                    clampedMultiply(row, destinationRowBlocks),
                                    block.bytes
                                )
                            )
                        )
                    );
                    if (sourceOffset > sourceSize || rowBytes > sourceSize - sourceOffset
                        || destinationOffset > copy.destination->memory->size
                        || rowBytes > copy.destination->memory->size - destinationOffset
                        || destinationOffset < copy.destination->memoryOffset
                        || destinationOffset - copy.destination->memoryOffset
                            > copy.destination->size
                        || rowBytes > copy.destination->size
                            - (destinationOffset - copy.destination->memoryOffset)) {
                        return VK_ERROR_MEMORY_MAP_FAILED;
                    }
                    std::memcpy(
                        copy.destination->memory->bytes.data()
                            + static_cast<std::size_t>(destinationOffset),
                        sourceBytes + static_cast<std::size_t>(sourceOffset),
                        static_cast<std::size_t>(rowBytes)
                    );
                    dirtyStart = std::min(dirtyStart, destinationOffset);
                    dirtyEnd = std::max(dirtyEnd, destinationOffset + rowBytes);
                }
            }
        }
    }
    if (dirtyStart != VK_WHOLE_SIZE && dirtyEnd > dirtyStart) {
        copy.destination->memory->dirtyOffset = std::min(
            copy.destination->memory->dirtyOffset,
            dirtyStart
        );
        copy.destination->memory->dirtyEnd = std::max(
            copy.destination->memory->dirtyEnd,
            dirtyEnd
        );
        const VkResult syncResult = syncMemoryToExternalFDLocked(
            copy.destination->memory,
            dirtyStart,
            dirtyEnd - dirtyStart
        );
        if (syncResult != VK_SUCCESS) return syncResult;
    }
    return VK_SUCCESS;
}

VkResult syncTransferredImageLocked(ImageState* image) {
    if (image == nullptr) {
        return VK_SUCCESS;
    }
    if ((image->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) != 0
        && image->memory == nullptr) {
        if (!image->dirty || image->sparseBytes.empty()) return VK_SUCCESS;
        const VkResult backingResult = ensureSparseImageBackingLocked(image);
        if (backingResult != VK_SUCCESS) return backingResult;
        const VkDeviceSize tightSize = clampedMultiply(
            imageLayerByteSize(image->format, image->extent, image->mipLevels),
            image->arrayLayers
        );
        if (tightSize == 0 || tightSize > image->sparseBytes.size()) {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        try {
            gState.bridge->writeImage(
                image->resourceID,
                image->sparseBytes.data(),
                tightSize
            );
            image->dirty = false;
            if (traceEnabled() || rayTracingTraceEnabled()) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: uploaded sparse image to Metal resource=%llu format=%d extent=%ux%u mips=%u layers=%u bytes=%llu\n",
                    static_cast<unsigned long long>(image->resourceID),
                    image->format,
                    image->extent.width,
                    image->extent.height,
                    image->mipLevels,
                    image->arrayLayers,
                    static_cast<unsigned long long>(tightSize)
                );
            }
        } catch (const std::exception& error) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: sparse image upload to Metal failed resource=%llu: %s\n",
                static_cast<unsigned long long>(image->resourceID),
                error.what()
            );
            return VK_ERROR_DEVICE_LOST;
        }
        return VK_SUCCESS;
    }
    if (image->memory == nullptr || image->memory->externalFD < 0) {
        return VK_SUCCESS;
    }
    image->memory->dirtyOffset =
        std::min(image->memory->dirtyOffset, image->memoryOffset);
    image->memory->dirtyEnd =
        std::max(image->memory->dirtyEnd, image->memoryOffset + image->size);
    const VkResult result = syncMemoryToExternalFDLocked(
        image->memory,
        image->memoryOffset,
        image->size
    );
    if (result == VK_SUCCESS) {
        image->memory->dirtyOffset = VK_WHOLE_SIZE;
        image->memory->dirtyEnd = 0;
    }
    return result;
}

VkResult bridgePrimitiveAccelerationStructureBuildLocked(
    const RecordedAccelerationStructureBuildKHR& build
);
VkResult bridgeInstanceAccelerationStructureBuildLocked(
    const RecordedAccelerationStructureBuildKHR& build
);
VkResult bridgeComputeDispatchLocked(const RecordedComputeDispatch& dispatch);

VkResult executeRecordedCommandsLocked(
    CommandBufferState* command,
    std::unordered_set<ImageState*>& dirtyImages
) {
    std::size_t copyIndex = 0;
    std::size_t updateIndex = 0;
    std::size_t fillIndex = 0;
    std::size_t clearIndex = 0;
    std::size_t imageCopyIndex = 0;
    std::size_t bufferToImageCopyIndex = 0;
    std::size_t imageToBufferCopyIndex = 0;
    std::size_t accelerationBuildIndex = 0;
    std::size_t computeDispatchIndex = 0;
    while (copyIndex < command->bufferCopies.size()
        || updateIndex < command->bufferUpdates.size()
        || fillIndex < command->bufferFills.size()
        || clearIndex < command->imageClears.size()
        || imageCopyIndex < command->imageCopies.size()
        || bufferToImageCopyIndex < command->bufferToImageCopies.size()
        || imageToBufferCopyIndex < command->imageToBufferCopies.size()
        || accelerationBuildIndex
            < command->accelerationStructureBuildsKHR.size()
        || computeDispatchIndex < command->computeDispatches.size()) {
        std::uint64_t nextSequence = UINT64_MAX;
        enum class RecordedKind {
            None,
            BufferCopy,
            BufferUpdate,
            BufferFill,
            ImageClear,
            ImageCopy,
            BufferToImageCopy,
            ImageToBufferCopy,
            AccelerationStructureBuild,
            ComputeDispatch,
        };
        RecordedKind kind = RecordedKind::None;
        auto consider = [&nextSequence, &kind](
                            std::uint64_t sequence,
                            RecordedKind candidate
                        ) {
            if (sequence < nextSequence) {
                nextSequence = sequence;
                kind = candidate;
            }
        };
        if (copyIndex < command->bufferCopies.size()) {
            consider(
                command->bufferCopies[copyIndex].sequence,
                RecordedKind::BufferCopy
            );
        }
        if (updateIndex < command->bufferUpdates.size()) {
            consider(
                command->bufferUpdates[updateIndex].sequence,
                RecordedKind::BufferUpdate
            );
        }
        if (fillIndex < command->bufferFills.size()) {
            consider(
                command->bufferFills[fillIndex].sequence,
                RecordedKind::BufferFill
            );
        }
        if (clearIndex < command->imageClears.size()) {
            consider(
                command->imageClears[clearIndex].sequence,
                RecordedKind::ImageClear
            );
        }
        if (imageCopyIndex < command->imageCopies.size()) {
            consider(
                command->imageCopies[imageCopyIndex].sequence,
                RecordedKind::ImageCopy
            );
        }
        if (bufferToImageCopyIndex < command->bufferToImageCopies.size()) {
            consider(
                command->bufferToImageCopies[bufferToImageCopyIndex].sequence,
                RecordedKind::BufferToImageCopy
            );
        }
        if (imageToBufferCopyIndex < command->imageToBufferCopies.size()) {
            consider(
                command->imageToBufferCopies[imageToBufferCopyIndex].sequence,
                RecordedKind::ImageToBufferCopy
            );
        }
        if (accelerationBuildIndex
            < command->accelerationStructureBuildsKHR.size()) {
            consider(
                command->accelerationStructureBuildsKHR[
                    accelerationBuildIndex
                ].sequence,
                RecordedKind::AccelerationStructureBuild
            );
        }
        if (computeDispatchIndex < command->computeDispatches.size()) {
            consider(
                command->computeDispatches[computeDispatchIndex].sequence,
                RecordedKind::ComputeDispatch
            );
        }

        VkResult result = VK_SUCCESS;
        switch (kind) {
        case RecordedKind::BufferCopy: {
            const auto& copy = command->bufferCopies[copyIndex++];
            for (const auto& region : copy.regions) {
                result = copyBufferBytesLocked(copy.source, copy.destination, region);
                if (result != VK_SUCCESS) break;
            }
            if (result == VK_SUCCESS && rayTracingTraceEnabled()
                && ((copy.source->flags | copy.destination->flags)
                    & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) != 0) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: executed sparse buffer copy source=%p destination=%p regions=%zu\n",
                    static_cast<void*>(copy.source),
                    static_cast<void*>(copy.destination),
                    copy.regions.size()
                );
            }
            break;
        }
        case RecordedKind::BufferUpdate: {
            const auto& update = command->bufferUpdates[updateIndex++];
            result = writeBufferBytesLocked(
                update.destination,
                update.offset,
                update.data.data(),
                update.data.size()
            );
            break;
        }
        case RecordedKind::BufferFill:
            result = executeBufferFillLocked(command->bufferFills[fillIndex++]);
            break;
        case RecordedKind::ImageClear: {
            const auto& clear = command->imageClears[clearIndex++];
            result = executeImageClearLocked(clear);
            if (result == VK_SUCCESS) dirtyImages.insert(clear.image);
            break;
        }
        case RecordedKind::ImageCopy: {
            const auto& copy = command->imageCopies[imageCopyIndex++];
            result = executeImageCopyLocked(copy);
            if (result == VK_SUCCESS) dirtyImages.insert(copy.destination);
            break;
        }
        case RecordedKind::BufferToImageCopy: {
            const auto& copy = command->bufferToImageCopies[
                bufferToImageCopyIndex++
            ];
            result = executeBufferToImageCopyLocked(copy);
            if (result == VK_SUCCESS) dirtyImages.insert(copy.destination);
            break;
        }
        case RecordedKind::ImageToBufferCopy: {
            const auto& copy = command->imageToBufferCopies[
                imageToBufferCopyIndex++
            ];
            result = executeImageToBufferCopyLocked(copy);
            break;
        }
        case RecordedKind::AccelerationStructureBuild: {
            const auto& build = command->accelerationStructureBuildsKHR[
                accelerationBuildIndex++
            ];
            result = build.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                ? bridgeInstanceAccelerationStructureBuildLocked(build)
                : bridgePrimitiveAccelerationStructureBuildLocked(build);
            if (result == VK_ERROR_FEATURE_NOT_PRESENT
                && build.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
                // RTX can submit a TLAS whose instance records retain
                // NVIDIA-only/null child addresses. Preserve command order
                // for all completed BLASes, then populate this same TLAS from
                // the visible USD manifest in the fallback below.
                if (rayTracingTraceEnabled()) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: deferring unsupported scene TLAS build to Metal BLAS fallback dst=%p sequence=%llu\n",
                        static_cast<void*>(build.destination),
                        static_cast<unsigned long long>(build.sequence)
                    );
                }
                result = VK_SUCCESS;
            }
            break;
        }
        case RecordedKind::ComputeDispatch: {
            const auto& dispatch = command->computeDispatches[
                computeDispatchIndex++
            ];
            result = bridgeComputeDispatchLocked(dispatch);
            // RTX keeps legal null/partially-bound descriptors in layouts
            // whose shader permutations do not necessarily read them. A
            // missing CPU backing makes only this optional Metal translation
            // ineligible; it must not poison the Vulkan queue.
            if (result == VK_ERROR_INITIALIZATION_FAILED
                || result == VK_ERROR_FEATURE_NOT_PRESENT
                || result == VK_ERROR_FORMAT_NOT_SUPPORTED
                || result == VK_ERROR_MEMORY_MAP_FAILED) {
                if (computeTraceEnabled()) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: skipped unsupported Metal compute pipeline=%llu result=%d sequence=%llu\n",
                        static_cast<unsigned long long>(
                            dispatch.pipeline == nullptr
                                ? 0
                                : dispatch.pipeline->bridgeComputePipelineID
                        ),
                        result,
                        static_cast<unsigned long long>(dispatch.sequence)
                    );
                }
                result = VK_SUCCESS;
            }
            break;
        }
        case RecordedKind::None:
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        if (result != VK_SUCCESS) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: recorded command execution failed kind=%d sequence=%llu result=%d\n",
                static_cast<int>(kind),
                static_cast<unsigned long long>(nextSequence),
                result
            );
            return result;
        }
    }
    return VK_SUCCESS;
}

VkResult uploadBufferForMetalLocked(BufferState* buffer) {
    if (buffer == nullptr || buffer->memory == nullptr || gState.bridge == nullptr) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    auto* memory = buffer->memory;
    const VkResult backingResult = ensureMemoryBackingLocked(memory, true);
    if (backingResult != VK_SUCCESS) return backingResult;
    if (buffer->memoryOffset > memory->size
        || buffer->size > memory->size - buffer->memoryOffset) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    // CUDA writes imported Vulkan allocations through the shared OPAQUE_FD.
    // Pull that exact buffer range back before uploading it to the host Metal
    // resource; such writes do not participate in Vulkan map/flush dirty bits.
    if (memory->externalFD >= 0) {
        const VkResult refreshResult = refreshMemoryFromExternalFDLocked(
            memory,
            buffer->memoryOffset,
            buffer->size
        );
        if (refreshResult != VK_SUCCESS) return refreshResult;
        try {
            gState.bridge->writeBuffer(
                memory->resourceID,
                memory->bytes.data() + buffer->memoryOffset,
                buffer->size,
                buffer->memoryOffset
            );
        } catch (const std::exception& error) {
            std::fprintf(stderr, "imb-vulkan-icd: shared Metal AS buffer upload failed: %s\n", error.what());
            return VK_ERROR_DEVICE_LOST;
        }
        return VK_SUCCESS;
    }
    if (memory->dirtyOffset != VK_WHOLE_SIZE && memory->dirtyEnd > memory->dirtyOffset) {
        try {
            gState.bridge->writeBuffer(
                memory->resourceID,
                memory->bytes.data() + memory->dirtyOffset,
                memory->dirtyEnd - memory->dirtyOffset,
                memory->dirtyOffset
            );
        } catch (const std::exception& error) {
            std::fprintf(stderr, "imb-vulkan-icd: Metal AS buffer upload failed: %s\n", error.what());
            return VK_ERROR_DEVICE_LOST;
        }
        memory->dirtyOffset = VK_WHOLE_SIZE;
        memory->dirtyEnd = 0;
    }
    return VK_SUCCESS;
}

VkResult bridgePrimitiveAccelerationStructureBuildLocked(
    const RecordedAccelerationStructureBuildKHR& build
) {
    if (build.destination == nullptr || build.destination->bridgeResourceID == 0
        || build.type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
        || build.mode != VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
        || build.geometries.empty() || build.geometries.size() != build.ranges.size()) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    std::vector<BridgePrimitiveAccelerationStructureGeometry> bridgeGeometries;
    try {
        bridgeGeometries.reserve(build.geometries.size());
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    std::uint64_t totalTriangleCount = 0;
    std::array<float, 3> triangleBoundsMinimum{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    std::array<float, 3> triangleBoundsMaximum{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
    };
    bool triangleBoundsValid = true;
    for (std::size_t index = 0; index < build.geometries.size(); ++index) {
        const auto& geometry = build.geometries[index];
        const auto& range = build.ranges[index];
        BridgePrimitiveAccelerationStructureGeometry bridgeGeometry{};
        bridgeGeometry.flags = geometry.flags;
        bridgeGeometry.primitiveCount = range.primitiveCount;
        if (range.primitiveCount == 0) return VK_ERROR_INITIALIZATION_FAILED;

        if (geometry.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
            const auto& triangles = geometry.geometry.triangles;
            if (triangles.sType != VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR
                || triangles.vertexFormat != VK_FORMAT_R32G32B32_SFLOAT
                || triangles.vertexStride < 12 || triangles.vertexStride > UINT32_MAX) {
                return VK_ERROR_FORMAT_NOT_SUPPORTED;
            }
            auto* vertexBuffer = bufferForDeviceAddressLocked(build.destination->device, triangles.vertexData.deviceAddress);
            if (vertexBuffer == nullptr || vertexBuffer->memory == nullptr) return VK_ERROR_MEMORY_MAP_FAILED;
            if (rayTracingTraceEnabled()) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: Metal BLAS input destination=%p vertexBuffer=%p address=%#llx size=%llu vertexAddress=%#llx memory=%p memoryOffset=%llu primitives=%u\n",
                    static_cast<void*>(build.destination),
                    static_cast<void*>(vertexBuffer),
                    static_cast<unsigned long long>(vertexBuffer->deviceAddress),
                    static_cast<unsigned long long>(vertexBuffer->size),
                    static_cast<unsigned long long>(triangles.vertexData.deviceAddress),
                    static_cast<void*>(vertexBuffer->memory),
                    static_cast<unsigned long long>(vertexBuffer->memoryOffset),
                    range.primitiveCount
                );
            }
            VkResult uploadResult = uploadBufferForMetalLocked(vertexBuffer);
            if (uploadResult != VK_SUCCESS) return uploadResult;
            const VkDeviceSize vertexAddressOffset = triangles.vertexData.deviceAddress - vertexBuffer->deviceAddress;
            VkDeviceSize vertexOffset = vertexBuffer->memoryOffset + vertexAddressOffset
                + static_cast<VkDeviceSize>(range.firstVertex) * triangles.vertexStride;
            bridgeGeometry.kind = 0;
            bridgeGeometry.dataResourceID = vertexBuffer->memory->resourceID;
            bridgeGeometry.stride = static_cast<std::uint32_t>(triangles.vertexStride);
            bridgeGeometry.vertexFormat = 1;
            BufferState* indexBuffer = nullptr;

            if (triangles.indexType == VK_INDEX_TYPE_NONE_KHR) {
                vertexOffset += range.primitiveOffset;
            } else {
                if (triangles.indexType != VK_INDEX_TYPE_UINT16
                    && triangles.indexType != VK_INDEX_TYPE_UINT32) {
                    return VK_ERROR_FORMAT_NOT_SUPPORTED;
                }
                indexBuffer = bufferForDeviceAddressLocked(
                    build.destination->device,
                    triangles.indexData.deviceAddress
                );
                if (indexBuffer == nullptr || indexBuffer->memory == nullptr) return VK_ERROR_MEMORY_MAP_FAILED;
                if (rayTracingTraceEnabled()) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: Metal BLAS index buffer=%p address=%#llx size=%llu indexAddress=%#llx memory=%p memoryOffset=%llu\n",
                        static_cast<void*>(indexBuffer),
                        static_cast<unsigned long long>(indexBuffer->deviceAddress),
                        static_cast<unsigned long long>(indexBuffer->size),
                        static_cast<unsigned long long>(triangles.indexData.deviceAddress),
                        static_cast<void*>(indexBuffer->memory),
                        static_cast<unsigned long long>(indexBuffer->memoryOffset)
                    );
                }
                uploadResult = uploadBufferForMetalLocked(indexBuffer);
                if (uploadResult != VK_SUCCESS) return uploadResult;
                bridgeGeometry.indexResourceID = indexBuffer->memory->resourceID;
                bridgeGeometry.indexOffset = indexBuffer->memoryOffset
                    + (triangles.indexData.deviceAddress - indexBuffer->deviceAddress)
                    + range.primitiveOffset;
                bridgeGeometry.indexType = triangles.indexType == VK_INDEX_TYPE_UINT16 ? 1U : 2U;
            }
            bridgeGeometry.dataOffset = vertexOffset;

            BufferState* transformBuffer = nullptr;
            if (triangles.transformData.deviceAddress != 0) {
                transformBuffer = bufferForDeviceAddressLocked(
                    build.destination->device,
                    triangles.transformData.deviceAddress
                );
                if (transformBuffer == nullptr || transformBuffer->memory == nullptr) {
                    return VK_ERROR_MEMORY_MAP_FAILED;
                }
                uploadResult = uploadBufferForMetalLocked(transformBuffer);
                if (uploadResult != VK_SUCCESS) return uploadResult;
                bridgeGeometry.transformResourceID = transformBuffer->memory->resourceID;
                bridgeGeometry.transformOffset = transformBuffer->memoryOffset
                    + (triangles.transformData.deviceAddress - transformBuffer->deviceAddress)
                    + range.transformOffset;
            }

            if (totalTriangleCount
                > std::numeric_limits<std::uint64_t>::max() - range.primitiveCount) {
                triangleBoundsValid = false;
            } else {
                totalTriangleCount += range.primitiveCount;
            }
            std::array<float, 12> geometryTransform{
                1.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 1.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 1.0F, 0.0F,
            };
            if (triangleBoundsValid && transformBuffer != nullptr) {
                if (bridgeGeometry.transformOffset
                    > transformBuffer->memory->bytes.size()
                    || sizeof(VkTransformMatrixKHR)
                    > transformBuffer->memory->bytes.size()
                        - bridgeGeometry.transformOffset) {
                    triangleBoundsValid = false;
                } else {
                    for (std::size_t component = 0;
                         component < geometryTransform.size();
                         ++component) {
                        std::uint32_t bits = 0;
                        std::memcpy(
                            &bits,
                            transformBuffer->memory->bytes.data()
                                + bridgeGeometry.transformOffset
                                + component * sizeof(std::uint32_t),
                            sizeof(bits)
                        );
                        geometryTransform[component] = std::bit_cast<float>(bits);
                        if (!std::isfinite(geometryTransform[component])) {
                            triangleBoundsValid = false;
                            break;
                        }
                    }
                }
            }
            const std::uint64_t vertexReferenceCount =
                static_cast<std::uint64_t>(range.primitiveCount) * 3;
            for (std::uint64_t vertexReference = 0;
                 triangleBoundsValid && vertexReference < vertexReferenceCount;
                 ++vertexReference) {
                std::uint64_t vertexIndex = vertexReference;
                if (indexBuffer != nullptr) {
                    const std::uint64_t indexStride =
                        triangles.indexType == VK_INDEX_TYPE_UINT16 ? 2 : 4;
                    if (vertexReference
                        > (std::numeric_limits<std::uint64_t>::max()
                            - bridgeGeometry.indexOffset) / indexStride) {
                        triangleBoundsValid = false;
                        break;
                    }
                    const std::uint64_t indexOffset = bridgeGeometry.indexOffset
                        + vertexReference * indexStride;
                    if (indexOffset > indexBuffer->memory->bytes.size()
                        || indexStride
                            > indexBuffer->memory->bytes.size() - indexOffset) {
                        triangleBoundsValid = false;
                        break;
                    }
                    if (indexStride == 2) {
                        std::uint16_t value = 0;
                        std::memcpy(
                            &value,
                            indexBuffer->memory->bytes.data() + indexOffset,
                            sizeof(value)
                        );
                        vertexIndex = value;
                    } else {
                        std::uint32_t value = 0;
                        std::memcpy(
                            &value,
                            indexBuffer->memory->bytes.data() + indexOffset,
                            sizeof(value)
                        );
                        vertexIndex = value;
                    }
                }
                if (vertexIndex
                    > (std::numeric_limits<std::uint64_t>::max()
                        - bridgeGeometry.dataOffset) / bridgeGeometry.stride) {
                    triangleBoundsValid = false;
                    break;
                }
                const std::uint64_t pointOffset = bridgeGeometry.dataOffset
                    + vertexIndex * bridgeGeometry.stride;
                if (pointOffset > vertexBuffer->memory->bytes.size()
                    || 12 > vertexBuffer->memory->bytes.size() - pointOffset) {
                    triangleBoundsValid = false;
                    break;
                }
                std::array<float, 3> point{};
                for (std::size_t component = 0; component < point.size(); ++component) {
                    std::uint32_t bits = 0;
                    std::memcpy(
                        &bits,
                        vertexBuffer->memory->bytes.data()
                            + pointOffset + component * sizeof(std::uint32_t),
                        sizeof(bits)
                    );
                    point[component] = std::bit_cast<float>(bits);
                    if (!std::isfinite(point[component])) {
                        triangleBoundsValid = false;
                        break;
                    }
                }
                if (!triangleBoundsValid) break;
                const std::array<float, 3> transformed{
                    geometryTransform[0] * point[0]
                        + geometryTransform[1] * point[1]
                        + geometryTransform[2] * point[2]
                        + geometryTransform[3],
                    geometryTransform[4] * point[0]
                        + geometryTransform[5] * point[1]
                        + geometryTransform[6] * point[2]
                        + geometryTransform[7],
                    geometryTransform[8] * point[0]
                        + geometryTransform[9] * point[1]
                        + geometryTransform[10] * point[2]
                        + geometryTransform[11],
                };
                for (std::size_t component = 0; component < 3; ++component) {
                    if (!std::isfinite(transformed[component])) {
                        triangleBoundsValid = false;
                        break;
                    }
                    triangleBoundsMinimum[component] = std::min(
                        triangleBoundsMinimum[component],
                        transformed[component]
                    );
                    triangleBoundsMaximum[component] = std::max(
                        triangleBoundsMaximum[component],
                        transformed[component]
                    );
                }
            }
        } else if (geometry.geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
            triangleBoundsValid = false;
            const auto& aabbs = geometry.geometry.aabbs;
            if (aabbs.sType != VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR
                || aabbs.stride < 24 || aabbs.stride > UINT32_MAX) {
                return VK_ERROR_FORMAT_NOT_SUPPORTED;
            }
            auto* dataBuffer = bufferForDeviceAddressLocked(build.destination->device, aabbs.data.deviceAddress);
            if (dataBuffer == nullptr || dataBuffer->memory == nullptr) return VK_ERROR_MEMORY_MAP_FAILED;
            const VkResult uploadResult = uploadBufferForMetalLocked(dataBuffer);
            if (uploadResult != VK_SUCCESS) return uploadResult;
            bridgeGeometry.kind = 1;
            bridgeGeometry.dataResourceID = dataBuffer->memory->resourceID;
            bridgeGeometry.dataOffset = dataBuffer->memoryOffset
                + (aabbs.data.deviceAddress - dataBuffer->deviceAddress)
                + range.primitiveOffset;
            bridgeGeometry.stride = static_cast<std::uint32_t>(aabbs.stride);
        } else {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        try {
            bridgeGeometries.push_back(bridgeGeometry);
        } catch (const std::bad_alloc&) {
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }

    try {
        gState.bridge->buildPrimitiveAccelerationStructure(
            build.destination->bridgeResourceID,
            build.flags,
            bridgeGeometries
        );
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: Metal primitive AS build failed: %s\n", error.what());
        return VK_ERROR_DEVICE_LOST;
    }
    build.destination->triangleCount = totalTriangleCount;
    build.destination->hasTriangleBounds =
        triangleBoundsValid && totalTriangleCount != 0;
    if (build.destination->hasTriangleBounds) {
        build.destination->triangleBoundsMinimum = triangleBoundsMinimum;
        build.destination->triangleBoundsMaximum = triangleBoundsMaximum;
    }
    build.destination->built = true;
    build.destination->builtFromSceneState = false;
    build.destination->sceneStateSequence = 0;
    build.destination->sceneStateMeshSequence = 0;
    std::fprintf(
        stderr,
        "imb-vulkan-icd: Metal primitive AS built host=%llu geometries=%zu triangles=%llu bounds=%s[(%.6g,%.6g,%.6g),(%.6g,%.6g,%.6g)]\n",
        static_cast<unsigned long long>(build.destination->bridgeResourceID),
        bridgeGeometries.size(),
        static_cast<unsigned long long>(build.destination->triangleCount),
        build.destination->hasTriangleBounds ? "" : "unavailable ",
        build.destination->triangleBoundsMinimum[0],
        build.destination->triangleBoundsMinimum[1],
        build.destination->triangleBoundsMinimum[2],
        build.destination->triangleBoundsMaximum[0],
        build.destination->triangleBoundsMaximum[1],
        build.destination->triangleBoundsMaximum[2]
    );
    return VK_SUCCESS;
}

VkResult bridgeInstanceAccelerationStructureBuildLocked(
    const RecordedAccelerationStructureBuildKHR& build
) {
    if (build.destination == nullptr || build.destination->bridgeResourceID == 0
        || build.type == VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
        || build.mode != VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR
        || build.geometries.size() != 1 || build.ranges.size() != 1) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    const auto& geometry = build.geometries[0];
    const auto& range = build.ranges[0];
    if (geometry.geometryType != VK_GEOMETRY_TYPE_INSTANCES_KHR
        || geometry.geometry.instances.sType
            != VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR
        || geometry.geometry.instances.data.deviceAddress == 0
        || range.primitiveCount == 0 || range.primitiveCount > 1048576) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    const auto& instanceData = geometry.geometry.instances;
    const VkDeviceSize inputStride = instanceData.arrayOfPointers == VK_TRUE
        ? sizeof(VkDeviceAddress)
        : sizeof(VkAccelerationStructureInstanceKHR);
    const VkDeviceAddress baseAddress = instanceData.data.deviceAddress;
    if (range.primitiveOffset > std::numeric_limits<VkDeviceAddress>::max() - baseAddress) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    const VkDeviceAddress firstAddress = baseAddress + range.primitiveOffset;

    std::vector<BridgeInstanceAccelerationStructureInstance> bridgeInstances;
    try {
        bridgeInstances.reserve(range.primitiveCount);
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    for (std::uint32_t index = 0; index < range.primitiveCount; ++index) {
        const VkDeviceAddress recordStride = static_cast<VkDeviceAddress>(index) * inputStride;
        if (recordStride > std::numeric_limits<VkDeviceAddress>::max() - firstAddress) {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        VkDeviceAddress recordAddress = firstAddress + recordStride;
        if (instanceData.arrayOfPointers == VK_TRUE) {
            imb::Bytes pointerBytes;
            const VkResult pointerResult = copyDeviceAddressBytesLocked(
                build.destination->device,
                recordAddress,
                sizeof(VkDeviceAddress),
                pointerBytes
            );
            if (pointerResult != VK_SUCCESS) return pointerResult;
            recordAddress = imb::readLittleEndian<std::uint64_t>(pointerBytes, 0);
            if (recordAddress == 0) return VK_ERROR_FEATURE_NOT_PRESENT;
        }

        imb::Bytes record;
        const VkResult recordResult = copyDeviceAddressBytesLocked(
            build.destination->device,
            recordAddress,
            sizeof(VkAccelerationStructureInstanceKHR),
            record
        );
        if (recordResult != VK_SUCCESS) return recordResult;

        BridgeInstanceAccelerationStructureInstance bridgeInstance{};
        for (std::size_t component = 0; component < bridgeInstance.transformationBits.size(); ++component) {
            bridgeInstance.transformationBits[component] =
                imb::readLittleEndian<std::uint32_t>(record, component * sizeof(std::uint32_t));
        }
        const std::uint32_t customIndexAndMask = imb::readLittleEndian<std::uint32_t>(record, 48);
        const std::uint32_t offsetAndFlags = imb::readLittleEndian<std::uint32_t>(record, 52);
        const VkDeviceAddress childAddress = imb::readLittleEndian<std::uint64_t>(record, 56);
        auto* child = accelerationStructureForDeviceAddressLocked(build.destination->device, childAddress);
        if (child == nullptr || child == build.destination || !child->built
            || child->bridgeResourceID == 0
            || child->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        bridgeInstance.userID = customIndexAndMask & 0x00ff'ffffU;
        bridgeInstance.mask = customIndexAndMask >> 24;
        bridgeInstance.intersectionFunctionTableOffset = offsetAndFlags & 0x00ff'ffffU;
        bridgeInstance.options = offsetAndFlags >> 24;
        if ((bridgeInstance.options & ~0xfU) != 0) return VK_ERROR_FEATURE_NOT_PRESENT;
        bridgeInstance.accelerationStructureResourceID = child->bridgeResourceID;
        try {
            bridgeInstances.push_back(bridgeInstance);
        } catch (const std::bad_alloc&) {
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
    }

    try {
        gState.bridge->buildInstanceAccelerationStructure(
            build.destination->bridgeResourceID,
            build.flags,
            bridgeInstances
        );
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: Metal instance AS build failed: %s\n", error.what());
        return VK_ERROR_DEVICE_LOST;
    }
    build.destination->built = true;
    build.destination->builtFromSceneState = false;
    build.destination->sceneStateSequence = 0;
    build.destination->sceneStateMeshSequence = 0;
    std::fprintf(
        stderr,
        "imb-vulkan-icd: Metal instance AS built host=%llu instances=%zu\n",
        static_cast<unsigned long long>(build.destination->bridgeResourceID),
        bridgeInstances.size()
    );
    return VK_SUCCESS;
}

std::pair<std::array<float, 3>, std::array<float, 3>> transformedSceneMeshBounds(
    const BridgeSceneMesh& mesh
) {
    std::array<float, 3> minimum{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
    };
    std::array<float, 3> maximum{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
    };
    for (std::uint32_t corner = 0; corner < 8; ++corner) {
        const std::array<float, 3> point{
            (corner & 1u) == 0
                ? mesh.boundsMinimum[0] : mesh.boundsMaximum[0],
            (corner & 2u) == 0
                ? mesh.boundsMinimum[1] : mesh.boundsMaximum[1],
            (corner & 4u) == 0
                ? mesh.boundsMinimum[2] : mesh.boundsMaximum[2],
        };
        const std::array<float, 3> transformed{
            mesh.worldTransform[0] * point[0]
                + mesh.worldTransform[1] * point[1]
                + mesh.worldTransform[2] * point[2]
                + mesh.worldTransform[3],
            mesh.worldTransform[4] * point[0]
                + mesh.worldTransform[5] * point[1]
                + mesh.worldTransform[6] * point[2]
                + mesh.worldTransform[7],
            mesh.worldTransform[8] * point[0]
                + mesh.worldTransform[9] * point[1]
                + mesh.worldTransform[10] * point[2]
                + mesh.worldTransform[11],
        };
        for (std::size_t component = 0; component < 3; ++component) {
            minimum[component] = std::min(minimum[component], transformed[component]);
            maximum[component] = std::max(maximum[component], transformed[component]);
        }
    }
    return {minimum, maximum};
}

float sceneBoundsError(
    const std::array<float, 3>& leftMinimum,
    const std::array<float, 3>& leftMaximum,
    const std::array<float, 3>& rightMinimum,
    const std::array<float, 3>& rightMaximum
) {
    float centerDistanceSquared = 0.0F;
    float extentDistanceSquared = 0.0F;
    float leftExtentSquared = 0.0F;
    float rightExtentSquared = 0.0F;
    for (std::size_t component = 0; component < 3; ++component) {
        const float leftCenter =
            (leftMinimum[component] + leftMaximum[component]) * 0.5F;
        const float rightCenter =
            (rightMinimum[component] + rightMaximum[component]) * 0.5F;
        const float leftExtent = leftMaximum[component] - leftMinimum[component];
        const float rightExtent = rightMaximum[component] - rightMinimum[component];
        const float centerDelta = leftCenter - rightCenter;
        const float extentDelta = leftExtent - rightExtent;
        centerDistanceSquared += centerDelta * centerDelta;
        extentDistanceSquared += extentDelta * extentDelta;
        leftExtentSquared += leftExtent * leftExtent;
        rightExtentSquared += rightExtent * rightExtent;
    }
    const float scale = std::max(
        std::max(std::sqrt(leftExtentSquared), std::sqrt(rightExtentSquared)),
        0.0001F
    );
    return (std::sqrt(centerDistanceSquared)
        + 0.5F * std::sqrt(extentDistanceSquared)) / scale;
}

std::uint64_t bridgeSceneMeshAccelerationStructureLocked(
    VkDevice device,
    const BridgeSceneMesh& mesh
) {
    auto appendHashBytes = [](std::uint64_t& hash, const void* data, std::size_t size) {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t index = 0; index < size; ++index) {
            hash ^= bytes[index];
            hash *= UINT64_C(1099511628211);
        }
    };
    std::uint64_t contentHash = UINT64_C(14695981039346656037);
    const auto appendScalar = [&appendHashBytes, &contentHash](const auto& value) {
        appendHashBytes(contentHash, &value, sizeof(value));
    };
    const auto appendVector = [&appendHashBytes, &contentHash](const auto& values) {
        if (!values.empty()) {
            appendHashBytes(
                contentHash,
                values.data(),
                values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type)
            );
        }
    };
    appendScalar(mesh.triangleCount);
    appendScalar(mesh.materialFlags);
    appendVector(mesh.vertices);
    appendVector(mesh.indices);
    appendVector(mesh.cornerNormals);
    appendVector(mesh.cornerUVs);
    appendScalar(mesh.textureWidth);
    appendScalar(mesh.textureHeight);
    if (mesh.textureRGBA8) appendVector(*mesh.textureRGBA8);
    appendScalar(mesh.roughness);
    appendScalar(mesh.metallic);
    for (const float component : mesh.emissionColor) appendScalar(component);
    appendScalar(mesh.emissionIntensity);
    appendScalar(mesh.opacity);
    appendScalar(mesh.opacityThreshold);
    appendScalar(mesh.opacityFromBaseAlpha);
    const auto appendTextureMap = [&appendScalar, &appendVector](
                                      const BridgeSceneTextureMap& map) {
        appendScalar(map.width);
        appendScalar(map.height);
        appendScalar(map.channel);
        if (map.rgba8) appendVector(*map.rgba8);
    };
    appendTextureMap(mesh.roughnessTexture);
    appendTextureMap(mesh.metallicTexture);
    appendTextureMap(mesh.emissionTexture);
    appendTextureMap(mesh.normalTexture);

    for (const auto& cached : gState.sceneMetalMeshes) {
        if (cached.device == device && cached.contentHash == contentHash) {
            return cached.accelerationStructureResourceID;
        }
    }
    if (gState.bridge == nullptr || mesh.vertices.empty() || mesh.indices.empty()
        || mesh.vertices.size() % 3 != 0 || mesh.indices.size() % 3 != 0
        || mesh.indices.size() / 3 != mesh.triangleCount) {
        return 0;
    }

    const bool hasAuthoredCornerNormals =
        mesh.cornerNormals.size() == mesh.indices.size() * 3;
    const bool hasFileTexture =
        mesh.cornerUVs.size() == mesh.indices.size() * 2
        && mesh.textureWidth > 0 && mesh.textureHeight > 0
        && mesh.textureRGBA8
        && mesh.textureRGBA8->size()
            == static_cast<std::size_t>(mesh.textureWidth)
                * mesh.textureHeight * 4;
    const auto hasTextureMap = [](const BridgeSceneTextureMap& map) {
        return map.width > 0 && map.height > 0
            && map.rgba8
            && map.rgba8->size()
                == static_cast<std::size_t>(map.width) * map.height * 4;
    };
    const bool hasRoughnessTexture = hasTextureMap(mesh.roughnessTexture);
    const bool hasMetallicTexture = hasTextureMap(mesh.metallicTexture);
    const bool hasEmissionTexture = hasTextureMap(mesh.emissionTexture);
    const bool hasNormalTexture = hasTextureMap(mesh.normalTexture);
    const bool hasAlphaCutout = (mesh.materialFlags & 64u) != 0;
    const bool hasStandardOpacity = (mesh.materialFlags & 128u) != 0;
    const bool hasParameterTextures = hasRoughnessTexture
        || hasMetallicTexture || hasEmissionTexture;
    const bool hasMaterialDescriptorTextures = hasParameterTextures
        || hasNormalTexture || hasAlphaCutout || hasStandardOpacity;
    const bool hasMaterialTextureUVs =
        mesh.cornerUVs.size() == mesh.indices.size() * 2
        && (hasFileTexture || hasMaterialDescriptorTextures);
    const bool hasMaterialParameters = (mesh.materialFlags & 16u) != 0;
    const bool hasEmission = (mesh.materialFlags & 32u) != 0;
    const bool usesEmissionVertexFormat = hasEmission
        || hasMaterialDescriptorTextures;
    const bool hasMaterialAttributes = hasMaterialParameters
        || hasEmission || hasMaterialDescriptorTextures;
    const bool hasInterleavedAttributes =
        hasAuthoredCornerNormals || hasMaterialTextureUVs
        || hasMaterialAttributes;
    std::vector<float> interleavedVertices;
    const float* vertexData = mesh.vertices.data();
    std::size_t vertexValueCount = mesh.vertices.size();
    if (hasInterleavedAttributes) {
        try {
            const std::size_t valuesPerCorner = hasNormalTexture
                ? 20 : (usesEmissionVertexFormat
                ? 14 : (hasMaterialParameters
                    ? 10 : (hasMaterialTextureUVs ? 8 : 6)));
            interleavedVertices.reserve(mesh.indices.size() * valuesPerCorner);
            for (std::size_t corner = 0; corner < mesh.indices.size(); ++corner) {
                const std::size_t pointOffset =
                    static_cast<std::size_t>(mesh.indices[corner]) * 3;
                const std::size_t normalOffset = corner * 3;
                interleavedVertices.insert(
                    interleavedVertices.end(),
                    mesh.vertices.begin() + pointOffset,
                    mesh.vertices.begin() + pointOffset + 3
                );
                std::array<float, 3> shadingNormal{};
                if (hasAuthoredCornerNormals) {
                    std::copy_n(
                        mesh.cornerNormals.begin() + normalOffset,
                        shadingNormal.size(),
                        shadingNormal.begin()
                    );
                } else {
                    const std::size_t triangleCorner = (corner / 3) * 3;
                    const auto point = [&mesh, triangleCorner](std::size_t index) {
                        const std::size_t offset = static_cast<std::size_t>(
                            mesh.indices[triangleCorner + index]
                        ) * 3;
                        return std::array<float, 3>{
                            mesh.vertices[offset], mesh.vertices[offset + 1],
                            mesh.vertices[offset + 2],
                        };
                    };
                    const auto a = point(0);
                    const auto b = point(1);
                    const auto c = point(2);
                    const std::array<float, 3> edgeA{
                        b[0] - a[0], b[1] - a[1], b[2] - a[2],
                    };
                    const std::array<float, 3> edgeB{
                        c[0] - a[0], c[1] - a[1], c[2] - a[2],
                    };
                    shadingNormal = {
                        edgeA[1] * edgeB[2] - edgeA[2] * edgeB[1],
                        edgeA[2] * edgeB[0] - edgeA[0] * edgeB[2],
                        edgeA[0] * edgeB[1] - edgeA[1] * edgeB[0],
                    };
                    const float length = std::sqrt(
                        shadingNormal[0] * shadingNormal[0]
                            + shadingNormal[1] * shadingNormal[1]
                            + shadingNormal[2] * shadingNormal[2]
                    );
                    if (!std::isfinite(length) || length <= 0.000001F) {
                        // Degenerate triangles never produce a useful hit, but
                        // abandoning the whole USD Mesh makes TLAS membership
                        // depend on a later native-BLAS bounds match. Keep the
                        // remaining valid triangles deterministic instead.
                        shadingNormal = {0.0F, 1.0F, 0.0F};
                    } else {
                        for (float& component : shadingNormal) component /= length;
                    }
                }
                interleavedVertices.insert(
                    interleavedVertices.end(),
                    shadingNormal.begin(),
                    shadingNormal.end()
                );
                if (hasMaterialTextureUVs) {
                    const std::size_t uvOffset = corner * 2;
                    interleavedVertices.insert(
                        interleavedVertices.end(),
                        mesh.cornerUVs.begin() + uvOffset,
                        mesh.cornerUVs.begin() + uvOffset + 2
                    );
                } else if (hasMaterialAttributes) {
                    interleavedVertices.push_back(0.0f);
                    interleavedVertices.push_back(0.0f);
                }
                if (hasMaterialAttributes) {
                    interleavedVertices.push_back(mesh.roughness);
                    interleavedVertices.push_back(mesh.metallic);
                }
                if (usesEmissionVertexFormat) {
                    interleavedVertices.insert(
                        interleavedVertices.end(),
                        mesh.emissionColor.begin(), mesh.emissionColor.end()
                    );
                    interleavedVertices.push_back(mesh.emissionIntensity);
                }
                if (hasNormalTexture) {
                    const std::size_t triangleCorner = (corner / 3) * 3;
                    const auto trianglePoint = [&mesh, triangleCorner](
                        std::size_t index
                    ) {
                        const std::size_t offset = static_cast<std::size_t>(
                            mesh.indices[triangleCorner + index]
                        ) * 3;
                        return std::array<float, 3>{
                            mesh.vertices[offset], mesh.vertices[offset + 1],
                            mesh.vertices[offset + 2],
                        };
                    };
                    const auto triangleUV = [&mesh, triangleCorner](
                        std::size_t index
                    ) {
                        const std::size_t offset =
                            (triangleCorner + index) * 2;
                        return std::array<float, 2>{
                            mesh.cornerUVs[offset], mesh.cornerUVs[offset + 1],
                        };
                    };
                    const auto pointA = trianglePoint(0);
                    const auto pointB = trianglePoint(1);
                    const auto pointC = trianglePoint(2);
                    const auto uvA = triangleUV(0);
                    const auto uvB = triangleUV(1);
                    const auto uvC = triangleUV(2);
                    const std::array<float, 3> edgeA{
                        pointB[0] - pointA[0],
                        pointB[1] - pointA[1],
                        pointB[2] - pointA[2],
                    };
                    const std::array<float, 3> edgeB{
                        pointC[0] - pointA[0],
                        pointC[1] - pointA[1],
                        pointC[2] - pointA[2],
                    };
                    const float deltaU1 = uvB[0] - uvA[0];
                    const float deltaV1 = uvB[1] - uvA[1];
                    const float deltaU2 = uvC[0] - uvA[0];
                    const float deltaV2 = uvC[1] - uvA[1];
                    const float determinant =
                        deltaU1 * deltaV2 - deltaV1 * deltaU2;
                    std::array<float, 3> tangent{};
                    std::array<float, 3> bitangent{};
                    float tangentLengthSquared = 0.0F;
                    float bitangentLengthSquared = 0.0F;
                    if (std::isfinite(determinant)
                        && std::abs(determinant) > 0.000000000001F) {
                        const float inverseDeterminant = 1.0F / determinant;
                        for (std::size_t component = 0; component < 3; ++component) {
                            tangent[component] = (
                                edgeA[component] * deltaV2
                                - edgeB[component] * deltaV1
                            ) * inverseDeterminant;
                            bitangent[component] = (
                                edgeB[component] * deltaU1
                                - edgeA[component] * deltaU2
                            ) * inverseDeterminant;
                            tangentLengthSquared +=
                                tangent[component] * tangent[component];
                            bitangentLengthSquared +=
                                bitangent[component] * bitangent[component];
                        }
                    }
                    const bool tangentBasisValid =
                        std::isfinite(tangentLengthSquared)
                        && std::isfinite(bitangentLengthSquared)
                        && tangentLengthSquared > 0.000000000001F
                        && bitangentLengthSquared > 0.000000000001F;
                    if (tangentBasisValid) {
                        const float inverseTangentLength =
                            1.0F / std::sqrt(tangentLengthSquared);
                        const float inverseBitangentLength =
                            1.0F / std::sqrt(bitangentLengthSquared);
                        for (std::size_t component = 0; component < 3; ++component) {
                            tangent[component] *= inverseTangentLength;
                            bitangent[component] *= inverseBitangentLength;
                        }
                    } else {
                        // Repeated UVs and zero-area UV islands are legal in
                        // production assets. Build a stable orthonormal basis
                        // from the shading normal for those corners rather than
                        // dropping the complete Mesh and relying on a
                        // nondeterministic native-BLAS bounds match.
                        std::array<float, 3> normal = shadingNormal;
                        const float normalLengthSquared =
                            normal[0] * normal[0] + normal[1] * normal[1]
                            + normal[2] * normal[2];
                        if (!std::isfinite(normalLengthSquared)
                            || normalLengthSquared <= 0.000000000001F) {
                            normal = {0.0F, 1.0F, 0.0F};
                        } else {
                            const float inverseNormalLength =
                                1.0F / std::sqrt(normalLengthSquared);
                            for (float& component : normal) {
                                component *= inverseNormalLength;
                            }
                        }
                        const std::array<float, 3> reference =
                            std::abs(normal[1]) < 0.999F
                            ? std::array<float, 3>{0.0F, 1.0F, 0.0F}
                            : std::array<float, 3>{1.0F, 0.0F, 0.0F};
                        tangent = {
                            reference[1] * normal[2] - reference[2] * normal[1],
                            reference[2] * normal[0] - reference[0] * normal[2],
                            reference[0] * normal[1] - reference[1] * normal[0],
                        };
                        tangentLengthSquared =
                            tangent[0] * tangent[0] + tangent[1] * tangent[1]
                            + tangent[2] * tangent[2];
                        if (!std::isfinite(tangentLengthSquared)
                            || tangentLengthSquared <= 0.000000000001F) {
                            return 0;
                        }
                        const float inverseTangentLength =
                            1.0F / std::sqrt(tangentLengthSquared);
                        for (float& component : tangent) {
                            component *= inverseTangentLength;
                        }
                        bitangent = {
                            normal[1] * tangent[2] - normal[2] * tangent[1],
                            normal[2] * tangent[0] - normal[0] * tangent[2],
                            normal[0] * tangent[1] - normal[1] * tangent[0],
                        };
                    }
                    interleavedVertices.insert(
                        interleavedVertices.end(), tangent.begin(), tangent.end()
                    );
                    interleavedVertices.insert(
                        interleavedVertices.end(),
                        bitangent.begin(), bitangent.end()
                    );
                }
            }
        } catch (const std::bad_alloc&) {
            return 0;
        }
        vertexData = interleavedVertices.data();
        vertexValueCount = interleavedVertices.size();
    }
    const std::uint64_t vertexBytes = static_cast<std::uint64_t>(
        vertexValueCount * sizeof(float)
    );
    const std::uint64_t indexBytes = hasInterleavedAttributes
        ? 0
        : static_cast<std::uint64_t>(
            mesh.indices.size() * sizeof(std::uint32_t)
        );
    BridgeSceneMetalMesh resources{};
    resources.device = device;
    resources.contentHash = contentHash;
    resources.pathHash = mesh.pathHash;
    const auto destroyCreatedResources = [&resources]() {
        if (gState.bridge == nullptr) return;
        const std::array<std::uint64_t, 4> resourceIDs{
            resources.accelerationStructureResourceID,
            resources.materialDescriptorResourceID,
            resources.indexBufferResourceID,
            resources.vertexBufferResourceID,
        };
        for (const auto resourceID : resourceIDs) {
            if (resourceID == 0) continue;
            try {
                gState.bridge->destroyBuffer(resourceID);
            } catch (const std::exception&) {
            }
        }
        releaseSceneTextureResourceLocked(resources.normalTextureResourceID);
        releaseSceneTextureResourceLocked(resources.emissionTextureResourceID);
        releaseSceneTextureResourceLocked(resources.metallicTextureResourceID);
        releaseSceneTextureResourceLocked(resources.roughnessTextureResourceID);
        releaseSceneTextureResourceLocked(resources.textureResourceID);
    };
    try {
        resources.vertexBufferResourceID = gState.bridge->createBuffer(vertexBytes);
        if (indexBytes != 0) {
            resources.indexBufferResourceID = gState.bridge->createBuffer(indexBytes);
        }
        if (hasFileTexture) {
            BridgeSceneTextureMap texture{};
            texture.width = mesh.textureWidth;
            texture.height = mesh.textureHeight;
            texture.rgba8 = mesh.textureRGBA8;
            resources.textureResourceID = acquireSceneTextureResourceLocked(
                device, texture
            );
        }
        const auto uploadParameterTexture = [device](
            const BridgeSceneTextureMap& map,
            std::uint64_t& resourceID
        ) {
            resourceID = acquireSceneTextureResourceLocked(device, map);
        };
        if (hasRoughnessTexture) {
            uploadParameterTexture(
                mesh.roughnessTexture,
                resources.roughnessTextureResourceID
            );
        }
        if (hasMetallicTexture) {
            uploadParameterTexture(
                mesh.metallicTexture,
                resources.metallicTextureResourceID
            );
        }
        if (hasEmissionTexture) {
            uploadParameterTexture(
                mesh.emissionTexture,
                resources.emissionTextureResourceID
            );
        }
        if (hasNormalTexture) {
            uploadParameterTexture(
                mesh.normalTexture,
                resources.normalTextureResourceID
            );
        }
        if (hasMaterialDescriptorTextures) {
            constexpr std::uint32_t kMaterialDescriptorMagic =
                UINT32_C(0x314d424d);
            const std::uint32_t descriptorVersion = hasStandardOpacity
                ? 3u : ((hasNormalTexture || hasAlphaCutout) ? 2u : 1u);
            const bool usesExtendedDescriptor = descriptorVersion >= 2u;
            std::vector<std::uint8_t> descriptor;
            descriptor.reserve(descriptorVersion == 3u
                ? 64 : (usesExtendedDescriptor ? 56 : 48));
            std::uint32_t descriptorFlags = 0;
            if (resources.textureResourceID != 0) descriptorFlags |= 1u;
            if (resources.roughnessTextureResourceID != 0) descriptorFlags |= 2u;
            if (resources.metallicTextureResourceID != 0) descriptorFlags |= 4u;
            if (resources.emissionTextureResourceID != 0) descriptorFlags |= 8u;
            if (resources.normalTextureResourceID != 0) descriptorFlags |= 16u;
            if ((mesh.materialFlags & 64u) != 0) descriptorFlags |= 32u;
            if (hasStandardOpacity) descriptorFlags |= 64u;
            if (mesh.opacityFromBaseAlpha) descriptorFlags |= 128u;
            const std::uint32_t packedChannels =
                mesh.roughnessTexture.channel
                | (mesh.metallicTexture.channel << 4)
                | (mesh.emissionTexture.channel << 8);
            imb::appendLittleEndian(descriptor, kMaterialDescriptorMagic);
            imb::appendLittleEndian(
                descriptor,
                descriptorVersion
            );
            imb::appendLittleEndian(descriptor, descriptorFlags);
            imb::appendLittleEndian(descriptor, packedChannels);
            const auto appendResourceID = [&descriptor](std::uint64_t resourceID) {
                imb::appendLittleEndian(
                    descriptor, static_cast<std::uint32_t>(resourceID)
                );
                imb::appendLittleEndian(
                    descriptor, static_cast<std::uint32_t>(resourceID >> 32)
                );
            };
            appendResourceID(resources.textureResourceID);
            appendResourceID(resources.roughnessTextureResourceID);
            appendResourceID(resources.metallicTextureResourceID);
            appendResourceID(resources.emissionTextureResourceID);
            if (usesExtendedDescriptor) {
                appendResourceID(resources.normalTextureResourceID);
            }
            if (descriptorVersion == 3u) {
                imb::appendLittleEndian(
                    descriptor,
                    std::bit_cast<std::uint32_t>(mesh.opacity)
                );
                imb::appendLittleEndian(
                    descriptor,
                    std::bit_cast<std::uint32_t>(mesh.opacityThreshold)
                );
            }
            resources.materialDescriptorResourceID =
                gState.bridge->createBuffer(descriptor.size());
            gState.bridge->writeBuffer(
                resources.materialDescriptorResourceID,
                descriptor.data(),
                descriptor.size()
            );
        }
        gState.bridge->writeBuffer(
            resources.vertexBufferResourceID,
            reinterpret_cast<const std::uint8_t*>(vertexData),
            vertexBytes
        );
        if (indexBytes != 0) {
            gState.bridge->writeBuffer(
                resources.indexBufferResourceID,
                reinterpret_cast<const std::uint8_t*>(mesh.indices.data()),
                indexBytes
            );
        }
        resources.accelerationStructureResourceID =
            gState.bridge->createAccelerationStructure(
                static_cast<std::uint32_t>(
                    VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR
                ),
                std::max<std::uint64_t>(vertexBytes + indexBytes, 1)
            );
        BridgePrimitiveAccelerationStructureGeometry geometry{};
        geometry.kind = 0;
        geometry.flags = 0;
        geometry.dataResourceID = resources.vertexBufferResourceID;
        geometry.dataOffset = 0;
        geometry.primitiveCount = mesh.triangleCount;
        geometry.stride = hasNormalTexture
            ? 20 * sizeof(float)
            : (usesEmissionVertexFormat
            ? 14 * sizeof(float)
            : (hasMaterialParameters
                ? 10 * sizeof(float)
                : (hasMaterialTextureUVs
                    ? 8 * sizeof(float)
                    : (hasAuthoredCornerNormals
                        ? 6 * sizeof(float) : 3 * sizeof(float)))));
        geometry.indexResourceID = resources.indexBufferResourceID;
        geometry.indexOffset = 0;
        geometry.indexType = hasInterleavedAttributes ? 0 : 2;
        geometry.vertexFormat = hasNormalTexture
            ? 6 : (usesEmissionVertexFormat
            ? 5 : (hasMaterialParameters
                ? 4 : (hasMaterialTextureUVs
                    ? 3 : (hasAuthoredCornerNormals ? 2 : 1))));
        gState.bridge->buildPrimitiveAccelerationStructure(
            resources.accelerationStructureResourceID,
            0,
            {geometry}
        );
        gState.sceneMetalMeshes.push_back(resources);
        return resources.accelerationStructureResourceID;
    } catch (...) {
        destroyCreatedResources();
        throw;
    }
}

VkResult bridgeFallbackInstanceAccelerationStructureBuildLocked(
    VkDevice device,
    AccelerationStructureKHRState** topLevelOutput
) {
    if (topLevelOutput == nullptr || gState.bridge == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *topLevelOutput = nullptr;
    AccelerationStructureKHRState* topLevel = nullptr;
    std::vector<AccelerationStructureKHRState*> children;
    for (auto* acceleration : gState.accelerationStructuresKHR) {
        if (acceleration == nullptr || acceleration->device != device
            || acceleration->bridgeResourceID == 0) {
            continue;
        }
        if (acceleration->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) {
            if (acceleration->built && !acceleration->builtFromSceneState) {
                *topLevelOutput = acceleration;
                return VK_SUCCESS;
            }
            if (topLevel == nullptr || acceleration->builtFromSceneState) {
                topLevel = acceleration;
            }
        } else if (acceleration->built) {
            children.push_back(acceleration);
        }
    }
    if (topLevel == nullptr) return VK_ERROR_FEATURE_NOT_PRESENT;
    std::sort(
        children.begin(),
        children.end(),
        [](const auto* left, const auto* right) {
            return left->bridgeResourceID < right->bridgeResourceID;
        }
    );

    const auto liveSceneHeader = readLiveSceneState(false);
    if (!liveSceneHeader.has_value() || !liveSceneHeader->hasMeshManifest) {
        if (rayTracingTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: refusing fallback TLAS without a live USD Mesh manifest\n"
            );
        }
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (topLevel->built && topLevel->builtFromSceneState
        && topLevel->sceneStateMeshSequence == liveSceneHeader->meshSequence) {
        if (topLevel->sceneStateSequence != liveSceneHeader->camera.sequence) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: reused fallback Metal instance AS host=%llu sceneSequence=%llu meshSequence=%llu reason=camera-only-update\n",
                static_cast<unsigned long long>(topLevel->bridgeResourceID),
                static_cast<unsigned long long>(
                    liveSceneHeader->camera.sequence
                ),
                static_cast<unsigned long long>(liveSceneHeader->meshSequence)
            );
        }
        topLevel->sceneStateSequence = liveSceneHeader->camera.sequence;
        *topLevelOutput = topLevel;
        return VK_SUCCESS;
    }

    const auto liveScene = readLiveSceneState();
    if (!liveScene.has_value() || !liveScene->hasMeshManifest) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    if (topLevel->built && topLevel->builtFromSceneState
        && topLevel->sceneStateMeshSequence == liveScene->meshSequence) {
        topLevel->sceneStateSequence = liveScene->camera.sequence;
        *topLevelOutput = topLevel;
        return VK_SUCCESS;
    }

    const std::array<float, 12> identity{
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
    };
    std::vector<BridgeInstanceAccelerationStructureInstance> instances;
    std::unordered_set<AccelerationStructureKHRState*> matchedChildren;
    std::unordered_set<std::uint64_t> activeSceneResources;
    std::size_t sceneGeometryMeshCount = 0;
    std::size_t reclaimedSceneResourceCount = 0;
    try {
        instances.reserve(liveScene->meshes.size());
        for (std::size_t meshIndex = 0;
             meshIndex < liveScene->meshes.size();
             ++meshIndex) {
            const auto& mesh = liveScene->meshes[meshIndex];
            const auto worldBounds = transformedSceneMeshBounds(mesh);
            AccelerationStructureKHRState* bestChild = nullptr;
            float bestLocalError = std::numeric_limits<float>::max();
            float bestWorldError = std::numeric_limits<float>::max();
            float bestError = std::numeric_limits<float>::max();
            std::uint64_t childResourceID = 0;
            bool usesSceneGeometry = false;
            if (!mesh.vertices.empty() && !mesh.indices.empty()) {
                childResourceID = bridgeSceneMeshAccelerationStructureLocked(
                    device,
                    mesh
                );
                usesSceneGeometry = childResourceID != 0;
                if (usesSceneGeometry) {
                    activeSceneResources.insert(childResourceID);
                }
            }
            if (!usesSceneGeometry) {
                for (auto* child : children) {
                    if (matchedChildren.contains(child)
                        || !child->hasTriangleBounds
                        || child->triangleCount != mesh.triangleCount) {
                        continue;
                    }
                    const float localError = sceneBoundsError(
                        child->triangleBoundsMinimum,
                        child->triangleBoundsMaximum,
                        mesh.boundsMinimum,
                        mesh.boundsMaximum
                    );
                    const float worldError = sceneBoundsError(
                        child->triangleBoundsMinimum,
                        child->triangleBoundsMaximum,
                        worldBounds.first,
                        worldBounds.second
                    );
                    const float error = std::min(localError, worldError);
                    if (error < bestError) {
                        bestChild = child;
                        bestLocalError = localError;
                        bestWorldError = worldError;
                        bestError = error;
                    }
                }
                constexpr float kMaximumBoundsMatchError = 0.2F;
                if (bestChild == nullptr || bestError > kMaximumBoundsMatchError) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: USD Mesh has no matching real BLAS pathHash=%#llx triangles=%u bestBoundsError=%.6g\n",
                        static_cast<unsigned long long>(mesh.pathHash),
                        mesh.triangleCount,
                        bestError
                    );
                    continue;
                }
                matchedChildren.insert(bestChild);
                childResourceID = bestChild->bridgeResourceID;
            }
            BridgeInstanceAccelerationStructureInstance instance{};
            const bool verticesAlreadyInWorld =
                !usesSceneGeometry
                && bestWorldError + 0.0001F < bestLocalError;
            const auto& transform = verticesAlreadyInWorld
                ? identity : mesh.worldTransform;
            for (std::size_t component = 0; component < transform.size(); ++component) {
                std::memcpy(
                    &instance.transformationBits[component],
                    &transform[component],
                    sizeof(std::uint32_t)
                );
            }
            instance.mask = 0xff;
            std::uint64_t textureResourceID = 0;
            std::uint64_t materialDescriptorResourceID = 0;
            const auto materialResource = std::find_if(
                gState.sceneMetalMeshes.begin(),
                gState.sceneMetalMeshes.end(),
                [device, childResourceID](const auto& candidate) {
                    return candidate.device == device
                        && candidate.accelerationStructureResourceID
                            == childResourceID;
                }
            );
            if (materialResource != gState.sceneMetalMeshes.end()) {
                textureResourceID = materialResource->textureResourceID;
                materialDescriptorResourceID =
                    materialResource->materialDescriptorResourceID;
            }
            if (materialDescriptorResourceID != 0
                && materialDescriptorResourceID <= UINT32_C(0x003fffff)) {
                instance.userID = UINT32_C(0x00c00000)
                    | static_cast<std::uint32_t>(
                        materialDescriptorResourceID
                    );
            } else if (textureResourceID != 0
                && textureResourceID <= UINT32_C(0x003fffff)) {
                instance.userID = UINT32_C(0x00400000)
                    | static_cast<std::uint32_t>(textureResourceID);
            } else if ((mesh.materialFlags & 2u) != 0) {
                const std::uint32_t packedRGB = mesh.materialFlags >> 8;
                const std::uint32_t red = packedRGB & 0xffu;
                const std::uint32_t green = (packedRGB >> 8) & 0xffu;
                // Metal exposes a 24-bit per-instance user ID.  The high two
                // bits select none/texture/base-color/material-descriptor, so
                // an inline base color has only six collision-free blue bits.
                // The previous seven-bit packing could set marker bit 22 for
                // blue >= 0.5, turning neutral white/gray materials into an
                // invalid descriptor and therefore the shader's blue fallback.
                const std::uint32_t blue6 = (packedRGB >> 18) & 0x3fu;
                instance.userID = UINT32_C(0x00800000)
                    | red | (green << 8) | (blue6 << 16);
            }
            instance.accelerationStructureResourceID = childResourceID;
            instances.push_back(instance);
            if (usesSceneGeometry) {
                ++sceneGeometryMeshCount;
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: built USD Mesh BLAS pathHash=%#llx vertices=%zu triangles=%u host=%llu instanceTransform=usd-world source=scene-state normals=%s material=%s texture=%ux%u roughness=%.3f metallic=%.3f emission=(%.3f,%.3f,%.3f)x%.3f parameterTextures=roughness:%ux%u[c%u],metallic:%ux%u[c%u],emission:%ux%u[rgb],normal:%ux%u[rgb-tangent] alphaCutout=%d opacity=%.3f threshold=%.3f opacityTexture=%d materialDescriptor=%llu\n",
                    static_cast<unsigned long long>(mesh.pathHash),
                    mesh.vertices.size() / 3,
                    mesh.triangleCount,
                    static_cast<unsigned long long>(childResourceID),
                    mesh.cornerNormals.empty()
                        ? "geometric-face" : "authored-corner",
                    textureResourceID != 0
                        ? "file-texture"
                        : ((mesh.materialFlags & 2u) != 0
                        ? ((mesh.materialFlags & 4u) != 0
                            ? "connected-fallback-color"
                            : ((mesh.materialFlags & 1u) != 0
                                ? "bound-base-color" : "display-color"))
                        : ((mesh.materialFlags & 1u) != 0 ? "bound" : "none")),
                    mesh.textureWidth,
                    mesh.textureHeight,
                    mesh.roughness,
                    mesh.metallic,
                    mesh.emissionColor[0],
                    mesh.emissionColor[1],
                    mesh.emissionColor[2],
                    mesh.emissionIntensity,
                    mesh.roughnessTexture.width,
                    mesh.roughnessTexture.height,
                    mesh.roughnessTexture.channel,
                    mesh.metallicTexture.width,
                    mesh.metallicTexture.height,
                    mesh.metallicTexture.channel,
                    mesh.emissionTexture.width,
                    mesh.emissionTexture.height,
                    mesh.normalTexture.width,
                    mesh.normalTexture.height,
                    (mesh.materialFlags & 64u) != 0 ? 1 : 0,
                    mesh.opacity,
                    mesh.opacityThreshold,
                    mesh.opacityFromBaseAlpha ? 1 : 0,
                    static_cast<unsigned long long>(
                        materialDescriptorResourceID
                    )
                );
            } else {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: matched USD Mesh pathHash=%#llx triangles=%u host=%llu localBoundsError=%.6g worldBoundsError=%.6g instanceTransform=%s material=%s\n",
                    static_cast<unsigned long long>(mesh.pathHash),
                    mesh.triangleCount,
                    static_cast<unsigned long long>(bestChild->bridgeResourceID),
                    bestLocalError,
                    bestWorldError,
                    verticesAlreadyInWorld ? "identity" : "usd-world",
                    (mesh.materialFlags & 2u) != 0
                        ? ((mesh.materialFlags & 4u) != 0
                            ? "connected-fallback-color"
                            : ((mesh.materialFlags & 1u) != 0
                                ? "bound-base-color" : "display-color"))
                        : ((mesh.materialFlags & 1u) != 0 ? "bound" : "none")
                );
            }
        }
        if (instances.empty()) return VK_ERROR_FEATURE_NOT_PRESENT;
        gState.bridge->buildInstanceAccelerationStructure(
            topLevel->bridgeResourceID,
            0,
            instances
        );
        for (auto iterator = gState.sceneMetalMeshes.begin();
             iterator != gState.sceneMetalMeshes.end();) {
            if (iterator->device != device
                || activeSceneResources.contains(
                    iterator->accelerationStructureResourceID
                )) {
                ++iterator;
                continue;
            }
            const std::array<std::uint64_t, 4> resourceIDs{
                iterator->accelerationStructureResourceID,
                iterator->materialDescriptorResourceID,
                iterator->indexBufferResourceID,
                iterator->vertexBufferResourceID,
            };
            for (const auto resourceID : resourceIDs) {
                if (resourceID == 0) continue;
                try {
                    gState.bridge->destroyBuffer(resourceID);
                } catch (const std::exception& error) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: stale animated USD Mesh resource destroy warning: %s\n",
                        error.what()
                    );
                }
            }
            releaseSceneTextureResourceLocked(iterator->normalTextureResourceID);
            releaseSceneTextureResourceLocked(iterator->emissionTextureResourceID);
            releaseSceneTextureResourceLocked(iterator->metallicTextureResourceID);
            releaseSceneTextureResourceLocked(iterator->roughnessTextureResourceID);
            releaseSceneTextureResourceLocked(iterator->textureResourceID);
            iterator = gState.sceneMetalMeshes.erase(iterator);
            ++reclaimedSceneResourceCount;
        }
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: fallback Metal instance AS build failed: %s\n",
            error.what()
        );
        return VK_ERROR_DEVICE_LOST;
    }
    topLevel->built = true;
    topLevel->builtFromSceneState = true;
    topLevel->sceneStateSequence = liveScene->camera.sequence;
    topLevel->sceneStateMeshSequence = liveScene->meshSequence;
    *topLevelOutput = topLevel;
    std::fprintf(
        stderr,
        "imb-vulkan-icd: fallback Metal instance AS built host=%llu matchedUSDMeshes=%zu sceneGeometryMeshes=%zu excludedInternalBLAS=%zu sceneSequence=%llu meshSequence=%llu reclaimedSceneResources=%zu reason=Kit-null-TLAS\n",
        static_cast<unsigned long long>(topLevel->bridgeResourceID),
        instances.size(),
        sceneGeometryMeshCount,
        children.size() - matchedChildren.size(),
        static_cast<unsigned long long>(liveScene->camera.sequence),
        static_cast<unsigned long long>(liveScene->meshSequence),
        reclaimedSceneResourceCount
    );
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateDeferredOperationKHR(
    VkDevice device,
    const VkAllocationCallbacks*,
    VkDeferredOperationKHR* deferredOperation
) {
    if (deferredOperation == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_DEVICE_LOST;
    auto* state = new (std::nothrow) DeferredOperationKHRState{device, VK_SUCCESS};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    gState.deferredOperationsKHR.insert(state);
    *deferredOperation = makeObjectHandle<VkDeferredOperationKHR>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyDeferredOperationKHR(
    VkDevice,
    VkDeferredOperationKHR deferredOperation,
    const VkAllocationCallbacks*
) {
    if (deferredOperation == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<DeferredOperationKHRState>(deferredOperation);
    if (gState.deferredOperationsKHR.erase(state) != 0) delete state;
}

VKAPI_ATTR std::uint32_t VKAPI_CALL imb_vkGetDeferredOperationMaxConcurrencyKHR(
    VkDevice device,
    VkDeferredOperationKHR deferredOperation
) {
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<DeferredOperationKHRState>(deferredOperation);
    return validDevice(device) && gState.deferredOperationsKHR.contains(state) && state->device == device ? 1U : 0U;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetDeferredOperationResultKHR(
    VkDevice device,
    VkDeferredOperationKHR deferredOperation
) {
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<DeferredOperationKHRState>(deferredOperation);
    if (!validDevice(device) || !gState.deferredOperationsKHR.contains(state) || state->device != device) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    return state->result;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkDeferredOperationJoinKHR(
    VkDevice device,
    VkDeferredOperationKHR deferredOperation
) {
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<DeferredOperationKHRState>(deferredOperation);
    if (!validDevice(device) || !gState.deferredOperationsKHR.contains(state) || state->device != device) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateAccelerationStructureKHR(
    VkDevice device,
    const VkAccelerationStructureCreateInfoKHR* createInfo,
    const VkAllocationCallbacks*,
    VkAccelerationStructureKHR* accelerationStructure
) {
    if (createInfo == nullptr || accelerationStructure == nullptr
        || createInfo->sType != VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR
        || createInfo->buffer == VK_NULL_HANDLE || createInfo->size == 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    auto* buffer = objectState<BufferState>(createInfo->buffer);
    if (!validDevice(device) || !gState.buffers.contains(buffer) || buffer->device != device
        || createInfo->offset > buffer->size || createInfo->size > buffer->size - createInfo->offset) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    auto* state = new (std::nothrow) AccelerationStructureKHRState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    state->buffer = buffer;
    state->offset = createInfo->offset;
    state->size = createInfo->size;
    state->type = createInfo->type;
    state->deviceAddress = createInfo->deviceAddress != 0
        ? createInfo->deviceAddress
        : buffer->deviceAddress + createInfo->offset;
    try {
        state->bridgeResourceID = gState.bridge->createAccelerationStructure(
            static_cast<std::uint32_t>(state->type),
            state->size
        );
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: Metal AS creation failed: %s\n", error.what());
        delete state;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    gState.accelerationStructuresKHR.insert(state);
    *accelerationStructure = makeObjectHandle<VkAccelerationStructureKHR>(state);
    if (traceEnabled() || rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: create AS KHR handle=%p buffer=%p offset=%llu size=%llu type=%d address=%#llx\n",
            static_cast<void*>(state),
            static_cast<void*>(buffer),
            static_cast<unsigned long long>(state->offset),
            static_cast<unsigned long long>(state->size),
            state->type,
            static_cast<unsigned long long>(state->deviceAddress)
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyAccelerationStructureKHR(
    VkDevice,
    VkAccelerationStructureKHR accelerationStructure,
    const VkAllocationCallbacks*
) {
    if (accelerationStructure == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<AccelerationStructureKHRState>(accelerationStructure);
    if (gState.accelerationStructuresKHR.erase(state) != 0) {
        if (state->bridgeResourceID != 0 && gState.bridge != nullptr) {
            try {
                gState.bridge->destroyBuffer(state->bridgeResourceID);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "imb-vulkan-icd: Metal AS destroy warning: %s\n", error.what());
            }
        }
        delete state;
    }
}

VKAPI_ATTR VkDeviceAddress VKAPI_CALL imb_vkGetAccelerationStructureDeviceAddressKHR(
    VkDevice device,
    const VkAccelerationStructureDeviceAddressInfoKHR* info
) {
    if (info == nullptr || info->accelerationStructure == VK_NULL_HANDLE) return 0;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<AccelerationStructureKHRState>(info->accelerationStructure);
    if (!validDevice(device) || !gState.accelerationStructuresKHR.contains(state) || state->device != device) return 0;
    return state->deviceAddress;
}

VkDeviceSize accelerationStructurePrimitiveBytes(
    const VkAccelerationStructureBuildGeometryInfoKHR& buildInfo,
    const std::uint32_t* primitiveCounts
) {
    VkDeviceSize bytes = 4096;
    for (std::uint32_t index = 0; index < buildInfo.geometryCount; ++index) {
        const VkDeviceSize primitiveCount = primitiveCounts == nullptr ? 1 : primitiveCounts[index];
        VkDeviceSize bytesPerPrimitive = 128;
        const VkAccelerationStructureGeometryKHR* geometry = buildInfo.pGeometries != nullptr
            ? &buildInfo.pGeometries[index]
            : (buildInfo.ppGeometries == nullptr ? nullptr : buildInfo.ppGeometries[index]);
        if (geometry != nullptr && geometry->geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
            bytesPerPrimitive = 64;
        } else if (geometry != nullptr && geometry->geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
            bytesPerPrimitive = sizeof(VkAccelerationStructureInstanceKHR) + 64;
        }
        bytes = saturatingAccelerationStructureAdd(
            bytes,
            saturatingAccelerationStructureMultiply(primitiveCount, bytesPerPrimitive)
        );
    }
    return alignedAccelerationStructureSize(bytes);
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetAccelerationStructureBuildSizesKHR(
    VkDevice device,
    VkAccelerationStructureBuildTypeKHR,
    const VkAccelerationStructureBuildGeometryInfoKHR* buildInfo,
    const std::uint32_t* maxPrimitiveCounts,
    VkAccelerationStructureBuildSizesInfoKHR* sizeInfo
) {
    if (buildInfo == nullptr || sizeInfo == nullptr) return;
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return;
    const VkDeviceSize size = accelerationStructurePrimitiveBytes(*buildInfo, maxPrimitiveCounts);
    sizeInfo->accelerationStructureSize = size;
    sizeInfo->buildScratchSize = alignedAccelerationStructureSize(
        saturatingAccelerationStructureAdd(4096, size / 2)
    );
    sizeInfo->updateScratchSize = alignedAccelerationStructureSize(
        saturatingAccelerationStructureAdd(4096, size / 4)
    );
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: AS KHR build sizes geometries=%u size=%llu scratch=%llu/%llu\n",
            buildInfo->geometryCount,
            static_cast<unsigned long long>(sizeInfo->accelerationStructureSize),
            static_cast<unsigned long long>(sizeInfo->buildScratchSize),
            static_cast<unsigned long long>(sizeInfo->updateScratchSize)
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkGetDeviceAccelerationStructureCompatibilityKHR(
    VkDevice device,
    const VkAccelerationStructureVersionInfoKHR* versionInfo,
    VkAccelerationStructureCompatibilityKHR* compatibility
) {
    if (compatibility == nullptr) return;
    std::lock_guard lock(gState.mutex);
    *compatibility = validDevice(device) && versionInfo != nullptr && versionInfo->pVersionData != nullptr
        ? VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR
        : VK_ACCELERATION_STRUCTURE_COMPATIBILITY_INCOMPATIBLE_KHR;
}

std::uint64_t mixRayTracingHash(std::uint64_t hash, std::uint64_t value) {
    for (unsigned int byte = 0; byte < sizeof(value); ++byte) {
        hash ^= static_cast<std::uint8_t>(value >> (byte * 8));
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

void writeRayTracingGroupHandle(
    const PipelineState& pipeline,
    std::uint32_t group,
    std::uint8_t* output
) {
    std::uint64_t value = mixRayTracingHash(pipeline.rayTracingHash, group);
    for (std::size_t byte = 0; byte < kShaderGroupHandleSize; ++byte) {
        if ((byte % sizeof(value)) == 0) value = mixRayTracingHash(value, byte / sizeof(value));
        output[byte] = static_cast<std::uint8_t>(value >> ((byte % sizeof(value)) * 8));
    }
}

std::int32_t rayTracingGroupForSBTAddressLocked(
    PipelineState* pipeline,
    VkDevice device,
    VkDeviceAddress address
) {
    if (pipeline == nullptr || address == 0) return -1;
    imb::Bytes actual;
    const VkResult copyResult = copyDeviceAddressBytesLocked(
        device,
        address,
        kShaderGroupHandleSize,
        actual
    );
    if (copyResult != VK_SUCCESS
        || actual.size() != kShaderGroupHandleSize) {
        if (rayTracingTraceEnabled()) {
            const auto* buffer = bufferForDeviceAddressLocked(device, address);
            std::fprintf(
                stderr,
                "imb-vulkan-icd: RT SBT unreadable address=%#llx result=%d buffer=%p base=%#llx size=%llu memory=%p memoryOffset=%llu\n",
                static_cast<unsigned long long>(address),
                copyResult,
                static_cast<const void*>(buffer),
                static_cast<unsigned long long>(buffer != nullptr ? buffer->deviceAddress : 0),
                static_cast<unsigned long long>(buffer != nullptr ? buffer->size : 0),
                static_cast<void*>(buffer != nullptr ? buffer->memory : nullptr),
                static_cast<unsigned long long>(buffer != nullptr ? buffer->memoryOffset : 0)
            );
        }
        return -1;
    }
    std::array<std::uint8_t, kShaderGroupHandleSize> expected{};
    for (std::uint32_t group = 0; group < pipeline->rayTracingGroups.size(); ++group) {
        writeRayTracingGroupHandle(*pipeline, group, expected.data());
        if (std::memcmp(actual.data(), expected.data(), expected.size()) == 0) {
            return static_cast<std::int32_t>(group);
        }
    }
    if (rayTracingTraceEnabled()) {
        writeRayTracingGroupHandle(*pipeline, 0, expected.data());
        std::fprintf(
            stderr,
            "imb-vulkan-icd: RT SBT handle mismatch address=%#llx actual=%02x%02x%02x%02x%02x%02x%02x%02x expected0=%02x%02x%02x%02x%02x%02x%02x%02x\n",
            static_cast<unsigned long long>(address),
            actual[0], actual[1], actual[2], actual[3],
            actual[4], actual[5], actual[6], actual[7],
            expected[0], expected[1], expected[2], expected[3],
            expected[4], expected[5], expected[6], expected[7]
        );
    }
    return -1;
}

std::int32_t inferRaygenGroupFromSparseRecordLocked(
    PipelineState* pipeline,
    VkDevice device,
    const VkStridedDeviceAddressRegionKHR& raygen
) {
    if (pipeline == nullptr || raygen.deviceAddress == 0 || raygen.stride == 0) return -1;
    auto* buffer = bufferForDeviceAddressLocked(device, raygen.deviceAddress);
    if (buffer == nullptr) return -1;
    const VkDeviceSize resourceOffset = raygen.deviceAddress - buffer->deviceAddress;
    const auto binding = std::find_if(
        buffer->sparseBindings.begin(),
        buffer->sparseBindings.end(),
        [resourceOffset](const SparseBufferBinding& candidate) {
            return resourceOffset >= candidate.resourceOffset
                && resourceOffset - candidate.resourceOffset < candidate.size;
        }
    );
    if (binding == buffer->sparseBindings.end()) return -1;
    const VkDeviceSize recordOffset = resourceOffset - binding->resourceOffset;
    if ((recordOffset % raygen.stride) != 0) return -1;
    const VkDeviceSize raygenOrdinal = recordOffset / raygen.stride;
    VkDeviceSize currentOrdinal = 0;
    for (std::size_t groupIndex = 0; groupIndex < pipeline->rayTracingGroups.size(); ++groupIndex) {
        const auto shaderIndex = pipeline->rayTracingGroups[groupIndex].generalShader;
        if (shaderIndex == VK_SHADER_UNUSED_KHR || shaderIndex >= pipeline->rayTracingStages.size()
            || pipeline->rayTracingStages[shaderIndex] != VK_SHADER_STAGE_RAYGEN_BIT_KHR) {
            continue;
        }
        if (currentOrdinal == raygenOrdinal) return static_cast<std::int32_t>(groupIndex);
        ++currentOrdinal;
    }
    return -1;
}

std::uint64_t rayTracingShaderHashForGroup(
    const PipelineState* pipeline,
    std::int32_t group
) {
    if (pipeline == nullptr || group < 0
        || static_cast<std::size_t>(group) >= pipeline->rayTracingGroups.size()) {
        return 0;
    }
    const auto shaderIndex = pipeline->rayTracingGroups[static_cast<std::size_t>(group)].generalShader;
    if (shaderIndex == VK_SHADER_UNUSED_KHR || shaderIndex >= pipeline->rayTracingShaders.size()
        || pipeline->rayTracingShaders[shaderIndex] == nullptr) {
        return 0;
    }
    return pipeline->rayTracingShaders[shaderIndex]->hash;
}

std::uint64_t observedIsaacRaygenHashForExtent(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t depth
) {
    if (depth != 1) return 0;
    if (width == 1280 && height == 720) return UINT64_C(0xf2dbfaa8274b5250);
    if (width == 2560 && height == 720) return UINT64_C(0x9eeed51da7b7135d);
    if (width == 160 && height == 90) return UINT64_C(0xa00a626692fa2fee);
    if (width == 256 && height == 144) return UINT64_C(0x72f6dc6c98605c7f);
    return 0;
}

std::int32_t uniqueRayTracingGroupForShaderHash(
    const PipelineState* pipeline,
    std::uint64_t shaderHash
) {
    if (pipeline == nullptr || shaderHash == 0) return -1;
    std::int32_t match = -1;
    for (std::size_t group = 0; group < pipeline->rayTracingGroups.size(); ++group) {
        if (rayTracingShaderHashForGroup(pipeline, static_cast<std::int32_t>(group))
            != shaderHash) {
            continue;
        }
        if (match >= 0) return -1;
        match = static_cast<std::int32_t>(group);
    }
    return match;
}

bool validRayTracingShaderStageNV(VkShaderStageFlagBits stage) {
    return stage == VK_SHADER_STAGE_RAYGEN_BIT_NV
        || stage == VK_SHADER_STAGE_ANY_HIT_BIT_NV
        || stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV
        || stage == VK_SHADER_STAGE_MISS_BIT_NV
        || stage == VK_SHADER_STAGE_INTERSECTION_BIT_NV
        || stage == VK_SHADER_STAGE_CALLABLE_BIT_NV;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateRayTracingPipelinesNV(
    VkDevice device,
    VkPipelineCache pipelineCache,
    std::uint32_t createInfoCount,
    const VkRayTracingPipelineCreateInfoNV* createInfos,
    const VkAllocationCallbacks*,
    VkPipeline* pipelines
) {
    if (createInfoCount == 0 || createInfos == nullptr || pipelines == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_DEVICE_LOST;
    if (pipelineCache != VK_NULL_HANDLE) {
        auto* cache = objectState<PipelineCacheState>(pipelineCache);
        if (!gState.pipelineCaches.contains(cache) || cache->device != device) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    for (std::uint32_t index = 0; index < createInfoCount; ++index) pipelines[index] = VK_NULL_HANDLE;

    std::vector<PipelineState*> created;
    created.reserve(createInfoCount);
    for (std::uint32_t index = 0; index < createInfoCount; ++index) {
        const auto& info = createInfos[index];
        auto* layout = objectState<PipelineLayoutState>(info.layout);
        if (info.sType != VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_NV
            || info.stageCount == 0 || info.pStages == nullptr
            || info.groupCount == 0 || info.pGroups == nullptr
            || info.maxRecursionDepth == 0 || info.maxRecursionDepth > 2
            || !gState.pipelineLayouts.contains(layout) || layout->device != device) {
            for (auto* state : created) { gState.pipelines.erase(state); delete state; }
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto* state = new (std::nothrow) PipelineState{};
        if (state == nullptr) {
            for (auto* previous : created) { gState.pipelines.erase(previous); delete previous; }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        state->device = device;
        state->layout = layout;
        state->rayTracingNV = true;
        state->maxRecursionDepth = info.maxRecursionDepth;
        state->rayTracingHash = UINT64_C(1469598103934665603);
        try {
            state->rayTracingShaders.reserve(info.stageCount);
            state->rayTracingStages.reserve(info.stageCount);
            state->rayTracingGroups.assign(info.pGroups, info.pGroups + info.groupCount);
        } catch (const std::bad_alloc&) {
            delete state;
            for (auto* previous : created) { gState.pipelines.erase(previous); delete previous; }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        bool valid = true;
        for (std::uint32_t stageIndex = 0; stageIndex < info.stageCount; ++stageIndex) {
            const auto& stage = info.pStages[stageIndex];
            auto* shader = objectState<ShaderModuleState>(stage.module);
            if (!validRayTracingShaderStageNV(stage.stage) || stage.pName == nullptr
                || !gState.shaderModules.contains(shader) || shader->device != device) {
                valid = false;
                break;
            }
            state->rayTracingShaders.push_back(shader);
            state->rayTracingStages.push_back(stage.stage);
            state->rayTracingHash = mixRayTracingHash(state->rayTracingHash, shader->hash);
            state->rayTracingHash = mixRayTracingHash(state->rayTracingHash, stage.stage);
        }
        for (std::uint32_t groupIndex = 0; valid && groupIndex < info.groupCount; ++groupIndex) {
            auto& group = state->rayTracingGroups[groupIndex];
            group.pNext = nullptr;
            const std::uint32_t shaderIndices[] = {
                group.generalShader,
                group.closestHitShader,
                group.anyHitShader,
                group.intersectionShader,
            };
            for (const std::uint32_t shaderIndex : shaderIndices) {
                if (shaderIndex != VK_SHADER_UNUSED_NV && shaderIndex >= info.stageCount) {
                    valid = false;
                    break;
                }
            }
            state->rayTracingHash = mixRayTracingHash(state->rayTracingHash, group.type);
            for (const std::uint32_t shaderIndex : shaderIndices) {
                state->rayTracingHash = mixRayTracingHash(state->rayTracingHash, shaderIndex);
            }
        }
        if (!valid) {
            delete state;
            for (auto* previous : created) { gState.pipelines.erase(previous); delete previous; }
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        state->isMetalRayProbe = state->rayTracingShaders.size() == 1
            && state->rayTracingStages.size() == 1
            && state->rayTracingStages[0] == VK_SHADER_STAGE_RAYGEN_BIT_KHR
            && state->rayTracingShaders[0] != nullptr
            && state->rayTracingShaders[0]->isAddUInt32;
        gState.pipelines.insert(state);
        created.push_back(state);
        pipelines[index] = makeObjectHandle<VkPipeline>(state);
        if (traceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: create RT pipeline NV handle=%p stages=%u groups=%u recursion=%u hash=%016llx\n",
                static_cast<void*>(state),
                info.stageCount,
                info.groupCount,
                info.maxRecursionDepth,
                static_cast<unsigned long long>(state->rayTracingHash)
            );
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateRayTracingPipelinesKHR(
    VkDevice device,
    VkDeferredOperationKHR deferredOperation,
    VkPipelineCache pipelineCache,
    std::uint32_t createInfoCount,
    const VkRayTracingPipelineCreateInfoKHR* createInfos,
    const VkAllocationCallbacks*,
    VkPipeline* pipelines
) {
    if (createInfoCount == 0 || createInfos == nullptr || pipelines == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_DEVICE_LOST;
    DeferredOperationKHRState* deferredState = nullptr;
    if (deferredOperation != VK_NULL_HANDLE) {
        deferredState = objectState<DeferredOperationKHRState>(deferredOperation);
        if (!gState.deferredOperationsKHR.contains(deferredState) || deferredState->device != device) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    if (pipelineCache != VK_NULL_HANDLE) {
        auto* cache = objectState<PipelineCacheState>(pipelineCache);
        if (!gState.pipelineCaches.contains(cache) || cache->device != device) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    for (std::uint32_t index = 0; index < createInfoCount; ++index) pipelines[index] = VK_NULL_HANDLE;

    std::vector<PipelineState*> created;
    try {
        created.reserve(createInfoCount);
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    for (std::uint32_t index = 0; index < createInfoCount; ++index) {
        const auto& info = createInfos[index];
        auto* layout = objectState<PipelineLayoutState>(info.layout);
        const std::uint32_t libraryCount = info.pLibraryInfo == nullptr ? 0 : info.pLibraryInfo->libraryCount;
        if (rayTracingTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: RT KHR request index=%u flags=%#x stages=%u groups=%u libraries=%u depth=%u layout=%p knownLayout=%d deferred=%d\n",
                index,
                info.flags,
                info.stageCount,
                info.groupCount,
                libraryCount,
                info.maxPipelineRayRecursionDepth,
                reinterpret_cast<void*>(info.layout),
                gState.pipelineLayouts.contains(layout) ? 1 : 0,
                deferredState == nullptr ? 0 : 1
            );
        }
        if (info.sType != VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR
            || (info.stageCount != 0 && info.pStages == nullptr)
            || (info.groupCount != 0 && info.pGroups == nullptr)
            || (info.stageCount == 0 && libraryCount == 0)
            || info.maxPipelineRayRecursionDepth == 0
            || info.maxPipelineRayRecursionDepth > kMaxRayRecursionDepth
            || !gState.pipelineLayouts.contains(layout) || layout->device != device) {
            if (rayTracingTraceEnabled()) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: RT KHR reject base sType=%d stagePtr=%d groupPtr=%d depth=%u layoutDevice=%d\n",
                    info.sType,
                    info.pStages != nullptr ? 1 : 0,
                    info.pGroups != nullptr ? 1 : 0,
                    info.maxPipelineRayRecursionDepth,
                    gState.pipelineLayouts.contains(layout) && layout->device == device ? 1 : 0
                );
            }
            for (auto* state : created) { gState.pipelines.erase(state); delete state; }
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        auto* state = new (std::nothrow) PipelineState{};
        if (state == nullptr) {
            for (auto* previous : created) { gState.pipelines.erase(previous); delete previous; }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        state->device = device;
        state->layout = layout;
        state->rayTracingKHR = true;
        state->maxRecursionDepth = info.maxPipelineRayRecursionDepth;
        state->rayTracingHash = UINT64_C(1469598103934665603);
        bool valid = true;
        try {
            state->rayTracingShaders.reserve(info.stageCount);
            state->rayTracingStages.reserve(info.stageCount);
            state->rayTracingGroups.reserve(info.groupCount);
            state->rayTracingLibraries.reserve(libraryCount);
        } catch (const std::bad_alloc&) {
            delete state;
            for (auto* previous : created) { gState.pipelines.erase(previous); delete previous; }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        for (std::uint32_t stageIndex = 0; stageIndex < info.stageCount; ++stageIndex) {
            const auto& stage = info.pStages[stageIndex];
            auto* shader = objectState<ShaderModuleState>(stage.module);
            if (!validRayTracingShaderStageNV(stage.stage) || stage.pName == nullptr
                || !gState.shaderModules.contains(shader) || shader->device != device) {
                if (rayTracingTraceEnabled()) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: RT KHR reject stage=%u type=%#x entry=%d module=%p knownModule=%d\n",
                        stageIndex,
                        stage.stage,
                        stage.pName != nullptr ? 1 : 0,
                        reinterpret_cast<void*>(stage.module),
                        gState.shaderModules.contains(shader) && shader->device == device ? 1 : 0
                    );
                }
                valid = false;
                break;
            }
            state->rayTracingShaders.push_back(shader);
            state->rayTracingStages.push_back(stage.stage);
            state->rayTracingHash = mixRayTracingHash(state->rayTracingHash, shader->hash);
            state->rayTracingHash = mixRayTracingHash(state->rayTracingHash, stage.stage);
        }
        for (std::uint32_t groupIndex = 0; valid && groupIndex < info.groupCount; ++groupIndex) {
            const auto& inputGroup = info.pGroups[groupIndex];
            if (inputGroup.sType != VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR
                || inputGroup.pShaderGroupCaptureReplayHandle != nullptr) {
                if (rayTracingTraceEnabled()) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: RT KHR reject group=%u sType=%d captureReplay=%d\n",
                        groupIndex,
                        inputGroup.sType,
                        inputGroup.pShaderGroupCaptureReplayHandle != nullptr ? 1 : 0
                    );
                }
                valid = false;
                break;
            }
            VkRayTracingShaderGroupCreateInfoNV group{};
            group.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_NV;
            group.type = inputGroup.type;
            group.generalShader = inputGroup.generalShader;
            group.closestHitShader = inputGroup.closestHitShader;
            group.anyHitShader = inputGroup.anyHitShader;
            group.intersectionShader = inputGroup.intersectionShader;
            const std::uint32_t shaderIndices[] = {
                group.generalShader,
                group.closestHitShader,
                group.anyHitShader,
                group.intersectionShader,
            };
            for (const std::uint32_t shaderIndex : shaderIndices) {
                if (shaderIndex != VK_SHADER_UNUSED_KHR && shaderIndex >= info.stageCount) {
                    if (rayTracingTraceEnabled()) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: RT KHR reject group=%u shaderIndex=%u stageCount=%u\n",
                            groupIndex,
                            shaderIndex,
                            info.stageCount
                        );
                    }
                    valid = false;
                    break;
                }
            }
            state->rayTracingGroups.push_back(group);
            state->rayTracingHash = mixRayTracingHash(state->rayTracingHash, group.type);
            for (const std::uint32_t shaderIndex : shaderIndices) {
                state->rayTracingHash = mixRayTracingHash(state->rayTracingHash, shaderIndex);
            }
        }
        if (info.pLibraryInfo != nullptr) {
            if (info.pLibraryInfo->libraryCount != 0 && info.pLibraryInfo->pLibraries == nullptr) {
                if (rayTracingTraceEnabled()) {
                    std::fprintf(stderr, "imb-vulkan-icd: RT KHR reject null library array\n");
                }
                valid = false;
            }
            for (std::uint32_t libraryIndex = 0; valid && libraryIndex < info.pLibraryInfo->libraryCount; ++libraryIndex) {
                auto* library = objectState<PipelineState>(info.pLibraryInfo->pLibraries[libraryIndex]);
                if (!gState.pipelines.contains(library) || library->device != device || !library->rayTracingKHR) {
                    if (rayTracingTraceEnabled()) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: RT KHR reject library=%u handle=%p known=%d rayKHR=%d\n",
                            libraryIndex,
                            reinterpret_cast<void*>(info.pLibraryInfo->pLibraries[libraryIndex]),
                            gState.pipelines.contains(library) && library->device == device ? 1 : 0,
                            gState.pipelines.contains(library) && library->rayTracingKHR ? 1 : 0
                        );
                    }
                    valid = false;
                    break;
                }
                state->rayTracingLibraries.push_back(library);
                state->rayTracingHash = mixRayTracingHash(state->rayTracingHash, library->rayTracingHash);

                // A linked KHR ray-tracing pipeline exposes shader groups from
                // its libraries in library-list order.  Keep a flattened view
                // so vkGetRayTracingShaderGroupHandlesKHR and later Metal
                // compilation use the executable pipeline's group numbering,
                // not only the final link create-info's (usually empty) arrays.
                if (state->rayTracingShaders.size() > UINT32_MAX
                    || library->rayTracingShaders.size()
                        > UINT32_MAX - state->rayTracingShaders.size()) {
                    valid = false;
                    break;
                }
                const std::uint32_t stageBase =
                    static_cast<std::uint32_t>(state->rayTracingShaders.size());
                try {
                    state->rayTracingShaders.insert(
                        state->rayTracingShaders.end(),
                        library->rayTracingShaders.begin(),
                        library->rayTracingShaders.end()
                    );
                    state->rayTracingStages.insert(
                        state->rayTracingStages.end(),
                        library->rayTracingStages.begin(),
                        library->rayTracingStages.end()
                    );
                    for (const auto& libraryGroup : library->rayTracingGroups) {
                        auto linkedGroup = libraryGroup;
                        auto rebaseShader = [stageBase](std::uint32_t shader) {
                            return shader == VK_SHADER_UNUSED_KHR ? shader : shader + stageBase;
                        };
                        linkedGroup.generalShader = rebaseShader(linkedGroup.generalShader);
                        linkedGroup.closestHitShader = rebaseShader(linkedGroup.closestHitShader);
                        linkedGroup.anyHitShader = rebaseShader(linkedGroup.anyHitShader);
                        linkedGroup.intersectionShader = rebaseShader(linkedGroup.intersectionShader);
                        state->rayTracingGroups.push_back(linkedGroup);
                    }
                } catch (const std::bad_alloc&) {
                    delete state;
                    for (auto* previous : created) {
                        gState.pipelines.erase(previous);
                        delete previous;
                    }
                    return VK_ERROR_OUT_OF_HOST_MEMORY;
                }
            }
        }
        if (!valid) {
            delete state;
            for (auto* previous : created) { gState.pipelines.erase(previous); delete previous; }
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        state->isMetalRayProbe = state->rayTracingShaders.size() == 1
            && state->rayTracingStages.size() == 1
            && state->rayTracingStages[0] == VK_SHADER_STAGE_RAYGEN_BIT_KHR
            && state->rayTracingShaders[0] != nullptr
            && state->rayTracingShaders[0]->isAddUInt32;
        gState.pipelines.insert(state);
        created.push_back(state);
        pipelines[index] = makeObjectHandle<VkPipeline>(state);
        if (traceEnabled() || rayTracingTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: create RT pipeline KHR handle=%p stages=%u groups=%u libraries=%u linkedStages=%zu linkedGroups=%zu recursion=%u hash=%016llx\n",
                static_cast<void*>(state),
                info.stageCount,
                info.groupCount,
                libraryCount,
                state->rayTracingShaders.size(),
                state->rayTracingGroups.size(),
                info.maxPipelineRayRecursionDepth,
                static_cast<unsigned long long>(state->rayTracingHash)
            );
            for (std::size_t stageIndex = 0; stageIndex < state->rayTracingShaders.size(); ++stageIndex) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: RT KHR stage pipeline=%p index=%zu type=%#x moduleHash=%016llx\n",
                    static_cast<void*>(state),
                    stageIndex,
                    state->rayTracingStages[stageIndex],
                    static_cast<unsigned long long>(state->rayTracingShaders[stageIndex]->hash)
                );
            }
        }
    }
    if (deferredState != nullptr) {
        deferredState->result = VK_SUCCESS;
        return VK_OPERATION_NOT_DEFERRED_KHR;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetRayTracingShaderGroupHandlesNV(
    VkDevice device,
    VkPipeline pipeline,
    std::uint32_t firstGroup,
    std::uint32_t groupCount,
    std::size_t dataSize,
    void* data
) {
    if (groupCount != 0 && data == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<PipelineState>(pipeline);
    const bool valid = validDevice(device) && gState.pipelines.contains(state) && state->device == device
        && (state->rayTracingNV || state->rayTracingKHR) && firstGroup <= state->rayTracingGroups.size()
        && groupCount <= state->rayTracingGroups.size() - firstGroup
        && dataSize >= static_cast<std::size_t>(groupCount) * kShaderGroupHandleSize;
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: RT group handles pipeline=%p first=%u count=%u bytes=%zu storedGroups=%zu valid=%d\n",
            static_cast<void*>(state),
            firstGroup,
            groupCount,
            dataSize,
            gState.pipelines.contains(state) ? state->rayTracingGroups.size() : 0,
            valid ? 1 : 0
        );
    }
    if (!valid) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    auto* output = static_cast<std::uint8_t*>(data);
    for (std::uint32_t group = 0; group < groupCount; ++group) {
        writeRayTracingGroupHandle(
            *state,
            firstGroup + group,
            output + static_cast<std::size_t>(group) * kShaderGroupHandleSize
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetRayTracingShaderGroupHandlesKHR(
    VkDevice device,
    VkPipeline pipeline,
    std::uint32_t firstGroup,
    std::uint32_t groupCount,
    std::size_t dataSize,
    void* data
) {
    return imb_vkGetRayTracingShaderGroupHandlesNV(
        device,
        pipeline,
        firstGroup,
        groupCount,
        dataSize,
        data
    );
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR(
    VkDevice,
    VkPipeline,
    std::uint32_t,
    std::uint32_t,
    std::size_t,
    void*
) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkDeviceSize VKAPI_CALL imb_vkGetRayTracingShaderGroupStackSizeKHR(
    VkDevice device,
    VkPipeline pipeline,
    std::uint32_t group,
    VkShaderGroupShaderKHR
) {
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<PipelineState>(pipeline);
    const bool valid = validDevice(device) && gState.pipelines.contains(state) && state->rayTracingKHR
        && group < state->rayTracingGroups.size();
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: RT group stack pipeline=%p group=%u storedGroups=%zu valid=%d\n",
            static_cast<void*>(state),
            group,
            gState.pipelines.contains(state) ? state->rayTracingGroups.size() : 0,
            valid ? 1 : 0
        );
    }
    if (!valid) {
        return 0;
    }
    return 64;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetAccelerationStructureHandleNV(
    VkDevice device,
    VkAccelerationStructureNV accelerationStructure,
    std::size_t dataSize,
    void* data
) {
    if (data == nullptr || dataSize < sizeof(std::uint64_t)) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<AccelerationStructureNVState>(accelerationStructure);
    if (!validDevice(device) || !gState.accelerationStructuresNV.contains(state)
        || state->device != device || state->memory == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::memcpy(data, &state->opaqueHandle, sizeof(state->opaqueHandle));
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCompileDeferredNV(
    VkDevice device,
    VkPipeline pipeline,
    std::uint32_t shader
) {
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<PipelineState>(pipeline);
    if (!validDevice(device) || !gState.pipelines.contains(state) || state->device != device
        || !state->rayTracingNV || shader >= state->rayTracingShaders.size()) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdBuildAccelerationStructureNV(
    VkCommandBuffer commandBuffer,
    const VkAccelerationStructureInfoNV* info,
    VkBuffer instanceData,
    VkDeviceSize instanceOffset,
    VkBool32 update,
    VkAccelerationStructureNV destination,
    VkAccelerationStructureNV source,
    VkBuffer scratch,
    VkDeviceSize scratchOffset
) {
    if (info == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* destinationState = objectState<AccelerationStructureNVState>(destination);
    auto* sourceState = objectState<AccelerationStructureNVState>(source);
    auto* instanceState = objectState<BufferState>(instanceData);
    auto* scratchState = objectState<BufferState>(scratch);
    const bool validSource = source == VK_NULL_HANDLE || gState.accelerationStructuresNV.contains(sourceState);
    const bool validInstance = instanceData == VK_NULL_HANDLE || gState.buffers.contains(instanceState);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.accelerationStructuresNV.contains(destinationState)
        || destinationState->memory == nullptr || !validSource || !validInstance
        || scratch == VK_NULL_HANDLE || !gState.buffers.contains(scratchState)
        || scratchState->memory == nullptr) {
        return;
    }
    command->accelerationStructureBuildsNV.push_back(RecordedAccelerationStructureBuildNV{
        destinationState,
        source == VK_NULL_HANDLE ? nullptr : sourceState,
        instanceData == VK_NULL_HANDLE ? nullptr : instanceState,
        instanceOffset,
        scratchState,
        scratchOffset,
        update == VK_TRUE,
    });
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: record build AS NV command=%p dst=%p src=%p update=%u geometries=%u instances=%u scratch=%p+%llu\n",
            static_cast<void*>(command),
            static_cast<void*>(destinationState),
            static_cast<void*>(source == VK_NULL_HANDLE ? nullptr : sourceState),
            update,
            info->geometryCount,
            info->instanceCount,
            static_cast<void*>(scratchState),
            static_cast<unsigned long long>(scratchOffset)
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyAccelerationStructureNV(
    VkCommandBuffer commandBuffer,
    VkAccelerationStructureNV destination,
    VkAccelerationStructureNV source,
    VkCopyAccelerationStructureModeNV mode
) {
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* destinationState = objectState<AccelerationStructureNVState>(destination);
    auto* sourceState = objectState<AccelerationStructureNVState>(source);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.accelerationStructuresNV.contains(destinationState)
        || !gState.accelerationStructuresNV.contains(sourceState)) {
        return;
    }
    command->accelerationStructureCopiesNV.push_back({destinationState, sourceState, mode});
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdTraceRaysNV(
    VkCommandBuffer commandBuffer,
    VkBuffer raygenBuffer,
    VkDeviceSize raygenOffset,
    VkBuffer missBuffer,
    VkDeviceSize missOffset,
    VkDeviceSize missStride,
    VkBuffer hitBuffer,
    VkDeviceSize hitOffset,
    VkDeviceSize hitStride,
    VkBuffer callableBuffer,
    VkDeviceSize callableOffset,
    VkDeviceSize callableStride,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t depth
) {
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* raygen = objectState<BufferState>(raygenBuffer);
    auto* miss = objectState<BufferState>(missBuffer);
    auto* hit = objectState<BufferState>(hitBuffer);
    auto* callable = objectState<BufferState>(callableBuffer);
    const auto validOptionalBuffer = [](VkBuffer handle, BufferState* state) {
        return handle == VK_NULL_HANDLE || gState.buffers.contains(state);
    };
    if (!gState.commandBuffers.contains(command) || !command->recording
        || command->pipeline == nullptr || !command->pipeline->rayTracingNV
        || raygenBuffer == VK_NULL_HANDLE || !gState.buffers.contains(raygen)
        || !validOptionalBuffer(missBuffer, miss)
        || !validOptionalBuffer(hitBuffer, hit)
        || !validOptionalBuffer(callableBuffer, callable)
        || width == 0 || height == 0 || depth == 0) {
        return;
    }
    command->rayTracesNV.push_back(RecordedRayTraceNV{
        command->pipeline,
        raygen,
        missBuffer == VK_NULL_HANDLE ? nullptr : miss,
        hitBuffer == VK_NULL_HANDLE ? nullptr : hit,
        callableBuffer == VK_NULL_HANDLE ? nullptr : callable,
        raygenOffset,
        missOffset,
        missStride,
        hitOffset,
        hitStride,
        callableOffset,
        callableStride,
        width,
        height,
        depth,
    });
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: record trace rays NV command=%p pipeline=%p extent=%ux%ux%u\n",
            static_cast<void*>(command),
            static_cast<void*>(command->pipeline),
            width,
            height,
            depth
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdWriteAccelerationStructuresPropertiesNV(
    VkCommandBuffer commandBuffer,
    std::uint32_t accelerationStructureCount,
    const VkAccelerationStructureNV* accelerationStructures,
    VkQueryType queryType,
    VkQueryPool queryPool,
    std::uint32_t firstQuery
) {
    if (accelerationStructureCount != 0 && accelerationStructures == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* pool = objectState<QueryPoolState>(queryPool);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.queryPools.contains(pool) || pool->type != queryType
        || firstQuery > pool->count || accelerationStructureCount > pool->count - firstQuery) {
        return;
    }
    for (std::uint32_t index = 0; index < accelerationStructureCount; ++index) {
        auto* state = objectState<AccelerationStructureNVState>(accelerationStructures[index]);
        if (!gState.accelerationStructuresNV.contains(state)) return;
        pool->values[firstQuery + index] = state->objectSize;
        pool->available[firstQuery + index] = true;
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdBuildAccelerationStructuresKHR(
    VkCommandBuffer commandBuffer,
    std::uint32_t infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR* infos,
    const VkAccelerationStructureBuildRangeInfoKHR* const* buildRangeInfos
) {
    if (infoCount != 0 && (infos == nullptr || buildRangeInfos == nullptr)) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    if (!gState.commandBuffers.contains(command) || !command->recording) {
        if (rayTracingTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: reject AS KHR record command=%p known=%d recording=%d infos=%u\n",
                static_cast<void*>(command),
                gState.commandBuffers.contains(command) ? 1 : 0,
                gState.commandBuffers.contains(command) && command->recording ? 1 : 0,
                infoCount
            );
        }
        return;
    }
    for (std::uint32_t index = 0; index < infoCount; ++index) {
        const auto& info = infos[index];
        auto* destination = objectState<AccelerationStructureKHRState>(info.dstAccelerationStructure);
        auto* source = objectState<AccelerationStructureKHRState>(info.srcAccelerationStructure);
        auto* scratch = bufferForDeviceAddressLocked(command->device, info.scratchData.deviceAddress);
        if (!gState.accelerationStructuresKHR.contains(destination)
            || (info.srcAccelerationStructure != VK_NULL_HANDLE
                && !gState.accelerationStructuresKHR.contains(source))
            || info.scratchData.deviceAddress == 0
            || scratch == nullptr
            || (info.geometryCount != 0 && buildRangeInfos[index] == nullptr)) {
            if (rayTracingTraceEnabled()) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: reject AS KHR build index=%u dst=%p knownDst=%d src=%p knownSrc=%d type=%d mode=%d geometries=%u scratch=%#llx knownScratch=%d ranges=%p\n",
                    index,
                    static_cast<void*>(destination),
                    gState.accelerationStructuresKHR.contains(destination) ? 1 : 0,
                    static_cast<void*>(info.srcAccelerationStructure == VK_NULL_HANDLE ? nullptr : source),
                    info.srcAccelerationStructure == VK_NULL_HANDLE
                        || gState.accelerationStructuresKHR.contains(source) ? 1 : 0,
                    info.type,
                    info.mode,
                    info.geometryCount,
                    static_cast<unsigned long long>(info.scratchData.deviceAddress),
                    scratch != nullptr ? 1 : 0,
                    static_cast<const void*>(
                        info.geometryCount == 0 ? nullptr : buildRangeInfos[index]
                    )
                );
            }
            return;
        }
        RecordedAccelerationStructureBuildKHR build{};
        build.sequence = command->nextCommandSequence++;
        build.destination = destination;
        build.source = info.srcAccelerationStructure == VK_NULL_HANDLE ? nullptr : source;
        build.type = info.type;
        build.flags = info.flags;
        build.mode = info.mode;
        build.scratchAddress = info.scratchData.deviceAddress;
        try {
            if (info.geometryCount != 0) {
                build.geometries.reserve(info.geometryCount);
                for (std::uint32_t geometryIndex = 0; geometryIndex < info.geometryCount; ++geometryIndex) {
                    const VkAccelerationStructureGeometryKHR* input = info.pGeometries != nullptr
                        ? &info.pGeometries[geometryIndex]
                        : (info.ppGeometries == nullptr ? nullptr : info.ppGeometries[geometryIndex]);
                    if (input == nullptr
                        || input->sType != VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR) {
                        return;
                    }
                    auto geometry = *input;
                    geometry.pNext = nullptr;
                    if (geometry.geometryType == VK_GEOMETRY_TYPE_TRIANGLES_KHR) {
                        geometry.geometry.triangles.pNext = nullptr;
                    } else if (geometry.geometryType == VK_GEOMETRY_TYPE_AABBS_KHR) {
                        geometry.geometry.aabbs.pNext = nullptr;
                    } else if (geometry.geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR) {
                        geometry.geometry.instances.pNext = nullptr;
                    } else {
                        return;
                    }
                    build.geometries.push_back(geometry);
                }
                build.ranges.assign(buildRangeInfos[index], buildRangeInfos[index] + info.geometryCount);
            }
            command->accelerationStructureBuildsKHR.push_back(std::move(build));
        } catch (const std::bad_alloc&) {
            return;
        }
        if (traceEnabled() || rayTracingTraceEnabled()) {
            const auto& recorded = command->accelerationStructureBuildsKHR.back();
            std::fprintf(
                stderr,
                "imb-vulkan-icd: record build AS KHR command=%p sequence=%llu dst=%p src=%p type=%d mode=%d geometries=%u firstType=%d firstPrimitives=%u scratch=%#llx\n",
                static_cast<void*>(command),
                static_cast<unsigned long long>(recorded.sequence),
                static_cast<void*>(destination),
                static_cast<void*>(info.srcAccelerationStructure == VK_NULL_HANDLE ? nullptr : source),
                info.type,
                info.mode,
                info.geometryCount,
                info.geometryCount == 0
                    ? -1
                    : static_cast<int>(recorded.geometries[0].geometryType),
                info.geometryCount == 0 ? 0 : recorded.ranges[0].primitiveCount,
                static_cast<unsigned long long>(info.scratchData.deviceAddress)
            );
        }
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdBuildAccelerationStructuresIndirectKHR(
    VkCommandBuffer commandBuffer,
    std::uint32_t infoCount,
    const VkAccelerationStructureBuildGeometryInfoKHR* infos,
    const VkDeviceAddress* indirectDeviceAddresses,
    const std::uint32_t* indirectStrides,
    const std::uint32_t* const* maxPrimitiveCounts
) {
    if (traceEnabled() || rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: indirect AS KHR build command=%p infos=%u data=%p strides=%p maxCounts=%p ignored because feature is not advertised\n",
            reinterpret_cast<void*>(commandBuffer),
            infoCount,
            static_cast<const void*>(indirectDeviceAddresses),
            static_cast<const void*>(indirectStrides),
            static_cast<const void*>(maxPrimitiveCounts)
        );
        for (std::uint32_t index = 0; infos != nullptr && index < infoCount; ++index) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: indirect AS KHR info=%u dst=%p src=%p type=%d mode=%d geometries=%u address=%#llx stride=%u\n",
                index,
                reinterpret_cast<void*>(infos[index].dstAccelerationStructure),
                reinterpret_cast<void*>(infos[index].srcAccelerationStructure),
                infos[index].type,
                infos[index].mode,
                infos[index].geometryCount,
                static_cast<unsigned long long>(
                    indirectDeviceAddresses == nullptr ? 0 : indirectDeviceAddresses[index]
                ),
                indirectStrides == nullptr ? 0 : indirectStrides[index]
            );
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkBuildAccelerationStructuresKHR(
    VkDevice,
    VkDeferredOperationKHR,
    std::uint32_t,
    const VkAccelerationStructureBuildGeometryInfoKHR*,
    const VkAccelerationStructureBuildRangeInfoKHR* const*
) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCopyAccelerationStructureKHR(
    VkDevice,
    VkDeferredOperationKHR,
    const VkCopyAccelerationStructureInfoKHR*
) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCopyAccelerationStructureToMemoryKHR(
    VkDevice,
    VkDeferredOperationKHR,
    const VkCopyAccelerationStructureToMemoryInfoKHR*
) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCopyMemoryToAccelerationStructureKHR(
    VkDevice,
    VkDeferredOperationKHR,
    const VkCopyMemoryToAccelerationStructureInfoKHR*
) {
    return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkWriteAccelerationStructuresPropertiesKHR(
    VkDevice device,
    std::uint32_t accelerationStructureCount,
    const VkAccelerationStructureKHR* accelerationStructures,
    VkQueryType,
    std::size_t dataSize,
    void* data,
    std::size_t stride
) {
    if (accelerationStructureCount != 0 && (accelerationStructures == nullptr || data == nullptr)) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (accelerationStructureCount != 0
        && (stride < sizeof(std::uint64_t)
            || dataSize < static_cast<std::size_t>(accelerationStructureCount - 1) * stride + sizeof(std::uint64_t))) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_DEVICE_LOST;
    auto* output = static_cast<std::uint8_t*>(data);
    for (std::uint32_t index = 0; index < accelerationStructureCount; ++index) {
        auto* state = objectState<AccelerationStructureKHRState>(accelerationStructures[index]);
        if (!gState.accelerationStructuresKHR.contains(state) || state->device != device) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        const std::uint64_t value = state->size;
        std::memcpy(output + static_cast<std::size_t>(index) * stride, &value, sizeof(value));
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyAccelerationStructureKHR(
    VkCommandBuffer commandBuffer,
    const VkCopyAccelerationStructureInfoKHR* info
) {
    if (info == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* destination = objectState<AccelerationStructureKHRState>(info->dst);
    auto* source = objectState<AccelerationStructureKHRState>(info->src);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.accelerationStructuresKHR.contains(destination)
        || !gState.accelerationStructuresKHR.contains(source)) {
        return;
    }
    const auto sequence = command->nextCommandSequence++;
    command->accelerationStructureCopiesKHR.push_back({sequence, destination, source, info->mode});
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: recorded AS copy KHR sequence=%llu mode=%u src=%p srcBuilt=%d srcType=%u dst=%p dstBuilt=%d dstType=%u\n",
            static_cast<unsigned long long>(sequence),
            static_cast<unsigned>(info->mode),
            static_cast<void*>(source),
            source->built ? 1 : 0,
            static_cast<unsigned>(source->type),
            static_cast<void*>(destination),
            destination->built ? 1 : 0,
            static_cast<unsigned>(destination->type)
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyAccelerationStructureToMemoryKHR(
    VkCommandBuffer,
    const VkCopyAccelerationStructureToMemoryInfoKHR*
) {
    if (traceEnabled()) {
        std::fprintf(stderr, "imb-vulkan-icd: AS KHR serialization command unsupported\n");
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyMemoryToAccelerationStructureKHR(
    VkCommandBuffer,
    const VkCopyMemoryToAccelerationStructureInfoKHR*
) {
    if (traceEnabled()) {
        std::fprintf(stderr, "imb-vulkan-icd: AS KHR deserialization command unsupported\n");
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdWriteAccelerationStructuresPropertiesKHR(
    VkCommandBuffer commandBuffer,
    std::uint32_t accelerationStructureCount,
    const VkAccelerationStructureKHR* accelerationStructures,
    VkQueryType queryType,
    VkQueryPool queryPool,
    std::uint32_t firstQuery
) {
    if (accelerationStructureCount != 0 && accelerationStructures == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* pool = objectState<QueryPoolState>(queryPool);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.queryPools.contains(pool) || pool->type != queryType
        || firstQuery > pool->count || accelerationStructureCount > pool->count - firstQuery) {
        return;
    }
    for (std::uint32_t index = 0; index < accelerationStructureCount; ++index) {
        auto* state = objectState<AccelerationStructureKHRState>(accelerationStructures[index]);
        if (!gState.accelerationStructuresKHR.contains(state)) return;
        pool->values[firstQuery + index] = state->size;
        pool->available[firstQuery + index] = true;
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdTraceRaysKHR(
    VkCommandBuffer commandBuffer,
    const VkStridedDeviceAddressRegionKHR* raygen,
    const VkStridedDeviceAddressRegionKHR* miss,
    const VkStridedDeviceAddressRegionKHR* hit,
    const VkStridedDeviceAddressRegionKHR* callable,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t depth
) {
    if (raygen == nullptr || miss == nullptr || hit == nullptr || callable == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    const bool valid = gState.commandBuffers.contains(command) && command->recording
        && command->pipeline != nullptr && command->pipeline->rayTracingKHR
        && raygen->deviceAddress != 0
        && bufferForDeviceAddressLocked(command->device, raygen->deviceAddress) != nullptr
        && width != 0 && height != 0 && depth != 0;
    std::int32_t raygenGroup = valid && rayTracingTraceEnabled()
        ? rayTracingGroupForSBTAddressLocked(command->pipeline, command->device, raygen->deviceAddress)
        : -1;
    if (raygenGroup < 0 && valid && rayTracingTraceEnabled()) {
        raygenGroup = inferRaygenGroupFromSparseRecordLocked(
            command->pipeline,
            command->device,
            *raygen
        );
    }
    const std::int32_t missGroup = valid && rayTracingTraceEnabled() && miss->deviceAddress != 0
        ? rayTracingGroupForSBTAddressLocked(command->pipeline, command->device, miss->deviceAddress)
        : -1;
    const std::int32_t hitGroup = valid && rayTracingTraceEnabled() && hit->deviceAddress != 0
        ? rayTracingGroupForSBTAddressLocked(command->pipeline, command->device, hit->deviceAddress)
        : -1;
    const std::uint64_t raygenShaderHash = rayTracingShaderHashForGroup(
        command->pipeline,
        raygenGroup
    );
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: RT trace command=%p pipeline=%p extent=%ux%ux%u raygen=%#llx+%llu/%llu miss=%#llx+%llu/%llu hit=%#llx+%llu/%llu groups=%d/%d/%d raygenHash=%016llx valid=%d\n",
            static_cast<void*>(command),
            static_cast<void*>(gState.commandBuffers.contains(command) ? command->pipeline : nullptr),
            width,
            height,
            depth,
            static_cast<unsigned long long>(raygen->deviceAddress),
            static_cast<unsigned long long>(raygen->stride),
            static_cast<unsigned long long>(raygen->size),
            static_cast<unsigned long long>(miss->deviceAddress),
            static_cast<unsigned long long>(miss->stride),
            static_cast<unsigned long long>(miss->size),
            static_cast<unsigned long long>(hit->deviceAddress),
            static_cast<unsigned long long>(hit->stride),
            static_cast<unsigned long long>(hit->size),
            raygenGroup,
            missGroup,
            hitGroup,
            static_cast<unsigned long long>(raygenShaderHash),
            valid ? 1 : 0
        );
    }
    if (!valid) {
        return;
    }
    command->rayTracesKHR.push_back({command->pipeline, *raygen, *miss, *hit, *callable, width, height, depth});
    command->rayTracesKHR.back().descriptorSets = command->rayTracingDescriptorSets;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: record trace rays KHR command=%p pipeline=%p extent=%ux%ux%u\n",
            static_cast<void*>(command),
            static_cast<void*>(command->pipeline),
            width,
            height,
            depth
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdTraceRaysIndirectKHR(
    VkCommandBuffer,
    const VkStridedDeviceAddressRegionKHR*,
    const VkStridedDeviceAddressRegionKHR*,
    const VkStridedDeviceAddressRegionKHR*,
    const VkStridedDeviceAddressRegionKHR*,
    VkDeviceAddress
) {
    if (traceEnabled()) {
        std::fprintf(stderr, "imb-vulkan-icd: indirect trace rays KHR ignored because feature is not advertised\n");
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdSetRayTracingPipelineStackSizeKHR(
    VkCommandBuffer commandBuffer,
    std::uint32_t pipelineStackSize
) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: RT pipeline stack command=%p size=%u\n",
            static_cast<void*>(commandBuffer),
            pipelineStackSize
        );
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateDescriptorPool(
    VkDevice device,
    const VkDescriptorPoolCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkDescriptorPool* descriptorPool
) {
    if (createInfo == nullptr || descriptorPool == nullptr
        || (createInfo->poolSizeCount != 0 && createInfo->pPoolSizes == nullptr)) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    auto* state = new (std::nothrow) DescriptorPoolState{
        device,
        std::max<std::uint32_t>(1, createInfo->maxSets),
        0,
    };
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    gState.descriptorPools.insert(state);
    *descriptorPool = makeObjectHandle<VkDescriptorPool>(state);
    return VK_SUCCESS;
}

void destroyDescriptorSetsForPoolLocked(DescriptorPoolState* pool) {
    for (auto iterator = gState.descriptorSets.begin(); iterator != gState.descriptorSets.end();) {
        auto* set = *iterator;
        if (set->pool == pool) {
            iterator = gState.descriptorSets.erase(iterator);
            delete set;
        } else {
            ++iterator;
        }
    }
    pool->allocatedSets = 0;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyDescriptorPool(
    VkDevice,
    VkDescriptorPool descriptorPool,
    const VkAllocationCallbacks*
) {
    if (descriptorPool == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* pool = objectState<DescriptorPoolState>(descriptorPool);
    if (gState.descriptorPools.erase(pool) == 0) return;
    destroyDescriptorSetsForPoolLocked(pool);
    delete pool;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkResetDescriptorPool(
    VkDevice,
    VkDescriptorPool descriptorPool,
    VkDescriptorPoolResetFlags
) {
    std::lock_guard lock(gState.mutex);
    auto* pool = objectState<DescriptorPoolState>(descriptorPool);
    if (!gState.descriptorPools.contains(pool)) return VK_ERROR_INITIALIZATION_FAILED;
    destroyDescriptorSetsForPoolLocked(pool);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkAllocateDescriptorSets(
    VkDevice device,
    const VkDescriptorSetAllocateInfo* allocateInfo,
    VkDescriptorSet* descriptorSets
) {
    if (allocateInfo == nullptr || descriptorSets == nullptr || allocateInfo->descriptorSetCount == 0
        || allocateInfo->pSetLayouts == nullptr) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    auto* pool = objectState<DescriptorPoolState>(allocateInfo->descriptorPool);
    if (!validDevice(device) || !gState.descriptorPools.contains(pool)
        || pool->device != device || allocateInfo->descriptorSetCount > pool->maxSets - pool->allocatedSets) {
        return VK_ERROR_OUT_OF_POOL_MEMORY;
    }
    for (std::uint32_t index = 0; index < allocateInfo->descriptorSetCount; ++index) {
        auto* layout = objectState<DescriptorSetLayoutState>(allocateInfo->pSetLayouts[index]);
        if (!gState.descriptorSetLayouts.contains(layout) || layout->device != device) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    for (std::uint32_t index = 0; index < allocateInfo->descriptorSetCount; ++index) {
        auto* layout = objectState<DescriptorSetLayoutState>(allocateInfo->pSetLayouts[index]);
        auto* state = new (std::nothrow) DescriptorSetState{};
        if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
        state->device = device;
        state->pool = pool;
        state->layout = layout;
        gState.descriptorSets.insert(state);
        descriptorSets[index] = makeObjectHandle<VkDescriptorSet>(state);
        ++pool->allocatedSets;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkFreeDescriptorSets(
    VkDevice,
    VkDescriptorPool descriptorPool,
    std::uint32_t descriptorSetCount,
    const VkDescriptorSet* descriptorSets
) {
    if (descriptorSetCount != 0 && descriptorSets == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    auto* pool = objectState<DescriptorPoolState>(descriptorPool);
    if (!gState.descriptorPools.contains(pool)) return VK_ERROR_INITIALIZATION_FAILED;
    for (std::uint32_t index = 0; index < descriptorSetCount; ++index) {
        auto* set = objectState<DescriptorSetState>(descriptorSets[index]);
        if (!gState.descriptorSets.contains(set) || set->pool != pool) return VK_ERROR_INITIALIZATION_FAILED;
    }
    for (std::uint32_t index = 0; index < descriptorSetCount; ++index) {
        auto* set = objectState<DescriptorSetState>(descriptorSets[index]);
        gState.descriptorSets.erase(set);
        delete set;
        --pool->allocatedSets;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkUpdateDescriptorSets(
    VkDevice,
    std::uint32_t descriptorWriteCount,
    const VkWriteDescriptorSet* descriptorWrites,
    std::uint32_t descriptorCopyCount,
    const VkCopyDescriptorSet*
) {
    if (descriptorCopyCount != 0 || (descriptorWriteCount != 0 && descriptorWrites == nullptr)) return;
    std::lock_guard lock(gState.mutex);
    for (std::uint32_t index = 0; index < descriptorWriteCount; ++index) {
        const auto& write = descriptorWrites[index];
        auto* set = objectState<DescriptorSetState>(write.dstSet);
        if (traceEnabled()
            || (rayTracingTraceEnabled()
                && (write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    || write.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR))) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: descriptor write set=%p binding=%u array=%u count=%u type=%d pNext=%p images=%p buffers=%p texels=%p\n",
                static_cast<void*>(set),
                write.dstBinding,
                write.dstArrayElement,
                write.descriptorCount,
                write.descriptorType,
                write.pNext,
                static_cast<const void*>(write.pImageInfo),
                static_cast<const void*>(write.pBufferInfo),
                static_cast<const void*>(write.pTexelBufferView)
            );
            if (write.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
                auto* descriptorNext = reinterpret_cast<const VkBaseInStructure*>(write.pNext);
                std::uint32_t chainIndex = 0;
                while (descriptorNext != nullptr && chainIndex < 8) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: RT acceleration pNext index=%u address=%p sType=%d next=%p\n",
                        chainIndex,
                        static_cast<const void*>(descriptorNext),
                        descriptorNext->sType,
                        static_cast<const void*>(descriptorNext->pNext)
                    );
                    descriptorNext = descriptorNext->pNext;
                    ++chainIndex;
                }
            }
            if (write.pImageInfo != nullptr) {
                for (std::uint32_t descriptorIndex = 0; descriptorIndex < write.descriptorCount; ++descriptorIndex) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: descriptor image index=%u sampler=%p view=%p layout=%d\n",
                        descriptorIndex,
                        reinterpret_cast<void*>(write.pImageInfo[descriptorIndex].sampler),
                        reinterpret_cast<void*>(write.pImageInfo[descriptorIndex].imageView),
                        write.pImageInfo[descriptorIndex].imageLayout
                    );
                }
            }
            if (write.pBufferInfo != nullptr) {
                for (std::uint32_t descriptorIndex = 0; descriptorIndex < write.descriptorCount; ++descriptorIndex) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: descriptor buffer index=%u buffer=%p offset=%llu range=%llu\n",
                        descriptorIndex,
                        reinterpret_cast<void*>(write.pBufferInfo[descriptorIndex].buffer),
                        static_cast<unsigned long long>(write.pBufferInfo[descriptorIndex].offset),
                        static_cast<unsigned long long>(write.pBufferInfo[descriptorIndex].range)
                    );
                }
            }
            if (write.pTexelBufferView != nullptr) {
                for (std::uint32_t descriptorIndex = 0;
                     descriptorIndex < write.descriptorCount;
                     ++descriptorIndex) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: descriptor texel index=%u view=%p\n",
                        descriptorIndex,
                        reinterpret_cast<void*>(write.pTexelBufferView[descriptorIndex])
                    );
                }
            }
        }
        if (gState.descriptorSets.contains(set)
            && (write.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                || write.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
            && write.pImageInfo != nullptr) {
            for (std::uint32_t descriptorIndex = 0; descriptorIndex < write.descriptorCount; ++descriptorIndex) {
                auto* view = objectState<ImageViewState>(write.pImageInfo[descriptorIndex].imageView);
                if (gState.imageViews.contains(view) && view->image != nullptr) {
                    set->sampledImages[write.dstArrayElement + descriptorIndex] = view->image;
                    const std::uint64_t key = (static_cast<std::uint64_t>(write.dstBinding) << 32)
                        | (write.dstArrayElement + descriptorIndex);
                    set->computeImages[key] = {
                        view->image,
                        write.descriptorType,
                    };
                }
            }
        }
        if (gState.descriptorSets.contains(set)
            && write.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER
            && write.pImageInfo != nullptr) {
            for (std::uint32_t descriptorIndex = 0;
                 descriptorIndex < write.descriptorCount;
                 ++descriptorIndex) {
                auto* sampler = objectState<SamplerState>(
                    write.pImageInfo[descriptorIndex].sampler
                );
                if (gState.samplers.contains(sampler) && sampler->device == set->device) {
                    const std::uint64_t key =
                        (static_cast<std::uint64_t>(write.dstBinding) << 32)
                        | (write.dstArrayElement + descriptorIndex);
                    set->computeSamplers[key] = sampler;
                }
            }
        }
        if (gState.descriptorSets.contains(set)
            && write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
            && write.pImageInfo != nullptr) {
            for (std::uint32_t descriptorIndex = 0; descriptorIndex < write.descriptorCount; ++descriptorIndex) {
                auto* view = objectState<ImageViewState>(write.pImageInfo[descriptorIndex].imageView);
                if (gState.imageViews.contains(view) && view->image != nullptr) {
                    const std::uint64_t key = (static_cast<std::uint64_t>(write.dstBinding) << 32)
                        | (write.dstArrayElement + descriptorIndex);
                    set->storageImages[key] = view->image;
                    set->computeImages[key] = {
                        view->image,
                        write.descriptorType,
                    };
                    view->image->storageDescriptorSequence =
                        gState.nextStorageDescriptorSequence++;
                    if (rayTracingTraceEnabled()) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: RT storage image set=%p binding=%u array=%u image=%p extent=%ux%u format=%d host=%llu\n",
                            static_cast<void*>(set),
                            write.dstBinding,
                            write.dstArrayElement + descriptorIndex,
                            static_cast<void*>(view->image),
                            view->image->extent.width,
                            view->image->extent.height,
                            view->image->format,
                            static_cast<unsigned long long>(view->image->resourceID)
                        );
                    }
                }
            }
        }
        if (gState.descriptorSets.contains(set)
            && (write.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                || write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
            && write.pTexelBufferView != nullptr) {
            for (std::uint32_t descriptorIndex = 0;
                 descriptorIndex < write.descriptorCount;
                 ++descriptorIndex) {
                auto* view = objectState<BufferViewState>(
                    write.pTexelBufferView[descriptorIndex]
                );
                if (!gState.bufferViews.contains(view) || view->buffer == nullptr
                    || !gState.buffers.contains(view->buffer)) {
                    continue;
                }
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(write.dstBinding) << 32)
                    | (write.dstArrayElement + descriptorIndex);
                set->computeBuffers[key] = {
                    view->buffer,
                    view->offset,
                    view->range,
                    write.descriptorType,
                    view->format,
                };
            }
        }
        if (gState.descriptorSets.contains(set)
            && (write.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                || write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                || write.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                || write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
            && write.pBufferInfo != nullptr) {
            for (std::uint32_t descriptorIndex = 0; descriptorIndex < write.descriptorCount; ++descriptorIndex) {
                auto* buffer = objectState<BufferState>(
                    write.pBufferInfo[descriptorIndex].buffer
                );
                if (!gState.buffers.contains(buffer)
                    || write.pBufferInfo[descriptorIndex].offset > buffer->size) {
                    continue;
                }
                const VkDeviceSize range =
                    write.pBufferInfo[descriptorIndex].range == VK_WHOLE_SIZE
                    ? buffer->size - write.pBufferInfo[descriptorIndex].offset
                    : write.pBufferInfo[descriptorIndex].range;
                if (range == 0
                    || range > buffer->size - write.pBufferInfo[descriptorIndex].offset) {
                    continue;
                }
                const std::uint64_t key =
                    (static_cast<std::uint64_t>(write.dstBinding) << 32)
                    | (write.dstArrayElement + descriptorIndex);
                set->computeBuffers[key] = {
                    buffer,
                    write.pBufferInfo[descriptorIndex].offset,
                    range,
                    write.descriptorType,
                };
            }
        }
        if (gState.descriptorSets.contains(set)
            && write.descriptorType == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {
            const VkWriteDescriptorSetAccelerationStructureKHR* accelerationWrite = nullptr;
            auto* next = reinterpret_cast<const VkBaseInStructure*>(write.pNext);
            while (next != nullptr) {
                if (next->sType == VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR) {
                    accelerationWrite = reinterpret_cast<const VkWriteDescriptorSetAccelerationStructureKHR*>(next);
                    break;
                }
                next = next->pNext;
            }
            if (accelerationWrite != nullptr
                && accelerationWrite->accelerationStructureCount == write.descriptorCount
                && (write.descriptorCount == 0 || accelerationWrite->pAccelerationStructures != nullptr)) {
                for (std::uint32_t descriptorIndex = 0; descriptorIndex < write.descriptorCount; ++descriptorIndex) {
                    const VkAccelerationStructureKHR rawAcceleration =
                        accelerationWrite->pAccelerationStructures[descriptorIndex];
                    auto* acceleration = objectState<AccelerationStructureKHRState>(
                        rawAcceleration
                    );
                    if (rayTracingTraceEnabled()) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: RT acceleration raw set=%p binding=%u array=%u handle=%p state=%p known=%d\n",
                            static_cast<void*>(set),
                            write.dstBinding,
                            write.dstArrayElement + descriptorIndex,
                            reinterpret_cast<void*>(rawAcceleration),
                            static_cast<void*>(acceleration),
                            gState.accelerationStructuresKHR.contains(acceleration) ? 1 : 0
                        );
                    }
                    if (gState.accelerationStructuresKHR.contains(acceleration)) {
                        const std::uint64_t key = (static_cast<std::uint64_t>(write.dstBinding) << 32)
                            | (write.dstArrayElement + descriptorIndex);
                        set->accelerationStructuresKHR[key] = acceleration;
                        if (rayTracingTraceEnabled()) {
                            std::fprintf(
                                stderr,
                                "imb-vulkan-icd: RT acceleration set=%p binding=%u array=%u as=%p type=%d built=%d host=%llu\n",
                                static_cast<void*>(set),
                                write.dstBinding,
                                write.dstArrayElement + descriptorIndex,
                                static_cast<void*>(acceleration),
                                acceleration->type,
                                acceleration->built ? 1 : 0,
                                static_cast<unsigned long long>(acceleration->bridgeResourceID)
                            );
                        }
                    }
                }
            }
        }
        if (!gState.descriptorSets.contains(set) || write.dstBinding != 0 || write.dstArrayElement != 0
            || write.descriptorCount != 1 || write.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            || write.pBufferInfo == nullptr) {
            continue;
        }
        auto* buffer = objectState<BufferState>(write.pBufferInfo[0].buffer);
        if (!gState.buffers.contains(buffer)) continue;
        set->buffer = buffer;
        set->offset = write.pBufferInfo[0].offset;
        set->range = write.pBufferInfo[0].range == VK_WHOLE_SIZE
            ? buffer->size - set->offset
            : write.pBufferInfo[0].range;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateCommandPool(
    VkDevice device,
    const VkCommandPoolCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkCommandPool* commandPool
) {
    if (createInfo == nullptr || commandPool == nullptr || createInfo->queueFamilyIndex != 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    auto* state = new (std::nothrow) CommandPoolState{device};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    gState.commandPools.insert(state);
    *commandPool = makeObjectHandle<VkCommandPool>(state);
    return VK_SUCCESS;
}

void destroyCommandBuffersForPoolLocked(CommandPoolState* pool) {
    for (auto iterator = gState.commandBuffers.begin(); iterator != gState.commandBuffers.end();) {
        auto* commandBuffer = *iterator;
        if (commandBuffer->pool == pool) {
            iterator = gState.commandBuffers.erase(iterator);
            delete commandBuffer;
        } else {
            ++iterator;
        }
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyCommandPool(
    VkDevice,
    VkCommandPool commandPool,
    const VkAllocationCallbacks*
) {
    if (commandPool == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* pool = objectState<CommandPoolState>(commandPool);
    if (gState.commandPools.erase(pool) == 0) return;
    destroyCommandBuffersForPoolLocked(pool);
    delete pool;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkResetCommandPool(
    VkDevice,
    VkCommandPool commandPool,
    VkCommandPoolResetFlags
) {
    std::lock_guard lock(gState.mutex);
    auto* pool = objectState<CommandPoolState>(commandPool);
    if (!gState.commandPools.contains(pool)) return VK_ERROR_INITIALIZATION_FAILED;
    for (auto* commandBuffer : gState.commandBuffers) {
        if (commandBuffer->pool == pool) {
            commandBuffer->recording = false;
            commandBuffer->executable = false;
            commandBuffer->pipeline = nullptr;
            commandBuffer->descriptorSet = nullptr;
            commandBuffer->hasPushConstant = false;
            commandBuffer->computePushConstants.clear();
            commandBuffer->groupCountX = 0;
            commandBuffer->framebuffer = nullptr;
            commandBuffer->insideRenderPass = false;
            commandBuffer->clearRGBA8 = 0;
            commandBuffer->drewTriangle = false;
            commandBuffer->computeDescriptorSets.fill(nullptr);
            commandBuffer->rayTracingDescriptorSets.fill(nullptr);
            commandBuffer->accelerationStructureBuildsNV.clear();
            commandBuffer->accelerationStructureCopiesNV.clear();
            commandBuffer->rayTracesNV.clear();
            commandBuffer->accelerationStructureBuildsKHR.clear();
            commandBuffer->accelerationStructureCopiesKHR.clear();
            commandBuffer->rayTracesKHR.clear();
            commandBuffer->computeDispatches.clear();
            commandBuffer->nextCommandSequence = 1;
            commandBuffer->bufferCopies.clear();
            commandBuffer->bufferUpdates.clear();
            commandBuffer->bufferFills.clear();
            commandBuffer->imageClears.clear();
            commandBuffer->imageCopies.clear();
            commandBuffer->bufferToImageCopies.clear();
            commandBuffer->imageToBufferCopies.clear();
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkAllocateCommandBuffers(
    VkDevice device,
    const VkCommandBufferAllocateInfo* allocateInfo,
    VkCommandBuffer* commandBuffers
) {
    if (allocateInfo == nullptr || commandBuffers == nullptr || allocateInfo->commandBufferCount == 0
        || allocateInfo->level != VK_COMMAND_BUFFER_LEVEL_PRIMARY) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    auto* pool = objectState<CommandPoolState>(allocateInfo->commandPool);
    if (!validDevice(device) || !gState.commandPools.contains(pool) || pool->device != device) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    for (std::uint32_t index = 0; index < allocateInfo->commandBufferCount; ++index) {
        auto* state = new (std::nothrow) CommandBufferState{};
        if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
        set_loader_magic_value(&state->loaderData);
        state->device = device;
        state->pool = pool;
        gState.commandBuffers.insert(state);
        commandBuffers[index] = reinterpret_cast<VkCommandBuffer>(state);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkFreeCommandBuffers(
    VkDevice,
    VkCommandPool commandPool,
    std::uint32_t commandBufferCount,
    const VkCommandBuffer* commandBuffers
) {
    if (commandBufferCount != 0 && commandBuffers == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* pool = objectState<CommandPoolState>(commandPool);
    if (!gState.commandPools.contains(pool)) return;
    for (std::uint32_t index = 0; index < commandBufferCount; ++index) {
        auto* state = reinterpret_cast<CommandBufferState*>(commandBuffers[index]);
        if (gState.commandBuffers.contains(state) && state->pool == pool) {
            gState.commandBuffers.erase(state);
            delete state;
        }
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkBeginCommandBuffer(
    VkCommandBuffer commandBuffer,
    const VkCommandBufferBeginInfo* beginInfo
) {
    if (commandBuffer == VK_NULL_HANDLE || beginInfo == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    auto* state = reinterpret_cast<CommandBufferState*>(commandBuffer);
    if (!gState.commandBuffers.contains(state) || state->recording) return VK_ERROR_INITIALIZATION_FAILED;
    state->recording = true;
    state->executable = false;
    state->pipeline = nullptr;
    state->descriptorSet = nullptr;
    state->hasPushConstant = false;
    state->computePushConstants.clear();
    state->groupCountX = 0;
    state->framebuffer = nullptr;
    state->insideRenderPass = false;
    state->clearRGBA8 = 0;
    state->drewTriangle = false;
    state->graphicsDescriptorSets.fill(nullptr);
    state->computeDescriptorSets.fill(nullptr);
    state->rayTracingDescriptorSets.fill(nullptr);
    state->vertexBuffer = nullptr;
    state->vertexBufferOffset = 0;
    state->indexBuffer = nullptr;
    state->indexBufferOffset = 0;
    state->indexType = VK_INDEX_TYPE_UINT16;
    state->textureIndex = 0;
    state->scissor = {};
    state->uiDraws.clear();
    state->accelerationStructureBuildsNV.clear();
    state->accelerationStructureCopiesNV.clear();
    state->rayTracesNV.clear();
    state->accelerationStructureBuildsKHR.clear();
    state->accelerationStructureCopiesKHR.clear();
    state->rayTracesKHR.clear();
    state->computeDispatches.clear();
    state->nextCommandSequence = 1;
    state->bufferCopies.clear();
    state->bufferUpdates.clear();
    state->bufferFills.clear();
    state->imageClears.clear();
    state->imageCopies.clear();
    state->bufferToImageCopies.clear();
    state->imageToBufferCopies.clear();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkEndCommandBuffer(VkCommandBuffer commandBuffer) {
    std::lock_guard lock(gState.mutex);
    auto* state = reinterpret_cast<CommandBufferState*>(commandBuffer);
    if (!gState.commandBuffers.contains(state) || !state->recording) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    state->recording = false;
    state->executable = true;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkResetCommandBuffer(
    VkCommandBuffer commandBuffer,
    VkCommandBufferResetFlags
) {
    std::lock_guard lock(gState.mutex);
    auto* state = reinterpret_cast<CommandBufferState*>(commandBuffer);
    if (!gState.commandBuffers.contains(state)) return VK_ERROR_INITIALIZATION_FAILED;
    state->recording = false;
    state->executable = false;
    state->pipeline = nullptr;
    state->descriptorSet = nullptr;
    state->hasPushConstant = false;
    state->computePushConstants.clear();
    state->groupCountX = 0;
    state->framebuffer = nullptr;
    state->insideRenderPass = false;
    state->clearRGBA8 = 0;
    state->drewTriangle = false;
    state->graphicsDescriptorSets.fill(nullptr);
    state->computeDescriptorSets.fill(nullptr);
    state->rayTracingDescriptorSets.fill(nullptr);
    state->vertexBuffer = nullptr;
    state->vertexBufferOffset = 0;
    state->indexBuffer = nullptr;
    state->indexBufferOffset = 0;
    state->indexType = VK_INDEX_TYPE_UINT16;
    state->textureIndex = 0;
    state->scissor = {};
    state->uiDraws.clear();
    state->accelerationStructureBuildsNV.clear();
    state->accelerationStructureCopiesNV.clear();
    state->rayTracesNV.clear();
    state->accelerationStructureBuildsKHR.clear();
    state->accelerationStructureCopiesKHR.clear();
    state->rayTracesKHR.clear();
    state->computeDispatches.clear();
    state->nextCommandSequence = 1;
    state->bufferCopies.clear();
    state->bufferUpdates.clear();
    state->bufferFills.clear();
    state->imageClears.clear();
    state->imageCopies.clear();
    state->bufferToImageCopies.clear();
    state->imageToBufferCopies.clear();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdBindPipeline(
    VkCommandBuffer commandBuffer,
    VkPipelineBindPoint pipelineBindPoint,
    VkPipeline pipeline
) {
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* pipelineState = objectState<PipelineState>(pipeline);
    if (gState.commandBuffers.contains(command) && command->recording
        && (pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE
            || pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS
            || pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_NV)
        && gState.pipelines.contains(pipelineState)) {
        command->pipeline = pipelineState;
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdBindDescriptorSets(
    VkCommandBuffer commandBuffer,
    VkPipelineBindPoint pipelineBindPoint,
    VkPipelineLayout layout,
    std::uint32_t firstSet,
    std::uint32_t descriptorSetCount,
    const VkDescriptorSet* descriptorSets,
    std::uint32_t dynamicOffsetCount,
    const std::uint32_t*
) {
    if (descriptorSets == nullptr) return;
    if (traceEnabled()
        || (rayTracingTraceEnabled()
            && pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: bind descriptors command=%p point=%d layout=%p first=%u count=%u dynamic=%u",
            static_cast<void*>(commandBuffer),
            pipelineBindPoint,
            reinterpret_cast<void*>(layout),
            firstSet,
            descriptorSetCount,
            dynamicOffsetCount
        );
        for (std::uint32_t index = 0; index < descriptorSetCount; ++index) {
            std::fprintf(stderr, " set[%u]=%p", index, reinterpret_cast<void*>(descriptorSets[index]));
        }
        std::fputc('\n', stderr);
    }
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* layoutState = objectState<PipelineLayoutState>(layout);
    auto* set = objectState<DescriptorSetState>(descriptorSets[0]);
    if (gState.commandBuffers.contains(command) && command->recording
        && pipelineBindPoint == VK_PIPELINE_BIND_POINT_GRAPHICS
        && gState.pipelineLayouts.contains(layoutState)
        && firstSet <= command->graphicsDescriptorSets.size()
        && descriptorSetCount <= command->graphicsDescriptorSets.size() - firstSet) {
        bool valid = true;
        for (std::uint32_t index = 0; index < descriptorSetCount; ++index) {
            auto* graphicsSet = objectState<DescriptorSetState>(descriptorSets[index]);
            if (!gState.descriptorSets.contains(graphicsSet)) {
                valid = false;
                break;
            }
            command->graphicsDescriptorSets[firstSet + index] = graphicsSet;
        }
        if (!valid) command->graphicsDescriptorSets.fill(nullptr);
    }
    if (gState.commandBuffers.contains(command) && command->recording
        && pipelineBindPoint == VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR
        && gState.pipelineLayouts.contains(layoutState)
        && firstSet <= command->rayTracingDescriptorSets.size()
        && descriptorSetCount <= command->rayTracingDescriptorSets.size() - firstSet
        && dynamicOffsetCount == 0) {
        bool valid = true;
        for (std::uint32_t index = 0; index < descriptorSetCount; ++index) {
            auto* raySet = objectState<DescriptorSetState>(descriptorSets[index]);
            if (!gState.descriptorSets.contains(raySet)) {
                valid = false;
                break;
            }
            command->rayTracingDescriptorSets[firstSet + index] = raySet;
        }
        if (!valid) command->rayTracingDescriptorSets.fill(nullptr);
    }
    if (gState.commandBuffers.contains(command) && command->recording
        && pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE
        && gState.pipelineLayouts.contains(layoutState)
        && firstSet <= command->computeDescriptorSets.size()
        && descriptorSetCount <= command->computeDescriptorSets.size() - firstSet
        && dynamicOffsetCount == 0) {
        bool valid = true;
        for (std::uint32_t index = 0; index < descriptorSetCount; ++index) {
            auto* computeSet = objectState<DescriptorSetState>(descriptorSets[index]);
            if (!gState.descriptorSets.contains(computeSet)) {
                valid = false;
                break;
            }
            command->computeDescriptorSets[firstSet + index] = computeSet;
        }
        if (!valid) command->computeDescriptorSets.fill(nullptr);
    }
    if (gState.commandBuffers.contains(command) && command->recording
        && pipelineBindPoint == VK_PIPELINE_BIND_POINT_COMPUTE && firstSet == 0
        && descriptorSetCount == 1 && dynamicOffsetCount == 0
        && gState.pipelineLayouts.contains(layoutState) && gState.descriptorSets.contains(set)
        && !layoutState->setLayouts.empty() && set->layout == layoutState->setLayouts[0]) {
        command->descriptorSet = set;
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdPushConstants(
    VkCommandBuffer commandBuffer,
    VkPipelineLayout layout,
    VkShaderStageFlags stageFlags,
    std::uint32_t offset,
    std::uint32_t size,
    const void* values
) {
    if (values == nullptr) return;
    if (traceEnabled()) {
        std::uint32_t firstValue = 0;
        if (size >= sizeof(firstValue)) std::memcpy(&firstValue, values, sizeof(firstValue));
        std::fprintf(
            stderr,
            "imb-vulkan-icd: push constants command=%p layout=%p stages=%#x offset=%u size=%u first=%#x\n",
            static_cast<void*>(commandBuffer),
            reinterpret_cast<void*>(layout),
            stageFlags,
            offset,
            size,
            firstValue
        );
    }
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* layoutState = objectState<PipelineLayoutState>(layout);
    if (gState.commandBuffers.contains(command) && command->recording
        && gState.pipelineLayouts.contains(layoutState)
        && (stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) != 0
        && size > 0 && offset <= 4096 && size <= 4096 - offset) {
        try {
            if (command->computePushConstants.size() < offset + size) {
                command->computePushConstants.resize(offset + size, 0);
            }
            std::memcpy(
                command->computePushConstants.data() + offset,
                values,
                size
            );
        } catch (const std::bad_alloc&) {
            command->computePushConstants.clear();
        }
    }
    if (gState.commandBuffers.contains(command) && command->recording
        && gState.pipelineLayouts.contains(layoutState) && stageFlags == VK_SHADER_STAGE_COMPUTE_BIT
        && offset == 0 && size == sizeof(std::uint32_t)) {
        std::memcpy(&command->addend, values, sizeof(command->addend));
        command->hasPushConstant = true;
    }
    if (gState.commandBuffers.contains(command) && command->recording
        && gState.pipelineLayouts.contains(layoutState)
        && (stageFlags & VK_SHADER_STAGE_FRAGMENT_BIT) != 0
        && offset == 0 && size == sizeof(std::uint32_t)) {
        std::memcpy(&command->textureIndex, values, sizeof(command->textureIndex));
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdDispatch(
    VkCommandBuffer commandBuffer,
    std::uint32_t groupCountX,
    std::uint32_t groupCountY,
    std::uint32_t groupCountZ
) {
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || groupCountX == 0 || groupCountY == 0 || groupCountZ == 0) {
        return;
    }
    if (groupCountY == 1 && groupCountZ == 1) {
        command->groupCountX = groupCountX;
    }
    std::size_t requiredPushConstantSize = 0;
    if (command->pipeline != nullptr && command->pipeline->layout != nullptr) {
        for (const auto& range : command->pipeline->layout->pushConstantRanges) {
            if ((range.stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) != 0) {
                requiredPushConstantSize = std::max<std::size_t>(
                    requiredPushConstantSize,
                    static_cast<std::size_t>(range.offset) + range.size
                );
            }
        }
    }
    if (genericComputeBridgeEnabled() && command->pipeline != nullptr
        && gState.pipelines.contains(command->pipeline)
        && command->pipeline->layout != nullptr) {
        const bool bridgeEligible =
            command->pipeline->bridgeComputePipelineID != 0
            && (!(command->pipeline->usesFloat64Matrix
                    && command->pipeline->requiresSoftwareFloat64Execution)
                || validatedSoftwareFloat64MatrixShader(
                    command->pipeline->computeHash
                )
                || float64MatrixExecutionEnabled())
            && command->computePushConstants.size() >= requiredPushConstantSize;
        command->computeDispatches.push_back(RecordedComputeDispatch{
            command->nextCommandSequence++,
            command->pipeline,
            bridgeEligible,
            command->computeDescriptorSets,
            groupCountX,
            groupCountY,
            groupCountZ,
            command->computePushConstants,
        });
        if (traceEnabled() || computeTraceEnabled() || rayTracingTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: recorded compute dispatch hash=%016llx pipeline=%llu eligible=%d groups=%ux%ux%u sequence=%llu\n",
                static_cast<unsigned long long>(
                    command->pipeline->computeHash
                ),
                static_cast<unsigned long long>(
                    command->pipeline->bridgeComputePipelineID
                ),
                bridgeEligible ? 1 : 0,
                groupCountX,
                groupCountY,
                groupCountZ,
                static_cast<unsigned long long>(
                    command->computeDispatches.back().sequence
                )
            );
        }
    }
}

std::uint32_t packClearRGBA8(const VkClearColorValue& color);

VKAPI_ATTR void VKAPI_CALL imb_vkCmdPipelineBarrier(
    VkCommandBuffer commandBuffer,
    VkPipelineStageFlags sourceStageMask,
    VkPipelineStageFlags destinationStageMask,
    VkDependencyFlags dependencyFlags,
    std::uint32_t memoryBarrierCount,
    const VkMemoryBarrier*,
    std::uint32_t bufferMemoryBarrierCount,
    const VkBufferMemoryBarrier*,
    std::uint32_t imageMemoryBarrierCount,
    const VkImageMemoryBarrier* imageMemoryBarriers
) {
    if (!traceEnabled()) return;
    std::fprintf(
        stderr,
        "imb-vulkan-icd: pipeline barrier command=%p stages=%#x->%#x dependency=%#x memory=%u buffers=%u images=%u\n",
        static_cast<void*>(commandBuffer),
        sourceStageMask,
        destinationStageMask,
        dependencyFlags,
        memoryBarrierCount,
        bufferMemoryBarrierCount,
        imageMemoryBarrierCount
    );
    for (std::uint32_t index = 0; index < imageMemoryBarrierCount; ++index) {
        const auto& barrier = imageMemoryBarriers[index];
        std::fprintf(
            stderr,
            "imb-vulkan-icd: image barrier image=%p layout=%d->%d access=%#x->%#x mip=%u+%u layer=%u+%u\n",
            reinterpret_cast<void*>(barrier.image),
            barrier.oldLayout,
            barrier.newLayout,
            barrier.srcAccessMask,
            barrier.dstAccessMask,
            barrier.subresourceRange.baseMipLevel,
            barrier.subresourceRange.levelCount,
            barrier.subresourceRange.baseArrayLayer,
            barrier.subresourceRange.layerCount
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyBuffer(
    VkCommandBuffer commandBuffer,
    VkBuffer sourceBuffer,
    VkBuffer destinationBuffer,
    std::uint32_t regionCount,
    const VkBufferCopy* regions
) {
    if (regionCount == 0 || regions == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* source = objectState<BufferState>(sourceBuffer);
    auto* destination = objectState<BufferState>(destinationBuffer);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.buffers.contains(source) || !gState.buffers.contains(destination)
        || source->device != command->device || destination->device != command->device) {
        return;
    }
    for (std::uint32_t index = 0; index < regionCount; ++index) {
        if (regions[index].size == 0 || regions[index].srcOffset > source->size
            || regions[index].size > source->size - regions[index].srcOffset
            || regions[index].dstOffset > destination->size
            || regions[index].size > destination->size - regions[index].dstOffset) {
            return;
        }
    }
    try {
        RecordedBufferCopy copy;
        copy.sequence = command->nextCommandSequence++;
        copy.source = source;
        copy.destination = destination;
        copy.regions.assign(regions, regions + regionCount);
        command->bufferCopies.push_back(std::move(copy));
        if (rayTracingTraceEnabled()
            && ((source->flags | destination->flags) & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) != 0) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: record sparse buffer copy command=%p source=%p destination=%p regions=%u first=%llu->%llu bytes=%llu\n",
                static_cast<void*>(command),
                static_cast<void*>(source),
                static_cast<void*>(destination),
                regionCount,
                static_cast<unsigned long long>(regions[0].srcOffset),
                static_cast<unsigned long long>(regions[0].dstOffset),
                static_cast<unsigned long long>(regions[0].size)
            );
        }
    } catch (const std::bad_alloc&) {
        command->bufferCopies.clear();
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyBuffer2(
    VkCommandBuffer commandBuffer,
    const VkCopyBufferInfo2* copyInfo
) {
    if (copyInfo == nullptr || copyInfo->regionCount == 0 || copyInfo->pRegions == nullptr) return;
    std::vector<VkBufferCopy> regions;
    try {
        regions.reserve(copyInfo->regionCount);
        for (std::uint32_t index = 0; index < copyInfo->regionCount; ++index) {
            regions.push_back({
                copyInfo->pRegions[index].srcOffset,
                copyInfo->pRegions[index].dstOffset,
                copyInfo->pRegions[index].size,
            });
        }
    } catch (const std::bad_alloc&) {
        return;
    }
    imb_vkCmdCopyBuffer(
        commandBuffer,
        copyInfo->srcBuffer,
        copyInfo->dstBuffer,
        static_cast<std::uint32_t>(regions.size()),
        regions.data()
    );
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdUpdateBuffer(
    VkCommandBuffer commandBuffer,
    VkBuffer destinationBuffer,
    VkDeviceSize destinationOffset,
    VkDeviceSize dataSize,
    const void* data
) {
    if (data == nullptr || dataSize == 0 || dataSize > 65536
        || (destinationOffset % 4) != 0 || (dataSize % 4) != 0) {
        return;
    }
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* destination = objectState<BufferState>(destinationBuffer);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.buffers.contains(destination) || destination->device != command->device
        || destinationOffset > destination->size || dataSize > destination->size - destinationOffset) {
        return;
    }
    try {
        RecordedBufferUpdate update;
        update.sequence = command->nextCommandSequence++;
        update.destination = destination;
        update.offset = destinationOffset;
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        update.data.assign(bytes, bytes + static_cast<std::size_t>(dataSize));
        command->bufferUpdates.push_back(std::move(update));
    } catch (const std::bad_alloc&) {
        command->bufferUpdates.clear();
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdFillBuffer(
    VkCommandBuffer commandBuffer,
    VkBuffer destinationBuffer,
    VkDeviceSize destinationOffset,
    VkDeviceSize size,
    std::uint32_t data
) {
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* destination = objectState<BufferState>(destinationBuffer);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.buffers.contains(destination) || destination->device != command->device
        || destinationOffset >= destination->size || (destinationOffset % 4) != 0) {
        return;
    }
    const VkDeviceSize fillSize =
        size == VK_WHOLE_SIZE ? destination->size - destinationOffset : size;
    if (fillSize == 0 || fillSize > destination->size - destinationOffset
        || (fillSize % 4) != 0) {
        return;
    }
    try {
        command->bufferFills.push_back({
            command->nextCommandSequence++,
            destination,
            destinationOffset,
            fillSize,
            data,
        });
    } catch (const std::bad_alloc&) {
        command->bufferFills.clear();
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdClearColorImage(
    VkCommandBuffer commandBuffer,
    VkImage image,
    VkImageLayout,
    const VkClearColorValue* color,
    std::uint32_t rangeCount,
    const VkImageSubresourceRange* ranges
) {
    if (color == nullptr || rangeCount == 0 || ranges == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* destination = objectState<ImageState>(image);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.images.contains(destination) || destination->device != command->device
        || destination->samples != VK_SAMPLE_COUNT_1_BIT
        || (formatAspectMask(destination->format) & VK_IMAGE_ASPECT_COLOR_BIT) == 0) {
        return;
    }
    for (std::uint32_t index = 0; index < rangeCount; ++index) {
        const auto& range = ranges[index];
        const std::uint32_t levelCount = range.levelCount == VK_REMAINING_MIP_LEVELS
            ? destination->mipLevels - std::min(range.baseMipLevel, destination->mipLevels)
            : range.levelCount;
        const std::uint32_t layerCount = range.layerCount == VK_REMAINING_ARRAY_LAYERS
            ? destination->arrayLayers
                - std::min(range.baseArrayLayer, destination->arrayLayers)
            : range.layerCount;
        if (range.aspectMask != VK_IMAGE_ASPECT_COLOR_BIT || levelCount == 0
            || layerCount == 0 || range.baseMipLevel >= destination->mipLevels
            || levelCount > destination->mipLevels - range.baseMipLevel
            || range.baseArrayLayer >= destination->arrayLayers
            || layerCount > destination->arrayLayers - range.baseArrayLayer) {
            return;
        }
    }
    try {
        RecordedImageClear clear;
        clear.sequence = command->nextCommandSequence++;
        clear.image = destination;
        clear.color = *color;
        clear.ranges.assign(ranges, ranges + rangeCount);
        command->imageClears.push_back(std::move(clear));
    } catch (const std::bad_alloc&) {
        command->imageClears.clear();
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyImage(
    VkCommandBuffer commandBuffer,
    VkImage sourceImage,
    VkImageLayout,
    VkImage destinationImage,
    VkImageLayout,
    std::uint32_t regionCount,
    const VkImageCopy* regions
) {
    if (regionCount == 0 || regions == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* source = objectState<ImageState>(sourceImage);
    auto* destination = objectState<ImageState>(destinationImage);
    const auto sourceBlock = gState.images.contains(source)
        ? formatBlockInfo(source->format)
        : FormatBlockInfo{};
    const auto destinationBlock = gState.images.contains(destination)
        ? formatBlockInfo(destination->format)
        : FormatBlockInfo{};
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.images.contains(source) || !gState.images.contains(destination)
        || source->device != command->device || destination->device != command->device
        || source->samples != destination->samples
        || sourceBlock.width != destinationBlock.width
        || sourceBlock.height != destinationBlock.height
        || sourceBlock.depth != destinationBlock.depth
        || sourceBlock.bytes != destinationBlock.bytes || sourceBlock.bytes == 0) {
        return;
    }
    for (std::uint32_t index = 0; index < regionCount; ++index) {
        const auto& region = regions[index];
        if (region.srcSubresource.aspectMask != formatAspectMask(source->format)
            || region.dstSubresource.aspectMask != formatAspectMask(destination->format)
            || region.srcSubresource.mipLevel >= source->mipLevels
            || region.dstSubresource.mipLevel >= destination->mipLevels
            || region.srcSubresource.layerCount == 0
            || region.srcSubresource.layerCount != region.dstSubresource.layerCount
            || region.srcSubresource.baseArrayLayer >= source->arrayLayers
            || region.srcSubresource.layerCount
                > source->arrayLayers - region.srcSubresource.baseArrayLayer
            || region.dstSubresource.baseArrayLayer >= destination->arrayLayers
            || region.dstSubresource.layerCount
                > destination->arrayLayers - region.dstSubresource.baseArrayLayer
            || region.srcOffset.x < 0 || region.srcOffset.y < 0 || region.srcOffset.z < 0
            || region.dstOffset.x < 0 || region.dstOffset.y < 0 || region.dstOffset.z < 0
            || region.extent.width == 0 || region.extent.height == 0
            || region.extent.depth == 0) {
            return;
        }
        const VkExtent3D sourceExtent{
            std::max(1U, source->extent.width >> region.srcSubresource.mipLevel),
            std::max(1U, source->extent.height >> region.srcSubresource.mipLevel),
            std::max(1U, source->extent.depth >> region.srcSubresource.mipLevel),
        };
        const VkExtent3D destinationExtent{
            std::max(1U, destination->extent.width >> region.dstSubresource.mipLevel),
            std::max(1U, destination->extent.height >> region.dstSubresource.mipLevel),
            std::max(1U, destination->extent.depth >> region.dstSubresource.mipLevel),
        };
        if (static_cast<std::uint32_t>(region.srcOffset.x) > sourceExtent.width
            || region.extent.width
                > sourceExtent.width - static_cast<std::uint32_t>(region.srcOffset.x)
            || static_cast<std::uint32_t>(region.srcOffset.y) > sourceExtent.height
            || region.extent.height
                > sourceExtent.height - static_cast<std::uint32_t>(region.srcOffset.y)
            || static_cast<std::uint32_t>(region.srcOffset.z) > sourceExtent.depth
            || region.extent.depth
                > sourceExtent.depth - static_cast<std::uint32_t>(region.srcOffset.z)
            || static_cast<std::uint32_t>(region.dstOffset.x) > destinationExtent.width
            || region.extent.width
                > destinationExtent.width - static_cast<std::uint32_t>(region.dstOffset.x)
            || static_cast<std::uint32_t>(region.dstOffset.y) > destinationExtent.height
            || region.extent.height
                > destinationExtent.height - static_cast<std::uint32_t>(region.dstOffset.y)
            || static_cast<std::uint32_t>(region.dstOffset.z) > destinationExtent.depth
            || region.extent.depth
                > destinationExtent.depth - static_cast<std::uint32_t>(region.dstOffset.z)
            || (static_cast<std::uint32_t>(region.srcOffset.x) % sourceBlock.width) != 0
            || (static_cast<std::uint32_t>(region.srcOffset.y) % sourceBlock.height) != 0
            || (static_cast<std::uint32_t>(region.srcOffset.z) % sourceBlock.depth) != 0
            || (static_cast<std::uint32_t>(region.dstOffset.x) % sourceBlock.width) != 0
            || (static_cast<std::uint32_t>(region.dstOffset.y) % sourceBlock.height) != 0
            || (static_cast<std::uint32_t>(region.dstOffset.z) % sourceBlock.depth) != 0) {
            return;
        }
    }
    try {
        RecordedImageCopy copy;
        copy.sequence = command->nextCommandSequence++;
        copy.source = source;
        copy.destination = destination;
        copy.regions.assign(regions, regions + regionCount);
        command->imageCopies.push_back(std::move(copy));
    } catch (const std::bad_alloc&) {
        command->imageCopies.clear();
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyImage2(
    VkCommandBuffer commandBuffer,
    const VkCopyImageInfo2* copyInfo
) {
    if (copyInfo == nullptr || copyInfo->regionCount == 0 || copyInfo->pRegions == nullptr) {
        return;
    }
    std::vector<VkImageCopy> regions;
    try {
        regions.reserve(copyInfo->regionCount);
        for (std::uint32_t index = 0; index < copyInfo->regionCount; ++index) {
            const auto& source = copyInfo->pRegions[index];
            regions.push_back({
                source.srcSubresource,
                source.srcOffset,
                source.dstSubresource,
                source.dstOffset,
                source.extent,
            });
        }
    } catch (const std::bad_alloc&) {
        return;
    }
    imb_vkCmdCopyImage(
        commandBuffer,
        copyInfo->srcImage,
        copyInfo->srcImageLayout,
        copyInfo->dstImage,
        copyInfo->dstImageLayout,
        static_cast<std::uint32_t>(regions.size()),
        regions.data()
    );
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyBufferToImage(
    VkCommandBuffer commandBuffer,
    VkBuffer sourceBuffer,
    VkImage destinationImage,
    VkImageLayout destinationImageLayout,
    std::uint32_t regionCount,
    const VkBufferImageCopy* regions
) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: copy buffer-to-image command=%p buffer=%p image=%p layout=%d regions=%u\n",
            static_cast<void*>(commandBuffer),
            reinterpret_cast<void*>(sourceBuffer),
            reinterpret_cast<void*>(destinationImage),
            destinationImageLayout,
            regionCount
        );
        for (std::uint32_t index = 0; index < regionCount; ++index) {
            const auto& region = regions[index];
            std::fprintf(
                stderr,
                "imb-vulkan-icd: copy region=%u offset=%llu row=%u height=%u mip=%u layer=%u+%u imageOffset=%d,%d,%d extent=%ux%ux%u\n",
                index,
                static_cast<unsigned long long>(region.bufferOffset),
                region.bufferRowLength,
                region.bufferImageHeight,
                region.imageSubresource.mipLevel,
                region.imageSubresource.baseArrayLayer,
                region.imageSubresource.layerCount,
                region.imageOffset.x,
                region.imageOffset.y,
                region.imageOffset.z,
                region.imageExtent.width,
                region.imageExtent.height,
                region.imageExtent.depth
            );
        }
    }
    if (regionCount == 0 || regions == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* source = objectState<BufferState>(sourceBuffer);
    auto* destination = objectState<ImageState>(destinationImage);
    const auto block = gState.images.contains(destination)
        ? formatBlockInfo(destination->format)
        : FormatBlockInfo{};
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.buffers.contains(source) || !gState.images.contains(destination)
        || source->device != command->device || destination->device != command->device
        || source->memory == nullptr
        || (destination->memory == nullptr
            && (destination->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) == 0)
        || destination->samples != VK_SAMPLE_COUNT_1_BIT
        || block.width == 0 || block.height == 0 || block.depth == 0
        || block.bytes == 0) {
        return;
    }
    try {
        RecordedBufferToImageCopy copy;
        copy.sequence = command->nextCommandSequence++;
        copy.source = source;
        copy.destination = destination;
        copy.regions.assign(regions, regions + regionCount);
        command->bufferToImageCopies.push_back(std::move(copy));
    } catch (const std::bad_alloc&) {
        command->bufferToImageCopies.clear();
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyImageToBuffer(
    VkCommandBuffer commandBuffer,
    VkImage sourceImage,
    VkImageLayout sourceImageLayout,
    VkBuffer destinationBuffer,
    std::uint32_t regionCount,
    const VkBufferImageCopy* regions
) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: copy image-to-buffer command=%p image=%p layout=%d buffer=%p regions=%u\n",
            static_cast<void*>(commandBuffer),
            reinterpret_cast<void*>(sourceImage),
            sourceImageLayout,
            reinterpret_cast<void*>(destinationBuffer),
            regionCount
        );
    }
    if (regionCount == 0 || regions == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* source = objectState<ImageState>(sourceImage);
    auto* destination = objectState<BufferState>(destinationBuffer);
    const auto block = gState.images.contains(source)
        ? formatBlockInfo(source->format)
        : FormatBlockInfo{};
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.images.contains(source) || !gState.buffers.contains(destination)
        || source->device != command->device
        || destination->device != command->device
        || (source->memory == nullptr
            && (source->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) == 0)
        || destination->memory == nullptr
        || source->samples != VK_SAMPLE_COUNT_1_BIT
        || block.width == 0 || block.height == 0
        || block.depth == 0 || block.bytes == 0) {
        return;
    }
    try {
        RecordedImageToBufferCopy copy;
        copy.sequence = command->nextCommandSequence++;
        copy.source = source;
        copy.destination = destination;
        copy.regions.assign(regions, regions + regionCount);
        command->imageToBufferCopies.push_back(std::move(copy));
    } catch (const std::bad_alloc&) {
        command->imageToBufferCopies.clear();
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyImageToBuffer2(
    VkCommandBuffer commandBuffer,
    const VkCopyImageToBufferInfo2* copyInfo
) {
    if (copyInfo == nullptr || copyInfo->regionCount == 0 || copyInfo->pRegions == nullptr) {
        return;
    }
    std::vector<VkBufferImageCopy> regions;
    try {
        regions.reserve(copyInfo->regionCount);
        for (std::uint32_t index = 0; index < copyInfo->regionCount; ++index) {
            const auto& source = copyInfo->pRegions[index];
            VkBufferImageCopy destination{};
            destination.bufferOffset = source.bufferOffset;
            destination.bufferRowLength = source.bufferRowLength;
            destination.bufferImageHeight = source.bufferImageHeight;
            destination.imageSubresource = source.imageSubresource;
            destination.imageOffset = source.imageOffset;
            destination.imageExtent = source.imageExtent;
            regions.push_back(destination);
        }
    } catch (const std::bad_alloc&) {
        return;
    }
    imb_vkCmdCopyImageToBuffer(
        commandBuffer,
        copyInfo->srcImage,
        copyInfo->srcImageLayout,
        copyInfo->dstBuffer,
        static_cast<std::uint32_t>(regions.size()),
        regions.data()
    );
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdSetViewport(
    VkCommandBuffer commandBuffer,
    std::uint32_t firstViewport,
    std::uint32_t viewportCount,
    const VkViewport* viewports
) {
    if (!traceEnabled()) return;
    for (std::uint32_t index = 0; index < viewportCount; ++index) {
        const auto& viewport = viewports[index];
        std::fprintf(
            stderr,
            "imb-vulkan-icd: viewport command=%p index=%u first=%u xy=%g,%g wh=%g,%g depth=%g,%g\n",
            static_cast<void*>(commandBuffer),
            index,
            firstViewport,
            static_cast<double>(viewport.x),
            static_cast<double>(viewport.y),
            static_cast<double>(viewport.width),
            static_cast<double>(viewport.height),
            static_cast<double>(viewport.minDepth),
            static_cast<double>(viewport.maxDepth)
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdSetScissor(
    VkCommandBuffer commandBuffer,
    std::uint32_t firstScissor,
    std::uint32_t scissorCount,
    const VkRect2D* scissors
) {
    if (traceEnabled()) for (std::uint32_t index = 0; index < scissorCount; ++index) {
        const auto& scissor = scissors[index];
        std::fprintf(
            stderr,
            "imb-vulkan-icd: scissor command=%p index=%u first=%u xy=%d,%d wh=%u,%u\n",
            static_cast<void*>(commandBuffer),
            index,
            firstScissor,
            scissor.offset.x,
            scissor.offset.y,
            scissor.extent.width,
            scissor.extent.height
        );
    }
    if (scissorCount == 0 || scissors == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    if (gState.commandBuffers.contains(command) && command->recording && firstScissor == 0) {
        command->scissor = scissors[0];
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdSetDepthBounds(
    VkCommandBuffer commandBuffer,
    float minimumDepthBounds,
    float maximumDepthBounds
) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: depth bounds command=%p min=%g max=%g\n",
            static_cast<void*>(commandBuffer),
            static_cast<double>(minimumDepthBounds),
            static_cast<double>(maximumDepthBounds)
        );
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdClearAttachments(
    VkCommandBuffer commandBuffer,
    std::uint32_t attachmentCount,
    const VkClearAttachment* attachments,
    std::uint32_t rectangleCount,
    const VkClearRect* rectangles
) {
    if (traceEnabled()) std::fprintf(
        stderr,
        "imb-vulkan-icd: clear attachments command=%p attachments=%u rectangles=%u\n",
        static_cast<void*>(commandBuffer),
        attachmentCount,
        rectangleCount
    );
    if (traceEnabled()) for (std::uint32_t index = 0; index < attachmentCount; ++index) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: clear attachment=%u aspect=%#x color=%g,%g,%g,%g\n",
            index,
            attachments[index].aspectMask,
            static_cast<double>(attachments[index].clearValue.color.float32[0]),
            static_cast<double>(attachments[index].clearValue.color.float32[1]),
            static_cast<double>(attachments[index].clearValue.color.float32[2]),
            static_cast<double>(attachments[index].clearValue.color.float32[3])
        );
    }
    if (traceEnabled()) for (std::uint32_t index = 0; index < rectangleCount; ++index) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: clear rect=%u xy=%d,%d wh=%u,%u layer=%u+%u\n",
            index,
            rectangles[index].rect.offset.x,
            rectangles[index].rect.offset.y,
            rectangles[index].rect.extent.width,
            rectangles[index].rect.extent.height,
            rectangles[index].baseArrayLayer,
            rectangles[index].layerCount
        );
    }
    if (attachmentCount == 0 || attachments == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    if (gState.commandBuffers.contains(command) && command->recording && command->insideRenderPass
        && (attachments[0].aspectMask & VK_IMAGE_ASPECT_COLOR_BIT) != 0) {
        command->clearRGBA8 = packClearRGBA8(attachments[0].clearValue.color);
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdBindVertexBuffers(
    VkCommandBuffer commandBuffer,
    std::uint32_t firstBinding,
    std::uint32_t bindingCount,
    const VkBuffer* buffers,
    const VkDeviceSize* offsets
) {
    if (traceEnabled()) for (std::uint32_t index = 0; index < bindingCount; ++index) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: bind vertex command=%p binding=%u buffer=%p offset=%llu\n",
            static_cast<void*>(commandBuffer),
            firstBinding + index,
            reinterpret_cast<void*>(buffers[index]),
            static_cast<unsigned long long>(offsets[index])
        );
    }
    if (bindingCount == 0 || buffers == nullptr || offsets == nullptr) return;
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* buffer = objectState<BufferState>(buffers[0]);
    if (gState.commandBuffers.contains(command) && command->recording && firstBinding == 0
        && gState.buffers.contains(buffer)) {
        command->vertexBuffer = buffer;
        command->vertexBufferOffset = offsets[0];
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdBindIndexBuffer(
    VkCommandBuffer commandBuffer,
    VkBuffer buffer,
    VkDeviceSize offset,
    VkIndexType indexType
) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: bind index command=%p buffer=%p offset=%llu type=%d\n",
            static_cast<void*>(commandBuffer),
            reinterpret_cast<void*>(buffer),
            static_cast<unsigned long long>(offset),
            indexType
        );
    }
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* bufferState = objectState<BufferState>(buffer);
    if (gState.commandBuffers.contains(command) && command->recording
        && gState.buffers.contains(bufferState)) {
        command->indexBuffer = bufferState;
        command->indexBufferOffset = offset;
        command->indexType = indexType;
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdDrawIndexed(
    VkCommandBuffer commandBuffer,
    std::uint32_t indexCount,
    std::uint32_t instanceCount,
    std::uint32_t firstIndex,
    std::int32_t vertexOffset,
    std::uint32_t firstInstance
) {
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: draw indexed command=%p count=%u instances=%u first=%u vertexOffset=%d firstInstance=%u\n",
            static_cast<void*>(commandBuffer),
            indexCount,
            instanceCount,
            firstIndex,
            vertexOffset,
            firstInstance
        );
    }
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    if (!gState.commandBuffers.contains(command) || !command->recording || !command->insideRenderPass
        || command->pipeline == nullptr || !command->pipeline->isKitUI
        || command->vertexBuffer == nullptr || command->indexBuffer == nullptr
        || command->indexType != VK_INDEX_TYPE_UINT32 || indexCount == 0
        || instanceCount != 1 || firstInstance != 0) {
        return;
    }
    ImageState* texture = nullptr;
    if (command->graphicsDescriptorSets[0] != nullptr) {
        const auto found = command->graphicsDescriptorSets[0]->sampledImages.find(command->textureIndex);
        if (found != command->graphicsDescriptorSets[0]->sampledImages.end()) texture = found->second;
    }
    try {
        command->uiDraws.push_back(RecordedUIDraw{
            texture,
            indexCount,
            firstIndex,
            vertexOffset,
            command->scissor,
        });
    } catch (const std::bad_alloc&) {
        command->uiDraws.clear();
    }
}

std::uint32_t packClearRGBA8(const VkClearColorValue& color) {
    const auto channel = [](float value) {
        return static_cast<std::uint32_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    return channel(color.float32[0])
        | (channel(color.float32[1]) << 8)
        | (channel(color.float32[2]) << 16)
        | (channel(color.float32[3]) << 24);
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdBeginRenderPass(
    VkCommandBuffer commandBuffer,
    const VkRenderPassBeginInfo* beginInfo,
    VkSubpassContents contents
) {
    if (beginInfo == nullptr || contents != VK_SUBPASS_CONTENTS_INLINE) return;
    if (traceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: begin render command=%p pass=%p framebuffer=%p area=%d,%d %ux%u clears=%u\n",
            static_cast<void*>(commandBuffer),
            reinterpret_cast<void*>(beginInfo->renderPass),
            reinterpret_cast<void*>(beginInfo->framebuffer),
            beginInfo->renderArea.offset.x,
            beginInfo->renderArea.offset.y,
            beginInfo->renderArea.extent.width,
            beginInfo->renderArea.extent.height,
            beginInfo->clearValueCount
        );
    }
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* renderPass = objectState<RenderPassState>(beginInfo->renderPass);
    auto* framebuffer = objectState<FramebufferState>(beginInfo->framebuffer);
    if (!gState.commandBuffers.contains(command) || !command->recording || command->insideRenderPass
        || !gState.renderPasses.contains(renderPass) || !gState.framebuffers.contains(framebuffer)
        || framebuffer->renderPass != renderPass || framebuffer->colorImage == nullptr) {
        return;
    }
    command->framebuffer = framebuffer;
    command->insideRenderPass = true;
    command->clearRGBA8 = beginInfo->clearValueCount != 0 && beginInfo->pClearValues != nullptr
        ? packClearRGBA8(beginInfo->pClearValues[0].color)
        : 0xff000000U;
    command->drewTriangle = false;
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdEndRenderPass(VkCommandBuffer commandBuffer) {
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    if (gState.commandBuffers.contains(command) && command->recording && command->insideRenderPass) {
        command->insideRenderPass = false;
    }
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdDraw(
    VkCommandBuffer commandBuffer,
    std::uint32_t vertexCount,
    std::uint32_t instanceCount,
    std::uint32_t firstVertex,
    std::uint32_t firstInstance
) {
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    if (gState.commandBuffers.contains(command) && command->recording && command->insideRenderPass
        && command->framebuffer != nullptr && command->pipeline != nullptr
        && command->pipeline->graphics && command->pipeline->isFixedTriangle
        && vertexCount == 3 && instanceCount == 1 && firstVertex == 0 && firstInstance == 0) {
        command->drewTriangle = true;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateSemaphore(
    VkDevice device,
    const VkSemaphoreCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkSemaphore* semaphore
) {
    if (createInfo == nullptr || semaphore == nullptr || createInfo->flags != 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    auto* state = new (std::nothrow) SemaphoreState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    const auto* next = reinterpret_cast<const VkBaseInStructure*>(createInfo->pNext);
    while (next != nullptr) {
        if (next->sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO) {
            const auto* typeInfo = reinterpret_cast<const VkSemaphoreTypeCreateInfo*>(next);
            if (typeInfo->semaphoreType != VK_SEMAPHORE_TYPE_BINARY
                && typeInfo->semaphoreType != VK_SEMAPHORE_TYPE_TIMELINE) {
                delete state;
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            state->timeline = typeInfo->semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE;
            state->value = state->timeline ? typeInfo->initialValue : 0;
            state->signaled = state->timeline && typeInfo->initialValue != 0;
        } else if (next->sType == VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO) {
            state->exportHandleTypes =
                reinterpret_cast<const VkExportSemaphoreCreateInfo*>(next)->handleTypes;
        }
        next = next->pNext;
    }
    if ((state->exportHandleTypes
            & ~VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) != 0) {
        delete state;
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    gState.semaphores.insert(state);
    *semaphore = makeObjectHandle<VkSemaphore>(state);
    if (externalMemoryTraceEnabled() || rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: semaphore handle=%p timeline=%d value=%llu exportTypes=%#x\n",
            static_cast<void*>(state),
            state->timeline ? 1 : 0,
            static_cast<unsigned long long>(state->value),
            state->exportHandleTypes
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroySemaphore(
    VkDevice,
    VkSemaphore semaphore,
    const VkAllocationCallbacks*
) {
    if (semaphore == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<SemaphoreState>(semaphore);
    if (gState.semaphores.erase(state) != 0) {
        if (state->externalFD >= 0) ::close(state->externalFD);
        delete state;
    }
}

VkResult syncSemaphoreToExternalFDLocked(SemaphoreState* state) {
    if (state == nullptr || state->externalFD < 0) return VK_SUCCESS;
    const std::uint64_t payload = state->timeline
        ? state->value
        : static_cast<std::uint64_t>(state->signaled ? 1 : 0);
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&payload);
    std::size_t written = 0;
    while (written < sizeof(payload)) {
        const ssize_t count = ::pwrite(
            state->externalFD,
            bytes + written,
            sizeof(payload) - written,
            static_cast<off_t>(written)
        );
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return VK_ERROR_INVALID_EXTERNAL_HANDLE;
        written += static_cast<std::size_t>(count);
    }
    return VK_SUCCESS;
}

VkResult refreshSemaphoreFromExternalFDLocked(SemaphoreState* state) {
    if (state == nullptr || state->externalFD < 0) return VK_SUCCESS;
    std::uint64_t payload = 0;
    auto* bytes = reinterpret_cast<std::uint8_t*>(&payload);
    std::size_t read = 0;
    while (read < sizeof(payload)) {
        const ssize_t count = ::pread(
            state->externalFD,
            bytes + read,
            sizeof(payload) - read,
            static_cast<off_t>(read)
        );
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return VK_ERROR_INVALID_EXTERNAL_HANDLE;
        read += static_cast<std::size_t>(count);
    }
    if (state->timeline) {
        state->value = std::max(state->value, payload);
        state->signaled = state->value != 0;
    } else {
        state->signaled = payload != 0;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetSemaphoreFdKHR(
    VkDevice device,
    const VkSemaphoreGetFdInfoKHR* info,
    int* fd
) {
    if (info == nullptr || fd == nullptr
        || info->sType != VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR
        || info->handleType != VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<SemaphoreState>(info->semaphore);
    if (!validDevice(device) || !gState.semaphores.contains(state)
        || state->device != device
        || (state->exportHandleTypes
            & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) == 0) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    if (state->externalFD < 0) {
        char path[] = "/tmp/imb-vulkan-semaphore.XXXXXX";
        const int backingFD = ::mkstemp(path);
        if (backingFD < 0) return VK_ERROR_TOO_MANY_OBJECTS;
        ::unlink(path);
        if (::ftruncate(backingFD, static_cast<off_t>(sizeof(std::uint64_t))) != 0) {
            ::close(backingFD);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        state->externalFD = backingFD;
    }
    const VkResult syncResult = syncSemaphoreToExternalFDLocked(state);
    if (syncResult != VK_SUCCESS) return syncResult;
    const int output = ::dup(state->externalFD);
    if (output < 0) return VK_ERROR_TOO_MANY_OBJECTS;
    *fd = output;
    if (externalMemoryTraceEnabled() || rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: exported semaphore=%p OPAQUE_FD=%d backingFD=%d value=%llu\n",
            static_cast<void*>(state),
            output,
            state->externalFD,
            static_cast<unsigned long long>(
                state->timeline ? state->value : (state->signaled ? 1 : 0)
            )
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkImportSemaphoreFdKHR(
    VkDevice device,
    const VkImportSemaphoreFdInfoKHR* info
) {
    if (info == nullptr
        || info->sType != VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR
        || info->handleType != VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
        || info->fd < 0
        || (info->flags & ~VK_SEMAPHORE_IMPORT_TEMPORARY_BIT) != 0) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<SemaphoreState>(info->semaphore);
    struct stat status {};
    if (!validDevice(device) || !gState.semaphores.contains(state)
        || state->device != device
        || ::fstat(info->fd, &status) != 0
        || status.st_size < static_cast<off_t>(sizeof(std::uint64_t))) {
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }
    if (state->externalFD >= 0) ::close(state->externalFD);
    state->externalFD = info->fd;
    state->exportHandleTypes |= VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    const VkResult refreshResult = refreshSemaphoreFromExternalFDLocked(state);
    if (refreshResult != VK_SUCCESS) {
        state->externalFD = -1;
        return refreshResult;
    }
    if (externalMemoryTraceEnabled() || rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: imported semaphore=%p OPAQUE_FD=%d flags=%#x value=%llu\n",
            static_cast<void*>(state),
            info->fd,
            info->flags,
            static_cast<unsigned long long>(
                state->timeline ? state->value : (state->signaled ? 1 : 0)
            )
        );
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetSemaphoreCounterValue(
    VkDevice device,
    VkSemaphore semaphore,
    std::uint64_t* value
) {
    if (value == nullptr || semaphore == VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<SemaphoreState>(semaphore);
    if (!validDevice(device) || !gState.semaphores.contains(state) || state->device != device) {
        return VK_ERROR_DEVICE_LOST;
    }
    const VkResult refreshResult = refreshSemaphoreFromExternalFDLocked(state);
    if (refreshResult != VK_SUCCESS) return refreshResult;
    // Kit's Vulkan backend resolves and calls the KHR entry point even for its
    // binary-semaphore fallback.  Expose a stable 0/1 counter in that case.
    *value = state->timeline ? state->value : (state->signaled ? 1U : 0U);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkSignalSemaphore(
    VkDevice device,
    const VkSemaphoreSignalInfo* signalInfo
) {
    if (signalInfo == nullptr || signalInfo->semaphore == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<SemaphoreState>(signalInfo->semaphore);
    if (!validDevice(device) || !gState.semaphores.contains(state) || state->device != device) {
        return VK_ERROR_DEVICE_LOST;
    }
    if (state->timeline) state->value = std::max(state->value, signalInfo->value);
    state->signaled = true;
    return syncSemaphoreToExternalFDLocked(state);
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkWaitSemaphores(
    VkDevice device,
    const VkSemaphoreWaitInfo* waitInfo,
    std::uint64_t timeout
) {
    if (waitInfo == nullptr
        || (waitInfo->semaphoreCount != 0
            && (waitInfo->pSemaphores == nullptr || waitInfo->pValues == nullptr))) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_DEVICE_LOST;
    bool anySatisfied = false;
    bool allSatisfied = true;
    for (std::uint32_t index = 0; index < waitInfo->semaphoreCount; ++index) {
        auto* state = objectState<SemaphoreState>(waitInfo->pSemaphores[index]);
        if (!gState.semaphores.contains(state) || state->device != device) {
            return VK_ERROR_DEVICE_LOST;
        }
        const VkResult refreshResult = refreshSemaphoreFromExternalFDLocked(state);
        if (refreshResult != VK_SUCCESS) return refreshResult;
        const bool satisfied = state->timeline
            ? state->value >= waitInfo->pValues[index]
            : state->signaled;
        if (!satisfied && semaphoreTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: semaphore wait unsatisfied handle=%p timeline=%d actual=%llu requested=%llu timeout=%llu flags=%#x\n",
                static_cast<void*>(state),
                state->timeline ? 1 : 0,
                static_cast<unsigned long long>(
                    state->timeline ? state->value : (state->signaled ? 1U : 0U)
                ),
                static_cast<unsigned long long>(waitInfo->pValues[index]),
                static_cast<unsigned long long>(timeout),
                waitInfo->flags
            );
        }
        anySatisfied = anySatisfied || satisfied;
        allSatisfied = allSatisfied && satisfied;
    }
    const bool waitAny = (waitInfo->flags & VK_SEMAPHORE_WAIT_ANY_BIT) != 0;
    return (waitAny ? anySatisfied : allSatisfied) ? VK_SUCCESS : VK_TIMEOUT;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateQueryPool(
    VkDevice device,
    const VkQueryPoolCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkQueryPool* queryPool
) {
    if (createInfo == nullptr || queryPool == nullptr || createInfo->queryCount == 0
        || createInfo->flags != 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_DEVICE_LOST;
    auto* state = new (std::nothrow) QueryPoolState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    state->type = createInfo->queryType;
    state->count = createInfo->queryCount;
    state->pipelineStatistics = createInfo->pipelineStatistics;
    try {
        state->values.assign(state->count, 0);
        state->available.assign(state->count, false);
    } catch (const std::bad_alloc&) {
        delete state;
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    gState.queryPools.insert(state);
    *queryPool = makeObjectHandle<VkQueryPool>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyQueryPool(
    VkDevice,
    VkQueryPool queryPool,
    const VkAllocationCallbacks*
) {
    if (queryPool == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<QueryPoolState>(queryPool);
    if (gState.queryPools.erase(state) != 0) delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetQueryPoolResults(
    VkDevice device,
    VkQueryPool queryPool,
    std::uint32_t firstQuery,
    std::uint32_t queryCount,
    std::size_t dataSize,
    void* data,
    VkDeviceSize stride,
    VkQueryResultFlags flags
) {
    if (queryCount != 0 && data == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<QueryPoolState>(queryPool);
    if (!validDevice(device) || !gState.queryPools.contains(state) || state->device != device
        || firstQuery > state->count || queryCount > state->count - firstQuery) {
        return VK_ERROR_DEVICE_LOST;
    }
    std::uint32_t valueCount = 1;
    if (state->type == VK_QUERY_TYPE_PIPELINE_STATISTICS) {
        valueCount = 0;
        auto bits = state->pipelineStatistics;
        while (bits != 0) {
            valueCount += bits & 1U;
            bits >>= 1U;
        }
        valueCount = std::max(valueCount, 1U);
    }
    const bool withAvailability = (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0;
    const std::size_t valueSize = (flags & VK_QUERY_RESULT_64_BIT) != 0
        ? sizeof(std::uint64_t)
        : sizeof(std::uint32_t);
    const std::size_t rowSize = static_cast<std::size_t>(valueCount + (withAvailability ? 1U : 0U))
        * valueSize;
    if (queryCount != 0) {
        const std::size_t required = static_cast<std::size_t>(queryCount - 1U) * stride + rowSize;
        if (dataSize < required || stride < rowSize) return VK_ERROR_INITIALIZATION_FAILED;
    }
    auto* bytes = static_cast<std::uint8_t*>(data);
    for (std::uint32_t query = 0; query < queryCount; ++query) {
        auto* row = bytes + static_cast<std::size_t>(query) * stride;
        std::memset(row, 0, rowSize);
        const std::uint32_t queryIndex = firstQuery + query;
        const std::uint64_t storedValue = state->values[queryIndex];
        if (valueSize == sizeof(std::uint64_t)) {
            std::memcpy(row, &storedValue, sizeof(storedValue));
        } else {
            const std::uint32_t storedValue32 = static_cast<std::uint32_t>(storedValue);
            std::memcpy(row, &storedValue32, sizeof(storedValue32));
        }
        if (withAvailability) {
            auto* availability = row + static_cast<std::size_t>(valueCount) * valueSize;
            if (valueSize == sizeof(std::uint64_t)) {
                const std::uint64_t ready = state->available[queryIndex] ? 1 : 0;
                std::memcpy(availability, &ready, sizeof(ready));
            } else {
                const std::uint32_t ready = state->available[queryIndex] ? 1 : 0;
                std::memcpy(availability, &ready, sizeof(ready));
            }
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkResetQueryPool(
    VkDevice device,
    VkQueryPool queryPool,
    std::uint32_t firstQuery,
    std::uint32_t queryCount
) {
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<QueryPoolState>(queryPool);
    if (!validDevice(device) || !gState.queryPools.contains(state)
        || firstQuery > state->count || queryCount > state->count - firstQuery) return;
    std::fill_n(state->values.begin() + firstQuery, queryCount, std::uint64_t{0});
    std::fill_n(state->available.begin() + firstQuery, queryCount, false);
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdResetQueryPool(
    VkCommandBuffer commandBuffer,
    VkQueryPool queryPool,
    std::uint32_t firstQuery,
    std::uint32_t queryCount
) {
    std::lock_guard lock(gState.mutex);
    auto* command = reinterpret_cast<CommandBufferState*>(commandBuffer);
    auto* state = objectState<QueryPoolState>(queryPool);
    if (!gState.commandBuffers.contains(command) || !command->recording
        || !gState.queryPools.contains(state)
        || firstQuery > state->count || queryCount > state->count - firstQuery) return;
    std::fill_n(state->values.begin() + firstQuery, queryCount, std::uint64_t{0});
    std::fill_n(state->available.begin() + firstQuery, queryCount, false);
}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdBeginQuery(
    VkCommandBuffer,
    VkQueryPool,
    std::uint32_t,
    VkQueryControlFlags
) {}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdEndQuery(
    VkCommandBuffer,
    VkQueryPool,
    std::uint32_t
) {}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdWriteTimestamp(
    VkCommandBuffer,
    VkPipelineStageFlagBits,
    VkQueryPool,
    std::uint32_t
) {}

VKAPI_ATTR void VKAPI_CALL imb_vkCmdCopyQueryPoolResults(
    VkCommandBuffer,
    VkQueryPool,
    std::uint32_t,
    std::uint32_t,
    VkBuffer,
    VkDeviceSize,
    VkDeviceSize,
    VkQueryResultFlags
) {}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkCreateFence(
    VkDevice device,
    const VkFenceCreateInfo* createInfo,
    const VkAllocationCallbacks*,
    VkFence* fence
) {
    if (createInfo == nullptr || fence == nullptr
        || (createInfo->flags & ~VK_FENCE_CREATE_SIGNALED_BIT) != 0) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    if (!validDevice(device)) return VK_ERROR_INITIALIZATION_FAILED;
    auto* state = new (std::nothrow) FenceState{};
    if (state == nullptr) return VK_ERROR_OUT_OF_HOST_MEMORY;
    state->device = device;
    state->signaled = (createInfo->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
    gState.fences.insert(state);
    *fence = makeObjectHandle<VkFence>(state);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL imb_vkDestroyFence(
    VkDevice,
    VkFence fence,
    const VkAllocationCallbacks*
) {
    if (fence == VK_NULL_HANDLE) return;
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<FenceState>(fence);
    if (gState.fences.erase(state) != 0) delete state;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkResetFences(
    VkDevice,
    std::uint32_t fenceCount,
    const VkFence* fences
) {
    if (fenceCount != 0 && fences == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    for (std::uint32_t index = 0; index < fenceCount; ++index) {
        auto* state = objectState<FenceState>(fences[index]);
        if (!gState.fences.contains(state) || (state->submitted && !state->signaled)) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    for (std::uint32_t index = 0; index < fenceCount; ++index) {
        auto* state = objectState<FenceState>(fences[index]);
        state->bridgeFenceID = 0;
        state->resultMemory = nullptr;
        state->resultImage = nullptr;
        state->submitted = false;
        state->signaled = false;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkGetFenceStatus(VkDevice, VkFence fence) {
    std::lock_guard lock(gState.mutex);
    auto* state = objectState<FenceState>(fence);
    if (!gState.fences.contains(state)) return VK_ERROR_DEVICE_LOST;
    return state->signaled ? VK_SUCCESS : VK_NOT_READY;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkWaitForFences(
    VkDevice,
    std::uint32_t fenceCount,
    const VkFence* fences,
    VkBool32 waitAll,
    std::uint64_t timeout
) {
    if (fenceCount == 0 || fences == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    if (timeout == 0) {
        for (std::uint32_t index = 0; index < fenceCount; ++index) {
            auto* state = objectState<FenceState>(fences[index]);
            if (!gState.fences.contains(state)) return VK_ERROR_DEVICE_LOST;
            if ((waitAll != VK_FALSE && !state->signaled) || (waitAll == VK_FALSE && state->signaled)) {
                return waitAll == VK_FALSE && state->signaled ? VK_SUCCESS : VK_TIMEOUT;
            }
        }
        return VK_SUCCESS;
    }
    for (std::uint32_t index = 0; index < fenceCount; ++index) {
        auto* state = objectState<FenceState>(fences[index]);
        const VkResult result = completeFenceLocked(state);
        if (result == VK_SUCCESS && waitAll == VK_FALSE) return VK_SUCCESS;
        if (result != VK_SUCCESS) return result;
    }
    return VK_SUCCESS;
}

void replaceSparseBufferBindingLocked(
    BufferState* buffer,
    const VkSparseMemoryBind& bind
) {
    const VkDeviceSize bindEnd = bind.resourceOffset + bind.size;
    std::vector<SparseBufferBinding> updated;
    updated.reserve(buffer->sparseBindings.size() + 2);
    for (const auto& existing : buffer->sparseBindings) {
        const VkDeviceSize existingEnd = existing.resourceOffset + existing.size;
        if (existingEnd <= bind.resourceOffset || existing.resourceOffset >= bindEnd) {
            updated.push_back(existing);
            continue;
        }
        if (existing.resourceOffset < bind.resourceOffset) {
            auto left = existing;
            left.size = bind.resourceOffset - existing.resourceOffset;
            updated.push_back(left);
        }
        if (existingEnd > bindEnd) {
            auto right = existing;
            const VkDeviceSize removed = bindEnd - existing.resourceOffset;
            right.resourceOffset = bindEnd;
            right.size = existingEnd - bindEnd;
            right.memoryOffset += removed;
            updated.push_back(right);
        }
    }
    if (bind.memory != VK_NULL_HANDLE) {
        updated.push_back({
            bind.resourceOffset,
            bind.size,
            objectState<DeviceMemoryState>(bind.memory),
            bind.memoryOffset,
            bind.flags,
        });
    }
    std::sort(
        updated.begin(),
        updated.end(),
        [](const SparseBufferBinding& left, const SparseBufferBinding& right) {
            return left.resourceOffset < right.resourceOffset;
        }
    );
    buffer->sparseBindings = std::move(updated);
}

void replaceSparseImageBindingLocked(
    ImageState* image,
    const VkSparseImageMemoryBind& bind
) {
    image->sparseBindings.erase(
        std::remove_if(
            image->sparseBindings.begin(),
            image->sparseBindings.end(),
            [&bind](const SparseImageBinding& existing) {
                return existing.subresource.aspectMask == bind.subresource.aspectMask
                    && existing.subresource.mipLevel == bind.subresource.mipLevel
                    && existing.subresource.arrayLayer == bind.subresource.arrayLayer
                    && existing.offset.x == bind.offset.x
                    && existing.offset.y == bind.offset.y
                    && existing.offset.z == bind.offset.z
                    && existing.extent.width == bind.extent.width
                    && existing.extent.height == bind.extent.height
                    && existing.extent.depth == bind.extent.depth;
            }
        ),
        image->sparseBindings.end()
    );
    if (bind.memory != VK_NULL_HANDLE) {
        image->sparseBindings.push_back({
            bind.subresource,
            bind.offset,
            bind.extent,
            objectState<DeviceMemoryState>(bind.memory),
            bind.memoryOffset,
            bind.flags,
        });
    }
}

void replaceSparseImageOpaqueBindingLocked(
    ImageState* image,
    const VkSparseMemoryBind& bind
) {
    image->sparseOpaqueBindings.erase(
        std::remove_if(
            image->sparseOpaqueBindings.begin(),
            image->sparseOpaqueBindings.end(),
            [&bind](const SparseImageOpaqueBinding& existing) {
                return existing.resourceOffset == bind.resourceOffset
                    && existing.size == bind.size;
            }
        ),
        image->sparseOpaqueBindings.end()
    );
    if (bind.memory != VK_NULL_HANDLE) {
        image->sparseOpaqueBindings.push_back({
            bind.resourceOffset,
            bind.size,
            objectState<DeviceMemoryState>(bind.memory),
            bind.memoryOffset,
            bind.flags,
        });
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkQueueBindSparse(
    VkQueue queue,
    std::uint32_t bindInfoCount,
    const VkBindSparseInfo* bindInfos,
    VkFence fence
) {
    if (bindInfoCount != 0 && bindInfos == nullptr) return VK_ERROR_INITIALIZATION_FAILED;
    std::lock_guard lock(gState.mutex);
    const VkDevice device = deviceForQueue(queue);
    if (device == VK_NULL_HANDLE) return VK_ERROR_DEVICE_LOST;
    if (rayTracingTraceEnabled()) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: queue bind sparse queue=%p infos=%u fence=%p\n",
            reinterpret_cast<void*>(queue),
            bindInfoCount,
            reinterpret_cast<void*>(fence)
        );
        for (std::uint32_t infoIndex = 0; infoIndex < bindInfoCount; ++infoIndex) {
            const auto& info = bindInfos[infoIndex];
            std::fprintf(
                stderr,
                "imb-vulkan-icd: queue bind sparse info=%u waits=%u buffers=%u opaqueImages=%u tiledImages=%u signals=%u\n",
                infoIndex,
                info.waitSemaphoreCount,
                info.bufferBindCount,
                info.imageOpaqueBindCount,
                info.imageBindCount,
                info.signalSemaphoreCount
            );
        }
    }

    FenceState* fenceState = nullptr;
    if (fence != VK_NULL_HANDLE) {
        fenceState = objectState<FenceState>(fence);
        if (!gState.fences.contains(fenceState) || fenceState->device != device
            || fenceState->submitted || fenceState->signaled) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }

    for (std::uint32_t infoIndex = 0; infoIndex < bindInfoCount; ++infoIndex) {
        const auto& info = bindInfos[infoIndex];
        if ((info.waitSemaphoreCount != 0 && info.pWaitSemaphores == nullptr)
            || (info.signalSemaphoreCount != 0 && info.pSignalSemaphores == nullptr)
            || (info.bufferBindCount != 0 && info.pBufferBinds == nullptr)
            || (info.imageOpaqueBindCount != 0 && info.pImageOpaqueBinds == nullptr)
            || (info.imageBindCount != 0 && info.pImageBinds == nullptr)) {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        for (std::uint32_t semaphoreIndex = 0;
             semaphoreIndex < info.waitSemaphoreCount;
             ++semaphoreIndex) {
            auto* semaphore = objectState<SemaphoreState>(info.pWaitSemaphores[semaphoreIndex]);
            if (!gState.semaphores.contains(semaphore) || semaphore->device != device
                || (!semaphore->timeline && !semaphore->signaled)) {
                return VK_ERROR_INITIALIZATION_FAILED;
            }
        }
        for (std::uint32_t semaphoreIndex = 0;
             semaphoreIndex < info.signalSemaphoreCount;
             ++semaphoreIndex) {
            auto* semaphore = objectState<SemaphoreState>(info.pSignalSemaphores[semaphoreIndex]);
            if (!gState.semaphores.contains(semaphore) || semaphore->device != device) {
                return VK_ERROR_INITIALIZATION_FAILED;
            }
        }
        for (std::uint32_t bufferIndex = 0; bufferIndex < info.bufferBindCount; ++bufferIndex) {
            const auto& bufferInfo = info.pBufferBinds[bufferIndex];
            auto* buffer = objectState<BufferState>(bufferInfo.buffer);
            if (!gState.buffers.contains(buffer) || buffer->device != device
                || (buffer->flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT) == 0
                || (bufferInfo.bindCount != 0 && bufferInfo.pBinds == nullptr)) {
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            for (std::uint32_t bindIndex = 0; bindIndex < bufferInfo.bindCount; ++bindIndex) {
                const auto& bind = bufferInfo.pBinds[bindIndex];
                auto* memory = objectState<DeviceMemoryState>(bind.memory);
                if (bind.size == 0 || bind.resourceOffset > buffer->size
                    || bind.size > buffer->size - bind.resourceOffset
                    || (bind.resourceOffset % 65536) != 0
                    || (bind.memoryOffset % 65536) != 0
                    || (bind.flags & ~VK_SPARSE_MEMORY_BIND_METADATA_BIT) != 0
                    || (bind.memory != VK_NULL_HANDLE
                        && (!gState.memories.contains(memory) || memory->device != device
                            || bind.memoryOffset > memory->size
                            || bind.size > memory->size - bind.memoryOffset))) {
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
            }
        }
        for (std::uint32_t imageIndex = 0; imageIndex < info.imageOpaqueBindCount; ++imageIndex) {
            const auto& imageInfo = info.pImageOpaqueBinds[imageIndex];
            auto* image = objectState<ImageState>(imageInfo.image);
            if (!gState.images.contains(image) || image->device != device
                || (image->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) == 0
                || (image->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) == 0
                || image->sparseTileBytes == 0
                || (imageInfo.bindCount != 0 && imageInfo.pBinds == nullptr)) {
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            for (std::uint32_t bindIndex = 0; bindIndex < imageInfo.bindCount; ++bindIndex) {
                const auto& bind = imageInfo.pBinds[bindIndex];
                auto* memory = objectState<DeviceMemoryState>(bind.memory);
                if (bind.size == 0 || bind.resourceOffset > image->size
                    || bind.size > image->size - bind.resourceOffset
                    || (bind.resourceOffset % image->sparseTileBytes) != 0
                    || (bind.size % image->sparseTileBytes) != 0
                    || (bind.memoryOffset % image->sparseTileBytes) != 0
                    || bind.flags != 0
                    || (bind.memory != VK_NULL_HANDLE
                        && (!gState.memories.contains(memory) || memory->device != device
                            || bind.memoryOffset > memory->size
                            || bind.size > memory->size - bind.memoryOffset))) {
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
            }
        }
        for (std::uint32_t imageIndex = 0; imageIndex < info.imageBindCount; ++imageIndex) {
            const auto& imageInfo = info.pImageBinds[imageIndex];
            auto* image = objectState<ImageState>(imageInfo.image);
            if (!gState.images.contains(image) || image->device != device
                || (image->flags & VK_IMAGE_CREATE_SPARSE_BINDING_BIT) == 0
                || (image->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) == 0
                || image->sparseTileBytes == 0
                || (imageInfo.bindCount != 0 && imageInfo.pBinds == nullptr)) {
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            for (std::uint32_t bindIndex = 0; bindIndex < imageInfo.bindCount; ++bindIndex) {
                const auto& bind = imageInfo.pBinds[bindIndex];
                auto* memory = objectState<DeviceMemoryState>(bind.memory);
                if (bind.subresource.mipLevel >= image->mipLevels
                    || bind.subresource.arrayLayer >= image->arrayLayers) {
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
                const VkExtent3D mipExtent{
                    std::max(1U, image->extent.width >> bind.subresource.mipLevel),
                    std::max(1U, image->extent.height >> bind.subresource.mipLevel),
                    std::max(1U, image->extent.depth >> bind.subresource.mipLevel),
                };
                const auto granularity = image->sparseGranularity;
                const std::uint64_t tileCount =
                    ((bind.extent.width + granularity.width - 1) / granularity.width)
                    * static_cast<std::uint64_t>(
                        (bind.extent.height + granularity.height - 1) / granularity.height
                    )
                    * ((bind.extent.depth + granularity.depth - 1) / granularity.depth);
                const std::uint64_t requiredBytes = clampedMultiply(
                    tileCount,
                    image->sparseTileBytes
                );
                if (bind.subresource.aspectMask != formatAspectMask(image->format)
                    || bind.offset.x < 0 || bind.offset.y < 0 || bind.offset.z < 0
                    || bind.extent.width == 0 || bind.extent.height == 0 || bind.extent.depth == 0
                    || static_cast<std::uint32_t>(bind.offset.x) > mipExtent.width
                    || bind.extent.width > mipExtent.width - static_cast<std::uint32_t>(bind.offset.x)
                    || static_cast<std::uint32_t>(bind.offset.y) > mipExtent.height
                    || bind.extent.height > mipExtent.height - static_cast<std::uint32_t>(bind.offset.y)
                    || static_cast<std::uint32_t>(bind.offset.z) > mipExtent.depth
                    || bind.extent.depth > mipExtent.depth - static_cast<std::uint32_t>(bind.offset.z)
                    || (static_cast<std::uint32_t>(bind.offset.x) % granularity.width) != 0
                    || (static_cast<std::uint32_t>(bind.offset.y) % granularity.height) != 0
                    || (static_cast<std::uint32_t>(bind.offset.z) % granularity.depth) != 0
                    || (bind.memoryOffset % image->sparseTileBytes) != 0
                    || bind.flags != 0
                    || (bind.memory != VK_NULL_HANDLE
                        && (!gState.memories.contains(memory) || memory->device != device
                            || bind.memoryOffset > memory->size
                            || requiredBytes > memory->size - bind.memoryOffset))) {
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
            }
        }
    }

    try {
        for (std::uint32_t infoIndex = 0; infoIndex < bindInfoCount; ++infoIndex) {
            const auto& info = bindInfos[infoIndex];
            for (std::uint32_t semaphoreIndex = 0;
                 semaphoreIndex < info.waitSemaphoreCount;
                 ++semaphoreIndex) {
                auto* semaphore = objectState<SemaphoreState>(info.pWaitSemaphores[semaphoreIndex]);
                if (!semaphore->timeline) semaphore->signaled = false;
            }
            for (std::uint32_t bufferIndex = 0; bufferIndex < info.bufferBindCount; ++bufferIndex) {
                const auto& bufferInfo = info.pBufferBinds[bufferIndex];
                auto* buffer = objectState<BufferState>(bufferInfo.buffer);
                for (std::uint32_t bindIndex = 0; bindIndex < bufferInfo.bindCount; ++bindIndex) {
                    const auto& bind = bufferInfo.pBinds[bindIndex];
                    replaceSparseBufferBindingLocked(buffer, bind);
                    if (rayTracingTraceEnabled()) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: sparse buffer=%p resource=%llu size=%llu memory=%p memoryOffset=%llu residentRanges=%zu\n",
                            static_cast<void*>(buffer),
                            static_cast<unsigned long long>(bind.resourceOffset),
                            static_cast<unsigned long long>(bind.size),
                            static_cast<void*>(objectState<DeviceMemoryState>(bind.memory)),
                            static_cast<unsigned long long>(bind.memoryOffset),
                            buffer->sparseBindings.size()
                        );
                    }
                }
            }
            for (std::uint32_t imageIndex = 0; imageIndex < info.imageOpaqueBindCount; ++imageIndex) {
                const auto& imageInfo = info.pImageOpaqueBinds[imageIndex];
                auto* image = objectState<ImageState>(imageInfo.image);
                const auto requirements = sparseImageRequirements(
                    image->format,
                    image->type,
                    image->extent,
                    image->mipLevels,
                    image->arrayLayers
                );
                for (std::uint32_t bindIndex = 0; bindIndex < imageInfo.bindCount; ++bindIndex) {
                    const auto& bind = imageInfo.pBinds[bindIndex];
                    if ((bind.flags & VK_SPARSE_MEMORY_BIND_METADATA_BIT) == 0
                        && requirements.imageMipTailFirstLod < image->mipLevels) {
                        const VkResult backingResult = ensureSparseImageBackingLocked(image);
                        if (backingResult != VK_SUCCESS) return backingResult;
                        const VkDeviceSize bindEnd = bind.resourceOffset + bind.size;
                        for (std::uint32_t layer = 0; layer < image->arrayLayers; ++layer) {
                            const VkDeviceSize tailStart = requirements.imageMipTailOffset
                                + requirements.imageMipTailStride * layer;
                            const VkDeviceSize tailEnd = tailStart + requirements.imageMipTailSize;
                            if (bind.resourceOffset < tailEnd && bindEnd > tailStart) {
                                for (std::uint32_t mip = requirements.imageMipTailFirstLod;
                                     mip < image->mipLevels;
                                     ++mip) {
                                    const VkExtent3D mipExtent{
                                        std::max(1U, image->extent.width >> mip),
                                        std::max(1U, image->extent.height >> mip),
                                        std::max(1U, image->extent.depth >> mip),
                                    };
                                    const auto granularity = image->metalSparseGranularity;
                                    gState.bridge->updateSparseImageMapping(
                                        image->resourceID,
                                        bind.memory != VK_NULL_HANDLE,
                                        mip,
                                        layer,
                                        0,
                                        0,
                                        0,
                                        std::max(
                                            1U,
                                            (mipExtent.width + granularity.width - 1)
                                                / granularity.width
                                        ),
                                        std::max(
                                            1U,
                                            (mipExtent.height + granularity.height - 1)
                                                / granularity.height
                                        ),
                                        std::max(
                                            1U,
                                            (mipExtent.depth + granularity.depth - 1)
                                                / granularity.depth
                                        )
                                    );
                                }
                            }
                        }
                    }
                    replaceSparseImageOpaqueBindingLocked(image, bind);
                    if (rayTracingTraceEnabled()) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: sparse image opaque=%p host=%llu offset=%llu size=%llu mapped=%d\n",
                            static_cast<void*>(image),
                            static_cast<unsigned long long>(image->resourceID),
                            static_cast<unsigned long long>(bind.resourceOffset),
                            static_cast<unsigned long long>(bind.size),
                            bind.memory != VK_NULL_HANDLE ? 1 : 0
                        );
                    }
                }
            }
            for (std::uint32_t imageIndex = 0; imageIndex < info.imageBindCount; ++imageIndex) {
                const auto& imageInfo = info.pImageBinds[imageIndex];
                auto* image = objectState<ImageState>(imageInfo.image);
                const VkResult backingResult = ensureSparseImageBackingLocked(image);
                if (backingResult != VK_SUCCESS) return backingResult;
                const auto granularity = image->metalSparseGranularity;
                for (std::uint32_t bindIndex = 0; bindIndex < imageInfo.bindCount; ++bindIndex) {
                    const auto& bind = imageInfo.pBinds[bindIndex];
                    const auto tileX = static_cast<std::uint32_t>(bind.offset.x)
                        / granularity.width;
                    const auto tileY = static_cast<std::uint32_t>(bind.offset.y)
                        / granularity.height;
                    const auto tileZ = static_cast<std::uint32_t>(bind.offset.z)
                        / granularity.depth;
                    const auto tileWidth =
                        (bind.extent.width + granularity.width - 1) / granularity.width;
                    const auto tileHeight =
                        (bind.extent.height + granularity.height - 1) / granularity.height;
                    const auto tileDepth =
                        (bind.extent.depth + granularity.depth - 1) / granularity.depth;
                    gState.bridge->updateSparseImageMapping(
                        image->resourceID,
                        bind.memory != VK_NULL_HANDLE,
                        bind.subresource.mipLevel,
                        bind.subresource.arrayLayer,
                        tileX,
                        tileY,
                        tileZ,
                        tileWidth,
                        tileHeight,
                        tileDepth
                    );
                    replaceSparseImageBindingLocked(image, bind);
                    if (rayTracingTraceEnabled()) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: sparse image tiles=%p host=%llu mip=%u slice=%u tile=%u,%u,%u %ux%ux%u mapped=%d\n",
                            static_cast<void*>(image),
                            static_cast<unsigned long long>(image->resourceID),
                            bind.subresource.mipLevel,
                            bind.subresource.arrayLayer,
                            tileX,
                            tileY,
                            tileZ,
                            tileWidth,
                            tileHeight,
                            tileDepth,
                            bind.memory != VK_NULL_HANDLE ? 1 : 0
                        );
                    }
                }
            }
            for (std::uint32_t semaphoreIndex = 0;
                 semaphoreIndex < info.signalSemaphoreCount;
                 ++semaphoreIndex) {
                auto* semaphore = objectState<SemaphoreState>(info.pSignalSemaphores[semaphoreIndex]);
                if (!semaphore->timeline) semaphore->signaled = true;
            }
        }
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: sparse image mapping failed: %s\n", error.what());
        return VK_ERROR_DEVICE_LOST;
    }

    if (fenceState != nullptr) fenceState->signaled = true;
    if (rayTracingTraceEnabled()) {
        std::fprintf(stderr, "imb-vulkan-icd: queue bind sparse result=VK_SUCCESS\n");
    }
    return VK_SUCCESS;
}

VkResult completeQueueSubmitSynchronizationLocked(
    VkDevice device,
    std::uint32_t submitCount,
    const VkSubmitInfo* submits
) {
    for (std::uint32_t submitIndex = 0; submitIndex < submitCount; ++submitIndex) {
        const auto& submit = submits[submitIndex];
        const VkTimelineSemaphoreSubmitInfo* timelineInfo = nullptr;
        auto* next = reinterpret_cast<const VkBaseInStructure*>(submit.pNext);
        while (next != nullptr) {
            if (next->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
                timelineInfo = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(next);
                break;
            }
            next = next->pNext;
        }
        for (std::uint32_t index = 0; index < submit.waitSemaphoreCount; ++index) {
            auto* semaphore = objectState<SemaphoreState>(submit.pWaitSemaphores[index]);
            if (!gState.semaphores.contains(semaphore) || semaphore->device != device) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: queue submit synchronization rejected invalid wait semaphore submit=%u index=%u handle=%p\n",
                    submitIndex,
                    index,
                    static_cast<void*>(semaphore)
                );
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            const VkResult refreshResult = refreshSemaphoreFromExternalFDLocked(semaphore);
            if (refreshResult != VK_SUCCESS) return refreshResult;
            if (!semaphore->timeline) semaphore->signaled = false;
            const VkResult syncResult = syncSemaphoreToExternalFDLocked(semaphore);
            if (syncResult != VK_SUCCESS) return syncResult;
        }
        for (std::uint32_t index = 0; index < submit.signalSemaphoreCount; ++index) {
            auto* semaphore = objectState<SemaphoreState>(submit.pSignalSemaphores[index]);
            if (!gState.semaphores.contains(semaphore) || semaphore->device != device) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: queue submit synchronization rejected invalid signal semaphore submit=%u index=%u handle=%p\n",
                    submitIndex,
                    index,
                    static_cast<void*>(semaphore)
                );
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            if (semaphore->timeline) {
                const bool hasValue = timelineInfo != nullptr
                    && index < timelineInfo->signalSemaphoreValueCount
                    && timelineInfo->pSignalSemaphoreValues != nullptr;
                const std::uint64_t signalValue = hasValue
                    ? timelineInfo->pSignalSemaphoreValues[index]
                    : semaphore->value + 1;
                semaphore->value = std::max(semaphore->value, signalValue);
            }
            semaphore->signaled = true;
            if (semaphoreTraceEnabled()) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: semaphore exact completion handle=%p timeline=%d value=%llu submit=%u signal=%u\n",
                    static_cast<void*>(semaphore),
                    semaphore->timeline ? 1 : 0,
                    static_cast<unsigned long long>(
                        semaphore->timeline
                            ? semaphore->value
                            : (semaphore->signaled ? 1U : 0U)
                    ),
                    submitIndex,
                    index
                );
            }
            const VkResult syncResult = syncSemaphoreToExternalFDLocked(semaphore);
            if (syncResult != VK_SUCCESS) return syncResult;
        }
    }
    return VK_SUCCESS;
}

VkResult uploadComputeImageForMetalLocked(ImageState* image) {
    if (image == nullptr || !gState.images.contains(image) || gState.bridge == nullptr) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if ((image->flags & VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT) != 0) {
        return ensureSparseImageBackingLocked(image);
    }
    if (image->memory == nullptr
        || (image->format != VK_FORMAT_R8G8B8A8_UNORM
            && image->format != VK_FORMAT_B8G8R8A8_UNORM)
        || (image->type != VK_IMAGE_TYPE_2D && image->type != VK_IMAGE_TYPE_3D)
        || image->extent.depth == 0 || image->mipLevels != 1
        || image->arrayLayers != 1) {
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    const VkResult memoryResult = ensureMemoryBackingLocked(image->memory, false);
    if (memoryResult != VK_SUCCESS) return memoryResult;
    const VkResult imageResult = ensureImageBackingLocked(image);
    if (imageResult != VK_SUCCESS) return imageResult;
    const VkDeviceSize byteCount = static_cast<VkDeviceSize>(image->extent.width)
        * image->extent.height * image->extent.depth * 4;
    if (image->memoryOffset > image->memory->size
        || byteCount > image->memory->size - image->memoryOffset) {
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    if (image->memory->externalFD >= 0) {
        const VkResult refreshResult = refreshMemoryFromExternalFDLocked(
            image->memory,
            image->memoryOffset,
            byteCount
        );
        if (refreshResult != VK_SUCCESS) return refreshResult;
    }
    try {
        gState.bridge->writeImage(
            image->resourceID,
            image->memory->bytes.data() + image->memoryOffset,
            byteCount
        );
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: Metal compute image upload failed: %s\n", error.what());
        return VK_ERROR_DEVICE_LOST;
    }
    return VK_SUCCESS;
}

VkResult bridgeComputeDispatchLocked(const RecordedComputeDispatch& dispatch) {
    if (dispatch.pipeline == nullptr || !gState.pipelines.contains(dispatch.pipeline)
        || !dispatch.bridgeEligible
        || dispatch.pipeline->bridgeComputePipelineID == 0 || gState.bridge == nullptr
        || dispatch.groupCountX == 0 || dispatch.groupCountY == 0
        || dispatch.groupCountZ == 0) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    std::vector<BridgeComputeBinding> bindings;
    struct WritableBuffer {
        const DescriptorSetState::BufferBinding* descriptor = nullptr;
        VkDeviceSize range = 0;
    };
    std::vector<WritableBuffer> writableBuffers;
    std::vector<ImageState*> writableImages;
    try {
        const std::size_t setCount = std::min<std::size_t>(
            dispatch.descriptorSets.size(),
            dispatch.pipeline->layout->setLayouts.size()
        );
        for (std::uint32_t setIndex = 0; setIndex < setCount; ++setIndex) {
            auto* set = dispatch.descriptorSets[setIndex];
            if (set == nullptr) continue;
            if (!gState.descriptorSets.contains(set)) return VK_ERROR_INITIALIZATION_FAILED;
            for (const auto& [key, descriptor] : set->computeBuffers) {
                if (descriptor.buffer == nullptr || descriptor.buffer->memory == nullptr
                    || !gState.buffers.contains(descriptor.buffer)) {
                    return VK_ERROR_MEMORY_MAP_FAILED;
                }
                if (descriptor.offset > descriptor.buffer->size) {
                    return VK_ERROR_MEMORY_MAP_FAILED;
                }
                const VkDeviceSize availableRange =
                    descriptor.buffer->size - descriptor.offset;
                const VkDeviceSize descriptorRange =
                    descriptor.range == VK_WHOLE_SIZE
                        ? availableRange
                        : descriptor.range;
                if (descriptorRange == 0 || descriptorRange > availableRange) {
                    return VK_ERROR_MEMORY_MAP_FAILED;
                }
                const VkResult uploadResult = uploadBufferForMetalLocked(descriptor.buffer);
                if (uploadResult != VK_SUCCESS) {
                    if (computeTraceEnabled()) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: compute buffer unavailable hash=%016llx set=%u binding=%u array=%u result=%d buffer=%p memory=%p sparseBindings=%zu range=%llu\n",
                            static_cast<unsigned long long>(
                                dispatch.pipeline->computeHash
                            ),
                            setIndex,
                            static_cast<std::uint32_t>(key >> 32),
                            static_cast<std::uint32_t>(key),
                            uploadResult,
                            static_cast<void*>(descriptor.buffer),
                            static_cast<void*>(descriptor.buffer->memory),
                            descriptor.buffer->sparseBindings.size(),
                            static_cast<unsigned long long>(descriptor.range)
                        );
                    }
                    return uploadResult;
                }
                const bool writable =
                    descriptor.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                    || descriptor.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC
                    || descriptor.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                const bool texelBuffer =
                    descriptor.type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER
                    || descriptor.type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
                bindings.push_back(BridgeComputeBinding{
                    setIndex,
                    static_cast<std::uint32_t>(key >> 32),
                    static_cast<std::uint32_t>(key),
                    texelBuffer
                        ? (writable
                            ? IMB_COMPUTE_BINDING_TEXEL_BUFFER_READ_WRITE
                            : IMB_COMPUTE_BINDING_TEXEL_BUFFER_READ)
                        : (writable
                            ? IMB_COMPUTE_BINDING_BUFFER_READ_WRITE
                            : IMB_COMPUTE_BINDING_BUFFER_READ),
                    texelBuffer
                        ? static_cast<std::uint32_t>(descriptor.format)
                        : 0,
                    descriptor.buffer->memory->resourceID,
                    descriptor.buffer->memoryOffset + descriptor.offset,
                    descriptorRange,
                });
                if (writable) {
                    writableBuffers.push_back({&descriptor, descriptorRange});
                }
            }
            for (const auto& [key, descriptor] : set->computeImages) {
                if (descriptor.image == nullptr || !gState.images.contains(descriptor.image)) {
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
                const bool writable = descriptor.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                const VkResult uploadResult =
                    uploadComputeImageForMetalLocked(descriptor.image);
                if (uploadResult != VK_SUCCESS) {
                    if (computeTraceEnabled()) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: compute image unavailable hash=%016llx set=%u binding=%u array=%u result=%d image=%p memory=%p sparseBindings=%zu format=%d type=%d extent=%ux%ux%u mips=%u layers=%u samples=%#x flags=%#x\n",
                            static_cast<unsigned long long>(
                                dispatch.pipeline->computeHash
                            ),
                            setIndex,
                            static_cast<std::uint32_t>(key >> 32),
                            static_cast<std::uint32_t>(key),
                            uploadResult,
                            static_cast<void*>(descriptor.image),
                            static_cast<void*>(descriptor.image->memory),
                            descriptor.image->sparseBindings.size(),
                            descriptor.image->format,
                            descriptor.image->type,
                            descriptor.image->extent.width,
                            descriptor.image->extent.height,
                            descriptor.image->extent.depth,
                            descriptor.image->mipLevels,
                            descriptor.image->arrayLayers,
                            descriptor.image->samples,
                            descriptor.image->flags
                        );
                    }
                    return uploadResult;
                }
                bindings.push_back(BridgeComputeBinding{
                    setIndex,
                    static_cast<std::uint32_t>(key >> 32),
                    static_cast<std::uint32_t>(key),
                    writable
                        ? IMB_COMPUTE_BINDING_TEXTURE_READ_WRITE
                        : IMB_COMPUTE_BINDING_TEXTURE_READ,
                    0,
                    descriptor.image->resourceID,
                    0,
                    0,
                });
                if (writable) writableImages.push_back(descriptor.image);
            }
            for (const auto& [key, sampler] : set->computeSamplers) {
                if (sampler == nullptr || !gState.samplers.contains(sampler)
                    || sampler->device != set->device) {
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
                const std::uint32_t samplerOptions = bridgeSamplerOptions(*sampler);
                if ((samplerOptions & ~IMB_COMPUTE_SAMPLER_OPTIONS_MASK) != 0) {
                    return VK_ERROR_FEATURE_NOT_PRESENT;
                }
                bindings.push_back(BridgeComputeBinding{
                    setIndex,
                    static_cast<std::uint32_t>(key >> 32),
                    static_cast<std::uint32_t>(key),
                    IMB_COMPUTE_BINDING_SAMPLER,
                    samplerOptions,
                    0,
                    bridgeSamplerFloatPair(sampler->minLod, sampler->maxLod),
                    bridgeSamplerFloatPair(
                        sampler->mipLodBias,
                        sampler->anisotropyEnable != VK_FALSE
                            ? sampler->maxAnisotropy
                            : 1.0f
                    ),
                });
            }
        }

        const std::uint64_t fence = gState.bridge->submitCompute(
            dispatch.pipeline->bridgeComputePipelineID,
            dispatch.groupCountX,
            dispatch.groupCountY,
            dispatch.groupCountZ,
            bindings,
            dispatch.pushConstants
        );
        gState.bridge->waitFence(fence);

        for (const auto& writable : writableBuffers) {
            const auto* descriptor = writable.descriptor;
            auto* memory = descriptor->buffer->memory;
            const VkDeviceSize offset = descriptor->buffer->memoryOffset + descriptor->offset;
            if (offset > memory->size || writable.range > memory->size - offset) {
                return VK_ERROR_MEMORY_MAP_FAILED;
            }
            gState.bridge->readBuffer(
                memory->resourceID,
                memory->bytes.data() + offset,
                writable.range,
                offset
            );
            const VkResult syncResult =
                syncMemoryToExternalFDLocked(memory, offset, writable.range);
            if (syncResult != VK_SUCCESS) return syncResult;
        }
        for (auto* image : writableImages) {
            if (image->memory == nullptr
                || (image->format != VK_FORMAT_R8G8B8A8_UNORM
                    && image->format != VK_FORMAT_B8G8R8A8_UNORM)
                || image->mipLevels != 1 || image->arrayLayers != 1) {
                return VK_ERROR_FORMAT_NOT_SUPPORTED;
            }
            const VkDeviceSize byteCount = static_cast<VkDeviceSize>(image->extent.width)
                * image->extent.height * image->extent.depth * 4;
            if (image->memoryOffset > image->memory->size
                || byteCount > image->memory->size - image->memoryOffset) {
                return VK_ERROR_MEMORY_MAP_FAILED;
            }
            gState.bridge->readImage(
                image->resourceID,
                image->memory->bytes.data() + image->memoryOffset,
                byteCount
            );
            const VkResult syncResult = syncMemoryToExternalFDLocked(
                image->memory,
                image->memoryOffset,
                byteCount
            );
            if (syncResult != VK_SUCCESS) return syncResult;
            image->dirty = true;
        }
        if (computeTraceEnabled()) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: executed Metal compute pipeline=%llu groups=%ux%ux%u bindings=%zu fence=%llu\n",
                static_cast<unsigned long long>(
                    dispatch.pipeline->bridgeComputePipelineID
                ),
                dispatch.groupCountX,
                dispatch.groupCountY,
                dispatch.groupCountZ,
                bindings.size(),
                static_cast<unsigned long long>(fence)
            );
        }
        return VK_SUCCESS;
    } catch (const std::bad_alloc&) {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: Metal compute dispatch failed: %s\n", error.what());
        return VK_ERROR_DEVICE_LOST;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkQueueSubmit(
    VkQueue queue,
    std::uint32_t submitCount,
    const VkSubmitInfo* submits,
    VkFence fence
) {
    if (submitCount != 0 && submits == nullptr) {
        std::fprintf(stderr, "imb-vulkan-icd: queue submit rejected null submit array count=%u\n", submitCount);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    std::lock_guard lock(gState.mutex);
    const VkDevice device = deviceForQueue(queue);
    if (device == VK_NULL_HANDLE) return VK_ERROR_DEVICE_LOST;

    FenceState* fenceState = nullptr;
    if (fence != VK_NULL_HANDLE) {
        fenceState = objectState<FenceState>(fence);
        if (!gState.fences.contains(fenceState) || fenceState->device != device
            || fenceState->submitted || fenceState->signaled) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: queue submit rejected fence=%p known=%d deviceMatch=%d submitted=%d signaled=%d\n",
                static_cast<void*>(fenceState),
                gState.fences.contains(fenceState) ? 1 : 0,
                gState.fences.contains(fenceState) && fenceState->device == device ? 1 : 0,
                gState.fences.contains(fenceState) && fenceState->submitted ? 1 : 0,
                gState.fences.contains(fenceState) && fenceState->signaled ? 1 : 0
            );
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }

    // Host-visible buffer contents are the backing store for this ICD. Replay
    // transfers, supported compute dispatches, and KHR acceleration-structure
    // builds in their recorded Vulkan order. Kit reuses geometry ring ranges
    // between Mesh builds, so batching producers before/after every BLAS would
    // make multiple BLASes read the same Mesh.
    std::unordered_set<ImageState*> transferredImages;
    for (std::uint32_t submitIndex = 0; submitIndex < submitCount; ++submitIndex) {
        const auto& submit = submits[submitIndex];
        if (submit.commandBufferCount != 0 && submit.pCommandBuffers == nullptr) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: queue submit rejected null command array submit=%u count=%u\n",
                submitIndex,
                submit.commandBufferCount
            );
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        for (std::uint32_t commandIndex = 0; commandIndex < submit.commandBufferCount; ++commandIndex) {
            auto* command = reinterpret_cast<CommandBufferState*>(submit.pCommandBuffers[commandIndex]);
            if (!gState.commandBuffers.contains(command) || !command->executable
                || command->device != device) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: queue submit rejected command submit=%u index=%u handle=%p known=%d executable=%d deviceMatch=%d\n",
                    submitIndex,
                    commandIndex,
                    static_cast<void*>(command),
                    gState.commandBuffers.contains(command) ? 1 : 0,
                    gState.commandBuffers.contains(command) && command->executable ? 1 : 0,
                    gState.commandBuffers.contains(command) && command->device == device ? 1 : 0
                );
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            const VkResult commandResult =
                executeRecordedCommandsLocked(command, transferredImages);
            if (commandResult != VK_SUCCESS) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: queue submit recorded command failed submit=%u index=%u result=%d copies=%zu updates=%zu fills=%zu clears=%zu imageCopies=%zu bufferToImage=%zu imageToBuffer=%zu AS=%zu compute=%zu\n",
                    submitIndex,
                    commandIndex,
                    commandResult,
                    command->bufferCopies.size(),
                    command->bufferUpdates.size(),
                    command->bufferFills.size(),
                    command->imageClears.size(),
                    command->imageCopies.size(),
                    command->bufferToImageCopies.size(),
                    command->imageToBufferCopies.size(),
                    command->accelerationStructureBuildsKHR.size(),
                    command->computeDispatches.size()
                );
                return commandResult;
            }
        }
    }
    for (auto* image : transferredImages) {
        const VkResult syncResult = syncTransferredImageLocked(image);
        if (syncResult != VK_SUCCESS) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: queue submit transferred image synchronization failed image=%p resource=%llu result=%d\n",
                static_cast<void*>(image),
                static_cast<unsigned long long>(image == nullptr ? 0 : image->resourceID),
                syncResult
            );
            return syncResult;
        }
    }

    const bool isSimpleSubmit = submitCount == 1 && submits[0].commandBufferCount == 1
        && submits[0].pCommandBuffers != nullptr && submits[0].waitSemaphoreCount == 0
        && submits[0].signalSemaphoreCount == 0;
    bool isMetalAddSubmit = isSimpleSubmit;
    bool isMetalRasterSubmit = isSimpleSubmit;
    bool isMetalUISubmit = false;
    bool containsUnsupportedRayTracing = false;
    bool nullSceneRTXInitializationCandidate = true;
    std::size_t unsupportedNVBuildCount = 0;
    std::size_t unsupportedNVCopyCount = 0;
    std::size_t unsupportedNVTraceCount = 0;
    std::size_t unsupportedKHRCopyCount = 0;
    std::size_t totalKHRTraceCount = 0;
    std::size_t unresolvedKHRTraceCount = 0;
    std::size_t unknownKHRTraceCount = 0;
    std::vector<const RecordedRayTraceKHR*> allKHRTraces;
    std::vector<const RecordedRayTraceKHR*> nullSceneRTXTraces;
    std::vector<const RecordedRayTraceKHR*> sceneRayTraces;
    const RecordedRayTraceKHR* metalRayTrace = nullptr;
    const RecordedRayTraceKHR* mainSceneRayTrace = nullptr;
    CommandBufferState* addCommand = nullptr;
    CommandBufferState* rasterCommand = nullptr;
    CommandBufferState* uiCommand = nullptr;
    std::vector<CommandBufferState*> uiCommands;
    if (isSimpleSubmit) {
        addCommand = reinterpret_cast<CommandBufferState*>(submits[0].pCommandBuffers[0]);
        rasterCommand = addCommand;
        isMetalAddSubmit = gState.commandBuffers.contains(addCommand) && addCommand->executable
            && addCommand->device == device && addCommand->pipeline != nullptr
            && addCommand->pipeline->isAddUInt32 && addCommand->descriptorSet != nullptr
            && addCommand->descriptorSet->buffer != nullptr
            && addCommand->computeDispatches.empty();
        isMetalRasterSubmit = gState.commandBuffers.contains(rasterCommand) && rasterCommand->executable
            && rasterCommand->device == device && rasterCommand->pipeline != nullptr
            && rasterCommand->pipeline->isFixedTriangle && rasterCommand->framebuffer != nullptr
            && rasterCommand->framebuffer->colorImage != nullptr && rasterCommand->drewTriangle
            && !rasterCommand->insideRenderPass;
    }

    for (std::uint32_t submitIndex = 0; submitIndex < submitCount; ++submitIndex) {
        const auto& submit = submits[submitIndex];
        if (submit.commandBufferCount != 0 && submit.pCommandBuffers == nullptr) continue;
        for (std::uint32_t commandIndex = 0; commandIndex < submit.commandBufferCount; ++commandIndex) {
            auto* candidate = reinterpret_cast<CommandBufferState*>(submit.pCommandBuffers[commandIndex]);
            if (gState.commandBuffers.contains(candidate) && candidate->executable
                && candidate->device == device && candidate->pipeline != nullptr
                && candidate->pipeline->isKitUI && candidate->framebuffer != nullptr
                && candidate->framebuffer->colorImage != nullptr && !candidate->uiDraws.empty()
                && !candidate->insideRenderPass) {
                uiCommands.push_back(candidate);
                // Full can submit its docked workspace and an undocked
                // viewport window together. The viewport is commonly recorded
                // last, so "last command wins" replaces Stage/Property/Content
                // with only the viewport. The main workspace has the richer
                // indexed UI list; retain that command as the presentation
                // surface while still acknowledging the whole Vulkan submit.
                if (uiCommand == nullptr
                    || candidate->uiDraws.size() > uiCommand->uiDraws.size()) {
                    uiCommand = candidate;
                }
                isMetalUISubmit = true;
            }
            if (gState.commandBuffers.contains(candidate) && candidate->executable
                && candidate->device == device) {
                unsupportedNVBuildCount += candidate->accelerationStructureBuildsNV.size();
                unsupportedNVCopyCount += candidate->accelerationStructureCopiesNV.size();
                unsupportedNVTraceCount += candidate->rayTracesNV.size();
                unsupportedKHRCopyCount += candidate->accelerationStructureCopiesKHR.size();
                totalKHRTraceCount += candidate->rayTracesKHR.size();
                if (!candidate->accelerationStructureBuildsNV.empty()
                    || !candidate->accelerationStructureCopiesNV.empty()
                    || !candidate->rayTracesNV.empty()
                    || !candidate->accelerationStructureCopiesKHR.empty()) {
                    containsUnsupportedRayTracing = true;
                    nullSceneRTXInitializationCandidate = false;
                }
                for (const auto& trace : candidate->rayTracesKHR) {
                    allKHRTraces.push_back(&trace);
                    std::int32_t raygenGroup = trace.pipeline == nullptr
                        ? -1
                        : rayTracingGroupForSBTAddressLocked(
                            trace.pipeline,
                            device,
                            trace.raygen.deviceAddress
                        );
                    if (raygenGroup < 0 && trace.pipeline != nullptr) {
                        raygenGroup = inferRaygenGroupFromSparseRecordLocked(
                            trace.pipeline,
                            device,
                            trace.raygen
                        );
                    }
                    bool recoveredObservedRaygen = false;
                    if (raygenGroup < 0 && trace.pipeline != nullptr) {
                        const std::uint64_t expectedHash = observedIsaacRaygenHashForExtent(
                            trace.width,
                            trace.height,
                            trace.depth
                        );
                        raygenGroup = uniqueRayTracingGroupForShaderHash(
                            trace.pipeline,
                            expectedHash
                        );
                        recoveredObservedRaygen = raygenGroup >= 0;
                        if (recoveredObservedRaygen) {
                            std::fprintf(
                                stderr,
                                "imb-vulkan-icd: recovered known Isaac raygen from exact extent=%ux%ux%u group=%d hash=%016llx after unresolved SBT record\n",
                                trace.width,
                                trace.height,
                                trace.depth,
                                raygenGroup,
                                static_cast<unsigned long long>(expectedHash)
                            );
                        }
                    }
                    const std::uint64_t raygenHash = rayTracingShaderHashForGroup(
                        trace.pipeline,
                        raygenGroup
                    );
                    if (raygenHash == 0) ++unresolvedKHRTraceCount;
                    if (rayTracingTraceEnabled() && trace.pipeline != nullptr) {
                        const std::int32_t missGroup = trace.miss.deviceAddress == 0
                            ? -1
                            : rayTracingGroupForSBTAddressLocked(
                                trace.pipeline,
                                device,
                                trace.miss.deviceAddress
                            );
                        const std::int32_t hitGroup = trace.hit.deviceAddress == 0
                            ? -1
                            : rayTracingGroupForSBTAddressLocked(
                                trace.pipeline,
                                device,
                                trace.hit.deviceAddress
                            );
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: RT submit SBT extent=%ux%ux%u groups=%d/%d/%d raygenHash=%016llx\n",
                            trace.width,
                            trace.height,
                            trace.depth,
                            raygenGroup,
                            missGroup,
                            hitGroup,
                            static_cast<unsigned long long>(
                                raygenHash
                            )
                        );
                    }
                    if (trace.pipeline != nullptr && trace.pipeline->isMetalRayProbe
                        && metalRayTrace == nullptr) {
                        metalRayTrace = &trace;
                    } else {
                        containsUnsupportedRayTracing = true;
                        bool hasAccelerationStructure = false;
                        for (auto* set : trace.descriptorSets) {
                            if (set == nullptr || !gState.descriptorSets.contains(set)) continue;
                            for (const auto& [_, acceleration] : set->accelerationStructuresKHR) {
                                if (acceleration != nullptr
                                    && gState.accelerationStructuresKHR.contains(acceleration)
                                    && acceleration->built
                                    && acceleration->bridgeResourceID != 0) {
                                    hasAccelerationStructure = true;
                                    break;
                                }
                            }
                            if (hasAccelerationStructure) break;
                        }
                        const bool knownNullSceneRaygen = raygenHash == UINT64_C(0xf2dbfaa8274b5250)
                            || raygenHash == UINT64_C(0x9eeed51da7b7135d)
                            || raygenHash == UINT64_C(0xa00a626692fa2fee)
                            || raygenHash == UINT64_C(0x72f6dc6c98605c7f);
                        const bool knownSceneRayTrace =
                            raygenHash == UINT64_C(0xf2dbfaa8274b5250)
                            && trace.width >= 16 && trace.height >= 16
                            && trace.width <= 8192 && trace.height <= 8192
                            && trace.depth == 1;
                        if (!knownSceneRayTrace && !knownNullSceneRaygen) {
                            ++unknownKHRTraceCount;
                        }
                        if (knownSceneRayTrace) {
                            sceneRayTraces.push_back(&trace);
                            if (mainSceneRayTrace == nullptr
                                || (trace.width == 1280 && trace.height == 720)) {
                                mainSceneRayTrace = &trace;
                            }
                        }
                        if (knownSceneRayTrace) {
                            // The scene's TLAS can be null until the visible
                            // USD Mesh fallback is built below. Viewport and
                            // Camera Render Products may coexist in this one
                            // command buffer and must all be executed.
                        } else if (!hasAccelerationStructure && knownNullSceneRaygen) {
                            nullSceneRTXTraces.push_back(&trace);
                        } else {
                            nullSceneRTXInitializationCandidate = false;
                        }
                    }
                }
            }
        }
    }

    // Isaac RTX creates the real scene TLAS object but leaves its descriptors
    // null when its NVIDIA-only scene-instance producer is unavailable.  The
    // scene BLASes above still contain the real Hydra geometry and are now
    // built by Metal. Match the visible USD Mesh manifest to those BLASes,
    // apply the authored world transforms, exclude renderer-internal BLASes,
    // and route viewport or camera-render-product traces through Metal.
    bool deferSceneRayTracingUntilManifest = false;
    if (mainSceneRayTrace != nullptr
        && gState.bridge != nullptr
        && (gState.bridge->capabilities().bits & IMB_CAP_METAL_RAY_DISPATCH) != 0) {
        const auto liveSceneHeader = readLiveSceneState(false);
        if (emptyStageGridPresentationEnabled()
            || !liveSceneHeader.has_value()
            || !liveSceneHeader->hasMeshManifest) {
            // Full may start rendering while the stage extension is still
            // walking the real USD hierarchy and publishing its mesh
            // manifest. Rejecting that first known scene trace poisons Kit's
            // render graph before the real Metal TLAS can be built. Defer only
            // a pure KHR-trace submit with resolved shader records; once the
            // manifest appears the normal fallback-TLAS path below is used.
            deferSceneRayTracingUntilManifest = unsupportedNVBuildCount == 0
                && unsupportedNVCopyCount == 0
                && unsupportedNVTraceCount == 0
                && unsupportedKHRCopyCount == 0
                && unresolvedKHRTraceCount == 0
                && !allKHRTraces.empty();
        } else {
            AccelerationStructureKHRState* fallbackTopLevel = nullptr;
            const VkResult fallbackResult = bridgeFallbackInstanceAccelerationStructureBuildLocked(
                device,
                &fallbackTopLevel
            );
            if (fallbackResult == VK_SUCCESS && fallbackTopLevel != nullptr) {
                metalRayTrace = mainSceneRayTrace;
                containsUnsupportedRayTracing = false;
                nullSceneRTXInitializationCandidate = false;
            } else if (fallbackResult != VK_ERROR_FEATURE_NOT_PRESENT) {
                return fallbackResult;
            }
        }
    }
    std::vector<const RecordedRayTraceKHR*> metalRayTraces;
    if (metalRayTrace == mainSceneRayTrace && !sceneRayTraces.empty()) {
        metalRayTraces = sceneRayTraces;
    } else if (metalRayTrace != nullptr) {
        metalRayTraces.push_back(metalRayTrace);
    }

    // NV ray-tracing commands are deliberately never acknowledged as complete
    // until they have a real Metal acceleration-structure / shader backend.
    // Keeping the recorded object graph here lets us trace Kit's exact workload
    // without silently turning scene construction or trace dispatches into no-ops.
    const auto& initializationTraces = deferSceneRayTracingUntilManifest
        ? allKHRTraces : nullSceneRTXTraces;
    if ((deferSceneRayTracingUntilManifest
            || (containsUnsupportedRayTracing && nullSceneRTXInitializationCandidate))
        && !initializationTraces.empty()) {
        std::unordered_set<ImageState*> clearedImages;
        try {
            if (!gState.bridge
                || (gState.bridge->capabilities().bits & IMB_CAP_NOOP_COMMAND) == 0) {
                return VK_ERROR_FEATURE_NOT_PRESENT;
            }
            for (const auto* trace : initializationTraces) {
                for (auto* set : trace->descriptorSets) {
                    if (set == nullptr || !gState.descriptorSets.contains(set)) continue;
                    for (const auto& [_, image] : set->storageImages) {
                        if (image == nullptr || !gState.images.contains(image)
                            || image->memory == nullptr
                            || image->extent.width != trace->width
                            || image->extent.height != trace->height
                            || !clearedImages.insert(image).second) {
                            continue;
                        }
                        auto* memory = image->memory;
                        const VkResult backingResult = ensureMemoryBackingLocked(memory, false);
                        if (backingResult != VK_SUCCESS) return backingResult;
                        if (image->memoryOffset > memory->size
                            || image->size > memory->size - image->memoryOffset) {
                            return VK_ERROR_MEMORY_MAP_FAILED;
                        }
                        std::memset(
                            memory->bytes.data()
                                + static_cast<std::size_t>(image->memoryOffset),
                            0,
                            static_cast<std::size_t>(image->size)
                        );
                        const VkResult syncResult = syncMemoryToExternalFDLocked(
                            memory,
                            image->memoryOffset,
                            image->size
                        );
                        if (syncResult != VK_SUCCESS) return syncResult;
                        image->dirty = true;
                    }
                }
            }
            const std::uint64_t bridgeFence = gState.bridge->submitNoop();
            gState.bridge->waitFence(bridgeFence);
            // These null-TLAS initialization submits can contain RTX work the
            // bridge deliberately does not execute. Advancing a timeline
            // semaphore directly to Kit's requested value would report that
            // all of that work completed and can release dependent TaskGroups
            // too early. Preserve the conservative one-step completion used by
            // this fallback path; ordinary UI submissions below use the exact
            // VkTimelineSemaphoreSubmitInfo values after their Metal fence.
            for (std::uint32_t submitIndex = 0;
                 submitIndex < submitCount;
                 ++submitIndex) {
                const auto& submit = submits[submitIndex];
                for (std::uint32_t index = 0;
                     index < submit.waitSemaphoreCount;
                     ++index) {
                    auto* semaphore = objectState<SemaphoreState>(
                        submit.pWaitSemaphores[index]
                    );
                    if (gState.semaphores.contains(semaphore)
                        && !semaphore->timeline) {
                        semaphore->signaled = false;
                    }
                }
                for (std::uint32_t index = 0;
                     index < submit.signalSemaphoreCount;
                     ++index) {
                    auto* semaphore = objectState<SemaphoreState>(
                        submit.pSignalSemaphores[index]
                    );
                    if (gState.semaphores.contains(semaphore)) {
                        semaphore->signaled = true;
                        if (semaphore->timeline) ++semaphore->value;
                        if (semaphoreTraceEnabled()) {
                            std::fprintf(
                                stderr,
                                "imb-vulkan-icd: semaphore conservative null-TLAS completion handle=%p timeline=%d value=%llu submit=%u signal=%u\n",
                                static_cast<void*>(semaphore),
                                semaphore->timeline ? 1 : 0,
                                static_cast<unsigned long long>(
                                    semaphore->timeline
                                        ? semaphore->value
                                        : (semaphore->signaled ? 1U : 0U)
                                ),
                                submitIndex,
                                index
                            );
                        }
                    }
                }
            }
            if (fenceState != nullptr) fenceState->signaled = true;
            if (deferSceneRayTracingUntilManifest) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: deferred pre-manifest Isaac scene RTX traces=%zu unknownTraces=%zu clearedStorageImages=%zu fence=%llu\n",
                    initializationTraces.size(),
                    unknownKHRTraceCount,
                    clearedImages.size(),
                    static_cast<unsigned long long>(bridgeFence)
                );
            } else {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: executed Metal-backed null-TLAS RTX initialization traces=%zu clearedStorageImages=%zu fence=%llu\n",
                    initializationTraces.size(),
                    clearedImages.size(),
                    static_cast<unsigned long long>(bridgeFence)
                );
            }
            return VK_SUCCESS;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "imb-vulkan-icd: null-TLAS RTX initialization failed: %s\n", error.what());
            return VK_ERROR_DEVICE_LOST;
        }
    }
    if (containsUnsupportedRayTracing) {
        std::fprintf(
            stderr,
            "imb-vulkan-icd: ray-tracing submit rejected: Metal ray dispatch/copy is not connected yet nvBuilds=%zu nvCopies=%zu nvTraces=%zu khrCopies=%zu khrTraces=%zu unresolvedKHRTraces=%zu unknownKHRTraces=%zu nullSceneTraces=%zu sceneTraces=%zu\n",
            unsupportedNVBuildCount,
            unsupportedNVCopyCount,
            unsupportedNVTraceCount,
            unsupportedKHRCopyCount,
            totalKHRTraceCount,
            unresolvedKHRTraceCount,
            unknownKHRTraceCount,
            nullSceneRTXTraces.size(),
            sceneRayTraces.size()
        );
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }

    if (!metalRayTraces.empty()) {
        if (!gState.bridge
            || (gState.bridge->capabilities().bits & IMB_CAP_METAL_RAY_DISPATCH) == 0) {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        for (const auto* currentRayTrace : metalRayTraces) {
        // A single Kit queue submit can contain both the interactive viewport
        // and one or more Camera Render Products. Process each recognized
        // scene trace instead of silently retaining only the first dispatch.
        metalRayTrace = currentRayTrace;
        ImageState* targetImage = nullptr;
        AccelerationStructureKHRState* topLevel = nullptr;
        for (auto* set : metalRayTrace->descriptorSets) {
            if (set == nullptr || !gState.descriptorSets.contains(set)) continue;
            for (const auto& [_, image] : set->storageImages) {
                if (image != nullptr && gState.images.contains(image)
                    && image->extent.width == metalRayTrace->width
                    && image->extent.height == metalRayTrace->height
                    && image->format == VK_FORMAT_R8G8B8A8_UNORM
                    && image->memory != nullptr) {
                    targetImage = image;
                    break;
                }
            }
            for (const auto& [_, acceleration] : set->accelerationStructuresKHR) {
                if (acceleration != nullptr && gState.accelerationStructuresKHR.contains(acceleration)
                    && acceleration->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                    && acceleration->built && acceleration->bridgeResourceID != 0) {
                    topLevel = acceleration;
                    break;
                }
            }
        }
        if (targetImage == nullptr) {
            std::uint64_t newestSequence = 0;
            for (auto* image : gState.images) {
                if (image == nullptr || image->device != device
                    || image->extent.width != metalRayTrace->width
                    || image->extent.height != metalRayTrace->height
                    || image->format != VK_FORMAT_R8G8B8A8_UNORM
                    || image->memory == nullptr) {
                    continue;
                }
                if (rayTracingTraceEnabled()) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: Metal scene global image candidate=%p sequence=%llu memory=%p offset=%llu size=%llu host=%llu\n",
                        static_cast<void*>(image),
                        static_cast<unsigned long long>(image->storageDescriptorSequence),
                        static_cast<void*>(image->memory),
                        static_cast<unsigned long long>(image->memoryOffset),
                        static_cast<unsigned long long>(image->size),
                        static_cast<unsigned long long>(image->resourceID)
                    );
                }
                if (targetImage == nullptr
                    || image->storageDescriptorSequence >= newestSequence) {
                    targetImage = image;
                    newestSequence = image->storageDescriptorSequence;
                }
            }
        }
        if (topLevel == nullptr) {
            for (auto* acceleration : gState.accelerationStructuresKHR) {
                if (acceleration != nullptr && acceleration->device == device
                    && acceleration->type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
                    && acceleration->built && acceleration->bridgeResourceID != 0) {
                    topLevel = acceleration;
                    break;
                }
            }
        }
        if (targetImage == nullptr || topLevel == nullptr || targetImage->memory == nullptr) {
            if (rayTracingTraceEnabled()) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: Metal scene dispatch missing target=%p tlas=%p targetMemory=%p\n",
                    static_cast<void*>(targetImage),
                    static_cast<void*>(topLevel),
                    static_cast<void*>(targetImage == nullptr ? nullptr : targetImage->memory)
                );
            }
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        const std::uint64_t byteCount = static_cast<std::uint64_t>(targetImage->extent.width)
            * targetImage->extent.height * 4;
        if (targetImage->memoryOffset > targetImage->memory->size
            || byteCount > targetImage->memory->size - targetImage->memoryOffset) {
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        try {
            const VkResult memoryResult = ensureMemoryBackingLocked(targetImage->memory, false);
            if (memoryResult != VK_SUCCESS) return memoryResult;
            const VkResult imageResult = ensureImageBackingLocked(targetImage);
            if (imageResult != VK_SUCCESS) return imageResult;
            const bool isSceneRayTrace = std::find(
                sceneRayTraces.begin(),
                sceneRayTraces.end(),
                metalRayTrace
            ) != sceneRayTraces.end();
            const bool isViewportSceneRayTrace = isSceneRayTrace
                && metalRayTrace->width == 1280
                && metalRayTrace->height == 720;
            std::optional<BridgeRayCamera> liveCamera =
                isSceneRayTrace
                ? readLiveCameraState()
                : std::nullopt;
            BridgeSceneTextureReferenceGuard lightTextureReferences;
            if (liveCamera.has_value()) {
                for (auto& light : liveCamera->additionalLights) {
                    const auto acquireLightTexture = [device,
                        &lightTextureReferences](
                        std::uint32_t width,
                        std::uint32_t height,
                        const std::shared_ptr<
                            const std::vector<std::uint8_t>
                        >& rgba8
                    ) {
                        BridgeSceneTextureMap texture{};
                        texture.width = width;
                        texture.height = height;
                        texture.rgba8 = rgba8;
                        const std::uint64_t resourceID =
                            acquireSceneTextureResourceLocked(device, texture);
                        if (resourceID != 0) {
                            lightTextureReferences.resourceIDs.push_back(
                                resourceID
                            );
                        }
                        return resourceID;
                    };
                    if ((light.textureFlags
                            & IMB_TRACE_LIGHT_TEXTURE_RECT_EMISSION) != 0) {
                        light.emissionTextureResourceID = acquireLightTexture(
                            light.emissionTextureWidth,
                            light.emissionTextureHeight,
                            light.emissionTextureRGBA8
                        );
                        if (light.emissionTextureResourceID == 0) {
                            throw std::runtime_error(
                                "RectLight texture Metal resource creation failed"
                            );
                        }
                    }
                    if ((light.textureFlags
                            & IMB_TRACE_LIGHT_TEXTURE_IES_PROFILE) != 0) {
                        light.iesTextureResourceID = acquireLightTexture(
                            light.iesTextureWidth,
                            light.iesTextureHeight,
                            light.iesTextureRGBA8
                        );
                        if (light.iesTextureResourceID == 0) {
                            throw std::runtime_error(
                                "IES profile Metal resource creation failed"
                            );
                        }
                    }
                }
            }
            if (rayTracingTraceEnabled() && liveCamera.has_value()) {
                static std::uint64_t lastLoggedCameraSequence = 0;
                if (liveCamera->sequence != lastLoggedCameraSequence) {
                    lastLoggedCameraSequence = liveCamera->sequence;
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: live Kit camera sequence=%llu position=(%.3f,%.3f,%.3f) forward=(%.3f,%.3f,%.3f) fov=%.4f near=%.4f far=%.1f\n",
                        static_cast<unsigned long long>(liveCamera->sequence),
                        liveCamera->position[0],
                        liveCamera->position[1],
                        liveCamera->position[2],
                        liveCamera->forward[0],
                        liveCamera->forward[1],
                        liveCamera->forward[2],
                        liveCamera->verticalFOVRadians,
                        liveCamera->nearDistance,
                        liveCamera->farDistance
                    );
                    if (liveCamera->hasSphereLight) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: live USD SphereLight position=(%.3f,%.3f,%.3f) color=(%.3f,%.3f,%.3f) intensity=%.3f radius=%.3f\n",
                            liveCamera->sphereLightPosition[0],
                            liveCamera->sphereLightPosition[1],
                            liveCamera->sphereLightPosition[2],
                            liveCamera->sphereLightColor[0],
                            liveCamera->sphereLightColor[1],
                            liveCamera->sphereLightColor[2],
                            liveCamera->sphereLightIntensity,
                            liveCamera->sphereLightRadius
                        );
                    }
                    if (liveCamera->hasDistantLight) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: live USD DistantLight direction=(%.3f,%.3f,%.3f) color=(%.3f,%.3f,%.3f) intensity=%.3f angle=%.3fdeg\n",
                            liveCamera->distantLightDirection[0],
                            liveCamera->distantLightDirection[1],
                            liveCamera->distantLightDirection[2],
                            liveCamera->distantLightColor[0],
                            liveCamera->distantLightColor[1],
                            liveCamera->distantLightColor[2],
                            liveCamera->distantLightIntensity,
                            liveCamera->distantLightAngleDegrees
                        );
                    }
                    if (liveCamera->hasDomeLight) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: live USD DomeLight color=(%.3f,%.3f,%.3f) intensity=%.3f\n",
                            liveCamera->domeLightColor[0],
                            liveCamera->domeLightColor[1],
                            liveCamera->domeLightColor[2],
                            liveCamera->domeLightIntensity
                        );
                    }
                    if (!liveCamera->additionalLights.empty()) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: live %s USD light list count=%zu",
                            liveCamera->completeLightList ? "complete" : "additional",
                            liveCamera->additionalLights.size()
                        );
                        for (const auto& light : liveCamera->additionalLights) {
                            std::fprintf(
                                stderr,
                                " kind=%u schema=%u pathHash=0x%llx",
                                light.kind,
                                light.schema,
                                static_cast<unsigned long long>(light.pathHash)
                            );
                            if (light.kind == IMB_TRACE_LIGHT_KIND_POSITIONAL
                                && (light.schema == IMB_TRACE_LIGHT_SCHEMA_RECT
                                    || light.schema == IMB_TRACE_LIGHT_SCHEMA_DISK
                                    || light.schema
                                        == IMB_TRACE_LIGHT_SCHEMA_CYLINDER)) {
                                std::fprintf(
                                    stderr,
                                    " axisU=(%.3f,%.3f,%.3f)x%.3f axisV=(%.3f,%.3f,%.3f)x%.3f",
                                    light.values[8],
                                    light.values[9],
                                    light.values[10],
                                    light.values[11],
                                    light.values[12],
                                    light.values[13],
                                    light.values[14],
                                    light.values[15]
                                );
                                if (light.schema
                                    == IMB_TRACE_LIGHT_SCHEMA_CYLINDER) {
                                    std::fprintf(
                                        stderr,
                                        " radialW=%.3f",
                                        light.values[7]
                                    );
                                }
                            }
                            if ((light.shapingFlags
                                    & IMB_TRACE_LIGHT_SHAPING_APPLIED) != 0) {
                                std::fprintf(
                                    stderr,
                                    " shapingAxis=(%.3f,%.3f,%.3f) cone=%.3f softness=%.3f focus=%.3f focusTint=(%.3f,%.3f,%.3f)",
                                    light.shapingAxis[0],
                                    light.shapingAxis[1],
                                    light.shapingAxis[2],
                                    light.shapingConeAngleDegrees,
                                    light.shapingConeSoftness,
                                    light.shapingFocus,
                                    light.shapingFocusTint[0],
                                    light.shapingFocusTint[1],
                                    light.shapingFocusTint[2]
                                );
                            }
                            if (light.textureFlags != 0) {
                                std::fprintf(
                                    stderr,
                                    " lightTextures=%#x rect=%llu ies=%llu angleScale=%.3f multiplier=%.6f",
                                    light.textureFlags,
                                    static_cast<unsigned long long>(
                                        light.emissionTextureResourceID
                                    ),
                                    static_cast<unsigned long long>(
                                        light.iesTextureResourceID
                                    ),
                                    light.iesAngleScale,
                                    light.iesMultiplier
                                );
                            }
                        }
                        std::fputc('\n', stderr);
                    }
                }
            }
            const std::uint64_t bridgeFence = gState.bridge->submitRayTrace(
                targetImage->resourceID,
                topLevel->bridgeResourceID,
                metalRayTrace->width,
                metalRayTrace->height,
                isSceneRayTrace
                    ? UINT32_C(0xff302018)
                    : UINT32_C(0xff000000),
                isSceneRayTrace
                    ? (sceneGridPresentationEnabled()
                        ? UINT32_C(0xffe08c31)
                        : UINT32_C(0xffe08c30))
                    : UINT32_C(0xff00ff00),
                liveCamera.has_value() ? &*liveCamera : nullptr
            );
            gState.bridge->waitFence(bridgeFence);
            gState.bridge->readImage(
                targetImage->resourceID,
                targetImage->memory->bytes.data() + targetImage->memoryOffset,
                byteCount
            );
            const char* sensorFramePath =
                std::getenv("IMB_CAMERA_SENSOR_FRAME_FILE");
            if (isSceneRayTrace && liveCamera.has_value()
                && sensorFramePath != nullptr && sensorFramePath[0] != '\0'
                && !gState.cameraSensorFrameAttempted) {
                gState.cameraSensorFrameAttempted = true;
                gState.cameraSensorFrameDevice = device;
                gState.cameraSensorFramePublished = publishMetalCameraSensorFrame(
                    sensorFramePath,
                    targetImage->extent.width,
                    targetImage->extent.height,
                    targetImage->memory->bytes.data() + targetImage->memoryOffset
                );
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: Metal Camera sensor frame publication %s path=%s sourceResolution=%ux%u requestedResolution=%sx%s bytes=%llu\n",
                    gState.cameraSensorFramePublished ? "passed" : "failed",
                    sensorFramePath == nullptr ? "(unset)" : sensorFramePath,
                    targetImage->extent.width,
                    targetImage->extent.height,
                    std::getenv("IMB_CAMERA_SENSOR_WIDTH") == nullptr
                        ? "(source)" : std::getenv("IMB_CAMERA_SENSOR_WIDTH"),
                    std::getenv("IMB_CAMERA_SENSOR_HEIGHT") == nullptr
                        ? "(source)" : std::getenv("IMB_CAMERA_SENSOR_HEIGHT"),
                    static_cast<unsigned long long>(byteCount)
                );
            }
            std::uint64_t hitPixelCount = 0;
            if (traceEnabled() || rayTracingTraceEnabled()) {
                const auto* pixels = targetImage->memory->bytes.data()
                    + static_cast<std::size_t>(targetImage->memoryOffset);
                const std::uint32_t missRGBA8 = isSceneRayTrace
                    ? UINT32_C(0xff302018)
                    : UINT32_C(0xff000000);
                std::array<std::uint8_t, 4> missBytes{};
                std::memcpy(missBytes.data(), &missRGBA8, missBytes.size());
                for (std::uint64_t offset = 0; offset < byteCount; offset += 4) {
                    if (std::memcmp(pixels + offset, missBytes.data(), missBytes.size()) != 0) {
                        ++hitPixelCount;
                    }
                }
            }
            const VkResult syncResult = syncMemoryToExternalFDLocked(
                targetImage->memory,
                targetImage->memoryOffset,
                byteCount
            );
            if (syncResult != VK_SUCCESS) return syncResult;
            if (isViewportSceneRayTrace && scenePresentationEnabled()) {
                const auto* sceneBytes = targetImage->memory->bytes.data()
                    + static_cast<std::size_t>(targetImage->memoryOffset);
                gState.latestMetalSceneDevice = device;
                gState.latestMetalSceneWidth = targetImage->extent.width;
                gState.latestMetalSceneHeight = targetImage->extent.height;
                gState.latestMetalSceneRGBA8.assign(
                    sceneBytes,
                    sceneBytes + static_cast<std::size_t>(byteCount)
                );
            }
            targetImage->dirty = true;
            if (traceEnabled() || rayTracingTraceEnabled()) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: submitted real Metal %s %ux%u tlas=%llu image=%llu byteCount=%llu hitPixels=%llu\n",
                    isSceneRayTrace
                        ? (isViewportSceneRayTrace
                            ? "Isaac viewport scene ray dispatch"
                            : "Isaac camera render-product ray dispatch")
                        : "ray probe",
                    metalRayTrace->width,
                    metalRayTrace->height,
                    static_cast<unsigned long long>(topLevel->bridgeResourceID),
                    static_cast<unsigned long long>(targetImage->resourceID),
                    static_cast<unsigned long long>(byteCount),
                    static_cast<unsigned long long>(hitPixelCount)
                );
            }
        } catch (const std::exception& error) {
            std::fprintf(stderr, "imb-vulkan-icd: Metal ray probe submit failed: %s\n", error.what());
            return VK_ERROR_DEVICE_LOST;
        }
        // Continue with any additional Camera Render Product in this submit.
        }
        const VkResult semaphoreResult = completeQueueSubmitSynchronizationLocked(
            device,
            submitCount,
            submits
        );
        if (semaphoreResult != VK_SUCCESS) return semaphoreResult;
        if (fenceState != nullptr) fenceState->signaled = true;
        return VK_SUCCESS;
    }

    if (!isMetalAddSubmit && !isMetalRasterSubmit && !isMetalUISubmit) {
        for (std::uint32_t submitIndex = 0; submitIndex < submitCount; ++submitIndex) {
            const auto& submit = submits[submitIndex];
            const VkTimelineSemaphoreSubmitInfo* timelineInfo = nullptr;
            auto* next = reinterpret_cast<const VkBaseInStructure*>(submit.pNext);
            while (next != nullptr) {
                if (next->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
                    timelineInfo = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(next);
                    break;
                }
                next = next->pNext;
            }
            if ((submit.commandBufferCount != 0 && submit.pCommandBuffers == nullptr)
                || (submit.waitSemaphoreCount != 0 && submit.pWaitSemaphores == nullptr)
                || (submit.signalSemaphoreCount != 0 && submit.pSignalSemaphores == nullptr)) {
                return VK_ERROR_INITIALIZATION_FAILED;
            }
            for (std::uint32_t index = 0; index < submit.commandBufferCount; ++index) {
                auto* command = reinterpret_cast<CommandBufferState*>(submit.pCommandBuffers[index]);
                if (!gState.commandBuffers.contains(command) || !command->executable
                    || command->device != device) {
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
            }
            for (std::uint32_t index = 0; index < submit.waitSemaphoreCount; ++index) {
                auto* semaphore = objectState<SemaphoreState>(submit.pWaitSemaphores[index]);
                if (!gState.semaphores.contains(semaphore) || semaphore->device != device) {
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
                if (!semaphore->timeline) semaphore->signaled = false;
            }
            for (std::uint32_t index = 0; index < submit.signalSemaphoreCount; ++index) {
                auto* semaphore = objectState<SemaphoreState>(submit.pSignalSemaphores[index]);
                if (!gState.semaphores.contains(semaphore) || semaphore->device != device) {
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
                if (semaphore->timeline) {
                    const bool hasValue = timelineInfo != nullptr
                        && index < timelineInfo->signalSemaphoreValueCount
                        && timelineInfo->pSignalSemaphoreValues != nullptr;
                    const std::uint64_t signalValue = hasValue
                        ? timelineInfo->pSignalSemaphoreValues[index]
                        : semaphore->value + 1;
                    semaphore->value = std::max(semaphore->value, signalValue);
                }
                semaphore->signaled = true;
            }
        }
        if (fenceState != nullptr) fenceState->signaled = true;
        return VK_SUCCESS;
    }

    if (!gState.bridge) return VK_ERROR_DEVICE_LOST;
    if (isMetalUISubmit) {
        if (uiPresentationTraceEnabled() && uiCommands.size() > 1) {
            std::fprintf(
                stderr,
                "imb-vulkan-icd: Kit UI submit candidates=%zu selectedDraws=%zu\n",
                uiCommands.size(),
                uiCommand->uiDraws.size()
            );
            for (const auto* candidate : uiCommands) {
                const auto* candidateImage = candidate->framebuffer->colorImage;
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: Kit UI candidate command=%p extent=%ux%u draws=%zu selected=%d\n",
                    static_cast<const void*>(candidate),
                    candidateImage->extent.width,
                    candidateImage->extent.height,
                    candidate->uiDraws.size(),
                    candidate == uiCommand ? 1 : 0
                );
            }
        }
        auto* image = uiCommand->framebuffer->colorImage;
        const std::uint32_t rootWidth = image->extent.width;
        const std::uint32_t rootHeight = image->extent.height;
        const bool layoutReady = fullWorkspaceLayoutReady();
        bool hasRightDock = false;
        bool hasBottomDock = false;
        std::uint64_t layoutSignature = 1469598103934665603ULL;
        for (const auto& draw : uiCommand->uiDraws) {
            const std::int64_t x = std::max<std::int64_t>(draw.scissor.offset.x, 0);
            const std::int64_t y = std::max<std::int64_t>(draw.scissor.offset.y, 0);
            const std::uint64_t width = draw.scissor.extent.width;
            const std::uint64_t height = draw.scissor.extent.height;
            const std::uint64_t components[] = {
                static_cast<std::uint64_t>(x),
                static_cast<std::uint64_t>(y),
                width,
                height,
            };
            for (const auto component : components) {
                layoutSignature ^= component;
                layoutSignature *= 1099511628211ULL;
            }
            hasRightDock = hasRightDock || (
                x * 100 >= static_cast<std::int64_t>(rootWidth) * 60
                && width * 100 <= static_cast<std::uint64_t>(rootWidth) * 45
                && height * 100 >= static_cast<std::uint64_t>(rootHeight) * 25
            );
            hasBottomDock = hasBottomDock || (
                y * 100 >= static_cast<std::int64_t>(rootHeight) * 50
                && x * 100 <= static_cast<std::int64_t>(rootWidth) * 10
                && width * 100 >= static_cast<std::uint64_t>(rootWidth) * 45
                && height * 100 >= static_cast<std::uint64_t>(rootHeight) * 15
            );
        }
        auto completeSkippedUISubmit = [&]() -> VkResult {
            const VkResult semaphoreResult = completeQueueSubmitSynchronizationLocked(
                device,
                submitCount,
                submits
            );
            if (semaphoreResult != VK_SUCCESS) return semaphoreResult;
            if (fenceState != nullptr) {
                fenceState->bridgeFenceID = 0;
                fenceState->resultMemory = nullptr;
                fenceState->resultImage = nullptr;
                fenceState->submitted = false;
                fenceState->signaled = true;
            }
            return VK_SUCCESS;
        };
        if (uiSnapshotLayoutChangesEnabled()) {
            static std::unordered_set<std::uint64_t> capturedLayoutSignatures;
            if (!capturedLayoutSignatures.insert(layoutSignature).second) {
                return completeSkippedUISubmit();
            }
            std::fprintf(
                stderr,
                "imb-vulkan-icd: snapshot Kit UI root %ux%u draws=%zu target=%llu rightDock=%d bottomDock=%d signature=%#llx\n",
                rootWidth,
                rootHeight,
                uiCommand->uiDraws.size(),
                static_cast<unsigned long long>(image->resourceID),
                hasRightDock ? 1 : 0,
                hasBottomDock ? 1 : 0,
                static_cast<unsigned long long>(layoutSignature)
            );
        }
        if (fullWorkspaceOnlyEnabled()) {
            if (!layoutReady || !hasRightDock || !hasBottomDock) {
                static std::uint64_t lastSkippedLayoutSignature = 0;
                if (uiPresentationTraceEnabled()
                    && layoutSignature != lastSkippedLayoutSignature) {
                    lastSkippedLayoutSignature = layoutSignature;
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: skipped Kit UI root %ux%u draws=%zu layoutReady=%d rightDock=%d bottomDock=%d signature=%#llx\n",
                        rootWidth,
                        rootHeight,
                        uiCommand->uiDraws.size(),
                        layoutReady ? 1 : 0,
                        hasRightDock ? 1 : 0,
                        hasBottomDock ? 1 : 0,
                        static_cast<unsigned long long>(layoutSignature)
                    );
                }
                return completeSkippedUISubmit();
            }
            static std::uint64_t lastSelectedLayoutSignature = 0;
            if (uiPresentationTraceEnabled()
                && layoutSignature != lastSelectedLayoutSignature) {
                lastSelectedLayoutSignature = layoutSignature;
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: selected Isaac Full workspace root %ux%u draws=%zu signature=%#llx\n",
                    rootWidth,
                    rootHeight,
                    uiCommand->uiDraws.size(),
                    static_cast<unsigned long long>(layoutSignature)
                );
            }
        }
        auto* vertexBuffer = uiCommand->vertexBuffer;
        auto* indexBuffer = uiCommand->indexBuffer;
        if (image == nullptr || image->format != VK_FORMAT_B8G8R8A8_UNORM
            || vertexBuffer == nullptr || indexBuffer == nullptr
            || vertexBuffer->memory == nullptr || indexBuffer->memory == nullptr
            || uiCommand->indexType != VK_INDEX_TYPE_UINT32) {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        try {
            const VkResult imageResult = ensureImageBackingLocked(image);
            if (imageResult != VK_SUCCESS) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: Kit UI root image unavailable result=%d format=%d type=%d extent=%ux%ux%u mips=%u layers=%u samples=%#x\n",
                    imageResult,
                    image->format,
                    image->type,
                    image->extent.width,
                    image->extent.height,
                    image->extent.depth,
                    image->mipLevels,
                    image->arrayLayers,
                    image->samples
                );
                return imageResult;
            }
            const VkResult vertexResult = ensureMemoryBackingLocked(vertexBuffer->memory, true);
            if (vertexResult != VK_SUCCESS) return vertexResult;
            const VkResult indexResult = ensureMemoryBackingLocked(indexBuffer->memory, true);
            if (indexResult != VK_SUCCESS) return indexResult;

            std::array<DeviceMemoryState*, 2> drawMemories{
                vertexBuffer->memory,
                indexBuffer->memory,
            };
            for (std::size_t memoryIndex = 0; memoryIndex < drawMemories.size(); ++memoryIndex) {
                auto* memory = drawMemories[memoryIndex];
                if (memoryIndex != 0 && memory == drawMemories[0]) continue;
                if (memory->dirtyOffset != VK_WHOLE_SIZE && memory->dirtyEnd > memory->dirtyOffset) {
                    gState.bridge->writeBuffer(
                        memory->resourceID,
                        memory->bytes.data() + memory->dirtyOffset,
                        memory->dirtyEnd - memory->dirtyOffset,
                        memory->dirtyOffset
                    );
                    memory->dirtyOffset = VK_WHOLE_SIZE;
                    memory->dirtyEnd = 0;
                }
            }

            std::unordered_set<ImageState*> uploadedTextures;
            std::unordered_set<ImageState*> scenePresentationTextures;
            std::vector<BridgeUIDraw> bridgeDraws;
            bridgeDraws.reserve(uiCommand->uiDraws.size());
            for (const auto& draw : uiCommand->uiDraws) {
                std::uint64_t textureID = 0;
                const bool hasValidTexture = draw.texture != nullptr
                    && gState.images.contains(draw.texture);
                if (rayTracingTraceEnabled()
                    && draw.scissor.extent.width >= 600
                    && draw.scissor.extent.height >= 300) {
                    std::fprintf(
                        stderr,
                        "imb-vulkan-icd: Kit UI large draw texture=%p format=%d extent=%ux%u host=%llu sequence=%llu dirty=%d scissor=%d,%d %ux%u\n",
                        static_cast<void*>(draw.texture),
                        !hasValidTexture ? VK_FORMAT_UNDEFINED : draw.texture->format,
                        !hasValidTexture ? 0U : draw.texture->extent.width,
                        !hasValidTexture ? 0U : draw.texture->extent.height,
                        static_cast<unsigned long long>(
                            !hasValidTexture ? 0 : draw.texture->resourceID
                        ),
                        static_cast<unsigned long long>(
                            !hasValidTexture ? 0 : draw.texture->storageDescriptorSequence
                        ),
                        hasValidTexture && draw.texture->dirty ? 1 : 0,
                        draw.scissor.offset.x,
                        draw.scissor.offset.y,
                        draw.scissor.extent.width,
                        draw.scissor.extent.height
                    );
                }
                if (hasValidTexture
                    && draw.texture->format == VK_FORMAT_R8G8B8A8_UNORM
                    && draw.texture->memory != nullptr) {
                    const VkResult textureMemoryResult = ensureMemoryBackingLocked(draw.texture->memory, false);
                    if (textureMemoryResult != VK_SUCCESS) return textureMemoryResult;
                    const VkResult textureImageResult = ensureImageBackingLocked(draw.texture);
                    if (textureImageResult != VK_SUCCESS) {
                        std::fprintf(
                            stderr,
                            "imb-vulkan-icd: Kit UI texture unavailable result=%d format=%d type=%d extent=%ux%ux%u mips=%u layers=%u samples=%#x\n",
                            textureImageResult,
                            draw.texture->format,
                            draw.texture->type,
                            draw.texture->extent.width,
                            draw.texture->extent.height,
                            draw.texture->extent.depth,
                            draw.texture->mipLevels,
                            draw.texture->arrayLayers,
                            draw.texture->samples
                        );
                        return textureImageResult;
                    }
                    const std::uint64_t textureBytes =
                        static_cast<std::uint64_t>(draw.texture->extent.width)
                        * draw.texture->extent.height * 4;
                    const bool isLargeViewportTexture =
                        draw.scissor.extent.width >= 600
                        && draw.scissor.extent.height >= 300
                        // Kit's RGBA font/icon atlas can also be used by a
                        // full-window draw, but it is a tall 1024x3240 image.
                        // Only landscape render products are viewports.
                        && static_cast<std::uint64_t>(draw.texture->extent.width) * 3
                            >= static_cast<std::uint64_t>(draw.texture->extent.height) * 4;
                    if (emptyStageGridPresentationEnabled()
                        && isLargeViewportTexture
                        && scenePresentationTextures.insert(draw.texture).second) {
                        if (draw.texture->memoryOffset > draw.texture->memory->size
                            || textureBytes > draw.texture->memory->size
                                - draw.texture->memoryOffset) {
                            return VK_ERROR_MEMORY_MAP_FAILED;
                        }
                        const std::optional<BridgeRayCamera> gridCamera =
                            readLiveCameraState();
                        const std::uint64_t gridFence = gState.bridge->submitRayTrace(
                            draw.texture->resourceID,
                            0,
                            draw.texture->extent.width,
                            draw.texture->extent.height,
                            0,
                            0,
                            gridCamera.has_value() ? &*gridCamera : nullptr,
                            true
                        );
                        gState.bridge->waitFence(gridFence);
                        gState.bridge->readImage(
                            draw.texture->resourceID,
                            draw.texture->memory->bytes.data()
                                + draw.texture->memoryOffset,
                            textureBytes
                        );
                        const VkResult gridSyncResult = syncMemoryToExternalFDLocked(
                            draw.texture->memory,
                            draw.texture->memoryOffset,
                            textureBytes
                        );
                        if (gridSyncResult != VK_SUCCESS) return gridSyncResult;
                        draw.texture->dirty = true;
                        bool& gridLogEmitted = gridCamera.has_value()
                            ? gState.emptyStageGridLoggedWithCamera
                            : gState.emptyStageGridLoggedWithoutCamera;
                        if (!gridLogEmitted) {
                            std::fprintf(
                                stderr,
                                "imb-vulkan-icd: rendered Metal empty-stage grid into Kit viewport image=%llu extent=%ux%u liveCamera=%d scissor=%d,%d %ux%u\n",
                                static_cast<unsigned long long>(draw.texture->resourceID),
                                draw.texture->extent.width,
                                draw.texture->extent.height,
                                gridCamera.has_value() ? 1 : 0,
                                draw.scissor.offset.x,
                                draw.scissor.offset.y,
                                draw.scissor.extent.width,
                                draw.scissor.extent.height
                            );
                            gridLogEmitted = true;
                        }
                    } else if (gState.latestMetalSceneDevice == device
                        && draw.texture->extent.width == gState.latestMetalSceneWidth
                        && draw.texture->extent.height == gState.latestMetalSceneHeight
                        && isLargeViewportTexture
                        && gState.latestMetalSceneRGBA8.size() == textureBytes
                        && scenePresentationTextures.insert(draw.texture).second) {
                        if (draw.texture->memoryOffset > draw.texture->memory->size
                            || textureBytes > draw.texture->memory->size - draw.texture->memoryOffset) {
                            return VK_ERROR_MEMORY_MAP_FAILED;
                        }
                        std::memcpy(
                            draw.texture->memory->bytes.data() + draw.texture->memoryOffset,
                            gState.latestMetalSceneRGBA8.data(),
                            static_cast<std::size_t>(textureBytes)
                        );
                        const VkResult sceneSyncResult = syncMemoryToExternalFDLocked(
                            draw.texture->memory,
                            draw.texture->memoryOffset,
                            textureBytes
                        );
                        if (sceneSyncResult != VK_SUCCESS) return sceneSyncResult;
                        draw.texture->dirty = true;
                        if (rayTracingTraceEnabled()) {
                            std::fprintf(
                                stderr,
                                "imb-vulkan-icd: synchronized Metal scene into Kit viewport image=%llu extent=%ux%u scissor=%d,%d %ux%u\n",
                                static_cast<unsigned long long>(draw.texture->resourceID),
                                draw.texture->extent.width,
                                draw.texture->extent.height,
                                draw.scissor.offset.x,
                                draw.scissor.offset.y,
                                draw.scissor.extent.width,
                                draw.scissor.extent.height
                            );
                        }
                    }
                    if (draw.texture->dirty && uploadedTextures.insert(draw.texture).second) {
                        if (draw.texture->memoryOffset > draw.texture->memory->size
                            || textureBytes > draw.texture->memory->size - draw.texture->memoryOffset) {
                            return VK_ERROR_MEMORY_MAP_FAILED;
                        }
                        gState.bridge->writeImage(
                            draw.texture->resourceID,
                            draw.texture->memory->bytes.data() + draw.texture->memoryOffset,
                            textureBytes
                        );
                        draw.texture->dirty = false;
                    }
                    textureID = draw.texture->resourceID;
                }
                bridgeDraws.push_back(BridgeUIDraw{
                    textureID,
                    draw.indexCount,
                    draw.firstIndex,
                    draw.vertexOffset,
                    static_cast<std::uint32_t>(std::max(draw.scissor.offset.x, 0)),
                    static_cast<std::uint32_t>(std::max(draw.scissor.offset.y, 0)),
                    draw.scissor.extent.width,
                    draw.scissor.extent.height,
                });
            }

            const std::uint64_t bridgeFence = gState.bridge->submitIndexedUI(
                image->resourceID,
                vertexBuffer->memory->resourceID,
                indexBuffer->memory->resourceID,
                vertexBuffer->memoryOffset + uiCommand->vertexBufferOffset,
                indexBuffer->memoryOffset + uiCommand->indexBufferOffset,
                image->extent.width,
                image->extent.height,
                uiCommand->clearRGBA8,
                bridgeDraws
            );
            // Kit reuses its ring buffers immediately after the queue fence is
            // observed. Complete the Metal work here so every presented UI
            // frame is durable on the host and the shared upload range cannot
            // be overwritten while Metal is still reading it.
            gState.bridge->waitFence(bridgeFence);
            // A Kit UI submit can also contain RTX command buffers that this
            // fallback path does not execute. Complete its visible Metal UI
            // work conservatively instead of advancing a timeline directly to
            // the submit's final value and releasing those RTX dependencies
            // before their producer exists.
            for (std::uint32_t submitIndex = 0;
                 submitIndex < submitCount;
                 ++submitIndex) {
                const auto& submit = submits[submitIndex];
                for (std::uint32_t index = 0;
                     index < submit.waitSemaphoreCount;
                     ++index) {
                    auto* semaphore = objectState<SemaphoreState>(
                        submit.pWaitSemaphores[index]
                    );
                    if (gState.semaphores.contains(semaphore)
                        && !semaphore->timeline) {
                        semaphore->signaled = false;
                    }
                }
                for (std::uint32_t index = 0;
                     index < submit.signalSemaphoreCount;
                     ++index) {
                    auto* semaphore = objectState<SemaphoreState>(
                        submit.pSignalSemaphores[index]
                    );
                    if (gState.semaphores.contains(semaphore)) {
                        semaphore->signaled = true;
                        if (semaphore->timeline) ++semaphore->value;
                        if (semaphoreTraceEnabled()) {
                            std::fprintf(
                                stderr,
                                "imb-vulkan-icd: semaphore conservative UI completion handle=%p timeline=%d value=%llu submit=%u signal=%u\n",
                                static_cast<void*>(semaphore),
                                semaphore->timeline ? 1 : 0,
                                static_cast<unsigned long long>(
                                    semaphore->timeline
                                        ? semaphore->value
                                        : (semaphore->signaled ? 1U : 0U)
                                ),
                                submitIndex,
                                index
                            );
                        }
                    }
                }
            }
            if (fenceState != nullptr) {
                fenceState->bridgeFenceID = 0;
                fenceState->resultMemory = nullptr;
                fenceState->resultImage = nullptr;
                fenceState->submitted = false;
                fenceState->signaled = true;
            }
            if (uiPresentationTraceEnabled()) {
                std::fprintf(
                    stderr,
                    "imb-vulkan-icd: submitted real Kit UI frame %ux%u draws=%zu target=%llu fence=%llu signature=%#llx\n",
                    image->extent.width,
                    image->extent.height,
                    bridgeDraws.size(),
                    static_cast<unsigned long long>(image->resourceID),
                    static_cast<unsigned long long>(bridgeFence),
                    static_cast<unsigned long long>(layoutSignature)
                );
            }
            return VK_SUCCESS;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "imb-vulkan-icd: Kit UI queue submit failed: %s\n", error.what());
            return VK_ERROR_DEVICE_LOST;
        }
    }
    if (isMetalRasterSubmit) {
        auto* image = rasterCommand->framebuffer->colorImage;
        auto* memory = image->memory;
        const std::uint64_t byteCount = static_cast<std::uint64_t>(image->extent.width)
            * image->extent.height * 4;
        if (memory == nullptr || image->format != VK_FORMAT_R8G8B8A8_UNORM
            || image->extent.depth != 1 || rasterCommand->framebuffer->width != image->extent.width
            || rasterCommand->framebuffer->height != image->extent.height
            || image->memoryOffset > memory->size || byteCount > memory->size - image->memoryOffset) {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        try {
            const VkResult memoryResult = ensureMemoryBackingLocked(memory, false);
            if (memoryResult != VK_SUCCESS) return memoryResult;
            const VkResult imageResult = ensureImageBackingLocked(image);
            if (imageResult != VK_SUCCESS) return imageResult;
            const std::uint64_t bridgeFence = gState.bridge->submitTriangle(
                image->resourceID,
                rasterCommand->clearRGBA8
            );
            if (fenceState != nullptr) {
                fenceState->bridgeFenceID = bridgeFence;
                fenceState->resultMemory = nullptr;
                fenceState->resultImage = image;
                fenceState->submitted = true;
            } else {
                gState.bridge->waitFence(bridgeFence);
                gState.bridge->readImage(
                    image->resourceID,
                    memory->bytes.data() + image->memoryOffset,
                    byteCount
                );
            }
            return VK_SUCCESS;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "imb-vulkan-icd: raster queue submit failed: %s\n", error.what());
            return VK_ERROR_DEVICE_LOST;
        }
    }

    auto* command = addCommand;
    auto* buffer = command->descriptorSet->buffer;
    auto* memory = buffer->memory;
    const VkDeviceSize byteCount = static_cast<VkDeviceSize>(command->groupCountX) * sizeof(std::uint32_t);
    if (memory == nullptr || buffer->memoryOffset != 0 || command->descriptorSet->offset != 0
        || command->descriptorSet->range < byteCount || buffer->size < byteCount) {
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    try {
        const VkResult backingResult = ensureMemoryBackingLocked(memory, true);
        if (backingResult != VK_SUCCESS) return backingResult;
        gState.bridge->writeBuffer(memory->resourceID, memory->bytes.data(), memory->size);
        const std::uint64_t bridgeFence = gState.bridge->submitAdd(
            memory->resourceID,
            command->groupCountX,
            command->addend
        );
        if (fenceState != nullptr) {
            fenceState->bridgeFenceID = bridgeFence;
            fenceState->resultMemory = memory;
            fenceState->resultImage = nullptr;
            fenceState->submitted = true;
        } else {
            gState.bridge->waitFence(bridgeFence);
            gState.bridge->readBuffer(memory->resourceID, memory->bytes.data(), memory->size);
        }
        return VK_SUCCESS;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "imb-vulkan-icd: queue submit failed: %s\n", error.what());
        return VK_ERROR_DEVICE_LOST;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkDeviceWaitIdle(VkDevice device) {
    std::lock_guard lock(gState.mutex);
    if (!gState.devices.contains(device)) return VK_ERROR_DEVICE_LOST;
    for (auto* fence : gState.fences) {
        if (fence->device == device && fence->submitted && !fence->signaled) {
            const VkResult result = completeFenceLocked(fence);
            if (result != VK_SUCCESS) return result;
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL imb_vkQueueWaitIdle(VkQueue queue) {
    std::lock_guard lock(gState.mutex);
    const VkDevice device = deviceForQueue(queue);
    return device == VK_NULL_HANDLE ? VK_ERROR_DEVICE_LOST : imb_vkDeviceWaitIdle(device);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL imb_vkGetInstanceProcAddr(VkInstance instance, const char* name);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL imb_vkGetDeviceProcAddr(VkDevice device, const char* name) {
    if (name == nullptr) return nullptr;
    if (traceEnabled()) std::fprintf(stderr, "imb-vulkan-icd: vkGetDeviceProcAddr %s\n", name);
    if (std::strcmp(name, "vkDestroyDevice") == 0) return toVoidFunction(imb_vkDestroyDevice);
    if (std::strcmp(name, "vkGetDeviceQueue") == 0) return toVoidFunction(imb_vkGetDeviceQueue);
    if (std::strcmp(name, "vkGetDeviceQueue2") == 0) return toVoidFunction(imb_vkGetDeviceQueue2);
    if (std::strcmp(name, "vkDeviceWaitIdle") == 0) return toVoidFunction(imb_vkDeviceWaitIdle);
    if (std::strcmp(name, "vkQueueWaitIdle") == 0) return toVoidFunction(imb_vkQueueWaitIdle);
    if (std::strcmp(name, "vkQueueBindSparse") == 0) return toVoidFunction(imb_vkQueueBindSparse);
    if (std::strcmp(name, "vkCreateBuffer") == 0) return toVoidFunction(imb_vkCreateBuffer);
    if (std::strcmp(name, "vkDestroyBuffer") == 0) return toVoidFunction(imb_vkDestroyBuffer);
    if (std::strcmp(name, "vkGetBufferMemoryRequirements") == 0) return toVoidFunction(imb_vkGetBufferMemoryRequirements);
    if (std::strcmp(name, "vkGetBufferMemoryRequirements2") == 0
        || std::strcmp(name, "vkGetBufferMemoryRequirements2KHR") == 0) return toVoidFunction(imb_vkGetBufferMemoryRequirements2);
    if (std::strcmp(name, "vkGetDeviceBufferMemoryRequirements") == 0
        || std::strcmp(name, "vkGetDeviceBufferMemoryRequirementsKHR") == 0) {
        return toVoidFunction(imb_vkGetDeviceBufferMemoryRequirements);
    }
    if (std::strcmp(name, "vkCreateImage") == 0) return toVoidFunction(imb_vkCreateImage);
    if (std::strcmp(name, "vkDestroyImage") == 0) return toVoidFunction(imb_vkDestroyImage);
    if (std::strcmp(name, "vkGetImageMemoryRequirements") == 0) return toVoidFunction(imb_vkGetImageMemoryRequirements);
    if (std::strcmp(name, "vkGetImageMemoryRequirements2") == 0
        || std::strcmp(name, "vkGetImageMemoryRequirements2KHR") == 0) return toVoidFunction(imb_vkGetImageMemoryRequirements2);
    if (std::strcmp(name, "vkGetDeviceImageMemoryRequirements") == 0
        || std::strcmp(name, "vkGetDeviceImageMemoryRequirementsKHR") == 0) {
        return toVoidFunction(imb_vkGetDeviceImageMemoryRequirements);
    }
    if (std::strcmp(name, "vkGetImageSparseMemoryRequirements") == 0) {
        return toVoidFunction(imb_vkGetImageSparseMemoryRequirements);
    }
    if (std::strcmp(name, "vkGetImageSparseMemoryRequirements2") == 0
        || std::strcmp(name, "vkGetImageSparseMemoryRequirements2KHR") == 0) {
        return toVoidFunction(imb_vkGetImageSparseMemoryRequirements2);
    }
    if (std::strcmp(name, "vkGetDeviceImageSparseMemoryRequirements") == 0
        || std::strcmp(name, "vkGetDeviceImageSparseMemoryRequirementsKHR") == 0) {
        return toVoidFunction(imb_vkGetDeviceImageSparseMemoryRequirements);
    }
    if (std::strcmp(name, "vkGetImageSubresourceLayout") == 0) return toVoidFunction(imb_vkGetImageSubresourceLayout);
    if (std::strcmp(name, "vkAllocateMemory") == 0) return toVoidFunction(imb_vkAllocateMemory);
    if (std::strcmp(name, "vkFreeMemory") == 0) return toVoidFunction(imb_vkFreeMemory);
    if (std::strcmp(name, "vkGetMemoryFdKHR") == 0) return toVoidFunction(imb_vkGetMemoryFdKHR);
    if (std::strcmp(name, "vkGetMemoryFdPropertiesKHR") == 0) return toVoidFunction(imb_vkGetMemoryFdPropertiesKHR);
    if (std::strcmp(name, "vkGetSemaphoreFdKHR") == 0) return toVoidFunction(imb_vkGetSemaphoreFdKHR);
    if (std::strcmp(name, "vkImportSemaphoreFdKHR") == 0) return toVoidFunction(imb_vkImportSemaphoreFdKHR);
    if (std::strcmp(name, "vkBindBufferMemory") == 0) return toVoidFunction(imb_vkBindBufferMemory);
    if (std::strcmp(name, "vkBindBufferMemory2") == 0
        || std::strcmp(name, "vkBindBufferMemory2KHR") == 0) return toVoidFunction(imb_vkBindBufferMemory2);
    if (std::strcmp(name, "vkBindImageMemory") == 0) return toVoidFunction(imb_vkBindImageMemory);
    if (std::strcmp(name, "vkBindImageMemory2") == 0
        || std::strcmp(name, "vkBindImageMemory2KHR") == 0) return toVoidFunction(imb_vkBindImageMemory2);
    if (std::strcmp(name, "vkMapMemory") == 0) return toVoidFunction(imb_vkMapMemory);
    if (std::strcmp(name, "vkUnmapMemory") == 0) return toVoidFunction(imb_vkUnmapMemory);
    if (std::strcmp(name, "vkFlushMappedMemoryRanges") == 0) return toVoidFunction(imb_vkFlushMappedMemoryRanges);
    if (std::strcmp(name, "vkInvalidateMappedMemoryRanges") == 0) return toVoidFunction(imb_vkInvalidateMappedMemoryRanges);
    if (std::strcmp(name, "vkCreateShaderModule") == 0) return toVoidFunction(imb_vkCreateShaderModule);
    if (std::strcmp(name, "vkDestroyShaderModule") == 0) return toVoidFunction(imb_vkDestroyShaderModule);
    if (std::strcmp(name, "vkCreateBufferView") == 0) return toVoidFunction(imb_vkCreateBufferView);
    if (std::strcmp(name, "vkDestroyBufferView") == 0) return toVoidFunction(imb_vkDestroyBufferView);
    if (std::strcmp(name, "vkCreateImageView") == 0) return toVoidFunction(imb_vkCreateImageView);
    if (std::strcmp(name, "vkDestroyImageView") == 0) return toVoidFunction(imb_vkDestroyImageView);
    if (std::strcmp(name, "vkCreateSampler") == 0) return toVoidFunction(imb_vkCreateSampler);
    if (std::strcmp(name, "vkDestroySampler") == 0) return toVoidFunction(imb_vkDestroySampler);
    if (std::strcmp(name, "vkCreateRenderPass") == 0) return toVoidFunction(imb_vkCreateRenderPass);
    if (std::strcmp(name, "vkDestroyRenderPass") == 0) return toVoidFunction(imb_vkDestroyRenderPass);
    if (std::strcmp(name, "vkCreateFramebuffer") == 0) return toVoidFunction(imb_vkCreateFramebuffer);
    if (std::strcmp(name, "vkDestroyFramebuffer") == 0) return toVoidFunction(imb_vkDestroyFramebuffer);
    if (std::strcmp(name, "vkCreatePipelineCache") == 0) return toVoidFunction(imb_vkCreatePipelineCache);
    if (std::strcmp(name, "vkDestroyPipelineCache") == 0) return toVoidFunction(imb_vkDestroyPipelineCache);
    if (std::strcmp(name, "vkGetPipelineCacheData") == 0) return toVoidFunction(imb_vkGetPipelineCacheData);
    if (std::strcmp(name, "vkMergePipelineCaches") == 0) return toVoidFunction(imb_vkMergePipelineCaches);
    if (std::strcmp(name, "vkCreateDescriptorSetLayout") == 0) return toVoidFunction(imb_vkCreateDescriptorSetLayout);
    if (std::strcmp(name, "vkDestroyDescriptorSetLayout") == 0) return toVoidFunction(imb_vkDestroyDescriptorSetLayout);
    if (std::strcmp(name, "vkGetDescriptorSetLayoutSupport") == 0
        || std::strcmp(name, "vkGetDescriptorSetLayoutSupportKHR") == 0) return toVoidFunction(imb_vkGetDescriptorSetLayoutSupport);
    if (std::strcmp(name, "vkCreatePipelineLayout") == 0) return toVoidFunction(imb_vkCreatePipelineLayout);
    if (std::strcmp(name, "vkDestroyPipelineLayout") == 0) return toVoidFunction(imb_vkDestroyPipelineLayout);
    if (std::strcmp(name, "vkCreateComputePipelines") == 0) return toVoidFunction(imb_vkCreateComputePipelines);
    if (std::strcmp(name, "vkCreateGraphicsPipelines") == 0) return toVoidFunction(imb_vkCreateGraphicsPipelines);
    if (std::strcmp(name, "vkCreateRayTracingPipelinesNV") == 0) return toVoidFunction(imb_vkCreateRayTracingPipelinesNV);
    if (std::strcmp(name, "vkCreateRayTracingPipelinesKHR") == 0) return toVoidFunction(imb_vkCreateRayTracingPipelinesKHR);
    if (std::strcmp(name, "vkGetRayTracingShaderGroupHandlesKHR") == 0) return toVoidFunction(imb_vkGetRayTracingShaderGroupHandlesKHR);
    if (std::strcmp(name, "vkGetRayTracingShaderGroupHandlesNV") == 0) return toVoidFunction(imb_vkGetRayTracingShaderGroupHandlesNV);
    if (std::strcmp(name, "vkGetRayTracingCaptureReplayShaderGroupHandlesKHR") == 0) return toVoidFunction(imb_vkGetRayTracingCaptureReplayShaderGroupHandlesKHR);
    if (std::strcmp(name, "vkGetRayTracingShaderGroupStackSizeKHR") == 0) return toVoidFunction(imb_vkGetRayTracingShaderGroupStackSizeKHR);
    if (std::strcmp(name, "vkCompileDeferredNV") == 0) return toVoidFunction(imb_vkCompileDeferredNV);
    if (std::strcmp(name, "vkDestroyPipeline") == 0) return toVoidFunction(imb_vkDestroyPipeline);
    if (std::strcmp(name, "vkCreateAccelerationStructureNV") == 0) return toVoidFunction(imb_vkCreateAccelerationStructureNV);
    if (std::strcmp(name, "vkDestroyAccelerationStructureNV") == 0) return toVoidFunction(imb_vkDestroyAccelerationStructureNV);
    if (std::strcmp(name, "vkGetAccelerationStructureMemoryRequirementsNV") == 0) return toVoidFunction(imb_vkGetAccelerationStructureMemoryRequirementsNV);
    if (std::strcmp(name, "vkBindAccelerationStructureMemoryNV") == 0) return toVoidFunction(imb_vkBindAccelerationStructureMemoryNV);
    if (std::strcmp(name, "vkGetAccelerationStructureHandleNV") == 0) return toVoidFunction(imb_vkGetAccelerationStructureHandleNV);
    if (std::strcmp(name, "vkCreateAccelerationStructureKHR") == 0) return toVoidFunction(imb_vkCreateAccelerationStructureKHR);
    if (std::strcmp(name, "vkDestroyAccelerationStructureKHR") == 0) return toVoidFunction(imb_vkDestroyAccelerationStructureKHR);
    if (std::strcmp(name, "vkGetAccelerationStructureDeviceAddressKHR") == 0) return toVoidFunction(imb_vkGetAccelerationStructureDeviceAddressKHR);
    if (std::strcmp(name, "vkGetAccelerationStructureBuildSizesKHR") == 0) return toVoidFunction(imb_vkGetAccelerationStructureBuildSizesKHR);
    if (std::strcmp(name, "vkGetDeviceAccelerationStructureCompatibilityKHR") == 0) return toVoidFunction(imb_vkGetDeviceAccelerationStructureCompatibilityKHR);
    if (std::strcmp(name, "vkBuildAccelerationStructuresKHR") == 0) return toVoidFunction(imb_vkBuildAccelerationStructuresKHR);
    if (std::strcmp(name, "vkCopyAccelerationStructureKHR") == 0) return toVoidFunction(imb_vkCopyAccelerationStructureKHR);
    if (std::strcmp(name, "vkCopyAccelerationStructureToMemoryKHR") == 0) return toVoidFunction(imb_vkCopyAccelerationStructureToMemoryKHR);
    if (std::strcmp(name, "vkCopyMemoryToAccelerationStructureKHR") == 0) return toVoidFunction(imb_vkCopyMemoryToAccelerationStructureKHR);
    if (std::strcmp(name, "vkWriteAccelerationStructuresPropertiesKHR") == 0) return toVoidFunction(imb_vkWriteAccelerationStructuresPropertiesKHR);
    if (std::strcmp(name, "vkCreateDeferredOperationKHR") == 0) return toVoidFunction(imb_vkCreateDeferredOperationKHR);
    if (std::strcmp(name, "vkDestroyDeferredOperationKHR") == 0) return toVoidFunction(imb_vkDestroyDeferredOperationKHR);
    if (std::strcmp(name, "vkGetDeferredOperationMaxConcurrencyKHR") == 0) return toVoidFunction(imb_vkGetDeferredOperationMaxConcurrencyKHR);
    if (std::strcmp(name, "vkGetDeferredOperationResultKHR") == 0) return toVoidFunction(imb_vkGetDeferredOperationResultKHR);
    if (std::strcmp(name, "vkDeferredOperationJoinKHR") == 0) return toVoidFunction(imb_vkDeferredOperationJoinKHR);
    if (std::strcmp(name, "vkGetBufferDeviceAddress") == 0
        || std::strcmp(name, "vkGetBufferDeviceAddressKHR") == 0) return toVoidFunction(imb_vkGetBufferDeviceAddressKHR);
    if (std::strcmp(name, "vkGetBufferOpaqueCaptureAddress") == 0
        || std::strcmp(name, "vkGetBufferOpaqueCaptureAddressKHR") == 0) return toVoidFunction(imb_vkGetBufferOpaqueCaptureAddressKHR);
    if (std::strcmp(name, "vkGetDeviceMemoryOpaqueCaptureAddress") == 0
        || std::strcmp(name, "vkGetDeviceMemoryOpaqueCaptureAddressKHR") == 0) return toVoidFunction(imb_vkGetDeviceMemoryOpaqueCaptureAddressKHR);
    if (std::strcmp(name, "vkCreateDescriptorPool") == 0) return toVoidFunction(imb_vkCreateDescriptorPool);
    if (std::strcmp(name, "vkDestroyDescriptorPool") == 0) return toVoidFunction(imb_vkDestroyDescriptorPool);
    if (std::strcmp(name, "vkResetDescriptorPool") == 0) return toVoidFunction(imb_vkResetDescriptorPool);
    if (std::strcmp(name, "vkAllocateDescriptorSets") == 0) return toVoidFunction(imb_vkAllocateDescriptorSets);
    if (std::strcmp(name, "vkFreeDescriptorSets") == 0) return toVoidFunction(imb_vkFreeDescriptorSets);
    if (std::strcmp(name, "vkUpdateDescriptorSets") == 0) return toVoidFunction(imb_vkUpdateDescriptorSets);
    if (std::strcmp(name, "vkCreateCommandPool") == 0) return toVoidFunction(imb_vkCreateCommandPool);
    if (std::strcmp(name, "vkDestroyCommandPool") == 0) return toVoidFunction(imb_vkDestroyCommandPool);
    if (std::strcmp(name, "vkResetCommandPool") == 0) return toVoidFunction(imb_vkResetCommandPool);
    if (std::strcmp(name, "vkAllocateCommandBuffers") == 0) return toVoidFunction(imb_vkAllocateCommandBuffers);
    if (std::strcmp(name, "vkFreeCommandBuffers") == 0) return toVoidFunction(imb_vkFreeCommandBuffers);
    if (std::strcmp(name, "vkBeginCommandBuffer") == 0) return toVoidFunction(imb_vkBeginCommandBuffer);
    if (std::strcmp(name, "vkEndCommandBuffer") == 0) return toVoidFunction(imb_vkEndCommandBuffer);
    if (std::strcmp(name, "vkResetCommandBuffer") == 0) return toVoidFunction(imb_vkResetCommandBuffer);
    if (std::strcmp(name, "vkCmdBindPipeline") == 0) return toVoidFunction(imb_vkCmdBindPipeline);
    if (std::strcmp(name, "vkCmdBindDescriptorSets") == 0) return toVoidFunction(imb_vkCmdBindDescriptorSets);
    if (std::strcmp(name, "vkCmdPushConstants") == 0) return toVoidFunction(imb_vkCmdPushConstants);
    if (std::strcmp(name, "vkCmdDispatch") == 0) return toVoidFunction(imb_vkCmdDispatch);
    if (std::strcmp(name, "vkCmdPipelineBarrier") == 0) return toVoidFunction(imb_vkCmdPipelineBarrier);
    if (std::strcmp(name, "vkCmdCopyBuffer") == 0) return toVoidFunction(imb_vkCmdCopyBuffer);
    if (std::strcmp(name, "vkCmdCopyBuffer2") == 0
        || std::strcmp(name, "vkCmdCopyBuffer2KHR") == 0) return toVoidFunction(imb_vkCmdCopyBuffer2);
    if (std::strcmp(name, "vkCmdUpdateBuffer") == 0) return toVoidFunction(imb_vkCmdUpdateBuffer);
    if (std::strcmp(name, "vkCmdFillBuffer") == 0) return toVoidFunction(imb_vkCmdFillBuffer);
    if (std::strcmp(name, "vkCmdClearColorImage") == 0) {
        return toVoidFunction(imb_vkCmdClearColorImage);
    }
    if (std::strcmp(name, "vkCmdCopyImage") == 0) return toVoidFunction(imb_vkCmdCopyImage);
    if (std::strcmp(name, "vkCmdCopyImage2") == 0
        || std::strcmp(name, "vkCmdCopyImage2KHR") == 0) {
        return toVoidFunction(imb_vkCmdCopyImage2);
    }
    if (std::strcmp(name, "vkCmdCopyBufferToImage") == 0) return toVoidFunction(imb_vkCmdCopyBufferToImage);
    if (std::strcmp(name, "vkCmdCopyImageToBuffer") == 0) return toVoidFunction(imb_vkCmdCopyImageToBuffer);
    if (std::strcmp(name, "vkCmdCopyImageToBuffer2") == 0
        || std::strcmp(name, "vkCmdCopyImageToBuffer2KHR") == 0) {
        return toVoidFunction(imb_vkCmdCopyImageToBuffer2);
    }
    if (std::strcmp(name, "vkCmdSetViewport") == 0) return toVoidFunction(imb_vkCmdSetViewport);
    if (std::strcmp(name, "vkCmdSetScissor") == 0) return toVoidFunction(imb_vkCmdSetScissor);
    if (std::strcmp(name, "vkCmdSetDepthBounds") == 0) return toVoidFunction(imb_vkCmdSetDepthBounds);
    if (std::strcmp(name, "vkCmdClearAttachments") == 0) return toVoidFunction(imb_vkCmdClearAttachments);
    if (std::strcmp(name, "vkCmdBindVertexBuffers") == 0) return toVoidFunction(imb_vkCmdBindVertexBuffers);
    if (std::strcmp(name, "vkCmdBindIndexBuffer") == 0) return toVoidFunction(imb_vkCmdBindIndexBuffer);
    if (std::strcmp(name, "vkCmdDrawIndexed") == 0) return toVoidFunction(imb_vkCmdDrawIndexed);
    if (std::strcmp(name, "vkCmdBeginRenderPass") == 0) return toVoidFunction(imb_vkCmdBeginRenderPass);
    if (std::strcmp(name, "vkCmdEndRenderPass") == 0) return toVoidFunction(imb_vkCmdEndRenderPass);
    if (std::strcmp(name, "vkCmdDraw") == 0) return toVoidFunction(imb_vkCmdDraw);
    if (std::strcmp(name, "vkCmdBuildAccelerationStructureNV") == 0) return toVoidFunction(imb_vkCmdBuildAccelerationStructureNV);
    if (std::strcmp(name, "vkCmdCopyAccelerationStructureNV") == 0) return toVoidFunction(imb_vkCmdCopyAccelerationStructureNV);
    if (std::strcmp(name, "vkCmdTraceRaysNV") == 0) return toVoidFunction(imb_vkCmdTraceRaysNV);
    if (std::strcmp(name, "vkCmdWriteAccelerationStructuresPropertiesNV") == 0) return toVoidFunction(imb_vkCmdWriteAccelerationStructuresPropertiesNV);
    if (std::strcmp(name, "vkCmdBuildAccelerationStructuresKHR") == 0) return toVoidFunction(imb_vkCmdBuildAccelerationStructuresKHR);
    if (std::strcmp(name, "vkCmdBuildAccelerationStructuresIndirectKHR") == 0) return toVoidFunction(imb_vkCmdBuildAccelerationStructuresIndirectKHR);
    if (std::strcmp(name, "vkCmdCopyAccelerationStructureKHR") == 0) return toVoidFunction(imb_vkCmdCopyAccelerationStructureKHR);
    if (std::strcmp(name, "vkCmdCopyAccelerationStructureToMemoryKHR") == 0) return toVoidFunction(imb_vkCmdCopyAccelerationStructureToMemoryKHR);
    if (std::strcmp(name, "vkCmdCopyMemoryToAccelerationStructureKHR") == 0) return toVoidFunction(imb_vkCmdCopyMemoryToAccelerationStructureKHR);
    if (std::strcmp(name, "vkCmdWriteAccelerationStructuresPropertiesKHR") == 0) return toVoidFunction(imb_vkCmdWriteAccelerationStructuresPropertiesKHR);
    if (std::strcmp(name, "vkCmdTraceRaysKHR") == 0) return toVoidFunction(imb_vkCmdTraceRaysKHR);
    if (std::strcmp(name, "vkCmdTraceRaysIndirectKHR") == 0) return toVoidFunction(imb_vkCmdTraceRaysIndirectKHR);
    if (std::strcmp(name, "vkCmdSetRayTracingPipelineStackSizeKHR") == 0) return toVoidFunction(imb_vkCmdSetRayTracingPipelineStackSizeKHR);
    if (std::strcmp(name, "vkCreateQueryPool") == 0) return toVoidFunction(imb_vkCreateQueryPool);
    if (std::strcmp(name, "vkDestroyQueryPool") == 0) return toVoidFunction(imb_vkDestroyQueryPool);
    if (std::strcmp(name, "vkGetQueryPoolResults") == 0) return toVoidFunction(imb_vkGetQueryPoolResults);
    if (std::strcmp(name, "vkResetQueryPool") == 0
        || std::strcmp(name, "vkResetQueryPoolEXT") == 0) return toVoidFunction(imb_vkResetQueryPool);
    if (std::strcmp(name, "vkCmdResetQueryPool") == 0) return toVoidFunction(imb_vkCmdResetQueryPool);
    if (std::strcmp(name, "vkCmdBeginQuery") == 0) return toVoidFunction(imb_vkCmdBeginQuery);
    if (std::strcmp(name, "vkCmdEndQuery") == 0) return toVoidFunction(imb_vkCmdEndQuery);
    if (std::strcmp(name, "vkCmdWriteTimestamp") == 0) return toVoidFunction(imb_vkCmdWriteTimestamp);
    if (std::strcmp(name, "vkCmdCopyQueryPoolResults") == 0) return toVoidFunction(imb_vkCmdCopyQueryPoolResults);
    if (std::strcmp(name, "vkCreateSemaphore") == 0) return toVoidFunction(imb_vkCreateSemaphore);
    if (std::strcmp(name, "vkDestroySemaphore") == 0) return toVoidFunction(imb_vkDestroySemaphore);
    if (std::strcmp(name, "vkGetSemaphoreCounterValue") == 0
        || std::strcmp(name, "vkGetSemaphoreCounterValueKHR") == 0) return toVoidFunction(imb_vkGetSemaphoreCounterValue);
    if (std::strcmp(name, "vkSignalSemaphore") == 0
        || std::strcmp(name, "vkSignalSemaphoreKHR") == 0) return toVoidFunction(imb_vkSignalSemaphore);
    if (std::strcmp(name, "vkWaitSemaphores") == 0
        || std::strcmp(name, "vkWaitSemaphoresKHR") == 0) return toVoidFunction(imb_vkWaitSemaphores);
    if (std::strcmp(name, "vkCreateFence") == 0) return toVoidFunction(imb_vkCreateFence);
    if (std::strcmp(name, "vkDestroyFence") == 0) return toVoidFunction(imb_vkDestroyFence);
    if (std::strcmp(name, "vkResetFences") == 0) return toVoidFunction(imb_vkResetFences);
    if (std::strcmp(name, "vkGetFenceStatus") == 0) return toVoidFunction(imb_vkGetFenceStatus);
    if (std::strcmp(name, "vkWaitForFences") == 0) return toVoidFunction(imb_vkWaitForFences);
    if (std::strcmp(name, "vkQueueSubmit") == 0) return toVoidFunction(imb_vkQueueSubmit);
    if (device != VK_NULL_HANDLE && isCoreDeviceFunctionName(name)) {
        return reinterpret_cast<PFN_vkVoidFunction>(gDiagnosticDeviceFunctions[diagnosticSlotFor(name)]);
    }
    return nullptr;
}

PFN_vkVoidFunction getInstanceFunction(VkInstance instance, const char* name) {
    if (name == nullptr) return nullptr;
    if (std::strcmp(name, "vkGetInstanceProcAddr") == 0) {
        return toVoidFunction(imb_vkGetInstanceProcAddr);
    }
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) return toVoidFunction(imb_vkGetDeviceProcAddr);
    if (std::strcmp(name, "vkEnumerateInstanceExtensionProperties") == 0) return toVoidFunction(imb_vkEnumerateInstanceExtensionProperties);
    if (std::strcmp(name, "vkEnumerateInstanceLayerProperties") == 0) return toVoidFunction(imb_vkEnumerateInstanceLayerProperties);
    if (std::strcmp(name, "vkEnumerateInstanceVersion") == 0) return toVoidFunction(imb_vkEnumerateInstanceVersion);
    if (std::strcmp(name, "vkCreateInstance") == 0) return toVoidFunction(imb_vkCreateInstance);
    if (instance == VK_NULL_HANDLE) return nullptr;
    if (std::strcmp(name, "vkDestroyInstance") == 0) return toVoidFunction(imb_vkDestroyInstance);
    if (std::strcmp(name, "vkEnumeratePhysicalDevices") == 0) return toVoidFunction(imb_vkEnumeratePhysicalDevices);
    if (std::strcmp(name, "vkGetPhysicalDeviceFeatures") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceFeatures);
    if (std::strcmp(name, "vkGetPhysicalDeviceProperties") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceProperties);
    if (std::strcmp(name, "vkGetPhysicalDeviceMemoryProperties") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceMemoryProperties);
    if (std::strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceQueueFamilyProperties);
    if (std::strcmp(name, "vkGetPhysicalDeviceFormatProperties") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceFormatProperties);
    if (std::strcmp(name, "vkGetPhysicalDeviceImageFormatProperties") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceImageFormatProperties);
    if (std::strcmp(name, "vkGetPhysicalDeviceSparseImageFormatProperties") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceSparseImageFormatProperties);
    if (std::strcmp(name, "vkDestroySurfaceKHR") == 0) return toVoidFunction(imb_vkDestroySurfaceKHR);
    if (std::strcmp(name, "vkGetPhysicalDeviceSurfaceSupportKHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceSurfaceSupportKHR);
    if (std::strcmp(name, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceSurfaceCapabilitiesKHR);
    if (std::strcmp(name, "vkGetPhysicalDeviceSurfaceFormatsKHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceSurfaceFormatsKHR);
    if (std::strcmp(name, "vkGetPhysicalDeviceSurfacePresentModesKHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceSurfacePresentModesKHR);
    if (std::strcmp(name, "vkCreateXlibSurfaceKHR") == 0) return toVoidFunction(imb_vkCreateXlibSurfaceKHR);
    if (std::strcmp(name, "vkGetPhysicalDeviceXlibPresentationSupportKHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceXlibPresentationSupportKHR);
    if (std::strcmp(name, "vkGetPhysicalDeviceFeatures2") == 0 || std::strcmp(name, "vkGetPhysicalDeviceFeatures2KHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceFeatures2);
    if (std::strcmp(name, "vkGetPhysicalDeviceProperties2") == 0 || std::strcmp(name, "vkGetPhysicalDeviceProperties2KHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceProperties2);
    if (std::strcmp(name, "vkGetPhysicalDeviceFormatProperties2") == 0 || std::strcmp(name, "vkGetPhysicalDeviceFormatProperties2KHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceFormatProperties2);
    if (std::strcmp(name, "vkGetPhysicalDeviceImageFormatProperties2") == 0 || std::strcmp(name, "vkGetPhysicalDeviceImageFormatProperties2KHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceImageFormatProperties2);
    if (std::strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties2") == 0 || std::strcmp(name, "vkGetPhysicalDeviceQueueFamilyProperties2KHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceQueueFamilyProperties2);
    if (std::strcmp(name, "vkGetPhysicalDeviceMemoryProperties2") == 0 || std::strcmp(name, "vkGetPhysicalDeviceMemoryProperties2KHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceMemoryProperties2);
    if (std::strcmp(name, "vkGetPhysicalDeviceSparseImageFormatProperties2") == 0 || std::strcmp(name, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceSparseImageFormatProperties2);
    if (std::strcmp(name, "vkGetPhysicalDeviceExternalBufferProperties") == 0 || std::strcmp(name, "vkGetPhysicalDeviceExternalBufferPropertiesKHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceExternalBufferProperties);
    if (std::strcmp(name, "vkGetPhysicalDeviceExternalFenceProperties") == 0 || std::strcmp(name, "vkGetPhysicalDeviceExternalFencePropertiesKHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceExternalFenceProperties);
    if (std::strcmp(name, "vkGetPhysicalDeviceExternalSemaphoreProperties") == 0 || std::strcmp(name, "vkGetPhysicalDeviceExternalSemaphorePropertiesKHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceExternalSemaphoreProperties);
    if (std::strcmp(name, "vkEnumeratePhysicalDeviceGroups") == 0 || std::strcmp(name, "vkEnumeratePhysicalDeviceGroupsKHR") == 0) return toVoidFunction(imb_vkEnumeratePhysicalDeviceGroups);
    if (std::strcmp(name, "vkCreateDebugUtilsMessengerEXT") == 0) return toVoidFunction(imb_vkCreateDebugUtilsMessengerEXT);
    if (std::strcmp(name, "vkDestroyDebugUtilsMessengerEXT") == 0) return toVoidFunction(imb_vkDestroyDebugUtilsMessengerEXT);
    if (std::strcmp(name, "vkSubmitDebugUtilsMessageEXT") == 0) return toVoidFunction(imb_vkSubmitDebugUtilsMessageEXT);
    if (std::strcmp(name, "vkCreateDebugReportCallbackEXT") == 0) return toVoidFunction(imb_vkCreateDebugReportCallbackEXT);
    if (std::strcmp(name, "vkDestroyDebugReportCallbackEXT") == 0) return toVoidFunction(imb_vkDestroyDebugReportCallbackEXT);
    if (std::strcmp(name, "vkDebugReportMessageEXT") == 0) return toVoidFunction(imb_vkDebugReportMessageEXT);
    if (std::strcmp(name, "vkGetPhysicalDeviceSurfaceCapabilities2KHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceSurfaceCapabilities2KHR);
    if (std::strcmp(name, "vkGetPhysicalDeviceSurfaceFormats2KHR") == 0) return toVoidFunction(imb_vkGetPhysicalDeviceSurfaceFormats2KHR);
    if (std::strcmp(name, "vkEnumerateDeviceExtensionProperties") == 0) return toVoidFunction(imb_vkEnumerateDeviceExtensionProperties);
    if (std::strcmp(name, "vkEnumerateDeviceLayerProperties") == 0) return toVoidFunction(imb_vkEnumerateDeviceLayerProperties);
    if (std::strcmp(name, "vkCreateDevice") == 0) return toVoidFunction(imb_vkCreateDevice);
    const auto deviceFunction = imb_vkGetDeviceProcAddr(VK_NULL_HANDLE, name);
    if (deviceFunction == nullptr && traceEnabled()) {
        std::fprintf(stderr, "imb-vulkan-icd: unresolved function %s\n", name);
    }
    return deviceFunction;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL imb_vkGetInstanceProcAddr(VkInstance instance, const char* name) {
    return getInstanceFunction(instance, name);
}

}  // namespace

extern "C" {

__attribute__((visibility("default")))
VKAPI_ATTR VkResult VKAPI_CALL vk_icdNegotiateLoaderICDInterfaceVersion(std::uint32_t* supportedVersion) {
    if (supportedVersion == nullptr || *supportedVersion == 0) return VK_ERROR_INCOMPATIBLE_DRIVER;
    *supportedVersion = std::min(*supportedVersion, std::uint32_t{5});
    return VK_SUCCESS;
}

__attribute__((visibility("default")))
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vk_icdGetInstanceProcAddr(VkInstance instance, const char* name) {
    return getInstanceFunction(instance, name);
}

}  // extern "C"
