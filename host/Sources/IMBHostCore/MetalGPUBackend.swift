import Foundation

#if canImport(Metal)
import Metal
import MetalKit
#endif

public enum GPUBackendError: Error, CustomStringConvertible {
    case unavailable(String)
    case resourceNotFound(UInt64)
    case outOfBounds
    case unsupported(String)
    case commandFailed(String)

    public var description: String {
        switch self {
        case .unavailable(let message): message
        case .resourceNotFound(let id): "GPU resource \(id) was not found"
        case .outOfBounds: "GPU resource access is out of bounds"
        case .unsupported(let message): message
        case .commandFailed(let message): message
        }
    }
}

public struct IndexedUIDraw: Sendable, Equatable {
    public let textureID: UInt64
    public let indexCount: UInt32
    public let firstIndex: UInt32
    public let vertexOffset: Int32
    public let scissorX: UInt32
    public let scissorY: UInt32
    public let scissorWidth: UInt32
    public let scissorHeight: UInt32

    public init(
        textureID: UInt64,
        indexCount: UInt32,
        firstIndex: UInt32,
        vertexOffset: Int32,
        scissorX: UInt32,
        scissorY: UInt32,
        scissorWidth: UInt32,
        scissorHeight: UInt32
    ) {
        self.textureID = textureID
        self.indexCount = indexCount
        self.firstIndex = firstIndex
        self.vertexOffset = vertexOffset
        self.scissorX = scissorX
        self.scissorY = scissorY
        self.scissorWidth = scissorWidth
        self.scissorHeight = scissorHeight
    }
}

public enum ComputeBindingKind: UInt32, Sendable {
    case bufferRead = 1
    case bufferReadWrite = 2
    case textureRead = 3
    case textureReadWrite = 4
    case texelBufferRead = 5
    case texelBufferReadWrite = 6
}

public struct ComputeBinding: Sendable, Equatable {
    public let descriptorSet: UInt32
    public let binding: UInt32
    public let arrayElement: UInt32
    public let kind: ComputeBindingKind
    public let format: UInt32
    public let resourceID: UInt64
    public let offset: UInt64
    public let length: UInt64

    public init(
        descriptorSet: UInt32,
        binding: UInt32,
        arrayElement: UInt32,
        kind: ComputeBindingKind,
        format: UInt32 = 0,
        resourceID: UInt64,
        offset: UInt64,
        length: UInt64
    ) {
        self.descriptorSet = descriptorSet
        self.binding = binding
        self.arrayElement = arrayElement
        self.kind = kind
        self.format = format
        self.resourceID = resourceID
        self.offset = offset
        self.length = length
    }
}

public enum PrimitiveAccelerationStructureGeometryKind: UInt32, Sendable {
    case triangles = 0
    case boundingBoxes = 1
}

public struct PrimitiveAccelerationStructureGeometry: Sendable, Equatable {
    public let kind: PrimitiveAccelerationStructureGeometryKind
    public let flags: UInt32
    public let dataResourceID: UInt64
    public let dataOffset: UInt64
    public let primitiveCount: UInt32
    public let stride: UInt32
    public let indexResourceID: UInt64
    public let indexOffset: UInt64
    public let indexType: UInt32
    public let vertexFormat: UInt32
    public let transformResourceID: UInt64
    public let transformOffset: UInt64

    public init(
        kind: PrimitiveAccelerationStructureGeometryKind,
        flags: UInt32,
        dataResourceID: UInt64,
        dataOffset: UInt64,
        primitiveCount: UInt32,
        stride: UInt32,
        indexResourceID: UInt64 = 0,
        indexOffset: UInt64 = 0,
        indexType: UInt32 = 0,
        vertexFormat: UInt32 = 0,
        transformResourceID: UInt64 = 0,
        transformOffset: UInt64 = 0
    ) {
        self.kind = kind
        self.flags = flags
        self.dataResourceID = dataResourceID
        self.dataOffset = dataOffset
        self.primitiveCount = primitiveCount
        self.stride = stride
        self.indexResourceID = indexResourceID
        self.indexOffset = indexOffset
        self.indexType = indexType
        self.vertexFormat = vertexFormat
        self.transformResourceID = transformResourceID
        self.transformOffset = transformOffset
    }
}

public struct InstanceAccelerationStructureInstance: Sendable, Equatable {
    public let transformationMatrix: [Float]
    public let options: UInt32
    public let mask: UInt32
    public let intersectionFunctionTableOffset: UInt32
    public let userID: UInt32
    public let accelerationStructureResourceID: UInt64

    public init(
        transformationMatrix: [Float],
        options: UInt32,
        mask: UInt32,
        intersectionFunctionTableOffset: UInt32,
        userID: UInt32,
        accelerationStructureResourceID: UInt64
    ) {
        self.transformationMatrix = transformationMatrix
        self.options = options
        self.mask = mask
        self.intersectionFunctionTableOffset = intersectionFunctionTableOffset
        self.userID = userID
        self.accelerationStructureResourceID = accelerationStructureResourceID
    }
}

public struct SparseImageProperties: Sendable, Equatable {
    public let tileWidth: UInt32
    public let tileHeight: UInt32
    public let tileDepth: UInt32
    public let tileSizeBytes: UInt64

    public init(tileWidth: UInt32, tileHeight: UInt32, tileDepth: UInt32, tileSizeBytes: UInt64) {
        self.tileWidth = tileWidth
        self.tileHeight = tileHeight
        self.tileDepth = tileDepth
        self.tileSizeBytes = tileSizeBytes
    }
}

public struct RayCamera: Sendable, Equatable {
    public let position: SIMD3<Float>
    public let forward: SIMD3<Float>
    public let up: SIMD3<Float>
    public let verticalFOVRadians: Float
    public let nearDistance: Float
    public let farDistance: Float

    public init(
        position: SIMD3<Float>,
        forward: SIMD3<Float>,
        up: SIMD3<Float>,
        verticalFOVRadians: Float,
        nearDistance: Float,
        farDistance: Float
    ) {
        self.position = position
        self.forward = forward
        self.up = up
        self.verticalFOVRadians = verticalFOVRadians
        self.nearDistance = nearDistance
        self.farDistance = farDistance
    }
}

public struct RaySphereLight: Sendable, Equatable {
    public let position: SIMD3<Float>
    public let color: SIMD3<Float>
    public let intensity: Float
    public let radius: Float

    public init(
        position: SIMD3<Float>,
        color: SIMD3<Float>,
        intensity: Float,
        radius: Float
    ) {
        self.position = position
        self.color = color
        self.intensity = intensity
        self.radius = radius
    }
}

public struct RayDistantLight: Sendable, Equatable {
    /// World-space USD emission direction (local -Z after the prim transform).
    public let direction: SIMD3<Float>
    public let color: SIMD3<Float>
    public let intensity: Float
    public let angleDegrees: Float

    public init(
        direction: SIMD3<Float>,
        color: SIMD3<Float>,
        intensity: Float,
        angleDegrees: Float
    ) {
        self.direction = direction
        self.color = color
        self.intensity = intensity
        self.angleDegrees = angleDegrees
    }
}

public struct RayDomeLight: Sendable, Equatable {
    public let color: SIMD3<Float>
    public let intensity: Float

    public init(color: SIMD3<Float>, intensity: Float) {
        self.color = color
        self.intensity = intensity
    }
}

/// Narrow backend contract used by the protocol session.
///
/// It deliberately covers only the verified Metal milestones: shared buffers,
/// byte upload/readback, one deterministic UInt32 compute kernel, the fixed
/// probe triangle, Kit's indexed UI pipeline, and command-buffer fences.
public protocol BridgeGPUBackend: AnyObject {
    var supportsSPIRVCompute: Bool { get }
    var supportsAccelerationStructures: Bool { get }
    var supportsRayDispatch: Bool { get }
    var supportsSparseImages: Bool { get }
    func createBuffer(id: UInt64, size: UInt64, options: UInt32) throws
    func createImage(
        id: UInt64,
        width: UInt32,
        height: UInt32,
        format: UInt32,
        options: UInt32
    ) throws
    func querySparseImageProperties(
        format: UInt32,
        textureType: UInt32,
        sampleCount: UInt32
    ) throws -> SparseImageProperties
    func createSparseImage(
        id: UInt64,
        virtualSize: UInt64,
        width: UInt32,
        height: UInt32,
        format: UInt32,
        mipLevels: UInt32,
        arrayLayers: UInt32,
        sampleCount: UInt32,
        textureType: UInt32
    ) throws
    func updateSparseImageMapping(
        id: UInt64,
        map: Bool,
        mipLevel: UInt32,
        slice: UInt32,
        tileX: UInt32,
        tileY: UInt32,
        tileZ: UInt32,
        tileWidth: UInt32,
        tileHeight: UInt32,
        tileDepth: UInt32
    ) throws
    func destroyBuffer(id: UInt64) throws
    func destroyImage(id: UInt64) throws
    func createComputePipeline(id: UInt64, spirv: Data, entryPoint: String) throws
    func destroyComputePipeline(id: UInt64) throws
    func createAccelerationStructure(id: UInt64, type: UInt32, requestedSize: UInt64) throws
    func destroyAccelerationStructure(id: UInt64) throws
    func buildPrimitiveAccelerationStructure(
        id: UInt64,
        buildFlags: UInt32,
        geometries: [PrimitiveAccelerationStructureGeometry]
    ) throws
    func buildInstanceAccelerationStructure(
        id: UInt64,
        buildFlags: UInt32,
        instances: [InstanceAccelerationStructureInstance]
    ) throws
    func writeBuffer(id: UInt64, offset: UInt64, data: Data) throws
    func writeImage(id: UInt64, data: Data) throws
    func readBuffer(id: UInt64, offset: UInt64, length: UInt64) throws -> Data
    func readImage(id: UInt64) throws -> Data
    func submitNoop(fenceID: UInt64) throws
    func submitAddUInt32(
        bufferID: UInt64,
        elementCount: UInt32,
        addend: UInt32,
        fenceID: UInt64
    ) throws
    func submitCompute(
        pipelineID: UInt64,
        groupCountX: UInt32,
        groupCountY: UInt32,
        groupCountZ: UInt32,
        bindings: [ComputeBinding],
        pushConstants: Data,
        fenceID: UInt64
    ) throws
    func submitTriangle(imageID: UInt64, clearRGBA8: UInt32, fenceID: UInt64) throws
    func submitIndexedUI(
        imageID: UInt64,
        vertexBufferID: UInt64,
        indexBufferID: UInt64,
        vertexBufferOffset: UInt64,
        indexBufferOffset: UInt64,
        width: UInt32,
        height: UInt32,
        clearRGBA8: UInt32,
        draws: [IndexedUIDraw],
        fenceID: UInt64
    ) throws
    func submitRayTrace(
        imageID: UInt64,
        accelerationStructureID: UInt64,
        width: UInt32,
        height: UInt32,
        missRGBA8: UInt32,
        hitRGBA8: UInt32,
        camera: RayCamera?,
        sphereLight: RaySphereLight?,
        distantLight: RayDistantLight?,
        domeLight: RayDomeLight?,
        fenceID: UInt64
    ) throws
    func waitFence(id: UInt64) throws -> Bool
    func reset()
}

#if canImport(Metal)
public final class MetalGPUBackend: BridgeGPUBackend, @unchecked Sendable {
    private struct ComputePipelineKey: Hashable {
        let spirv: Data
        let entryPoint: String
    }

    private struct ComputePipelineResource {
        let pipeline: any MTLComputePipelineState
        let function: any MTLFunction
        let argumentBufferSets: Set<Int>
        let pushConstantBufferIndex: Int?
        let threadsPerThreadgroup: MTLSize
    }

    private struct ImageResource {
        let texture: any MTLTexture
        let sparseHeap: (any MTLHeap)?
        let width: Int
        let height: Int
        let format: UInt32
        let mipLevels: Int
        let arrayLayers: Int
        let sparse: Bool
    }

    private struct TriangleNormalData {
        let values: [SIMD4<Float>]
        let triangleCount: UInt32
        let normalsPerTriangle: UInt32
    }

    private struct TriangleUVData {
        let values: [SIMD2<Float>]
        let triangleCount: UInt32
    }

    private struct TriangleTangentData {
        // Two float4 values per triangle: tangent, then bitangent.
        let values: [SIMD4<Float>]
        let triangleCount: UInt32
    }

    private struct MaterialParameterData {
        let roughness: Float
        let metallic: Float
        let emissionColor: SIMD3<Float>
        let emissionIntensity: Float
    }

    private struct MaterialTextureDescriptor {
        let resourceIDs: [UInt64]
        let channels: SIMD4<UInt32>
        let alphaCutout: Bool
    }

    private struct AccelerationStructureResource {
        let type: UInt32
        let requestedSize: UInt64
        var structure: (any MTLAccelerationStructure)?
        var childStructureIDs: [UInt64]
        // A Metal acceleration structure retains the intersection geometry,
        // but it does not expose triangle vertices to a compute intersection
        // shader. Preserve one face normal per triangle while the Vulkan input
        // buffers are still alive, then build a world-space lookup for each
        // TLAS instance.
        var localTriangleNormals: TriangleNormalData?
        var localTriangleUVs: TriangleUVData?
        var localTriangleTangents: TriangleTangentData?
        var worldTriangleNormalBuffer: (any MTLBuffer)?
        var instanceNormalRangeBuffer: (any MTLBuffer)?
        var normalInstanceCount: UInt32
        var triangleUVBuffer: (any MTLBuffer)?
        var instanceUVRangeBuffer: (any MTLBuffer)?
        var uvInstanceCount: UInt32
        var instanceTextureIndexBuffer: (any MTLBuffer)?
        var instanceMaterialTextureChannelBuffer: (any MTLBuffer)?
        var worldTriangleTangentBuffer: (any MTLBuffer)?
        var instanceTangentRangeBuffer: (any MTLBuffer)?
        var tangentInstanceCount: UInt32
        var instanceNormalTextureIndexBuffer: (any MTLBuffer)?
        var materialTextureResourceIDs: [UInt64]
        var localMaterialParameters: MaterialParameterData?
        var instanceMaterialParameterBuffer: (any MTLBuffer)?
        var materialInstanceCount: UInt32
    }

    private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void imb_add_u32(
        device uint *values [[buffer(0)]],
        constant uint &addend [[buffer(1)]],
        uint index [[thread_position_in_grid]])
    {
        values[index] += addend;
    }

    struct IMBRasterVertex {
        float4 position [[position]];
        float4 color;
    };

    vertex IMBRasterVertex imb_triangle_vertex(uint index [[vertex_id]])
    {
        IMBRasterVertex output;
        if (index == 0) {
            output.position = float4(-0.8, -0.8, 0.0, 1.0);
            output.color = float4(1.0, 0.0, 0.0, 1.0);
        } else if (index == 1) {
            output.position = float4(0.8, -0.8, 0.0, 1.0);
            output.color = float4(0.0, 1.0, 0.0, 1.0);
        } else {
            output.position = float4(0.0, 0.8, 0.0, 1.0);
            output.color = float4(0.0, 0.0, 1.0, 1.0);
        }
        return output;
    }

    fragment float4 imb_triangle_fragment(IMBRasterVertex input [[stage_in]])
    {
        return input.color;
    }

    struct IMBUIRasterVertex {
        float4 position [[position]];
        float2 uv;
        float4 color;
    };

    vertex IMBUIRasterVertex imb_ui_vertex(
        const device uint *vertexWords [[buffer(0)]],
        constant float2 &invScreenSize [[buffer(1)]],
        uint vertexID [[vertex_id]])
    {
        const uint wordOffset = vertexID * 5;
        const float2 storedPosition = as_type<float2>(uint2(
            vertexWords[wordOffset],
            vertexWords[wordOffset + 1]
        ));
        const float2 storedUV = as_type<float2>(uint2(
            vertexWords[wordOffset + 2],
            vertexWords[wordOffset + 3]
        ));
        const uint storedColor = vertexWords[wordOffset + 4];
        float2 position = storedPosition * invScreenSize;
        position = position * 2.0 - 1.0;
        position.y = -position.y;
        IMBUIRasterVertex output;
        output.position = float4(position, 0.0, 1.0);
        output.uv = storedUV;
        output.color = float4(
            float(storedColor & 0xff),
            float((storedColor >> 8) & 0xff),
            float((storedColor >> 16) & 0xff),
            float((storedColor >> 24) & 0xff)
        ) / 255.0;
        return output;
    }

    fragment float4 imb_ui_fragment(
        IMBUIRasterVertex input [[stage_in]],
        texture2d<float> texture [[texture(0)]],
        sampler textureSampler [[sampler(0)]])
    {
        return input.color * texture.sample(textureSampler, input.uv);
    }

    #include <metal_raytracing>
    using namespace raytracing;

    float imb_material_channel(float4 value, uint channel)
    {
        if (channel == 0u) return value.r;
        if (channel == 1u) return value.g;
        if (channel == 2u) return value.b;
        return value.a;
    }

    float3 imb_srgb_to_linear(float3 encoded)
    {
        encoded = clamp(encoded, float3(0.0), float3(1.0));
        const float3 low = encoded / 12.92;
        const float3 high = pow(
            (encoded + 0.055) / 1.055, float3(2.4)
        );
        return select(high, low, encoded <= 0.04045);
    }

    float3 imb_linear_to_srgb(float3 linear)
    {
        linear = max(linear, float3(0.0));
        const float3 low = linear * 12.92;
        const float3 high = 1.055 * pow(linear, float3(1.0 / 2.4)) - 0.055;
        return clamp(select(high, low, linear <= 0.0031308), 0.0, 1.0);
    }

    kernel void imb_trace_probe(
        instance_acceleration_structure accelerationStructure [[buffer(0)]],
        constant uint2 &extent [[buffer(1)]],
        constant uint2 &colors [[buffer(2)]],
        constant float4 &cameraPositionAndFov [[buffer(3)]],
        constant float4 &cameraForwardAndNear [[buffer(4)]],
        constant float4 &cameraUpAndFar [[buffer(5)]],
        constant uint &cameraOptions [[buffer(6)]],
        constant float4 &sphereLightPositionAndIntensity [[buffer(7)]],
        constant float4 &sphereLightColorAndRadius [[buffer(8)]],
        constant float4 &distantLightDirectionAndIntensity [[buffer(9)]],
        constant float4 &distantLightColorAndAngle [[buffer(10)]],
        constant float4 &domeLightColorAndIntensity [[buffer(11)]],
        const device float4 *triangleNormals [[buffer(12)]],
        const device uint4 *instanceNormalRanges [[buffer(13)]],
        constant uint &normalInstanceCount [[buffer(14)]],
        const device float2 *triangleUVs [[buffer(15)]],
        const device uint4 *instanceUVRanges [[buffer(16)]],
        constant uint &uvInstanceCount [[buffer(17)]],
        const device uint4 *instanceTextureIndices [[buffer(18)]],
        constant uint &textureInstanceCount [[buffer(19)]],
        constant uint &materialTextureCount [[buffer(20)]],
        const device float4 *instanceMaterialParameters [[buffer(21)]],
        constant uint &materialInstanceCount [[buffer(22)]],
        const device uint *instanceMaterialTextureChannels [[buffer(23)]],
        const device float4 *triangleTangents [[buffer(24)]],
        const device uint4 *instanceTangentRanges [[buffer(25)]],
        constant uint &tangentInstanceCount [[buffer(26)]],
        const device uint *instanceNormalTextureIndices [[buffer(27)]],
        constant uint &rayPixelStep [[buffer(28)]],
        texture2d<float, access::write> output [[texture(0)]],
        texture2d<float, access::sample> sceneMaterialTexture [[texture(1)]],
        array<texture2d<float, access::sample>, 126> materialTextures [[texture(2)]],
        uint2 threadID [[thread_position_in_grid]])
    {
        const uint pixelStep = max(rayPixelStep, 1u);
        const uint2 pixelOrigin = threadID * pixelStep;
        if (pixelOrigin.x >= extent.x || pixelOrigin.y >= extent.y) return;

        const float2 samplePixel = min(
            float2(pixelOrigin) + float(pixelStep) * 0.5,
            float2(extent) - 0.5
        );
        const float2 normalized = samplePixel / float2(extent);
        const bool isaacSceneView = colors.y == 0xffe08c30 || colors.y == 0xffe08c31;
        const bool simpleGridView = colors.y == 0xffe08c31;
        ray probeRay;
        if (isaacSceneView) {
            const bool useLiveCamera = (cameraOptions & 1u) != 0;
            const float3 cameraPosition = useLiveCamera
                ? cameraPositionAndFov.xyz
                : float3(4.5, 4.5, 3.5);
            const float3 forward = useLiveCamera
                ? normalize(cameraForwardAndNear.xyz)
                : normalize(float3(0.0, 0.0, 0.5) - cameraPosition);
            const float3 upHint = useLiveCamera
                ? normalize(cameraUpAndFar.xyz)
                : float3(0.0, 0.0, 1.0);
            const float3 right = normalize(cross(forward, upHint));
            const float3 up = normalize(cross(right, forward));
            const float2 plane = normalized * 2.0 - 1.0;
            const float aspect = float(extent.x) / float(extent.y);
            const float halfHeight = useLiveCamera
                ? tan(clamp(cameraPositionAndFov.w, 0.01, 3.13) * 0.5)
                : 0.52;
            probeRay.origin = cameraPosition;
            probeRay.direction = normalize(
                forward + right * plane.x * aspect * halfHeight - up * plane.y * halfHeight
            );
        } else {
            const float2 plane = normalized * 2.2 - 1.1;
            probeRay.origin = float3(plane.x, -plane.y, -2.0);
            probeRay.direction = float3(0.0, 0.0, 1.0);
        }
        probeRay.min_distance = (cameraOptions & 1u) != 0
            ? max(cameraForwardAndNear.w, 0.0001)
            : 0.001;
        probeRay.max_distance = (cameraOptions & 1u) != 0
            ? max(cameraUpAndFar.w, probeRay.min_distance + 0.001)
            : 100.0;

        intersector<triangle_data, instancing> triangleIntersector;
        constexpr sampler materialSampler(
            coord::normalized,
            address::repeat,
            filter::linear,
            mip_filter::none
        );
        auto intersection = triangleIntersector.intersect(
            probeRay, accelerationStructure
        );
        bool acceptedTriangleHit = false;
        // Metal's triangle intersector returns the nearest geometric hit.  For
        // the explicitly tagged Warehouse masked materials, sample the baked
        // opacity and continue the same ray past transparent triangles.  This
        // reproduces their binary 0.3333 MDL opacity mask without treating all
        // unrelated texture alpha channels as transparency.
        for (uint alphaLayer = 0u; alphaLayer < 64u; ++alphaLayer) {
            if (intersection.type == intersection_type::none) break;
            bool rejectCutout = false;
            if (intersection.instance_id < uvInstanceCount
                && intersection.instance_id < textureInstanceCount) {
                const uint candidateChannels =
                    instanceMaterialTextureChannels[intersection.instance_id];
                const uint4 candidateTextures =
                    instanceTextureIndices[intersection.instance_id];
                const uint4 candidateUVRange =
                    instanceUVRanges[intersection.instance_id];
                if ((candidateChannels & 0x80000000u) != 0u
                    && candidateTextures.x < materialTextureCount
                    && intersection.primitive_id < candidateUVRange.y
                    && candidateUVRange.z == 3u) {
                    const uint candidateUVIndex = candidateUVRange.x
                        + intersection.primitive_id * 3u;
                    const float2 candidateBarycentric =
                        intersection.triangle_barycentric_coord;
                    const float2 candidateUV =
                        triangleUVs[candidateUVIndex]
                            * (1.0 - candidateBarycentric.x
                                - candidateBarycentric.y)
                        + triangleUVs[candidateUVIndex + 1u]
                            * candidateBarycentric.x
                        + triangleUVs[candidateUVIndex + 2u]
                            * candidateBarycentric.y;
                    const float candidateAlpha =
                        materialTextures[candidateTextures.x].sample(
                            materialSampler,
                            float2(candidateUV.x, 1.0 - candidateUV.y)
                        ).a;
                    rejectCutout = candidateAlpha < 0.3333;
                }
            }
            if (!rejectCutout) {
                acceptedTriangleHit = true;
                break;
            }
            const float nextMinimum = intersection.distance + max(
                0.0001, intersection.distance * 0.000001
            );
            probeRay.min_distance = min(nextMinimum, probeRay.max_distance);
            intersection = triangleIntersector.intersect(
                probeRay, accelerationStructure
            );
        }
        float4 color;
        float gridDistance = INFINITY;
        if (simpleGridView && probeRay.direction.z < -0.0001) {
            gridDistance = -probeRay.origin.z / probeRay.direction.z;
        }
        const bool hasTriangleHit = acceptedTriangleHit;
        const bool hasGridHit = simpleGridView
            && gridDistance >= probeRay.min_distance
            && gridDistance <= probeRay.max_distance
            && (!hasTriangleHit || gridDistance <= intersection.distance + 0.02);
        if (hasGridHit) {
            const float3 worldPosition = probeRay.origin + probeRay.direction * gridDistance;
            const float2 minorCell = abs(worldPosition.xy - round(worldPosition.xy));
            const float2 majorCell = abs(worldPosition.xy / 5.0 - round(worldPosition.xy / 5.0)) * 5.0;
            const float lineWidth = clamp(gridDistance * 0.0015, 0.012, 0.075);
            const float minorLine = 1.0 - smoothstep(
                lineWidth,
                lineWidth * 2.2,
                min(minorCell.x, minorCell.y)
            );
            const float majorLine = 1.0 - smoothstep(
                lineWidth * 1.35,
                lineWidth * 2.8,
                min(majorCell.x, majorCell.y)
            );
            const float distanceFade = clamp(1.15 - gridDistance * 0.045, 0.35, 1.0);
            float3 gridColor;
            if ((cameraOptions & 4u) != 0) {
                constexpr sampler materialSampler(
                    coord::normalized,
                    address::repeat,
                    filter::linear,
                    mip_filter::linear
                );
                // Simple Grid's OmniPBR material uses projected world-space
                // UVs and inputs:texture_scale=(0.5, 0.5).
                const float2 materialUV = worldPosition.xy * 0.5;
                gridColor = imb_srgb_to_linear(
                    sceneMaterialTexture.sample(
                        materialSampler,
                        materialUV
                    ).rgb
                );
            } else {
                const float3 floorColor = float3(0.025, 0.075, 0.13);
                const float3 minorColor = float3(0.20, 0.48, 0.72) * distanceFade;
                const float3 majorColor = float3(0.62, 0.82, 1.0) * distanceFade;
                gridColor = mix(floorColor, minorColor, minorLine * 0.72);
                gridColor = mix(gridColor, majorColor, majorLine);
            }
            const float axisWidth = lineWidth * 1.8;
            const float xAxis = 1.0 - smoothstep(axisWidth, axisWidth * 2.0, abs(worldPosition.y));
            const float yAxis = 1.0 - smoothstep(axisWidth, axisWidth * 2.0, abs(worldPosition.x));
            gridColor = mix(gridColor, float3(0.86, 0.18, 0.14), xAxis);
            gridColor = mix(gridColor, float3(0.18, 0.74, 0.32), yAxis);
            color = float4(gridColor, 1.0);
        } else if (isaacSceneView && hasTriangleHit) {
            const uint materialID = intersection.user_instance_id;
            const uint materialMarker = materialID & 0x00c00000u;
            const bool hasAuthoredBaseColor = materialMarker == 0x00800000u;
            const float3 authoredBaseColor = float3(
                float(materialID & 0xffu) / 255.0,
                float((materialID >> 8) & 0xffu) / 255.0,
                // The upper two bits are the inline-color marker.  Blue uses
                // the remaining six bits so it can never impersonate the
                // material-descriptor marker.
                float((materialID >> 16) & 0x3fu) / 63.0
            );
            float3 resolvedBaseColor = (
                hasAuthoredBaseColor
                    ? authoredBaseColor
                    : float3(0.12, 0.48, 0.95)
            );
            uint4 textureIndices = uint4(0xffffffffu);
            uint textureChannels = 0u;
            float2 materialUV = float2(0.0);
            bool canSampleMaterialTextures = false;
            if (intersection.instance_id < uvInstanceCount
                && intersection.instance_id < textureInstanceCount) {
                textureIndices = instanceTextureIndices[intersection.instance_id];
                textureChannels =
                    instanceMaterialTextureChannels[intersection.instance_id];
                const uint4 uvRange = instanceUVRanges[intersection.instance_id];
                if (intersection.primitive_id < uvRange.y
                    && uvRange.z == 3u) {
                    const uint uvIndex = uvRange.x
                        + intersection.primitive_id * 3u;
                    const float2 barycentric =
                        intersection.triangle_barycentric_coord;
                    const float2 uv = triangleUVs[uvIndex]
                            * (1.0 - barycentric.x - barycentric.y)
                        + triangleUVs[uvIndex + 1] * barycentric.x
                        + triangleUVs[uvIndex + 2] * barycentric.y;
                    materialUV = float2(uv.x, 1.0 - uv.y);
                    canSampleMaterialTextures = true;
                }
            }
            if (canSampleMaterialTextures
                && textureIndices.x < materialTextureCount) {
                resolvedBaseColor = imb_srgb_to_linear(
                    materialTextures[textureIndices.x].sample(
                        materialSampler, materialUV
                    ).rgb
                );
            }
            float roughness = 0.5;
            float metallic = 0.0;
            float3 emissiveRadiance = float3(0.0);
            bool hasMaterialParameters = false;
            if (intersection.instance_id < materialInstanceCount) {
                const uint materialOffset = intersection.instance_id * 2u;
                const float4 parameters =
                    instanceMaterialParameters[materialOffset];
                if (parameters.w > 0.5) {
                    roughness = clamp(parameters.x, 0.0, 1.0);
                    metallic = clamp(parameters.y, 0.0, 1.0);
                    const float4 emission =
                        instanceMaterialParameters[materialOffset + 1u];
                    emissiveRadiance = clamp(
                        emission.xyz, float3(0.0), float3(1.0)
                    ) * clamp(emission.w, 0.0, 1000000.0);
                    hasMaterialParameters = true;
                }
            }
            if (canSampleMaterialTextures
                && textureIndices.y < materialTextureCount) {
                const float4 sampleValue =
                    materialTextures[textureIndices.y].sample(
                        materialSampler, materialUV
                    );
                roughness = clamp(imb_material_channel(
                    sampleValue, textureChannels & 0x7u
                ), 0.0, 1.0);
                hasMaterialParameters = true;
            }
            if (canSampleMaterialTextures
                && textureIndices.z < materialTextureCount) {
                const float4 sampleValue =
                    materialTextures[textureIndices.z].sample(
                        materialSampler, materialUV
                    );
                metallic = clamp(imb_material_channel(
                    sampleValue, (textureChannels >> 4) & 0x7u
                ), 0.0, 1.0);
                hasMaterialParameters = true;
            }
            if (canSampleMaterialTextures
                && textureIndices.w < materialTextureCount) {
                const float3 emissionTexture = clamp(
                    imb_srgb_to_linear(
                        materialTextures[textureIndices.w].sample(
                            materialSampler, materialUV
                        ).rgb
                    ),
                    float3(0.0),
                    float3(1.0)
                );
                emissiveRadiance *= emissionTexture;
            }
            const float3 baseColor = resolvedBaseColor;
            if ((cameraOptions & (2u | 8u | 16u)) != 0) {
                const float3 hitPosition =
                    probeRay.origin + probeRay.direction * intersection.distance;
                float3 surfaceNormal = normalize(-probeRay.direction);
                if (intersection.instance_id < normalInstanceCount) {
                    const uint4 normalRange =
                        instanceNormalRanges[intersection.instance_id];
                    if (intersection.primitive_id < normalRange.y) {
                        const uint normalIndex = normalRange.x
                            + intersection.primitive_id * normalRange.z;
                        float3 geometryNormal = triangleNormals[normalIndex].xyz;
                        if (normalRange.z == 3u) {
                            const float2 barycentric =
                                intersection.triangle_barycentric_coord;
                            geometryNormal =
                                triangleNormals[normalIndex].xyz
                                    * (1.0 - barycentric.x - barycentric.y)
                                + triangleNormals[normalIndex + 1].xyz
                                    * barycentric.x
                                + triangleNormals[normalIndex + 2].xyz
                                    * barycentric.y;
                        }
                        if (dot(geometryNormal, geometryNormal) > 0.000001) {
                            surfaceNormal = normalize(geometryNormal);
                            // The USD Meshes used by the bounded bridge are
                            // double-sided. Keep the real face plane while
                            // orienting its normal toward the incoming ray.
                            if (dot(surfaceNormal, -probeRay.direction) < 0.0) {
                                surfaceNormal = -surfaceNormal;
                            }
                        }
                    }
                }
                if (canSampleMaterialTextures
                    && intersection.instance_id < tangentInstanceCount
                    && intersection.instance_id < textureInstanceCount) {
                    const uint normalTextureIndex =
                        instanceNormalTextureIndices[intersection.instance_id];
                    const uint4 tangentRange =
                        instanceTangentRanges[intersection.instance_id];
                    if (normalTextureIndex < materialTextureCount
                        && intersection.primitive_id < tangentRange.y
                        && tangentRange.z == 2u) {
                        const uint tangentIndex = tangentRange.x
                            + intersection.primitive_id * 2u;
                        float3 tangent = triangleTangents[tangentIndex].xyz;
                        float3 bitangent = triangleTangents[tangentIndex + 1u].xyz;
                        tangent -= surfaceNormal * dot(tangent, surfaceNormal);
                        const float tangentLengthSquared = dot(tangent, tangent);
                        if (tangentLengthSquared > 0.000001) {
                            tangent *= rsqrt(tangentLengthSquared);
                            bitangent -= surfaceNormal
                                * dot(bitangent, surfaceNormal);
                            bitangent -= tangent * dot(bitangent, tangent);
                            const float bitangentLengthSquared =
                                dot(bitangent, bitangent);
                            if (bitangentLengthSquared > 0.000001) {
                                bitangent *= rsqrt(bitangentLengthSquared);
                                if (dot(cross(tangent, bitangent), surfaceNormal) < 0.0) {
                                    bitangent = -bitangent;
                                }
                                const float3 tangentNormal =
                                    materialTextures[normalTextureIndex].sample(
                                        materialSampler, materialUV
                                    ).rgb * 2.0 - 1.0;
                                if (dot(tangentNormal, tangentNormal) > 0.000001) {
                                    surfaceNormal = normalize(
                                        tangent * tangentNormal.x
                                        + bitangent * tangentNormal.y
                                        + surfaceNormal * tangentNormal.z
                                    );
                                    if (dot(surfaceNormal, -probeRay.direction) < 0.0) {
                                        surfaceNormal = -surfaceNormal;
                                    }
                                }
                            }
                        }
                    }
                }
                float3 illumination = float3(0.18);
                float3 specularIllumination = float3(0.0);
                const float3 viewDirection = normalize(-probeRay.direction);
                const float3 reflectance = mix(
                    float3(0.04), resolvedBaseColor, metallic
                );
                const float specularExponent = mix(
                    128.0, 4.0, roughness * roughness
                );
                const float specularEnergy = mix(1.0, 0.18, roughness);
                if ((cameraOptions & 2u) != 0) {
                    const float3 toLight = sphereLightPositionAndIntensity.xyz - hitPosition;
                    const float lightDistanceSquared = max(dot(toLight, toLight), 0.0001);
                    const float3 lightDirection = toLight * rsqrt(lightDistanceSquared);
                    const float diffuse = max(dot(surfaceNormal, lightDirection), 0.0);
                    float visibility = 1.0;
                    if (diffuse > 0.0 && (cameraOptions & 32u) != 0) {
                        ray shadowRay;
                        shadowRay.origin = hitPosition + surfaceNormal * 0.002;
                        shadowRay.direction = lightDirection;
                        shadowRay.min_distance = 0.001;
                        shadowRay.max_distance = max(
                            sqrt(lightDistanceSquared) - 0.004,
                            shadowRay.min_distance
                        );
                        const auto shadowIntersection = triangleIntersector.intersect(
                            shadowRay,
                            accelerationStructure
                        );
                        visibility = shadowIntersection.type == intersection_type::none
                            ? 1.0 : 0.0;
                    }
                    const float radius = max(sphereLightColorAndRadius.w, 1.0);
                    const float attenuation = 1.0 / (
                        1.0 + lightDistanceSquared / (radius * radius * 64.0)
                    );
                    const float strength = clamp(
                        log2(1.0 + max(sphereLightPositionAndIntensity.w, 0.0)) * 0.12,
                        0.15,
                        2.5
                    );
                    const float3 lightColor = max(
                        sphereLightColorAndRadius.xyz,
                        float3(0.0)
                    );
                    illumination += lightColor * (0.25 + 0.75 * diffuse * visibility)
                        * strength * attenuation;
                    if (hasMaterialParameters && diffuse > 0.0 && visibility > 0.0) {
                        const float3 halfUnnormalized =
                            lightDirection + viewDirection;
                        const float3 halfVector = halfUnnormalized * rsqrt(max(
                            dot(halfUnnormalized, halfUnnormalized), 0.000001
                        ));
                        const float specular = pow(
                            max(dot(surfaceNormal, halfVector), 0.0),
                            specularExponent
                        ) * specularEnergy;
                        specularIllumination += lightColor * reflectance
                            * specular * strength * attenuation * visibility;
                    }
                }
                if ((cameraOptions & 8u) != 0) {
                    // USD DistantLight emits along its transformed local -Z;
                    // negate that vector for the surface-to-light direction.
                    const float3 lightDirection = normalize(
                        -distantLightDirectionAndIntensity.xyz
                    );
                    const float angularWrap = clamp(
                        distantLightColorAndAngle.w / 180.0,
                        0.0,
                        1.0
                    ) * 0.12;
                    const float diffuse = clamp(
                        dot(surfaceNormal, lightDirection) + angularWrap,
                        0.0,
                        1.0
                    );
                    float visibility = 1.0;
                    if (diffuse > 0.0 && (cameraOptions & 32u) != 0) {
                        ray shadowRay;
                        shadowRay.origin = hitPosition + surfaceNormal * 0.002;
                        shadowRay.direction = lightDirection;
                        shadowRay.min_distance = 0.001;
                        shadowRay.max_distance = 1000000.0;
                        const auto shadowIntersection = triangleIntersector.intersect(
                            shadowRay,
                            accelerationStructure
                        );
                        visibility = shadowIntersection.type == intersection_type::none
                            ? 1.0 : 0.0;
                    }
                    const float strength = clamp(
                        log2(1.0 + max(distantLightDirectionAndIntensity.w, 0.0))
                            * 0.11,
                        0.12,
                        2.5
                    );
                    const float3 lightColor = max(
                        distantLightColorAndAngle.xyz,
                        float3(0.0)
                    );
                    illumination += lightColor
                        * (0.18 + 0.82 * diffuse * visibility) * strength;
                    if (hasMaterialParameters && diffuse > 0.0 && visibility > 0.0) {
                        const float3 halfUnnormalized =
                            lightDirection + viewDirection;
                        const float3 halfVector = halfUnnormalized * rsqrt(max(
                            dot(halfUnnormalized, halfUnnormalized), 0.000001
                        ));
                        const float specular = pow(
                            max(dot(surfaceNormal, halfVector), 0.0),
                            specularExponent
                        ) * specularEnergy;
                        specularIllumination += lightColor * reflectance
                            * specular * strength * visibility;
                    }
                }
                if ((cameraOptions & 16u) != 0) {
                    const float strength = clamp(
                        log2(1.0 + max(domeLightColorAndIntensity.w, 0.0)) * 0.08,
                        0.10,
                        2.0
                    );
                    illumination += max(
                        domeLightColorAndIntensity.xyz,
                        float3(0.0)
                    ) * strength * 0.55;
                    if (hasMaterialParameters) {
                        specularIllumination += max(
                            domeLightColorAndIntensity.xyz,
                            float3(0.0)
                        ) * reflectance * strength
                            * mix(0.45, 0.08, roughness);
                    }
                }
                const float diffuseWeight = hasMaterialParameters
                    ? (1.0 - metallic) : 1.0;
                color = float4(
                    baseColor * illumination * diffuseWeight
                        + specularIllumination
                        + emissiveRadiance,
                    1.0
                );
            } else {
                color = float4(baseColor + emissiveRadiance, 1.0);
            }
        } else {
            const uint packed = intersection.type == intersection_type::none ? colors.x : colors.y;
            color = float4(
                float(packed & 0xff),
                float((packed >> 8) & 0xff),
                float((packed >> 16) & 0xff),
                float((packed >> 24) & 0xff)
            ) / 255.0;
        }
        if (hasGridHit || (isaacSceneView && hasTriangleHit)) {
            color.rgb = imb_linear_to_srgb(color.rgb);
        }
        for (uint offsetY = 0; offsetY < pixelStep; ++offsetY) {
            for (uint offsetX = 0; offsetX < pixelStep; ++offsetX) {
                const uint2 outputPixel = pixelOrigin + uint2(offsetX, offsetY);
                if (outputPixel.x < extent.x && outputPixel.y < extent.y) {
                    output.write(color, outputPixel);
                }
            }
        }
    }
    """

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let addPipeline: any MTLComputePipelineState
    private let trianglePipeline: any MTLRenderPipelineState
    private let uiPipeline: any MTLRenderPipelineState
    private let uiSampler: any MTLSamplerState
    private let whiteTexture: any MTLTexture
    private let sceneMaterialTexture: any MTLTexture
    private let hasSceneMaterialTexture: Bool
    private let rayTracePipeline: (any MTLComputePipelineState)?
    private let spirvCompiler: SPIRVCrossCompiler?
    private var buffers: [UInt64: any MTLBuffer] = [:]
    private var images: [UInt64: ImageResource] = [:]
    private var computePipelines: [UInt64: ComputePipelineResource] = [:]
    private var computePipelineCache: [ComputePipelineKey: ComputePipelineResource] = [:]
    private var accelerationStructures: [UInt64: AccelerationStructureResource] = [:]
    private var commandBuffers: [UInt64: any MTLCommandBuffer] = [:]
    private var frameImages: [UInt64: UInt64] = [:]
    private var frameOutputDisabledAfterError = false
    private var rayQualityReported = false

    public static func makeDefault() -> MetalGPUBackend? {
        guard let device = MTLCreateSystemDefaultDevice() else {
            return nil
        }
        let compiler: SPIRVCrossCompiler?
        if let path = ProcessInfo.processInfo.environment["IMB_SPIRV_CROSS"], !path.isEmpty {
            compiler = try? SPIRVCrossCompiler(executableURL: URL(fileURLWithPath: path))
        } else {
            compiler = nil
        }
        let sceneMaterialTextureURL: URL?
        if let path = ProcessInfo.processInfo.environment["IMB_SCENE_MATERIAL_TEXTURE"],
           !path.isEmpty {
            sceneMaterialTextureURL = URL(fileURLWithPath: path)
        } else {
            sceneMaterialTextureURL = nil
        }
        return try? MetalGPUBackend(
            device: device,
            spirvCompiler: compiler,
            sceneMaterialTextureURL: sceneMaterialTextureURL
        )
    }

    public init(
        device: any MTLDevice,
        spirvCompiler: SPIRVCrossCompiler? = nil,
        sceneMaterialTextureURL: URL? = nil
    ) throws {
        guard let queue = device.makeCommandQueue() else {
            throw GPUBackendError.unavailable("Metal command queue creation failed")
        }
        let library: any MTLLibrary
        do {
            library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        } catch {
            throw GPUBackendError.unavailable("Metal shader compilation failed: \(error)")
        }
        guard let function = library.makeFunction(name: "imb_add_u32") else {
            throw GPUBackendError.unavailable("Metal compute function imb_add_u32 was not found")
        }
        do {
            self.addPipeline = try device.makeComputePipelineState(function: function)
        } catch {
            throw GPUBackendError.unavailable("Metal compute pipeline creation failed: \(error)")
        }
        if device.supportsRaytracing,
           let rayFunction = library.makeFunction(name: "imb_trace_probe") {
            do {
                self.rayTracePipeline = try device.makeComputePipelineState(function: rayFunction)
            } catch {
                throw GPUBackendError.unavailable("Metal ray tracing pipeline creation failed: \(error)")
            }
        } else {
            self.rayTracePipeline = nil
        }
        guard let vertexFunction = library.makeFunction(name: "imb_triangle_vertex"),
              let fragmentFunction = library.makeFunction(name: "imb_triangle_fragment")
        else {
            throw GPUBackendError.unavailable("Metal triangle shader functions were not found")
        }
        let renderDescriptor = MTLRenderPipelineDescriptor()
        renderDescriptor.label = "IMB fixed triangle pipeline"
        renderDescriptor.vertexFunction = vertexFunction
        renderDescriptor.fragmentFunction = fragmentFunction
        renderDescriptor.colorAttachments[0].pixelFormat = .rgba8Unorm
        do {
            self.trianglePipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)
        } catch {
            throw GPUBackendError.unavailable("Metal render pipeline creation failed: \(error)")
        }
        guard let uiVertexFunction = library.makeFunction(name: "imb_ui_vertex"),
              let uiFragmentFunction = library.makeFunction(name: "imb_ui_fragment")
        else {
            throw GPUBackendError.unavailable("Metal Kit UI shader functions were not found")
        }
        let uiDescriptor = MTLRenderPipelineDescriptor()
        uiDescriptor.label = "IMB Kit indexed UI pipeline"
        uiDescriptor.vertexFunction = uiVertexFunction
        uiDescriptor.fragmentFunction = uiFragmentFunction
        let uiColor = uiDescriptor.colorAttachments[0]!
        uiColor.pixelFormat = .bgra8Unorm
        uiColor.isBlendingEnabled = true
        uiColor.rgbBlendOperation = .add
        uiColor.sourceRGBBlendFactor = .sourceAlpha
        uiColor.destinationRGBBlendFactor = .oneMinusSourceAlpha
        uiColor.alphaBlendOperation = .add
        uiColor.sourceAlphaBlendFactor = .one
        uiColor.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        do {
            self.uiPipeline = try device.makeRenderPipelineState(descriptor: uiDescriptor)
        } catch {
            throw GPUBackendError.unavailable("Metal Kit UI pipeline creation failed: \(error)")
        }
        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .linear
        samplerDescriptor.magFilter = .linear
        samplerDescriptor.mipFilter = .linear
        samplerDescriptor.sAddressMode = .clampToEdge
        samplerDescriptor.tAddressMode = .clampToEdge
        guard let uiSampler = device.makeSamplerState(descriptor: samplerDescriptor) else {
            throw GPUBackendError.unavailable("Metal Kit UI sampler creation failed")
        }
        self.uiSampler = uiSampler
        let whiteDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: 1,
            height: 1,
            mipmapped: false
        )
        whiteDescriptor.storageMode = .shared
        whiteDescriptor.usage = .shaderRead
        guard let whiteTexture = device.makeTexture(descriptor: whiteDescriptor) else {
            throw GPUBackendError.unavailable("Metal fallback UI texture creation failed")
        }
        var whitePixel: UInt32 = 0xffff_ffff
        whiteTexture.replace(
            region: MTLRegionMake2D(0, 0, 1, 1),
            mipmapLevel: 0,
            withBytes: &whitePixel,
            bytesPerRow: 4
        )
        self.whiteTexture = whiteTexture
        if let sceneMaterialTextureURL {
            do {
                let texture = try MTKTextureLoader(device: device).newTexture(
                    URL: sceneMaterialTextureURL,
                    options: [
                        .SRGB: false,
                        .origin: MTKTextureLoader.Origin.bottomLeft,
                        .generateMipmaps: true,
                    ]
                )
                texture.label = "IMB real USD OmniPBR diffuse texture"
                self.sceneMaterialTexture = texture
                self.hasSceneMaterialTexture = true
                FileHandle.standardError.write(Data(
                    "imb-host: loaded real scene material texture \(sceneMaterialTextureURL.path) \(texture.width)x\(texture.height)\n".utf8
                ))
            } catch {
                self.sceneMaterialTexture = whiteTexture
                self.hasSceneMaterialTexture = false
                FileHandle.standardError.write(Data(
                    "imb-host: scene material texture load failed, using analytic fallback: \(error)\n".utf8
                ))
            }
        } else {
            self.sceneMaterialTexture = whiteTexture
            self.hasSceneMaterialTexture = false
        }
        self.spirvCompiler = spirvCompiler
        self.device = device
        self.queue = queue
    }

    public var supportsSPIRVCompute: Bool { spirvCompiler != nil }
    public var supportsAccelerationStructures: Bool { device.supportsRaytracing }
    public var supportsRayDispatch: Bool { device.supportsRaytracing && rayTracePipeline != nil }
    public var supportsSparseImages: Bool { device.sparseTileSizeInBytes > 0 }

    public func createBuffer(id: UInt64, size: UInt64, options: UInt32) throws {
        guard options == 0 else {
            throw GPUBackendError.unsupported("nonzero Metal buffer options are not defined by the current IMB protocol")
        }
        guard size > 0, size <= UInt64(Int.max), size <= UInt64(device.maxBufferLength) else {
            throw GPUBackendError.outOfBounds
        }
        guard buffers[id] == nil, images[id] == nil else {
            throw GPUBackendError.unsupported("GPU resource ID \(id) already exists")
        }
        guard let buffer = device.makeBuffer(length: Int(size), options: .storageModeShared) else {
            throw GPUBackendError.unavailable("Metal failed to allocate a \(size)-byte shared buffer")
        }
        buffers[id] = buffer
    }

    public func createImage(
        id: UInt64,
        width: UInt32,
        height: UInt32,
        format: UInt32,
        options: UInt32
    ) throws {
        guard (format == 1 || format == 2), options == 0 else {
            throw GPUBackendError.unsupported("only option-free RGBA8/BGRA8_UNORM images are supported")
        }
        guard width > 0, height > 0, width <= 4096, height <= 4096,
              buffers[id] == nil, images[id] == nil
        else {
            throw GPUBackendError.outOfBounds
        }
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format == 1 ? .rgba8Unorm : .bgra8Unorm,
            width: Int(width),
            height: Int(height),
            mipmapped: false
        )
        descriptor.storageMode = .shared
        descriptor.usage = [.renderTarget, .shaderRead, .shaderWrite]
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw GPUBackendError.unavailable("Metal failed to allocate a \(width)x\(height) RGBA8 image")
        }
        images[id] = ImageResource(
            texture: texture,
            sparseHeap: nil,
            width: Int(width),
            height: Int(height),
            format: format,
            mipLevels: 1,
            arrayLayers: 1,
            sparse: false
        )
    }

    private func metalPixelFormat(_ format: UInt32) throws -> MTLPixelFormat {
        switch format {
        case 1:
            return .rgba8Unorm
        case 2:
            return .bgra8Unorm
        case 3:
            // Swift does not expose a stable case spelling for BC3 across all
            // SDK overlays. The Metal ABI value is stable in MTLPixelFormat.h.
            guard let format = MTLPixelFormat(rawValue: 134) else {
                throw GPUBackendError.unsupported("Metal BC3_UNORM is unavailable")
            }
            return format
        case 4:
            return .r16Unorm
        case 5:
            return .rgba16Unorm
        case 6:
            guard let format = MTLPixelFormat(rawValue: 135) else {
                throw GPUBackendError.unsupported("Metal BC3_sRGB is unavailable")
            }
            return format
        case 7:
            guard let format = MTLPixelFormat(rawValue: 142) else {
                throw GPUBackendError.unsupported("Metal BC5_RGUnorm is unavailable")
            }
            return format
        default:
            throw GPUBackendError.unsupported("unsupported sparse image format \(format)")
        }
    }

    private func metalTexelBufferFormat(
        _ vulkanFormat: UInt32
    ) throws -> (pixelFormat: MTLPixelFormat, bytesPerTexel: UInt64) {
        switch vulkanFormat {
        case 9: return (.r8Unorm, 1)
        case 10: return (.r8Snorm, 1)
        case 13: return (.r8Uint, 1)
        case 14: return (.r8Sint, 1)
        case 16: return (.rg8Unorm, 2)
        case 17: return (.rg8Snorm, 2)
        case 20: return (.rg8Uint, 2)
        case 21: return (.rg8Sint, 2)
        case 37: return (.rgba8Unorm, 4)
        case 38: return (.rgba8Snorm, 4)
        case 41: return (.rgba8Uint, 4)
        case 42: return (.rgba8Sint, 4)
        case 70: return (.r16Unorm, 2)
        case 71: return (.r16Snorm, 2)
        case 74: return (.r16Uint, 2)
        case 75: return (.r16Sint, 2)
        case 76: return (.r16Float, 2)
        case 77: return (.rg16Unorm, 4)
        case 78: return (.rg16Snorm, 4)
        case 81: return (.rg16Uint, 4)
        case 82: return (.rg16Sint, 4)
        case 83: return (.rg16Float, 4)
        case 91: return (.rgba16Unorm, 8)
        case 92: return (.rgba16Snorm, 8)
        case 95: return (.rgba16Uint, 8)
        case 96: return (.rgba16Sint, 8)
        case 97: return (.rgba16Float, 8)
        case 98: return (.r32Uint, 4)
        case 99: return (.r32Sint, 4)
        case 100: return (.r32Float, 4)
        case 101: return (.rg32Uint, 8)
        case 102: return (.rg32Sint, 8)
        case 103: return (.rg32Float, 8)
        case 107: return (.rgba32Uint, 16)
        case 108: return (.rgba32Sint, 16)
        case 109: return (.rgba32Float, 16)
        default:
            throw GPUBackendError.unsupported(
                "unsupported Vulkan texel-buffer format \(vulkanFormat)"
            )
        }
    }

    public func querySparseImageProperties(
        format: UInt32,
        textureType: UInt32,
        sampleCount: UInt32
    ) throws -> SparseImageProperties {
        guard supportsSparseImages, textureType == 1, sampleCount == 1 else {
            throw GPUBackendError.unsupported("Metal sparse images currently require 2D single-sample textures")
        }
        let pixelFormat = try metalPixelFormat(format)
        let tile = device.sparseTileSize(
            with: .type2D,
            pixelFormat: pixelFormat,
            sampleCount: Int(sampleCount)
        )
        guard tile.width > 0, tile.height > 0, tile.depth > 0 else {
            throw GPUBackendError.unsupported("Metal returned an empty sparse tile size")
        }
        return SparseImageProperties(
            tileWidth: UInt32(tile.width),
            tileHeight: UInt32(tile.height),
            tileDepth: UInt32(tile.depth),
            tileSizeBytes: UInt64(device.sparseTileSizeInBytes)
        )
    }

    public func createSparseImage(
        id: UInt64,
        virtualSize: UInt64,
        width: UInt32,
        height: UInt32,
        format: UInt32,
        mipLevels: UInt32,
        arrayLayers: UInt32,
        sampleCount: UInt32,
        textureType: UInt32
    ) throws {
        guard supportsSparseImages, textureType == 1, sampleCount == 1,
              width > 0, height > 0, width <= 16_384, height <= 16_384,
              mipLevels > 0, arrayLayers > 0,
              buffers[id] == nil, images[id] == nil
        else {
            throw GPUBackendError.outOfBounds
        }
        let pixelFormat = try metalPixelFormat(format)
        let tileBytes = UInt64(device.sparseTileSizeInBytes)
        guard tileBytes > 0, virtualSize > 0, virtualSize <= UInt64(Int.max) else {
            throw GPUBackendError.outOfBounds
        }
        let roundedSize = ((virtualSize + tileBytes - 1) / tileBytes) * tileBytes
        guard roundedSize <= UInt64(Int.max) else {
            throw GPUBackendError.outOfBounds
        }

        let heapDescriptor = MTLHeapDescriptor()
        heapDescriptor.type = .sparse
        heapDescriptor.storageMode = .private
        heapDescriptor.size = Int(roundedSize)
        guard let heap = device.makeHeap(descriptor: heapDescriptor) else {
            throw GPUBackendError.unavailable(
                "Metal failed to allocate a \(roundedSize)-byte sparse texture heap"
            )
        }

        let descriptor = MTLTextureDescriptor()
        descriptor.textureType = arrayLayers == 1 ? .type2D : .type2DArray
        descriptor.pixelFormat = pixelFormat
        descriptor.width = Int(width)
        descriptor.height = Int(height)
        descriptor.depth = 1
        descriptor.mipmapLevelCount = Int(mipLevels)
        descriptor.arrayLength = Int(arrayLayers)
        descriptor.sampleCount = 1
        descriptor.storageMode = .private
        // Block-compressed formats are sampleable but not writable in either
        // Vulkan or Metal. Advertising shaderWrite makes Metal reject the
        // descriptor even though sparse BC residency itself is supported.
        descriptor.usage = (format == 3 || format == 6 || format == 7)
            ? [.shaderRead]
            : [.shaderRead, .shaderWrite]
        guard let texture = heap.makeTexture(descriptor: descriptor) else {
            throw GPUBackendError.unavailable(
                "Metal failed to create a \(width)x\(height) sparse texture"
            )
        }
        images[id] = ImageResource(
            texture: texture,
            sparseHeap: heap,
            width: Int(width),
            height: Int(height),
            format: format,
            mipLevels: Int(mipLevels),
            arrayLayers: Int(arrayLayers),
            sparse: true
        )
    }

    public func updateSparseImageMapping(
        id: UInt64,
        map: Bool,
        mipLevel: UInt32,
        slice: UInt32,
        tileX: UInt32,
        tileY: UInt32,
        tileZ: UInt32,
        tileWidth: UInt32,
        tileHeight: UInt32,
        tileDepth: UInt32
    ) throws {
        guard let image = images[id] else {
            throw GPUBackendError.resourceNotFound(id)
        }
        guard image.sparse, mipLevel < image.mipLevels, slice < image.arrayLayers,
              tileWidth > 0, tileHeight > 0, tileDepth > 0
        else {
            throw GPUBackendError.outOfBounds
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeResourceStateCommandEncoder()
        else {
            throw GPUBackendError.unavailable("Metal sparse mapping command creation failed")
        }
        encoder.updateTextureMapping?(
            image.texture,
            mode: map ? .map : .unmap,
            region: MTLRegionMake3D(
                Int(tileX),
                Int(tileY),
                Int(tileZ),
                Int(tileWidth),
                Int(tileHeight),
                Int(tileDepth)
            ),
            mipLevel: Int(mipLevel),
            slice: Int(slice)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status == .error {
            throw GPUBackendError.unavailable(
                "Metal sparse mapping failed: \(commandBuffer.error?.localizedDescription ?? "unknown error")"
            )
        }
    }

    public func destroyBuffer(id: UInt64) throws {
        guard buffers.removeValue(forKey: id) != nil else {
            throw GPUBackendError.resourceNotFound(id)
        }
    }

    public func destroyImage(id: UInt64) throws {
        guard images.removeValue(forKey: id) != nil else {
            throw GPUBackendError.resourceNotFound(id)
        }
    }

    public func createComputePipeline(id: UInt64, spirv: Data, entryPoint: String) throws {
        guard let spirvCompiler else {
            throw GPUBackendError.unsupported("SPIR-V compute translation is not configured")
        }
        guard buffers[id] == nil, images[id] == nil, computePipelines[id] == nil else {
            throw GPUBackendError.unsupported("GPU resource ID \(id) already exists")
        }
        let key = ComputePipelineKey(spirv: spirv, entryPoint: entryPoint)
        if let resource = computePipelineCache[key] {
            computePipelines[id] = resource
            return
        }
        let source: String
        do {
            source = try spirvCompiler.translateToMSL(
                spirv: spirv,
                computeEntryPoint: entryPoint
            )
        } catch {
            throw GPUBackendError.commandFailed("SPIR-V to MSL translation failed: \(error)")
        }
        let library: any MTLLibrary
        do {
            library = try device.makeLibrary(source: source, options: nil)
        } catch {
            throw GPUBackendError.commandFailed("translated Metal shader compilation failed: \(error)")
        }
        guard let function = library.makeFunction(name: "imb_compute_main") else {
            throw GPUBackendError.commandFailed("translated Metal compute entry point was not found")
        }
        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            var argumentBufferSets: Set<Int> = []
            for descriptorSet in 0..<32 where source.contains(
                "spvDescriptorSet\(descriptorSet) [[buffer(\(descriptorSet))]]"
            ) {
                argumentBufferSets.insert(descriptorSet)
            }
            let localSize = Self.computeLocalSize(from: spirv)
                ?? MTLSize(width: max(1, pipeline.threadExecutionWidth), height: 1, depth: 1)
            let threadCount = localSize.width * localSize.height * localSize.depth
            guard localSize.width > 0, localSize.height > 0, localSize.depth > 0,
                  threadCount > 0,
                  threadCount <= pipeline.maxTotalThreadsPerThreadgroup
            else {
                throw GPUBackendError.unsupported(
                    "SPIR-V local size exceeds the Metal pipeline threadgroup limit"
                )
            }
            let resource = ComputePipelineResource(
                pipeline: pipeline,
                function: function,
                argumentBufferSets: argumentBufferSets,
                pushConstantBufferIndex: Self.pushConstantBufferIndex(in: source),
                threadsPerThreadgroup: localSize
            )
            computePipelines[id] = resource
            computePipelineCache[key] = resource
        } catch {
            if let backendError = error as? GPUBackendError {
                throw backendError
            }
            throw GPUBackendError.commandFailed("translated Metal compute pipeline creation failed: \(error)")
        }
    }

    private static func computeLocalSize(from spirv: Data) -> MTLSize? {
        guard spirv.count >= 20, spirv.count % 4 == 0 else { return nil }
        var wordIndex = 5
        let wordCount = spirv.count / 4
        while wordIndex < wordCount {
            guard let instruction: UInt32 = try? spirv.readLittleEndian(at: wordIndex * 4) else {
                return nil
            }
            let instructionWordCount = Int(instruction >> 16)
            let opcode = UInt16(instruction & 0xffff)
            guard instructionWordCount > 0, wordIndex + instructionWordCount <= wordCount else {
                return nil
            }
            // OpExecutionMode %entry LocalSize x y z
            if opcode == 16, instructionWordCount >= 6,
               let mode: UInt32 = try? spirv.readLittleEndian(at: (wordIndex + 2) * 4),
               mode == 17,
               let x: UInt32 = try? spirv.readLittleEndian(at: (wordIndex + 3) * 4),
               let y: UInt32 = try? spirv.readLittleEndian(at: (wordIndex + 4) * 4),
               let z: UInt32 = try? spirv.readLittleEndian(at: (wordIndex + 5) * 4),
               x > 0, y > 0, z > 0 {
                return MTLSize(width: Int(x), height: Int(y), depth: Int(z))
            }
            wordIndex += instructionWordCount
        }
        return nil
    }

    private static func pushConstantBufferIndex(in source: String) -> Int? {
        guard let kernelStart = source.range(of: "kernel void imb_compute_main"),
              let bodyStart = source[kernelStart.lowerBound...].firstIndex(of: "{")
        else {
            return nil
        }
        let signature = source[kernelStart.lowerBound..<bodyStart]
        guard let constant = signature.range(of: "constant "),
              let marker = signature.range(
                  of: "[[buffer(",
                  range: constant.upperBound..<signature.endIndex
              )
        else {
            return nil
        }
        let digitsStart = marker.upperBound
        guard let digitsEnd = signature[digitsStart...].firstIndex(of: ")") else {
            return nil
        }
        return Int(signature[digitsStart..<digitsEnd])
    }

    public func destroyComputePipeline(id: UInt64) throws {
        guard computePipelines.removeValue(forKey: id) != nil else {
            throw GPUBackendError.resourceNotFound(id)
        }
    }

    public func createAccelerationStructure(id: UInt64, type: UInt32, requestedSize: UInt64) throws {
        guard device.supportsRaytracing else {
            throw GPUBackendError.unsupported("Metal ray tracing is unavailable on this device")
        }
        guard (type == 0 || type == 1 || type == 2), requestedSize > 0,
              buffers[id] == nil, images[id] == nil, computePipelines[id] == nil,
              accelerationStructures[id] == nil
        else {
            throw GPUBackendError.unsupported("invalid or duplicate acceleration structure resource \(id)")
        }
        accelerationStructures[id] = AccelerationStructureResource(
            type: type,
            requestedSize: requestedSize,
            structure: nil,
            childStructureIDs: [],
            localTriangleNormals: nil,
            localTriangleUVs: nil,
            localTriangleTangents: nil,
            worldTriangleNormalBuffer: nil,
            instanceNormalRangeBuffer: nil,
            normalInstanceCount: 0,
            triangleUVBuffer: nil,
            instanceUVRangeBuffer: nil,
            uvInstanceCount: 0,
            instanceTextureIndexBuffer: nil,
            instanceMaterialTextureChannelBuffer: nil,
            worldTriangleTangentBuffer: nil,
            instanceTangentRangeBuffer: nil,
            tangentInstanceCount: 0,
            instanceNormalTextureIndexBuffer: nil,
            materialTextureResourceIDs: [],
            localMaterialParameters: nil,
            instanceMaterialParameterBuffer: nil,
            materialInstanceCount: 0
        )
    }

    public func destroyAccelerationStructure(id: UInt64) throws {
        guard accelerationStructures.removeValue(forKey: id) != nil else {
            throw GPUBackendError.resourceNotFound(id)
        }
    }

    public func buildPrimitiveAccelerationStructure(
        id: UInt64,
        buildFlags: UInt32,
        geometries: [PrimitiveAccelerationStructureGeometry]
    ) throws {
        guard var resource = accelerationStructures[id] else {
            throw GPUBackendError.resourceNotFound(id)
        }
        guard device.supportsRaytracing, resource.type == 1 || resource.type == 2,
              !geometries.isEmpty, geometries.count <= 65_535,
              buildFlags & ~UInt32(0x1f) == 0
        else {
            throw GPUBackendError.unsupported("invalid primitive acceleration structure build")
        }

        let descriptor = MTLPrimitiveAccelerationStructureDescriptor()
        var usage: MTLAccelerationStructureUsage = []
        if buildFlags & 0x1 != 0 { usage.insert(.refit) }
        if buildFlags & 0x8 != 0 { usage.insert(.preferFastBuild) }
        descriptor.usage = usage
        descriptor.geometryDescriptors = try geometries.map { geometry in
            guard geometry.primitiveCount > 0, geometry.flags & ~UInt32(0x3) == 0 else {
                throw GPUBackendError.unsupported("invalid primitive geometry flags or count")
            }
            let dataBuffer = try requireBuffer(geometry.dataResourceID)
            switch geometry.kind {
            case .triangles:
                guard (1...6).contains(geometry.vertexFormat),
                      geometry.stride >= (geometry.vertexFormat == 6
                        ? 80 : (geometry.vertexFormat == 5
                        ? 56 : (geometry.vertexFormat == 4
                            ? 40 : (geometry.vertexFormat == 3
                                ? 32 : (geometry.vertexFormat == 2 ? 24 : 12))))),
                      geometry.stride % 4 == 0,
                      geometry.dataOffset <= UInt64(dataBuffer.length)
                else {
                    throw GPUBackendError.unsupported("Metal triangle geometry requires packed float3 vertices")
                }
                let triangle = MTLAccelerationStructureTriangleGeometryDescriptor()
                triangle.vertexBuffer = dataBuffer
                triangle.vertexBufferOffset = try checkedInt(geometry.dataOffset)
                triangle.vertexFormat = .float3
                triangle.vertexStride = Int(geometry.stride)
                triangle.triangleCount = Int(geometry.primitiveCount)
                triangle.opaque = geometry.flags & 0x1 != 0
                triangle.allowDuplicateIntersectionFunctionInvocation = geometry.flags & 0x2 == 0

                if geometry.indexType == 0 {
                    guard geometry.indexResourceID == 0, geometry.indexOffset == 0 else {
                        throw GPUBackendError.unsupported("non-indexed triangle geometry has an index buffer")
                    }
                    let vertexCount = UInt64(geometry.primitiveCount) * 3
                    let vertexLength = try stridedRangeLength(
                        count: vertexCount,
                        stride: UInt64(geometry.stride),
                        elementSize: geometry.vertexFormat == 6
                            ? 80 : (geometry.vertexFormat == 5
                            ? 56 : (geometry.vertexFormat == 4
                                ? 40 : (geometry.vertexFormat == 3
                                    ? 32 : (geometry.vertexFormat == 2 ? 24 : 12))))
                    )
                    try validateRange(
                        offset: geometry.dataOffset,
                        length: vertexLength,
                        bufferLength: UInt64(dataBuffer.length)
                    )
                } else {
                    guard geometry.indexType == 1 || geometry.indexType == 2,
                          geometry.indexResourceID != 0
                    else {
                        throw GPUBackendError.unsupported("invalid Metal triangle index type")
                    }
                    let indexBuffer = try requireBuffer(geometry.indexResourceID)
                    let indexSize: UInt64 = geometry.indexType == 1 ? 2 : 4
                    let indexLength = UInt64(geometry.primitiveCount) * 3 * indexSize
                    try validateRange(
                        offset: geometry.indexOffset,
                        length: indexLength,
                        bufferLength: UInt64(indexBuffer.length)
                    )
                    triangle.indexBuffer = indexBuffer
                    triangle.indexBufferOffset = try checkedInt(geometry.indexOffset)
                    triangle.indexType = geometry.indexType == 1 ? .uint16 : .uint32
                }

                if geometry.transformResourceID != 0 {
                    let transformBuffer = try requireBuffer(geometry.transformResourceID)
                    try validateRange(
                        offset: geometry.transformOffset,
                        length: 48,
                        bufferLength: UInt64(transformBuffer.length)
                    )
                    triangle.transformationMatrixBuffer = transformBuffer
                    triangle.transformationMatrixBufferOffset = try checkedInt(geometry.transformOffset)
                } else if geometry.transformOffset != 0 {
                    throw GPUBackendError.unsupported("triangle transform offset has no transform buffer")
                }
                return triangle

            case .boundingBoxes:
                guard geometry.stride >= 24,
                      geometry.stride % 4 == 0,
                      geometry.indexResourceID == 0,
                      geometry.indexOffset == 0,
                      geometry.indexType == 0,
                      geometry.vertexFormat == 0,
                      geometry.transformResourceID == 0,
                      geometry.transformOffset == 0
                else {
                    throw GPUBackendError.unsupported("invalid Metal bounding-box geometry")
                }
                let rangeLength = try stridedRangeLength(
                    count: UInt64(geometry.primitiveCount),
                    stride: UInt64(geometry.stride),
                    elementSize: 24
                )
                try validateRange(
                    offset: geometry.dataOffset,
                    length: rangeLength,
                    bufferLength: UInt64(dataBuffer.length)
                )
                let boxes = MTLAccelerationStructureBoundingBoxGeometryDescriptor()
                boxes.boundingBoxBuffer = dataBuffer
                boxes.boundingBoxBufferOffset = try checkedInt(geometry.dataOffset)
                boxes.boundingBoxStride = Int(geometry.stride)
                boxes.boundingBoxCount = Int(geometry.primitiveCount)
                boxes.opaque = geometry.flags & 0x1 != 0
                boxes.allowDuplicateIntersectionFunctionInvocation = geometry.flags & 0x2 == 0
                return boxes
            }
        }

        let sizes = device.accelerationStructureSizes(descriptor: descriptor)
        guard sizes.accelerationStructureSize > 0, sizes.buildScratchBufferSize > 0,
              let structure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(
                length: sizes.buildScratchBufferSize,
                options: .storageModePrivate
              ),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeAccelerationStructureCommandEncoder()
        else {
            throw GPUBackendError.unavailable("Metal acceleration structure allocation failed")
        }
        structure.label = "IMB primitive acceleration structure \(id)"
        commandBuffer.label = "IMB build primitive acceleration structure \(id)"
        encoder.label = "IMB Metal BLAS build"
        encoder.build(
            accelerationStructure: structure,
            descriptor: descriptor,
            scratchBuffer: scratch,
            scratchBufferOffset: 0
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw GPUBackendError.commandFailed(
                commandBuffer.error?.localizedDescription ?? "Metal acceleration structure build failed"
            )
        }
        resource.structure = structure
        resource.childStructureIDs = []
        resource.localTriangleNormals = geometries.count == 1
            ? try? triangleNormals(for: geometries[0])
            : nil
        resource.localTriangleUVs = geometries.count == 1
            ? try? triangleUVs(for: geometries[0])
            : nil
        resource.localTriangleTangents = geometries.count == 1
            ? try? triangleTangents(for: geometries[0])
            : nil
        resource.worldTriangleNormalBuffer = nil
        resource.instanceNormalRangeBuffer = nil
        resource.normalInstanceCount = 0
        resource.triangleUVBuffer = nil
        resource.instanceUVRangeBuffer = nil
        resource.uvInstanceCount = 0
        resource.instanceTextureIndexBuffer = nil
        resource.instanceMaterialTextureChannelBuffer = nil
        resource.worldTriangleTangentBuffer = nil
        resource.instanceTangentRangeBuffer = nil
        resource.tangentInstanceCount = 0
        resource.instanceNormalTextureIndexBuffer = nil
        resource.materialTextureResourceIDs = []
        resource.localMaterialParameters = geometries.count == 1
            ? try? materialParameters(for: geometries[0])
            : nil
        resource.instanceMaterialParameterBuffer = nil
        resource.materialInstanceCount = 0
        accelerationStructures[id] = resource
    }

    public func buildInstanceAccelerationStructure(
        id: UInt64,
        buildFlags: UInt32,
        instances: [InstanceAccelerationStructureInstance]
    ) throws {
        guard var resource = accelerationStructures[id] else {
            throw GPUBackendError.resourceNotFound(id)
        }
        guard device.supportsRaytracing, resource.type == 0 || resource.type == 2,
              !instances.isEmpty, instances.count <= 1_048_576,
              buildFlags & ~UInt32(0x1f) == 0,
              MemoryLayout<MTLAccelerationStructureUserIDInstanceDescriptor>.size == 68
        else {
            throw GPUBackendError.unsupported("invalid instance acceleration structure build")
        }

        var descriptorBytes = Data()
        descriptorBytes.reserveCapacity(instances.count * 68)
        var instancedStructures: [any MTLAccelerationStructure] = []
        instancedStructures.reserveCapacity(instances.count)
        var worldTriangleNormals: [SIMD4<Float>] = []
        var instanceNormalRanges: [SIMD4<UInt32>] = []
        instanceNormalRanges.reserveCapacity(instances.count)
        var worldTriangleTangents: [SIMD4<Float>] = []
        var instanceTangentRanges: [SIMD4<UInt32>] = []
        instanceTangentRanges.reserveCapacity(instances.count)
        var triangleUVs: [SIMD2<Float>] = []
        var instanceUVRanges: [SIMD4<UInt32>] = []
        instanceUVRanges.reserveCapacity(instances.count)
        var instanceTextureIndices: [SIMD4<UInt32>] = []
        instanceTextureIndices.reserveCapacity(instances.count)
        var instanceMaterialTextureChannels: [UInt32] = []
        instanceMaterialTextureChannels.reserveCapacity(instances.count)
        var instanceNormalTextureIndices: [UInt32] = []
        instanceNormalTextureIndices.reserveCapacity(instances.count)
        var materialTextureResourceIDs: [UInt64] = []
        var instanceMaterialParameters: [SIMD4<Float>] = []
        instanceMaterialParameters.reserveCapacity(instances.count * 2)
        var hasMaterialParameters = false
        for instance in instances {
            guard instance.transformationMatrix.count == 12,
                  instance.transformationMatrix.allSatisfy(\.isFinite),
                  instance.options & ~UInt32(0xf) == 0,
                  instance.mask <= 0xff,
                  instance.intersectionFunctionTableOffset <= 0x00ff_ffff,
                  instance.userID <= 0x00ff_ffff,
                  let child = accelerationStructures[instance.accelerationStructureResourceID],
                  child.type == 1 || child.type == 2,
                  let childStructure = child.structure
            else {
                throw GPUBackendError.unsupported("invalid Metal acceleration structure instance")
            }
            for component in instance.transformationMatrix {
                descriptorBytes.appendLittleEndian(component.bitPattern)
            }
            descriptorBytes.appendLittleEndian(instance.options)
            descriptorBytes.appendLittleEndian(instance.mask)
            descriptorBytes.appendLittleEndian(instance.intersectionFunctionTableOffset)
            descriptorBytes.appendLittleEndian(UInt32(instancedStructures.count))
            descriptorBytes.appendLittleEndian(instance.userID)
            instancedStructures.append(childStructure)
            if let parameters = child.localMaterialParameters {
                instanceMaterialParameters.append(SIMD4<Float>(
                    parameters.roughness, parameters.metallic, 0, 1
                ))
                instanceMaterialParameters.append(SIMD4<Float>(
                    parameters.emissionColor,
                    parameters.emissionIntensity
                ))
                hasMaterialParameters = true
            } else {
                instanceMaterialParameters.append(SIMD4<Float>(0.5, 0, 0, 0))
                instanceMaterialParameters.append(SIMD4<Float>(0, 0, 0, 0))
            }

            let normalOffset = worldTriangleNormals.count
            if let localNormals = child.localTriangleNormals,
               normalOffset <= Int(UInt32.max),
               localNormals.values.count <= Int(UInt32.max) - normalOffset,
               localNormals.normalsPerTriangle > 0,
               localNormals.values.count
                    == Int(localNormals.triangleCount * localNormals.normalsPerTriangle) {
                worldTriangleNormals.append(contentsOf: localNormals.values.map {
                    transformedNormal($0, by: instance.transformationMatrix)
                })
                instanceNormalRanges.append(SIMD4<UInt32>(
                    UInt32(normalOffset),
                    localNormals.triangleCount,
                    localNormals.normalsPerTriangle,
                    0
                ))
            } else {
                instanceNormalRanges.append(SIMD4<UInt32>(0, 0, 0, 0))
            }

            let tangentOffset = worldTriangleTangents.count
            if let localTangents = child.localTriangleTangents,
               tangentOffset <= Int(UInt32.max),
               localTangents.values.count <= Int(UInt32.max) - tangentOffset,
               localTangents.values.count == Int(localTangents.triangleCount) * 2 {
                worldTriangleTangents.append(contentsOf: localTangents.values.map {
                    transformedDirection($0, by: instance.transformationMatrix)
                })
                instanceTangentRanges.append(SIMD4<UInt32>(
                    UInt32(tangentOffset), localTangents.triangleCount, 2, 0
                ))
            } else {
                instanceTangentRanges.append(SIMD4<UInt32>(0, 0, 0, 0))
            }

            let uvOffset = triangleUVs.count
            if let localUVs = child.localTriangleUVs,
               uvOffset <= Int(UInt32.max),
               localUVs.values.count <= Int(UInt32.max) - uvOffset,
               localUVs.values.count == Int(localUVs.triangleCount) * 3 {
                triangleUVs.append(contentsOf: localUVs.values)
                instanceUVRanges.append(SIMD4<UInt32>(
                    UInt32(uvOffset), localUVs.triangleCount, 3, 0
                ))
                let appendTextureIndex = { (textureResourceID: UInt64) -> UInt32 in
                    guard textureResourceID != 0,
                          self.images[textureResourceID] != nil
                    else {
                        return UInt32.max
                    }
                    if let existing = materialTextureResourceIDs.firstIndex(
                        of: textureResourceID
                    ) {
                        return UInt32(existing)
                    }
                    guard materialTextureResourceIDs.count < 126 else {
                        return UInt32.max
                    }
                    let textureIndex = UInt32(materialTextureResourceIDs.count)
                    materialTextureResourceIDs.append(textureResourceID)
                    return textureIndex
                }
                if let descriptor = materialTextureDescriptor(
                    userID: instance.userID
                ) {
                    instanceTextureIndices.append(SIMD4<UInt32>(
                        appendTextureIndex(descriptor.resourceIDs[0]),
                        appendTextureIndex(descriptor.resourceIDs[1]),
                        appendTextureIndex(descriptor.resourceIDs[2]),
                        appendTextureIndex(descriptor.resourceIDs[3])
                    ))
                    instanceMaterialTextureChannels.append(
                        descriptor.channels.x
                        | (descriptor.channels.y << 4)
                        | (descriptor.channels.z << 8)
                        | (descriptor.alphaCutout ? 0x8000_0000 : 0)
                    )
                    instanceNormalTextureIndices.append(
                        appendTextureIndex(descriptor.resourceIDs[4])
                    )
                } else {
                    let textureResourceID = UInt64(
                        instance.userID & 0x003f_ffff
                    )
                    let hasLegacyBaseTexture =
                        instance.userID & 0x00c0_0000 == 0x0040_0000
                    instanceTextureIndices.append(SIMD4<UInt32>(
                        hasLegacyBaseTexture
                            ? appendTextureIndex(textureResourceID)
                            : UInt32.max,
                        UInt32.max,
                        UInt32.max,
                        UInt32.max
                    ))
                    instanceMaterialTextureChannels.append(0)
                    instanceNormalTextureIndices.append(UInt32.max)
                }
            } else {
                instanceUVRanges.append(SIMD4<UInt32>(0, 0, 0, 0))
                instanceTextureIndices.append(SIMD4<UInt32>(repeating: UInt32.max))
                instanceMaterialTextureChannels.append(0)
                instanceNormalTextureIndices.append(UInt32.max)
            }
        }

        let instanceBuffer = descriptorBytes.withUnsafeBytes { bytes in
            guard let baseAddress = bytes.baseAddress else { return nil as (any MTLBuffer)? }
            return device.makeBuffer(
                bytes: baseAddress,
                length: descriptorBytes.count,
                options: .storageModeShared
            )
        }
        guard let instanceBuffer else {
            throw GPUBackendError.unavailable("Metal instance descriptor buffer allocation failed")
        }
        instanceBuffer.label = "IMB acceleration structure instances \(id)"

        let descriptor = MTLInstanceAccelerationStructureDescriptor()
        descriptor.instanceDescriptorBuffer = instanceBuffer
        descriptor.instanceDescriptorBufferOffset = 0
        descriptor.instanceDescriptorStride = 68
        descriptor.instanceCount = instances.count
        descriptor.instancedAccelerationStructures = instancedStructures
        descriptor.instanceDescriptorType = .userID
        descriptor.instanceTransformationMatrixLayout = .rowMajor
        var usage: MTLAccelerationStructureUsage = []
        if buildFlags & 0x1 != 0 { usage.insert(.refit) }
        if buildFlags & 0x8 != 0 { usage.insert(.preferFastBuild) }
        if #available(macOS 26.0, *) {
            if buildFlags & 0x4 != 0 { usage.insert(.preferFastIntersection) }
            if buildFlags & 0x10 != 0 { usage.insert(.minimizeMemory) }
        }
        descriptor.usage = usage

        let sizes = device.accelerationStructureSizes(descriptor: descriptor)
        guard sizes.accelerationStructureSize > 0, sizes.buildScratchBufferSize > 0,
              let structure = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = device.makeBuffer(
                length: sizes.buildScratchBufferSize,
                options: .storageModePrivate
              ),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeAccelerationStructureCommandEncoder()
        else {
            throw GPUBackendError.unavailable("Metal instance acceleration structure allocation failed")
        }
        structure.label = "IMB instance acceleration structure \(id)"
        commandBuffer.label = "IMB build instance acceleration structure \(id)"
        encoder.label = "IMB Metal TLAS build"
        encoder.build(
            accelerationStructure: structure,
            descriptor: descriptor,
            scratchBuffer: scratch,
            scratchBufferOffset: 0
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw GPUBackendError.commandFailed(
                commandBuffer.error?.localizedDescription ?? "Metal instance acceleration structure build failed"
            )
        }
        resource.structure = structure
        resource.childStructureIDs = instances.map(\.accelerationStructureResourceID)
        if !worldTriangleNormals.isEmpty {
            resource.worldTriangleNormalBuffer = worldTriangleNormals.withUnsafeBytes { bytes in
                guard let baseAddress = bytes.baseAddress else {
                    return nil as (any MTLBuffer)?
                }
                return device.makeBuffer(
                    bytes: baseAddress,
                    length: bytes.count,
                    options: .storageModeShared
                )
            }
            resource.instanceNormalRangeBuffer = instanceNormalRanges.withUnsafeBytes { bytes in
                guard let baseAddress = bytes.baseAddress else {
                    return nil as (any MTLBuffer)?
                }
                return device.makeBuffer(
                    bytes: baseAddress,
                    length: bytes.count,
                    options: .storageModeShared
                )
            }
            guard resource.worldTriangleNormalBuffer != nil,
                  resource.instanceNormalRangeBuffer != nil
            else {
                throw GPUBackendError.unavailable(
                    "Metal triangle-normal lookup allocation failed"
                )
            }
            resource.normalInstanceCount = UInt32(instanceNormalRanges.count)
        } else {
            resource.worldTriangleNormalBuffer = nil
            resource.instanceNormalRangeBuffer = nil
            resource.normalInstanceCount = 0
        }
        if !worldTriangleTangents.isEmpty {
            resource.worldTriangleTangentBuffer =
                worldTriangleTangents.withUnsafeBytes { bytes in
                    guard let baseAddress = bytes.baseAddress else {
                        return nil as (any MTLBuffer)?
                    }
                    return device.makeBuffer(
                        bytes: baseAddress,
                        length: bytes.count,
                        options: .storageModeShared
                    )
                }
            resource.instanceTangentRangeBuffer =
                instanceTangentRanges.withUnsafeBytes { bytes in
                    guard let baseAddress = bytes.baseAddress else {
                        return nil as (any MTLBuffer)?
                    }
                    return device.makeBuffer(
                        bytes: baseAddress,
                        length: bytes.count,
                        options: .storageModeShared
                    )
                }
            guard resource.worldTriangleTangentBuffer != nil,
                  resource.instanceTangentRangeBuffer != nil
            else {
                throw GPUBackendError.unavailable(
                    "Metal triangle-tangent lookup allocation failed"
                )
            }
            resource.tangentInstanceCount = UInt32(instanceTangentRanges.count)
        } else {
            resource.worldTriangleTangentBuffer = nil
            resource.instanceTangentRangeBuffer = nil
            resource.tangentInstanceCount = 0
        }
        if !triangleUVs.isEmpty, !materialTextureResourceIDs.isEmpty {
            resource.triangleUVBuffer = triangleUVs.withUnsafeBytes { bytes in
                guard let baseAddress = bytes.baseAddress else {
                    return nil as (any MTLBuffer)?
                }
                return device.makeBuffer(
                    bytes: baseAddress,
                    length: bytes.count,
                    options: .storageModeShared
                )
            }
            resource.instanceUVRangeBuffer = instanceUVRanges.withUnsafeBytes { bytes in
                guard let baseAddress = bytes.baseAddress else {
                    return nil as (any MTLBuffer)?
                }
                return device.makeBuffer(
                    bytes: baseAddress,
                    length: bytes.count,
                    options: .storageModeShared
                )
            }
            resource.instanceTextureIndexBuffer = instanceTextureIndices.withUnsafeBytes { bytes in
                guard let baseAddress = bytes.baseAddress else {
                    return nil as (any MTLBuffer)?
                }
                return device.makeBuffer(
                    bytes: baseAddress,
                    length: bytes.count,
                    options: .storageModeShared
                )
            }
            resource.instanceMaterialTextureChannelBuffer =
                instanceMaterialTextureChannels.withUnsafeBytes { bytes in
                    guard let baseAddress = bytes.baseAddress else {
                        return nil as (any MTLBuffer)?
                    }
                    return device.makeBuffer(
                        bytes: baseAddress,
                        length: bytes.count,
                        options: .storageModeShared
                    )
                }
            resource.instanceNormalTextureIndexBuffer =
                instanceNormalTextureIndices.withUnsafeBytes { bytes in
                    guard let baseAddress = bytes.baseAddress else {
                        return nil as (any MTLBuffer)?
                    }
                    return device.makeBuffer(
                        bytes: baseAddress,
                        length: bytes.count,
                        options: .storageModeShared
                    )
                }
            guard resource.triangleUVBuffer != nil,
                  resource.instanceUVRangeBuffer != nil,
                  resource.instanceTextureIndexBuffer != nil,
                  resource.instanceMaterialTextureChannelBuffer != nil,
                  resource.instanceNormalTextureIndexBuffer != nil
            else {
                throw GPUBackendError.unavailable(
                    "Metal triangle-UV lookup allocation failed"
                )
            }
            resource.uvInstanceCount = UInt32(instanceUVRanges.count)
            resource.materialTextureResourceIDs = materialTextureResourceIDs
        } else {
            resource.triangleUVBuffer = nil
            resource.instanceUVRangeBuffer = nil
            resource.uvInstanceCount = 0
            resource.instanceTextureIndexBuffer = nil
            resource.instanceMaterialTextureChannelBuffer = nil
            resource.instanceNormalTextureIndexBuffer = nil
            resource.materialTextureResourceIDs = []
        }
        if hasMaterialParameters {
            resource.instanceMaterialParameterBuffer =
                instanceMaterialParameters.withUnsafeBytes { bytes in
                    guard let baseAddress = bytes.baseAddress else {
                        return nil as (any MTLBuffer)?
                    }
                    return device.makeBuffer(
                        bytes: baseAddress,
                        length: bytes.count,
                        options: .storageModeShared
                    )
                }
            guard resource.instanceMaterialParameterBuffer != nil else {
                throw GPUBackendError.unavailable(
                    "Metal instance-material lookup allocation failed"
                )
            }
            resource.materialInstanceCount = UInt32(instances.count)
        } else {
            resource.instanceMaterialParameterBuffer = nil
            resource.materialInstanceCount = 0
        }
        accelerationStructures[id] = resource
    }

    public func writeBuffer(id: UInt64, offset: UInt64, data: Data) throws {
        let buffer = try requireBuffer(id)
        try validateRange(offset: offset, length: UInt64(data.count), bufferLength: UInt64(buffer.length))
        guard !data.isEmpty else { return }
        data.withUnsafeBytes { source in
            if let baseAddress = source.baseAddress {
                memcpy(buffer.contents().advanced(by: Int(offset)), baseAddress, data.count)
            }
        }
    }

    public func writeImage(id: UInt64, data: Data) throws {
        guard let image = images[id] else {
            throw GPUBackendError.resourceNotFound(id)
        }
        let blockWidth: Int
        let blockHeight: Int
        let blockBytes: Int
        switch image.format {
        case 1, 2:
            (blockWidth, blockHeight, blockBytes) = (1, 1, 4)
        case 3, 6, 7:
            (blockWidth, blockHeight, blockBytes) = (4, 4, 16)
        case 4:
            (blockWidth, blockHeight, blockBytes) = (1, 1, 2)
        case 5:
            (blockWidth, blockHeight, blockBytes) = (1, 1, 8)
        default:
            throw GPUBackendError.unsupported("unsupported Metal image upload format \(image.format)")
        }

        struct UploadPlan {
            let tightOffset: Int
            let stagingOffset: Int
            let rowBytes: Int
            let rowPitch: Int
            let rowCount: Int
            let width: Int
            let height: Int
            let mipLevel: Int
            let slice: Int
        }
        // minimumTextureBufferAlignment(for:) asserts inside Metal for BC
        // pixel formats. Buffer-to-texture blits use Metal's 256-byte row and
        // offset alignment for those compressed blocks.
        let alignment = blockWidth > 1 || blockHeight > 1
            ? 256
            : max(
                1,
                device.minimumTextureBufferAlignment(for: image.texture.pixelFormat)
            )
        let align = { (value: Int) -> Int in
            ((value + alignment - 1) / alignment) * alignment
        }
        var plans: [UploadPlan] = []
        var tightOffset = 0
        var stagingSize = 0
        for slice in 0..<image.arrayLayers {
            var width = image.width
            var height = image.height
            for mipLevel in 0..<image.mipLevels {
                let rowBytes = ((width + blockWidth - 1) / blockWidth) * blockBytes
                let rowCount = (height + blockHeight - 1) / blockHeight
                let tightByteCount = rowBytes * rowCount
                let stagingOffset = align(stagingSize)
                let rowPitch = align(rowBytes)
                plans.append(UploadPlan(
                    tightOffset: tightOffset,
                    stagingOffset: stagingOffset,
                    rowBytes: rowBytes,
                    rowPitch: rowPitch,
                    rowCount: rowCount,
                    width: width,
                    height: height,
                    mipLevel: mipLevel,
                    slice: slice
                ))
                tightOffset += tightByteCount
                stagingSize = stagingOffset + rowPitch * rowCount
                width = max(1, width / 2)
                height = max(1, height / 2)
            }
        }
        guard data.count == tightOffset else {
            throw GPUBackendError.outOfBounds
        }

        if !image.sparse, image.mipLevels == 1, image.arrayLayers == 1,
           blockWidth == 1, blockHeight == 1 {
            data.withUnsafeBytes { source in
                guard let baseAddress = source.baseAddress else { return }
                image.texture.replace(
                    region: MTLRegionMake2D(0, 0, image.width, image.height),
                    mipmapLevel: 0,
                    withBytes: baseAddress,
                    bytesPerRow: plans[0].rowBytes
                )
            }
            return
        }

        guard stagingSize > 0,
              let staging = device.makeBuffer(
                  length: stagingSize,
                  options: .storageModeShared
              )
        else {
            throw GPUBackendError.unavailable("Metal image upload buffer allocation failed")
        }
        data.withUnsafeBytes { source in
            guard let sourceBase = source.baseAddress else { return }
            let destinationBase = staging.contents()
            for plan in plans {
                for row in 0..<plan.rowCount {
                    memcpy(
                        destinationBase.advanced(by: plan.stagingOffset + row * plan.rowPitch),
                        sourceBase.advanced(by: plan.tightOffset + row * plan.rowBytes),
                        plan.rowBytes
                    )
                }
            }
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeBlitCommandEncoder()
        else {
            throw GPUBackendError.unavailable("Metal image upload command creation failed")
        }
        for plan in plans {
            encoder.copy(
                from: staging,
                sourceOffset: plan.stagingOffset,
                sourceBytesPerRow: plan.rowPitch,
                sourceBytesPerImage: plan.rowPitch * plan.rowCount,
                sourceSize: MTLSize(
                    width: plan.width,
                    height: plan.height,
                    depth: 1
                ),
                to: image.texture,
                destinationSlice: plan.slice,
                destinationLevel: plan.mipLevel,
                destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
            )
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if commandBuffer.status == .error {
            throw GPUBackendError.unavailable(
                "Metal image upload failed: \(commandBuffer.error?.localizedDescription ?? "unknown error")"
            )
        }
    }

    public func readBuffer(id: UInt64, offset: UInt64, length: UInt64) throws -> Data {
        let buffer = try requireBuffer(id)
        try validateRange(offset: offset, length: length, bufferLength: UInt64(buffer.length))
        guard length > 0 else { return Data() }
        return Data(bytes: buffer.contents().advanced(by: Int(offset)), count: Int(length))
    }

    public func readImage(id: UInt64) throws -> Data {
        guard let image = images[id] else {
            throw GPUBackendError.resourceNotFound(id)
        }
        let bytesPerRow = image.width * 4
        var result = Data(count: bytesPerRow * image.height)
        result.withUnsafeMutableBytes { destination in
            guard let baseAddress = destination.baseAddress else { return }
            image.texture.getBytes(
                baseAddress,
                bytesPerRow: bytesPerRow,
                from: MTLRegionMake2D(0, 0, image.width, image.height),
                mipmapLevel: 0
            )
        }
        return result
    }

    public func submitNoop(fenceID: UInt64) throws {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw GPUBackendError.unavailable("Metal command buffer creation failed")
        }
        commandBuffer.label = "IMB no-op fence \(fenceID)"
        commandBuffer.commit()
        commandBuffers[fenceID] = commandBuffer
    }

    public func submitAddUInt32(
        bufferID: UInt64,
        elementCount: UInt32,
        addend: UInt32,
        fenceID: UInt64
    ) throws {
        guard elementCount > 0 else {
            throw GPUBackendError.outOfBounds
        }
        let buffer = try requireBuffer(bufferID)
        let byteCount = UInt64(elementCount) * UInt64(MemoryLayout<UInt32>.size)
        try validateRange(offset: 0, length: byteCount, bufferLength: UInt64(buffer.length))
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw GPUBackendError.unavailable("Metal compute command creation failed")
        }

        commandBuffer.label = "IMB add-u32 fence \(fenceID)"
        encoder.label = "IMB add-u32"
        encoder.setComputePipelineState(addPipeline)
        encoder.setBuffer(buffer, offset: 0, index: 0)
        var addendValue = addend
        encoder.setBytes(&addendValue, length: MemoryLayout<UInt32>.size, index: 1)
        let width = min(Int(elementCount), addPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: Int(elementCount), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffers[fenceID] = commandBuffer
    }

    public func submitCompute(
        pipelineID: UInt64,
        groupCountX: UInt32,
        groupCountY: UInt32,
        groupCountZ: UInt32,
        bindings: [ComputeBinding],
        pushConstants: Data,
        fenceID: UInt64
    ) throws {
        guard groupCountX > 0, groupCountY > 0, groupCountZ > 0,
              let resource = computePipelines[pipelineID],
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw GPUBackendError.resourceNotFound(pipelineID)
        }

        commandBuffer.label = "IMB SPIR-V compute fence \(fenceID)"
        encoder.label = "IMB translated SPIR-V compute"
        encoder.setComputePipelineState(resource.pipeline)
        if let pushConstantBufferIndex = resource.pushConstantBufferIndex {
            guard !pushConstants.isEmpty, pushConstants.count <= 4_096 else {
                throw GPUBackendError.unsupported(
                    "Metal compute pipeline requires a bounded push-constant payload"
                )
            }
            pushConstants.withUnsafeBytes { bytes in
                if let baseAddress = bytes.baseAddress {
                    encoder.setBytes(
                        baseAddress,
                        length: pushConstants.count,
                        index: pushConstantBufferIndex
                    )
                }
            }
        } else if !pushConstants.isEmpty {
            throw GPUBackendError.unsupported(
                "compute push constants were supplied to a pipeline without a push-constant buffer"
            )
        }

        var argumentBuffers: [any MTLBuffer] = []
        var texelBufferTextures: [any MTLTexture] = []
        let bindingsBySet = Dictionary(grouping: bindings, by: { Int($0.descriptorSet) })
        for (descriptorSet, setBindings) in bindingsBySet.sorted(by: { $0.key < $1.key }) {
            guard descriptorSet >= 0, descriptorSet < 32,
                  resource.argumentBufferSets.contains(descriptorSet)
            else {
                throw GPUBackendError.unsupported(
                    "Metal compute pipeline has no argument buffer for descriptor set \(descriptorSet)"
                )
            }
            let argumentEncoder = resource.function.makeArgumentEncoder(
                bufferIndex: descriptorSet
            )
            guard argumentEncoder.encodedLength > 0,
                  let argumentBuffer = device.makeBuffer(
                      length: argumentEncoder.encodedLength,
                      options: .storageModeShared
                  )
            else {
                throw GPUBackendError.unavailable(
                    "Metal argument buffer allocation failed for descriptor set \(descriptorSet)"
                )
            }
            argumentEncoder.setArgumentBuffer(argumentBuffer, offset: 0)
            for binding in setBindings {
                let metalIndexValue = UInt64(binding.binding) + UInt64(binding.arrayElement)
                guard metalIndexValue <= UInt64(Int.max) else {
                    throw GPUBackendError.outOfBounds
                }
                let metalIndex = Int(metalIndexValue)
                switch binding.kind {
                case .bufferRead, .bufferReadWrite:
                    let buffer = try requireBuffer(binding.resourceID)
                    try validateRange(
                        offset: binding.offset,
                        length: binding.length,
                        bufferLength: UInt64(buffer.length)
                    )
                    argumentEncoder.setBuffer(
                        buffer,
                        offset: try checkedInt(binding.offset),
                        index: metalIndex
                    )
                    var usage: MTLResourceUsage = .read
                    if binding.kind == .bufferReadWrite {
                        usage.insert(.write)
                    }
                    encoder.useResource(buffer, usage: usage)
                case .texelBufferRead, .texelBufferReadWrite:
                    let buffer = try requireBuffer(binding.resourceID)
                    try validateRange(
                        offset: binding.offset,
                        length: binding.length,
                        bufferLength: UInt64(buffer.length)
                    )
                    let texel = try metalTexelBufferFormat(binding.format)
                    guard binding.length % texel.bytesPerTexel == 0,
                          binding.length / texel.bytesPerTexel > 0,
                          binding.length / texel.bytesPerTexel <= UInt64(Int.max)
                    else {
                        throw GPUBackendError.outOfBounds
                    }
                    let minimumAlignment = UInt64(
                        device.minimumTextureBufferAlignment(
                            for: texel.pixelFormat
                        )
                    )
                    guard minimumAlignment > 0,
                          binding.offset % minimumAlignment == 0
                    else {
                        throw GPUBackendError.unsupported(
                            "Metal texel-buffer offset does not meet \(minimumAlignment)-byte alignment"
                        )
                    }
                    let usage: MTLTextureUsage = binding.kind == .texelBufferReadWrite
                        ? [.shaderRead, .shaderWrite]
                        : [.shaderRead]
                    let descriptor = MTLTextureDescriptor.textureBufferDescriptor(
                        with: texel.pixelFormat,
                        width: Int(binding.length / texel.bytesPerTexel),
                        resourceOptions: .storageModeShared,
                        usage: usage
                    )
                    guard let texture = buffer.makeTexture(
                        descriptor: descriptor,
                        offset: try checkedInt(binding.offset),
                        bytesPerRow: try checkedInt(binding.length)
                    ) else {
                        throw GPUBackendError.unavailable(
                            "Metal failed to create a buffer-backed texture"
                        )
                    }
                    argumentEncoder.setTexture(texture, index: metalIndex)
                    var resourceUsage: MTLResourceUsage = .read
                    if binding.kind == .texelBufferReadWrite {
                        resourceUsage.insert(.write)
                    }
                    encoder.useResource(texture, usage: resourceUsage)
                    texelBufferTextures.append(texture)
                case .textureRead, .textureReadWrite:
                    guard binding.offset == 0, binding.length == 0,
                          let image = images[binding.resourceID]
                    else {
                        throw GPUBackendError.resourceNotFound(binding.resourceID)
                    }
                    argumentEncoder.setTexture(image.texture, index: metalIndex)
                    var usage: MTLResourceUsage = .read
                    if binding.kind == .textureReadWrite {
                        usage.insert(.write)
                    }
                    encoder.useResource(image.texture, usage: usage)
                }
            }
            encoder.setBuffer(argumentBuffer, offset: 0, index: descriptorSet)
            argumentBuffers.append(argumentBuffer)
        }

        encoder.dispatchThreadgroups(
            MTLSize(
                width: Int(groupCountX),
                height: Int(groupCountY),
                depth: Int(groupCountZ)
            ),
            threadsPerThreadgroup: resource.threadsPerThreadgroup
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffers[fenceID] = commandBuffer
        _ = argumentBuffers
        _ = texelBufferTextures
    }

    public func submitTriangle(imageID: UInt64, clearRGBA8: UInt32, fenceID: UInt64) throws {
        guard let image = images[imageID] else {
            throw GPUBackendError.resourceNotFound(imageID)
        }
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw GPUBackendError.unavailable("Metal render command buffer creation failed")
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = image.texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(
            red: Double(clearRGBA8 & 0xff) / 255.0,
            green: Double((clearRGBA8 >> 8) & 0xff) / 255.0,
            blue: Double((clearRGBA8 >> 16) & 0xff) / 255.0,
            alpha: Double((clearRGBA8 >> 24) & 0xff) / 255.0
        )
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw GPUBackendError.unavailable("Metal render command encoder creation failed")
        }
        commandBuffer.label = "IMB triangle fence \(fenceID)"
        encoder.label = "IMB fixed offscreen triangle"
        encoder.setRenderPipelineState(trianglePipeline)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffers[fenceID] = commandBuffer
    }

    public func submitIndexedUI(
        imageID: UInt64,
        vertexBufferID: UInt64,
        indexBufferID: UInt64,
        vertexBufferOffset: UInt64,
        indexBufferOffset: UInt64,
        width: UInt32,
        height: UInt32,
        clearRGBA8: UInt32,
        draws: [IndexedUIDraw],
        fenceID: UInt64
    ) throws {
        guard let image = images[imageID], image.format == 2,
              image.width == Int(width), image.height == Int(height),
              width > 0, height > 0, !draws.isEmpty
        else {
            throw GPUBackendError.unsupported("Kit UI requires a matching BGRA8 target and at least one draw")
        }
        let vertexBuffer = try requireBuffer(vertexBufferID)
        let indexBuffer = try requireBuffer(indexBufferID)
        guard vertexBufferOffset <= UInt64(vertexBuffer.length),
              indexBufferOffset <= UInt64(indexBuffer.length)
        else {
            throw GPUBackendError.outOfBounds
        }
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw GPUBackendError.unavailable("Metal Kit UI command buffer creation failed")
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = image.texture
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(
            red: Double(clearRGBA8 & 0xff) / 255.0,
            green: Double((clearRGBA8 >> 8) & 0xff) / 255.0,
            blue: Double((clearRGBA8 >> 16) & 0xff) / 255.0,
            alpha: Double((clearRGBA8 >> 24) & 0xff) / 255.0
        )
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            throw GPUBackendError.unavailable("Metal Kit UI render encoder creation failed")
        }
        commandBuffer.label = "IMB Kit UI fence \(fenceID)"
        encoder.label = "IMB real Isaac Kit UI"
        encoder.setRenderPipelineState(uiPipeline)
        encoder.setViewport(MTLViewport(
            originX: 0,
            originY: 0,
            width: Double(width),
            height: Double(height),
            znear: 0,
            zfar: 1
        ))
        var invScreenSize = SIMD2<Float>(1.0 / Float(width), 1.0 / Float(height))
        encoder.setVertexBytes(&invScreenSize, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        encoder.setFragmentSamplerState(uiSampler, index: 0)

        for draw in draws where draw.indexCount > 0 {
            let maximumX = min(UInt64(width), UInt64(draw.scissorX) + UInt64(draw.scissorWidth))
            let maximumY = min(UInt64(height), UInt64(draw.scissorY) + UInt64(draw.scissorHeight))
            guard UInt64(draw.scissorX) < maximumX, UInt64(draw.scissorY) < maximumY else { continue }
            let indexOffset = indexBufferOffset + UInt64(draw.firstIndex) * 4
            let indexBytes = UInt64(draw.indexCount) * 4
            let vertexByteOffset = Int64(vertexBufferOffset) + Int64(draw.vertexOffset) * 20
            guard indexOffset <= UInt64(indexBuffer.length),
                  indexBytes <= UInt64(indexBuffer.length) - indexOffset,
                  vertexByteOffset >= 0,
                  vertexByteOffset <= Int64(vertexBuffer.length)
            else {
                throw GPUBackendError.outOfBounds
            }
            encoder.setScissorRect(MTLScissorRect(
                x: Int(draw.scissorX),
                y: Int(draw.scissorY),
                width: Int(maximumX - UInt64(draw.scissorX)),
                height: Int(maximumY - UInt64(draw.scissorY))
            ))
            encoder.setVertexBuffer(vertexBuffer, offset: Int(vertexByteOffset), index: 0)
            let texture = images[draw.textureID]?.texture ?? whiteTexture
            encoder.setFragmentTexture(texture, index: 0)
            encoder.drawIndexedPrimitives(
                type: .triangle,
                indexCount: Int(draw.indexCount),
                indexType: .uint32,
                indexBuffer: indexBuffer,
                indexBufferOffset: Int(indexOffset),
                instanceCount: 1,
                baseVertex: 0,
                baseInstance: 0
            )
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffers[fenceID] = commandBuffer
        frameImages[fenceID] = imageID
    }

    public func submitRayTrace(
        imageID: UInt64,
        accelerationStructureID: UInt64,
        width: UInt32,
        height: UInt32,
        missRGBA8: UInt32,
        hitRGBA8: UInt32,
        camera: RayCamera?,
        sphereLight: RaySphereLight?,
        distantLight: RayDistantLight?,
        domeLight: RayDomeLight?,
        fenceID: UInt64
    ) throws {
        guard let pipeline = rayTracePipeline, device.supportsRaytracing else {
            throw GPUBackendError.unavailable("Metal ray dispatch is unavailable")
        }
        guard let image = images[imageID], image.width == Int(width), image.height == Int(height),
              width > 0, height > 0
        else {
            throw GPUBackendError.unsupported("ray dispatch requires a matching non-empty image")
        }
        guard let acceleration = accelerationStructures[accelerationStructureID],
              acceleration.type == 0,
              let structure = acceleration.structure
        else {
            throw GPUBackendError.unsupported("ray dispatch requires a built top-level acceleration structure")
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            throw GPUBackendError.unavailable("Metal ray dispatch command creation failed")
        }
        commandBuffer.label = "IMB ray dispatch fence \(fenceID)"
        encoder.label = "IMB real Metal acceleration-structure intersection"
        encoder.setComputePipelineState(pipeline)
        encoder.setAccelerationStructure(structure, bufferIndex: 0)
        encoder.setTexture(image.texture, index: 0)
        encoder.setTexture(sceneMaterialTexture, index: 1)
        for textureIndex in 0..<126 {
            let materialTexture: any MTLTexture
            if textureIndex < acceleration.materialTextureResourceIDs.count,
               let resource = images[
                    acceleration.materialTextureResourceIDs[textureIndex]
               ] {
                materialTexture = resource.texture
            } else {
                materialTexture = whiteTexture
            }
            encoder.setTexture(materialTexture, index: 2 + textureIndex)
            encoder.useResource(materialTexture, usage: .read)
        }
        var extent = SIMD2<UInt32>(width, height)
        var colors = SIMD2<UInt32>(missRGBA8, hitRGBA8)
        var cameraPositionAndFov = SIMD4<Float>(
            camera?.position.x ?? 0,
            camera?.position.y ?? 0,
            camera?.position.z ?? 0,
            camera?.verticalFOVRadians ?? 0
        )
        var cameraForwardAndNear = SIMD4<Float>(
            camera?.forward.x ?? 0,
            camera?.forward.y ?? 0,
            camera?.forward.z ?? 0,
            camera?.nearDistance ?? 0
        )
        var cameraUpAndFar = SIMD4<Float>(
            camera?.up.x ?? 0,
            camera?.up.y ?? 0,
            camera?.up.z ?? 0,
            camera?.farDistance ?? 0
        )
        var cameraOptions: UInt32 = (camera == nil ? 0 : 1)
            | (sphereLight == nil ? 0 : 2)
            | (hasSceneMaterialTexture ? 4 : 0)
            | (distantLight == nil ? 0 : 8)
            | (domeLight == nil ? 0 : 16)
        let shadowMode = ProcessInfo.processInfo.environment[
            "IMB_RAY_SHADOWS"
        ]?.lowercased() ?? "always"
        let shadowInstanceLimit = Int(
            ProcessInfo.processInfo.environment[
                "IMB_RAY_SHADOW_INSTANCE_LIMIT"
            ] ?? "256"
        ) ?? 256
        let tracesHardShadows: Bool
        switch shadowMode {
        case "1", "on", "true", "always":
            tracesHardShadows = true
        case "0", "off", "false", "never":
            tracesHardShadows = false
        default:
            tracesHardShadows = acceleration.childStructureIDs.count
                <= max(shadowInstanceLimit, 0)
        }
        if tracesHardShadows {
            cameraOptions |= 32
        }
        let pixelStepMode = ProcessInfo.processInfo.environment[
            "IMB_RAY_PIXEL_STEP"
        ]?.lowercased() ?? "1"
        let requestedPixelStep = Int(pixelStepMode)
        let rayPixelStepValue: Int
        if let requestedPixelStep {
            rayPixelStepValue = min(max(requestedPixelStep, 1), 8)
        } else if pixelStepMode == "adaptive" {
            if acceleration.childStructureIDs.count > 512 {
                rayPixelStepValue = 4
            } else if acceleration.childStructureIDs.count > 256 {
                rayPixelStepValue = 2
            } else {
                rayPixelStepValue = 1
            }
        } else {
            rayPixelStepValue = 1
        }
        var rayPixelStep = UInt32(rayPixelStepValue)
        if !rayQualityReported {
            let message = "imb-host: Metal ray quality hardShadows="
                + "\(tracesHardShadows) mode=\(shadowMode) "
                + "pixelStep=\(rayPixelStepValue) "
                + "pixelStepMode=\(pixelStepMode) "
                + "instances=\(acceleration.childStructureIDs.count) "
                + "adaptiveLimit=\(max(shadowInstanceLimit, 0))\n"
            FileHandle.standardError.write(Data(message.utf8))
            rayQualityReported = true
        }
        var sphereLightPositionAndIntensity = SIMD4<Float>(
            sphereLight?.position.x ?? 0,
            sphereLight?.position.y ?? 0,
            sphereLight?.position.z ?? 0,
            sphereLight?.intensity ?? 0
        )
        var sphereLightColorAndRadius = SIMD4<Float>(
            sphereLight?.color.x ?? 0,
            sphereLight?.color.y ?? 0,
            sphereLight?.color.z ?? 0,
            sphereLight?.radius ?? 0
        )
        var distantLightDirectionAndIntensity = SIMD4<Float>(
            distantLight?.direction.x ?? 0,
            distantLight?.direction.y ?? 0,
            distantLight?.direction.z ?? 0,
            distantLight?.intensity ?? 0
        )
        var distantLightColorAndAngle = SIMD4<Float>(
            distantLight?.color.x ?? 0,
            distantLight?.color.y ?? 0,
            distantLight?.color.z ?? 0,
            distantLight?.angleDegrees ?? 0
        )
        var domeLightColorAndIntensity = SIMD4<Float>(
            domeLight?.color.x ?? 0,
            domeLight?.color.y ?? 0,
            domeLight?.color.z ?? 0,
            domeLight?.intensity ?? 0
        )
        encoder.setBytes(&extent, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        encoder.setBytes(&colors, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
        encoder.setBytes(
            &cameraPositionAndFov,
            length: MemoryLayout<SIMD4<Float>>.size,
            index: 3
        )
        encoder.setBytes(
            &cameraForwardAndNear,
            length: MemoryLayout<SIMD4<Float>>.size,
            index: 4
        )
        encoder.setBytes(
            &cameraUpAndFar,
            length: MemoryLayout<SIMD4<Float>>.size,
            index: 5
        )
        encoder.setBytes(&cameraOptions, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(
            &sphereLightPositionAndIntensity,
            length: MemoryLayout<SIMD4<Float>>.size,
            index: 7
        )
        encoder.setBytes(
            &sphereLightColorAndRadius,
            length: MemoryLayout<SIMD4<Float>>.size,
            index: 8
        )
        encoder.setBytes(
            &distantLightDirectionAndIntensity,
            length: MemoryLayout<SIMD4<Float>>.size,
            index: 9
        )
        encoder.setBytes(
            &distantLightColorAndAngle,
            length: MemoryLayout<SIMD4<Float>>.size,
            index: 10
        )
        encoder.setBytes(
            &domeLightColorAndIntensity,
            length: MemoryLayout<SIMD4<Float>>.size,
            index: 11
        )
        var fallbackNormal = SIMD4<Float>(0, 0, 0, 0)
        var fallbackRange = SIMD4<UInt32>(0, 0, 0, 0)
        var normalInstanceCount = acceleration.normalInstanceCount
        if let normalBuffer = acceleration.worldTriangleNormalBuffer,
           let rangeBuffer = acceleration.instanceNormalRangeBuffer {
            encoder.setBuffer(normalBuffer, offset: 0, index: 12)
            encoder.setBuffer(rangeBuffer, offset: 0, index: 13)
            encoder.useResource(normalBuffer, usage: .read)
            encoder.useResource(rangeBuffer, usage: .read)
        } else {
            encoder.setBytes(
                &fallbackNormal,
                length: MemoryLayout<SIMD4<Float>>.size,
                index: 12
            )
            encoder.setBytes(
                &fallbackRange,
                length: MemoryLayout<SIMD4<UInt32>>.size,
                index: 13
            )
            normalInstanceCount = 0
        }
        encoder.setBytes(
            &normalInstanceCount,
            length: MemoryLayout<UInt32>.size,
            index: 14
        )
        var fallbackUV = SIMD2<Float>(0, 0)
        var fallbackUVRange = SIMD4<UInt32>(0, 0, 0, 0)
        var fallbackTextureIndices = SIMD4<UInt32>(repeating: UInt32.max)
        var fallbackMaterialTextureChannels: UInt32 = 0
        var fallbackNormalTextureIndex = UInt32.max
        var uvInstanceCount = acceleration.uvInstanceCount
        var textureInstanceCount = acceleration.uvInstanceCount
        var materialTextureCount = UInt32(
            min(acceleration.materialTextureResourceIDs.count, 126)
        )
        if let uvBuffer = acceleration.triangleUVBuffer,
           let uvRangeBuffer = acceleration.instanceUVRangeBuffer,
           let textureIndexBuffer = acceleration.instanceTextureIndexBuffer,
           let textureChannelBuffer =
               acceleration.instanceMaterialTextureChannelBuffer,
           let normalTextureIndexBuffer =
               acceleration.instanceNormalTextureIndexBuffer {
            encoder.setBuffer(uvBuffer, offset: 0, index: 15)
            encoder.setBuffer(uvRangeBuffer, offset: 0, index: 16)
            encoder.setBuffer(textureIndexBuffer, offset: 0, index: 18)
            encoder.setBuffer(textureChannelBuffer, offset: 0, index: 23)
            encoder.setBuffer(normalTextureIndexBuffer, offset: 0, index: 27)
            encoder.useResource(uvBuffer, usage: .read)
            encoder.useResource(uvRangeBuffer, usage: .read)
            encoder.useResource(textureIndexBuffer, usage: .read)
            encoder.useResource(textureChannelBuffer, usage: .read)
            encoder.useResource(normalTextureIndexBuffer, usage: .read)
        } else {
            encoder.setBytes(
                &fallbackUV,
                length: MemoryLayout<SIMD2<Float>>.size,
                index: 15
            )
            encoder.setBytes(
                &fallbackUVRange,
                length: MemoryLayout<SIMD4<UInt32>>.size,
                index: 16
            )
            encoder.setBytes(
                &fallbackTextureIndices,
                length: MemoryLayout<SIMD4<UInt32>>.size,
                index: 18
            )
            encoder.setBytes(
                &fallbackMaterialTextureChannels,
                length: MemoryLayout<UInt32>.size,
                index: 23
            )
            encoder.setBytes(
                &fallbackNormalTextureIndex,
                length: MemoryLayout<UInt32>.size,
                index: 27
            )
            uvInstanceCount = 0
            textureInstanceCount = 0
            materialTextureCount = 0
        }
        encoder.setBytes(
            &uvInstanceCount,
            length: MemoryLayout<UInt32>.size,
            index: 17
        )
        encoder.setBytes(
            &textureInstanceCount,
            length: MemoryLayout<UInt32>.size,
            index: 19
        )
        encoder.setBytes(
            &materialTextureCount,
            length: MemoryLayout<UInt32>.size,
            index: 20
        )
        var fallbackMaterialParameters = SIMD4<Float>(0.5, 0, 0, 0)
        var materialInstanceCount = acceleration.materialInstanceCount
        if let materialBuffer = acceleration.instanceMaterialParameterBuffer {
            encoder.setBuffer(materialBuffer, offset: 0, index: 21)
            encoder.useResource(materialBuffer, usage: .read)
        } else {
            encoder.setBytes(
                &fallbackMaterialParameters,
                length: MemoryLayout<SIMD4<Float>>.size,
                index: 21
            )
            materialInstanceCount = 0
        }
        encoder.setBytes(
            &materialInstanceCount,
            length: MemoryLayout<UInt32>.size,
            index: 22
        )
        var fallbackTangent = SIMD4<Float>(0, 0, 0, 0)
        var fallbackTangentRange = SIMD4<UInt32>(0, 0, 0, 0)
        var tangentInstanceCount = acceleration.tangentInstanceCount
        if let tangentBuffer = acceleration.worldTriangleTangentBuffer,
           let tangentRangeBuffer = acceleration.instanceTangentRangeBuffer {
            encoder.setBuffer(tangentBuffer, offset: 0, index: 24)
            encoder.setBuffer(tangentRangeBuffer, offset: 0, index: 25)
            encoder.useResource(tangentBuffer, usage: .read)
            encoder.useResource(tangentRangeBuffer, usage: .read)
        } else {
            encoder.setBytes(
                &fallbackTangent,
                length: MemoryLayout<SIMD4<Float>>.size,
                index: 24
            )
            encoder.setBytes(
                &fallbackTangentRange,
                length: MemoryLayout<SIMD4<UInt32>>.size,
                index: 25
            )
            tangentInstanceCount = 0
        }
        encoder.setBytes(
            &tangentInstanceCount,
            length: MemoryLayout<UInt32>.size,
            index: 26
        )
        encoder.setBytes(
            &rayPixelStep,
            length: MemoryLayout<UInt32>.size,
            index: 28
        )
        for childID in acceleration.childStructureIDs {
            guard let child = accelerationStructures[childID]?.structure else {
                throw GPUBackendError.resourceNotFound(childID)
            }
            encoder.useResource(child, usage: .read)
        }
        let threadWidth = pipeline.threadExecutionWidth
        let threadHeight = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / threadWidth, 8))
        let dispatchWidth = (Int(width) + rayPixelStepValue - 1)
            / rayPixelStepValue
        let dispatchHeight = (Int(height) + rayPixelStepValue - 1)
            / rayPixelStepValue
        encoder.dispatchThreads(
            MTLSize(width: dispatchWidth, height: dispatchHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadWidth, height: threadHeight, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffers[fenceID] = commandBuffer
        frameImages[fenceID] = imageID
    }

    public func waitFence(id: UInt64) throws -> Bool {
        guard let commandBuffer = commandBuffers.removeValue(forKey: id) else {
            throw GPUBackendError.resourceNotFound(id)
        }
        commandBuffer.waitUntilCompleted()
        guard commandBuffer.status == .completed else {
            throw GPUBackendError.commandFailed(
                commandBuffer.error?.localizedDescription ?? "Metal command buffer did not complete"
            )
        }
        if let imageID = frameImages.removeValue(forKey: id) {
            do {
                try writeFrameIfRequested(imageID: imageID, fenceID: id)
            } catch {
                // IMB_FRAME_OUTPUT is an optional viewer/debug sink. A full
                // disk or an unwritable path must not turn a completed Metal
                // command buffer into VK_ERROR_DEVICE_LOST in the guest.
                frameOutputDisabledAfterError = true
                let message = "imb-container-host: disabling frame output after write failure: \(error)\n"
                FileHandle.standardError.write(Data(message.utf8))
            }
        }
        return true
    }

    public func reset() {
        for commandBuffer in commandBuffers.values {
            commandBuffer.waitUntilCompleted()
        }
        commandBuffers.removeAll()
        frameImages.removeAll()
        buffers.removeAll()
        images.removeAll()
        computePipelines.removeAll()
        computePipelineCache.removeAll()
        accelerationStructures.removeAll()
    }

    private func requireBuffer(_ id: UInt64) throws -> any MTLBuffer {
        guard let buffer = buffers[id] else {
            throw GPUBackendError.resourceNotFound(id)
        }
        return buffer
    }

    private func materialTextureDescriptor(
        userID: UInt32
    ) -> MaterialTextureDescriptor? {
        guard userID & 0x00c0_0000 == 0x00c0_0000 else { return nil }
        let descriptorID = UInt64(userID & 0x003f_ffff)
        guard descriptorID != 0,
              let buffer = buffers[descriptorID],
              buffer.length >= 48
        else {
            return nil
        }
        let prefix = (0..<4).map { index in
            UInt32(littleEndian: buffer.contents().advanced(by: index * 4)
                .assumingMemoryBound(to: UInt32.self).pointee)
        }
        let version = prefix[1]
        let wordCount = version == 2 ? 14 : 12
        guard prefix[0] == 0x314d_424d,
              version == 1 || version == 2,
              buffer.length >= wordCount * 4
        else {
            return nil
        }
        let words = (0..<wordCount).map { index in
            UInt32(littleEndian: buffer.contents().advanced(by: index * 4)
                .assumingMemoryBound(to: UInt32.self).pointee)
        }
        let flags = words[2]
        let supportedFlags: UInt32 = version == 2 ? 0x3f : 0x0f
        guard flags & ~supportedFlags == 0,
              flags & (version == 2 ? 0x3e : 0x0e) != 0
        else {
            return nil
        }
        var resourceIDs = stride(
            from: 4,
            through: version == 2 ? 12 : 10,
            by: 2
        ).map { index in
            UInt64(words[index]) | (UInt64(words[index + 1]) << 32)
        }
        if version == 1 { resourceIDs.append(0) }
        for index in 0..<5 {
            let present = flags & (UInt32(1) << UInt32(index)) != 0
            if present {
                guard resourceIDs[index] != 0,
                      images[resourceIDs[index]] != nil
                else {
                    return nil
                }
            } else if resourceIDs[index] != 0 {
                return nil
            }
        }
        let packedChannels = words[3]
        let roughnessChannel = packedChannels & 0x7
        let metallicChannel = (packedChannels >> 4) & 0x7
        let emissionChannel = (packedChannels >> 8) & 0x7
        guard (flags & 2 == 0 || roughnessChannel <= 3),
              (flags & 4 == 0 || metallicChannel <= 3),
              (flags & 8 == 0 || emissionChannel == 4)
        else {
            return nil
        }
        return MaterialTextureDescriptor(
            resourceIDs: resourceIDs,
            channels: SIMD4<UInt32>(
                roughnessChannel, metallicChannel, emissionChannel, 0
            ),
            alphaCutout: flags & 0x20 != 0
        )
    }

    private func validateRange(offset: UInt64, length: UInt64, bufferLength: UInt64) throws {
        guard offset <= bufferLength, length <= bufferLength - offset else {
            throw GPUBackendError.outOfBounds
        }
    }

    private func checkedInt(_ value: UInt64) throws -> Int {
        guard value <= UInt64(Int.max) else { throw GPUBackendError.outOfBounds }
        return Int(value)
    }

    private func stridedRangeLength(count: UInt64, stride: UInt64, elementSize: UInt64) throws -> UInt64 {
        guard count > 0 else { throw GPUBackendError.outOfBounds }
        let (span, multipliedOverflow) = (count - 1).multipliedReportingOverflow(by: stride)
        let (length, addedOverflow) = span.addingReportingOverflow(elementSize)
        guard !multipliedOverflow, !addedOverflow else { throw GPUBackendError.outOfBounds }
        return length
    }

    private func triangleNormals(
        for geometry: PrimitiveAccelerationStructureGeometry
    ) throws -> TriangleNormalData? {
        guard geometry.kind == .triangles,
              (1...6).contains(geometry.vertexFormat)
        else {
            return nil
        }
        let vertexBuffer = try requireBuffer(geometry.dataResourceID)
        let indexBuffer = geometry.indexType == 0
            ? nil : try requireBuffer(geometry.indexResourceID)
        let transform: [Float]?
        if geometry.transformResourceID != 0 {
            let transformBuffer = try requireBuffer(geometry.transformResourceID)
            try validateRange(
                offset: geometry.transformOffset,
                length: 48,
                bufferLength: UInt64(transformBuffer.length)
            )
            transform = (0..<12).map { component in
                readFloat(
                    from: transformBuffer,
                    offset: Int(geometry.transformOffset) + component * 4
                )
            }
        } else {
            transform = nil
        }

        var normals: [SIMD4<Float>] = []
        let normalsPerTriangle: UInt32 = geometry.vertexFormat >= 2 ? 3 : 1
        normals.reserveCapacity(Int(geometry.primitiveCount * normalsPerTriangle))
        for triangleIndex in 0..<Int(geometry.primitiveCount) {
            var points: [SIMD3<Float>] = []
            points.reserveCapacity(3)
            var authoredNormals: [SIMD4<Float>] = []
            if geometry.vertexFormat >= 2 {
                authoredNormals.reserveCapacity(3)
            }
            for corner in 0..<3 {
                let linearIndex = triangleIndex * 3 + corner
                let vertexIndex: UInt64
                switch geometry.indexType {
                case 0:
                    vertexIndex = UInt64(linearIndex)
                case 1:
                    guard let indexBuffer else { return nil }
                    let offset = try checkedInt(geometry.indexOffset) + linearIndex * 2
                    guard offset <= indexBuffer.length - 2 else { return nil }
                    vertexIndex = UInt64(
                        indexBuffer.contents().advanced(by: offset)
                            .assumingMemoryBound(to: UInt16.self).pointee
                    )
                case 2:
                    guard let indexBuffer else { return nil }
                    let offset = try checkedInt(geometry.indexOffset) + linearIndex * 4
                    guard offset <= indexBuffer.length - 4 else { return nil }
                    vertexIndex = UInt64(
                        indexBuffer.contents().advanced(by: offset)
                            .assumingMemoryBound(to: UInt32.self).pointee
                    )
                default:
                    return nil
                }
                let (strideOffset, overflow) = vertexIndex.multipliedReportingOverflow(
                    by: UInt64(geometry.stride)
                )
                let (vertexOffset, addedOverflow) = geometry.dataOffset
                    .addingReportingOverflow(strideOffset)
                guard !overflow, !addedOverflow,
                      vertexOffset <= UInt64(vertexBuffer.length),
                      UInt64(vertexBuffer.length) - vertexOffset
                        >= (geometry.vertexFormat == 6
                            ? 80 : (geometry.vertexFormat == 5
                            ? 56 : (geometry.vertexFormat == 4
                                ? 40 : (geometry.vertexFormat == 3
                                    ? 32 : (geometry.vertexFormat == 2 ? 24 : 12)))))
                else {
                    return nil
                }
                let offset = try checkedInt(vertexOffset)
                var point = SIMD3<Float>(
                    readFloat(from: vertexBuffer, offset: offset),
                    readFloat(from: vertexBuffer, offset: offset + 4),
                    readFloat(from: vertexBuffer, offset: offset + 8)
                )
                if let transform {
                    point = SIMD3<Float>(
                        transform[0] * point.x + transform[1] * point.y
                            + transform[2] * point.z + transform[3],
                        transform[4] * point.x + transform[5] * point.y
                            + transform[6] * point.z + transform[7],
                        transform[8] * point.x + transform[9] * point.y
                            + transform[10] * point.z + transform[11]
                    )
                }
                points.append(point)
                if geometry.vertexFormat >= 2 {
                    var normal = SIMD4<Float>(
                        readFloat(from: vertexBuffer, offset: offset + 12),
                        readFloat(from: vertexBuffer, offset: offset + 16),
                        readFloat(from: vertexBuffer, offset: offset + 20),
                        0
                    )
                    if let transform {
                        normal = transformedNormal(normal, by: transform)
                    } else {
                        let lengthSquared = normal.x * normal.x
                            + normal.y * normal.y + normal.z * normal.z
                        guard lengthSquared > 0.000000000001,
                              lengthSquared.isFinite
                        else {
                            return nil
                        }
                        normal *= 1 / sqrt(lengthSquared)
                    }
                    guard normal.x.isFinite, normal.y.isFinite, normal.z.isFinite,
                          normal.x * normal.x + normal.y * normal.y
                            + normal.z * normal.z > 0.000001
                    else {
                        return nil
                    }
                    authoredNormals.append(normal)
                }
            }
            if geometry.vertexFormat >= 2 {
                guard authoredNormals.count == 3 else { return nil }
                normals.append(contentsOf: authoredNormals)
                continue
            }
            let edgeA = points[1] - points[0]
            let edgeB = points[2] - points[0]
            let cross = SIMD3<Float>(
                edgeA.y * edgeB.z - edgeA.z * edgeB.y,
                edgeA.z * edgeB.x - edgeA.x * edgeB.z,
                edgeA.x * edgeB.y - edgeA.y * edgeB.x
            )
            let lengthSquared = cross.x * cross.x + cross.y * cross.y
                + cross.z * cross.z
            if lengthSquared > 0.000000000001, lengthSquared.isFinite {
                let inverseLength = 1 / sqrt(lengthSquared)
                normals.append(SIMD4<Float>(cross * inverseLength, 0))
            } else {
                normals.append(SIMD4<Float>(0, 0, 0, 0))
            }
        }
        return TriangleNormalData(
            values: normals,
            triangleCount: geometry.primitiveCount,
            normalsPerTriangle: normalsPerTriangle
        )
    }

    private func triangleUVs(
        for geometry: PrimitiveAccelerationStructureGeometry
    ) throws -> TriangleUVData? {
        guard geometry.kind == .triangles,
              (3...6).contains(geometry.vertexFormat)
        else {
            return nil
        }
        let vertexBuffer = try requireBuffer(geometry.dataResourceID)
        let indexBuffer = geometry.indexType == 0
            ? nil : try requireBuffer(geometry.indexResourceID)
        var values: [SIMD2<Float>] = []
        values.reserveCapacity(Int(geometry.primitiveCount) * 3)
        for triangleIndex in 0..<Int(geometry.primitiveCount) {
            for corner in 0..<3 {
                let linearIndex = triangleIndex * 3 + corner
                let vertexIndex: UInt64
                switch geometry.indexType {
                case 0:
                    vertexIndex = UInt64(linearIndex)
                case 1:
                    guard let indexBuffer else { return nil }
                    let offset = try checkedInt(geometry.indexOffset) + linearIndex * 2
                    guard indexBuffer.length >= 2, offset <= indexBuffer.length - 2 else {
                        return nil
                    }
                    vertexIndex = UInt64(
                        indexBuffer.contents().advanced(by: offset)
                            .assumingMemoryBound(to: UInt16.self).pointee
                    )
                case 2:
                    guard let indexBuffer else { return nil }
                    let offset = try checkedInt(geometry.indexOffset) + linearIndex * 4
                    guard indexBuffer.length >= 4, offset <= indexBuffer.length - 4 else {
                        return nil
                    }
                    vertexIndex = UInt64(
                        indexBuffer.contents().advanced(by: offset)
                            .assumingMemoryBound(to: UInt32.self).pointee
                    )
                default:
                    return nil
                }
                let (strideOffset, multipliedOverflow) = vertexIndex
                    .multipliedReportingOverflow(by: UInt64(geometry.stride))
                let (vertexOffset, addedOverflow) = geometry.dataOffset
                    .addingReportingOverflow(strideOffset)
                guard !multipliedOverflow, !addedOverflow,
                      vertexOffset <= UInt64(vertexBuffer.length),
                      UInt64(vertexBuffer.length) - vertexOffset >= 32
                else {
                    return nil
                }
                let offset = try checkedInt(vertexOffset)
                let uv = SIMD2<Float>(
                    readFloat(from: vertexBuffer, offset: offset + 24),
                    readFloat(from: vertexBuffer, offset: offset + 28)
                )
                guard uv.x.isFinite, uv.y.isFinite else { return nil }
                values.append(uv)
            }
        }
        return TriangleUVData(values: values, triangleCount: geometry.primitiveCount)
    }

    private func materialParameters(
        for geometry: PrimitiveAccelerationStructureGeometry
    ) throws -> MaterialParameterData? {
        guard geometry.kind == .triangles,
              (4...6).contains(geometry.vertexFormat)
        else {
            return nil
        }
        let vertexBuffer = try requireBuffer(geometry.dataResourceID)
        let firstVertexIndex: UInt64
        switch geometry.indexType {
        case 0:
            firstVertexIndex = 0
        case 1:
            let indexBuffer = try requireBuffer(geometry.indexResourceID)
            let offset = try checkedInt(geometry.indexOffset)
            guard indexBuffer.length >= 2, offset <= indexBuffer.length - 2 else {
                return nil
            }
            firstVertexIndex = UInt64(
                indexBuffer.contents().advanced(by: offset)
                    .assumingMemoryBound(to: UInt16.self).pointee
            )
        case 2:
            let indexBuffer = try requireBuffer(geometry.indexResourceID)
            let offset = try checkedInt(geometry.indexOffset)
            guard indexBuffer.length >= 4, offset <= indexBuffer.length - 4 else {
                return nil
            }
            firstVertexIndex = UInt64(
                indexBuffer.contents().advanced(by: offset)
                    .assumingMemoryBound(to: UInt32.self).pointee
            )
        default:
            return nil
        }
        let (strideOffset, multipliedOverflow) = firstVertexIndex
            .multipliedReportingOverflow(by: UInt64(geometry.stride))
        let (vertexOffset, addedOverflow) = geometry.dataOffset
            .addingReportingOverflow(strideOffset)
        guard !multipliedOverflow, !addedOverflow,
              vertexOffset <= UInt64(vertexBuffer.length),
              UInt64(vertexBuffer.length) - vertexOffset
                >= (geometry.vertexFormat == 6
                    ? 80 : (geometry.vertexFormat == 5 ? 56 : 40))
        else {
            return nil
        }
        let offset = try checkedInt(vertexOffset)
        let roughness = readFloat(from: vertexBuffer, offset: offset + 32)
        let metallic = readFloat(from: vertexBuffer, offset: offset + 36)
        guard roughness.isFinite, metallic.isFinite,
              (0...1).contains(roughness), (0...1).contains(metallic)
        else {
            return nil
        }
        var emissionColor = SIMD3<Float>(repeating: 0)
        var emissionIntensity: Float = 0
        if geometry.vertexFormat >= 5 {
            emissionColor = SIMD3<Float>(
                readFloat(from: vertexBuffer, offset: offset + 40),
                readFloat(from: vertexBuffer, offset: offset + 44),
                readFloat(from: vertexBuffer, offset: offset + 48)
            )
            emissionIntensity = readFloat(
                from: vertexBuffer, offset: offset + 52
            )
            guard emissionColor.x.isFinite, emissionColor.y.isFinite,
                  emissionColor.z.isFinite, emissionIntensity.isFinite,
                  (0...1).contains(emissionColor.x),
                  (0...1).contains(emissionColor.y),
                  (0...1).contains(emissionColor.z),
                  (0...1_000_000).contains(emissionIntensity)
            else {
                return nil
            }
        }
        return MaterialParameterData(
            roughness: roughness,
            metallic: metallic,
            emissionColor: emissionColor,
            emissionIntensity: emissionIntensity
        )
    }

    private func triangleTangents(
        for geometry: PrimitiveAccelerationStructureGeometry
    ) throws -> TriangleTangentData? {
        guard geometry.kind == .triangles,
              geometry.vertexFormat == 6,
              geometry.indexType == 0,
              geometry.stride >= 80
        else {
            return nil
        }
        let vertexBuffer = try requireBuffer(geometry.dataResourceID)
        var values: [SIMD4<Float>] = []
        values.reserveCapacity(Int(geometry.primitiveCount) * 2)
        for triangleIndex in 0..<Int(geometry.primitiveCount) {
            let vertexIndex = UInt64(triangleIndex * 3)
            let (strideOffset, multipliedOverflow) = vertexIndex
                .multipliedReportingOverflow(by: UInt64(geometry.stride))
            let (vertexOffset, addedOverflow) = geometry.dataOffset
                .addingReportingOverflow(strideOffset)
            guard !multipliedOverflow, !addedOverflow,
                  vertexOffset <= UInt64(vertexBuffer.length),
                  UInt64(vertexBuffer.length) - vertexOffset >= 80
            else {
                return nil
            }
            let offset = try checkedInt(vertexOffset)
            var tangent = SIMD4<Float>(
                readFloat(from: vertexBuffer, offset: offset + 56),
                readFloat(from: vertexBuffer, offset: offset + 60),
                readFloat(from: vertexBuffer, offset: offset + 64),
                0
            )
            var bitangent = SIMD4<Float>(
                readFloat(from: vertexBuffer, offset: offset + 68),
                readFloat(from: vertexBuffer, offset: offset + 72),
                readFloat(from: vertexBuffer, offset: offset + 76),
                0
            )
            let tangentLengthSquared = tangent.x * tangent.x
                + tangent.y * tangent.y + tangent.z * tangent.z
            let bitangentLengthSquared = bitangent.x * bitangent.x
                + bitangent.y * bitangent.y + bitangent.z * bitangent.z
            guard tangentLengthSquared.isFinite,
                  bitangentLengthSquared.isFinite,
                  tangentLengthSquared > 0.000000000001,
                  bitangentLengthSquared > 0.000000000001
            else {
                return nil
            }
            tangent *= 1 / sqrt(tangentLengthSquared)
            bitangent *= 1 / sqrt(bitangentLengthSquared)
            values.append(tangent)
            values.append(bitangent)
        }
        return TriangleTangentData(
            values: values,
            triangleCount: geometry.primitiveCount
        )
    }

    private func transformedNormal(
        _ normal: SIMD4<Float>,
        by transform: [Float]
    ) -> SIMD4<Float> {
        guard transform.count == 12 else { return SIMD4<Float>(0, 0, 0, 0) }
        // Cofactor(A) * n is det(A) * inverse-transpose(A) * n. It preserves
        // the winding produced by transforming the triangle and avoids a
        // divide for singular transforms.
        let x = (transform[5] * transform[10] - transform[6] * transform[9]) * normal.x
            + (transform[6] * transform[8] - transform[4] * transform[10]) * normal.y
            + (transform[4] * transform[9] - transform[5] * transform[8]) * normal.z
        let y = (transform[2] * transform[9] - transform[1] * transform[10]) * normal.x
            + (transform[0] * transform[10] - transform[2] * transform[8]) * normal.y
            + (transform[1] * transform[8] - transform[0] * transform[9]) * normal.z
        let z = (transform[1] * transform[6] - transform[2] * transform[5]) * normal.x
            + (transform[2] * transform[4] - transform[0] * transform[6]) * normal.y
            + (transform[0] * transform[5] - transform[1] * transform[4]) * normal.z
        let lengthSquared = x * x + y * y + z * z
        guard lengthSquared > 0.000000000001, lengthSquared.isFinite else {
            return SIMD4<Float>(0, 0, 0, 0)
        }
        let inverseLength = 1 / sqrt(lengthSquared)
        return SIMD4<Float>(x * inverseLength, y * inverseLength, z * inverseLength, 0)
    }

    private func transformedDirection(
        _ direction: SIMD4<Float>,
        by transform: [Float]
    ) -> SIMD4<Float> {
        guard transform.count == 12 else { return SIMD4<Float>(0, 0, 0, 0) }
        let transformed = SIMD4<Float>(
            transform[0] * direction.x + transform[1] * direction.y
                + transform[2] * direction.z,
            transform[4] * direction.x + transform[5] * direction.y
                + transform[6] * direction.z,
            transform[8] * direction.x + transform[9] * direction.y
                + transform[10] * direction.z,
            0
        )
        let lengthSquared = transformed.x * transformed.x
            + transformed.y * transformed.y + transformed.z * transformed.z
        guard lengthSquared.isFinite, lengthSquared > 0.000000000001 else {
            return SIMD4<Float>(0, 0, 0, 0)
        }
        return transformed * (1 / sqrt(lengthSquared))
    }

    private func readFloat(from buffer: any MTLBuffer, offset: Int) -> Float {
        Float(bitPattern: buffer.contents().advanced(by: offset)
            .assumingMemoryBound(to: UInt32.self).pointee)
    }

    private func writeFrameIfRequested(imageID: UInt64, fenceID: UInt64) throws {
        guard !frameOutputDisabledAfterError,
              let path = ProcessInfo.processInfo.environment["IMB_FRAME_OUTPUT"], !path.isEmpty,
              let image = images[imageID]
        else {
            return
        }
        // In native-viewer mode the RGBA ray target is an input texture for
        // Kit's later BGRA UI composition. Publishing it directly races the UI
        // frame and makes the viewer alternate between "scene only" and the
        // complete Isaac application. Only the final BGRA Kit UI target should
        // become the viewer's frame in that mode.
        if ProcessInfo.processInfo.environment["IMB_FRAME_OUTPUT_UI_ONLY"] == "1",
           image.format != 2 {
            return
        }
        let pixels = try readImage(id: imageID)
        var ppm = Data("P6\n\(image.width) \(image.height)\n255\n".utf8)
        ppm.reserveCapacity(ppm.count + image.width * image.height * 3)
        for pixel in stride(from: 0, to: pixels.count, by: 4) {
            if image.format == 2 {
                ppm.append(pixels[pixel + 2])
                ppm.append(pixels[pixel + 1])
                ppm.append(pixels[pixel])
            } else {
                ppm.append(pixels[pixel])
                ppm.append(pixels[pixel + 1])
                ppm.append(pixels[pixel + 2])
            }
        }
        let outputPath: String
        if ProcessInfo.processInfo.environment["IMB_FRAME_OUTPUT_ALL_LATEST"] == "1" {
            // Keep one current diagnostic frame per Metal image instead of
            // writing an unbounded file for every Kit update.
            outputPath = "\(path).image-\(imageID).ppm"
        } else if ProcessInfo.processInfo.environment["IMB_FRAME_OUTPUT_ALL"] == "1" {
            outputPath = "\(path).image-\(imageID).fence-\(fenceID).ppm"
        } else {
            outputPath = path
        }
        try ppm.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
    }
}
#else
public final class MetalGPUBackend: BridgeGPUBackend, @unchecked Sendable {
    public static func makeDefault() -> MetalGPUBackend? { nil }
    public var supportsSPIRVCompute: Bool { false }
    public var supportsAccelerationStructures: Bool { false }
    public var supportsRayDispatch: Bool { false }
    public var supportsSparseImages: Bool { false }
    public func createBuffer(id: UInt64, size: UInt64, options: UInt32) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func createImage(id: UInt64, width: UInt32, height: UInt32, format: UInt32, options: UInt32) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func querySparseImageProperties(format: UInt32, textureType: UInt32, sampleCount: UInt32) throws -> SparseImageProperties { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func createSparseImage(id: UInt64, virtualSize: UInt64, width: UInt32, height: UInt32, format: UInt32, mipLevels: UInt32, arrayLayers: UInt32, sampleCount: UInt32, textureType: UInt32) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func updateSparseImageMapping(id: UInt64, map: Bool, mipLevel: UInt32, slice: UInt32, tileX: UInt32, tileY: UInt32, tileZ: UInt32, tileWidth: UInt32, tileHeight: UInt32, tileDepth: UInt32) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func destroyBuffer(id: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func destroyImage(id: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func createComputePipeline(id: UInt64, spirv: Data, entryPoint: String) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func destroyComputePipeline(id: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func createAccelerationStructure(id: UInt64, type: UInt32, requestedSize: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func destroyAccelerationStructure(id: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func buildPrimitiveAccelerationStructure(id: UInt64, buildFlags: UInt32, geometries: [PrimitiveAccelerationStructureGeometry]) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func buildInstanceAccelerationStructure(id: UInt64, buildFlags: UInt32, instances: [InstanceAccelerationStructureInstance]) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func writeBuffer(id: UInt64, offset: UInt64, data: Data) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func writeImage(id: UInt64, data: Data) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func readBuffer(id: UInt64, offset: UInt64, length: UInt64) throws -> Data { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func readImage(id: UInt64) throws -> Data { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func submitNoop(fenceID: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func submitAddUInt32(bufferID: UInt64, elementCount: UInt32, addend: UInt32, fenceID: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func submitCompute(pipelineID: UInt64, groupCountX: UInt32, groupCountY: UInt32, groupCountZ: UInt32, bindings: [ComputeBinding], pushConstants: Data, fenceID: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func submitTriangle(imageID: UInt64, clearRGBA8: UInt32, fenceID: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func submitIndexedUI(imageID: UInt64, vertexBufferID: UInt64, indexBufferID: UInt64, vertexBufferOffset: UInt64, indexBufferOffset: UInt64, width: UInt32, height: UInt32, clearRGBA8: UInt32, draws: [IndexedUIDraw], fenceID: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func submitRayTrace(imageID: UInt64, accelerationStructureID: UInt64, width: UInt32, height: UInt32, missRGBA8: UInt32, hitRGBA8: UInt32, camera: RayCamera?, sphereLight: RaySphereLight?, distantLight: RayDistantLight?, domeLight: RayDomeLight?, fenceID: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func waitFence(id: UInt64) throws -> Bool { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func reset() {}
}
#endif
