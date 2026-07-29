import Foundation

#if canImport(Metal)
import Metal
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

    private struct AccelerationStructureResource {
        let type: UInt32
        let requestedSize: UInt64
        var structure: (any MTLAccelerationStructure)?
        var childStructureIDs: [UInt64]
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

    kernel void imb_trace_probe(
        instance_acceleration_structure accelerationStructure [[buffer(0)]],
        constant uint2 &extent [[buffer(1)]],
        constant uint2 &colors [[buffer(2)]],
        texture2d<float, access::write> output [[texture(0)]],
        uint2 threadID [[thread_position_in_grid]])
    {
        if (threadID.x >= extent.x || threadID.y >= extent.y) return;

        const float2 normalized = (float2(threadID) + 0.5) / float2(extent);
        const bool isaacSceneView = colors.y == 0xffe08c30 || colors.y == 0xffe08c31;
        const bool simpleGridView = colors.y == 0xffe08c31;
        ray probeRay;
        if (isaacSceneView) {
            const float3 cameraPosition = float3(4.5, 4.5, 3.5);
            const float3 cameraTarget = float3(0.0, 0.0, 0.5);
            const float3 forward = normalize(cameraTarget - cameraPosition);
            const float3 right = normalize(cross(forward, float3(0.0, 0.0, 1.0)));
            const float3 up = normalize(cross(right, forward));
            const float2 plane = normalized * 2.0 - 1.0;
            const float aspect = float(extent.x) / float(extent.y);
            probeRay.origin = cameraPosition;
            probeRay.direction = normalize(
                forward + right * plane.x * aspect * 0.52 - up * plane.y * 0.52
            );
        } else {
            const float2 plane = normalized * 2.2 - 1.1;
            probeRay.origin = float3(plane.x, -plane.y, -2.0);
            probeRay.direction = float3(0.0, 0.0, 1.0);
        }
        probeRay.min_distance = 0.001;
        probeRay.max_distance = 100.0;

        intersector<triangle_data, instancing> triangleIntersector;
        const auto intersection = triangleIntersector.intersect(probeRay, accelerationStructure);
        float4 color;
        float gridDistance = INFINITY;
        if (simpleGridView && probeRay.direction.z < -0.0001) {
            gridDistance = -probeRay.origin.z / probeRay.direction.z;
        }
        const bool hasTriangleHit = intersection.type != intersection_type::none;
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
            const float3 floorColor = float3(0.025, 0.075, 0.13);
            const float3 minorColor = float3(0.20, 0.48, 0.72) * distanceFade;
            const float3 majorColor = float3(0.62, 0.82, 1.0) * distanceFade;
            float3 gridColor = mix(floorColor, minorColor, minorLine * 0.72);
            gridColor = mix(gridColor, majorColor, majorLine);
            const float axisWidth = lineWidth * 1.8;
            const float xAxis = 1.0 - smoothstep(axisWidth, axisWidth * 2.0, abs(worldPosition.y));
            const float yAxis = 1.0 - smoothstep(axisWidth, axisWidth * 2.0, abs(worldPosition.x));
            gridColor = mix(gridColor, float3(0.86, 0.18, 0.14), xAxis);
            gridColor = mix(gridColor, float3(0.18, 0.74, 0.32), yAxis);
            color = float4(gridColor, 1.0);
        } else if (isaacSceneView && hasTriangleHit) {
            const float shade = clamp(1.25 - intersection.distance * 0.12, 0.35, 1.0);
            const float instanceTint = 0.82 + 0.06 * float(intersection.instance_id % 3);
            color = float4(float3(0.12, 0.48, 0.95) * shade * instanceTint, 1.0);
        } else {
            const uint packed = intersection.type == intersection_type::none ? colors.x : colors.y;
            color = float4(
                float(packed & 0xff),
                float((packed >> 8) & 0xff),
                float((packed >> 16) & 0xff),
                float((packed >> 24) & 0xff)
            ) / 255.0;
        }
        output.write(color, threadID);
    }
    """

    private let device: any MTLDevice
    private let queue: any MTLCommandQueue
    private let addPipeline: any MTLComputePipelineState
    private let trianglePipeline: any MTLRenderPipelineState
    private let uiPipeline: any MTLRenderPipelineState
    private let uiSampler: any MTLSamplerState
    private let whiteTexture: any MTLTexture
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
        return try? MetalGPUBackend(device: device, spirvCompiler: compiler)
    }

    public init(device: any MTLDevice, spirvCompiler: SPIRVCrossCompiler? = nil) throws {
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
            throw GPUBackendError.unsupported("nonzero Metal buffer options are not defined by IMB protocol 1.10")
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
        descriptor.usage = [.shaderRead, .shaderWrite]
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
            childStructureIDs: []
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
                guard geometry.vertexFormat == 1,
                      geometry.stride >= 12,
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
                        elementSize: 12
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
        let bytesPerRow = image.width * 4
        guard data.count == bytesPerRow * image.height else {
            throw GPUBackendError.outOfBounds
        }
        data.withUnsafeBytes { source in
            guard let baseAddress = source.baseAddress else { return }
            image.texture.replace(
                region: MTLRegionMake2D(0, 0, image.width, image.height),
                mipmapLevel: 0,
                withBytes: baseAddress,
                bytesPerRow: bytesPerRow
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
        var extent = SIMD2<UInt32>(width, height)
        var colors = SIMD2<UInt32>(missRGBA8, hitRGBA8)
        encoder.setBytes(&extent, length: MemoryLayout<SIMD2<UInt32>>.size, index: 1)
        encoder.setBytes(&colors, length: MemoryLayout<SIMD2<UInt32>>.size, index: 2)
        for childID in acceleration.childStructureIDs {
            guard let child = accelerationStructures[childID]?.structure else {
                throw GPUBackendError.resourceNotFound(childID)
            }
            encoder.useResource(child, usage: .read)
        }
        let threadWidth = pipeline.threadExecutionWidth
        let threadHeight = max(1, min(pipeline.maxTotalThreadsPerThreadgroup / threadWidth, 8))
        encoder.dispatchThreads(
            MTLSize(width: Int(width), height: Int(height), depth: 1),
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
        if ProcessInfo.processInfo.environment["IMB_FRAME_OUTPUT_ALL"] == "1" {
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
    public func submitRayTrace(imageID: UInt64, accelerationStructureID: UInt64, width: UInt32, height: UInt32, missRGBA8: UInt32, hitRGBA8: UInt32, fenceID: UInt64) throws { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func waitFence(id: UInt64) throws -> Bool { throw GPUBackendError.unavailable("Metal is unavailable") }
    public func reset() {}
}
#endif
