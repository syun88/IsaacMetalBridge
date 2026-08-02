import Foundation
import Testing
@testable import IMBHostCore

private let testMetal = MetalCapabilities(
    available: true,
    deviceName: "Test Metal Device",
    unifiedMemory: true,
    rayTracing: false,
    maxBufferLength: 4096
)

private let testMetalWithRayTracing = MetalCapabilities(
    available: true,
    deviceName: "Test Metal RT Device",
    unifiedMemory: true,
    rayTracing: true,
    maxBufferLength: 4096
)

private final class TestGPUBackend: BridgeGPUBackend, @unchecked Sendable {
    private struct Image {
        let width: Int
        let height: Int
        var pixels: Data
    }

    private var buffers: [UInt64: Data] = [:]
    private var images: [UInt64: Image] = [:]
    private var fences: Set<UInt64> = []
    private var computePipelines: Set<UInt64> = []
    private var accelerationStructures: Set<UInt64> = []
    private(set) var lastRayCamera: RayCamera?
    private(set) var lastRaySphereLight: RaySphereLight?
    private(set) var lastRayDistantLight: RayDistantLight?
    private(set) var lastRayDomeLight: RayDomeLight?

    var supportsSPIRVCompute: Bool { true }
    var supportsAccelerationStructures: Bool { true }
    var supportsRayDispatch: Bool { true }
    var supportsSparseImages: Bool { true }

    func createBuffer(id: UInt64, size: UInt64, options: UInt32) throws {
        guard size <= UInt64(Int.max), options == 0 else { throw GPUBackendError.outOfBounds }
        buffers[id] = Data(repeating: 0, count: Int(size))
    }

    func createImage(id: UInt64, width: UInt32, height: UInt32, format: UInt32, options: UInt32) throws {
        guard width > 0, height > 0, (format == 1 || format == 2), options == 0 else {
            throw GPUBackendError.unsupported("invalid test image")
        }
        let byteCount = Int(width) * Int(height) * 4
        images[id] = Image(
            width: Int(width),
            height: Int(height),
            pixels: Data(repeating: 0, count: byteCount)
        )
    }

    func querySparseImageProperties(
        format: UInt32,
        textureType: UInt32,
        sampleCount: UInt32
    ) throws -> SparseImageProperties {
        guard (format >= 1 && format <= 5),
              textureType == 1,
              sampleCount == 1
        else {
            throw GPUBackendError.unsupported("invalid test sparse image query")
        }
        return SparseImageProperties(
            tileWidth: (format == 3 || format == 4) ? 128 : 64,
            tileHeight: format == 5 ? 32 : (format == 4 ? 64 : (format == 3 ? 128 : 64)),
            tileDepth: 1,
            tileSizeBytes: 16_384
        )
    }

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
    ) throws {
        guard virtualSize > 0, width > 0, height > 0,
              (format >= 1 && format <= 5),
              mipLevels > 0, arrayLayers > 0, sampleCount == 1, textureType == 1
        else {
            throw GPUBackendError.unsupported("invalid test sparse image")
        }
        images[id] = Image(width: Int(width), height: Int(height), pixels: Data())
    }

    func updateSparseImageMapping(
        id: UInt64,
        map _: Bool,
        mipLevel _: UInt32,
        slice _: UInt32,
        tileX _: UInt32,
        tileY _: UInt32,
        tileZ _: UInt32,
        tileWidth: UInt32,
        tileHeight: UInt32,
        tileDepth: UInt32
    ) throws {
        guard images[id] != nil, tileWidth > 0, tileHeight > 0, tileDepth > 0 else {
            throw GPUBackendError.outOfBounds
        }
    }

    func destroyBuffer(id: UInt64) throws {
        guard buffers.removeValue(forKey: id) != nil else { throw GPUBackendError.resourceNotFound(id) }
    }

    func destroyImage(id: UInt64) throws {
        guard images.removeValue(forKey: id) != nil else { throw GPUBackendError.resourceNotFound(id) }
    }

    func createComputePipeline(id: UInt64, spirv: Data, entryPoint: String) throws {
        guard spirv.count >= 4, entryPoint == "main" else {
            throw GPUBackendError.unsupported("invalid test compute pipeline")
        }
        computePipelines.insert(id)
    }

    func destroyComputePipeline(id: UInt64) throws {
        guard computePipelines.remove(id) != nil else { throw GPUBackendError.resourceNotFound(id) }
    }

    func submitCompute(
        pipelineID: UInt64,
        groupCountX: UInt32,
        groupCountY: UInt32,
        groupCountZ: UInt32,
        bindings: [ComputeBinding],
        pushConstants _: Data,
        fenceID: UInt64
    ) throws {
        guard computePipelines.contains(pipelineID),
              groupCountX > 0, groupCountY > 0, groupCountZ > 0,
              bindings.allSatisfy({
                  ($0.kind == .bufferRead || $0.kind == .bufferReadWrite
                      || $0.kind == .texelBufferRead || $0.kind == .texelBufferReadWrite)
                      ? buffers[$0.resourceID] != nil
                      : images[$0.resourceID] != nil
              })
        else {
            throw GPUBackendError.unsupported("invalid test compute dispatch")
        }
        fences.insert(fenceID)
    }

    func createAccelerationStructure(id: UInt64, type: UInt32, requestedSize: UInt64) throws {
        guard type <= 2, requestedSize > 0 else {
            throw GPUBackendError.unsupported("invalid test acceleration structure")
        }
        accelerationStructures.insert(id)
    }

    func destroyAccelerationStructure(id: UInt64) throws {
        guard accelerationStructures.remove(id) != nil else { throw GPUBackendError.resourceNotFound(id) }
    }

    func buildPrimitiveAccelerationStructure(
        id: UInt64,
        buildFlags: UInt32,
        geometries: [PrimitiveAccelerationStructureGeometry]
    ) throws {
        guard accelerationStructures.contains(id), buildFlags & ~UInt32(0x1f) == 0,
              !geometries.isEmpty
        else {
            throw GPUBackendError.unsupported("invalid test acceleration structure build")
        }
    }

    func buildInstanceAccelerationStructure(
        id: UInt64,
        buildFlags: UInt32,
        instances: [InstanceAccelerationStructureInstance]
    ) throws {
        guard accelerationStructures.contains(id), buildFlags & ~UInt32(0x1f) == 0,
              !instances.isEmpty,
              instances.allSatisfy({ accelerationStructures.contains($0.accelerationStructureResourceID) })
        else {
            throw GPUBackendError.unsupported("invalid test instance acceleration structure build")
        }
    }

    func writeBuffer(id: UInt64, offset: UInt64, data: Data) throws {
        guard var buffer = buffers[id],
              offset <= UInt64(buffer.count),
              UInt64(data.count) <= UInt64(buffer.count) - offset
        else { throw GPUBackendError.outOfBounds }
        buffer.replaceSubrange(Int(offset)..<(Int(offset) + data.count), with: data)
        buffers[id] = buffer
    }

    func writeImage(id: UInt64, data: Data) throws {
        guard var image = images[id], data.count == image.pixels.count else {
            throw GPUBackendError.outOfBounds
        }
        image.pixels = data
        images[id] = image
    }

    func readBuffer(id: UInt64, offset: UInt64, length: UInt64) throws -> Data {
        guard let buffer = buffers[id],
              offset <= UInt64(buffer.count),
              length <= UInt64(buffer.count) - offset
        else { throw GPUBackendError.outOfBounds }
        return buffer.subdata(in: Int(offset)..<(Int(offset + length)))
    }

    func readImage(id: UInt64) throws -> Data {
        guard let image = images[id] else { throw GPUBackendError.resourceNotFound(id) }
        return image.pixels
    }

    func submitNoop(fenceID: UInt64) throws {
        fences.insert(fenceID)
    }

    func submitAddUInt32(bufferID: UInt64, elementCount: UInt32, addend: UInt32, fenceID: UInt64) throws {
        guard var buffer = buffers[bufferID], UInt64(elementCount) * 4 <= UInt64(buffer.count) else {
            throw GPUBackendError.outOfBounds
        }
        for index in 0..<Int(elementCount) {
            let offset = index * 4
            let value: UInt32 = try buffer.readLittleEndian(at: offset)
            var replacement = Data()
            replacement.appendLittleEndian(value &+ addend)
            buffer.replaceSubrange(offset..<(offset + 4), with: replacement)
        }
        buffers[bufferID] = buffer
        fences.insert(fenceID)
    }

    func submitTriangle(imageID: UInt64, clearRGBA8: UInt32, fenceID: UInt64) throws {
        guard var image = images[imageID] else { throw GPUBackendError.resourceNotFound(imageID) }
        let clear = Data([
            UInt8(clearRGBA8 & 0xff),
            UInt8((clearRGBA8 >> 8) & 0xff),
            UInt8((clearRGBA8 >> 16) & 0xff),
            UInt8((clearRGBA8 >> 24) & 0xff),
        ])
        for offset in stride(from: 0, to: image.pixels.count, by: 4) {
            image.pixels.replaceSubrange(offset..<(offset + 4), with: clear)
        }
        let centerOffset = ((image.height / 2) * image.width + image.width / 2) * 4
        image.pixels.replaceSubrange(centerOffset..<(centerOffset + 4), with: Data([255, 0, 0, 255]))
        images[imageID] = image
        fences.insert(fenceID)
    }

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
    ) throws {
        guard images[imageID] != nil,
              buffers[vertexBufferID] != nil,
              buffers[indexBufferID] != nil,
              !draws.isEmpty
        else {
            throw GPUBackendError.resourceNotFound(imageID)
        }
        fences.insert(fenceID)
    }

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
    ) throws {
        guard var image = images[imageID],
              accelerationStructures.contains(accelerationStructureID),
              image.width == Int(width), image.height == Int(height)
        else {
            throw GPUBackendError.resourceNotFound(accelerationStructureID)
        }
        let miss = Data([
            UInt8(missRGBA8 & 0xff), UInt8((missRGBA8 >> 8) & 0xff),
            UInt8((missRGBA8 >> 16) & 0xff), UInt8((missRGBA8 >> 24) & 0xff),
        ])
        for offset in stride(from: 0, to: image.pixels.count, by: 4) {
            image.pixels.replaceSubrange(offset..<(offset + 4), with: miss)
        }
        let centerOffset = ((image.height / 2) * image.width + image.width / 2) * 4
        image.pixels.replaceSubrange(centerOffset..<(centerOffset + 4), with: Data([
            UInt8(hitRGBA8 & 0xff), UInt8((hitRGBA8 >> 8) & 0xff),
            UInt8((hitRGBA8 >> 16) & 0xff), UInt8((hitRGBA8 >> 24) & 0xff),
        ]))
        images[imageID] = image
        lastRayCamera = camera
        lastRaySphereLight = sphereLight
        lastRayDistantLight = distantLight
        lastRayDomeLight = domeLight
        fences.insert(fenceID)
    }

    func waitFence(id: UInt64) throws -> Bool {
        guard fences.remove(id) != nil else { throw GPUBackendError.resourceNotFound(id) }
        return true
    }

    func reset() {
        buffers.removeAll()
        images.removeAll()
        fences.removeAll()
        computePipelines.removeAll()
        accelerationStructures.removeAll()
    }
}

private func request(
    _ type: MessageType,
    id: UInt64,
    resourceID: UInt64 = 0,
    payload: Data = Data(),
    magic: UInt32 = IMBProtocol.magic,
    major: UInt16 = IMBProtocol.major,
    minor: UInt16 = IMBProtocol.minor
) -> Frame {
    Frame(
        header: MessageHeader(
            magic: magic,
            versionMajor: major,
            versionMinor: minor,
            messageType: type.rawValue,
            requestID: id,
            resourceID: resourceID
        ),
        payload: payload
    )
}

private func helloPayload(
    minMajor: UInt16 = IMBProtocol.major,
    minMinor: UInt16 = IMBProtocol.minor,
    maxMajor: UInt16 = IMBProtocol.major,
    maxMinor: UInt16 = IMBProtocol.minor
) -> Data {
    var data = Data()
    data.appendLittleEndian(minMajor)
    data.appendLittleEndian(minMinor)
    data.appendLittleEndian(maxMajor)
    data.appendLittleEndian(maxMinor)
    return data
}

private func negotiate(_ session: BridgeSession) {
    let result = session.handle(request(.hello, id: 1, payload: helloPayload()))
    #expect(result.response.header.messageType == MessageType.helloReply.rawValue)
}

private func errorCode(_ frame: Frame) throws -> UInt32 {
    try frame.payload.readLittleEndian(at: 0)
}

@Test func headerEncodingMatchesTheCABI() throws {
    let header = MessageHeader(messageType: MessageType.ping.rawValue, requestID: 0x0102_0304_0506_0708)
    let data = header.encoded()
    #expect(data.count == 32)
    #expect(Array(data.prefix(4)) == [0x49, 0x4d, 0x42, 0x31])
    #expect(try MessageHeader.decode(data) == header)
}

@Test func handshakeCapabilitiesAndPing() throws {
    let session = BridgeSession(metal: testMetal, backend: TestGPUBackend())
    negotiate(session)

    let capabilities = session.handle(request(.queryCapabilities, id: 2)).response
    #expect(capabilities.header.messageType == MessageType.capabilitiesReply.rawValue)
    let bits: UInt64 = try capabilities.payload.readLittleEndian(at: 0)
    let maxBuffer: UInt64 = try capabilities.payload.readLittleEndian(at: 8)
    let nameLength: UInt32 = try capabilities.payload.readLittleEndian(at: 16)
    #expect(bits & Capability.metalAvailable.rawValue != 0)
    #expect(bits & Capability.unifiedMemory.rawValue != 0)
    #expect(bits & Capability.rayTracing.rawValue == 0)
    #expect(bits & Capability.metalBuffer.rawValue != 0)
    #expect(bits & Capability.metalCompute.rawValue != 0)
    #expect(bits & Capability.resourceIO.rawValue != 0)
    #expect(bits & Capability.realFence.rawValue != 0)
    #expect(bits & Capability.metalImage.rawValue != 0)
    #expect(bits & Capability.metalRaster.rawValue != 0)
    #expect(bits & Capability.metalUIRaster.rawValue != 0)
    #expect(bits & Capability.metalSPIRVCompute.rawValue != 0)
    #expect(bits & Capability.metalSparseImage.rawValue != 0)
    #expect(bits & Capability.metalRayDispatch.rawValue == 0)
    #expect(bits & Capability.simulatedFence.rawValue == 0)
    #expect(maxBuffer == 4096)
    #expect(nameLength == UInt32("Test Metal Device".utf8.count))

    let bytes = Data([1, 2, 3, 4])
    let pong = session.handle(request(.ping, id: 3, payload: bytes)).response
    #expect(pong.header.messageType == MessageType.pong.rawValue)
    #expect(pong.payload == bytes)
}

@Test func invalidMagicAndHandshakeRequiredAreRejected() throws {
    let invalid = BridgeSession(metal: testMetal).handle(
        request(.hello, id: 1, payload: helloPayload(), magic: 0)
    ).response
    #expect(invalid.header.messageType == MessageType.error.rawValue)
    #expect(try errorCode(invalid) == ErrorCode.invalidMagic.rawValue)

    let missing = BridgeSession(metal: testMetal).handle(request(.ping, id: 2)).response
    #expect(try errorCode(missing) == ErrorCode.handshakeRequired.rawValue)
}

@Test func versionMismatchIsRejected() throws {
    let session = BridgeSession(metal: testMetal, backend: TestGPUBackend())
    let mismatch = session.handle(
        request(.hello, id: 1, payload: helloPayload(minMajor: 2, maxMajor: 2))
    ).response
    #expect(mismatch.header.messageType == MessageType.error.rawValue)
    #expect(try errorCode(mismatch) == ErrorCode.unsupportedVersion.rawValue)
}

@Test func resourceLifecycleAndInvalidDestroy() throws {
    let session = BridgeSession(metal: testMetal, backend: TestGPUBackend())
    negotiate(session)
    var createPayload = Data()
    createPayload.appendLittleEndian(UInt64(1024))
    createPayload.appendLittleEndian(UInt32(1))
    createPayload.appendLittleEndian(UInt32(0))

    let created = session.handle(request(.createResource, id: 2, payload: createPayload)).response
    let resourceID = created.header.resourceID
    #expect(resourceID != 0)
    #expect(session.resourceCount == 1)

    let destroyed = session.handle(request(.destroyResource, id: 3, resourceID: resourceID)).response
    #expect(destroyed.header.messageType == MessageType.destroyResource.rawValue)
    #expect(session.resourceCount == 0)

    let duplicate = session.handle(request(.destroyResource, id: 4, resourceID: resourceID)).response
    #expect(try errorCode(duplicate) == ErrorCode.resourceNotFound.rawValue)
}

@Test func spirvComputePipelineLifecycleAndValidation() throws {
    let session = BridgeSession(metal: testMetal, backend: TestGPUBackend())
    negotiate(session)

    var create = Data()
    create.appendLittleEndian(UInt32(4))
    create.appendLittleEndian(UInt32(4))
    create.appendLittleEndian(UInt32(0))
    create.appendLittleEndian(UInt32(0))
    create.appendLittleEndian(UInt32(0x0723_0203))
    create.append(Data("main".utf8))
    let created = session.handle(request(.createComputePipeline, id: 2, payload: create)).response
    #expect(created.header.messageType == MessageType.createComputePipeline.rawValue)
    #expect(created.header.resourceID != 0)
    #expect(session.resourceCount == 1)

    let destroyed = session.handle(
        request(.destroyResource, id: 3, resourceID: created.header.resourceID)
    ).response
    #expect(destroyed.header.messageType == MessageType.destroyResource.rawValue)
    #expect(session.resourceCount == 0)

    var invalid = create
    invalid.replaceSubrange(16..<20, with: Data(repeating: 0, count: 4))
    let rejected = session.handle(request(.createComputePipeline, id: 4, payload: invalid)).response
    #expect(rejected.header.messageType == MessageType.error.rawValue)
    #expect(try errorCode(rejected) == ErrorCode.invalidPayload.rawValue)
}

@Test func genericComputeDispatchTracksBindingsAndFenceLifetime() throws {
    let session = BridgeSession(metal: testMetal, backend: TestGPUBackend())
    negotiate(session)

    var bufferCreate = Data()
    bufferCreate.appendLittleEndian(UInt64(16))
    bufferCreate.appendLittleEndian(UInt32(1))
    bufferCreate.appendLittleEndian(UInt32(0))
    let buffer = session.handle(
        request(.createResource, id: 2, payload: bufferCreate)
    ).response

    var pipelineCreate = Data()
    pipelineCreate.appendLittleEndian(UInt32(4))
    pipelineCreate.appendLittleEndian(UInt32(4))
    pipelineCreate.appendLittleEndian(UInt32(0))
    pipelineCreate.appendLittleEndian(UInt32(0))
    pipelineCreate.appendLittleEndian(UInt32(0x0723_0203))
    pipelineCreate.append(Data("main".utf8))
    let pipeline = session.handle(
        request(.createComputePipeline, id: 3, payload: pipelineCreate)
    ).response

    var command = Data()
    command.appendLittleEndian(UInt16(5))
    command.appendLittleEndian(UInt16(0))
    command.appendLittleEndian(UInt32(0))
    command.appendLittleEndian(UInt32(4))
    command.appendLittleEndian(UInt32(3))
    command.appendLittleEndian(UInt32(1))
    command.appendLittleEndian(UInt32(1))
    command.appendLittleEndian(UInt32(4))
    command.appendLittleEndian(UInt32(0))
    command.appendLittleEndian(UInt32(0))
    command.appendLittleEndian(UInt32(0))
    command.appendLittleEndian(UInt32(0))
    command.appendLittleEndian(ComputeBindingKind.bufferReadWrite.rawValue)
    command.appendLittleEndian(UInt32(0))
    command.appendLittleEndian(UInt32(0))
    command.appendLittleEndian(buffer.header.resourceID)
    command.appendLittleEndian(UInt64(0))
    command.appendLittleEndian(UInt64(16))
    command.appendLittleEndian(UInt32(5))
    #expect(command.count == 84)

    let submitted = session.handle(request(
        .submitCommand,
        id: 4,
        resourceID: pipeline.header.resourceID,
        payload: command
    )).response
    #expect(submitted.header.messageType == MessageType.submitCommand.rawValue)

    let busyBuffer = session.handle(
        request(.destroyResource, id: 5, resourceID: buffer.header.resourceID)
    ).response
    #expect(try errorCode(busyBuffer) == ErrorCode.resourceInUse.rawValue)

    let busyPipeline = session.handle(
        request(.destroyResource, id: 6, resourceID: pipeline.header.resourceID)
    ).response
    #expect(try errorCode(busyPipeline) == ErrorCode.resourceInUse.rawValue)

    let waited = session.handle(request(
        .waitFence,
        id: 7,
        resourceID: submitted.header.resourceID
    )).response
    #expect(try waited.payload.readLittleEndian(at: 0) as UInt32 == 1)

    let destroyedBuffer = session.handle(
        request(.destroyResource, id: 8, resourceID: buffer.header.resourceID)
    ).response
    #expect(destroyedBuffer.header.messageType == MessageType.destroyResource.rawValue)

    let destroyedPipeline = session.handle(
        request(.destroyResource, id: 9, resourceID: pipeline.header.resourceID)
    ).response
    #expect(destroyedPipeline.header.messageType == MessageType.destroyResource.rawValue)
}

@Test func primitiveAccelerationStructureLifecycleAndValidation() throws {
    let session = BridgeSession(metal: testMetalWithRayTracing, backend: TestGPUBackend())
    negotiate(session)

    var bufferCreate = Data()
    bufferCreate.appendLittleEndian(UInt64(36))
    bufferCreate.appendLittleEndian(UInt32(1))
    bufferCreate.appendLittleEndian(UInt32(0))
    let buffer = session.handle(request(.createResource, id: 2, payload: bufferCreate)).response
    #expect(buffer.header.resourceID != 0)

    var create = Data()
    create.appendLittleEndian(UInt64(4096))
    create.appendLittleEndian(UInt32(1))
    create.appendLittleEndian(UInt32(0))
    let accelerationStructure = session.handle(
        request(.createAccelerationStructure, id: 3, payload: create)
    ).response
    #expect(accelerationStructure.header.messageType == MessageType.createAccelerationStructure.rawValue)
    #expect(accelerationStructure.header.resourceID != 0)

    var build = Data()
    build.appendLittleEndian(UInt32(1))
    build.appendLittleEndian(UInt32(0x4))
    build.appendLittleEndian(UInt32(0))
    build.appendLittleEndian(UInt32(0))
    build.appendLittleEndian(UInt32(0))
    build.appendLittleEndian(UInt32(1))
    build.appendLittleEndian(buffer.header.resourceID)
    build.appendLittleEndian(UInt64(0))
    build.appendLittleEndian(UInt32(1))
    build.appendLittleEndian(UInt32(12))
    build.appendLittleEndian(UInt64(0))
    build.appendLittleEndian(UInt64(0))
    build.appendLittleEndian(UInt32(0))
    build.appendLittleEndian(UInt32(1))
    build.appendLittleEndian(UInt64(0))
    build.appendLittleEndian(UInt64(0))
    #expect(build.count == 88)
    let built = session.handle(request(
        .buildPrimitiveAccelerationStructure,
        id: 4,
        resourceID: accelerationStructure.header.resourceID,
        payload: build
    )).response
    #expect(built.header.messageType == MessageType.buildPrimitiveAccelerationStructure.rawValue)

    var invalid = build
    invalid.replaceSubrange(16..<20, with: withUnsafeBytes(of: UInt32(3).littleEndian) { Data($0) })
    let rejected = session.handle(request(
        .buildPrimitiveAccelerationStructure,
        id: 5,
        resourceID: accelerationStructure.header.resourceID,
        payload: invalid
    )).response
    #expect(try errorCode(rejected) == ErrorCode.invalidPayload.rawValue)

    create.replaceSubrange(8..<12, with: withUnsafeBytes(of: UInt32(0).littleEndian) { Data($0) })
    let topLevel = session.handle(
        request(.createAccelerationStructure, id: 6, payload: create)
    ).response
    #expect(topLevel.header.messageType == MessageType.createAccelerationStructure.rawValue)

    var instanceBuild = Data()
    instanceBuild.appendLittleEndian(UInt32(1))
    instanceBuild.appendLittleEndian(UInt32(0x4))
    instanceBuild.appendLittleEndian(UInt32(0))
    instanceBuild.appendLittleEndian(UInt32(0))
    for component: Float in [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    ] {
        instanceBuild.appendLittleEndian(component.bitPattern)
    }
    instanceBuild.appendLittleEndian(UInt32(1))
    instanceBuild.appendLittleEndian(UInt32(0xff))
    instanceBuild.appendLittleEndian(UInt32(3))
    instanceBuild.appendLittleEndian(UInt32(42))
    instanceBuild.appendLittleEndian(accelerationStructure.header.resourceID)
    instanceBuild.appendLittleEndian(UInt64(0))
    #expect(instanceBuild.count == 96)
    let instanceBuilt = session.handle(request(
        .buildInstanceAccelerationStructure,
        id: 7,
        resourceID: topLevel.header.resourceID,
        payload: instanceBuild
    )).response
    #expect(instanceBuilt.header.messageType == MessageType.buildInstanceAccelerationStructure.rawValue)

    let topDestroyed = session.handle(request(
        .destroyResource,
        id: 8,
        resourceID: topLevel.header.resourceID
    )).response
    #expect(topDestroyed.header.messageType == MessageType.destroyResource.rawValue)

    let destroyed = session.handle(request(
        .destroyResource,
        id: 9,
        resourceID: accelerationStructure.header.resourceID
    )).response
    #expect(destroyed.header.messageType == MessageType.destroyResource.rawValue)
}

@Test func rayDispatchCommandUsesImageAndTopLevelAccelerationStructure() throws {
    let backend = TestGPUBackend()
    let session = BridgeSession(metal: testMetalWithRayTracing, backend: backend)
    negotiate(session)

    var imageCreate = Data()
    imageCreate.appendLittleEndian(UInt64(16 * 16 * 4))
    imageCreate.appendLittleEndian(UInt32(2))
    imageCreate.appendLittleEndian(UInt32(0))
    imageCreate.appendLittleEndian(UInt32(16))
    imageCreate.appendLittleEndian(UInt32(16))
    imageCreate.appendLittleEndian(UInt32(1))
    imageCreate.appendLittleEndian(UInt32(0))
    let image = session.handle(request(.createResource, id: 2, payload: imageCreate)).response

    var accelerationCreate = Data()
    accelerationCreate.appendLittleEndian(UInt64(4096))
    accelerationCreate.appendLittleEndian(UInt32(0))
    accelerationCreate.appendLittleEndian(UInt32(0))
    let acceleration = session.handle(
        request(.createAccelerationStructure, id: 3, payload: accelerationCreate)
    ).response

    var command = Data()
    command.appendLittleEndian(UInt16(4))
    command.appendLittleEndian(UInt16(0))
    command.appendLittleEndian(UInt32(0))
    command.appendLittleEndian(acceleration.header.resourceID)
    command.appendLittleEndian(UInt32(16))
    command.appendLittleEndian(UInt32(16))
    command.appendLittleEndian(UInt32(0xff00_0000))
    command.appendLittleEndian(UInt32(0xff00_ff00))
    command.appendLittleEndian(UInt32(15))
    command.appendLittleEndian(UInt32(0))
    let expectedCamera = RayCamera(
        position: SIMD3<Float>(7, 8, 9),
        forward: SIMD3<Float>(-1, 0, 0),
        up: SIMD3<Float>(0, 0, 1),
        verticalFOVRadians: 0.9,
        nearDistance: 0.1,
        farDistance: 1_000
    )
    for value in [
        expectedCamera.position.x,
        expectedCamera.position.y,
        expectedCamera.position.z,
        expectedCamera.forward.x,
        expectedCamera.forward.y,
        expectedCamera.forward.z,
        expectedCamera.up.x,
        expectedCamera.up.y,
        expectedCamera.up.z,
        expectedCamera.verticalFOVRadians,
        expectedCamera.nearDistance,
        expectedCamera.farDistance,
    ] {
        command.appendLittleEndian(value.bitPattern)
    }
    let expectedSphereLight = RaySphereLight(
        position: SIMD3<Float>(2, 3, 4),
        color: SIMD3<Float>(1, 0.75, 0.5),
        intensity: 12_500,
        radius: 0.75
    )
    for value in [
        expectedSphereLight.position.x,
        expectedSphereLight.position.y,
        expectedSphereLight.position.z,
        expectedSphereLight.color.x,
        expectedSphereLight.color.y,
        expectedSphereLight.color.z,
        expectedSphereLight.intensity,
        expectedSphereLight.radius,
    ] {
        command.appendLittleEndian(value.bitPattern)
    }
    let expectedDistantLight = RayDistantLight(
        direction: SIMD3<Float>(0.2, -0.4, -0.9),
        color: SIMD3<Float>(1, 0.93, 0.82),
        intensity: 2_500,
        angleDegrees: 1
    )
    for value in [
        expectedDistantLight.direction.x,
        expectedDistantLight.direction.y,
        expectedDistantLight.direction.z,
        expectedDistantLight.color.x,
        expectedDistantLight.color.y,
        expectedDistantLight.color.z,
        expectedDistantLight.intensity,
        expectedDistantLight.angleDegrees,
    ] {
        command.appendLittleEndian(value.bitPattern)
    }
    let expectedDomeLight = RayDomeLight(
        color: SIMD3<Float>(0.28, 0.36, 0.5),
        intensity: 400
    )
    for value in [
        expectedDomeLight.color.x,
        expectedDomeLight.color.y,
        expectedDomeLight.color.z,
        expectedDomeLight.intensity,
    ] {
        command.appendLittleEndian(value.bitPattern)
    }
    let submitted = session.handle(request(
        .submitCommand,
        id: 4,
        resourceID: image.header.resourceID,
        payload: command
    )).response
    #expect(submitted.header.messageType == MessageType.submitCommand.rawValue)
    #expect(backend.lastRayCamera == expectedCamera)
    #expect(backend.lastRaySphereLight == expectedSphereLight)
    #expect(backend.lastRayDistantLight == expectedDistantLight)
    #expect(backend.lastRayDomeLight == expectedDomeLight)

    let busyAcceleration = session.handle(request(
        .destroyResource,
        id: 5,
        resourceID: acceleration.header.resourceID
    )).response
    #expect(try errorCode(busyAcceleration) == ErrorCode.resourceInUse.rawValue)

    let waited = session.handle(request(
        .waitFence,
        id: 6,
        resourceID: submitted.header.resourceID
    )).response
    #expect(try waited.payload.readLittleEndian(at: 0) as UInt32 == 1)
}

@Test func noOpFenceAndInvalidMessagePayload() throws {
    let session = BridgeSession(metal: testMetal, backend: TestGPUBackend())
    negotiate(session)
    var command = Data()
    command.appendLittleEndian(UInt16(0))
    command.appendLittleEndian(UInt16(0))
    command.appendLittleEndian(UInt32(0))
    let submitted = session.handle(request(.submitCommand, id: 2, payload: command)).response
    let fenceID = submitted.header.resourceID
    #expect(fenceID != 0)

    let waited = session.handle(request(.waitFence, id: 3, resourceID: fenceID)).response
    let signaled: UInt32 = try waited.payload.readLittleEndian(at: 0)
    #expect(signaled == 1)

    let invalid = session.handle(request(.queryCapabilities, id: 4, payload: Data([0]))).response
    #expect(try errorCode(invalid) == ErrorCode.invalidPayload.rawValue)
}

@Test func bufferUploadMetalStyleComputeFenceAndReadback() throws {
    let session = BridgeSession(metal: testMetal, backend: TestGPUBackend())
    negotiate(session)

    var create = Data()
    create.appendLittleEndian(UInt64(16))
    create.appendLittleEndian(UInt32(1))
    create.appendLittleEndian(UInt32(0))
    let created = session.handle(request(.createResource, id: 2, payload: create)).response
    let bufferID = created.header.resourceID

    var write = Data()
    write.appendLittleEndian(UInt64(0))
    write.appendLittleEndian(UInt32(16))
    write.appendLittleEndian(UInt32(0))
    for value: UInt32 in [1, 2, 3, 4] { write.appendLittleEndian(value) }
    let written = session.handle(request(.writeResource, id: 3, resourceID: bufferID, payload: write)).response
    #expect(written.header.messageType == MessageType.writeResource.rawValue)

    var command = Data()
    command.appendLittleEndian(UInt16(1))
    command.appendLittleEndian(UInt16(0))
    command.appendLittleEndian(UInt32(0))
    command.appendLittleEndian(UInt32(4))
    command.appendLittleEndian(UInt32(5))
    let submitted = session.handle(request(.submitCommand, id: 4, resourceID: bufferID, payload: command)).response
    let fenceID = submitted.header.resourceID

    let busyDestroy = session.handle(request(.destroyResource, id: 5, resourceID: bufferID)).response
    #expect(try errorCode(busyDestroy) == ErrorCode.resourceInUse.rawValue)

    let waited = session.handle(request(.waitFence, id: 6, resourceID: fenceID)).response
    #expect(try waited.payload.readLittleEndian(at: 0) as UInt32 == 1)

    var read = Data()
    read.appendLittleEndian(UInt64(0))
    read.appendLittleEndian(UInt64(16))
    let result = session.handle(request(.readResource, id: 7, resourceID: bufferID, payload: read)).response
    let values: [UInt32] = try (0..<4).map { try result.payload.readLittleEndian(at: $0 * 4) }
    #expect(values == [6, 7, 8, 9])

    var invalidRead = Data()
    invalidRead.appendLittleEndian(UInt64(12))
    invalidRead.appendLittleEndian(UInt64(8))
    let invalid = session.handle(request(.readResource, id: 8, resourceID: bufferID, payload: invalidRead)).response
    #expect(try errorCode(invalid) == ErrorCode.outOfBounds.rawValue)
}

@Test func imageTriangleFenceAndReadback() throws {
    let session = BridgeSession(metal: testMetal, backend: TestGPUBackend())
    negotiate(session)

    let width: UInt32 = 8
    let height: UInt32 = 8
    var create = Data()
    create.appendLittleEndian(UInt64(width) * UInt64(height) * 4)
    create.appendLittleEndian(UInt32(2))
    create.appendLittleEndian(UInt32(0))
    create.appendLittleEndian(width)
    create.appendLittleEndian(height)
    create.appendLittleEndian(UInt32(1))
    create.appendLittleEndian(UInt32(0))
    let created = session.handle(request(.createResource, id: 2, payload: create)).response
    let imageID = created.header.resourceID
    #expect(imageID != 0)

    let clearRGBA8: UInt32 = 0xff20_1810
    var command = Data()
    command.appendLittleEndian(UInt16(2))
    command.appendLittleEndian(UInt16(0))
    command.appendLittleEndian(UInt32(0))
    command.appendLittleEndian(clearRGBA8)
    command.appendLittleEndian(UInt32(0))
    let submitted = session.handle(
        request(.submitCommand, id: 3, resourceID: imageID, payload: command)
    ).response
    let fenceID = submitted.header.resourceID
    #expect(fenceID != 0)

    let waited = session.handle(request(.waitFence, id: 4, resourceID: fenceID)).response
    #expect(try waited.payload.readLittleEndian(at: 0) as UInt32 == 1)

    var read = Data()
    read.appendLittleEndian(UInt64(0))
    read.appendLittleEndian(UInt64(width) * UInt64(height) * 4)
    let result = session.handle(request(.readResource, id: 5, resourceID: imageID, payload: read)).response
    #expect(result.payload.count == Int(width * height * 4))
    #expect(Array(result.payload.prefix(4)) == [0x10, 0x18, 0x20, 0xff])
    let center = ((Int(height) / 2) * Int(width) + Int(width) / 2) * 4
    #expect(Array(result.payload[center..<(center + 4)]) == [255, 0, 0, 255])
}
