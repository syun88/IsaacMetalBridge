import Foundation

public struct SessionResult: Sendable {
    public let response: Frame
    public let shouldShutdown: Bool

    public init(response: Frame, shouldShutdown: Bool = false) {
        self.response = response
        self.shouldShutdown = shouldShutdown
    }
}

public final class BridgeSession: @unchecked Sendable {
    private struct Resource {
        let size: UInt64
        let kind: UInt32
        let options: UInt32
        let width: UInt32
        let height: UInt32
        let depth: UInt32
        let format: UInt32
        let mipLevels: UInt32
        let arrayLayers: UInt32
    }

    private struct ImageUpload {
        var data: Data
        var nextOffset: UInt64
    }

    private let metal: MetalCapabilities
    private let backend: (any BridgeGPUBackend)?
    private var negotiated = false
    private var resources: [UInt64: Resource] = [:]
    private var imageUploads: [UInt64: ImageUpload] = [:]
    private var fences: Set<UInt64> = []
    private var fenceResources: [UInt64: Set<UInt64>] = [:]
    private var nextResourceID: UInt64 = 1
    private var nextFenceID: UInt64 = 1

    /// Creates a live session backed by the default Metal device when possible.
    public convenience init() {
        self.init(metal: .detect(), backend: MetalGPUBackend.makeDefault())
    }

    /// Dependency-injection initializer used by validation tests.
    public init(metal: MetalCapabilities, backend: (any BridgeGPUBackend)? = nil) {
        self.metal = metal
        self.backend = backend
    }

    deinit {
        backend?.reset()
    }

    public var resourceCount: Int { resources.count }

    private func tightlyPackedImageSize(_ resource: Resource) -> UInt64? {
        guard resource.kind == 2, resource.width > 0, resource.height > 0,
              resource.mipLevels > 0, resource.arrayLayers > 0
        else { return nil }
        let blockWidth: UInt64
        let blockHeight: UInt64
        let blockBytes: UInt64
        switch resource.format {
        case 1, 2, 8, 9, 10:
            (blockWidth, blockHeight, blockBytes) = (1, 1, 4)
        case 3, 6, 7:
            (blockWidth, blockHeight, blockBytes) = (4, 4, 16)
        case 4:
            (blockWidth, blockHeight, blockBytes) = (1, 1, 2)
        case 5:
            (blockWidth, blockHeight, blockBytes) = (1, 1, 8)
        default:
            return nil
        }
        var width = UInt64(resource.width)
        var height = UInt64(resource.height)
        var depth = UInt64(resource.depth)
        var layerBytes: UInt64 = 0
        for _ in 0..<resource.mipLevels {
            let blocksX = (width + blockWidth - 1) / blockWidth
            let blocksY = (height + blockHeight - 1) / blockHeight
            let blocksZ = depth
            let (xyBlockCount, xyOverflow) = blocksX.multipliedReportingOverflow(by: blocksY)
            let (blockCount, blockOverflow) = xyBlockCount.multipliedReportingOverflow(by: blocksZ)
            let (mipBytes, byteOverflow) = blockCount.multipliedReportingOverflow(by: blockBytes)
            let (nextLayerBytes, addOverflow) = layerBytes.addingReportingOverflow(mipBytes)
            guard !xyOverflow, !blockOverflow, !byteOverflow, !addOverflow else { return nil }
            layerBytes = nextLayerBytes
            width = max(1, width / 2)
            height = max(1, height / 2)
            depth = max(1, depth / 2)
        }
        let (totalBytes, overflow) = layerBytes.multipliedReportingOverflow(
            by: UInt64(resource.arrayLayers)
        )
        return overflow ? nil : totalBytes
    }

    public func handle(_ frame: Frame) -> SessionResult {
        let header = frame.header
        guard header.magic == IMBProtocol.magic else {
            return failure(frame, .invalidMagic, "invalid protocol magic")
        }
        guard frame.payload.count <= IMBProtocol.maxPayloadLength,
              header.payloadLength == UInt32(frame.payload.count),
              header.requestID != 0,
              header.flags == 0
        else {
            return failure(frame, .invalidHeader, "invalid length, request ID, or request flags")
        }
        guard let type = MessageType(rawValue: header.messageType) else {
            return failure(frame, .unsupportedMessage, "unknown message type \(header.messageType)")
        }

        if !negotiated {
            guard type == .hello else {
                return failure(frame, .handshakeRequired, "HELLO must be the first request")
            }
            return handleHello(frame)
        }

        guard header.versionMajor == IMBProtocol.major,
              header.versionMinor == IMBProtocol.minor
        else {
            return failure(
                frame,
                .unsupportedVersion,
                "session requires protocol \(IMBProtocol.major).\(IMBProtocol.minor)"
            )
        }

        switch type {
        case .queryCapabilities:
            guard frame.payload.isEmpty else {
                return failure(frame, .invalidPayload, "QUERY_CAPABILITIES has no payload")
            }
            var payload = Data()
            let name = Data(metal.deviceName.utf8)
            payload.appendLittleEndian(capabilityBits)
            payload.appendLittleEndian(metal.maxBufferLength)
            payload.appendLittleEndian(UInt32(name.count))
            payload.appendLittleEndian(UInt32(0))
            payload.append(name)
            return success(frame, type: .capabilitiesReply, payload: payload)

        case .ping:
            return success(frame, type: .pong, payload: frame.payload)

        case .querySparseImageProperties:
            guard frame.header.resourceID == 0,
                  frame.payload.count == 16,
                  let format: UInt32 = try? frame.payload.readLittleEndian(at: 0),
                  let textureType: UInt32 = try? frame.payload.readLittleEndian(at: 4),
                  let sampleCount: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                  let reserved: UInt32 = try? frame.payload.readLittleEndian(at: 12),
                  reserved == 0,
                  let backend,
                  backend.supportsSparseImages
            else {
                return failure(frame, .invalidPayload, "invalid sparse image properties query")
            }
            do {
                let properties = try backend.querySparseImageProperties(
                    format: format,
                    textureType: textureType,
                    sampleCount: sampleCount
                )
                var payload = Data()
                payload.appendLittleEndian(properties.tileWidth)
                payload.appendLittleEndian(properties.tileHeight)
                payload.appendLittleEndian(properties.tileDepth)
                payload.appendLittleEndian(UInt32(0))
                payload.appendLittleEndian(properties.tileSizeBytes)
                return success(frame, type: .querySparseImageProperties, payload: payload)
            } catch {
                return backendFailure(frame, error)
            }

        case .createResource:
            guard frame.payload.count >= 16,
                  let size: UInt64 = try? frame.payload.readLittleEndian(at: 0),
                  let kind: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                  let options: UInt32 = try? frame.payload.readLittleEndian(at: 12),
                  let backend
            else {
                return failure(frame, .invalidPayload, "invalid CREATE_RESOURCE payload")
            }
            let id = nextResourceID
            let resource: Resource
            do {
                switch kind {
                case 1:
                    guard frame.payload.count == 16,
                          size > 0,
                          size <= metal.maxBufferLength,
                          options == 0
                    else {
                        return failure(frame, .invalidPayload, "invalid option-free buffer resource")
                    }
                    try backend.createBuffer(id: id, size: size, options: options)
                    resource = Resource(
                        size: size,
                        kind: kind,
                        options: options,
                        width: 0,
                        height: 0,
                        depth: 1,
                        format: 0,
                        mipLevels: 1,
                        arrayLayers: 1
                    )
                case 2:
                    guard let width: UInt32 = try? frame.payload.readLittleEndian(at: 16),
                          let height: UInt32 = try? frame.payload.readLittleEndian(at: 20),
                          let format: UInt32 = try? frame.payload.readLittleEndian(at: 24)
                    else {
                        return failure(frame, .invalidPayload, "invalid image resource")
                    }
                    var resourceMipLevels: UInt32 = 1
                    var resourceArrayLayers: UInt32 = 1
                    var resourceDepth: UInt32 = 1
                    if options == 0 || ImageOption.decodedDepth(from: options) != nil {
                        let depth = ImageOption.decodedDepth(from: options) ?? 1
                        guard frame.payload.count == 32,
                              let reserved: UInt32 = try? frame.payload.readLittleEndian(at: 28),
                              width > 0,
                              height > 0,
                              depth > 0,
                              width <= 4096,
                              height <= 4096,
                              depth <= 4096,
                              (format == 1 || format == 2 || format == 8 || format == 9),
                              reserved == 0,
                              size == UInt64(width) * UInt64(height) * UInt64(depth) * 4,
                              size <= UInt64(IMBProtocol.maxPayloadLength)
                        else {
                            return failure(
                                frame,
                                .invalidPayload,
                                "invalid ordinary 2D/3D RGBA8/BGRA8 image resource"
                            )
                        }
                        try backend.createImage(
                            id: id,
                            width: width,
                            height: height,
                            format: format,
                            options: options
                        )
                        resourceDepth = depth
                    } else {
                        guard options == ImageOption.sparse,
                              frame.payload.count == 48,
                              let mipLevels: UInt32 = try? frame.payload.readLittleEndian(at: 28),
                              let arrayLayers: UInt32 = try? frame.payload.readLittleEndian(at: 32),
                              let sampleCount: UInt32 = try? frame.payload.readLittleEndian(at: 36),
                              let textureType: UInt32 = try? frame.payload.readLittleEndian(at: 40),
                              let reserved: UInt32 = try? frame.payload.readLittleEndian(at: 44),
                              width > 0,
                              height > 0,
                              width <= 16_384,
                              height <= 16_384,
                              (format >= 1 && format <= 10),
                              mipLevels > 0,
                              mipLevels <= 32,
                              arrayLayers > 0,
                              sampleCount == 1,
                              textureType == 1,
                              reserved == 0,
                              size > 0,
                              size <= metal.maxBufferLength
                        else {
                            return failure(frame, .invalidPayload, "invalid sparse 2D image resource")
                        }
                        try backend.createSparseImage(
                            id: id,
                            virtualSize: size,
                            width: width,
                            height: height,
                            format: format,
                            mipLevels: mipLevels,
                            arrayLayers: arrayLayers,
                            sampleCount: sampleCount,
                            textureType: textureType
                        )
                        resourceMipLevels = mipLevels
                        resourceArrayLayers = arrayLayers
                    }
                    resource = Resource(
                        size: size,
                        kind: kind,
                        options: options,
                        width: width,
                        height: height,
                        depth: resourceDepth,
                        format: format,
                        mipLevels: resourceMipLevels,
                        arrayLayers: resourceArrayLayers
                    )
                default:
                    return failure(frame, .invalidPayload, "unknown resource kind \(kind)")
                }
            } catch {
                return backendFailure(frame, error)
            }
            resources[id] = resource
            nextResourceID += 1
            return success(frame, type: .createResource, resourceID: id)

        case .updateSparseImageMapping:
            guard frame.payload.count == 40,
                  let resource = resources[header.resourceID],
                  resource.kind == 2,
                  resource.options == 1,
                  let mode: UInt32 = try? frame.payload.readLittleEndian(at: 0),
                  let mipLevel: UInt32 = try? frame.payload.readLittleEndian(at: 4),
                  let slice: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                  let reserved: UInt32 = try? frame.payload.readLittleEndian(at: 12),
                  let tileX: UInt32 = try? frame.payload.readLittleEndian(at: 16),
                  let tileY: UInt32 = try? frame.payload.readLittleEndian(at: 20),
                  let tileZ: UInt32 = try? frame.payload.readLittleEndian(at: 24),
                  let tileWidth: UInt32 = try? frame.payload.readLittleEndian(at: 28),
                  let tileHeight: UInt32 = try? frame.payload.readLittleEndian(at: 32),
                  let tileDepth: UInt32 = try? frame.payload.readLittleEndian(at: 36),
                  mode <= 1,
                  reserved == 0,
                  tileWidth > 0,
                  tileHeight > 0,
                  tileDepth > 0,
                  let backend
            else {
                return failure(frame, .invalidPayload, "invalid sparse image mapping")
            }
            do {
                try backend.updateSparseImageMapping(
                    id: header.resourceID,
                    map: mode == 0,
                    mipLevel: mipLevel,
                    slice: slice,
                    tileX: tileX,
                    tileY: tileY,
                    tileZ: tileZ,
                    tileWidth: tileWidth,
                    tileHeight: tileHeight,
                    tileDepth: tileDepth
                )
                return success(
                    frame,
                    type: .updateSparseImageMapping,
                    resourceID: header.resourceID
                )
            } catch {
                return backendFailure(frame, error)
            }

        case .createComputePipeline:
            guard frame.header.resourceID == 0,
                  frame.payload.count >= 21,
                  let spirvLength: UInt32 = try? frame.payload.readLittleEndian(at: 0),
                  let entryPointLength: UInt32 = try? frame.payload.readLittleEndian(at: 4),
                  let flags: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                  let reserved: UInt32 = try? frame.payload.readLittleEndian(at: 12),
                  spirvLength >= 4,
                  spirvLength % 4 == 0,
                  entryPointLength > 0,
                  entryPointLength <= 1024,
                  flags == 0,
                  reserved == 0,
                  UInt64(spirvLength) + UInt64(entryPointLength) + 16 == UInt64(frame.payload.count),
                  let backend,
                  backend.supportsSPIRVCompute
            else {
                return failure(frame, .invalidPayload, "invalid CREATE_COMPUTE_PIPELINE payload or unavailable translator")
            }
            let spirvEnd = 16 + Int(spirvLength)
            let spirv = frame.payload.subdata(in: 16..<spirvEnd)
            let entryPointData = frame.payload.subdata(in: spirvEnd..<frame.payload.count)
            guard let magic: UInt32 = try? spirv.readLittleEndian(at: 0),
                  magic == 0x0723_0203,
                  !entryPointData.contains(0),
                  let entryPoint = String(data: entryPointData, encoding: .utf8),
                  !entryPoint.isEmpty
            else {
                return failure(frame, .invalidPayload, "invalid SPIR-V magic or UTF-8 compute entry point")
            }
            let id = nextResourceID
            let pipelineFlags: UInt32
            do {
                pipelineFlags = try backend.createComputePipeline(
                    id: id,
                    spirv: spirv,
                    entryPoint: entryPoint
                )
            } catch {
                return backendFailure(frame, error)
            }
            resources[id] = Resource(
                size: UInt64(spirvLength),
                kind: 3,
                options: 0,
                width: 0,
                height: 0,
                depth: 1,
                format: 0,
                mipLevels: 1,
                arrayLayers: 1
            )
            nextResourceID += 1
            var responsePayload = Data()
            responsePayload.appendLittleEndian(pipelineFlags)
            return success(
                frame,
                type: .createComputePipeline,
                resourceID: id,
                payload: responsePayload
            )

        case .createAccelerationStructure:
            guard frame.header.resourceID == 0,
                  frame.payload.count == 16,
                  let requestedSize: UInt64 = try? frame.payload.readLittleEndian(at: 0),
                  let accelerationStructureType: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                  let flags: UInt32 = try? frame.payload.readLittleEndian(at: 12),
                  requestedSize > 0,
                  accelerationStructureType <= 2,
                  flags == 0,
                  let backend,
                  backend.supportsAccelerationStructures
            else {
                return failure(frame, .invalidPayload, "invalid CREATE_ACCELERATION_STRUCTURE payload or unavailable Metal RT")
            }
            let id = nextResourceID
            do {
                try backend.createAccelerationStructure(
                    id: id,
                    type: accelerationStructureType,
                    requestedSize: requestedSize
                )
            } catch {
                return backendFailure(frame, error)
            }
            resources[id] = Resource(
                size: requestedSize,
                kind: 4,
                options: 0,
                width: 0,
                height: 0,
                depth: 1,
                format: accelerationStructureType,
                mipLevels: 1,
                arrayLayers: 1
            )
            nextResourceID += 1
            return success(frame, type: .createAccelerationStructure, resourceID: id)

        case .buildPrimitiveAccelerationStructure:
            guard frame.header.resourceID != 0,
                  frame.payload.count >= 88,
                  let target = resources[frame.header.resourceID],
                  target.kind == 4,
                  let geometryCount: UInt32 = try? frame.payload.readLittleEndian(at: 0),
                  let buildFlags: UInt32 = try? frame.payload.readLittleEndian(at: 4),
                  let mode: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                  let reserved: UInt32 = try? frame.payload.readLittleEndian(at: 12),
                  geometryCount > 0,
                  geometryCount <= 65_535,
                  buildFlags & ~UInt32(0x1f) == 0,
                  mode == 0,
                  reserved == 0,
                  frame.payload.count == 16 + Int(geometryCount) * 72,
                  let backend,
                  backend.supportsAccelerationStructures
            else {
                return failure(frame, .invalidPayload, "invalid BUILD_PRIMITIVE_ACCELERATION_STRUCTURE payload")
            }
            var geometries: [PrimitiveAccelerationStructureGeometry] = []
            geometries.reserveCapacity(Int(geometryCount))
            for index in 0..<Int(geometryCount) {
                let offset = 16 + index * 72
                guard let rawKind: UInt32 = try? frame.payload.readLittleEndian(at: offset),
                      let kind = PrimitiveAccelerationStructureGeometryKind(rawValue: rawKind),
                      let flags: UInt32 = try? frame.payload.readLittleEndian(at: offset + 4),
                      let dataResourceID: UInt64 = try? frame.payload.readLittleEndian(at: offset + 8),
                      let dataOffset: UInt64 = try? frame.payload.readLittleEndian(at: offset + 16),
                      let primitiveCount: UInt32 = try? frame.payload.readLittleEndian(at: offset + 24),
                      let stride: UInt32 = try? frame.payload.readLittleEndian(at: offset + 28),
                      let indexResourceID: UInt64 = try? frame.payload.readLittleEndian(at: offset + 32),
                      let indexOffset: UInt64 = try? frame.payload.readLittleEndian(at: offset + 40),
                      let indexType: UInt32 = try? frame.payload.readLittleEndian(at: offset + 48),
                      let vertexFormat: UInt32 = try? frame.payload.readLittleEndian(at: offset + 52),
                      let transformResourceID: UInt64 = try? frame.payload.readLittleEndian(at: offset + 56),
                      let transformOffset: UInt64 = try? frame.payload.readLittleEndian(at: offset + 64),
                      dataResourceID != 0,
                      resources[dataResourceID]?.kind == 1,
                      indexResourceID == 0 || resources[indexResourceID]?.kind == 1,
                      transformResourceID == 0 || resources[transformResourceID]?.kind == 1
                else {
                    return failure(frame, .invalidPayload, "invalid acceleration structure geometry \(index)")
                }
                geometries.append(PrimitiveAccelerationStructureGeometry(
                    kind: kind,
                    flags: flags,
                    dataResourceID: dataResourceID,
                    dataOffset: dataOffset,
                    primitiveCount: primitiveCount,
                    stride: stride,
                    indexResourceID: indexResourceID,
                    indexOffset: indexOffset,
                    indexType: indexType,
                    vertexFormat: vertexFormat,
                    transformResourceID: transformResourceID,
                    transformOffset: transformOffset
                ))
            }
            do {
                try backend.buildPrimitiveAccelerationStructure(
                    id: frame.header.resourceID,
                    buildFlags: buildFlags,
                    geometries: geometries
                )
            } catch {
                return backendFailure(frame, error)
            }
            return success(
                frame,
                type: .buildPrimitiveAccelerationStructure,
                resourceID: frame.header.resourceID
            )

        case .buildInstanceAccelerationStructure:
            guard frame.header.resourceID != 0,
                  frame.payload.count >= 96,
                  let target = resources[frame.header.resourceID],
                  target.kind == 4,
                  target.format == 0 || target.format == 2,
                  let instanceCount: UInt32 = try? frame.payload.readLittleEndian(at: 0),
                  let buildFlags: UInt32 = try? frame.payload.readLittleEndian(at: 4),
                  let mode: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                  let reserved: UInt32 = try? frame.payload.readLittleEndian(at: 12),
                  instanceCount > 0,
                  instanceCount <= 1_048_576,
                  buildFlags & ~UInt32(0x1f) == 0,
                  mode == 0,
                  reserved == 0,
                  UInt64(frame.payload.count) == 16 + UInt64(instanceCount) * 80,
                  let backend,
                  backend.supportsAccelerationStructures
            else {
                return failure(frame, .invalidPayload, "invalid BUILD_INSTANCE_ACCELERATION_STRUCTURE payload")
            }
            var instances: [InstanceAccelerationStructureInstance] = []
            instances.reserveCapacity(Int(instanceCount))
            for index in 0..<Int(instanceCount) {
                let offset = 16 + index * 80
                var transformationMatrix: [Float] = []
                transformationMatrix.reserveCapacity(12)
                for component in 0..<12 {
                    guard let bits: UInt32 = try? frame.payload.readLittleEndian(
                        at: offset + component * 4
                    ) else {
                        return failure(frame, .invalidPayload, "invalid instance transform \(index)")
                    }
                    transformationMatrix.append(Float(bitPattern: bits))
                }
                guard transformationMatrix.allSatisfy(\.isFinite),
                      let options: UInt32 = try? frame.payload.readLittleEndian(at: offset + 48),
                      let mask: UInt32 = try? frame.payload.readLittleEndian(at: offset + 52),
                      let intersectionOffset: UInt32 = try? frame.payload.readLittleEndian(at: offset + 56),
                      let userID: UInt32 = try? frame.payload.readLittleEndian(at: offset + 60),
                      let childID: UInt64 = try? frame.payload.readLittleEndian(at: offset + 64),
                      let instanceReserved: UInt64 = try? frame.payload.readLittleEndian(at: offset + 72),
                      options & ~UInt32(0xf) == 0,
                      mask <= 0xff,
                      intersectionOffset <= 0x00ff_ffff,
                      userID <= 0x00ff_ffff,
                      childID != frame.header.resourceID,
                      let child = resources[childID],
                      child.kind == 4,
                      child.format == 1 || child.format == 2,
                      instanceReserved == 0
                else {
                    return failure(frame, .invalidPayload, "invalid acceleration structure instance \(index)")
                }
                instances.append(InstanceAccelerationStructureInstance(
                    transformationMatrix: transformationMatrix,
                    options: options,
                    mask: mask,
                    intersectionFunctionTableOffset: intersectionOffset,
                    userID: userID,
                    accelerationStructureResourceID: childID
                ))
            }
            do {
                try backend.buildInstanceAccelerationStructure(
                    id: frame.header.resourceID,
                    buildFlags: buildFlags,
                    instances: instances
                )
            } catch {
                return backendFailure(frame, error)
            }
            return success(
                frame,
                type: .buildInstanceAccelerationStructure,
                resourceID: frame.header.resourceID
            )

        case .destroyResource:
            guard frame.payload.isEmpty else {
                return failure(frame, .invalidPayload, "DESTROY_RESOURCE has no payload")
            }
            guard resources[header.resourceID] != nil else {
                return failure(frame, .resourceNotFound, "unknown resource \(header.resourceID)")
            }
            guard !fenceResources.values.contains(where: { $0.contains(header.resourceID) }) else {
                return failure(frame, .resourceInUse, "resource \(header.resourceID) has an unfinished command")
            }
            do {
                if resources[header.resourceID]?.kind == 1 {
                    try backend?.destroyBuffer(id: header.resourceID)
                } else if resources[header.resourceID]?.kind == 2 {
                    try backend?.destroyImage(id: header.resourceID)
                } else if resources[header.resourceID]?.kind == 3 {
                    try backend?.destroyComputePipeline(id: header.resourceID)
                } else {
                    try backend?.destroyAccelerationStructure(id: header.resourceID)
                }
            } catch {
                return backendFailure(frame, error)
            }
            imageUploads.removeValue(forKey: header.resourceID)
            resources.removeValue(forKey: header.resourceID)
            return success(frame, type: .destroyResource, resourceID: header.resourceID)

        case .writeResource:
            guard frame.payload.count >= 16,
                  let resource = resources[header.resourceID],
                  let offset: UInt64 = try? frame.payload.readLittleEndian(at: 0),
                  let length: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                  let reserved: UInt32 = try? frame.payload.readLittleEndian(at: 12),
                  reserved == 0,
                  Int(length) == frame.payload.count - 16,
                  let backend,
                  resource.kind == 1 || resource.kind == 2
            else {
                return failure(frame, .invalidPayload, "invalid WRITE_RESOURCE payload or resource")
            }
            do {
                let data = frame.payload.subdata(in: 16..<frame.payload.count)
                if resource.kind == 1 {
                    try backend.writeBuffer(id: header.resourceID, offset: offset, data: data)
                } else {
                    guard let tightSize = tightlyPackedImageSize(resource),
                          tightSize <= UInt64(Int.max),
                          tightSize <= resource.size,
                          offset <= tightSize,
                          UInt64(length) <= tightSize - offset
                    else {
                        return failure(frame, .outOfBounds, "image upload exceeds the tightly packed mip data")
                    }
                    var upload: ImageUpload
                    if offset == 0 {
                        upload = ImageUpload(
                            data: Data(count: Int(tightSize)),
                            nextOffset: 0
                        )
                    } else if let pending = imageUploads[header.resourceID] {
                        upload = pending
                    } else {
                        return failure(frame, .outOfBounds, "image upload must begin at offset zero")
                    }
                    guard upload.nextOffset == offset else {
                        return failure(frame, .outOfBounds, "image upload chunks must be contiguous")
                    }
                    let endOffset = offset + UInt64(length)
                    upload.data.replaceSubrange(
                        Int(offset)..<Int(endOffset),
                        with: data
                    )
                    upload.nextOffset = endOffset
                    if endOffset == tightSize {
                        try backend.writeImage(id: header.resourceID, data: upload.data)
                        imageUploads.removeValue(forKey: header.resourceID)
                    } else {
                        imageUploads[header.resourceID] = upload
                    }
                }
            } catch {
                return backendFailure(frame, error)
            }
            return success(frame, type: .writeResource, resourceID: header.resourceID)

        case .readResource:
            guard frame.payload.count == 16,
                  let resource = resources[header.resourceID],
                  let offset: UInt64 = try? frame.payload.readLittleEndian(at: 0),
                  let length: UInt64 = try? frame.payload.readLittleEndian(at: 8),
                  length <= UInt64(IMBProtocol.maxPayloadLength),
                  let backend,
                  resource.kind == 1 || resource.kind == 2
            else {
                return failure(frame, .invalidPayload, "invalid READ_RESOURCE payload or resource")
            }
            do {
                let payload: Data
                if resource.kind == 1 {
                    payload = try backend.readBuffer(id: header.resourceID, offset: offset, length: length)
                } else {
                    guard offset == 0, length == resource.size else {
                        return failure(frame, .outOfBounds, "image readback requires the complete tightly packed image")
                    }
                    payload = try backend.readImage(id: header.resourceID)
                }
                return success(frame, type: .readResource, resourceID: header.resourceID, payload: payload)
            } catch {
                return backendFailure(frame, error)
            }

        case .submitCommand:
            return handleSubmit(frame)

        case .waitFence:
            guard frame.payload.isEmpty else {
                return failure(frame, .invalidPayload, "WAIT_FENCE has no payload")
            }
            guard fences.contains(header.resourceID), let backend else {
                return failure(frame, .resourceNotFound, "unknown fence \(header.resourceID)")
            }
            let signaled: Bool
            do {
                signaled = try backend.waitFence(id: header.resourceID)
            } catch {
                return backendFailure(frame, error)
            }
            fences.remove(header.resourceID)
            fenceResources.removeValue(forKey: header.resourceID)
            var payload = Data()
            payload.appendLittleEndian(UInt32(signaled ? 1 : 0))
            payload.appendLittleEndian(UInt32(0))
            return success(frame, type: .waitFence, resourceID: header.resourceID, payload: payload)

        case .shutdown:
            guard frame.payload.isEmpty else {
                return failure(frame, .invalidPayload, "SHUTDOWN has no payload")
            }
            resources.removeAll()
            fences.removeAll()
            fenceResources.removeAll()
            backend?.reset()
            return SessionResult(
                response: responseFrame(frame, type: .shutdown),
                shouldShutdown: true
            )

        case .hello, .helloReply, .capabilitiesReply, .pong, .error:
            return failure(frame, .unsupportedMessage, "message \(type) is not a valid request in this state")
        }
    }

    private func handleHello(_ frame: Frame) -> SessionResult {
        guard frame.payload.count == 8,
              let minMajor: UInt16 = try? frame.payload.readLittleEndian(at: 0),
              let minMinor: UInt16 = try? frame.payload.readLittleEndian(at: 2),
              let maxMajor: UInt16 = try? frame.payload.readLittleEndian(at: 4),
              let maxMinor: UInt16 = try? frame.payload.readLittleEndian(at: 6)
        else {
            return failure(frame, .invalidPayload, "HELLO payload must be 8 bytes")
        }

        let supported = (minMajor < IMBProtocol.major || (minMajor == IMBProtocol.major && minMinor <= IMBProtocol.minor))
            && (maxMajor > IMBProtocol.major || (maxMajor == IMBProtocol.major && maxMinor >= IMBProtocol.minor))
        guard supported else {
            return failure(
                frame,
                .unsupportedVersion,
                "no compatible version; host supports \(IMBProtocol.major).\(IMBProtocol.minor)"
            )
        }

        negotiated = true
        var payload = Data()
        payload.appendLittleEndian(IMBProtocol.major)
        payload.appendLittleEndian(IMBProtocol.minor)
        payload.appendLittleEndian(UInt32(0))
        return success(frame, type: .helloReply, payload: payload)
    }

    private func handleSubmit(_ frame: Frame) -> SessionResult {
        guard frame.payload.count >= 8,
              let command: UInt16 = try? frame.payload.readLittleEndian(at: 0),
              let reserved16: UInt16 = try? frame.payload.readLittleEndian(at: 2),
              let reserved32: UInt32 = try? frame.payload.readLittleEndian(at: 4),
              reserved16 == 0,
              reserved32 == 0
        else {
            return failure(frame, .invalidPayload, "invalid SUBMIT_COMMAND payload")
        }
        guard let backend else {
            return failure(frame, .backendUnavailable, "Metal command backend is unavailable")
        }

        let fenceID = nextFenceID
        var commandResources: Set<UInt64> = []
        do {
            switch command {
            case 0:
                guard frame.payload.count == 8, frame.header.resourceID == 0 else {
                    return failure(frame, .invalidPayload, "NOOP requires an 8-byte payload and no resource")
                }
                try backend.submitNoop(fenceID: fenceID)
            case 1:
                guard frame.payload.count == 16,
                      let resource = resources[frame.header.resourceID],
                      resource.kind == 1,
                      let elementCount: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                      let addend: UInt32 = try? frame.payload.readLittleEndian(at: 12)
                else {
                    return failure(frame, .invalidPayload, "ADD_U32 requires a Metal buffer and a 16-byte payload")
                }
                try backend.submitAddUInt32(
                    bufferID: frame.header.resourceID,
                    elementCount: elementCount,
                    addend: addend,
                    fenceID: fenceID
                )
                commandResources.insert(frame.header.resourceID)
            case 2:
                guard frame.payload.count == 16,
                      let resource = resources[frame.header.resourceID],
                      resource.kind == 2,
                      let clearRGBA8: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                      let reservedColor: UInt32 = try? frame.payload.readLittleEndian(at: 12),
                      reservedColor == 0
                else {
                    return failure(frame, .invalidPayload, "DRAW_TRIANGLE requires an RGBA8 image and a 16-byte payload")
                }
                try backend.submitTriangle(
                    imageID: frame.header.resourceID,
                    clearRGBA8: clearRGBA8,
                    fenceID: fenceID
                )
                commandResources.insert(frame.header.resourceID)
            case 3:
                guard frame.payload.count >= 56,
                      let target = resources[frame.header.resourceID],
                      target.kind == 2,
                      target.format == 2,
                      let vertexBufferID: UInt64 = try? frame.payload.readLittleEndian(at: 8),
                      let indexBufferID: UInt64 = try? frame.payload.readLittleEndian(at: 16),
                      let vertexBufferOffset: UInt64 = try? frame.payload.readLittleEndian(at: 24),
                      let indexBufferOffset: UInt64 = try? frame.payload.readLittleEndian(at: 32),
                      let width: UInt32 = try? frame.payload.readLittleEndian(at: 40),
                      let height: UInt32 = try? frame.payload.readLittleEndian(at: 44),
                      let clearRGBA8: UInt32 = try? frame.payload.readLittleEndian(at: 48),
                      let drawCount: UInt32 = try? frame.payload.readLittleEndian(at: 52),
                      drawCount > 0,
                      drawCount <= 100_000,
                      frame.payload.count == 56 + Int(drawCount) * 40,
                      let vertexBuffer = resources[vertexBufferID], vertexBuffer.kind == 1,
                      let indexBuffer = resources[indexBufferID], indexBuffer.kind == 1,
                      width == target.width,
                      height == target.height
                else {
                    return failure(frame, .invalidPayload, "DRAW_INDEXED_UI requires BGRA8 target, buffers, and valid draw records")
                }
                var draws: [IndexedUIDraw] = []
                draws.reserveCapacity(Int(drawCount))
                for index in 0..<Int(drawCount) {
                    let offset = 56 + index * 40
                    guard let textureID: UInt64 = try? frame.payload.readLittleEndian(at: offset),
                          let indexCount: UInt32 = try? frame.payload.readLittleEndian(at: offset + 8),
                          let firstIndex: UInt32 = try? frame.payload.readLittleEndian(at: offset + 12),
                          let rawVertexOffset: UInt32 = try? frame.payload.readLittleEndian(at: offset + 16),
                          let scissorX: UInt32 = try? frame.payload.readLittleEndian(at: offset + 20),
                          let scissorY: UInt32 = try? frame.payload.readLittleEndian(at: offset + 24),
                          let scissorWidth: UInt32 = try? frame.payload.readLittleEndian(at: offset + 28),
                          let scissorHeight: UInt32 = try? frame.payload.readLittleEndian(at: offset + 32),
                          let reserved: UInt32 = try? frame.payload.readLittleEndian(at: offset + 36),
                          reserved == 0,
                          textureID == 0 || resources[textureID]?.kind == 2
                    else {
                        return failure(frame, .invalidPayload, "invalid UI draw record \(index)")
                    }
                    draws.append(IndexedUIDraw(
                        textureID: textureID,
                        indexCount: indexCount,
                        firstIndex: firstIndex,
                        vertexOffset: Int32(bitPattern: rawVertexOffset),
                        scissorX: scissorX,
                        scissorY: scissorY,
                        scissorWidth: scissorWidth,
                        scissorHeight: scissorHeight
                    ))
                }
                try backend.submitIndexedUI(
                    imageID: frame.header.resourceID,
                    vertexBufferID: vertexBufferID,
                    indexBufferID: indexBufferID,
                    vertexBufferOffset: vertexBufferOffset,
                    indexBufferOffset: indexBufferOffset,
                    width: width,
                    height: height,
                    clearRGBA8: clearRGBA8,
                    draws: draws,
                    fenceID: fenceID
                )
                commandResources.formUnion([frame.header.resourceID, vertexBufferID, indexBufferID])
                commandResources.formUnion(draws.lazy.map(\.textureID).filter { $0 != 0 })
            case 4:
                guard frame.payload.count == 168,
                      let target = resources[frame.header.resourceID],
                      target.kind == 2,
                      let accelerationStructureID: UInt64 = try? frame.payload.readLittleEndian(at: 8),
                      let width: UInt32 = try? frame.payload.readLittleEndian(at: 16),
                      let height: UInt32 = try? frame.payload.readLittleEndian(at: 20),
                      let missRGBA8: UInt32 = try? frame.payload.readLittleEndian(at: 24),
                      let hitRGBA8: UInt32 = try? frame.payload.readLittleEndian(at: 28),
                      let options: UInt32 = try? frame.payload.readLittleEndian(at: 32),
                      let reserved1: UInt32 = try? frame.payload.readLittleEndian(at: 36),
                      width == target.width,
                      height == target.height,
                      options & ~UInt32(31) == 0,
                      reserved1 == 0,
                      backend.supportsRayDispatch
                else {
                    return failure(
                        frame,
                        .invalidPayload,
                        "TRACE_RAYS requires a matching image and valid options"
                    )
                }
                let emptyStageGrid = options & 16 != 0
                let accelerationStructure = resources[accelerationStructureID]
                guard emptyStageGrid
                    ? accelerationStructureID == 0 && options & UInt32(14) == 0
                    : accelerationStructure?.kind == 4 && accelerationStructure?.format == 0
                else {
                    return failure(
                        frame,
                        .invalidPayload,
                        "TRACE_RAYS requires a built top-level Metal acceleration structure unless EMPTY_STAGE_GRID is set"
                    )
                }
                var camera: RayCamera?
                if options & 1 != 0 {
                    let values: [Float] = (0..<12).compactMap { index in
                        guard let bits: UInt32 = try? frame.payload.readLittleEndian(
                            at: 40 + index * 4
                        ) else {
                            return nil
                        }
                        return Float(bitPattern: bits)
                    }
                    guard values.count == 12,
                          values.allSatisfy(\.isFinite),
                          values[9] > 0.01,
                          values[9] < 3.13,
                          values[10] > 0,
                          values[11] > values[10],
                          values[3] * values[3] + values[4] * values[4]
                            + values[5] * values[5] > 0.000_001,
                          values[6] * values[6] + values[7] * values[7]
                            + values[8] * values[8] > 0.000_001
                    else {
                        return failure(frame, .invalidPayload, "TRACE_RAYS live camera is invalid")
                    }
                    camera = RayCamera(
                        position: SIMD3<Float>(values[0], values[1], values[2]),
                        forward: SIMD3<Float>(values[3], values[4], values[5]),
                        up: SIMD3<Float>(values[6], values[7], values[8]),
                        verticalFOVRadians: values[9],
                        nearDistance: values[10],
                        farDistance: values[11]
                    )
                }
                var sphereLight: RaySphereLight?
                if options & 2 != 0 {
                    let values: [Float] = (0..<8).compactMap { index in
                        guard let bits: UInt32 = try? frame.payload.readLittleEndian(
                            at: 88 + index * 4
                        ) else {
                            return nil
                        }
                        return Float(bitPattern: bits)
                    }
                    guard values.count == 8,
                          values.allSatisfy(\.isFinite),
                          values[3] >= 0,
                          values[4] >= 0,
                          values[5] >= 0,
                          values[6] >= 0,
                          values[7] > 0
                    else {
                        return failure(frame, .invalidPayload, "TRACE_RAYS live SphereLight is invalid")
                    }
                    sphereLight = RaySphereLight(
                        position: SIMD3<Float>(values[0], values[1], values[2]),
                        color: SIMD3<Float>(values[3], values[4], values[5]),
                        intensity: values[6],
                        radius: values[7]
                    )
                }
                var distantLight: RayDistantLight?
                if options & 4 != 0 {
                    let values: [Float] = (0..<8).compactMap { index in
                        guard let bits: UInt32 = try? frame.payload.readLittleEndian(
                            at: 120 + index * 4
                        ) else {
                            return nil
                        }
                        return Float(bitPattern: bits)
                    }
                    guard values.count == 8,
                          values.allSatisfy(\.isFinite),
                          values[0] * values[0] + values[1] * values[1]
                            + values[2] * values[2] > 0.000_001,
                          values[3] >= 0,
                          values[4] >= 0,
                          values[5] >= 0,
                          values[6] >= 0,
                          values[7] >= 0,
                          values[7] <= 180
                    else {
                        return failure(frame, .invalidPayload, "TRACE_RAYS live DistantLight is invalid")
                    }
                    distantLight = RayDistantLight(
                        direction: SIMD3<Float>(values[0], values[1], values[2]),
                        color: SIMD3<Float>(values[3], values[4], values[5]),
                        intensity: values[6],
                        angleDegrees: values[7]
                    )
                }
                var domeLight: RayDomeLight?
                if options & 8 != 0 {
                    let values: [Float] = (0..<4).compactMap { index in
                        guard let bits: UInt32 = try? frame.payload.readLittleEndian(
                            at: 152 + index * 4
                        ) else {
                            return nil
                        }
                        return Float(bitPattern: bits)
                    }
                    guard values.count == 4,
                          values.allSatisfy(\.isFinite),
                          values.allSatisfy({ $0 >= 0 })
                    else {
                        return failure(frame, .invalidPayload, "TRACE_RAYS live DomeLight is invalid")
                    }
                    domeLight = RayDomeLight(
                        color: SIMD3<Float>(values[0], values[1], values[2]),
                        intensity: values[3]
                    )
                }
                if emptyStageGrid {
                    try backend.submitEmptyStageGrid(
                        imageID: frame.header.resourceID,
                        width: width,
                        height: height,
                        camera: camera,
                        fenceID: fenceID
                    )
                    commandResources.insert(frame.header.resourceID)
                } else {
                    try backend.submitRayTrace(
                        imageID: frame.header.resourceID,
                        accelerationStructureID: accelerationStructureID,
                        width: width,
                        height: height,
                        missRGBA8: missRGBA8,
                        hitRGBA8: hitRGBA8,
                        camera: camera,
                        sphereLight: sphereLight,
                        distantLight: distantLight,
                        domeLight: domeLight,
                        fenceID: fenceID
                    )
                    commandResources.formUnion([
                        frame.header.resourceID,
                        accelerationStructureID,
                    ])
                }
            case 5:
                guard frame.payload.count >= 32,
                      let pipeline = resources[frame.header.resourceID],
                      pipeline.kind == 3,
                      let groupCountX: UInt32 = try? frame.payload.readLittleEndian(at: 8),
                      let groupCountY: UInt32 = try? frame.payload.readLittleEndian(at: 12),
                      let groupCountZ: UInt32 = try? frame.payload.readLittleEndian(at: 16),
                      let bindingCount: UInt32 = try? frame.payload.readLittleEndian(at: 20),
                      let pushConstantLength: UInt32 = try? frame.payload.readLittleEndian(at: 24),
                      let reserved1: UInt32 = try? frame.payload.readLittleEndian(at: 28),
                      groupCountX > 0, groupCountY > 0, groupCountZ > 0,
                      bindingCount <= 100_000,
                      pushConstantLength <= 4_096,
                      reserved1 == 0,
                      frame.payload.count
                          == 32 + Int(bindingCount) * 48 + Int(pushConstantLength)
                else {
                    return failure(
                        frame,
                        .invalidPayload,
                        "DISPATCH_COMPUTE requires a Metal pipeline and valid workgroups"
                    )
                }
                var bindings: [ComputeBinding] = []
                bindings.reserveCapacity(Int(bindingCount))
                var occupiedBindings: Set<String> = []
                for index in 0..<Int(bindingCount) {
                    let offset = 32 + index * 48
                    guard let descriptorSet: UInt32 = try? frame.payload.readLittleEndian(at: offset),
                          let binding: UInt32 = try? frame.payload.readLittleEndian(at: offset + 4),
                          let arrayElement: UInt32 = try? frame.payload.readLittleEndian(at: offset + 8),
                          let rawKind: UInt32 = try? frame.payload.readLittleEndian(at: offset + 12),
                          let kind = ComputeBindingKind(rawValue: rawKind),
                          let format: UInt32 = try? frame.payload.readLittleEndian(at: offset + 16),
                          let reserved: UInt32 = try? frame.payload.readLittleEndian(at: offset + 20),
                          let resourceID: UInt64 = try? frame.payload.readLittleEndian(at: offset + 24),
                          let resourceOffset: UInt64 = try? frame.payload.readLittleEndian(at: offset + 32),
                          let length: UInt64 = try? frame.payload.readLittleEndian(at: offset + 40),
                          reserved == 0,
                          descriptorSet < 32
                    else {
                        return failure(frame, .invalidPayload, "invalid compute binding \(index)")
                    }
                    let key = "\(descriptorSet):\(binding):\(arrayElement)"
                    guard occupiedBindings.insert(key).inserted else {
                        return failure(frame, .invalidPayload, "duplicate compute binding \(key)")
                    }
                    switch kind {
                    case .bufferRead, .bufferReadWrite,
                         .texelBufferRead, .texelBufferReadWrite:
                        guard let resource = resources[resourceID] else {
                            return failure(frame, .invalidPayload, "invalid compute buffer resource \(index)")
                        }
                        let isTexel = kind == .texelBufferRead
                            || kind == .texelBufferReadWrite
                        guard resource.kind == 1, length > 0,
                              resourceOffset <= resource.size,
                              length <= resource.size - resourceOffset,
                              isTexel ? format != 0 : format == 0
                        else {
                            return failure(frame, .outOfBounds, "invalid compute buffer binding \(index)")
                        }
                    case .textureRead, .textureReadWrite:
                        guard let resource = resources[resourceID] else {
                            return failure(frame, .invalidPayload, "invalid compute texture resource \(index)")
                        }
                        guard resource.kind == 2, resourceOffset == 0,
                              length == 0, format == 0
                        else {
                            return failure(frame, .invalidPayload, "invalid compute texture binding \(index)")
                        }
                    case .sampler:
                        let minLod = Float(bitPattern: UInt32(truncatingIfNeeded: resourceOffset))
                        let maxLod = Float(bitPattern: UInt32(truncatingIfNeeded: resourceOffset >> 32))
                        let mipLodBias = Float(bitPattern: UInt32(truncatingIfNeeded: length))
                        let maxAnisotropy = Float(bitPattern: UInt32(truncatingIfNeeded: length >> 32))
                        guard resourceID == 0,
                              format & ~UInt32(0x001f_ffff) == 0,
                              minLod.isFinite, maxLod.isFinite,
                              mipLodBias.isFinite, maxAnisotropy.isFinite,
                              minLod <= maxLod,
                              maxAnisotropy >= 1, maxAnisotropy <= 16,
                              ((format >> 3) & 0x7) <= 4,
                              ((format >> 6) & 0x7) <= 4,
                              ((format >> 9) & 0x7) <= 4,
                              ((format >> 14) & 0x7) <= 7,
                              ((format >> 18) & 0x7) <= 5
                        else {
                            return failure(frame, .invalidPayload, "invalid compute sampler binding \(index)")
                        }
                    }
                    bindings.append(ComputeBinding(
                        descriptorSet: descriptorSet,
                        binding: binding,
                        arrayElement: arrayElement,
                        kind: kind,
                        format: format,
                        resourceID: resourceID,
                        offset: resourceOffset,
                        length: length
                    ))
                    if resourceID != 0 {
                        commandResources.insert(resourceID)
                    }
                }
                try backend.submitCompute(
                    pipelineID: frame.header.resourceID,
                    groupCountX: groupCountX,
                    groupCountY: groupCountY,
                    groupCountZ: groupCountZ,
                    bindings: bindings,
                    pushConstants: frame.payload.subdata(
                        in: (32 + Int(bindingCount) * 48)..<frame.payload.count
                    ),
                    fenceID: fenceID
                )
                commandResources.insert(frame.header.resourceID)
            default:
                return failure(frame, .unsupportedCommand, "unknown command \(command)")
            }
        } catch {
            return backendFailure(frame, error)
        }

        fences.insert(fenceID)
        if !commandResources.isEmpty {
            fenceResources[fenceID] = commandResources
        }
        nextFenceID += 1
        return success(frame, type: .submitCommand, resourceID: fenceID)
    }

    private func success(
        _ request: Frame,
        type: MessageType,
        resourceID: UInt64 = 0,
        payload: Data = Data()
    ) -> SessionResult {
        SessionResult(response: responseFrame(request, type: type, resourceID: resourceID, payload: payload))
    }

    private func responseFrame(
        _ request: Frame,
        type: MessageType,
        resourceID: UInt64 = 0,
        payload: Data = Data()
    ) -> Frame {
        Frame(
            header: MessageHeader(
                messageType: type.rawValue,
                flags: IMBProtocol.responseFlag,
                requestID: request.header.requestID,
                resourceID: resourceID
            ),
            payload: payload
        )
    }

    private func failure(_ request: Frame, _ code: ErrorCode, _ message: String) -> SessionResult {
        let messageData = Data(message.utf8)
        var payload = Data()
        payload.appendLittleEndian(code.rawValue)
        payload.appendLittleEndian(UInt32(messageData.count))
        payload.append(messageData)
        return SessionResult(
            response: responseFrame(request, type: .error, resourceID: request.header.resourceID, payload: payload)
        )
    }

    private var capabilityBits: UInt64 {
        var bits = metal.capabilityBits
        guard backend != nil else {
            return bits
        }
        bits |= Capability.noopCommand.rawValue
            | Capability.metalBuffer.rawValue
            | Capability.metalCompute.rawValue
            | Capability.resourceIO.rawValue
            | Capability.realFence.rawValue
            | Capability.metalImage.rawValue
            | Capability.metalRaster.rawValue
            | Capability.metalUIRaster.rawValue
        if backend?.supportsSPIRVCompute == true {
            bits |= Capability.metalSPIRVCompute.rawValue
        }
        if metal.rayTracing, backend?.supportsAccelerationStructures == true {
            bits |= Capability.metalAccelerationStructure.rawValue
        }
        if metal.rayTracing, backend?.supportsRayDispatch == true {
            bits |= Capability.metalRayDispatch.rawValue
        }
        if backend?.supportsSparseImages == true {
            bits |= Capability.metalSparseImage.rawValue
        }
        return bits
    }

    private func backendFailure(_ request: Frame, _ error: Error) -> SessionResult {
        let code: ErrorCode
        if let backendError = error as? GPUBackendError {
            switch backendError {
            case .unavailable:
                code = .backendUnavailable
            case .resourceNotFound:
                code = .resourceNotFound
            case .outOfBounds:
                code = .outOfBounds
            case .unsupported:
                code = .unsupportedCommand
            case .commandFailed:
                code = .gpuFailure
            }
        } else {
            code = .internal
        }
        return failure(request, code, String(describing: error))
    }
}
