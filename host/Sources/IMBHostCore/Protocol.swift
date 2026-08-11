import Foundation

public enum IMBProtocol {
    public static let magic: UInt32 = 0x3142_4d49
    public static let major: UInt16 = 1
    public static let minor: UInt16 = 22
    public static let headerSize = 32
    public static let maxPayloadLength = 16 * 1024 * 1024
    public static let responseFlag: UInt16 = 1
}

public enum MessageType: UInt16, Sendable {
    case hello = 1
    case helloReply = 2
    case queryCapabilities = 3
    case capabilitiesReply = 4
    case ping = 5
    case pong = 6
    case createResource = 7
    case destroyResource = 8
    case submitCommand = 9
    case waitFence = 10
    case error = 11
    case shutdown = 12
    case writeResource = 13
    case readResource = 14
    case createComputePipeline = 15
    case createAccelerationStructure = 16
    case buildPrimitiveAccelerationStructure = 17
    case buildInstanceAccelerationStructure = 18
    case querySparseImageProperties = 19
    case updateSparseImageMapping = 20
}

public enum ErrorCode: UInt32, Sendable {
    case none = 0
    case invalidMagic = 1
    case invalidHeader = 2
    case unsupportedVersion = 3
    case unsupportedMessage = 4
    case invalidPayload = 5
    case handshakeRequired = 6
    case resourceNotFound = 7
    case resourceInUse = 8
    case unsupportedCommand = 9
    case `internal` = 10
    case backendUnavailable = 11
    case outOfBounds = 12
    case gpuFailure = 13
}

public enum ComputePipelineFlag {
    public static let softwareFP64ExecutionRequired: UInt32 = 1 << 0
    public static let serializedAtomic64ExecutionRequired: UInt32 = 1 << 1
}

public enum ImageOption {
    public static let sparse: UInt32 = 1 << 0
    public static let texture3D: UInt32 = 1 << 1
    public static let depthShift: UInt32 = 16
    public static let depthMask: UInt32 = 0xffff << depthShift
    public static let texture3DMask: UInt32 = texture3D | depthMask

    public static func encodedTexture3D(depth: UInt32) -> UInt32? {
        guard depth > 0, depth <= 0xffff else { return nil }
        return texture3D | (depth << depthShift)
    }

    public static func decodedDepth(from options: UInt32) -> UInt32? {
        guard options & ~texture3DMask == 0,
              options & texture3D != 0
        else { return nil }
        let depth = (options & depthMask) >> depthShift
        return depth > 0 ? depth : nil
    }
}

public enum Capability: UInt64, Sendable {
    case metalAvailable = 1
    case unifiedMemory = 2
    case rayTracing = 4
    case noopCommand = 8
    case simulatedFence = 16
    case metalBuffer = 32
    case metalCompute = 64
    case resourceIO = 128
    case realFence = 256
    case metalImage = 512
    case metalRaster = 1024
    case metalUIRaster = 2048
    case metalSPIRVCompute = 4096
    case metalAccelerationStructure = 8192
    case metalRayDispatch = 16384
    case metalSparseImage = 32768
}

public struct MessageHeader: Equatable, Sendable {
    public var magic: UInt32
    public var versionMajor: UInt16
    public var versionMinor: UInt16
    public var messageType: UInt16
    public var flags: UInt16
    public var payloadLength: UInt32
    public var requestID: UInt64
    public var resourceID: UInt64

    public init(
        magic: UInt32 = IMBProtocol.magic,
        versionMajor: UInt16 = IMBProtocol.major,
        versionMinor: UInt16 = IMBProtocol.minor,
        messageType: UInt16,
        flags: UInt16 = 0,
        payloadLength: UInt32 = 0,
        requestID: UInt64,
        resourceID: UInt64 = 0
    ) {
        self.magic = magic
        self.versionMajor = versionMajor
        self.versionMinor = versionMinor
        self.messageType = messageType
        self.flags = flags
        self.payloadLength = payloadLength
        self.requestID = requestID
        self.resourceID = resourceID
    }

    public func encoded() -> Data {
        var data = Data()
        data.appendLittleEndian(magic)
        data.appendLittleEndian(versionMajor)
        data.appendLittleEndian(versionMinor)
        data.appendLittleEndian(messageType)
        data.appendLittleEndian(flags)
        data.appendLittleEndian(payloadLength)
        data.appendLittleEndian(requestID)
        data.appendLittleEndian(resourceID)
        return data
    }

    public static func decode(_ data: Data) throws -> MessageHeader {
        guard data.count == IMBProtocol.headerSize else {
            throw WireError.invalidHeader("expected 32 header bytes, got \(data.count)")
        }
        return MessageHeader(
            magic: try data.readLittleEndian(at: 0),
            versionMajor: try data.readLittleEndian(at: 4),
            versionMinor: try data.readLittleEndian(at: 6),
            messageType: try data.readLittleEndian(at: 8),
            flags: try data.readLittleEndian(at: 10),
            payloadLength: try data.readLittleEndian(at: 12),
            requestID: try data.readLittleEndian(at: 16),
            resourceID: try data.readLittleEndian(at: 24)
        )
    }
}

public struct Frame: Equatable, Sendable {
    public var header: MessageHeader
    public var payload: Data

    public init(header: MessageHeader, payload: Data = Data()) {
        self.header = header
        self.payload = payload
        self.header.payloadLength = UInt32(payload.count)
    }

    public func encoded() -> Data {
        var data = header.encoded()
        data.append(payload)
        return data
    }
}

public enum WireError: Error, Equatable, CustomStringConvertible {
    case invalidHeader(String)
    case truncatedPayload

    public var description: String {
        switch self {
        case .invalidHeader(let message): message
        case .truncatedPayload: "truncated payload"
        }
    }
}

extension Data {
    public mutating func appendLittleEndian<T: FixedWidthInteger>(_ value: T) {
        var value = value.littleEndian
        Swift.withUnsafeBytes(of: &value) { append(contentsOf: $0) }
    }

    public func readLittleEndian<T: FixedWidthInteger>(at offset: Int) throws -> T {
        let width = MemoryLayout<T>.size
        guard offset >= 0, offset + width <= count else {
            throw WireError.invalidHeader("integer at offset \(offset) exceeds data length \(count)")
        }
        var result: T = 0
        for index in 0..<width {
            result |= T(self[offset + index]) << (index * 8)
        }
        return result
    }
}
