import Foundation

public enum ViewerInputProtocol {
    public static let magic: UInt32 = 0x3149_4d49 // little-endian bytes: I M I 1
    public static let version: UInt16 = 1
    public static let recordSize = 48
}

public enum ViewerInputKind: UInt16, Sendable {
    case mouseMove = 1
    case leftMouseDown = 2
    case leftMouseUp = 3
    case rightMouseDown = 4
    case rightMouseUp = 5
    case middleMouseDown = 6
    case middleMouseUp = 7
    case scroll = 8
    case keyDown = 9
    case keyUp = 10
    case character = 11
}

public struct ViewerInputRecord: Equatable, Sendable {
    public var kind: ViewerInputKind
    public var sequence: UInt64
    public var x: Float
    public var y: Float
    public var deltaX: Float
    public var deltaY: Float
    public var code: UInt32
    public var modifiers: UInt32
    public var targetWidth: UInt32
    public var targetHeight: UInt32

    public init(
        kind: ViewerInputKind,
        sequence: UInt64,
        x: Float = 0,
        y: Float = 0,
        deltaX: Float = 0,
        deltaY: Float = 0,
        code: UInt32 = 0,
        modifiers: UInt32 = 0,
        targetWidth: UInt32 = 0,
        targetHeight: UInt32 = 0
    ) {
        self.kind = kind
        self.sequence = sequence
        self.x = x
        self.y = y
        self.deltaX = deltaX
        self.deltaY = deltaY
        self.code = code
        self.modifiers = modifiers
        self.targetWidth = targetWidth
        self.targetHeight = targetHeight
    }

    public func encoded() -> Data {
        var data = Data()
        data.reserveCapacity(ViewerInputProtocol.recordSize)
        data.appendLittleEndian(ViewerInputProtocol.magic)
        data.appendLittleEndian(ViewerInputProtocol.version)
        data.appendLittleEndian(kind.rawValue)
        data.appendLittleEndian(sequence)
        data.appendLittleEndian(x.bitPattern)
        data.appendLittleEndian(y.bitPattern)
        data.appendLittleEndian(deltaX.bitPattern)
        data.appendLittleEndian(deltaY.bitPattern)
        data.appendLittleEndian(code)
        data.appendLittleEndian(modifiers)
        data.appendLittleEndian(targetWidth)
        data.appendLittleEndian(targetHeight)
        return data
    }

    public static func decode(_ data: Data) throws -> ViewerInputRecord {
        guard data.count == ViewerInputProtocol.recordSize else {
            throw WireError.invalidHeader(
                "expected \(ViewerInputProtocol.recordSize) input bytes, got \(data.count)"
            )
        }
        let magic: UInt32 = try data.readLittleEndian(at: 0)
        let version: UInt16 = try data.readLittleEndian(at: 4)
        let rawKind: UInt16 = try data.readLittleEndian(at: 6)
        guard magic == ViewerInputProtocol.magic,
              version == ViewerInputProtocol.version,
              let kind = ViewerInputKind(rawValue: rawKind)
        else {
            throw WireError.invalidHeader("invalid viewer input record")
        }
        let xBits: UInt32 = try data.readLittleEndian(at: 16)
        let yBits: UInt32 = try data.readLittleEndian(at: 20)
        let deltaXBits: UInt32 = try data.readLittleEndian(at: 24)
        let deltaYBits: UInt32 = try data.readLittleEndian(at: 28)
        return ViewerInputRecord(
            kind: kind,
            sequence: try data.readLittleEndian(at: 8),
            x: Float(bitPattern: xBits),
            y: Float(bitPattern: yBits),
            deltaX: Float(bitPattern: deltaXBits),
            deltaY: Float(bitPattern: deltaYBits),
            code: try data.readLittleEndian(at: 32),
            modifiers: try data.readLittleEndian(at: 36),
            targetWidth: try data.readLittleEndian(at: 40),
            targetHeight: try data.readLittleEndian(at: 44)
        )
    }
}
