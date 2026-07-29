import Foundation

public enum BridgeStreamError: Error, CustomStringConvertible {
    case unexpectedEOF
    case oversizedPayload(UInt32)

    public var description: String {
        switch self {
        case .unexpectedEOF:
            "unexpected end of protocol stream"
        case .oversizedPayload(let size):
            "payload length \(size) exceeds protocol maximum"
        }
    }
}

/// Serves one IMB session over a pair of byte-stream file handles.
///
/// The handles may be stdin/stdout pipes or the single full-duplex file handle
/// returned by Apple container's `ContainerClient.dial(id:port:)` API.
public enum BridgeStreamServer {
    public static func serve(
        input: FileHandle,
        output: FileHandle,
        session: BridgeSession,
        log: (String) -> Void = { _ in }
    ) throws {
        while let headerData = try readExactly(
            input,
            count: IMBProtocol.headerSize,
            allowCleanEOF: true
        ) {
            let header = try MessageHeader.decode(headerData)
            guard header.payloadLength <= IMBProtocol.maxPayloadLength else {
                throw BridgeStreamError.oversizedPayload(header.payloadLength)
            }
            let payload = try readExactly(input, count: Int(header.payloadLength)) ?? Data()
            let result = session.handle(Frame(header: header, payload: payload))
            try output.write(contentsOf: result.response.encoded())
            if result.shouldShutdown {
                log("session shutdown")
                return
            }
        }
    }

    private static func readExactly(
        _ handle: FileHandle,
        count: Int,
        allowCleanEOF: Bool = false
    ) throws -> Data? {
        var data = Data()
        while data.count < count {
            guard let chunk = try handle.read(upToCount: count - data.count), !chunk.isEmpty else {
                if allowCleanEOF && data.isEmpty {
                    return nil
                }
                throw BridgeStreamError.unexpectedEOF
            }
            data.append(chunk)
        }
        return data
    }
}
