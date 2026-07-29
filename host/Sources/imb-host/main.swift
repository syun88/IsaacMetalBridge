import Foundation
import IMBHostCore

func log(_ message: String) {
    FileHandle.standardError.write(Data("imb-host: \(message)\n".utf8))
}

let metal = MetalCapabilities.detect()
log("Metal available=\(metal.available) device=\(metal.deviceName.isEmpty ? "none" : metal.deviceName) unifiedMemory=\(metal.unifiedMemory) rayTracing=\(metal.rayTracing) maxBufferLength=\(metal.maxBufferLength)")

let session = BridgeSession()

do {
    try BridgeStreamServer.serve(
        input: .standardInput,
        output: .standardOutput,
        session: session,
        log: log
    )
} catch {
    log("fatal protocol I/O error: \(error)")
    exit(1)
}
