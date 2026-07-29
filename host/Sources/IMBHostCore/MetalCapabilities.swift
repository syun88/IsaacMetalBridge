import Foundation
#if canImport(Metal)
import Metal
#endif

public struct MetalCapabilities: Equatable, Sendable {
    public let available: Bool
    public let deviceName: String
    public let unifiedMemory: Bool
    public let rayTracing: Bool
    public let maxBufferLength: UInt64

    public init(
        available: Bool,
        deviceName: String,
        unifiedMemory: Bool,
        rayTracing: Bool,
        maxBufferLength: UInt64
    ) {
        self.available = available
        self.deviceName = deviceName
        self.unifiedMemory = unifiedMemory
        self.rayTracing = rayTracing
        self.maxBufferLength = maxBufferLength
    }

    public static func detect() -> MetalCapabilities {
        #if canImport(Metal)
        guard let device = MTLCreateSystemDefaultDevice() else {
            return MetalCapabilities(
                available: false,
                deviceName: "",
                unifiedMemory: false,
                rayTracing: false,
                maxBufferLength: 0
            )
        }
        return MetalCapabilities(
            available: true,
            deviceName: device.name,
            unifiedMemory: device.hasUnifiedMemory,
            rayTracing: device.supportsRaytracing,
            maxBufferLength: UInt64(device.maxBufferLength)
        )
        #else
        return MetalCapabilities(
            available: false,
            deviceName: "",
            unifiedMemory: false,
            rayTracing: false,
            maxBufferLength: 0
        )
        #endif
    }

    public var capabilityBits: UInt64 {
        var bits: UInt64 = 0
        if available { bits |= Capability.metalAvailable.rawValue }
        if unifiedMemory { bits |= Capability.unifiedMemory.rawValue }
        if rayTracing { bits |= Capability.rayTracing.rawValue }
        return bits
    }
}
