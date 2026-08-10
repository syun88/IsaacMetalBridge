import Foundation
import IMBHostCore
import Metal

func firstLine(_ value: String) -> String {
    value.split(whereSeparator: { $0.isNewline }).first.map(String.init) ?? value
}

func scalarData<T>(_ value: T) -> Data {
    var copy = value
    return withUnsafeBytes(of: &copy) { Data($0) }
}

func floatArrayData(_ values: [Float]) -> Data {
    values.withUnsafeBytes { Data($0) }
}

func packedFloatPair(_ low: Float, _ high: Float) -> UInt64 {
    UInt64(low.bitPattern) | (UInt64(high.bitPattern) << 32)
}

/// Executes the smallest captured Isaac RTX compute shader which performs a
/// real binary64-storage read on the Apple GPU. The fixture selects only the
/// world-transform output path, dispatches one 32x32 workgroup (one invocation
/// is inside the 1x1 bounds), and verifies that the decoded position reaches
/// the output matrix. This exercises the actual captured SPIR-V, argument
/// buffer remapping, software-FP64 lowering, and Metal dispatch together.
func executeIsaacFP64TransformFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> SIMD3<Float> {
    let expectedHash = "f5dd5704d7491f17"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the FP64 dispatch fixture requires (expectedHash).spv"
        )
    }

    let backend = try MetalGPUBackend(device: device, spirvCompiler: translator)
    let pipelineID: UInt64 = 1
    let flags = try backend.createComputePipeline(
        id: pipelineID,
        spirv: Data(contentsOf: shader),
        entryPoint: "main"
    )
    guard flags & ComputePipelineFlag.softwareFP64ExecutionRequired != 0 else {
        throw GPUBackendError.commandFailed(
            "captured transform shader was not classified as software-FP64 execution"
        )
    }

    struct Resource {
        let id: UInt64
        let size: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let kind: ComputeBindingKind
    }
    let resources: [Resource] = [
        // Set 2: compacted IDs are intentionally in a different order from
        // these sparse Vulkan bindings.
        Resource(id: 10, size: 4_096, descriptorSet: 2, binding: 4, kind: .bufferRead),
        Resource(id: 11, size: 4_096, descriptorSet: 2, binding: 2, kind: .bufferRead),
        Resource(id: 12, size: 4_096, descriptorSet: 2, binding: 3, kind: .bufferRead),
        Resource(id: 13, size: 32, descriptorSet: 2, binding: 10, kind: .bufferRead),
        Resource(id: 14, size: 64, descriptorSet: 2, binding: 6, kind: .bufferRead),
        Resource(id: 15, size: 80, descriptorSet: 2, binding: 7, kind: .bufferRead),
        Resource(id: 16, size: 704, descriptorSet: 2, binding: 0, kind: .bufferRead),
        Resource(id: 17, size: 4, descriptorSet: 2, binding: 5, kind: .bufferRead),
        Resource(id: 18, size: 4_036, descriptorSet: 3, binding: 0, kind: .bufferRead),
        Resource(id: 19, size: 48, descriptorSet: 5, binding: 0, kind: .bufferRead),
        Resource(id: 20, size: 256, descriptorSet: 5, binding: 1, kind: .bufferRead),
        Resource(id: 21, size: 16, descriptorSet: 6, binding: 0, kind: .bufferRead),
        Resource(id: 22, size: 4, descriptorSet: 6, binding: 6, kind: .bufferReadWrite),
        Resource(id: 23, size: 24, descriptorSet: 6, binding: 1, kind: .bufferReadWrite),
        Resource(id: 24, size: 24, descriptorSet: 6, binding: 2, kind: .bufferReadWrite),
        Resource(id: 25, size: 96, descriptorSet: 6, binding: 3, kind: .bufferReadWrite),
        Resource(id: 26, size: 12, descriptorSet: 6, binding: 4, kind: .bufferReadWrite),
    ]
    for resource in resources {
        try backend.createBuffer(id: resource.id, size: resource.size, options: 0)
    }

    // One logical invocation in bounds.
    try backend.writeBuffer(id: 21, offset: 0, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 21, offset: 4, data: scalarData(UInt32(1)))
    // Direct index path plus the transform-output bit, avoiding unrelated
    // output branches: 0x00100000 | 0x02000000.
    try backend.writeBuffer(id: 19, offset: 0, data: scalarData(UInt64(0x0210_0000)))
    // Prevent a divide-by-zero in the float transform scale.
    try backend.writeBuffer(id: 18, offset: 2_912, data: scalarData(Float(1)))

    let input = SIMD3<Double>(1.5, -2.25, 3.75)
    try backend.writeBuffer(id: 15, offset: 48, data: scalarData(input.x.bitPattern))
    try backend.writeBuffer(id: 15, offset: 56, data: scalarData(input.y.bitPattern))
    try backend.writeBuffer(id: 15, offset: 64, data: scalarData(input.z.bitPattern))

    let bindings = resources.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.kind,
            resourceID: $0.id,
            offset: 0,
            length: $0.size
        )
    }
    try backend.submitCompute(
        pipelineID: pipelineID,
        groupCountX: 1,
        groupCountY: 1,
        groupCountZ: 1,
        bindings: bindings,
        pushConstants: Data(),
        fenceID: 100
    )
    guard try backend.waitFence(id: 100) else {
        throw GPUBackendError.commandFailed("FP64 fixture fence was not signaled")
    }

    let counter = try backend.readBuffer(id: 22, offset: 0, length: 4).withUnsafeBytes {
        $0.loadUnaligned(as: UInt32.self)
    }
    guard counter == 1 else {
        throw GPUBackendError.commandFailed(
            "FP64 fixture executed (counter) in-bounds invocations instead of 1"
        )
    }
    let outputData = try backend.readBuffer(id: 25, offset: 80, length: 12)
    let output = outputData.withUnsafeBytes { bytes in
        SIMD3<Float>(
            bytes.loadUnaligned(fromByteOffset: 0, as: Float.self),
            bytes.loadUnaligned(fromByteOffset: 4, as: Float.self),
            bytes.loadUnaligned(fromByteOffset: 8, as: Float.self)
        )
    }
    let expected = SIMD3<Float>(Float(input.x), Float(input.y), Float(input.z))
    guard output == expected else {
        throw GPUBackendError.commandFailed(
            "FP64 fixture output (output) did not match (expected)"
        )
    }
    return output
}

/// Executes a captured RTX camera/depth reconstruction shader which requires
/// two separate Vulkan samplers. The FP64 transform moves the test point onto
/// a clipping plane exactly; that selects the zero-distance branch and writes
/// a deterministic low-intensity white pixel. The same plane would not be hit
/// without the binary64 scale/offset operation.
func executeIsaacFP64SamplerFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> [UInt8] {
    let expectedHash = "64d94f5901ee0e4f"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the sampler FP64 dispatch fixture requires \(expectedHash).spv"
        )
    }

    let backend = try MetalGPUBackend(device: device, spirvCompiler: translator)
    let pipelineID: UInt64 = 1
    let flags = try backend.createComputePipeline(
        id: pipelineID,
        spirv: Data(contentsOf: shader),
        entryPoint: "main"
    )
    guard flags & ComputePipelineFlag.softwareFP64ExecutionRequired != 0 else {
        throw GPUBackendError.commandFailed(
            "captured sampler shader was not classified as software-FP64 execution"
        )
    }

    struct BufferResource {
        let id: UInt64
        let size: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let kind: ComputeBindingKind
        let format: UInt32
    }
    let buffers = [
        BufferResource(id: 10, size: 4_036, descriptorSet: 0, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 11, size: 256, descriptorSet: 0, binding: 11, kind: .texelBufferRead, format: 109),
        BufferResource(id: 12, size: 256, descriptorSet: 0, binding: 12, kind: .texelBufferRead, format: 109),
        BufferResource(id: 13, size: 256, descriptorSet: 0, binding: 13, kind: .texelBufferRead, format: 109),
        BufferResource(id: 14, size: 48, descriptorSet: 2, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 15, size: 16, descriptorSet: 3, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 16, size: 16, descriptorSet: 4, binding: 0, kind: .bufferRead, format: 0),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    // One output invocation and finite three-level input filtering.
    try backend.writeBuffer(id: 16, offset: 0, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 16, offset: 8, data: scalarData(Float(128)))
    try backend.writeBuffer(id: 15, offset: 0, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 15, offset: 8, data: scalarData(SIMD2<UInt32>(8, 8)))

    // Select the mode-8 ray branch and the single camera record.
    try backend.writeBuffer(id: 10, offset: 2_816, data: scalarData(Int32(8)))
    try backend.writeBuffer(id: 10, offset: 2_792, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 2_912, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 10, offset: 2_916, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 2_920, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_272, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_276, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_296, data: scalarData(Float.pi * 2))
    try backend.writeBuffer(id: 10, offset: 3_360, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 10, offset: 3_648, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 3_736, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_740, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_744, data: scalarData(UInt32(1)))

    // Identity camera-to-world transform. The shader's row-major load is
    // transposed, and identity is invariant under that conversion.
    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    try backend.writeBuffer(id: 10, offset: 672, data: floatArrayData(identity))
    try backend.writeBuffer(id: 10, offset: 2_888, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 10, offset: 2_892, data: scalarData(Float(10)))

    // FP64 operation: z = 1 / 2 + 1 = 1.5 exactly. Plane z=1.5 makes
    // the resulting clip distance exactly zero.
    try backend.writeBuffer(id: 10, offset: 2_848, data: scalarData(Double(1).bitPattern))
    try backend.writeBuffer(
        id: 10,
        offset: 3_008,
        data: scalarData(SIMD4<Float>(0, 0, -1, 1.5))
    )

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let arrayElement: UInt32
        let writable: Bool
        let pixel: [UInt8]
    }
    let images = [
        ImageResource(id: 20, descriptorSet: 0, binding: 8, arrayElement: 0, writable: false, pixel: [0, 0, 255, 255]),
        ImageResource(id: 21, descriptorSet: 4, binding: 1, arrayElement: 0, writable: false, pixel: [255, 0, 0, 255]),
        ImageResource(id: 22, descriptorSet: 4, binding: 1, arrayElement: 1, writable: false, pixel: [0, 0, 0, 0]),
        ImageResource(id: 23, descriptorSet: 4, binding: 1, arrayElement: 2, writable: false, pixel: [0, 0, 0, 0]),
        ImageResource(id: 24, descriptorSet: 4, binding: 1, arrayElement: 3, writable: false, pixel: [0, 0, 0, 0]),
        ImageResource(id: 25, descriptorSet: 3, binding: 1, arrayElement: 0, writable: false, pixel: [255, 0, 0, 255]),
        ImageResource(id: 26, descriptorSet: 4, binding: 2, arrayElement: 0, writable: false, pixel: [255, 255, 255, 255]),
        ImageResource(id: 27, descriptorSet: 4, binding: 3, arrayElement: 0, writable: true, pixel: [0, 0, 0, 0]),
    ]
    for image in images {
        try backend.createImage(id: image.id, width: 1, height: 1, format: 1, options: 0)
        try backend.writeImage(id: image.id, data: Data(image.pixel))
    }

    var bindings = buffers.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.kind,
            format: $0.format,
            resourceID: $0.id,
            offset: 0,
            length: $0.size
        )
    }
    bindings.append(contentsOf: images.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: $0.arrayElement,
            kind: $0.writable ? .textureReadWrite : .textureRead,
            resourceID: $0.id,
            offset: 0,
            length: 0
        )
    })
    let samplerOptions: UInt32 = 1 | 2 | 4 | (1 << 3) | (1 << 6) | (1 << 9)
    for (descriptorSet, binding) in [(UInt32(3), UInt32(2)), (UInt32(7), UInt32(7))] {
        bindings.append(ComputeBinding(
            descriptorSet: descriptorSet,
            binding: binding,
            arrayElement: 0,
            kind: .sampler,
            format: samplerOptions,
            resourceID: 0,
            offset: packedFloatPair(0, 16),
            length: packedFloatPair(0, 1)
        ))
    }

    try backend.submitCompute(
        pipelineID: pipelineID,
        groupCountX: 1,
        groupCountY: 1,
        groupCountZ: 1,
        bindings: bindings,
        pushConstants: Data(),
        fenceID: 100
    )
    guard try backend.waitFence(id: 100) else {
        throw GPUBackendError.commandFailed("sampler FP64 fixture fence was not signaled")
    }
    let pixel = [UInt8](try backend.readImage(id: 27))
    guard pixel.count == 4, pixel[0] == 255,
          (18...20).contains(Int(pixel[1])),
          pixel[1] == pixel[2], pixel[3] == 255
    else {
        throw GPUBackendError.commandFailed(
            "sampler FP64 fixture output \(pixel) did not match the expected clipping-plane pixel"
        )
    }
    return pixel
}

/// Executes a captured RTX depth-to-world-position shader. Its final storage
/// image is the software-FP64 scale/offset result itself, so the checked RGB
/// value distinguishes the lowered operation from the untransformed ray.
func executeIsaacFP64PositionFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> [UInt8] {
    let expectedHash = "75321ea922defdfc"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the position FP64 dispatch fixture requires \(expectedHash).spv"
        )
    }

    let backend = try MetalGPUBackend(device: device, spirvCompiler: translator)
    let pipelineID: UInt64 = 1
    let flags = try backend.createComputePipeline(
        id: pipelineID,
        spirv: Data(contentsOf: shader),
        entryPoint: "main"
    )
    guard flags & ComputePipelineFlag.softwareFP64ExecutionRequired != 0 else {
        throw GPUBackendError.commandFailed(
            "captured position shader was not classified as software-FP64 execution"
        )
    }

    struct BufferResource {
        let id: UInt64
        let size: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let kind: ComputeBindingKind
        let format: UInt32
    }
    let buffers = [
        BufferResource(id: 10, size: 1_200, descriptorSet: 0, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 11, size: 4_036, descriptorSet: 1, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 12, size: 256, descriptorSet: 1, binding: 11, kind: .texelBufferRead, format: 109),
        BufferResource(id: 13, size: 256, descriptorSet: 1, binding: 12, kind: .texelBufferRead, format: 109),
        BufferResource(id: 14, size: 256, descriptorSet: 1, binding: 13, kind: .texelBufferRead, format: 109),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    // One invocation and a 0.1 output multiplier (0.001 * shader's 100).
    try backend.writeBuffer(id: 10, offset: 0, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 160, data: scalarData(SIMD3<Float>(0.001, 0, 0)))

    // Mode 7 reconstructs a panoramic -Z camera ray. One camera record keeps the large
    // texel-buffer fallbacks inactive while still binding their real types.
    try backend.writeBuffer(id: 11, offset: 2_792, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 2_816, data: scalarData(Int32(7)))
    try backend.writeBuffer(id: 11, offset: 2_888, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 11, offset: 2_892, data: scalarData(Float(10)))
    try backend.writeBuffer(id: 11, offset: 2_912, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 11, offset: 2_916, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 11, offset: 2_920, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 11, offset: 3_368, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 11, offset: 3_648, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 3_744, data: scalarData(UInt32(1)))

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    try backend.writeBuffer(id: 11, offset: 672, data: floatArrayData(identity))

    // FP64 operation: (1,0,-1) / 2 + (1,2,3) = (1.5,2,2.5).
    // The 0.1 multiplier above makes the RGBA8 result approximately
    // [38.25, 51, 63.75, 255].
    for (offset, value) in [(2_832, 1.0), (2_840, 2.0), (2_848, 3.0)] {
        try backend.writeBuffer(
            id: 11,
            offset: UInt64(offset),
            data: scalarData(Double(value).bitPattern)
        )
    }

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let writable: Bool
        let pixel: [UInt8]
    }
    let images = [
        ImageResource(id: 20, descriptorSet: 0, binding: 26, writable: false, pixel: [255, 0, 0, 255]),
        ImageResource(id: 21, descriptorSet: 1, binding: 8, writable: false, pixel: [0, 0, 255, 255]),
        ImageResource(id: 22, descriptorSet: 0, binding: 1, writable: true, pixel: [0, 0, 0, 0]),
    ]
    for image in images {
        try backend.createImage(id: image.id, width: 1, height: 1, format: 1, options: 0)
        try backend.writeImage(id: image.id, data: Data(image.pixel))
    }

    var bindings = buffers.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.kind,
            format: $0.format,
            resourceID: $0.id,
            offset: 0,
            length: $0.size
        )
    }
    bindings.append(contentsOf: images.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.writable ? .textureReadWrite : .textureRead,
            resourceID: $0.id,
            offset: 0,
            length: 0
        )
    })
    let samplerOptions: UInt32 = 1 | 2 | 4 | (1 << 3) | (1 << 6) | (1 << 9)
    bindings.append(ComputeBinding(
        descriptorSet: 2,
        binding: 7,
        arrayElement: 0,
        kind: .sampler,
        format: samplerOptions,
        resourceID: 0,
        offset: packedFloatPair(0, 16),
        length: packedFloatPair(0, 1)
    ))

    try backend.submitCompute(
        pipelineID: pipelineID,
        groupCountX: 1,
        groupCountY: 1,
        groupCountZ: 1,
        bindings: bindings,
        pushConstants: Data(),
        fenceID: 100
    )
    guard try backend.waitFence(id: 100) else {
        throw GPUBackendError.commandFailed("position FP64 fixture fence was not signaled")
    }
    let pixel = [UInt8](try backend.readImage(id: 22))
    guard pixel.count == 4,
          (37...39).contains(Int(pixel[0])),
          (50...52).contains(Int(pixel[1])),
          (63...65).contains(Int(pixel[2])),
          pixel[3] == 255
    else {
        throw GPUBackendError.commandFailed(
            "position FP64 fixture output \(pixel) did not match the transformed world position"
        )
    }
    return pixel
}

/// Executes the captured buffer-output variant of the RTX panoramic
/// depth-to-world-position shader. This has a materially different descriptor
/// ABI from the image-output fixture above: five storage buffers receive the
/// reconstructed attributes and the first one receives the software-FP64
/// transformed world position.
func executeIsaacFP64StructuredPositionFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> SIMD3<Float> {
    let expectedHash = "361fde9aceebb12b"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the structured position FP64 dispatch fixture requires \(expectedHash).spv"
        )
    }

    let backend = try MetalGPUBackend(device: device, spirvCompiler: translator)
    let pipelineID: UInt64 = 1
    let flags = try backend.createComputePipeline(
        id: pipelineID,
        spirv: Data(contentsOf: shader),
        entryPoint: "main"
    )
    guard flags & ComputePipelineFlag.softwareFP64ExecutionRequired != 0 else {
        throw GPUBackendError.commandFailed(
            "captured structured position shader was not classified as software-FP64 execution"
        )
    }

    struct BufferResource {
        let id: UInt64
        let size: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let kind: ComputeBindingKind
        let format: UInt32
    }
    let buffers = [
        BufferResource(id: 10, size: 4_036, descriptorSet: 0, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 11, size: 256, descriptorSet: 0, binding: 11, kind: .texelBufferRead, format: 109),
        BufferResource(id: 12, size: 256, descriptorSet: 0, binding: 12, kind: .texelBufferRead, format: 109),
        BufferResource(id: 13, size: 256, descriptorSet: 0, binding: 13, kind: .texelBufferRead, format: 109),
        BufferResource(id: 30, size: 12, descriptorSet: 2, binding: 1, kind: .bufferReadWrite, format: 0),
        BufferResource(id: 31, size: 12, descriptorSet: 2, binding: 2, kind: .bufferReadWrite, format: 0),
        BufferResource(id: 32, size: 4, descriptorSet: 2, binding: 3, kind: .bufferReadWrite, format: 0),
        BufferResource(id: 33, size: 16, descriptorSet: 2, binding: 4, kind: .bufferReadWrite, format: 0),
        BufferResource(id: 34, size: 4, descriptorSet: 2, binding: 5, kind: .bufferReadWrite, format: 0),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    // One camera record and one output pixel. Mode 7 reconstructs a panoramic
    // ray through the center of a two-radian-wide projection.
    try backend.writeBuffer(id: 10, offset: 2_792, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 2_816, data: scalarData(Int32(7)))
    try backend.writeBuffer(id: 10, offset: 2_888, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 10, offset: 2_892, data: scalarData(Float(10)))
    try backend.writeBuffer(id: 10, offset: 2_912, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 10, offset: 2_916, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 2_920, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_368, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 10, offset: 3_648, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 3_736, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_740, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_744, data: scalarData(UInt32(1)))

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    try backend.writeBuffer(id: 10, offset: 672, data: floatArrayData(identity))

    // FP64 operation: (1,0,-1) / 2 + (1,2,3) = (1.5,2,2.5).
    for (offset, value) in [(2_832, 1.0), (2_840, 2.0), (2_848, 3.0)] {
        try backend.writeBuffer(
            id: 10,
            offset: UInt64(offset),
            data: scalarData(Double(value).bitPattern)
        )
    }

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let writable: Bool
        let pixel: [UInt8]
    }
    let images = [
        ImageResource(id: 20, descriptorSet: 0, binding: 8, writable: false, pixel: [0, 0, 255, 255]),
        ImageResource(id: 21, descriptorSet: 1, binding: 1, writable: false, pixel: [255, 0, 0, 255]),
        ImageResource(id: 22, descriptorSet: 1, binding: 3, writable: false, pixel: [0, 0, 0, 0]),
        ImageResource(id: 23, descriptorSet: 1, binding: 4, writable: false, pixel: [0, 0, 0, 0]),
        ImageResource(id: 24, descriptorSet: 1, binding: 8, writable: false, pixel: [0, 0, 0, 0]),
        ImageResource(id: 25, descriptorSet: 2, binding: 0, writable: true, pixel: [0, 0, 0, 255]),
    ]
    for image in images {
        try backend.createImage(id: image.id, width: 1, height: 1, format: 1, options: 0)
        try backend.writeImage(id: image.id, data: Data(image.pixel))
    }

    var bindings = buffers.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.kind,
            format: $0.format,
            resourceID: $0.id,
            offset: 0,
            length: $0.size
        )
    }
    bindings.append(contentsOf: images.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.writable ? .textureReadWrite : .textureRead,
            resourceID: $0.id,
            offset: 0,
            length: 0
        )
    })
    let samplerOptions: UInt32 = 1 | 2 | 4 | (1 << 3) | (1 << 6) | (1 << 9)
    bindings.append(ComputeBinding(
        descriptorSet: 3,
        binding: 7,
        arrayElement: 0,
        kind: .sampler,
        format: samplerOptions,
        resourceID: 0,
        offset: packedFloatPair(0, 16),
        length: packedFloatPair(0, 1)
    ))

    try backend.submitCompute(
        pipelineID: pipelineID,
        groupCountX: 1,
        groupCountY: 1,
        groupCountZ: 1,
        bindings: bindings,
        pushConstants: Data(),
        fenceID: 100
    )
    guard try backend.waitFence(id: 100) else {
        throw GPUBackendError.commandFailed("structured position FP64 fixture fence was not signaled")
    }
    let outputData = try backend.readBuffer(id: 30, offset: 0, length: 12)
    let output = outputData.withUnsafeBytes { bytes in
        SIMD3<Float>(
            bytes.loadUnaligned(fromByteOffset: 0, as: Float.self),
            bytes.loadUnaligned(fromByteOffset: 4, as: Float.self),
            bytes.loadUnaligned(fromByteOffset: 8, as: Float.self)
        )
    }
    let expected = SIMD3<Float>(1.5, 2, 2.5)
    guard abs(output.x - expected.x) < 0.0001,
          abs(output.y - expected.y) < 0.0001,
          abs(output.z - expected.z) < 0.0001
    else {
        throw GPUBackendError.commandFailed(
            "structured position FP64 fixture output \(output) did not match \(expected)"
        )
    }
    return output
}

/// Executes a captured RTX motion-vector shader that reconstructs two depth
/// samples, applies separate software-FP64 world origins, and writes their
/// difference. Identical rays isolate the checked result to the two FP64
/// origin transforms.
func executeIsaacFP64MotionFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> [UInt8] {
    let expectedHash = "2eb5c9558c6d3e5a"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the motion FP64 dispatch fixture requires \(expectedHash).spv"
        )
    }

    let backend = try MetalGPUBackend(device: device, spirvCompiler: translator)
    let pipelineID: UInt64 = 1
    let flags = try backend.createComputePipeline(
        id: pipelineID,
        spirv: Data(contentsOf: shader),
        entryPoint: "main"
    )
    guard flags & ComputePipelineFlag.softwareFP64ExecutionRequired != 0 else {
        throw GPUBackendError.commandFailed(
            "captured motion shader was not classified as software-FP64 execution"
        )
    }

    struct BufferResource {
        let id: UInt64
        let size: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let kind: ComputeBindingKind
        let format: UInt32
    }
    let buffers = [
        BufferResource(id: 10, size: 1_200, descriptorSet: 0, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 11, size: 4_036, descriptorSet: 1, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 12, size: 256, descriptorSet: 1, binding: 11, kind: .texelBufferRead, format: 109),
        BufferResource(id: 13, size: 256, descriptorSet: 1, binding: 12, kind: .texelBufferRead, format: 109),
        BufferResource(id: 14, size: 256, descriptorSet: 1, binding: 13, kind: .texelBufferRead, format: 109),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    // A 2x2 extent keeps the second reconstruction's half-pixel coordinate
    // from clamping to zero. The 1x1 camera-record grid below still lets only
    // the first invocation reach the output write.
    try backend.writeBuffer(id: 10, offset: 0, data: scalarData(SIMD2<UInt32>(2, 2)))
    try backend.writeBuffer(id: 11, offset: 2_792, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 2_816, data: scalarData(Int32(7)))
    try backend.writeBuffer(id: 11, offset: 2_888, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 11, offset: 2_892, data: scalarData(Float(10)))
    try backend.writeBuffer(id: 11, offset: 2_912, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 11, offset: 2_916, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 11, offset: 2_920, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 11, offset: 3_368, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 11, offset: 3_648, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 3_744, data: scalarData(UInt32(1)))

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    try backend.writeBuffer(id: 11, offset: 672, data: floatArrayData(identity))
    try backend.writeBuffer(id: 11, offset: 800, data: floatArrayData(identity))

    // The two reconstructed positions are identical. Their FP64 origins are
    // (0.25,0.5,0.75) and zero, so subtraction writes that exact RGB value.
    for (offset, value) in [(2_832, 0.25), (2_840, 0.5), (2_848, 0.75)] {
        try backend.writeBuffer(
            id: 11,
            offset: UInt64(offset),
            data: scalarData(Double(value).bitPattern)
        )
    }

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let writable: Bool
        let pixel: [UInt8]
    }
    let images = [
        ImageResource(id: 20, descriptorSet: 0, binding: 26, writable: false, pixel: [255, 0, 0, 255]),
        ImageResource(id: 21, descriptorSet: 0, binding: 7, writable: false, pixel: [0, 0, 0, 0]),
        ImageResource(id: 22, descriptorSet: 1, binding: 8, writable: false, pixel: [0, 0, 255, 255]),
        ImageResource(id: 23, descriptorSet: 0, binding: 1, writable: true, pixel: [0, 0, 0, 0]),
    ]
    for image in images {
        try backend.createImage(id: image.id, width: 2, height: 2, format: 1, options: 0)
        try backend.writeImage(
            id: image.id,
            data: Data(Array(repeating: image.pixel, count: 4).flatMap { $0 })
        )
    }

    var bindings = buffers.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.kind,
            format: $0.format,
            resourceID: $0.id,
            offset: 0,
            length: $0.size
        )
    }
    bindings.append(contentsOf: images.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.writable ? .textureReadWrite : .textureRead,
            resourceID: $0.id,
            offset: 0,
            length: 0
        )
    })
    let samplerOptions: UInt32 = 1 | 2 | 4 | (1 << 3) | (1 << 6) | (1 << 9)
    bindings.append(ComputeBinding(
        descriptorSet: 2,
        binding: 7,
        arrayElement: 0,
        kind: .sampler,
        format: samplerOptions,
        resourceID: 0,
        offset: packedFloatPair(0, 16),
        length: packedFloatPair(0, 1)
    ))

    try backend.submitCompute(
        pipelineID: pipelineID,
        groupCountX: 1,
        groupCountY: 1,
        groupCountZ: 1,
        bindings: bindings,
        pushConstants: Data(),
        fenceID: 100
    )
    guard try backend.waitFence(id: 100) else {
        throw GPUBackendError.commandFailed("motion FP64 fixture fence was not signaled")
    }
    let pixel = Array([UInt8](try backend.readImage(id: 23)).prefix(4))
    guard pixel.count == 4,
          (63...65).contains(Int(pixel[0])),
          (127...129).contains(Int(pixel[1])),
          (190...192).contains(Int(pixel[2])),
          pixel[3] == 255
    else {
        throw GPUBackendError.commandFailed(
            "motion FP64 fixture output \(pixel) did not match the two-origin difference"
        )
    }
    return pixel
}

/// Executes a captured RTX projection shader whose source position comes from
/// storage buffers. The checked output depth is the length of
/// `(position - FP64 origin) * FP64 scale`, projected by the panorama path.
func executeIsaacFP64ProjectedDepthFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> [UInt8] {
    let expectedHash = "610a7b9aed9f7b98"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the projected-depth FP64 dispatch fixture requires \(expectedHash).spv"
        )
    }

    let backend = try MetalGPUBackend(device: device, spirvCompiler: translator)
    let pipelineID: UInt64 = 1
    let flags = try backend.createComputePipeline(
        id: pipelineID,
        spirv: Data(contentsOf: shader),
        entryPoint: "main"
    )
    guard flags & ComputePipelineFlag.softwareFP64ExecutionRequired != 0 else {
        throw GPUBackendError.commandFailed(
            "captured projected-depth shader was not classified as software-FP64 execution"
        )
    }

    struct BufferResource {
        let id: UInt64
        let size: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let kind: ComputeBindingKind
        let format: UInt32
    }
    let buffers = [
        BufferResource(id: 10, size: 704, descriptorSet: 0, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 11, size: 4_036, descriptorSet: 1, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 12, size: 160, descriptorSet: 3, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 13, size: 256, descriptorSet: 1, binding: 11, kind: .texelBufferRead, format: 109),
        BufferResource(id: 14, size: 256, descriptorSet: 1, binding: 12, kind: .texelBufferRead, format: 109),
        BufferResource(id: 15, size: 256, descriptorSet: 1, binding: 13, kind: .texelBufferRead, format: 109),
        BufferResource(id: 30, size: 64, descriptorSet: 0, binding: 6, kind: .bufferRead, format: 0),
        BufferResource(id: 31, size: 80, descriptorSet: 0, binding: 7, kind: .bufferRead, format: 0),
        BufferResource(id: 32, size: 112, descriptorSet: 0, binding: 8, kind: .bufferRead, format: 0),
        BufferResource(id: 33, size: 192, descriptorSet: 0, binding: 9, kind: .bufferRead, format: 0),
        BufferResource(id: 40, size: 12, descriptorSet: 3, binding: 8, kind: .bufferRead, format: 0),
        BufferResource(id: 41, size: 12, descriptorSet: 3, binding: 9, kind: .bufferRead, format: 0),
        BufferResource(id: 42, size: 4, descriptorSet: 3, binding: 10, kind: .bufferRead, format: 0),
        BufferResource(id: 43, size: 16, descriptorSet: 3, binding: 11, kind: .bufferRead, format: 0),
        BufferResource(id: 44, size: 4, descriptorSet: 3, binding: 12, kind: .bufferRead, format: 0),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    // One panorama camera record and one pixel.
    try backend.writeBuffer(id: 11, offset: 2_784, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 2_816, data: scalarData(Int32(7)))
    try backend.writeBuffer(id: 11, offset: 2_912, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 11, offset: 2_916, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 11, offset: 2_920, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 11, offset: 3_648, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 3_736, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 11, offset: 3_740, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 11, offset: 3_744, data: scalarData(UInt32(1)))

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    try backend.writeBuffer(id: 11, offset: 608, data: floatArrayData(identity))

    // Source position is zero. Subtracting origin (-0.25,-0.5,-0.75) and
    // multiplying by one produces (0.25,0.5,0.75), whose length is sqrt(0.875).
    for (offset, value) in [(2_832, -0.25), (2_840, -0.5), (2_848, -0.75)] {
        try backend.writeBuffer(
            id: 11,
            offset: UInt64(offset),
            data: scalarData(Double(value).bitPattern)
        )
    }

    // The input color's positive alpha enters the projection block. Position
    // buffers remain zero so only the FP64 origin controls projected depth.
    try backend.writeBuffer(id: 43, offset: 0, data: scalarData(SIMD4<Float>(0.1, 0.2, 0.3, 1)))

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let writable: Bool
        let pixel: [UInt8]
    }
    let images = [
        ImageResource(id: 20, descriptorSet: 1, binding: 9, writable: false, pixel: [0, 0, 255, 255]),
        ImageResource(id: 21, descriptorSet: 3, binding: 6, writable: true, pixel: [0, 0, 0, 0]),
        ImageResource(id: 22, descriptorSet: 3, binding: 1, writable: true, pixel: [0, 0, 0, 0]),
        ImageResource(id: 23, descriptorSet: 3, binding: 3, writable: true, pixel: [0, 0, 0, 0]),
        ImageResource(id: 24, descriptorSet: 3, binding: 4, writable: true, pixel: [0, 0, 0, 0]),
        ImageResource(id: 25, descriptorSet: 3, binding: 7, writable: true, pixel: [0, 0, 0, 0]),
        ImageResource(id: 26, descriptorSet: 3, binding: 5, writable: true, pixel: [0, 0, 0, 0]),
    ]
    for image in images {
        try backend.createImage(id: image.id, width: 1, height: 1, format: 1, options: 0)
        try backend.writeImage(id: image.id, data: Data(image.pixel))
    }

    var bindings = buffers.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.kind,
            format: $0.format,
            resourceID: $0.id,
            offset: 0,
            length: $0.size
        )
    }
    bindings.append(contentsOf: images.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.writable ? .textureReadWrite : .textureRead,
            resourceID: $0.id,
            offset: 0,
            length: 0
        )
    })
    let samplerOptions: UInt32 = 1 | 2 | 4 | (1 << 3) | (1 << 6) | (1 << 9)
    bindings.append(ComputeBinding(
        descriptorSet: 4,
        binding: 7,
        arrayElement: 0,
        kind: .sampler,
        format: samplerOptions,
        resourceID: 0,
        offset: packedFloatPair(0, 16),
        length: packedFloatPair(0, 1)
    ))

    try backend.submitCompute(
        pipelineID: pipelineID,
        groupCountX: 1,
        groupCountY: 1,
        groupCountZ: 1,
        bindings: bindings,
        pushConstants: Data(),
        fenceID: 100
    )
    guard try backend.waitFence(id: 100) else {
        throw GPUBackendError.commandFailed("projected-depth FP64 fixture fence was not signaled")
    }
    let pixel = [UInt8](try backend.readImage(id: 24))
    guard pixel.count == 4,
          (238...240).contains(Int(pixel[0])),
          pixel[1] == 0,
          pixel[2] == 0,
          pixel[3] == 0
    else {
        throw GPUBackendError.commandFailed(
            "projected-depth FP64 fixture output \(pixel) did not match sqrt(0.875)"
        )
    }
    return pixel
}

/// Executes the captured final-composite shader through its early clipping
/// output. The panoramic camera ray has y=0 at its origin; software FP64 adds
/// a y origin of 2, placing it exactly on plane -y+2=0. The resulting zero
/// ray length takes the early output path before the shader's unrelated
/// lighting and bindless-material branches.
func executeIsaacFP64CompositeClipFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> [UInt8] {
    let expectedHash = "b258118a36125408"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the composite clipping FP64 dispatch fixture requires \(expectedHash).spv"
        )
    }

    let backend = try MetalGPUBackend(device: device, spirvCompiler: translator)
    let pipelineID: UInt64 = 1
    let flags = try backend.createComputePipeline(
        id: pipelineID,
        spirv: Data(contentsOf: shader),
        entryPoint: "main"
    )
    guard flags & ComputePipelineFlag.softwareFP64ExecutionRequired != 0 else {
        throw GPUBackendError.commandFailed(
            "captured composite shader was not classified as software-FP64 execution"
        )
    }

    struct BufferResource {
        let id: UInt64
        let size: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
    }
    let buffers = [
        BufferResource(id: 10, size: 4_036, descriptorSet: 3, binding: 0),
        BufferResource(id: 11, size: 96, descriptorSet: 7, binding: 0),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    // A 16x16 input matches the captured shader's complete workgroup. Mode 7
    // creates finite panoramic rays without touching the mode-8 input map.
    let inverseExtent = SIMD2<Float>(repeating: 1.0 / 16.0)
    try backend.writeBuffer(id: 10, offset: 2_792, data: scalarData(inverseExtent))
    try backend.writeBuffer(id: 10, offset: 2_816, data: scalarData(Int32(7)))
    try backend.writeBuffer(id: 10, offset: 2_888, data: scalarData(Float(0)))
    try backend.writeBuffer(id: 10, offset: 2_892, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 10, offset: 2_912, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 10, offset: 3_272, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_276, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_368, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 10, offset: 3_648, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 3_744, data: scalarData(UInt32(1)))

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    try backend.writeBuffer(id: 10, offset: 672, data: floatArrayData(identity))

    // FP64 operation: rayOrigin.y / 2 + 2 = 2. Plane -y+2=0
    // therefore produces an exact zero clipping distance. Without the FP64
    // origin the same plane lies in front of the ray and cannot take this
    // early branch.
    try backend.writeBuffer(
        id: 10,
        offset: 2_840,
        data: scalarData(Double(2).bitPattern)
    )
    try backend.writeBuffer(
        id: 10,
        offset: 3_008,
        data: scalarData(SIMD4<Float>(0, -1, 0, 2))
    )

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let format: UInt32
        let writable: Bool
        let fillPixel: [UInt8]
    }
    let images = [
        ImageResource(
            id: 20, descriptorSet: 5, binding: 2, format: 1,
            writable: false, fillPixel: [0, 0, 0, 0]
        ),
        ImageResource(
            id: 21, descriptorSet: 5, binding: 3, format: 8,
            writable: false, fillPixel: [0, 0, 0, 0]
        ),
        ImageResource(
            id: 22, descriptorSet: 5, binding: 4, format: 8,
            writable: false, fillPixel: [0, 0, 0, 0]
        ),
        ImageResource(
            id: 23, descriptorSet: 5, binding: 1, format: 1,
            writable: false, fillPixel: [0, 0, 0, 0]
        ),
        ImageResource(
            id: 24, descriptorSet: 7, binding: 20, format: 1,
            writable: true, fillPixel: [255, 0, 255, 255]
        ),
    ]
    for image in images {
        try backend.createImage(
            id: image.id,
            width: 16,
            height: 16,
            format: image.format,
            options: 0
        )
        var pixels = Data()
        pixels.reserveCapacity(16 * 16 * 4)
        for _ in 0..<(16 * 16) {
            pixels.append(contentsOf: image.fillPixel)
        }
        try backend.writeImage(id: image.id, data: pixels)
    }

    var bindings = buffers.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: .bufferRead,
            resourceID: $0.id,
            offset: 0,
            length: $0.size
        )
    }
    bindings.append(contentsOf: images.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.writable ? .textureReadWrite : .textureRead,
            resourceID: $0.id,
            offset: 0,
            length: 0
        )
    })

    try backend.submitCompute(
        pipelineID: pipelineID,
        groupCountX: 1,
        groupCountY: 1,
        groupCountZ: 1,
        bindings: bindings,
        pushConstants: Data(),
        fenceID: 100
    )
    guard try backend.waitFence(id: 100) else {
        throw GPUBackendError.commandFailed("composite FP64 fixture fence was not signaled")
    }
    let output = [UInt8](try backend.readImage(id: 24))
    let pixel = Array(output.prefix(4))
    guard pixel == [0, 0, 0, 255] else {
        throw GPUBackendError.commandFailed(
            "composite FP64 fixture output \(pixel) did not take the clipping branch"
        )
    }
    return pixel
}

/// Executes the captured RTX volumetric integration shader with real 3D
/// sampled and storage textures. A binary64 camera origin shifts every noise
/// lookup into the bright half of a two-voxel volume; the resulting red
/// accumulation therefore proves the FP64 value, 3D argument binding, and
/// writable Metal texture path in one dispatch.
func executeIsaacFP64VolumeFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> [UInt8] {
    let expectedHash = "c93eebdfc4dd964b"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the volumetric FP64 dispatch fixture requires \(expectedHash).spv"
        )
    }

    let backend = try MetalGPUBackend(device: device, spirvCompiler: translator)
    let pipelineID: UInt64 = 1
    let flags = try backend.createComputePipeline(
        id: pipelineID,
        spirv: Data(contentsOf: shader),
        entryPoint: "main"
    )
    guard flags & ComputePipelineFlag.softwareFP64ExecutionRequired != 0 else {
        throw GPUBackendError.commandFailed(
            "captured volumetric shader was not classified as software-FP64 execution"
        )
    }

    struct BufferResource {
        let id: UInt64
        let size: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let kind: ComputeBindingKind
        let format: UInt32
    }
    let buffers = [
        BufferResource(id: 10, size: 496, descriptorSet: 0, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 11, size: 4_036, descriptorSet: 1, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 12, size: 4_096, descriptorSet: 0, binding: 6, kind: .bufferRead, format: 0),
        BufferResource(id: 13, size: 4_096, descriptorSet: 0, binding: 7, kind: .bufferRead, format: 0),
        BufferResource(id: 14, size: 256, descriptorSet: 1, binding: 11, kind: .texelBufferRead, format: 109),
        BufferResource(id: 15, size: 256, descriptorSet: 1, binding: 12, kind: .texelBufferRead, format: 109),
        BufferResource(id: 16, size: 256, descriptorSet: 1, binding: 13, kind: .texelBufferRead, format: 109),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    // One camera, one z slice, and a finite mode-0 unprojection. The three
    // matrices used by that branch are identity matrices.
    try backend.writeBuffer(id: 11, offset: 3_744, data: scalarData(UInt32(1)))
    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    for offset: UInt64 in [336, 672, 928] {
        try backend.writeBuffer(id: 11, offset: offset, data: floatArrayData(identity))
    }

    // Establish the no-origin baseline first. The fixture later changes only
    // the binary64 x value and compares both real Metal dispatches.
    for (offset, value): (UInt64, Double) in [(1_824, 0), (1_832, 0), (1_840, 0)] {
        try backend.writeBuffer(
            id: 11,
            offset: offset,
            data: scalarData(value.bitPattern)
        )
    }

    // Volumetric constants from block_natural_1.
    for (offset, value): (UInt64, UInt32) in [
        (0, 16), (4, 16), (8, 1), // output extent
        (12, 1),                  // z slices
        (44, 1),                  // noise enabled
        (76, 1),                  // one octave
        (92, 16),                 // xy normalization
    ] {
        try backend.writeBuffer(id: 10, offset: offset, data: scalarData(value))
    }
    try backend.writeBuffer(id: 10, offset: 16, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 24, data: scalarData(Float(0)))
    try backend.writeBuffer(id: 10, offset: 28, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 10, offset: 32, data: floatArrayData([0, 0, 0]))
    try backend.writeBuffer(id: 10, offset: 48, data: floatArrayData([1, 1, 1]))
    try backend.writeBuffer(id: 10, offset: 60, data: scalarData(Float(100)))
    try backend.writeBuffer(id: 10, offset: 64, data: floatArrayData([0, 0, 1]))
    try backend.writeBuffer(id: 10, offset: 80, data: floatArrayData([0, 0, 0]))
    try backend.writeBuffer(id: 10, offset: 96, data: floatArrayData([1, 1, 1]))
    try backend.writeBuffer(id: 10, offset: 108, data: scalarData(Float(0)))
    try backend.writeBuffer(id: 10, offset: 112, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 10, offset: 116, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 10, offset: 120, data: scalarData(Float(0)))

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let width: UInt32
        let height: UInt32
        let depth: UInt32
        let writable: Bool
        let pixels: [UInt8]
    }
    let images = [
        // S10 noise: dark x=0, white x=1.
        ImageResource(
            id: 20, descriptorSet: 0, binding: 2,
            width: 2, height: 1, depth: 1, writable: false,
            pixels: [0, 0, 0, 255, 255, 255, 255, 255]
        ),
        ImageResource(
            id: 21, descriptorSet: 0, binding: 1,
            width: 1, height: 1, depth: 1, writable: false,
            pixels: [255, 0, 0, 255]
        ),
        ImageResource(
            id: 22, descriptorSet: 0, binding: 9,
            width: 1, height: 1, depth: 1, writable: false,
            pixels: [0, 0, 0, 0]
        ),
        ImageResource(
            id: 23, descriptorSet: 0, binding: 3,
            width: 16, height: 16, depth: 1, writable: true,
            pixels: Array(repeating: UInt8(255), count: 16 * 16 * 4)
        ),
    ]
    for image in images {
        guard let options = ImageOption.encodedTexture3D(depth: image.depth) else {
            throw GPUBackendError.outOfBounds
        }
        try backend.createImage(
            id: image.id,
            width: image.width,
            height: image.height,
            format: 1,
            options: options
        )
        try backend.writeImage(id: image.id, data: Data(image.pixels))
    }
    for (id, descriptorSet, binding) in [
        (UInt64(24), UInt32(0), UInt32(5)),
        (UInt64(25), UInt32(1), UInt32(8)),
        (UInt64(26), UInt32(1), UInt32(9)),
    ] {
        try backend.createImage(id: id, width: 1, height: 1, format: 1, options: 0)
        try backend.writeImage(id: id, data: Data([0, 0, 0, 0]))
        _ = descriptorSet
        _ = binding
    }

    var bindings = buffers.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.kind,
            format: $0.format,
            resourceID: $0.id,
            offset: 0,
            length: $0.size
        )
    }
    bindings.append(contentsOf: images.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.writable ? .textureReadWrite : .textureRead,
            resourceID: $0.id,
            offset: 0,
            length: 0
        )
    })
    bindings.append(contentsOf: [
        ComputeBinding(descriptorSet: 0, binding: 5, arrayElement: 0, kind: .textureRead, resourceID: 24, offset: 0, length: 0),
        ComputeBinding(descriptorSet: 1, binding: 8, arrayElement: 0, kind: .textureRead, resourceID: 25, offset: 0, length: 0),
        ComputeBinding(descriptorSet: 1, binding: 9, arrayElement: 0, kind: .textureRead, resourceID: 26, offset: 0, length: 0),
    ])
    let samplerOptions: UInt32 = 1 | 2 | 4 | (1 << 3) | (1 << 6) | (1 << 9)
    for (descriptorSet, binding) in [(UInt32(0), UInt32(12)), (UInt32(2), UInt32(7))] {
        bindings.append(ComputeBinding(
            descriptorSet: descriptorSet,
            binding: binding,
            arrayElement: 0,
            kind: .sampler,
            format: samplerOptions,
            resourceID: 0,
            offset: packedFloatPair(0, 16),
            length: packedFloatPair(0, 1)
        ))
    }

    try backend.submitCompute(
        pipelineID: pipelineID,
        groupCountX: 1,
        groupCountY: 1,
        groupCountZ: 1,
        bindings: bindings,
        pushConstants: Data(),
        fenceID: 99
    )
    guard try backend.waitFence(id: 99) else {
        throw GPUBackendError.commandFailed("volumetric FP64 baseline fence was not signaled")
    }
    let baseline = Array([UInt8](try backend.readImage(id: 23)).prefix(4))
    guard baseline == [0, 0, 0, 0] else {
        throw GPUBackendError.commandFailed(
            "volumetric FP64 baseline \(baseline) sampled the bright noise voxel"
        )
    }

    // Changing only this captured dvec3 component moves the noise lookup from
    // the dark voxel to the bright voxel.
    try backend.writeBuffer(
        id: 11,
        offset: 1_824,
        data: scalarData(Double(0.75).bitPattern)
    )
    try backend.submitCompute(
        pipelineID: pipelineID,
        groupCountX: 1,
        groupCountY: 1,
        groupCountZ: 1,
        bindings: bindings,
        pushConstants: Data(),
        fenceID: 100
    )
    guard try backend.waitFence(id: 100) else {
        throw GPUBackendError.commandFailed("volumetric FP64 fixture fence was not signaled")
    }
    let output = [UInt8](try backend.readImage(id: 23))
    let pixel = Array(output.prefix(4))
    guard pixel.count == 4, pixel[0] >= 8, pixel[1] == 0,
          pixel[2] == 0, pixel[3] == 0
    else {
        throw GPUBackendError.commandFailed(
            "volumetric FP64 fixture output \(pixel) did not contain red accumulated density"
        )
    }
    return pixel
}

guard CommandLine.arguments.count == 3 || CommandLine.arguments.count == 4 else {
    FileHandle.standardError.write(
        Data("usage: imb-shader-probe SPIRV_CROSS SHADER_DIRECTORY [LIMIT]\n".utf8)
    )
    exit(2)
}

let translatorURL = URL(fileURLWithPath: CommandLine.arguments[1])
let shaderDirectory = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
let limit = CommandLine.arguments.count == 4 ? Int(CommandLine.arguments[3]) : nil
let mslDumpDirectory = ProcessInfo.processInfo.environment["IMB_MSL_DUMP_DIR"].map {
    URL(fileURLWithPath: $0, isDirectory: true)
}

guard let device = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write(Data("imb-shader-probe: Metal is unavailable\n".utf8))
    exit(1)
}

do {
    let translator = try SPIRVCrossCompiler(executableURL: translatorURL)
    if let mslDumpDirectory {
        try FileManager.default.createDirectory(
            at: mslDumpDirectory,
            withIntermediateDirectories: true
        )
    }
    var shaders = try FileManager.default.contentsOfDirectory(
        at: shaderDirectory,
        includingPropertiesForKeys: nil
    ).filter { $0.pathExtension == "spv" }
    shaders.sort { $0.lastPathComponent < $1.lastPathComponent }
    if let selectedHash = ProcessInfo.processInfo.environment["IMB_SHADER_HASH"],
       !selectedHash.isEmpty {
        shaders = shaders.filter {
            $0.deletingPathExtension().lastPathComponent == selectedHash
        }
        guard !shaders.isEmpty else {
            throw GPUBackendError.resourceNotFound(0)
        }
    }
    if let limit {
        shaders = Array(shaders.prefix(limit))
    }

    var translatedCount = 0
    var metalCompiledCount = 0
    var translationFailures = 0
    var metalFailures = 0

    for shader in shaders {
        let spirv = try Data(contentsOf: shader)
        let source: String
        do {
            source = try translator.translateToMSL(spirv: spirv)
            translatedCount += 1
            if let mslDumpDirectory {
                let output = mslDumpDirectory
                    .appendingPathComponent(shader.deletingPathExtension().lastPathComponent)
                    .appendingPathExtension("metal")
                try Data(source.utf8).write(to: output, options: .atomic)
            }
        } catch {
            translationFailures += 1
            print("translate-failed \(shader.lastPathComponent): \(firstLine(String(describing: error)))")
            continue
        }

        do {
            let library = try device.makeLibrary(source: source, options: nil)
            guard !library.functionNames.isEmpty else {
                throw GPUBackendError.commandFailed("Metal library has no entry point")
            }
            metalCompiledCount += 1
        } catch {
            metalFailures += 1
            print("metal-failed \(shader.lastPathComponent): \(firstLine(String(describing: error)))")
        }
    }

    print(
        "imb-shader-probe: device=\(device.name) total=\(shaders.count) "
        + "translated=\(translatedCount) translationFailures=\(translationFailures) "
        + "metalCompiled=\(metalCompiledCount) metalFailures=\(metalFailures)"
    )
    if ProcessInfo.processInfo.environment["IMB_EXECUTE_FP64_FIXTURE"] == "1" {
        guard shaders.count == 1, let shader = shaders.first else {
            throw GPUBackendError.unsupported("select one supported fixture shader hash")
        }
        switch shader.deletingPathExtension().lastPathComponent {
        case "f5dd5704d7491f17":
            let output = try executeIsaacFP64TransformFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=f5dd5704d7491f17 "
                + "decodedPosition=\(output.x),\(output.y),\(output.z)"
            )
        case "64d94f5901ee0e4f":
            let pixel = try executeIsaacFP64SamplerFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=64d94f5901ee0e4f "
                + "clippingPlanePixel=\(pixel)"
            )
        case "75321ea922defdfc":
            let pixel = try executeIsaacFP64PositionFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=75321ea922defdfc "
                + "worldPositionPixel=\(pixel)"
            )
        case "361fde9aceebb12b":
            let output = try executeIsaacFP64StructuredPositionFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=361fde9aceebb12b "
                + "structuredWorldPosition=\(output.x),\(output.y),\(output.z)"
            )
        case "2eb5c9558c6d3e5a":
            let pixel = try executeIsaacFP64MotionFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=2eb5c9558c6d3e5a "
                + "motionOriginDifferencePixel=\(pixel)"
            )
        case "610a7b9aed9f7b98":
            let pixel = try executeIsaacFP64ProjectedDepthFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=610a7b9aed9f7b98 "
                + "projectedDepthPixel=\(pixel)"
            )
        case "b258118a36125408":
            let pixel = try executeIsaacFP64CompositeClipFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=b258118a36125408 "
                + "compositeClipPixel=\(pixel)"
            )
        case "c93eebdfc4dd964b":
            let pixel = try executeIsaacFP64VolumeFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=c93eebdfc4dd964b "
                + "volumePixel=\(pixel)"
            )
        default:
            throw GPUBackendError.unsupported(
                "no actual Metal dispatch fixture exists for the selected shader"
            )
        }
    }
    exit(translationFailures == 0 && metalFailures == 0 ? 0 : 1)
} catch {
    FileHandle.standardError.write(Data("imb-shader-probe: \(error)\n".utf8))
    exit(1)
}
