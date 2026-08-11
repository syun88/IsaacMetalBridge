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

func doubleArrayData(_ values: [Double]) -> Data {
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

/// Executes the captured volume-composite sibling through both sides of its
/// clipping branch. With a zero binary64 origin the shader reaches its normal
/// output path and halves the initialized magenta color. Changing only the
/// binary64 y origin to 2 places the ray on plane -y+2=0, so the shader keeps
/// the original pixel. The comparison proves that this exact shader executes
/// its software-FP64 camera-origin arithmetic on Metal.
func executeIsaacFP64VolumeCompositeFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> (baseline: [UInt8], clipped: [UInt8]) {
    let expectedHash = "bdd2d21d53978c2e"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the volume-composite FP64 dispatch fixture requires \(expectedHash).spv"
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
            "captured volume-composite shader was not classified as software-FP64 execution"
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
        BufferResource(id: 10, size: 4_036, descriptorSet: 3, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 11, size: 704, descriptorSet: 4, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 12, size: 96, descriptorSet: 7, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 13, size: 32, descriptorSet: 6, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 14, size: 256, descriptorSet: 3, binding: 11, kind: .texelBufferRead, format: 109),
        BufferResource(id: 15, size: 256, descriptorSet: 3, binding: 12, kind: .texelBufferRead, format: 109),
        BufferResource(id: 16, size: 256, descriptorSet: 3, binding: 13, kind: .texelBufferRead, format: 109),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    // Configure one panoramic camera tile. The generated MSL uses the common
    // UBO's actual 3264-byte inverse-extent field for this shader; keeping the
    // exact reflected offsets here catches layout regressions as well.
    try backend.writeBuffer(
        id: 10,
        offset: 3_264,
        data: scalarData(SIMD2<Float>(repeating: 1.0 / 16.0))
    )
    try backend.writeBuffer(id: 10, offset: 2_816, data: scalarData(Int32(7)))
    try backend.writeBuffer(id: 10, offset: 2_888, data: scalarData(Float(0)))
    try backend.writeBuffer(id: 10, offset: 2_892, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 10, offset: 2_912, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 10, offset: 2_916, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 2_920, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 2_960, data: scalarData(UInt32(16)))
    try backend.writeBuffer(id: 10, offset: 3_272, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_276, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_368, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 10, offset: 3_648, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 3_744, data: scalarData(UInt32(1)))

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    try backend.writeBuffer(id: 10, offset: 672, data: floatArrayData(identity))
    try backend.writeBuffer(
        id: 10,
        offset: 3_008,
        data: scalarData(SIMD4<Float>(0, -1, 0, 2))
    )

    // The non-clipped path preserves alpha and applies inputScale*outputScale
    // to RGB. Volume and atmospheric branches stay disabled, isolating the
    // camera-origin clipping decision from unrelated material state.
    try backend.writeBuffer(id: 13, offset: 0, data: scalarData(Float(0.5)))
    try backend.writeBuffer(id: 13, offset: 4, data: scalarData(Float(1)))

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
    let magenta = [UInt8](arrayLiteral: 255, 0, 255, 255)
    let images = [
        ImageResource(
            id: 20, descriptorSet: 3, binding: 8,
            width: 1, height: 1, depth: 1, writable: false,
            pixels: [0, 0, 0, 0]
        ),
        ImageResource(
            id: 21, descriptorSet: 7, binding: 19,
            width: 16, height: 16, depth: 1, writable: false,
            pixels: Array(repeating: UInt8(0), count: 16 * 16 * 4)
        ),
        ImageResource(
            id: 22, descriptorSet: 7, binding: 20,
            width: 16, height: 16, depth: 1, writable: true,
            pixels: Array(repeating: magenta, count: 16 * 16).flatMap { $0 }
        ),
        ImageResource(
            id: 23, descriptorSet: 7, binding: 17,
            width: 1, height: 1, depth: 1, writable: false,
            pixels: [0, 0, 0, 0]
        ),
    ]
    for image in images {
        let options: UInt32
        if image.binding == 17 {
            guard let encoded = ImageOption.encodedTexture3D(depth: image.depth) else {
                throw GPUBackendError.outOfBounds
            }
            options = encoded
        } else {
            options = 0
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

    func dispatchAndRead(fenceID: UInt64) throws -> [UInt8] {
        try backend.submitCompute(
            pipelineID: pipelineID,
            groupCountX: 1,
            groupCountY: 1,
            groupCountZ: 1,
            bindings: bindings,
            pushConstants: Data(),
            fenceID: fenceID
        )
        guard try backend.waitFence(id: fenceID) else {
            throw GPUBackendError.commandFailed(
                "volume-composite FP64 fixture fence \(fenceID) was not signaled"
            )
        }
        return Array([UInt8](try backend.readImage(id: 22)).prefix(4))
    }

    let baseline = try dispatchAndRead(fenceID: 99)
    guard (127...128).contains(Int(baseline[0])), baseline[1] == 0,
          (127...128).contains(Int(baseline[2])), baseline[3] == 255
    else {
        throw GPUBackendError.commandFailed(
            "volume-composite baseline \(baseline) did not take the scaled output path"
        )
    }

    // Change only the captured binary64 y component. As in the related final
    // composite shader, rayOrigin.y / 2 + 2 lands exactly on -y+2=0.
    try backend.writeBuffer(
        id: 10,
        offset: 2_840,
        data: scalarData(Double(2).bitPattern)
    )
    try backend.writeImage(
        id: 22,
        data: Data(Array(repeating: magenta, count: 16 * 16).flatMap { $0 })
    )
    let clipped = try dispatchAndRead(fenceID: 100)
    guard clipped == magenta else {
        throw GPUBackendError.commandFailed(
            "volume-composite clipped output \(clipped) did not preserve the source pixel"
        )
    }
    return (baseline, clipped)
}

/// Executes both software-binary64 clipping sites in the captured RTX depth
/// consistency shader. The first site clips the current ray and the second
/// clips every valid four-neighbor ray. SPIRV-Cross's generated dataflow
/// assigns both clipped distances back to local ray-length variables, but the
/// subsequent reprojections deliberately use the original origin, direction,
/// and input depth. Two dispatches change only the binary64 world origin and
/// therefore must preserve both the depth image and the neighbor-completion
/// marker. A signed-normalized validity image is required to reach the real
/// neighbor path used by this shader.
func executeIsaacFP64DepthConsistencyFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> (baselineDepth: [UInt8], shiftedDepth: [UInt8], neighborMarker: [UInt8]) {
    let expectedHash = "d1b78c3914cb1874"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the depth-consistency FP64 dispatch fixture requires \(expectedHash).spv"
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
            "captured depth-consistency shader was not classified as software-FP64 execution"
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

    let extent = SIMD2<UInt32>(3, 3)
    let inverseExtent = SIMD2<Float>(repeating: 1.0 / 3.0)
    try backend.writeBuffer(id: 10, offset: 0, data: scalarData(extent))
    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    // Reflected common-UBO offsets: panorama projection and ray transforms.
    for offset: UInt64 in [608, 672] {
        try backend.writeBuffer(id: 11, offset: offset, data: floatArrayData(identity))
    }
    try backend.writeBuffer(id: 11, offset: 2_784, data: scalarData(SIMD2<Float>(3, 3)))
    try backend.writeBuffer(id: 11, offset: 2_792, data: scalarData(inverseExtent))
    try backend.writeBuffer(id: 11, offset: 2_816, data: scalarData(Int32(7)))
    try backend.writeBuffer(id: 11, offset: 2_888, data: scalarData(Float(0)))
    try backend.writeBuffer(id: 11, offset: 2_892, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 11, offset: 2_912, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 11, offset: 2_916, data: scalarData(UInt32(3)))
    try backend.writeBuffer(id: 11, offset: 2_920, data: scalarData(UInt32(3)))
    try backend.writeBuffer(
        id: 11,
        offset: 3_008,
        data: scalarData(SIMD4<Float>(0, -1, 0, 0))
    )
    try backend.writeBuffer(id: 11, offset: 3_264, data: scalarData(inverseExtent))
    try backend.writeBuffer(id: 11, offset: 3_272, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 11, offset: 3_276, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 11, offset: 3_648, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 3_736, data: scalarData(UInt32(3)))
    try backend.writeBuffer(id: 11, offset: 3_740, data: scalarData(UInt32(3)))
    try backend.writeBuffer(id: 11, offset: 3_744, data: scalarData(UInt32(1)))

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let format: UInt32
        let writable: Bool
        let pixels: [UInt8]
    }
    let pixelCount = 9
    let depthPixel = [UInt8](arrayLiteral: 128, 0, 0, 255)
    let zeroPixel = [UInt8](arrayLiteral: 0, 0, 0, 0)
    let neighborSeed = [UInt8](arrayLiteral: 64, 0, 0, 255)
    // 0x81 is -127 in signed normalized RGBA8 and makes every valid neighbor
    // enter the shader's second FP64 clipping/reprojection path.
    let validNeighbor = [UInt8](arrayLiteral: 0x81, 0, 0, 0)
    let images = [
        ImageResource(id: 20, descriptorSet: 0, binding: 26, format: 1, writable: false, pixels: Array(repeating: depthPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 21, descriptorSet: 0, binding: 6, format: 1, writable: false, pixels: Array(repeating: depthPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 22, descriptorSet: 0, binding: 2, format: 1, writable: true, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 23, descriptorSet: 0, binding: 1, format: 1, writable: true, pixels: Array(repeating: neighborSeed, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 24, descriptorSet: 0, binding: 7, format: 9, writable: false, pixels: Array(repeating: validNeighbor, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 25, descriptorSet: 1, binding: 8, format: 1, writable: false, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 26, descriptorSet: 1, binding: 9, format: 1, writable: false, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
    ]
    for image in images {
        try backend.createImage(id: image.id, width: 3, height: 3, format: image.format, options: 0)
        try backend.writeImage(id: image.id, data: Data(image.pixels))
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

    func writeOriginY(_ value: Double) throws {
        for (offset, component) in [(2_832, 0.0), (2_840, value), (2_848, 0.0)] {
            try backend.writeBuffer(
                id: 11,
                offset: UInt64(offset),
                data: scalarData(component.bitPattern)
            )
        }
    }
    func resetOutputs() throws {
        try backend.writeImage(
            id: 22,
            data: Data(Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 })
        )
        try backend.writeImage(
            id: 23,
            data: Data(Array(repeating: neighborSeed, count: pixelCount).flatMap { $0 })
        )
    }
    func dispatchAndRead(fenceID: UInt64) throws -> (depth: [UInt8], marker: [UInt8]) {
        try backend.submitCompute(
            pipelineID: pipelineID,
            groupCountX: 1,
            groupCountY: 1,
            groupCountZ: 1,
            bindings: bindings,
            pushConstants: Data(),
            fenceID: fenceID
        )
        guard try backend.waitFence(id: fenceID) else {
            throw GPUBackendError.commandFailed(
                "depth-consistency FP64 fixture fence \(fenceID) was not signaled"
            )
        }
        return (
            [UInt8](try backend.readImage(id: 22)),
            [UInt8](try backend.readImage(id: 23))
        )
    }

    try resetOutputs()
    try writeOriginY(-2)
    let baseline = try dispatchAndRead(fenceID: 99)
    try resetOutputs()
    try writeOriginY(0)
    let shifted = try dispatchAndRead(fenceID: 100)

    let expectedDepth = Array(repeating: [UInt8](arrayLiteral: 128, 0, 0, 0), count: pixelCount).flatMap { $0 }
    let expectedMarker = Array(repeating: [UInt8](arrayLiteral: 64, 0, 0, 0), count: pixelCount).flatMap { $0 }
    guard baseline.depth == expectedDepth,
          shifted.depth == expectedDepth,
          baseline.marker == expectedMarker,
          shifted.marker == expectedMarker
    else {
        throw GPUBackendError.commandFailed(
            "depth-consistency outputs differed: baselineDepth=\(baseline.depth) "
            + "shiftedDepth=\(shifted.depth) baselineMarker=\(baseline.marker) "
            + "shiftedMarker=\(shifted.marker)"
        )
    }
    return (baseline.depth, shifted.depth, shifted.marker)
}

/// Executes both software-binary64 clipping sites in the captured temporal
/// consistency shader. Each site updates a local maximum ray length, while the
/// immediately following reprojection uses the original sampled depth. The
/// fixture changes only the binary64 camera origins used by those clipping
/// sites and requires the final four-neighbor consistency result to remain
/// identical after both loops have completed.
func executeIsaacFP64TemporalConsistencyFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> (baseline: [UInt8], shifted: [UInt8]) {
    let expectedHash = "0a553b2a8825d0ff"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the temporal-consistency FP64 fixture requires \(expectedHash).spv"
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
            "captured temporal-consistency shader was not classified as software-FP64 execution"
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
        BufferResource(id: 10, size: 48, descriptorSet: 0, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 11, size: 4_036, descriptorSet: 1, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 12, size: 4_036, descriptorSet: 2, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 13, size: 256, descriptorSet: 1, binding: 11, kind: .texelBufferRead, format: 109),
        BufferResource(id: 14, size: 256, descriptorSet: 1, binding: 12, kind: .texelBufferRead, format: 109),
        BufferResource(id: 15, size: 256, descriptorSet: 1, binding: 13, kind: .texelBufferRead, format: 109),
        BufferResource(id: 16, size: 256, descriptorSet: 2, binding: 11, kind: .texelBufferRead, format: 109),
        BufferResource(id: 17, size: 256, descriptorSet: 2, binding: 12, kind: .texelBufferRead, format: 109),
        BufferResource(id: 18, size: 256, descriptorSet: 2, binding: 13, kind: .texelBufferRead, format: 109),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    // Enter the four-neighbor consistency path and make its completed result
    // select the explicit blue diagnostic output.
    try backend.writeBuffer(id: 10, offset: 0, data: scalarData(Float(10)))
    try backend.writeBuffer(id: 10, offset: 4, data: scalarData(Float(2)))
    try backend.writeBuffer(id: 10, offset: 8, data: scalarData(UInt32(1)))

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let extent = SIMD2<UInt32>(3, 3)
    let inverseExtent = SIMD2<Float>(repeating: 1.0 / 3.0)
    for bufferID: UInt64 in [11, 12] {
        // Reflected camera transform and panorama projection fields.
        for offset: UInt64 in [608, 672] {
            try backend.writeBuffer(id: bufferID, offset: offset, data: floatArrayData(identity))
        }
        try backend.writeBuffer(id: bufferID, offset: 2_784, data: scalarData(SIMD2<Float>(3, 3)))
        try backend.writeBuffer(id: bufferID, offset: 2_792, data: scalarData(inverseExtent))
        try backend.writeBuffer(id: bufferID, offset: 2_816, data: scalarData(Int32(7)))
        try backend.writeBuffer(id: bufferID, offset: 2_888, data: scalarData(Float(0)))
        try backend.writeBuffer(id: bufferID, offset: 2_892, data: scalarData(Float(1)))
        try backend.writeBuffer(id: bufferID, offset: 2_912, data: scalarData(Float(2)))
        try backend.writeBuffer(id: bufferID, offset: 2_916, data: scalarData(UInt32(3)))
        try backend.writeBuffer(id: bufferID, offset: 2_920, data: scalarData(UInt32(3)))
        try backend.writeBuffer(id: bufferID, offset: 2_960, data: scalarData(UInt32(3)))
        try backend.writeBuffer(id: bufferID, offset: 2_964, data: scalarData(UInt32(3)))
        try backend.writeBuffer(
            id: bufferID,
            offset: 3_008,
            data: scalarData(SIMD4<Float>(0, 0, 1, 0))
        )
        try backend.writeBuffer(id: bufferID, offset: 3_272, data: scalarData(UInt32(1)))
        try backend.writeBuffer(id: bufferID, offset: 3_276, data: scalarData(UInt32(1)))
        try backend.writeBuffer(id: bufferID, offset: 3_648, data: scalarData(SIMD2<Float>(1, 1)))
        try backend.writeBuffer(id: bufferID, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
        try backend.writeBuffer(id: bufferID, offset: 3_736, data: scalarData(UInt32(3)))
        try backend.writeBuffer(id: bufferID, offset: 3_740, data: scalarData(UInt32(3)))
        try backend.writeBuffer(id: bufferID, offset: 3_744, data: scalarData(UInt32(1)))
    }

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let format: UInt32
        let writable: Bool
        let pixels: [UInt8]
    }
    let pixelCount = 9
    let depthPixel = [UInt8](arrayLiteral: 128, 0, 0, 255)
    let zeroPixel = [UInt8](arrayLiteral: 0, 0, 0, 0)
    let images = [
        ImageResource(id: 20, descriptorSet: 0, binding: 7, format: 1, writable: true, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 21, descriptorSet: 0, binding: 1, format: 1, writable: false, pixels: Array(repeating: depthPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 22, descriptorSet: 0, binding: 2, format: 1, writable: false, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 23, descriptorSet: 0, binding: 3, format: 8, writable: false, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 24, descriptorSet: 0, binding: 4, format: 1, writable: false, pixels: Array(repeating: depthPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 25, descriptorSet: 0, binding: 6, format: 8, writable: false, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 26, descriptorSet: 0, binding: 5, format: 1, writable: false, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 27, descriptorSet: 1, binding: 8, format: 1, writable: false, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 28, descriptorSet: 2, binding: 8, format: 1, writable: false, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
        ImageResource(id: 29, descriptorSet: 2, binding: 9, format: 1, writable: false, pixels: Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 }),
    ]
    for image in images {
        try backend.createImage(id: image.id, width: extent.x, height: extent.y, format: image.format, options: 0)
        try backend.writeImage(id: image.id, data: Data(image.pixels))
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
    for descriptorSet: UInt32 in [0, 3] {
        bindings.append(ComputeBinding(
            descriptorSet: descriptorSet,
            binding: descriptorSet == 0 ? 8 : 7,
            arrayElement: 0,
            kind: .sampler,
            format: samplerOptions,
            resourceID: 0,
            offset: packedFloatPair(0, 16),
            length: packedFloatPair(0, 1)
        ))
    }

    func writeOriginZ(_ value: Double) throws {
        for bufferID: UInt64 in [11, 12] {
            for (offset, component) in [(2_832, 0.0), (2_840, 0.0), (2_848, value)] {
                try backend.writeBuffer(
                    id: bufferID,
                    offset: UInt64(offset),
                    data: scalarData(component.bitPattern)
                )
            }
        }
    }
    func dispatchAndRead(fenceID: UInt64) throws -> [UInt8] {
        try backend.writeImage(
            id: 20,
            data: Data(Array(repeating: zeroPixel, count: pixelCount).flatMap { $0 })
        )
        try backend.submitCompute(
            pipelineID: pipelineID,
            groupCountX: 1,
            groupCountY: 1,
            groupCountZ: 1,
            bindings: bindings,
            pushConstants: Data(),
            fenceID: fenceID
        )
        guard try backend.waitFence(id: fenceID) else {
            throw GPUBackendError.commandFailed(
                "temporal-consistency FP64 fixture fence \(fenceID) was not signaled"
            )
        }
        return [UInt8](try backend.readImage(id: 20))
    }

    try writeOriginZ(-2)
    let baseline = try dispatchAndRead(fenceID: 101)
    try writeOriginZ(0)
    let shifted = try dispatchAndRead(fenceID: 102)
    let expected = Array(
        repeating: [UInt8](arrayLiteral: 0, 0, 255, 255),
        count: pixelCount
    ).flatMap { $0 }
    guard baseline == expected, shifted == expected else {
        throw GPUBackendError.commandFailed(
            "temporal-consistency outputs differed: baseline=\(baseline) shifted=\(shifted)"
        )
    }
    return (baseline, shifted)
}

/// Executes the captured RTX splat-record generation shader and observes its
/// software-binary64 camera-origin clipping through the shader's real storage
/// buffer output. The baseline origin leaves every finite ray length nonzero;
/// moving only origin.y onto the enabled clipping plane makes the upper
/// panoramic rays take the exact-zero branch, reducing the four-level atomic
/// record count produced by the same Metal dispatch.
func executeIsaacFP64SplatRecordFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> (baseline: [UInt32], clipped: [UInt32]) {
    let expectedHash = "75cdce75f76c7184"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the splat-record FP64 dispatch fixture requires \(expectedHash).spv"
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
            "captured splat-record shader was not classified as software-FP64 execution"
        )
    }

    struct BufferResource {
        let id: UInt64
        let size: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let arrayElement: UInt32
        let kind: ComputeBindingKind
        let format: UInt32
    }
    let recordBufferSize: UInt64 = 65_536
    let buffers = [
        BufferResource(id: 10, size: 4_036, descriptorSet: 0, binding: 0, arrayElement: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 11, size: 48, descriptorSet: 2, binding: 0, arrayElement: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 12, size: 512, descriptorSet: 2, binding: 3, arrayElement: 0, kind: .bufferReadWrite, format: 0),
        BufferResource(id: 13, size: recordBufferSize, descriptorSet: 2, binding: 2, arrayElement: 0, kind: .bufferReadWrite, format: 0),
        BufferResource(id: 14, size: recordBufferSize, descriptorSet: 2, binding: 2, arrayElement: 1, kind: .bufferReadWrite, format: 0),
        BufferResource(id: 15, size: recordBufferSize, descriptorSet: 2, binding: 2, arrayElement: 2, kind: .bufferReadWrite, format: 0),
        BufferResource(id: 16, size: recordBufferSize, descriptorSet: 2, binding: 2, arrayElement: 3, kind: .bufferReadWrite, format: 0),
        BufferResource(id: 17, size: 256, descriptorSet: 0, binding: 11, arrayElement: 0, kind: .texelBufferRead, format: 109),
        BufferResource(id: 18, size: 256, descriptorSet: 0, binding: 12, arrayElement: 0, kind: .texelBufferRead, format: 109),
        BufferResource(id: 19, size: 256, descriptorSet: 0, binding: 13, arrayElement: 0, kind: .texelBufferRead, format: 109),
        BufferResource(id: 23, size: 16, descriptorSet: 3, binding: 0, arrayElement: 0, kind: .bufferRead, format: 0),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    // One 32x32 panorama and one camera record. The exact reflected matrix
    // offsets are the camera-space and camera-to-world transforms used by this
    // shader's non-mode-0 ray and circle-of-confusion calculations.
    for offset: UInt64 in [608, 672] {
        try backend.writeBuffer(id: 10, offset: offset, data: floatArrayData(identity))
    }
    try backend.writeBuffer(id: 10, offset: 2_792, data: scalarData(SIMD2<Float>(repeating: 1.0 / 32.0)))
    try backend.writeBuffer(id: 10, offset: 2_816, data: scalarData(Int32(7)))
    try backend.writeBuffer(id: 10, offset: 2_888, data: scalarData(Float(0)))
    try backend.writeBuffer(id: 10, offset: 2_892, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 10, offset: 2_912, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 10, offset: 2_916, data: scalarData(UInt32(32)))
    try backend.writeBuffer(id: 10, offset: 2_920, data: scalarData(UInt32(32)))
    // The plane is -y=0. With origin.y=-2 the finite rays remain nonzero;
    // origin.y=0 later puts the ray origin exactly on the plane.
    try backend.writeBuffer(
        id: 10,
        offset: 3_008,
        data: scalarData(SIMD4<Float>(0, -1, 0, 0))
    )
    try backend.writeBuffer(id: 10, offset: 3_264, data: scalarData(SIMD2<Float>(repeating: 1.0 / 32.0)))
    try backend.writeBuffer(id: 10, offset: 3_272, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_276, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 10, offset: 3_648, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 10, offset: 3_736, data: scalarData(UInt32(32)))
    try backend.writeBuffer(id: 10, offset: 3_740, data: scalarData(UInt32(32)))
    try backend.writeBuffer(id: 10, offset: 3_744, data: scalarData(UInt32(1)))

    // Force the shader down its per-sample record path. The optical constants
    // deliberately make a 0.5 depth sample generate a magnitude above one,
    // while very large hierarchy thresholds prevent coarser reductions from
    // hiding the four individual FP64-controlled zero decisions.
    try backend.writeBuffer(id: 11, offset: 0, data: scalarData(Float(128)))
    try backend.writeBuffer(id: 11, offset: 4, data: scalarData(Float(2)))
    for offset: UInt64 in [12, 16, 20] {
        try backend.writeBuffer(id: 11, offset: offset, data: scalarData(Float(1_000_000)))
    }
    try backend.writeBuffer(id: 11, offset: 24, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 11, offset: 28, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 11, offset: 32, data: scalarData(Float(128)))
    try backend.writeBuffer(id: 23, offset: 0, data: scalarData(SIMD2<Float>(repeating: 1.0 / 32.0)))
    try backend.writeBuffer(id: 23, offset: 8, data: scalarData(SIMD2<UInt32>(32, 32)))

    // Both sampled images use the actual 32x32 footprint accessed by the
    // generated gather/read operations. RGBA8 normalized depth 128 is finite
    // and the colored source makes any generated record data nontrivial.
    try backend.createImage(id: 20, width: 1, height: 1, format: 1, options: 0)
    try backend.writeImage(id: 20, data: Data([0, 0, 0, 0]))
    try backend.createImage(id: 21, width: 32, height: 32, format: 1, options: 0)
    try backend.writeImage(
        id: 21,
        data: Data(Array(repeating: [UInt8](arrayLiteral: 64, 128, 191, 255), count: 32 * 32).flatMap { $0 })
    )
    try backend.createImage(id: 22, width: 32, height: 32, format: 1, options: 0)
    try backend.writeImage(
        id: 22,
        data: Data(Array(repeating: [UInt8](arrayLiteral: 128, 0, 0, 255), count: 32 * 32).flatMap { $0 })
    )

    var bindings = buffers.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: $0.arrayElement,
            kind: $0.kind,
            format: $0.format,
            resourceID: $0.id,
            offset: 0,
            length: $0.size
        )
    }
    bindings.append(contentsOf: [
        ComputeBinding(descriptorSet: 0, binding: 8, arrayElement: 0, kind: .textureRead, resourceID: 20, offset: 0, length: 0),
        ComputeBinding(descriptorSet: 2, binding: 1, arrayElement: 0, kind: .textureRead, resourceID: 21, offset: 0, length: 0),
        ComputeBinding(descriptorSet: 3, binding: 1, arrayElement: 0, kind: .textureRead, resourceID: 22, offset: 0, length: 0),
    ])
    let samplerOptions: UInt32 = 1 | 2 | 4 | (1 << 3) | (1 << 6) | (1 << 9)
    for descriptor: (UInt32, UInt32) in [(3, 2), (7, 7)] {
        bindings.append(ComputeBinding(
            descriptorSet: descriptor.0,
            binding: descriptor.1,
            arrayElement: 0,
            kind: .sampler,
            format: samplerOptions,
            resourceID: 0,
            offset: packedFloatPair(0, 16),
            length: packedFloatPair(0, 1)
        ))
    }

    func writeOriginY(_ value: Double) throws {
        try backend.writeBuffer(id: 10, offset: 2_832, data: scalarData(Double(0).bitPattern))
        try backend.writeBuffer(id: 10, offset: 2_840, data: scalarData(value.bitPattern))
        try backend.writeBuffer(id: 10, offset: 2_848, data: scalarData(Double(0).bitPattern))
    }
    func clearOutputs() throws {
        try backend.writeBuffer(id: 12, offset: 0, data: Data(repeating: 0, count: 512))
        for id: UInt64 in 13...16 {
            try backend.writeBuffer(
                id: id,
                offset: 0,
                data: Data(repeating: 0, count: Int(recordBufferSize))
            )
        }
    }
    func dispatchAndRead(fenceID: UInt64) throws -> [UInt32] {
        try backend.submitCompute(
            pipelineID: pipelineID,
            groupCountX: 1,
            groupCountY: 1,
            groupCountZ: 1,
            bindings: bindings,
            pushConstants: Data(),
            fenceID: fenceID
        )
        guard try backend.waitFence(id: fenceID) else {
            throw GPUBackendError.commandFailed(
                "splat-record FP64 fixture fence \(fenceID) was not signaled"
            )
        }
        let counterData = try backend.readBuffer(id: 12, offset: 0, length: 512)
        return [4, 132, 260, 388].map { offset in
            counterData.withUnsafeBytes {
                $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
            }
        }
    }

    try clearOutputs()
    try writeOriginY(-2)
    let baseline = try dispatchAndRead(fenceID: 99)
    try clearOutputs()
    try writeOriginY(0)
    let clipped = try dispatchAndRead(fenceID: 100)
    let expectedBaseline: [UInt32] = [64, 32, 128, 800]
    let expectedClipped: [UInt32] = [32, 32, 64, 384]
    guard baseline == expectedBaseline, clipped == expectedClipped else {
        throw GPUBackendError.commandFailed(
            "splat-record FP64 counters differed: baseline=\(baseline) clipped=\(clipped)"
        )
    }
    return (baseline, clipped)
}

/// Executes the captured render-output inspection shader through its mode-3
/// position reconstruction path. The reconstructed local position is zero for
/// the one-voxel fixture, so the shader's real software-binary64
/// `origin + local / scale` operation is observable directly in the RGBA8
/// output. Two dispatches that differ only in the dvec3 origin prove that this
/// exact (large, production-captured) shader executes FP64-derived color on
/// Metal rather than merely compiling it.
func executeIsaacFP64RenderOutputFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> (baseline: [UInt8], shifted: [UInt8]) {
    let expectedHash = "d38fa4ca78558061"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported(
            "the render-output FP64 dispatch fixture requires \(expectedHash).spv"
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
            "captured render-output shader was not classified as software-FP64 execution"
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
        BufferResource(id: 11, size: 80, descriptorSet: 2, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 12, size: 4_036, descriptorSet: 3, binding: 0, kind: .bufferRead, format: 0),
        BufferResource(id: 13, size: 256, descriptorSet: 3, binding: 11, kind: .texelBufferRead, format: 109),
        BufferResource(id: 14, size: 256, descriptorSet: 3, binding: 12, kind: .texelBufferRead, format: 109),
        BufferResource(id: 15, size: 256, descriptorSet: 3, binding: 13, kind: .texelBufferRead, format: 109),
    ]
    for buffer in buffers {
        try backend.createBuffer(id: buffer.id, size: buffer.size, options: 0)
    }

    // One output pixel and one voxel select mode 3 (position reconstruction).
    // A zero depth sample plus identity transforms reconstructs local (0,0,0).
    try backend.writeBuffer(id: 11, offset: 0, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 8, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 11, offset: 16, data: scalarData(SIMD3<UInt32>(1, 1, 1)))
    try backend.writeBuffer(id: 11, offset: 28, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 11, offset: 32, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 11, offset: 60, data: scalarData(Int32(3)))

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    // Reflected nested camera transforms used by both halves of the mode-0
    // depth/position conversion. Keeping the real offsets checks ABI layout.
    for offset: UInt64 in [336, 864, 928] {
        try backend.writeBuffer(id: 12, offset: offset, data: floatArrayData(identity))
    }
    try backend.writeBuffer(id: 12, offset: 2_816, data: scalarData(Int32(0)))
    try backend.writeBuffer(id: 12, offset: 2_912, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 12, offset: 2_916, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 12, offset: 2_920, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 12, offset: 3_264, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 12, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 12, offset: 3_744, data: scalarData(UInt32(1)))

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let is3D: Bool
        let writable: Bool
    }
    let images = [
        ImageResource(id: 20, descriptorSet: 0, binding: 3, is3D: true, writable: false),
        ImageResource(id: 21, descriptorSet: 0, binding: 2, is3D: true, writable: false),
        ImageResource(id: 22, descriptorSet: 1, binding: 2, is3D: true, writable: false),
        ImageResource(id: 23, descriptorSet: 2, binding: 1, is3D: false, writable: false),
        ImageResource(id: 24, descriptorSet: 3, binding: 8, is3D: false, writable: false),
        ImageResource(id: 25, descriptorSet: 3, binding: 9, is3D: false, writable: false),
        ImageResource(id: 26, descriptorSet: 3, binding: 1, is3D: false, writable: true),
    ]
    for image in images {
        let options: UInt32
        if image.is3D {
            guard let encoded = ImageOption.encodedTexture3D(depth: 1) else {
                throw GPUBackendError.outOfBounds
            }
            options = encoded
        } else {
            options = 0
        }
        try backend.createImage(id: image.id, width: 1, height: 1, format: 1, options: options)
        try backend.writeImage(id: image.id, data: Data([0, 0, 0, image.writable ? 255 : 0]))
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
    for descriptor: (UInt32, UInt32) in [(0, 12), (4, 7)] {
        bindings.append(ComputeBinding(
            descriptorSet: descriptor.0,
            binding: descriptor.1,
            arrayElement: 0,
            kind: .sampler,
            format: samplerOptions,
            resourceID: 0,
            offset: packedFloatPair(0, 16),
            length: packedFloatPair(0, 1)
        ))
    }

    func writeOrigin(_ value: SIMD3<Double>) throws {
        try backend.writeBuffer(id: 12, offset: 2_832, data: scalarData(value.x.bitPattern))
        try backend.writeBuffer(id: 12, offset: 2_840, data: scalarData(value.y.bitPattern))
        try backend.writeBuffer(id: 12, offset: 2_848, data: scalarData(value.z.bitPattern))
    }
    func dispatchAndRead(fenceID: UInt64) throws -> [UInt8] {
        try backend.submitCompute(
            pipelineID: pipelineID,
            groupCountX: 1,
            groupCountY: 1,
            groupCountZ: 1,
            bindings: bindings,
            pushConstants: Data(),
            fenceID: fenceID
        )
        guard try backend.waitFence(id: fenceID) else {
            throw GPUBackendError.commandFailed(
                "render-output FP64 fixture fence \(fenceID) was not signaled"
            )
        }
        return Array([UInt8](try backend.readImage(id: 26)).prefix(4))
    }

    try writeOrigin(SIMD3<Double>(0.25, 0.5, 0.75))
    let baseline = try dispatchAndRead(fenceID: 99)
    try writeOrigin(SIMD3<Double>(0.75, 0.25, 0.5))
    let shifted = try dispatchAndRead(fenceID: 100)
    guard baseline != shifted else {
        throw GPUBackendError.commandFailed(
            "render-output FP64 origin did not affect output: \(baseline)"
        )
    }
    return (baseline, shifted)
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

/// Executes the captured Isaac shader which owns the final two UInt64
/// compare-exchange instructions. Only pixel (0, 0) is in bounds; the bridge
/// serializes the 16x16 logical workgroup on one Metal thread. A successful
/// insertion changes one all-ones hash slot and increments the real counter.
func executeIsaacAtomic64CASFixture(
    device: any MTLDevice,
    translator: SPIRVCrossCompiler,
    shader: URL
) throws -> (counter: UInt32, occupiedSlots: Int, insertedKey: UInt64) {
    let expectedHash = "f72cb1589be7be96"
    guard shader.deletingPathExtension().lastPathComponent == expectedHash else {
        throw GPUBackendError.unsupported("the atomic64 fixture requires \(expectedHash).spv")
    }

    let backend = try MetalGPUBackend(device: device, spirvCompiler: translator)
    let pipelineID: UInt64 = 1
    let flags = try backend.createComputePipeline(
        id: pipelineID,
        spirv: Data(contentsOf: shader),
        entryPoint: "main"
    )
    let expectedFlags = ComputePipelineFlag.softwareFP64ExecutionRequired
        | ComputePipelineFlag.serializedAtomic64ExecutionRequired
    guard flags & expectedFlags == expectedFlags else {
        throw GPUBackendError.commandFailed(
            "captured atomic shader flags \(flags) did not include software FP64 and serialized atomic64"
        )
    }

    struct BufferResource {
        let id: UInt64
        let size: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let writable: Bool
    }
    let buffers: [BufferResource] = [
        BufferResource(id: 100, size: 65_536, descriptorSet: 0, binding: 3, writable: false),
        BufferResource(id: 101, size: 704, descriptorSet: 2, binding: 0, writable: false),
        BufferResource(id: 102, size: 65_536, descriptorSet: 2, binding: 1, writable: false),
        BufferResource(id: 103, size: 65_536, descriptorSet: 2, binding: 6, writable: false),
        BufferResource(id: 104, size: 65_536, descriptorSet: 2, binding: 7, writable: false),
        BufferResource(id: 105, size: 65_536, descriptorSet: 2, binding: 27, writable: false),
        BufferResource(id: 106, size: 4_036, descriptorSet: 3, binding: 0, writable: false),
        BufferResource(id: 107, size: 65_536, descriptorSet: 3, binding: 4, writable: true),
        BufferResource(id: 108, size: 176, descriptorSet: 6, binding: 0, writable: false),
        BufferResource(id: 109, size: 65_536, descriptorSet: 6, binding: 3, writable: true),
        BufferResource(id: 110, size: 65_536, descriptorSet: 6, binding: 6, writable: true),
        BufferResource(id: 111, size: 65_536, descriptorSet: 6, binding: 7, writable: true),
        BufferResource(id: 112, size: 96, descriptorSet: 7, binding: 0, writable: false),
        BufferResource(id: 113, size: 272, descriptorSet: 8, binding: 0, writable: false),
        BufferResource(id: 114, size: 32, descriptorSet: 9, binding: 0, writable: false),
        BufferResource(id: 115, size: 65_536, descriptorSet: 9, binding: 3, writable: true),
        BufferResource(id: 116, size: 65_536, descriptorSet: 9, binding: 4, writable: true),
        BufferResource(id: 117, size: 112, descriptorSet: 10, binding: 0, writable: false),
        BufferResource(id: 118, size: 64, descriptorSet: 10, binding: 1, writable: true),
        BufferResource(id: 119, size: 65_536, descriptorSet: 10, binding: 3, writable: true),
        BufferResource(id: 120, size: 4, descriptorSet: 10, binding: 5, writable: true),
    ]
    for resource in buffers {
        try backend.createBuffer(id: resource.id, size: resource.size, options: 0)
    }

    struct TexelResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
    }
    let texels = [
        TexelResource(id: 130, descriptorSet: 3, binding: 11),
        TexelResource(id: 131, descriptorSet: 3, binding: 12),
        TexelResource(id: 132, descriptorSet: 3, binding: 13),
    ]
    for texel in texels {
        try backend.createBuffer(id: texel.id, size: 4_096, options: 0)
    }

    struct ImageResource {
        let id: UInt64
        let descriptorSet: UInt32
        let binding: UInt32
        let uintFormat: Bool
        let writable: Bool
        let texture3D: Bool
    }
    let images: [ImageResource] = [
        ImageResource(id: 200, descriptorSet: 3, binding: 1, uintFormat: false, writable: true, texture3D: false),
        ImageResource(id: 201, descriptorSet: 3, binding: 2, uintFormat: true, writable: false, texture3D: false),
        ImageResource(id: 202, descriptorSet: 3, binding: 3, uintFormat: true, writable: false, texture3D: false),
        ImageResource(id: 203, descriptorSet: 3, binding: 8, uintFormat: false, writable: false, texture3D: false),
        ImageResource(id: 204, descriptorSet: 3, binding: 9, uintFormat: false, writable: false, texture3D: false),
        ImageResource(id: 205, descriptorSet: 4, binding: 1, uintFormat: false, writable: false, texture3D: false),
        ImageResource(id: 206, descriptorSet: 4, binding: 3, uintFormat: true, writable: false, texture3D: false),
        ImageResource(id: 207, descriptorSet: 4, binding: 4, uintFormat: true, writable: false, texture3D: false),
        ImageResource(id: 208, descriptorSet: 4, binding: 5, uintFormat: false, writable: false, texture3D: false),
        ImageResource(id: 209, descriptorSet: 4, binding: 8, uintFormat: true, writable: false, texture3D: false),
        ImageResource(id: 210, descriptorSet: 5, binding: 1, uintFormat: false, writable: false, texture3D: false),
        ImageResource(id: 211, descriptorSet: 5, binding: 2, uintFormat: true, writable: false, texture3D: false),
        ImageResource(id: 212, descriptorSet: 5, binding: 3, uintFormat: false, writable: false, texture3D: false),
        ImageResource(id: 213, descriptorSet: 5, binding: 5, uintFormat: false, writable: false, texture3D: false),
        ImageResource(id: 214, descriptorSet: 6, binding: 1, uintFormat: false, writable: false, texture3D: false),
        ImageResource(id: 215, descriptorSet: 6, binding: 2, uintFormat: true, writable: false, texture3D: false),
        ImageResource(id: 216, descriptorSet: 6, binding: 4, uintFormat: false, writable: false, texture3D: false),
        ImageResource(id: 217, descriptorSet: 6, binding: 5, uintFormat: false, writable: true, texture3D: false),
        ImageResource(id: 218, descriptorSet: 10, binding: 6, uintFormat: false, writable: true, texture3D: true),
    ]
    for image in images {
        let options: UInt32 = image.texture3D
            ? ImageOption.texture3D | (1 << ImageOption.depthShift)
            : 0
        try backend.createImage(
            id: image.id,
            width: 1,
            height: 1,
            format: image.uintFormat ? 8 : 1,
            options: options
        )
    }

    // One pixel is active. Identity transforms and unit scales keep case 7's
    // captured key construction finite and deterministic.
    try backend.writeBuffer(id: 106, offset: 2_960, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 106, offset: 2_964, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 106, offset: 2_916, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 106, offset: 2_920, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 106, offset: 2_928, data: scalarData(SIMD4<Float>(0, 0, 0, 1)))
    try backend.writeBuffer(id: 106, offset: 3_728, data: scalarData(SIMD2<UInt32>(1, 1)))
    try backend.writeBuffer(id: 106, offset: 3_736, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 106, offset: 3_744, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 106, offset: 592, data: scalarData(SIMD3<Float>(0, 0, 1)))
    let identityFloat: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ]
    let identityDouble = identityFloat.map(Double.init)
    try backend.writeBuffer(id: 106, offset: 0, data: doubleArrayData(identityDouble))
    try backend.writeBuffer(id: 106, offset: 128, data: doubleArrayData(identityDouble))
    for offset: UInt64 in [2_336, 2_400, 2_464, 2_528, 2_592, 2_656, 2_720] {
        try backend.writeBuffer(id: 106, offset: offset, data: floatArrayData(identityFloat))
    }
    try backend.writeBuffer(id: 108, offset: 4, data: scalarData(UInt32(7)))
    try backend.writeBuffer(id: 108, offset: 48, data: scalarData(SIMD2<Float>(1, 1)))
    try backend.writeBuffer(id: 108, offset: 80, data: scalarData(SIMD2<Float>(0, 0)))
    try backend.writeBuffer(id: 108, offset: 112, data: scalarData(Float(1)))

    // Eight empty UInt64 slots. Offset 100 is the 64-bit multiplicative-hash
    // shift (64 - log2(8)); case-7 enable and finite quantization constants
    // select the insertion path.
    try backend.writeBuffer(id: 117, offset: 0, data: scalarData(UInt32(8)))
    try backend.writeBuffer(id: 117, offset: 32, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 117, offset: 36, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 117, offset: 44, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 117, offset: 48, data: scalarData(UInt32(1)))
    try backend.writeBuffer(id: 117, offset: 76, data: scalarData(Float(1)))
    try backend.writeBuffer(id: 117, offset: 100, data: scalarData(UInt32(61)))
    try backend.writeBuffer(id: 118, offset: 0, data: Data(repeating: 0xff, count: 64))

    var bindings = buffers.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: $0.writable ? .bufferReadWrite : .bufferRead,
            resourceID: $0.id,
            offset: 0,
            length: $0.size
        )
    }
    bindings.append(contentsOf: texels.map {
        ComputeBinding(
            descriptorSet: $0.descriptorSet,
            binding: $0.binding,
            arrayElement: 0,
            kind: .texelBufferRead,
            format: 100,
            resourceID: $0.id,
            offset: 0,
            length: 4_096
        )
    })
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
    for (descriptorSet, binding) in [
        (UInt32(6), UInt32(8)),
        (UInt32(6), UInt32(9)),
        (UInt32(7), UInt32(7)),
    ] {
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
        fenceID: 103
    )
    guard try backend.waitFence(id: 103) else {
        throw GPUBackendError.commandFailed("atomic64 fixture fence was not signaled")
    }
    let counter = try backend.readBuffer(id: 120, offset: 0, length: 4).withUnsafeBytes {
        UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self))
    }
    let table = try backend.readBuffer(id: 118, offset: 0, length: 64)
    let slots: [UInt64] = table.withUnsafeBytes { bytes in
        (0..<8).map {
            UInt64(littleEndian: bytes.loadUnaligned(
                fromByteOffset: $0 * MemoryLayout<UInt64>.size,
                as: UInt64.self
            ))
        }
    }
    let occupied = slots.filter { $0 != UInt64.max }
    guard counter == 1, occupied.count == 1, let insertedKey = occupied.first else {
        throw GPUBackendError.commandFailed(
            "captured atomic64 insertion produced counter=\(counter) slots=\(slots)"
        )
    }
    return (counter, occupied.count, insertedKey)
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
        case "bdd2d21d53978c2e":
            let output = try executeIsaacFP64VolumeCompositeFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=bdd2d21d53978c2e "
                + "baselinePixel=\(output.baseline) clippedPixel=\(output.clipped)"
            )
        case "75cdce75f76c7184":
            let output = try executeIsaacFP64SplatRecordFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=75cdce75f76c7184 "
                + "baselineCounters=\(output.baseline) clippedCounters=\(output.clipped)"
            )
        case "d1b78c3914cb1874":
            let output = try executeIsaacFP64DepthConsistencyFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=d1b78c3914cb1874 "
                + "baselineDepth=\(output.baselineDepth) shiftedDepth=\(output.shiftedDepth) "
                + "neighborMarker=\(output.neighborMarker)"
            )
        case "0a553b2a8825d0ff":
            let output = try executeIsaacFP64TemporalConsistencyFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=0a553b2a8825d0ff "
                + "baselinePixel=\(Array(output.baseline.prefix(4))) "
                + "shiftedPixel=\(Array(output.shifted.prefix(4)))"
            )
        case "d38fa4ca78558061":
            let output = try executeIsaacFP64RenderOutputFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=d38fa4ca78558061 "
                + "baselinePixel=\(output.baseline) shiftedPixel=\(output.shifted)"
            )
        case "f72cb1589be7be96":
            let output = try executeIsaacAtomic64CASFixture(
                device: device,
                translator: translator,
                shader: shader
            )
            print(
                "imb-shader-probe: actualMetalDispatch=passed shader=f72cb1589be7be96 "
                + "serializedAtomic64=passed counter=\(output.counter) "
                + "occupiedSlots=\(output.occupiedSlots) "
                + "insertedKey=0x\(String(output.insertedKey, radix: 16))"
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
