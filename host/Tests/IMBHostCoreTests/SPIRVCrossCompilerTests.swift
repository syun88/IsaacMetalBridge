import Foundation
import Testing
@testable import IMBHostCore

@Test func softwareFloat64SupportLeavesOrdinaryMSLUnchanged() {
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void main0(uint index [[thread_position_in_grid]]) {}
    """

    #expect(SPIRVCrossCompiler.addSoftwareFloat64SupportIfNeeded(to: source) == source)
}

@Test func softwareFloat64SupportNormalizesOnlyNumericLongFloatSuffixes() {
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    struct spvDoubleValue { int self; };
    constant float value = 1.25lf;
    """

    let lowered = SPIRVCrossCompiler.addSoftwareFloat64SupportIfNeeded(to: source)
    #expect(lowered.contains("constant float value = 1.25f;"))
    #expect(lowered.contains("int self;"))
    #expect(lowered.contains("struct spvDouble"))
    #expect(lowered.contains(SPIRVCrossCompiler.softwareFloat64DeclarationOnlyMarker))
}

@Test func softwareFloat64SupportDistinguishesEntryPointUseFromBufferDeclarations() {
    let declarationOnly = """
    #include <metal_stdlib>
    using namespace metal;
    struct Storage
    {
        spvDouble4x4 unusedMatrix;
    };
    kernel void main0(uint index [[thread_position_in_grid]]) {}
    """
    let executableUse = """
    #include <metal_stdlib>
    using namespace metal;
    struct Storage
    {
        spvDouble value;
    };
    kernel void main0(device Storage& storage [[buffer(0)]])
    {
        spvDouble temporary = storage.value;
    }
    """

    #expect(
        SPIRVCrossCompiler.addSoftwareFloat64SupportIfNeeded(to: declarationOnly)
            .contains(SPIRVCrossCompiler.softwareFloat64DeclarationOnlyMarker)
    )
    #expect(
        SPIRVCrossCompiler.addSoftwareFloat64SupportIfNeeded(to: executableUse)
            .contains(SPIRVCrossCompiler.softwareFloat64ExecutionRequiredMarker)
    )
}

@Test func metalArgumentBindingMapTracksCompactedSparseBindingsAndAliases() throws {
    func stringWords(_ value: String) -> [UInt32] {
        var bytes = Array(value.utf8) + [0]
        while bytes.count % 4 != 0 { bytes.append(0) }
        return stride(from: 0, to: bytes.count, by: 4).map { offset in
            UInt32(bytes[offset])
                | (UInt32(bytes[offset + 1]) << 8)
                | (UInt32(bytes[offset + 2]) << 16)
                | (UInt32(bytes[offset + 3]) << 24)
        }
    }

    func opName(_ id: UInt32, _ name: String) -> [UInt32] {
        let encoded = stringWords(name)
        return [UInt32((encoded.count + 2) << 16) | 5, id] + encoded
    }

    func opDecorate(_ id: UInt32, _ decoration: UInt32, _ value: UInt32) -> [UInt32] {
        [UInt32(4 << 16) | 71, id, decoration, value]
    }

    var words: [UInt32] = [0x0723_0203, 0x0001_0000, 0, 32, 0]
    words += opName(10, "_Sparse")
    words += opDecorate(10, 34, 2)
    words += opDecorate(10, 33, 10)
    words += opName(11, "_AliasA")
    words += opDecorate(11, 34, 2)
    words += opDecorate(11, 33, 4)
    words += opName(12, "_AliasB")
    words += opDecorate(12, 34, 2)
    words += opDecorate(12, 33, 4)
    words += opName(13, "_13")
    words += opDecorate(13, 34, 0)
    words += opDecorate(13, 33, 0)
    words += opName(14, "_Direct")
    words += opDecorate(14, 34, 9)
    words += opDecorate(14, 33, 3)
    words += opName(15, "_Tex")
    words += opDecorate(15, 34, 9)
    words += opDecorate(15, 33, 4)
    let spirv = words.withUnsafeBytes { Data($0) }
    let msl = """
    struct spvDescriptorSetBuffer2
    {
        device uint* _Sparse [[id(0)]];
        device uint* _AliasA [[id(1)]];
        // Overlapping binding: device int* _AliasB [[id(2)]];
    };
    struct spvDescriptorSetBuffer0
    {
        device uint* m_13 [[id(0)]];
    };
    kernel void imb_compute_main(
        device spvDescriptorSetBuffer0& spvDescriptorSet0 [[buffer(0)]],
        device uint* _Direct [[buffer(8)]],
        device void* spvBufferAliasSet10Binding1 [[buffer(9)]],
        texture2d<float> _Tex [[texture(3)]],
        uint3 gl_GlobalInvocationID [[thread_position_in_grid]]) {}
    """

    let map = SPIRVCrossCompiler.metalArgumentBindingMap(spirv: spirv, msl: msl)
    #expect(map.emittedFieldCount == 7)
    #expect(map.mappedFieldCount == 7)
    #expect(map.indicesByVulkanBinding[
        SPIRVDescriptorBinding(descriptorSet: 2, binding: 10)
    ] == [0])
    #expect(map.indicesByVulkanBinding[
        SPIRVDescriptorBinding(descriptorSet: 2, binding: 4)
    ] == [1, 2])
    #expect(map.indicesByVulkanBinding[
        SPIRVDescriptorBinding(descriptorSet: 0, binding: 0)
    ] == [0])
    #expect(map.directBufferIndicesByVulkanBinding[
        SPIRVDescriptorBinding(descriptorSet: 9, binding: 3)
    ] == [8])
    #expect(map.directBufferIndicesByVulkanBinding[
        SPIRVDescriptorBinding(descriptorSet: 10, binding: 1)
    ] == [9])
    #expect(map.directTextureIndicesByVulkanBinding[
        SPIRVDescriptorBinding(descriptorSet: 9, binding: 4)
    ] == [3])
}

@Test func pushConstantReflectionDoesNotSelectAnEarlierDirectUBO() {
    func stringWords(_ value: String) -> [UInt32] {
        var bytes = Array(value.utf8) + [0]
        while bytes.count % 4 != 0 { bytes.append(0) }
        return stride(from: 0, to: bytes.count, by: 4).map { offset in
            UInt32(bytes[offset])
                | (UInt32(bytes[offset + 1]) << 8)
                | (UInt32(bytes[offset + 2]) << 16)
                | (UInt32(bytes[offset + 3]) << 24)
        }
    }

    func opName(_ id: UInt32, _ name: String) -> [UInt32] {
        let encoded = stringWords(name)
        return [UInt32((encoded.count + 2) << 16) | 5, id] + encoded
    }

    var words: [UInt32] = [0x0723_0203, 0x0001_0000, 0, 64, 0]
    words += opName(20, "_DirectConstants")
    words += opName(30, "_Push")
    words += [UInt32(4 << 16) | 59, 1, 20, 2] // Uniform OpVariable
    words += [UInt32(4 << 16) | 59, 2, 30, 9] // PushConstant OpVariable
    let spirv = words.withUnsafeBytes { Data($0) }
    let msl = """
    kernel void imb_compute_main(
        constant DirectConstants& _DirectConstants [[buffer(8)]],
        constant PushConstants& _Push [[buffer(11)]],
        uint3 gl_GlobalInvocationID [[thread_position_in_grid]]) {}
    """

    #expect(MetalGPUBackend.pushConstantBufferIndex(in: msl, spirv: spirv) == 11)

    let directOnlyWords = Array(words.dropLast(4))
    let directOnlySPIRV = directOnlyWords.withUnsafeBytes { Data($0) }
    #expect(MetalGPUBackend.pushConstantBufferIndex(in: msl, spirv: directOnlySPIRV) == nil)
}

@Test func serializedAtomic64SupportWrapsOnlyIndependentGlobalInvocations() throws {
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void imb_compute_main(
        device ulong* values [[buffer(0)]],
        uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
    {
        const ulong old = spvAtomicCompareExchange64(
            &values[gl_GlobalInvocationID.x],
            ulong(-1),
            ulong(gl_GlobalInvocationID.x));
        values[gl_GlobalInvocationID.x] = old == ulong(-1)
            ? ulong(gl_GlobalInvocationID.x)
            : old;
    }
    """
    let lowered = try SPIRVCrossCompiler.addSerializedAtomic64SupportIfNeeded(to: source)
    #expect(lowered.contains(SPIRVCrossCompiler.serializedAtomic64ExecutionMarker))
    #expect(lowered.contains("imbSerialGrid [[buffer(30)]]"))
    #expect(lowered.contains("for (uint imbSerialX = 0u;"))
    #expect(lowered.contains("const uint3 gl_GlobalInvocationID"))

    let coordinated = source.replacingOccurrences(
        of: "uint3 gl_GlobalInvocationID [[thread_position_in_grid]]",
        with: "uint3 gl_GlobalInvocationID [[thread_position_in_grid]], "
            + "uint3 localID [[thread_position_in_threadgroup]]"
    )
    #expect(throws: SPIRVCrossCompilerError.self) {
        try SPIRVCrossCompiler.addSerializedAtomic64SupportIfNeeded(to: coordinated)
    }
}

#if canImport(Metal)
import Metal

@Test func softwareFloat64SupportRunsOnTheDefaultMetalDevice() async throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    // Mentioning the patched SPIRV-Cross storage name enables the preamble.
    struct ProbeStorage { spvDouble value; };
    struct ProbeMatrixStorage { spvDouble4x4 matrix; spvDouble marker; };
    static_assert(sizeof(spvDouble4) == 32, "binary64 vector ABI changed");
    static_assert(sizeof(spvDouble4x4) == 128, "binary64 matrix ABI changed");
    static_assert(sizeof(ProbeMatrixStorage) == 136, "matrix member ABI changed");
    kernel void probeBinary64(
        device const ulong *input [[buffer(0)]],
        device float *decoded [[buffer(1)]],
        device ulong *encoded [[buffer(2)]],
        device ulong *packed [[buffer(3)]],
        uint index [[thread_position_in_grid]])
    {
        decoded[index] = spvDoubleToFloat(input[index]);
        encoded[index] = spvFloatToDouble(decoded[index]);
        uint2 halves = uint2(uint(input[index]), uint(input[index] >> 32));
        packed[index] = unsupported_GLSLstd450PackDouble2x32(halves).bits;
    }
    """
    let lowered = SPIRVCrossCompiler.addSoftwareFloat64SupportIfNeeded(to: source)
    #expect(lowered.contains(SPIRVCrossCompiler.softwareFloat64ExecutionRequiredMarker))
    let library = try await device.makeLibrary(source: lowered, options: nil)
    let function = try #require(library.makeFunction(name: "probeBinary64"))
    let pipeline = try await device.makeComputePipelineState(function: function)

    let values: [Double] = [1.5, -2.25, -0.0, .infinity]
    let byteCount = values.count * MemoryLayout<UInt64>.stride
    let input = try #require(device.makeBuffer(length: byteCount, options: .storageModeShared))
    let decoded = try #require(
        device.makeBuffer(
            length: values.count * MemoryLayout<Float>.stride,
            options: .storageModeShared
        )
    )
    let encoded = try #require(device.makeBuffer(length: byteCount, options: .storageModeShared))
    let packed = try #require(device.makeBuffer(length: byteCount, options: .storageModeShared))

    let inputWords = input.contents().bindMemory(to: UInt64.self, capacity: values.count)
    for (index, value) in values.enumerated() {
        inputWords[index] = value.bitPattern
    }

    let queue = try #require(device.makeCommandQueue())
    let commandBuffer = try #require(queue.makeCommandBuffer())
    let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(decoded, offset: 0, index: 1)
    encoder.setBuffer(encoded, offset: 0, index: 2)
    encoder.setBuffer(packed, offset: 0, index: 3)
    encoder.dispatchThreads(
        MTLSize(width: values.count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: values.count, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    await commandBuffer.completed()
    #expect(commandBuffer.status == .completed)

    let decodedValues = decoded.contents().bindMemory(to: Float.self, capacity: values.count)
    let encodedWords = encoded.contents().bindMemory(to: UInt64.self, capacity: values.count)
    let packedWords = packed.contents().bindMemory(to: UInt64.self, capacity: values.count)
    for (index, value) in values.enumerated() {
        #expect(decodedValues[index].bitPattern == Float(value).bitPattern)
        #expect(encodedWords[index] == Double(Float(value)).bitPattern)
        #expect(packedWords[index] == value.bitPattern)
    }
}

@Test func serializedAtomic64SupportRunsOnTheDefaultMetalDevice() async throws {
    let device = try #require(MTLCreateSystemDefaultDevice())
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void imb_compute_main(
        device ulong* values [[buffer(0)]],
        uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
    {
        const ulong old = spvAtomicCompareExchange64(
            &values[gl_GlobalInvocationID.x],
            ulong(-1),
            ulong(gl_GlobalInvocationID.x + 10u));
        values[gl_GlobalInvocationID.x] = old == ulong(-1)
            ? ulong(gl_GlobalInvocationID.x + 10u)
            : old;
    }
    """
    let lowered = try SPIRVCrossCompiler.addSerializedAtomic64SupportIfNeeded(to: source)
    let library = try await device.makeLibrary(source: lowered, options: nil)
    let function = try #require(library.makeFunction(name: "imb_compute_main"))
    let pipeline = try await device.makeComputePipelineState(function: function)
    let buffer = try #require(
        device.makeBuffer(
            length: 4 * MemoryLayout<UInt64>.stride,
            options: .storageModeShared
        )
    )
    let values = buffer.contents().bindMemory(to: UInt64.self, capacity: 4)
    for index in 0..<4 { values[index] = .max }
    let queue = try #require(device.makeCommandQueue())
    let commandBuffer = try #require(queue.makeCommandBuffer())
    let encoder = try #require(commandBuffer.makeComputeCommandEncoder())
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(buffer, offset: 0, index: 0)
    var grid = SIMD3<UInt32>(4, 1, 1)
    encoder.setBytes(
        &grid,
        length: MemoryLayout<SIMD3<UInt32>>.stride,
        index: SPIRVCrossCompiler.serializedAtomic64GridBufferIndex
    )
    encoder.dispatchThreads(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    await commandBuffer.completed()
    #expect(commandBuffer.status == .completed)
    #expect((0..<4).map { values[$0] } == [10, 11, 12, 13])
}
#endif
