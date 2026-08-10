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
    """

    let map = SPIRVCrossCompiler.metalArgumentBindingMap(spirv: spirv, msl: msl)
    #expect(map.emittedFieldCount == 4)
    #expect(map.mappedFieldCount == 4)
    #expect(map.indicesByVulkanBinding[
        SPIRVDescriptorBinding(descriptorSet: 2, binding: 10)
    ] == [0])
    #expect(map.indicesByVulkanBinding[
        SPIRVDescriptorBinding(descriptorSet: 2, binding: 4)
    ] == [1, 2])
    #expect(map.indicesByVulkanBinding[
        SPIRVDescriptorBinding(descriptorSet: 0, binding: 0)
    ] == [0])
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
#endif
