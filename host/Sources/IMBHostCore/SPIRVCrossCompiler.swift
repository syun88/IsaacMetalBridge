import Foundation

public enum SPIRVCrossCompilerError: Error, CustomStringConvertible {
    case executableUnavailable(String)
    case processFailed(status: Int32, diagnostic: String)
    case invalidUTF8
    case unsupportedSerializedAtomic64(String)

    public var description: String {
        switch self {
        case .executableUnavailable(let path):
            "SPIRV-Cross executable is unavailable at \(path)"
        case .processFailed(let status, let diagnostic):
            "SPIRV-Cross exited with status \(status): \(diagnostic)"
        case .invalidUTF8:
            "SPIRV-Cross produced non-UTF-8 MSL"
        case .unsupportedSerializedAtomic64(let diagnostic):
            "serialized 64-bit atomic lowering is unavailable: \(diagnostic)"
        }
    }
}

struct SPIRVDescriptorBinding: Hashable, Sendable {
    let descriptorSet: UInt32
    let binding: UInt32
}

struct MetalArgumentBindingMap: Sendable {
    let indicesByVulkanBinding: [SPIRVDescriptorBinding: [Int]]
    let directBufferIndicesByVulkanBinding: [SPIRVDescriptorBinding: [Int]]
    let directTextureIndicesByVulkanBinding: [SPIRVDescriptorBinding: [Int]]
    let directSamplerIndicesByVulkanBinding: [SPIRVDescriptorBinding: [Int]]
    let emittedFieldCount: Int
    let mappedFieldCount: Int
}

/// Translates Vulkan SPIR-V into MSL suitable for Apple Silicon.
///
/// Isaac RTX uses large descriptor arrays, so all Vulkan descriptor sets are
/// emitted as tier-2 Metal argument buffers in device address space. Keeping
/// translation behind this small type also lets the runtime pin and bundle a
/// known SPIRV-Cross build instead of depending on a system-wide installation.
public final class SPIRVCrossCompiler: @unchecked Sendable {
    public let executableURL: URL

    public init(executableURL: URL) throws {
        guard FileManager.default.isExecutableFile(atPath: executableURL.path) else {
            throw SPIRVCrossCompilerError.executableUnavailable(executableURL.path)
        }
        self.executableURL = executableURL
    }

    public func translateToMSL(spirv: Data, computeEntryPoint: String? = nil) throws -> String {
        let process = Process()
        let input = Pipe()
        let output = Pipe()
        let diagnostics = Pipe()
        process.executableURL = executableURL
        var arguments = [
            "-",
            "--msl",
            "--msl-version", "31000",
            "--msl-argument-buffers",
            "--msl-argument-buffer-tier", "1",
            "--msl-runtime-array-rich-descriptor",
            "--msl-replace-recursive-inputs",
            // Vulkan texel buffers must remain native Metal texture_buffer
            // arguments. The default 4096-wide texture2d emulation has an
            // incompatible argument type for MTLBuffer.makeTexture views.
            "--msl-texture-buffer-native",
        ]
        for descriptorSet in 0..<32 {
            arguments += ["--msl-device-argument-buffer", String(descriptorSet)]
        }
        if let computeEntryPoint {
            arguments += [
                "--rename-entry-point", computeEntryPoint, "imb_compute_main", "comp",
                "--entry", "imb_compute_main",
                "--stage", "comp",
            ]
        }
        process.arguments = arguments
        process.standardInput = input
        process.standardOutput = output
        process.standardError = diagnostics

        try process.run()
        try input.fileHandleForWriting.write(contentsOf: spirv)
        try input.fileHandleForWriting.close()

        // Drain stdout before waiting because generated RTX MSL can exceed the
        // kernel pipe buffer by several megabytes.
        let translated = output.fileHandleForReading.readDataToEndOfFile()
        let diagnosticData = diagnostics.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationReason == .exit, process.terminationStatus == 0 else {
            let diagnostic = String(data: diagnosticData, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? "no diagnostic"
            throw SPIRVCrossCompilerError.processFailed(
                status: process.terminationStatus,
                diagnostic: diagnostic.isEmpty ? "no diagnostic" : diagnostic
            )
        }
        guard let source = String(data: translated, encoding: .utf8) else {
            throw SPIRVCrossCompilerError.invalidUTF8
        }
        let withSoftwareFloat64 = Self.addSoftwareFloat64SupportIfNeeded(to: source)
        return try Self.addSerializedAtomic64SupportIfNeeded(to: withSoftwareFloat64)
    }

    /// Correlates Vulkan descriptor set/binding decorations in SPIR-V with
    /// the compact `[[id(N)]]` values emitted inside each Metal argument
    /// buffer. SPIRV-Cross is free to compact sparse Vulkan bindings, so the
    /// Vulkan binding number cannot be passed directly to MTLArgumentEncoder.
    ///
    /// RTX shaders preserve OpName on descriptor variables. SPIRV-Cross uses
    /// that same name for the argument-buffer field, including overlapping
    /// aliases. One Vulkan binding can therefore map to more than one Metal
    /// argument ID when the shader aliases the descriptor with multiple types.
    static func metalArgumentBindingMap(spirv: Data, msl source: String) -> MetalArgumentBindingMap {
        guard spirv.count >= 20, spirv.count % MemoryLayout<UInt32>.size == 0 else {
            return MetalArgumentBindingMap(
                indicesByVulkanBinding: [:],
                directBufferIndicesByVulkanBinding: [:],
                directTextureIndicesByVulkanBinding: [:],
                directSamplerIndicesByVulkanBinding: [:],
                emittedFieldCount: 0,
                mappedFieldCount: 0
            )
        }

        let words: [UInt32] = spirv.withUnsafeBytes { bytes in
            let count = bytes.count / MemoryLayout<UInt32>.size
            return (0..<count).map {
                UInt32(littleEndian: bytes.loadUnaligned(
                    fromByteOffset: $0 * MemoryLayout<UInt32>.size,
                    as: UInt32.self
                ))
            }
        }
        guard words.first == 0x0723_0203 else {
            return MetalArgumentBindingMap(
                indicesByVulkanBinding: [:],
                directBufferIndicesByVulkanBinding: [:],
                directTextureIndicesByVulkanBinding: [:],
                directSamplerIndicesByVulkanBinding: [:],
                emittedFieldCount: 0,
                mappedFieldCount: 0
            )
        }

        var names: [UInt32: String] = [:]
        var descriptorSets: [UInt32: UInt32] = [:]
        var bindings: [UInt32: UInt32] = [:]
        var wordIndex = 5
        while wordIndex < words.count {
            let instruction = words[wordIndex]
            let wordCount = Int(instruction >> 16)
            let opcode = UInt16(instruction & 0xffff)
            guard wordCount > 0, wordIndex + wordCount <= words.count else { break }
            switch opcode {
            case 5 where wordCount >= 3: // OpName
                let target = words[wordIndex + 1]
                var bytes: [UInt8] = []
                nameLoop: for stringWord in words[(wordIndex + 2)..<(wordIndex + wordCount)] {
                    for shift in stride(from: 0, through: 24, by: 8) {
                        let byte = UInt8((stringWord >> UInt32(shift)) & 0xff)
                        if byte == 0 { break nameLoop }
                        bytes.append(byte)
                    }
                }
                if let name = String(bytes: bytes, encoding: .utf8), !name.isEmpty {
                    names[target] = name
                }
            case 71 where wordCount >= 4: // OpDecorate
                let target = words[wordIndex + 1]
                let decoration = words[wordIndex + 2]
                if decoration == 33 { // Binding
                    bindings[target] = words[wordIndex + 3]
                } else if decoration == 34 { // DescriptorSet
                    descriptorSets[target] = words[wordIndex + 3]
                }
            default:
                break
            }
            wordIndex += wordCount
        }

        var bindingsByName: [String: [SPIRVDescriptorBinding]] = [:]
        for (target, name) in names {
            guard let descriptorSet = descriptorSets[target], let binding = bindings[target] else {
                continue
            }
            bindingsByName[name, default: []].append(SPIRVDescriptorBinding(
                descriptorSet: descriptorSet,
                binding: binding
            ))
        }

        let setPattern = try? NSRegularExpression(
            pattern: #"^\s*struct\s+spvDescriptorSetBuffer([0-9]+)\s*$"#
        )
        let fieldPattern = try? NSRegularExpression(
            pattern: #"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[\[id\(([0-9]+)\)\]\]"#
        )
        var currentSet: UInt32?
        var emittedFieldCount = 0
        var mappedFieldCount = 0
        var result: [SPIRVDescriptorBinding: Set<Int>] = [:]

        func descriptorBindings(for fieldName: String, descriptorSet: UInt32?)
            -> [SPIRVDescriptorBinding]
        {
            var matches = bindingsByName[fieldName] ?? []
            if let descriptorSet {
                matches = matches.filter { $0.descriptorSet == descriptorSet }
            }
            if matches.isEmpty {
                // When a SPIR-V resource has only its generated numeric name
                // (for example `_10` for result ID 10), SPIRV-Cross makes the
                // MSL identifier legal by spelling it `m_10`.
                let generatedIDText: Substring?
                if fieldName.hasPrefix("m_") {
                    generatedIDText = fieldName.dropFirst(2)
                } else if fieldName.hasPrefix("_") {
                    generatedIDText = fieldName.dropFirst()
                } else {
                    generatedIDText = nil
                }
                if let generatedIDText,
                   let generatedID = UInt32(generatedIDText),
                   let generatedSet = descriptorSets[generatedID],
                   descriptorSet == nil || descriptorSet == generatedSet,
                   let binding = bindings[generatedID] {
                    matches = [SPIRVDescriptorBinding(
                        descriptorSet: generatedSet,
                        binding: binding
                    )]
                }
            }
            return matches
        }

        for rawLine in source.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(rawLine)
            let fullRange = NSRange(line.startIndex..<line.endIndex, in: line)
            if currentSet == nil,
               let match = setPattern?.firstMatch(in: line, range: fullRange),
               let range = Range(match.range(at: 1), in: line),
               let value = UInt32(line[range]) {
                currentSet = value
                continue
            }
            guard let descriptorSet = currentSet else { continue }
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("};") {
                currentSet = nil
                continue
            }
            guard let match = fieldPattern?.firstMatch(in: line, range: fullRange),
                  let nameRange = Range(match.range(at: 1), in: line),
                  let indexRange = Range(match.range(at: 2), in: line),
                  let metalIndex = Int(line[indexRange])
            else {
                continue
            }
            emittedFieldCount += 1
            let fieldName = String(line[nameRange])
            let matches = descriptorBindings(for: fieldName, descriptorSet: descriptorSet)
            guard matches.count == 1, let descriptorBinding = matches.first else { continue }
            mappedFieldCount += 1
            result[descriptorBinding, default: []].insert(metalIndex)
        }

        // Metal supports only eight argument-buffer slots. SPIRV-Cross emits
        // descriptors from higher Vulkan sets as direct kernel arguments and
        // aliases overlapping storage-buffer types through one raw pointer.
        // Track those direct buffer/texture/sampler indices alongside compact
        // argument-buffer IDs so the backend can bind every active resource.
        let kernelPattern = "kernel void imb_compute_main"
        let directPattern = try? NSRegularExpression(
            pattern: #"\b([A-Za-z_][A-Za-z0-9_]*)\s*\[\[(buffer|texture|sampler)\(([0-9]+)\)\]\]"#
        )
        let aliasPattern = try? NSRegularExpression(
            pattern: #"^spvBufferAliasSet([0-9]+)Binding([0-9]+)$"#
        )
        var directBuffers: [SPIRVDescriptorBinding: Set<Int>] = [:]
        var directTextures: [SPIRVDescriptorBinding: Set<Int>] = [:]
        var directSamplers: [SPIRVDescriptorBinding: Set<Int>] = [:]
        if let kernelStart = source.range(of: kernelPattern),
           let bodyStart = source[kernelStart.lowerBound...].firstIndex(of: "{") {
            let signature = String(source[kernelStart.lowerBound..<bodyStart])
            let signatureRange = NSRange(signature.startIndex..<signature.endIndex, in: signature)
            for match in directPattern?.matches(in: signature, range: signatureRange) ?? [] {
                guard let nameRange = Range(match.range(at: 1), in: signature),
                      let kindRange = Range(match.range(at: 2), in: signature),
                      let indexRange = Range(match.range(at: 3), in: signature),
                      let metalIndex = Int(signature[indexRange])
                else {
                    continue
                }
                let fieldName = String(signature[nameRange])
                var matches = descriptorBindings(for: fieldName, descriptorSet: nil)
                if matches.isEmpty,
                   let aliasMatch = aliasPattern?.firstMatch(
                       in: fieldName,
                       range: NSRange(fieldName.startIndex..<fieldName.endIndex, in: fieldName)
                   ),
                   let setRange = Range(aliasMatch.range(at: 1), in: fieldName),
                   let bindingRange = Range(aliasMatch.range(at: 2), in: fieldName),
                   let descriptorSet = UInt32(fieldName[setRange]),
                   let binding = UInt32(fieldName[bindingRange]) {
                    matches = [SPIRVDescriptorBinding(
                        descriptorSet: descriptorSet,
                        binding: binding
                    )]
                }
                // Descriptor-set argument buffers, push constants, builtins,
                // and the bridge's hidden serial-grid constant deliberately
                // have no original descriptor binding and are ignored here.
                guard matches.count == 1, let descriptorBinding = matches.first else { continue }
                emittedFieldCount += 1
                mappedFieldCount += 1
                switch String(signature[kindRange]) {
                case "buffer":
                    directBuffers[descriptorBinding, default: []].insert(metalIndex)
                case "texture":
                    directTextures[descriptorBinding, default: []].insert(metalIndex)
                case "sampler":
                    directSamplers[descriptorBinding, default: []].insert(metalIndex)
                default:
                    break
                }
            }
        }

        return MetalArgumentBindingMap(
            indicesByVulkanBinding: result.mapValues { $0.sorted() },
            directBufferIndicesByVulkanBinding: directBuffers.mapValues { $0.sorted() },
            directTextureIndicesByVulkanBinding: directTextures.mapValues { $0.sorted() },
            directSamplerIndicesByVulkanBinding: directSamplers.mapValues { $0.sorted() },
            emittedFieldCount: emittedFieldCount,
            mappedFieldCount: mappedFieldCount
        )
    }

    static let serializedAtomic64ExecutionMarker =
        "// IMB_SERIALIZED_ATOMIC64_EXECUTION_REQUIRED=1\n"
    static let serializedAtomic64GridBufferIndex = 30

    /// Metal has no 64-bit integer atomics, including in MSL 4.0 on Apple M4.
    /// A shader which uses only GlobalInvocationID and no threadgroup/subgroup
    /// coordination can nevertheless preserve Vulkan compare-exchange exactly
    /// by executing every logical invocation in a deterministic loop on one
    /// Metal thread. The patched SPIRV-Cross emits the helper call below only
    /// for 64-bit OpAtomicCompareExchange; all other 64-bit atomic operations
    /// remain rejected.
    static func addSerializedAtomic64SupportIfNeeded(to source: String) throws -> String {
        guard source.contains("spvAtomicCompareExchange64(") else { return source }
        guard !source.contains("thread_position_in_threadgroup"),
              !source.contains("threadgroup_position_in_grid"),
              !source.contains("simdgroup"),
              !source.contains("threadgroup_barrier")
        else {
            throw SPIRVCrossCompilerError.unsupportedSerializedAtomic64(
                "the shader uses threadgroup or subgroup execution state"
            )
        }
        let marker = "using namespace metal;\n"
        guard source.range(of: marker) != nil else {
            throw SPIRVCrossCompilerError.unsupportedSerializedAtomic64(
                "the Metal namespace marker is missing"
            )
        }
        let kernelToken = source.contains("kernel void imb_compute_main")
            ? "kernel void imb_compute_main"
            : "kernel void main0"
        guard let kernelStart = source.range(of: kernelToken),
              let bodyStart = source[kernelStart.lowerBound...].firstIndex(of: "{")
        else {
            throw SPIRVCrossCompilerError.unsupportedSerializedAtomic64(
                "the translated compute entry point is missing"
            )
        }
        let signatureRange = kernelStart.lowerBound..<bodyStart
        let signature = String(source[signatureRange])
        let globalIDParameter = "uint3 gl_GlobalInvocationID [[thread_position_in_grid]]"
        guard signature.components(separatedBy: globalIDParameter).count == 2 else {
            throw SPIRVCrossCompilerError.unsupportedSerializedAtomic64(
                "GlobalInvocationID is not a single uint3 kernel parameter"
            )
        }
        let hiddenBufferAttribute = "[[buffer(\(serializedAtomic64GridBufferIndex))]]"
        guard !signature.contains(hiddenBufferAttribute) else {
            throw SPIRVCrossCompilerError.unsupportedSerializedAtomic64(
                "Metal buffer index \(serializedAtomic64GridBufferIndex) is already occupied"
            )
        }

        var depth = 0
        var bodyEnd: String.Index?
        for index in source.indices[bodyStart...] {
            switch source[index] {
            case "{": depth += 1
            case "}":
                depth -= 1
                if depth == 0 {
                    bodyEnd = index
                    break
                }
            default: break
            }
            if bodyEnd != nil { break }
        }
        guard bodyEnd != nil else {
            throw SPIRVCrossCompilerError.unsupportedSerializedAtomic64(
                "the translated compute body is unbalanced"
            )
        }

        let replacementParameter = """
        uint3 imbSerialThreadID [[thread_position_in_grid]], constant uint3& imbSerialGrid \(hiddenBufferAttribute)
        """
        var serialized = source.replacingOccurrences(
            of: globalIDParameter,
            with: replacementParameter,
            range: signatureRange
        )
        // The replacement does not change any text before the opening brace,
        // so reacquire the kernel braces in the updated String before insertion.
        guard let updatedKernelStart = serialized.range(of: kernelToken),
              let updatedBodyStart = serialized[updatedKernelStart.lowerBound...].firstIndex(of: "{")
        else {
            throw SPIRVCrossCompilerError.unsupportedSerializedAtomic64(
                "the rewritten compute entry point is missing"
            )
        }
        depth = 0
        var updatedBodyEnd: String.Index?
        for index in serialized.indices[updatedBodyStart...] {
            switch serialized[index] {
            case "{": depth += 1
            case "}":
                depth -= 1
                if depth == 0 {
                    updatedBodyEnd = index
                    break
                }
            default: break
            }
            if updatedBodyEnd != nil { break }
        }
        guard let updatedBodyEnd else {
            throw SPIRVCrossCompilerError.unsupportedSerializedAtomic64(
                "the rewritten compute body is unbalanced"
            )
        }
        let loopPrefix = """

            if (any(imbSerialThreadID != uint3(0u)))
                return;
            for (uint imbSerialZ = 0u; imbSerialZ < imbSerialGrid.z; ++imbSerialZ)
            for (uint imbSerialY = 0u; imbSerialY < imbSerialGrid.y; ++imbSerialY)
            for (uint imbSerialX = 0u; imbSerialX < imbSerialGrid.x; ++imbSerialX)
            {
                const uint3 gl_GlobalInvocationID = uint3(imbSerialX, imbSerialY, imbSerialZ);
        """
        let loopSuffix = """
            }

        """
        serialized.insert(contentsOf: loopSuffix, at: updatedBodyEnd)
        serialized.insert(contentsOf: loopPrefix, at: serialized.index(after: updatedBodyStart))

        let helper = """
        \(serializedAtomic64ExecutionMarker)
        inline ulong spvAtomicCompareExchange64(
            device ulong* object,
            ulong expected,
            ulong desired)
        {
            const ulong observed = *object;
            if (observed == expected)
                *object = desired;
            return observed;
        }

        """
        guard let updatedMarkerRange = serialized.range(of: marker) else {
            throw SPIRVCrossCompilerError.unsupportedSerializedAtomic64(
                "the rewritten Metal namespace marker is missing"
            )
        }
        serialized.insert(contentsOf: helper, at: updatedMarkerRange.upperBound)
        return serialized
    }

    /// SPIR-V uses binary64 for several RTX world-position buffers, while
    /// Apple GPU families expose no native MSL `double`.  The pinned
    /// SPIRV-Cross patch names those values `spvDouble` without changing their
    /// 8-byte physical layout.  This preamble decodes the IEEE-754 payload to
    /// float for arithmetic and encodes it again for buffer stores.  It is an
    /// ABI-preserving precision lowering, not a reinterpretation of the low
    /// 32 bits as a float.
    static func addSoftwareFloat64SupportIfNeeded(to source: String) -> String {
        guard source.contains("spvDouble") else { return source }
        let marker = "using namespace metal;\n"
        guard source.range(of: marker) != nil else { return source }

        let numericLongFloatSuffix = try? NSRegularExpression(
            pattern: #"(?<![A-Za-z0-9_])([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)lf\b"#
        )
        let fullRange = NSRange(source.startIndex..<source.endIndex, in: source)
        let normalized = numericLongFloatSuffix?.stringByReplacingMatches(
            in: source,
            range: fullRange,
            withTemplate: "$1f"
        ) ?? source
        guard let normalizedMarkerRange = normalized.range(of: marker) else { return normalized }
        let executionMarker = requiresSoftwareFloat64Execution(in: normalized)
            ? softwareFloat64ExecutionRequiredMarker
            : softwareFloat64DeclarationOnlyMarker
        return normalized.replacingCharacters(
            in: normalizedMarkerRange,
            with: marker + executionMarker + Self.softwareFloat64MSL
        )
    }

    static let softwareFloat64ExecutionRequiredMarker =
        "// IMB_SOFTWARE_FP64_EXECUTION_REQUIRED=1\n"
    static let softwareFloat64DeclarationOnlyMarker =
        "// IMB_SOFTWARE_FP64_EXECUTION_REQUIRED=0\n"

    /// SPIRV-Cross must spell every member of a Vulkan uniform/storage struct,
    /// including members which the selected entry point never reads. Merely
    /// seeing `spvDouble4x4` in one of those declarations is therefore not a
    /// reason to suppress an otherwise ordinary float shader. Generated MSL
    /// uses balanced, brace-delimited struct blocks; ignore those declarations
    /// and conservatively retain any software-FP64 token in a constant, helper,
    /// signature, or kernel body as executable use.
    static func requiresSoftwareFloat64Execution(in source: String) -> Bool {
        var insideStruct = false
        var structBraceDepth = 0
        var sawStructOpeningBrace = false

        for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
            let text = String(line)
            let trimmed = text.trimmingCharacters(in: .whitespaces)
            if !insideStruct && trimmed.hasPrefix("struct ") {
                insideStruct = true
                structBraceDepth = 0
                sawStructOpeningBrace = false
            }

            if !insideStruct && text.contains("spvDouble") {
                return true
            }

            if insideStruct {
                let openingCount = text.reduce(into: 0) { count, character in
                    if character == "{" { count += 1 }
                }
                let closingCount = text.reduce(into: 0) { count, character in
                    if character == "}" { count += 1 }
                }
                if openingCount > 0 { sawStructOpeningBrace = true }
                structBraceDepth += openingCount - closingCount
                if sawStructOpeningBrace && structBraceDepth <= 0 {
                    insideStruct = false
                    structBraceDepth = 0
                }
            }
        }
        return false
    }

    private static let softwareFloat64MSL = #"""

// IsaacMetalBridge software binary64 storage support. Apple GPU hardware has
// no native double arithmetic, so values retain their Vulkan ABI in memory and
// are rounded to float only while an operation is evaluated.
inline float spvDoubleToFloat(ulong bits)
{
    uint sign = uint(bits >> 63);
    uint exponent = uint((bits >> 52) & 0x7fful);
    ulong fraction = bits & 0x000ffffffffffffful;
    float value;
    if (exponent == 0u)
    {
        // Every binary64 subnormal is below the binary32 representable range.
        value = 0.0f;
    }
    else if (exponent == 0x7ffu)
    {
        value = as_type<float>(fraction == 0ul ? 0x7f800000u : 0x7fc00000u);
    }
    else
    {
        int unbiased = int(exponent) - 1023;
        if (unbiased > 127)
            value = as_type<float>(0x7f800000u);
        else if (unbiased < -149)
            value = 0.0f;
        else
            value = ldexp(1.0f + float(fraction) * 0x1.0p-52f, unbiased);
    }
    return sign != 0u ? -value : value;
}

inline ulong spvFloatToDouble(float value)
{
    uint bits = as_type<uint>(value);
    ulong sign = ulong(bits >> 31) << 63;
    uint exponent = (bits >> 23) & 0xffu;
    uint fraction = bits & 0x007fffffu;
    if (exponent == 0xffu)
    {
        ulong payload = fraction == 0u ? 0ul : (ulong(fraction) << 29) | (1ul << 51);
        return sign | (0x7fful << 52) | payload;
    }
    if (exponent == 0u)
    {
        if (fraction == 0u)
            return sign;
        uint highest = 31u - clz(fraction);
        int unbiased = int(highest) - 149;
        uint leading = 1u << highest;
        ulong payload = ulong(fraction - leading) << (52u - highest);
        return sign | (ulong(unbiased + 1023) << 52) | payload;
    }
    return sign | (ulong(int(exponent) - 127 + 1023) << 52) | (ulong(fraction) << 29);
}

struct spvDouble
{
    ulong bits;

    spvDouble() thread : bits(0ul) {}
    spvDouble(float value) thread : bits(spvFloatToDouble(value)) {}
    spvDouble(int value) thread : bits(spvFloatToDouble(float(value))) {}
    spvDouble(uint value) thread : bits(spvFloatToDouble(float(value))) {}
    spvDouble(const device spvDouble& value) thread : bits(value.bits) {}
    spvDouble(const constant spvDouble& value) thread : bits(value.bits) {}
    operator float() const thread { return spvDoubleToFloat(bits); }
};

// SPIRV-Cross deliberately leaves GLSLstd450 PackDouble2x32 unsupported on
// native MSL.  For software binary64 the operation is only a bit assembly, so
// it can remain exact without requiring double arithmetic.
inline spvDouble unsupported_GLSLstd450PackDouble2x32(uint2 value)
{
    spvDouble result;
    result.bits = ulong(value.x) | (ulong(value.y) << 32);
    return result;
}

inline spvDouble operator +(spvDouble lhs, spvDouble rhs)
{
    return spvDouble(float(lhs) + float(rhs));
}
inline spvDouble operator -(spvDouble lhs, spvDouble rhs)
{
    return spvDouble(float(lhs) - float(rhs));
}
inline spvDouble operator *(spvDouble lhs, spvDouble rhs)
{
    return spvDouble(float(lhs) * float(rhs));
}
inline spvDouble operator /(spvDouble lhs, spvDouble rhs)
{
    return spvDouble(float(lhs) / float(rhs));
}
inline spvDouble operator +(float lhs, spvDouble rhs) { return spvDouble(lhs + float(rhs)); }
inline spvDouble operator +(spvDouble lhs, float rhs) { return spvDouble(float(lhs) + rhs); }
inline spvDouble operator -(float lhs, spvDouble rhs) { return spvDouble(lhs - float(rhs)); }
inline spvDouble operator -(spvDouble lhs, float rhs) { return spvDouble(float(lhs) - rhs); }
inline spvDouble operator *(float lhs, spvDouble rhs) { return spvDouble(lhs * float(rhs)); }
inline spvDouble operator *(spvDouble lhs, float rhs) { return spvDouble(float(lhs) * rhs); }
inline spvDouble operator /(float lhs, spvDouble rhs) { return spvDouble(lhs / float(rhs)); }
inline spvDouble operator /(spvDouble lhs, float rhs) { return spvDouble(float(lhs) / rhs); }
inline spvDouble operator -(spvDouble value)
{
    return spvDouble(-float(value));
}
inline bool operator ==(spvDouble lhs, spvDouble rhs) { return float(lhs) == float(rhs); }
inline bool operator !=(spvDouble lhs, spvDouble rhs) { return float(lhs) != float(rhs); }
inline bool operator <(spvDouble lhs, spvDouble rhs) { return float(lhs) < float(rhs); }
inline bool operator <=(spvDouble lhs, spvDouble rhs) { return float(lhs) <= float(rhs); }
inline bool operator >(spvDouble lhs, spvDouble rhs) { return float(lhs) > float(rhs); }
inline bool operator >=(spvDouble lhs, spvDouble rhs) { return float(lhs) >= float(rhs); }

struct spvDouble3
{
    spvDouble x;
    spvDouble y;
    spvDouble z;

    spvDouble3() thread : x(0.0f), y(0.0f), z(0.0f) {}
    spvDouble3(float value) thread : x(value), y(value), z(value) {}
    spvDouble3(float3 value) thread : x(value.x), y(value.y), z(value.z) {}
    spvDouble3(spvDouble xValue, spvDouble yValue, spvDouble zValue) thread
        : x(xValue), y(yValue), z(zValue) {}
    spvDouble3(const device spvDouble3& value) thread
        : x(value.x), y(value.y), z(value.z) {}
    spvDouble3(const constant spvDouble3& value) thread
        : x(value.x), y(value.y), z(value.z) {}
    operator float3() const thread { return float3(float(x), float(y), float(z)); }
};

typedef spvDouble3 packed_spvDouble3;

inline spvDouble3 operator +(spvDouble3 lhs, spvDouble3 rhs)
{
    return spvDouble3(float3(lhs) + float3(rhs));
}
inline spvDouble3 operator -(spvDouble3 lhs, spvDouble3 rhs)
{
    return spvDouble3(float3(lhs) - float3(rhs));
}
inline spvDouble3 operator *(spvDouble3 lhs, spvDouble rhs)
{
    return spvDouble3(float3(lhs) * float(rhs));
}
inline spvDouble3 operator *(spvDouble lhs, spvDouble3 rhs)
{
    return rhs * lhs;
}
inline spvDouble3 operator /(spvDouble3 lhs, spvDouble rhs)
{
    return spvDouble3(float3(lhs) / float(rhs));
}

// Four-component values occur both as arithmetic temporaries and as the
// 32-byte columns of a binary64 4x4 matrix.  The anonymous union supplies the
// vector-style `.xyz` read used by SPIRV-Cross while preserving four adjacent
// 8-byte IEEE-754 storage cells and scalar (8-byte) alignment.
struct spvDouble4
{
    union
    {
        struct
        {
            spvDouble x;
            spvDouble y;
            spvDouble z;
            spvDouble w;
        };
        struct
        {
            spvDouble3 xyz;
            spvDouble xyzPadding;
        };
    };

    spvDouble4() thread : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    spvDouble4(float value) thread : x(value), y(value), z(value), w(value) {}
    spvDouble4(float4 value) thread
        : x(value.x), y(value.y), z(value.z), w(value.w) {}
    spvDouble4(spvDouble xValue, spvDouble yValue, spvDouble zValue, spvDouble wValue) thread
        : x(xValue), y(yValue), z(zValue), w(wValue) {}
    spvDouble4(const device spvDouble4& value) thread
        : x(value.x), y(value.y), z(value.z), w(value.w) {}
    spvDouble4(const constant spvDouble4& value) thread
        : x(value.x), y(value.y), z(value.z), w(value.w) {}
    operator float4() const thread
    {
        return float4(float(x), float(y), float(z), float(w));
    }
};

struct spvDouble4x4
{
    spvDouble4 columns[4];
};

// SPIRV-Cross marks relaxed-layout float4 arrays as packed when an adjacent
// binary64 member lowers their containing struct alignment.  Metal template
// deduction does not consider packed_float4 and float4 the same T, so provide
// the element-wise ABI-safe copy that the generated store requires.
template<uint N>
inline void spvArrayCopyFromStackToDevice(
    device packed_float4 (&destination)[N],
    thread const float4 (&source)[N])
{
    for (uint index = 0; index < N; ++index)
        destination[index] = packed_float4(source[index]);
}

"""#
}
