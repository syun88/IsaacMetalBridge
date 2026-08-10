import Foundation

public enum SPIRVCrossCompilerError: Error, CustomStringConvertible {
    case executableUnavailable(String)
    case processFailed(status: Int32, diagnostic: String)
    case invalidUTF8

    public var description: String {
        switch self {
        case .executableUnavailable(let path):
            "SPIRV-Cross executable is unavailable at \(path)"
        case .processFailed(let status, let diagnostic):
            "SPIRV-Cross exited with status \(status): \(diagnostic)"
        case .invalidUTF8:
            "SPIRV-Cross produced non-UTF-8 MSL"
        }
    }
}

struct SPIRVDescriptorBinding: Hashable, Sendable {
    let descriptorSet: UInt32
    let binding: UInt32
}

struct MetalArgumentBindingMap: Sendable {
    let indicesByVulkanBinding: [SPIRVDescriptorBinding: [Int]]
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
        return Self.addSoftwareFloat64SupportIfNeeded(to: source)
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
            var matches = (bindingsByName[fieldName] ?? []).filter {
                $0.descriptorSet == descriptorSet
            }
            if matches.isEmpty {
                // When a SPIR-V resource has only its generated numeric name
                // (for example `_10` for result ID 10), SPIRV-Cross makes the
                // MSL identifier legal by spelling it `m_10`. Recover that
                // exact result ID from the original decorations instead of
                // dropping an otherwise valid argument-buffer field.
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
                   descriptorSets[generatedID] == descriptorSet,
                   let binding = bindings[generatedID] {
                    matches = [SPIRVDescriptorBinding(
                        descriptorSet: descriptorSet,
                        binding: binding
                    )]
                }
            }
            guard matches.count == 1, let descriptorBinding = matches.first else { continue }
            mappedFieldCount += 1
            result[descriptorBinding, default: []].insert(metalIndex)
        }

        return MetalArgumentBindingMap(
            indicesByVulkanBinding: result.mapValues { $0.sorted() },
            emittedFieldCount: emittedFieldCount,
            mappedFieldCount: mappedFieldCount
        )
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
