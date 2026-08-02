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
        return source
    }
}
