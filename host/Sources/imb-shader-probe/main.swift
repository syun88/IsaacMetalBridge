import Foundation
import IMBHostCore
import Metal

func firstLine(_ value: String) -> String {
    value.split(whereSeparator: { $0.isNewline }).first.map(String.init) ?? value
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

guard let device = MTLCreateSystemDefaultDevice() else {
    FileHandle.standardError.write(Data("imb-shader-probe: Metal is unavailable\n".utf8))
    exit(1)
}

do {
    let translator = try SPIRVCrossCompiler(executableURL: translatorURL)
    var shaders = try FileManager.default.contentsOfDirectory(
        at: shaderDirectory,
        includingPropertiesForKeys: nil
    ).filter { $0.pathExtension == "spv" }
    shaders.sort { $0.lastPathComponent < $1.lastPathComponent }
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
    exit(translationFailures == 0 && metalFailures == 0 ? 0 : 1)
} catch {
    FileHandle.standardError.write(Data("imb-shader-probe: \(error)\n".utf8))
    exit(1)
}
