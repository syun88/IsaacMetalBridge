import ContainerAPIClient
import Foundation
import IMBHostCore

enum ArgumentsError: Error, CustomStringConvertible {
    case usage(String)

    var description: String {
        switch self {
        case .usage(let message): message
        }
    }
}

struct Arguments {
    let containerID: String
    let vsockPort: UInt32
    let sessions: Int
    let waitForExit: Bool

    static func parse(_ values: [String]) throws -> Arguments {
        var containerID: String?
        var vsockPort: UInt32?
        var sessions = 1
        var waitForExit = false
        var index = 0
        while index < values.count {
            switch values[index] {
            case "--wait-exit":
                waitForExit = true
                index += 1
                continue
            case "--container":
                guard index + 1 < values.count else { throw usageError() }
                guard !values[index + 1].isEmpty else { throw usageError() }
                containerID = values[index + 1]
            case "--vsock-port":
                guard index + 1 < values.count else { throw usageError() }
                guard let port = UInt32(values[index + 1]), port > 0 else { throw usageError() }
                vsockPort = port
            case "--sessions":
                guard index + 1 < values.count else { throw usageError() }
                guard let count = Int(values[index + 1]), (1...64).contains(count) else { throw usageError() }
                sessions = count
            default:
                throw usageError()
            }
            index += 2
        }
        guard let containerID, let vsockPort else {
            throw usageError()
        }
        return Arguments(
            containerID: containerID,
            vsockPort: vsockPort,
            sessions: sessions,
            waitForExit: waitForExit
        )
    }

    private static func usageError() -> ArgumentsError {
        ArgumentsError.usage(
            "usage: imb-container-host --container <id> --vsock-port <1-4294967295> "
                + "[--sessions <1-64>] [--wait-exit]"
        )
    }
}

func log(_ message: String) {
    FileHandle.standardError.write(Data("imb-container-host: \(message)\n".utf8))
}

func isCleanProtocolDisconnect(_ error: any Error) -> Bool {
    var current = error as NSError
    for _ in 0..<4 {
        if current.domain == NSPOSIXErrorDomain,
           current.code == POSIXErrorCode.EPIPE.rawValue
            || current.code == POSIXErrorCode.ECONNRESET.rawValue
            || current.code == POSIXErrorCode.ENOTCONN.rawValue {
            return true
        }
        guard let underlying = current.userInfo[NSUnderlyingErrorKey] as? NSError else {
            break
        }
        current = underlying
    }
    return false
}

do {
    let arguments = try Arguments.parse(Array(CommandLine.arguments.dropFirst()))
    let metal = MetalCapabilities.detect()
    log(
        "Metal available=\(metal.available) "
            + "device=\(metal.deviceName.isEmpty ? "none" : metal.deviceName) "
            + "unifiedMemory=\(metal.unifiedMemory) "
            + "rayTracing=\(metal.rayTracing) "
            + "maxBufferLength=\(metal.maxBufferLength)"
    )
    let client = ContainerClient()
    let exitMonitor: Task<Int32, any Error>?
    if arguments.waitForExit {
        // `container run --detach` has already bootstrapped and started the init
        // process. Bootstrap is idempotent, so this obtains a ClientProcess
        // handle and registers the wait before a fast crash can be cleaned up.
        let process = try await client.bootstrap(
            id: arguments.containerID,
            stdio: [nil, nil, nil]
        )
        exitMonitor = Task {
            try await process.wait()
        }
    } else {
        exitMonitor = nil
    }
    var observedExitCode: Int32?
    sessionLoop: for sessionIndex in 1...arguments.sessions {
        let maximumAttempts = sessionIndex == 1 ? 1 : 200
        var stream: FileHandle?
        var lastDialError: (any Error)?
        for attempt in 1...maximumAttempts {
            do {
                log(
                    "dialing container=\(arguments.containerID) vsockPort=\(arguments.vsockPort) "
                        + "session=\(sessionIndex)/\(arguments.sessions) attempt=\(attempt)"
                )
                stream = try await client.dial(id: arguments.containerID, port: arguments.vsockPort)
                break
            } catch {
                lastDialError = error
                if attempt < maximumAttempts {
                    try await Task.sleep(nanoseconds: 100_000_000)
                }
            }
        }
        guard let stream else {
            throw lastDialError ?? ArgumentsError.usage("vsock dial failed without an error")
        }
        log("vsock connected session=\(sessionIndex)/\(arguments.sessions)")
        do {
            try BridgeStreamServer.serve(
                input: stream,
                output: stream,
                session: BridgeSession(),
                log: log
            )
        } catch BridgeStreamError.unexpectedEOF {
            // Kit's finite --/app/quitAfter path can close the guest vsock
            // while its Vulkan driver is partway through a final request.
            // With --wait-exit, the container init status is authoritative:
            // accept only a verified zero exit and continue to report crashes.
            guard let exitMonitor else {
                throw BridgeStreamError.unexpectedEOF
            }
            let exitCode = try await exitMonitor.value
            observedExitCode = exitCode
            if exitCode != 0 {
                log("container init exited status=\(exitCode) after protocol EOF")
                exit(exitCode)
            }
            log("protocol stream closed during clean container shutdown")
            log("session complete session=\(sessionIndex)/\(arguments.sessions)")
            break sessionLoop
        } catch {
            // A finite Kit run can close the guest endpoint after reading a
            // request but before the host writes its reply. Foundation wraps
            // that EPIPE/ECONNRESET in NSCocoaErrorDomain. Treat only those
            // transport disconnects as clean, and only after the container's
            // init process has independently reported exit status zero.
            guard isCleanProtocolDisconnect(error), let exitMonitor else {
                throw error
            }
            let exitCode = try await exitMonitor.value
            observedExitCode = exitCode
            if exitCode != 0 {
                log("container init exited status=\(exitCode) after protocol disconnect")
                exit(exitCode)
            }
            log("protocol reply raced clean container shutdown")
            log("session complete session=\(sessionIndex)/\(arguments.sessions)")
            break sessionLoop
        }
        log("session complete session=\(sessionIndex)/\(arguments.sessions)")
    }
    if let exitMonitor {
        let exitCode: Int32
        if let observedExitCode {
            exitCode = observedExitCode
        } else {
            exitCode = try await exitMonitor.value
        }
        log("container init exited status=\(exitCode)")
        if exitCode != 0 {
            exit(exitCode)
        }
    }
} catch {
    log("fatal: \(error)")
    exit(1)
}
