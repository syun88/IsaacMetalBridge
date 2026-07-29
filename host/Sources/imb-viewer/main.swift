import AppKit
import CoreGraphics
import Foundation
import IMBHostCore

private enum ViewerError: Error, CustomStringConvertible {
    case usage

    var description: String {
        switch self {
        case .usage:
            "usage: imb-viewer --frame <PPM path> [--input <event path>] [--title <window title>]"
        }
    }
}

private struct ViewerArguments {
    let frameURL: URL
    let inputURL: URL?
    let title: String

    static func parse(_ values: [String]) throws -> ViewerArguments {
        var framePath: String?
        var inputPath: String?
        var title = "Isaac Sim 6.0.1 — Apple Metal Bridge"
        var index = 0
        while index < values.count {
            guard index + 1 < values.count else { throw ViewerError.usage }
            switch values[index] {
            case "--frame":
                framePath = values[index + 1]
            case "--input":
                inputPath = values[index + 1]
            case "--title":
                title = values[index + 1]
            default:
                throw ViewerError.usage
            }
            index += 2
        }
        guard let framePath, !framePath.isEmpty else { throw ViewerError.usage }
        return ViewerArguments(
            frameURL: URL(fileURLWithPath: framePath).standardizedFileURL,
            inputURL: inputPath.map { URL(fileURLWithPath: $0).standardizedFileURL },
            title: title
        )
    }
}

private final class InputEventFileWriter {
    private let handle: FileHandle
    private var sequence: UInt64 = 0

    init(url: URL) throws {
        let parent = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        guard FileManager.default.createFile(atPath: url.path, contents: Data()) else {
            throw CocoaError(.fileWriteUnknown)
        }
        handle = try FileHandle(forWritingTo: url)
    }

    deinit {
        try? handle.close()
    }

    func append(
        kind: ViewerInputKind,
        x: Float = 0,
        y: Float = 0,
        deltaX: Float = 0,
        deltaY: Float = 0,
        code: UInt32 = 0,
        modifiers: UInt32 = 0,
        targetWidth: UInt32 = 0,
        targetHeight: UInt32 = 0
    ) {
        sequence &+= 1
        let record = ViewerInputRecord(
            kind: kind,
            sequence: sequence,
            x: x,
            y: y,
            deltaX: deltaX,
            deltaY: deltaY,
            code: code,
            modifiers: modifiers,
            targetWidth: targetWidth,
            targetHeight: targetHeight
        )
        do {
            try handle.write(contentsOf: record.encoded())
        } catch {
            FileHandle.standardError.write(Data("imb-viewer: input write failed: \(error)\n".utf8))
        }
    }
}

@MainActor
private final class InteractiveImageView: NSImageView {
    private let inputWriter: InputEventFileWriter?
    private var tracking: NSTrackingArea?
    var targetPixelSize = NSSize(width: 1_440, height: 900)

    init(inputWriter: InputEventFileWriter?) {
        self.inputWriter = inputWriter
        super.init(frame: .zero)
    }

    required init?(coder: NSCoder) {
        nil
    }

    override var acceptsFirstResponder: Bool { true }

    override func updateTrackingAreas() {
        if let tracking {
            removeTrackingArea(tracking)
        }
        let newTracking = NSTrackingArea(
            rect: .zero,
            options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
            owner: self,
            userInfo: nil
        )
        addTrackingArea(newTracking)
        tracking = newTracking
        super.updateTrackingAreas()
    }

    override func mouseMoved(with event: NSEvent) {
        sendMouse(.mouseMove, event: event)
    }

    override func mouseDragged(with event: NSEvent) {
        sendMouse(.mouseMove, event: event)
    }

    override func rightMouseDragged(with event: NSEvent) {
        sendMouse(.mouseMove, event: event)
    }

    override func otherMouseDragged(with event: NSEvent) {
        sendMouse(.mouseMove, event: event)
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        sendMouse(.leftMouseDown, event: event)
    }

    override func mouseUp(with event: NSEvent) {
        sendMouse(.leftMouseUp, event: event)
    }

    override func rightMouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        sendMouse(.rightMouseDown, event: event)
    }

    override func rightMouseUp(with event: NSEvent) {
        sendMouse(.rightMouseUp, event: event)
    }

    override func otherMouseDown(with event: NSEvent) {
        guard event.buttonNumber == 2 else { return }
        window?.makeFirstResponder(self)
        sendMouse(.middleMouseDown, event: event)
    }

    override func otherMouseUp(with event: NSEvent) {
        guard event.buttonNumber == 2 else { return }
        sendMouse(.middleMouseUp, event: event)
    }

    override func scrollWheel(with event: NSEvent) {
        guard let point = targetPoint(for: event) else { return }
        inputWriter?.append(
            kind: .scroll,
            x: point.x,
            y: point.y,
            deltaX: Float(event.scrollingDeltaX),
            deltaY: Float(event.scrollingDeltaY),
            modifiers: modifierFlags(for: event),
            targetWidth: UInt32(targetPixelSize.width),
            targetHeight: UInt32(targetPixelSize.height)
        )
    }

    override func keyDown(with event: NSEvent) {
        inputWriter?.append(
            kind: .keyDown,
            code: UInt32(event.keyCode),
            modifiers: modifierFlags(for: event),
            targetWidth: UInt32(targetPixelSize.width),
            targetHeight: UInt32(targetPixelSize.height)
        )
        if !event.isARepeat, let characters = event.characters {
            for scalar in characters.unicodeScalars where !CharacterSet.controlCharacters.contains(scalar) {
                inputWriter?.append(
                    kind: .character,
                    code: scalar.value,
                    modifiers: modifierFlags(for: event),
                    targetWidth: UInt32(targetPixelSize.width),
                    targetHeight: UInt32(targetPixelSize.height)
                )
            }
        }
    }

    override func keyUp(with event: NSEvent) {
        inputWriter?.append(
            kind: .keyUp,
            code: UInt32(event.keyCode),
            modifiers: modifierFlags(for: event),
            targetWidth: UInt32(targetPixelSize.width),
            targetHeight: UInt32(targetPixelSize.height)
        )
    }

    private func sendMouse(_ kind: ViewerInputKind, event: NSEvent) {
        guard let point = targetPoint(for: event) else { return }
        inputWriter?.append(
            kind: kind,
            x: point.x,
            y: point.y,
            modifiers: modifierFlags(for: event),
            targetWidth: UInt32(targetPixelSize.width),
            targetHeight: UInt32(targetPixelSize.height)
        )
    }

    private func targetPoint(for event: NSEvent) -> (x: Float, y: Float)? {
        guard targetPixelSize.width > 0, targetPixelSize.height > 0 else { return nil }
        let point = convert(event.locationInWindow, from: nil)
        let scale = min(bounds.width / targetPixelSize.width, bounds.height / targetPixelSize.height)
        guard scale > 0 else { return nil }
        let imageSize = NSSize(
            width: targetPixelSize.width * scale,
            height: targetPixelSize.height * scale
        )
        let imageRect = NSRect(
            x: bounds.midX - imageSize.width / 2,
            y: bounds.midY - imageSize.height / 2,
            width: imageSize.width,
            height: imageSize.height
        )
        guard imageRect.contains(point) else { return nil }
        let x = (point.x - imageRect.minX) / scale
        let y = targetPixelSize.height - ((point.y - imageRect.minY) / scale)
        return (
            Float(min(max(x, 0), targetPixelSize.width - 1)),
            Float(min(max(y, 0), targetPixelSize.height - 1))
        )
    }

    private func modifierFlags(for event: NSEvent) -> UInt32 {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        var result: UInt32 = 0
        if flags.contains(.shift) { result |= 1 }
        if flags.contains(.control) { result |= 2 }
        if flags.contains(.option) { result |= 4 }
        if flags.contains(.command) { result |= 8 }
        return result
    }
}

private enum PPMDecoder {
    static func image(at url: URL) -> NSImage? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        var cursor = 0

        func skipSeparators() {
            while cursor < data.count {
                let byte = data[cursor]
                if byte == 35 {
                    while cursor < data.count, data[cursor] != 10 { cursor += 1 }
                } else if byte == 9 || byte == 10 || byte == 13 || byte == 32 {
                    cursor += 1
                } else {
                    return
                }
            }
        }

        func nextToken() -> String? {
            skipSeparators()
            let start = cursor
            while cursor < data.count {
                let byte = data[cursor]
                if byte == 9 || byte == 10 || byte == 13 || byte == 32 || byte == 35 { break }
                cursor += 1
            }
            guard cursor > start else { return nil }
            return String(data: data[start..<cursor], encoding: .ascii)
        }

        guard nextToken() == "P6",
              let widthToken = nextToken(), let width = Int(widthToken), width > 0,
              let heightToken = nextToken(), let height = Int(heightToken), height > 0,
              nextToken() == "255",
              cursor < data.count
        else {
            return nil
        }

        if data[cursor] == 13, cursor + 1 < data.count, data[cursor + 1] == 10 {
            cursor += 2
        } else if data[cursor] == 9 || data[cursor] == 10 || data[cursor] == 32 {
            cursor += 1
        } else {
            return nil
        }

        let pixelCount = width.multipliedReportingOverflow(by: height)
        guard !pixelCount.overflow else { return nil }
        let rgbByteCount = pixelCount.partialValue.multipliedReportingOverflow(by: 3)
        guard !rgbByteCount.overflow,
              cursor <= data.count,
              rgbByteCount.partialValue == data.count - cursor
        else {
            return nil
        }

        var rgba = Data(count: pixelCount.partialValue * 4)
        rgba.withUnsafeMutableBytes { destination in
            data.withUnsafeBytes { source in
                guard let destinationBase = destination.bindMemory(to: UInt8.self).baseAddress,
                      let sourceBase = source.bindMemory(to: UInt8.self).baseAddress
                else {
                    return
                }
                let pixels = sourceBase.advanced(by: cursor)
                for pixel in 0..<pixelCount.partialValue {
                    destinationBase[pixel * 4] = pixels[pixel * 3]
                    destinationBase[pixel * 4 + 1] = pixels[pixel * 3 + 1]
                    destinationBase[pixel * 4 + 2] = pixels[pixel * 3 + 2]
                    destinationBase[pixel * 4 + 3] = 255
                }
            }
        }

        guard let provider = CGDataProvider(data: rgba as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(
                  width: width,
                  height: height,
                  bitsPerComponent: 8,
                  bitsPerPixel: 32,
                  bytesPerRow: width * 4,
                  space: colorSpace,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                  provider: provider,
                  decode: nil,
                  shouldInterpolate: false,
                  intent: .defaultIntent
              )
        else {
            return nil
        }
        return NSImage(cgImage: image, size: NSSize(width: width, height: height))
    }
}

@MainActor
private final class ViewerAppDelegate: NSObject, NSApplicationDelegate {
    private let frameURL: URL
    private let windowTitle: String
    private let inputWriter: InputEventFileWriter?
    private var window: NSWindow?
    private var imageView: InteractiveImageView?
    private var placeholder: NSTextField?
    private var refreshTimer: Timer?
    private var lastSignature: String?

    init(frameURL: URL, inputURL: URL?, title: String) throws {
        self.frameURL = frameURL
        self.windowTitle = title
        inputWriter = try inputURL.map(InputEventFileWriter.init(url:))
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let visibleFrame = NSScreen.main?.visibleFrame
            ?? NSRect(x: 0, y: 0, width: 1_280, height: 800)
        var width = min(1_280, visibleFrame.width * 0.9)
        var height = width * 900 / 1_440
        if height > visibleFrame.height * 0.9 {
            height = visibleFrame.height * 0.9
            width = height * 1_440 / 900
        }

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: width, height: height),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = windowTitle
        window.minSize = NSSize(width: 720, height: 450)
        window.isReleasedWhenClosed = false
        window.backgroundColor = .black
        window.acceptsMouseMovedEvents = true
        window.center()

        let content = NSView()
        content.wantsLayer = true
        content.layer?.backgroundColor = NSColor.black.cgColor
        let imageView = InteractiveImageView(inputWriter: inputWriter)
        imageView.translatesAutoresizingMaskIntoConstraints = false
        imageView.imageAlignment = .alignCenter
        imageView.imageScaling = .scaleProportionallyUpOrDown
        imageView.animates = false
        let placeholder = NSTextField(labelWithString: "Starting the real Isaac Sim…")
        placeholder.translatesAutoresizingMaskIntoConstraints = false
        placeholder.textColor = .secondaryLabelColor
        placeholder.font = .systemFont(ofSize: 18, weight: .medium)

        content.addSubview(imageView)
        content.addSubview(placeholder)
        NSLayoutConstraint.activate([
            imageView.leadingAnchor.constraint(equalTo: content.leadingAnchor),
            imageView.trailingAnchor.constraint(equalTo: content.trailingAnchor),
            imageView.topAnchor.constraint(equalTo: content.topAnchor),
            imageView.bottomAnchor.constraint(equalTo: content.bottomAnchor),
            placeholder.centerXAnchor.constraint(equalTo: content.centerXAnchor),
            placeholder.centerYAnchor.constraint(equalTo: content.centerYAnchor),
        ])
        window.contentView = content
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        self.window = window
        self.imageView = imageView
        self.placeholder = placeholder
        window.makeFirstResponder(imageView)
        refreshFrame()
        refreshTimer = Timer.scheduledTimer(
            timeInterval: 1.0 / 20.0,
            target: self,
            selector: #selector(refreshFrame),
            userInfo: nil,
            repeats: true
        )
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    @objc private func refreshFrame() {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: frameURL.path),
              let modificationDate = attributes[.modificationDate] as? Date,
              let byteCount = attributes[.size] as? NSNumber
        else {
            return
        }
        let signature = "\(modificationDate.timeIntervalSinceReferenceDate):\(byteCount.uint64Value)"
        guard signature != lastSignature, let image = PPMDecoder.image(at: frameURL) else { return }
        lastSignature = signature
        imageView?.image = image
        imageView?.targetPixelSize = image.size
        placeholder?.isHidden = true
    }
}

do {
    let arguments = try ViewerArguments.parse(Array(CommandLine.arguments.dropFirst()))
    let application = NSApplication.shared
    application.setActivationPolicy(.regular)

    let mainMenu = NSMenu()
    let applicationItem = NSMenuItem()
    mainMenu.addItem(applicationItem)
    let applicationMenu = NSMenu()
    applicationMenu.addItem(
        withTitle: "Quit Isaac Sim Viewer",
        action: #selector(NSApplication.terminate(_:)),
        keyEquivalent: "q"
    )
    applicationItem.submenu = applicationMenu
    application.mainMenu = mainMenu

    let delegate = try ViewerAppDelegate(
        frameURL: arguments.frameURL,
        inputURL: arguments.inputURL,
        title: arguments.title
    )
    application.delegate = delegate
    application.run()
} catch {
    FileHandle.standardError.write(Data("imb-viewer: \(error)\n".utf8))
    exit(2)
}
