// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "IsaacMetalBridgeHost",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "IMBHostCore", targets: ["IMBHostCore"]),
        .executable(name: "imb-host", targets: ["imb-host"]),
        .executable(name: "imb-viewer", targets: ["imb-viewer"]),
        .executable(name: "imb-shader-probe", targets: ["imb-shader-probe"]),
    ],
    targets: [
        .target(
            name: "IMBHostCore",
            linkerSettings: [.linkedFramework("Metal")]
        ),
        .executableTarget(
            name: "imb-host",
            dependencies: ["IMBHostCore"]
        ),
        .executableTarget(
            name: "imb-viewer",
            dependencies: ["IMBHostCore"],
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("CoreGraphics"),
            ]
        ),
        .executableTarget(
            name: "imb-shader-probe",
            dependencies: ["IMBHostCore"],
            linkerSettings: [.linkedFramework("Metal")]
        ),
        .testTarget(
            name: "IMBHostCoreTests",
            dependencies: ["IMBHostCore"]
        ),
    ]
)
