// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "IsaacMetalBridgeContainerAdapter",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "imb-container-host", targets: ["imb-container-host"]),
    ],
    dependencies: [
        .package(path: ".."),
        .package(path: "../../third_party/apple-container"),
    ],
    targets: [
        .executableTarget(
            name: "imb-container-host",
            dependencies: [
                .product(name: "IMBHostCore", package: "host"),
                .product(name: "ContainerAPIClient", package: "apple-container"),
            ]
        ),
    ]
)
