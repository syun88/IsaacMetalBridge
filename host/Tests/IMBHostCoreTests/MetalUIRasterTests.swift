import Foundation
import Testing
@testable import IMBHostCore

#if canImport(Metal)
private func appendFloat(_ value: Float, to data: inout Data) {
    data.appendLittleEndian(value.bitPattern)
}

private func appendUIVertex(
    x: Float,
    y: Float,
    u: Float = 0,
    v: Float = 0,
    rgba8: UInt32,
    to data: inout Data
) {
    appendFloat(x, to: &data)
    appendFloat(y, to: &data)
    appendFloat(u, to: &data)
    appendFloat(v, to: &data)
    data.appendLittleEndian(rgba8)
}

@Test func metalOrdinary3DImageRoundTripsEveryDepthSlice() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    let options = try #require(ImageOption.encodedTexture3D(depth: 2))
    try backend.createImage(
        id: 200,
        width: 2,
        height: 2,
        format: 1,
        options: options
    )
    let voxels = Data([
        255, 0, 0, 255, 0, 255, 0, 255,
        0, 0, 255, 255, 255, 255, 0, 255,
        16, 32, 48, 64, 80, 96, 112, 128,
        144, 160, 176, 192, 208, 224, 240, 255,
    ])
    try backend.writeImage(id: 200, data: voxels)
    #expect(try backend.readImage(id: 200) == voxels)
}

@Test func metalIndexedUIFillsAKnownRectangle() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    try backend.createBuffer(id: 1, size: 80, options: 0)
    try backend.createBuffer(id: 2, size: 24, options: 0)
    try backend.createImage(id: 3, width: 100, height: 100, format: 2, options: 0)

    var vertices = Data()
    appendUIVertex(x: 10, y: 10, rgba8: 0xff00_00ff, to: &vertices)
    appendUIVertex(x: 90, y: 10, rgba8: 0xff00_00ff, to: &vertices)
    appendUIVertex(x: 90, y: 90, rgba8: 0xff00_00ff, to: &vertices)
    appendUIVertex(x: 10, y: 90, rgba8: 0xff00_00ff, to: &vertices)
    try backend.writeBuffer(id: 1, offset: 0, data: vertices)

    var indices = Data()
    for index: UInt32 in [0, 1, 2, 2, 3, 0] {
        indices.appendLittleEndian(index)
    }
    try backend.writeBuffer(id: 2, offset: 0, data: indices)

    try backend.submitIndexedUI(
        imageID: 3,
        vertexBufferID: 1,
        indexBufferID: 2,
        vertexBufferOffset: 0,
        indexBufferOffset: 0,
        width: 100,
        height: 100,
        clearRGBA8: 0xff00_0000,
        draws: [IndexedUIDraw(
            textureID: 0,
            indexCount: 6,
            firstIndex: 0,
            vertexOffset: 0,
            scissorX: 0,
            scissorY: 0,
            scissorWidth: 100,
            scissorHeight: 100
        )],
        fenceID: 1
    )
    #expect(try backend.waitFence(id: 1))

    let pixels = try backend.readImage(id: 3)
    let center = (50 * 100 + 50) * 4
    let corner = 4
    #expect(Array(pixels[center..<(center + 4)]) == [0, 0, 255, 255])
    #expect(Array(pixels[corner..<(corner + 4)]) == [0, 0, 0, 255])
}

@Test func metalIndexedUIHandlesIsaacRingBufferOffsets() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    let ringSize: UInt64 = 33_554_432
    let vertexBindingOffset: UInt64 = 23_568
    let firstIndex: UInt32 = 48
    let vertexOffset: Int32 = 32
    try backend.createBuffer(id: 1, size: ringSize, options: 0)
    try backend.createImage(id: 2, width: 1_440, height: 900, format: 2, options: 0)

    var vertices = Data()
    appendUIVertex(x: 0, y: 0, rgba8: 0xff24_2424, to: &vertices)
    appendUIVertex(x: 1_440, y: 0, rgba8: 0xff24_2424, to: &vertices)
    appendUIVertex(x: 1_440, y: 900, rgba8: 0xff24_2424, to: &vertices)
    appendUIVertex(x: 0, y: 900, rgba8: 0xff24_2424, to: &vertices)
    let vertexWriteOffset = vertexBindingOffset + UInt64(vertexOffset) * 20
    try backend.writeBuffer(id: 1, offset: vertexWriteOffset, data: vertices)

    var indices = Data()
    for index: UInt32 in [0, 1, 2, 2, 3, 0] {
        indices.appendLittleEndian(index)
    }
    try backend.writeBuffer(id: 1, offset: UInt64(firstIndex) * 4, data: indices)

    try backend.submitIndexedUI(
        imageID: 2,
        vertexBufferID: 1,
        indexBufferID: 1,
        vertexBufferOffset: vertexBindingOffset,
        indexBufferOffset: 0,
        width: 1_440,
        height: 900,
        clearRGBA8: 0xff45_4545,
        draws: [IndexedUIDraw(
            textureID: 0,
            indexCount: 6,
            firstIndex: firstIndex,
            vertexOffset: vertexOffset,
            scissorX: 0,
            scissorY: 0,
            scissorWidth: 1_440,
            scissorHeight: 900
        )],
        fenceID: 1
    )
    #expect(try backend.waitFence(id: 1))

    let pixels = try backend.readImage(id: 2)
    let center = (450 * 1_440 + 720) * 4
    #expect(Array(pixels[center..<(center + 4)]) == [36, 36, 36, 255])
}

@Test func metalBuildsPrimitiveTriangleAccelerationStructure() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    #expect(backend.supportsAccelerationStructures)
    try backend.createBuffer(id: 1, size: 36, options: 0)

    var vertices = Data()
    appendFloat(-1, to: &vertices)
    appendFloat(-1, to: &vertices)
    appendFloat(0, to: &vertices)
    appendFloat(1, to: &vertices)
    appendFloat(-1, to: &vertices)
    appendFloat(0, to: &vertices)
    appendFloat(0, to: &vertices)
    appendFloat(1, to: &vertices)
    appendFloat(0, to: &vertices)
    try backend.writeBuffer(id: 1, offset: 0, data: vertices)

    try backend.createAccelerationStructure(id: 2, type: 1, requestedSize: 4096)
    try backend.buildPrimitiveAccelerationStructure(
        id: 2,
        buildFlags: 0x4,
        geometries: [PrimitiveAccelerationStructureGeometry(
            kind: .triangles,
            flags: 1,
            dataResourceID: 1,
            dataOffset: 0,
            primitiveCount: 1,
            stride: 12,
            vertexFormat: 1
        )]
    )
    try backend.createAccelerationStructure(id: 3, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 3,
        buildFlags: 0x4,
        instances: [InstanceAccelerationStructureInstance(
            transformationMatrix: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
            ],
            options: 0,
            mask: 0xff,
            intersectionFunctionTableOffset: 0,
            // Bit 23 marks a scene-state material color; low RGB fields encode
            // half red (R8, G8, B7) for the Metal intersection shader. Keep
            // headroom so this probe can distinguish direct lighting without
            // relying on the old non-physical distance/instance tint.
            userID: 0x0080_0080,
            accelerationStructureResourceID: 2
        )]
    )
    #expect(backend.supportsRayDispatch)
    try backend.createImage(id: 4, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 4,
        accelerationStructureID: 3,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xff00_ff00,
        camera: nil,
        sphereLight: nil,
        distantLight: nil,
        domeLight: nil,
        fenceID: 9
    )
    #expect(try backend.waitFence(id: 9))
    let rayPixels = try backend.readImage(id: 4)
    let center = (32 * 64 + 32) * 4
    let corner = 0
    #expect(Array(rayPixels[center..<(center + 4)]) == [0, 255, 0, 255])
    #expect(Array(rayPixels[corner..<(corner + 4)]) == [0, 0, 0, 255])

    try backend.createImage(id: 5, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 5,
        accelerationStructureID: 3,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: RayCamera(
            position: SIMD3<Float>(0, 0, -2),
            forward: SIMD3<Float>(0, 0, 1),
            up: SIMD3<Float>(0, 1, 0),
            verticalFOVRadians: 0.9,
            nearDistance: 0.01,
            farDistance: 100
        ),
        sphereLight: RaySphereLight(
            position: SIMD3<Float>(0, 0, -1),
            color: SIMD3<Float>(1, 1, 1),
            intensity: 100_000,
            radius: 0.25
        ),
        distantLight: RayDistantLight(
            direction: SIMD3<Float>(0, 0, 1),
            color: SIMD3<Float>(1, 0.85, 0.65),
            intensity: 2_500,
            angleDegrees: 1
        ),
        domeLight: RayDomeLight(
            color: SIMD3<Float>(0.28, 0.36, 0.5),
            intensity: 400
        ),
        fenceID: 10
    )
    #expect(try backend.waitFence(id: 10))
    let litPixels = try backend.readImage(id: 5)
    let litCenter = Array(litPixels[center..<(center + 4)])
    #expect(litCenter != Array(rayPixels[center..<(center + 4)]))
    #expect(litCenter[0] >= litCenter[1])
    #expect(litCenter[0] >= litCenter[2])
    #expect(litCenter[0] > 64)
    #expect(litCenter[3] == 255)

    let liveCamera = RayCamera(
        position: SIMD3<Float>(0, 0, -2),
        forward: SIMD3<Float>(0, 0, 1),
        up: SIMD3<Float>(0, 1, 0),
        verticalFOVRadians: 0.9,
        nearDistance: 0.01,
        farDistance: 100
    )
    try backend.createImage(id: 6, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 6,
        accelerationStructureID: 3,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: liveCamera,
        sphereLight: nil,
        distantLight: nil,
        domeLight: nil,
        fenceID: 11
    )
    #expect(try backend.waitFence(id: 11))
    let unlitCenter = Array(try backend.readImage(id: 6)[center..<(center + 4)])
    #expect(unlitCenter[0] > unlitCenter[1])
    #expect(unlitCenter[0] > unlitCenter[2])

    // Inline scene colors share Metal's 24-bit user ID with a two-bit kind
    // marker.  Use a neutral 0.7 gray whose blue component sets the old
    // collision bit: it must remain an inline color instead of being mistaken
    // for a material-texture descriptor and falling back to cyan.
    try backend.createAccelerationStructure(id: 22, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 22,
        buildFlags: 0x4,
        instances: [InstanceAccelerationStructureInstance(
            transformationMatrix: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
            ],
            options: 0,
            mask: 0xff,
            intersectionFunctionTableOffset: 0,
            // R8=179, G8=179, B6=44 with marker 10 in bits 23...22.
            userID: 0x00ac_b3b3,
            accelerationStructureResourceID: 2
        )]
    )
    try backend.createImage(id: 23, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 23,
        accelerationStructureID: 22,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: liveCamera,
        sphereLight: nil,
        distantLight: nil,
        domeLight: nil,
        fenceID: 15
    )
    #expect(try backend.waitFence(id: 15))
    let neutralCenter = Array(
        try backend.readImage(id: 23)[center..<(center + 4)]
    )
    #expect(abs(Int(neutralCenter[0]) - Int(neutralCenter[1])) <= 2)
    #expect(abs(Int(neutralCenter[1]) - Int(neutralCenter[2])) <= 2)
    #expect(neutralCenter[0] > 180)
    #expect(neutralCenter[3] == 255)

    try backend.createImage(id: 7, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 7,
        accelerationStructureID: 3,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: liveCamera,
        sphereLight: nil,
        distantLight: RayDistantLight(
            direction: SIMD3<Float>(0, 0, 1),
            color: SIMD3<Float>(1, 0.93, 0.82),
            intensity: 2_500,
            angleDegrees: 1
        ),
        domeLight: nil,
        fenceID: 12
    )
    #expect(try backend.waitFence(id: 12))
    let distantCenter = Array(try backend.readImage(id: 7)[center..<(center + 4)])
    #expect(distantCenter != unlitCenter)
    #expect(distantCenter[0] > unlitCenter[0])

    try backend.createImage(id: 8, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 8,
        accelerationStructureID: 3,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: liveCamera,
        sphereLight: nil,
        distantLight: nil,
        domeLight: RayDomeLight(
            color: SIMD3<Float>(0.28, 0.36, 0.5),
            intensity: 400
        ),
        fenceID: 13
    )
    #expect(try backend.waitFence(id: 13))
    let domeCenter = Array(try backend.readImage(id: 8)[center..<(center + 4)])
    #expect(domeCenter != unlitCenter)
    #expect(domeCenter != distantCenter)

    // Rotate the same BLAS 60 degrees around Y. The center ray still hits the
    // triangle at the origin, but a real transformed face normal receives
    // only half as much of the Z-aligned distant light. A view-facing fallback
    // would incorrectly produce the same brightness as the identity instance.
    let cosine: Float = 0.5
    let sine: Float = 0.866_025_4
    try backend.createAccelerationStructure(id: 20, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 20,
        buildFlags: 0x4,
        instances: [InstanceAccelerationStructureInstance(
            transformationMatrix: [
                cosine, 0, sine, 0,
                0, 1, 0, 0,
                -sine, 0, cosine, 0,
            ],
            options: 0,
            mask: 0xff,
            intersectionFunctionTableOffset: 0,
            userID: 0x0080_0080,
            accelerationStructureResourceID: 2
        )]
    )
    try backend.createImage(id: 21, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 21,
        accelerationStructureID: 20,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: liveCamera,
        sphereLight: nil,
        distantLight: RayDistantLight(
            direction: SIMD3<Float>(0, 0, 1),
            color: SIMD3<Float>(1, 0.93, 0.82),
            intensity: 2_500,
            angleDegrees: 1
        ),
        domeLight: nil,
        fenceID: 14
    )
    #expect(try backend.waitFence(id: 14))
    let rotatedCenter = Array(try backend.readImage(id: 21)[center..<(center + 4)])
    #expect(rotatedCenter[0] > 0)
    #expect(rotatedCenter[0] < distantCenter[0])
    try backend.destroyAccelerationStructure(id: 22)
    try backend.destroyAccelerationStructure(id: 20)
    try backend.destroyAccelerationStructure(id: 3)
    try backend.destroyAccelerationStructure(id: 2)
}

@Test func metalRayLightingInterpolatesAuthoredCornerNormals() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    #expect(backend.supportsRayDispatch)

    // vertexFormat=2 interleaves float3 position and float3 authored normal.
    // The world-space left and top corners face the -Z light, while the
    // world-space right corner faces +X. The camera's right basis is -X here,
    // so correct barycentric interpolation makes screen-left measurably darker.
    try backend.createBuffer(id: 30, size: 72, options: 0)
    var vertices = Data()
    for (position, normal): (SIMD3<Float>, SIMD3<Float>) in [
        (SIMD3<Float>(-1, -1, 0), SIMD3<Float>(0, 0, -1)),
        (SIMD3<Float>(1, -1, 0), SIMD3<Float>(1, 0, 0)),
        (SIMD3<Float>(0, 1, 0), SIMD3<Float>(0, 0, -1)),
    ] {
        appendFloat(position.x, to: &vertices)
        appendFloat(position.y, to: &vertices)
        appendFloat(position.z, to: &vertices)
        appendFloat(normal.x, to: &vertices)
        appendFloat(normal.y, to: &vertices)
        appendFloat(normal.z, to: &vertices)
    }
    try backend.writeBuffer(id: 30, offset: 0, data: vertices)

    try backend.createAccelerationStructure(id: 31, type: 1, requestedSize: 4096)
    try backend.buildPrimitiveAccelerationStructure(
        id: 31,
        buildFlags: 0x4,
        geometries: [PrimitiveAccelerationStructureGeometry(
            kind: .triangles,
            flags: 1,
            dataResourceID: 30,
            dataOffset: 0,
            primitiveCount: 1,
            stride: 24,
            vertexFormat: 2
        )]
    )
    try backend.createAccelerationStructure(id: 32, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 32,
        buildFlags: 0x4,
        instances: [InstanceAccelerationStructureInstance(
            transformationMatrix: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
            ],
            options: 0,
            mask: 0xff,
            intersectionFunctionTableOffset: 0,
            userID: 0x0080_00ff,
            accelerationStructureResourceID: 31
        )]
    )

    try backend.createImage(id: 33, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 33,
        accelerationStructureID: 32,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: RayCamera(
            position: SIMD3<Float>(0, 0, -2),
            forward: SIMD3<Float>(0, 0, 1),
            up: SIMD3<Float>(0, 1, 0),
            verticalFOVRadians: 0.9,
            nearDistance: 0.01,
            farDistance: 100
        ),
        sphereLight: nil,
        distantLight: RayDistantLight(
            direction: SIMD3<Float>(0, 0, 1),
            color: SIMD3<Float>(1, 1, 1),
            intensity: 1,
            angleDegrees: 0
        ),
        domeLight: nil,
        fenceID: 34
    )
    #expect(try backend.waitFence(id: 34))

    let pixels = try backend.readImage(id: 33)
    let leftOffset = (38 * 64 + 14) * 4
    let rightOffset = (38 * 64 + 49) * 4
    let left = Array(pixels[leftOffset..<(leftOffset + 4)])
    let right = Array(pixels[rightOffset..<(rightOffset + 4)])
    #expect(left[0] > 0)
    #expect(right[0] > 0)
    #expect(left[0] < right[0])
}

@Test func metalRayShadingSamplesFileTextureUVs() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    #expect(backend.supportsRayDispatch)

    try backend.createImage(id: 40, width: 2, height: 2, format: 1, options: 0)
    try backend.writeImage(id: 40, data: Data([
        255, 0, 0, 255, 0, 255, 0, 255,
        255, 0, 0, 255, 0, 255, 0, 255,
    ]))

    // vertexFormat=3 is float3 position + float3 normal + float2 UV.
    try backend.createBuffer(id: 41, size: 96, options: 0)
    var vertices = Data()
    for (position, uv): (SIMD3<Float>, SIMD2<Float>) in [
        (SIMD3<Float>(-1, -1, 0), SIMD2<Float>(0, 0)),
        (SIMD3<Float>(1, -1, 0), SIMD2<Float>(1, 0)),
        (SIMD3<Float>(0, 1, 0), SIMD2<Float>(0.5, 1)),
    ] {
        for value in [position.x, position.y, position.z, 0, 0, -1, uv.x, uv.y] {
            appendFloat(value, to: &vertices)
        }
    }
    try backend.writeBuffer(id: 41, offset: 0, data: vertices)
    try backend.createAccelerationStructure(id: 42, type: 1, requestedSize: 4096)
    try backend.buildPrimitiveAccelerationStructure(
        id: 42,
        buildFlags: 0x4,
        geometries: [PrimitiveAccelerationStructureGeometry(
            kind: .triangles,
            flags: 1,
            dataResourceID: 41,
            dataOffset: 0,
            primitiveCount: 1,
            stride: 32,
            vertexFormat: 3
        )]
    )
    try backend.createAccelerationStructure(id: 43, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 43,
        buildFlags: 0x4,
        instances: [InstanceAccelerationStructureInstance(
            transformationMatrix: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
            ],
            options: 0,
            mask: 0xff,
            intersectionFunctionTableOffset: 0,
            userID: 0x0040_0000 | 40,
            accelerationStructureResourceID: 42
        )]
    )
    try backend.createImage(id: 44, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 44,
        accelerationStructureID: 43,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: RayCamera(
            position: SIMD3<Float>(0, 0, -2),
            forward: SIMD3<Float>(0, 0, 1),
            up: SIMD3<Float>(0, 1, 0),
            verticalFOVRadians: 0.9,
            nearDistance: 0.01,
            farDistance: 100
        ),
        sphereLight: nil,
        distantLight: nil,
        domeLight: nil,
        fenceID: 45
    )
    #expect(try backend.waitFence(id: 45))
    let pixels = try backend.readImage(id: 44)
    let screenLeftOffset = (38 * 64 + 14) * 4
    let screenRightOffset = (38 * 64 + 49) * 4
    let screenLeft = Array(pixels[screenLeftOffset..<(screenLeftOffset + 4)])
    let screenRight = Array(pixels[screenRightOffset..<(screenRightOffset + 4)])
    // The camera right basis is -X, so screen-left samples the green texel at
    // high U and screen-right samples the red texel at low U.
    #expect(screenLeft[1] > screenLeft[0])
    #expect(screenRight[0] > screenRight[1])
}

@Test func metalRayShadingSkipsTaggedAlphaCutoutHit() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    #expect(backend.supportsRayDispatch)

    // Transparent red front material. MBM1 v2 flag bit 5 marks only this
    // material as a binary 0.3333 alpha cutout.
    try backend.createImage(id: 110, width: 1, height: 1, format: 1, options: 0)
    try backend.writeImage(id: 110, data: Data([255, 0, 0, 0]))
    try backend.createBuffer(id: 111, size: 96, options: 0)
    var frontVertices = Data()
    for position: SIMD3<Float> in [
        SIMD3<Float>(-1, -1, 0),
        SIMD3<Float>(1, -1, 0),
        SIMD3<Float>(0, 1, 0),
    ] {
        for value in [
            position.x, position.y, position.z,
            0, 0, -1,
            0.5, 0.5,
        ] {
            appendFloat(value, to: &frontVertices)
        }
    }
    try backend.writeBuffer(id: 111, offset: 0, data: frontVertices)
    try backend.createAccelerationStructure(id: 112, type: 1, requestedSize: 4096)
    try backend.buildPrimitiveAccelerationStructure(
        id: 112,
        buildFlags: 0x4,
        geometries: [PrimitiveAccelerationStructureGeometry(
            kind: .triangles,
            flags: 1,
            dataResourceID: 111,
            dataOffset: 0,
            primitiveCount: 1,
            stride: 32,
            vertexFormat: 3
        )]
    )
    try backend.createBuffer(id: 113, size: 56, options: 0)
    var cutoutDescriptor = Data()
    for word: UInt32 in [
        0x314d_424d, 2, 0x21, 0,
        110, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
    ] {
        cutoutDescriptor.appendLittleEndian(word)
    }
    try backend.writeBuffer(id: 113, offset: 0, data: cutoutDescriptor)

    // Opaque green triangle behind the transparent front triangle.
    try backend.createBuffer(id: 114, size: 36, options: 0)
    var backVertices = Data()
    for position: SIMD3<Float> in [
        SIMD3<Float>(-1, -1, 0.25),
        SIMD3<Float>(1, -1, 0.25),
        SIMD3<Float>(0, 1, 0.25),
    ] {
        for value in [position.x, position.y, position.z] {
            appendFloat(value, to: &backVertices)
        }
    }
    try backend.writeBuffer(id: 114, offset: 0, data: backVertices)
    try backend.createAccelerationStructure(id: 115, type: 1, requestedSize: 4096)
    try backend.buildPrimitiveAccelerationStructure(
        id: 115,
        buildFlags: 0x4,
        geometries: [PrimitiveAccelerationStructureGeometry(
            kind: .triangles,
            flags: 1,
            dataResourceID: 114,
            dataOffset: 0,
            primitiveCount: 1,
            stride: 12,
            vertexFormat: 1
        )]
    )

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    ]
    try backend.createAccelerationStructure(id: 116, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 116,
        buildFlags: 0x4,
        instances: [
            InstanceAccelerationStructureInstance(
                transformationMatrix: identity,
                options: 0,
                mask: 0xff,
                intersectionFunctionTableOffset: 0,
                userID: 0x00c0_0000 | 113,
                accelerationStructureResourceID: 112
            ),
            InstanceAccelerationStructureInstance(
                transformationMatrix: identity,
                options: 0,
                mask: 0xff,
                intersectionFunctionTableOffset: 0,
                userID: 0x0080_ff00,
                accelerationStructureResourceID: 115
            ),
        ]
    )
    try backend.createImage(id: 117, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 117,
        accelerationStructureID: 116,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: RayCamera(
            position: SIMD3<Float>(0, 0, -2),
            forward: SIMD3<Float>(0, 0, 1),
            up: SIMD3<Float>(0, 1, 0),
            verticalFOVRadians: 0.9,
            nearDistance: 0.01,
            farDistance: 100
        ),
        sphereLight: nil,
        distantLight: nil,
        domeLight: nil,
        fenceID: 118
    )
    #expect(try backend.waitFence(id: 118))
    let center = (32 * 64 + 32) * 4
    let pixel = Array(try backend.readImage(id: 117)[center..<(center + 4)])
    #expect(pixel[1] > 240)
    #expect(pixel[0] < 10)
    #expect(pixel[2] < 10)
    #expect(pixel[3] == 255)
}

@Test func metalRayShadingUsesRoughnessAndMetallic() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    #expect(backend.supportsRayDispatch)

    func buildMaterialTriangle(
        bufferID: UInt64,
        accelerationID: UInt64,
        roughness: Float,
        metallic: Float
    ) throws {
        // vertexFormat=4 extends position/normal/UV with one bounded
        // roughness/metallic pair repeated at each triangle corner.
        try backend.createBuffer(id: bufferID, size: 120, options: 0)
        var vertices = Data()
        for position: SIMD3<Float> in [
            SIMD3<Float>(-1, -1, 0),
            SIMD3<Float>(1, -1, 0),
            SIMD3<Float>(0, 1, 0),
        ] {
            for value in [
                position.x, position.y, position.z,
                0, 0, -1,
                0, 0,
                roughness, metallic,
            ] {
                appendFloat(value, to: &vertices)
            }
        }
        try backend.writeBuffer(id: bufferID, offset: 0, data: vertices)
        try backend.createAccelerationStructure(
            id: accelerationID, type: 1, requestedSize: 4096
        )
        try backend.buildPrimitiveAccelerationStructure(
            id: accelerationID,
            buildFlags: 0x4,
            geometries: [PrimitiveAccelerationStructureGeometry(
                kind: .triangles,
                flags: 1,
                dataResourceID: bufferID,
                dataOffset: 0,
                primitiveCount: 1,
                stride: 40,
                vertexFormat: 4
            )]
        )
    }

    try buildMaterialTriangle(
        bufferID: 50, accelerationID: 51, roughness: 1, metallic: 0
    )
    try buildMaterialTriangle(
        bufferID: 52, accelerationID: 53, roughness: 0.05, metallic: 1
    )
    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    ]
    for (topLevelID, childID) in [(UInt64(54), UInt64(51)), (55, 53)] {
        try backend.createAccelerationStructure(
            id: topLevelID, type: 0, requestedSize: 4096
        )
        try backend.buildInstanceAccelerationStructure(
            id: topLevelID,
            buildFlags: 0x4,
            instances: [InstanceAccelerationStructureInstance(
                transformationMatrix: identity,
                options: 0,
                mask: 0xff,
                intersectionFunctionTableOffset: 0,
                userID: 0x0080_0080,
                accelerationStructureResourceID: childID
            )]
        )
    }
    let camera = RayCamera(
        position: SIMD3<Float>(0, 0, -2),
        forward: SIMD3<Float>(0, 0, 1),
        up: SIMD3<Float>(0, 1, 0),
        verticalFOVRadians: 0.9,
        nearDistance: 0.01,
        farDistance: 100
    )
    let light = RaySphereLight(
        position: SIMD3<Float>(0, 0, -2),
        color: SIMD3<Float>(1, 1, 1),
        intensity: 100,
        radius: 1
    )
    for (imageID, topLevelID, fenceID) in [
        (UInt64(56), UInt64(54), UInt64(58)),
        (57, 55, 59),
    ] {
        try backend.createImage(
            id: imageID, width: 64, height: 64, format: 1, options: 0
        )
        try backend.submitRayTrace(
            imageID: imageID,
            accelerationStructureID: topLevelID,
            width: 64,
            height: 64,
            missRGBA8: 0xff00_0000,
            hitRGBA8: 0xffe0_8c30,
            camera: camera,
            sphereLight: light,
            distantLight: nil,
            domeLight: nil,
            fenceID: fenceID
        )
        #expect(try backend.waitFence(id: fenceID))
    }
    let center = (32 * 64 + 32) * 4
    let roughDielectric = Array(
        try backend.readImage(id: 56)[center..<(center + 4)]
    )
    let smoothMetal = Array(
        try backend.readImage(id: 57)[center..<(center + 4)]
    )
    #expect(roughDielectric[0] > 0)
    #expect(smoothMetal[0] > 0)
    #expect(abs(Int(roughDielectric[0]) - Int(smoothMetal[0])) >= 8)
}

@Test func metalRayShadingAddsDirectEmissionWithoutLights() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    #expect(backend.supportsRayDispatch)

    // vertexFormat=5 extends the scalar material record with float3 emission
    // and intensity. No light is supplied, so green output proves that the
    // value is self-emission rather than diffuse or specular illumination.
    try backend.createBuffer(id: 60, size: 168, options: 0)
    var vertices = Data()
    for position: SIMD3<Float> in [
        SIMD3<Float>(-1, -1, 0),
        SIMD3<Float>(1, -1, 0),
        SIMD3<Float>(0, 1, 0),
    ] {
        for value in [
            position.x, position.y, position.z,
            0, 0, -1,
            0, 0,
            0.5, 0,
            0, 0.7, 0, 1,
        ] {
            appendFloat(value, to: &vertices)
        }
    }
    try backend.writeBuffer(id: 60, offset: 0, data: vertices)
    try backend.createAccelerationStructure(id: 61, type: 1, requestedSize: 4096)
    try backend.buildPrimitiveAccelerationStructure(
        id: 61,
        buildFlags: 0x4,
        geometries: [PrimitiveAccelerationStructureGeometry(
            kind: .triangles,
            flags: 1,
            dataResourceID: 60,
            dataOffset: 0,
            primitiveCount: 1,
            stride: 56,
            vertexFormat: 5
        )]
    )
    try backend.createAccelerationStructure(id: 62, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 62,
        buildFlags: 0x4,
        instances: [InstanceAccelerationStructureInstance(
            transformationMatrix: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
            ],
            options: 0,
            mask: 0xff,
            intersectionFunctionTableOffset: 0,
            userID: 0x0080_0010,
            accelerationStructureResourceID: 61
        )]
    )
    try backend.createImage(id: 63, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 63,
        accelerationStructureID: 62,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: RayCamera(
            position: SIMD3<Float>(0, 0, -2),
            forward: SIMD3<Float>(0, 0, 1),
            up: SIMD3<Float>(0, 1, 0),
            verticalFOVRadians: 0.9,
            nearDistance: 0.01,
            farDistance: 100
        ),
        sphereLight: nil,
        distantLight: nil,
        domeLight: nil,
        fenceID: 64
    )
    #expect(try backend.waitFence(id: 64))
    let center = (32 * 64 + 32) * 4
    let pixel = Array(try backend.readImage(id: 63)[center..<(center + 4)])
    #expect(pixel[1] > pixel[0])
    #expect(pixel[1] >= 160)
}

@Test func metalRayShadingSamplesEmissionTextureFromMaterialDescriptor() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    #expect(backend.supportsRayDispatch)

    // The two colors must remain spatially distinct. A bridge that replaces
    // this texture with one averaged color cannot satisfy both assertions.
    try backend.createImage(id: 70, width: 2, height: 2, format: 1, options: 0)
    try backend.writeImage(id: 70, data: Data([
        255, 0, 0, 255, 0, 255, 0, 255,
        255, 0, 0, 255, 0, 255, 0, 255,
    ]))

    // vertexFormat=5 carries the shared UV plus a white x1 emission factor;
    // the material descriptor supplies the emission texture itself.
    try backend.createBuffer(id: 71, size: 168, options: 0)
    var vertices = Data()
    for (position, uv): (SIMD3<Float>, SIMD2<Float>) in [
        (SIMD3<Float>(-1, -1, 0), SIMD2<Float>(0, 0)),
        (SIMD3<Float>(1, -1, 0), SIMD2<Float>(1, 0)),
        (SIMD3<Float>(0, 1, 0), SIMD2<Float>(0.5, 1)),
    ] {
        for value in [
            position.x, position.y, position.z,
            0, 0, -1,
            uv.x, uv.y,
            0.5, 0,
            1, 1, 1, 1,
        ] {
            appendFloat(value, to: &vertices)
        }
    }
    try backend.writeBuffer(id: 71, offset: 0, data: vertices)
    try backend.createAccelerationStructure(id: 72, type: 1, requestedSize: 4096)
    try backend.buildPrimitiveAccelerationStructure(
        id: 72,
        buildFlags: 0x4,
        geometries: [PrimitiveAccelerationStructureGeometry(
            kind: .triangles,
            flags: 1,
            dataResourceID: 71,
            dataOffset: 0,
            primitiveCount: 1,
            stride: 56,
            vertexFormat: 5
        )]
    )

    // MBM1 v1: flags=emission, RGB channel selector=4, then four UInt64
    // resource IDs in base/roughness/metallic/emission order.
    try backend.createBuffer(id: 73, size: 48, options: 0)
    var descriptor = Data()
    for word: UInt32 in [
        0x314d_424d, 1, 8, 4 << 8,
        0, 0,
        0, 0,
        0, 0,
        70, 0,
    ] {
        descriptor.appendLittleEndian(word)
    }
    try backend.writeBuffer(id: 73, offset: 0, data: descriptor)

    try backend.createAccelerationStructure(id: 74, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 74,
        buildFlags: 0x4,
        instances: [InstanceAccelerationStructureInstance(
            transformationMatrix: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
            ],
            options: 0,
            mask: 0xff,
            intersectionFunctionTableOffset: 0,
            userID: 0x00c0_0000 | 73,
            accelerationStructureResourceID: 72
        )]
    )
    try backend.createImage(id: 75, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 75,
        accelerationStructureID: 74,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: RayCamera(
            position: SIMD3<Float>(0, 0, -2),
            forward: SIMD3<Float>(0, 0, 1),
            up: SIMD3<Float>(0, 1, 0),
            verticalFOVRadians: 0.9,
            nearDistance: 0.01,
            farDistance: 100
        ),
        sphereLight: nil,
        distantLight: nil,
        domeLight: nil,
        fenceID: 76
    )
    #expect(try backend.waitFence(id: 76))
    let pixels = try backend.readImage(id: 75)
    let screenLeftOffset = (38 * 64 + 14) * 4
    let screenRightOffset = (38 * 64 + 49) * 4
    let screenLeft = Array(pixels[screenLeftOffset..<(screenLeftOffset + 4)])
    let screenRight = Array(pixels[screenRightOffset..<(screenRightOffset + 4)])
    #expect(screenLeft[1] > screenLeft[0])
    #expect(screenRight[0] > screenRight[1])
}

@Test func metalRayShadingAppliesTangentSpaceNormalTexture() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    #expect(backend.supportsRayDispatch)

    try backend.createImage(id: 80, width: 2, height: 2, format: 1, options: 0)
    try backend.writeImage(id: 80, data: Data([
        230, 128, 204, 255, 26, 128, 204, 255,
        230, 128, 204, 255, 26, 128, 204, 255,
    ]))

    // vertexFormat=6 extends format 5 with a real tangent and bitangent. The
    // two normal-map texels tilt the same planar triangle in opposite X
    // directions, so one side faces the off-axis light and the other does not.
    try backend.createBuffer(id: 81, size: 240, options: 0)
    var vertices = Data()
    for (position, uv): (SIMD3<Float>, SIMD2<Float>) in [
        (SIMD3<Float>(-1, -1, 0), SIMD2<Float>(0, 0)),
        (SIMD3<Float>(1, -1, 0), SIMD2<Float>(1, 0)),
        (SIMD3<Float>(0, 1, 0), SIMD2<Float>(0.5, 1)),
    ] {
        for value in [
            position.x, position.y, position.z,
            0, 0, -1,
            uv.x, uv.y,
            0.5, 0,
            0, 0, 0, 0,
            1, 0, 0,
            0, 1, 0,
        ] {
            appendFloat(value, to: &vertices)
        }
    }
    try backend.writeBuffer(id: 81, offset: 0, data: vertices)
    try backend.createAccelerationStructure(id: 82, type: 1, requestedSize: 4096)
    try backend.buildPrimitiveAccelerationStructure(
        id: 82,
        buildFlags: 0x4,
        geometries: [PrimitiveAccelerationStructureGeometry(
            kind: .triangles,
            flags: 1,
            dataResourceID: 81,
            dataOffset: 0,
            primitiveCount: 1,
            stride: 80,
            vertexFormat: 6
        )]
    )

    try backend.createBuffer(id: 83, size: 56, options: 0)
    var descriptor = Data()
    for word: UInt32 in [
        0x314d_424d, 2, 16, 0,
        0, 0,
        0, 0,
        0, 0,
        0, 0,
        80, 0,
    ] {
        descriptor.appendLittleEndian(word)
    }
    try backend.writeBuffer(id: 83, offset: 0, data: descriptor)
    try backend.createAccelerationStructure(id: 84, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 84,
        buildFlags: 0x4,
        instances: [InstanceAccelerationStructureInstance(
            transformationMatrix: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
            ],
            options: 0,
            mask: 0xff,
            intersectionFunctionTableOffset: 0,
            userID: 0x00c0_0000 | 83,
            accelerationStructureResourceID: 82
        )]
    )
    try backend.createImage(id: 85, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 85,
        accelerationStructureID: 84,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: RayCamera(
            position: SIMD3<Float>(0, 0, -2),
            forward: SIMD3<Float>(0, 0, 1),
            up: SIMD3<Float>(0, 1, 0),
            verticalFOVRadians: 0.9,
            nearDistance: 0.01,
            farDistance: 100
        ),
        sphereLight: RaySphereLight(
            position: SIMD3<Float>(2, 0, -1),
            color: SIMD3<Float>(1, 1, 1),
            intensity: 100_000,
            radius: 0.25
        ),
        distantLight: nil,
        domeLight: nil,
        fenceID: 86
    )
    #expect(try backend.waitFence(id: 86))
    let pixels = try backend.readImage(id: 85)
    let screenLeftOffset = (38 * 64 + 14) * 4
    let screenRightOffset = (38 * 64 + 49) * 4
    let screenLeft = Array(pixels[screenLeftOffset..<(screenLeftOffset + 4)])
    let screenRight = Array(pixels[screenRightOffset..<(screenRightOffset + 4)])
    #expect(Int(screenRight[2]) - Int(screenLeft[2]) >= 12)
}

@Test func metalRayLightingUsesRealTLASOcclusion() throws {
    let backend = try #require(MetalGPUBackend.makeDefault())
    #expect(backend.supportsRayDispatch)

    // Triangle 0 is the lit receiver at z=0. Triangle 1 is a small diagonal
    // blocker centered halfway from the receiver to the SphereLight. It is far
    // enough to the right that the primary center-camera ray misses it.
    try backend.createBuffer(id: 100, size: 72, options: 0)
    var vertices = Data()
    for point: SIMD3<Float> in [
        SIMD3<Float>(-1, -1, 0),
        SIMD3<Float>(1, -1, 0),
        SIMD3<Float>(0, 1, 0),
        SIMD3<Float>(0.5, 0.25, -0.5),
        SIMD3<Float>(0.676_776_7, -0.2, -0.323_223_3),
        SIMD3<Float>(0.323_223_3, -0.2, -0.676_776_7),
    ] {
        appendFloat(point.x, to: &vertices)
        appendFloat(point.y, to: &vertices)
        appendFloat(point.z, to: &vertices)
    }
    try backend.writeBuffer(id: 100, offset: 0, data: vertices)

    try backend.createAccelerationStructure(id: 101, type: 1, requestedSize: 4096)
    try backend.buildPrimitiveAccelerationStructure(
        id: 101,
        buildFlags: 0x4,
        geometries: [PrimitiveAccelerationStructureGeometry(
            kind: .triangles,
            flags: 1,
            dataResourceID: 100,
            dataOffset: 0,
            primitiveCount: 1,
            stride: 12,
            vertexFormat: 1
        )]
    )
    try backend.createAccelerationStructure(id: 102, type: 1, requestedSize: 4096)
    try backend.buildPrimitiveAccelerationStructure(
        id: 102,
        buildFlags: 0x4,
        geometries: [PrimitiveAccelerationStructureGeometry(
            kind: .triangles,
            flags: 1,
            dataResourceID: 100,
            dataOffset: 36,
            primitiveCount: 1,
            stride: 12,
            vertexFormat: 1
        )]
    )

    let identity: [Float] = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
    ]
    let receiver = InstanceAccelerationStructureInstance(
        transformationMatrix: identity,
        options: 0,
        mask: 0xff,
        intersectionFunctionTableOffset: 0,
        userID: 0x0080_00ff,
        accelerationStructureResourceID: 101
    )
    let blocker = InstanceAccelerationStructureInstance(
        transformationMatrix: identity,
        options: 0,
        mask: 0xff,
        intersectionFunctionTableOffset: 0,
        userID: 0x0080_00ff,
        accelerationStructureResourceID: 102
    )
    try backend.createAccelerationStructure(id: 103, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 103,
        buildFlags: 0x4,
        instances: [receiver]
    )
    try backend.createAccelerationStructure(id: 104, type: 0, requestedSize: 4096)
    try backend.buildInstanceAccelerationStructure(
        id: 104,
        buildFlags: 0x4,
        instances: [receiver, blocker]
    )

    let camera = RayCamera(
        position: SIMD3<Float>(0, 0, -2),
        forward: SIMD3<Float>(0, 0, 1),
        up: SIMD3<Float>(0, 1, 0),
        verticalFOVRadians: 0.9,
        nearDistance: 0.01,
        farDistance: 100
    )
    let light = RaySphereLight(
        position: SIMD3<Float>(1, 0, -1),
        color: SIMD3<Float>(1, 1, 1),
        intensity: 100_000,
        radius: 0.25
    )
    try backend.createImage(id: 105, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 105,
        accelerationStructureID: 103,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: camera,
        sphereLight: light,
        distantLight: nil,
        domeLight: nil,
        fenceID: 200
    )
    #expect(try backend.waitFence(id: 200))
    try backend.createImage(id: 106, width: 64, height: 64, format: 1, options: 0)
    try backend.submitRayTrace(
        imageID: 106,
        accelerationStructureID: 104,
        width: 64,
        height: 64,
        missRGBA8: 0xff00_0000,
        hitRGBA8: 0xffe0_8c30,
        camera: camera,
        sphereLight: light,
        distantLight: nil,
        domeLight: nil,
        fenceID: 201
    )
    #expect(try backend.waitFence(id: 201))

    let center = (32 * 64 + 32) * 4
    let lit = Array(try backend.readImage(id: 105)[center..<(center + 4)])
    let shadowed = Array(try backend.readImage(id: 106)[center..<(center + 4)])
    #expect(lit[0] > 0)
    #expect(shadowed[0] > 0)
    #expect(shadowed[0] < lit[0])
}
#endif
