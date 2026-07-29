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
            userID: 7,
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
        fenceID: 9
    )
    #expect(try backend.waitFence(id: 9))
    let rayPixels = try backend.readImage(id: 4)
    let center = (32 * 64 + 32) * 4
    let corner = 0
    #expect(Array(rayPixels[center..<(center + 4)]) == [0, 255, 0, 255])
    #expect(Array(rayPixels[corner..<(corner + 4)]) == [0, 0, 0, 255])
    try backend.destroyAccelerationStructure(id: 3)
    try backend.destroyAccelerationStructure(id: 2)
}
#endif
