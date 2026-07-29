import Foundation
import Testing
@testable import IMBHostCore

@Test func viewerInputRecordHasStableCrossLanguageLayout() throws {
    let record = ViewerInputRecord(
        kind: .leftMouseDown,
        sequence: 0x0102_0304_0506_0708,
        x: 123.5,
        y: 456.25,
        deltaX: -2.5,
        deltaY: 7.75,
        code: 42,
        modifiers: 5,
        targetWidth: 1_440,
        targetHeight: 900
    )
    let encoded = record.encoded()
    #expect(encoded.count == ViewerInputProtocol.recordSize)
    #expect(Array(encoded.prefix(8)) == [0x49, 0x4d, 0x49, 0x31, 0x01, 0x00, 0x02, 0x00])
    #expect(try ViewerInputRecord.decode(encoded) == record)
}

@Test func viewerInputRecordRejectsWrongMagicAndLength() throws {
    let record = ViewerInputRecord(kind: .keyDown, sequence: 1, code: 36).encoded()
    #expect(throws: WireError.self) {
        try ViewerInputRecord.decode(record.dropLast())
    }
    var invalid = record
    invalid[0] = 0
    #expect(throws: WireError.self) {
        try ViewerInputRecord.decode(invalid)
    }
}
