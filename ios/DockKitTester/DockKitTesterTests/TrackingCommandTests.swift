import Foundation
import XCTest
@testable import DockKitTesterCore

final class TrackingCommandTests: XCTestCase {
    func testDecodesV143SnakeCasePayload() throws {
        let json = #"{"type":"tracking","version":"1.0","source_version":"1.6","sequence":42,"target_locked":true,"target_id":7,"error_x":0.18,"error_y":-0.04,"confidence":0.91,"timestamp_ms":1781770000000}"#

        let command = try JSONDecoder().decode(TrackingCommand.self, from: Data(json.utf8))

        XCTAssertEqual(command.type, "tracking")
        XCTAssertEqual(command.version, "1.0")
        XCTAssertEqual(command.sourceVersion, "1.6")
        XCTAssertEqual(command.sequence, 42)
        XCTAssertTrue(command.targetLocked)
        XCTAssertEqual(command.targetId, 7)
        XCTAssertEqual(command.errorX, 0.18)
        XCTAssertEqual(command.errorY, -0.04)
        XCTAssertEqual(command.confidence, 0.91)
        XCTAssertEqual(command.timestampMs, 1_781_770_000_000)
    }

    func testSafeDecoderReturnsFailureForMissingFields() {
        let json = Data(#"{"type":"tracking"}"#.utf8)

        let result = JSONDecoder().decodeSafely(TrackingCommand.self, from: json)

        if case .success = result {
            XCTFail("Expected malformed command to fail decoding")
        }
    }

    func testClamp() {
        XCTAssertEqual(clamp(2.0, min: -1.0, max: 1.0), 1.0)
        XCTAssertEqual(clamp(-2.0, min: -1.0, max: 1.0), -1.0)
        XCTAssertEqual(clamp(0.2, min: -1.0, max: 1.0), 0.2)
    }

    func testTrackingRequiresConfidenceAndFiniteErrors() {
        XCTAssertTrue(makeCommand(sequence: 1, confidence: 0.8).isTrackable())
        XCTAssertFalse(makeCommand(sequence: 2, confidence: 0.2).isTrackable())
        XCTAssertFalse(makeCommand(sequence: 3, confidence: 0.8, errorX: .infinity).isTrackable())
    }

    func testSequenceValidatorRejectsDuplicateAndOutOfOrderCommands() {
        var validator = TrackingCommandSequenceValidator()

        XCTAssertTrue(validator.accept(makeCommand(sequence: 10)))
        XCTAssertFalse(validator.accept(makeCommand(sequence: 10)))
        XCTAssertFalse(validator.accept(makeCommand(sequence: 9)))
        XCTAssertTrue(validator.accept(makeCommand(sequence: 11)))
        validator.reset()
        XCTAssertTrue(validator.accept(makeCommand(sequence: 1)))
    }

    private func makeCommand(
        sequence: Int64,
        confidence: Double = 0.9,
        errorX: Double = 0.1
    ) -> TrackingCommand {
        TrackingCommand(
            type: "tracking",
            sequence: sequence,
            targetLocked: true,
            targetId: 7,
            errorX: errorX,
            errorY: 0,
            confidence: confidence
        )
    }
}
