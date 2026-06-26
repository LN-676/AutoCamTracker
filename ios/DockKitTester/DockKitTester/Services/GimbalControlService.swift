import Foundation
import Combine

@MainActor
final class GimbalControlService: ObservableObject {
    @Published private(set) var currentVelocity = GimbalVelocity.zero
    @Published private(set) var calibration = GimbalCalibrationProfile.conservative
    @Published private(set) var lastStopReason: String?

    private let dockKitManager: DockKitMotorControlling
    private let logger: AppLogger
    private var calculator = GimbalVelocityCalculator()
    private var commandGeneration = 0
    private let calibrationKey = "AutoCamTrackerGimbalCalibrationV164"

    init(dockKitManager: DockKitMotorControlling, logger: AppLogger) {
        self.dockKitManager = dockKitManager
        self.logger = logger
        calibration = Self.loadCalibration(key: calibrationKey)
        calculator.applyCalibration(calibration)
    }

    func setYawInverted(_ inverted: Bool) {
        updateCalibration { $0.yawInverted = inverted }
        logger.log(.info, "Tracking yaw direction \(inverted ? "inverted" : "normal").")
    }

    func setPitchInverted(_ inverted: Bool) {
        updateCalibration { $0.pitchInverted = inverted }
        logger.log(.info, "Tracking pitch direction \(inverted ? "inverted" : "normal").")
    }

    func setMaxYawSpeed(_ value: Double) {
        updateCalibration { $0.maxYawSpeed = value }
    }

    func setMaxPitchSpeed(_ value: Double) {
        updateCalibration { $0.maxPitchSpeed = value }
    }

    func setDeadZone(_ value: Double) {
        updateCalibration { $0.deadZone = value }
    }

    func setMinimumErrorImprovement(_ value: Double) {
        updateCalibration { $0.minimumErrorImprovement = value }
    }

    func setMaxNonImprovingUpdates(_ value: Double) {
        updateCalibration { $0.maxNonImprovingUpdates = Int(value.rounded()) }
    }

    func resetCalibration() {
        calibration = .conservative
        saveCalibration()
        calculator.applyCalibration(calibration)
        logger.log(.info, "Tracking calibration reset to conservative defaults.")
    }

    func execute(_ command: GimbalCommand) async {
        switch command {
        case .stop:
            await emergencyStop(reason: "manual Stop")
        case .recenter:
            commandGeneration += 1
            calculator.reset()
            currentVelocity = .zero
            logger.log(.info, "Recenter requested.")
            await dockKitManager.recenter()
        default:
            commandGeneration += 1
            let generation = commandGeneration
            let velocity = calculator.velocity(for: command)
            currentVelocity = velocity
            await dockKitManager.setAngularVelocity(
                yaw: velocity.yaw,
                pitch: velocity.pitch,
                roll: velocity.roll
            )
            if generation != commandGeneration {
                await dockKitManager.stop()
            }
        }
    }

    func apply(_ trackingCommand: TrackingCommand) async {
        guard trackingCommand.type == "tracking" else {
            logger.log(.error, "Ignored V1.64 message with unsupported type: \(trackingCommand.type).")
            await emergencyStop(reason: "invalid V1.64 message")
            return
        }

        guard trackingCommand.isTrackable() else {
            calculator.reset()
            if currentVelocity != .zero {
                await emergencyStop(reason: "target unavailable or confidence below safety threshold")
            }
            return
        }

        commandGeneration += 1
        let generation = commandGeneration
        let velocity = calculator.velocity(for: trackingCommand)
        currentVelocity = velocity
        if let reason = calculator.safetyStopReason {
            await emergencyStop(reason: reason)
            return
        }

        await dockKitManager.setAngularVelocity(
            yaw: velocity.yaw,
            pitch: velocity.pitch,
            roll: velocity.roll
        )
        if generation != commandGeneration {
            await dockKitManager.stop()
        }
    }

    func testAngularVelocity() async {
        commandGeneration += 1
        let generation = commandGeneration
        let velocity = GimbalVelocity(yaw: 0.15, pitch: 0, roll: 0)
        currentVelocity = velocity
        logger.log(.info, "Angular velocity test: yaw +0.15 rad/s for 350 ms.")
        await dockKitManager.setAngularVelocity(yaw: velocity.yaw, pitch: 0, roll: 0)
        try? await Task.sleep(for: .milliseconds(350))
        if generation == commandGeneration {
            await emergencyStop(reason: "angular velocity test completed")
        }
    }

    func emergencyStop(reason: String) async {
        commandGeneration += 1
        calculator.reset()
        currentVelocity = .zero
        lastStopReason = reason
        logger.log(.warning, "Safety stop: \(reason).")
        await dockKitManager.stop()
    }

    private func updateCalibration(_ mutate: (inout GimbalCalibrationProfile) -> Void) {
        mutate(&calibration)
        calibration = calibration
        saveCalibration()
        calculator.applyCalibration(calibration)
    }

    private func saveCalibration() {
        guard let data = try? JSONEncoder().encode(calibration) else { return }
        UserDefaults.standard.set(data, forKey: calibrationKey)
    }

    private static func loadCalibration(key: String) -> GimbalCalibrationProfile {
        guard let data = UserDefaults.standard.data(forKey: key),
              let profile = try? JSONDecoder().decode(GimbalCalibrationProfile.self, from: data) else {
            return .conservative
        }
        return profile
    }
}
