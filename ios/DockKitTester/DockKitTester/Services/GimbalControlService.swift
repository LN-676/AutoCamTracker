import Foundation
import Combine

@MainActor
final class GimbalControlService: ObservableObject {
    @Published private(set) var currentVelocity = GimbalVelocity.zero

    private let dockKitManager: DockKitMotorControlling
    private let logger: AppLogger
    private var calculator = GimbalVelocityCalculator()
    private var commandGeneration = 0

    init(dockKitManager: DockKitMotorControlling, logger: AppLogger) {
        self.dockKitManager = dockKitManager
        self.logger = logger
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
            logger.log(.error, "Ignored V1.62 message with unsupported type: \(trackingCommand.type).")
            await emergencyStop(reason: "invalid V1.62 message")
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
        logger.log(.warning, "Safety stop: \(reason).")
        await dockKitManager.stop()
    }
}
