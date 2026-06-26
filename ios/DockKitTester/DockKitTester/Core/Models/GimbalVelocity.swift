import Foundation

struct GimbalVelocity: Equatable, Sendable {
    var yaw: Double
    var pitch: Double
    var roll: Double

    static let zero = GimbalVelocity(yaw: 0, pitch: 0, roll: 0)
}

struct GimbalControlConfiguration: Equatable, Sendable {
    var manualSpeed = 0.2
    var maxYawSpeed = 0.35
    var maxPitchSpeed = 0.22
    var deadZone = 0.05
    var smoothingOldWeight = 0.7
    var kpYaw = 1.0
    var kpPitch = 1.0
    var yawDirection = 1.0
    var pitchDirection = 1.0
    var minimumErrorImprovement = 0.01
    var maxNonImprovingUpdates = 8
}

struct GimbalVelocityCalculator: Sendable {
    var configuration: GimbalControlConfiguration
    private(set) var previous = GimbalVelocity.zero
    private(set) var safetyStopReason: String?
    private var previousErrorMagnitude: Double?
    private var nonImprovingUpdates = 0

    init(configuration: GimbalControlConfiguration = .init()) {
        self.configuration = configuration
    }

    mutating func setTrackingAxisInversion(yawInverted: Bool, pitchInverted: Bool) {
        configuration.yawDirection = yawInverted ? -1 : 1
        configuration.pitchDirection = pitchInverted ? -1 : 1
        reset()
    }

    mutating func velocity(for command: GimbalCommand) -> GimbalVelocity {
        let speed = configuration.manualSpeed
        let output: GimbalVelocity
        switch command {
        case .panLeft:
            output = .init(yaw: -speed, pitch: 0, roll: 0)
        case .panRight:
            output = .init(yaw: speed, pitch: 0, roll: 0)
        case .tiltUp:
            output = .init(yaw: 0, pitch: -speed, roll: 0)
        case .tiltDown:
            output = .init(yaw: 0, pitch: speed, roll: 0)
        case .stop, .recenter:
            output = .zero
        }
        previous = output
        return output
    }

    mutating func velocity(for tracking: TrackingCommand) -> GimbalVelocity {
        safetyStopReason = nil
        guard tracking.isTrackable() else {
            reset()
            return .zero
        }

        let errorX = abs(tracking.errorX) < configuration.deadZone ? 0 : tracking.errorX
        let errorY = abs(tracking.errorY) < configuration.deadZone ? 0 : tracking.errorY
        if errorX == 0, errorY == 0 {
            reset()
            return .zero
        }

        guard shouldContinueTracking(errorX: errorX, errorY: errorY) else {
            previous = .zero
            previousErrorMagnitude = nil
            nonImprovingUpdates = 0
            safetyStopReason = "tracking error did not improve; check yaw/pitch direction"
            return .zero
        }

        let requestedYaw = clamp(
            errorX * configuration.kpYaw * configuration.yawDirection,
            min: -configuration.maxYawSpeed,
            max: configuration.maxYawSpeed
        )
        let requestedPitch = clamp(
            -errorY * configuration.kpPitch * configuration.pitchDirection,
            min: -configuration.maxPitchSpeed,
            max: configuration.maxPitchSpeed
        )
        let newWeight = 1 - configuration.smoothingOldWeight
        let output = GimbalVelocity(
            yaw: previous.yaw * configuration.smoothingOldWeight + requestedYaw * newWeight,
            pitch: previous.pitch * configuration.smoothingOldWeight + requestedPitch * newWeight,
            roll: 0
        )
        previous = output
        return output
    }

    mutating func reset() {
        previous = .zero
        previousErrorMagnitude = nil
        nonImprovingUpdates = 0
        safetyStopReason = nil
    }

    private mutating func shouldContinueTracking(errorX: Double, errorY: Double) -> Bool {
        let magnitude = sqrt(errorX * errorX + errorY * errorY)
        defer { previousErrorMagnitude = magnitude }

        guard let previousErrorMagnitude else {
            nonImprovingUpdates = 0
            return true
        }

        if magnitude < previousErrorMagnitude - configuration.minimumErrorImprovement {
            nonImprovingUpdates = 0
            return true
        }

        nonImprovingUpdates += 1
        return nonImprovingUpdates < configuration.maxNonImprovingUpdates
    }
}
