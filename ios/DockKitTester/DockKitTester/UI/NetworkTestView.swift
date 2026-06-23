import SwiftUI

struct NetworkTestView: View {
    @ObservedObject var client: V13NetworkClient
    let canInjectCommand: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("AutoCamTracker V1.5")
                    .font(.headline)
                Spacer()
                Text(client.status.rawValue)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }

            TextField("ws://Mac-IP:8765/ws/tracking", text: $client.serverURL)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .textFieldStyle(.roundedBorder)

            HStack {
                Button("Connect") {
                    Task { await client.connect() }
                }
                .buttonStyle(.borderedProminent)

                Button("Disconnect") {
                    Task { await client.disconnect() }
                }
                .buttonStyle(.bordered)
            }

            if let command = client.lastCommand {
                Text(
                    String(
                        format: "locked=%@  error_x=%.3f  error_y=%.3f  confidence=%.2f",
                        String(command.targetLocked),
                        command.errorX,
                        command.errorY,
                        command.confidence
                    )
                )
                .font(.system(.caption, design: .monospaced))
            }

            Text("Camera frames sent: \(client.cameraFramesSent)")
                .font(.system(.caption, design: .monospaced))

            Button("Inject Local Fake JSON") {
                Task { await client.sendFakeCommand() }
            }
            .buttonStyle(.bordered)
            .disabled(!canInjectCommand)

            Text("Wi-Fi and USB network links use the same WebSocket URL. Missing tracking data for 500 ms always triggers STOP.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
    }
}
