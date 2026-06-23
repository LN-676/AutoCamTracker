import SwiftUI

struct ContentView: View {
    @ObservedObject var dockKitManager: DockKitManager
    @ObservedObject var cameraSession: CameraSessionService
    @ObservedObject var controlService: GimbalControlService
    @ObservedObject var networkClient: V13NetworkClient
    @ObservedObject var logger: AppLogger
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        TabView {
            cameraTab
                .tabItem { Label("相機", systemImage: "camera.fill") }

            gimbalTab
                .tabItem { Label("雲台", systemImage: "move.3d") }

            connectionTab
                .tabItem { Label("連線", systemImage: "network") }

            logTab
                .tabItem { Label("紀錄", systemImage: "list.bullet.rectangle") }
        }
        .tint(.yellow)
        .task { await prepareServices() }
        .onChange(of: scenePhase) { _, newPhase in
            Task {
                if newPhase == .active {
                    await cameraSession.start()
                } else {
                    await controlService.emergencyStop(reason: "app left foreground")
                    await cameraSession.stop()
                }
            }
        }
    }

    private var cameraTab: some View {
        NavigationStack {
            CameraControlPage(
                cameraSession: cameraSession,
                dockKitManager: dockKitManager,
                networkClient: networkClient
            )
            .navigationTitle("AutoCam Camera")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbarBackground(.black, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
        }
    }

    private var gimbalTab: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    safetyNotice
                    StatusPanelView(
                        manager: dockKitManager,
                        onTestVelocity: { await controlService.testAngularVelocity() }
                    )
                    ManualControlPadView(
                        isDocked: dockKitManager.isDocked,
                        isManualControlReady: dockKitManager.isManualControlReady,
                        onCommand: { await controlService.execute($0) }
                    )
                    velocityPanel
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("雲台控制")
        }
    }

    private var connectionTab: some View {
        NavigationStack {
            ScrollView {
                NetworkTestView(
                    client: networkClient,
                    canInjectCommand: dockKitManager.isManualControlReady
                )
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("電腦連線")
        }
    }

    private var logTab: some View {
        NavigationStack {
            ScrollView {
                LogConsoleView(logger: logger)
                    .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("系統紀錄")
        }
    }

    private var safetyNotice: some View {
        Label(
            "先進入 Manual Mode 並確認 Tracking OFF。方向鍵會持續輸出速度，測完請立即按 STOP。",
            systemImage: "exclamationmark.triangle.fill"
        )
        .font(.footnote.weight(.semibold))
        .foregroundStyle(.orange)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))
    }

    private var velocityPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("目前命令")
                .font(.headline)
            Text(
                String(
                    format: "yaw %.3f   pitch %.3f   roll %.3f rad/s",
                    controlService.currentVelocity.yaw,
                    controlService.currentVelocity.pitch,
                    controlService.currentVelocity.roll
                )
            )
            .font(.system(.body, design: .monospaced))
        }
        .panelStyle()
    }

    private func prepareServices() async {
        cameraSession.onJPEGFrame = { [weak networkClient] data in
            Task { @MainActor in await networkClient?.sendCameraFrame(data) }
        }
        networkClient.onCommand = { [weak controlService] command in
            await controlService?.apply(command)
        }
        networkClient.onTimeout = { [weak controlService] in
            await controlService?.emergencyStop(reason: "V1.42 timeout or disconnect")
        }
        await cameraSession.start()
        await dockKitManager.startListening()
    }
}

private struct CameraControlPage: View {
    @ObservedObject var cameraSession: CameraSessionService
    @ObservedObject var dockKitManager: DockKitManager
    @ObservedObject var networkClient: V13NetworkClient
    @Environment(\.verticalSizeClass) private var verticalSizeClass

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                ZStack(alignment: .top) {
                    CameraPreviewView(
                        session: cameraSession.session,
                        onFocus: cameraSession.focus(at:)
                    )
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .clipped()

                    statusStrip
                        .padding(.horizontal, 12)
                        .padding(.top, 10)

                    if !cameraSession.isRunning {
                        ContentUnavailableView(
                            "相機未啟動",
                            systemImage: "camera.slash",
                            description: Text(cameraSession.lastError ?? "正在等待相機權限")
                        )
                        .foregroundStyle(.white)
                    }
                }
                .frame(height: previewHeight(for: geometry.size))

                controls
            }
            .background(.black)
        }
        .background(.black)
    }

    private var statusStrip: some View {
        HStack(spacing: 8) {
            statusChip(
                cameraSession.isRunning ? "相機" : "相機關閉",
                icon: cameraSession.isRunning ? "camera.fill" : "camera.slash",
                active: cameraSession.isRunning
            )
            statusChip(
                dockKitManager.isDocked ? "雲台" : "未接雲台",
                icon: "move.3d",
                active: dockKitManager.isDocked
            )
            statusChip(
                networkActive ? "電腦" : "未連線",
                icon: "network",
                active: networkActive
            )
            Spacer()
            statusChip(
                cameraSession.streamOrientation.label,
                icon: cameraSession.streamOrientation == .portrait ? "iphone" : "iphone.landscape",
                active: true
            )
        }
    }

    private var controls: some View {
        VStack(spacing: 14) {
            HStack(spacing: 12) {
                ForEach(zoomPresets, id: \.self) { factor in
                    Button {
                        cameraSession.setDisplayZoom(factor)
                    } label: {
                        Text(formatZoom(factor))
                            .font(.subheadline.weight(.bold))
                            .foregroundStyle(isSelected(factor) ? .black : .yellow)
                            .frame(width: 46, height: 34)
                            .background(
                                isSelected(factor) ? Color.yellow : Color.white.opacity(0.12),
                                in: Capsule()
                            )
                    }
                }
            }

            HStack(spacing: 12) {
                Image(systemName: "minus.magnifyingglass")
                Slider(
                    value: Binding(
                        get: { cameraSession.displayZoomFactor },
                        set: { cameraSession.setDisplayZoom($0) }
                    ),
                    in: cameraSession.minimumDisplayZoomFactor...max(
                        cameraSession.minimumDisplayZoomFactor,
                        cameraSession.maximumDisplayZoomFactor
                    )
                )
                Image(systemName: "plus.magnifyingglass")
            }
            .foregroundStyle(.yellow)

            HStack {
                Label("點按畫面對焦", systemImage: "viewfinder")
                Spacer()
                Text("目前 \(formatZoom(cameraSession.displayZoomFactor))")
                    .fontWeight(.semibold)
            }
            .font(.caption)
            .foregroundStyle(.white.opacity(0.75))
        }
        .padding(.horizontal, 20)
        .padding(.vertical, verticalSizeClass == .compact ? 10 : 18)
        .background(.black)
    }

    private var zoomPresets: [CGFloat] {
        let candidates: [CGFloat] = [cameraSession.minimumDisplayZoomFactor, 1, 2, 5]
        var result: [CGFloat] = []
        for factor in candidates where factor >= cameraSession.minimumDisplayZoomFactor && factor <= cameraSession.maximumDisplayZoomFactor {
            if !result.contains(where: { abs($0 - factor) < 0.05 }) { result.append(factor) }
        }
        return result
    }

    private var networkActive: Bool {
        networkClient.status == .connected || networkClient.status == .receiving
    }

    private func previewHeight(for size: CGSize) -> CGFloat {
        verticalSizeClass == .compact ? size.height * 0.68 : size.height * 0.72
    }

    private func statusChip(_ text: String, icon: String, active: Bool) -> some View {
        Label(text, systemImage: icon)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.white)
            .padding(.horizontal, 9)
            .padding(.vertical, 6)
            .background(active ? Color.black.opacity(0.58) : Color.red.opacity(0.72), in: Capsule())
    }

    private func formatZoom(_ value: CGFloat) -> String {
        value < 1 ? String(format: "%.1f×", value) : String(format: "%.0f×", value)
    }

    private func isSelected(_ factor: CGFloat) -> Bool {
        abs(cameraSession.displayZoomFactor - factor) < 0.08
    }
}

private extension View {
    func panelStyle() -> some View {
        frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 14))
    }
}
