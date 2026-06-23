import AVFoundation
import SwiftUI

struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    let onFocus: (CGPoint) -> Void

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.onFocus = onFocus
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.session = session
        uiView.onFocus = onFocus
    }
}

final class PreviewView: UIView {
    var onFocus: ((CGPoint) -> Void)?

    override init(frame: CGRect) {
        super.init(frame: frame)
        addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(focusTapped(_:))))
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(focusTapped(_:))))
    }

    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        guard let connection = previewLayer.connection,
              let orientation = window?.windowScene?.interfaceOrientation else { return }
        let angle: CGFloat
        switch orientation {
        case .portrait: angle = 90
        case .portraitUpsideDown: angle = 270
        case .landscapeLeft: angle = 0
        case .landscapeRight: angle = 180
        default: return
        }
        if connection.isVideoRotationAngleSupported(angle) {
            connection.videoRotationAngle = angle
        }
    }

    @objc private func focusTapped(_ recognizer: UITapGestureRecognizer) {
        let location = recognizer.location(in: self)
        onFocus?(previewLayer.captureDevicePointConverted(fromLayerPoint: location))
        showFocusReticle(at: location)
    }

    private func showFocusReticle(at point: CGPoint) {
        let reticle = UIView(frame: CGRect(x: 0, y: 0, width: 72, height: 72))
        reticle.center = point
        reticle.layer.borderWidth = 1.5
        reticle.layer.borderColor = UIColor.systemYellow.cgColor
        reticle.layer.cornerRadius = 4
        reticle.alpha = 0
        addSubview(reticle)

        reticle.transform = CGAffineTransform(scaleX: 1.25, y: 1.25)
        UIView.animate(withDuration: 0.18, animations: {
            reticle.alpha = 1
            reticle.transform = .identity
        }) { _ in
            UIView.animate(withDuration: 0.3, delay: 0.65, options: []) {
                reticle.alpha = 0
            } completion: { _ in
                reticle.removeFromSuperview()
            }
        }
    }
}
