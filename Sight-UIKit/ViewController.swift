//
//  ViewController.swift
//  Sight-UIKit
//
//  Created by Helen Olhausen on 16/11/2024.
//

import UIKit
import AVFoundation
import Combine
import Vision

final class ViewController: UIViewController {

    private let sightLabel: UILabel = {
        let label = UILabel()
        label.font = .systemFont(ofSize: 32, weight: .semibold)
        label.numberOfLines = 0
        label.textColor = .white
        label.translatesAutoresizingMaskIntoConstraints = false
        label.textAlignment = .center
        return label
    }()
    
    private var sightString: String = "" {
        didSet {
            sightLabel.text = sightString
        }
    }
    
    private struct Constants {
        static let videoQueue = "videoQueue"
    }
    
    fileprivate enum SightError: LocalizedError {
        case failToLoadModel
        case noBestResult
        case unexpectedType
        case predictionFailed(error: String)

        var errorDescription: String? {
            switch self {
            case .failToLoadModel: return "Can't load model"
            case .noBestResult: return "No best result found"
            case .unexpectedType: return "Unexpected result type"
            case .predictionFailed(let error): return error
            }
        }
    }
    
    fileprivate var cancellables = Set<AnyCancellable>()
    fileprivate lazy var mlModel: VNCoreMLModel? = {
        // Returns nil if an error is thrown
        try? VNCoreMLModel(for: Resnet50(configuration: MLModelConfiguration()).model)
    }()
    fileprivate var frameSubject = PassthroughSubject<CMSampleBuffer, Never>()
    private var lastFrameHash: Int?
    private let speechSynthesizer = AVSpeechSynthesizer()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupPrediction()
    }

    private func setupUI() {
        setupVideo()
        view.addSubview(sightLabel)
        NSLayoutConstraint.activate([
            sightLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 32),
            sightLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -32),
            sightLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            sightLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor)
        ])
    }
    
    private func CVPixelBufferGetHash(_ pixelBuffer: CVPixelBuffer) -> Int {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)
        let hash = baseAddress?.hashValue ?? 0
        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        return hash
    }
    
    private func isFrameStable(_ sampleBuffer: CMSampleBuffer) -> Bool {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return false
        }
        let currentFrameHash = CVPixelBufferGetHash(pixelBuffer)
        
        // Compare the current frame hash to the previous one
        let isStable = currentFrameHash == lastFrameHash
        
        // Update the lastFrameHash before returning
        lastFrameHash = currentFrameHash
        
        return isStable
    }
    
    private func setupPrediction() {
        frameSubject
            .throttle(for: .seconds(1), scheduler: RunLoop.main, latest: true)
            .filter { [weak self] sampleBuffer in
                // Check if the camera is stable based on frame hash.
                self?.isFrameStable(sampleBuffer) ?? false
            }
            .sink { [weak self] sampleBuffer in
                
                guard let strongSelf = self else {
                    // TODO: Refactor: add to result for error
                    return
                }
                guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                    // TODO: Refactor: add to result for error
                    return
                }
                
                // Every time a frame is captured, we will make a prediction of what we are seeing this is where ML and Vision comes in, this prediction will return the result asynchronously
                strongSelf.performPredictionWith(cvPixelBuffer: pixelBuffer)
                    .removeDuplicates { old, new in
                        return old.trimmingCharacters(in: .whitespacesAndNewlines) == new.trimmingCharacters(in: .whitespacesAndNewlines)
                    }
                    .receive(on: RunLoop.main)
                    .sink(receiveCompletion: { result in
                        switch result {
                        case .finished:
                            break
                        case .failure(let failure):
                            DispatchQueue.main.async { [weak self] in
                                self?.sightLabel.textColor = .red
                                self?.handlePrediction(text: failure.localizedDescription)
                            }
                        }
                    }, receiveValue: { [weak self] resultString in
                        self?.handlePrediction(text: resultString)
                    })
                    .store(in: &strongSelf.cancellables)
            }
            .store(in: &cancellables)
    }
    
    private func handlePrediction(text: String) {
        DispatchQueue.main.async { [weak self] in
            if self?.sightString != text {
                self?.sightString = text
                self?.speak(text: text)
            }
        }
    }
    
    private func speak(text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        speechSynthesizer.speak(utterance)
    }
}

// MARK: AV
extension ViewController {
    fileprivate func setupVideo() {
        guard let camera = AVCaptureDevice.default(for: .video) else {
            sightString = "No video camera available to capture images"
            return
        }
        
        let videoCaptureSession = AVCaptureSession()
        do {
            videoCaptureSession.addInput(try AVCaptureDeviceInput(device: camera))
        } catch {
            sightString = error.localizedDescription
        }
        
        videoCaptureSession.sessionPreset = .high
       
        // setup video output
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: Constants.videoQueue))
        setFrameRate(30) // Set FPS to 30
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        
        // wire up the video output to the video capture session
        videoCaptureSession.addOutput(videoOutput)
        
        // add the preview layer
        let previewLayer = AVCaptureVideoPreviewLayer(session: videoCaptureSession)
        previewLayer.frame = view.frame
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        // start the session in a background thread
        DispatchQueue.global(qos: .background).async {
            videoCaptureSession.startRunning()
        }
    }
    
    func setFrameRate(_ fps: Int) {
        guard let videoDevice = AVCaptureDevice.default(for: .video) else { return }
        
        do {
            try videoDevice.lockForConfiguration()
            
            // Set the frame rate (FPS)
            videoDevice.activeVideoMinFrameDuration = CMTime(value: 1, timescale: Int32(fps))
            videoDevice.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: Int32(fps))
            
            videoDevice.unlockForConfiguration()
        } catch {
            print("Error setting frame rate: \(error)")
        }
    }
}

// MARK: AV Delegate
extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        frameSubject.send(sampleBuffer)
    }
}

// MARK: Vision & CoreML
extension ViewController {
    fileprivate func performPredictionWith(cvPixelBuffer pixelBuffer: CVPixelBuffer) -> Future<String, Error> {
        return Future() { promise in
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .upMirrored)
            do {
                guard let mlModel = self.mlModel else {
                    promise(.failure(SightError.failToLoadModel))
                    return
                }
                
                // set up the vision request using the model
                let classificationRequest = VNCoreMLRequest(model: mlModel,
                                                            completionHandler: { request, error in
                    
                    guard let observations = request.results as? [VNClassificationObservation]  else {
                        promise(.failure(SightError.unexpectedType))
                        return
                    }
                    guard let best = observations.first else {
                        promise(.failure(SightError.noBestResult))
                        return
                    }
                    
                    promise(.success(best.identifier))
                    
                })
                classificationRequest.imageCropAndScaleOption = .centerCrop
                
                try imageRequestHandler.perform([classificationRequest])
                
            } catch {
                promise(.failure(SightError.predictionFailed(error: error.localizedDescription)))
            }
        }
    }
}

