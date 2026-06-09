// Copyright © 2025 Apple Inc.

import Foundation
import Hub
import HuggingFace
import SwiftUI

/// View model for the experimental **ExecuTorch** tab.
///
/// This is deliberately isolated from ``LLMEvaluator`` (the MLX flow) so the
/// existing tabs are never affected. It downloads a `.pte` + tokenizer from the
/// Hugging Face Hub using the same infrastructure as the MLX models, then drives
/// an ``ExecuTorchRunner`` for generation. Phase 1 uses ``MockExecuTorchRunner``.
@Observable
@MainActor
class ExecuTorchEvaluator {

    // MARK: - Model selection

    var model = ExecuTorchModel.available[0] {
        didSet {
            guard model.id != oldValue.id else { return }
            loadState = .idle
            modelInfo = ""
            output = ""
            errorMessage = nil
        }
    }

    // MARK: - UI / generation state

    var prompt = ""
    var output = ""
    var modelInfo = ""
    var running = false
    var errorMessage: String?

    // MARK: - Download progress

    var downloadProgress: Double?
    var progressDescription: String?

    // MARK: - Metrics

    var tokensPerSecond: Double = 0
    var timeToFirstToken: Double = 0
    var totalTokens: Int = 0
    var totalTime: Double = 0

    // MARK: - Load state

    enum LoadState {
        case idle
        case loading
        case loaded(URL)
    }

    var loadState = LoadState.idle

    var isLoading: Bool {
        if case .loading = loadState { return true }
        return false
    }

    var isLoaded: Bool {
        if case .loaded = loadState { return true }
        return false
    }

    // MARK: - Engine

    private let runner: ExecuTorchRunner
    private var generationTask: Task<Void, Never>?

    /// Tracks first-token timing for live metrics.
    private var generationStart: TimeInterval = 0
    private var firstTokenTime: TimeInterval?

    init(runner: ExecuTorchRunner = ExecuTorchEvaluator.makeRunner()) {
        self.runner = runner
    }

    var backendName: String { runner.backendName }
    var isMockRunner: Bool { runner.isMock }

    /// Selects the real ExecuTorch runner once the framework is linked (Phase 2),
    /// otherwise falls back to the mock so the app always builds and runs.
    static func makeRunner() -> ExecuTorchRunner {
        #if canImport(ExecuTorch)
            // Phase 2: return RealExecuTorchRunner() once the framework is added.
            return MockExecuTorchRunner()
        #else
            return MockExecuTorchRunner()
        #endif
    }

    // MARK: - Download + load

    /// Downloads the selected model's `.pte` + tokenizer from Hugging Face and
    /// hands the local directory to the runner. Safe to call repeatedly.
    func loadModel() async {
        guard !isLoading, !isLoaded else { return }
        errorMessage = nil
        loadState = .loading
        modelInfo = "\(model.shortName) indiriliyor..."
        downloadProgress = 0
        progressDescription = nil

        do {
            let hub = HubApi()
            let repo = Hub.Repo(id: model.repoId)
            let directory = try await hub.snapshot(
                from: repo,
                matching: [model.pteGlob, model.tokenizerGlob]
            ) { @Sendable [weak self] progress in
                let fraction = progress.fractionCompleted
                Task { @MainActor in
                    guard let self else { return }
                    self.downloadProgress = fraction
                    self.progressDescription = "\(Int(fraction * 100))%"
                }
            }

            modelInfo = "\(model.shortName) yükleniyor..."
            downloadProgress = nil
            progressDescription = nil

            try await runner.load(modelDirectory: directory)

            loadState = .loaded(directory)
            modelInfo = "\(model.shortName) • \(runner.backendName) backend"
        } catch {
            loadState = .idle
            downloadProgress = nil
            progressDescription = nil
            modelInfo = ""
            errorMessage = error.localizedDescription
        }
    }

    // MARK: - Generation

    func generate() {
        guard !running, !prompt.isEmpty else { return }
        let currentPrompt = prompt
        resetMetrics()
        errorMessage = nil

        generationTask = Task {
            running = true
            defer { running = false }

            if !isLoaded {
                await loadModel()
            }
            guard isLoaded else { return }

            generationStart = Date.timeIntervalSinceReferenceDate
            firstTokenTime = nil

            do {
                for try await token in runner.generate(prompt: currentPrompt) {
                    appendToken(token)
                }
            } catch is CancellationError {
                // Expected when the user taps Stop.
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    func cancel() {
        generationTask?.cancel()
        generationTask = nil
        running = false
    }

    // MARK: - Helpers

    private func appendToken(_ token: String) {
        let now = Date.timeIntervalSinceReferenceDate
        if firstTokenTime == nil {
            firstTokenTime = now
            timeToFirstToken = (now - generationStart) * 1000  // ms
        }
        output += token
        totalTokens += 1
        let elapsed = now - (firstTokenTime ?? generationStart)
        if elapsed > 0 {
            tokensPerSecond = Double(totalTokens) / elapsed
            totalTime = elapsed
        }
    }

    private func resetMetrics() {
        output = ""
        totalTokens = 0
        tokensPerSecond = 0
        timeToFirstToken = 0
        totalTime = 0
    }
}
