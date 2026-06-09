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

    /// Selects the real ExecuTorch runner when the `ExecuTorchLLM` framework is
    /// linked (Phase 2), otherwise falls back to the mock so the app always builds
    /// and runs even without the binary dependency.
    static func makeRunner() -> ExecuTorchRunner {
        #if canImport(ExecuTorchLLM)
            return RealExecuTorchRunner()
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

            // 1) The `.pte` program from the model repo.
            let progressHandler: @Sendable (Progress) -> Void = { [weak self] progress in
                let fraction = progress.fractionCompleted
                Task { @MainActor in
                    guard let self else { return }
                    self.downloadProgress = fraction
                    self.progressDescription = "\(Int(fraction * 100))%"
                }
            }

            let modelDir = try await hub.snapshot(
                from: Hub.Repo(id: model.repoId),
                matching: [model.pteGlob],
                progressHandler: progressHandler
            )

            // 2) The tokenizer, possibly from a different (vocab-compatible) repo.
            let tokenizerDir = try await hub.snapshot(
                from: Hub.Repo(id: model.effectiveTokenizerRepoId),
                matching: [model.tokenizerGlob],
                progressHandler: progressHandler
            )

            let modelURL = try Self.resolveFile(in: modelDir, glob: model.pteGlob)
            let tokenizerURL = try Self.resolveFile(in: tokenizerDir, glob: model.tokenizerGlob)

            modelInfo = "\(model.shortName) yükleniyor..."
            downloadProgress = nil
            progressDescription = nil

            try await runner.load(
                modelURL: modelURL,
                tokenizerURL: tokenizerURL,
                specialTokens: model.specialTokens)

            loadState = .loaded(modelURL)
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
        // Apply the model's chat template so instruction-tuned models answer the
        // prompt instead of merely continuing the text.
        let currentPrompt = model.formattedPrompt(prompt)
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

    /// Resolves a single downloaded file inside a snapshot directory. `glob` is
    /// either an extension pattern (`*.pte`) or a literal filename
    /// (`tokenizer.json`); the search recurses so nested snapshots still match.
    nonisolated static func resolveFile(in directory: URL, glob: String) throws -> URL {
        let fileManager = FileManager.default
        let all = fileManager.enumerator(
            at: directory, includingPropertiesForKeys: nil)?
            .compactMap { $0 as? URL } ?? []

        if glob.hasPrefix("*.") {
            let ext = String(glob.dropFirst(2))
            if let match = all.first(where: { $0.pathExtension == ext }) { return match }
        } else if let match = all.first(where: { $0.lastPathComponent == glob }) {
            return match
        }
        throw ExecuTorchLoadError.missingFile(glob: glob, directory: directory)
    }
}

/// Errors surfaced while preparing an ExecuTorch model for the experimental tab.
/// Defined outside the `#if canImport(ExecuTorchLLM)` guard so the mock build can
/// reference it too.
enum ExecuTorchLoadError: LocalizedError {
    case missingFile(glob: String, directory: URL)

    var errorDescription: String? {
        switch self {
        case .missingFile(let glob, _):
            return "İndirilen dosyalar arasında '\(glob)' ile eşleşen bulunamadı"
        }
    }
}
