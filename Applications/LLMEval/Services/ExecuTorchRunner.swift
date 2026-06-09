// Copyright © 2025 Apple Inc.

import Foundation

/// Abstraction (the integration *seam*) over an ExecuTorch inference engine.
///
/// Phase 1 ships only ``MockExecuTorchRunner`` so the whole experimental tab —
/// dynamic `.pte` download, streaming UI and metrics — works end-to-end **without
/// adding any binary dependency**. Phase 2 adds a real conformer behind
/// `#if canImport(ExecuTorch)` (see ``ExecuTorchEvaluator/makeRunner()``); this
/// protocol is the single point that has to change.
protocol ExecuTorchRunner: Sendable {

    /// Whether this runner is the mock stand-in (vs a real ExecuTorch engine).
    var isMock: Bool { get }

    /// Backend label shown in the UI (e.g. `Mock`, `XNNPACK`, `Core ML`).
    var backendName: String { get }

    /// Load a `.pte` program + tokenizer from a previously downloaded directory.
    /// `specialTokens` is used by tiktoken-style tokenizers (e.g. Llama 3); it can
    /// be empty for tokenizers that already embed their special tokens (HF json).
    func load(modelDirectory: URL, specialTokens: [String]) async throws

    /// Stream generated text. Tokens are delivered in order; the stream finishes
    /// when generation completes, throws on error, or is cancelled.
    func generate(prompt: String) -> AsyncThrowingStream<String, Error>
}

/// A stand-in engine that simulates streaming output and plausible timing.
///
/// It lets the experimental tab exercise the full flow (real Hugging Face download
/// → UI → metrics) before the real ExecuTorch framework is linked. Replace with a
/// real ``ExecuTorchRunner`` conformer in Phase 2.
final class MockExecuTorchRunner: ExecuTorchRunner {

    let isMock = true
    let backendName = "Mock"

    func load(modelDirectory: URL, specialTokens: [String]) async throws {
        // Simulate engine warm-up so the loading overlay is exercised.
        try await Task.sleep(for: .milliseconds(400))
    }

    func generate(prompt: String) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                let response = """
                    [MOCK ExecuTorch] Bu yanıt, gerçek ExecuTorch motoru bağlanana kadar \
                    akışı ve metrikleri göstermek için üretildi. Sorunuz: "\(prompt)". \
                    Faz 2'de bu metin, indirilen .pte modelinden gerçek zamanlı gelecek.
                    """
                do {
                    for word in response.split(
                        separator: " ", omittingEmptySubsequences: false)
                    {
                        try Task.checkCancellation()
                        try await Task.sleep(for: .milliseconds(45))
                        continuation.yield(String(word) + " ")
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
