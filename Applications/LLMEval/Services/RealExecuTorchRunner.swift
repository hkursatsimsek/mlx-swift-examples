// Copyright © 2025 Apple Inc.

#if canImport(ExecuTorchLLM)

    import ExecuTorchLLM
    import Foundation

    /// Real ``ExecuTorchRunner`` backed by ExecuTorch's `TextLLMRunner`.
    ///
    /// Compiled only when the `executorch_llm` SwiftPM product (module
    /// `ExecuTorchLLM`) is linked into the LLMEval target — see the package wiring
    /// in `project.pbxproj` (`https://github.com/pytorch/executorch.git`, branch
    /// `swiftpm-1.3.1`, products: `executorch`, `executorch_llm`, `backend_xnnpack`,
    /// `kernels_llm`, `kernels_optimized`, plus the `-all_load` linker flag).
    ///
    /// Marked `@unchecked Sendable`: it wraps the Objective-C-bridged `TextLLMRunner`
    /// (not `Sendable`). `load(...)` completes before `generate(...)` is invoked, and
    /// generation runs on a single detached task, so the wrapped runner is never used
    /// concurrently.
    final class RealExecuTorchRunner: ExecuTorchRunner, @unchecked Sendable {

        let isMock = false
        let backendName = "XNNPACK"

        private let sequenceLength: Int
        private let maxNewTokens: Int
        private let temperature: Double

        @available(*, deprecated, message: "Wraps ExecuTorch's experimental LLM API")
        private var runner: TextRunner?

        // A lower temperature keeps a small, heavily-quantized model more focused
        // and reduces (but does not eliminate) made-up answers.
        init(sequenceLength: Int = 2048, maxNewTokens: Int = 512, temperature: Double = 0.6) {
            self.sequenceLength = sequenceLength
            self.maxNewTokens = maxNewTokens
            self.temperature = temperature
        }

        @available(*, deprecated, message: "Wraps ExecuTorch's experimental LLM API")
        func load(modelURL: URL, tokenizerURL: URL, specialTokens: [String]) async throws {
            let runner = TextRunner(
                modelPath: modelURL.path,
                tokenizerPath: tokenizerURL.path,
                specialTokens: specialTokens
            )
            try runner.load()
            self.runner = runner
        }

        @available(*, deprecated, message: "Wraps ExecuTorch's experimental LLM API")
        func generate(prompt: String) -> AsyncThrowingStream<String, Error> {
            AsyncThrowingStream { continuation in
                let task = Task.detached { [self] in
                    guard let runner = self.runner else {
                        continuation.finish(throwing: ExecuTorchRunnerError.notLoaded)
                        return
                    }
                    do {
                        try runner.generate(
                            prompt,
                            Config {
                                $0.temperature = self.temperature
                                $0.sequenceLength = self.sequenceLength
                                $0.maximumNewTokens = self.maxNewTokens
                                // Stream only the model's answer, not the prompt.
                                // ExecuTorch's GenerationConfig echoes the prompt by
                                // default, which dumps the whole chat template into
                                // the output view.
                                $0.isEchoEnabled = false
                            }
                        ) { token in
                            continuation.yield(token)
                        }
                        continuation.finish()
                    } catch {
                        continuation.finish(throwing: error)
                    }
                }
                continuation.onTermination = { [self] _ in
                    self.runner?.stop()
                    task.cancel()
                }
            }
        }
    }

    enum ExecuTorchRunnerError: LocalizedError {
        case notLoaded

        var errorDescription: String? {
            switch self {
            case .notLoaded: return "ExecuTorch modeli henüz yüklenmedi"
            }
        }
    }

#endif
