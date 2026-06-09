// Copyright © 2025 Apple Inc.

import Foundation

/// A selectable ExecuTorch model hosted on the Hugging Face Hub.
///
/// The `.pte` program and tokenizer are downloaded dynamically at runtime
/// (see ``ExecuTorchEvaluator/loadModel()``), mirroring how the MLX flow fetches
/// safetensors — no manual bundling of model files is required.
///
/// > Note: The repo ids below point at the `executorch-community` org, which
/// > publishes iOS-compatible `.pte` programs. Verify a given repo actually hosts
/// > a `.pte` exported with a backend you intend to run before relying on it.
struct ExecuTorchModel: Identifiable, Hashable {

    /// Hugging Face repo id, e.g. `executorch-community/Llama-3.2-1B-Instruct-ExecuTorch`.
    let repoId: String

    /// Glob matching the ExecuTorch program file inside the repo.
    let pteGlob: String

    /// Glob matching the tokenizer file(s) inside the repo.
    let tokenizerGlob: String

    /// Human-readable name shown in the picker.
    let displayName: String

    /// Approximate on-disk download size.
    let size: String

    /// Special tokens passed to the ExecuTorch text runner. Required for
    /// tiktoken-style tokenizers (e.g. Llama 3); leave empty for tokenizers that
    /// already embed their special tokens (HF `tokenizer.json`).
    ///
    /// > Note: For a Llama-3 `tokenizer.model` (tiktoken) the full ordered list of
    /// > special/reserved tokens must be supplied or generation will misbehave.
    /// > Fill this in per model when validating real inference on device.
    let specialTokens: [String]

    /// How the user prompt should be wrapped before being sent to the model.
    let promptStyle: PromptStyle

    enum PromptStyle: Sendable {
        /// Send the raw prompt unchanged (base models / no chat template).
        case raw
        /// Wrap with the Llama 3 chat template.
        case llama3
    }

    var id: String { repoId }

    var shortName: String { repoId.components(separatedBy: "/").last ?? repoId }

    /// Wraps a user prompt according to ``promptStyle`` so instruction-tuned models
    /// respond as a chat assistant instead of merely continuing the text.
    func formattedPrompt(
        _ userPrompt: String, system: String = "You are a helpful assistant."
    ) -> String {
        switch promptStyle {
        case .raw:
            return userPrompt
        case .llama3:
            return """
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                \(system)<|eot_id|><|start_header_id|>user<|end_header_id|>

                \(userPrompt)<|eot_id|><|start_header_id|>assistant<|end_header_id|>


                """
        }
    }

    static let available: [ExecuTorchModel] = [
        // Recommended for real answers: instruction-tuned, uses the Llama 3 chat
        // template + the explicit tiktoken special-token list. Validate on device.
        ExecuTorchModel(
            repoId: "executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET",
            pteGlob: "*.pte",
            tokenizerGlob: "tokenizer.model",
            displayName: "Llama 3.2 1B Instruct (INT4)",
            size: "~1.1 GB",
            specialTokens: ExecuTorchModel.llama3SpecialTokens,
            promptStyle: .llama3
        ),
        // Tiny smoke-test model. It is a 135M *base* model (not instruction-tuned),
        // so it continues text rather than answering questions — expect low quality.
        ExecuTorchModel(
            repoId: "executorch-community/SmolLM2-135M",
            pteGlob: "*.pte",
            tokenizerGlob: "tokenizer.json",
            displayName: "SmolLM2 135M (base)",
            size: "~270 MB",
            specialTokens: [],
            promptStyle: .raw
        ),
    ]

    /// The canonical 256-entry Llama-3 special-token list (Meta `tokenizer.py`
    /// construction). Required for the tiktoken `tokenizer.model`; verify against
    /// the exact model revision when validating real inference on device.
    static let llama3SpecialTokens: [String] = {
        let base = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",
            "<|eot_id|>",
            "<|python_tag|>",
        ]
        let reserved = (0 ..< (256 - base.count)).map {
            "<|reserved_special_token_\($0 + 2)|>"
        }
        return Array(base.dropLast()) + reserved + [base[base.count - 1]]
    }()
}
