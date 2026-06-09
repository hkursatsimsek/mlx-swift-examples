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

    var id: String { repoId }

    var shortName: String { repoId.components(separatedBy: "/").last ?? repoId }

    static let available: [ExecuTorchModel] = [
        ExecuTorchModel(
            repoId: "executorch-community/Qwen2.5-0.5B-Instruct-ExecuTorch",
            pteGlob: "*.pte",
            tokenizerGlob: "tokenizer*",
            displayName: "Qwen2.5 0.5B Instruct",
            size: "~0.5 GB"
        ),
        ExecuTorchModel(
            repoId: "executorch-community/Llama-3.2-1B-Instruct-ExecuTorch",
            pteGlob: "*.pte",
            tokenizerGlob: "tokenizer*",
            displayName: "Llama 3.2 1B Instruct",
            size: "~1.2 GB"
        ),
    ]
}
