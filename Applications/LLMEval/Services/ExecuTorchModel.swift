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

    /// Optional separate HF repo to fetch the tokenizer from. Use this when the
    /// model repo only ships a format ExecuTorch's loader can't read (e.g. a
    /// tiktoken `tokenizer.model`) and a vocab-compatible HF `tokenizer.json` lives
    /// in another (non-gated) mirror. `nil` → the tokenizer is fetched from `repoId`.
    let tokenizerRepoId: String?

    /// Glob matching the tokenizer file inside ``tokenizerRepoId`` (or ``repoId``).
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
        /// ChatML format used by Qwen3 (and Qwen2) models.
        case qwen3
        /// Phi-4 Mini chat format (`<|system|>…<|end|>`).
        case phi4
    }

    var id: String { repoId }

    var shortName: String { repoId.components(separatedBy: "/").last ?? repoId }

    /// Wraps a user prompt according to ``promptStyle`` so instruction-tuned models
    /// respond as a chat assistant instead of merely continuing the text.
    func formattedPrompt(
        _ userPrompt: String, system: String = "Sen yardımsever bir Türkçe asistansın. Her zaman Türkçe yanıt ver."
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
        case .qwen3:
            // ChatML format; `/no_think` suffix disables the chain-of-thought
            // scratchpad so the model responds directly (faster, smaller output).
            return "<|im_start|>system\n\(system)<|im_end|>\n<|im_start|>user\n\(userPrompt)<|im_end|>\n<|im_start|>assistant\n"
        case .phi4:
            return "<|system|>\n\(system)<|end|>\n<|user|>\n\(userPrompt)<|end|>\n<|assistant|>\n"
        }
    }

    static let available: [ExecuTorchModel] = [
        // Recommended for real answers: instruction-tuned, uses the Llama 3 chat
        // template. Use the model repo's *own* tiktoken `tokenizer.model` — the one
        // the `.pte` was exported against. ExecuTorch routes it through its native
        // tiktoken tokenizer (the path Meta tests for this model). A vocab-identical
        // HF `tokenizer.json` mirror tokenizes regular text differently here and
        // yields nonsense output, so we pass the explicit special-token list instead.
        ExecuTorchModel(
            repoId: "executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET",
            pteGlob: "*.pte",
            tokenizerRepoId: nil,
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
            tokenizerRepoId: nil,
            tokenizerGlob: "tokenizer.json",
            displayName: "SmolLM2 135M (base)",
            size: "~270 MB",
            specialTokens: [],
            promptStyle: .raw
        ),

        // Strong reasoning model from the PyTorch team. INT8 embeddings + INT4
        // weights give a good speed/quality balance. Runs at ~14.8 tok/s on
        // iPhone 15 Pro (~3.4 GB). Tokenizer.json ships in the same repo.
        ExecuTorchModel(
            repoId: "pytorch/Qwen3-4B-INT8-INT4",
            pteGlob: "model.pte",
            tokenizerRepoId: nil,
            tokenizerGlob: "tokenizer.json",
            displayName: "Qwen3 4B (INT8+INT4)",
            size: "~3.4 GB",
            specialTokens: [],
            promptStyle: .qwen3
        ),

        // Microsoft Phi-4 Mini quantized by the PyTorch team. Fastest of the
        // larger models (~17.3 tok/s on iPhone 15 Pro, ~3.2 GB). Excellent at
        // instruction following and reasoning for its size. Tokenizer.json ships
        // in the same repo; specialTokens not required for HF tokenizer format.
        ExecuTorchModel(
            repoId: "pytorch/Phi-4-mini-instruct-INT8-INT4",
            pteGlob: "model.pte",
            tokenizerRepoId: nil,
            tokenizerGlob: "tokenizer.json",
            displayName: "Phi-4 Mini Instruct (INT8+INT4)",
            size: "~3.2 GB",
            specialTokens: [],
            promptStyle: .phi4
        ),

        // NOTE: All Gemma variants (2B / 3B / 4B, any org) are gated on HF
        // because Google's Gemma license requires an accepted agreement.
        // They cannot be downloaded without a user-scoped HF token, which
        // this app does not support. Use Qwen3 or Phi-4 Mini instead.
    ]

    /// Repo the tokenizer is actually downloaded from.
    var effectiveTokenizerRepoId: String { tokenizerRepoId ?? repoId }

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
        // Meta's `tokenizer.py` lays these out as: the 11 named tokens above (ids
        // 128000–128010, so `<|python_tag|>` is 128010), then reserved_special_token
        // 2…246 filling 128011–128255. Keep this exact order — the ids must line up
        // with the model's embedding table.
        let reserved = (0 ..< (256 - base.count)).map {
            "<|reserved_special_token_\($0 + 2)|>"
        }
        return base + reserved
    }()
}
