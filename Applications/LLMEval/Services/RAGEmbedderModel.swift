// Copyright © 2025 Apple Inc.

import Accelerate
import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
import Tokenizers

/// How a query/passage should be prefixed before embedding. Several modern
/// retrieval models were trained with asymmetric prompts and lose a large
/// amount of accuracy when the prefixes are omitted.
enum RAGPromptStyle: Sendable {
    /// No prefix (classic sentence-transformers / BGE dense usage).
    case none
    /// E5 family: `"query: …"` for questions, `"passage: …"` for documents.
    case e5
    /// Qwen3-Embedding: an instruction prefix on queries, raw text on passages.
    case qwenInstruct

    func apply(_ text: String, isQuery: Bool) -> String {
        switch self {
        case .none:
            return text
        case .e5:
            return (isQuery ? "query: " : "passage: ") + text
        case .qwenInstruct:
            guard isQuery else { return text }
            return
                "Instruct: Given a search query, retrieve relevant passages that answer it\nQuery: "
                + text
        }
    }
}

/// A selectable embedding model for the RAG tab. Mirrors `LLMEvaluator.EvalModel`
/// so the UI picker can present the same kind of chooser.
struct RAGEmbedderModel: Identifiable, Hashable, Sendable {
    let id: String
    let configuration: ModelConfiguration
    let shortName: String
    let size: String
    /// Whether the model handles non-English text (e.g. Turkish) well.
    let multilingual: Bool
    let promptStyle: RAGPromptStyle
    /// Padding token strings to try, in order, before falling back to EOS.
    /// XLM-RoBERTa based models (e5, bge-m3) use `<pad>`, not `[PAD]`.
    let padTokenCandidates: [String]

    static func == (lhs: RAGEmbedderModel, rhs: RAGEmbedderModel) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

extension RAGEmbedderModel {
    /// all-MiniLM-L6-v2 — fast but English-only. Kept for comparison / backwards
    /// compatibility; it is the reason Turkish retrieval was failing.
    static let minilm = RAGEmbedderModel(
        id: EmbedderRegistry.minilm_l6.name,
        configuration: EmbedderRegistry.minilm_l6,
        shortName: "all-MiniLM-L6-v2",
        size: "~90 MB",
        multilingual: false,
        promptStyle: .none,
        padTokenCandidates: ["[PAD]"]
    )

    /// multilingual-e5-small — small, fast, strong multilingual (incl. Turkish).
    /// Recommended default for Turkish corpora.
    static let multilingualE5Small = RAGEmbedderModel(
        id: EmbedderRegistry.multilingual_e5_small.name,
        configuration: EmbedderRegistry.multilingual_e5_small,
        shortName: "multilingual-e5-small",
        size: "~120 MB",
        multilingual: true,
        promptStyle: .e5,
        padTokenCandidates: ["<pad>"]
    )

    /// bge-m3 — highest multilingual quality, larger download / slower.
    static let bgeM3 = RAGEmbedderModel(
        id: EmbedderRegistry.bge_m3.name,
        configuration: EmbedderRegistry.bge_m3,
        shortName: "bge-m3",
        size: "~1.2 GB",
        multilingual: true,
        promptStyle: .none,
        padTokenCandidates: ["<pad>"]
    )

    /// Qwen3-Embedding-0.6B (4bit) — multilingual, instruction-tuned queries.
    static let qwen3Embedding = RAGEmbedderModel(
        id: EmbedderRegistry.qwen3_embedding.name,
        configuration: EmbedderRegistry.qwen3_embedding,
        shortName: "Qwen3-Embedding-0.6B",
        size: "~600 MB",
        multilingual: true,
        promptStyle: .qwenInstruct,
        padTokenCandidates: ["<|endoftext|>"]
    )

    static let all: [RAGEmbedderModel] = [
        minilm, multilingualE5Small, bgeM3, qwen3Embedding,
    ]
}

/// Shared, isolation-free embedding helper used by both `RAGEvaluator` (live
/// querying/indexing) and `RAGBenchmark` (model comparison). It applies the
/// model's prompt prefix and resolves the correct padding token before pooling.
enum RAGEmbedding {

    /// Embed a batch of texts with the given model/container.
    /// - Parameter isQuery: `true` for questions, `false` for document passages.
    /// - Returns: one L2-normalized vector per non-empty input text.
    static func embed(
        texts: [String],
        isQuery: Bool,
        model: RAGEmbedderModel,
        container: EmbedderModelContainer
    ) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }

        let prepared = texts.map { model.promptStyle.apply($0, isQuery: isQuery) }
        let padCandidates = model.padTokenCandidates

        return try await container.perform { context in
            let tokenizer = context.tokenizer
            let padToken =
                padCandidates.lazy
                .compactMap { tokenizer.convertTokenToId($0) }
                .first ?? tokenizer.eosTokenId ?? 0

            let encoded = prepared.compactMap { text -> [Int]? in
                let tokens = tokenizer.encode(text: text, addSpecialTokens: true)
                return tokens.isEmpty ? nil : tokens
            }
            guard !encoded.isEmpty else { return [[Float]]() }

            let maxLength = encoded.map(\.count).max() ?? 0
            let padded = stacked(
                encoded.map { tokens in
                    MLXArray(tokens + Array(repeating: padToken, count: maxLength - tokens.count))
                })

            let mask = padded .!= MLXArray(padToken)
            let tokenTypes = MLXArray.zeros(like: padded)

            let output = context.model(
                padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask)

            let pooled = context.pooling(
                output, mask: mask, normalize: true, applyLayerNorm: false)
            pooled.eval()

            return (0..<pooled.shape[0]).map { i in
                pooled[i].asArray(Float.self)
            }
        }
    }

    /// L2-normalize a vector, sanitizing non-finite values. Pooling already
    /// normalizes, so this is mostly a safety net for `dot`-product search.
    static func normalize(_ v: [Float]) -> [Float] {
        guard !v.isEmpty else { return [] }
        let sanitized = v.map { $0.isFinite ? $0 : Float(0) }
        var sumSq: Float = 0
        vDSP_svesq(sanitized, 1, &sumSq, vDSP_Length(sanitized.count))
        guard sumSq > 1e-9 else { return sanitized }
        var divisor = sqrt(sumSq)
        var out = [Float](repeating: 0, count: sanitized.count)
        vDSP_vsdiv(sanitized, 1, &divisor, &out, 1, vDSP_Length(sanitized.count))
        return out
    }
}
