// Copyright © 2025 Apple Inc.

import Foundation
import Hub
import HuggingFace
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import SwiftUI
import Tokenizers

/// On-device retrieval benchmark that compares embedding models on the bundled
/// Turkish corpus using a small labeled gold set. For each model it indexes the
/// same documents and measures how often the topically-correct source document
/// is retrieved (hit@k, MRR). This lets the user see, concretely, which model
/// retrieves Turkish content most accurately.
@Observable
@MainActor
final class RAGBenchmark {

    /// A labeled question whose answer belongs to `expectedDoc`.
    struct GoldQuery: Sendable {
        let question: String
        /// Bundled document file name, e.g. "izin_politikasi.txt".
        let expectedDoc: String
    }

    struct ModelResult: Identifiable {
        let id = UUID()
        let model: RAGEmbedderModel
        let hitAt1: Double
        let hitAt3: Double
        let hitAt5: Double
        let mrr: Double
        let avgTopScore: Double
        let failure: String?
    }

    var isRunning = false
    var progress: Double = 0
    var progressText = ""
    var results: [ModelResult] = []
    var error: String?

    /// Topic-robust gold set over the 8 bundled documents. The expected source is
    /// determined by topic, so it stays valid regardless of exact wording.
    static let goldQueries: [GoldQuery] = [
        .init(question: "Yıllık izin hakkı kaç gündür?", expectedDoc: "izin_politikasi.txt"),
        .init(question: "Doğum izni ne kadar sürer?", expectedDoc: "izin_politikasi.txt"),
        .init(question: "Hastalık izni için rapor gerekir mi?", expectedDoc: "izin_politikasi.txt"),
        .init(question: "Maaş ödemeleri ayın hangi günü yapılır?", expectedDoc: "maas_politikasi.txt"),
        .init(question: "Maaş zammı nasıl belirlenir?", expectedDoc: "maas_politikasi.txt"),
        .init(question: "Haftalık çalışma saati kaçtır?", expectedDoc: "calisma_saatleri.txt"),
        .init(question: "Mesai başlangıç saati nedir?", expectedDoc: "calisma_saatleri.txt"),
        .init(question: "İşe alım süreci kaç aşamadan oluşur?", expectedDoc: "ise_alim.txt"),
        .init(question: "Mülakat sonrası geri dönüş ne zaman yapılır?", expectedDoc: "ise_alim.txt"),
        .init(question: "MLX nedir ve ne işe yarar?", expectedDoc: "mlx-overview.txt"),
        .init(question: "Swift'te async/await nasıl kullanılır?", expectedDoc: "swift-concurrency.txt"),
        .init(question: "Actor nedir?", expectedDoc: "swift-concurrency.txt"),
        .init(question: "LLM quantization neden yapılır?", expectedDoc: "llm-quantization.txt"),
        .init(question: "4-bit kuantizasyon nedir?", expectedDoc: "llm-quantization.txt"),
        .init(
            question: "Transformer'da attention mekanizması nasıl çalışır?",
            expectedDoc: "transformer-attention.txt"),
        .init(question: "Self-attention nedir?", expectedDoc: "transformer-attention.txt"),
    ]

    func run(models: [RAGEmbedderModel]) {
        guard !isRunning else { return }
        Task { await runAsync(models) }
    }

    private func runAsync(_ models: [RAGEmbedderModel]) async {
        isRunning = true
        error = nil
        results = []
        progress = 0

        let documents = Self.loadBundledDocuments()
        guard !documents.isEmpty else {
            error = "Örnek dokümanlar uygulama paketinde bulunamadı."
            isRunning = false
            return
        }

        let chunks = documents.flatMap { RAGChunker.chunk($0) }
        let totalSteps = max(models.count, 1)

        for (modelIndex, model) in models.enumerated() {
            progressText = "\(model.shortName) değerlendiriliyor…"
            do {
                let result = try await evaluate(model: model, chunks: chunks)
                results.append(result)
            } catch {
                results.append(
                    ModelResult(
                        model: model, hitAt1: 0, hitAt3: 0, hitAt5: 0, mrr: 0, avgTopScore: 0,
                        failure: error.localizedDescription))
            }
            progress = Double(modelIndex + 1) / Double(totalSteps)
        }

        progressText = "Tamamlandı"
        isRunning = false
    }

    private func evaluate(model: RAGEmbedderModel, chunks: [RAGChunk]) async throws -> ModelResult {
        let downloader = #hubDownloader()
        let tokenizerLoader = #huggingFaceTokenizerLoader()

        let container = try await EmbedderModelFactory.shared.loadContainer(
            from: downloader,
            using: tokenizerLoader,
            configuration: model.configuration
        ) { [weak self] progress in
            Task { @MainActor in
                let pct = Int(progress.fractionCompleted * 100)
                self?.progressText = "\(model.shortName) indiriliyor (\(pct)%)"
            }
        }

        // Build the index for this model.
        var index = RAGVectorIndex()
        let batchSize = 8
        var start = 0
        while start < chunks.count {
            let end = min(start + batchSize, chunks.count)
            let batch = Array(chunks[start..<end])
            let vectors = try await RAGEmbedding.embed(
                texts: batch.map { $0.text }, isQuery: false, model: model, container: container)
            for (i, vector) in vectors.enumerated() where batch.indices.contains(i) {
                index.add(RAGIndexEntry(chunk: batch[i], embedding: RAGEmbedding.normalize(vector)))
            }
            start = end
        }

        // Run the gold queries.
        var reciprocalRanks: [Double] = []
        var hits1 = 0, hits3 = 0, hits5 = 0
        var topScores: [Double] = []

        for gold in Self.goldQueries {
            let queryVectors = try await RAGEmbedding.embed(
                texts: [gold.question], isQuery: true, model: model, container: container)
            guard let raw = queryVectors.first else { continue }
            let results = index.search(query: RAGEmbedding.normalize(raw), topK: 5)
            if let top = results.first { topScores.append(Double(top.score)) }

            let rank = results.firstIndex { $0.documentName == gold.expectedDoc }
            if let rank {
                reciprocalRanks.append(1.0 / Double(rank + 1))
                if rank < 1 { hits1 += 1 }
                if rank < 3 { hits3 += 1 }
                if rank < 5 { hits5 += 1 }
            } else {
                reciprocalRanks.append(0)
            }
        }

        let n = Double(max(Self.goldQueries.count, 1))
        return ModelResult(
            model: model,
            hitAt1: Double(hits1) / n,
            hitAt3: Double(hits3) / n,
            hitAt5: Double(hits5) / n,
            mrr: reciprocalRanks.reduce(0, +) / n,
            avgTopScore: topScores.isEmpty ? 0 : topScores.reduce(0, +) / Double(topScores.count),
            failure: nil
        )
    }

    private static func loadBundledDocuments() -> [RAGDocument] {
        RAGEvaluator.bundledDocNames.compactMap { name -> RAGDocument? in
            guard let url = Bundle.main.url(forResource: name, withExtension: "txt"),
                let contents = try? String(contentsOf: url, encoding: .utf8),
                !contents.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { return nil }
            return RAGDocument(name: "\(name).txt", contents: contents)
        }
    }
}
