// Copyright © 2025 Apple Inc.

import Accelerate
import Foundation
import Hub
import HuggingFace
import MLX
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import MLXLLM
import PDFKit
import SwiftUI
import Tokenizers
import UniformTypeIdentifiers

@Observable
@MainActor
class RAGEvaluator {

    // MARK: - Embedder state

    enum EmbedderState {
        case idle
        case loading
        case loaded(EmbedderModelContainer)
    }

    var embedderState = EmbedderState.idle
    var embedderInfo = ""
    var embedderDownloadProgress: Double?

    // MARK: - Index state

    var index = RAGVectorIndex()
    var documentCount = 0
    var isIndexing = false
    var indexingProgress: Double = 0
    var indexError: String?
    var corpusName = ""

    // MARK: - Query state

    var query = ""
    var searchResults: [RAGSearchResult] = []
    var isSearching = false

    // MARK: - Generation state

    var llm: LLMEvaluator
    var answer = ""
    var isGenerating = false
    var generationError: String?

    // MARK: - File picker state

    var isSelectingFolder = false

    static let embedderConfig = EmbedderRegistry.minilm_l6

    static let bundledDocNames = [
        "izin_politikasi", "maas_politikasi", "calisma_saatleri", "ise_alim",
        "mlx-overview", "swift-concurrency", "llm-quantization", "transformer-attention",
    ]

    init(llm: LLMEvaluator) {
        self.llm = llm
    }

    // MARK: - Embedder loading

    func loadEmbedder() async throws -> EmbedderModelContainer {
        while true {
            switch embedderState {
            case .idle:
                return try await performEmbedderLoad()
            case .loading:
                try await Task.sleep(for: .milliseconds(100))
            case .loaded(let container):
                return container
            }
        }
    }

    private func performEmbedderLoad() async throws -> EmbedderModelContainer {
        embedderState = .loading
        embedderInfo = "Embedding modeli indiriliyor..."
        embedderDownloadProgress = 0.0

        do {
            let downloader = #hubDownloader()
            let tokenizerLoader = #huggingFaceTokenizerLoader()

            let container = try await EmbedderModelFactory.shared.loadContainer(
                from: downloader,
                using: tokenizerLoader,
                configuration: Self.embedderConfig
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.embedderDownloadProgress = progress.fractionCompleted
                    let pct = Int(progress.fractionCompleted * 100)
                    self?.embedderInfo = "Embedding modeli indiriliyor (\(pct)%)"
                }
            }

            embedderInfo = "all-MiniLM-L6-v2 hazır"
            embedderDownloadProgress = nil
            embedderState = .loaded(container)
            return container
        } catch {
            embedderState = .idle
            embedderDownloadProgress = nil
            throw error
        }
    }

    // MARK: - Corpus selection

    func selectCorpusFolder() {
        #if os(macOS)
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.folder, .text, .pdf]
        panel.prompt = "Kaynak Seç"

        if panel.runModal() == .OK, !panel.urls.isEmpty {
            Task { await handleSelectedURLs(panel.urls) }
        }
        #else
        isSelectingFolder = true
        #endif
    }

    func handleSelectedURLs(_ urls: [URL]) async {
        indexError = nil
        var documents: [RAGDocument] = []
        var displayName: String?

        for url in urls {
            let needsScope = url.startAccessingSecurityScopedResource()

            let isDir =
                (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            if isDir {
                do {
                    let folderDocs = try loadDocuments(from: url)
                    documents.append(contentsOf: folderDocs)
                    if displayName == nil { displayName = url.lastPathComponent }
                } catch {
                    indexError = error.localizedDescription
                }
            } else if let doc = loadDocument(from: url) {
                documents.append(doc)
            }

            if needsScope { url.stopAccessingSecurityScopedResource() }
        }

        guard !documents.isEmpty else {
            if indexError == nil {
                indexError =
                    "Seçilen kaynaklarda işlenebilir doküman (.txt, .md, .pdf) bulunamadı."
            }
            return
        }

        corpusName =
            displayName
            ?? (urls.count == 1
                ? urls[0].lastPathComponent : "\(documents.count) doküman")
        await buildIndexFromDocuments(documents)
    }

    func useBundledDocs(selectedNames: [String]) {
        let docs = selectedNames.compactMap { name -> RAGDocument? in
            guard let url = Bundle.main.url(forResource: name, withExtension: "txt"),
                let contents = try? String(contentsOf: url, encoding: .utf8),
                !contents.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { return nil }
            return RAGDocument(name: "\(name).txt", contents: contents)
        }
        guard !docs.isEmpty else {
            indexError = "Örnek dokümanlar uygulama paketinde bulunamadı."
            return
        }
        corpusName = "Örnek Dokümanlar"
        Task { await buildIndexFromDocuments(docs) }
    }

    // MARK: - Indexing

    func buildIndexFromDocuments(_ documents: [RAGDocument]) async {
        guard !isIndexing else { return }

        isIndexing = true
        indexingProgress = 0
        indexError = nil
        index = RAGVectorIndex()
        documentCount = 0
        searchResults = []
        answer = ""

        do {
            let container = try await loadEmbedder()

            let chunks = documents.flatMap { RAGChunker.chunk($0) }
            guard !chunks.isEmpty else {
                indexError = "Belgelerden geçerli parça çıkarılamadı."
                isIndexing = false
                return
            }

            let batchSize = 8
            var newIndex = RAGVectorIndex()
            var processed = 0

            var batchStart = 0
            while batchStart < chunks.count {
                let batchEnd = min(batchStart + batchSize, chunks.count)
                let batch = Array(chunks[batchStart..<batchEnd])

                let vectors = try await embedTexts(batch.map { $0.text }, container: container)

                for (i, vector) in vectors.enumerated() {
                    guard batch.indices.contains(i) else { continue }
                    let normalized = normalizeVector(vector)
                    newIndex.add(RAGIndexEntry(chunk: batch[i], embedding: normalized))
                }

                processed += batch.count
                indexingProgress = Double(processed) / Double(chunks.count)
                batchStart = batchEnd
            }

            self.index = newIndex
            documentCount = documents.count

        } catch {
            indexError = error.localizedDescription
        }

        isIndexing = false
        indexingProgress = 0
    }

    // MARK: - Search + Generation

    func runRAGQuery() {
        guard !query.isEmpty, documentCount > 0 else { return }
        let currentQuery = query

        Task {
            isSearching = true
            answer = ""
            generationError = nil
            searchResults = []

            do {
                let container = try await loadEmbedder()
                let queryVectors = try await embedTexts([currentQuery], container: container)

                guard let rawVector = queryVectors.first else {
                    isSearching = false
                    return
                }

                let normalizedQuery = normalizeVector(rawVector)
                searchResults = index.search(query: normalizedQuery, topK: 3)
                isSearching = false

                guard !searchResults.isEmpty else {
                    answer = "İlgili doküman bulunamadı."
                    return
                }

                await generateAnswer(query: currentQuery, results: searchResults)

            } catch {
                isSearching = false
                generationError = error.localizedDescription
            }
        }
    }

    private func generateAnswer(query: String, results: [RAGSearchResult]) async {
        guard !llm.running else {
            generationError = "Evaluate sekmesinde üretim devam ediyor, lütfen bekleyin."
            return
        }

        isGenerating = true
        answer = ""

        let context = results.enumerated().map { i, r in
            let label = r.chunkLabel.isEmpty ? "" : " (\(r.chunkLabel))"
            return "[\(i + 1)] \(r.documentName)\(label):\n\(r.chunk.text)"
        }.joined(separator: "\n\n---\n\n")

        let systemPrompt = """
            Sen yalnızca verilen kaynak dokümanları kullanarak yanıt veren bir asistansın. \
            Yorum, varsayım, çıkarım veya genel bilgi kullanamazsın. \
            Kaynakta açıkça yazmayan hiçbir bilgiyi cevaba ekleyemezsin.
            """

        let ragPrompt = """
            Aşağıdaki kaynaklardan yararlanarak soruyu yanıtla.

            KURALLAR:
            1. ÖNCE: Sorunun cevabını içeren cümleyi kaynaklardan AYNEN alıntıla.
               Format: > "[alıntı]" (Kaynak: [n] [doküman_adı])
            2. SONRA: Bu alıntıya dayanarak kısa ve net cevabı yaz.
            3. Kaynakta açıkça yazmayan bilgiyi çıkarma, yorum yapma, varsayım kurma.
            4. Sayısal/kategorik kuralları en spesifik eşleşmeye göre uygula; sınır durumlarda \
               en alt sınırı sağlayan kuralı seç ve gerekçeyi alıntıyla göster.
            5. Cevap kaynaklarda yoksa: "Verilen kaynaklarda bu sorunun açık bir cevabı bulunmuyor." de.

            === KAYNAKLAR ===
            \(context)

            === SORU ===
            \(query)
            """

        do {
            let container = try await llm.load()

            MLXRandom.seed(UInt64(Date.timeIntervalSinceReferenceDate * 1000))

            let userInput = UserInput(chat: [
                .system(systemPrompt),
                .user(ragPrompt),
            ])

            let lmInput = try await container.prepare(input: userInput)
            let parameters = GenerateParameters(maxTokens: 512, temperature: 0.0)
            let stream = try await container.generate(input: lmInput, parameters: parameters)

            var iterator = stream.makeAsyncIterator()
            while let item = await iterator.next() {
                if let chunk = item.chunk, !chunk.isEmpty {
                    answer += chunk
                }
            }
        } catch {
            generationError = error.localizedDescription
        }

        isGenerating = false
    }

    // MARK: - Private helpers

    private func loadDocuments(from root: URL) throws -> [RAGDocument] {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: root,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        var documents: [RAGDocument] = []

        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: [.isRegularFileKey])
            guard values.isRegularFile == true else { continue }
            guard let contents = readSupportedFile(at: fileURL) else { continue }

            let name = relativeName(for: fileURL, root: root)
            documents.append(RAGDocument(name: name, contents: contents))
            if documents.count >= 200 { break }
        }

        return documents
    }

    private func loadDocument(from fileURL: URL) -> RAGDocument? {
        guard let contents = readSupportedFile(at: fileURL) else { return nil }
        return RAGDocument(name: fileURL.lastPathComponent, contents: contents)
    }

    private func readSupportedFile(at fileURL: URL) -> String? {
        let ext = fileURL.pathExtension.lowercased()
        let raw: String?
        switch ext {
        case "txt", "md":
            raw = try? String(contentsOf: fileURL, encoding: .utf8)
        case "pdf":
            raw = extractPDFText(at: fileURL)
        default:
            return nil
        }
        guard let text = raw,
            !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }
        return text
    }

    private func extractPDFText(at url: URL) -> String? {
        guard let pdf = PDFDocument(url: url) else { return nil }
        var pages: [String] = []
        for i in 0..<pdf.pageCount {
            if let s = pdf.page(at: i)?.string, !s.isEmpty {
                pages.append(s)
            }
        }
        let joined = pages.joined(separator: "\n\n")
        return normalizePDFText(joined)
    }

    private func normalizePDFText(_ text: String) -> String {
        var s = text
        while s.contains("\n\n\n") {
            s = s.replacingOccurrences(of: "\n\n\n", with: "\n\n")
        }
        while s.contains("   ") {
            s = s.replacingOccurrences(of: "   ", with: "  ")
        }
        return s
    }

    private func relativeName(for url: URL, root: URL) -> String {
        let path = url.standardizedFileURL.path
        let base = root.standardizedFileURL.path
        guard path.hasPrefix(base) else { return url.lastPathComponent }
        let remainder = path.dropFirst(base.count)
        return remainder.hasPrefix("/") ? String(remainder.dropFirst()) : String(remainder)
    }

    private func embedTexts(_ texts: [String], container: EmbedderModelContainer) async throws
        -> [[Float]]
    {
        guard !texts.isEmpty else { return [] }

        return try await container.perform { context in
            let tokenizer = context.tokenizer
            let padToken = tokenizer.convertTokenToId("[PAD]") ?? tokenizer.eosTokenId ?? 0

            let encoded = texts.compactMap { text -> [Int]? in
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

            let pooled = context.pooling(output, mask: mask, normalize: true, applyLayerNorm: false)
            pooled.eval()

            return (0..<pooled.shape[0]).map { i in
                pooled[i].asArray(Float.self)
            }
        }
    }

    private func normalizeVector(_ v: [Float]) -> [Float] {
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
