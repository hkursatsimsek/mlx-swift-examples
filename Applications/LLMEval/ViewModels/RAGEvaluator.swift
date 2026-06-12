// Copyright © 2025 Apple Inc.

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

    /// One summary per indexed collection (namespace). Drives the corpus list UI
    /// and the query-scope picker. Kept in sync with `index` and disk.
    var collections: [RAGCollectionInfo] = []

    // MARK: - Query state

    var query = ""
    var searchResults: [RAGSearchResult] = []
    var isSearching = false

    /// Which collection a search is restricted to. `nil` = search every
    /// collection ("Tümü").
    var queryScope: String?

    // MARK: - Generation state

    var llm: LLMEvaluator
    var answer = ""
    var isGenerating = false
    var generationError: String?

    // MARK: - File picker state

    var isSelectingFolder = false

    // MARK: - Embedder selection

    /// Available embedding models the user can pick from. Defaults to a
    /// multilingual model so Turkish corpora work out of the box.
    static let availableEmbedders = RAGEmbedderModel.all

    /// The selected embedding model. Changing it invalidates the current index
    /// (different models produce incompatible vector spaces), so the corpus must
    /// be re-indexed afterwards.
    var selectedEmbedder: RAGEmbedderModel = .multilingualE5Small {
        didSet {
            guard selectedEmbedder.id != oldValue.id else { return }
            embedderState = .idle
            embedderInfo = ""
            embedderDownloadProgress = nil
            // Different model = incompatible vector space. Drop the live index and
            // instead load any collections that were previously persisted *for the
            // new embedder* (so switching back and forth doesn't force re-indexing).
            index = RAGVectorIndex()
            collections = []
            documentCount = 0
            corpusName = ""
            searchResults = []
            answer = ""
            indexError = nil
            queryScope = nil
            loadPersistedCollections()
        }
    }

    static let bundledDocNames = [
        "izin_politikasi", "maas_politikasi", "calisma_saatleri", "ise_alim",
        "mlx-overview", "swift-concurrency", "llm-quantization", "transformer-attention",
    ]

    init(llm: LLMEvaluator) {
        self.llm = llm
        loadPersistedCollections()
    }

    // MARK: - Persistence

    /// Rebuild the in-memory index from disk for the current embedder. Cheap
    /// (JSON read, no embedding) and idempotent. Vectors are already computed, so
    /// previously indexed corpora are queryable immediately without re-embedding.
    private func loadPersistedCollections() {
        let persisted = RAGIndexStore.loadAll(embedderId: selectedEmbedder.id)
        guard !persisted.isEmpty else { return }

        var rebuilt = RAGVectorIndex()
        var infos: [RAGCollectionInfo] = []
        for pc in persisted {
            for pe in pc.entries {
                let document = RAGDocument(
                    name: pe.chunk.documentName, contents: "", collection: pc.name)
                let chunk = RAGChunk(
                    document: document, text: pe.chunk.text,
                    index: pe.chunk.index, totalChunks: pe.chunk.totalChunks)
                rebuilt.add(RAGIndexEntry(chunk: chunk, embedding: pe.embedding))
            }
            infos.append(
                RAGCollectionInfo(
                    name: pc.name, documentCount: pc.documentCount,
                    chunkCount: pc.entries.count))
        }

        index = rebuilt
        collections = infos
        documentCount = infos.reduce(0) { $0 + $1.documentCount }
        corpusName = infos.count == 1 ? infos[0].name : "\(infos.count) koleksiyon"
    }

    private func persist(
        collection: String, entries: [RAGIndexEntry], documentCount: Int
    ) {
        let persistedEntries = entries.map { entry in
            PersistedEntry(
                chunk: PersistedChunk(
                    documentName: entry.chunk.document.name, text: entry.chunk.text,
                    index: entry.chunk.index, totalChunks: entry.chunk.totalChunks),
                embedding: entry.embedding)
        }
        let payload = PersistedCollection(
            name: collection, embedderId: selectedEmbedder.id,
            documentCount: documentCount, createdAt: Date(), entries: persistedEntries)
        do {
            try RAGIndexStore.save(payload)
        } catch {
            indexError = "İndeks diske kaydedilemedi: \(error.localizedDescription)"
        }
    }

    /// Remove a single collection from memory and disk.
    func deleteCollection(_ name: String) {
        index.removeCollection(name)
        collections.removeAll { $0.name == name }
        documentCount = collections.reduce(0) { $0 + $1.documentCount }
        if queryScope == name { queryScope = nil }
        if collections.isEmpty { corpusName = "" }
        RAGIndexStore.delete(name: name, embedderId: selectedEmbedder.id)
    }

    /// Wipe every collection for the current embedder (memory + disk).
    func clearAll() {
        index = RAGVectorIndex()
        collections = []
        documentCount = 0
        corpusName = ""
        searchResults = []
        answer = ""
        queryScope = nil
        RAGIndexStore.deleteAll(embedderId: selectedEmbedder.id)
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
                configuration: selectedEmbedder.configuration
            ) { [weak self] progress in
                Task { @MainActor in
                    self?.embedderDownloadProgress = progress.fractionCompleted
                    let pct = Int(progress.fractionCompleted * 100)
                    self?.embedderInfo = "Embedding modeli indiriliyor (\(pct)%)"
                }
            }

            embedderInfo = "\(selectedEmbedder.shortName) hazır"
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

        let collectionName =
            displayName
            ?? (urls.count == 1
                ? urls[0].lastPathComponent : "\(documents.count) doküman")
        corpusName = collectionName
        let stamped = documents.map {
            RAGDocument(name: $0.name, contents: $0.contents, collection: collectionName)
        }
        await buildIndexFromDocuments(stamped)
    }

    func useBundledDocs(selectedNames: [String]) {
        let collectionName = "Örnek Dokümanlar"
        let docs = selectedNames.compactMap { name -> RAGDocument? in
            guard let url = Bundle.main.url(forResource: name, withExtension: "txt"),
                let contents = try? String(contentsOf: url, encoding: .utf8),
                !contents.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { return nil }
            return RAGDocument(
                name: "\(name).txt", contents: contents, collection: collectionName)
        }
        guard !docs.isEmpty else {
            indexError = "Örnek dokümanlar uygulama paketinde bulunamadı."
            return
        }
        corpusName = collectionName
        Task { await buildIndexFromDocuments(docs) }
    }

    // MARK: - Indexing

    /// Embed `documents` and add them as a collection. The collection name comes
    /// from `RAGDocument.collection` (stamped at load time). Unlike before, this
    /// **accumulates** — existing collections are kept — and persists the new
    /// collection to disk so it survives app restarts.
    func buildIndexFromDocuments(_ documents: [RAGDocument]) async {
        guard !isIndexing else { return }
        let collectionName = documents.first?.collection ?? corpusName

        isIndexing = true
        indexingProgress = 0
        indexError = nil
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
            var newEntries: [RAGIndexEntry] = []
            var processed = 0

            var batchStart = 0
            while batchStart < chunks.count {
                let batchEnd = min(batchStart + batchSize, chunks.count)
                let batch = Array(chunks[batchStart..<batchEnd])

                let vectors = try await embedTexts(
                    batch.map { $0.text }, isQuery: false, container: container)

                for (i, vector) in vectors.enumerated() {
                    guard batch.indices.contains(i) else { continue }
                    let normalized = RAGEmbedding.normalize(vector)
                    newEntries.append(RAGIndexEntry(chunk: batch[i], embedding: normalized))
                }

                processed += batch.count
                indexingProgress = Double(processed) / Double(chunks.count)
                batchStart = batchEnd
            }

            // Re-index = replace the same-named collection in place, keep others.
            index.removeCollection(collectionName)
            for entry in newEntries { index.add(entry) }

            collections.removeAll { $0.name == collectionName }
            collections.append(
                RAGCollectionInfo(
                    name: collectionName, documentCount: documents.count,
                    chunkCount: newEntries.count))
            documentCount = collections.reduce(0) { $0 + $1.documentCount }

            persist(
                collection: collectionName, entries: newEntries,
                documentCount: documents.count)

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
                let queryVectors = try await embedTexts(
                    [currentQuery], isQuery: true, container: container)

                guard let rawVector = queryVectors.first else {
                    isSearching = false
                    return
                }

                let normalizedQuery = RAGEmbedding.normalize(rawVector)
                searchResults = index.search(
                    query: normalizedQuery, topK: 5, collection: queryScope)
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

    private func embedTexts(
        _ texts: [String], isQuery: Bool, container: EmbedderModelContainer
    ) async throws -> [[Float]] {
        try await RAGEmbedding.embed(
            texts: texts, isQuery: isQuery, model: selectedEmbedder, container: container)
    }
}
