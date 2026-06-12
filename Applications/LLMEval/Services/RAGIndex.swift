// Copyright © 2025 Apple Inc.

import Accelerate
import Foundation

struct RAGDocument {
    let name: String
    let contents: String
    /// Source-derived namespace (e.g. selected folder name). Lets the index hold
    /// several isolated collections at once — the on-device equivalent of a
    /// per-tenant/per-topic namespace. Defaults to "Genel" for callers that don't
    /// care about isolation (e.g. the benchmark).
    var collection: String = "Genel"
}

/// Lightweight, UI-facing summary of one indexed collection (namespace).
struct RAGCollectionInfo: Identifiable, Hashable {
    let name: String
    let documentCount: Int
    let chunkCount: Int
    var id: String { name }
}

struct RAGChunk: Identifiable {
    let id = UUID()
    let document: RAGDocument
    let text: String
    let index: Int
    let totalChunks: Int
}

struct RAGIndexEntry {
    let chunk: RAGChunk
    let embedding: [Float]
}

struct RAGSearchResult: Identifiable {
    let id = UUID()
    let chunk: RAGChunk
    let score: Float

    var documentName: String { chunk.document.name }
    var chunkLabel: String {
        chunk.totalChunks > 1 ? "bölüm \(chunk.index + 1)/\(chunk.totalChunks)" : ""
    }
    var preview: String { String(chunk.text.prefix(300)) }
}

struct RAGVectorIndex {
    private(set) var entries: [RAGIndexEntry] = []

    mutating func add(_ entry: RAGIndexEntry) {
        entries.append(entry)
    }

    /// Drop every entry belonging to a collection (used on re-index / delete).
    mutating func removeCollection(_ collection: String) {
        entries.removeAll { $0.chunk.document.collection == collection }
    }

    /// Distinct collection names in first-seen order.
    var collectionNames: [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for entry in entries {
            let name = entry.chunk.document.collection
            if seen.insert(name).inserted { ordered.append(name) }
        }
        return ordered
    }

    func chunkCount(in collection: String) -> Int {
        entries.reduce(0) { $0 + ($1.chunk.document.collection == collection ? 1 : 0) }
    }

    /// Search the index. When `collection` is non-nil the scan is scoped to that
    /// namespace only; otherwise it spans every collection ("Tümü").
    func search(query: [Float], topK: Int, collection: String? = nil) -> [RAGSearchResult] {
        guard !entries.isEmpty, !query.isEmpty else { return [] }
        let q = sanitize(query)
        let scope = collection.map { name in
            entries.filter { $0.chunk.document.collection == name }
        } ?? entries
        guard !scope.isEmpty else { return [] }
        return scope
            .map { entry -> (RAGIndexEntry, Float) in (entry, dot(q, entry.embedding)) }
            .sorted { $0.1 > $1.1 }
            .prefix(topK)
            .map { RAGSearchResult(chunk: $0.0.chunk, score: $0.1) }
    }

    private func dot(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    private func sanitize(_ v: [Float]) -> [Float] {
        v.map { $0.isFinite ? $0 : 0 }
    }
}

enum RAGChunker {
    static let targetSize = 400
    static let maxSize = 600

    static func chunk(_ document: RAGDocument) -> [RAGChunk] {
        let raw = splitParagraphs(document.contents)
        let merged = mergeSmall(raw)
        let split = merged.flatMap { splitOversized($0) }
        let total = max(split.count, 1)
        return split.enumerated().map { i, text in
            RAGChunk(document: document, text: text, index: i, totalChunks: total)
        }
    }

    private static func splitParagraphs(_ text: String) -> [String] {
        text
            .components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private static func mergeSmall(_ paragraphs: [String]) -> [String] {
        var result: [String] = []
        var buffer = ""

        for paragraph in paragraphs {
            if buffer.isEmpty {
                buffer = paragraph
                continue
            }
            let candidate = buffer + "\n\n" + paragraph
            if candidate.count <= targetSize {
                buffer = candidate
            } else {
                result.append(buffer)
                buffer = paragraph
            }
        }
        if !buffer.isEmpty { result.append(buffer) }
        return result
    }

    private static func splitOversized(_ chunk: String) -> [String] {
        guard chunk.count > maxSize else { return [chunk] }

        let sentences = chunk
            .components(separatedBy: CharacterSet(charactersIn: ".!?\n"))
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        var result: [String] = []
        var buffer = ""
        for sentence in sentences {
            let piece = sentence + "."
            if buffer.isEmpty {
                buffer = piece
            } else if (buffer.count + piece.count + 1) <= maxSize {
                buffer += " " + piece
            } else {
                result.append(buffer)
                buffer = piece
            }
        }
        if !buffer.isEmpty { result.append(buffer) }
        return result.isEmpty ? [chunk] : result
    }
}
