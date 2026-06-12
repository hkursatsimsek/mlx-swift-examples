// Copyright © 2025 Apple Inc.

import Foundation

// MARK: - Persisted (Codable) representation
//
// The live model (`RAGChunk`/`RAGIndexEntry`) carries non-Codable identity
// (`UUID`) and a full `RAGDocument.contents` we don't need on disk, so we
// persist a lean DTO instead. Only `chunk.text` + the embedding are required to
// rebuild a queryable index; document contents are recomputed-free because
// retrieval and the LLM prompt only ever use `chunk.text`.

struct PersistedChunk: Codable {
    let documentName: String
    let text: String
    let index: Int
    let totalChunks: Int
}

struct PersistedEntry: Codable {
    let chunk: PersistedChunk
    let embedding: [Float]
}

/// One indexed collection (namespace) as stored on disk. `embedderId` guards
/// against mixing incompatible vector spaces — collections are kept in a folder
/// keyed by the embedder that produced them.
struct PersistedCollection: Codable {
    let name: String
    let embedderId: String
    let documentCount: Int
    let createdAt: Date
    let entries: [PersistedEntry]
}

/// On-disk store for RAG collections.
///
/// Layout: `<Application Support>/RAGIndex/<embedderId>/<collection>.json`
///
/// Application Support is chosen over Caches (no eviction under storage
/// pressure) and over Documents (app-managed data, not user-browsable). This
/// removes the re-index-on-every-launch problem: a collection is embedded once
/// and reloaded from disk thereafter. Mirrors the JSON index pattern already
/// used by `Tools/embedder-tool`.
enum RAGIndexStore {

    private static let rootFolder = "RAGIndex"

    // MARK: Directories

    private static func baseDirectory() throws -> URL {
        let support = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true)
        return support.appendingPathComponent(rootFolder, isDirectory: true)
    }

    /// Folder holding all collections produced by `embedderId` (created if needed).
    static func directory(embedderId: String) throws -> URL {
        let dir = try baseDirectory()
            .appendingPathComponent(sanitize(embedderId), isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private static func fileURL(name: String, embedderId: String) throws -> URL {
        try directory(embedderId: embedderId)
            .appendingPathComponent(sanitize(name) + ".json")
    }

    // MARK: CRUD

    static func save(_ collection: PersistedCollection) throws {
        let url = try fileURL(name: collection.name, embedderId: collection.embedderId)
        let encoder = JSONEncoder()
        let data = try encoder.encode(collection)
        try data.write(to: url, options: .atomic)
    }

    /// Load every collection for an embedder, oldest first. Corrupt/unreadable
    /// files are skipped rather than failing the whole load.
    static func loadAll(embedderId: String) -> [PersistedCollection] {
        guard let dir = try? directory(embedderId: embedderId),
            let urls = try? FileManager.default.contentsOfDirectory(
                at: dir, includingPropertiesForKeys: nil)
        else { return [] }

        let decoder = JSONDecoder()
        return urls
            .filter { $0.pathExtension == "json" }
            .compactMap { try? Data(contentsOf: $0) }
            .compactMap { try? decoder.decode(PersistedCollection.self, from: $0) }
            .sorted { $0.createdAt < $1.createdAt }
    }

    static func delete(name: String, embedderId: String) {
        guard let url = try? fileURL(name: name, embedderId: embedderId) else { return }
        try? FileManager.default.removeItem(at: url)
    }

    static func deleteAll(embedderId: String) {
        guard let dir = try? directory(embedderId: embedderId) else { return }
        try? FileManager.default.removeItem(at: dir)
    }

    // MARK: Helpers

    /// Map an arbitrary string (HF repo id with slashes, Turkish collection
    /// names) to a safe path component. The authoritative name lives inside the
    /// JSON, so light collisions only risk overwriting a same-sanitized sibling.
    private static func sanitize(_ s: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_."))
        let mapped = s.unicodeScalars.map { allowed.contains($0) ? Character($0) : "_" }
        let result = String(mapped)
        return result.isEmpty ? "untitled" : result
    }
}
