// Copyright © 2025 Apple Inc.

import Foundation
import SwiftData

enum BenchmarkStore {

    @MainActor
    static func save(_ result: InferenceResult, in context: ModelContext) {
        context.insert(result)
        try? context.save()
    }

    @MainActor
    static func delete(_ result: InferenceResult, in context: ModelContext) {
        context.delete(result)
        try? context.save()
    }

    @MainActor
    static func deleteAll(for promptText: String, in context: ModelContext) {
        let descriptor = FetchDescriptor<InferenceResult>()
        let all = (try? context.fetch(descriptor)) ?? []
        all.filter { $0.prompt == promptText }.forEach { context.delete($0) }
        try? context.save()
    }
}
