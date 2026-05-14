// Copyright © 2025 Apple Inc.

import SwiftData
import SwiftUI

struct ComparisonView: View {
    @Environment(\.modelContext) private var modelContext

    @Query(sort: \InferenceResult.timestamp, order: .reverse)
    private var allResults: [InferenceResult]

    @State private var searchText = ""
    @State private var sortByDate = true

    private var distinctPrompts: [String] {
        var seen = Set<String>()
        let ordered = allResults.compactMap { result -> String? in
            seen.insert(result.prompt).inserted ? result.prompt : nil
        }
        let filtered = searchText.isEmpty
            ? ordered
            : ordered.filter { $0.localizedCaseInsensitiveContains(searchText) }
        return sortByDate ? filtered : filtered.sorted()
    }

    var body: some View {
        Group {
            if allResults.isEmpty {
                emptyState
            } else {
                ScrollView {
                    VStack(spacing: 0) {
                        searchSortBar
                            .padding(.horizontal)
                            .padding(.top, 8)
                            .padding(.bottom, 12)

                        LazyVStack(alignment: .leading, spacing: 24) {
                            ForEach(distinctPrompts, id: \.self) { prompt in
                                ComparisonPromptSection(
                                    promptText: prompt,
                                    results: allResults.filter { $0.prompt == prompt }
                                )
                            }

                            MetricsLegendView()
                                .padding(.top, 8)
                        }
                        .padding(.horizontal)
                        .padding(.bottom, 20)
                    }
                }
            }
        }
        .navigationTitle("Compare")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var searchSortBar: some View {
        HStack(spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .font(.subheadline)

                TextField("Search prompts", text: $searchText)
                    .textFieldStyle(.plain)

                if !searchText.isEmpty {
                    Button { searchText = "" } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 10))

            Menu {
                Button {
                    sortByDate = true
                } label: {
                    Label("Newest First",
                          systemImage: sortByDate ? "checkmark" : "clock.arrow.circlepath")
                }
                Button {
                    sortByDate = false
                } label: {
                    Label("Prompt A–Z",
                          systemImage: sortByDate ? "textformat.abc" : "checkmark")
                }
            } label: {
                Image(systemName: sortByDate ? "clock.arrow.circlepath" : "textformat.abc")
                    .font(.subheadline)
                    .foregroundStyle(.primary)
                    .frame(width: 38, height: 38)
                    .background(Color(.systemGray6), in: RoundedRectangle(cornerRadius: 10))
            }
        }
    }

    private var emptyState: some View {
        ContentUnavailableView(
            "No Benchmarks Yet",
            systemImage: "chart.bar.doc.horizontal",
            description: Text(
                "Run a generation from the Evaluate tab to record results here."
            )
        )
    }
}

// MARK: - Per-Prompt Section

struct ComparisonPromptSection: View {
    let promptText: String
    let results: [InferenceResult]

    @Environment(\.modelContext) private var modelContext
    @State private var showDeleteConfirmation = false

    // Kazanan IDs — birden fazla sonuç varsa hesaplanır
    private var tpsWinnerID: PersistentIdentifier? {
        guard results.count > 1 else { return nil }
        return results.max(by: { $0.tokensPerSecond < $1.tokensPerSecond })?.persistentModelID
    }

    private var ttftWinnerID: PersistentIdentifier? {
        guard results.count > 1 else { return nil }
        return results.min(by: { $0.timeToFirstToken < $1.timeToFirstToken })?.persistentModelID
    }

    private var timeWinnerID: PersistentIdentifier? {
        guard results.count > 1 else { return nil }
        return results.min(by: { $0.totalTime < $1.totalTime })?.persistentModelID
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            sectionHeader

            ForEach(results) { result in
                let id = result.persistentModelID
                InferenceResultCard(
                    result: result,
                    winsOnTps: id == tpsWinnerID,
                    winsOnTtft: id == ttftWinnerID,
                    winsOnTime: id == timeWinnerID
                )
                .contextMenu {
                    Button(role: .destructive) {
                        BenchmarkStore.delete(result, in: modelContext)
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
        }
    }

    private var sectionHeader: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(promptText)
                .font(.subheadline)
                .fontWeight(.semibold)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)

            HStack {
                Text("\(results.count) result\(results.count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                Button(role: .destructive) {
                    showDeleteConfirmation = true
                } label: {
                    Text("Delete All")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
                .confirmationDialog(
                    "Delete all results for this prompt?",
                    isPresented: $showDeleteConfirmation,
                    titleVisibility: .visible
                ) {
                    Button("Delete All", role: .destructive) {
                        BenchmarkStore.deleteAll(for: promptText, in: modelContext)
                    }
                }
            }
        }
    }
}

// MARK: - Metrics Legend

struct MetricsLegendView: View {
    private struct Row {
        let metric: String
        let unit: String
        let better: String
        let betterIsHigh: Bool?  // true = yüksek iyi, false = düşük iyi, nil = bağlamsal
        let description: String
    }

    private let rows: [Row] = [
        Row(metric: "tok/s", unit: "token/sn", better: "Yüksek ↑",  betterIsHigh: true,
            description: "Üretim hızı"),
        Row(metric: "TTFT",  unit: "ms",        better: "Düşük ↓",   betterIsHigh: false,
            description: "İlk token gecikmesi"),
        Row(metric: "Time",  unit: "saniye",    better: "Düşük ↓",   betterIsHigh: false,
            description: "Toplam üretim süresi"),
        Row(metric: "Tokens", unit: "adet",     better: "Bağlamsal", betterIsHigh: nil,
            description: "Üretilen token sayısı"),
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("Metrik Rehberi")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundStyle(.secondary)
            }

            VStack(spacing: 0) {
                headerRow
                ForEach(rows.indices, id: \.self) { i in
                    Divider()
                    dataRow(rows[i])
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(
                RoundedRectangle(cornerRadius: 10)
                    .strokeBorder(Color(.systemGray4), lineWidth: 0.5)
            )
        }
    }

    private var headerRow: some View {
        HStack(spacing: 0) {
            cell("Metrik",    width: 60,  isHeader: true)
            dividerV
            cell("Birimi",    width: 70,  isHeader: true)
            dividerV
            cell("İyi olan",  width: 80,  isHeader: true)
            dividerV
            cell("Ne ölçer",  width: nil, isHeader: true)
        }
        .background(Color(.systemGray5))
    }

    private func dataRow(_ row: Row) -> some View {
        HStack(spacing: 0) {
            cell(row.metric, width: 60, isHeader: false, mono: true)
            dividerV
            cell(row.unit,   width: 70, isHeader: false)
            dividerV
            betterCell(row)
            dividerV
            cell(row.description, width: nil, isHeader: false)
        }
        .background(Color(.systemGray6))
    }

    private func betterCell(_ row: Row) -> some View {
        Text(row.better)
            .font(.caption2)
            .fontWeight(.medium)
            .foregroundStyle(
                row.betterIsHigh == nil ? Color.secondary : Color.green
            )
            .frame(width: 80, alignment: .leading)
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
    }

    private func cell(
        _ text: String, width: CGFloat?, isHeader: Bool, mono: Bool = false
    ) -> some View {
        Text(text)
            .font(isHeader ? .caption2 : .caption2)
            .fontWeight(isHeader ? .semibold : .regular)
            .foregroundStyle(isHeader ? Color.secondary : Color.primary)
            .monospaced(mono)
            .frame(
                minWidth: width,
                maxWidth: width == nil ? .infinity : width,
                alignment: .leading
            )
            .padding(.horizontal, 8)
            .padding(.vertical, 8)
    }

    private var dividerV: some View {
        Rectangle()
            .fill(Color(.systemGray4))
            .frame(width: 0.5)
    }
}
