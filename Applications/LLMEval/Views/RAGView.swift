// Copyright © 2025 Apple Inc.

import SwiftUI
import UniformTypeIdentifiers

struct RAGView: View {
    @Bindable var rag: RAGEvaluator
    @State private var showBundledDocPicker = false
    @State private var selectedBundledDocs: Set<String> = []
    @State private var benchmark = RAGBenchmark()
    @State private var showBenchmark = false

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    llmModelSection
                    embedderModelSection
                    corpusSection
                    querySection
                    if !rag.searchResults.isEmpty {
                        retrievedSection
                    }
                    if rag.isGenerating || !rag.answer.isEmpty {
                        answerSection
                            .id("answerSection")
                    }
                    if let err = rag.generationError {
                        Label(err, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                            .font(.caption)
                    }
                    embedderStatusView
                }
                .padding()
            }
            .onTapGesture { hideKeyboard() }
            .onChange(of: rag.isGenerating) { _, newValue in
                guard newValue else { return }
                DispatchQueue.main.async {
                    withAnimation { proxy.scrollTo("answerSection", anchor: .top) }
                }
            }
        }
        .navigationTitle("RAG Test")
        .navigationBarTitleDisplayMode(.inline)
        .fileImporter(
            isPresented: $rag.isSelectingFolder,
            allowedContentTypes: [.folder, .text, .pdf],
            allowsMultipleSelection: true
        ) { result in
            switch result {
            case .success(let urls):
                guard !urls.isEmpty else { return }
                Task { await rag.handleSelectedURLs(urls) }
            case .failure(let error):
                rag.indexError = error.localizedDescription
            }
        }
        .overlay {
            if let progress = rag.embedderDownloadProgress {
                LoadingOverlayView(
                    modelInfo: rag.embedderInfo,
                    downloadProgress: progress,
                    progressDescription: nil
                )
            }
        }
        .sheet(isPresented: $showBundledDocPicker) {
            bundledDocPickerSheet
        }
        .sheet(isPresented: $showBenchmark) {
            benchmarkSheet
        }
    }

    // MARK: - Embedder model section

    private var embedderModelSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Anlamsal Arama Modeli (Embedding)")
                        .font(.headline)
                    Spacer()
                    Button {
                        showBenchmark = true
                    } label: {
                        Label("Karşılaştır", systemImage: "chart.bar.xaxis")
                            .font(.caption)
                    }
                    .disabled(rag.isIndexing || rag.isSearching || rag.isGenerating)
                }

                Picker("Embedding", selection: Binding(
                    get: { rag.selectedEmbedder.id },
                    set: { id in
                        if let model = RAGEvaluator.availableEmbedders.first(where: { $0.id == id }) {
                            rag.selectedEmbedder = model
                        }
                    }
                )) {
                    ForEach(RAGEvaluator.availableEmbedders) { model in
                        Text("\(model.multilingual ? "🌍 " : "")\(model.shortName)  ·  \(model.size)")
                            .tag(model.id)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: .infinity)
                .disabled(rag.isIndexing || rag.isSearching || rag.isGenerating)

                if !rag.selectedEmbedder.multilingual {
                    Label(
                        "Bu model yalnızca İngilizce için iyidir; Türkçe dokümanlarda zayıf sonuç verebilir.",
                        systemImage: "exclamationmark.triangle.fill"
                    )
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("Çok dilli model — Türkçe dokümanlar için önerilir.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Benchmark sheet

    private var benchmarkSheet: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(
                        "Örnek Türkçe dokümanlar üzerinde \(RAGBenchmark.goldQueries.count) etiketli soruyla modellerin isabet oranı (hit@k) ve MRR değeri ölçülür."
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)

                    if benchmark.isRunning {
                        VStack(alignment: .leading, spacing: 6) {
                            ProgressView(value: benchmark.progress)
                            Text(benchmark.progressText)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if let err = benchmark.error {
                        Label(err, systemImage: "exclamationmark.triangle")
                            .font(.caption)
                            .foregroundStyle(.red)
                    }

                    if !benchmark.results.isEmpty {
                        benchmarkTable
                    }

                    Button {
                        benchmark.run(models: RAGEvaluator.availableEmbedders)
                    } label: {
                        Text(benchmark.isRunning ? "Çalışıyor…" : "Tüm Modelleri Çalıştır")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(benchmark.isRunning)

                    Text(
                        "Not: Modeller sırayla indirilip değerlendirilir; ilk çalıştırmada büyük indirmeler olabilir."
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("Model Karşılaştırma")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Kapat") { showBenchmark = false }
                        .disabled(benchmark.isRunning)
                }
            }
        }
    }

    private var bestHitAt3: Double {
        benchmark.results.map(\.hitAt3).max() ?? 0
    }

    private var benchmarkTable: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Model").font(.caption).fontWeight(.semibold)
                    .frame(maxWidth: .infinity, alignment: .leading)
                Group {
                    Text("hit@1")
                    Text("hit@3")
                    Text("hit@5")
                    Text("MRR")
                }
                .font(.caption2)
                .fontWeight(.semibold)
                .frame(width: 44)
            }
            .padding(.vertical, 6)
            Divider()
            ForEach(benchmark.results) { result in
                HStack {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(result.model.shortName)
                            .font(.caption)
                            .lineLimit(1)
                        if let failure = result.failure {
                            Text(failure)
                                .font(.caption2)
                                .foregroundStyle(.red)
                                .lineLimit(1)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    Group {
                        Text(pct(result.hitAt1))
                        Text(pct(result.hitAt3))
                            .fontWeight(
                                result.hitAt3 == bestHitAt3 && result.failure == nil
                                    ? .bold : .regular)
                        Text(pct(result.hitAt5))
                        Text(String(format: "%.2f", result.mrr))
                    }
                    .font(.caption2)
                    .monospacedDigit()
                    .frame(width: 44)
                }
                .padding(.vertical, 6)
                .background(
                    result.hitAt3 == bestHitAt3 && result.failure == nil
                        ? Color.green.opacity(0.12) : Color.clear)
                Divider()
            }
        }
    }

    private func pct(_ value: Double) -> String {
        "\(Int((value * 100).rounded()))%"
    }

    // MARK: - LLM model section

    private var llmModelSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 8) {
                Text("Cevap Üretici Model")
                    .font(.headline)

                Picker("Model", selection: Binding(
                    get: { rag.llm.modelConfiguration.name },
                    set: { name in
                        if let evalModel = LLMEvaluator.availableModels.first(where: { $0.id == name }) {
                            rag.llm.modelConfiguration = evalModel.configuration
                        }
                    }
                )) {
                    ForEach(LLMEvaluator.availableModels) { evalModel in
                        Text("\(evalModel.isCompatibleWithDevice ? "" : "⚠️ ")\(evalModel.shortName)  ·  \(evalModel.size)")
                            .tag(evalModel.id)
                    }
                }
                .labelsHidden()
                .frame(maxWidth: .infinity)
                .disabled(rag.llm.running || rag.llm.isLoading || rag.isGenerating)

                if let warning = rag.llm.memoryWarning {
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                } else if rag.llm.isLoading {
                    HStack(spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text(rag.llm.modelInfo.isEmpty ? "Model yükleniyor..." : rag.llm.modelInfo)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                } else if !rag.llm.modelInfo.isEmpty {
                    Text(rag.llm.modelInfo)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
            }
        }
    }

    // MARK: - Corpus section

    private var corpusSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Doküman Kaynağı")
                            .font(.headline)
                        Text(rag.corpusName.isEmpty ? "Henüz seçilmedi" : rag.corpusName)
                            .font(.caption)
                            .foregroundStyle(rag.corpusName.isEmpty ? .secondary : .primary)
                    }
                    Spacer()
                    Button("Kaynak Seç") { rag.selectCorpusFolder() }
                        .disabled(rag.isIndexing)
                    Button("Örnek Dokümanlar") {
                        selectedBundledDocs = Set(RAGEvaluator.bundledDocNames)
                        showBundledDocPicker = true
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(rag.isIndexing)
                }

                if rag.isIndexing {
                    VStack(alignment: .leading, spacing: 4) {
                        ProgressView(value: rag.indexingProgress)
                        Text("İndeksleniyor... \(Int(rag.indexingProgress * 100))%")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else if rag.documentCount > 0 {
                    Label(
                        "\(rag.documentCount) doküman indekslendi",
                        systemImage: "checkmark.circle.fill"
                    )
                    .foregroundStyle(.green)
                    .font(.caption)
                }

                if let err = rag.indexError {
                    Label(err, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .font(.caption)
                }
            }
        }
    }

    // MARK: - Query section

    private var querySection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                Text("Soru")
                    .font(.headline)
                HStack(alignment: .bottom, spacing: 8) {
                    ZStack(alignment: .topTrailing) {
                        TextField("Soru yaz...", text: $rag.query, axis: .vertical)
                            .textFieldStyle(.roundedBorder)
                            .lineLimit(2...5)
                            .padding(.trailing, rag.query.isEmpty ? 0 : 28)
                            .disabled(rag.isSearching || rag.isGenerating || rag.documentCount == 0)
                        if !rag.query.isEmpty {
                            Button {
                                rag.query = ""
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.top, 8)
                            .padding(.trailing, 8)
                        }
                    }
                    Button {
                        rag.runRAGQuery()
                    } label: {
                        if rag.isSearching {
                            ProgressView().controlSize(.small)
                        } else {
                            Text("Ara")
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(
                        rag.query.isEmpty || rag.documentCount == 0 || rag.isSearching
                            || rag.isGenerating)
                }
                if rag.documentCount == 0 {
                    Text("Önce bir doküman kaynağı seçin.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    // MARK: - Retrieved docs section

    private var retrievedSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Text("Bulunan Bağlam")
                    .font(.headline)
                if rag.searchResults.count > 3 {
                    ScrollView {
                        retrievedRows
                    }
                    .frame(maxHeight: 360)
                } else {
                    retrievedRows
                }
            }
        }
    }

    private var retrievedRows: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(rag.searchResults) { result in
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(result.documentName)
                            .font(.caption)
                            .fontWeight(.semibold)
                            .lineLimit(1)
                        if !result.chunkLabel.isEmpty {
                            Text(result.chunkLabel)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 1)
                                .background(Color.secondary.opacity(0.15))
                                .clipShape(Capsule())
                        }
                        Spacer()
                        Text(String(format: "%.3f", result.score))
                            .font(.caption)
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                    Text(result.preview)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(4)
                }
                if result.id != rag.searchResults.last?.id {
                    Divider()
                }
            }
        }
    }

    // MARK: - Answer section

    private var answerSection: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Cevap")
                        .font(.headline)
                    Spacer()
                    if rag.isGenerating {
                        ProgressView().controlSize(.small)
                    }
                }
                if rag.answer.isEmpty && rag.isGenerating {
                    Text("Üretiliyor...")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text(rag.answer)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    // MARK: - Embedder status

    private var embedderStatusView: some View {
        Group {
            if !rag.embedderInfo.isEmpty {
                Text(rag.embedderInfo)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Bundled doc picker sheet

    private var bundledDocPickerSheet: some View {
        NavigationStack {
            List {
                ForEach(RAGEvaluator.bundledDocNames, id: \.self) { (name: String) in
                    HStack {
                        Text(name.replacingOccurrences(of: "_", with: " ").capitalized)
                        Spacer()
                        if selectedBundledDocs.contains(name) {
                            Image(systemName: "checkmark")
                                .foregroundStyle(.tint)
                        }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        if selectedBundledDocs.contains(name) {
                            selectedBundledDocs.remove(name)
                        } else {
                            selectedBundledDocs.insert(name)
                        }
                    }
                }
            }
            .navigationTitle("Örnek Dokümanlar")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("İptal") { showBundledDocPicker = false }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("İndeksle") {
                        showBundledDocPicker = false
                        rag.useBundledDocs(selectedNames: Array(selectedBundledDocs))
                    }
                    .disabled(selectedBundledDocs.isEmpty)
                }
            }
        }
    }

    // MARK: - Private helpers

    private func hideKeyboard() {
        UIApplication.shared.sendAction(
            #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}
