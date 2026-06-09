// Copyright © 2024 Apple Inc.

import SwiftData
import SwiftUI

@main
struct LLMEvalApp: App {
    @State private var llm = LLMEvaluator()

    var body: some Scene {
        WindowGroup {
            if LLMEvaluator.isGPUFamilySupported {
                TabView {
                    NavigationStack {
                        ContentView(llm: llm)
                    }
                    .tabItem {
                        Label("Evaluate", systemImage: "waveform.and.magnifyingglass")
                    }

                    NavigationStack {
                        ComparisonView()
                    }
                    .tabItem {
                        Label("Compare", systemImage: "chart.bar.doc.horizontal")
                    }

                    NavigationStack {
                        RAGView(rag: RAGEvaluator(llm: llm))
                    }
                    .tabItem {
                        Label("RAG", systemImage: "doc.text.magnifyingglass")
                    }

                    NavigationStack {
                        ExecuTorchView(evaluator: ExecuTorchEvaluator())
                    }
                    .tabItem {
                        Label("ExecuTorch", systemImage: "cpu")
                    }
                }
                .environment(DeviceStat())
            } else {
                UnsupportedDeviceView()
            }
        }
        .modelContainer(for: InferenceResult.self)
    }
}
