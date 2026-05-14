// Copyright © 2024 Apple Inc.

import SwiftData
import SwiftUI

@main
struct LLMEvalApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                NavigationStack {
                    ContentView()
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
            }
            .environment(DeviceStat())
        }
        .modelContainer(for: InferenceResult.self)
    }
}
